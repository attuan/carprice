"""機能A の LLM フォールバック（設計書 05）を実測する。

## 何を確かめたいか

機能A は「埋め込み → 近傍で分類 → confidence が閾値未満なら LLM へ」という
造りになっている（設計書 01〜05）。04 までは APIキー無しで実装・測定済みで、
05 だけが `QueueOnlyFallback`（答えずにレビュー待ちへ積むだけ）だった。

そこにキーが来たので、実物の `ClaudeFallback` を差して問う:

> **近傍分類が自信を持てなかった行を LLM に回すと、実際に正しくなるのか。
> 1行いくらか。**

これが成り立たないなら、信頼度ルーティング（設計書の AdaptivePredictor、
PRD「信頼度ルーティング」）の前提そのものが崩れる。

## 2つのデータセットで測る

同じ問いを、性質の違う2つの非構造列で測る。**列名も宣言する値も
データセットごとに違う**（それぞれの原典の言葉で考える）。共通なのは
「値の名前だけを宣言する（PRD 機能A の 01・案b）」という起点と、
正解を人手ルールの結果に取ることだけ。

    sienta   … タイトル → グレード名。宣言値10個。正解は正規表現版のグレード名
    vehicles … model → 車種の芯。宣言値60個。正解は手書きルールの正規化結果

シエンタ側は「グレード名がタイトルに文字どおり書いてある」ので LLM には
読み取るだけの課題になる。Craigslist の `model` は 19,739 種類の自由記述で、
区切りも語順も揃っていない。**同じ仕組みが後者でも成り立つか**が
Craigslist を足す理由（`docs/2026-09-01-feature-fallback.md` の続き）。

## 測り方

エスカレーション率を 0% → 30% と上げながら、生成した列が人手ルール版と
どれだけ一致するか（中間ラベル accuracy）を見る。
`escalate_rate` は「confidence の低い順に何割を回すか」なので、
5% に回る行は 15% に回る行の部分集合になる。**プロンプトのキャッシュが
効くので、率を上げても追加ぶんしか課金されない。**

費用を抑えるため 1 fold・test を抽出して測る。
**下流の MAE はここでは測らない。** 下流まで見るには訓練側の行も
同じ率でエスカレーションする必要があり、桁が変わるため。

## 実行

    .venv/bin/python scripts/run_feature_fallback.py
    .venv/bin/python scripts/run_feature_fallback.py --dataset vehicles
    .venv/bin/python scripts/run_feature_fallback.py --n-test 400 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_feature import COL, DECLARED_VALUES, GENERATED  # noqa: E402
from demo_feature_vehicles import (  # noqa: E402
    GENERATED as V_GENERATED, declared_values,
)
from eval_protocol import N_SPLITS, SEED, VEHICLES, load_dataset  # noqa: E402
from run_baselines_vehicles import (  # noqa: E402
    N_SAMPLE, RULE_COL, TEXT as V_TEXT, normalize_model,
)

from unfold import Feature, QueueOnlyFallback  # noqa: E402
from unfold.fallback import ClaudeFallback  # noqa: E402
from unfold.llm import ClaudeClient  # noqa: E402

RATES = (0.0, 0.05, 0.15, 0.30)


# --- 測定対象 ---------------------------------------------------------


@dataclass(frozen=True)
class FallbackTask:
    """1つのデータセットについて、この測定が知っている必要のあること。

    以降の処理はここに書かれた列名しか見ない。データセット固有の知識
    （元データの列名・宣言する値の作り方）は `build` の中に閉じ込める。
    """

    name: str
    build: Callable[[], tuple[pd.DataFrame, list[str]]]  # → (データ, 宣言値)
    source: str          # 分類の入力にする非構造列
    truth: str           # 正解とみなす列（人手ルールの結果）
    generated: str       # 機能A が作る列の名前
    out: Path            # 結果の書き出し先
    note: str            # 画面に出す1行説明


def _build_sienta() -> tuple[pd.DataFrame, list[str]]:
    """シエンタ 5,507行。宣言値はカタログを見れば書ける10グレード。"""
    return load_dataset(), list(DECLARED_VALUES)


def _build_vehicles() -> tuple[pd.DataFrame, list[str]]:
    """Craigslist 60,000行の抽出。宣言値は「よく出る model を60個」。

    正解は手書きルール（区切り記号以降を捨てて先頭2語）の正規化結果。
    シエンタの正規表現版グレード名にあたる、人間側の線である。
    宣言値も正解も **price を一切見ていない**ので目的変数のリークにはあたらない。
    """
    df = load_dataset(dataset=VEHICLES, sample=N_SAMPLE)
    df[RULE_COL] = df[V_TEXT].map(normalize_model)
    return df, declared_values(df)


TASKS = {
    "sienta": FallbackTask(
        name="シエンタ（単一車種・日本語）",
        build=_build_sienta,
        source=COL,
        truth="グレード名",
        generated=GENERATED,
        out=ROOT / "results" / "feature_fallback.csv",
        note="タイトルからグレード名を作る。グレード名は文字どおり書かれている",
    ),
    "vehicles": FallbackTask(
        name="Craigslist（複数車種・英語）",
        build=_build_vehicles,
        source=V_TEXT,
        truth=RULE_COL,
        generated=V_GENERATED,
        out=ROOT / "results" / "feature_fallback_vehicles.csv",
        note="model（19,739種類の自由記述）から車種の芯を作る。区切りも語順も揃っていない",
    ),
}


# --- 測定 -------------------------------------------------------------


def one_rate(task: FallbackTask, train: pd.DataFrame, test: pd.DataFrame,
             values: list[str], rate: float, client: ClaudeClient,
             dry_run: bool) -> dict:
    """あるエスカレーション率で1回だけ測る。"""
    fb = (QueueOnlyFallback() if (rate == 0.0 or dry_run)
          else ClaudeFallback(client=client))
    f = Feature(source=task.source, type="category", values=values,
                k="auto", escalate_rate=rate if rate > 0 else None,
                threshold=0.9, fallback=fb, name=task.generated)
    f.fit(train)
    pred = f.transform(test).astype(str).to_numpy()
    truth = test[task.truth].astype(str).to_numpy()

    prov = f._prov()
    llm_rows = (prov["由来"] == "llm").to_numpy()
    ok = pred == truth
    # 宣言した値に無い正解は、そもそも当てようがない。
    # LLM の効果は「宣言値に含まれる行」で見ないと過小評価になる
    in_scope = np.isin(truth, values)

    return {
        "エスカレーション率": rate,
        "実際に回した行": int(llm_rows.sum()),
        "accuracy": float(ok.mean()),
        "宣言値に限った accuracy": float(ok[in_scope].mean()),
        "回した行の accuracy": (float(ok[llm_rows].mean())
                                if llm_rows.any() else float("nan")),
        "回した行の宣言値内 accuracy": (
            float(ok[llm_rows & in_scope].mean())
            if (llm_rows & in_scope).any() else float("nan")),
        "費用_usd": float(prov["コスト"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="機能A の LLM フォールバックを実測")
    ap.add_argument("--dataset", default="sienta", choices=sorted(TASKS),
                    help="測定するデータセット（既定 sienta）")
    ap.add_argument("--n-test", type=int, default=400,
                    help="採点に使う test の行数（費用を抑えるため抽出する）")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="LLM を呼ばず、回る行数だけ数える")
    args = ap.parse_args()

    task = TASKS[args.dataset]
    client = ClaudeClient(model=args.model, effort=args.effort,
                          max_workers=args.workers)
    if not args.dry_run and not client.available():
        print("APIキーがありません。.env にキーを書いてください。")
        raise SystemExit(1)

    df, values = task.build()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(iter(kf.split(df)))       # fold1 だけ使う
    train = df.iloc[tr_idx].reset_index(drop=True)
    test_all = df.iloc[te_idx].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(test_all),
                              size=min(args.n_test, len(test_all)),
                              replace=False))
    test = test_all.iloc[pick].reset_index(drop=True)

    truth_all = test[task.truth].astype(str)
    covered = float(np.isin(truth_all.to_numpy(), values).mean())
    n_levels = int(df[task.truth].astype(str).nunique())
    print(f"\n=== {task.name} ===")
    print(task.note)
    print(f"fold1 のみ / 訓練 {len(train):,} 行 → 採点 {len(test)} 行")
    print(f"宣言した{len(values)}値がカバーするのは test の {covered:.1%}"
          f"（正解は全体で {n_levels:,} 水準あり、残りは当てようがない裾）")
    print("参照事例は値の名前だけ（PRD 機能A の 01・案b）。人手ラベルは使わない。\n")

    rows = [one_rate(task, train, test, values, r, client, args.dry_run)
            for r in RATES]
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))

    base = res.iloc[0]
    print(f"\n基準（0% = LLM を呼ばない）: accuracy {base['accuracy']:.3f} / "
          f"宣言値内 {base['宣言値に限った accuracy']:.3f}")
    for _, r in res.iloc[1:].iterrows():
        d = r["宣言値に限った accuracy"] - base["宣言値に限った accuracy"]
        n = int(r["実際に回した行"])
        per = r["費用_usd"] / n if n else float("nan")
        print(f"  {r['エスカレーション率']:.0%} 回すと "
              f"宣言値内 accuracy {r['宣言値に限った accuracy']:.3f}"
              f"（{d:+.3f}） / {n} 行 / ${r['費用_usd']:.3f}"
              f"（1行 ${per:.4f}）")

    task.out.parent.mkdir(exist_ok=True)
    res.to_csv(task.out, index=False, encoding="utf-8-sig")
    print(f"\n結果: {task.out.relative_to(ROOT)}")
    if not args.dry_run:
        print("\n--- 実測費用（累計）---")
        for k, v in client.summary().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
