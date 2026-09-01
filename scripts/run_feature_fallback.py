"""機能A の LLM フォールバック（設計書 05）を実測する。

## 何を確かめたいか

機能A は「埋め込み → 近傍で分類 → confidence が閾値未満なら LLM へ」という
造りになっている（設計書 01〜05）。04 までは APIキー無しで実装・測定済みで、
05 だけが `QueueOnlyFallback`（答えずにレビュー待ちへ積むだけ）だった。

そこにキーが来たので、実物の `ClaudeFallback` を差して問う:

> **近傍分類が自信を持てなかった行を LLM に回すと、実際に正しくなるのか。
> 1行いくらか。**

これが成り立たないなら、信頼度ルーティング（設計書の AdaptivePredictor、
PRD §6.3）の前提そのものが崩れる。

## 測り方

エスカレーション率を 0% → 30% と上げながら、生成した列が正規表現版の
グレード名とどれだけ一致するか（中間ラベル accuracy）を見る。
`escalate_rate` は「confidence の低い順に何割を回すか」なので、
5% に回る行は 15% に回る行の部分集合になる。**プロンプトのキャッシュが
効くので、率を上げても追加ぶんしか課金されない。**

費用を抑えるため 1 fold・test を抽出して測る。
**下流の MAE はここでは測らない。** 下流まで見るには訓練側の行も
同じ率でエスカレーションする必要があり（4,405 行）、桁が変わるため。

## 実行

    .venv/bin/python scripts/run_feature_fallback.py
    .venv/bin/python scripts/run_feature_fallback.py --n-test 400 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_feature import COL, DECLARED_VALUES, GENERATED  # noqa: E402
from eval_protocol import N_SPLITS, SEED, load_dataset  # noqa: E402

from unfold import Feature, QueueOnlyFallback  # noqa: E402
from unfold.fallback import ClaudeFallback  # noqa: E402
from unfold.llm import ClaudeClient  # noqa: E402

RATES = (0.0, 0.05, 0.15, 0.30)


def one_rate(train: pd.DataFrame, test: pd.DataFrame, rate: float,
             client: ClaudeClient, dry_run: bool) -> dict:
    """あるエスカレーション率で1回だけ測る。"""
    fb = (QueueOnlyFallback() if (rate == 0.0 or dry_run)
          else ClaudeFallback(client=client))
    f = Feature(source=COL, type="category", values=DECLARED_VALUES,
                k="auto", escalate_rate=rate if rate > 0 else None,
                threshold=0.9, fallback=fb, name=GENERATED)
    f.fit(train)
    pred = f.transform(test).astype(str).to_numpy()
    truth = test["グレード名"].astype(str).to_numpy()

    prov = f._prov()
    llm_rows = (prov["由来"] == "llm").to_numpy()
    ok = pred == truth
    # 宣言した10値に無いグレードは、そもそも正解になりようがない。
    # LLM の効果は「宣言値に含まれる行」で見ないと過小評価になる
    in_scope = np.isin(truth, DECLARED_VALUES)

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
    ap.add_argument("--n-test", type=int, default=400,
                    help="採点に使う test の行数（費用を抑えるため抽出する）")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="LLM を呼ばず、回る行数だけ数える")
    args = ap.parse_args()

    client = ClaudeClient(model=args.model, effort=args.effort,
                          max_workers=args.workers)
    if not args.dry_run and not client.available():
        print("ANTHROPIC_API_KEY がありません。.env にキーを書いてください。")
        raise SystemExit(1)

    df = load_dataset()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(iter(kf.split(df)))       # fold1 だけ使う
    train = df.iloc[tr_idx].reset_index(drop=True)
    test_all = df.iloc[te_idx].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(test_all),
                              size=min(args.n_test, len(test_all)),
                              replace=False))
    test = test_all.iloc[pick].reset_index(drop=True)

    covered = float(np.isin(test["グレード名"].astype(str), DECLARED_VALUES).mean())
    print(f"\nfold1 のみ / 訓練 {len(train):,} 行 → 採点 {len(test)} 行")
    print(f"宣言した10値がカバーするのは test の {covered:.1%}"
          f"（残りは 116 水準の裾で、正解になりようがない）")
    print("参照事例は値の名前だけ（PRD §7-1 の案b）。人手ラベルは使わない。\n")

    rows = [one_rate(train, test, r, client, args.dry_run) for r in RATES]
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

    out = ROOT / "results" / "feature_fallback.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n結果: {out.relative_to(ROOT)}")
    if not args.dry_run:
        print("\n--- 実測費用（累計）---")
        for k, v in client.summary().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
