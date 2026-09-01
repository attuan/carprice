"""`python -m unfold.demo` — ライブラリ全体を一度に動かして見せる入口。

**APIキーが無くても最後まで走る。**既定では LLM を1回も呼ばないので無料。
使うデータは git に入っている 500 行の抜粋（`sampledata/sample/vehicles_sample500.csv`）
なので、**clone 直後・データ取得なしで動く。**

    .venv/bin/python -m unfold.demo              # 無料。LLM は呼ばない
    .venv/bin/python -m unfold.demo --run 20     # 20 行を予測する（うち3割が LLM に回る）

順番は実際の使い方どおりにしてある。

1. **リーク検査** — 同じ車が train と test に入っていないか（`check_duplicates`）
2. **スクリーニング** — そのデータでテキストが効くか（`screen`。LLM を呼ばない）
3. **機能A** — 非構造列を型付き列にする（`Feature`）
4. **見積もり** — LLM に何行回すと、いくら・何秒か（`AdaptivePredictor.plan`）
5. **実行** — `--run N` を付けたときだけ。予測と来歴（`explain` / `report` / `cost`）
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from unfold import (
    AdaptivePredictor,
    Feature,
    UnfoldWarning,
    check_duplicates,
    check_overlap,
    screen,
)
from unfold.llm import ClaudeClient

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sampledata" / "sample" / "vehicles_sample500.csv"
BASE_YEAR = 2021          # データの掲載時期（posting_date は 2021-04〜05）
SEED = 42

NUM = ["車齢", "走行距離_mile"]
CAT = ["メーカー", "州", "燃料", "変速機"]
TEXT = "車種名"


def load() -> pd.DataFrame:
    """抜粋データを、そのまま学習に使える形にする。

    `clean_vehicles.py` の軽い版。**価格の外れ値は必ず切る**
    （元データは最大 $3,736,928,711 で、平均を取ると壊れる）。
    """
    df = pd.read_csv(SAMPLE)
    df = df[df["price"].between(1_000, 100_000)]
    df = df[df["year"].notna() & df["manufacturer"].notna()
            & df["odometer"].notna()]
    out = pd.DataFrame({
        "物件ID": df["id"],
        "価格_usd": df["price"].astype(float),
        "車齢": BASE_YEAR - df["year"],
        "走行距離_mile": df["odometer"],
        "メーカー": df["manufacturer"].str.lower().str.strip(),
        "車種名": df["model"].fillna("").str.lower().str.strip(),
        "燃料": df["fuel"].fillna("不明"),
        "変速機": df["transmission"].fillna("不明"),
        "州": df["state"],
        "車台番号": df["VIN"],
    })
    return out.reset_index(drop=True)


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def step1_leak(df: pd.DataFrame) -> pd.DataFrame:
    hr("1. リーク検査 — 同じ車が二重に入っていないか")
    print("同じ車が複数の地域に出稿されているデータでは、ランダムに分割すると")
    print("同じ車が train と test の両方に入り、答えを見て答えることになる。\n")

    print("[全列で照合]")
    print(check_duplicates(df, ignore=["物件ID"]))
    print("\n[車台番号（VIN）で照合]")
    vin = df[df["車台番号"].notna()]
    print(check_duplicates(vin, keys=["車台番号"]))

    before = len(df)
    df = df[~df["車台番号"].notna() | ~df["車台番号"].duplicated()]
    df = df.reset_index(drop=True)
    print(f"\n→ 1台1行に潰した: {before} 行 → {len(df)} 行")
    return df


def split(df: pd.DataFrame, n_test: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    test = df.iloc[idx[:n_test]].reset_index(drop=True)
    train = df.iloc[idx[n_test:]].reset_index(drop=True)
    print(f"\n訓練 {len(train)} 行 / 評価 {len(test)} 行に分けた。")
    print(check_overlap(train, test, ignore=["物件ID", "車台番号"]))
    return train, test


def step2_screen(df: pd.DataFrame) -> None:
    hr("2. スクリーニング — このデータでテキストは効くか（LLM を呼ばない）")
    print("機能B はテキストが価格を左右するデータでしか効かない。")
    print("先に無料で判定しておくと、効かないデータに課金しなくて済む。\n")
    print(screen(df, target="価格_usd", text=TEXT, unit="USD",
                 numeric=NUM, categorical=CAT))


def step3_feature(train: pd.DataFrame, test: pd.DataFrame) -> None:
    hr("3. 機能A（Feature）— 非構造列を型のついた列にする")
    print("車種名（自由な文字列）から、車体の種類を表す列を作る。")
    print("中身は「埋め込み → 既存の値の名前の近傍で分類」で、")
    print("**確信度が低い行だけ LLM に回る**（キーが無ければ人手レビュー待ちに積む）。\n")

    f = Feature(source="車種名", type="category",
                values=["pickup truck", "sedan", "suv", "van", "coupe"],
                escalate_rate=0.15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnfoldWarning)
        train = train.copy()
        train["車体"] = f.fit_transform(train)

    print("状態:")
    for k, v in f.status().items():
        print(f"  {k}: {v}")
    print("\n割り当てられた値:")
    print(train["車体"].value_counts().to_string())
    q = f.review_queue()
    if len(q):
        print(f"\nレビュー待ち {len(q)} 件のうち先頭3件"
              "（確信できなかった行。承認すると次回の教師ラベルになる）:")
        cols = [c for c in ["テキスト", "LLMの答え", "推測"] if c in q.columns]
        print(q[cols].head(3).to_string(index=False))


def step4_plan(train: pd.DataFrame, test: pd.DataFrame, rate: float,
               client: ClaudeClient) -> AdaptivePredictor:
    hr("4. 見積もり — 何行を LLM に回すと、いくら・何秒か（LLM を呼ばない）")
    print("機能B は1行につき1回 LLM を呼ぶので、行数がそのまま費用と時間になる。")
    print("信頼度ルーティングは「統計モデルが迷っている行」だけを回す。\n")

    model = AdaptivePredictor(
        target="価格_usd", unit="USD", numeric=NUM, categorical=CAT, text=TEXT,
        n_examples=3, escalate_rate=rate, client=client)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnfoldWarning)
        model.fit(train)
    for k, v in model.plan(test).items():
        print(f"  {k}: {v}")
    return model


def step5_run(model: AdaptivePredictor, test: pd.DataFrame, n_rows: int) -> None:
    hr(f"5. 実行 — 先頭 {n_rows} 行を予測する")
    sub = test.iloc[:n_rows].reset_index(drop=True)
    truth = sub["価格_usd"].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnfoldWarning)
        pred = model.predict(sub, verbose=True)

    print("\n" + model.report())
    print(f"\n  この {len(sub)} 行の MAE: "
          f"{np.mean(np.abs(pred - truth)):,.2f} USD")

    prov = model.provenance()
    base_col = [c for c in prov.columns if c.startswith("証拠_")][0]
    base = prov[base_col].to_numpy(dtype=float)
    print(f"  1行も呼ばなかった場合（{base_col[3:]}だけ）: "
          f"{np.mean(np.abs(base - truth)):,.2f} USD")

    print("\n--- 行ごとの経路 ---")
    print(model.route().round(2).head(10).to_string(index=False))

    llm_rows = model.route().query("経路 == 'llm'")["行"].tolist()
    if llm_rows:
        print("\n--- explain()（LLM に回った行の1つめ）---")
        print(model.explain(int(llm_rows[0]))[:1500])

    print("\n--- cost() ---")
    for k, v in model.cost().items():
        print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="unfold を一通り動かす（既定では LLM を呼ばない）")
    ap.add_argument("--run", type=int, default=0, metavar="行数",
                    help="実際に予測する行数。このうち --rate の割合だけが "
                         "LLM に回る。0 なら1回も呼ばない（既定）")
    ap.add_argument("--rate", type=float, default=0.3,
                    help="LLM に回す割合（信頼度ルーティングの閾値）")
    args = ap.parse_args()

    if not SAMPLE.exists():
        raise SystemExit(f"抜粋データがありません: {SAMPLE}")

    print(f"データ: {SAMPLE.relative_to(ROOT)}")
    df = load()
    print(f"  {len(df)} 行 / 価格 中央値 {df['価格_usd'].median():,.0f} USD")

    df = step1_leak(df)
    train, test = split(df)
    step2_screen(df)
    step3_feature(train, test)

    client = ClaudeClient()
    model = step4_plan(train, test, args.rate, client)

    if args.run <= 0:
        hr("ここまで LLM を1回も呼んでいない（費用 $0）")
        print("実際に投げるには:")
        print("    .venv/bin/python -m unfold.demo --run 20")
        print("  20 行のうち 6 行が LLM に回る。1行あたり約 $0.0086 なので約 $0.05。")
        return
    if not client.available():
        hr("APIキーがありません")
        print(".env の ANTHROPIC_API_KEY を設定すると、ここから先が動く。")
        print("  確認: .venv/bin/python scripts/check_api_key.py")
        return
    step5_run(model, test, args.run)


if __name__ == "__main__":
    main()
