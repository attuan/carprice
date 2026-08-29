"""Kaggle Craigslist 中古車データ（複数車種）のクレンジング。

シエンタ（単一車種）で引いたベースラインを、**複数車種**に広げて再検証するための
学習用データを作る。狙いは docs/2026-08-29-baseline.md の宿題:

    「シエンタ単一車種ではグレード抽出が正規表現で足りてしまう。
      複数車種に広げると表記が破綻するので、そこが LLM／埋め込みの出番になる」

vehicles.csv の `model` 列（29,668 種類の自由記述）が、シエンタの `グレード名` に
あたる非構造列。ここを「そのままカテゴリ」「手書きルール」「TF-IDF」「埋め込み」で
処理し分けて比較するのが目的。

出力: sampledata/processed/vehicles_multi_clean.parquet

再現:
    .venv/bin/python scripts/clean_vehicles.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sampledata" / "raw" / "vehicles.csv"
OUT = ROOT / "sampledata" / "processed" / "vehicles_multi_clean.parquet"
# 重複を残した版。「重複排除をサボると精度がどれだけ水増しされるか」を
# 測るためだけに使う（scripts/check_duplicate_leak.py）。
OUT_DUP = ROOT / "sampledata" / "processed" / "vehicles_multi_withdup.parquet"

# 掲載期間は 2021-04〜05 なので、車齢はこの年を基準にする。
BASE_YEAR = 2021

# 除外条件とその理由:
#   price 1,000〜100,000 USD … 中央値 13,950 に対し最大 37億。0円や桁違いの
#                              入力ミスを落とす。CLAUDE.md の既知問題への対応
#   year  1990〜2022         … 1900年代の入力ミスとクラシックカーを落とす
#   odometer 100〜400,000    … 最大 1,000万マイルという明らかな異常値がある
#   model / manufacturer     … 本実験の主役の列なので欠損は使えない
FILTERS = """
    price BETWEEN 1000 AND 100000
    AND year BETWEEN 1990 AND 2022
    AND odometer BETWEEN 100 AND 400000
    AND model IS NOT NULL AND manufacturer IS NOT NULL
"""


def main(dedup: bool = True) -> None:
    if not SRC.exists():
        raise SystemExit(
            f"{SRC} がありません。先に `python3 scripts/download_data.py` を実行してください。"
        )

    src = f"read_csv_auto('{SRC}', ignore_errors=true)"
    con = duckdb.connect()

    n_all = con.sql(f"SELECT count(*) FROM {src}").fetchone()[0]

    # --- 1. 行のフィルタ -------------------------------------------------
    con.sql(f"CREATE TABLE filtered AS SELECT * FROM {src} WHERE {FILTERS}")
    n_filtered = con.sql("SELECT count(*) FROM filtered").fetchone()[0]

    # --- 2. 重複排除 ------------------------------------------------------
    # 同じ車が複数の region に出稿されている（同一 VIN が最大 261 件、価格は同一）。
    # ランダム分割の CV では同じ車が train と test の両方に入りリークするので、
    # ここで必ず1台1行にする。VIN が無い行は内容の一致で潰す。
    con.sql(
        """
        CREATE TABLE dedup AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY coalesce(
                    VIN,
                    concat_ws('|', manufacturer, model, year, odometer, price,
                              coalesce(description, ''))
                )
                ORDER BY id
            ) AS rn
            FROM filtered
        ) WHERE rn = 1
        """
    )
    n_dedup = con.sql("SELECT count(*) FROM dedup").fetchone()[0]

    # 重複を残した版を作るときは、この段階の表を差し替えるだけ
    if not dedup:
        con.sql("DROP TABLE dedup")
        con.sql("CREATE TABLE dedup AS SELECT * FROM filtered")

    # --- 3. 列の整形 ------------------------------------------------------
    # 文字列は小文字・空白正規化だけ。model の表記ゆれをここで潰すと
    # 「非構造テキストをどう扱うか」という実験の問い自体が消えてしまう。
    con.sql(
        f"""
        CREATE TABLE clean AS
        SELECT
            id                                          AS 物件ID,
            price                                       AS 価格_usd,
            {BASE_YEAR} - year                          AS 車齢,
            year                                        AS 年式,
            odometer                                    AS 走行距離_mile,
            lower(trim(manufacturer))                   AS メーカー,
            regexp_replace(lower(trim(model)), '\\s+', ' ', 'g') AS 車種名,
            condition                                   AS 状態,
            cylinders                                   AS 気筒数,
            fuel                                        AS 燃料,
            title_status                                AS 名義状態,
            transmission                                AS 変速機,
            drive                                       AS 駆動,
            size                                        AS サイズ,
            type                                        AS ボディ,
            paint_color                                 AS 色,
            state                                       AS 州,
            region                                      AS 地域,
            length(description)                         AS 説明文字数,
            description                                 AS 説明文,
            VIN                                         AS 車台番号
        FROM dedup
        """
    )

    out = OUT if dedup else OUT_DUP
    out.parent.mkdir(parents=True, exist_ok=True)
    con.sql(f"COPY clean TO '{out}' (FORMAT PARQUET)")

    # --- 4. 要約 ----------------------------------------------------------
    print(f"元データ            {n_all:,} 行")
    print(f"フィルタ後          {n_filtered:,} 行  ({n_all - n_filtered:,} 行を除外)")
    print(f"重複排除後          {n_dedup:,} 行  ({n_filtered - n_dedup:,} 行が重複出稿)")
    if not dedup:
        print("※ --no-dedup 指定のため重複はそのまま残している")
    print(f"→ {out.relative_to(ROOT)}")
    print()
    print(con.sql(
        """
        SELECT count(*) AS 行数,
               count(DISTINCT メーカー) AS メーカー数,
               count(DISTINCT 車種名) AS 車種名の種類,
               round(median(価格_usd)) AS 価格中央値,
               round(avg(価格_usd)) AS 価格平均,
               round(median(走行距離_mile)) AS 走行距離中央値,
               round(median(車齢), 1) AS 車齢中央値
        FROM clean
        """
    ).df().T.to_string())
    print()
    print("欠損率（%）")
    cols = [r[0] for r in con.sql("DESCRIBE clean").fetchall()]
    miss = con.sql(
        "SELECT " + ", ".join(
            f'round(100.0 * sum(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) / count(*), 1) AS "{c}"'
            for c in cols
        ) + " FROM clean"
    ).df().T
    print(miss.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-dedup", action="store_true",
                    help="重複出稿を残したまま出力する（リークの実害を測る用）")
    main(dedup=not ap.parse_args().no_dedup)
