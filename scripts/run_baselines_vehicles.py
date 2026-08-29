"""ベースライン（複数車種）— シエンタで引いた線を Craigslist 20万件で引き直す。

docs/2026-08-29-baseline.md の宿題への回答。シエンタ（単一車種）では
「グレード名」を正規表現で抜けてしまったので、埋め込みや LLM を使う動機が出なかった。
複数車種では非構造テキストの表記が破綻するはずで、そこが unfold 機能A の本番になる。

対応関係:

    シエンタの「グレード名（116水準・タイトルから正規表現で抽出）」
      ↓
    Craigslist の「車種名 model（19,739水準・自由記述）」

はしご:

    0  中央値                     予測しない場合の下限
    A1 線形回帰・構造化列         従来手法の再現。車種名を使わない
    A2 LightGBM・構造化列         モデルを木に替えただけの効果
    B1 + 車種名そのまま           19,739水準をそのままカテゴリに突っ込む
    B2 + 車種名を手書きルールで正規化   シエンタでの「正規表現で抽出」に相当
    C1 + 車種名の単語TF-IDF       ルールを書かずに語で持つ
    C2 + 車種名の文字TF-IDF       同上・部分文字列で表記ゆれを吸収
    E  C2 + 説明文の単語TF-IDF    自由記述本体（平均2,972文字）も足した上限

実行:
    .venv/bin/python scripts/run_baselines_vehicles.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import (  # noqa: E402
    VEHICLES, cross_validate, load_dataset, show_leaderboard,
)
from features import make_lgbm, make_linear, predict_median  # noqa: E402

TARGET = VEHICLES.target

# 実験に使う行数。20万行を全部使うと TF-IDF・埋め込みまで含めた比較が
# 現実的な時間に収まらないので、seed 固定で間引く。
N_SAMPLE = 60_000

# 構造化列。中古車サイトが選択式で持っている項目。
NUM = ["車齢", "走行距離_mile"]
BOOL: list[str] = []
CAT = ["メーカー", "状態", "気筒数", "燃料", "名義状態",
       "変速機", "駆動", "サイズ", "ボディ", "色", "州"]

TEXT = "車種名"      # 非構造テキスト（本実験の主役）
DESC = "説明文"      # 自由記述本体

RULE_COL = "車種名_ルール正規化"


# --- 手書きルールによる車種名の正規化 ---------------------------------

# Craigslist の model 列は「車種 + グレード + 装備 + 宣伝文」が混ざっている。
#   ford  : 'f-150 raptor arizona raptor*rust free*icon level kit*tech pkg*...'
#   ram   : '2500 slt / quad cab / 4x4 / leather / 5.9 l high output / ...'
#   bmw   : 'x5 3.0i awd 126k miles 3.0l v6 pano roof heated leather'
# シエンタでやったのと同じ発想（正規表現で車種の芯だけ抜く）を、素直に書いてみる。
_SEP = re.compile(r"[*/|!,()\[\]]")           # ここから先は宣伝文とみなす
_NOISE = re.compile(r"[^a-z0-9\- ]+")


def normalize_model(s: str) -> str:
    """区切り記号以降を捨て、先頭2語だけを車種名とみなす。"""
    if not isinstance(s, str):
        return ""
    head = _SEP.split(s)[0]
    head = _NOISE.sub(" ", head.lower())
    return " ".join(head.split()[:2])


def main() -> None:
    df = load_dataset(dataset=VEHICLES, sample=N_SAMPLE)
    df[RULE_COL] = df[TEXT].map(normalize_model)

    print(f"  車種名: {df[TEXT].nunique():,} 種類 "
          f"→ 手書きルール正規化後 {df[RULE_COL].nunique():,} 種類")
    print(f"  メーカー {df['メーカー'].nunique()} 社")

    runs = [
        ("0  中央値", predict_median(TARGET), "予測しない場合の下限"),
        ("A1 線形回帰・構造化列",
         make_linear(TARGET, NUM, BOOL, CAT),
         "Ridge + one-hot。車種名を使わない従来手法の再現"),
        ("A2 LightGBM・構造化列",
         make_lgbm(TARGET, NUM, BOOL, CAT),
         "A1と同じ列。モデルを木に替えた効果"),
        ("B1 + 車種名そのまま",
         make_lgbm(TARGET, NUM, BOOL, CAT + [TEXT]),
         f"自由記述{df[TEXT].nunique():,}水準をそのままカテゴリ扱い"),
        ("B2 + 車種名を手書きルールで正規化",
         make_lgbm(TARGET, NUM, BOOL, CAT + [RULE_COL]),
         f"区切り記号以降を捨て先頭2語。{df[RULE_COL].nunique():,}水準。"
         f"シエンタの正規表現抽出に相当"),
        ("C1 + 車種名の単語TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "word"),
         "min_df=5, max_features=300。ルールを書かずに語で持つ"),
        ("C2 + 車種名の文字TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char"),
         "char_wb 2-4gram, max_features=1000。表記ゆれを部分文字列で吸収"),
        ("E  B2 + 説明文の単語TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT + [RULE_COL], DESC, "word"),
         "自由記述本体（平均2,972文字）を足した場合の上積み"),
    ]

    print()
    for name, fn, note in runs:
        print(f"[{name}]")
        t0 = time.time()
        cross_validate(name, fn, df, note=note, dataset=VEHICLES)
        print(f"  ({time.time() - t0:.0f} 秒)\n")

    print("=" * 78)
    print(f"リーダーボード（{VEHICLES.leaderboard.name}）")
    print("=" * 78)
    lb = show_leaderboard(VEHICLES)
    print(lb[["手法", "MAE", "MAE_std", "RMSE", "MAPE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
