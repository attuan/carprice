"""ベースライン — unfold（機能A/機能B）が超えるべき線を引く。

段階的に情報を足していく「はしご」になっていて、隣り合う行の差が
そのまま「何を足したら何万円縮んだか」になる。

    0  中央値                    予測しない場合の下限
    A1 線形回帰・従来列          スクレイピング当時の回帰分析の再現＝出発点
    A2 LightGBM・従来列          モデルを木に替えただけの効果
    B  LightGBM・構造化列フル    クレンジングで増えた列（グレード等）の効果
    C1 B + 装備テキスト(単語)    LLMも埋め込みも使わずテキストを入れた効果
    C2 B + 装備テキスト(文字)    同上・部分文字列で表記ゆれを吸収した版

**C が本命の比較対象。** unfold の機能A は「非構造テキストを埋め込みと
LLM で特徴量にする」機構だが、TF-IDF で同じだけ縮むならその機構は要らない。
機能A の価値は C からの上積みで測る。

実行:
    .venv/bin/python scripts/run_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import (  # noqa: E402
    EXTRA_CAT, EXTRA_NUM, LEGACY_BOOL, LEGACY_CAT, LEGACY_NUM,
    TARGET, TEXT_COL, cross_validate, load_dataset, show_leaderboard,
)
from features import make_lgbm, make_linear, predict_median  # noqa: E402


def main() -> None:
    df = load_dataset()

    legacy_num, legacy_boo, legacy_cat = LEGACY_NUM, LEGACY_BOOL, LEGACY_CAT
    full_num = LEGACY_NUM + EXTRA_NUM
    full_cat = LEGACY_CAT + EXTRA_CAT

    runs = [
        ("0  中央値", predict_median(TARGET), "予測しない場合の下限"),
        ("A1 線形回帰・従来列",
         make_linear(TARGET, legacy_num, legacy_boo, legacy_cat),
         "Ridge + one-hot。スクレイピング当時の回帰分析の再現"),
        ("A2 LightGBM・従来列",
         make_lgbm(TARGET, legacy_num, legacy_boo, legacy_cat),
         "A1と同じ列。モデルを木に替えた効果"),
        ("B  LightGBM・構造化列フル",
         make_lgbm(TARGET, full_num, legacy_boo, full_cat),
         "+グレード名・色・車検残月数・装備数"),
        ("C1 B+装備テキスト(単語TF-IDF)",
         make_lgbm(TARGET, full_num, legacy_boo, full_cat, TEXT_COL, "word"),
         "min_df=5, max_features=300。LLMなしでテキストを入れた到達点"),
        ("C2 B+装備テキスト(文字TF-IDF)",
         make_lgbm(TARGET, full_num, legacy_boo, full_cat, TEXT_COL, "char"),
         "char_wb 2-4gram, max_features=1000。表記ゆれを部分文字列で吸収"),
        ("C1' C1のlog(価格)版",
         make_lgbm(TARGET, full_num, legacy_boo, full_cat, TEXT_COL, "word", log_target=True),
         "目的変数を対数化。右に裾を引く分布への対応"),
    ]

    print()
    for name, fn, note in runs:
        print(f"[{name}]")
        cross_validate(name, fn, df, note=note)
        print()

    print("=" * 78)
    print("リーダーボード（results/leaderboard.csv）")
    print("=" * 78)
    lb = show_leaderboard()
    print(lb[["手法", "MAE", "MAE_std", "RMSE", "MAPE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
