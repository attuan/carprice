"""複数車種（Craigslist）の結果の作図。results/vehicles_*.png を吐く。

leaderboard を読むだけなので再学習はしない。
    .venv/bin/python scripts/plot_vehicles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import ROOT, SIENTA, VEHICLES  # noqa: E402

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

OUT = ROOT / "results"
RULE = "#4a7fb0"    # 人手のルール
EMB = "#2e6b4f"     # 埋め込み
BOTH = "#1f4d5a"    # 埋め込み + 文字TF-IDF の併用
PLAIN = "#9aa0a6"


def _last(path: Path) -> pd.DataFrame:
    lb = pd.read_csv(path, encoding="utf-8-sig")
    return lb.drop_duplicates("手法", keep="last").set_index("手法")


def ladder(lb: pd.DataFrame) -> None:
    items = [
        ("A1 線形回帰\n構造化列", "A1 線形回帰・構造化列", PLAIN),
        ("A2 LightGBM\n構造化列", "A2 LightGBM・構造化列", PLAIN),
        ("B1 +車種名\nそのまま", "B1 + 車種名そのまま", RULE),
        ("B2 +車種名\n手書きルール", "B2 + 車種名を手書きルールで正規化", RULE),
        ("C1 +車種名\n単語TF-IDF", "C1 + 車種名の単語TF-IDF", RULE),
        ("C2 +車種名\n文字TF-IDF", "C2 + 車種名の文字TF-IDF", RULE),
        ("D1 +車種名\n埋め込み", "D1 構造化列+車種名の埋め込み", EMB),
        ("D4 埋め込み\n+文字TF-IDF", "D4 埋め込み+文字TF-IDF", BOTH),
    ]
    labels = [i[0] for i in items]
    vals = np.array([lb.loc[i[1], "MAE"] for i in items])
    errs = [lb.loc[i[1], "MAE_std"] for i in items]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.bar(labels, vals, yerr=errs, capsize=4,
                  color=[i[2] for i in items])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,.0f}",
                ha="center", fontsize=10, fontweight="bold")
    for i in range(1, len(vals)):
        ax.annotate(f"−{vals[i-1]-vals[i]:,.0f}",
                    xy=(i - 0.5, (vals[i - 1] + vals[i]) / 2 - 260),
                    ha="center", fontsize=9, color="#b03030")

    ax.set_ylabel("MAE（USD・5-fold CV）")
    ax.set_title("複数車種のはしご — Craigslist 60,000件 / 価格中央値 $11,995"
                 "（緑系＝埋め込みを使った構成）", fontsize=12)
    ax.set_ylim(0, vals.max() * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "vehicles_ladder.png", dpi=150)
    print("保存:", OUT / "vehicles_ladder.png")


def unseen(path: Path) -> None:
    """訓練データに無かった車種名の行だけを取り出した比較。"""
    d = pd.read_csv(path, encoding="utf-8-sig")
    labels = ["B2 手書きルール", "C2 文字TF-IDF", "D1 埋め込み", "D4 併用"]
    colors = [RULE, RULE, EMB, BOTH]
    x = np.arange(len(d))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    b1 = ax.bar(x - w / 2, d["既知MAE"], w, color="#c9d6e3", label="既知の車種名")
    b2 = ax.bar(x + w / 2, d["未知MAE"], w, color=colors, label="未知の車種名")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 60,
                    f"{b.get_height():,.0f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE（USD）")
    ax.set_title("訓練データに無い車種名が来たとき — 手法別の MAE", fontsize=12)
    ax.legend()
    ax.set_ylim(0, d["未知MAE"].max() * 1.22)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "vehicles_unseen.png", dpi=150)
    print("保存:", OUT / "vehicles_unseen.png")


def cross_dataset(v: pd.DataFrame, s: pd.DataFrame) -> None:
    """単一車種と複数車種で「人手のルール」と「埋め込み」の効き方を比べる。

    価格の単位が違う（万円 / USD）ので、絶対値ではなく
    「構造化列だけの LightGBM から何%縮んだか」で揃える。
    """
    sets = [
        ("シエンタ 5,507件\n（単一車種）",
         s.loc["A2 LightGBM・従来列", "MAE"],
         s.loc["Ab1 +グレード名", "MAE"],
         s.loc["D2 従来列+タイトル埋め込み(グレード名なし)", "MAE"]),
        ("Craigslist 60,000件\n（複数車種）",
         v.loc["A2 LightGBM・構造化列", "MAE"],
         v.loc["B2 + 車種名を手書きルールで正規化", "MAE"],
         v.loc["D1 構造化列+車種名の埋め込み", "MAE"]),
    ]
    x = np.arange(len(sets))
    w = 0.35
    rule = [(a - r) / a * 100 for _, a, r, _ in sets]
    emb = [(a - e) / a * 100 for _, a, _, e in sets]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = ax.bar(x - w / 2, rule, w, color=RULE, label="人手のルール（正規表現）")
    b2 = ax.bar(x + w / 2, emb, w, color=EMB, label="埋め込み（ルールなし）")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                    f"{b.get_height():.1f}%", ha="center", fontsize=10,
                    fontweight="bold")
    ax.set_xticks(x, [s[0] for s in sets])
    ax.set_ylabel("構造化列のみ（A2）からの MAE 改善率")
    ax.set_title("車種を増やすと、人手のルールだけが効かなくなる", fontsize=12)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(rule + emb) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "vehicles_vs_sienta.png", dpi=150)
    print("保存:", OUT / "vehicles_vs_sienta.png")


def main() -> None:
    v = _last(VEHICLES.leaderboard)
    ladder(v)
    breakdown = OUT / "vehicles_unseen_breakdown.csv"
    if breakdown.exists():
        unseen(breakdown)
    cross_dataset(v, _last(SIENTA.leaderboard))


if __name__ == "__main__":
    main()
