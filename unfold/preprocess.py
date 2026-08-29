"""埋め込みに入れる前のテキスト整形。

**既定で定数語の除去を行う。**（`scripts/text_variants.py` の V2 と同じ規則）
2026-08-29 の測定で、コーパス内の出現率が閾値以上のトークンを落とすと

  ・近傍5件による価格予測の MAE が 31.45 → 28.92 万円
  ・教師あり学習（LightGBM に埋め込みを渡す構成）の MAE が 13.68 → 13.29 万円
  ・埋め込みの計算時間も短くなる

と、測った3つすべてで悪化しなかった（→ `docs/2026-08-29-denoise.md`）。
人がストップワードのリストを書くのではなく**データから決まる**ので、
「ルールを書かない」という機能A の趣旨にも合う。

一方、同じ測定で「最初の区切り記号までで切る」規則は下流精度を悪化させたので
既定には入れない。削る量は多いほど良いわけではない。
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# 装備の羅列を区切っている文字。全角スペース・スラッシュ・中黒など
SEP = r"[\s/・,、|｜]+"
IDEO_SPACE = "　"


def tokenize(text: str) -> list[str]:
    """区切り文字で分割する。形態素解析はしない（言語非依存にするため）。"""
    return [t for t in re.split(SEP, text.replace(IDEO_SPACE, " ")) if t]


def constant_tokens(texts: pd.Series, threshold: float = 0.9) -> list[str]:
    """出現率が threshold 以上のトークン（＝行を区別しない語）を返す。

    「シエンタ」のように全行に出る語は、どの行がどの行に似ているかを
    まったく決めないのに、ベクトルの向きを揃えてしまう。
    """
    n = max(len(texts), 1)
    df = Counter()
    for s in texts.fillna(""):
        df.update(set(tokenize(s)))
    return sorted(w for w, k in df.items() if k / n >= threshold)


def drop_constant_tokens(texts: pd.Series, stop: list[str] | None = None,
                         threshold: float = 0.9) -> tuple[pd.Series, list[str]]:
    """定数語を落としたテキストと、落とした語の一覧を返す。

    stop を渡すとその一覧を使う（**学習時に決めた語を推論時に使い回すため**。
    推論データで数え直すと、行数が少ないときに別の語が消えて表現がずれる）。
    """
    if stop is None:
        stop = constant_tokens(texts, threshold)
    stopset = set(stop)
    out = texts.fillna("").map(
        lambda s: " ".join(t for t in tokenize(s) if t not in stopset))
    return out, list(stop)
