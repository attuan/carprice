"""unfold — 非構造データのための機械学習ライブラリ（機能A の骨組み）。

伊藤さんの設計書（dialogs/unfold-landing.html）のうち、**機能A（Feature）**を
LLM 呼び出しなしで動くところまで実装したもの。機能B（LLMPredictor）と
信頼度ルーティング（AdaptivePredictor）は差し込み口だけ用意してある。

    from unfold import Feature

    df["グレード"] = Feature(
        source="タイトル",
        type="category",
        values=["G", "Z", "X", "G クエロ"],
    ).fit_transform(df)

設計書どおり scikit-learn 互換（`fit` / `transform` / `fit_transform`）で、
そこに `explain` / `confidence` / `examples` / `cost` が乗る。

**なぜ LLM を呼ばずに動くのか。** 機能A の中身は「埋め込み → 既存の教師ラベルの
近傍で分類」であって、LLM は confidence が閾値を下回った行のフォールバックにしか
出てこない（設計書の 01〜05）。したがって 01〜04 は APIキーなしで実装・測定できる。
05 は `LLMFallback` という差し込み口にしてあり、既定の `QueueOnlyFallback` は
**答えずにレビュー待ちのキューに積むだけ**。APIキーが決まったら実装を差し替える。
"""

from unfold.encoders import (
    CharTfidfEncoder,
    Encoder,
    PrecomputedEncoder,
    SentenceTransformerEncoder,
)
from unfold.errors import UnfoldError
from unfold.fallback import LLMFallback, QueueOnlyFallback
from unfold.feature import Feature
from unfold.preprocess import drop_constant_tokens

__all__ = [
    "Feature",
    "Encoder",
    "CharTfidfEncoder",
    "SentenceTransformerEncoder",
    "PrecomputedEncoder",
    "LLMFallback",
    "QueueOnlyFallback",
    "UnfoldError",
    "drop_constant_tokens",
]

__version__ = "0.0.1"
