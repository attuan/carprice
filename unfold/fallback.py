"""confidence が低い行の逃がし先（設計書 05「Uncertain? Escalate to the LLM.」）。

**ここが唯一 LLM を呼ぶ場所**であり、2026-08-29 時点ではまだ呼べない
（APIキーの入手経路が未確定。PRD §7-9）。そこで差し込み口だけ定義し、
既定は「答えずにレビュー待ちとして積むだけ」の `QueueOnlyFallback` にしてある。

これは手抜きではなく、設計書の能動学習ループの片側そのものでもある。
仕様書には "Everything the fallback answers is queued as a labeling candidate.
Accept it and it joins the ground truth" とあり、**LLM の答えも人の承認も
同じキューを通る**。キューを先に作っておけば、LLM 実装は後から差し込める。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Answer:
    """フォールバックが返す1行ぶんの答え。"""
    value: str | None
    confidence: float
    cost: float = 0.0
    origin: str = "llm"


@runtime_checkable
class LLMFallback(Protocol):
    """LLM 実装が満たすべき形。APIキーが決まったらこれを実装して差し込む。"""

    #: 1行あたりの推定費用（PRD S7 の「1行あたりの単価」に対応）
    cost_per_call: float

    def can_answer(self) -> bool:
        """いま実際に呼べるか（キー未設定なら False）。"""

    def answer(self, texts: list[str], values: list[str] | None,
               context: list[dict]) -> list[Answer]:
        """texts の各行に対して値を返す。context には近傍事例が入る。"""


@dataclass
class QueueOnlyFallback:
    """LLM を呼ばず、レビュー待ちのキューに積むだけの既定実装。

    APIキーが無い状態でも機能A の 01〜04（埋め込み → 近傍分類 → 確信度判定）を
    最後まで動かし、**05 に落ちた行が何%あるか**を測れるようにするためのもの。
    その割合がそのまま「LLM を呼ぶ必要がある行の割合」＝費用の見積もりになる。
    """

    cost_per_call: float = 0.0
    queued: list[dict] = field(default_factory=list)

    def can_answer(self) -> bool:
        return False

    def answer(self, texts: list[str], values: list[str] | None,
               context: list[dict]) -> list[Answer]:
        raise NotImplementedError(
            "QueueOnlyFallback は答えません。LLM を使うには LLMFallback を"
            "実装して Feature(fallback=...) に渡してください。")

    def enqueue(self, row: int, text: str, guess: str | None,
                confidence: float) -> None:
        self.queued.append({"行番号": row, "テキスト": text,
                            "分類器の推測": guess, "confidence": confidence})
