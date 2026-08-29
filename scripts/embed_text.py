"""シエンタの非構造テキストを埋め込みベクトルに変換する。

**このスクリプトは隔離環境 .venv-embed で動かす。**（主環境には torch を入れない）

    bash scripts/setup_embed_env.sh                # 初回だけ
    .venv-embed/bin/python scripts/embed_text.py

埋め込み（embedding）とは、文章を数百個の数値の並び（ベクトル）に変換し、
意味の近さを距離で測れるようにする技術。TF-IDF は「同じ単語が入っているか」しか
見ないが、埋め込みなら「バックカメラ」と「リアカメラ」が別の単語でも近いと分かる。
unfold 機能A（Feature）はこの埋め込みを土台にしている。

2つの列を埋め込む。狙いが違うので両方作る。

  装備テキスト … TF-IDF（ベースライン C）と同じ入力。埋め込みが TF-IDF に勝つかを見る
  タイトル     … グレード名を含む生の文章。**正規表現を使わずにグレードの情報を
                  取り出せるか**を見る。unfold 機能A の核心にあたる問い

出力は主環境（numpy 2.x）でも読める parquet にする。torch のオブジェクトは一切渡さない。
行の順番は入力の parquet と完全に同じで、対応付けは位置（行番号）で行う。
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sampledata" / "processed" / "usedsienta_clean.parquet"
OUT_DIR = ROOT / "sampledata" / "processed"

# multilingual-e5-small: 日本語を含む100言語対応、384次元、118Mパラメータ。
# CPU だけで数千件を数分で回せる大きさなので、Intel Mac でも実用になる。
MODEL_NAME = "intfloat/multilingual-e5-small"
DIM_TAG = "e5small"

# e5 系のモデルは入力に接頭辞を付けて学習されている。付けないと精度が落ちる。
# 文書同士を比べる用途なので "query: " を使う。
PREFIX = "query: "

TARGETS = {
    "equipment": "装備テキスト",
    "title": "タイトル",
}


def main() -> None:
    df = pd.read_parquet(SRC)
    print(f"入力: {SRC.name}  {len(df):,} 行")

    print(f"モデル読み込み: {MODEL_NAME}（初回はダウンロードに数分かかります）")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 128  # 装備の羅列は長くないので短く切って高速化

    for tag, col in TARGETS.items():
        texts = (PREFIX + df[col].fillna("")).tolist()
        n_empty = int(df[col].isna().sum())
        print(f"\n[{col}] {len(texts):,} 件を埋め込み中"
              f"（うち欠損 {n_empty} 件は空文字として扱う）")

        t0 = time.time()
        vecs = model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,  # 長さ1に揃える。後で内積＝コサイン類似度になる
            show_progress_bar=True,
        )
        print(f"  {time.time() - t0:.1f} 秒 / 形 {vecs.shape}")

        out = pd.DataFrame(vecs, columns=[f"emb_{i}" for i in range(vecs.shape[1])])
        # 行の対応は位置で取る。物件ID は 1,890 件欠損しているのでキーに使えない
        out.insert(0, "行番号", range(len(out)))
        dst = OUT_DIR / f"usedsienta_emb_{tag}_{DIM_TAG}.parquet"
        out.to_parquet(dst, index=False)
        print(f"  保存: {dst.relative_to(ROOT)}"
              f"（{dst.stat().st_size / 1024**2:.1f} MB）")


if __name__ == "__main__":
    main()
