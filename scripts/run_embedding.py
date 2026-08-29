"""埋め込みを特徴量にして精度を測る（主環境 .venv で実行）。

前提: 先に隔離環境で埋め込みを計算しておくこと。

    bash scripts/setup_embed_env.sh
    .venv-embed/bin/python scripts/embed_text.py
    .venv/bin/python scripts/run_embedding.py     # ← これ

ベースライン（run_baselines.py）で分かったのは次の2点だった。

  ・効いたのは グレード名（−3.91万円）。タイトル文から正規表現で抜いた列
  ・装備テキストを TF-IDF で入れても −0.67万円しか出ない

そこでここでは狙いの違う2つの問いを立てる。

  D1「埋め込みは TF-IDF に勝つか」
      装備テキストを埋め込みに替えて C2（MAE 12.21）と比べる。
      同じ入力・同じ下流モデルなので、表現方法だけの比較になる。

  D2「正規表現なしで、グレードの情報を取り出せるか」★本命
      グレード名の列を使わず、生のタイトル文の埋め込みだけを渡す。
      Ab1（正規表現でグレード名を抜いた場合 = MAE 13.60）に届けば、
      **人手のルールなしで同じ情報が取れた**ことになる。
      これは unfold 機能A が成り立つかどうかの核心にあたる。
      シエンタ単一車種では正規表現で足りてしまうが、複数車種に広げると
      表記が破綻して正規表現が書けなくなる。そこで効くかの前哨戦。

埋め込みは学習済みモデルの出力で、目的変数（価格）を一切見ていない。
したがって全データ分を先に計算しておいても交差検証のリークにはならない。
一方 PCA は訓練データの分布を使うので、fold の中で fit する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import (  # noqa: E402
    DATA, EXTRA_CAT, EXTRA_NUM, LEGACY_BOOL, LEGACY_CAT, LEGACY_NUM,
    ROOT, TARGET, cross_validate, load_dataset,
)
from features import LGBM_PARAMS, as_category as _as_category, numeric_frame as _numeric_frame  # noqa: E402

EMB_DIR = ROOT / "sampledata" / "processed"
EMB_FILES = {
    "equip": EMB_DIR / "usedsienta_emb_equipment_e5small.parquet",
    "title": EMB_DIR / "usedsienta_emb_title_e5small.parquet",
}


def attach_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """埋め込み列を df に横付けする。

    埋め込みは元の clean parquet（5,513行）の順で保存してあるのに対し、
    load_dataset() は価格欠損の6行を落として 5,507 行にしている。
    同じ手順で行を選び直して位置を揃える。
    """
    raw = pd.read_parquet(DATA, columns=[TARGET])
    keep = raw[TARGET].notna().to_numpy()

    out = df.copy()
    for tag, path in EMB_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} がありません。先に埋め込みを計算してください:\n"
                f"  .venv-embed/bin/python scripts/embed_text.py"
            )
        emb = pd.read_parquet(path).drop(columns=["行番号"])
        emb = emb[keep].reset_index(drop=True)
        if len(emb) != len(df):
            raise ValueError(f"{path.name}: 行数が合いません {len(emb)} != {len(df)}")
        emb.columns = [f"{tag}_{c}" for c in emb.columns]
        out = pd.concat([out, emb], axis=1)
    return out


def make_lgbm_emb(num, boo, cat, emb_tag: str | None = None,
                  pca_dim: int | None = None):
    """LightGBM に埋め込み列を足して学習する。

    emb_tag が None なら埋め込みなし（run_baselines.make_lgbm と同じ）。
    pca_dim を指定すると、fold 内の訓練データで PCA を学習して次元を落とす。
    """
    def fit_predict(train, test):
        Xtr = _numeric_frame(train, num, boo)
        Xte = _numeric_frame(test, num, boo)
        if cat:
            Ctr, Cte = _as_category(train, test, cat)
            Xtr = pd.concat([Xtr, Ctr], axis=1)
            Xte = pd.concat([Xte, Cte], axis=1)

        if emb_tag:
            cols = [c for c in train.columns if c.startswith(f"{emb_tag}_emb_")]
            Etr = train[cols].to_numpy(dtype="float32")
            Ete = test[cols].to_numpy(dtype="float32")
            if pca_dim:
                # PCA は訓練データの分布を使うので必ず fold の中で fit する
                pca = PCA(n_components=pca_dim, random_state=42)
                Etr = pca.fit_transform(Etr)
                Ete = pca.transform(Ete)
                names = [f"{emb_tag}_pc{i}" for i in range(Etr.shape[1])]
            else:
                names = cols
            Xtr = pd.concat([Xtr, pd.DataFrame(Etr, columns=names, index=Xtr.index)], axis=1)
            Xte = pd.concat([Xte, pd.DataFrame(Ete, columns=names, index=Xte.index)], axis=1)

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(Xtr, train[TARGET], categorical_feature=cat or "auto")
        return model.predict(Xte)
    return fit_predict


def show_neighbours(df: pd.DataFrame, n_show: int = 3, k: int = 3) -> None:
    """タイトル埋め込みで「意味の近い車」を引いてみる（定性確認）。

    unfold 機能A の 03「近い正解例を探す」に相当する動き。
    数字ではなく、埋め込みが何を近いと見なしているかを目で見るためのもの。
    """
    cols = [c for c in df.columns if c.startswith("title_emb_")]
    V = df[cols].to_numpy(dtype="float32")   # 長さ1に正規化済み
    sim = V @ V.T
    np.fill_diagonal(sim, -1)

    print("\n" + "=" * 78)
    print("タイトル埋め込みで最も近い車を引く（正規表現を一切使わない検索）")
    print("=" * 78)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(df), n_show, replace=False):
        r = df.iloc[i]
        print(f"\n■ 基準: {r['グレード名']:<12} {r[TARGET]:6.1f}万円  "
              f"{r['車齢']}年  {r['走行距離_km']:,.0f}km")
        print(f"   {r['タイトル'][:64]}")
        for j in np.argsort(-sim[i])[:k]:
            s = df.iloc[j]
            print(f"   └ 類似{sim[i, j]:.3f}  {s['グレード名']:<12} "
                  f"{s[TARGET]:6.1f}万円  {s['タイトル'][:44]}")


def main() -> None:
    df = attach_embeddings(load_dataset())
    print(f"埋め込みを横付け: {df.shape[1]} 列（うち埋め込み 768 列）\n")

    FULL_NUM = LEGACY_NUM + EXTRA_NUM
    FULL_CAT = LEGACY_CAT + EXTRA_CAT

    runs = [
        # D1: 装備テキストを TF-IDF ではなく埋め込みで入れる。C2(12.21) と比較
        ("D1 B+装備テキスト埋め込み",
         make_lgbm_emb(FULL_NUM, LEGACY_BOOL, FULL_CAT, emb_tag="equip"),
         "C2(TF-IDF 12.21)との比較。同じ入力・表現方法だけ違う"),

        # D2: グレード名を使わず、生タイトルの埋め込みだけ。Ab1(13.60) と比較 ★本命
        ("D2 従来列+タイトル埋め込み(グレード名なし)",
         make_lgbm_emb(LEGACY_NUM, LEGACY_BOOL, LEGACY_CAT, emb_tag="title"),
         "正規表現なしでグレード情報を取れるか。Ab1(+グレード名 13.60)が目標"),

        # D2b: 384次元は 4,400 行に対して多すぎる可能性。圧縮すると変わるか
        ("D2b D2のPCA64次元版",
         make_lgbm_emb(LEGACY_NUM, LEGACY_BOOL, LEGACY_CAT,
                       emb_tag="title", pca_dim=64),
         "埋め込みを64次元に圧縮。次元数が効いているかの確認"),

        # D3: 全部乗せ。現時点の上限を見る
        ("D3 構造化列フル+タイトル埋め込み",
         make_lgbm_emb(FULL_NUM, LEGACY_BOOL, FULL_CAT, emb_tag="title"),
         "グレード名も埋め込みも両方入れた全部乗せ"),
    ]

    for name, fn, note in runs:
        print(f"[{name}]")
        cross_validate(name, fn, df, note=note)
        print()

    show_neighbours(df)


if __name__ == "__main__":
    main()
