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

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import (  # noqa: E402
    EXTRA_CAT, EXTRA_NUM, LEGACY_BOOL, LEGACY_CAT, LEGACY_NUM,
    TARGET, TEXT_COL, cross_validate, load_dataset, show_leaderboard,
)

SEED = 42

LGBM_PARAMS = dict(
    n_estimators=700,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=SEED,
    verbose=-1,
)


# --- 特徴量づくり -----------------------------------------------------

def _numeric_frame(df: pd.DataFrame, num: list[str], boo: list[str]) -> pd.DataFrame:
    """数値列と bool 列を取り出す。bool は 0/1 にするだけ。"""
    out = df[num].astype("float64").copy()
    for c in boo:
        out[c] = df[c].astype("float64")
    return out


def _as_category(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """train に出た水準だけをカテゴリとして固定する。

    test にしか無い水準（例: train に無かったグレード）は NaN になり、
    LightGBM は欠損として扱う。train を見て決めるのが原則。
    """
    tr, te = train[cols].copy(), test[cols].copy()
    for c in cols:
        cats = pd.Index(tr[c].dropna().unique())
        tr[c] = pd.Categorical(tr[c], categories=cats)
        # 未知の水準は先に NaN に落としてから Categorical にする
        te[c] = pd.Categorical(te[c].where(te[c].isin(cats)), categories=cats)
    return tr, te


def _tfidf(train: pd.DataFrame, test: pd.DataFrame, **kw):
    """装備テキストを TF-IDF に変換する。fit は train のみ。

    欠損（518件）は空文字にする。装備の記載が無いこと自体が
    「情報を出していない店」を意味するので、行は落とさない。
    """
    tr_txt = train[TEXT_COL].fillna("").to_numpy()
    te_txt = test[TEXT_COL].fillna("").to_numpy()
    vec = TfidfVectorizer(**kw)
    A = vec.fit_transform(tr_txt).toarray()
    B = vec.transform(te_txt).toarray()
    names = [f"tfidf_{i}" for i in range(A.shape[1])]
    return (pd.DataFrame(A, columns=names, index=train.index),
            pd.DataFrame(B, columns=names, index=test.index))


# --- 各ベースライン ---------------------------------------------------

def predict_median(train, test):
    """0. 何も見ずに train の中央値を返す。これを下回るモデルは無意味。"""
    return np.full(len(test), train[TARGET].median())


def make_linear(num, boo, cat):
    """A1. 線形回帰 + one-hot ダミー = 従来手法の再現。

    Ridge にしているのは、都道府県47水準の one-hot で最小二乗が
    不安定になるのを防ぐため。正則化以外は素の線形回帰と同じ。
    """
    def fit_predict(train, test):
        Xtr_n = _numeric_frame(train, num, boo)
        Xte_n = _numeric_frame(test, num, boo)
        # 線形回帰は欠損を扱えないので中央値で埋める（train の中央値）
        med = Xtr_n.median()
        Xtr_n, Xte_n = Xtr_n.fillna(med), Xte_n.fillna(med)

        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xtr_c = enc.fit_transform(train[cat].fillna("欠損").astype(str))
        Xte_c = enc.transform(test[cat].fillna("欠損").astype(str))

        Xtr = np.hstack([Xtr_n.to_numpy(), Xtr_c])
        Xte = np.hstack([Xte_n.to_numpy(), Xte_c])

        model = Ridge(alpha=1.0)
        model.fit(Xtr, train[TARGET])
        return model.predict(Xte)
    return fit_predict


def make_lgbm(num, boo, cat, text: str | None = None, log_target: bool = False):
    """A2 / B / C. LightGBM。text を渡すと TF-IDF 列を足す。"""
    tfidf_kw = {
        "word": dict(analyzer="word", token_pattern=r"\S+",
                     min_df=5, max_features=300),
        "char": dict(analyzer="char_wb", ngram_range=(2, 4),
                     min_df=10, max_features=1000),
    }

    def fit_predict(train, test):
        Xtr = _numeric_frame(train, num, boo)
        Xte = _numeric_frame(test, num, boo)
        if cat:
            Ctr, Cte = _as_category(train, test, cat)
            Xtr = pd.concat([Xtr, Ctr], axis=1)
            Xte = pd.concat([Xte, Cte], axis=1)
        if text:
            Ttr, Tte = _tfidf(train, test, **tfidf_kw[text])
            Xtr = pd.concat([Xtr, Ttr], axis=1)
            Xte = pd.concat([Xte, Tte], axis=1)

        y = train[TARGET].to_numpy(dtype=float)
        if log_target:
            y = np.log(y)

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(Xtr, y, categorical_feature=cat or "auto")
        pred = model.predict(Xte)
        return np.exp(pred) if log_target else pred
    return fit_predict


def main() -> None:
    df = load_dataset()

    legacy_num, legacy_boo, legacy_cat = LEGACY_NUM, LEGACY_BOOL, LEGACY_CAT
    full_num = LEGACY_NUM + EXTRA_NUM
    full_cat = LEGACY_CAT + EXTRA_CAT

    runs = [
        ("0  中央値", predict_median, "予測しない場合の下限"),
        ("A1 線形回帰・従来列",
         make_linear(legacy_num, legacy_boo, legacy_cat),
         "Ridge + one-hot。スクレイピング当時の回帰分析の再現"),
        ("A2 LightGBM・従来列",
         make_lgbm(legacy_num, legacy_boo, legacy_cat),
         "A1と同じ列。モデルを木に替えた効果"),
        ("B  LightGBM・構造化列フル",
         make_lgbm(full_num, legacy_boo, full_cat),
         "+グレード名・色・車検残月数・装備数"),
        ("C1 B+装備テキスト(単語TF-IDF)",
         make_lgbm(full_num, legacy_boo, full_cat, text="word"),
         "min_df=5, max_features=300。LLMなしでテキストを入れた到達点"),
        ("C2 B+装備テキスト(文字TF-IDF)",
         make_lgbm(full_num, legacy_boo, full_cat, text="char"),
         "char_wb 2-4gram, max_features=1000。表記ゆれを部分文字列で吸収"),
        ("C1' C1のlog(価格)版",
         make_lgbm(full_num, legacy_boo, full_cat, text="word", log_target=True),
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
