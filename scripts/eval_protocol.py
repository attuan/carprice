"""評価プロトコル — すべての実験をここに通して同じ土俵に乗せる。

CLAUDE.md の「同じ指標で精度を記録する」を機械的に守るための共通基盤。
以降 unfold の機能A（特徴量生成）・機能B（LLMPredictor）を試すときも、
必ず `cross_validate()` を経由させて `results/leaderboard.csv` に積む。

固定しているもの（一度決めたら変えない。変えるなら過去の行も引き直す）:

- データ    : sampledata/processed/usedsienta_clean.parquet（シエンタ 5,513行）
- 目的変数  : 車両本体価格_万円
              ※ 支払総額ではない。支払総額には店舗ごとの諸費用が乗っていて、
                 車両そのものの価値以外の分散が混ざるため。
- 分割      : KFold(n_splits=5, shuffle=True, random_state=42)
- 指標      : MAE / RMSE / MAPE / R2（すべて万円スケールで計算）
              主指標は MAE。中央値185万円に対して外れ値が穏やかなので、
              RMSE より解釈しやすい（MAE 10 = 平均10万円ずれる）。

使い方:

    from eval_protocol import load_dataset, cross_validate

    df = load_dataset()
    cross_validate("手法名", fit_predict, df, note="何をしたか")

`fit_predict(train_df, test_df) -> np.ndarray` は fold ごとに呼ばれる。
**前処理も含めて train だけで fit すること。** TF-IDF や埋め込みを
全データで fit すると test の情報が漏れる。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "sampledata" / "processed" / "usedsienta_clean.parquet"
LEADERBOARD = ROOT / "results" / "leaderboard.csv"

TARGET = "車両本体価格_万円"
SEED = 42
N_SPLITS = 5

# --- 特徴量の区分 -----------------------------------------------------
# 「従来手法」の列。スクレイピング当時に回帰分析へ入れていたものと同じ構成。
# ここが出発点で、これを超えられるかが本プロジェクトの問い。
LEGACY_NUM = ["車齢", "走行距離_km"]
LEGACY_BOOL = ["修復歴あり", "保証付", "ハイブリッド"]
LEGACY_CAT = ["車検区分", "都道府県"]

# クレンジングで新たに取り出せた構造化列。
# 車検残月数は「期限指定」の車にしか無く欠損2,926件だが、欠損自体が
# 「車検整備付」を意味するので落とさず欠損のまま木に渡す。
EXTRA_NUM = ["車検残月数", "装備数"]
EXTRA_CAT = ["グレード名", "色_基本"]

# 非構造テキスト。unfold 機能A の入力になる列。
TEXT_COL = "装備テキスト"

# 除外する列とその理由:
#   車名(シエンタのみ)・排気量(1.5のみ)     … 定数
#   年式                                    … 車齢と完全に従属
#   支払総額_万円                            … 目的変数のリーク
#   車検満了                                 … 車検残月数として数値化済み
#   物件ID・店舗・タイトル・グレード_原文     … ID的／テキスト。別途扱う
#   色                                       … 色_基本に集約済み
#   走行距離_異常・装備記載あり・タイトル切れ疑い・タイトル文字数
#                                            … クレンジングの品質フラグ。
#                                               予測に使うのは筋が悪いので外す


def load_dataset(verbose: bool = True) -> pd.DataFrame:
    """学習用データを読む。目的変数が欠損の行（「応談」表記）だけ落とす。"""
    df = pd.read_parquet(DATA)
    n_before = len(df)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    if verbose:
        print(f"データ: {DATA.relative_to(ROOT)}")
        print(f"  {n_before:,} 行 → {len(df):,} 行"
              f"（価格欠損 {n_before - len(df)} 行を除外）")
        print(f"  価格: 中央値 {df[TARGET].median():.1f} 万円 / "
              f"平均 {df[TARGET].mean():.1f} 万円 / "
              f"範囲 {df[TARGET].min():.0f}〜{df[TARGET].max():.0f} 万円")
    return df


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err / y_true)) * 100),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    }


def cross_validate(
    name: str,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    df: pd.DataFrame | None = None,
    note: str = "",
    record: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """5-fold CV を回し、結果を leaderboard.csv に1行追記する。

    fit_predict は (train_df, test_df) を受け取り test_df の予測値
    （万円スケール）を返す。中で log 変換するのは自由だが、返す時点で
    万円に戻しておくこと。指標は必ず万円で計算する。
    """
    if df is None:
        df = load_dataset(verbose=False)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    per_fold: list[dict[str, float]] = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)
        y_pred = np.asarray(fit_predict(train_df, test_df), dtype=float)
        y_true = test_df[TARGET].to_numpy(dtype=float)
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"{name}: 予測の形が合いません {y_pred.shape} != {y_true.shape}"
            )
        m = _metrics(y_true, y_pred)
        per_fold.append(m)
        if verbose:
            print(f"  fold{fold}  MAE {m['MAE']:6.2f}  RMSE {m['RMSE']:6.2f}"
                  f"  MAPE {m['MAPE']:5.1f}%  R2 {m['R2']:.3f}")

    agg = {k: float(np.mean([f[k] for f in per_fold])) for k in per_fold[0]}
    agg["MAE_std"] = float(np.std([f["MAE"] for f in per_fold]))

    if verbose:
        print(f"  {'平均':6}  MAE {agg['MAE']:6.2f} (±{agg['MAE_std']:.2f})"
              f"  RMSE {agg['RMSE']:6.2f}  MAPE {agg['MAPE']:5.1f}%"
              f"  R2 {agg['R2']:.3f}")

    if record:
        _append(name, agg, len(df), note)
    return agg


def _append(name: str, agg: dict[str, float], n_rows: int, note: str) -> None:
    LEADERBOARD.parent.mkdir(exist_ok=True)
    header = ["実行日時", "手法", "MAE", "MAE_std", "RMSE", "MAPE", "R2",
              "行数", "fold数", "seed", "備考"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        name,
        f"{agg['MAE']:.3f}", f"{agg['MAE_std']:.3f}", f"{agg['RMSE']:.3f}",
        f"{agg['MAPE']:.2f}", f"{agg['R2']:.4f}",
        n_rows, N_SPLITS, SEED, note,
    ]
    is_new = not LEADERBOARD.exists()
    with LEADERBOARD.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)


def show_leaderboard() -> pd.DataFrame:
    """これまでの全実験を MAE 昇順で表示する。"""
    if not LEADERBOARD.exists():
        print("まだ結果がありません。")
        return pd.DataFrame()
    lb = pd.read_csv(LEADERBOARD, encoding="utf-8-sig")
    return lb.sort_values("MAE").reset_index(drop=True)
