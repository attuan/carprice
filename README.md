# carprice

株式会社INDXのインターンにて作成する中古車価格の自動予測プログラムです。
AutoMLエージェントを構成し、AIによってモデル選択、予測、特徴量選択を行えるようにするつもりです。

## セットアップ

### 1. Python環境

Python 3.12 を使います（システム標準の 3.9 では動きません）。

```bash
brew install python@3.12 libomp                        # libomp は xgboost/lightgbm に必要
/usr/local/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

### 2. データの取得

このリポジトリには 1.4GB の生データを含めていません。

```bash
kaggle auth login              # 初回のみ（ブラウザで認証）
python scripts/download_data.py
```

取得せずに動作確認だけしたい場合は `sampledata/sample/` の抜粋データが使えます。

### 3. APIキー（LLMを使う段階になったら）

```bash
cp .env.example .env           # .env は git 管理外
```

## unfold（ライブラリ本体）

`unfold/` が設計書（`dialogs/unfold-landing.html`）の実装です。
現在あるのは**機能A（`Feature`）の骨組み**で、LLM 呼び出しなしで動きます。

```python
from unfold import Feature

# 非構造列（タイトル）から、型のついた列を作る
df["グレード"] = Feature(
    source="タイトル",
    type="category",
    values=["G", "Z", "X", "G クエロ"],
).fit_transform(df)
```

`fit` / `transform` のほかに、設計書どおりの検査 API があります。

```python
f.explain(0)        # そのセルの来歴（値・confidence・参照した事例・費用）
f.confidence()      # 行ごとの確信度
f.cost()            # LLM に回る行の割合と費用の見積もり
f.review_queue()    # 確信度が低くレビュー待ちになった行
```

測定結果は `docs/2026-08-29-feature-skeleton.md`。テストは次のとおり。

```bash
.venv/bin/python -m pytest tests -q
```

## 分析の始め方

```bash
source .venv/bin/activate
jupyter lab                    # ブラウザで開く（Colab に近い操作感）
```

VS Code で `notebooks/01_explore_vehicles.ipynb` を直接開いても同じことができます。
その場合は右上の「カーネルの選択」で `.venv` を選んでください。

`notebooks/01_explore_vehicles.ipynb` が出発点です。
上から順に実行すると、vehicles.csv の全体像の確認から
`sampledata/processed/vehicles_clean.parquet`（380,907行）の生成までが一通り走ります。

## dialogsについて

これまでの議論によって作られた議事録や、イメージをまとめてあるフォルダです。
`entrysheet.md` はこのインターンに参加するにあたって最初のアイデアがまとめられてあります。
`20260812中古車...` はAI作成の議事録で、方針が端的にまとまってあります。
`unfold-landing.html` は伊藤さんによるアイデアをまとめた設計書の一つになります。

## sampledataについて

分析をかけるであろうデータです。「再現できるかどうか」で4つに分けています。

| ディレクトリ | git管理 | 中身 |
|---|---|---|
| `raw/` | ✕ | 外部から再取得できる巨大な生データ |
| `scraped/` | ○ | 自分たちで集めた、再取得できないデータ |
| `processed/` | ✕ | 分析途中の中間データ。コードで再生成できる |
| `sample/` | ○ | 動作確認用の小さな抜粋 |

- `raw/vehicles.csv` — Kaggleから取得したCraigslistの中古車データ（1.4GB, 英語）。
  https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
- `scraped/usedsientaL.edit.omit.csv` — 2026年3月頃にカーセンサーから
  octoparseでデータスクレイピングしてきたものです（203KB, 日本語）。
- `sample/vehicles_sample500.csv` — `raw/vehicles.csv` の先頭500件（1.5MB）。

生データを git に入れていないのは、git が全履歴を全員に複製する仕組みで、
一度巨大ファイルを入れると削除しても永久に残るためです。
データそのものではなく、取得手順（`scripts/download_data.py`）を管理しています。
