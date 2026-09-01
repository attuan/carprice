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

`.env` は作成済みです。エディタで開き、`ANTHROPIC_API_KEY=` の右にキーを貼るだけで動きます。
（clone 直後で `.env` が無い場合は `cp .env.example .env` で作る。`.env` は git 管理外）

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

キーは https://console.anthropic.com/settings/keys で発行します。
貼ったら疎通確認する（実際に1回だけ Claude を呼びます。費用は $0.001 程度）:

```bash
.venv/bin/python scripts/check_api_key.py
.venv/bin/python scripts/check_api_key.py --no-call   # 課金なしで読み込みだけ確認
```

## unfold（ライブラリ本体）

`unfold/` が設計書（`dialogs/unfold-landing.html`）の実装です。機能は2つあります。

### 機能A — `Feature`（特徴量生成）

非構造列を、宣言するだけで型のついた列にします。中身は「埋め込み → 近傍で分類」で、
LLM は確信度が低い行のフォールバックにしか出てきません。

```python
from unfold import Feature

df["グレード"] = Feature(
    source="タイトル",
    type="category",
    values=["G", "Z", "X", "G クエロ"],
).fit_transform(df)
```

### 機能B — `LLMPredictor`（LLM Predict）

LLM に生データを渡して当てさせるのではなく、**LightGBM・XGBoost・近傍検索に
先に解かせ、その予測値と「実際の価格が分かっている似た事例」を証拠として渡し、
最終判断だけさせます**。類似事例は行ごとに引き直します（フューショット）。

```python
from unfold import LLMPredictor

model = LLMPredictor(
    target="車両本体価格_万円", unit="万円",
    numeric=["車齢", "走行距離_km"], categorical=["グレード名"],
    text="装備テキスト",
)
pred = model.fit(train_df).predict(test_df)
```

正解ラベルを別途用意する必要はありません。訓練データの価格がそのまま
フューショットの例になります。

### 使う前に — そのデータで効くかを確かめる

機能B は「テキストが価格を左右するデータ」でしか効きません（シエンタでは負け、
Craigslist では勝ちました）。**LLM を呼ぶ前に**、そのデータでテキストが効くかを
無料で判定できます。

```python
from unfold import screen

print(screen(df, target="価格_usd", text="車種名", unit="USD",
             numeric=["車齢", "走行距離_mile"], categorical=["メーカー", "州"]))
# テキスト寄与率 15.4%（閾値 10%） → 判定: 試す価値あり
```

### 自由記述を渡すときの注意

出品テキストには**売り値がそのまま書いてあることが多い**です
（Craigslist の説明文は 43.7% の行が該当）。そのまま渡すと予測ではなく
答えの読み取りになるので、`unfold` は**既定で金額を伏せます**。

```python
model = LLMPredictor(..., long_text="説明文")   # 金額は自動で〈金額〉に伏せられる
```

詳しくは `docs/2026-09-01-description-leak.md`。

### 共通 — 来歴（provenance）の検査 API

どちらの機能も、設計書どおり「なぜその値になったか」を辿れます。

```python
model.explain(0)      # その行の来歴（証拠・参照した事例・LLM の理由）
model.confidence()    # 行ごとの確信度
model.examples()      # 推論に使った類似事例
model.cost()          # かかった費用と1行あたりの単価
model.provenance()    # 全行の来歴を1つの表に
```

LLM の応答は `sampledata/processed/llm_cache/` にディスクキャッシュされます。
**同じ行を2度課金しない**ので、実装をいじって測り直すのは無料です。

測定結果は `docs/` の日付つきファイルにあります。テストは次のとおり
（LLM 部分は差し替えているので **APIキー無しでも全部通ります**）。

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
