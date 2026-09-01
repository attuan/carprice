# carprice

株式会社INDXのインターンにて作成する中古車価格の自動予測プログラムです。

成果物は **`unfold`** という scikit-learn 互換の Python ライブラリで、機能は2つあります。

- **機能A（`Feature`）** — 画像や自由記述などの非構造な列を、宣言するだけで型のついた列にする
- **機能B（`LLMPredictor`）** — 統計モデルに先に解かせ、その予測値と類似事例を証拠として
  LLM に渡し、最終判断だけさせる

そこに **信頼度ルーティング（`AdaptivePredictor`）** が乗り、「どの行を LLM に回すか」を決めます。

構想は3段階で変わってきました（詳細は `CLAUDE.md`）。当初は「LLM にモデル選択をさせる」案も
並べていましたが、伊藤さんの設計書（`dialogs/unfold-landing.html`）の段階で**その案は消え**、
上の2機能に統合されています。**古い資料を読むときはどの時点のものかに注意してください。**

## ドキュメント

測定結果・設計判断・経緯はすべて `docs/` にあります（23本）。
**`docs/README.md` が入口**で、どれから読むかと、本文に出てくる
`P3` `S6` `R2` といった記号の意味をまとめてあります。

- **結果だけ知りたい** → `docs/results-summary.md`
- **経緯を追いたい** → `docs/progress-log.md`
- **なぜそう作ったのか** → `docs/PRD.md`

---

## セットアップ

### 1. Python環境

Python 3.12 を使います（システム標準の 3.9 では動きません）。

Ubuntu（計算ノード）:

```bash
sudo apt update && sudo apt install -y python3.12-venv fonts-noto-cjk
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

macOS:

```bash
brew install python@3.12 libomp                        # libomp は xgboost/lightgbm に必要
/usr/local/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

`fonts-noto-cjk` はグラフの日本語表示に必要です（macOS は Hiragino Sans が標準搭載）。

**環境は2つに分かれています。**

| | 中身 | いつ使うか |
|---|---|---|
| `.venv`（主環境） | pandas / LightGBM / XGBoost / `unfold` 本体 | ふだんはこちら |
| `.venv-embed` | torch / sentence-transformers / TabPFN | 埋め込みの計算と TabPFN のときだけ |

**`unfold` は torch 無しで動きます**（既定のエンコーダは文字 TF-IDF）。
主環境に torch を入れないのは、うっかり torch へ依存してもテストが通ってしまうのを
防ぐためです。`.venv-embed` が要る作業をするときだけ、次を実行します。

```bash
bash scripts/setup_embed_env.sh        # requirements-embed.txt から構築
.venv-embed/bin/python scripts/embed_text.py       # 使うときは python を切り替える
```

詳しい経緯は `docs/2026-09-01-embed-env-rebuild.md` にあります。

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

`unfold/` が設計書（`dialogs/unfold-landing.html`）の実装です。
機能は2つ（`Feature` / `LLMPredictor`）で、そこに「どの行を LLM に回すか」を
決める信頼度ルーティング（`AdaptivePredictor`）が乗ります。

### まず動かす

**APIキーもデータ取得も要りません。**git に入っている 500 行の抜粋を使い、
リーク検査 → スクリーニング → 機能A → 費用の見積もり、までを一度に流します。
**既定では LLM を1回も呼ばない**ので無料です。

```bash
.venv/bin/python -m unfold.demo              # 無料
.venv/bin/python -m unfold.demo --run 20     # 20行を予測（うち6行が LLM に回る。約 $0.05）
```

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

### 信頼度ルーティング — `AdaptivePredictor`（どの行を LLM に回すか）

機能B は**1行につき1回 LLM を呼ぶ**ので、行数がそのまま費用と時間になります
（6万行なら約 $516・約17時間）。そこで「LLM を呼ぶ前に手に入る信号」だけで
呼ぶ行を選び、残りは統計モデルに任せます。閾値ひとつで
「全行呼ぶ」と「1行も呼ばない」の間を連続に動かせます。

```python
from unfold import AdaptivePredictor

model = AdaptivePredictor(target="価格_usd", unit="USD",
                          numeric=["車齢", "走行距離_mile"],
                          categorical=["メーカー", "州"], text="車種名",
                          escalate_rate=0.3)     # 信号の強い上位3割だけ回す
model.fit(train)
model.plan(test)      # 呼ぶ前に「何行・いくら・何秒」（LLM を呼ばないので無料）
pred = model.predict(test)
model.curve(test)     # 割合を振ったときの精度・費用・レイテンシ
model.approve()       # LLM の答えを承認 → 次回はその行を呼ばずに返る
```

既定の信号は **LightGBM と XGBoost の予測の食い違い**（統計モデル自身が
迷っている行に回す）で、実測で最良でした。`signal="unseen"` にすると
訓練データに無い車種名・グレードを含む行を優先します。

Craigslist の 60 行で上位30%だけ回した実測は
**MAE 2,767 → 2,476（費用 $0.155）**。詳しくは `docs/2026-09-01-adaptive.md`。

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

### 共通 — リーク検査（重複レコードの検知）

**同じ車が train と test の両方に入っていると、予測ではなく答えの読み取りに
なります。**Craigslist のデータは同じ車が複数の地域に出稿されており、
重複を残したまま評価すると R² が 0.880 → 0.914 に水増しされました。
しかも**テキストを特徴量にするほど水増しが大きくなる**ので、
`unfold` は fit / predict のときに自動で検知して警告します。

```python
from unfold import check_duplicates, check_overlap

print(check_duplicates(df, keys=["車台番号"]))   # 1つの表の中の重複
print(check_overlap(train, test))                # train と test にまたがる重複
```

例外にはしません（重複が意図的なこともあるため）。うるさければ
`LLMPredictor(..., check_leakage=False)` で切れます。

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

**中間データが2つあるので、取り違えないでください。**

| ファイル | 行数 | 列名 | 重複排除 | 用途 |
|---|---:|---|---|---|
| `vehicles_clean.parquet` | 380,907 | 英語（元のまま） | **していない** | ノートブックでの探索用 |
| `vehicles_multi_clean.parquet` | 200,374 | 日本語 | **済み**（1台1行） | **測定はすべてこちら** |

**精度を測るときは必ず `vehicles_multi_clean.parquet`**（`scripts/clean_vehicles.py` が生成）を使います。
同じ車が複数の地域に重複出稿されているため、重複を残したまま交差検証すると
train と test に同じ車が入り、成績が良く見えてしまいます。

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
