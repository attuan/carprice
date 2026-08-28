# carprice

株式会社INDX のインターン（2026/8/28〜）で作成する、中古車価格の自動予測プログラム。
LLM を使った AutoML エージェントを構成し、特徴量生成・モデル選択・予測を自動化することが目標。

## 分析アプローチ（3方式を全部試す方針）

1. LLM で直接価格を予測する
2. LLM でモデル選択を自動化する（統計モデルは従来通り）
3. LLM で特徴量生成を自動化する

どれをメインにするかは未決定。比較できるよう、各方式の精度を同じ指標で記録すること。

背景: 従来の回帰分析では定性情報をダミー変数でしか扱えず精度に限界があった。
そこを LLM で埋められるかが本プロジェクトの主題。

## ディレクトリ

- `dialogs/` — 議事録・設計資料。**編集しない**（記録なので）
  - `entrysheet.md` — 最初の企画案
  - `2026...ミーティング.md` — 方針が端的にまとまっている。迷ったらまずここ
  - `unfold-landing.html` — 伊藤さんによる設計書
- `scripts/` — データ取得などの補助スクリプト
- `sampledata/` — データ。用途ごとに4つに分かれている（下記）

## データ

置き場所のルールはこの4つ。新しいデータを足すときは必ずどれかに分類する。

| ディレクトリ | git管理 | 中身 |
|---|---|---|
| `sampledata/raw/` | ✕ | 外部から再取得できる巨大な生データ |
| `sampledata/scraped/` | ○ | 自分たちで集めた、再取得できないデータ |
| `sampledata/processed/` | ✕ | 分析途中の中間データ。コードで再生成できる |
| `sampledata/sample/` | ○ | 動作確認用の小さな抜粋 |

**判断基準は「再現できるか」。** 再取得・再生成できるものは git に入れず、
手順（スクリプト）の方を git に入れる。git は全履歴を全員に配るので、
一度でも巨大ファイルを入れると削除しても永久に残る。

### `sampledata/scraped/usedsientaL.edit.omit.csv`（203KB, 日本語）

2026年3月にカーセンサーから Octoparse でスクレイピングしたトヨタ シエンタのデータ。
列: `index, 車両本体価格(万円), 走行距離(km), 車歴, 修復歴, 保証, ハイブリッド, 車検, 都道府県`
数値列はすでにダミー変数化済み（0/1）。都道府県だけが文字列。
**再取得できないので git 管理下に置いている。**

### `sampledata/raw/vehicles.csv`（1.4GB, 英語, git管理外）

Kaggle の Craigslist 中古車データ。clone 直後は存在しないので、まず取得する:

```bash
python3 scripts/download_data.py
```

**巨大なので絶対に全体を読み込まないこと。** まずはサンプルで組み立てる:

```bash
head -20 sampledata/sample/vehicles_sample500.csv
```

`description` 列がフィールド内に改行を含むため、**`head` で行数を切ると
CSV レコードが壊れる。** 抜粋を作るときは必ず csv パーサを使うこと
（`sampledata/sample/vehicles_sample500.csv` はその方法で生成済み・500行）。

pandas で本番データを読むときも `nrows=` か `chunksize=` を必ず指定する。
欠損が多い（1行目から `year`, `manufacturer` などが空）ので、まず欠損率の確認から入る。

## 環境

Python 3.12.14（Homebrew）+ venv。**システムの `/usr/bin/python3`（3.9）は使わない。**

```bash
source .venv/bin/activate     # 有効化。以後 python / pip は venv のものになる
```

有効化せずに単発で動かす場合は `.venv/bin/python スクリプト.py` と直接叩く。
依存は `requirements.txt`。追加したら `pip install -r requirements.txt` を再実行。

未セットアップの環境での構築手順:

```bash
brew install python@3.12 libomp          # libomp は xgboost/lightgbm に必要
/usr/local/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

APIキーは `.env`（git管理外）に置く。雛形は `.env.example`。

### 巨大データの扱い方

`sampledata/raw/vehicles.csv` は 426,880行 / 1.4GB。

実測値（このマシン: Intel i7-8850H / 16GB）:

| 方法 | 全件集計の所要 | ピークメモリ |
|---|---|---|
| DuckDB（ファイルを直接SQL） | 1.3秒 | 0.22 GB |
| pandas 全件読み込み | 55.9秒 | 6.80 GB |

**探索・集計は DuckDB を使うこと。** 40倍速く、メモリは30分の1。
pandas 全件読みは 6.8GB 使うので、16GB 機ではブラウザ等と併用すると苦しい。
さらに `.copy()` や merge をすると倍近くまで膨らむので、
pandas に渡すのは必要な列・行に絞り込んだ後にする。

```python
import duckdb
duckdb.sql("SELECT manufacturer, count(*) FROM "
           "read_csv_auto('sampledata/raw/vehicles.csv', ignore_errors=true) "
           "GROUP BY 1 ORDER BY 2 DESC").show()
```

モデリングで部分集合を DataFrame にする段階になったら pandas に渡す。
中間データは `sampledata/processed/` に **parquet** で保存する（CSVより小さく速い）。

### 既知のデータ品質問題

`vehicles.csv` の `price` は外れ値が激しい。平均 $75,199 に対し中央値 $13,950、
最大は $3,736,928,711。**平均を使う前に必ず外れ値処理をすること。**

## ノートブック

`notebooks/` に連番で置く（`01_explore_vehicles.ipynb` など）。
カーネルは必ず `.venv` のものを使う。ノートブック内では `!pip install` を使わず、
`requirements.txt` に追記して `pip install -r requirements.txt` する。

グラフに日本語を使う場合は先頭で以下を設定する（未設定だと豆腐□になる）:

```python
plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False
```

出力込みでコミットしている（結果を共有するため）。差分が読みにくくなってきたら
`nbstripout` の導入を検討する。

## ルール

- 説明は日本語で書く
- 巨大ファイル（100MB 超）を git にコミットしない。`.gitignore` を確認してから `git add` する
  - このルールは `.claude/hooks/block_large_git_add.py` が機械的に強制している。
    `git add` / `git commit` の直前にサイズを検査し、100MB 以上は拒否、25MB 以上は確認を求める。
    閾値は環境変数 `CARPRICE_GIT_DENY_MB` / `CARPRICE_GIT_ASK_MB` で変えられる。
- 生成した中間ファイルは `sampledata/processed/` に置く（git 管理外）
- コミットメッセージは日本語で可
