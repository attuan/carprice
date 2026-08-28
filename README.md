# carprice

株式会社INDXのインターンにて作成する中古車価格の自動予測プログラムです。
AutoMLエージェントを構成し、AIによってモデル選択、予測、特徴量選択を行えるようにするつもりです。

## セットアップ

このリポジトリには 1.4GB の生データを含めていません。clone 後に取得してください。

```bash
python3 scripts/download_data.py
```

（初回は `pip3 install kaggle` と Kaggle API トークンの設定が必要です。
詳細は [scripts/download_data.py](scripts/download_data.py) の冒頭コメントを参照）

取得せずに動作確認だけしたい場合は `sampledata/sample/` の抜粋データが使えます。

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
