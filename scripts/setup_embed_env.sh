#!/usr/bin/env bash
# 埋め込み計算専用の隔離環境を作る。
#
# なぜ分けるのか:
#   このマシンは Intel Mac なので PyTorch は 2.2.2 が上限（それ以降は
#   Apple Silicon 向けしかビルドされていない）。2.2.2 は numpy 1.x 時代の
#   ビルドで、主環境 .venv の numpy 2.x とは共存できない。
#   主環境に入れると pandas / LightGBM ごと壊れるため、別の venv に隔離する。
#
# 使い方:
#   bash scripts/setup_embed_env.sh                     # 環境を作る
#   .venv-embed/bin/python scripts/embed_text.py        # 埋め込みを計算する
#   .venv/bin/python scripts/run_embedding.py           # 主環境で精度を測る
#
# 埋め込みは parquet に落として受け渡すので、主環境は torch に一切触らない。
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d .venv-embed ]; then
  echo ".venv-embed は既にあります。作り直すなら rm -rf .venv-embed してから実行してください。"
  exit 0
fi

PY312=/usr/local/opt/python@3.12/bin/python3.12
[ -x "$PY312" ] || PY312=$(command -v python3.12)

"$PY312" -m venv .venv-embed
.venv-embed/bin/pip install --upgrade pip -q
.venv-embed/bin/pip install -q -r requirements-embed.txt

echo "--- 確認 ---"
.venv-embed/bin/python - <<'PY'
import numpy, torch, sentence_transformers as st
print("numpy :", numpy.__version__, "(1.x であること)")
print("torch :", torch.__version__)
print("st    :", st.__version__)
PY
