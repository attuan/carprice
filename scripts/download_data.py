#!/usr/bin/env python3
"""Kaggle の Craigslist 中古車データを sampledata/raw/ に取得する。

このリポジトリには 1.4GB の生データを含めていないため、
clone した直後は各自でこのスクリプトを実行する必要がある。

    python3 scripts/download_data.py

事前準備（初回のみ）:
  仮想環境を有効化した上で、次のどちらかで Kaggle 認証を通す。

  A) ブラウザ認証（推奨・トークン管理不要）
       kaggle auth login
  B) APIトークン
       https://www.kaggle.com/settings の "API" で Generate New Token を押し、
       export KAGGLE_API_TOKEN=xxxxx  もしくは ~/.kaggle/access_token に保存
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "austinreese/craigslist-carstrucks-data"
RAW_DIR = Path(__file__).resolve().parent.parent / "sampledata" / "raw"
TARGET = RAW_DIR / "vehicles.csv"
EXPECTED_BYTES = 1_447_955_215  # 2021-05-06 版のサイズ


def already_present() -> bool:
    if not TARGET.exists():
        return False
    size = TARGET.stat().st_size
    if size == EXPECTED_BYTES:
        print(f"取得済み: {TARGET} ({size:,} bytes)")
        return True
    print(
        f"警告: {TARGET} は存在しますがサイズが想定と異なります。\n"
        f"  実際  : {size:,} bytes\n"
        f"  想定  : {EXPECTED_BYTES:,} bytes\n"
        "  データセットが更新された可能性があります。再取得する場合は"
        "このファイルを削除してから再実行してください。"
    )
    return True


def main() -> int:
    if already_present():
        return 0

    if shutil.which("kaggle") is None:
        print(
            "kaggle コマンドが見つかりません。仮想環境を有効化してください:\n"
            "  source .venv/bin/activate\n"
            "未セットアップなら: python3.12 -m venv .venv && "
            ".venv/bin/pip install -r requirements.txt\n"
            "認証手順はこのファイル冒頭のコメントを参照。\n"
            "\n"
            "手動で取得する場合は次の URL から vehicles.csv をダウンロードし、\n"
            f"  https://www.kaggle.com/datasets/{DATASET}\n"
            f"{RAW_DIR} に置いてください。",
            file=sys.stderr,
        )
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Kaggle から取得中: {DATASET}（1.4GB あるため数分かかります）")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR)]
    )
    if result.returncode != 0:
        print("kaggle コマンドが失敗しました。認証設定を確認してください。", file=sys.stderr)
        return result.returncode

    archives = list(RAW_DIR.glob("*.zip"))
    if not archives:
        print("zip が見つかりません。取得結果を確認してください。", file=sys.stderr)
        return 1

    for archive in archives:
        print(f"展開中: {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(RAW_DIR)
        archive.unlink()

    if not TARGET.exists():
        print(f"展開後も {TARGET} が見つかりません。", file=sys.stderr)
        return 1

    print(f"完了: {TARGET} ({TARGET.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
