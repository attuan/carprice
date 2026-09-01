#!/usr/bin/env python3
"""通しドキュメント4本（PRD / progress-log / related-work / README）の更新漏れを機械的に探す。

「内容が正しいか」は人（または LLM）にしか判定できないが、
**内容以前のズレ**は文字列の突き合わせで見つかる。ここで見るのはその4つだけ。

  A. 本数        docs/README.md と README.md に書いた「NN本」が実ファイル数と合っているか
  B. 索引漏れ    docs/2026-*.md が docs/README.md の表に載っているか（逆に、消えた行がないか）
  C. 記号        P / S / R 番号の集合が PRD.md・docs/README.md・related-work.md で揃っているか
  D. 鮮度        results/ や docs/2026-*.md より通しドキュメントのほうが古いコミットのままでないか

D だけは git の履歴を見る。測定を足したのに progress-log を直していない、が典型的な取りこぼしで、
これは日付の大小だけで検出できる。**内容の正しさは見ていない**ので、
ここが通っても「更新済み」の証明にはならない。判断の入口として使う。

使い方:

    python3 scripts/check_docs.py            # 指摘があれば表示して終了コード 1
    python3 scripts/check_docs.py --quiet    # 終了コードだけ使う（フックから呼ぶとき）

標準ライブラリのみ / Python 3.8 以降。venv の有効化に依存させない。
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# 「通し」のドキュメント。日付がつかず、状況が動いたら直す必要があるもの
STANDING = ["docs/PRD.md", "docs/progress-log.md", "docs/related-work.md",
            "docs/README.md", "README.md"]

# 通しドキュメントより新しくなっていたら「反映漏れかもしれない」と疑う対象
SOURCES = ["results/", "unfold/", "scripts/"]

# 記号の定義元。ここに載っている番号が、参照側にも同じだけあるはず
SYMBOLS = {
    # PRD の P は取り消し線つき（`~~P5~~【実施不能】`）で出ることがあるので ~ を挟んで許す
    "P": [("docs/PRD.md", r"P(\d+(?:\.\d+)?)~*【"), ("docs/README.md", r"^\| P(\d+(?:\.\d+)?) \|")],
    "S": [("docs/PRD.md", r"^\| S(\d) \|"), ("docs/README.md", r"^\| S(\d) \|")],
    "R": [("docs/related-work.md", r"\bR(\d)\b"), ("docs/README.md", r"^\| R(\d) \|")],
}


def git(*args):
    """git を読み取り専用で呼ぶ。失敗したら None（呼び出し側は「指摘しない」に倒す）。"""
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=20)
    except Exception:
        return None
    return p.stdout.decode("utf-8", "replace") if p.returncode == 0 else None


def last_commit_epoch(path):
    """path を最後に触ったコミットの時刻（epoch 秒）。履歴になければ None。"""
    out = git("log", "-1", "--format=%ct", "--", path)
    out = (out or "").strip()
    return int(out) if out.isdigit() else None


def dated_docs():
    return sorted(p.name for p in DOCS.glob("2026-*.md"))


def read(rel):
    try:
        return (ROOT / rel).read_text(encoding="utf-8")
    except OSError:
        return ""


def check_count(issues):
    """docs/*.md の本数（docs/README.md 自身は数えない）が、文中の「NN本」と合っているか。"""
    actual = len([p for p in DOCS.glob("*.md") if p.name != "README.md"])
    for rel in ("docs/README.md", "README.md"):
        for written in re.findall(r"(\d+)\s*本のドキュメント|ドキュメント[はが]?\s*(\d+)\s*本|"
                                  r"`docs/`\s*にあります（(\d+)本）", read(rel)):
            n = next((x for x in written if x), None)
            if n and int(n) != actual:
                issues.append(f"{rel}: ドキュメントの本数が {n} 本と書いてあるが、実際は {actual} 本")


def check_index(issues):
    """日付つきドキュメントが docs/README.md の表から漏れていないか（両方向）。"""
    index = read("docs/README.md")
    listed = set(re.findall(r"`(2026-[\w.-]+\.md)`", index))
    actual = set(dated_docs())
    for name in sorted(actual - listed):
        issues.append(f"docs/README.md: `{name}` が索引に載っていない")
    for name in sorted(listed - actual):
        issues.append(f"docs/README.md: `{name}` を索引に載せているが、ファイルが存在しない")


def check_symbols(issues):
    """P / S / R 番号の集合が、定義元と参照側で一致しているか。"""
    for kind, ((src_rel, src_re), (ref_rel, ref_re)) in SYMBOLS.items():
        src = set(re.findall(src_re, read(src_rel), re.MULTILINE))
        ref = set(re.findall(ref_re, read(ref_rel), re.MULTILINE))
        if not src or not ref:
            continue  # 片方が拾えないときは書式が変わったとき。誤検知を出さない
        for n in sorted(ref - src):
            issues.append(f"{ref_rel}: {kind}{n} を参照しているが、{src_rel} に定義がない")
        for n in sorted(src - ref):
            issues.append(f"{ref_rel}: {src_rel} にある {kind}{n} が一覧に出てこない")


def check_freshness(issues):
    """通しドキュメントを最後に直して以降、素材だけが動いたコミットが積もっていないか。

    4本を1つのまとまりとして「最後にどれかを直した時刻」を基準にする。
    ドキュメントごとに個別に見ると、related-work.md のように
    めったに動かないファイルが毎回引っかかって警報が意味を失うため。
    """
    times = [t for t in (last_commit_epoch(rel) for rel in STANDING) if t is not None]
    if not times:
        return
    base = max(times)

    paths = SOURCES + [f"docs/{n}" for n in dated_docs()]
    out = git("log", "--format=%ct\t%h\t%s", "--", *paths)
    if out is None:
        return
    pending = []
    for line in out.strip().split("\n"):
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) > base:
            pending.append(f"{parts[1]} {parts[2]}")
    if pending:
        issues.append("通しドキュメントを最後に直してから、素材だけが動いたコミットが "
                      f"{len(pending)} 件ある: " + " / ".join(pending[:5]) +
                      (" ..." if len(pending) > 5 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true", help="終了コードだけ返す")
    args = ap.parse_args()

    issues = []
    for check in (check_count, check_index, check_symbols, check_freshness):
        try:
            check(issues)
        except Exception as e:  # 検査の不具合で作業を止めない
            if not args.quiet:
                print(f"（{check.__name__} は実行できなかった: {e}）", file=sys.stderr)

    if not args.quiet:
        if issues:
            print("ドキュメントの更新漏れの疑い:")
            for m in issues:
                print(f"  - {m}")
            print("\n直すときは /update-docs を使う（判断が要る書き換えはスキル側の手順に従う）。")
        else:
            print("通しドキュメント4本に、機械的に見つかるズレはない。")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
