#!/usr/bin/env python3
"""巨大ファイルが git のステージに乗るのを未然に止める PreToolUse フック。

git の履歴は全員に配られ、一度入った巨大ファイルは削除しても永久に残る。
CLAUDE.md の「100MB 超をコミットしない」を、お願いではなく機械的に強制する。

Bash ツールの実行直前に呼ばれ、コマンドが git add / git commit なら
追加されうるファイルを先読みしてサイズを見る。

  DENY_MB 以上 -> 拒否する
  ASK_MB  以上 -> ユーザーに確認を求める
  それ未満     -> 何もしない

検出には `git ls-files` だけを使う（読み取り専用）。
`git add --dry-run` は名前に反して index.lock を取るので、フックから呼んではいけない。
本来の git コマンドとぶつかったり、中断時に stale な lock を残したりする。

判定に失敗したときは必ず「通す」側に倒す。フックの不具合で作業が止まるほうが困るため。
標準ライブラリのみ / Python 3.8 以降で動く（venv の有効化に依存させない）。
"""
import json
import os
import shlex
import subprocess
import sys

DENY_MB = int(os.environ.get("CARPRICE_GIT_DENY_MB", "100"))
ASK_MB = int(os.environ.get("CARPRICE_GIT_ASK_MB", "25"))

# コマンドの区切り。これで分割して1つずつ git かどうかを見る
SEPARATORS = {"&&", "||", ";", "|", "&", "\n"}
# git 本体のオプションのうち、値を1つ取るもの（サブコマンド判定でスキップする）
GIT_OPTS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}


def run(args, cwd):
    """git を呼ぶ。失敗したら None を返す（呼び出し側は「通す」に倒す）。"""
    try:
        p = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=20)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    return p.stdout.decode("utf-8", "replace")


def lines(out):
    return [l for l in (out or "").split("\n") if l]


def split_segments(command):
    """シェルのコマンド列を、区切り記号で複数のコマンドに分ける。"""
    try:
        tokens = shlex.split(command, comments=True)
    except ValueError:
        return []
    segments, current = [], []
    for tok in tokens:
        if tok in SEPARATORS:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments


def parse_git(tokens):
    """git のコマンドなら (サブコマンド, 残りの引数) を返す。git でなければ None。"""
    if not tokens or os.path.basename(tokens[0]) != "git":
        return None
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in GIT_OPTS_WITH_VALUE:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        return tokens[i], tokens[i + 1:]
    return None


def has_flag(args, long_name, short_letter):
    """--force / -f / -fv のような短縮形の連結にも対応する。"""
    for a in args:
        if a == long_name:
            return True
        if a.startswith("-") and not a.startswith("--") and short_letter in a[1:]:
            return True
    return False


def split_pathspecs(args):
    """引数をフラグとパス指定に分ける。`--` 以降は全部パス指定。"""
    paths, seen_ddash = [], False
    for a in args:
        if seen_ddash:
            paths.append(a)
        elif a == "--":
            seen_ddash = True
        elif not a.startswith("-"):
            paths.append(a)
    return paths or ["."]


def targets_for_add(args, cwd):
    """`git add` で新しくステージに乗りうるファイルを、読み取り専用で列挙する。"""
    pathspecs = split_pathspecs(args)
    force = has_flag(args, "--force", "f")
    # -u / --update は追跡中の変更だけが対象。未追跡ファイルは巻き込まない
    update_only = has_flag(args, "--update", "u") and not has_flag(args, "--all", "A")

    flags = ["-m"] if update_only else ["-o", "-m"]
    found = lines(run(["git", "ls-files"] + flags + ["--exclude-standard", "--"] + pathspecs, cwd))
    if force:
        # -f は .gitignore を無視するので、ignore 済みファイルも入りうる
        found += lines(run(["git", "ls-files", "-o", "-i", "--exclude-standard", "--"] + pathspecs,
                           cwd))
    return found


def targets_for_commit(args, cwd):
    """`git commit` の対象。ステージ済みと、-a なら変更済みの追跡ファイル。"""
    found = lines(run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"], cwd))
    if has_flag(args, "--all", "a"):
        found += lines(run(["git", "ls-files", "-m", "--exclude-standard"], cwd))
    return found


def collect_targets(command, cwd):
    """このコマンドで git に入りうるパスを集める。"""
    found = []
    for tokens in split_segments(command):
        parsed = parse_git(tokens)
        if parsed is None:
            continue
        sub, args = parsed
        if sub in ("add", "stage"):
            found += targets_for_add(args, cwd)
        elif sub == "commit":
            found += targets_for_commit(args, cwd)
    return found


def size_of(path, cwd, root):
    """パスの実サイズ。cwd 基準 -> リポジトリルート基準 の順で探す。"""
    for base in (cwd, root):
        if not base:
            continue
        full = os.path.join(base, path)
        if os.path.isfile(full):
            try:
                return os.path.getsize(full)
            except OSError:
                return None
    return None


def decide(items, event, decision):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": event,
            "permissionDecision": decision,
            "permissionDecisionReason": items,
        }
    }, ensure_ascii=False))


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if payload.get("tool_name") != "Bash":
        sys.exit(0)
    command = (payload.get("tool_input") or {}).get("command") or ""
    if "git" not in command:
        sys.exit(0)

    cwd = payload.get("cwd") or os.getcwd()
    root = (run(["git", "rev-parse", "--show-toplevel"], cwd) or "").strip() or None

    found = collect_targets(command, cwd)
    if not found:
        sys.exit(0)

    deny_limit = DENY_MB * 1024 * 1024
    ask_limit = ASK_MB * 1024 * 1024
    denied, asked = [], []
    for path in sorted(set(found)):
        size = size_of(path, cwd, root)
        if size is None:
            continue
        mb = size / 1024 / 1024
        if size >= deny_limit:
            denied.append((path, mb))
        elif size >= ask_limit:
            asked.append((path, mb))

    if not denied and not asked:
        sys.exit(0)

    def listing(items):
        return "\n".join("  - {} ({:,.1f} MB)".format(p, m) for p, m in items)

    guidance = (
        "\n\ngit の履歴は全員に配られ、一度入った巨大ファイルは後から削除しても永久に残ります。"
        "\n対処:"
        "\n  1. 再取得・再生成できるデータなら sampledata/raw/ か sampledata/processed/ に置く"
        "\n     （どちらも .gitignore 済み）。git に入れるのは取得・生成の手順のほう。"
        "\n  2. 動作確認用なら sampledata/sample/ に小さい抜粋を作って、それをコミットする。"
        "\n  3. それでも本当に入れる必要があるなら .gitignore を見直したうえで、"
        "\n     環境変数 CARPRICE_GIT_DENY_MB で閾値を一時的に上げる。"
    )

    if denied:
        reason = ("{}MB 以上のファイルを git に追加しようとしています。中止しました。\n".format(DENY_MB)
                  + listing(denied) + guidance)
        decide(reason, "PreToolUse", "deny")
        # JSON 形式を解釈しないハーネスでも止まるよう、終了コード 2 でも拒否する
        sys.stderr.write(reason + "\n")
        sys.exit(2)

    reason = ("{}MB 以上のファイルが含まれています。本当に git に入れますか。\n".format(ASK_MB)
              + listing(asked) + guidance)
    decide(reason, "PreToolUse", "ask")
    sys.exit(0)


if __name__ == "__main__":
    main()
