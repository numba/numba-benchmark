"""CI planning: resolve Numba commits and emit the benchmark matrix."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = os.environ.get("NUMBA_REPO", "https://github.com/numba/numba.git")


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def remote_sha(rev):
    if re.fullmatch(r"[0-9a-f]{40}", rev):
        return rev
    output = subprocess.check_output(["git", "ls-remote", REPO, rev], text=True)
    refs = {}
    for line in output.splitlines():
        sha, name = line.split("\t")
        refs[name] = sha
    peeled = [sha for name, sha in refs.items() if name.endswith("^{}")]
    if peeled:
        return peeled[0]
    if len(refs) == 1:
        return next(iter(refs.values()))
    raise SystemExit(f"cannot resolve {rev!r} to a single Numba commit")


def clone_history(ref, directory):
    """Bare commit-only clone of the Numba history ending at ref."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        subprocess.run(
            ["git", "clone", "--bare", "--filter=tree:0", REPO, directory],
            check=True,
        )
        subprocess.run(["git", "fetch", "origin", ref], cwd=directory, check=True)
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "--bare",
                "--filter=tree:0",
                "--single-branch",
                "--branch",
                ref,
                REPO,
                directory,
            ],
            check=True,
        )


def commit_position(commit, cwd):
    """First-parent position of a commit, which orders results across runs."""
    return int(git("rev-list", "--count", "--first-parent", commit, cwd=cwd))


def pending(commits):
    return [
        commit
        for commit in commits
        if not Path(f"results/data/{commit}.json").exists()
    ]


def bounded_count(value, name, maximum=50):
    try:
        count = int(value)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {value!r}")
    return max(1, min(count, maximum))


def historic_range(from_ref, to_ref, ref, limit):
    with tempfile.TemporaryDirectory() as clone:
        clone_history(ref, clone)
        end = git("rev-parse", "--verify", f"{to_ref or ref}^{{commit}}", cwd=clone)
        if from_ref:
            start = git("rev-parse", "--verify", f"{from_ref}^{{commit}}", cwd=clone)
            commits = git(
                "rev-list", "--first-parent", "--reverse", f"{start}..{end}", cwd=clone
            ).splitlines()
            commits = pending(commits)[:limit]
        else:
            commits = git(
                "rev-list", "--first-parent", "--reverse", end, cwd=clone
            ).splitlines()
            commits = pending(commits)[-limit:]
        return [(commit, commit_position(commit, cwd=clone)) for commit in commits]


def cover(count, ref):
    with tempfile.TemporaryDirectory() as clone:
        clone_history(ref, clone)
        head = git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=clone)
        commits = git(
            "rev-list", "--first-parent", f"--max-count={count}", head, cwd=clone
        ).splitlines()
        oldest = commits[-1]
        try:
            parent = git("rev-parse", "--verify", f"{oldest}^", cwd=clone)
        except subprocess.CalledProcessError:
            parent = ""
    return head, parent, commits, pending(commits)


def plan():
    event = os.environ["EVENT"]
    from_ref = (os.environ.get("DISPATCH_FROM") or "").strip()
    to_ref = (os.environ.get("DISPATCH_TO") or "").strip()
    last = (os.environ.get("DISPATCH_LAST") or "").strip()
    ref = (os.environ.get("DISPATCH_REF") or "main").strip()

    if event == "workflow_dispatch" and (last or from_ref or to_ref):
        if last:
            limit = bounded_count(last, "last")
            from_ref = ""
        else:
            limit = bounded_count(os.environ.get("DISPATCH_LIMIT") or 10, "limit")
        return historic_range(from_ref, to_ref, ref, limit)

    sha = remote_sha(ref)
    if Path(f"results/data/{sha}.json").exists():
        return []
    with tempfile.TemporaryDirectory() as clone:
        clone_history(ref, clone)
        sha = git("rev-parse", "--verify", f"{sha}^{{commit}}", cwd=clone)
        position = commit_position(sha, cwd=clone)
    return [(sha, position)]


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python src/ci_utils.py plan | cover --count N --ref REF"
        )

    if sys.argv[1] == "cover":
        parser = argparse.ArgumentParser(prog="python src/ci_utils.py cover")
        parser.add_argument("--count", type=int, default=30)
        parser.add_argument("--ref", default="main")
        args = parser.parse_args(sys.argv[2:])
        head, parent, commits, missing = cover(args.count, args.ref)
        print(f"to={head}")
        print(f"from={parent}")
        print(f"count={len(commits)}")
        print(f"present={len(commits) - len(missing)}")
        print(f"missing_count={len(missing)}")
        print(f"commits={','.join(commits)}")
        print(f"missing={','.join(missing)}")
        return

    if len(sys.argv) != 2 or sys.argv[1] != "plan":
        raise SystemExit(
            "usage: python src/ci_utils.py plan | cover --count N --ref REF"
        )

    commits = plan()
    matrix = {
        "include": [
            {"commit": commit, "position": position} for commit, position in commits
        ]
    }
    print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")
    print(f"has_work={'true' if commits else 'false'}")
    print(f"commits={','.join(commit for commit, _ in commits)}")


if __name__ == "__main__":
    main()
