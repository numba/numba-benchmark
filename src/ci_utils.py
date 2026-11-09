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
    """Pending commits of a range, plus how many were left for a later run."""
    with tempfile.TemporaryDirectory() as clone:
        clone_history(ref, clone)
        end = git("rev-parse", "--verify", f"{to_ref or ref}^{{commit}}", cwd=clone)
        if from_ref:
            start = git("rev-parse", "--verify", f"{from_ref}^{{commit}}", cwd=clone)
            commits = git(
                "rev-list", "--first-parent", "--reverse", f"{start}..{end}", cwd=clone
            ).splitlines()
            missing = pending(commits)
            commits = missing[:limit]
        else:
            commits = git(
                "rev-list", "--first-parent", "--reverse", end, cwd=clone
            ).splitlines()
            missing = pending(commits)
            commits = missing[-limit:]
        remaining = len(missing) - len(commits)
        return [
            (commit, commit_position(commit, cwd=clone)) for commit in commits
        ], remaining


def last_commits(ref, limit):
    """The newest ``limit`` first-parent commits of ref, oldest first.

    Unlike a backfill this keeps the window fixed: commits that already have
    results are scheduled again so a rerun measures the same trailing window.
    The publish step then replaces their stored measurements with the newest
    ones instead of appending a second record.
    """
    with tempfile.TemporaryDirectory() as clone:
        clone_history(ref, clone)
        head = git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=clone)
        commits = git(
            "rev-list", "--first-parent", f"--max-count={limit}", head, cwd=clone
        ).splitlines()
        commits.reverse()
        return [(commit, commit_position(commit, cwd=clone)) for commit in commits], 0


def cover(count, ref, from_ref="", to_ref=""):
    """Commits an explicit from..to range, or the last count commits of ref."""
    if to_ref and not from_ref:
        raise SystemExit("--to requires --from; use --count for the last N commits")
    with tempfile.TemporaryDirectory() as clone:
        clone_history(ref, clone)
        head = git("rev-parse", "--verify", f"{to_ref or ref}^{{commit}}", cwd=clone)
        if from_ref:
            start = git("rev-parse", "--verify", f"{from_ref}^{{commit}}", cwd=clone)
            commits = git(
                "rev-list", "--first-parent", "--reverse", f"{start}..{head}",
                cwd=clone,
            ).splitlines()
            parent = start
        else:
            commits = git(
                "rev-list", "--first-parent", f"--max-count={count}", head, cwd=clone
            ).splitlines()
            parent = ""
            if commits:
                try:
                    parent = git(
                        "rev-parse", "--verify", f"{commits[-1]}^", cwd=clone
                    )
                except subprocess.CalledProcessError:
                    parent = ""
    return head, parent, commits, pending(commits)


def cover_commits(value):
    """Validate an explicit comma-separated list of full commit SHAs."""
    commits = [sha.strip() for sha in value.split(",") if sha.strip()]
    for commit in commits:
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise SystemExit(
                f"commit must be a full 40-character lowercase hex SHA, got {commit!r}"
            )
    return commits


def has_results():
    return any(Path("results/data").glob("*.json"))


def pending_head(ref, limit, window=50):
    """Unbenchmarked commits reachable from ref, oldest first.

    Covers the contiguous unmeasured run at the tip and any holes inside the
    most recent ``window`` first-parent commits, so a failed publish or a
    skipped commit does not strand older history.
    """
    sha = remote_sha(ref)
    with tempfile.TemporaryDirectory() as clone:
        clone_history(ref, clone)
        head = git("rev-parse", "--verify", f"{sha}^{{commit}}", cwd=clone)
        history = git("rev-list", "--first-parent", head, cwd=clone).splitlines()
        backlog = []
        for commit in history:
            if Path(f"results/data/{commit}.json").exists():
                break
            backlog.append(commit)
        recent = list(reversed(history[:window]))
        candidates = set(backlog) | set(pending(recent))
        commits = [commit for commit in reversed(history) if commit in candidates]
        if has_results():
            selected = commits[:limit]
        else:
            selected = commits[-limit:]
        remaining = len(commits) - len(selected)
        return [
            (commit, commit_position(commit, cwd=clone)) for commit in selected
        ], remaining


def plan_nightly():
    ref = (os.environ.get("NIGHTLY_REF") or "main").strip()
    limit = bounded_count(os.environ.get("NIGHTLY_LIMIT") or 10, "limit")
    return pending_head(ref, limit)


def plan_historic():
    from_ref = (os.environ.get("DISPATCH_FROM") or "").strip()
    to_ref = (os.environ.get("DISPATCH_TO") or "").strip()
    last = (os.environ.get("DISPATCH_LAST") or "").strip()
    ref = (os.environ.get("DISPATCH_REF") or "main").strip()

    if last:
        return last_commits(ref, bounded_count(last, "last"))

    limit = bounded_count(os.environ.get("DISPATCH_LIMIT") or 10, "limit")
    return historic_range(from_ref, to_ref, ref, limit)


USAGE = (
    "usage: python src/ci_utils.py plan nightly | plan historic | "
    "cover [--count N | --from REF --to REF | --commits SHA,...] --ref REF"
)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(USAGE)

    if sys.argv[1] == "cover":
        parser = argparse.ArgumentParser(prog="python src/ci_utils.py cover")
        parser.add_argument("--count", type=int, default=30)
        parser.add_argument("--from", dest="from_ref", default="")
        parser.add_argument("--to", dest="to_ref", default="")
        parser.add_argument("--ref", default="main")
        parser.add_argument("--commits", default="")
        args = parser.parse_args(sys.argv[2:])
        if args.commits:
            commits = cover_commits(args.commits)
            head = commits[-1] if commits else ""
            parent = commits[0] if commits else ""
            missing = pending(commits)
        else:
            head, parent, commits, missing = cover(
                args.count, args.ref, args.from_ref, args.to_ref
            )
        print(f"to={head}")
        print(f"from={parent}")
        print(f"count={len(commits)}")
        print(f"present={len(commits) - len(missing)}")
        print(f"missing_count={len(missing)}")
        print(f"commits={','.join(commits)}")
        print(f"missing={','.join(missing)}")
        return

    mode = sys.argv[2] if len(sys.argv) == 3 else ""
    if sys.argv[1] != "plan" or mode not in ("nightly", "historic"):
        raise SystemExit(USAGE)

    commits, remaining = (
        plan_nightly() if mode == "nightly" else plan_historic()
    )
    matrix = {
        "include": [
            {"commit": commit, "position": position} for commit, position in commits
        ]
    }
    print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")
    print(f"has_work={'true' if commits else 'false'}")
    print(f"commits={','.join(commit for commit, _ in commits)}")
    print(f"remaining={remaining}")
    if remaining:
        print(
            f"{remaining} pending commit(s) were not scheduled; rerun to continue",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
