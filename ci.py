"""CI planning: resolve Numba commits and emit the benchmark matrix."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from cases import ENVIRONMENTS

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


def historic_range(from_ref, to_ref, ref, limit):
    with tempfile.TemporaryDirectory() as clone:
        if re.fullmatch(r"[0-9a-f]{40}", ref):
            subprocess.run(
                ["git", "clone", "--bare", "--filter=blob:none", REPO, clone],
                check=True,
            )
            subprocess.run(["git", "fetch", "origin", ref], cwd=clone, check=True)
        else:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--bare",
                    "--filter=blob:none",
                    "--single-branch",
                    "--branch",
                    ref,
                    REPO,
                    clone,
                ],
                check=True,
            )
        start = git("rev-parse", "--verify", f"{from_ref}^{{commit}}", cwd=clone)
        end = git("rev-parse", "--verify", f"{to_ref}^{{commit}}", cwd=clone)
        commits = git(
            "rev-list", "--first-parent", "--reverse", f"{start}..{end}", cwd=clone
        ).splitlines()
    return [
        commit for commit in commits if not Path(f"results/{commit}.json").exists()
    ][:limit]


def plan():
    event = os.environ["EVENT"]
    from_ref = (os.environ.get("DISPATCH_FROM") or "").strip()
    to_ref = (os.environ.get("DISPATCH_TO") or "").strip()
    ref = (os.environ.get("DISPATCH_REF") or "main").strip()
    call_ref = (os.environ.get("CALL_REF") or "").strip()

    if event == "workflow_call":
        if not call_ref:
            raise SystemExit("workflow_call requires numba-ref")
        return [remote_sha(call_ref)]
    if event == "workflow_dispatch" and from_ref and to_ref:
        limit = max(1, min(int(os.environ.get("DISPATCH_LIMIT") or 10), 50))
        return historic_range(from_ref, to_ref, ref, limit)

    sha = remote_sha(ref)
    return [] if Path(f"results/{sha}.json").exists() else [sha]


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "plan":
        raise SystemExit("usage: python ci.py plan")

    commits = plan()
    matrix = {
        "include": [
            {"commit": commit, **environment}
            for commit in commits
            for environment in ENVIRONMENTS
        ]
    }
    print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")
    print(f"has_work={'true' if commits else 'false'}")
    print(f"commits={','.join(commits)}")


if __name__ == "__main__":
    main()
