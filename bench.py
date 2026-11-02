"""Minimal NumPy-vs-Numba benchmark: measure, merge, compare, report.

Usage:
    python bench.py run --commit SHA --os linux --python 3.13 --numpy 2.2 --out FILE
    python bench.py merge DIR --out results/
    python bench.py compare --results results/ --commit SHA
    python bench.py report --results results/ --out site/index.html

Numba's BLAS-backed np.dot needs scipy in the environment being measured.
"""

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from cases import CASES, ENVIRONMENT_KEYS


def utcnow():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def env_key(os_name, python_minor, numpy_minor):
    return f"{os_name}|{python_minor}|{numpy_minor}"


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


def cmd_run(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        sys.exit("--commit must be a full 40-character lowercase hex SHA")

    from benchmark import run_benchmarks

    key = env_key(args.os_name, args.python_minor, args.numpy_minor)
    print(f"measuring {key}", file=sys.stderr)
    rows, environment = run_benchmarks()

    timestamp = utcnow()
    document = {
        "commit": args.commit,
        "date": timestamp,
        "measurements": {
            key: {
                "os": args.os_name,
                **environment,
                "timestamp": timestamp,
                "cases": rows,
            }
        },
    }
    write_json(args.out, document)
    print(f"wrote {args.out}", file=sys.stderr)


def load_results(directory):
    documents = []
    directory = Path(directory)
    if not directory.exists():
        return documents
    for path in sorted(directory.rglob("*.json")):
        document = json.loads(path.read_text())
        if "commit" in document and "measurements" in document:
            documents.append(document)
    documents.sort(key=lambda document: (document.get("date", ""), document["commit"]))
    return documents


def merge_documents(documents):
    merged = {}
    for document in documents:
        commit = document["commit"]
        date = document.get("date", "")
        target = merged.setdefault(
            commit, {"commit": commit, "date": date, "measurements": {}}
        )
        if date and (not target["date"] or date < target["date"]):
            target["date"] = date
        for key, entry in document["measurements"].items():
            current = target["measurements"].get(key)
            if current is None or entry.get("timestamp", "") >= current.get(
                "timestamp", ""
            ):
                target["measurements"][key] = entry
    return merged


def cmd_merge(args):
    documents = load_results(args.directory)
    merged = merge_documents(documents)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not merged:
        print("no result artifacts found")
        return

    for commit, document in sorted(merged.items()):
        path = out_dir / f"{commit}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            document = merge_documents([existing, document])[commit]
        write_json(path, document)
        print(f"{path}: {len(document['measurements'])} measurement(s)")


def case_order(case_id):
    return CASES.index(case_id) if case_id in CASES else len(CASES)


def env_order(key):
    if key in ENVIRONMENT_KEYS:
        return ENVIRONMENT_KEYS.index(key)
    return len(ENVIRONMENT_KEYS)


def milliseconds(ns):
    return f"{ns / 1e6:.3f}"


def percent_change(current, base):
    return f"{(current - base) / base * 100:+.1f}%"


def cmd_compare(args):
    documents = load_results(args.results)
    index = next(
        (
            i
            for i, document in enumerate(documents)
            if document["commit"] == args.commit
        ),
        None,
    )
    if index is None:
        sys.exit(f"no results for {args.commit} in {args.results}")

    document = documents[index]
    keys = sorted(document["measurements"], key=env_order)
    if not keys:
        print("no measurements")
        return

    for key in keys:
        entry = document["measurements"][key]
        baseline = None
        for previous in reversed(documents[:index]):
            if key in previous["measurements"]:
                baseline = previous
                break

        print(f"### {key}")
        if baseline is None:
            print("No earlier result for this environment.\n")
            continue

        print(f"Baseline: `{baseline['commit'][:10]}`\n")
        previous_cases = {
            case["id"]: case for case in baseline["measurements"][key]["cases"]
        }
        print("| Case | NumPy ms | Numba ms | Speedup | Numba change |")
        print("|---|---:|---:|---:|---:|")
        for case in sorted(entry["cases"], key=lambda case: case_order(case["id"])):
            case_id = case["id"]
            old = previous_cases.get(case_id)
            change = percent_change(case["numba_ns"], old["numba_ns"]) if old else "new"
            print(
                f"| {case_id} | {milliseconds(case['numpy_ns'])} | "
                f"{milliseconds(case['numba_ns'])} | {case['speedup']:.2f}x | {change} |"
            )
        print()


def cmd_report(args):
    documents = load_results(args.results)

    environments = list(ENVIRONMENT_KEYS)
    cases = list(CASES)
    for document in documents:
        for key, entry in document["measurements"].items():
            if key not in environments:
                environments.append(key)
            for case in entry["cases"]:
                if case["id"] not in cases:
                    cases.append(case["id"])
    environments.sort(key=env_order)
    cases.sort(key=case_order)

    data = {env: {case: [] for case in cases} for env in environments}
    for document in documents:
        for env in environments:
            entry = document["measurements"].get(env)
            by_case = {case["id"]: case for case in entry["cases"]} if entry else {}
            for case in cases:
                point = by_case.get(case)
                data[env][case].append(
                    None
                    if point is None
                    else {
                        "speedup": point["speedup"],
                        "numpy_ns": point["numpy_ns"],
                        "numba_ns": point["numba_ns"],
                        "compile_ns": point["compile_ns"],
                    }
                )

    payload = {
        "labels": [document["commit"][:10] for document in documents],
        "dates": [document["date"] for document in documents],
        "envs": environments,
        "cases": cases,
        "data": data,
    }
    summary = (
        f"{len(documents)} commit(s) &middot; {len(environments)} environment(s) "
        f"&middot; generated {utcnow()}"
    )
    template = Path(__file__).with_name("template.html").read_text()
    html = template.replace("__SUMMARY__", summary).replace(
        "__DATA__", json.dumps(payload)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{out}: {len(documents)} commit(s), {len(cases)} case(s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="measure one environment")
    run.add_argument("--commit", required=True)
    run.add_argument("--os", dest="os_name", required=True)
    run.add_argument("--python", dest="python_minor", required=True)
    run.add_argument("--numpy", dest="numpy_minor", required=True)
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)

    merge = subparsers.add_parser("merge", help="combine per-environment artifacts")
    merge.add_argument("directory")
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=cmd_merge)

    compare = subparsers.add_parser(
        "compare", help="compare a commit with the previous one"
    )
    compare.add_argument("--results", required=True)
    compare.add_argument("--commit", required=True)
    compare.set_defaults(func=cmd_compare)

    report = subparsers.add_parser("report", help="build the Chart.js page")
    report.add_argument("--results", default="results/")
    report.add_argument("--out", default="site/index.html")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
