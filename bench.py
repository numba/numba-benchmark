"""Minimal NumPy-vs-Numba benchmark: measure, merge, compare, report.

Usage:
    python bench.py run --commit SHA --os linux --python 3.13 --numpy 2.2 --out FILE
    python bench.py merge DIR --out results/
    python bench.py compare --results results/ --commit SHA
    python bench.py report --results results/ --out site/index.html
    python bench.py cases
    python bench.py backfill-order --results results/

Numba's BLAS-backed np.dot needs scipy in the environment being measured.
The registry is imported lazily by `cases` and `backfill-order` only, so
`merge`, `compare`, and `report` run without numpy or numba.
"""

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from env_matrix import ENVIRONMENT_KEYS


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
    rows, skipped, environment = run_benchmarks()

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
                "skipped": skipped,
            }
        },
    }
    if args.position is not None:
        document["position"] = args.position
    write_json(args.out, document)
    print(
        f"wrote {args.out} ({len(rows)} case(s), {len(skipped)} skipped)",
        file=sys.stderr,
    )


def document_order(document):
    """Numba history order, not benchmark run time, so backfills sort right."""
    position = document.get("position")
    if position is None:
        return (1, 0, document.get("date", ""), document["commit"])
    return (0, position, document.get("date", ""), document["commit"])


def load_results(directory):
    documents = []
    directory = Path(directory)
    if not directory.exists():
        return documents
    for path in sorted(directory.rglob("*.json")):
        document = json.loads(path.read_text())
        if "commit" in document and "measurements" in document:
            documents.append(document)
    documents.sort(key=document_order)
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
        if document.get("position") is not None:
            target.setdefault("position", document["position"])
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


def case_orders(documents):
    """Smallest order seen per case id across measured and skipped rows."""
    orders = {}
    for document in documents:
        for entry in document["measurements"].values():
            for case in entry.get("cases", []):
                if "order" in case:
                    orders[case["id"]] = min(
                        orders.get(case["id"], case["order"]), case["order"]
                    )
            for skip in entry.get("skipped", []):
                if "order" in skip:
                    orders[skip["id"]] = min(
                        orders.get(skip["id"], skip["order"]), skip["order"]
                    )
    return orders


def case_key(case_id, orders):
    return (orders.get(case_id, sys.maxsize), case_id)


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
    orders = case_orders(documents)
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
        for case in sorted(
            entry["cases"], key=lambda case: case_key(case["id"], orders)
        ):
            case_id = case["id"]
            old = previous_cases.get(case_id)
            change = percent_change(case["numba_ns"], old["numba_ns"]) if old else "new"
            print(
                f"| {case_id} | {milliseconds(case['numpy_ns'])} | "
                f"{milliseconds(case['numba_ns'])} | {case['speedup']:.2f}x | {change} |"
            )
        print()

        skipped = entry.get("skipped") or []
        if skipped:
            print("Skipped:")
            for skip in sorted(skipped, key=lambda skip: case_key(skip["id"], orders)):
                reason = skip.get("reason") or (
                    f"requires numba {skip['requires_numba']}"
                )
                print(f"- `{skip['id']}`: {reason}")
            print()


def cmd_report(args):
    documents = load_results(args.results)
    orders = case_orders(documents)

    environments = list(ENVIRONMENT_KEYS)
    cases = []
    skips = {}
    for document in documents:
        for key, entry in document["measurements"].items():
            if key not in environments:
                environments.append(key)
            for case in entry["cases"]:
                if case["id"] not in cases:
                    cases.append(case["id"])
            skipped = entry.get("skipped", [])
            for skip in skipped:
                if skip["id"] not in cases:
                    cases.append(skip["id"])
            # The newest measurement for an environment decides its skips.
            skips[key] = {
                skip["id"]: skip.get("reason")
                or f"requires numba {skip['requires_numba']}"
                for skip in skipped
            }
    environments.sort(key=env_order)
    cases.sort(key=lambda case: case_key(case, orders))

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
        "skips": skips,
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


def cmd_cases(args):
    from cases import iter_cases, select_cases

    cases = iter_cases()
    try:
        import numba
    except ImportError:
        status = {case.id: "unknown" for case in cases}
    else:
        runnable, skipped = select_cases(numba.__version__)
        status = {case.id: "ok" for case in runnable}
        status.update({skip["id"]: "skipped" for skip in skipped})

    print("order | id | requires_numba | status")
    for case in cases:
        print(
            f"{case.order} | {case.id} | "
            f"{case.requires_numba or '-'} | {status[case.id]}"
        )


def cmd_backfill_order(args):
    from cases import iter_cases

    orders = {case.id: case.order for case in iter_cases()}
    changed = 0
    for path in sorted(Path(args.results).rglob("*.json")):
        document = json.loads(path.read_text())
        if "commit" not in document or "measurements" not in document:
            continue
        touched = False
        for entry in document["measurements"].values():
            for case in entry.get("cases", []):
                if "order" not in case and case.get("id") in orders:
                    case["order"] = orders[case["id"]]
                    touched = True
        if touched:
            write_json(path, document)
            changed += 1
    print(f"backfilled {changed} result file(s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="measure one environment")
    run.add_argument("--commit", required=True)
    run.add_argument("--os", dest="os_name", required=True)
    run.add_argument("--python", dest="python_minor", required=True)
    run.add_argument("--numpy", dest="numpy_minor", required=True)
    run.add_argument(
        "--position",
        type=int,
        default=None,
        help="first-parent position of the commit in Numba history",
    )
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

    cases = subparsers.add_parser("cases", help="list registered cases")
    cases.set_defaults(func=cmd_cases)

    backfill = subparsers.add_parser(
        "backfill-order", help="write case order into existing results"
    )
    backfill.add_argument("--results", default="results/")
    backfill.set_defaults(func=cmd_backfill_order)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
