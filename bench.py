"""Minimal NumPy-vs-Numba benchmark: measure, merge, compare, report.

Usage:
    python bench.py run --commit SHA --os linux --python 3.13 --numpy 2.2 --out FILE
    python bench.py merge DIR --out results/
    python bench.py compare --results results/ --commit SHA
    python bench.py report --results results/ --out site/index.html
    python bench.py summary --results results/ [--commits SHA[,SHA...]]
                            [--artifact-url URL] [--out FILE]
    python bench.py cases
    python bench.py backfill-order --results results/

Numba's BLAS-backed np.dot needs scipy in the environment being measured.
The registry is imported lazily by `cases` and `backfill-order` only, so
`merge`, `compare`, `report`, and `summary` run without numpy or numba.
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


def sort_key(text):
    return (text.lower(), text)


def case_sections(case_ids):
    groups = {}
    for case_id in case_ids:
        name = case_id.split("/", 1)[0] or "other"
        groups.setdefault(name, []).append(case_id)
    return [
        {"name": name, "cases": sorted(ids, key=sort_key)}
        for name, ids in sorted(groups.items(), key=lambda item: sort_key(item[0]))
    ]


def report_payload(documents):
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
    cases.sort(key=sort_key)

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

    return {
        "labels": [document["commit"][:10] for document in documents],
        "dates": [document["date"] for document in documents],
        "envs": environments,
        "cases": cases,
        "sections": case_sections(cases),
        "data": data,
        "skips": skips,
    }


def comparison_markdown(documents, commit, heading_level=3):
    index = next(
        (i for i, document in enumerate(documents) if document["commit"] == commit),
        None,
    )
    if index is None:
        return f"No results for {commit}."

    document = documents[index]
    orders = case_orders(documents)
    keys = sorted(document["measurements"], key=env_order)
    if not keys:
        return "No measurements."

    heading = "#" * heading_level
    lines = []
    for key in keys:
        entry = document["measurements"][key]
        baseline = None
        for previous in reversed(documents[:index]):
            if key in previous["measurements"]:
                baseline = previous
                break

        lines.append(f"{heading} {key}")
        if baseline is None:
            lines.append("No earlier result for this environment.\n")
            continue

        lines.append(f"Baseline: `{baseline['commit'][:10]}`\n")
        previous_cases = {
            case["id"]: case for case in baseline["measurements"][key]["cases"]
        }
        lines.append("| Case | NumPy ms | Numba ms | Speedup | Numba change |")
        lines.append("|---|---:|---:|---:|---:|")
        for case in sorted(
            entry["cases"], key=lambda case: case_key(case["id"], orders)
        ):
            case_id = case["id"]
            old = previous_cases.get(case_id)
            change = percent_change(case["numba_ns"], old["numba_ns"]) if old else "new"
            lines.append(
                f"| {case_id} | {milliseconds(case['numpy_ns'])} | "
                f"{milliseconds(case['numba_ns'])} | {case['speedup']:.2f}x | {change} |"
            )
        lines.append("")

        skipped = entry.get("skipped") or []
        if skipped:
            lines.append("Skipped:")
            for skip in sorted(skipped, key=lambda skip: case_key(skip["id"], orders)):
                reason = skip.get("reason") or (
                    f"requires numba {skip['requires_numba']}"
                )
                lines.append(f"- `{skip['id']}`: {reason}")
            lines.append("")

    return "\n".join(lines).rstrip("\n")


def cmd_compare(args):
    documents = load_results(args.results)
    if not any(document["commit"] == args.commit for document in documents):
        sys.exit(f"no results for {args.commit} in {args.results}")
    print(comparison_markdown(documents, args.commit))


def cmd_report(args):
    documents = load_results(args.results)
    if args.commits:
        wanted = {sha.strip() for sha in args.commits.split(",") if sha.strip()}
        documents = [document for document in documents if document["commit"] in wanted]
    payload = report_payload(documents)
    summary = (
        f"{len(documents)} commit(s) &middot; {len(payload['envs'])} environment(s) "
        f"&middot; generated {utcnow()}"
    )
    template = Path(__file__).with_name("template.html").read_text()
    html = template.replace("__SUMMARY__", summary).replace(
        "__DATA__", json.dumps(payload)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{out}: {len(documents)} commit(s), {len(payload['cases'])} case(s)")


MAX_CHART_POINTS = 20


def summary_text(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "'")
    )


def mermaid_text(text):
    return " ".join(str(text).replace('"', "'").split())


def environment_label(key):
    return mermaid_text(key.replace("|", " "))


def speedup_text(value):
    return f"{round(value, 2)}"


def case_chart(payload, case_id):
    """Mermaid chart lines and notes for one case, or skip bullets."""
    envs = payload["envs"]
    data = payload["data"]
    labels = payload["labels"]

    def point(env, index):
        return data[env][case_id][index]

    skipped = {
        env: payload["skips"][env][case_id]
        for env in envs
        if case_id in payload["skips"].get(env, {})
    }
    skip_notes = [f"- {env}: {summary_text(reason)}" for env, reason in skipped.items()]
    chart_envs = [
        env
        for env in envs
        if env not in skipped and any(point(env, index) for index in range(len(labels)))
    ]
    indexes = sorted(
        {
            index
            for env in chart_envs
            for index in range(len(labels))
            if point(env, index)
        }
    )

    if not indexes:
        return skip_notes or ["No measurements."]

    title = mermaid_text(f"{summary_text(case_id)} — speedup")
    chart = ["```mermaid", "xychart-beta", f'  title "{title}"']
    lines = []
    notes = []

    if len(indexes) == 1:
        index = indexes[0]
        bar_envs = [env for env in chart_envs if point(env, index)]
        chart.append(
            "  x-axis ["
            + ", ".join(json.dumps(environment_label(env)) for env in bar_envs)
            + "]"
        )
        chart.append('  y-axis "Speedup (NumPy / Numba)"')
        chart.append(
            "  bar ["
            + ", ".join(speedup_text(point(env, index)["speedup"]) for env in bar_envs)
            + "]"
        )
        lines = [environment_label(env) for env in bar_envs]
    else:
        axis = indexes[-MAX_CHART_POINTS:]
        chart.append(
            "  x-axis [" + ", ".join(json.dumps(labels[index]) for index in axis) + "]"
        )
        chart.append('  y-axis "Speedup (NumPy / Numba)"')
        for env in chart_envs:
            missing = [index for index in indexes if not point(env, index)]
            if missing:
                notes.append(
                    f"- not measured: {environment_label(env)} at "
                    + ", ".join(labels[index] for index in missing)
                )
                continue
            chart.append(
                f'  line "{environment_label(env)}" ['
                + ", ".join(
                    speedup_text(point(env, index)["speedup"]) for index in axis
                )
                + "]"
            )
            lines.append(environment_label(env))
        if not lines:
            return skip_notes + notes or ["No measurements."]

    chart.append("```")
    caption = "Lines: " + ", ".join(lines)
    if len(indexes) > MAX_CHART_POINTS:
        caption += f" — showing last {MAX_CHART_POINTS} of {len(indexes)} commits"
    result = ["\n".join(chart)]
    if skip_notes or notes:
        result += [""] + skip_notes + notes
    result += ["", summary_text(caption)]
    return result


def summary_markdown(documents, planned=None, artifact_url=""):
    planned = planned or []
    selected = (
        [document for document in documents if document["commit"] in planned]
        if planned
        else documents
    )
    payload = report_payload(selected)
    count = (
        f"{len(selected)} of {len(planned)} commit(s)"
        if planned
        else f"{len(selected)} commit(s)"
    )
    lines = [f"## Numba benchmarks — {count}", ""]

    active = [
        env
        for env in payload["envs"]
        if any(
            point for case in payload["cases"] for point in payload["data"][env][case]
        )
    ]
    lines.append(f"{len(active)} environment(s) · generated {utcnow()}")
    if artifact_url:
        lines.append(
            f"[Full interactive report]({artifact_url}) "
            "(download and open `index.html`)"
        )

    missing_envs = [key for key in ENVIRONMENT_KEYS if key not in active]
    if missing_envs:
        lines.append(
            "Not measured: " + ", ".join(key.replace("|", " ") for key in missing_envs)
        )
    missing = [sha for sha in planned if sha not in {d["commit"] for d in documents}]
    if missing:
        lines.append(
            "Missing from results/: " + ", ".join(f"`{sha}`" for sha in missing)
        )
    lines.append("")

    for section in payload["sections"]:
        lines.append("<details open>")
        lines.append(
            f"<summary><b>{summary_text(section['name'])}</b> — "
            f"{len(section['cases'])} benchmark(s)</summary>"
        )
        lines.append("")
        for case_id in section["cases"]:
            lines.append(f"### {summary_text(case_id)}")
            lines.extend(case_chart(payload, case_id))
            lines.append("")
        lines.append("</details>")
        lines.append("")

    if selected:
        lines.append("<details><summary>Changes vs previous commit</summary>")
        lines.append("")
        for document in selected:
            lines.append(f"### {document['commit'][:10]}")
            lines.append(comparison_markdown(documents, document["commit"], 4))
            lines.append("")
        lines.append("</details>")

    return "\n".join(lines).rstrip("\n") + "\n"


def cmd_summary(args):
    documents = load_results(args.results)
    if not documents:
        text = "No results yet.\n"
    else:
        planned = [
            sha.strip() for sha in (args.commits or "").split(",") if sha.strip()
        ]
        text = summary_markdown(documents, planned, args.artifact_url)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    else:
        print(text, end="")


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
    report.add_argument("--commits", default="", help="comma-separated SHAs to include")
    report.add_argument("--out", default="site/index.html")
    report.set_defaults(func=cmd_report)

    summary = subparsers.add_parser("summary", help="write a Markdown run summary")
    summary.add_argument("--results", default="results/")
    summary.add_argument(
        "--commits", default="", help="comma-separated SHAs the run planned"
    )
    summary.add_argument("--artifact-url", default="")
    summary.add_argument("--out", default="")
    summary.set_defaults(func=cmd_summary)

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
