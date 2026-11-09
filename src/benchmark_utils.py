"""Timing and measurement of the NumPy and Numba callables from the registry."""

import platform
import sys
import time

from cases import build_case, seed_for, select_cases

TARGET_NS = 50_000_000
REPEATS = 5


def time_call(fn, args, compiled):
    """Return (compile_ns, best_average_ns). Compile is the first-call time."""
    compile_ns = 0
    started = time.perf_counter_ns()
    fn(*args)
    first_ns = time.perf_counter_ns() - started
    if compiled:
        compile_ns = first_ns

    fn(*args)
    started = time.perf_counter_ns()
    fn(*args)
    single_ns = max(time.perf_counter_ns() - started, 1)
    inner = max(1, min(TARGET_NS // single_ns, 100_000))

    best = None
    for _ in range(REPEATS):
        started = time.perf_counter_ns()
        for _ in range(inner):
            fn(*args)
        average_ns = (time.perf_counter_ns() - started) / inner
        best = average_ns if best is None else min(best, average_ns)
    return compile_ns, int(best)


def collect_sysinfo():
    """The `numba --sys-json` payload, or an error record if it cannot be read."""
    try:
        from numba.misc.numba_sysinfo import get_sysinfo

        info = get_sysinfo()
    except Exception as exc:  # sysinfo must never fail a benchmark run
        return {"error": f"{type(exc).__name__}: {exc}"}
    for key in ("Start", "Start UTC"):
        value = info.get(key)
        if hasattr(value, "isoformat"):
            info[key] = value.isoformat()
    return info


def run_benchmarks():
    try:
        import llvmlite
        import numba
        import numpy as np
    except ImportError as exc:
        sys.exit(f"run requires numpy, numba, and scipy: {exc}")

    runnable, skipped = select_cases(numba.__version__)

    rows = []
    for case in runnable:
        numpy_fn, numba_fn, args = build_case(
            case, np.random.RandomState(seed_for(case.id))
        )
        _, numpy_ns = time_call(numpy_fn, args, compiled=False)
        compile_ns, numba_ns = time_call(numba_fn, args, compiled=True)
        rows.append(
            {
                "id": case.id,
                "order": case.order,
                "numpy_ns": numpy_ns,
                "numba_ns": numba_ns,
                "compile_ns": compile_ns,
                "speedup": round(numpy_ns / numba_ns, 3),
            }
        )
        print(
            f"{case.id:32s} numpy {numpy_ns / 1e6:9.3f} ms  "
            f"numba {numba_ns / 1e6:9.3f} ms  "
            f"{numpy_ns / numba_ns:6.2f}x",
            file=sys.stderr,
        )

    for skip in skipped:
        print(f"{skip['id']:32s} skipped: {skip['reason']}", file=sys.stderr)

    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
        "sysinfo": collect_sysinfo(),
    }
    return rows, skipped, environment
