"""Timing and measurement of the NumPy and Numba callables from cases.py."""

import platform
import sys
import time

from cases import build_cases

TARGET_NS = 50_000_000
REPEATS = 5


def time_call(fn, compiled):
    """Return (compile_ns, best_average_ns). Compile is the first-call time."""
    compile_ns = 0
    started = time.perf_counter_ns()
    fn()
    first_ns = time.perf_counter_ns() - started
    if compiled:
        compile_ns = first_ns

    fn()
    started = time.perf_counter_ns()
    fn()
    single_ns = max(time.perf_counter_ns() - started, 1)
    inner = max(1, min(TARGET_NS // single_ns, 100_000))

    best = None
    for _ in range(REPEATS):
        started = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        average_ns = (time.perf_counter_ns() - started) / inner
        best = average_ns if best is None else min(best, average_ns)
    return compile_ns, int(best)


def run_benchmarks():
    try:
        import llvmlite
        import numba
        import numpy as np
    except ImportError as exc:
        sys.exit(f"run requires numpy, numba, and scipy: {exc}")

    rows = []
    for case_id, numpy_fn, numba_fn in build_cases():
        _, numpy_ns = time_call(numpy_fn, compiled=False)
        compile_ns, numba_ns = time_call(numba_fn, compiled=True)
        rows.append(
            {
                "id": case_id,
                "numpy_ns": numpy_ns,
                "numba_ns": numba_ns,
                "compile_ns": compile_ns,
                "speedup": round(numpy_ns / numba_ns, 3),
            }
        )
        print(
            f"{case_id:24s} numpy {numpy_ns / 1e6:9.3f} ms  "
            f"numba {numba_ns / 1e6:9.3f} ms  "
            f"{numpy_ns / numba_ns:6.2f}x",
            file=sys.stderr,
        )

    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
    }
    return rows, environment
