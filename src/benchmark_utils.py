"""Timing and measurement of the NumPy and Numba callables from the registry.

Each case is measured in three separate steps:

1. compile: the first Numba call, timed with Numba's ``numba:compile``
   events (see ``compile_time``), so the reported compile time excludes
   executing the freshly compiled kernel;
2. warm-up: one untimed call, so the loop below times compiled code;
3. benchmark loop: ``REPEATS`` timed batches of ``inner`` calls each,
   keeping the fastest average.
"""

import math
import platform
import sys
import time

from cases import build_case, seed_for, select_cases

TARGET_NS = 50_000_000
REPEATS = 5
MAX_INNER = 100_000


def measure_compile(fn, args):
    """Compile ``fn(*args)`` and return ``(compile_ns, compile_count)``."""
    from compile_time import CompileTimeTracker

    tracker = CompileTimeTracker()
    with tracker.monitor():
        fn(*args)
    return int(tracker.compile_time * 1e9), tracker.compile_count


def calibrate(fn, args):
    """Number of calls that should take about ``TARGET_NS`` nanoseconds."""
    started = time.perf_counter_ns()
    fn(*args)
    single_ns = max(time.perf_counter_ns() - started, 1)
    return max(1, min(TARGET_NS // single_ns, MAX_INNER))


def measure(fn, args):
    """Best average ``fn(*args)`` time in ns over ``REPEATS`` timed batches."""
    fn(*args)

    inner = calibrate(fn, args)
    best_ns = math.inf

    # --- benchmark loop: REPEATS batches of `inner` calls -----------------
    for _ in range(REPEATS):
        started = time.perf_counter_ns()
        for _ in range(inner):
            fn(*args)
        best_ns = min(best_ns, (time.perf_counter_ns() - started) / inner)
    # ----------------------------------------------------------------------

    return int(best_ns)


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

        # The NumPy reference needs no compilation.
        numpy_ns = measure(numpy_fn, args)

        # The first Numba call compiles the kernel, timed through Numba's
        # events; measure() then warms it up before the benchmark loop.
        compile_ns, compile_count = measure_compile(numba_fn, args)
        numba_ns = measure(numba_fn, args)

        rows.append(
            {
                "id": case.id,
                "order": case.order,
                "numpy_ns": numpy_ns,
                "numba_ns": numba_ns,
                "compile_ns": compile_ns,
                "compile_count": compile_count,
                "speedup": round(numpy_ns / numba_ns, 3),
            }
        )
        print(
            f"{case.id:32s} numpy {numpy_ns / 1e6:9.3f} ms  "
            f"numba {numba_ns / 1e6:9.3f} ms  "
            f"{numpy_ns / numba_ns:6.2f}x  "
            f"compile {compile_ns / 1e6:9.3f} ms ({compile_count}x)",
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
