"""Compile-time monitoring for the Numba kernels in a benchmark run.

Numba broadcasts a ``numba:compile`` event around every dispatcher
compilation, so listening for it measures the compiler rather than the first
call, which also includes the execution of the freshly compiled kernel.

Example:
    tracker = CompileTimeTracker()
    with tracker.monitor():
        kernel(*args)
    print(tracker.compile_time, tracker.compile_count)
"""

import contextlib
import time

from numba.core import event


class _CompileListener(event.Listener):
    """Sum the outermost ``numba:compile`` events seen while installed.

    Compilation can nest (a dispatcher may compile another one during type
    inference); only the outermost event is counted so its time is not
    counted twice.
    """

    def __init__(self):
        self._stack = []
        self.total = 0.0
        self.count = 0

    def on_start(self, evt):
        self._stack.append(time.perf_counter())

    def on_end(self, evt):
        if not self._stack:
            return
        started = self._stack.pop()
        if self._stack:
            return
        self.total += time.perf_counter() - started
        self.count += 1


class CompileTimeTracker:
    """Records Numba compile time and count for the ``monitor()`` block.

    ``compile_time`` is in seconds. A ``compile_count`` of zero after the
    block means the dispatcher was already compiled for those argument types,
    so there was no compilation to time.
    """

    def __init__(self):
        self.compile_time: float | None = None
        self.compile_count: int | None = None

    @contextlib.contextmanager
    def monitor(self):
        """Yield self; set ``compile_time`` and ``compile_count`` on exit."""
        listener = _CompileListener()
        with event.install_listener("numba:compile", listener):
            yield self
        self.compile_time = listener.total
        self.compile_count = listener.count
