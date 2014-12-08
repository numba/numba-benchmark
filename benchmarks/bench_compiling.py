"""
Benchmarks of compilation time.
"""

import numpy as np

from numba import jit


def no_op(x):
    return x

def mandel(x, y, max_iters):
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return 255

mandel_sig = "int32(float64, float64, int32)"

def force_obj(x):
    object()

def lift(x):
    # Outer needs object mode because of np.empty()
    a = np.empty((2, 3))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            # Inner is nopython-compliant
            a[i, j] = x
    return a


class NoPythonCompilation:

    def time_jit_noop(self):
        jit("int32(int32)", nopython=True)(no_op)

    def time_jit_mandel(self):
        jit(mandel_sig, nopython=True)(mandel)


class PyObjectCompilation:

    def time_jit_noop_fallback(self):
        jit("int32(int32)")(force_obj)

    def time_jit_noop_forceobj(self):
        jit("int32(int32)", forceobj=True)(force_obj)

    def time_jit_mandel_forceobj(self):
        jit(mandel_sig, forceobj=True)(mandel)


class LoopLiftedCompilation:

    def time_lift(self):
        # The only way to time the entire thing (including the inner
        # function formed by the loop) is to call the function wrapper.
        f = jit(lift)
        f(1.0)

