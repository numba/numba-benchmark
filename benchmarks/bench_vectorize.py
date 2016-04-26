"""
Benchmarks for ``@vectorize`` ufuncs.
"""

import numpy as np

from numba import vectorize


@vectorize(["float32(float32, float32)",
            "float64(float64, float64)",
            "complex64(complex64, complex64)",
            "complex128(complex128, complex128)"])
def mul(x, y):
    return x * y


@vectorize(["float32(float32, float32)",
            "float64(float64, float64)"])
def rel_diff(x, y):
    # XXX for float32 performance, we should write `np.float32(2)`, but
    # that's not the natural way to write this code...
    return 2 * (x - y) / (x + y)


class Vectorize:

    n = 10000
    dtypes = ('float32', 'float64', 'complex64', 'complex128')

    def setup(self):
        self.samples = {}
        self.out = {}
        for dtype in self.dtypes:
            self.samples[dtype] = np.linspace(0.1, 1, self.n, dtype=dtype)
            self.out[dtype] = np.zeros(self.n, dtype=dtype)

    def _binary_func(func, dtype):
        def f(self):
            func(self.samples[dtype], self.samples[dtype], self.out[dtype])
        return f

    for dtype in dtypes:
        locals()['time_mul_%s' % dtype] = _binary_func(mul, dtype)

    time_rel_diff_float32 = _binary_func(rel_diff, 'float32')
    time_rel_diff_float64 = _binary_func(rel_diff, 'float64')

    del _binary_func
