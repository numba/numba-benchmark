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


class TimeSuite:

    n = 10000
    dtypes = ('float32', 'float64', 'complex64', 'complex128')

    def setup(self):
        self.samples = {}
        self.out = {}
        for dtype in self.dtypes:
            self.samples[dtype] = np.linspace(0, 1, self.n, dtype=dtype)
            self.out[dtype] = np.zeros(self.n, dtype=dtype)

    def _binary_func(func, dtype):
        def f(self):
            func(self.samples[dtype], self.samples[dtype], self.out[dtype])
        return f

    for dtype in dtypes:
        locals()['time_mul_%s' % dtype] = _binary_func(mul, dtype)

    del _binary_func
