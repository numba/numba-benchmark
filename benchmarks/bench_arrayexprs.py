"""
Benchmarks for array expressions.
"""

import numpy as np

from numba import jit


@jit(nopython=True)
def sum(a, b):
    return a + b

@jit(nopython=True)
def sq_diff(a, b):
    return (a - b) * (a + b)

@jit(nopython=True)
def rel_diff(a, b):
    return (a - b) / (a + b)


def setup():
    ArrayExpressions.setupClass()


class ArrayExpressions:

    n = 100000
    dtypes = ('float32', 'float64')

    @classmethod
    def setupClass(cls):
        cls.samples = {}
        random = np.random.RandomState(0)
        for dtype in cls.dtypes:
            arrays = [random.uniform(1.0, 2.0, size=cls.n).astype(dtype)
                      for i in range(2)]
            cls.samples[dtype] = arrays

    @classmethod
    def _binary_func(cls, func, dtype):
        def f(self):
            func(*self.samples[dtype])
        return f

    @classmethod
    def generate_benchmarks(cls):
        for dtype in cls.dtypes:
            for func in (sum, sq_diff, rel_diff):
                bench_func = cls._binary_func(func, dtype)
                setattr(cls, 'time_%s_%s' % (func.__name__, dtype), bench_func)


ArrayExpressions.generate_benchmarks()
