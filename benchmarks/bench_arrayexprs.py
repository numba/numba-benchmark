"""
Benchmarks for array expressions.
"""

import numpy as np


# @jit(nopython=True)
def sum(a, b):
    return a + b

# @jit(nopython=True)
def sq_diff(a, b):
    return (a - b) * (a + b)

# @jit(nopython=True)
def rel_diff(a, b):
    return (a - b) / (a + b)

# @jit(nopython=True)
def square(a, b):
    # Note this is currently slower than `a ** 2 + b`, due to how LLVM
    # seems to lower the power intrinsic.  It's still faster than the naive
    # lowering as `exp(2 * log(a))`, though
    return a ** 2

# @jit(nopython=True)
def cube(a, b):
    return a ** 3


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
    def _binary_func(cls, func, dtype, fname):
        def f(self):
            f = getattr(self, func)
            f(*self.samples[dtype])
        f.__name__ = fname
        return f

    @classmethod
    def generate_benchmarks(cls):
        for dtype in cls.dtypes:
            for func in (sum, sq_diff, rel_diff, square, cube):
                fname = 'time_%s_%s' % (func.__name__, dtype)
                bench_func = cls._binary_func(func.__name__, dtype, fname)
                setattr(cls, fname, bench_func)

    def setup(self):
        from numba import jit

        jitter = jit(nopython=True)
        self.sum = jitter(sum)
        self.sq_diff = jitter(sq_diff)
        self.rel_diff = jitter(rel_diff)
        self.square = jitter(square)
        self.cube = jitter(cube)


ArrayExpressions.generate_benchmarks()
