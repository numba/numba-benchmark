"""
Benchmarks for argument dispatching and call overhead of ``@jit`` functions.
"""

import numpy as np

from numba import jit


rec_dtype = np.dtype([('a', np.float64),
                      ('b', np.int32),
                      ('c', np.complex64),
                      ])

samples = {
    'bool': True,
    'int': 100000,
    'float': 0.5,
    'complex': 0.5 + 1.0j,
    'arr1': np.zeros(10, dtype=np.int64),
    'arr2': np.zeros(20, dtype=np.float64).reshape(2, 2, 5),
    'arr3': np.zeros(10, dtype=rec_dtype),
    'recarr': np.recarray(10, dtype=rec_dtype),
    }

@jit(nopython=True)
def binary(x, y):
    pass

@jit(forceobj=True)
def binary_pyobj(x, y):
    pass


def setup():
    for tp in samples.values():
        binary(tp, tp)
    binary_pyobj(object(), object())


class NoPythonDispatch:

    # We repeat 1000 times so as to make the overhead of benchmark launching
    # negligible.

    def time_dispatch_scalar(self):
        f = samples['float']
        for i in range(1000):
            binary(f, f)

    def time_dispatch_array_1d(self):
        arr = samples['arr1']
        for i in range(1000):
            binary(arr, arr)

    def time_dispatch_array_3d(self):
        arr = samples['arr2']
        for i in range(1000):
            binary(arr, arr)

    def time_dispatch_array_records(self):
        arr = samples['arr3']
        for i in range(1000):
            binary(arr, arr)

    def time_dispatch_recarray(self):
        arr = samples['recarr']
        for i in range(1000):
            binary(arr, arr)


class PyObjectDispatch:

    def time_dispatch_pyobject(self):
        x = object()
        for i in range(1000):
            binary_pyobj(x, x)
