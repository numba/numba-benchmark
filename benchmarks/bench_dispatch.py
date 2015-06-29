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
    'array_1d': np.zeros(10, dtype=np.int64),
    'array_3d': np.zeros(20, dtype=np.float64).reshape(2, 2, 5),
    'array_records': np.zeros(10, dtype=rec_dtype),
    'recarray': np.recarray(10, dtype=rec_dtype),
    'tuple': (0.5, 1.0j, ()),
    'record': np.empty(1, dtype=rec_dtype)[0],
    'bytearray': bytearray(3),
    }

@jit(nopython=True)
def binary(x, y):
    pass

@jit(forceobj=True)
def binary_pyobj(x, y):
    pass


def setup():
    """
    Precompile jitted functions.  This will register many specializations
    to choose from.
    """
    for tp in samples.values():
        binary(tp, tp)
    binary_pyobj(object(), object())


class NoPythonDispatch:
    """
    Time dispatching to a jitted function's specializations based on argument
    types.
    This stresses two things:
    - the typing of arguments (from argument value to typecode)
    - the selection of the best specialization amongst all the known ones
    """

    # We repeat 1000 times so as to make the overhead of benchmark launching
    # negligible.

    @classmethod
    def generate_benchmarks(cls, names):
        for name in names:
            def timefunc(self, arg=samples[name]):
                func = binary
                for i in range(1000):
                    func(arg, arg)
            timefunc.__name__ = "time_dispatch_" + name
            setattr(cls, timefunc.__name__, timefunc)


NoPythonDispatch.generate_benchmarks(samples.keys())


class PyObjectDispatch:

    def time_dispatch_pyobject(self):
        x = object()
        for i in range(1000):
            binary_pyobj(x, x)
