"""
Benchmarks for jitclass method dispatching and call overhead
"""

import numpy as numpy


def setup():
    from numba import njit, jitclass, uint32

    @jitclass([('val', uint32)])
    class Box:
        def __init__(self, val):
            self.val = val

        def inc(self):
            self.val += 1

        @property
        def value(self):
            return self.val

        @value.setter
        def value(self, new_value):
            self.val = new_value

    @njit
    def method_call(N):
        b = Box(0)
        for i in range(N):
            b.inc()
        return b

    @njit
    def property_access(N):
        acc = 0
        b = Box(10)
        for i in range(N):
            acc += b.value

        return acc

    @njit
    def property_setting(N):
        b = Box(0)
        for i in range(N):
            b.value = i

        return b

    @njit
    def constructor_call(N):
        l = []
        for i in range(N):
            l.append(Box(i))

        return l

    globals().update(locals())


class JitClassDispatch:

    N = 10 ** 8
    funcs = [
            'constructor_call',
            'method_call',
            'property_access',
            'property_setting',
        ]

    def setup(self):
        for fn in self.funcs:
            globals()[fn](1)

    @classmethod
    def generate_benchmarks(cls):
        for fn in cls.funcs:
            fname = "time_{}".format(fn)
            def f(self):
                globals()[fn](self.N)
            f.__name__ = fname
            setattr(cls, fname, f)

JitClassDispatch.generate_benchmarks()
