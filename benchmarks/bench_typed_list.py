"""
Benchmarks for numba.typed.List
"""

import numpy as np
from numba import njit
from numba.typed import List
from numba.typed.typedlist import _sort
from numba.core.registry import dispatcher_registry
from numba.core.typing import Signature
from numba.core.types import ListType, int64, none, boolean

SIZE = 10**5
SEED = 23


@njit
def make_random_typed_list(n):
    tl = List()
    np.random.seed(SEED)
    for i in range(n):
        tl.append(np.random.randint(0, 100))
    return tl


def make_random_python_list(n):
    pl = list()
    np.random.seed(SEED)
    for i in range(n):
        pl.append(np.random.randint(0, 100))
    return pl


class SortSuite:

    def setup(self):
        self.tl = make_random_typed_list(SIZE)
        self.tl.sort()
        self.dispatcher = dispatcher_registry['cpu'](_sort.py_func)
        self.signature = Signature(none,
                                   [ListType(int64), none, boolean],
                                   None)

    def time_execute_sort(self):
        self.tl.sort()

    def time_compile_sort(self):
        self.dispatcher.compile(self.signature)


class ConstructionSuite:

    def setup(self):
        self.pl = make_random_python_list(SIZE)
        List(self.pl)

    def time_construct_from_python_list(self):
        List(self.pl)

    def time_construct_in_njit_function(self):
        make_random_typed_list(SIZE)


class ReductionSuite:

    def setup(self):
        self.tl = make_random_typed_list(SIZE)

        def reduction_sum(tl):
            agg = 0
            for i in tl:
                agg += i
            return agg

        self.reduction_sum = njit(reduction_sum)
        self.reduction_sum(self.tl)

    def time_reduction_sum(self):
        self.reduction_sum(self.tl)