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


def clear_dispatcher(dispatcher):
    dispatcher._make_finalizer()()
    dispatcher._reset_overloads()
    dispatcher._cache.flush()
    dispatcher._can_compile = True


class BaseSuite:
    min_run_count = 5


class SortSuite(BaseSuite):

    def setup(self):
        self.tl = make_random_typed_list(SIZE)
        self.tl.sort()
        self.dispatcher = dispatcher_registry['cpu'](_sort.py_func)
        self.signature = Signature(none,
                                   [ListType(int64), none, boolean],
                                   None)
        clear_dispatcher(self.dispatcher)

    def time_execute_sort(self):
        self.tl.sort()

    def time_compile_sort(self):
        self.dispatcher.compile(self.signature)


class ConstructionSuite(BaseSuite):

    def setup(self):
        self.pl = make_random_python_list(SIZE)
        List(self.pl)

    def time_construct_from_python_list(self):
        List(self.pl)

    def time_construct_in_njit_function(self):
        make_random_typed_list(SIZE)


class ReductionSuite(BaseSuite):

    def setup(self):
        self.tl = make_random_typed_list(SIZE)

        def reduction_sum(tl):
            agg = 0
            for i in tl:
                agg += i
            return agg

        self.reduction_sum_fastmath = njit(reduction_sum, fastmath=True)
        self.reduction_sum_no_fastmath = njit(reduction_sum)

        self.reduction_sum_fastmath(self.tl)
        self.reduction_sum_no_fastmath(self.tl)

    def time_reduction_sum_fastmath(self):
        self.reduction_sum_fastmath(self.tl)

    def time_reduction_sum_no_fastmath(self):
        self.reduction_sum_no_fastmath(self.tl)
