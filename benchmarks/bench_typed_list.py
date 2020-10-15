"""
Benchmarks for numba.typed.List
"""

import numpy as np
from numba import njit
from numba.typed import List
from numba.typed.typedlist import _sort
from numba.core.registry import dispatcher_registry
from numba.core.typing import Signature
from numba.core.types import ListType, int64, float64, none, boolean, Array

SIZE = 10**5
SEED = 23

def check_getitem_unchecked():

    @njit
    def use_getitem_unchecked():
        tl = List((1, 2, 3, 4))
        return tl.getitem_unchecked(0)

    try:
        use_getitem_unchecked()
    except Exception:
        return False
    else:
        return True


have_getitem_unchecked = check_getitem_unchecked()


@njit
def make_random_typed_list_int(n):
    tl = List()
    np.random.seed(SEED)
    for i in range(n):
        tl.append(np.random.randint(0, 100))
    return tl

@njit
def make_random_typed_list_float(n):
    tl = List()
    np.random.seed(SEED)
    for i in range(n):
        tl.append(np.random.randn())
    return tl

@njit
def make_random_typed_list_array(n):
    tl = List()
    np.random.seed(SEED)
    for i in range(n):
        tl.append(np.random.randn(100, 100))
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
        self.tl = make_random_typed_list_int(SIZE)
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
        make_random_typed_list_int(SIZE)


class ReductionSuite(BaseSuite):

    def setup(self):
        raise NotImplementedError

    def post_setup(self):

        self.reduction_sum = self.define_function()

        self.reduction_sum_fastmath = njit(self.reduction_sum, fastmath=True)
        self.reduction_sum_no_fastmath = njit(self.reduction_sum)

        self.reduction_sum_fastmath(self.tl)
        self.reduction_sum_no_fastmath(self.tl)

        self.fastmath_dispatcher = dispatcher_registry['cpu'](
            self.reduction_sum_fastmath.py_func)
        self.fastmath_dispatcher.targetoptions['fastmath'] = True

        self.no_fastmath_dispatcher = dispatcher_registry['cpu'](
            self.reduction_sum_no_fastmath.py_func)

        self.signature = Signature(int64, [ListType(int64)], None)

        clear_dispatcher(self.fastmath_dispatcher)
        clear_dispatcher(self.no_fastmath_dispatcher)

    def time_execute_reduction_sum_fastmath(self):
        self.reduction_sum_fastmath(self.tl)

    def time_compile_reduction_sum_fastmath(self):
        self.fastmath_dispatcher.compile(self.signature)

    def time_execute_reduction_sum_no_fastmath(self):
        self.reduction_sum_no_fastmath(self.tl)

    def time_compile_reduction_sum_no_fastmath(self):
        self.no_fastmath_dispatcher.compile(self.signature)


class IteratorReductionSuiteInt(ReductionSuite):

    def setup(self):

        self.tl = make_random_typed_list_int(SIZE)
        self.post_setup()

    def define_function(self):

        def reduction_sum(tl):
            agg = 0
            for i in tl:
                agg += i
            return agg

        return reduction_sum

class IteratorReductionSuiteFloat(ReductionSuite):

    def setup(self):

        self.tl = make_random_typed_list_float(SIZE)
        self.post_setup()

    def define_function(self):

        def reduction_sum(tl):
            agg = 0.0
            for i in tl:
                agg += i
            return agg

        return reduction_sum

class ForLoopReductionSuiteInt(ReductionSuite):

    def setup(self):

        self.tl = make_random_typed_list_int(SIZE)
        self.post_setup()

    def define_function(self):

        def reduction_sum(tl):
            agg = 0
            length = len(tl)
            for i in range(length):
                agg += tl[i]
            return agg

        return reduction_sum

class ForLoopReductionSuiteFloat(ReductionSuite):

    def setup(self):

        self.tl = make_random_typed_list_float(SIZE)
        self.post_setup()

    def define_function(self):

        def reduction_sum(tl):
            agg = 0.0
            length = len(tl)
            for i in range(length):
                agg += tl[i]
            return agg

        return reduction_sum


class GetitemUncheckedLoopReductionSuiteInt(ReductionSuite):

    def setup(self):

        self.tl = make_random_typed_list_int(SIZE)
        self.post_setup()

    def define_function(self):

        def reduction_sum_regular(tl):
            agg = 0
            length = len(tl)
            for i in range(length):
                agg += tl[i]
            return agg

        def reduction_sum_unchecked(tl):
            agg = 0
            length = len(tl)
            for i in range(length):
                agg += tl.getitem_unchecked(i)
            return agg

        return reduction_sum_unchecked if have_getitem_unchecked else reduction_sum_regular

class GetitemUncheckedReductionSuiteFloat(ReductionSuite):

    def setup(self):

        self.tl = make_random_typed_list_float(SIZE)
        self.post_setup()

    def define_function(self):

        def reduction_sum_regular(tl):
            agg = 0.0
            length = len(tl)
            for i in range(length):
                agg += tl[i]
            return agg

        def reduction_sum_unchecked(tl):
            agg = 0.0
            length = len(tl)
            for i in range(length):
                agg += tl.getitem_unchecked(i)
            return agg

        return reduction_sum_unchecked if have_getitem_unchecked else reduction_sum_regular


class ArrayListSuite(ReductionSuite):

    def setup(self):

        self.tl = make_random_typed_list_array(SIZE/100)
        self.signature = Signature(float64, [ListType(Array(float64, 1, 'C'))], None)
        self.post_setup()

    def define_function(self):

        def array_reduction(tl):
            agg = 0.0
            for i in tl:
                agg += i.sum()
            return agg

        return array_reduction
