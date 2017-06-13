"""
Benchmark array sorting.
"""

from __future__ import division

import math

import numpy as np

def setup():
    from numba import jit


    @jit(nopython=True)
    def real_sort(x):
        x.sort()

    def sort(x):
        # We *have* to do a copy, otherwise repeating the benchmark will
        # produce skewed results in the later iterations, as an already-sorted
        # array will be passed.
        # We prefer do the copying outside of the JITted function, as we want
        # to measure sort() performance, not the performance of Numba's copy().
        real_sort(x.copy())

    @jit(nopython=True)
    def argsort(x):
        return x.argsort()

    @jit(nopython=True)
    def median(x):
        return np.median(x)

    globals().update(locals())


class BaseArraySorting:

    def setup(self):
        s = 1.0
        e = 100.0
        self.sorted_array = np.linspace(s, e, self.n, dtype=self.dtype)
        self.triangle_array = np.concatenate([
            np.linspace(s, e, self.n // 4, dtype=self.dtype),
            np.linspace(e, s, 3 * self.n // 4, dtype=self.dtype)
            ])
        rnd = np.random.RandomState(42)
        self.random_array = rnd.uniform(s, e, self.n).astype(self.dtype)
        # Note the amount of duplicates depends on `e - s`, so those
        # values shouldn't be changed lightly.
        self.duplicates_array = np.floor(self.random_array)


class ArraySorting(BaseArraySorting):
    n = 100000
    # Reduce memory size to minimize possible cache aliasing effects and
    # other oddities.
    dtype = np.float32

    def setup(self):
        BaseArraySorting.setup(self)
        # Warm up
        dummy = np.arange(10, dtype=self.dtype)
        sort(dummy)
        argsort(dummy)

    def time_sort_sorted_array(self):
        """
        Sort an already sorted array.
        """
        sort(self.sorted_array)

    def time_sort_triangle_array(self):
        """
        Sort a "triangular" array: ascending then descending.
        """
        sort(self.triangle_array)

    def time_sort_random_array(self):
        """
        Sort a random array.
        """
        sort(self.random_array)

    def time_sort_duplicates_array(self):
        """
        Sort a random array with many duplicates.
        """
        sort(self.duplicates_array)

    def time_argsort_random_array(self):
        """
        Arg-sort a random array.
        """
        argsort(self.random_array)


class ArrayMedian(BaseArraySorting):
    n = 100000
    # Reduce memory size to minimize possible cache aliasing effects and
    # other oddities.
    dtype = np.float32

    def setup(self):
        BaseArraySorting.setup(self)
        # Warm up
        dummy = np.arange(10, dtype=self.dtype)
        median(dummy)

    def time_median_sorted_array(self):
        """
        Median of a sorted array.
        """
        median(self.sorted_array)

    def time_median_triangle_array(self):
        """
        Median of a "triangular" array: ascending then descending.
        """
        median(self.triangle_array)

    def time_median_random_array_even(self):
        """
        Median of an even-sized random array.
        """
        arr = self.random_array
        assert arr.size & 1 == 0
        median(arr)

    def time_median_random_array_odd(self):
        """
        Median of an odd-sized random array.
        """
        arr = self.random_array[:-1]
        assert arr.size & 1 == 1
        median(arr)

    def time_median_duplicates_array(self):
        """
        Median of a random array with many duplicates.
        """
        median(self.duplicates_array)
