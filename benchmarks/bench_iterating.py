"""
Benchmarks for the various ways of iterating over the values of an array.
"""

import numpy as np

from numba import jit

N = 500

arr1 = np.zeros(N * N, dtype=np.float64)
arr2c = arr1.reshape((N, N))
arr2f = arr2c.T
arr2a = np.concatenate((arr2c, arr2c))[::2]


# XXX: also add benchmarks iterating over two arrays in lock step?
# (e.g. to measure zip() overhead)


@jit(nopython=True)
def array_iter_1d(arr):
    total = 0.0
    for val in arr:
        total += val
    return total

@jit(nopython=True)
def flat_iter(arr):
    total = 0.0
    for val in arr.flat:
        total += val
    return total

@jit(nopython=True)
def flat_index(arr):
    total = 0.0
    flat = arr.flat
    for i in range(arr.size):
        total += flat[i]
    return total

@jit(nopython=True)
def ndindex(arr):
    total = 0.0
    for ind in np.ndindex(arr.shape):
        total += arr[ind]
    return total

@jit(nopython=True)
def range1d(arr):
    total = 0
    n, = arr.shape
    for i in range(n):
        total += arr[i]
    return total

@jit(nopython=True)
def range2d(arr):
    total = 0
    m, n = arr.shape
    for i in range(m):
        for j in range(n):
            total += arr[i, j]
    return total


class NumpyIterators:

    # These are the dimensions-agnostic iteration methods

    def time_flat_iter_1d(self):
        flat_iter(arr1)

    def time_flat_iter_2d_C(self):
        flat_iter(arr2c)

    def time_flat_iter_2d_fortran(self):
        flat_iter(arr2f)

    def time_flat_iter_2d_non_contiguous(self):
        flat_iter(arr2a)

    def time_flat_index_1d(self):
        flat_index(arr1)

    def time_flat_index_2d_C(self):
        flat_index(arr2c)

    def time_flat_index_2d_fortran(self):
        flat_index(arr2f)

    def time_flat_index_2d_non_contiguous(self):
        flat_index(arr2a)

    def time_ndindex_1d(self):
        ndindex(arr1)

    def time_ndindex_2d(self):
        ndindex(arr2c)

    # When the number of dimensions is known / hardcoded

    def time_array_iter_1d(self):
        array_iter_1d(arr1)

    def time_range_index_1d(self):
        range1d(arr1)

    def time_range_index_2d(self):
        range2d(arr2c)
