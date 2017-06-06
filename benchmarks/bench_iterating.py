"""
Benchmarks for the various ways of iterating over the values of an array.
"""

import numpy as np


# This choice of dtype is intentional.  It seems to allow better
# vectorization with LLVM 3.7 than either float32 or float64 (at least here
# on SandyBridge), and therefore helps reveal iterator inefficiences.
dtype = np.int32
zero = dtype(0)
N = 1000

arr1 = np.zeros(N * N, dtype=dtype)
arr2c = arr1.reshape((N, N))
arr2f = arr2c.T
arr2a = np.concatenate((arr2c, arr2c))[::2]
# 2d with a very small inner dimension
arr2c2 = arr1.reshape((N * N // 5, 5))
arr2f2 = arr2c2.copy(order='F')
arr2a2 = np.concatenate((arr2c2, arr2c2))[::2]


def setup():
    from numba import jit


    @jit(nopython=True)
    def array_iter_1d(arr):
        total = zero
        for val in arr:
            total += val
        return total

    @jit(nopython=True)
    def flat_iter(arr):
        total = zero
        for val in arr.flat:
            total += val
        return total

    @jit(nopython=True)
    def flat_index(arr):
        total = zero
        flat = arr.flat
        for i in range(arr.size):
            total += flat[i]
        return total

    @jit(nopython=True)
    def ndindex(arr):
        total = zero
        for ind in np.ndindex(arr.shape):
            total += arr[ind]
        return total

    @jit(nopython=True)
    def range1d(arr):
        total = zero
        n, = arr.shape
        for i in range(n):
            total += arr[i]
        return total

    @jit(nopython=True)
    def range2d(arr):
        total = zero
        m, n = arr.shape
        for i in range(m):
            for j in range(n):
                total += arr[i, j]
        return total

    @jit(nopython=True)
    def nditer1(a):
        total = zero
        for u in np.nditer(a):
            total += u.item()
        return total

    @jit(nopython=True)
    def nditer2(a, b):
        total = zero
        for u, v in np.nditer((a, b)):
            total += u.item() * v.item()
        return total

    @jit(nopython=True)
    def nditer3(a, b, out):
        total = zero
        for u, v, res in np.nditer((a, b, out)):
            res.itemset(u.item() * v.item())
        return total

    @jit(nopython=True)
    def zip_iter(a, b):
        total = zero
        for u, v in zip(a, b):
            total += u * v
        return total

    @jit(nopython=True)
    def zip_flat(a, b):
        total = zero
        for u, v in zip(a.flat, b.flat):
            total += u * v
        return total

    globals().update(locals())

class MonoArrayIterators:

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

    def time_nditer_iter_1d(self):
        nditer1(arr1)

    def time_nditer_iter_2d_C(self):
        nditer1(arr2c)

    def time_nditer_iter_2d_C_small_inner_dim(self):
        nditer1(arr2c2)

    def time_nditer_iter_2d_fortran(self):
        nditer1(arr2f)

    def time_nditer_iter_2d_non_contiguous(self):
        nditer1(arr2a)

    # When the number of dimensions is known / hardcoded

    def time_array_iter_1d(self):
        array_iter_1d(arr1)

    def time_range_index_1d(self):
        range1d(arr1)

    def time_range_index_2d(self):
        range2d(arr2c)


class MultiArrayIterators:

    # These are the dimensions-agnostic iteration methods

    def time_nditer_two_1d(self):
        nditer2(arr1, arr1)

    def time_nditer_two_2d_C_C(self):
        nditer2(arr2c, arr2c)

    def time_nditer_two_2d_F_F(self):
        nditer2(arr2f, arr2f)

    def time_nditer_two_2d_F_C(self):
        nditer2(arr2f, arr2c)

    def time_nditer_two_2d_C_A(self):
        nditer2(arr2c, arr2a)

    def time_nditer_two_2d_A_A(self):
        nditer2(arr2a, arr2a)

    def time_nditer_two_2d_C_C_small_inner_dim(self):
        nditer2(arr2c2, arr2c2)

    def time_nditer_two_2d_F_F_small_inner_dim(self):
        nditer2(arr2f2, arr2f2)

    def time_nditer_two_2d_F_C_small_inner_dim(self):
        nditer2(arr2f2, arr2c2)

    def time_nditer_two_2d_C_A_small_inner_dim(self):
        nditer2(arr2c2, arr2a2)

    def time_zip_flat_two_1d(self):
        zip_flat(arr1, arr1)

    def time_zip_flat_two_2d_C_C(self):
        zip_flat(arr2c, arr2c)

    def time_zip_flat_two_2d_C_C_small_inner_dim(self):
        zip_flat(arr2c2, arr2c2)

    def time_zip_flat_two_2d_F_F(self):
        zip_flat(arr2f, arr2f)

    def time_zip_flat_two_2d_A_A(self):
        zip_flat(arr2a, arr2a)

    def time_nditer_three_2d_C_C_C(self):
        nditer3(arr2c, arr2c, arr2c)

    def time_nditer_three_2d_F_F_F(self):
        nditer3(arr2f, arr2f, arr2f)

    def time_nditer_three_2d_A_A_A(self):
        nditer3(arr2a, arr2a, arr2a)

    # When the number of dimensions is known / hardcoded

    def time_zip_iter_two_1d(self):
        zip_iter(arr1, arr1)
