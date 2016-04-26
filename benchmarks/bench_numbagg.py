"""
Benchmark some functions from the `numbagg` project:
https://github.com/shoyer/numbagg
"""

import numpy as np

from numba import jit, guvectorize


n = 10000

no_nans = np.random.RandomState(0).randn(n)
some_nans = no_nans.copy()
some_nans[some_nans < -1] = np.nan
some_nans_2d = some_nans.reshape((100, n // 100))
some_nans_2d_reversed = some_nans_2d[::-1]


@jit(nopython=True)
def nanmean(a):
    asum = 0.0
    count = 0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > 0:
        return asum / count
    else:
        return np.nan


@guvectorize(['void(float64[:], float64[:])'], '(n)->()')
def gu_nanmean(a, res):
    asum = 0.0
    count = 0
    for ai in a:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > 0:
        res[0] = asum / count
    else:
        res[0] = np.nan


@guvectorize(['void(float64[:], intp[:], float64[:])'], '(n),()->(n)')
def move_nanmean(a, window_arr, out):
    window = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window - 1):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
        out[i] = np.nan
    i = window - 1
    ai = a[i]
    if ai == ai:
        asum += ai
        count += 1
    if count > 0:
        out[i] = asum / count
    else:
        out[i] = np.nan
    for i in range(window, len(a)):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
        aold = a[i - window]
        if aold == aold:
            asum -= aold
            count -= 1
        if count > 0:
            out[i] = asum / count
        else:
            out[i] = np.nan


def setup():
    """
    Precompile jitted functions.
    """
    nanmean(some_nans)


class Numbagg:

    def time_nanmean_jit_1d(self):
        nanmean(some_nans)

    def time_nanmean_jit_2d(self):
        nanmean(some_nans_2d)

    def time_nanmean_jit_2d_reversed(self):
        nanmean(some_nans_2d_reversed)

    def time_nanmean_gufunc(self):
        gu_nanmean(some_nans_2d)

    def time_move_nanmean(self):
        arr = some_nans
        res = move_nanmean(arr, np.asarray(10))
        assert res.shape == arr.shape

    def time_move_nanmean_2d(self):
        arr = some_nans_2d
        res = move_nanmean(arr, np.asarray(10))
        assert res.shape == arr.shape

