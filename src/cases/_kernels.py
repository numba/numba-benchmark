"""Numba kernels imported from case bodies after the version gate."""

import numpy as np
from numba import njit


@njit
def nb_sum(x):
    return np.sum(x)


@njit
def nb_mean(x):
    return np.mean(x)


@njit
def nb_add(x, y):
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = x[i] + y[i]
    return out


@njit
def nb_dot(a, b):
    return np.dot(a, b)


@njit
def nb_sort(x):
    return np.sort(x)


@njit
def nb_cumsum(x):
    out = np.empty_like(x)
    acc = 0.0
    for i in range(x.size):
        acc += x[i]
        out[i] = acc
    return out


@njit
def nb_where(cond, x, y):
    return np.where(cond, x, y)


@njit
def nb_searchsorted(a, v):
    return np.searchsorted(a, v)


@njit
def nb_sum_axis(x, axis):
    return np.sum(x, axis=axis)


@njit
def nb_sum_pair(a, b):
    return np.sum(a) + np.sum(b)


@njit
def nb_cumsum_flat(x):
    return np.cumsum(x)


@njit
def nb_cumsum_axis(x, axis):
    return np.cumsum(x, axis=axis)
