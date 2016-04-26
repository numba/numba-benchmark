"""
Centered difference benchmark, adapted from
http://nbviewer.ipython.org/gist/ketch/ae87a94f4ef0793d5d52
"""

import numpy as np

from numba import jit


N = 250
u1 = np.random.rand(N * N)
D1 = np.zeros_like(u1)

u2c = u1.reshape((N, N))
D2c = np.zeros_like(u2c)

u2f = u2c.T
D2f = np.zeros_like(u2f)

u2a = np.concatenate((u2c, u2c))[::2]
D2a = np.zeros_like(u2a)

dx = 1.5


@jit(nopython=True)
def centered_difference_range1d(u, D, dx=1.):
    m, = u.shape
    for i in range(1, m - 1):
        D[i] = (u[i-1] + u[i+1] - 2.0*u[i]) / dx**2
    return D

@jit(nopython=True)
def centered_difference_range2d(u, D, dx=1.):
    m, n = u.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            D[i,j] = (u[i+1,j] + u[i,j+1] + u[i-1,j] + u[i,j-1] - 4.0*u[i,j]) / dx**2
    return D


class CenteredDifference:

    def time_centered_difference_1d(self):
        centered_difference_range1d(u1, D1, dx)

    def time_centered_difference_2d_C(self):
        centered_difference_range2d(u2c, D2c, dx)

    def time_centered_difference_2d_fortran(self):
        centered_difference_range2d(u2f, D2f, dx)

    def time_centered_difference_2d_non_contiguous(self):
        centered_difference_range2d(u2a, D2a, dx)

