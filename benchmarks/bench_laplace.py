"""
Benchmark Laplace equation solving.

From the Numpy benchmark suite, original code at
https://github.com/yarikoptic/numpy-vbench/commit/a192bfd43043d413cc5d27526a9b28ad343b2499
"""

import numpy as np

from numba import jit


dx = 0.1
dy = 0.1
dx2 = (dx * dx)
dy2 = (dy * dy)

@jit(nopython=True)
def laplace(N, Niter):
    u = np.zeros((N, N))
    u[0] = 1
    for i in range(Niter):
        u[1:(-1), 1:(-1)] = ((((u[2:, 1:(-1)] + u[:(-2), 1:(-1)]) * dy2) +
                              ((u[1:(-1), 2:] + u[1:(-1), :(-2)]) * dx2))
                             / (2 * (dx2 + dy2)))
    return u


class Laplace:
    N = 150
    Niter = 200

    def setup(self):
        # Warm up
        self.run_laplace(10, 10)

    def run_laplace(self, N, Niter):
        u = laplace(N, Niter)

    def time_laplace(self):
        self.run_laplace(self.N, self.Niter)
