"""
Ising model benchmark, adapted from
http://matthewrocklin.com/blog/work/2015/02/28/Ising/
"""

from math import exp, log, e, sqrt

import numpy as np

from numba import jit


kT = 2 / log(1 + sqrt(2), e)

N = 200

random = np.random.RandomState(0)

x_start = random.randint(2, size=(N, N)).astype('i8')
x_start[x_start == 0] = -1

N_iterations = 10


@jit(nopython=True)
def _update(x, i, j):
    n, m = x.shape
    dE = 2* x[i, j] * (
                     x[(i-1)%n, (j-1)%m]
                   + x[(i-1)%n,  j     ]
                   + x[(i-1)%n, (j+1)%m]

                   + x[ i     , (j-1)%m]
                   + x[ i     , (j+1)%m]

                   + x[(i+1)%n, (j-1)%m]
                   + x[(i+1)%n,  j     ]
                   + x[(i+1)%n, (j+1)%m]
                   )
    if dE <= 0 or exp(-dE / kT) > np.random.random():
        x[i, j] *= -1

@jit(nopython=True)
def update(x):
    n, m = x.shape

    for i in range(n):
        for j in range(0, m, 2):  # Even columns first to avoid overlap
            _update(x, j, i)

    for i in range(n):
        for j in range(1, m, 2):  # Odd columns second to avoid overlap
            _update(x, j, i)


class IsingModel:

    def time_ising(self):
        x = x_start.copy()
        for i in range(N_iterations):
            update(x)

