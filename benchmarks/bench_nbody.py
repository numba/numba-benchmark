"""
Benchmark an implementation of the N-body simulation.

As in the CUDA version, we only compute accelerations and don't care to
update speeds and positions.
"""

from __future__ import division

import math
import sys

import numpy as np

from numba import jit, float32, float64


eps_2 = np.float32(1e-6)
zero = np.float32(0.0)
one = np.float32(1.0)


@jit
def run_numba_nbody(positions, weights):
    accelerations = np.zeros_like(positions)
    n = weights.shape[0]
    for i in range(n):
        ax = zero
        ay = zero
        for j in range(n):
            rx = positions[j,0] - positions[i,0]
            ry = positions[j,1] - positions[i,1]
            sqr_dist = rx * rx + ry * ry + eps_2
            sixth_dist = sqr_dist * sqr_dist * sqr_dist
            inv_dist_cube = one / math.sqrt(sixth_dist)
            s = weights[j] * inv_dist_cube
            ax += s * rx
            ay += s * ry
        accelerations[i,0] = ax
        accelerations[i,1] = ay
    return accelerations


def run_numpy_nbody(positions, weights):
    accelerations = np.zeros_like(positions)
    n = weights.size
    for j in range(n):
        # Compute influence of j'th body on all bodies
        r = positions[j] - positions
        rx = r[:,0]
        ry = r[:,1]
        sqr_dist = rx * rx + ry * ry + eps_2
        sixth_dist = sqr_dist * sqr_dist * sqr_dist
        inv_dist_cube = one / np.sqrt(sixth_dist)
        s = weights[j] * inv_dist_cube
        accelerations += (r.transpose() * s).transpose()
    return accelerations


def make_nbody_samples(n_bodies):
    positions = np.random.RandomState(0).uniform(-1.0, 1.0, (n_bodies, 2))
    weights = np.random.RandomState(0).uniform(1.0, 2.0, n_bodies)
    return positions.astype(np.float32), weights.astype(np.float32)


class NBody:
    n_bodies = 4096

    def setup(self):
        # Sanity check our implementation
        p, w = make_nbody_samples(10)
        numba_res = run_numba_nbody(p, w)
        numpy_res = run_numpy_nbody(p, w)
        assert np.allclose(numba_res, numpy_res, 1e-4), (numba_res, numpy_res)
        # Actual benchmark samples
        self.positions, self.weights = make_nbody_samples(self.n_bodies)

    def time_numba_nbody(self):
        run_numba_nbody(self.positions, self.weights)

