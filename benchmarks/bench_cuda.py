"""
Benchmarks for the CUDA backend.
"""

from __future__ import division

import math
import sys

import numpy as np

from numba import cuda, float32, float64


def addmul(x, y, out):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i >= x.shape[0]:
        return
    out[i] = x[i] + y[i] * math.fabs(x[i])

addmul_f32 = cuda.jit(argtypes=(float32[:], float32[:], float32[:]))(addmul)
addmul_f64 = cuda.jit(argtypes=(float64[:], float64[:], float64[:]))(addmul)


@cuda.jit(argtypes=())
def no_op():
    pass


# N-body simulation.  We actually only run the step which computes the
# accelerations from the positions and weights of the bodies (updating
# speeds and positions is relatively uninteresting).

# CUDA version adapted from http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html

eps_2 = np.float32(1e-6)
zero = np.float32(0.0)
one = np.float32(1.0)

@cuda.jit(argtypes=(float32, float32,
                    float32, float32, float32,
                    float32, float32),
          device=True, inline=True)
def body_body_interaction(xi, yi, xj, yj, wj, axi, ayi):
    """
    Compute the influence of body j on the acceleration of body i.
    """
    rx = xj - xi
    ry = yj - yi
    sqr_dist = rx * rx + ry * ry + eps_2
    sixth_dist = sqr_dist * sqr_dist * sqr_dist
    inv_dist_cube = one / math.sqrt(sixth_dist)
    s = wj * inv_dist_cube
    axi += rx * s
    ayi += ry * s
    return axi, ayi

@cuda.jit(argtypes=(float32, float32, float32, float32,
                    float32[:,:], float32[:]), device=True, inline=True)
def tile_calculation(xi, yi, axi, ayi, positions, weights):
    """
    Compute the contribution of this block's tile to the acceleration
    of body i.
    """
    for j in range(cuda.blockDim.x):
        xj = positions[j,0]
        yj = positions[j,1]
        wj = weights[j]
        axi, ayi = body_body_interaction(xi, yi, xj, yj, wj, axi, ayi)
    return axi, ayi


tile_size = 128

# Don't JIT this function at the top-level as it breaks until Numba 0.16.
def calculate_forces(positions, weights, accelerations):
    """
    Calculate accelerations produced on all bodies by mutual gravitational
    forces.
    """
    sh_positions = cuda.shared.array((tile_size, 2), float32)
    sh_weights = cuda.shared.array(tile_size, float32)
    i = cuda.grid(1)
    axi = 0.0
    ayi = 0.0
    xi = positions[i,0]
    yi = positions[i,1]
    for j in range(0, len(weights), tile_size):
        index = (j // tile_size) * cuda.blockDim.x + cuda.threadIdx.x
        sh_index = cuda.threadIdx.x
        sh_positions[sh_index,0] = positions[index,0]
        sh_positions[sh_index,1] = positions[index,1]
        sh_weights[sh_index] = weights[index]
        cuda.syncthreads()
        axi, ayi = tile_calculation(xi, yi, axi, ayi,
                                    sh_positions, sh_weights)
        cuda.syncthreads()
    accelerations[i,0] = axi
    accelerations[i,1] = ayi


class NBodyCUDARunner:

    def __init__(self, positions, weights):
        self.calculate_forces = cuda.jit(
            argtypes=(float32[:,:], float32[:], float32[:,:])
            )(calculate_forces)
        self.accelerations = np.zeros_like(positions)
        self.n_bodies = len(weights)
        self.stream = cuda.stream()
        self.d_pos = cuda.to_device(positions, self.stream)
        self.d_wei = cuda.to_device(weights, self.stream)
        self.d_acc = cuda.to_device(self.accelerations, self.stream)
        self.stream.synchronize()

    def run(self):
        blockdim = tile_size
        griddim = int(math.ceil(self.n_bodies / blockdim))
        self.calculate_forces[griddim, blockdim, self.stream](
            self.d_pos, self.d_wei, self.d_acc)
        self.stream.synchronize()

    def results(self):
        self.d_acc.copy_to_host(self.accelerations, self.stream)
        self.stream.synchronize()
        return self.accelerations


def run_cpu_nbody(positions, weights):
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


# Taken from numba.cuda.tests.cudapy.test_blackscholes

N = 16384

RISKFREE = 0.02
VOLATILITY = 0.30

A1 = 0.31938153
A2 = -0.356563782
A3 = 1.781477937
A4 = -1.821255978
A5 = 1.330274429
RSQRT2PI = 0.39894228040143267793994605993438

callResultGold = np.zeros(N)
putResultGold = np.zeros(N)

stockPrice = np.random.RandomState(0).uniform(5.0, 30.0, N)
optionStrike = np.random.RandomState(1).uniform(1.0, 100.0, N)
optionYears = np.random.RandomState(2).uniform(0.25, 10.0, N)

args = (callResultGold, putResultGold, stockPrice, optionStrike,
        optionYears, RISKFREE, VOLATILITY)


@cuda.jit(argtypes=(float64,), restype=float64, device=True, inline=True)
def cnd_cuda(d):
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


@cuda.jit(argtypes=(float64[:], float64[:], float64[:], float64[:], float64[:],
                    float64, float64))
def black_scholes_cuda(callResult, putResult, S, X, T, R, V):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i >= S.shape[0]:
        return
    sqrtT = math.sqrt(T[i])
    d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_cuda(d1)
    cndd2 = cnd_cuda(d2)

    expRT = math.exp((-1. * R) * T[i])
    callResult[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
    putResult[i] = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))


class Synthetic:
    """
    Micro-Benchmarks.
    """
    n = 4 * 256 * 1024

    def setup(self):
        self.stream = cuda.stream()
        self.f32 = np.zeros(self.n, dtype=np.float32)
        self.d_f32 = cuda.to_device(self.f32, self.stream)
        self.f64 = np.zeros(self.n, dtype=np.float64)
        self.d_f64 = cuda.to_device(self.f64, self.stream)
        self.stream.synchronize()

    def time_addmul_f32(self):
        blockdim = 512, 1
        griddim = int(math.ceil(float(self.n) / blockdim[0])), 1
        for i in range(10):
            addmul_f32[griddim, blockdim, self.stream](
                self.d_f32, self.d_f32, self.d_f32)
        self.stream.synchronize()

    def time_addmul_f64(self):
        blockdim = 512, 1
        griddim = int(math.ceil(float(self.n) / blockdim[0])), 1
        for i in range(10):
            addmul_f64[griddim, blockdim, self.stream](
                self.d_f64, self.d_f64, self.d_f64)
        self.stream.synchronize()

    def time_run_empty_kernel(self):
        no_op[1, 1, self.stream]()
        self.stream.synchronize()


class BlackScholes:

    def setup(self):
        self.stream = cuda.stream()
        self.d_callResult = cuda.to_device(callResultGold, self.stream)
        self.d_putResult = cuda.to_device(putResultGold, self.stream)
        self.d_stockPrice = cuda.to_device(stockPrice, self.stream)
        self.d_optionStrike = cuda.to_device(optionStrike, self.stream)
        self.d_optionYears = cuda.to_device(optionYears, self.stream)
        self.stream.synchronize()

    def time_blackscholes(self):
        blockdim = 512, 1
        griddim = int(math.ceil(float(N) / blockdim[0])), 1
        for i in range(10):
            black_scholes_cuda[griddim, blockdim, self.stream](
                self.d_callResult, self.d_putResult,
                self.d_stockPrice, self.d_optionStrike, self.d_optionYears,
                RISKFREE, VOLATILITY)
        self.stream.synchronize()


class NBody:
    n_bodies = 4096

    def setup(self):
        # Sanity check our implementation
        p, w = make_nbody_samples(tile_size * 2)
        runner = NBodyCUDARunner(p, w)
        runner.run()
        cuda_res = runner.results()
        cpu_res = run_cpu_nbody(p, w)
        assert np.allclose(cuda_res, cpu_res, 1e-4), (cuda_res, cpu_res)
        # Make actual benchmark samples and prepare data transfer
        self.positions, self.weights = make_nbody_samples(self.n_bodies)
        self.runner = NBodyCUDARunner(self.positions, self.weights)

    def time_cpu_nbody(self):
        run_cpu_nbody(self.positions, self.weights)

    def time_cuda_nbody(self):
        self.runner.run()


class DataTransfer:

    def setup(self):
        self.stream = cuda.stream()
        self.small_data = np.zeros(512, dtype=np.float64)
        self.large_data = np.zeros(512 * 1024, dtype=np.float64)
        self.d_small_data = cuda.to_device(self.small_data, self.stream)
        self.d_large_data = cuda.to_device(self.large_data, self.stream)
        self.stream.synchronize()

    def time_transfer_to_gpu_small(self):
        for i in range(10):
            cuda.to_device(self.small_data, self.stream)
        self.stream.synchronize()

    def time_transfer_to_gpu_large(self):
        for i in range(10):
            cuda.to_device(self.large_data, self.stream)
        self.stream.synchronize()

    def time_transfer_from_gpu_small(self):
        for i in range(10):
            self.d_small_data.copy_to_host(self.small_data, self.stream)
        self.stream.synchronize()

    def time_transfer_from_gpu_large(self):
        for i in range(10):
            self.d_large_data.copy_to_host(self.large_data, self.stream)
        self.stream.synchronize()
