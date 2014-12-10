"""
Benchmarks for the CUDA backend.
"""

import math

import numpy as np

from numba import cuda, float32, float64


def addmul(x, y, out):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i >= x.shape[0]:
        return
    out[i] = x[i] + y[i] * math.fabs(x[i])

addmul_f32 = cuda.jit(argtypes=(float32[:], float32[:], float32[:]))(addmul)
addmul_f64 = cuda.jit(argtypes=(float64[:], float64[:], float64[:]))(addmul)


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
