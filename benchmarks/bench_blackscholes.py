"""
Benchmark an implementation of the Black–Scholes model.
"""

import math

import numpy as np

from numba import jit


# Taken from numba.tests.test_blackscholes

# XXX this data should be shared with bench_cuda.py
# (see https://github.com/spacetelescope/asv/issues/129)
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


@jit(nopython=True)
def cnd(d):
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


@jit(nopython=True)
def blackscholes(callResult, putResult, stockPrice, optionStrike,
                 optionYears, Riskfree, Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    for i in range(len(S)):
        sqrtT = math.sqrt(T[i])
        d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
        d2 = d1 - V * sqrtT
        cndd1 = cnd(d1)
        cndd2 = cnd(d2)

        expRT = math.exp((-1. * R) * T[i])
        callResult[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
        putResult[i] = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))


def setup():
    """
    Precompile jitted functions.
    """
    blackscholes(*args)


class BlackScholes:

    def time_blackscholes(self):
        for i in range(10):
            blackscholes(*args)
