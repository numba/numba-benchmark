"""
Benchmark various operations on sets.
"""

from __future__ import division

import numpy as np


def setup():
    global unique, setops
    from numba import jit

    # Set benchmarks
    # Notes:
    # - unless we want to benchmark marshalling a set or list back to Python,
    #   we return a single value to avoid conversion costs

    @jit(nopython=True)
    def unique(seq):
        l = []
        seen = set()
        for v in seq:
            if v not in seen:
                seen.add(v)
                l.append(v)
        return l[-1]


    @jit(nopython=True)
    def setops(a, b):
        sa = set(a)
        sb = set(b)
        return len(sa & sb), len(sa | sb), len(sa ^ sb), len(sa - sb), len(sb - sa)


class IntegerSets:
    N = 100000
    dtype = np.int32

    def setup(self):
        self.rnd = np.random.RandomState(42)
        self.seq = self.duplicates_array(self.N)
        self.a = self.sparse_array(self.N)
        self.b = self.sparse_array(self.N)
        # Warm up
        self.run_unique(5)
        self.run_setops(5)

    def duplicates_array(self, n):
        """
        Get a 1d array with many duplicate values.
        """
        a = np.arange(int(np.sqrt(n)), dtype=self.dtype)
        # XXX rnd.choice() can take an integer to sample from arange()
        return self.rnd.choice(a, n)

    def sparse_array(self, n):
        """
        Get a 1d array with values spread around.
        """
        # Note two calls to sparse_array() should generate reasonable overlap
        a = np.arange(int(n ** 1.3), dtype=self.dtype)
        return self.rnd.choice(a, n)

    def run_unique(self, n):
        unique(self.seq[:n])

    def run_setops(self, n):
        setops(self.a[:n], self.b[:n])

    def time_unique(self):
        self.run_unique(self.N)

    def time_setops(self):
        self.run_setops(self.N)
