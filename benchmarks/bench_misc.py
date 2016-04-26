"""
Miscellaneous benchmarks.
"""

import subprocess
import sys
import time

import numpy as np

from numba import jit


@jit(nopython=True)
def grouped_sum(values, labels, target):
    for i in range(len(values)):
        idx = labels[i]
        target[idx] += values[i]


class InitializationTime:
    # Measure wall clock time, not CPU time of calling process
    timer = time.time
    number = 1
    repeat = 10

    def time_new_process_import_numba(self):
        subprocess.check_call([sys.executable, "-c", "from numba import jit"])


class IndirectIndexing:

    # The motivation for this benchmark stems from
    # https://github.com/numba/numba/pull/929

    def setup(self):
        n_in = 200000
        n_out = 20
        self.values = np.random.RandomState(0).randn(n_in)
        self.labels = np.random.RandomState(0).randint(n_out, size=n_in).astype('intp')
        self.unsigned_labels = self.labels.astype('uintp')
        self.targets = np.zeros(n_out)
        # Warm up JIT
        grouped_sum(self.values, self.labels, self.targets)
        grouped_sum(self.values, self.unsigned_labels, self.targets)

    def time_signed_indirect_indexing(self):
        self.targets[:] = 0
        grouped_sum(self.values, self.labels, self.targets)

    def time_unsigned_indirect_indexing(self):
        self.targets[:] = 0
        grouped_sum(self.values, self.unsigned_labels, self.targets)
