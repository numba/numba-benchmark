"""
Miscellaneous benchmarks.
"""

import subprocess
import sys
import time

from numba import jit


class InitializationTime:
    # Measure wall clock time, not CPU time of calling process
    timer = time.time
    number = 1
    repeat = 10

    def time_new_process_import_numba(self):
        subprocess.check_call([sys.executable, "-c", "from numba import jit"])
