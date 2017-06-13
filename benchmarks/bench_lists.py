"""
Benchmark various operations on lists.

Note: sorting has its own benchmark file.
"""

from __future__ import division



# List benchmarks
# Notes:
# - unless we want to benchmark marshalling a list back to Python,
#   we return a single value to avoid conversion costs
# - we don't want LLVM to optimize away the whole list creation, so
#   we make the returned value unknown at compile-time.

def setup():
    from numba import jit

    @jit(nopython=True)
    def list_append(n, i):
        l = []
        for v in range(n):
            l.append(v)
        return l[i]

    @jit(nopython=True)
    def list_extend(n, i):
        l = []
        l.extend(range(n // 2))
        l.extend(range(n // 2))
        return l[i]

    @jit(nopython=True)
    def list_call(n, i):
        l = list(range(n))
        return l[i]


    @jit(nopython=True)
    def list_return(n):
        return [0] * n

    @jit(nopython=True)
    def list_pop(n):
        l = list(range(n))
        v = 0
        while len(l) > 0:
            v = v ^ l.pop()
        return v

    @jit(nopython=True)
    def list_insert(n, i):
        l = [0]
        for v in range(n):
            l.insert(0, v)
        return l[i]

    globals().update(locals())

class ListConstruction:
    n = 100000

    def setup(self):
        # Warm up
        list_append(1, 0)
        list_extend(1, 0)
        list_call(1, 0)

    def time_list_append(self):
        list_append(self.n, 0)

    def time_list_extend(self):
        list_extend(self.n, 0)

    def time_list_call(self):
        list_call(self.n, 0)


class ListReturn:
    n = 100000

    def setup(self):
        # Warm up
        list_return(1)

    def time_list_return(self):
        list_return(self.n)


class ListMutation:
    n = 100000

    def setup(self):
        # Warm up
        list_pop(1)
        list_insert(1, 0)

    def time_list_pop(self):
        list_pop(self.n)

    def time_list_insert(self):
        # list.insert() is quadratic, so reduce the effort
        list_insert(self.n // 10, 0)
