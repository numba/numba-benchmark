"""The original twelve benchmark cases, now registry entries."""

from . import case

N = 1_000_000
N_SORT = 200_000
N_DOT = 256


@case("sum/float64/C")
def sum_float64_c(rng):
    import numpy as np

    from ._kernels import nb_sum

    x = rng.rand(N)
    return np.sum, nb_sum, (x,)


@case("sum/float32/strided")
def sum_float32_strided(rng):
    import numpy as np

    from ._kernels import nb_sum

    x = rng.rand(N).astype(np.float32)[::2]
    return np.sum, nb_sum, (x,)


@case("sum/int64/C")
def sum_int64_c(rng):
    import numpy as np

    from ._kernels import nb_sum

    x = rng.randint(0, 1000, N).astype(np.int64)
    return np.sum, nb_sum, (x,)


@case("sum/complex128/C")
def sum_complex128_c(rng):
    import numpy as np

    from ._kernels import nb_sum

    x = rng.rand(N) + 1j * rng.rand(N)
    return np.sum, nb_sum, (x,)


@case("mean/float64/F")
def mean_float64_f(rng):
    import numpy as np

    from ._kernels import nb_mean

    x = np.asfortranarray(rng.rand(1000, 1000))
    return np.mean, nb_mean, (x,)


@case("mean/float32/negative")
def mean_float32_negative(rng):
    import numpy as np

    from ._kernels import nb_mean

    x = rng.rand(N).astype(np.float32)[::-1]
    return np.mean, nb_mean, (x,)


@case("add/float64/C")
def add_float64_c(rng):
    import numpy as np

    from ._kernels import nb_add

    x = rng.rand(N)
    y = rng.rand(N)
    return np.add, nb_add, (x, y)


@case("dot/float64/C")
def dot_float64_c(rng):
    import numpy as np

    from ._kernels import nb_dot

    a = rng.rand(N_DOT, N_DOT)
    b = rng.rand(N_DOT, N_DOT)
    return np.dot, nb_dot, (a, b)


@case("sort/float64/C")
def sort_float64_c(rng):
    import numpy as np

    from ._kernels import nb_sort

    x = rng.rand(N_SORT)
    return np.sort, nb_sort, (x,)


@case("cumsum/float64/C")
def cumsum_float64_c(rng):
    import numpy as np

    from ._kernels import nb_cumsum

    x = rng.rand(N)
    return np.cumsum, nb_cumsum, (x,)


@case("where/float64/C")
def where_float64_c(rng):
    import numpy as np

    from ._kernels import nb_where

    cond = rng.rand(N) > 0.5
    x = rng.rand(N)
    y = rng.rand(N)
    return np.where, nb_where, (cond, x, y)


@case("searchsorted/float64/C")
def searchsorted_float64_c(rng):
    import numpy as np

    from ._kernels import nb_searchsorted

    haystack = np.sort(rng.rand(N_SORT))
    needles = rng.rand(N_SORT)
    return np.searchsorted, nb_searchsorted, (haystack, needles)
