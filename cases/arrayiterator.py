"""Cases that exercise Numba's ArrayIterator-based reductions.

``np.sum`` (flat, axis, tuple axis) and ``np.cumsum`` (flat, axis) are lowered
through ``numba.np.arraymath.ArrayIterator``. The cases cover the layout and
axis combinations that ArrayIterator's loop nest has regressed on: C and
Fortran flat walks, non-contiguous (A layout) flat walks, inner and outer axis
reductions, runtime axes, and cumulative sums.

Context: numba/numba#9679, #9819, #10625, #10657, #10850, #10866, #10898,
#10936.
"""

from . import case

N_PAIR = 1_000_000
N_SIDE = 1_000
N_CUBE = 64
SUM_AXIS_SINCE = ">=0.67"
CUMSUM_AXIS_SINCE = ">=0.67"


@case("sum/float64/F")
def sum_float64_f(rng):
    import numpy as np

    from ._kernels import nb_sum

    x = np.asfortranarray(rng.rand(N_SIDE, N_SIDE))
    return np.sum, nb_sum, (x,)


@case("sum/float32/F")
def sum_float32_f(rng):
    import numpy as np

    from ._kernels import nb_sum

    x = np.asfortranarray(rng.rand(N_SIDE, N_SIDE).astype(np.float32))
    return np.sum, nb_sum, (x,)


@case("sum/float32/C")
def sum_float32_c(rng):
    import numpy as np

    from ._kernels import nb_sum

    x = rng.rand(N_PAIR).astype(np.float32)
    return np.sum, nb_sum, (x,)


@case("sum/float64/A")
def sum_float64_a(rng):
    import numpy as np

    from ._kernels import nb_sum

    x = rng.rand(N_CUBE, 2 * N_CUBE, N_CUBE + N_CUBE // 2).transpose(2, 0, 1)
    return np.sum, nb_sum, (x,)


@case("sum/float64/3d/C")
def sum_float64_3d_c(rng):
    import numpy as np

    from ._kernels import nb_sum

    x = rng.rand(N_CUBE, N_CUBE, N_CUBE)
    return np.sum, nb_sum, (x,)


@case("sum/uint8/C")
def sum_uint8_c(rng):
    import numpy as np

    from ._kernels import nb_sum

    x = rng.randint(0, 100, N_PAIR).astype(np.uint8)
    return np.sum, nb_sum, (x,)


@case("sum/int64/axis0/C")
def sum_int64_axis0_c(rng):
    import numpy as np

    from ._kernels import nb_sum_axis

    x = rng.randint(0, 1000, (N_SIDE, N_SIDE)).astype(np.int64)
    return np.sum, nb_sum_axis, (x, 0)


@case("sum/int64/axis1/C")
def sum_int64_axis1_c(rng):
    import numpy as np

    from ._kernels import nb_sum_axis

    x = rng.randint(0, 1000, (N_SIDE, N_SIDE)).astype(np.int64)
    return np.sum, nb_sum_axis, (x, 1)


@case("sum/float64/axis0/F")
def sum_float64_axis0_f(rng):
    import numpy as np

    from ._kernels import nb_sum_axis

    x = np.asfortranarray(rng.rand(N_SIDE, N_SIDE))
    return np.sum, nb_sum_axis, (x, 0)


@case("sum/float64/axis1/F")
def sum_float64_axis1_f(rng):
    import numpy as np

    from ._kernels import nb_sum_axis

    x = np.asfortranarray(rng.rand(N_SIDE, N_SIDE))
    return np.sum, nb_sum_axis, (x, 1)


@case("sum/int64/axis/3d", requires_numba=SUM_AXIS_SINCE)
def sum_int64_axis_3d(rng):
    import numpy as np

    from ._kernels import nb_sum_axis

    x = rng.randint(0, 1000, (N_CUBE, N_CUBE, N_CUBE)).astype(np.int64)
    return np.sum, nb_sum_axis, (x, 1)


@case("sum/axis_tuple/float64/C", requires_numba=SUM_AXIS_SINCE)
def sum_axis_tuple_float64_c(rng):
    import numpy as np

    from ._kernels import nb_sum_axis

    x = rng.rand(N_CUBE, N_CUBE, N_CUBE)
    return np.sum, nb_sum_axis, (x, (0, 2))


@case("sum/pair/float64/C")
def sum_pair_float64_c(rng):
    import numpy as np

    from ._kernels import nb_sum_pair

    a = rng.rand(N_PAIR)
    b = rng.rand(N_PAIR)
    return (lambda a, b: np.sum(a) + np.sum(b)), nb_sum_pair, (a, b)


@case("cumsum/float64/F")
def cumsum_float64_f(rng):
    import numpy as np

    from ._kernels import nb_cumsum_flat

    x = np.asfortranarray(rng.rand(N_SIDE, N_SIDE))
    return np.cumsum, nb_cumsum_flat, (x,)


@case("cumsum/float64/axis0/F", requires_numba=CUMSUM_AXIS_SINCE)
def cumsum_float64_axis0_f(rng):
    import numpy as np

    from ._kernels import nb_cumsum_axis

    x = np.asfortranarray(rng.rand(N_SIDE, N_SIDE))
    return np.cumsum, nb_cumsum_axis, (x, 0)


@case("cumsum/float64/axis0/C", requires_numba=CUMSUM_AXIS_SINCE)
def cumsum_float64_axis0_c(rng):
    import numpy as np

    from ._kernels import nb_cumsum_axis

    x = rng.rand(N_SIDE, N_SIDE)
    return np.cumsum, nb_cumsum_axis, (x, 0)


@case("cumsum/float64/axis1/C", requires_numba=CUMSUM_AXIS_SINCE)
def cumsum_float64_axis1_c(rng):
    import numpy as np

    from ._kernels import nb_cumsum_axis

    x = rng.rand(N_SIDE, N_SIDE)
    return np.cumsum, nb_cumsum_axis, (x, 1)


@case("cumsum/int64/axis1/3d", requires_numba=CUMSUM_AXIS_SINCE)
def cumsum_int64_axis1_3d(rng):
    import numpy as np

    from ._kernels import nb_cumsum_axis

    x = rng.randint(0, 1000, (N_CUBE, N_CUBE, N_CUBE)).astype(np.int64)
    return np.cumsum, nb_cumsum_axis, (x, 1)
