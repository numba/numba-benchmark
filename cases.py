"""Benchmark environments, cases, kernels, and inputs."""

try:
    import numpy as np
    from numba import njit
except ImportError:  # planning, merge, compare, and report do not need the stack
    np = None
    njit = None

N = 1_000_000
N_SORT = 200_000
N_DOT = 256

ENVIRONMENTS = [
    {
        "os": "linux",
        "runner": "ubuntu-24.04",
        "python": "3.13",
        "numpy": "2.2",
        "spec": "numpy>=2.2,<2.3",
    },
    {
        "os": "linux",
        "runner": "ubuntu-24.04",
        "python": "3.12",
        "numpy": "2.1",
        "spec": "numpy>=2.1,<2.2",
    },
    {
        "os": "macos",
        "runner": "macos-14",
        "python": "3.13",
        "numpy": "2.2",
        "spec": "numpy>=2.2,<2.3",
    },
    {
        "os": "macos",
        "runner": "macos-14",
        "python": "3.12",
        "numpy": "2.1",
        "spec": "numpy>=2.1,<2.2",
    },
]

ENVIRONMENT_KEYS = [
    f"{environment['os']}|{environment['python']}|{environment['numpy']}"
    for environment in ENVIRONMENTS
]

CASES = [
    "sum/float64/C",
    "sum/float32/strided",
    "sum/int64/C",
    "sum/complex128/C",
    "mean/float64/F",
    "mean/float32/negative",
    "add/float64/C",
    "dot/float64/C",
    "sort/float64/C",
    "cumsum/float64/C",
    "where/float64/C",
    "searchsorted/float64/C",
]

if njit is not None:

    @njit
    def nb_sum(x):
        return np.sum(x)

    @njit
    def nb_mean(x):
        return np.mean(x)

    @njit
    def nb_add(x, y):
        out = np.empty_like(x)
        for i in range(x.size):
            out[i] = x[i] + y[i]
        return out

    @njit
    def nb_dot(a, b):
        return np.dot(a, b)

    @njit
    def nb_sort(x):
        return np.sort(x)

    @njit
    def nb_cumsum(x):
        out = np.empty_like(x)
        acc = 0.0
        for i in range(x.size):
            acc += x[i]
            out[i] = acc
        return out

    @njit
    def nb_where(cond, x, y):
        return np.where(cond, x, y)

    @njit
    def nb_searchsorted(a, v):
        return np.searchsorted(a, v)


def build_cases():
    if np is None or njit is None:
        raise RuntimeError("numpy and numba are required to build benchmark cases")

    rng = np.random.RandomState(0)

    f64 = rng.rand(N)
    f32 = rng.rand(N).astype(np.float32)
    i64 = rng.randint(0, 1000, N).astype(np.int64)
    c128 = rng.rand(N) + 1j * rng.rand(N)
    mean_f = np.asfortranarray(rng.rand(1000, 1000))
    add_a = rng.rand(N)
    add_b = rng.rand(N)
    dot_a = rng.rand(N_DOT, N_DOT)
    dot_b = rng.rand(N_DOT, N_DOT)
    sort_x = rng.rand(N_SORT)
    cum_x = rng.rand(N)
    where_c = rng.rand(N) > 0.5
    where_a = rng.rand(N)
    where_b = rng.rand(N)
    haystack = np.sort(rng.rand(N_SORT))
    needles = rng.rand(N_SORT)

    f32_strided = f32[::2]
    f32_negative = f32[::-1]

    return [
        ("sum/float64/C", lambda: np.sum(f64), lambda: nb_sum(f64)),
        (
            "sum/float32/strided",
            lambda: np.sum(f32_strided),
            lambda: nb_sum(f32_strided),
        ),
        ("sum/int64/C", lambda: np.sum(i64), lambda: nb_sum(i64)),
        ("sum/complex128/C", lambda: np.sum(c128), lambda: nb_sum(c128)),
        ("mean/float64/F", lambda: np.mean(mean_f), lambda: nb_mean(mean_f)),
        (
            "mean/float32/negative",
            lambda: np.mean(f32_negative),
            lambda: nb_mean(f32_negative),
        ),
        ("add/float64/C", lambda: add_a + add_b, lambda: nb_add(add_a, add_b)),
        ("dot/float64/C", lambda: np.dot(dot_a, dot_b), lambda: nb_dot(dot_a, dot_b)),
        ("sort/float64/C", lambda: np.sort(sort_x), lambda: nb_sort(sort_x)),
        ("cumsum/float64/C", lambda: np.cumsum(cum_x), lambda: nb_cumsum(cum_x)),
        (
            "where/float64/C",
            lambda: np.where(where_c, where_a, where_b),
            lambda: nb_where(where_c, where_a, where_b),
        ),
        (
            "searchsorted/float64/C",
            lambda: np.searchsorted(haystack, needles),
            lambda: nb_searchsorted(haystack, needles),
        ),
    ]
