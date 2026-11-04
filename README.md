# numba-benchmark

NumPy-vs-Numba benchmarks for every Numba `main` commit, with a nightly run,
historic backfill, an HTML report artifact, and a Markdown run summary.

## Files

| File | Job |
|---|---|
| `cases/` | Case registry (`Case`, `@case`, discovery, version gates) and the benchmark cases. |
| `benchmark.py` | Timing: compile time, warm-up, auto-sized loop, best of 5. |
| `bench.py` | `run`, `merge`, `compare`, `report`, `summary`, `cases`, and `backfill-order`. |
| `ci.py` | Resolves Numba commits and emits the CI matrix. |
| `template.html` | Chart.js report page template, one section per case prefix. |
| `tests/` | Stdlib-only report and CLI tests; no numpy or numba required. |
| `.github/workflows/bench.yml` | Tests on push/PR, nightly and historic benchmark runs. |
| `results/<sha>.json` | One result file per Numba commit (generated). |
| `site/index.html` | Report page (generated). |

## Local

```bash
pip install 'numpy>=2.2,<2.3' scipy
pip install --pre -i https://pypi.anaconda.org/numba/label/dev/simple llvmlite
pip install 'git+https://github.com/numba/numba.git@<numba-sha>'
python bench.py run --commit <numba-sha> --os linux --python 3.13 --numpy 2.2 --out run.json
mkdir -p artifacts && cp run.json artifacts/
python bench.py merge artifacts --out results/
python bench.py compare --results results/ --commit <numba-sha>
python bench.py report --results results/ --out site/index.html
python bench.py summary --results results/ --out summary.md
python bench.py cases
```

`scipy` is needed because Numba's `np.dot` uses BLAS. Numba `main` also needs a
matching llvmlite dev build; install it from Numba's dev wheel index before
Numba, as Numba's own CI does. Results are keyed by `os|python|numpy`;
re-running a key keeps the newest measurement.

`merge`, `compare`, `report`, `summary`, and the tests run without numpy or numba.
`python bench.py cases` prints the registry order and, when numba is installed,
whether each case runs or is skipped by its version gate.

## Cases

A case is one module in `cases/`; it needs no registration elsewhere and no
`order`. The build function takes one `numpy.random.RandomState` and returns
`(numpy_fn, numba_fn, args)`:

```python
@case("sum/int64/C", requires_numba=">=0.61")
def sum_int64_c(rng):
    import numpy as np
    from ._kernels import nb_sum

    x = rng.randint(0, 1000, 1_000_000).astype(np.int64)
    return np.sum, nb_sum, (x,)
```

`order` defaults to `10, 20, 30, …` in discovery order; pass `order=` to
interleave. `requires_numba` takes comma-separated clauses such as
`">=0.55,<0.60"`; a skipped case is never built or imported. Case modules must
be import-safe: keep numpy/numba imports inside the build body.

Inputs are seeded from the case id (`blake2b`), so adding or reordering cases
never changes another case's data. Point `NUMBA_BENCH_CASES` at extra files or
directories (`:`-separated on POSIX, `;` on Windows) to add cases without
editing the repo.

## CI

- **Tests** (`test` job): pushes and pull requests run
  `python -m unittest discover -s tests` without numpy or numba.
- **Nightly** (`17 4 * * *`): benchmarks the current Numba `main` on four
  environments and stops if `results/<sha>.json` already exists.
- **Historic**: run *benchmarks* from the Actions tab with `last`, `from`, `to`,
  `limit` (default 10, max 50), and `ref` to backfill first-parent commits.
  `last` benchmarks the last `N` commits of `ref` (max 50) without needing a
  hash and takes precedence over `from`/`to`; `from` is exclusive and `to` is
  inclusive. Leave `from` empty to backfill the last `limit` commits ending at
  `to` (or at `ref`'s head). Results record each commit's first-parent
  `position`, so backfilled commits keep history order on the chart even though
  they are measured concurrently.

Results are committed to the default branch. When a run adds results, the
workflow uploads the generated page as the `benchmark-report` artifact and
writes the `summary` output to the run summary, which links the artifact; the
link expires with it. Runs that add no results write no summary.
