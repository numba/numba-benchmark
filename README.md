# numba-benchmark

NumPy-vs-Numba benchmarks for every Numba `main` commit, with a nightly run,
historic backfill, an HTML report artifact, and a Markdown run summary.

## Files

| File | Job |
|---|---|
| `src/cases/` | Case registry (`Case`, `@case`, discovery, version gates) and the benchmark cases. |
| `src/benchmark.py` | Timing: compile time, warm-up, auto-sized loop, best of 5. |
| `src/bench.py` | `run`, `merge`, `compare`, `report`, `summary`, `cases`, and `backfill-order`. |
| `src/ci.py` | Resolves Numba commits and emits the commit matrix. |
| `src/env_matrix.py` | Environment matrix used by the CI. |
| `src/template.html` | Chart.js report page template, one section per case prefix. |
| `.github/workflows/benchmark-environment.yml` | Reusable workflow: benchmarks one commit in one environment and uploads its environment JSON. |
| `.github/workflows/benchmark-commit.yml` | Reusable workflow: runs the environment matrix for one commit and returns its `<commit>.json`. |
| `.github/workflows/bench.yml` | Nightly and historic runs; calls `benchmark-commit.yml` once per commit, then publishes. |
| `results/` | Git submodule pointing at [numba-benchmark-results](https://github.com/numba/numba-benchmark-results). |
| `results/data/<sha>.json` | One result file per Numba commit (generated). |
| `results/COMMITS.md` | Commit index table (generated). |
| `results/scripts/results_readme.py` | Regenerates `results/COMMITS.md`, the commit index (lives in the results repository). |
| `site/index.html` | Report page (generated). |

Results live in the separate
[numba-benchmark-results](https://github.com/numba/numba-benchmark-results)
repository, mounted at `results/` as a git submodule. Clone with submodules, or
initialize it in an existing clone:

```bash
git clone --recurse-submodules https://github.com/numba/numba-benchmark.git
# or, in an existing clone:
git submodule update --init
```

## Local

```bash
pip install 'numpy>=2.2,<2.3' scipy
pip install --pre -i https://pypi.anaconda.org/numba/label/dev/simple llvmlite
pip install 'git+https://github.com/numba/numba.git@<numba-sha>'
python src/bench.py run --commit <numba-sha> --os linux --python 3.13 --numpy 2.2 --out run.json
mkdir -p artifacts && cp run.json artifacts/
python src/bench.py merge artifacts --out results/data/
python src/bench.py compare --results results/ --commit <numba-sha>
python src/bench.py report --results results/ --out site/index.html
python src/bench.py summary --results results/ --out summary.md
python src/bench.py cases
```

`scipy` is needed because Numba's `np.dot` uses BLAS. Numba `main` also needs a
matching llvmlite dev build; install it from Numba's dev wheel index before
Numba, as Numba's own CI does. Results are keyed by `os|python|numpy`;
re-running a key keeps the newest measurement.

`merge`, `compare`, `report`, and `summary` run without numpy or numba.
`python src/bench.py cases` prints the registry order and, when numba is installed,
whether each case runs or is skipped by its version gate.

## Cases

A case is one module in `src/cases/`; it needs no registration elsewhere and no
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

The CI is a chain of reusable workflows:

- `benchmark-environment.yml` (`workflow_call`) benchmarks one commit in one
  environment from `src/env_matrix.py` and uploads the environment JSON.
- `benchmark-commit.yml` (`workflow_call`) takes one Numba commit, builds the
  environment matrix, calls `benchmark-environment.yml` once per environment,
  merges the per-environment artifacts into `<commit>.json`, uploads it as
  `bench-result-<commit>`, and returns the same JSON in its `result` output.
- `bench.yml` plans the commits, calls `benchmark-commit.yml` once per commit,
  then `publish` downloads every `bench-result-*` artifact and merges them into
  `results/data/`.

Callers from another repository pass `repository` and `ref` so
`benchmark-commit.yml` checks out this benchmark repository and forwards the
same values to the environment workflow.

- **Nightly** (`17 4 * * *`): benchmarks the current Numba `main` commit and
  stops if `results/data/<sha>.json` already exists.
- **Historic**: run *benchmarks* from the Actions tab with `last`, `from`, `to`,
  `limit` (default 10, max 50), and `ref` to backfill first-parent commits.
  `last` benchmarks the last `N` commits of `ref` (max 50) without needing a
  hash and takes precedence over `from`/`to`; `from` is exclusive and `to` is
  inclusive. Leave `from` empty to backfill the last `limit` commits ending at
  `to` (or at `ref`'s head). Results record each commit's first-parent
  `position`, so backfilled commits keep history order on the chart even though
  they are measured concurrently.

Both paths call `benchmark-commit.yml` once per planned commit, then `publish`
downloads every `bench-result-*` artifact and merges them into `results/data/`.

Results are committed to the results submodule's `main` branch, and the parent
repository commits the updated submodule pointer. This needs a
`RESULTS_PUSH_TOKEN` repository secret with write access to
`numba-benchmark-results`. When a run adds results, the workflow uploads the
generated page as the `benchmark-report` artifact and writes the `summary`
output to the run summary, which links the artifact; the link expires with it.
The same run refreshes `results/COMMITS.md`, reusing existing rows and inserting
new commits in Numba first-parent order. Runs that add no results write no
summary.
