# numba-benchmark

NumPy-vs-Numba benchmarks for every Numba `main` commit, with a nightly run,
historic backfill, and a Chart.js graph.

## Files

| File | Job |
|---|---|
| `cases.py` | Environments, 12 cases, `@njit` kernels, seeded inputs. |
| `benchmark.py` | Timing: compile time, warm-up, auto-sized loop, best of 5. |
| `bench.py` | `run`, `merge`, `compare`, and `report` commands. |
| `ci.py` | Resolves Numba commits and emits the CI matrix. |
| `template.html` | Chart.js report page template. |
| `.github/workflows/bench.yml` | Nightly and historic backfill. |
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
```

`scipy` is needed because Numba's `np.dot` uses BLAS. Numba `main` also needs a
matching llvmlite dev build; install it from Numba's dev wheel index before
Numba, as Numba's own CI does. Results are keyed by `os|python|numpy`;
re-running a key keeps the newest measurement.

## CI

- **Nightly** (`17 4 * * *`): benchmarks the current Numba `main` on four
  environments and stops if `results/<sha>.json` already exists.
- **Historic**: run *benchmarks* from the Actions tab with `from`, `to`,
  `limit` (default 10, max 50), and `ref` to backfill first-parent commits.

Results are committed to the default branch. The report deploys through GitHub
Pages, so set the Pages source to "GitHub Actions".
