
# Numba benchmark suite

This is a series of benchmarks designed to measure and record the various
performance aspects of [Numba](https://numba.pydata.org). The results are
published at https://numba.pydata.org/numba-benchmark/

The benchmarks can be run with the [`asv`](https://github.com/spacetelescope/asv)
utility.

Current version is executed with `python=3.6` with, the ASV version saved as
submodule under `./asv`

## Setup

```bash
conda create -n numba-benchmark python=3.8 conda-forge::asv
conda activate numba-benchmark
# Get benchmark repo
git clone https://github.com/numba/numba-benchmark
cd numba-benchmark
```

## Run

```bash
asv run
```
