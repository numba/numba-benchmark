
# Numba benchmark suite

This is a series of benchmarks designed to measure and record the various
performance aspects of `Numba <http://numba.pydata.org/>`_.

The benchmarks can be run with the
`asv <https://github.com/spacetelescope/asv>`_ utility.

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

## Comparing two commits

Run `asv` on the first commit:

```bash
asv run "-1 abcdefg
```

Run `asv`  on the second commit:

```bash
asv run "-1 1235567
```

Compare them:

```bash
asv compare "abcdefg" "1234567"
```

Useful options to consider:

```bash
asv run --verbose --show-stderr  -b 'bench_typed_list' "-1 abcdefg"
```

```bash
asv compare --machine machine.local "abcdefg" "1234567"
```
