
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
asv run "-1 d34db33f"
```

Run `asv`  on the second commit:

```bash
asv run "-1 v0.42.0"
```

Note that the `-1 abcxyz` argument will be handed over to `git log`. The `-1`
is used to select only a single commit and the subsequent argument is the
commit-ish for the commit you would to run the benchmarks on. So, this can be a
hexadecimal commit identifier or a branch or tag.

Compare them:

```bash
asv compare "abcdefg" "1234567"
```

Useful options to consider:

```bash
asv run --verbose --show-stderr  -b 'bench_typed_list' "-1 d34db33f"
```

This is useful, because `--show-stderr` will output any errors enountered and
`-b` can be used to limit the benchmark suits that will be run.

```bash
asv compare --machine machine.local "d34db33f" "v0.42.0"
```

This will restrict the comparison to only those benchmarks collected on the
machine `machine.local`.
