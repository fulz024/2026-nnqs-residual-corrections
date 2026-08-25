# Reproducibility

## Method implementation

From a fresh clone:

```bash
python -m pip install -e ".[build,test,figures]"
cmake -S cpp_lib -B cpp_lib/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_lib/build -j 4
python -m pytest -q
python verification/check_hubbard.py
```

The tests check the public C++ symbol whitelist, Hermiticity and known Hubbard
diagonals, the zero-residual eigenvector limit, the projected internal solve,
the PT2/rPT2/dBW identities, and the Hamiltonian provenance/retrieval contract.

To verify distributed consistency on a machine with MPI:

```bash
torchrun --standalone --nproc-per-node=1 verification/mpi_consistency.py --output one.json
torchrun --standalone --nproc-per-node=2 verification/mpi_consistency.py --output two.json
python verification/compare_reports.py one.json two.json
```

## Molecular Hamiltonians

Third-party FCIDUMPs are intentionally absent.  Inspect their provenance and
status before retrieval:

```bash
python scripts/hamiltonian_artifacts.py list
```

The verified CDFCI, Dice, and Li--Chan inputs can then be fetched directly
from the pinned author repositories into the Git-ignored local directory:

```bash
python scripts/hamiltonian_artifacts.py \
  --destination data/hamiltonians fetch
```

This verifies both the original bytes and the exact analysis-copy bytes. It
does not fetch author-generated or derived records. Follow the
commands in `data/README.md` for Cr2 frozen-core reduction, the Fe2S2
canonical-RHF rotation, and NH3/HCN generation.

## Paper figures

The plot-ready source tables are versioned in `paper/figure_data/`; compact
derived diagnostics live in `data/diagnostics/`. Regenerate all paper figures
with

```bash
python paper/figures/make_paper_figures.py
```

Generated PDF/PNG/SVG files are intentionally ignored. The source tables,
column definitions (`docs/data_dictionary.md`), plotting script, and palette
are the public reproduction package.

## Reproduction scope

The method starts from a frozen determinant support, represented by the two
arrays documented in `docs/usage.md`. Neural-network training is outside this
interface and is not needed to evaluate a new support. The included source
tables are sufficient to reproduce the paper figures.
