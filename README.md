# NNQS residual energy corrections

This is the article-specific public implementation of frozen-support energy
reporting for neural-network quantum states (NNQS). It evaluates

- the variational energy restricted to a retained determinant support,
  $E_{\mathcal S}$;
- the full-residual diagonal PT2 correction, including the projected residual
  left inside the retained support;
- the corresponding rPT2 and diagonal Brillouin--Wigner (dBW) reports.

The repository intentionally does not contain NNQS training/model code. The
boundary between training and reporting is two ordinary NumPy arrays exported
from any wavefunction program:

```text
support/
  states_uint64.npy   # one packed determinant per row, at most 64 spin orbitals
  nnqs_vector.npy     # real or complex amplitude on the same rows
```

This keeps the reporting layer independent of the training code and lets it be used with
any NNQS implementation or a non-neural selected wavefunction.

## Quick start

Install the Python and build dependencies and compile the small C++ coupling
module:

```bash
python -m pip install -e ".[build,test]"
cmake -S cpp_lib -B cpp_lib/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_lib/build -j 4
python -m pytest -q
```

Run a report in one process:

```bash
python scripts/run_residual_report.py \
  --fcidump /path/to/FCIDUMP \
  --support /path/to/exported_support \
  --output results/residual_report.json
```

The same command uses the PyTorch/NCCL owner-reduction path when
launched with `torchrun`:

```bash
torchrun --standalone --nproc-per-node=4 scripts/run_residual_report.py \
  --fcidump /path/to/FCIDUMP \
  --support /path/to/exported_support \
  --output results/residual_report.json
```

Source rows are divided among ranks. Contributions to the same external
determinant are sent through the production SplitMix64 hash-owner route and
summed coherently on the device before
their squared residual enters PT2. No stochastic estimator is used: the
evaluator always constructs the full connected-space PT2 for the supplied
support.

## Implementation scope

The C++ extension exports an FCIDUMP reader, a
Hamiltonian object, connected determinant generation, and Hamiltonian
diagonals. Its only consumers are restricted $E_{\mathcal S}$ and residual
PT2. Determinants are represented by one 64-bit word, so the current
implementation supports at most 64 spin orbitals.

See [docs/method.md](docs/method.md) for the equations,
[docs/usage.md](docs/usage.md) for the data contract, and
[docs/reproducibility.md](docs/reproducibility.md) for verification and figure
reproduction.

## Repository contents

- `residual_energy/`: the standalone distributed evaluator.
- `cpp_lib/`: the article-only determinant coupling kernel.
- `paper/figure_data/` and `paper/figures/`: plot-ready source tables and the
  scripts used to regenerate article figures.
- `data/diagnostics/`: compact derived diagnostics used by the figures.
- `benchmark_cases/fcidump/`: small generated Hubbard inputs for tests.
- `verification/`: independent enumerable-space checks.

Third-party molecular FCIDUMPs are not committed.  Their pinned original
locations, upstream and analysis-copy SHA-256 values, and any local
transformations are recorded in `data/hamiltonian_manifest.csv`.  Use
`scripts/hamiltonian_artifacts.py` to retrieve verified originals directly
from their authors; see `data/README.md` for the Cr2, Fe2S2, NH3, and HCN
recipes.  The destination directory is ignored by Git.

The repository starts from an exported determinant support. Training software,
checkpoints, and sampler state are therefore not required to run the correction
or reproduce the figures from the included source tables.

## License

Source code and documentation are released under the
[BSD 3-Clause License](LICENSE). Author-generated figure tables, derived
diagnostics, the Hamiltonian provenance manifest, and generated Hubbard
FCIDUMPs are released under
[Creative Commons Attribution 4.0 International](DATA_LICENSE).

Third-party molecular Hamiltonians are neither redistributed nor covered by
these licenses. Their original locations, checksums, and upstream terms are
recorded in `data/hamiltonian_manifest.csv`.
