# Usage and input contract

## Exporting a frozen support

Create a directory with two arrays:

- `states_uint64.npy`: one-dimensional `numpy.uint64`, one packed determinant
  per entry. Bit (p) is one when spin orbital (p) is occupied.
- `nnqs_vector.npy`: one-dimensional real or complex amplitudes evaluated on
  exactly those determinants.

Rows may arrive in any order; the evaluator sorts them. Duplicate determinant
IDs, non-finite amplitudes, empty supports, and mismatched lengths are rejected.
The amplitudes need not be normalized. Sample multiplicities are not used:
this is a finite-vector report, so repeated Monte Carlo walkers must first be
deduplicated into a determinant support.

The Hamiltonian is supplied as an RHF or UHF FCIDUMP. RHF spatial orbitals are
expanded in interleaved alpha/beta order.

## Command line

```bash
python scripts/run_residual_report.py \
  --fcidump molecule.FCIDUMP \
  --support support_export \
  --output results/molecule.json
```

Optional controls:

- `--denominator-epsilon 1e-12`: denominators smaller in absolute value are
  sign-preservingly regularized. The reported minimum denominator should be
  inspected when this control is active.

Every nonzero Slater--Condon single and double connection from the retained
support is included.

Launch distributed reports with PyTorch's standard process launcher:

```bash
torchrun --standalone --nproc-per-node=4 scripts/run_residual_report.py ...
```

Every rank reads the compact support arrays and FCIDUMP. Hamiltonian source
rows are partitioned; external determinant records are locally combined on
the selected CPU/GPU device and routed with `torch.distributed` all-to-all to
deterministic owners. Multi-node jobs use hierarchical node/lane exchange when
`LOCAL_WORLD_SIZE` is available. For heterogeneous nodes, set
`NNQS_NODE_RANKS` to a comma-separated rank count for each node. Only rank zero writes the JSON
file.

## Interpreting the JSON

The primary fields are `restricted_energy`, `pt2_energy`, `rpt2_energy`, and
`dbw_energy`. `internal_pt2` and `external_pt2` expose the full-residual split.
Useful health diagnostics include `first_order_norm_sq`, `residual_norm_sq`,
`minimum_abs_denominator`, `internal_orthogonality_error`, and
`internal_equation_error`.

`dbw_energy` is `null` when the reference-connected scalar Newton solve fails;
`dbw_error` then contains the reason. Such a failure should not be interpreted
as a valid energy correction.
