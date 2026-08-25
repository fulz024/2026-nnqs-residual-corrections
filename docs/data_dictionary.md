# Figure-data dictionary

The CSV files in `paper/figure_data/` are the plotting inputs used by
`paper/figures/make_paper_figures.py`.

- `large_pt2_checkpoints.csv`: checkpoint-wise PT2, rPT2, and dBW reports.
- `large_training_trajectory.csv`: restricted-training energy trajectories.
- `n2_support_budget_triad.csv`: support-size and seed convergence.
- `large_pt34_warning.csv`: PT3/PT4 coefficient-ratio diagnostics; not primary
  energy reports.
- `small_intruder_summary.csv` and `n2_negative_axis_scan_seed334.csv`:
  convergence-radius and avoided-crossing diagnostics.
- `small_pt_generated_rr.csv`: PT-generated residual-Ritz diagnostics.
- `hpc_pt2_*.csv`: backend timing, memory, and scaling measurements.
- `data/diagnostics/large_system_failure_diagnosis/`: compact energy-level
  data for the root-locality figure.

Column names encode units where ambiguity is likely: `_mha` denotes
millihartree, `_sec` seconds, and `_bytes` bytes.  Energies without a suffix are
in hartree.  `seed` is the support-sampling seed, not a repeated stochastic
tail estimator unless the table explicitly names such an estimator.

Every published table should ultimately be traceable to a raw JSON record in
the companion data deposit.  The committed CSVs are immutable source data for
the archived paper release, not caches regenerated silently during plotting.
