# Paper figure artifact

Run `python paper/figures/make_paper_figures.py` from the repository root.
Figure source tables are versioned in `figure_data/`; generated graphics are
written to `figures/generated/` and are not committed.

All panels are generated from committed source tables. The root-locality panel
reads the compact derived data in `data/diagnostics/`.
