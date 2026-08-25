# Hamiltonian provenance and redistribution boundary

## Policy used by this repository

The reproducibility target is the electronic Hamiltonian, not possession of a
second untracked copy of someone else's file.  Accordingly:

1. author-generated Hamiltonians are accompanied by generation scripts,
   geometry, basis, orbital convention, and output checksums;
2. unmodified third-party Hamiltonians are fetched from their original host at
   a pinned revision and verified locally;
3. Hamiltonians derived from third-party inputs publish the transformation and
   hashes of both input and output, but not the input or derived bytes.

The authoritative byte-level details are in
`data/hamiltonian_manifest.csv`.  This document explains the scientific
construction.

## CDFCI inputs: N2, H2O, C2, and Cr2

The N2/cc-pVDZ, H2O/cc-pVDZ, C2/cc-pVDZ, and 48-electron/42-orbital
Cr2/Ahlrichs inputs come from `CDFCI/CDFCI` at commit
`7d045ffe00569e80a8b32826fc29b350206aae3b`.  The repository is BSD
3-Clause, but the FCIDUMPs are still not copied into this release.  The
retrieval tool downloads the exact raw file from the original repository,
checks its SHA-256, and reproduces the line-ending convention of the analysis
copy where necessary.

The paper's Cr2 CAS(24e,30o) input is obtained from the 48e/42o source by
freezing the one-based spatial orbitals

```text
1, 2, 3, 4, 14, 18, 24, 25, 26, 27, 35, 39
```

and retaining

```text
5-13, 15-17, 19-23, 28-34, 36-38, 40-42.
```

These symmetry-blocked indices freeze the Mg core on both Cr atoms.  The
closed-shell core is folded into the active one-electron integrals and gives
an effective core constant of `-1915.9783034306633 Eh`.  The committed script
checks three determinant energies before and after reduction; the analysis
run's largest invariance error was `4.55e-13 Eh`.

## NH3 and HCN at G2 geometries

The NH3 and HCN FCIDUMPs are author-generated.  Coordinates and SHCI
comparison energies were transcribed from the `geometries` and `SHCI.csv`
ancillary files of Yao et al., J. Chem. Phys. 153, 124117 (2020),
arXiv:2004.10059v3.  `scripts/generate_g2_hamiltonians.py` performs canonical
RHF with spherical cc-pVDZ, writes the all-electron FCIDUMP, and folds the
closed-shell core into the active Hamiltonian.  Spatial orbital 1 is frozen
for NH3; orbitals 1 and 2 are frozen for HCN.

Both the all-electron and frozen-core checksums are recorded.  The SHCI values
are external comparison energies, not ingredients in the Hamiltonian
generation.

## Mn(salen)

The diagnostic Mn(salen) CAS(28e,22o) input is the file
`tests/SHCI/integrals/mn_salen_FCIDUMP_nosym` from `sanshar/Dice` at commit
`f0f0850de73f2f02953ff6552315889d47255b6f`.  Dice declares GPL-3.0-or-later;
to avoid extending that statement beyond what the upstream authors intended
for the data file, this repository does not redistribute it.  The retrieval
tool obtains it from Dice and verifies both the upstream and analysis-copy
hashes.  The calculation selects `MS2=2` at run time.

## Fe2S2

The Fe2S2 CAS(30e,20o) Hamiltonian is the localized-DFT-orbital active-space
model of Li and Chan, *J. Chem. Theory Comput.* **2017**, *13*, 2681--2695,
DOI `10.1021/acs.jctc.7b00270`.  The authors publish the exact FCIDUMP in
`zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters`; this repository
pins commit `84433831a680f70665cf43fc8692e0210da92be3` and the path
`Fe2S2_and_Fe4S4/Fe2S2/fe2s2`.  The upstream raw file reproduces our source
checksum after the recorded LF-to-CRLF conversion.  The repository reports
the singlet DMRG result at bond dimension 8000 as `-116.6056091 Eh`.

The upstream repository does not include a license file.  We therefore do not
redistribute the FCIDUMP even though it is publicly downloadable; the local
retrieval tool obtains it from the authors' repository and verifies it.

Our paper does not use the localized-orbital file directly.  We performed a
unitary canonical-RHF rotation within the complete CAS(30e,20o), leaving the
many-body spectrum and the Li--Chan DMRG reference unchanged.  This rotated
Hamiltonian is an author-generated derivative of the cited input.  Its SHA-256
is recorded in the manifest, and `scripts/prepare_fe2s2_rhf.py` documents and
reproduces the transformation.

## Citation and license note

Repository license labels in the manifest describe the upstream repository at
the pinned revision.  They are not legal conclusions about ownership of every
integral file.  Users retrieving third-party inputs remain responsible for the
upstream terms and for citing the underlying data and method papers.
