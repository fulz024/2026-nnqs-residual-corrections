#!/usr/bin/env python3
"""Generate the paper's NH3 and HCN frozen-core cc-pVDZ Hamiltonians.

Coordinates and comparison energies are transcribed from the ancillary files
of Yao et al., J. Chem. Phys. 153, 124117 (2020), arXiv:2004.10059v3.
The coordinates are in bohr.  The FCIDUMPs themselves are generated locally.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pyscf import ci, gto, scf
from pyscf.tools import fcidump

from freeze_fcidump_core import freeze_core


@dataclass(frozen=True)
class Molecule:
    key: str
    atoms_bohr: str
    frozen_core_orbitals: str
    reference_energy: float
    reference_uncertainty: float


SYSTEMS = {
    "nh3": Molecule(
        key="NH3-cc-pVDZ-G2-FC",
        atoms_bohr="""
N   0.0000000000   0.0000000000   0.2149752284
H  -0.8857051218   1.5340862715  -0.5016466607
H  -0.8857051218  -1.5340862715  -0.5016466607
H   1.7714102436   0.0000000000  -0.5016466607
""",
        frozen_core_orbitals="1",
        reference_energy=-56.40247687620448,
        reference_uncertainty=5.784514789285814e-06,
    ),
    "hcn": Molecule(
        key="HCN-cc-pVDZ-G2-FC",
        atoms_bohr="""
H   0.0000000000   0.0000000000  -2.8401825705
C   0.0000000000   0.0000000000  -0.8276999826
N   0.0000000000   0.0000000000   1.3516076155
""",
        frozen_core_orbitals="1,2",
        reference_energy=-93.18988417813952,
        reference_uncertainty=7.190780067389824e-06,
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(spec: Molecule, destination: Path) -> dict[str, object]:
    output_dir = destination / spec.key
    output_dir.mkdir(parents=True, exist_ok=True)
    all_electron = output_dir / "FCIDUMP.all_electron"
    active = output_dir / "FCIDUMP"

    molecule = gto.M(
        atom=spec.atoms_bohr,
        unit="Bohr",
        basis="cc-pvdz",
        charge=0,
        spin=0,
        cart=False,
        symmetry=False,
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-12
    mean_field.max_cycle = 200
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(f"RHF did not converge for {spec.key}")
    fcidump.from_scf(mean_field, str(all_electron), tol=1.0e-15)

    frozen = freeze_core(
        all_electron,
        active,
        spec.frozen_core_orbitals,
        spec.reference_energy,
        "SHCI weighted-quadratic extrapolation (Yao et al. 2020)",
    )
    ncore = len(spec.frozen_core_orbitals.split(","))
    cisd = ci.CISD(mean_field, frozen=ncore)
    cisd.conv_tol = 1.0e-10
    cisd.max_cycle = 400
    cisd.kernel()
    if not cisd.converged:
        raise RuntimeError(f"CISD did not converge for {spec.key}")

    nocc = molecule.nelectron // 2
    record: dict[str, object] = {
        "system": spec.key,
        "source": "Yao et al., JCP 153, 124117 (2020), arXiv:2004.10059v3",
        "source_geometry_file": "arXiv ancillary file 'geometries'",
        "source_energy_file": "arXiv ancillary file 'SHCI.csv'",
        "basis": "cc-pVDZ (spherical)",
        "geometry_unit": "Bohr",
        "geometry": [
            line.strip() for line in spec.atoms_bohr.splitlines() if line.strip()
        ],
        "all_electron_n_electrons": int(molecule.nelectron),
        "all_electron_n_spatial_orbitals": int(molecule.nao_nr()),
        "frozen_core_orbitals_one_based": [
            int(value) for value in spec.frozen_core_orbitals.split(",")
        ],
        "active_n_electrons": frozen["active_n_electrons"],
        "active_n_spatial_orbitals": frozen["active_n_spatial_orbitals"],
        "rhf_energy": float(mean_field.e_tot),
        "rhf_homo_lumo_gap": float(
            mean_field.mo_energy[nocc] - mean_field.mo_energy[nocc - 1]
        ),
        "frozen_core_cisd_energy": float(cisd.e_tot),
        "frozen_core_cisd_hf_coefficient_squared": float(cisd.ci[0] ** 2),
        "shci_reference_energy": spec.reference_energy,
        "shci_reference_uncertainty": spec.reference_uncertainty,
        "all_electron_fcidump_sha256": _sha256(all_electron),
        "active_fcidump_sha256": _sha256(active),
    }
    (output_dir / "PROVENANCE.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "systems", nargs="*", choices=sorted(SYSTEMS), default=sorted(SYSTEMS)
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "hamiltonians",
    )
    args = parser.parse_args()
    records = [generate(SYSTEMS[key], args.destination) for key in args.systems]
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
