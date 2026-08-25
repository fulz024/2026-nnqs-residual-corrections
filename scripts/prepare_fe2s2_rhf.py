#!/usr/bin/env python3
"""Rotate the Li--Chan Fe2S2 CAS(30e,20o) Hamiltonian to canonical RHF orbitals.

The input is treated as an orthonormal one-particle basis.  Several
deterministic RHF guesses are optimized; the lowest converged, stability-tested
solution is selected and the complete active-space Hamiltonian is transformed
unitarily.  No orbitals or determinants are truncated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyscf import ao2mo, gto, scf
from pyscf.tools import fcidump


EXPECTED_SOURCE_SHA256 = (
    "670659bdbe5bd2a8953c79ec0fdf6c24eccae2ebc2d7bb4499e4feb57912d608"
)
EXPECTED_OUTPUT_SHA256 = (
    "7ae46b9b71285889f03f9cf1f4f02efe4f69d3d0101da2f60a71c342954e3e8d"
)
DMRG_REFERENCE = -116.6056091


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _determinant_energy(
    h1: np.ndarray, eri: np.ndarray, occupied: np.ndarray, ecore: float
) -> float:
    indices = np.flatnonzero(occupied > 0)
    energy = 2.0 * np.einsum("ii->", h1[np.ix_(indices, indices)])
    for first in indices:
        for second in indices:
            energy += (
                2.0 * eri[first, first, second, second]
                - eri[first, second, second, first]
            )
    return float(energy + ecore)


def _projector(coefficients: np.ndarray, nocc: int) -> np.ndarray:
    occupied = coefficients[:, :nocc]
    return 2.0 * occupied @ occupied.T


def _build_rhf(h1: np.ndarray, eri8: np.ndarray, nelec: int, ecore: float):
    molecule = gto.M(verbose=0)
    molecule.nelectron = int(nelec)
    molecule.spin = 0
    molecule.incore_anyway = True
    mean_field = scf.RHF(molecule)
    mean_field.get_hcore = lambda *args: h1
    mean_field.get_ovlp = lambda *args: np.eye(h1.shape[0])
    mean_field.energy_nuc = lambda *args: float(ecore)
    mean_field._eri = eri8
    mean_field.conv_tol = 1.0e-11
    mean_field.conv_tol_grad = 1.0e-8
    mean_field.max_cycle = 300
    mean_field.diis_space = 12
    mean_field.verbose = 3
    return mean_field


def _initial_densities(
    h1: np.ndarray, nocc: int, random_starts: int, seed: int
):
    norb = h1.shape[0]
    identity = np.eye(norb)
    yield "input_first_nocc", _projector(identity, nocc)
    _, one_electron_coefficients = np.linalg.eigh(h1)
    yield "one_electron", _projector(one_electron_coefficients, nocc)
    generator = np.random.default_rng(seed)
    for index in range(random_starts):
        perturbation = generator.normal(size=(norb, norb))
        perturbation -= perturbation.T
        scale = (index + 1) / max(1, random_starts)
        rotation = np.eye(norb) + 0.35 * scale * perturbation
        coefficients, _ = np.linalg.qr(one_electron_coefficients @ rotation)
        yield f"random_{index:02d}", _projector(coefficients, nocc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-starts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    input_hash = _sha256(args.input)
    if input_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"unexpected source SHA-256 {input_hash}; expected "
            f"{EXPECTED_SOURCE_SHA256}"
        )
    data = fcidump.read(str(args.input), molpro_orbsym=False, verbose=False)
    norb = int(data["NORB"])
    nelec = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    if (norb, nelec, ms2) != (20, 30, 0):
        raise ValueError(f"unexpected sector: NORB={norb}, NELEC={nelec}, MS2={ms2}")

    nocc = nelec // 2
    h1 = np.asarray(data["H1"], dtype=np.float64)
    eri8 = np.asarray(data["H2"], dtype=np.float64)
    eri = ao2mo.restore(1, eri8, norb)
    ecore = float(data.get("ECORE", 0.0))
    first_15 = np.zeros(norb, dtype=np.int8)
    first_15[:nocc] = 1
    input_determinant_energy = _determinant_energy(h1, eri, first_15, ecore)

    starts: list[dict[str, object]] = []
    solutions = []
    for label, density in _initial_densities(
        h1, nocc, args.random_starts, args.seed
    ):
        mean_field = _build_rhf(h1, eri8, nelec, ecore)
        energy = float(mean_field.kernel(dm0=density))
        gradient_norm = float(
            np.linalg.norm(mean_field.get_grad(mean_field.mo_coeff, mean_field.mo_occ))
        )
        used_newton = False
        if not mean_field.converged and gradient_norm < 5.0e-2:
            refined = scf.newton(mean_field)
            refined.conv_tol = 1.0e-11
            refined.conv_tol_grad = 1.0e-7
            refined.max_cycle = 100
            refined.verbose = 3
            energy = float(refined.kernel(dm0=mean_field.make_rdm1()))
            mean_field = refined
            gradient_norm = float(
                np.linalg.norm(
                    mean_field.get_grad(mean_field.mo_coeff, mean_field.mo_occ)
                )
            )
            used_newton = True
        record = {
            "label": label,
            "converged": bool(mean_field.converged),
            "energy": energy,
            "gradient_norm": gradient_norm,
            "cycles": int(getattr(mean_field, "cycles", -1)),
            "used_newton": used_newton,
        }
        starts.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if mean_field.converged and np.isfinite(energy):
            solutions.append((energy, label, mean_field))
    if not solutions:
        raise RuntimeError("no RHF initial guess converged")

    energy, label, best = min(solutions, key=lambda item: item[0])
    stable_orbitals, _, stable_internal, _ = best.stability(
        internal=True, external=False, return_status=True
    )
    if stable_internal is False:
        stable = _build_rhf(h1, eri8, nelec, ecore)
        stable_energy = float(
            stable.kernel(dm0=stable.make_rdm1(stable_orbitals, best.mo_occ))
        )
        if stable.converged and stable_energy < energy:
            best, energy, label = stable, stable_energy, f"{label}+internal_stability"
        _, _, stable_internal, _ = best.stability(
            internal=True, external=False, return_status=True
        )

    coefficients = np.asarray(best.mo_coeff, dtype=np.float64)
    h1_mo = coefficients.T @ h1 @ coefficients
    eri_mo = ao2mo.incore.full(eri8, coefficients, compact=False).reshape(
        norb, norb, norb, norb
    )
    canonical_determinant_energy = _determinant_energy(
        h1_mo, eri_mo, first_15, ecore
    )
    if abs(canonical_determinant_energy - energy) > 1.0e-8:
        raise RuntimeError("canonical determinant energy does not match RHF energy")
    if abs(np.linalg.norm(eri_mo) - np.linalg.norm(eri)) > 1.0e-8:
        raise RuntimeError("two-electron Frobenius norm changed under rotation")

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "FCIDUMP"
    fcidump.from_integrals(
        str(output),
        h1_mo,
        eri_mo,
        norb,
        nelec,
        nuc=ecore,
        ms=ms2,
        orbsym=[1] * norb,
        tol=1.0e-14,
    )
    # The analysis output was written on Linux; make the byte representation
    # platform-independent before hashing.
    output.write_bytes(output.read_bytes().replace(b"\r\n", b"\n"))
    np.savez_compressed(
        args.output_dir / "rhf_orbitals.npz",
        mo_coeff=coefficients,
        mo_energy=np.asarray(best.mo_energy),
        mo_occ=np.asarray(best.mo_occ),
        density=np.asarray(best.make_rdm1()),
    )
    output_hash = _sha256(output)
    summary = {
        "source": "Li and Chan, JCTC 13, 2681-2695 (2017)",
        "source_doi": "10.1021/acs.jctc.7b00270",
        "input": str(args.input.resolve()),
        "input_sha256": input_hash,
        "output": str(output.resolve()),
        "output_sha256": output_hash,
        "expected_analysis_output_sha256": EXPECTED_OUTPUT_SHA256,
        "exact_analysis_hash_reproduced": output_hash == EXPECTED_OUTPUT_SHA256,
        "norb": norb,
        "nelec": nelec,
        "ms2": ms2,
        "ecore": ecore,
        "input_first_15_determinant_energy": input_determinant_energy,
        "selected_start": label,
        "rhf_energy": energy,
        "rhf_gap_to_dmrg_reference": energy - DMRG_REFERENCE,
        "canonical_first_15_determinant_energy": canonical_determinant_energy,
        "final_gradient_norm": float(
            np.linalg.norm(best.get_grad(best.mo_coeff, best.mo_occ))
        ),
        "internally_stable": bool(stable_internal),
        "orbital_orthogonality_error": float(
            np.linalg.norm(coefficients.T @ coefficients - np.eye(norb))
        ),
        "starts": starts,
    }
    (args.output_dir / "rhf_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
