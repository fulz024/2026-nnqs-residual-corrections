#!/usr/bin/env python3
"""Freeze explicitly selected doubly occupied orbitals in an RHF FCIDUMP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump


def _indices(text: str, n_orbitals: int) -> np.ndarray:
    one_based = sorted(
        {int(token.strip()) for token in text.replace(";", ",").split(",") if token.strip()}
    )
    if not one_based or one_based[0] < 1 or one_based[-1] > n_orbitals:
        raise ValueError(f"core indices must lie in 1..{n_orbitals}: {one_based}")
    return np.asarray([value - 1 for value in one_based], dtype=np.int64)


def _determinant_energy(
    h1: np.ndarray, eri: np.ndarray, occupied: np.ndarray, ecore: float
) -> float:
    one_body = 2.0 * np.einsum(
        "ii->", h1[np.ix_(occupied, occupied)], optimize=True
    )
    block = eri[np.ix_(occupied, occupied, occupied, occupied)]
    coulomb = 2.0 * np.einsum("iijj->", block, optimize=True)
    exchange = np.einsum("ijji->", block, optimize=True)
    return float(ecore + one_body + coulomb - exchange)


def _read_fcidump_compat(path: Path) -> dict:
    """Read the standalone ``&FCI`` header used by the pinned CDFCI files."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines or lines[0].strip().upper() != "&FCI":
        return fcidump.read(str(path), verbose=False)
    if len(lines) < 2:
        raise ValueError(f"incomplete FCIDUMP header: {path}")
    lines[0] = f"&FCI {lines[1].lstrip()}"
    del lines[1]
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".FCIDUMP", delete=False
        ) as temporary:
            temporary.writelines(lines)
            temporary_name = temporary.name
        return fcidump.read(temporary_name, verbose=False)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def freeze_core(
    source: Path,
    output: Path,
    core_orbitals: str,
    reference_energy: float | None = None,
    reference_algorithm: str = "",
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    data = _read_fcidump_compat(source)
    n_orbitals = int(data["NORB"])
    n_electrons = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    uhf = str(data.get("UHF", "FALSE")).strip().upper()
    if uhf in {"1", "TRUE", ".TRUE.", "T"}:
        raise ValueError("only RHF FCIDUMPs are supported")

    core = _indices(core_orbitals, n_orbitals)
    core_set = set(core.tolist())
    active = np.asarray(
        [index for index in range(n_orbitals) if index not in core_set],
        dtype=np.int64,
    )
    active_electrons = n_electrons - 2 * int(core.shape[0])
    if active_electrons <= 0:
        raise ValueError("freezing the requested core removes all electrons")

    h1 = np.asarray(data["H1"], dtype=np.float64)
    eri = np.asarray(ao2mo.restore(1, data["H2"], n_orbitals), dtype=np.float64)
    source_ecore = float(data.get("ECORE", 0.0))
    h1_active = np.ascontiguousarray(h1[np.ix_(active, active)])
    eri_active = np.ascontiguousarray(
        eri[np.ix_(active, active, active, active)]
    )
    h1_active += 2.0 * np.einsum(
        "pqii->pq", eri[np.ix_(active, active, core, core)], optimize=True
    )
    h1_active -= np.einsum(
        "piiq->pq", eri[np.ix_(active, core, core, active)], optimize=True
    )

    eri_core = eri[np.ix_(core, core, core, core)]
    effective_core = (
        source_ecore
        + 2.0 * np.einsum("ii->", h1[np.ix_(core, core)], optimize=True)
        + 2.0 * np.einsum("iijj->", eri_core, optimize=True)
        - np.einsum("ijji->", eri_core, optimize=True)
    )
    orbsym = data.get("ORBSYM")
    active_orbsym = (
        [int(orbsym[index]) for index in active] if orbsym is not None else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fcidump.from_integrals(
        str(output),
        h1_active,
        eri_active,
        int(active.shape[0]),
        active_electrons,
        nuc=float(effective_core),
        ms=ms2,
        orbsym=active_orbsym,
        tol=1.0e-15,
    )
    # Make output hashes independent of the platform running the recipe.
    output.write_bytes(output.read_bytes().replace(b"\r\n", b"\n"))

    n_active_pairs = active_electrons // 2
    test_occupations = [
        np.arange(n_active_pairs, dtype=np.int64),
        np.arange(active.shape[0] - n_active_pairs, active.shape[0], dtype=np.int64),
        np.linspace(0, active.shape[0] - 1, n_active_pairs, dtype=np.int64),
    ]
    maximum_error = 0.0
    for active_occupied in test_occupations:
        original_occupied = np.sort(np.concatenate((core, active[active_occupied])))
        maximum_error = max(
            maximum_error,
            abs(
                _determinant_energy(h1, eri, original_occupied, source_ecore)
                - _determinant_energy(
                    h1_active, eri_active, active_occupied, float(effective_core)
                )
            ),
        )
    if maximum_error > 1.0e-9:
        raise RuntimeError(
            f"frozen-core determinant-energy check failed: {maximum_error:.3e} Ha"
        )

    record: dict[str, object] = {
        "source_fcidump": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_fcidump": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_n_electrons": n_electrons,
        "source_n_spatial_orbitals": n_orbitals,
        "active_n_electrons": active_electrons,
        "active_n_spatial_orbitals": int(active.shape[0]),
        "ms2": ms2,
        "frozen_core_orbitals_one_based": [int(index + 1) for index in core],
        "active_orbitals_one_based": [int(index + 1) for index in active],
        "effective_core_energy": float(effective_core),
        "max_determinant_energy_invariance_error_ha": maximum_error,
        "reference_energy": reference_energy,
        "reference_algorithm": reference_algorithm,
    }
    output.with_name("frozen_core_metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-orbitals", required=True)
    parser.add_argument("--reference-energy", type=float)
    parser.add_argument("--reference-algorithm", default="")
    args = parser.parse_args()
    record = freeze_core(
        args.input,
        args.output,
        args.core_orbitals,
        args.reference_energy,
        args.reference_algorithm,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
