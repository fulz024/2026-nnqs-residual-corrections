"""Thin import boundary around the article-only C++ coupling generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from cpp_lib.build import residual_coupling_module as _backend
except ImportError as error:  # pragma: no cover - exercised by installation checks
    raise ImportError(
        "The residual coupling extension is not built. Follow the build steps in README.md."
    ) from error


class CouplingGenerator:
    """Hamiltonian connectivity required by restricted energy and residual PT2."""

    def __init__(self, fcidump: str | Path):
        parsed = _backend.Fcidump(str(Path(fcidump)))
        self._hamiltonian = _backend.Hamiltonian(parsed)
        self.norb = int(self._hamiltonian.norb)
        self.nelec = int(self._hamiltonian.nelec)
        self.ms2 = int(self._hamiltonian.ms2)
        self.uhf = bool(self._hamiltonian.uhf)

    def generate(self, states: np.ndarray) -> dict[str, np.ndarray]:
        values = np.ascontiguousarray(states, dtype=np.uint64).reshape(-1)
        raw = _backend.generate_couplings(self._hamiltonian, values)
        return {str(key): np.asarray(value) for key, value in raw.items()}

    def diagonal(self, states: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(states, dtype=np.uint64).reshape(-1)
        return np.asarray(
            _backend.diagonal(self._hamiltonian, values),
            dtype=np.float64,
        )
