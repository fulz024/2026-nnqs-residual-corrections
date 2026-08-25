from __future__ import annotations

import numpy as np

from residual_energy import evaluate_residual_report
from residual_energy.coupling import CouplingGenerator

from conftest import HUBBARD4


def dense_hamiltonian(states: np.ndarray) -> np.ndarray:
    graph = CouplingGenerator(HUBBARD4).generate(states)
    lookup = {int(state): row for row, state in enumerate(states)}
    matrix = np.zeros((states.size, states.size), dtype=np.float64)
    offset = 0
    for row, length in enumerate(graph["coupled_states_length"]):
        stop = offset + int(length)
        for target, coefficient in zip(
            graph["coupled_states"][offset:stop],
            graph["coefficients"][offset:stop],
        ):
            matrix[row, lookup[int(target)]] = coefficient
        offset = stop
    return matrix


def test_full_space_eigenvector_has_zero_residual(hubbard4_states: np.ndarray) -> None:
    matrix = dense_hamiltonian(hubbard4_states)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    report = evaluate_residual_report(
        str(HUBBARD4),
        hubbard4_states,
        eigenvectors[:, 0],
    )
    assert report.external_size == 0
    np.testing.assert_allclose(report.restricted_energy, eigenvalues[0], atol=2.0e-14)
    assert report.residual_norm_sq < 1.0e-26
    assert abs(report.pt2_correction) < 1.0e-26
    assert report.dbw_error is None


def test_pt2_rpt2_and_dbw_identities(hubbard4_states: np.ndarray) -> None:
    rng = np.random.default_rng(73)
    support = hubbard4_states[4:20]
    vector = rng.normal(size=support.size) + 1j * rng.normal(size=support.size)
    report = evaluate_residual_report(str(HUBBARD4), support, vector)

    assert report.external_size > 0
    np.testing.assert_allclose(
        report.pt2_correction,
        report.internal_pt2 + report.external_pt2,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        report.rpt2_correction,
        report.pt2_correction / (1.0 + report.first_order_norm_sq),
        atol=2.0e-14,
    )
    assert report.dbw_error is None
    assert report.dbw_correction is not None
    assert report.dbw_energy is not None
    np.testing.assert_allclose(
        report.dbw_energy,
        report.restricted_energy + report.dbw_correction,
        atol=2.0e-14,
    )
    assert report.internal_orthogonality_error < 1.0e-12
    assert report.internal_equation_error < 1.0e-12
