from __future__ import annotations

import numpy as np

from cpp_lib.build import residual_coupling_module as backend
from residual_energy.coupling import CouplingGenerator

from conftest import HUBBARD4


def test_extension_exports_only_article_boundary() -> None:
    public = {name for name in dir(backend) if not name.startswith("_")}
    assert public == {"Fcidump", "Hamiltonian", "diagonal", "generate_couplings"}


def test_hubbard_graph_is_hermitian(hubbard4_states: np.ndarray) -> None:
    graph = CouplingGenerator(HUBBARD4).generate(hubbard4_states)
    index = {int(state): row for row, state in enumerate(hubbard4_states)}
    matrix = np.zeros((len(index), len(index)), dtype=np.float64)
    offset = 0
    for row, length in enumerate(graph["coupled_states_length"]):
        stop = offset + int(length)
        for target, coefficient in zip(
            graph["coupled_states"][offset:stop],
            graph["coefficients"][offset:stop],
        ):
            matrix[row, index[int(target)]] = coefficient
        offset = stop
    np.testing.assert_allclose(matrix, matrix.T, atol=0.0, rtol=0.0)

    # In this Hubbard convention the diagonal is U times the number of
    # doubly occupied spatial sites. Orbitals 0--3 occupy two complete sites.
    row = int(np.flatnonzero(hubbard4_states == np.uint64(0b00001111))[0])
    assert matrix[row, row] == 8.0
