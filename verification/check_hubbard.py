#!/usr/bin/env python3
"""Independent small-space sanity check for the released evaluator."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from residual_energy import evaluate_residual_report
from residual_energy.coupling import CouplingGenerator


FCIDUMP = ROOT / "benchmark_cases" / "fcidump" / "Hub1d4_U4.0_t1.0_Ne4_Sz0_obc.FCIDUMP"


def main() -> None:
    states = np.asarray(
        [
            sum(1 << orbital for orbital in occupied)
            for occupied in itertools.combinations(range(8), 4)
            if sum(orbital % 2 == 0 for orbital in occupied) == 2
        ],
        dtype=np.uint64,
    )
    graph = CouplingGenerator(FCIDUMP).generate(states)
    index = {int(state): row for row, state in enumerate(states)}
    matrix = np.zeros((states.size, states.size))
    offset = 0
    for row, length in enumerate(graph["coupled_states_length"]):
        stop = offset + int(length)
        for target, coefficient in zip(
            graph["coupled_states"][offset:stop],
            graph["coefficients"][offset:stop],
        ):
            matrix[row, index[int(target)]] = coefficient
        offset = stop
    values, vectors = np.linalg.eigh(matrix)
    report = evaluate_residual_report(str(FCIDUMP), states, vectors[:, 0])
    result = {
        "dense_ground_energy": float(values[0]),
        "reported_energy": report.restricted_energy,
        "energy_difference": report.restricted_energy - float(values[0]),
        "residual_norm_sq": report.residual_norm_sq,
        "pt2_correction": report.pt2_correction,
        "hermiticity_error": float(np.max(np.abs(matrix - matrix.T))),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if abs(result["energy_difference"]) > 2.0e-12:
        raise SystemExit("energy regression failed")
    if result["residual_norm_sq"] > 1.0e-24:
        raise SystemExit("residual regression failed")


if __name__ == "__main__":
    main()
