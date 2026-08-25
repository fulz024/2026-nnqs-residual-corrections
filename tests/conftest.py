from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HUBBARD4 = ROOT / "benchmark_cases" / "fcidump" / "Hub1d4_U4.0_t1.0_Ne4_Sz0_obc.FCIDUMP"


@pytest.fixture(scope="session")
def hubbard4_states() -> np.ndarray:
    values = []
    for occupied in itertools.combinations(range(8), 4):
        if sum(orbital % 2 == 0 for orbital in occupied) == 2:
            values.append(sum(1 << orbital for orbital in occupied))
    return np.asarray(values, dtype=np.uint64)
