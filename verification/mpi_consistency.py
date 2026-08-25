#!/usr/bin/env python3
"""Produce a deterministic report for one-versus-many-rank comparison."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from residual_energy import evaluate_residual_report
from residual_energy.distributed import Communicator


FCIDUMP = ROOT / "benchmark_cases" / "fcidump" / "Hub1d6_U4.0_t1.0_Ne6_Sz0_obc.FCIDUMP"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    comm = Communicator()
    states = np.asarray(
        [
            sum(1 << orbital for orbital in occupied)
            for occupied in itertools.combinations(range(12), 6)
            if sum(orbital % 2 == 0 for orbital in occupied) == 3
        ],
        dtype=np.uint64,
    )
    support = states[17:217]
    rng = np.random.default_rng(20260821)
    vector = rng.normal(size=support.size) + 1j * rng.normal(size=support.size)
    report = evaluate_residual_report(
        str(FCIDUMP),
        support,
        vector,
        communicator=comm,
    )
    if comm.rank == 0:
        args.output.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
