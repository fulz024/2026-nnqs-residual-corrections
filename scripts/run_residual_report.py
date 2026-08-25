#!/usr/bin/env python3
"""Compute a paper-facing residual-correction JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from residual_energy import evaluate_residual_report
from residual_energy.distributed import Communicator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcidump", required=True, type=Path)
    parser.add_argument(
        "--support",
        required=True,
        type=Path,
        help="directory containing states_uint64.npy and nnqs_vector.npy",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--denominator-epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comm = Communicator()
    states = np.load(args.support / "states_uint64.npy", allow_pickle=False)
    vector = np.load(args.support / "nnqs_vector.npy", allow_pickle=False)
    report = evaluate_residual_report(
        str(args.fcidump),
        states,
        vector,
        denominator_epsilon=args.denominator_epsilon,
        communicator=comm,
    )
    if comm.rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
