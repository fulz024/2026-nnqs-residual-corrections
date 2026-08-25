#!/usr/bin/env python3
"""Compare the invariant numerical fields of two MPI reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FIELDS = (
    "restricted_energy",
    "internal_pt2",
    "external_pt2",
    "pt2_correction",
    "rpt2_correction",
    "dbw_correction",
    "residual_norm_sq",
    "first_order_norm_sq",
    "minimum_abs_denominator",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    first = json.loads(args.first.read_text(encoding="utf-8"))
    second = json.loads(args.second.read_text(encoding="utf-8"))
    for field in FIELDS:
        np.testing.assert_allclose(first[field], second[field], rtol=2.0e-12, atol=2.0e-12)
    if first["support_size"] != second["support_size"]:
        raise AssertionError("support sizes differ")
    if first["external_size"] != second["external_size"]:
        raise AssertionError("external owner reductions differ")
    print("MPI reports agree within 2e-12")


if __name__ == "__main__":
    main()
