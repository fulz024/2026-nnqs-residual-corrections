#!/usr/bin/env python3
"""List, retrieve, and verify Hamiltonians without redistributing them.

Only rows marked ``verified_upstream`` can be downloaded. The bytes are
fetched from the pinned upstream URL, checked before any transformation, and
then checked again against the exact analysis-copy digest. Locally generated
and derived artifacts require their documented recipes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "hamiltonian_manifest.csv"
DEFAULT_DESTINATION = ROOT / "data" / "hamiltonians"


def _load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    records: dict[str, dict[str, str]] = {}
    for row in rows:
        artifact_id = row["artifact_id"]
        if not artifact_id or artifact_id in records:
            raise ValueError(f"invalid or duplicate artifact_id: {artifact_id!r}")
        records[artifact_id] = row
    return records


def _safe_target(root: Path, logical_path: str) -> Path:
    logical = PurePosixPath(logical_path)
    if logical.is_absolute() or not logical.parts or ".." in logical.parts:
        raise ValueError(f"unsafe logical path: {logical_path!r}")
    root = root.resolve()
    target = root.joinpath(*logical.parts).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"logical path escapes destination: {logical_path!r}")
    return target


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _integer(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "").strip()
    return int(value) if value else None


def _check_bytes(
    data: bytes,
    *,
    expected_size: int | None,
    expected_sha256: str,
    label: str,
) -> None:
    if expected_size is not None and len(data) != expected_size:
        raise ValueError(
            f"{label} byte count mismatch: got {len(data)}, "
            f"expected {expected_size}"
        )
    actual = _sha256(data)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: got {actual}, expected {expected_sha256}"
        )


def _materialize(data: bytes, operation: str) -> bytes:
    if operation == "copy":
        return data
    if operation == "lf_to_crlf":
        # Normalize first so the result is independent of the client platform.
        return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    raise ValueError(
        f"{operation!r} is a documented recipe, not an automatic download "
        "operation"
    )


def _selected(
    records: dict[str, dict[str, str]], artifact_ids: list[str]
) -> list[dict[str, str]]:
    if not artifact_ids:
        return list(records.values())
    missing = [value for value in artifact_ids if value not in records]
    if missing:
        raise KeyError(f"unknown artifact_id(s): {', '.join(missing)}")
    return [records[value] for value in artifact_ids]


def command_list(rows: list[dict[str, str]]) -> int:
    headings = ("artifact_id", "retrieval_status", "provenance_class", "logical_path")
    widths = {
        key: max(len(key), *(len(row.get(key, "")) for row in rows))
        for key in headings
    }
    print("  ".join(key.ljust(widths[key]) for key in headings))
    for row in rows:
        print("  ".join(row.get(key, "").ljust(widths[key]) for key in headings))
    return 0


def command_fetch(
    rows: list[dict[str, str]], destination: Path, overwrite: bool
) -> int:
    fetchable = [
        row for row in rows if row["retrieval_status"] == "verified_upstream"
    ]
    refused = [
        row["artifact_id"]
        for row in rows
        if row["retrieval_status"] != "verified_upstream"
    ]
    if refused:
        print(
            "not fetched (requires a local generation recipe): "
            + ", ".join(refused),
            file=sys.stderr,
        )
    if not fetchable:
        return 2

    for row in fetchable:
        target = _safe_target(destination, row["logical_path"])
        if target.exists() and not overwrite:
            print(f"exists, verifying: {target}")
            data = target.read_bytes()
        else:
            request = Request(
                row["source_url"],
                headers={"User-Agent": "nnqs-residual-energy-corrections/0.2"},
            )
            with urlopen(request, timeout=120) as response:
                source_data = response.read()
            _check_bytes(
                source_data,
                expected_size=_integer(row, "source_bytes"),
                expected_sha256=row["source_sha256"],
                label=f"{row['artifact_id']} upstream",
            )
            data = _materialize(source_data, row["materialization"])
            _check_bytes(
                data,
                expected_size=_integer(row, "analysis_bytes"),
                expected_sha256=row["analysis_sha256"],
                label=f"{row['artifact_id']} analysis copy",
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            try:
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"fetched and verified: {target}")
            continue

        _check_bytes(
            data,
            expected_size=_integer(row, "analysis_bytes"),
            expected_sha256=row["analysis_sha256"],
            label=f"{row['artifact_id']} local copy",
        )
        print(f"verified: {target}")
    return 0


def command_verify(rows: list[dict[str, str]], destination: Path) -> int:
    failures = 0
    for row in rows:
        target = _safe_target(destination, row["logical_path"])
        if not target.is_file():
            print(f"MISSING  {row['artifact_id']}  {target}")
            failures += 1
            continue
        try:
            _check_bytes(
                target.read_bytes(),
                expected_size=_integer(row, "analysis_bytes"),
                expected_sha256=row["analysis_sha256"],
                label=f"{row['artifact_id']} local copy",
            )
        except ValueError as error:
            print(f"FAILED   {row['artifact_id']}  {error}")
            failures += 1
        else:
            print(f"OK       {row['artifact_id']}  {target}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="show provenance status")
    list_parser.add_argument("artifact_id", nargs="*")

    fetch_parser = subparsers.add_parser(
        "fetch", help="download only verified upstream artifacts"
    )
    fetch_parser.add_argument("artifact_id", nargs="*")
    fetch_parser.add_argument("--overwrite", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="verify local analysis-copy hashes"
    )
    verify_parser.add_argument("artifact_id", nargs="*")

    args = parser.parse_args()
    records = _load_manifest(args.manifest.resolve())
    rows = _selected(records, args.artifact_id)
    if args.command == "list":
        return command_list(rows)
    if args.command == "fetch":
        return command_fetch(rows, args.destination, args.overwrite)
    if args.command == "verify":
        return command_verify(rows, args.destination)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
