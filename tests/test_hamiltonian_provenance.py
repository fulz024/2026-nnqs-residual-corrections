from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "hamiltonian_manifest.csv"
SCRIPT = ROOT / "scripts" / "hamiltonian_artifacts.py"
SHA256 = re.compile(r"[0-9a-f]{64}")


def _rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _artifact_module():
    spec = importlib.util.spec_from_file_location("hamiltonian_artifacts", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_is_complete_and_pinned() -> None:
    rows = _rows()
    assert len(rows) == 12
    assert len({row["artifact_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["logical_path"]
        assert row["analysis_sha256"] == "" or SHA256.fullmatch(
            row["analysis_sha256"]
        )
        assert row["source_sha256"] == "" or SHA256.fullmatch(
            row["source_sha256"]
        )
        if row["retrieval_status"] == "verified_upstream":
            assert row["source_url"].startswith("https://")
            assert row["source_commit"] in row["source_url"]
            assert len(row["source_commit"]) == 40
            assert row["source_sha256"]
            assert row["redistribution"] == "not_redistributed"


def test_third_party_hamiltonians_are_not_committed() -> None:
    if (ROOT / ".git").is_dir():
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode().split("\0")
        committed_inputs = [
            path
            for value in tracked
            if value
            for path in [Path(value)]
            if (path.name.upper() == "FCIDUMP" or path.suffix.lower() == ".fcidump")
            and "benchmark_cases" not in path.parts
        ]
    else:
        committed_inputs = [
            path
            for path in (ROOT / "data").rglob("*")
            if path.is_file()
            and (path.name.upper() == "FCIDUMP" or path.suffix.lower() == ".fcidump")
        ]
    assert committed_inputs == []
    assert "data/hamiltonians/" in (ROOT / ".gitignore").read_text(
        encoding="utf-8"
    )


def test_materialization_and_path_guard(tmp_path: Path) -> None:
    artifacts = _artifact_module()
    source = b"first\nsecond\n"
    assert artifacts._materialize(source, "copy") == source
    assert artifacts._materialize(source, "lf_to_crlf") == b"first\r\nsecond\r\n"
    assert artifacts._materialize(b"first\r\nsecond\r\n", "lf_to_crlf") == (
        b"first\r\nsecond\r\n"
    )
    assert artifacts._safe_target(tmp_path, "N2/FCIDUMP") == (
        tmp_path / "N2" / "FCIDUMP"
    ).resolve()
    with pytest.raises(ValueError):
        artifacts._safe_target(tmp_path, "../FCIDUMP")
    with pytest.raises(ValueError):
        artifacts._materialize(source, "unknown")
