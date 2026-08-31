#!/usr/bin/env python3
"""Focused tests for the external comparison runner CLI."""

from __future__ import annotations

import csv
import subprocess
import tempfile
import sys
from pathlib import Path, PureWindowsPath


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_external_comparison.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_external_comparison as runner  # noqa: E402

REPORT_FAMILIES = REPO_ROOT / "tests" / "corpus" / "manifests" / "report_families.tsv"
REQUIRED_OUTPUT_FILES = {
    "project_observations.tsv",
    "baseline_observations.tsv",
    "dependency_status.tsv",
    "study.tsv",
    "summary.md",
    "manifest.tsv",
}

TARGET_EXPECTATIONS = {
    "qr-minnorm": {
        "fixture_key": "qr_underdetermined_minnorm_2x4",
        "subfamily": "qr_minnorm",
        "operation": "minnorm_solve",
        "required_helper": "tests/qr_external_dense_reference.py",
        "generator_command": "python3 scripts/run_external_comparison.py --target qr-minnorm",
        "artifact_pattern": "build/comparison/qr_minnorm/study.tsv",
        "expected_metrics": {
            "project_status",
            "baseline_status",
            "residual_norm",
            "solution_norm",
            "solution_values",
            "project_vs_baseline_max_abs_delta",
        },
        "success_message": "external-comparison: qr-minnorm project-vs-baseline comparison passed",
    },
    "qr-compatible-ls": {
        "fixture_key": "qr_overdetermined_compatible_5x3",
        "subfamily": "qr_compatible_ls",
        "operation": "least_squares_solve",
        "required_helper": "tests/qr_external_dense_reference.py",
        "generator_command": "python3 scripts/run_external_comparison.py --target qr-compatible-ls",
        "artifact_pattern": "build/comparison/qr_compatible_ls/study.tsv",
        "expected_metrics": {
            "project_status",
            "baseline_status",
            "residual_norm",
            "solution_norm",
            "solution_values",
            "project_vs_baseline_max_abs_delta",
        },
        "success_message": (
            "external-comparison: qr-compatible-ls project-vs-baseline comparison passed"
        ),
    },
    "partial-svd-diag6-k2": {
        "fixture_key": "partial_svd_diag6_k2",
        "subfamily": "partial_svd_diag6_k2",
        "operation": "partial_svd",
        "required_helper": "tests/svd_external_dense_reference.py",
        "generator_command": (
            "python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2"
        ),
        "artifact_pattern": "build/comparison/partial_svd_diag6_k2/study.tsv",
        "expected_metrics": {
            "project_status",
            "baseline_status",
            "singular_value_0",
            "singular_value_1",
            "singular_values_max_abs_delta",
            "residual_norm",
            "u_orthogonality",
            "v_orthogonality",
            "u_projector_diag",
            "v_projector_diag",
        },
        "success_message": (
            "external-comparison: partial-svd-diag6-k2 project-vs-baseline comparison passed"
        ),
    },
    "lu-nonsym-square-5": {
        "fixture_key": "lu_nonsym_square_5",
        "subfamily": "lu_nonsym_square_5",
        "operation": "square_solve",
        "required_helper": "tests/lu_external_dense_reference.py",
        "generator_command": "python3 scripts/run_external_comparison.py --target lu-nonsym-square-5",
        "artifact_pattern": "build/comparison/lu_nonsym_square_5/study.tsv",
        "expected_metrics": {
            "project_status",
            "baseline_status",
            "residual_norm",
            "solution_norm",
            "solution_values",
            "project_vs_baseline_max_abs_delta",
        },
        "success_message": (
            "external-comparison: lu-nonsym-square-5 project-vs-baseline comparison passed"
        ),
    },
    "cholesky-spd-tridiag-5": {
        "fixture_key": "cholesky_spd_tridiag_5",
        "subfamily": "cholesky_spd_tridiag_5",
        "operation": "cholesky_spd_solve",
        "required_helper": "tests/chol_external_dense_reference.py",
        "generator_command": (
            "python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5"
        ),
        "artifact_pattern": "build/comparison/cholesky_spd_tridiag_5/study.tsv",
        "expected_metrics": {
            "project_status",
            "baseline_status",
            "residual_norm",
            "solution_norm",
            "solution_values",
            "project_vs_baseline_max_abs_delta",
        },
        "success_message": (
            "external-comparison: cholesky-spd-tridiag-5 "
            "project-vs-baseline comparison passed"
        ),
    },
}


def run_command(args: list[str], *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True)
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(args)}")
    return result


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_manifest(path: Path) -> dict[str, str]:
    return {row["key"]: row["value"] for row in read_tsv(path)}


def read_report_family_rows() -> dict[str, dict[str, str]]:
    rows = read_tsv(REPORT_FAMILIES)
    return {row["subfamily"]: row for row in rows if row["report_family"] == "comparison"}


def expected_row_ids(fixture_key: str, metrics: set[str]) -> set[str]:
    return {f"comparison_{fixture_key}_{metric}_v1" for metric in metrics}


def assert_target_output(target: str, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    expected = TARGET_EXPECTATIONS[target]
    result = run_command(
        [
            "python3",
            str(SCRIPT),
            "--target",
            target,
            "--output-dir",
            str(output_dir),
        ]
    )
    assert expected["success_message"] in result.stdout

    missing = sorted(name for name in REQUIRED_OUTPUT_FILES if not (output_dir / name).is_file())
    if missing:
        raise AssertionError(f"{target} did not generate required files: {missing}")

    manifest = read_manifest(output_dir / "manifest.tsv")
    assert manifest["target"] == target
    assert manifest["fixture_key"] == expected["fixture_key"]
    assert manifest["study_path"] == str(output_dir / "study.tsv")

    rows = read_tsv(output_dir / "study.tsv")
    expected_metrics = expected["expected_metrics"]
    assert len(rows) == len(expected_metrics)
    assert {row["comparison_row_id"] for row in rows} == expected_row_ids(
        expected["fixture_key"], expected_metrics
    )
    assert {row["metric"] for row in rows} == expected_metrics
    assert {row["status"] for row in rows} == {"pass"}
    assert {row["report_family"] for row in rows} == {"comparison"}
    assert {row["subfamily"] for row in rows} == {expected["subfamily"]}
    assert {row["fixture_key"] for row in rows} == {expected["fixture_key"]}
    assert {row["operation"] for row in rows} == {expected["operation"]}
    assert {row["support_tier"] for row in rows} == {"local_only"}
    assert {row["artifact_path"] for row in rows} == {str(output_dir / "study.tsv")}

    dependency_rows = read_tsv(output_dir / "dependency_status.tsv")
    required_rows = {row["dependency"]: row for row in dependency_rows if row["required"] == "yes"}
    assert expected["required_helper"] in required_rows
    assert required_rows[expected["required_helper"]]["status"] == "pass"
    assert (
        required_rows[expected["required_helper"]]["caveat"]
        == "source-controlled dense reference helper; not an external package"
    )
    optional_rows = {row["dependency"]: row for row in dependency_rows if row["required"] == "no"}
    assert {"numpy", "scipy"} <= set(optional_rows)
    for dependency in ("numpy", "scipy"):
        row = optional_rows[dependency]
        assert row["status"] == "defer"
        assert row["status_reason"] == "optional_package_baseline_not_selected"
        assert row["caveat"] == "deferred rows are not pass evidence"

    if not expected.get("require_report_family_metadata", True):
        return

    metadata = read_report_family_rows()[expected["subfamily"]]
    assert metadata["row_meaning"] == "external_process_dense_reference_comparison"
    assert metadata["row_origin"] == "generated_local"
    assert metadata["status"] == "unknown"
    assert metadata["support_tier"] == "local_only"
    assert metadata["freshness_policy"] == "generated_compare_inputs"
    assert metadata["generator_command"] == expected["generator_command"]
    assert metadata["artifact_pattern"] == expected["artifact_pattern"]
    assert "state-of-the-art" in metadata["non_claims"]
    assert "parity" in metadata["non_claims"]


def test_unsupported_target_reports_supported_targets() -> None:
    result = run_command(
        ["python3", str(SCRIPT), "--target", "not-a-target"],
        expect_success=False,
    )
    assert "ERROR unsupported_target:" in result.stderr
    assert "supported targets:" in result.stderr
    assert "qr-compatible-ls" in result.stderr
    assert "qr-minnorm" in result.stderr
    assert "partial-svd-diag6-k2" in result.stderr
    assert "lu-nonsym-square-5" in result.stderr
    assert "cholesky-spd-tridiag-5" in result.stderr


def test_selected_targets_generate_expected_rows_and_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-external-comparison-test-") as tmp:
        tmp_root = Path(tmp)
        for target in sorted(TARGET_EXPECTATIONS):
            assert_target_output(target, tmp_root / target)


def test_cmake_path_literal_uses_forward_slashes_for_windows_paths() -> None:
    literal = runner.cmake_path_literal(
        PureWindowsPath(r"D:\a\linalg_sparse_orthogonal\build\sparse_lu_ortho.lib")
    )
    assert literal == "D:/a/linalg_sparse_orthogonal/build/sparse_lu_ortho.lib"
    assert "\\a" not in literal


def main() -> int:
    test_cmake_path_literal_uses_forward_slashes_for_windows_paths()
    test_unsupported_target_reports_supported_targets()
    test_selected_targets_generate_expected_rows_and_metadata()
    print("test-run-external-comparison: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
