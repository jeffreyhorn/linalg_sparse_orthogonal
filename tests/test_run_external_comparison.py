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
    "qr-incompatible-ls": {
        "fixture_key": "qr_overdetermined_incompatible_4x2",
        "subfamily": "qr_incompatible_ls",
        "operation": "least_squares_solve",
        "required_helper": "tests/qr_external_dense_reference.py",
        "generator_command": "python3 scripts/run_external_comparison.py --target qr-incompatible-ls",
        "artifact_pattern": "build/comparison/qr_incompatible_ls/study.tsv",
        "expected_metrics": {
            "project_status",
            "baseline_status",
            "residual_norm",
            "solution_norm",
            "solution_values",
            "project_vs_baseline_max_abs_delta",
        },
        "success_message": (
            "external-comparison: qr-incompatible-ls project-vs-baseline comparison passed"
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


def qr_incompatible_project_observations(
    *,
    residual_norm: str = "1.7320508075688772",
    solution_norm: str = "2.2360679774997898",
    solution_values: str = "2,-1",
    status: str = "SPARSE_SUCCESS",
) -> dict[str, str]:
    return {
        "status": status,
        "residual_norm": residual_norm,
        "solution_norm": solution_norm,
        "solution_values": solution_values,
        "project_probe_command": "synthetic qr_incompatible_ls probe",
    }


def qr_incompatible_baseline_observations(
    *,
    residual_norm: str = "1.7320508075688772",
    solution_norm: str = "2.2360679774997898",
    solution_values: str = "2,-1",
    status: str = "success",
) -> dict[str, str]:
    return {
        "status": status,
        "residual_norm": residual_norm,
        "solution_norm": solution_norm,
        "solution_values": solution_values,
        "baseline_command": "synthetic qr_external_dense_reference.py",
        "baseline_helper_path": "tests/qr_external_dense_reference.py",
        "baseline_python_executable": "python3",
        "baseline_python_version": "synthetic",
    }


def qr_incompatible_manifest(target: dict[str, object]) -> dict[str, str]:
    return {
        "baseline_name": runner.baseline_name(target),
        "baseline_type": "external-process-source-controlled-helper",
        "baseline_version": runner.baseline_version(target),
        "project_version": "synthetic",
        "source_commit": "synthetic",
        "source_branch": "sprint-191",
        "worktree_state": "clean",
        "platform": "test",
        "configuration": runner.comparison_configuration(target),
    }


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
    assert "qr-incompatible-ls" in result.stderr
    assert "qr-minnorm" in result.stderr
    assert "partial-svd-diag6-k2" in result.stderr
    assert "lu-nonsym-square-5" in result.stderr
    assert "cholesky-spd-tridiag-5" in result.stderr


def test_qr_incompatible_ls_fixture_contract() -> None:
    target = runner.TARGETS["qr-incompatible-ls"]
    entries, rows, cols = runner.descriptor_entries(target)
    assert rows == 4
    assert cols == 2
    assert entries == [
        (0, 0, 1.0),
        (1, 1, 1.0),
        (2, 0, 1.0),
        (2, 1, 1.0),
        (3, 0, 2.0),
        (3, 1, -1.0),
    ]
    assert target["rhs"] == [1.0, -2.0, 2.0, 5.0]
    assert target["expected_solution"] == [2.0, -1.0]
    assert target["expected_solution_norm"] == 2.2360679774997898
    assert target["expected_residual_norm"] == 1.7320508075688772
    assert target["baseline_value_count"] == 3

    project_rows = runner.project_observation_rows(
        {
            "status": "SPARSE_SUCCESS",
            "residual_norm": "1.7320508075688772",
            "solution_norm": "2.2360679774997898",
            "solution_values": "2,-1",
        },
        target,
    )
    baseline_rows = runner.baseline_observation_rows(
        {
            "status": "success",
            "residual_norm": "1.7320508075688772",
            "solution_norm": "2.2360679774997898",
            "solution_values": "2,-1",
        },
        target,
    )
    assert {row["status"] for row in project_rows} == {"pass"}
    assert {row["status"] for row in baseline_rows} == {"pass"}


def test_qr_incompatible_ls_reference_observations_and_dependencies() -> None:
    target = runner.TARGETS["qr-incompatible-ls"]
    observations = runner.run_baseline_reference(REPO_ROOT, target)
    assert observations["status"] == "success"
    assert observations["solution_values"] == "1.9999999999999998,-1"
    assert observations["residual_norm"] == "1.7320508075688772"
    assert abs(float(observations["solution_norm"]) - 2.2360679774997898) <= 1e-15
    assert observations["baseline_helper_path"] == "tests/qr_external_dense_reference.py"
    assert "qr_overdetermined_incompatible_4x2" in observations["baseline_command"]

    dependency_rows = runner.dependency_status_rows(REPO_ROOT, target)
    required_rows = {
        row["dependency"]: row for row in dependency_rows if row["required"] == "yes"
    }
    assert required_rows["python3"]["status"] == "pass"
    helper_row = required_rows["tests/qr_external_dense_reference.py"]
    assert helper_row["status"] == "pass"
    assert helper_row["status_reason"] == "baseline_helper_available"
    assert helper_row["caveat"] == "source-controlled dense reference helper; not an external package"

    optional_rows = {
        row["dependency"]: row for row in dependency_rows if row["required"] == "no"
    }
    assert optional_rows["numpy"]["status"] == "defer"
    assert optional_rows["scipy"]["status"] == "defer"


def test_qr_incompatible_ls_project_probe_observations() -> None:
    target = runner.TARGETS["qr-incompatible-ls"]
    observations, compiler = runner.run_project_probe(
        REPO_ROOT,
        runner.DEFAULT_LIBRARY,
        False,
        target,
        probe_build_system="compiler",
        cmake_generator=None,
        cmake_arch=None,
        cmake_config=runner.DEFAULT_CMAKE_CONFIG,
    )
    assert compiler
    assert observations["status"] == "SPARSE_SUCCESS"
    assert abs(float(observations["residual_norm"]) - 1.7320508075688772) <= 1e-10
    assert abs(float(observations["solution_norm"]) - 2.2360679774997898) <= 1e-10
    solution_values = [
        float(value) for value in observations["solution_values"].split(",")
    ]
    assert len(solution_values) == 2
    assert abs(solution_values[0] - 2.0) <= 1e-10
    assert abs(solution_values[1] + 1.0) <= 1e-10
    assert "qr_incompatible_ls_probe" in observations["project_probe_command"]

    project_rows = runner.project_observation_rows(observations, target)
    rows_by_metric = {row["metric"]: row for row in project_rows}
    assert rows_by_metric["project_status"]["status"] == "pass"
    assert rows_by_metric["residual_norm"]["status"] == "pass"
    assert rows_by_metric["residual_norm"]["status_reason"] == (
        "project_residual_matches_expected"
    )
    assert rows_by_metric["solution_norm"]["status"] == "pass"
    assert rows_by_metric["solution_values"]["status"] == "pass"


def test_qr_incompatible_ls_project_rows_reject_residual_mismatch() -> None:
    target = runner.TARGETS["qr-incompatible-ls"]
    rows = runner.project_observation_rows(
        {
            "status": "SPARSE_SUCCESS",
            "residual_norm": "0",
            "solution_norm": "2.2360679774997898",
            "solution_values": "2,-1",
        },
        target,
    )
    residual_row = next(row for row in rows if row["metric"] == "residual_norm")
    assert residual_row["status"] == "fail"
    assert residual_row["status_reason"] == "project_residual_expected_mismatch"


def test_qr_incompatible_ls_project_rows_reject_solution_mismatch() -> None:
    target = runner.TARGETS["qr-incompatible-ls"]
    rows = runner.project_observation_rows(
        {
            "status": "SPARSE_SUCCESS",
            "residual_norm": "1.7320508075688772",
            "solution_norm": "2.2360679774997898",
            "solution_values": "2.25,-1",
        },
        target,
    )
    rows_by_metric = {row["metric"]: row for row in rows}
    assert rows_by_metric["solution_values"]["status"] == "fail"
    assert rows_by_metric["solution_values"]["status_reason"] == (
        "project_solution_values_tolerance_miss"
    )


def test_qr_incompatible_ls_tolerance_boundaries_pass_and_fail() -> None:
    target = dict(runner.TARGETS["qr-incompatible-ls"])
    target["residual_tolerance"] = 1e-9
    target["solution_tolerance"] = 1e-9

    project_rows = runner.project_observation_rows(
        qr_incompatible_project_observations(
            residual_norm="1.7320508080688772",
            solution_norm="2.2360679779997898",
            solution_values="2.0000000005,-1",
        ),
        target,
    )
    baseline_rows = runner.baseline_observation_rows(
        qr_incompatible_baseline_observations(
            residual_norm="1.7320508080688772",
            solution_norm="2.2360679779997898",
            solution_values="2.0000000005,-1",
        ),
        target,
    )
    assert {row["status"] for row in project_rows} == {"pass"}
    assert {row["status"] for row in baseline_rows} == {"pass"}

    project_rows = runner.project_observation_rows(
        qr_incompatible_project_observations(
            residual_norm="1.7320508095688772",
            solution_norm="2.2360679794997898",
            solution_values="2.000000002,-1",
        ),
        target,
    )
    baseline_rows = runner.baseline_observation_rows(
        qr_incompatible_baseline_observations(
            residual_norm="1.7320508095688772",
            solution_norm="2.2360679794997898",
            solution_values="2.000000002,-1",
        ),
        target,
    )
    project_by_metric = {row["metric"]: row for row in project_rows}
    baseline_by_metric = {row["metric"]: row for row in baseline_rows}
    assert project_by_metric["residual_norm"]["status"] == "fail"
    assert project_by_metric["solution_norm"]["status"] == "fail"
    assert project_by_metric["solution_values"]["status"] == "fail"
    assert baseline_by_metric["baseline_residual_norm"]["status"] == "fail"
    assert baseline_by_metric["baseline_solution_norm"]["status"] == "fail"
    assert baseline_by_metric["baseline_solution_values"]["status"] == "fail"


def test_qr_incompatible_ls_study_rows_reject_tolerance_miss() -> None:
    target = dict(runner.TARGETS["qr-incompatible-ls"])
    target["residual_tolerance"] = 1e-9
    target["solution_tolerance"] = 1e-9
    passing_rows = runner.comparison_study_rows(
        artifact_path="build/comparison/qr_incompatible_ls/study.tsv",
        baseline_observations=qr_incompatible_baseline_observations(
            residual_norm="1.7320508080688772",
            solution_norm="2.2360679779997898",
            solution_values="2.0000000005,-1",
        ),
        compiler="synthetic",
        generated_at="2026-09-01T00:00:00Z",
        manifest=qr_incompatible_manifest(target),
        observations=qr_incompatible_project_observations(),
        target=target,
    )
    assert {row["status"] for row in passing_rows} == {"pass"}
    runner.validate_selected_study_rows(passing_rows, target)

    failing_rows = runner.comparison_study_rows(
        artifact_path="build/comparison/qr_incompatible_ls/study.tsv",
        baseline_observations=qr_incompatible_baseline_observations(
            residual_norm="1.7320508095688772",
            solution_norm="2.2360679794997898",
            solution_values="2.000000002,-1",
        ),
        compiler="synthetic",
        generated_at="2026-09-01T00:00:00Z",
        manifest=qr_incompatible_manifest(target),
        observations=qr_incompatible_project_observations(),
        target=target,
    )
    rows_by_metric = {row["metric"]: row for row in failing_rows}
    assert rows_by_metric["residual_norm"]["status_reason"] == (
        "project_baseline_residual_delta_tolerance_miss"
    )
    assert rows_by_metric["solution_norm"]["status_reason"] == (
        "project_baseline_solution_norm_delta_tolerance_miss"
    )
    assert rows_by_metric["solution_values"]["status_reason"] == (
        "project_baseline_solution_values_delta_tolerance_miss"
    )
    assert rows_by_metric["project_vs_baseline_max_abs_delta"]["status_reason"] == (
        "project_baseline_max_abs_delta_tolerance_miss"
    )
    try:
        runner.validate_selected_study_rows(failing_rows, target)
    except runner.ComparisonError as exc:
        assert exc.failure_class == "metric_tolerance_miss"
        assert "project_baseline_residual_delta_tolerance_miss" in str(exc)
        assert "project_baseline_max_abs_delta_tolerance_miss" in str(exc)
    else:
        raise AssertionError("failing QR incompatible study rows unexpectedly passed")


def test_qr_incompatible_ls_project_parser_rejects_missing_fields() -> None:
    try:
        runner.parse_key_values(
            "status=SPARSE_SUCCESS\nresidual_norm=1.7320508075688772\n",
            {"status", "residual_norm", "solution_norm", "solution_values"},
        )
    except runner.ComparisonError as exc:
        assert exc.failure_class == "project_probe_failed"
        assert "solution_norm" in str(exc)
        assert "solution_values" in str(exc)
    else:
        raise AssertionError("malformed QR incompatible project output unexpectedly parsed")


def test_qr_incompatible_ls_reference_parser_rejects_malformed_output() -> None:
    original_run_capture = runner.run_capture

    def malformed_output(*args: object, **kwargs: object) -> str:
        return "OK 2\n2\n-1\n"

    runner.run_capture = malformed_output
    try:
        try:
            runner.run_baseline_reference(REPO_ROOT, runner.TARGETS["qr-incompatible-ls"])
        except runner.ComparisonError as exc:
            assert exc.failure_class == "baseline_malformed_output"
            assert "baseline value count must be 3" in str(exc)
        else:
            raise AssertionError("malformed QR incompatible reference unexpectedly succeeded")
    finally:
        runner.run_capture = original_run_capture


def test_qr_incompatible_ls_reference_reports_command_failure() -> None:
    original_run_capture = runner.run_capture

    def command_failure(*args: object, **kwargs: object) -> str:
        raise runner.ComparisonError("baseline_command_failed", "synthetic helper failure")

    runner.run_capture = command_failure
    try:
        try:
            runner.run_baseline_reference(REPO_ROOT, runner.TARGETS["qr-incompatible-ls"])
        except runner.ComparisonError as exc:
            assert exc.failure_class == "baseline_command_failed"
            assert "synthetic helper failure" in str(exc)
        else:
            raise AssertionError("failed QR incompatible reference unexpectedly succeeded")
    finally:
        runner.run_capture = original_run_capture


def test_qr_incompatible_ls_dependency_reports_missing_helper() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-missing-qr-helper-") as tmp:
        root = Path(tmp)
        dependency_rows = runner.dependency_status_rows(
            root, runner.TARGETS["qr-incompatible-ls"]
        )
        required_rows = {
            row["dependency"]: row for row in dependency_rows if row["required"] == "yes"
        }
        helper_row = required_rows["tests/qr_external_dense_reference.py"]
        assert helper_row["status"] == "error"
        assert helper_row["status_reason"] == "baseline_helper_missing"

        try:
            runner.run_baseline_reference(root, runner.TARGETS["qr-incompatible-ls"])
        except runner.ComparisonError as exc:
            assert exc.failure_class == "missing_baseline_helper"
            assert "tests/qr_external_dense_reference.py" in str(exc)
        else:
            raise AssertionError("missing QR helper unexpectedly succeeded")


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


def test_ensure_library_fails_structurally_for_default_library_on_windows() -> None:
    original_default_library = runner.DEFAULT_LIBRARY
    original_platform_system = runner.platform.system
    with tempfile.TemporaryDirectory(prefix="sparse-missing-default-lib-") as tmp:
        missing_library = Path(tmp) / "build" / "libsparse_lu_ortho.a"
        runner.DEFAULT_LIBRARY = missing_library
        runner.platform.system = lambda: "Windows"
        try:
            try:
                runner.ensure_library(Path(tmp), missing_library)
            except runner.ComparisonError as exc:
                assert exc.failure_class == "project_build_failed"
                assert "default Unix static library is missing on Windows" in str(exc)
                assert "--library" in str(exc)
            else:
                raise AssertionError("missing Windows default library unexpectedly succeeded")
        finally:
            runner.DEFAULT_LIBRARY = original_default_library
            runner.platform.system = original_platform_system


def test_ensure_library_wraps_missing_make_as_comparison_error() -> None:
    original_default_library = runner.DEFAULT_LIBRARY
    original_platform_system = runner.platform.system
    original_subprocess_run = runner.subprocess.run
    with tempfile.TemporaryDirectory(prefix="sparse-missing-make-") as tmp:
        missing_library = Path(tmp) / "build" / "libsparse_lu_ortho.a"
        runner.DEFAULT_LIBRARY = missing_library
        runner.platform.system = lambda: "Linux"

        def raise_missing_make(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise OSError("make not found")

        runner.subprocess.run = raise_missing_make
        try:
            try:
                runner.ensure_library(Path(tmp), missing_library)
            except runner.ComparisonError as exc:
                assert exc.failure_class == "project_build_failed"
                assert "failed to invoke make" in str(exc)
                assert "make not found" in str(exc)
            else:
                raise AssertionError("missing make unexpectedly succeeded")
        finally:
            runner.DEFAULT_LIBRARY = original_default_library
            runner.platform.system = original_platform_system
            runner.subprocess.run = original_subprocess_run


def main() -> int:
    test_cmake_path_literal_uses_forward_slashes_for_windows_paths()
    test_ensure_library_fails_structurally_for_default_library_on_windows()
    test_ensure_library_wraps_missing_make_as_comparison_error()
    test_unsupported_target_reports_supported_targets()
    test_qr_incompatible_ls_fixture_contract()
    test_qr_incompatible_ls_reference_observations_and_dependencies()
    test_qr_incompatible_ls_project_probe_observations()
    test_qr_incompatible_ls_project_rows_reject_residual_mismatch()
    test_qr_incompatible_ls_project_rows_reject_solution_mismatch()
    test_qr_incompatible_ls_tolerance_boundaries_pass_and_fail()
    test_qr_incompatible_ls_study_rows_reject_tolerance_miss()
    test_qr_incompatible_ls_project_parser_rejects_missing_fields()
    test_qr_incompatible_ls_reference_parser_rejects_malformed_output()
    test_qr_incompatible_ls_reference_reports_command_failure()
    test_qr_incompatible_ls_dependency_reports_missing_helper()
    test_selected_targets_generate_expected_rows_and_metadata()
    print("test-run-external-comparison: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
