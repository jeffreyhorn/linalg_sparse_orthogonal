#!/usr/bin/env python3
"""Validate selected report target manifest parser diagnostics."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_corpus_schema import (  # noqa: E402
    CorpusValidationError,
    read_tsv,
    validate_selected_report_targets,
)


MANIFEST_PATH = REPO_ROOT / "tests" / "corpus" / "manifests" / "selected_report_targets.tsv"
REPORT_FAMILIES_PATH = REPO_ROOT / "tests" / "corpus" / "manifests" / "report_families.tsv"
WINDOWS_DEFERRAL_RECORD = (
    REPO_ROOT
    / "docs"
    / "planning"
    / "EPIC_16"
    / "SPRINT_182"
    / "artifacts"
    / "windows-report-freshness-deferral-decision.md"
)
WINDOWS_CHOLESKY_TARGET_ID = "SRT-COMP-CHOLESKY-SPD-TRIDIAG-5"
WINDOWS_CHOLESKY_WORKFLOW_FILE = ".github/workflows/windows-ci.yml"
WINDOWS_CHOLESKY_WORKFLOW_JOB = "selected-comparison-freshness"
WINDOWS_CHOLESKY_ARTIFACT = "sprint190-windows-selected-comparison-cholesky"
WINDOWS_CHOLESKY_EXPECTED_ROWS = "6"
WINDOWS_CHOLESKY_REQUIRED_FILES = (
    "project_observations.tsv",
    "baseline_observations.tsv",
    "dependency_status.tsv",
    "study.tsv",
    "summary.md",
    "manifest.tsv",
)


def manifest_rows() -> list[dict[str, str]]:
    return read_tsv(MANIFEST_PATH)


def report_family_rows() -> list[dict[str, str]]:
    return read_tsv(REPORT_FAMILIES_PATH)


def assert_invalid(rows: list[dict[str, str]], expected: str) -> None:
    try:
        validate_selected_report_targets(MANIFEST_PATH, rows, report_family_rows())
    except CorpusValidationError as exc:
        message = str(exc)
        if expected not in message:
            raise AssertionError(f"expected {expected!r} in {message!r}") from exc
        return
    raise AssertionError(f"expected validation failure containing {expected!r}")


def assert_invalid_with_all(rows: list[dict[str, str]], expected: list[str]) -> None:
    try:
        validate_selected_report_targets(MANIFEST_PATH, rows, report_family_rows())
    except CorpusValidationError as exc:
        message = str(exc)
        missing = [part for part in expected if part not in message]
        if missing:
            raise AssertionError(f"expected {missing!r} in {message!r}") from exc
        return
    raise AssertionError(f"expected validation failure containing {expected!r}")


def split_manifest_values(value: str) -> list[str]:
    if value == "none":
        return []
    return [part for part in value.split(";") if part]


def cholesky_row_index(rows: list[dict[str, str]]) -> int:
    matches = [
        index
        for index, row in enumerate(rows)
        if row["target_id"] == WINDOWS_CHOLESKY_TARGET_ID
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {WINDOWS_CHOLESKY_TARGET_ID} row, got {len(matches)}"
        )
    return matches[0]


def with_windows_cholesky_metadata(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = copy.deepcopy(rows)
    row = rows[cholesky_row_index(rows)]
    row["workflow_file"] += f";{WINDOWS_CHOLESKY_WORKFLOW_FILE}"
    row["workflow_job"] += f";{WINDOWS_CHOLESKY_WORKFLOW_JOB}"
    row["workflow_artifact"] += f";{WINDOWS_CHOLESKY_ARTIFACT}"
    row["workflow_platforms"] += ";windows"
    return rows


def assert_windows_cholesky_manifest_allowlist(rows: list[dict[str, str]]) -> None:
    windows_rows = [
        row
        for row in rows
        if "windows" in split_manifest_values(row["workflow_platforms"])
    ]
    if len(windows_rows) != 1:
        raise AssertionError(
            f"expected exactly one Windows selected target, got {len(windows_rows)}"
        )
    row = windows_rows[0]
    if row["target_id"] != WINDOWS_CHOLESKY_TARGET_ID:
        raise AssertionError(
            "only selected Cholesky may list windows, got "
            f"{row['target_id']}"
        )
    platforms = split_manifest_values(row["workflow_platforms"])
    files = split_manifest_values(row["workflow_file"])
    jobs = split_manifest_values(row["workflow_job"])
    artifacts = split_manifest_values(row["workflow_artifact"])
    try:
        windows_index = platforms.index("windows")
    except ValueError as exc:
        raise AssertionError(f"{WINDOWS_CHOLESKY_TARGET_ID} missing windows platform") from exc
    for field_name, values, expected in (
        ("workflow_file", files, WINDOWS_CHOLESKY_WORKFLOW_FILE),
        ("workflow_job", jobs, WINDOWS_CHOLESKY_WORKFLOW_JOB),
        ("workflow_artifact", artifacts, WINDOWS_CHOLESKY_ARTIFACT),
    ):
        if len(values) != len(platforms):
            raise AssertionError(
                f"{WINDOWS_CHOLESKY_TARGET_ID} {field_name} must align with workflow_platforms"
            )
        if values[windows_index] != expected:
            raise AssertionError(
                f"{WINDOWS_CHOLESKY_TARGET_ID} {field_name} windows entry must be {expected!r}"
            )
    if row["expected_rows"] != WINDOWS_CHOLESKY_EXPECTED_ROWS:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} expected_rows must remain "
            f"{WINDOWS_CHOLESKY_EXPECTED_ROWS}"
        )
    required_files = tuple(split_manifest_values(row["required_files"]))
    if required_files != WINDOWS_CHOLESKY_REQUIRED_FILES:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} required_files drifted for Windows promotion"
        )
    reused_artifacts = {
        artifact
        for artifact, platform in zip(artifacts, platforms)
        if platform != "windows" and artifact == WINDOWS_CHOLESKY_ARTIFACT
    }
    if reused_artifacts:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} reuses Windows artifact on non-Windows platform"
        )


def assert_no_windows_selected_platform(
    rows: list[dict[str, str]],
    deferral_path: Path = WINDOWS_DEFERRAL_RECORD,
) -> None:
    try:
        deferral_text = deferral_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AssertionError(
            f"Windows report freshness deferral record file is missing: {deferral_path}"
        ) from exc
    if "Windows report freshness remains formally deferred" not in deferral_text:
        raise AssertionError(
            "Windows report freshness deferral record marker text is missing"
        )
    for row in rows:
        platforms = split_manifest_values(row["workflow_platforms"])
        if "windows" in platforms:
            raise AssertionError(
                "selected_report_targets.tsv must not list windows while "
                "Windows report freshness remains formally deferred: "
                f"{row['target_id']}"
            )


def test_current_manifest_validates() -> None:
    validate_selected_report_targets(MANIFEST_PATH, manifest_rows(), report_family_rows())


def test_duplicate_target_id_fails_clearly() -> None:
    rows = manifest_rows()
    rows[1]["target_id"] = rows[0]["target_id"]
    assert_invalid(rows, "duplicate target_id")


def test_duplicate_target_key_fails_clearly() -> None:
    rows = manifest_rows()
    rows[1]["family"] = rows[0]["family"]
    rows[1]["subfamily"] = rows[0]["subfamily"]
    rows[1]["target_key"] = rows[0]["target_key"]
    assert_invalid(rows, "duplicate family/subfamily/target_key")


def test_unsupported_support_tier_fails_clearly() -> None:
    rows = manifest_rows()
    rows[0]["support_tier"] = "portable_performance"
    assert_invalid_with_all(
        rows,
        ["target_id=SRT-ORACLE-QR-PSVD-LOCAL", "invalid support_tier"],
    )


def test_unsupported_freshness_policy_fails_clearly() -> None:
    rows = manifest_rows()
    rows[0]["freshness_policy"] = "generated_without_commit_check"
    assert_invalid_with_all(
        rows,
        ["target_id=SRT-ORACLE-QR-PSVD-LOCAL", "invalid freshness_policy"],
    )


def test_missing_artifact_pattern_fails_clearly() -> None:
    rows = manifest_rows()
    rows[0]["artifact_pattern"] = "none"
    assert_invalid_with_all(
        rows,
        ["target_id=SRT-ORACLE-QR-PSVD-LOCAL", "selected targets require artifact_pattern"],
    )


def test_parent_traversal_artifact_pattern_fails_clearly() -> None:
    rows = manifest_rows()
    rows[0]["artifact_pattern"] = "../build/corpus/oracle/*.tsv"
    assert_invalid_with_all(
        rows,
        [
            "target_id=SRT-ORACLE-QR-PSVD-LOCAL",
            "artifact_pattern must be a repo-relative path",
        ],
    )


def test_bad_expected_rows_fails_clearly() -> None:
    rows = manifest_rows()
    rows[0]["expected_rows"] = "zero"
    assert_invalid_with_all(
        rows,
        ["target_id=SRT-ORACLE-QR-PSVD-LOCAL", "expected_rows must be a positive integer"],
    )


def test_missing_expected_row_ids_fails_clearly() -> None:
    rows = manifest_rows()
    rows[0]["expected_row_ids"] = "none"
    assert_invalid_with_all(
        rows,
        ["target_id=SRT-ORACLE-QR-PSVD-LOCAL", "countable selected targets require"],
    )


def test_missing_generated_required_files_fails_clearly() -> None:
    rows = manifest_rows()
    rows[0]["required_files"] = "none"
    assert_invalid_with_all(
        rows,
        ["target_id=SRT-ORACLE-QR-PSVD-LOCAL", "generated selected targets require required_files"],
    )


def test_missing_hosted_workflow_metadata_fails_clearly() -> None:
    rows = manifest_rows()
    rows[-1]["workflow_artifact"] = "none"
    assert_invalid_with_all(
        rows,
        ["target_id=SRT-BENCH-REFACTOR-CSC-NOS4", "hosted selected targets require"],
    )


def test_mismatched_workflow_artifact_platforms_fail_clearly() -> None:
    rows = manifest_rows()
    rows[1]["workflow_artifact"] = "linux-upload;macos-upload"
    rows[1]["workflow_platforms"] = "linux"
    assert_invalid_with_all(
        rows,
        [
            "target_id=SRT-COMP-QR-MINNORM",
            "workflow_artifact must contain either one shared artifact name",
        ],
    )


def test_missing_report_family_mapping_fails_clearly() -> None:
    rows = manifest_rows()
    rows[0]["subfamily"] = "unregistered_selected_target"
    assert_invalid(rows, "not found in report_families.tsv")


def test_artifact_expected_count_collision_fails_clearly() -> None:
    rows = manifest_rows()
    duplicate = copy.deepcopy(rows[1])
    duplicate["target_id"] = "SRT-COMP-QR-MINNORM-COUNT-DRIFT"
    duplicate["target_key"] = "qr-minnorm-count-drift"
    duplicate["expected_rows"] = "7"
    rows.append(duplicate)
    assert_invalid(rows, "duplicate artifact/generator key")


def test_unpromoted_report_families_remain_unselected() -> None:
    selected_families = {row["family"] for row in manifest_rows()}
    assert selected_families == {"oracle", "comparison", "benchmark"}
    assert not selected_families & {
        "package",
        "ci",
        "documentation",
        "sentinel",
        "guardrail",
        "deadcode",
        "coverage",
    }


def test_windows_report_freshness_deferral_keeps_manifest_unselected() -> None:
    assert_no_windows_selected_platform(manifest_rows())


def test_windows_deferral_record_missing_file_fails_clearly() -> None:
    missing_path = WINDOWS_DEFERRAL_RECORD.with_name("missing-windows-deferral.md")
    try:
        assert_no_windows_selected_platform(manifest_rows(), missing_path)
    except AssertionError as exc:
        message = str(exc)
        expected = f"Windows report freshness deferral record file is missing: {missing_path}"
        if expected not in message:
            raise AssertionError(f"expected missing-file diagnostic in {message!r}") from exc
        return
    raise AssertionError("expected missing Windows deferral record file to fail")


def test_windows_deferral_record_missing_marker_fails_clearly() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        markerless_path = Path(tmp_dir) / "windows-report-freshness-deferral.md"
        markerless_path.write_text("Windows deferral placeholder\n", encoding="utf-8")
        try:
            assert_no_windows_selected_platform(manifest_rows(), markerless_path)
        except AssertionError as exc:
            message = str(exc)
            expected = "Windows report freshness deferral record marker text is missing"
            if expected not in message:
                raise AssertionError(
                    f"expected missing-marker diagnostic in {message!r}"
                ) from exc
            return
        raise AssertionError("expected missing Windows deferral marker to fail")


def test_windows_platform_drift_fails_clearly() -> None:
    rows = manifest_rows()
    rows[1]["workflow_platforms"] = f"{rows[1]['workflow_platforms']};windows"
    try:
        assert_no_windows_selected_platform(rows)
    except AssertionError as exc:
        message = str(exc)
        expected = (
            "selected_report_targets.tsv must not list windows while "
            "Windows report freshness remains formally deferred"
        )
        if expected not in message or "SRT-COMP-QR-MINNORM" not in message:
            raise AssertionError(f"expected Windows drift diagnostic in {message!r}") from exc
        return
    raise AssertionError("expected Windows selected platform drift to fail")


def test_future_windows_cholesky_metadata_allowlist_accepts_exact_row() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    assert_windows_cholesky_manifest_allowlist(rows)


def test_future_windows_metadata_rejects_unselected_target() -> None:
    rows = manifest_rows()
    rows[1]["workflow_platforms"] = f"{rows[1]['workflow_platforms']};windows"
    rows[1]["workflow_file"] = f"{rows[1]['workflow_file']};{WINDOWS_CHOLESKY_WORKFLOW_FILE}"
    rows[1]["workflow_job"] = f"{rows[1]['workflow_job']};{WINDOWS_CHOLESKY_WORKFLOW_JOB}"
    rows[1]["workflow_artifact"] = f"{rows[1]['workflow_artifact']};{WINDOWS_CHOLESKY_ARTIFACT}"
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "only selected Cholesky may list windows" not in str(exc):
            raise
        return
    raise AssertionError("expected unselected Windows manifest target to fail")


def test_future_windows_metadata_rejects_wrong_artifact() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    row = rows[cholesky_row_index(rows)]
    row["workflow_artifact"] = row["workflow_artifact"].replace(
        WINDOWS_CHOLESKY_ARTIFACT,
        "sprint175-macos-selected-comparison-freshness",
    )
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "workflow_artifact windows entry" not in str(exc):
            raise
        return
    raise AssertionError("expected wrong Windows Cholesky artifact to fail")


def test_future_windows_metadata_rejects_row_count_drift() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    rows[cholesky_row_index(rows)]["expected_rows"] = "7"
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "expected_rows must remain 6" not in str(exc):
            raise
        return
    raise AssertionError("expected Windows Cholesky row-count drift to fail")


def test_future_windows_metadata_rejects_missing_artifact_file() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    row = rows[cholesky_row_index(rows)]
    row["required_files"] = row["required_files"].replace(";manifest.tsv", "")
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "required_files drifted" not in str(exc):
            raise
        return
    raise AssertionError("expected Windows Cholesky required-file drift to fail")


def main() -> int:
    test_current_manifest_validates()
    test_duplicate_target_id_fails_clearly()
    test_duplicate_target_key_fails_clearly()
    test_unsupported_support_tier_fails_clearly()
    test_unsupported_freshness_policy_fails_clearly()
    test_missing_artifact_pattern_fails_clearly()
    test_parent_traversal_artifact_pattern_fails_clearly()
    test_bad_expected_rows_fails_clearly()
    test_missing_expected_row_ids_fails_clearly()
    test_missing_generated_required_files_fails_clearly()
    test_missing_hosted_workflow_metadata_fails_clearly()
    test_mismatched_workflow_artifact_platforms_fail_clearly()
    test_missing_report_family_mapping_fails_clearly()
    test_artifact_expected_count_collision_fails_clearly()
    test_unpromoted_report_families_remain_unselected()
    test_windows_report_freshness_deferral_keeps_manifest_unselected()
    test_windows_deferral_record_missing_file_fails_clearly()
    test_windows_deferral_record_missing_marker_fails_clearly()
    test_windows_platform_drift_fails_clearly()
    test_future_windows_cholesky_metadata_allowlist_accepts_exact_row()
    test_future_windows_metadata_rejects_unselected_target()
    test_future_windows_metadata_rejects_wrong_artifact()
    test_future_windows_metadata_rejects_row_count_drift()
    test_future_windows_metadata_rejects_missing_artifact_file()
    print("test-selected-report-targets-manifest: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
