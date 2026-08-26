#!/usr/bin/env python3
"""Validate selected report target manifest parser diagnostics."""

from __future__ import annotations

import copy
import sys
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
    print("test-selected-report-targets-manifest: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
