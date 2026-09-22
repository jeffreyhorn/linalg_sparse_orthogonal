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
WINDOWS_QR_INCOMPATIBLE_TARGET_ID = "SRT-COMP-QR-INCOMPATIBLE-LS"
WINDOWS_CHOLESKY_WORKFLOW_FILE = ".github/workflows/windows-ci.yml"
WINDOWS_CHOLESKY_WORKFLOW_JOB = "selected-comparison-freshness"
WINDOWS_CHOLESKY_ARTIFACT = "sprint190-windows-selected-comparison-cholesky"
WINDOWS_CHOLESKY_EXPECTED_ROWS = "6"
WINDOWS_CHOLESKY_FAMILY = "comparison"
WINDOWS_CHOLESKY_SUBFAMILY = "cholesky_spd_tridiag_5"
WINDOWS_CHOLESKY_TARGET_KEY = "cholesky-spd-tridiag-5"
WINDOWS_CHOLESKY_ARTIFACT_PATTERN = "build/comparison/cholesky_spd_tridiag_5/study.tsv"
WINDOWS_CHOLESKY_GENERATOR_COMMAND = (
    "python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5"
)
WINDOWS_CHOLESKY_CURRENT_SUPPORT_TIER = "local_only"
WINDOWS_CHOLESKY_PROMOTED_SUPPORT_TIER = "hosted_selected"
WINDOWS_CHOLESKY_CURRENT_CLAIM_SCOPE = (
    "Selected Cholesky SPD tridiagonal solve comparison rows are fresh for the "
    "named fixture against the selected source-controlled dense Cholesky reference helper."
)
WINDOWS_CHOLESKY_PROMOTED_CLAIM_SCOPE = (
    "Selected Cholesky SPD tridiagonal solve comparison rows are fresh for the "
    "named fixture on reviewed Linux, macOS, and Windows hosted lanes against the "
    "selected source-controlled dense Cholesky reference helper."
)
WINDOWS_CHOLESKY_WORKFLOW_FILES = (
    ".github/workflows/ci.yml",
    ".github/workflows/macos-ci.yml",
)
WINDOWS_CHOLESKY_WORKFLOW_JOBS = (
    "generated-report-freshness",
    "selected-comparison-freshness",
)
WINDOWS_CHOLESKY_WORKFLOW_ARTIFACTS = (
    "sprint175-linux-selected-comparison-freshness",
    "sprint175-macos-selected-comparison-freshness",
)
WINDOWS_CHOLESKY_WORKFLOW_PLATFORMS = ("linux", "macos")
WINDOWS_CHOLESKY_PROMOTED_WORKFLOW_FILES = (
    ".github/workflows/ci.yml",
    ".github/workflows/macos-ci.yml",
    WINDOWS_CHOLESKY_WORKFLOW_FILE,
)
WINDOWS_CHOLESKY_PROMOTED_WORKFLOW_JOBS = (
    "generated-report-freshness",
    "selected-comparison-freshness",
    WINDOWS_CHOLESKY_WORKFLOW_JOB,
)
WINDOWS_CHOLESKY_PROMOTED_WORKFLOW_ARTIFACTS = (
    "sprint175-linux-selected-comparison-freshness",
    "sprint175-macos-selected-comparison-freshness",
    WINDOWS_CHOLESKY_ARTIFACT,
)
WINDOWS_CHOLESKY_PROMOTED_WORKFLOW_PLATFORMS = (
    "linux",
    "macos",
    "windows",
)
WINDOWS_QR_INCOMPATIBLE_EXPECTED_ROWS = "6"
WINDOWS_QR_INCOMPATIBLE_WORKFLOW_FILES = (
    ".github/workflows/ci.yml",
    ".github/workflows/macos-ci.yml",
)
WINDOWS_QR_INCOMPATIBLE_WORKFLOW_JOBS = (
    "generated-report-freshness",
    "selected-comparison-freshness",
)
WINDOWS_QR_INCOMPATIBLE_WORKFLOW_ARTIFACTS = (
    "sprint175-linux-selected-comparison-freshness",
    "sprint175-macos-selected-comparison-freshness",
)
WINDOWS_QR_INCOMPATIBLE_WORKFLOW_PLATFORMS = ("linux", "macos")
WINDOWS_QR_INCOMPATIBLE_WORKFLOW_FILE = ".github/workflows/windows-ci.yml"
WINDOWS_QR_INCOMPATIBLE_WORKFLOW_JOB = "selected-qr-incompatible-comparison-freshness"
WINDOWS_QR_INCOMPATIBLE_ARTIFACT = "sprint209-windows-selected-comparison-qr-incompatible"
SELECTED_BENCHMARK_TARGET_ID = "SRT-BENCH-REFACTOR-CSC-NOS4"
SELECTED_BENCHMARK_REQUIRED_NON_CLAIMS = (
    "no portable performance claim",
    "no release benchmark claim",
    "no algorithmic superiority claim",
    "no platform parity",
    "no state-of-the-art claim",
    "no package or ABI support claim",
    "no broad package-manager distribution claim",
    "no Windows selected benchmark freshness",
)
WINDOWS_CHOLESKY_REQUIRED_FILES = (
    "project_observations.tsv",
    "baseline_observations.tsv",
    "dependency_status.tsv",
    "study.tsv",
    "summary.md",
    "manifest.tsv",
)
WINDOWS_CHOLESKY_EXPECTED_ROW_IDS = (
    "comparison_cholesky_spd_tridiag_5_project_status_v1",
    "comparison_cholesky_spd_tridiag_5_baseline_status_v1",
    "comparison_cholesky_spd_tridiag_5_residual_norm_v1",
    "comparison_cholesky_spd_tridiag_5_solution_norm_v1",
    "comparison_cholesky_spd_tridiag_5_solution_values_v1",
    "comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1",
)
WINDOWS_CHOLESKY_REQUIRED_NON_CLAIMS = (
    "no broad Cholesky correctness",
    "no broad SPD coverage",
    "no broad reordering coverage",
    "no CSC-vs-linked-list parity",
    "no fill superiority",
    "no Windows report freshness",
    "no package-manager proof",
    "no shared-library ABI proof",
    "no performance superiority",
    "no state-of-the-art claim",
)
WINDOWS_CHOLESKY_PROMOTED_NON_CLAIMS = (
    "no broad Cholesky correctness",
    "no broad SPD coverage",
    "no broad reordering coverage",
    "no CSC-vs-linked-list parity",
    "no fill superiority",
    "no broad Windows report freshness",
    "no package-manager proof",
    "no shared-library ABI proof",
    "no performance superiority",
    "no state-of-the-art claim",
)
WINDOWS_QR_INCOMPATIBLE_FAMILY = "comparison"
WINDOWS_QR_INCOMPATIBLE_SUBFAMILY = "qr_incompatible_ls"
WINDOWS_QR_INCOMPATIBLE_TARGET_KEY = "qr-incompatible-ls"
WINDOWS_QR_INCOMPATIBLE_ARTIFACT_PATTERN = "build/comparison/qr_incompatible_ls/study.tsv"
WINDOWS_QR_INCOMPATIBLE_GENERATOR_COMMAND = (
    "python3 scripts/run_external_comparison.py --target qr-incompatible-ls"
)
WINDOWS_QR_INCOMPATIBLE_CLAIM_SCOPE = (
    "Selected QR incompatible least-squares comparison rows are fresh for the "
    "named fixture against the selected source-controlled dense reference helper."
)
WINDOWS_QR_INCOMPATIBLE_REQUIRED_FILES = (
    "project_observations.tsv",
    "baseline_observations.tsv",
    "dependency_status.tsv",
    "study.tsv",
    "summary.md",
    "manifest.tsv",
)
WINDOWS_QR_INCOMPATIBLE_EXPECTED_ROW_IDS = (
    "comparison_qr_overdetermined_incompatible_4x2_project_status_v1",
    "comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1",
    "comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1",
    "comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1",
    "comparison_qr_overdetermined_incompatible_4x2_solution_values_v1",
    "comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1",
)
WINDOWS_QR_INCOMPATIBLE_REQUIRED_NON_CLAIMS = (
    "no broad QR parity",
    "no broad least-squares parity",
    "no raw QR basis identity",
    "no Q sign or orientation claim",
    "no global rank-threshold policy",
    "no broad rank-deficient solve claim",
    "no NumPy parity",
    "no SciPy parity",
    "no LAPACK parity",
    "no SuiteSparse parity",
    "no Eigen parity",
    "no Windows report freshness",
    "no package-manager proof",
    "no shared-library ABI proof",
    "no performance superiority",
    "no state-of-the-art claim",
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


def cholesky_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[cholesky_row_index(rows)]


def qr_incompatible_row(rows: list[dict[str, str]]) -> dict[str, str]:
    matches = [
        row for row in rows if row["target_id"] == WINDOWS_QR_INCOMPATIBLE_TARGET_ID
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {WINDOWS_QR_INCOMPATIBLE_TARGET_ID} row, got {len(matches)}"
        )
    return matches[0]


def selected_benchmark_row(rows: list[dict[str, str]]) -> dict[str, str]:
    matches = [row for row in rows if row["target_id"] == SELECTED_BENCHMARK_TARGET_ID]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {SELECTED_BENCHMARK_TARGET_ID} row, got {len(matches)}"
        )
    return matches[0]


def with_windows_cholesky_metadata(
    rows: list[dict[str, str]], *, promote_claims: bool = True
) -> list[dict[str, str]]:
    rows = copy.deepcopy(rows)
    row = cholesky_row(rows)
    row["workflow_file"] += f";{WINDOWS_CHOLESKY_WORKFLOW_FILE}"
    row["workflow_job"] += f";{WINDOWS_CHOLESKY_WORKFLOW_JOB}"
    row["workflow_artifact"] += f";{WINDOWS_CHOLESKY_ARTIFACT}"
    row["workflow_platforms"] += ";windows"
    if promote_claims:
        row["support_tier"] = WINDOWS_CHOLESKY_PROMOTED_SUPPORT_TIER
        row["claim_scope"] = WINDOWS_CHOLESKY_PROMOTED_CLAIM_SCOPE
        row["non_claims"] = ";".join(WINDOWS_CHOLESKY_PROMOTED_NON_CLAIMS)
    return rows


def assert_current_windows_cholesky_redeferral_contract(rows: list[dict[str, str]]) -> None:
    row = cholesky_row(rows)
    exact_fields = {
        "family": WINDOWS_CHOLESKY_FAMILY,
        "subfamily": WINDOWS_CHOLESKY_SUBFAMILY,
        "target_key": WINDOWS_CHOLESKY_TARGET_KEY,
        "artifact_pattern": WINDOWS_CHOLESKY_ARTIFACT_PATTERN,
        "generator_command": WINDOWS_CHOLESKY_GENERATOR_COMMAND,
        "support_tier": WINDOWS_CHOLESKY_CURRENT_SUPPORT_TIER,
        "claim_scope": WINDOWS_CHOLESKY_CURRENT_CLAIM_SCOPE,
    }
    for field, expected in exact_fields.items():
        if row[field] != expected:
            raise AssertionError(
                f"{WINDOWS_CHOLESKY_TARGET_ID} {field} must remain {expected!r}"
            )
    if row["expected_rows"] != WINDOWS_CHOLESKY_EXPECTED_ROWS:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} expected_rows must remain "
            f"{WINDOWS_CHOLESKY_EXPECTED_ROWS}"
        )
    required_files = tuple(split_manifest_values(row["required_files"]))
    if required_files != WINDOWS_CHOLESKY_REQUIRED_FILES:
        raise AssertionError(f"{WINDOWS_CHOLESKY_TARGET_ID} required_files drifted")
    expected_row_ids = tuple(split_manifest_values(row["expected_row_ids"]))
    if expected_row_ids != WINDOWS_CHOLESKY_EXPECTED_ROW_IDS:
        raise AssertionError(f"{WINDOWS_CHOLESKY_TARGET_ID} expected_row_ids drifted")
    workflow_files = tuple(split_manifest_values(row["workflow_file"]))
    if workflow_files != WINDOWS_CHOLESKY_WORKFLOW_FILES:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} workflow_file metadata must remain "
            "the current Linux/macOS selected freshness pair while re-deferred"
        )
    workflow_jobs = tuple(split_manifest_values(row["workflow_job"]))
    if workflow_jobs != WINDOWS_CHOLESKY_WORKFLOW_JOBS:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} workflow_job metadata must remain "
            "the current Linux/macOS selected freshness pair while re-deferred"
        )
    workflow_artifacts = tuple(split_manifest_values(row["workflow_artifact"]))
    if workflow_artifacts != WINDOWS_CHOLESKY_WORKFLOW_ARTIFACTS:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} workflow_artifact metadata must remain "
            "the current Linux/macOS selected freshness pair while re-deferred"
        )
    workflow_platforms = tuple(split_manifest_values(row["workflow_platforms"]))
    if workflow_platforms != WINDOWS_CHOLESKY_WORKFLOW_PLATFORMS:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} workflow_platforms metadata must remain "
            "linux/macos while re-deferred"
        )
    non_claims = tuple(split_manifest_values(row["non_claims"]))
    if non_claims != WINDOWS_CHOLESKY_REQUIRED_NON_CLAIMS:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} non_claims must remain the full "
            "current Cholesky claim-boundary set"
        )


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
    if row["support_tier"] != WINDOWS_CHOLESKY_PROMOTED_SUPPORT_TIER:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} support_tier must be "
            f"{WINDOWS_CHOLESKY_PROMOTED_SUPPORT_TIER} when windows is listed"
        )
    if row["claim_scope"] != WINDOWS_CHOLESKY_PROMOTED_CLAIM_SCOPE:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} claim_scope must be the exact "
            "promoted Windows selected Cholesky scope"
        )
    exact_identity = {
        "family": WINDOWS_CHOLESKY_FAMILY,
        "subfamily": WINDOWS_CHOLESKY_SUBFAMILY,
        "target_key": WINDOWS_CHOLESKY_TARGET_KEY,
        "artifact_pattern": WINDOWS_CHOLESKY_ARTIFACT_PATTERN,
        "generator_command": WINDOWS_CHOLESKY_GENERATOR_COMMAND,
    }
    for field_name, expected in exact_identity.items():
        if row[field_name] != expected:
            raise AssertionError(
                f"{WINDOWS_CHOLESKY_TARGET_ID} {field_name} must be the exact "
                "selected Cholesky promoted identity contract"
            )
    non_claims = tuple(split_manifest_values(row["non_claims"]))
    if non_claims != WINDOWS_CHOLESKY_PROMOTED_NON_CLAIMS:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} non_claims must be the exact "
            "promoted Windows selected Cholesky claim-boundary set"
        )
    exact_metadata = {
        "workflow_platforms": WINDOWS_CHOLESKY_PROMOTED_WORKFLOW_PLATFORMS,
        "workflow_file": WINDOWS_CHOLESKY_PROMOTED_WORKFLOW_FILES,
        "workflow_job": WINDOWS_CHOLESKY_PROMOTED_WORKFLOW_JOBS,
        "workflow_artifact": WINDOWS_CHOLESKY_PROMOTED_WORKFLOW_ARTIFACTS,
    }
    for field_name, expected in exact_metadata.items():
        values = tuple(split_manifest_values(row[field_name]))
        if values != expected:
            raise AssertionError(
                f"{WINDOWS_CHOLESKY_TARGET_ID} {field_name} must be the exact "
                "Linux/macOS/Windows promoted metadata tuple"
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
    expected_row_ids = tuple(split_manifest_values(row["expected_row_ids"]))
    if expected_row_ids != WINDOWS_CHOLESKY_EXPECTED_ROW_IDS:
        raise AssertionError(
            f"{WINDOWS_CHOLESKY_TARGET_ID} expected_row_ids drifted for Windows promotion"
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


def test_selected_benchmark_manifest_records_distribution_non_claims() -> None:
    row = selected_benchmark_row(manifest_rows())
    non_claims = split_manifest_values(row["non_claims"])
    for non_claim in SELECTED_BENCHMARK_REQUIRED_NON_CLAIMS:
        if non_claim not in non_claims:
            raise AssertionError(
                f"{SELECTED_BENCHMARK_TARGET_ID} missing non_claim {non_claim!r}"
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


def test_cholesky_manifest_remains_redeferred_for_windows() -> None:
    assert_current_windows_cholesky_redeferral_contract(manifest_rows())


def test_cholesky_manifest_redeferral_contract_rejects_windows_metadata() -> None:
    rows = manifest_rows()
    row = cholesky_row(rows)
    row["workflow_artifact"] = f"{row['workflow_artifact']};{WINDOWS_CHOLESKY_ARTIFACT}"
    try:
        assert_current_windows_cholesky_redeferral_contract(rows)
    except AssertionError as exc:
        if "workflow_artifact metadata must remain" not in str(exc):
            raise
        return
    raise AssertionError("expected current Cholesky Windows artifact metadata to fail")


def assert_current_qr_incompatible_redeferral_contract(
    rows: list[dict[str, str]]
) -> None:
    row = qr_incompatible_row(rows)
    exact_fields = {
        "family": WINDOWS_QR_INCOMPATIBLE_FAMILY,
        "subfamily": WINDOWS_QR_INCOMPATIBLE_SUBFAMILY,
        "target_key": WINDOWS_QR_INCOMPATIBLE_TARGET_KEY,
        "artifact_pattern": WINDOWS_QR_INCOMPATIBLE_ARTIFACT_PATTERN,
        "generator_command": WINDOWS_QR_INCOMPATIBLE_GENERATOR_COMMAND,
        "claim_scope": WINDOWS_QR_INCOMPATIBLE_CLAIM_SCOPE,
    }
    for field, expected in exact_fields.items():
        if row[field] != expected:
            raise AssertionError(
                f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} {field} must remain {expected!r}"
            )
    if row["expected_rows"] != WINDOWS_QR_INCOMPATIBLE_EXPECTED_ROWS:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} expected_rows must remain "
            f"{WINDOWS_QR_INCOMPATIBLE_EXPECTED_ROWS}"
        )
    required_files = tuple(split_manifest_values(row["required_files"]))
    if required_files != WINDOWS_QR_INCOMPATIBLE_REQUIRED_FILES:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} required_files drifted"
        )
    expected_row_ids = tuple(split_manifest_values(row["expected_row_ids"]))
    if expected_row_ids != WINDOWS_QR_INCOMPATIBLE_EXPECTED_ROW_IDS:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} expected_row_ids drifted"
        )
    workflow_files = tuple(split_manifest_values(row["workflow_file"]))
    if WINDOWS_QR_INCOMPATIBLE_WORKFLOW_FILE in workflow_files:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} must not list the Sprint 209 "
            "Windows QR workflow while re-deferred"
        )
    if workflow_files != WINDOWS_QR_INCOMPATIBLE_WORKFLOW_FILES:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} workflow_file metadata must remain "
            "the current Linux/macOS selected freshness pair while re-deferred"
        )
    workflow_jobs = tuple(split_manifest_values(row["workflow_job"]))
    if WINDOWS_QR_INCOMPATIBLE_WORKFLOW_JOB in workflow_jobs:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} must not list the Sprint 209 "
            "Windows QR workflow job while re-deferred"
        )
    if workflow_jobs != WINDOWS_QR_INCOMPATIBLE_WORKFLOW_JOBS:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} workflow_job metadata must remain "
            "the current Linux/macOS selected freshness pair while re-deferred"
        )
    workflow_artifacts = tuple(split_manifest_values(row["workflow_artifact"]))
    if WINDOWS_QR_INCOMPATIBLE_ARTIFACT in workflow_artifacts:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} must not list the Sprint 209 "
            "Windows QR artifact while re-deferred"
        )
    if workflow_artifacts != WINDOWS_QR_INCOMPATIBLE_WORKFLOW_ARTIFACTS:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} workflow_artifact metadata must "
            "remain the current Linux/macOS selected freshness pair while re-deferred"
        )
    workflow_platforms = tuple(split_manifest_values(row["workflow_platforms"]))
    if "windows" in workflow_platforms:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} must not list windows "
            "without hosted MSVC proof"
        )
    if workflow_platforms != WINDOWS_QR_INCOMPATIBLE_WORKFLOW_PLATFORMS:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} workflow_platforms metadata must "
            "remain linux/macos while re-deferred"
        )
    if WINDOWS_CHOLESKY_ARTIFACT in workflow_artifacts:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} must not reuse the Cholesky "
            "Windows artifact"
        )
    if row["support_tier"] != "local_only":
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} support_tier must remain local_only"
        )
    non_claims = split_manifest_values(row["non_claims"])
    if tuple(non_claims) != WINDOWS_QR_INCOMPATIBLE_REQUIRED_NON_CLAIMS:
        raise AssertionError(
            f"{WINDOWS_QR_INCOMPATIBLE_TARGET_ID} non_claims must remain the full "
            "current QR incompatible claim-boundary set"
        )


def test_qr_incompatible_manifest_remains_redeferred_for_windows() -> None:
    assert_current_qr_incompatible_redeferral_contract(manifest_rows())


def test_qr_incompatible_manifest_redeferral_rejects_sprint209_workflow() -> None:
    rows = manifest_rows()
    row = qr_incompatible_row(rows)
    row["workflow_file"] = f"{row['workflow_file']};{WINDOWS_QR_INCOMPATIBLE_WORKFLOW_FILE}"
    try:
        assert_current_qr_incompatible_redeferral_contract(rows)
    except AssertionError as exc:
        if "must not list the Sprint 209 Windows QR workflow" not in str(exc):
            raise
        return
    raise AssertionError("expected Sprint 209 QR workflow metadata to fail")


def test_qr_incompatible_manifest_redeferral_rejects_sprint209_job() -> None:
    rows = manifest_rows()
    row = qr_incompatible_row(rows)
    row["workflow_job"] = f"{row['workflow_job']};{WINDOWS_QR_INCOMPATIBLE_WORKFLOW_JOB}"
    try:
        assert_current_qr_incompatible_redeferral_contract(rows)
    except AssertionError as exc:
        if "must not list the Sprint 209 Windows QR workflow job" not in str(exc):
            raise
        return
    raise AssertionError("expected Sprint 209 QR workflow job metadata to fail")


def test_qr_incompatible_manifest_redeferral_rejects_sprint209_artifact() -> None:
    rows = manifest_rows()
    row = qr_incompatible_row(rows)
    row["workflow_artifact"] = f"{row['workflow_artifact']};{WINDOWS_QR_INCOMPATIBLE_ARTIFACT}"
    try:
        assert_current_qr_incompatible_redeferral_contract(rows)
    except AssertionError as exc:
        if "must not list the Sprint 209 Windows QR artifact" not in str(exc):
            raise
        return
    raise AssertionError("expected Sprint 209 QR artifact metadata to fail")


def test_qr_incompatible_manifest_redeferral_rejects_windows_platform() -> None:
    rows = manifest_rows()
    row = qr_incompatible_row(rows)
    row["workflow_platforms"] = f"{row['workflow_platforms']};windows"
    try:
        assert_current_qr_incompatible_redeferral_contract(rows)
    except AssertionError as exc:
        if "must not list windows without hosted MSVC proof" not in str(exc):
            raise
        return
    raise AssertionError("expected Sprint 209 QR Windows platform metadata to fail")


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


def test_future_windows_metadata_rejects_local_only_support_tier() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows(), promote_claims=False)
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "support_tier must be hosted_selected" not in str(exc):
            raise
        return
    raise AssertionError("expected local-only Windows Cholesky support tier to fail")


def test_future_windows_metadata_rejects_invalid_support_tier() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    cholesky_row(rows)["support_tier"] = "windows_hosted"
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "support_tier must be hosted_selected" not in str(exc):
            raise
        return
    raise AssertionError("expected invalid Windows Cholesky support tier to fail")


def test_future_windows_metadata_rejects_unpromoted_claim_scope() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    cholesky_row(rows)["claim_scope"] = WINDOWS_CHOLESKY_CURRENT_CLAIM_SCOPE
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if (
            "claim_scope must be the exact promoted Windows selected Cholesky scope"
            not in str(exc)
        ):
            raise
        return
    raise AssertionError("expected unpromoted Windows Cholesky claim scope to fail")


def test_future_windows_metadata_rejects_broad_claim_scope() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    cholesky_row(rows)["claim_scope"] = (
        "Selected Cholesky rows are fresh on Windows for all hosted report paths."
    )
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if (
            "claim_scope must be the exact promoted Windows selected Cholesky scope"
            not in str(exc)
        ):
            raise
        return
    raise AssertionError("expected broad Windows Cholesky claim scope to fail")


def test_future_windows_metadata_rejects_identity_drift() -> None:
    drift_values = {
        "family": "oracle",
        "subfamily": "cholesky_windows",
        "target_key": "cholesky-windows",
        "artifact_pattern": "build/comparison/cholesky_windows/study.tsv",
        "generator_command": (
            "python3 scripts/run_external_comparison.py --target cholesky-windows"
        ),
    }
    for field_name, drift_value in drift_values.items():
        rows = with_windows_cholesky_metadata(manifest_rows())
        cholesky_row(rows)[field_name] = drift_value
        try:
            assert_windows_cholesky_manifest_allowlist(rows)
        except AssertionError as exc:
            expected = (
                f"{WINDOWS_CHOLESKY_TARGET_ID} {field_name} must be the exact "
                "selected Cholesky promoted identity contract"
            )
            if expected not in str(exc):
                raise
            continue
        raise AssertionError(f"expected Windows Cholesky {field_name} drift to fail")


def test_future_windows_metadata_rejects_windows_non_claim() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    row = cholesky_row(rows)
    row["non_claims"] = f"{row['non_claims']};no Windows report freshness"
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "non_claims must be the exact promoted Windows selected Cholesky" not in str(exc):
            raise
        return
    raise AssertionError("expected stale Windows report freshness non-claim to fail")


def test_future_windows_metadata_rejects_missing_promoted_non_claim() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    row = cholesky_row(rows)
    row["non_claims"] = row["non_claims"].replace(";no package-manager proof", "")
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "non_claims must be the exact promoted Windows selected Cholesky" not in str(exc):
            raise
        return
    raise AssertionError("expected missing promoted Windows Cholesky non-claim to fail")


def test_future_windows_metadata_rejects_wrong_artifact() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    row = cholesky_row(rows)
    row["workflow_artifact"] = row["workflow_artifact"].replace(
        WINDOWS_CHOLESKY_ARTIFACT,
        "sprint175-macos-selected-comparison-freshness",
    )
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "workflow_artifact must be the exact Linux/macOS/Windows" not in str(exc):
            raise
        return
    raise AssertionError("expected wrong Windows Cholesky artifact to fail")


def test_future_windows_metadata_rejects_missing_linux_macos_metadata() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    row = cholesky_row(rows)
    row["workflow_platforms"] = "windows"
    row["workflow_file"] = WINDOWS_CHOLESKY_WORKFLOW_FILE
    row["workflow_job"] = WINDOWS_CHOLESKY_WORKFLOW_JOB
    row["workflow_artifact"] = WINDOWS_CHOLESKY_ARTIFACT
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "workflow_platforms must be the exact Linux/macOS/Windows" not in str(exc):
            raise
        return
    raise AssertionError("expected missing Linux/macOS promoted metadata to fail")


def test_future_windows_metadata_rejects_reordered_platform_metadata() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    row = cholesky_row(rows)
    row["workflow_platforms"] = "windows;linux;macos"
    row["workflow_file"] = (
        f"{WINDOWS_CHOLESKY_WORKFLOW_FILE};.github/workflows/ci.yml;"
        ".github/workflows/macos-ci.yml"
    )
    row["workflow_job"] = (
        f"{WINDOWS_CHOLESKY_WORKFLOW_JOB};generated-report-freshness;"
        "selected-comparison-freshness"
    )
    row["workflow_artifact"] = (
        f"{WINDOWS_CHOLESKY_ARTIFACT};sprint175-linux-selected-comparison-freshness;"
        "sprint175-macos-selected-comparison-freshness"
    )
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "workflow_platforms must be the exact Linux/macOS/Windows" not in str(exc):
            raise
        return
    raise AssertionError("expected reordered Windows Cholesky promoted metadata to fail")


def test_future_windows_metadata_rejects_row_count_drift() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    cholesky_row(rows)["expected_rows"] = "7"
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "expected_rows must remain 6" not in str(exc):
            raise
        return
    raise AssertionError("expected Windows Cholesky row-count drift to fail")


def test_future_windows_metadata_rejects_missing_artifact_file() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    row = cholesky_row(rows)
    row["required_files"] = row["required_files"].replace(";manifest.tsv", "")
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "required_files drifted" not in str(exc):
            raise
        return
    raise AssertionError("expected Windows Cholesky required-file drift to fail")


def test_future_windows_metadata_rejects_expected_row_id_drift() -> None:
    rows = with_windows_cholesky_metadata(manifest_rows())
    row = cholesky_row(rows)
    row["expected_row_ids"] = row["expected_row_ids"].replace(
        "comparison_cholesky_spd_tridiag_5_project_status_v1",
        "comparison_cholesky_spd_tridiag_5_windows_project_status_v1",
        1,
    )
    try:
        assert_windows_cholesky_manifest_allowlist(rows)
    except AssertionError as exc:
        if "expected_row_ids drifted for Windows promotion" not in str(exc):
            raise
        return
    raise AssertionError("expected Windows Cholesky expected-row-id drift to fail")


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
    test_selected_benchmark_manifest_records_distribution_non_claims()
    test_mismatched_workflow_artifact_platforms_fail_clearly()
    test_missing_report_family_mapping_fails_clearly()
    test_artifact_expected_count_collision_fails_clearly()
    test_unpromoted_report_families_remain_unselected()
    test_windows_report_freshness_deferral_keeps_manifest_unselected()
    test_cholesky_manifest_remains_redeferred_for_windows()
    test_cholesky_manifest_redeferral_contract_rejects_windows_metadata()
    test_qr_incompatible_manifest_remains_redeferred_for_windows()
    test_qr_incompatible_manifest_redeferral_rejects_sprint209_workflow()
    test_qr_incompatible_manifest_redeferral_rejects_sprint209_job()
    test_qr_incompatible_manifest_redeferral_rejects_sprint209_artifact()
    test_qr_incompatible_manifest_redeferral_rejects_windows_platform()
    test_windows_deferral_record_missing_file_fails_clearly()
    test_windows_deferral_record_missing_marker_fails_clearly()
    test_windows_platform_drift_fails_clearly()
    test_future_windows_cholesky_metadata_allowlist_accepts_exact_row()
    test_future_windows_metadata_rejects_unselected_target()
    test_future_windows_metadata_rejects_local_only_support_tier()
    test_future_windows_metadata_rejects_invalid_support_tier()
    test_future_windows_metadata_rejects_unpromoted_claim_scope()
    test_future_windows_metadata_rejects_broad_claim_scope()
    test_future_windows_metadata_rejects_identity_drift()
    test_future_windows_metadata_rejects_windows_non_claim()
    test_future_windows_metadata_rejects_missing_promoted_non_claim()
    test_future_windows_metadata_rejects_wrong_artifact()
    test_future_windows_metadata_rejects_missing_linux_macos_metadata()
    test_future_windows_metadata_rejects_reordered_platform_metadata()
    test_future_windows_metadata_rejects_row_count_drift()
    test_future_windows_metadata_rejects_missing_artifact_file()
    test_future_windows_metadata_rejects_expected_row_id_drift()
    print("test-selected-report-targets-manifest: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
