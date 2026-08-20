#!/usr/bin/env python3
"""Emit a normalized report index from maintained report metadata.

The index is a navigation and freshness aid. It preserves report-family row
meaning and non-claim boundaries; it does not generate solver, benchmark,
coverage, package, or CI proof.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "tests" / "corpus"
DEFAULT_BUILD_ROOT = REPO_ROOT / "build"
DEFAULT_OUTPUT = DEFAULT_BUILD_ROOT / "report-index" / "normalized-index.tsv"
REPORT_FAMILY_MANIFEST = Path("manifests") / "report_families.tsv"
GENERATED_PREFIXES = ("build/", "coverage/")
ORACLE_STATUS_MAP = {
    "pass": "pass",
    "fail": "fail",
    "skip": "skip",
    "defer": "defer",
    "unsupported": "unsupported",
    "xfail": "xfail",
}
GENERATED_STATUS_MAP = {
    "pass": "pass",
    "fail": "fail",
    "skip": "skip",
    "defer": "defer",
    "unsupported": "unsupported",
    "xfail": "xfail",
    "report": "advisory",
    "measurement": "advisory",
    "unknown": "unknown",
}
STRICT_FRESHNESS_POLICIES = {"generated_compare_inputs"}
ADVISORY_FRESHNESS_POLICIES = {"generated_local_advisory", "hosted_ci_external"}
ERROR_SEVERITIES = {"error"}
CANONICAL_ORACLE_TARGET = "make report-index-oracle-freshness"
CANONICAL_ORACLE_COMMAND = (
    "python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd"
)
CANONICAL_ORACLE_REMEDIATION = (
    f"{CANONICAL_ORACLE_TARGET} (fallback: {CANONICAL_ORACLE_COMMAND})"
)
SELECTED_ORACLE_ROW_COUNTS = {
    "partial_svd": 26,
    "qr": 23,
    "unknown": 3,
}
SELECTED_ORACLE_TOTAL_ROWS = sum(SELECTED_ORACLE_ROW_COUNTS.values())
SELECTED_ORACLE_FIXTURE_KEYS = {
    "partial_svd_clustered_repeated_diag8x6_k3_v1",
    "partial_svd_fail_closed_diag6_k2_v1",
    "partial_svd_lowrank_rect5x7_k3_sparse_output_v1",
    "partial_svd_rankdef_diag6x4_k2_range_projector_v1",
    "qr_minnorm_3x6_exact_values",
    "qr_minnorm_5x10_exact_values",
    "qr_rank_deficient_6x4_nullspace_v1",
    "qr_rankdef_dependent_row_4x3_v1",
    "qr_rankdef_duplicate_5x4_v1",
    "qr_underdetermined_minnorm_2x4",
}
SELECTED_COMPARISON_ROW_IDS = {
    "comparison_qr_underdetermined_minnorm_2x4_project_status_v1",
    "comparison_qr_underdetermined_minnorm_2x4_baseline_status_v1",
    "comparison_qr_underdetermined_minnorm_2x4_residual_norm_v1",
    "comparison_qr_underdetermined_minnorm_2x4_solution_norm_v1",
    "comparison_qr_underdetermined_minnorm_2x4_solution_values_v1",
    "comparison_qr_underdetermined_minnorm_2x4_project_vs_baseline_max_abs_delta_v1",
    "comparison_qr_overdetermined_compatible_5x3_project_status_v1",
    "comparison_qr_overdetermined_compatible_5x3_baseline_status_v1",
    "comparison_qr_overdetermined_compatible_5x3_residual_norm_v1",
    "comparison_qr_overdetermined_compatible_5x3_solution_norm_v1",
    "comparison_qr_overdetermined_compatible_5x3_solution_values_v1",
    "comparison_qr_overdetermined_compatible_5x3_project_vs_baseline_max_abs_delta_v1",
    "comparison_partial_svd_diag6_k2_project_status_v1",
    "comparison_partial_svd_diag6_k2_baseline_status_v1",
    "comparison_partial_svd_diag6_k2_singular_value_0_v1",
    "comparison_partial_svd_diag6_k2_singular_value_1_v1",
    "comparison_partial_svd_diag6_k2_singular_values_max_abs_delta_v1",
    "comparison_partial_svd_diag6_k2_residual_norm_v1",
    "comparison_partial_svd_diag6_k2_u_orthogonality_v1",
    "comparison_partial_svd_diag6_k2_v_orthogonality_v1",
    "comparison_partial_svd_diag6_k2_u_projector_diag_v1",
    "comparison_partial_svd_diag6_k2_v_projector_diag_v1",
}
SELECTED_COMPARISON_ARTIFACTS = (
    "build/comparison/qr_minnorm/study.tsv",
    "build/comparison/qr_compatible_ls/study.tsv",
    "build/comparison/partial_svd_diag6_k2/study.tsv",
)
SELECTED_COMPARISON_ARTIFACT_DIAGNOSTIC = "artifacts=" + ",".join(
    SELECTED_COMPARISON_ARTIFACTS
)

NORMALIZED_FIELDS = [
    "row_id",
    "report_family",
    "subfamily",
    "native_row_id",
    "row_origin",
    "row_meaning",
    "status",
    "status_reason",
    "support_tier",
    "claim_scope",
    "non_claims",
    "generator_command",
    "source_commit",
    "source_branch",
    "generated_at_utc",
    "platform",
    "compiler",
    "configuration",
    "artifact_path",
    "freshness_status",
    "freshness_reason",
    "skip_or_defer_reason",
]


class ReportIndexError(RuntimeError):
    pass


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        raise ReportIndexError(f"{path}: empty TSV")
    width = len(rows[0])
    for index, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ReportIndexError(f"{path}:{index}: expected {width} columns, got {len(row)}")
    header = rows[0]
    if len(set(header)) != len(header):
        raise ReportIndexError(f"{path}: duplicate header fields")
    return [dict(zip(header, row)) for row in rows[1:]]


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=NORMALIZED_FIELDS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_commit() -> str:
    return run_text(["git", "rev-parse", "HEAD"])


def source_branch() -> str:
    branch = run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return "detached" if branch == "HEAD" else branch


def slug(value: str) -> str:
    lowered = value.strip().lower()
    collapsed = re.sub(r"[^a-z0-9]+", "_", lowered)
    return collapsed.strip("_") or "unknown"


def is_generated_artifact_pattern(pattern: str) -> bool:
    return pattern.startswith(GENERATED_PREFIXES)


def pattern_to_paths(pattern: str, build_root: Path, repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for part in pattern.split(";"):
        candidate = part.strip()
        if not candidate:
            continue
        if candidate.startswith("build/"):
            relative_pattern = candidate.removeprefix("build/")
            if any(char in relative_pattern for char in "*?[]"):
                paths.extend(sorted(build_root.glob(relative_pattern)))
            else:
                path = build_root / relative_pattern
                if path.exists():
                    paths.append(path)
        elif candidate.startswith("coverage/"):
            if any(char in candidate for char in "*?[]"):
                paths.extend(sorted(repo_root.glob(candidate)))
            else:
                path = repo_root / candidate
                if path.exists():
                    paths.append(path)
        else:
            path = repo_root / candidate
            if any(char in candidate for char in "*?[]"):
                paths.extend(sorted(repo_root.glob(candidate)))
            elif path.exists():
                paths.append(path)
    return sorted({path.resolve() for path in paths})


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def configuration_value(configuration: str, key: str) -> str:
    prefix = f"{key}="
    for part in configuration.split(";"):
        if part.startswith(prefix):
            return part.removeprefix(prefix)
    return ""


def configuration_field(key: str, value: str) -> str:
    escaped = value.replace("%", "%25").replace(";", "%3B").replace("\n", "%0A")
    return f"{key}={escaped}"


def configuration_fields(fields: list[tuple[str, str]]) -> str:
    return ";".join(configuration_field(key, value) for key, value in fields)


def base_row(contract: dict[str, str], commit: str, branch: str) -> dict[str, str]:
    return {
        "report_family": contract["report_family"],
        "subfamily": contract["subfamily"],
        "native_row_id": (
            f"{contract['report_family']}/{contract['subfamily']}/{contract['row_meaning']}"
        ),
        "row_origin": contract["row_origin"],
        "row_meaning": contract["row_meaning"],
        "status": contract["status"],
        "status_reason": "contract_row",
        "support_tier": contract["support_tier"],
        "claim_scope": contract["claim_scope"],
        "non_claims": contract["non_claims"],
        "generator_command": contract["generator_command"],
        "source_commit": commit,
        "source_branch": branch,
        "generated_at_utc": "not_applicable",
        "platform": "not_applicable",
        "compiler": "not_applicable",
        "configuration": f"freshness_policy={contract['freshness_policy']}",
        "artifact_path": contract["artifact_pattern"],
        "freshness_status": "source_controlled",
        "freshness_reason": "source_controlled_contract",
        "skip_or_defer_reason": "",
    }


def row_with_overrides(
    contract: dict[str, str], commit: str, branch: str, overrides: dict[str, str]
) -> dict[str, str]:
    row = base_row(contract, commit, branch)
    row.update(overrides)
    return row


def contract_row(contract: dict[str, str], commit: str, branch: str) -> dict[str, str]:
    row = base_row(contract, commit, branch)
    row["row_id"] = "_".join(
        [
            "report_contract",
            slug(contract["report_family"]),
            slug(contract["subfamily"]),
            slug(contract["row_meaning"]),
            "v1",
        ]
    )
    if contract["status"] == "defer":
        row["freshness_status"] = "deferred"
        row["freshness_reason"] = "governance_deferred"
        row["skip_or_defer_reason"] = contract["claim_scope"]
    return row


def corpus_fixture_rows(
    contract: dict[str, str], corpus_root: Path, repo_root: Path, commit: str, branch: str
) -> list[dict[str, str]]:
    path = corpus_root / "manifests" / "fixtures.tsv"
    rows: list[dict[str, str]] = []
    for fixture in read_tsv(path):
        rows.append(
            row_with_overrides(
                contract,
                commit,
                branch,
                {
                    "row_id": f"corpus_fixture_{slug(fixture['fixture_key'])}_v1",
                    "native_row_id": fixture["fixture_key"],
                    "status": "advisory",
                    "status_reason": fixture["expected_behavior"],
                    "support_tier": fixture["support_tier"],
                    "claim_scope": fixture["claim_scope"],
                    "non_claims": fixture["non_claims"],
                    "generator_command": fixture["validation_command"],
                    "configuration": (
                        f"fixture_family={fixture['fixture_family']};"
                        f"storage_kind={fixture['storage_kind']};"
                        f"generator_key={fixture['generator_key'] or 'not_applicable'};"
                        f"rows={fixture['rows']};cols={fixture['cols']};nnz={fixture['nnz']};"
                        f"rank_status={fixture['rank_status']}"
                    ),
                    "artifact_path": display_path(path, repo_root),
                },
            )
        )
    return rows


def corpus_generator_rows(
    contract: dict[str, str], corpus_root: Path, repo_root: Path, commit: str, branch: str
) -> list[dict[str, str]]:
    path = corpus_root / "manifests" / "generators.tsv"
    rows: list[dict[str, str]] = []
    for generator in read_tsv(path):
        rows.append(
            row_with_overrides(
                contract,
                commit,
                branch,
                {
                    "row_id": f"corpus_generator_{slug(generator['generator_key'])}_v1",
                    "native_row_id": generator["generator_key"],
                    "status": "advisory",
                    "status_reason": "generator_metadata",
                    "generator_command": generator["regeneration_command"],
                    "configuration": (
                        f"generator_version={generator['generator_version']};"
                        f"algorithm={generator['algorithm']};seed={generator['seed']};"
                        f"canonical_format={generator['canonical_format']};"
                        f"change_policy={generator['change_policy']}"
                    ),
                    "artifact_path": display_path(path, repo_root),
                },
            )
        )
    return rows


def corpus_optional_rows(
    contract: dict[str, str], corpus_root: Path, repo_root: Path, commit: str, branch: str
) -> list[dict[str, str]]:
    path = corpus_root / "manifests" / "optional_data.tsv"
    rows: list[dict[str, str]] = []
    for optional in read_tsv(path):
        state = optional["availability_state"]
        status = "advisory" if state == "available" else "defer" if state == "deferred" else "skip"
        reason = optional["defer_reason"] if status == "defer" else optional["skip_reason"]
        rows.append(
            row_with_overrides(
                contract,
                commit,
                branch,
                {
                    "row_id": f"corpus_optional_{slug(optional['optional_data_key'])}_v1",
                    "native_row_id": optional["optional_data_key"],
                    "status": status,
                    "status_reason": f"optional_data_{state}",
                    "claim_scope": "Optional-data policy evidence only.",
                    "non_claims": optional["claim_boundary"],
                    "generator_command": optional["validation_command"],
                    "configuration": (
                        f"availability_state={state};fixture_keys={optional['fixture_keys']};"
                        f"expected_location={optional['expected_location']}"
                    ),
                    "artifact_path": display_path(path, repo_root),
                    "freshness_status": "optional_data_skip",
                    "freshness_reason": "source_controlled_optional_data_policy",
                    "skip_or_defer_reason": reason,
                },
            )
        )
    return rows


def corpus_expected_rows(
    contract: dict[str, str], corpus_root: Path, repo_root: Path, commit: str, branch: str
) -> list[dict[str, str]]:
    expected_dir = corpus_root / "expected"
    rows: list[dict[str, str]] = []
    for path in sorted(expected_dir.glob("*.tsv")):
        for expected in read_tsv(path):
            rows.append(
                row_with_overrides(
                    contract,
                    commit,
                    branch,
                    {
                        "row_id": f"corpus_expected_{slug(expected['oracle_row_id'])}_v1",
                        "native_row_id": expected["oracle_row_id"],
                        "status": "advisory",
                        "status_reason": expected["status"],
                        "claim_scope": expected["claim_scope"],
                        "non_claims": expected["non_claims"],
                        "configuration": (
                            f"fixture_key={expected['fixture_key']};"
                            f"operation={expected['operation']};"
                            f"comparison_kind={expected['comparison_kind']};"
                            f"expected_result_kind={expected['expected_result_kind']};"
                            f"tolerance_kind={expected['tolerance_kind']};"
                            f"tolerance_value={expected['tolerance_value'] or 'not_applicable'}"
                        ),
                        "artifact_path": display_path(path, repo_root),
                    },
                )
            )
    return rows


def oracle_generated_rows(
    contract: dict[str, str],
    build_root: Path,
    repo_root: Path,
    commit: str,
    branch: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in pattern_to_paths(contract["artifact_pattern"], build_root, repo_root):
        artifact = display_path(path, repo_root)
        for oracle in read_tsv(path):
            solver_family = oracle.get("solver_family", "unknown")
            is_solver_backed = solver_family != "unknown"
            if contract["row_meaning"] == "observed_oracle_comparison" and is_solver_backed:
                continue
            if contract["row_meaning"] == "solver_backed_fixture_proof" and not is_solver_backed:
                continue
            comparison_status = oracle.get("comparison_status", "unknown")
            rows.append(
                row_with_overrides(
                    contract,
                    commit,
                    branch,
                    {
                        "row_id": f"oracle_{slug(oracle['oracle_row_id'])}_{slug(artifact)}_v1",
                        "native_row_id": oracle["oracle_row_id"],
                        "status": ORACLE_STATUS_MAP.get(comparison_status, "unknown"),
                        "status_reason": oracle.get("failure_class", "") or comparison_status,
                        "support_tier": oracle.get("support_tier", contract["support_tier"]),
                        "claim_scope": oracle.get("claim_scope", contract["claim_scope"]),
                        "non_claims": oracle.get("non_claims", contract["non_claims"]),
                        "generator_command": oracle.get(
                            "command", contract["generator_command"]
                        ),
                        "source_commit": oracle.get("source_commit", commit),
                        "source_branch": oracle.get("source_branch", branch),
                        "generated_at_utc": oracle.get("generated_at_utc", "unknown"),
                        "platform": oracle.get("platform", "unknown"),
                        "compiler": oracle.get("compiler", "unknown"),
                        "configuration": (
                            f"solver_family={solver_family};fixture_key={oracle['fixture_key']};"
                            f"operation={oracle['operation']};"
                            f"comparison_kind={oracle['comparison_kind']};"
                            f"{oracle.get('configuration', '')}"
                        ),
                        "artifact_path": artifact,
                        "freshness_status": "generated_present_unchecked",
                        "freshness_reason": "oracle_row_loaded;stale_rules_deferred_to_days10_11",
                        "skip_or_defer_reason": oracle.get("skip_or_defer_reason", ""),
                    },
                )
            )
    return rows


def comparison_generated_rows(
    contract: dict[str, str],
    build_root: Path,
    repo_root: Path,
    commit: str,
    branch: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in pattern_to_paths(contract["artifact_pattern"], build_root, repo_root):
        artifact = display_path(path, repo_root)
        for comparison in read_tsv(path):
            rows.append(
                row_with_overrides(
                    contract,
                    commit,
                    branch,
                    {
                        "row_id": comparison["comparison_row_id"],
                        "native_row_id": comparison["comparison_row_id"],
                        "status": GENERATED_STATUS_MAP.get(
                            comparison.get("status", "unknown"), "unknown"
                        ),
                        "status_reason": comparison.get("status_reason", ""),
                        "support_tier": comparison.get("support_tier", contract["support_tier"]),
                        "claim_scope": comparison.get("claim_scope", contract["claim_scope"]),
                        "non_claims": comparison.get("non_claims", contract["non_claims"]),
                        "generator_command": contract["generator_command"],
                        "source_commit": comparison.get("source_commit", commit),
                        "source_branch": comparison.get("source_branch", branch),
                        "generated_at_utc": comparison.get("generated_at_utc", "unknown"),
                        "platform": comparison.get("platform", "unknown"),
                        "compiler": comparison.get("compiler", "unknown"),
                        "configuration": (
                            f"subfamily={comparison.get('subfamily', contract['subfamily'])};"
                            f"fixture_key={comparison.get('fixture_key', 'unknown')};"
                            f"operation={comparison.get('operation', 'unknown')};"
                            f"metric={comparison.get('metric', 'unknown')};"
                            f"row_kind={comparison.get('row_kind', 'unknown')};"
                            f"tolerance_kind={comparison.get('tolerance_kind', 'unknown')};"
                            f"tolerance_value={comparison.get('tolerance_value', '')};"
                            f"baseline_type={comparison.get('baseline_type', 'unknown')};"
                            f"{comparison.get('configuration', '')}"
                        ),
                        "artifact_path": artifact,
                        "freshness_status": "generated_present_unchecked",
                        "freshness_reason": "comparison_row_loaded",
                        "skip_or_defer_reason": comparison.get("caveat", ""),
                    },
                )
            )
    return rows


def benchmark_generated_rows(
    contract: dict[str, str],
    build_root: Path,
    repo_root: Path,
    commit: str,
    branch: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in pattern_to_paths(contract["artifact_pattern"], build_root, repo_root):
        artifact = display_path(path, repo_root)
        manifest = read_manifest(path.parent / "manifest.txt")
        for benchmark in read_tsv(path):
            native_id = benchmark.get("artifact", artifact)
            rows.append(
                row_with_overrides(
                    contract,
                    commit,
                    branch,
                    {
                        "row_id": f"benchmark_{slug(native_id)}_{slug(artifact)}_v1",
                        "native_row_id": native_id,
                        "status": "advisory",
                        "status_reason": benchmark.get("category", "measurement"),
                        "claim_scope": contract["claim_scope"],
                        "non_claims": contract["non_claims"],
                        "generator_command": contract["generator_command"],
                        "source_commit": benchmark.get("git_commit", manifest.get("git_commit", commit)),
                        "source_branch": benchmark.get("git_branch", manifest.get("git_branch", branch)),
                        "generated_at_utc": benchmark.get(
                            "generated_at_utc", manifest.get("generated_at_utc", "unknown")
                        ),
                        "platform": benchmark.get("platform", manifest.get("platform", "unknown")),
                        "compiler": benchmark.get("compiler", manifest.get("compiler", "unknown")),
                        "configuration": configuration_fields(
                            [
                                ("surface", benchmark.get("surface", "unknown")),
                                ("category", benchmark.get("category", "unknown")),
                                ("report_label", benchmark.get("report_label", "unknown")),
                                ("build_mode", benchmark.get("build_mode", "unknown")),
                                ("omp_num_threads", benchmark.get("omp_num_threads", "unknown")),
                                ("command", benchmark.get("command", "unknown")),
                                ("relative_path", benchmark.get("relative_path", "unknown")),
                                ("row_report_family", benchmark.get("report_family", "unknown")),
                                ("row_status", benchmark.get("status", "unknown")),
                                ("row_support_tier", benchmark.get("support_tier", "unknown")),
                                ("claim_boundary", benchmark.get("claim_boundary", "unknown")),
                                ("fixture_or_workload", benchmark.get("fixture_or_workload", "unknown")),
                                ("matrix_size", benchmark.get("matrix_size", "unknown")),
                                ("repeat_semantics", benchmark.get("repeat_semantics", "unknown")),
                                ("warmup", benchmark.get("warmup", "unknown")),
                                ("variance", benchmark.get("variance", "unknown")),
                                ("baseline", benchmark.get("baseline", "unknown")),
                                ("threshold", benchmark.get("threshold", "unknown")),
                                ("backend_context", benchmark.get("backend_context", "unknown")),
                                ("methodology_notes", benchmark.get("methodology_notes", "unknown")),
                            ]
                        ),
                        "artifact_path": artifact,
                        "freshness_status": "generated_present_unchecked",
                        "freshness_reason": "benchmark_row_loaded;stale_rules_deferred_to_days10_11",
                    },
                )
            )
    return rows


def sentinel_generated_rows(
    contract: dict[str, str],
    build_root: Path,
    repo_root: Path,
    commit: str,
    branch: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    hard_gate_boundaries = {"local_wall_gate", "local_selected_regression_gate"}
    for path in pattern_to_paths(contract["artifact_pattern"], build_root, repo_root):
        artifact = display_path(path, repo_root)
        manifest = read_manifest(path.parent / "manifest.txt")
        for sentinel in read_tsv(path):
            claim_boundary = sentinel.get("claim_boundary", "")
            if (
                contract["row_meaning"] == "sentinel_hard_gate"
                and claim_boundary not in hard_gate_boundaries
            ):
                continue
            if (
                contract["row_meaning"] == "sentinel_advisory_measurement"
                and claim_boundary in hard_gate_boundaries
            ):
                continue
            native_id = "_".join(
                [
                    sentinel.get("sentinel_id", "unknown"),
                    sentinel.get("matrix_or_fixture", "unknown"),
                    sentinel.get("metric", "unknown"),
                ]
            )
            status = GENERATED_STATUS_MAP.get(sentinel.get("status", "unknown"), "unknown")
            rows.append(
                row_with_overrides(
                    contract,
                    commit,
                    branch,
                    {
                        "row_id": f"sentinel_{slug(native_id)}_{slug(artifact)}_v1",
                        "native_row_id": native_id,
                        "status": status,
                        "status_reason": sentinel.get("notes", sentinel.get("status", "")),
                        "support_tier": contract["support_tier"],
                        "claim_scope": contract["claim_scope"],
                        "non_claims": contract["non_claims"],
                        "generator_command": sentinel.get("command", contract["generator_command"]),
                        "source_commit": manifest.get("git_commit", commit),
                        "source_branch": manifest.get("git_branch", branch),
                        "generated_at_utc": manifest.get("generated_at_utc", "unknown"),
                        "platform": manifest.get("platform", "unknown"),
                        "compiler": manifest.get("compiler", "unknown"),
                        "configuration": configuration_fields(
                            [
                                ("sentinel_id", sentinel.get("sentinel_id", "unknown")),
                                ("claim_boundary", claim_boundary or "unknown"),
                                ("row_support_tier", sentinel.get("support_tier", "unknown")),
                                ("build_mode", sentinel.get("build_mode", "unknown")),
                                ("omp_num_threads", sentinel.get("omp_num_threads", "unknown")),
                                ("metric", sentinel.get("metric", "unknown")),
                                ("value", sentinel.get("value", "unknown")),
                                ("baseline", sentinel.get("baseline", "unknown")),
                                ("threshold", sentinel.get("threshold", "unknown")),
                                (
                                    "baseline_provenance",
                                    sentinel.get("baseline_provenance", "unknown"),
                                ),
                                ("repeat_semantics", sentinel.get("repeat_semantics", "unknown")),
                                ("warmup", sentinel.get("warmup", "unknown")),
                                ("variance", sentinel.get("variance", "unknown")),
                                ("backend_request", sentinel.get("backend_request", "unknown")),
                                ("backend_selected", sentinel.get("backend_selected", "unknown")),
                                ("backend_fallback", sentinel.get("backend_fallback", "unknown")),
                                ("methodology_notes", sentinel.get("methodology_notes", "unknown")),
                            ]
                        ),
                        "artifact_path": artifact,
                        "freshness_status": "generated_present_unchecked",
                        "freshness_reason": "sentinel_row_loaded;stale_rules_deferred_to_days10_11",
                        "skip_or_defer_reason": (
                            sentinel.get("notes", "") if status in {"skip", "defer"} else ""
                        ),
                    },
                )
            )
    return rows


def guardrail_generated_rows(
    contract: dict[str, str],
    build_root: Path,
    repo_root: Path,
    commit: str,
    branch: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in pattern_to_paths(contract["artifact_pattern"], build_root, repo_root):
        artifact = display_path(path, repo_root)
        manifest = read_manifest(path.parent / "manifest.txt")
        for guardrail in read_tsv(path):
            native_id = guardrail.get("lane_id", "unknown")
            status = GENERATED_STATUS_MAP.get(guardrail.get("status", "unknown"), "unknown")
            rows.append(
                row_with_overrides(
                    contract,
                    commit,
                    branch,
                    {
                        "row_id": f"guardrail_{slug(native_id)}_{slug(artifact)}_v1",
                        "native_row_id": native_id,
                        "status": status,
                        "status_reason": guardrail.get("notes", guardrail.get("status", "")),
                        "claim_scope": contract["claim_scope"],
                        "non_claims": contract["non_claims"],
                        "generator_command": guardrail.get("command", contract["generator_command"]),
                        "source_commit": manifest.get("git_commit", commit),
                        "source_branch": manifest.get("git_branch", branch),
                        "generated_at_utc": manifest.get("generated_at_utc", "unknown"),
                        "platform": manifest.get("platform", "unknown"),
                        "compiler": manifest.get("compiler", "unknown"),
                        "configuration": (
                            f"lane_id={native_id};category={guardrail.get('category', 'unknown')};"
                            f"artifact={guardrail.get('artifact', 'unknown')};"
                            f"supplemental={manifest.get('supplemental', 'unknown')}"
                        ),
                        "artifact_path": artifact,
                        "freshness_status": "generated_present_unchecked",
                        "freshness_reason": "guardrail_row_loaded;stale_rules_deferred_to_days10_11",
                        "skip_or_defer_reason": (
                            guardrail.get("notes", "") if status in {"skip", "defer"} else ""
                        ),
                    },
                )
            )
    return rows


def coverage_generated_rows(
    contract: dict[str, str],
    repo_root: Path,
    commit: str,
    branch: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in pattern_to_paths(contract["artifact_pattern"], DEFAULT_BUILD_ROOT, repo_root):
        artifact = display_path(path, repo_root)
        rows.append(
            row_with_overrides(
                contract,
                commit,
                branch,
                {
                    "row_id": f"coverage_{slug(artifact)}_v1",
                    "native_row_id": artifact,
                    "status": "advisory",
                    "status_reason": "coverage_artifact_present",
                    "configuration": (
                        "backend=unknown_from_artifact_presence;"
                        "threshold=makefile_cov_threshold;"
                        "source_scope=src"
                    ),
                    "artifact_path": artifact,
                    "freshness_status": "generated_present_unchecked",
                    "freshness_reason": "coverage_artifact_present;stale_rules_deferred_to_days10_11",
                },
            )
        )
    return rows


def deadcode_generated_rows(
    contract: dict[str, str],
    build_root: Path,
    repo_root: Path,
    commit: str,
    branch: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in pattern_to_paths(contract["artifact_pattern"], build_root, repo_root):
        artifact = display_path(path, repo_root)
        for deadcode in read_tsv(path):
            native_id = "_".join(
                [
                    deadcode.get("bucket", "unknown"),
                    deadcode.get("tool", "unknown"),
                    deadcode.get("symbol", "") or deadcode.get("path", "unknown"),
                    deadcode.get("line", "") or "summary",
                ]
            )
            rows.append(
                row_with_overrides(
                    contract,
                    commit,
                    branch,
                    {
                        "row_id": f"deadcode_{slug(native_id)}_{slug(artifact)}_v1",
                        "native_row_id": native_id,
                        "status": "advisory",
                        "status_reason": deadcode.get("bucket", "deadcode_classification"),
                        "claim_scope": contract["claim_scope"],
                        "non_claims": contract["non_claims"],
                        "configuration": (
                            f"bucket={deadcode.get('bucket', 'unknown')};"
                            f"tool={deadcode.get('tool', 'unknown')};"
                            f"symbol={deadcode.get('symbol', '') or 'not_applicable'};"
                            f"path={deadcode.get('path', 'unknown')};"
                            f"line={deadcode.get('line', '') or 'not_applicable'};"
                            f"disposition={deadcode.get('disposition', 'unknown')}"
                        ),
                        "artifact_path": artifact,
                        "freshness_status": "generated_present_unchecked",
                        "freshness_reason": "deadcode_row_loaded;stale_rules_deferred_to_days10_11",
                    },
                )
            )
    return rows


def package_proof_owner_rows(
    contract: dict[str, str], repo_root: Path, commit: str, branch: str
) -> list[dict[str, str]]:
    proof_owners = [
        {
            "proof_name": "make_install_pkg_config",
            "path": "tests/test_install.sh",
            "command": "bash tests/test_install.sh",
            "scope": "Unix-side Make install/uninstall plus pkg-config downstream proof.",
        },
        {
            "proof_name": "cmake_install_export",
            "path": "tests/test_cmake_install.sh",
            "command": "bash tests/test_cmake_install.sh",
            "scope": "Unix-side CMake install/export plus find_package downstream proof.",
        },
        {
            "proof_name": "pkg_config_template",
            "path": "sparse.pc.in",
            "command": "template consumed by Make and CMake install paths",
            "scope": "pkg-config metadata template for the maintained static archive link surface.",
        },
        {
            "proof_name": "cmake_package_config",
            "path": "cmake/SparseConfig.cmake.in",
            "command": "cmake configure_file and install export",
            "scope": "CMake package config template for the maintained static imported target.",
        },
        {
            "proof_name": "static_package_deferral",
            "path": "scripts/static_package_deferral_check.sh",
            "command": "bash scripts/static_package_deferral_check.sh",
            "scope": "Static-first package decision guardrail for deferred shared-library or ABI claims.",
        },
        {
            "proof_name": "package_manager_deferral",
            "path": "scripts/package_manager_deferral_check.sh",
            "command": "bash scripts/package_manager_deferral_check.sh",
            "scope": "Package-manager deferral guardrail for provider support non-claims.",
        },
    ]
    rows: list[dict[str, str]] = []
    for owner in proof_owners:
        path = repo_root / owner["path"]
        exists = path.exists()
        rows.append(
            row_with_overrides(
                contract,
                commit,
                branch,
                {
                    "row_id": f"package_{slug(owner['proof_name'])}_v1",
                    "native_row_id": owner["proof_name"],
                    "status": "advisory" if exists else "fail",
                    "status_reason": "source_controlled_proof_owner" if exists else "proof_owner_missing",
                    "claim_scope": owner["scope"],
                    "non_claims": contract["non_claims"],
                    "generator_command": owner["command"],
                    "configuration": "package_surface=static_first;artifact_kind=source_controlled",
                    "artifact_path": owner["path"],
                    "freshness_status": "source_controlled" if exists else "not_generated",
                    "freshness_reason": (
                        "source_controlled_package_proof_owner"
                        if exists
                        else "source_controlled_package_proof_owner_missing"
                    ),
                    "skip_or_defer_reason": "" if exists else "required proof-owner path missing",
                },
            )
        )
    return rows


def generated_artifact_row(
    contract: dict[str, str], path: Path, commit: str, branch: str, repo_root: Path
) -> dict[str, str]:
    row = base_row(contract, commit, branch)
    artifact = display_path(path, repo_root)
    row["row_id"] = "_".join(
        [
            "report_artifact",
            slug(contract["report_family"]),
            slug(contract["subfamily"]),
            slug(contract["row_meaning"]),
            slug(artifact),
            "v1",
        ]
    )
    row["native_row_id"] = artifact
    row["status"] = "unknown"
    row["status_reason"] = "generated_artifact_present"
    row["artifact_path"] = artifact
    row["freshness_status"] = "generated_present_unchecked"
    row["freshness_reason"] = "day6_presence_only;stale_rules_deferred_to_days10_11"
    row["generated_at_utc"] = "unknown"
    row["platform"] = "unknown"
    row["compiler"] = "unknown"
    return row


def not_generated_row(contract: dict[str, str], commit: str, branch: str) -> dict[str, str]:
    row = base_row(contract, commit, branch)
    row["row_id"] = "_".join(
        [
            "report_missing",
            slug(contract["report_family"]),
            slug(contract["subfamily"]),
            slug(contract["row_meaning"]),
            "v1",
        ]
    )
    row["status"] = "unknown"
    row["status_reason"] = "generated_report_missing"
    row["freshness_status"] = "not_generated"
    row["freshness_reason"] = "local_generated_artifact_not_found"
    row["skip_or_defer_reason"] = "missing generated rows are not pass evidence"
    if contract["report_family"] == "oracle":
        row["freshness_reason"] = (
            "local_generated_artifact_not_found;"
            f" expected_artifact={contract['artifact_pattern']};"
            f" remediation={CANONICAL_ORACLE_REMEDIATION}"
        )
        row["skip_or_defer_reason"] = (
            "missing generated oracle rows are not pass evidence;"
            f" run {CANONICAL_ORACLE_REMEDIATION}"
        )
    if contract["report_family"] == "comparison":
        row["freshness_reason"] = (
            "local_generated_artifact_not_found;"
            f" expected_artifact={contract['artifact_pattern']};"
            " remediation=make report-index-comparison-freshness"
        )
        row["skip_or_defer_reason"] = (
            "missing generated comparison rows are not pass evidence;"
            " run make report-index-comparison-freshness"
        )
    return row


def selected_contracts(
    contracts: Iterable[dict[str, str]], families: set[str]
) -> list[dict[str, str]]:
    selected = []
    for contract in contracts:
        if families and contract["report_family"] not in families:
            continue
        selected.append(contract)
    return selected


def emit_rows(
    contracts: list[dict[str, str]],
    *,
    corpus_root: Path,
    build_root: Path,
    repo_root: Path,
    include_generated: bool,
    commit: str,
    branch: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for contract in contracts:
        rows.append(contract_row(contract, commit, branch))
        if contract["report_family"] == "corpus" and contract["subfamily"] == "fixtures":
            rows.extend(corpus_fixture_rows(contract, corpus_root, repo_root, commit, branch))
            continue
        if contract["report_family"] == "corpus" and contract["subfamily"] == "generators":
            rows.extend(corpus_generator_rows(contract, corpus_root, repo_root, commit, branch))
            continue
        if contract["report_family"] == "corpus" and contract["subfamily"] == "optional_data":
            rows.extend(corpus_optional_rows(contract, corpus_root, repo_root, commit, branch))
            continue
        if contract["report_family"] == "corpus" and contract["subfamily"] == "expected":
            rows.extend(corpus_expected_rows(contract, corpus_root, repo_root, commit, branch))
            continue
        if contract["report_family"] == "oracle":
            if not include_generated:
                rows.append(not_generated_row(contract, commit, branch))
            else:
                oracle_rows = oracle_generated_rows(contract, build_root, repo_root, commit, branch)
                if oracle_rows:
                    rows.extend(oracle_rows)
                else:
                    rows.append(not_generated_row(contract, commit, branch))
            continue
        if contract["report_family"] == "comparison":
            if not include_generated:
                rows.append(not_generated_row(contract, commit, branch))
            else:
                comparison_rows = comparison_generated_rows(
                    contract, build_root, repo_root, commit, branch
                )
                if comparison_rows:
                    rows.extend(comparison_rows)
                else:
                    rows.append(not_generated_row(contract, commit, branch))
            continue
        if contract["report_family"] == "benchmark":
            generated_rows = (
                []
                if not include_generated
                else benchmark_generated_rows(contract, build_root, repo_root, commit, branch)
            )
            rows.extend(generated_rows or [not_generated_row(contract, commit, branch)])
            continue
        if contract["report_family"] == "sentinel":
            generated_rows = (
                []
                if not include_generated
                else sentinel_generated_rows(contract, build_root, repo_root, commit, branch)
            )
            rows.extend(generated_rows or [not_generated_row(contract, commit, branch)])
            continue
        if contract["report_family"] == "guardrail":
            generated_rows = (
                []
                if not include_generated
                else guardrail_generated_rows(contract, build_root, repo_root, commit, branch)
            )
            rows.extend(generated_rows or [not_generated_row(contract, commit, branch)])
            continue
        if contract["report_family"] == "coverage":
            generated_rows = (
                []
                if not include_generated
                else coverage_generated_rows(contract, repo_root, commit, branch)
            )
            rows.extend(generated_rows or [not_generated_row(contract, commit, branch)])
            continue
        if contract["report_family"] == "deadcode":
            generated_rows = (
                []
                if not include_generated
                else deadcode_generated_rows(contract, build_root, repo_root, commit, branch)
            )
            rows.extend(generated_rows or [not_generated_row(contract, commit, branch)])
            continue
        if contract["report_family"] == "package":
            rows.extend(package_proof_owner_rows(contract, repo_root, commit, branch))
            continue
        if not is_generated_artifact_pattern(contract["artifact_pattern"]):
            continue
        if not include_generated:
            rows.append(not_generated_row(contract, commit, branch))
            continue
        paths = pattern_to_paths(contract["artifact_pattern"], build_root, repo_root)
        if paths:
            rows.extend(
                generated_artifact_row(contract, path, commit, branch, repo_root)
                for path in paths
            )
        else:
            rows.append(not_generated_row(contract, commit, branch))
    return sort_rows(rows)


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            row["report_family"],
            row["subfamily"],
            row["row_origin"],
            row["row_meaning"],
            row["native_row_id"],
            row["artifact_path"],
            row["row_id"],
        ),
    )


def validate_unique_row_ids(rows: Iterable[dict[str, str]]) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = row["row_id"]
        if row_id in seen:
            raise ReportIndexError(f"duplicate normalized row_id {row_id!r}")
        seen.add(row_id)


def freshness_policy(row: dict[str, str]) -> str:
    configured = configuration_value(row.get("configuration", ""), "freshness_policy")
    if configured:
        return configured
    if row["report_family"] == "oracle":
        return "generated_compare_inputs"
    if row["report_family"] == "sentinel" and row["row_meaning"] == "sentinel_hard_gate":
        return "generated_compare_inputs"
    if row["report_family"] == "guardrail":
        return "generated_compare_inputs"
    if row["report_family"] in {"benchmark", "coverage", "deadcode", "report_index"}:
        return "generated_local_advisory"
    if row["report_family"] == "package":
        return "source_controlled"
    if row["freshness_status"] == "optional_data_skip":
        return "optional_data_skip"
    if row["freshness_status"] == "deferred":
        return "deferred_governance"
    return ""


def is_required_family(row: dict[str, str], required_families: set[str]) -> bool:
    return row["report_family"] in required_families


def oracle_artifact_detail(build_root: Path) -> str:
    resolved = display_path(build_root / "corpus" / "oracle", REPO_ROOT)
    return f"artifact=build/corpus/oracle/*.tsv; resolved_artifact={resolved}/*.tsv"


def oracle_manifest_detail(build_root: Path) -> str:
    resolved = display_path(build_root / "corpus-reports" / "manifest.txt", REPO_ROOT)
    return f"manifest=build/corpus-reports/manifest.txt; resolved_manifest={resolved}"


def stale_source_commit_reason(row: dict[str, str], current_commit: str) -> str:
    if row["report_family"] == "comparison":
        return (
            "source_commit does not match current HEAD; "
            f"recorded={row.get('source_commit', 'unknown')}; current={current_commit}; "
            f"artifact={row.get('artifact_path', 'unknown')}; "
            "run make report-index-comparison-freshness"
        )
    if row["report_family"] != "oracle":
        return "source_commit does not match current HEAD"
    return (
        "source_commit does not match current HEAD; "
        f"recorded={row.get('source_commit', 'unknown')}; current={current_commit}; "
        f"artifact={row.get('artifact_path', 'unknown')}; "
        f"run {CANONICAL_ORACLE_REMEDIATION}"
    )


def selected_oracle_policy_enabled(
    required_families: set[str], strict_generated: bool
) -> bool:
    return "oracle" in required_families or strict_generated


def selected_oracle_generated_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["report_family"] == "oracle"
        and row["row_origin"] == "generated_local"
        and row["row_id"].startswith("oracle_")
    ]


def selected_oracle_policy_diagnostics(
    rows: list[dict[str, str]],
    *,
    build_root: Path,
    required_families: set[str],
    strict_generated: bool,
) -> tuple[list[str], bool]:
    if not selected_oracle_policy_enabled(required_families, strict_generated):
        return [], False

    oracle_rows = selected_oracle_generated_rows(rows)
    if not oracle_rows:
        return [], False

    counts = {solver_family: 0 for solver_family in SELECTED_ORACLE_ROW_COUNTS}
    observed_solver_families: set[str] = set()
    observed_fixture_keys: set[str] = set()
    for row in oracle_rows:
        solver_family = configuration_value(row["configuration"], "solver_family") or "unknown"
        fixture_key = configuration_value(row["configuration"], "fixture_key")
        observed_solver_families.add(solver_family)
        if fixture_key:
            observed_fixture_keys.add(fixture_key)
        counts[solver_family] = counts.get(solver_family, 0) + 1

    diagnostics: list[str] = []
    has_error = False
    if len(oracle_rows) != SELECTED_ORACLE_TOTAL_ROWS or counts != SELECTED_ORACLE_ROW_COUNTS:
        has_error = True
        diagnostics.append(
            "freshness: error: oracle_selected_row_count: row_count_mismatch: "
            f"expected total={SELECTED_ORACLE_TOTAL_ROWS} counts={SELECTED_ORACLE_ROW_COUNTS}; "
            f"observed total={len(oracle_rows)} counts={counts}; "
            f"{oracle_artifact_detail(build_root)}; "
            f"run {CANONICAL_ORACLE_REMEDIATION}"
        )

    expected_solver_families = set(SELECTED_ORACLE_ROW_COUNTS)
    missing_solver_families = sorted(expected_solver_families - observed_solver_families)
    if missing_solver_families:
        has_error = True
        diagnostics.append(
            "freshness: error: oracle_selected_solver_families: missing_solver_family: "
            f"missing={','.join(missing_solver_families)}; "
            f"observed={','.join(sorted(observed_solver_families)) or 'none'}; "
            f"{oracle_artifact_detail(build_root)}; "
            f"run {CANONICAL_ORACLE_REMEDIATION}"
        )

    missing_fixture_keys = sorted(SELECTED_ORACLE_FIXTURE_KEYS - observed_fixture_keys)
    if missing_fixture_keys:
        has_error = True
        diagnostics.append(
            "freshness: error: oracle_selected_fixture_keys: missing_fixture_key: "
            f"missing={','.join(missing_fixture_keys)}; "
            f"{oracle_manifest_detail(build_root)}; "
            f"run {CANONICAL_ORACLE_REMEDIATION}"
        )

    return diagnostics, has_error


def selected_comparison_policy_enabled(
    required_families: set[str], strict_generated: bool
) -> bool:
    return "comparison" in required_families or strict_generated


def selected_comparison_generated_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["report_family"] == "comparison"
        and row["row_origin"] == "generated_local"
        and row["row_id"].startswith("comparison_")
    ]


def selected_comparison_policy_diagnostics(
    rows: list[dict[str, str]],
    *,
    required_families: set[str],
    strict_generated: bool,
) -> tuple[list[str], bool]:
    if not selected_comparison_policy_enabled(required_families, strict_generated):
        return [], False

    comparison_rows = selected_comparison_generated_rows(rows)
    if not comparison_rows:
        return [], False

    diagnostics: list[str] = []
    has_error = False
    row_ids = [row["row_id"] for row in comparison_rows]
    observed_ids = set(row_ids)
    missing = sorted(SELECTED_COMPARISON_ROW_IDS - observed_ids)
    duplicates = sorted(row_id for row_id in observed_ids if row_ids.count(row_id) > 1)
    unexpected = sorted(observed_ids - SELECTED_COMPARISON_ROW_IDS)
    if missing or duplicates or unexpected or len(comparison_rows) != len(SELECTED_COMPARISON_ROW_IDS):
        has_error = True
        diagnostics.append(
            "freshness: error: comparison_selected_rows: row_set_mismatch: "
            f"expected={len(SELECTED_COMPARISON_ROW_IDS)}; observed={len(comparison_rows)}; "
            f"missing={','.join(missing) or 'none'}; "
            f"duplicates={','.join(duplicates) or 'none'}; "
            f"unexpected={','.join(unexpected) or 'none'}; "
            f"{SELECTED_COMPARISON_ARTIFACT_DIAGNOSTIC}; "
            "run make report-index-comparison-freshness"
        )

    non_pass = [
        f"{row['row_id']}={row['status']}/{row['status_reason']}"
        for row in comparison_rows
        if row["row_id"] in SELECTED_COMPARISON_ROW_IDS and row["status"] != "pass"
    ]
    if non_pass:
        has_error = True
        diagnostics.append(
            "freshness: error: comparison_selected_status: non_pass_selected_row: "
            f"{';'.join(non_pass)}; {SELECTED_COMPARISON_ARTIFACT_DIAGNOSTIC}; "
            "run make report-index-comparison-freshness"
        )

    deferred = [
        row["row_id"]
        for row in comparison_rows
        if row["status"] in {"skip", "defer"}
    ]
    if deferred:
        diagnostics.append(
            "freshness: defer: comparison_selected_rows: skip_or_defer_not_proof: "
            f"{','.join(deferred)}"
        )
    return diagnostics, has_error


def is_selected_required_generated_row(
    row: dict[str, str],
    *,
    required_families: set[str],
    strict_generated: bool,
) -> bool:
    if row["freshness_status"] != "generated_present_unchecked":
        return False
    if row["row_origin"] != "generated_local":
        return False
    if row["report_family"] == "oracle":
        return selected_oracle_policy_enabled(required_families, strict_generated) and row[
            "row_id"
        ].startswith("oracle_")
    if row["report_family"] == "comparison":
        return selected_comparison_policy_enabled(
            required_families, strict_generated
        ) and row["row_id"].startswith("comparison_")
    return False


def freshness_severity(
    row: dict[str, str],
    *,
    state: str,
    current_commit: str,
    required_families: set[str],
    strict_generated: bool,
    advisory_ok: bool,
) -> tuple[str, str]:
    policy = freshness_policy(row)
    required = is_required_family(row, required_families)
    advisory_policy = policy in ADVISORY_FRESHNESS_POLICIES

    if row["status"] == "fail":
        if row["row_meaning"] in {"sentinel_hard_gate", "guardrail_lane"}:
            return ("error", "generated hard-gate or guardrail row reports fail")
        if row["report_family"] == "oracle" and (
            required or selected_oracle_policy_enabled(required_families, strict_generated)
        ):
            return (
                "error",
                "generated oracle row reports fail; "
                f"fixture_key={configuration_value(row['configuration'], 'fixture_key') or 'unknown'}; "
                f"artifact={row.get('artifact_path', 'unknown')}; "
                f"run {CANONICAL_ORACLE_REMEDIATION}",
            )
        if row["report_family"] == "comparison" and (
            required or selected_comparison_policy_enabled(required_families, strict_generated)
        ):
            return (
                "error",
                "generated comparison row reports fail; "
                f"artifact={row.get('artifact_path', 'unknown')}; "
                "run make report-index-comparison-freshness",
            )
        if row["report_family"] == "package":
            return ("error", "source-controlled package proof owner is missing")

    if state == "source_controlled":
        return ("advisory", "source-controlled row is governed by schema and Git review")
    if state == "optional_data_skip":
        return ("skip", row["skip_or_defer_reason"] or "optional data unavailable")
    if state == "deferred":
        return ("defer", row["skip_or_defer_reason"] or "governance deferred")
    if state == "unsupported":
        severity = "error" if required else "unsupported"
        return (severity, row["skip_or_defer_reason"] or "row is unsupported in this context")
    if state == "not_generated":
        if required:
            if row["report_family"] == "oracle":
                return (
                    "error",
                    "required generated family missing: oracle; "
                    "artifact=build/corpus/oracle/*.tsv; "
                    f"run {CANONICAL_ORACLE_REMEDIATION}",
                )
            if row["report_family"] == "comparison":
                return (
                    "error",
                    "required generated family missing: comparison; "
                    f"{SELECTED_COMPARISON_ARTIFACT_DIAGNOSTIC}; "
                    "run make report-index-comparison-freshness",
                )
            return ("error", f"required generated family missing: {row['report_family']}")
        if policy in STRICT_FRESHNESS_POLICIES:
            return ("warning", "local generated report is absent")
        return ("advisory", "local generated advisory report is absent")
    if state == "stale":
        if required or (strict_generated and not (advisory_ok and advisory_policy)):
            return ("error", stale_source_commit_reason(row, current_commit))
        if policy in STRICT_FRESHNESS_POLICIES:
            return ("warning", stale_source_commit_reason(row, current_commit))
        return ("advisory", "local measurement freshness is advisory")
    if state == "fresh":
        return ("advisory", "generated row source_commit matches current HEAD")
    if state == "generated_present_unchecked":
        if row.get("source_commit") not in {"", "unknown", "not_applicable", current_commit}:
            if required or (strict_generated and not (advisory_ok and advisory_policy)):
                return ("error", stale_source_commit_reason(row, current_commit))
            if policy in STRICT_FRESHNESS_POLICIES:
                return ("warning", stale_source_commit_reason(row, current_commit))
            return ("advisory", "source_commit differs, but local measurement freshness is advisory")
        if policy in ADVISORY_FRESHNESS_POLICIES:
            return ("advisory", "local generated row freshness is advisory")
        return ("warning", "generated row exists but strict freshness comparison is pending")
    return ("advisory", row.get("freshness_reason", "") or "freshness state is advisory")


def evaluate_freshness_state(row: dict[str, str], current_commit: str) -> str:
    state = row.get("freshness_status", "unknown")
    if state == "generated_present_unchecked":
        source = row.get("source_commit", "")
        if source not in {"", "unknown", "not_applicable", current_commit}:
            return "stale"
        if freshness_policy(row) in STRICT_FRESHNESS_POLICIES:
            return "generated_present_unchecked"
        return "fresh"
    return state


def freshness_diagnostics(
    rows: list[dict[str, str]],
    *,
    build_root: Path,
    current_commit: str,
    required_families: set[str],
    strict_generated: bool,
    advisory_ok: bool,
) -> tuple[list[str], bool]:
    diagnostics: list[str] = []
    has_error = False
    for row in rows:
        state = evaluate_freshness_state(row, current_commit)
        if state == "generated_present_unchecked" and is_selected_required_generated_row(
            row,
            required_families=required_families,
            strict_generated=strict_generated,
        ):
            state = "fresh"
        severity, reason = freshness_severity(
            row,
            state=state,
            current_commit=current_commit,
            required_families=required_families,
            strict_generated=strict_generated,
            advisory_ok=advisory_ok,
        )
        if severity in ERROR_SEVERITIES:
            has_error = True
        diagnostics.append(
            f"freshness: {severity}: {row['row_id']}: {state}: {reason}"
        )
    oracle_diagnostics, oracle_has_error = selected_oracle_policy_diagnostics(
        rows,
        build_root=build_root,
        required_families=required_families,
        strict_generated=strict_generated,
    )
    diagnostics.extend(oracle_diagnostics)
    has_error = has_error or oracle_has_error
    comparison_diagnostics, comparison_has_error = selected_comparison_policy_diagnostics(
        rows,
        required_families=required_families,
        strict_generated=strict_generated,
    )
    diagnostics.extend(comparison_diagnostics)
    has_error = has_error or comparison_has_error
    return diagnostics, has_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--include-generated", dest="include_generated", action="store_true")
    parser.add_argument("--no-generated", dest="include_generated", action="store_false")
    parser.add_argument("--require-generated", action="append", default=[])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-freshness", action="store_true")
    parser.add_argument("--strict-generated", action="store_true")
    parser.add_argument("--advisory-ok", action="store_true")
    parser.add_argument("--format", choices=["tsv"], default="tsv")
    parser.set_defaults(include_generated=True)
    args = parser.parse_args()

    manifest_path = args.corpus_root / REPORT_FAMILY_MANIFEST
    contracts = selected_contracts(read_tsv(manifest_path), set(args.family))
    if not contracts:
        raise ReportIndexError("no report-family contracts selected")

    commit = source_commit()
    branch = source_branch()
    rows = emit_rows(
        contracts,
        corpus_root=args.corpus_root,
        build_root=args.build_root,
        repo_root=REPO_ROOT,
        include_generated=args.include_generated,
        commit=commit,
        branch=branch,
    )
    validate_unique_row_ids(rows)

    required_families = set(args.require_generated)
    if args.check_freshness:
        diagnostics, has_error = freshness_diagnostics(
            rows,
            build_root=args.build_root,
            current_commit=commit,
            required_families=required_families,
            strict_generated=args.strict_generated,
            advisory_ok=args.advisory_ok,
        )
        for diagnostic in diagnostics:
            print(diagnostic)
        if has_error:
            return 1
        print(f"normalize-report-index: freshness ok ({len(rows)} rows)")
        return 0

    missing_required = sorted(
        {
            row["report_family"]
            for row in rows
            if row["freshness_status"] == "not_generated"
            and row["report_family"] in required_families
        }
    )
    if missing_required:
        for family in missing_required:
            print(f"normalize-report-index: required generated family missing: {family}")
        return 1

    if not args.check:
        write_tsv(args.output, rows)
        print(f"normalize-report-index: wrote {args.output} ({len(rows)} rows)")
    else:
        print(f"normalize-report-index: {len(rows)} rows ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReportIndexError as error:
        print(f"normalize-report-index: {error}", file=sys.stderr)
        raise SystemExit(1)
