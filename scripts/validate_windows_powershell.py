#!/usr/bin/env python3
"""Validate owned Windows PowerShell workflow material.

This script validates the selected Windows CI PowerShell surface without
executing the workflow commands. Local runs without pwsh return exit 2 after
structural checks; hosted Windows runs should pass --require-pwsh and fail
closed if pwsh is unavailable or any snippet does not parse.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "windows-ci.yml"
SELECTED_TARGET_MANIFEST = (
    REPO_ROOT / "tests" / "corpus" / "manifests" / "selected_report_targets.tsv"
)
WINDOWS_DEFERRAL_RECORD = (
    REPO_ROOT
    / "docs"
    / "planning"
    / "EPIC_16"
    / "SPRINT_182"
    / "artifacts"
    / "windows-report-freshness-deferral-decision.md"
)

DEFERRAL_MARKER = "Windows report freshness remains formally deferred"
WORKFLOW_SELECTED_CHOLESKY_MARKER = "Sprint 190 promotes one bounded selected Cholesky comparison"
WORKFLOW_SELECTED_QR_MARKER = "Sprint 209 adds one bounded selected QR incompatible"
WINDOWS_RUNNER = "windows-2022"
FORBIDDEN_SELECTED_FRESHNESS = (
    "report-index-oracle-freshness",
    "report-index-comparison-freshness",
    "bench-canonical-report-freshness",
    "check_bench_canonical_freshness.py",
    "sprint159-oracle-freshness",
    "sprint175-linux-selected-comparison-freshness",
    "sprint175-macos-selected-comparison-freshness",
    "sprint168-selected-performance-freshness",
)
SELECTED_REPORT_FAMILIES = {"oracle", "comparison", "benchmark"}
HOSTED_VALIDATION_JOB = "powershell-validation"
HOSTED_VALIDATION_STEP_NAME = "Validate owned Windows PowerShell workflow material"
HOSTED_VALIDATION_COMMAND = "python scripts/validate_windows_powershell.py --require-pwsh"
WINDOWS_SELECTED_CHOLESKY_JOB = "selected-comparison-freshness"
WINDOWS_SELECTED_CHOLESKY_ARTIFACT = "sprint190-windows-selected-comparison-cholesky"
WINDOWS_SELECTED_CHOLESKY_TARGET_ID = "SRT-COMP-CHOLESKY-SPD-TRIDIAG-5"
WINDOWS_SELECTED_CHOLESKY_FAMILY = "comparison"
WINDOWS_SELECTED_CHOLESKY_SUBFAMILY = "cholesky_spd_tridiag_5"
WINDOWS_SELECTED_CHOLESKY_TARGET_KEY = "cholesky-spd-tridiag-5"
WINDOWS_SELECTED_CHOLESKY_ARTIFACT_PATTERN = "build/comparison/cholesky_spd_tridiag_5/study.tsv"
WINDOWS_SELECTED_CHOLESKY_GENERATOR_COMMAND = (
    "python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5"
)
WINDOWS_SELECTED_CHOLESKY_SUPPORT_TIER = "local_only"
WINDOWS_SELECTED_CHOLESKY_CLAIM_SCOPE = (
    "Selected Cholesky SPD tridiagonal solve comparison rows are fresh for the "
    "named fixture against the selected source-controlled dense Cholesky reference helper."
)
WINDOWS_SELECTED_CHOLESKY_WORKFLOW_FILES = (
    ".github/workflows/ci.yml",
    ".github/workflows/macos-ci.yml",
)
WINDOWS_SELECTED_CHOLESKY_WORKFLOW_JOBS = (
    "generated-report-freshness",
    "selected-comparison-freshness",
)
WINDOWS_SELECTED_CHOLESKY_WORKFLOW_ARTIFACTS = (
    "sprint175-linux-selected-comparison-freshness",
    "sprint175-macos-selected-comparison-freshness",
)
WINDOWS_SELECTED_CHOLESKY_WORKFLOW_PLATFORMS = ("linux", "macos")
WINDOWS_SELECTED_CHOLESKY_NON_CLAIMS = (
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
WINDOWS_SELECTED_CHOLESKY_MANIFEST_REQUIRED_FILES = (
    "project_observations.tsv",
    "baseline_observations.tsv",
    "dependency_status.tsv",
    "study.tsv",
    "summary.md",
    "manifest.tsv",
)
WINDOWS_SELECTED_CHOLESKY_EXPECTED_ROWS = "6"
WINDOWS_SELECTED_CHOLESKY_EXPECTED_ROW_IDS = (
    "comparison_cholesky_spd_tridiag_5_project_status_v1",
    "comparison_cholesky_spd_tridiag_5_baseline_status_v1",
    "comparison_cholesky_spd_tridiag_5_residual_norm_v1",
    "comparison_cholesky_spd_tridiag_5_solution_norm_v1",
    "comparison_cholesky_spd_tridiag_5_solution_values_v1",
    "comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1",
)
WINDOWS_SELECTED_CHOLESKY_GENERATOR = (
    "python scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 "
    "--probe-build-system cmake --cmake-generator \"Visual Studio 17 2022\" "
    "--cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib"
)
WINDOWS_SELECTED_CHOLESKY_FRESHNESS = (
    "python scripts/normalize_report_index.py --family comparison --require-generated "
    "comparison --check-freshness --selected-target cholesky-spd-tridiag-5"
)
WINDOWS_SELECTED_CHOLESKY_REQUIRED_FILES = (
    "build/comparison/cholesky_spd_tridiag_5/project_observations.tsv",
    "build/comparison/cholesky_spd_tridiag_5/baseline_observations.tsv",
    "build/comparison/cholesky_spd_tridiag_5/dependency_status.tsv",
    "build/comparison/cholesky_spd_tridiag_5/study.tsv",
    "build/comparison/cholesky_spd_tridiag_5/summary.md",
    "build/comparison/cholesky_spd_tridiag_5/manifest.tsv",
)
WINDOWS_SELECTED_QR_JOB = "selected-qr-incompatible-comparison-freshness"
WINDOWS_SELECTED_QR_ARTIFACT = "sprint209-windows-selected-comparison-qr-incompatible"
WINDOWS_SELECTED_QR_TARGET_ID = "SRT-COMP-QR-INCOMPATIBLE-LS"
WINDOWS_SELECTED_QR_FAMILY = "comparison"
WINDOWS_SELECTED_QR_TARGET_KEY = "qr-incompatible-ls"
WINDOWS_SELECTED_QR_SUBFAMILY = "qr_incompatible_ls"
WINDOWS_SELECTED_QR_ARTIFACT_PATTERN = "build/comparison/qr_incompatible_ls/study.tsv"
WINDOWS_SELECTED_QR_GENERATOR_COMMAND = (
    "python3 scripts/run_external_comparison.py --target qr-incompatible-ls"
)
WINDOWS_SELECTED_QR_SUPPORT_TIER = "local_only"
WINDOWS_SELECTED_QR_CLAIM_SCOPE = (
    "Selected QR incompatible least-squares comparison rows are fresh for the "
    "named fixture against the selected source-controlled dense reference helper."
)
WINDOWS_SELECTED_QR_WORKFLOW_FILES = (
    ".github/workflows/ci.yml",
    ".github/workflows/macos-ci.yml",
)
WINDOWS_SELECTED_QR_WORKFLOW_JOBS = (
    "generated-report-freshness",
    "selected-comparison-freshness",
)
WINDOWS_SELECTED_QR_WORKFLOW_ARTIFACTS = (
    "sprint175-linux-selected-comparison-freshness",
    "sprint175-macos-selected-comparison-freshness",
)
WINDOWS_SELECTED_QR_WORKFLOW_PLATFORMS = ("linux", "macos")
WINDOWS_SELECTED_QR_NON_CLAIMS = (
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
WINDOWS_SELECTED_QR_MANIFEST_REQUIRED_FILES = (
    "project_observations.tsv",
    "baseline_observations.tsv",
    "dependency_status.tsv",
    "study.tsv",
    "summary.md",
    "manifest.tsv",
)
WINDOWS_SELECTED_QR_EXPECTED_ROWS = "6"
WINDOWS_SELECTED_QR_EXPECTED_ROW_IDS = (
    "comparison_qr_overdetermined_incompatible_4x2_project_status_v1",
    "comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1",
    "comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1",
    "comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1",
    "comparison_qr_overdetermined_incompatible_4x2_solution_values_v1",
    "comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1",
)
WINDOWS_SELECTED_QR_GENERATOR = (
    "python scripts/run_external_comparison.py --target qr-incompatible-ls "
    "--probe-build-system cmake --cmake-generator \"Visual Studio 17 2022\" "
    "--cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib"
)
WINDOWS_SELECTED_QR_FRESHNESS = (
    "python scripts/normalize_report_index.py --family comparison --require-generated "
    "comparison --check-freshness --selected-target qr-incompatible-ls"
)
WINDOWS_SELECTED_QR_REQUIRED_FILES = (
    "build/comparison/qr_incompatible_ls/project_observations.tsv",
    "build/comparison/qr_incompatible_ls/baseline_observations.tsv",
    "build/comparison/qr_incompatible_ls/dependency_status.tsv",
    "build/comparison/qr_incompatible_ls/study.tsv",
    "build/comparison/qr_incompatible_ls/summary.md",
    "build/comparison/qr_incompatible_ls/manifest.tsv",
)
CLAIM_BOUNDARY_MARKERS = {
    REPO_ROOT / "README.md": (
        "Windows Makefile parity",
        "Windows `pkg-config` parity",
        "Sprint 190 adds one guarded Windows hosted",
        "evidence for that exact path and re-deferred selected Windows freshness",
        "Sprint 209 adds one bounded Windows QR incompatible evidence-collection lane",
        "the QR incompatible least-squares target remains\noutside Windows selected freshness",
        "outside the one Sprint 190 Cholesky",
    ),
    REPO_ROOT / "INSTALL.md": (
        "hosted PowerShell validation ownership for selected Windows workflow snippets",
        "bounded selected Cholesky comparison freshness workflow",
        "Sprint 209 adds one bounded Windows QR incompatible evidence-collection lane",
        "The QR incompatible least-squares target remains outside Windows selected freshness",
        "broad report freshness, selected oracle freshness, selected benchmark freshness",
        "does not imply Windows Makefile parity",
        "runtime-loader behavior, or broad Windows parity",
    ),
    REPO_ROOT / "docs" / "maintainer_guide.md": (
        "Sprint 190 adds one bounded Windows hosted workflow\npath for `cholesky-spd-tridiag-5`",
        "Sprint 209 adds one bounded Windows QR incompatible evidence-collection lane",
        "The Sprint 182 deferral remains active for every Windows report freshness\nsurface outside the one Sprint 190 Cholesky workflow path",
        "The QR\nincompatible least-squares target remains outside Windows selected freshness",
        "make windows-powershell-validate",
        "python scripts/validate_windows_powershell.py --require-pwsh",
        "If a local PowerShell\ncheck is unavailable, record that as an environment residual",
        "unavailable local PowerShell checks out of pass evidence",
    ),
    REPO_ROOT / "tests" / "corpus" / "README.md": (
        "Sprint 190 wires one bounded Windows selected Cholesky comparison\nfreshness workflow",
        "Sprint 209 adds one bounded Windows QR incompatible\n"
        "evidence-collection lane",
        "The Sprint 182 deferral\nremains active for all other Windows report freshness",
        "The QR incompatible least-squares target remains outside Windows selected\nfreshness",
        "hosted Windows PowerShell validation lane owns selected workflow snippet\nparsing",
        "unavailable\nlocal PowerShell validation",
        "reinterpret those states as pass evidence",
    ),
    REPO_ROOT / "tests" / "corpus" / "schemas" / "report_index_fields.md": (
        "Sprint 190 adds one bounded Windows hosted workflow path\nfor `cholesky-spd-tridiag-5`",
        "Sprint 209 adds one bounded Windows QR incompatible\n"
        "evidence-collection lane",
        "The QR incompatible least-squares target remains outside Windows selected\nfreshness",
        "external-library parity; platform proof; package proof; ABI proof; performance\nproof; release proof; or state-of-the-art evidence",
    ),
}
UNSUPPORTED_WINDOWS_CLAIM_PATTERNS = (
    re.compile(
        r"Windows report freshness (?:is |now )?(?:supported|promoted|complete|closed)",
        re.I,
    ),
    re.compile(r"PowerShell validation (?:proves|promotes|closes) Windows report freshness", re.I),
    re.compile(
        r"(?:Windows selected (?:Cholesky|comparison|report)|"
        r"Windows (?:selected )?QR incompatible(?: selected)?|"
        r"selected Windows(?: (?:Cholesky|comparison|report|QR incompatible))?) freshness "
        r"(?:is |now )?(?:supported|promoted|complete|closed)",
        re.I,
    ),
    re.compile(
        r"(?:QR incompatible|qr-incompatible|least-squares).*Windows selected freshness "
        r"(?:is |now )?(?:supported|promoted|complete|closed)",
        re.I,
    ),
    re.compile(r"local unavailable PowerShell (?:is|counts as|proves) pass evidence", re.I),
    re.compile(r"Windows selected report artifacts? (?:are |now )?(?:published|uploaded)", re.I),
)


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Step:
    job_id: str
    name: str
    run: str
    shell: str


@dataclass(frozen=True)
class StepRequirement:
    job_id: str
    name_anchor: str
    tokens: tuple[str, ...]


STEP_REQUIREMENTS = (
    StepRequirement(
        "build-and-test",
        "Run enforced reviewed CMake configure path",
        ("cmake -S . -B build", "Visual Studio 17 2022"),
    ),
    StepRequirement(
        "build-and-test",
        "Run enforced reviewed CMake build path",
        ("cmake --build build", "Release"),
    ),
    StepRequirement(
        "build-and-test",
        "Inspect enforced Windows reviewed consumer CTest surface",
        ("EXPECTED_WINDOWS_CTEST_COUNT", "Total Tests:"),
    ),
    StepRequirement(
        "build-and-test",
        "Run enforced reviewed CMake execution path",
        ("ctest --test-dir build", "--output-on-failure"),
    ),
    StepRequirement(
        "install-and-downstream",
        "Run reviewed CMake install/downstream validation proof",
        ("sparse_lu_ortho.lib", "sparse.pc", "metadata-only", "find_package", "mismatch"),
    ),
    StepRequirement(
        WINDOWS_SELECTED_CHOLESKY_JOB,
        "Configure selected Cholesky comparison library",
        ("cmake -S . -B build", "Visual Studio 17 2022"),
    ),
    StepRequirement(
        WINDOWS_SELECTED_CHOLESKY_JOB,
        "Build selected Cholesky comparison library",
        ("cmake --build build", "Release", "sparse_lu_ortho"),
    ),
    StepRequirement(
        WINDOWS_SELECTED_QR_JOB,
        "Configure selected QR incompatible comparison library",
        ("cmake -S . -B build", "Visual Studio 17 2022"),
    ),
    StepRequirement(
        WINDOWS_SELECTED_QR_JOB,
        "Build selected QR incompatible comparison library",
        ("cmake --build build", "Release", "sparse_lu_ortho"),
    ),
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValidationError(f"missing required file: {rel(path)}") from exc


def fail(message: str) -> int:
    print(f"windows-powershell-validate: FAIL: {message}", file=sys.stderr, flush=True)
    return 1


def unavailable(message: str) -> int:
    print(
        f"windows-powershell-validate: UNAVAILABLE: {message}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "windows-powershell-validate: local unavailable PowerShell is not pass evidence",
        file=sys.stderr,
        flush=True,
    )
    return 2


def pass_msg(message: str) -> None:
    print(f"windows-powershell-validate: {message} ok", flush=True)


def find_job_block(text: str, job_id: str) -> str:
    marker = f"  {job_id}:\n"
    start = text.find(marker)
    if start == -1:
        raise ValidationError(f"windows workflow missing job {job_id!r}")
    next_start = len(text)
    search_pos = start + len(marker)
    for line_start in line_offsets(text, search_pos):
        line_end = text.find("\n", line_start)
        line = text[line_start : line_end + 1 if line_end != -1 else len(text)]
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            next_start = line_start
            break
    return text[start:next_start]


def line_offsets(text: str, start: int = 0) -> list[int]:
    offsets = []
    pos = text.find("\n", start)
    while pos != -1 and pos + 1 < len(text):
        offsets.append(pos + 1)
        pos = text.find("\n", pos + 1)
    return offsets


def field_value(block: str, field: str) -> str:
    prefix = f"    {field}:"
    for line in block.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"')
    raise ValidationError(f"job block missing {field!r}")


def upload_path_entries(job_block: str, artifact_name: str) -> tuple[str, ...]:
    lines = job_block.splitlines()
    in_step = False
    is_upload = False
    in_path = False
    current_name = ""
    current_fail_closed = False
    current_paths: list[str] = []
    upload_names: list[str] = []
    matching_uploads: list[tuple[bool, tuple[str, ...]]] = []

    def finish_step() -> None:
        if is_upload:
            upload_names.append(current_name)
            if current_name == artifact_name:
                matching_uploads.append((current_fail_closed, tuple(current_paths)))

    for line in lines:
        if line.startswith("      - "):
            if in_step:
                finish_step()
            in_step = True
            is_upload = False
            in_path = False
            current_name = ""
            current_fail_closed = False
            current_paths = []
            continue
        if not in_step:
            continue
        if line.strip() == "uses: actions/upload-artifact@v4":
            is_upload = True
            in_path = False
            continue
        if is_upload and line.startswith("          name: "):
            current_name = line.split(":", 1)[1].strip()
            continue
        if is_upload and line.strip() == "if-no-files-found: error":
            current_fail_closed = True
            continue
        if is_upload and line.strip() == "path: |":
            in_path = True
            continue
        if in_path:
            if line.startswith("            "):
                path = line.strip()
                if path:
                    current_paths.append(path)
                continue
            in_path = False
    if in_step:
        finish_step()
    if upload_names != [artifact_name]:
        raise ValidationError(
            "selected QR upload validation must have exactly one upload-artifact step "
            f"named {artifact_name!r}"
        )
    if len(matching_uploads) != 1:
        raise ValidationError(
            f"upload artifact {artifact_name!r} must have exactly one matching upload step"
        )
    fail_closed, paths = matching_uploads[0]
    if not fail_closed:
        raise ValidationError(
            f"upload artifact {artifact_name!r} must declare if-no-files-found: error"
        )
    if not paths:
        raise ValidationError(f"upload artifact {artifact_name!r} missing path entries")
    return paths


def parse_steps(job_id: str, job_block: str) -> list[Step]:
    lines = job_block.splitlines()
    steps: list[Step] = []
    current: dict[str, str] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("      - name: "):
            if current is not None:
                steps.append(
                    Step(
                        job_id=job_id,
                        name=current.get("name", ""),
                        run=current.get("run", ""),
                        shell=current.get("shell", ""),
                    )
                )
            current = {"name": line.split(":", 1)[1].strip()}
            i += 1
            continue
        if current is None:
            i += 1
            continue
        if line.startswith("        shell: "):
            current["shell"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.startswith("        run: |"):
            block_lines: list[str] = []
            i += 1
            while i < len(lines) and (
                lines[i].startswith("          ") or lines[i].strip() == ""
            ):
                block_lines.append(lines[i][10:] if lines[i].startswith("          ") else "")
                i += 1
            current["run"] = "\n".join(block_lines)
            continue
        if line.startswith("        run: "):
            current["run"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        i += 1
    if current is not None:
        steps.append(
            Step(
                job_id=job_id,
                name=current.get("name", ""),
                run=current.get("run", ""),
                shell=current.get("shell", ""),
            )
        )
    return steps


def split_manifest_values(value: str) -> list[str]:
    if value == "none":
        return []
    return [part for part in value.split(";") if part]


def selected_report_targets() -> list[dict[str, str]]:
    with SELECTED_TARGET_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def selected_report_freshness_tokens(rows: list[dict[str, str]]) -> tuple[str, ...]:
    tokens = set(FORBIDDEN_SELECTED_FRESHNESS)
    for row in rows:
        for field in ("generator_command", "workflow_job", "workflow_artifact"):
            tokens.update(split_manifest_values(row[field]))
    return tuple(sorted(token for token in tokens if token != "none"))


def validate_selected_report_references(rows: list[dict[str, str]]) -> None:
    for row in rows:
        target_id = row["target_id"]
        if row["family"] not in SELECTED_REPORT_FAMILIES:
            raise ValidationError(
                f"selected_report_targets.tsv has unexpected selected family "
                f"{row['family']!r}: {target_id}"
            )
        workflow_files = split_manifest_values(row["workflow_file"])
        workflow_jobs = split_manifest_values(row["workflow_job"])
        workflow_artifacts = split_manifest_values(row["workflow_artifact"])
        workflow_platforms = split_manifest_values(row["workflow_platforms"])
        if not workflow_files:
            raise ValidationError(f"{target_id} is missing workflow_file metadata")
        if not workflow_jobs:
            raise ValidationError(f"{target_id} is missing workflow_job metadata")
        if not workflow_artifacts:
            raise ValidationError(f"{target_id} is missing workflow_artifact metadata")
        if not workflow_platforms:
            raise ValidationError(f"{target_id} is missing workflow_platforms metadata")
        if len(workflow_artifacts) not in (1, len(workflow_platforms)):
            raise ValidationError(
                f"{target_id} workflow_artifact must contain one shared artifact "
                "or one artifact per workflow platform"
            )
        for workflow_file in workflow_files:
            workflow_path = REPO_ROOT / workflow_file
            if not workflow_path.is_file():
                raise ValidationError(
                    f"{target_id} references missing workflow_file {workflow_file!r}"
                )
        for workflow_artifact in workflow_artifacts:
            if not workflow_artifact.startswith("sprint"):
                raise ValidationError(
                    f"{target_id} workflow_artifact is not sprint-scoped: "
                    f"{workflow_artifact!r}"
                )
    pass_msg(f"selected report manifest references ({len(rows)} rows)")


def validate_manifest_windows_deferral(rows: list[dict[str, str]]) -> None:
    for row in rows:
        target_id = row["target_id"]
        workflow_files = split_manifest_values(row["workflow_file"])
        workflow_jobs = split_manifest_values(row["workflow_job"])
        workflow_artifacts = split_manifest_values(row["workflow_artifact"])
        platforms = split_manifest_values(row["workflow_platforms"])
        if "windows" in platforms:
            raise ValidationError(
                "selected_report_targets.tsv must not list windows while "
                f"Windows report freshness is deferred: {target_id}"
            )
        if target_id not in (
            WINDOWS_SELECTED_CHOLESKY_TARGET_ID,
            WINDOWS_SELECTED_QR_TARGET_ID,
        ):
            if (
                str(WINDOWS_WORKFLOW.relative_to(REPO_ROOT)) in workflow_files
                or WINDOWS_SELECTED_CHOLESKY_ARTIFACT in workflow_artifacts
            ):
                raise ValidationError(
                    "non-Cholesky selected rows must not reference Windows "
                    "selected Cholesky workflow metadata while Windows report "
                    f"freshness is deferred: {target_id}"
                )
        if target_id != WINDOWS_SELECTED_QR_TARGET_ID:
            if (
                WINDOWS_SELECTED_QR_JOB in workflow_jobs
                or WINDOWS_SELECTED_QR_ARTIFACT in workflow_artifacts
            ):
                raise ValidationError(
                    "non-QR selected rows must not reference Sprint 209 Windows "
                    "QR workflow metadata while QR freshness is re-deferred: "
                    f"{target_id}"
                )
    matches = [
        row for row in rows if row["target_id"] == WINDOWS_SELECTED_CHOLESKY_TARGET_ID
    ]
    if len(matches) != 1:
        raise ValidationError(
            "selected_report_targets.tsv must contain exactly one selected "
            f"Cholesky row, got {len(matches)}"
        )
    cholesky = matches[0]
    exact_fields = {
        "family": WINDOWS_SELECTED_CHOLESKY_FAMILY,
        "subfamily": WINDOWS_SELECTED_CHOLESKY_SUBFAMILY,
        "target_key": WINDOWS_SELECTED_CHOLESKY_TARGET_KEY,
        "artifact_pattern": WINDOWS_SELECTED_CHOLESKY_ARTIFACT_PATTERN,
        "generator_command": WINDOWS_SELECTED_CHOLESKY_GENERATOR_COMMAND,
        "support_tier": WINDOWS_SELECTED_CHOLESKY_SUPPORT_TIER,
        "required_files": ";".join(WINDOWS_SELECTED_CHOLESKY_MANIFEST_REQUIRED_FILES),
        "expected_rows": WINDOWS_SELECTED_CHOLESKY_EXPECTED_ROWS,
        "expected_row_ids": ";".join(WINDOWS_SELECTED_CHOLESKY_EXPECTED_ROW_IDS),
        "workflow_file": ";".join(WINDOWS_SELECTED_CHOLESKY_WORKFLOW_FILES),
        "workflow_job": ";".join(WINDOWS_SELECTED_CHOLESKY_WORKFLOW_JOBS),
        "workflow_artifact": ";".join(WINDOWS_SELECTED_CHOLESKY_WORKFLOW_ARTIFACTS),
        "workflow_platforms": ";".join(WINDOWS_SELECTED_CHOLESKY_WORKFLOW_PLATFORMS),
        "claim_scope": WINDOWS_SELECTED_CHOLESKY_CLAIM_SCOPE,
        "non_claims": ";".join(WINDOWS_SELECTED_CHOLESKY_NON_CLAIMS),
    }
    for field, expected in exact_fields.items():
        if cholesky[field] != expected:
            raise ValidationError(
                f"{WINDOWS_SELECTED_CHOLESKY_TARGET_ID} must retain deferred "
                f"Windows manifest {field} metadata"
            )

    qr_matches = [row for row in rows if row["target_id"] == WINDOWS_SELECTED_QR_TARGET_ID]
    if len(qr_matches) != 1:
        raise ValidationError(
            "selected_report_targets.tsv must contain exactly one selected "
            f"QR incompatible row, got {len(qr_matches)}"
        )
    qr = qr_matches[0]
    qr_workflow_files = split_manifest_values(qr["workflow_file"])
    qr_workflow_jobs = split_manifest_values(qr["workflow_job"])
    qr_workflow_artifacts = split_manifest_values(qr["workflow_artifact"])
    if str(WINDOWS_WORKFLOW.relative_to(REPO_ROOT)) in qr_workflow_files:
        raise ValidationError(
            f"{WINDOWS_SELECTED_QR_TARGET_ID} must not list Sprint 209 Windows "
            "QR workflow file metadata while re-deferred"
        )
    if WINDOWS_SELECTED_QR_JOB in qr_workflow_jobs:
        raise ValidationError(
            f"{WINDOWS_SELECTED_QR_TARGET_ID} must not list Sprint 209 Windows "
            "QR workflow job metadata while re-deferred"
        )
    if WINDOWS_SELECTED_QR_ARTIFACT in qr_workflow_artifacts:
        raise ValidationError(
            f"{WINDOWS_SELECTED_QR_TARGET_ID} must not list Sprint 209 Windows "
            "QR workflow artifact metadata while re-deferred"
        )

    qr_exact_fields = {
        "family": WINDOWS_SELECTED_QR_FAMILY,
        "subfamily": WINDOWS_SELECTED_QR_SUBFAMILY,
        "target_key": WINDOWS_SELECTED_QR_TARGET_KEY,
        "artifact_pattern": WINDOWS_SELECTED_QR_ARTIFACT_PATTERN,
        "generator_command": WINDOWS_SELECTED_QR_GENERATOR_COMMAND,
        "support_tier": WINDOWS_SELECTED_QR_SUPPORT_TIER,
        "required_files": ";".join(WINDOWS_SELECTED_QR_MANIFEST_REQUIRED_FILES),
        "expected_rows": WINDOWS_SELECTED_QR_EXPECTED_ROWS,
        "expected_row_ids": ";".join(WINDOWS_SELECTED_QR_EXPECTED_ROW_IDS),
        "workflow_file": ";".join(WINDOWS_SELECTED_QR_WORKFLOW_FILES),
        "workflow_job": ";".join(WINDOWS_SELECTED_QR_WORKFLOW_JOBS),
        "workflow_artifact": ";".join(WINDOWS_SELECTED_QR_WORKFLOW_ARTIFACTS),
        "workflow_platforms": ";".join(WINDOWS_SELECTED_QR_WORKFLOW_PLATFORMS),
        "claim_scope": WINDOWS_SELECTED_QR_CLAIM_SCOPE,
        "non_claims": ";".join(WINDOWS_SELECTED_QR_NON_CLAIMS),
    }
    for field, expected in qr_exact_fields.items():
        if qr[field] != expected:
            raise ValidationError(
                f"{WINDOWS_SELECTED_QR_TARGET_ID} must retain re-deferred "
                f"Windows QR manifest {field} metadata"
            )
    pass_msg(f"selected manifest has no windows workflow platforms ({len(rows)} rows)")


def validate_deferral_record() -> None:
    text = read_text(WINDOWS_DEFERRAL_RECORD)
    if DEFERRAL_MARKER not in text:
        raise ValidationError("Windows report freshness deferral marker is missing")
    pass_msg("Windows report freshness deferral record")


def validate_workflow_structure(
    text: str,
    forbidden_selected_freshness: tuple[str, ...] = FORBIDDEN_SELECTED_FRESHNESS,
) -> list[Step]:
    if WORKFLOW_SELECTED_CHOLESKY_MARKER not in text:
        raise ValidationError("windows workflow missing Sprint 190 selected Cholesky comment")
    pass_msg("windows workflow selected Cholesky comment")
    if WORKFLOW_SELECTED_QR_MARKER not in text:
        raise ValidationError("windows workflow missing Sprint 209 selected QR comment")
    pass_msg("windows workflow selected QR comment")

    selected_lane = validate_windows_selected_cholesky_lane(text)
    selected_qr_lane = validate_windows_selected_qr_lane(text)
    text_without_selected_lane = text.replace(selected_lane, "", 1).replace(
        selected_qr_lane, "", 1
    )

    steps: list[Step] = []
    for job_id in (
        "build-and-test",
        "install-and-downstream",
        HOSTED_VALIDATION_JOB,
        WINDOWS_SELECTED_CHOLESKY_JOB,
        WINDOWS_SELECTED_QR_JOB,
    ):
        block = find_job_block(text, job_id)
        runner = field_value(block, "runs-on")
        if runner != WINDOWS_RUNNER:
            raise ValidationError(f"{job_id} must run on {WINDOWS_RUNNER}, got {runner!r}")
        pass_msg(f"{job_id} runner")
        steps.extend(parse_steps(job_id, block))

    for needle in forbidden_selected_freshness:
        if needle in text_without_selected_lane:
            raise ValidationError(
                f"windows workflow must not run or upload selected report freshness {needle!r}"
            )
    if "actions/upload-artifact" in text_without_selected_lane:
        raise ValidationError(
            "windows workflow must not publish hosted artifacts outside the "
            "owned selected comparison freshness lanes"
        )
    pass_msg("windows selected report freshness bounded promotion")

    return steps


def validate_windows_selected_cholesky_lane(text: str) -> str:
    block = find_job_block(text, WINDOWS_SELECTED_CHOLESKY_JOB)
    if "timeout-minutes: 20" not in block:
        raise ValidationError(
            f"{WINDOWS_SELECTED_CHOLESKY_JOB} must declare timeout-minutes: 20"
        )
    for needle in (
        WINDOWS_SELECTED_CHOLESKY_GENERATOR,
        WINDOWS_SELECTED_CHOLESKY_FRESHNESS,
        "actions/upload-artifact@v4",
        f"name: {WINDOWS_SELECTED_CHOLESKY_ARTIFACT}",
        "if-no-files-found: error",
    ):
        if needle not in block:
            raise ValidationError(
                f"{WINDOWS_SELECTED_CHOLESKY_JOB} missing selected Cholesky token {needle!r}"
            )
    if "build/comparison/**" in block:
        raise ValidationError(
            f"{WINDOWS_SELECTED_CHOLESKY_JOB} must not use broad comparison artifact paths"
        )
    for required_file in WINDOWS_SELECTED_CHOLESKY_REQUIRED_FILES:
        if required_file not in block:
            raise ValidationError(
                f"{WINDOWS_SELECTED_CHOLESKY_JOB} missing upload path {required_file!r}"
            )
    return block


def validate_windows_selected_qr_lane(text: str) -> str:
    block = find_job_block(text, WINDOWS_SELECTED_QR_JOB)
    if "timeout-minutes: 20" not in block:
        raise ValidationError(f"{WINDOWS_SELECTED_QR_JOB} must declare timeout-minutes: 20")
    for needle in (
        WINDOWS_SELECTED_QR_GENERATOR,
        WINDOWS_SELECTED_QR_FRESHNESS,
        "actions/upload-artifact@v4",
        f"name: {WINDOWS_SELECTED_QR_ARTIFACT}",
        "if-no-files-found: error",
    ):
        if needle not in block:
            raise ValidationError(
                f"{WINDOWS_SELECTED_QR_JOB} missing selected QR token {needle!r}"
            )
    for forbidden in (
        "build/comparison/**",
        "build/comparison/\n",
        "sprint203-windows-selected-comparison-qr-incompatible",
    ):
        if forbidden in block:
            raise ValidationError(
                f"{WINDOWS_SELECTED_QR_JOB} must not use broad or stale QR artifact paths"
            )
    for required_file in WINDOWS_SELECTED_QR_REQUIRED_FILES:
        if required_file not in block:
            raise ValidationError(
                f"{WINDOWS_SELECTED_QR_JOB} missing upload path {required_file!r}"
            )
    actual_paths = upload_path_entries(block, WINDOWS_SELECTED_QR_ARTIFACT)
    if actual_paths != WINDOWS_SELECTED_QR_REQUIRED_FILES:
        raise ValidationError(
            f"{WINDOWS_SELECTED_QR_JOB} upload paths must match the exact selected "
            "QR six-file contract"
        )
    return block


def validate_claim_boundaries(overrides: dict[Path, str] | None = None) -> None:
    for path, markers in CLAIM_BOUNDARY_MARKERS.items():
        text = overrides[path] if overrides and path in overrides else read_text(path)
        for marker in markers:
            if marker not in text:
                raise ValidationError(
                    f"{rel(path)} missing Windows/PowerShell non-claim marker {marker!r}"
                )
        for pattern in UNSUPPORTED_WINDOWS_CLAIM_PATTERNS:
            match = pattern.search(text)
            if match:
                raise ValidationError(
                    f"{rel(path)} contains unsupported Windows/PowerShell claim "
                    f"{match.group(0)!r}"
                )
    pass_msg(f"Windows/PowerShell claim boundaries ({len(CLAIM_BOUNDARY_MARKERS)} files)")


def validate_hosted_validation_wiring(text: str) -> None:
    block = find_job_block(text, HOSTED_VALIDATION_JOB)
    runner = field_value(block, "runs-on")
    if runner != WINDOWS_RUNNER:
        raise ValidationError(
            f"{HOSTED_VALIDATION_JOB} must run on {WINDOWS_RUNNER}, got {runner!r}"
        )
    steps = parse_steps(HOSTED_VALIDATION_JOB, block)
    matches = [step for step in steps if step.name == HOSTED_VALIDATION_STEP_NAME]
    if len(matches) != 1:
        raise ValidationError(
            f"{HOSTED_VALIDATION_JOB} expected one step named "
            f"{HOSTED_VALIDATION_STEP_NAME!r}, got {len(matches)}"
        )
    step = matches[0]
    if HOSTED_VALIDATION_COMMAND not in step.run:
        raise ValidationError(
            f"{HOSTED_VALIDATION_JOB}:{step.name} must run "
            f"{HOSTED_VALIDATION_COMMAND!r}"
        )
    if step.shell != "cmd":
        raise ValidationError(
            f"{HOSTED_VALIDATION_JOB}:{step.name} must declare shell: cmd, "
            f"got {step.shell!r}"
        )
    pass_msg("hosted Windows PowerShell validation wiring")


def find_required_step(steps: list[Step], requirement: StepRequirement) -> Step:
    matches = [
        step
        for step in steps
        if step.job_id == requirement.job_id and requirement.name_anchor in step.name
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"{requirement.job_id} expected one step containing "
            f"{requirement.name_anchor!r}, got {len(matches)}"
        )
    return matches[0]


def validate_required_steps(steps: list[Step]) -> list[Step]:
    selected: list[Step] = []
    for requirement in STEP_REQUIREMENTS:
        step = find_required_step(steps, requirement)
        if step.shell != "pwsh":
            raise ValidationError(
                f"{step.job_id}:{step.name} must declare shell: pwsh, got {step.shell!r}"
            )
        if not step.run:
            raise ValidationError(f"{step.job_id}:{step.name} is missing run text")
        for token in requirement.tokens:
            if token not in step.run:
                raise ValidationError(f"{step.job_id}:{step.name} missing token {token!r}")
        selected.append(step)

    selected_keys = {(step.job_id, step.name) for step in selected}
    unowned_pwsh_steps = [
        f"{step.job_id}:{step.name}"
        for step in steps
        if step.shell == "pwsh" and (step.job_id, step.name) not in selected_keys
    ]
    if unowned_pwsh_steps:
        raise ValidationError(
            "windows workflow has unowned PowerShell steps: "
            + ", ".join(unowned_pwsh_steps)
        )
    pass_msg(f"selected PowerShell workflow steps ({len(selected)})")
    return selected


def parse_with_pwsh(pwsh: str, steps: list[Step]) -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-windows-pwsh-") as tmp:
        tmpdir = Path(tmp)
        for index, step in enumerate(steps, start=1):
            snippet = tmpdir / f"snippet-{index}.ps1"
            snippet.write_text(step.run + "\n", encoding="utf-8")
            env = os.environ.copy()
            env["SPARSE_PWSH_SNIPPET"] = str(snippet)
            result = subprocess.run(
                [
                    pwsh,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    (
                        "$ErrorActionPreference = 'Stop'; "
                        "$text = Get-Content -Raw -LiteralPath $env:SPARSE_PWSH_SNIPPET; "
                        "[scriptblock]::Create($text) | Out-Null"
                    ),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise ValidationError(
                    f"PowerShell parse failed for {step.job_id}:{step.name}: {detail}"
                )
    pass_msg(f"PowerShell parse validation ({len(steps)} snippets)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate selected Windows CI PowerShell workflow material."
    )
    parser.add_argument(
        "--require-pwsh",
        action="store_true",
        help="fail if pwsh is unavailable; intended for hosted Windows CI",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        selected_targets = selected_report_targets()
        workflow_text = read_text(WINDOWS_WORKFLOW)
        validate_selected_report_references(selected_targets)
        steps = validate_workflow_structure(
            workflow_text,
            selected_report_freshness_tokens(selected_targets),
        )
        validate_claim_boundaries()
        validate_hosted_validation_wiring(workflow_text)
        selected_steps = validate_required_steps(steps)
        validate_deferral_record()
        validate_manifest_windows_deferral(selected_targets)
        pwsh = shutil.which("pwsh")
        if pwsh is None:
            if args.require_pwsh:
                raise ValidationError("pwsh not found but --require-pwsh was set")
            return unavailable("pwsh not found; structural checks passed")
        parse_with_pwsh(pwsh, selected_steps)
    except ValidationError as exc:
        return fail(str(exc))

    print(
        "windows-powershell-validate: passed "
        f"({len(STEP_REQUIREMENTS)} selected PowerShell snippets)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
