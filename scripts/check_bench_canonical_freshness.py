#!/usr/bin/env python3
"""Check freshness for the selected canonical benchmark report row.

The checker is intentionally scoped to the Sprint 168 selected performance
publication lane. It validates report artifacts, selected-row identity,
methodology metadata, manifest agreement, and claim boundaries. It does not
compare timing values or infer performance superiority.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO_ROOT / "build" / "bench-reports" / "canonical"
REMEDIATION = "run make bench-canonical-report-freshness"
SELECTED_ARTIFACT = "bench_refactor_csc"
SELECTED_RELATIVE_PATH = "bench_refactor_csc.csv"
SELECTED_COMMAND = "tests/data/suitesparse/nos4.mtx --repeat 1"
SELECTED_FIXTURE = "nos4.mtx"
SELECTED_REPEAT = "configured_repeat_1"
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

REQUIRED_ARTIFACTS = (
    SELECTED_RELATIVE_PATH,
    "index.tsv",
    "manifest.txt",
)
REQUIRED_COLUMNS = (
    "surface",
    "category",
    "report_label",
    "generated_at_utc",
    "git_commit",
    "git_branch",
    "platform",
    "compiler",
    "runner_context",
    "build_flags",
    "cpu_model",
    "build_mode",
    "omp_num_threads",
    "artifact",
    "relative_path",
    "command",
    "report_family",
    "status",
    "support_tier",
    "claim_boundary",
    "fixture_or_workload",
    "matrix_size",
    "repeat_semantics",
    "warmup",
    "variance",
    "baseline",
    "threshold",
    "backend_context",
    "methodology_notes",
)
REQUIRED_NONEMPTY_FIELDS = (
    "surface",
    "category",
    "report_label",
    "generated_at_utc",
    "git_commit",
    "git_branch",
    "platform",
    "compiler",
    "runner_context",
    "build_flags",
    "cpu_model",
    "build_mode",
    "omp_num_threads",
    "artifact",
    "relative_path",
    "command",
    "report_family",
    "status",
    "support_tier",
    "claim_boundary",
    "fixture_or_workload",
    "repeat_semantics",
    "warmup",
    "variance",
    "baseline",
    "threshold",
    "backend_context",
    "methodology_notes",
)
SELECTED_VALUES = {
    "surface": "canonical",
    "category": "measurement",
    "artifact": SELECTED_ARTIFACT,
    "relative_path": SELECTED_RELATIVE_PATH,
    "command": SELECTED_COMMAND,
    "report_family": "benchmark",
    "status": "measurement",
    "fixture_or_workload": SELECTED_FIXTURE,
    "repeat_semantics": SELECTED_REPEAT,
    "baseline": "n/a",
    "threshold": "n/a",
}
MANIFEST_MATCH_FIELDS = (
    "report_label",
    "git_commit",
    "git_branch",
    "platform",
    "compiler",
    "runner_context",
    "build_flags",
    "cpu_model",
    "build_mode",
    "omp_num_threads",
    "support_tier",
    "claim_boundary",
    "baseline",
    "threshold",
    "methodology_notes",
)
LOCAL_SUPPORT_TIERS = {"local_only", "hosted_selected"}
LOCAL_CLAIM_BOUNDARIES = {"local_threshold_free", "hosted_selected_threshold_free"}
HOSTED_SUPPORT_TIER = "hosted_selected"
HOSTED_CLAIM_BOUNDARY = "hosted_selected_threshold_free"


class FreshnessError(RuntimeError):
    pass


def error(message: str) -> None:
    raise FreshnessError(f"{message}; {REMEDIATION}")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
    except OSError as exc:
        error(
            "freshness: error: benchmark_selected_artifact_missing: "
            f"artifact=index.tsv path={display_path(path)} detail={exc}"
        )

    if not rows:
        error(
            "freshness: error: benchmark_selected_schema: index.tsv is empty"
        )

    header = rows[0]
    width = len(header)
    for index, row in enumerate(rows, start=1):
        if len(row) != width:
            error(
                "freshness: error: benchmark_selected_schema: "
                f"row={index} expected_columns={width} observed_columns={len(row)}"
            )

    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        error(
            "freshness: error: benchmark_selected_schema: "
            f"missing_columns={','.join(missing)}"
        )
    if len(set(header)) != len(header):
        error("freshness: error: benchmark_selected_schema: duplicate header fields")

    return [dict(zip(header, row)) for row in rows[1:]]


def read_manifest(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        error(
            "freshness: error: benchmark_selected_artifact_missing: "
            f"artifact=manifest.txt path={display_path(path)} detail={exc}"
        )

    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("- ") or line == "artifacts:" or line == "notes:":
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def require_artifacts(report_dir: Path) -> None:
    if not report_dir.is_dir():
        error(
            "freshness: error: benchmark_selected_report_dir_missing: "
            f"expected={display_path(report_dir)}"
        )
    for artifact in REQUIRED_ARTIFACTS:
        path = report_dir / artifact
        if not path.is_file():
            error(
                "freshness: error: benchmark_selected_artifact_missing: "
                f"artifact={artifact} path={display_path(path)}"
            )


def selected_row(rows: list[dict[str, str]]) -> dict[str, str]:
    selected = [row for row in rows if row.get("artifact") == SELECTED_ARTIFACT]
    if not selected:
        error(
            "freshness: error: benchmark_selected_row_missing: "
            f"artifact={SELECTED_ARTIFACT}"
        )
    if len(selected) != 1:
        error(
            "freshness: error: benchmark_selected_row_duplicate: "
            f"artifact={SELECTED_ARTIFACT} observed_count={len(selected)}"
        )
    return selected[0]


def check_nonempty(row: dict[str, str]) -> None:
    for field in REQUIRED_NONEMPTY_FIELDS:
        if not row.get(field, "").strip():
            error(
                "freshness: error: benchmark_selected_metadata_missing: "
                f"field={field}"
            )


def check_selected_values(row: dict[str, str], report_dir: Path) -> None:
    for field, expected in SELECTED_VALUES.items():
        observed = row.get(field, "")
        if observed != expected:
            error(
                "freshness: error: benchmark_selected_value: "
                f"field={field} expected={expected} observed={observed}"
            )

    if not TIMESTAMP_RE.match(row["generated_at_utc"]):
        error(
            "freshness: error: benchmark_selected_value: "
            "field=generated_at_utc expected=YYYY-MM-DDTHH:MM:SSZ "
            f"observed={row['generated_at_utc']}"
        )

    notes = row["methodology_notes"].split(";")
    if "not_portable_performance_claim" not in notes:
        error(
            "freshness: error: benchmark_selected_value: "
            "field=methodology_notes expected_token=not_portable_performance_claim "
            f"observed={row['methodology_notes']}"
        )

    relative_path = report_dir / row["relative_path"]
    if not relative_path.is_file():
        error(
            "freshness: error: benchmark_selected_artifact_missing: "
            f"artifact={row['relative_path']} path={display_path(relative_path)}"
        )


def check_claim_boundary(row: dict[str, str], mode: str) -> None:
    support_tier = row["support_tier"]
    claim_boundary = row["claim_boundary"]
    if mode == "hosted":
        if support_tier != HOSTED_SUPPORT_TIER:
            error(
                "freshness: error: benchmark_selected_claim_boundary: "
                f"field=support_tier expected={HOSTED_SUPPORT_TIER} observed={support_tier}"
            )
        if claim_boundary != HOSTED_CLAIM_BOUNDARY:
            error(
                "freshness: error: benchmark_selected_claim_boundary: "
                f"field=claim_boundary expected={HOSTED_CLAIM_BOUNDARY} observed={claim_boundary}"
            )
        if row["runner_context"] == "local":
            error(
                "freshness: error: benchmark_selected_metadata_missing: "
                "field=runner_context hosted_value_must_not_be_local"
            )
        if row["build_flags"] == "not_recorded":
            error(
                "freshness: error: benchmark_selected_metadata_missing: "
                "field=build_flags hosted_value_must_not_be_not_recorded"
            )
        if row["report_label"] == "unlabeled":
            error(
                "freshness: error: benchmark_selected_metadata_missing: "
                "field=report_label hosted_value_must_not_be_unlabeled"
            )
        return

    if support_tier not in LOCAL_SUPPORT_TIERS:
        error(
            "freshness: error: benchmark_selected_claim_boundary: "
            f"field=support_tier allowed={','.join(sorted(LOCAL_SUPPORT_TIERS))} "
            f"observed={support_tier}"
        )
    if claim_boundary not in LOCAL_CLAIM_BOUNDARIES:
        error(
            "freshness: error: benchmark_selected_claim_boundary: "
            f"field=claim_boundary allowed={','.join(sorted(LOCAL_CLAIM_BOUNDARIES))} "
            f"observed={claim_boundary}"
        )


def check_manifest(row: dict[str, str], manifest: dict[str, str]) -> None:
    for field in MANIFEST_MATCH_FIELDS:
        observed = row.get(field, "")
        manifest_value = manifest.get(field, "")
        if observed != manifest_value:
            error(
                "freshness: error: benchmark_selected_manifest_mismatch: "
                f"field={field} row={observed} manifest={manifest_value}"
            )


def check_report(report_dir: Path, mode: str) -> None:
    require_artifacts(report_dir)
    rows = read_tsv(report_dir / "index.tsv")
    row = selected_row(rows)
    manifest = read_manifest(report_dir / "manifest.txt")
    check_nonempty(row)
    check_selected_values(row, report_dir)
    check_claim_boundary(row, mode)
    check_manifest(row, manifest)
    print(
        "bench-canonical-freshness: passed "
        f"(mode={mode}; artifact={SELECTED_ARTIFACT}; report_dir={display_path(report_dir)})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--mode", choices=("local", "hosted"), default="local")
    args = parser.parse_args()

    report_dir = args.report_dir
    if not report_dir.is_absolute():
        report_dir = REPO_ROOT / report_dir

    try:
        check_report(report_dir, args.mode)
    except FreshnessError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
