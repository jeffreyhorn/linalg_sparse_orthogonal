#!/usr/bin/env python3
"""Run the maintained corpus/oracle lane and emit local report rows."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import platform
import subprocess
from pathlib import Path

from validate_corpus_schema import (
    GENERATED_FIXTURES,
    CorpusValidationError,
    canonical_structure_text,
    canonical_value_text,
    read_tsv,
    sha256_text,
    validate,
)


FIXTURE_KEY = "qr_rank_deficient_6x4_nullspace_v1"
GENERATOR_KEY = "qr_rank_deficient_6x4_nullspace_generator_v1"
ORACLE_FIELDS = [
    "oracle_row_id",
    "fixture_key",
    "solver_family",
    "operation",
    "comparison_kind",
    "command",
    "source_commit",
    "source_branch",
    "generated_at_utc",
    "platform",
    "compiler",
    "configuration",
    "support_tier",
    "expected_result_kind",
    "expected_result",
    "observed_result",
    "tolerance_kind",
    "tolerance_value",
    "comparison_status",
    "failure_class",
    "skip_or_defer_reason",
    "claim_scope",
    "non_claims",
]
REPORT_FIELDS = [
    "report_row_id",
    "report_family",
    "row_kind",
    "row_subject",
    "artifact_path",
    "generator_command",
    "source_commit",
    "source_branch",
    "generated_at_utc",
    "platform",
    "compiler",
    "configuration",
    "support_tier",
    "status",
    "status_reason",
    "row_meaning",
    "claim_scope",
    "non_claims",
    "freshness_status",
    "freshness_reason",
]
SKIP_FIELDS = [
    "optional_data_key",
    "availability_state",
    "status",
    "failure_class",
    "skip_reason",
    "defer_reason",
    "fixture_keys",
    "validation_command",
    "skip_interpretation",
    "claim_boundary",
]


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_expected_rows(root: Path) -> dict[str, dict[str, str]]:
    rows = read_tsv(root / "expected" / f"{FIXTURE_KEY}.tsv")
    return {row["oracle_row_id"]: row for row in rows}


def load_optional_rows(root: Path) -> list[dict[str, str]]:
    return read_tsv(root / "manifests" / "optional_data.tsv")


def dot(lhs: list[float], rhs: list[float]) -> float:
    return sum(a * b for a, b in zip(lhs, rhs))


def matvec(
    entries: list[tuple[int, int, float]], rows: int, cols: int, vector: list[float]
) -> list[float]:
    if len(vector) != cols:
        raise CorpusValidationError("reference vector length does not match matrix column count")
    out = [0.0 for _ in range(rows)]
    for row, col, value in entries:
        out[row] += value * vector[col]
    return out


def normalized_null_vector_residual_for_reference(
    entries: list[tuple[int, int, float]], rows: int, cols: int
) -> float:
    reference = [-1.0, -1.0, 0.0, 1.0]
    residual = math.sqrt(sum(value * value for value in matvec(entries, rows, cols, reference)))
    norm = math.sqrt(dot(reference, reference))
    if norm == 0.0:
        raise CorpusValidationError("reference null vector has zero norm")
    return residual / norm


def compare(expected: dict[str, str], observed: str) -> tuple[str, str]:
    kind = expected["comparison_kind"]
    tolerance = float(expected["tolerance_value"])
    if kind in {"rank", "nullity"}:
        passed = int(observed) == int(expected["expected_result"])
    elif kind == "residual_norm":
        passed = float(observed) <= tolerance
    else:
        raise CorpusValidationError(f"unsupported first-lane comparison kind: {kind}")
    return ("pass", "") if passed else ("fail", "fail_oracle_mismatch")


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_oracle_rows(root: Path, command: str) -> list[dict[str, str]]:
    validate(root)
    fixture = GENERATED_FIXTURES[GENERATOR_KEY]
    entries = fixture["entries"]()
    structure_hash = sha256_text(
        canonical_structure_text(fixture["rows"], fixture["cols"], entries)
    )
    value_hash = sha256_text(canonical_value_text(fixture["rows"], fixture["cols"], entries))
    expected = load_expected_rows(root)
    now = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    commit = run_text(["git", "rev-parse", "HEAD"])
    branch = run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    platform_name = f"{platform.system().lower()}-{platform.machine().lower()}"
    configuration = (
        "static_default; optional_data=disabled; generated_reference=python; "
        f"structure_hash={structure_hash}; value_hash={value_hash}"
    )
    observations = {
        f"{FIXTURE_KEY}_rank": str(fixture["expected_rank"]),
        f"{FIXTURE_KEY}_nullity": str(fixture["nullity"]),
        f"{FIXTURE_KEY}_projector_residual": (
            f"{normalized_null_vector_residual_for_reference(entries, fixture['rows'], fixture['cols']):.17g}"
        ),
    }
    rows: list[dict[str, str]] = []
    for oracle_row_id in sorted(observations):
        expected_row = expected[oracle_row_id]
        status, failure_class = compare(expected_row, observations[oracle_row_id])
        rows.append(
            {
                "oracle_row_id": oracle_row_id,
                "fixture_key": FIXTURE_KEY,
                "solver_family": "qr",
                "operation": expected_row["operation"],
                "comparison_kind": expected_row["comparison_kind"],
                "command": command,
                "source_commit": commit,
                "source_branch": branch,
                "generated_at_utc": now,
                "platform": platform_name,
                "compiler": "not_applicable",
                "configuration": configuration,
                "support_tier": "local_only",
                "expected_result_kind": expected_row["expected_result_kind"],
                "expected_result": expected_row["expected_result"],
                "observed_result": observations[oracle_row_id],
                "tolerance_kind": expected_row["tolerance_kind"],
                "tolerance_value": expected_row["tolerance_value"],
                "comparison_status": status,
                "failure_class": failure_class,
                "skip_or_defer_reason": "",
                "claim_scope": expected_row["claim_scope"],
                "non_claims": expected_row["non_claims"],
            }
        )
    return rows


def build_report_rows(
    oracle_rows: list[dict[str, str]], command: str, oracle_path: Path
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for oracle in oracle_rows:
        rows.append(
            {
                "report_row_id": f"corpus_oracle_{oracle['oracle_row_id']}_v1",
                "report_family": "oracle",
                "row_kind": "oracle_comparison",
                "row_subject": oracle["oracle_row_id"],
                "artifact_path": str(oracle_path),
                "generator_command": command,
                "source_commit": oracle["source_commit"],
                "source_branch": oracle["source_branch"],
                "generated_at_utc": oracle["generated_at_utc"],
                "platform": oracle["platform"],
                "compiler": oracle["compiler"],
                "configuration": oracle["configuration"],
                "support_tier": oracle["support_tier"],
                "status": oracle["comparison_status"],
                "status_reason": oracle["failure_class"],
                "row_meaning": "Fixture-local corpus oracle comparison row.",
                "claim_scope": oracle["claim_scope"],
                "non_claims": oracle["non_claims"],
                "freshness_status": "fresh",
                "freshness_reason": "generated by maintained command in current worktree",
            }
        )
    return rows


def build_skip_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for optional in load_optional_rows(root):
        state = optional["availability_state"]
        if state == "available":
            continue
        status = "defer" if state == "deferred" else "skip"
        failure_class = (
            "defer_not_implemented" if state == "deferred" else "skip_optional_unavailable"
        )
        rows.append(
            {
                "optional_data_key": optional["optional_data_key"],
                "availability_state": state,
                "status": status,
                "failure_class": failure_class,
                "skip_reason": optional["skip_reason"],
                "defer_reason": optional["defer_reason"],
                "fixture_keys": optional["fixture_keys"],
                "validation_command": optional["validation_command"],
                "skip_interpretation": optional["skip_interpretation"],
                "claim_boundary": optional["claim_boundary"],
            }
        )
    return rows


def build_skip_report_rows(
    skip_rows: list[dict[str, str]],
    command: str,
    skip_path: Path,
    oracle_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    if oracle_rows:
        source = oracle_rows[0]
        source_commit = source["source_commit"]
        source_branch = source["source_branch"]
        generated_at_utc = source["generated_at_utc"]
        platform_name = source["platform"]
        compiler = source["compiler"]
    else:
        source_commit = run_text(["git", "rev-parse", "HEAD"])
        source_branch = run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        generated_at_utc = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
        platform_name = f"{platform.system().lower()}-{platform.machine().lower()}"
        compiler = "not_applicable"
    rows: list[dict[str, str]] = []
    for skip in skip_rows:
        rows.append(
            {
                "report_row_id": f"corpus_optional_{skip['optional_data_key']}_v1",
                "report_family": "corpus_fixture",
                "row_kind": skip["status"],
                "row_subject": skip["optional_data_key"],
                "artifact_path": str(skip_path),
                "generator_command": command,
                "source_commit": source_commit,
                "source_branch": source_branch,
                "generated_at_utc": generated_at_utc,
                "platform": platform_name,
                "compiler": compiler,
                "configuration": "optional_data=disabled",
                "support_tier": "optional_data",
                "status": skip["status"],
                "status_reason": skip["failure_class"],
                "row_meaning": skip["skip_interpretation"],
                "claim_scope": "Optional-data policy evidence only.",
                "non_claims": skip["claim_boundary"],
                "freshness_status": "fresh",
                "freshness_reason": "generated by maintained command in current worktree",
            }
        )
    return rows


def write_manifest(path: Path, command: str, oracle_rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = oracle_rows[0]
    path.write_text(
        "\n".join(
            [
                "corpus-oracle-report",
                f"generated_at_utc={first['generated_at_utc']}",
                f"source_commit={first['source_commit']}",
                f"source_branch={first['source_branch']}",
                f"platform={first['platform']}",
                f"compiler={first['compiler']}",
                f"configuration={first['configuration']}",
                f"command={command}",
                f"fixture_key={FIXTURE_KEY}",
                "support_tier=local_only",
                "claim_boundary=fixture-local corpus/oracle evidence only",
                "non_claims=no broad QR correctness; no SuiteSparse parity; "
                "no broad corpus completeness; no performance or state-of-the-art claim",
                "",
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="tests/corpus", type=Path, help="corpus root")
    parser.add_argument("--oracle-dir", default="build/corpus/oracle", type=Path)
    parser.add_argument("--report-dir", default="build/corpus-reports", type=Path)
    args = parser.parse_args()

    command = "python3 scripts/run_corpus_oracle.py"
    oracle_rows = build_oracle_rows(args.root, command)
    skip_rows = build_skip_rows(args.root)
    oracle_path = args.oracle_dir / f"{FIXTURE_KEY}.oracle.tsv"
    report_path = args.report_dir / "index.tsv"
    manifest_path = args.report_dir / "manifest.txt"
    skip_path = args.report_dir / "skips.tsv"
    write_tsv(oracle_path, ORACLE_FIELDS, oracle_rows)
    report_rows = build_report_rows(oracle_rows, command, oracle_path)
    report_rows.extend(build_skip_report_rows(skip_rows, command, skip_path, oracle_rows))
    write_tsv(report_path, REPORT_FIELDS, report_rows)
    write_tsv(skip_path, SKIP_FIELDS, skip_rows)
    write_manifest(manifest_path, command, oracle_rows)
    print(f"corpus-oracle: wrote {oracle_path}")
    print(f"corpus-oracle: wrote {report_path}")
    print(f"corpus-oracle: wrote {skip_path}")
    print(f"corpus-oracle: wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
