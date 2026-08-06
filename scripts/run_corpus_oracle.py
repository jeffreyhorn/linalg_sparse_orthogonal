#!/usr/bin/env python3
"""Run the maintained corpus/oracle lane and emit local report rows."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "tests" / "corpus"
DEFAULT_ORACLE_DIR = REPO_ROOT / "build" / "corpus" / "oracle"
DEFAULT_REPORT_DIR = REPO_ROOT / "build" / "corpus-reports"
DEFAULT_SOLVER_LIBRARY = REPO_ROOT / "build" / "libsparse_lu_ortho.a"
FIXTURE_KEY = "qr_rank_deficient_6x4_nullspace_v1"
GENERATOR_KEY = "qr_rank_deficient_6x4_nullspace_generator_v1"
FIRST_LANE_ORACLE_ROW_IDS = {
    f"{FIXTURE_KEY}_rank",
    f"{FIXTURE_KEY}_nullity",
    f"{FIXTURE_KEY}_projector_residual",
}
SOLVER_QR_ORACLE_ROW_IDS = {
    f"{FIXTURE_KEY}_qr_rank": f"{FIXTURE_KEY}_rank",
    f"{FIXTURE_KEY}_qr_nullity": f"{FIXTURE_KEY}_nullity",
    f"{FIXTURE_KEY}_qr_nullspace_residual": f"{FIXTURE_KEY}_projector_residual",
}
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


def utc_timestamp() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def current_source_branch() -> str:
    branch = run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return "detached" if branch == "HEAD" else branch


def load_expected_rows(root: Path) -> dict[str, dict[str, str]]:
    path = root / "expected" / f"{FIXTURE_KEY}.tsv"
    rows = read_tsv(path)
    expected_by_id: dict[str, dict[str, str]] = {}
    for line, row in enumerate(rows, start=2):
        oracle_row_id = row["oracle_row_id"]
        if oracle_row_id in expected_by_id:
            raise CorpusValidationError(
                f"{path}:{line}: duplicate oracle_row_id {oracle_row_id!r}"
            )
        if oracle_row_id in FIRST_LANE_ORACLE_ROW_IDS and row["status"] != "ready_for_oracle":
            raise CorpusValidationError(
                f"{path}:{line}: first-lane oracle row {oracle_row_id!r} "
                f"must have status 'ready_for_oracle', got {row['status']!r}"
            )
        expected_by_id[oracle_row_id] = row
    missing = sorted(FIRST_LANE_ORACLE_ROW_IDS - set(expected_by_id))
    if missing:
        raise CorpusValidationError(
            f"{path}: missing first-lane expected oracle rows: {', '.join(missing)}"
        )
    return expected_by_id


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


def c_literal_for_entries(entries: list[tuple[int, int, float]]) -> str:
    lines = ["    {" + f"{row}, {col}, {value:.17g}" + "}" for row, col, value in entries]
    return ",\n".join(lines)


def qr_probe_source(entries: list[tuple[int, int, float]], rows: int, cols: int) -> str:
    return f"""#include \"sparse_matrix.h\"
#include \"sparse_qr.h\"
#include \"sparse_types.h\"
#include <math.h>
#include <stdio.h>

typedef struct Entry {{
    idx_t row;
    idx_t col;
    double value;
}} Entry;

static const Entry entries[] = {{
{c_literal_for_entries(entries)}
}};

int main(void) {{
    SparseMatrix *A = sparse_create({rows}, {cols});
    if (!A) {{
        fprintf(stderr, \"sparse_create failed\\n\");
        return 2;
    }}
    const size_t nnz = sizeof(entries) / sizeof(entries[0]);
    for (size_t k = 0; k < nnz; ++k) {{
        if (sparse_insert(A, entries[k].row, entries[k].col, entries[k].value) != SPARSE_OK) {{
            fprintf(stderr, \"sparse_insert failed at %zu\\n\", k);
            sparse_free(A);
            return 3;
        }}
    }}

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    if (err != SPARSE_OK) {{
        fprintf(stderr, \"sparse_qr_factor failed: %d\\n\", (int)err);
        sparse_free(A);
        return 4;
    }}

    idx_t rank = sparse_qr_rank(&qr, 0.0);
    idx_t nullity = -1;
    err = sparse_qr_nullspace(&qr, 0.0, NULL, &nullity);
    if (err != SPARSE_OK) {{
        fprintf(stderr, \"sparse_qr_nullspace query failed: %d\\n\", (int)err);
        sparse_qr_free(&qr);
        sparse_free(A);
        return 5;
    }}
    if (nullity != 1) {{
        printf(\"rank=%d\\n\", (int)rank);
        printf(\"nullity=%d\\n\", (int)nullity);
        printf(\"normalized_residual=inf\\n\");
        sparse_qr_free(&qr);
        sparse_free(A);
        return 0;
    }}

    double basis[{cols}] = {{0.0}};
    err = sparse_qr_nullspace(&qr, 0.0, basis, &nullity);
    if (err != SPARSE_OK) {{
        fprintf(stderr, \"sparse_qr_nullspace basis failed: %d\\n\", (int)err);
        sparse_qr_free(&qr);
        sparse_free(A);
        return 6;
    }}

    double residual_sq = 0.0;
    for (idx_t row = 0; row < {rows}; ++row) {{
        double accum = 0.0;
        for (size_t k = 0; k < nnz; ++k) {{
            if (entries[k].row == row)
                accum += entries[k].value * basis[entries[k].col];
        }}
        residual_sq += accum * accum;
    }}
    double norm_sq = 0.0;
    for (idx_t col = 0; col < {cols}; ++col)
        norm_sq += basis[col] * basis[col];
    double normalized_residual = norm_sq > 0.0 ? sqrt(residual_sq) / sqrt(norm_sq) : INFINITY;

    printf(\"rank=%d\\n\", (int)rank);
    printf(\"nullity=%d\\n\", (int)nullity);
    printf(\"normalized_residual=%.17g\\n\", normalized_residual);

    sparse_qr_free(&qr);
    sparse_free(A);
    return 0;
}}
"""


def parse_probe_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    required = {"rank", "nullity", "normalized_residual"}
    missing = sorted(required - set(parsed))
    if missing:
        raise CorpusValidationError(f"solver QR probe did not emit required fields: {missing}")
    return parsed


def compiler_argv() -> list[str]:
    argv = shlex.split(os.environ.get("CC", "cc"))
    return argv if argv else ["cc"]


def compiler_identity(cc_argv: list[str]) -> str:
    try:
        completed = subprocess.run(
            [*cc_argv, "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        return shlex.join(cc_argv)
    first = completed.stdout.splitlines()[0].strip() if completed.stdout else shlex.join(cc_argv)
    return first.replace("\t", " ")


def run_solver_qr_probe(
    entries: list[tuple[int, int, float]], rows: int, cols: int, library: Path
) -> tuple[dict[str, str], str]:
    if not library.is_file():
        raise CorpusValidationError(
            f"solver QR probe requires built static library at {library}; run make first"
        )
    cc = compiler_argv()
    with tempfile.TemporaryDirectory(prefix="sparse_qr_probe.") as tmp:
        tmpdir = Path(tmp)
        source = tmpdir / "qr_probe.c"
        executable = tmpdir / "qr_probe"
        source.write_text(qr_probe_source(entries, rows, cols))
        compile_cmd = [
            *cc,
            "-std=c99",
            f"-I{REPO_ROOT / 'include'}",
            f"-I{REPO_ROOT / 'build' / 'include'}",
            str(source),
            str(library),
            "-lm",
            "-o",
            str(executable),
        ]
        compiled = subprocess.run(
            compile_cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if compiled.returncode != 0:
            raise CorpusValidationError(
                "solver QR probe compile failed:\n"
                + shlex.join(compile_cmd)
                + "\n"
                + compiled.stdout
            )
        completed = subprocess.run(
            [str(executable)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise CorpusValidationError(
                f"solver QR probe failed with exit {completed.returncode}:\n{completed.stdout}"
            )
        return parse_probe_output(completed.stdout), compiler_identity(cc)


def compare(expected: dict[str, str], observed: str) -> tuple[str, str]:
    kind = expected["comparison_kind"]
    tolerance = float(expected["tolerance_value"])
    if kind in {"rank", "nullity"}:
        if expected["tolerance_kind"] != "exact":
            raise CorpusValidationError(
                f"{expected['oracle_row_id']}: {kind} comparison requires tolerance_kind='exact'"
            )
        passed = int(observed) == int(expected["expected_result"])
    elif kind == "residual_norm":
        if expected["tolerance_kind"] != "absolute":
            raise CorpusValidationError(
                f"{expected['oracle_row_id']}: residual_norm comparison requires "
                "tolerance_kind='absolute'"
            )
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
    now = utc_timestamp()
    commit = run_text(["git", "rev-parse", "HEAD"])
    branch = current_source_branch()
    platform_name = f"{platform.system().lower()}-{platform.machine().lower()}"
    configuration = (
        "build_profile=static_default;optional_data_policy=disabled;generated_reference=python;"
        f"structure_hash={structure_hash};value_hash={value_hash}"
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
        if oracle_row_id not in expected:
            raise CorpusValidationError(
                f"missing expected result for first-lane oracle row {oracle_row_id!r}"
            )
        expected_row = expected[oracle_row_id]
        status, failure_class = compare(expected_row, observations[oracle_row_id])
        rows.append(
            {
                "oracle_row_id": oracle_row_id,
                "fixture_key": FIXTURE_KEY,
                "solver_family": "unknown",
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


def build_solver_qr_oracle_rows(
    root: Path, command: str, library: Path, *, validate_root: bool = True
) -> list[dict[str, str]]:
    if validate_root:
        validate(root)
    fixture = GENERATED_FIXTURES[GENERATOR_KEY]
    entries = fixture["entries"]()
    structure_hash = sha256_text(
        canonical_structure_text(fixture["rows"], fixture["cols"], entries)
    )
    value_hash = sha256_text(canonical_value_text(fixture["rows"], fixture["cols"], entries))
    expected = load_expected_rows(root)
    observations, compiler = run_solver_qr_probe(entries, fixture["rows"], fixture["cols"], library)
    now = utc_timestamp()
    commit = run_text(["git", "rev-parse", "HEAD"])
    branch = current_source_branch()
    platform_name = f"{platform.system().lower()}-{platform.machine().lower()}"
    configuration = (
        "build_profile=static_default;optional_data_policy=disabled;"
        "proof_owner=runtime_qr_probe;"
        f"structure_hash={structure_hash};value_hash={value_hash};qr_tolerance=1e-10"
    )
    observation_by_solver_id = {
        f"{FIXTURE_KEY}_qr_rank": observations["rank"],
        f"{FIXTURE_KEY}_qr_nullity": observations["nullity"],
        f"{FIXTURE_KEY}_qr_nullspace_residual": observations["normalized_residual"],
    }
    rows: list[dict[str, str]] = []
    for oracle_row_id in sorted(observation_by_solver_id):
        expected_row_id = SOLVER_QR_ORACLE_ROW_IDS[oracle_row_id]
        if expected_row_id not in expected:
            raise CorpusValidationError(
                f"missing expected result for solver QR oracle row {expected_row_id!r}"
            )
        expected_row = expected[expected_row_id]
        status, failure_class = compare(expected_row, observation_by_solver_id[oracle_row_id])
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
                "compiler": compiler,
                "configuration": configuration,
                "support_tier": "local_only",
                "expected_result_kind": expected_row["expected_result_kind"],
                "expected_result": expected_row["expected_result"],
                "observed_result": observation_by_solver_id[oracle_row_id],
                "tolerance_kind": expected_row["tolerance_kind"],
                "tolerance_value": expected_row["tolerance_value"],
                "comparison_status": status,
                "failure_class": failure_class,
                "skip_or_defer_reason": "",
                "claim_scope": (
                    "Fixture-local solver-backed QR rank/nullity/nullspace residual evidence."
                ),
                "non_claims": (
                    "no broad QR correctness; no raw-basis parity; "
                    "no global rank-threshold policy; no broad rank-deficient solve; "
                    "no minimum-norm or least-squares claim; no SuiteSparse parity; "
                    "no external-library parity; no platform parity; no performance or "
                    "state-of-the-art claim"
                ),
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
        source_branch = current_source_branch()
        generated_at_utc = utc_timestamp()
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
                "configuration": f"optional_data={skip['availability_state']}",
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
    solver_families = ",".join(sorted({row["solver_family"] for row in oracle_rows}))
    solver_qr_row_count = sum(1 for row in oracle_rows if row["solver_family"] == "qr")
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
                f"oracle_row_count={len(oracle_rows)}",
                f"solver_families={solver_families}",
                f"solver_qr_row_count={solver_qr_row_count}",
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
    parser.add_argument("--root", default=DEFAULT_CORPUS_ROOT, type=Path, help="corpus root")
    parser.add_argument("--oracle-dir", default=DEFAULT_ORACLE_DIR, type=Path)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR, type=Path)
    parser.add_argument(
        "--include-solver-qr",
        action="store_true",
        help="append solver-backed QR oracle rows from a temporary static-library probe",
    )
    parser.add_argument(
        "--solver-library",
        default=DEFAULT_SOLVER_LIBRARY,
        type=Path,
        help="static library used by --include-solver-qr",
    )
    args = parser.parse_args()

    command = shlex.join(sys.argv)
    oracle_rows = build_oracle_rows(args.root, command)
    if args.include_solver_qr:
        oracle_rows.extend(
            build_solver_qr_oracle_rows(
                args.root, command, args.solver_library, validate_root=False
            )
        )
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
