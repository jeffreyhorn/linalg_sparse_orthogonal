#!/usr/bin/env python3
"""Run narrow local external-comparison harnesses."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from validate_corpus_schema import GENERATED_FIXTURES


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "comparison" / "qr_minnorm"
DEFAULT_LIBRARY = REPO_ROOT / "build" / "libsparse_lu_ortho.a"

TARGET = "qr-minnorm"
FIXTURE_KEY = "qr_underdetermined_minnorm_2x4"
GENERATOR_KEY = "qr_underdetermined_minnorm_2x4_generator_v1"
RHS = [1.0, 1.0]
EXPECTED_SOLUTION = [0.5, 0.5, 0.5, 0.5]
EXPECTED_SOLUTION_NORM = 1.0
RESIDUAL_TOLERANCE = 1e-10
SOLUTION_TOLERANCE = 1e-10

MANIFEST_FIELDS = [
    "key",
    "value",
]

PROJECT_OBSERVATION_FIELDS = [
    "fixture_key",
    "metric",
    "value",
    "status",
    "status_reason",
]

BASELINE_OBSERVATION_FIELDS = PROJECT_OBSERVATION_FIELDS

DEPENDENCY_STATUS_FIELDS = [
    "dependency",
    "status",
    "status_reason",
    "required",
    "caveat",
]

STUDY_FIELDS = [
    "comparison_row_id",
    "report_family",
    "subfamily",
    "row_kind",
    "fixture_key",
    "operation",
    "metric",
    "baseline_name",
    "baseline_type",
    "baseline_version",
    "baseline_command",
    "baseline_python_executable",
    "baseline_python_version",
    "project_name",
    "project_version",
    "project_command",
    "source_commit",
    "source_branch",
    "worktree_state",
    "platform",
    "compiler",
    "configuration",
    "expected_value",
    "project_value",
    "baseline_value",
    "delta_value",
    "tolerance_kind",
    "tolerance_value",
    "status",
    "status_reason",
    "caveat",
    "artifact_path",
    "generated_at_utc",
    "support_tier",
    "claim_scope",
    "non_claims",
]

NON_CLAIMS = (
    "no broad QR parity;no NumPy parity;no SciPy parity;"
    "no external-library ecosystem parity;no performance claim;"
    "no package-manager proof;no hosted CI proof;no shared-library ABI proof;"
    "no state-of-the-art claim"
)


class ComparisonError(RuntimeError):
    """Raised when the comparison harness cannot produce valid output."""

    def __init__(self, failure_class: str, message: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class


def run_text(argv: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def run_capture(
    argv: list[str],
    *,
    cwd: Path | None = None,
    failure_class: str = "project_probe_failed",
) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise ComparisonError(
            failure_class,
            f"command failed with exit {completed.returncode}: {shlex.join(argv)}\n"
            f"{completed.stdout}",
        )
    return completed.stdout


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def current_branch(root: Path) -> str:
    try:
        return run_text(["git", "branch", "--show-current"], cwd=root) or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def current_commit(root: Path) -> str:
    try:
        return run_text(["git", "rev-parse", "HEAD"], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ComparisonError("missing_source_provenance", "unable to resolve source commit") from exc


def worktree_state(root: Path) -> str:
    try:
        return "dirty" if run_text(["git", "status", "--porcelain"], cwd=root) else "clean"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def project_version(root: Path) -> str:
    version_path = root / "VERSION"
    if not version_path.is_file():
        return "unknown"
    return version_path.read_text(encoding="utf-8").strip()


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


def ensure_library(root: Path, library: Path) -> None:
    if library.is_file():
        return
    target = str(library.relative_to(root)) if library.is_relative_to(root) else str(library)
    completed = subprocess.run(
        ["make", target],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0 or not library.is_file():
        raise ComparisonError(
            "project_build_failed",
            f"failed to build required static library {library}:\n{completed.stdout}",
        )


def c_literal_for_entries(entries: list[tuple[int, int, float]]) -> str:
    return "\n".join(
        f"    {{{row}, {col}, {value:.17g}}}," for row, col, value in entries
    )


def c_literal_for_values(values: list[float]) -> str:
    return ", ".join(f"{value:.17g}" for value in values)


def project_probe_source(entries: list[tuple[int, int, float]], rows: int, cols: int) -> str:
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

static const double rhs[{rows}] = {{{c_literal_for_values(RHS)}}};

static const char *status_name(sparse_err_t err) {{
    switch (err) {{
    case SPARSE_OK:
        return \"SPARSE_SUCCESS\";
    case SPARSE_ERR_NULL:
        return \"SPARSE_ERR_NULL\";
    case SPARSE_ERR_ALLOC:
        return \"SPARSE_ERR_ALLOC\";
    case SPARSE_ERR_BOUNDS:
        return \"SPARSE_ERR_BOUNDS\";
    case SPARSE_ERR_SINGULAR:
        return \"SPARSE_ERR_SINGULAR\";
    case SPARSE_ERR_SHAPE:
        return \"SPARSE_ERR_SHAPE\";
    case SPARSE_ERR_BADARG:
        return \"SPARSE_ERR_BADARG\";
    case SPARSE_ERR_NOT_CONVERGED:
        return \"SPARSE_ERR_NOT_CONVERGED\";
    case SPARSE_ERR_NUMERIC:
        return \"SPARSE_ERR_NUMERIC\";
    default:
        return \"SPARSE_ERR_OTHER\";
    }}
}}

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

    double x[{cols}];
    for (idx_t col = 0; col < {cols}; ++col)
        x[col] = 0.0;
    sparse_err_t err = sparse_qr_solve_minnorm(A, rhs, x, NULL);

    double residual_sq = 0.0;
    double norm_sq = 0.0;
    if (err == SPARSE_OK) {{
        for (idx_t row = 0; row < {rows}; ++row) {{
            double accum = 0.0;
            for (size_t k = 0; k < nnz; ++k) {{
                if (entries[k].row == row)
                    accum += entries[k].value * x[entries[k].col];
            }}
            double diff = accum - rhs[row];
            residual_sq += diff * diff;
        }}
        for (idx_t col = 0; col < {cols}; ++col)
            norm_sq += x[col] * x[col];
    }}

    printf(\"status=%s\\n\", status_name(err));
    printf(\"residual_norm=%.17g\\n\", err == SPARSE_OK ? sqrt(residual_sq) : INFINITY);
    printf(\"solution_norm=%.17g\\n\", err == SPARSE_OK ? sqrt(norm_sq) : INFINITY);
    printf(\"solution_values=\");
    for (idx_t col = 0; col < {cols}; ++col) {{
        if (col > 0)
            printf(\",\");
        printf(\"%.17g\", x[col]);
    }}
    printf(\"\\n\");

    sparse_free(A);
    return 0;
}}
"""


def parse_key_values(output: str, required: set[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    missing = sorted(required - set(parsed))
    if missing:
        raise ComparisonError(
            "project_probe_failed",
            f"project probe did not emit required fields: {', '.join(missing)}",
        )
    return parsed


def run_project_probe(root: Path, library: Path, keep_temp: bool) -> tuple[dict[str, str], str]:
    ensure_library(root, library)
    fixture = GENERATED_FIXTURES[GENERATOR_KEY]
    entries = fixture["entries"]()
    rows = int(fixture["rows"])
    cols = int(fixture["cols"])
    cc = compiler_argv()
    compiler = compiler_identity(cc)

    temp_dir = Path(tempfile.mkdtemp(prefix="sparse-comparison-"))
    try:
        source = temp_dir / "qr_minnorm_probe.c"
        binary = temp_dir / "qr_minnorm_probe"
        source.write_text(project_probe_source(entries, rows, cols), encoding="utf-8")
        compile_cmd = [
            *cc,
            "-std=c99",
            "-I",
            str(root / "include"),
            "-I",
            str(root / "build" / "include"),
            str(source),
            str(library),
            "-lm",
            "-o",
            str(binary),
        ]
        completed = subprocess.run(
            compile_cmd,
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise ComparisonError(
                "project_build_failed",
                f"failed to compile project comparison probe:\n{completed.stdout}",
            )
        output = run_capture([str(binary)], cwd=root)
        parsed = parse_key_values(
            output,
            {"status", "residual_norm", "solution_norm", "solution_values"},
        )
        parsed["project_probe_command"] = shlex.join(compile_cmd) + " && " + str(binary)
        return parsed, compiler
    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def python_version() -> str:
    return " ".join(sys.version.split())


def parse_vector(text: str) -> list[float]:
    if not text:
        raise ComparisonError("project_probe_failed", "empty solution_values field")
    try:
        return [float(value) for value in text.split(",")]
    except ValueError as exc:
        raise ComparisonError("project_probe_failed", f"malformed vector {text!r}") from exc


def parse_baseline_vector(text: str) -> list[float]:
    try:
        return [float(value) for value in text.split(",")]
    except ValueError as exc:
        raise ComparisonError("baseline_malformed_output", f"malformed baseline vector {text!r}") from exc


def run_baseline_reference(root: Path) -> dict[str, str]:
    helper = root / "tests" / "qr_external_dense_reference.py"
    if not helper.is_file():
        raise ComparisonError(
            "missing_baseline_helper",
            f"selected baseline helper is missing: {helper}",
        )

    command = [sys.executable, str(helper), FIXTURE_KEY]
    output = run_capture(command, cwd=root, failure_class="baseline_command_failed")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ComparisonError("baseline_malformed_output", "baseline emitted no output")

    header = lines[0].split()
    if len(header) != 2 or header[0] != "OK":
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline first line must be 'OK 6', got {lines[0]!r}",
        )
    try:
        value_count = int(header[1])
    except ValueError as exc:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline value count is not an integer: {header[1]!r}",
        ) from exc
    if value_count != 6:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline value count must be 6 for {FIXTURE_KEY}, got {value_count}",
        )

    value_lines = lines[1:]
    if len(value_lines) != value_count:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline emitted {len(value_lines)} values, expected {value_count}",
        )
    try:
        values = [float(value) for value in value_lines]
    except ValueError as exc:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline emitted non-numeric values: {', '.join(value_lines)}",
        ) from exc

    solution = values[:4]
    residual = values[4]
    solution_norm = values[5]
    return {
        "status": "success",
        "solution_values": ",".join(f"{value:.17g}" for value in solution),
        "residual_norm": f"{residual:.17g}",
        "solution_norm": f"{solution_norm:.17g}",
        "baseline_command": shlex.join(command),
        "baseline_helper_path": str(helper.relative_to(root)),
        "baseline_python_executable": sys.executable,
        "baseline_python_version": python_version(),
    }


def project_observation_rows(observations: dict[str, str]) -> list[dict[str, str]]:
    values = parse_vector(observations["solution_values"])
    if len(values) != len(EXPECTED_SOLUTION):
        raise ComparisonError(
            "project_probe_failed",
            f"project solution has {len(values)} values, expected {len(EXPECTED_SOLUTION)}",
        )
    max_abs_delta = max(
        abs(expected - observed) for expected, observed in zip(EXPECTED_SOLUTION, values)
    )
    rows = [
        {
            "fixture_key": FIXTURE_KEY,
            "metric": "project_status",
            "value": observations["status"],
            "status": "pass" if observations["status"] == "SPARSE_SUCCESS" else "fail",
            "status_reason": "project_status_match"
            if observations["status"] == "SPARSE_SUCCESS"
            else "project_status_mismatch",
        },
        {
            "fixture_key": FIXTURE_KEY,
            "metric": "residual_norm",
            "value": observations["residual_norm"],
            "status": "pass"
            if float(observations["residual_norm"]) <= RESIDUAL_TOLERANCE
            else "fail",
            "status_reason": "project_residual_within_tolerance"
            if float(observations["residual_norm"]) <= RESIDUAL_TOLERANCE
            else "project_residual_tolerance_miss",
        },
        {
            "fixture_key": FIXTURE_KEY,
            "metric": "solution_norm",
            "value": observations["solution_norm"],
            "status": "pass"
            if abs(float(observations["solution_norm"]) - EXPECTED_SOLUTION_NORM)
            <= SOLUTION_TOLERANCE
            else "fail",
            "status_reason": "project_solution_norm_within_tolerance"
            if abs(float(observations["solution_norm"]) - EXPECTED_SOLUTION_NORM)
            <= SOLUTION_TOLERANCE
            else "project_solution_norm_tolerance_miss",
        },
        {
            "fixture_key": FIXTURE_KEY,
            "metric": "solution_values",
            "value": observations["solution_values"],
            "status": "pass" if max_abs_delta <= SOLUTION_TOLERANCE else "fail",
            "status_reason": "project_solution_values_within_tolerance"
            if max_abs_delta <= SOLUTION_TOLERANCE
            else "project_solution_values_tolerance_miss",
        },
    ]
    return rows


def baseline_observation_rows(observations: dict[str, str]) -> list[dict[str, str]]:
    values = parse_baseline_vector(observations["solution_values"])
    if len(values) != len(EXPECTED_SOLUTION):
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline solution has {len(values)} values, expected {len(EXPECTED_SOLUTION)}",
        )
    max_abs_delta = max(
        abs(expected - observed) for expected, observed in zip(EXPECTED_SOLUTION, values)
    )
    residual = float(observations["residual_norm"])
    solution_norm = float(observations["solution_norm"])
    rows = [
        {
            "fixture_key": FIXTURE_KEY,
            "metric": "baseline_status",
            "value": observations["status"],
            "status": "pass" if observations["status"] == "success" else "fail",
            "status_reason": "baseline_status_success"
            if observations["status"] == "success"
            else "baseline_status_mismatch",
        },
        {
            "fixture_key": FIXTURE_KEY,
            "metric": "baseline_residual_norm",
            "value": observations["residual_norm"],
            "status": "pass" if residual <= RESIDUAL_TOLERANCE else "fail",
            "status_reason": "baseline_residual_within_tolerance"
            if residual <= RESIDUAL_TOLERANCE
            else "baseline_residual_tolerance_miss",
        },
        {
            "fixture_key": FIXTURE_KEY,
            "metric": "baseline_solution_norm",
            "value": observations["solution_norm"],
            "status": "pass"
            if abs(solution_norm - EXPECTED_SOLUTION_NORM) <= SOLUTION_TOLERANCE
            else "fail",
            "status_reason": "baseline_solution_norm_within_tolerance"
            if abs(solution_norm - EXPECTED_SOLUTION_NORM) <= SOLUTION_TOLERANCE
            else "baseline_solution_norm_tolerance_miss",
        },
        {
            "fixture_key": FIXTURE_KEY,
            "metric": "baseline_solution_values",
            "value": observations["solution_values"],
            "status": "pass" if max_abs_delta <= SOLUTION_TOLERANCE else "fail",
            "status_reason": "baseline_solution_values_within_tolerance"
            if max_abs_delta <= SOLUTION_TOLERANCE
            else "baseline_solution_values_tolerance_miss",
        },
    ]
    return rows


def dependency_status_rows(root: Path) -> list[dict[str, str]]:
    helper = root / "tests" / "qr_external_dense_reference.py"
    return [
        {
            "dependency": "python3",
            "status": "pass",
            "status_reason": "selected_interpreter_available",
            "required": "yes",
            "caveat": "current Python executable only; no package-manager inference",
        },
        {
            "dependency": "tests/qr_external_dense_reference.py",
            "status": "pass" if helper.is_file() else "error",
            "status_reason": "baseline_helper_available"
            if helper.is_file()
            else "baseline_helper_missing",
            "required": "yes",
            "caveat": "source-controlled dense reference helper; not an external package",
        },
        {
            "dependency": "numpy",
            "status": "defer",
            "status_reason": "optional_package_baseline_not_selected",
            "required": "no",
            "caveat": "deferred rows are not pass evidence",
        },
        {
            "dependency": "scipy",
            "status": "defer",
            "status_reason": "optional_package_baseline_not_selected",
            "required": "no",
            "caveat": "deferred rows are not pass evidence",
        },
    ]


def max_abs_delta(lhs: list[float], rhs: list[float], failure_class: str) -> float:
    if len(lhs) != len(rhs):
        raise ComparisonError(
            failure_class,
            f"vector lengths differ: {len(lhs)} vs {len(rhs)}",
        )
    return max(abs(left - right) for left, right in zip(lhs, rhs))


def format_float(value: float) -> str:
    return f"{value:.17g}"


def build_study_context(
    *,
    artifact_path: str,
    baseline_observations: dict[str, str],
    compiler: str,
    generated_at: str,
    manifest: dict[str, str],
    observations: dict[str, str],
) -> dict[str, str]:
    return {
        "report_family": "comparison",
        "subfamily": "qr_minnorm",
        "fixture_key": FIXTURE_KEY,
        "operation": "minnorm_solve",
        "baseline_name": manifest["baseline_name"],
        "baseline_type": manifest["baseline_type"],
        "baseline_version": manifest["baseline_version"],
        "baseline_command": baseline_observations["baseline_command"],
        "baseline_python_executable": baseline_observations["baseline_python_executable"],
        "baseline_python_version": baseline_observations["baseline_python_version"],
        "project_name": "sparse_lu_ortho",
        "project_version": manifest["project_version"],
        "project_command": observations["project_probe_command"],
        "source_commit": manifest["source_commit"],
        "source_branch": manifest["source_branch"],
        "worktree_state": manifest["worktree_state"],
        "platform": manifest["platform"],
        "compiler": compiler,
        "configuration": manifest["configuration"],
        "artifact_path": artifact_path,
        "generated_at_utc": generated_at,
        "support_tier": "local_only",
        "claim_scope": "fixture-local qr minimum-norm comparison only",
        "non_claims": NON_CLAIMS,
    }


def study_row(context: dict[str, str], **overrides: str) -> dict[str, str]:
    row = {field: "" for field in STUDY_FIELDS}
    row.update(context)
    row.update(overrides)
    return row


def comparison_study_rows(
    *,
    artifact_path: str,
    baseline_observations: dict[str, str],
    compiler: str,
    generated_at: str,
    manifest: dict[str, str],
    observations: dict[str, str],
) -> list[dict[str, str]]:
    context = build_study_context(
        artifact_path=artifact_path,
        baseline_observations=baseline_observations,
        compiler=compiler,
        generated_at=generated_at,
        manifest=manifest,
        observations=observations,
    )
    project_solution = parse_vector(observations["solution_values"])
    baseline_solution = parse_baseline_vector(baseline_observations["solution_values"])
    expected_solution = ",".join(format_float(value) for value in EXPECTED_SOLUTION)
    solution_delta = max_abs_delta(
        project_solution,
        baseline_solution,
        "metric_comparison_malformed",
    )
    project_residual = float(observations["residual_norm"])
    baseline_residual = float(baseline_observations["residual_norm"])
    residual_delta = abs(project_residual - baseline_residual)
    project_solution_norm = float(observations["solution_norm"])
    baseline_solution_norm = float(baseline_observations["solution_norm"])
    solution_norm_delta = abs(project_solution_norm - baseline_solution_norm)

    caveat = (
        "local generated artifact; dirty worktree allowed only as explicit provenance"
        if manifest["worktree_state"] == "dirty"
        else "local generated artifact"
    )

    rows = [
        study_row(
            context,
            comparison_row_id=f"comparison_{FIXTURE_KEY}_project_status_v1",
            row_kind="metric_comparison",
            metric="project_status",
            expected_value="SPARSE_SUCCESS",
            project_value=observations["status"],
            baseline_value="",
            delta_value="",
            tolerance_kind="status_only",
            tolerance_value="",
            status="pass" if observations["status"] == "SPARSE_SUCCESS" else "fail",
            status_reason="project_status_match"
            if observations["status"] == "SPARSE_SUCCESS"
            else "project_status_mismatch",
            caveat=caveat,
        ),
        study_row(
            context,
            comparison_row_id=f"comparison_{FIXTURE_KEY}_baseline_status_v1",
            row_kind="dependency_status",
            metric="baseline_status",
            expected_value="success",
            project_value="",
            baseline_value=baseline_observations["status"],
            delta_value="",
            tolerance_kind="status_only",
            tolerance_value="",
            status="pass" if baseline_observations["status"] == "success" else "fail",
            status_reason="baseline_status_success"
            if baseline_observations["status"] == "success"
            else "baseline_status_mismatch",
            caveat="required source-controlled dense-reference baseline; not an external package",
        ),
        study_row(
            context,
            comparison_row_id=f"comparison_{FIXTURE_KEY}_residual_norm_v1",
            row_kind="metric_comparison",
            metric="residual_norm",
            expected_value=f"<={format_float(RESIDUAL_TOLERANCE)}",
            project_value=observations["residual_norm"],
            baseline_value=baseline_observations["residual_norm"],
            delta_value=format_float(residual_delta),
            tolerance_kind="absolute",
            tolerance_value=format_float(RESIDUAL_TOLERANCE),
            status="pass" if residual_delta <= RESIDUAL_TOLERANCE else "fail",
            status_reason="project_baseline_residual_delta_within_tolerance"
            if residual_delta <= RESIDUAL_TOLERANCE
            else "project_baseline_residual_delta_tolerance_miss",
            caveat=caveat,
        ),
        study_row(
            context,
            comparison_row_id=f"comparison_{FIXTURE_KEY}_solution_norm_v1",
            row_kind="metric_comparison",
            metric="solution_norm",
            expected_value=format_float(EXPECTED_SOLUTION_NORM),
            project_value=observations["solution_norm"],
            baseline_value=baseline_observations["solution_norm"],
            delta_value=format_float(solution_norm_delta),
            tolerance_kind="absolute",
            tolerance_value=format_float(SOLUTION_TOLERANCE),
            status="pass" if solution_norm_delta <= SOLUTION_TOLERANCE else "fail",
            status_reason="project_baseline_solution_norm_delta_within_tolerance"
            if solution_norm_delta <= SOLUTION_TOLERANCE
            else "project_baseline_solution_norm_delta_tolerance_miss",
            caveat=caveat,
        ),
        study_row(
            context,
            comparison_row_id=f"comparison_{FIXTURE_KEY}_solution_values_v1",
            row_kind="metric_comparison",
            metric="solution_values",
            expected_value=expected_solution,
            project_value=observations["solution_values"],
            baseline_value=baseline_observations["solution_values"],
            delta_value=format_float(solution_delta),
            tolerance_kind="absolute_per_component",
            tolerance_value=format_float(SOLUTION_TOLERANCE),
            status="pass" if solution_delta <= SOLUTION_TOLERANCE else "fail",
            status_reason="project_baseline_solution_values_delta_within_tolerance"
            if solution_delta <= SOLUTION_TOLERANCE
            else "project_baseline_solution_values_delta_tolerance_miss",
            caveat=caveat,
        ),
        study_row(
            context,
            comparison_row_id=f"comparison_{FIXTURE_KEY}_project_vs_baseline_max_abs_delta_v1",
            row_kind="metric_comparison",
            metric="project_vs_baseline_max_abs_delta",
            expected_value=f"<={format_float(SOLUTION_TOLERANCE)}",
            project_value=observations["solution_values"],
            baseline_value=baseline_observations["solution_values"],
            delta_value=format_float(solution_delta),
            tolerance_kind="absolute",
            tolerance_value=format_float(SOLUTION_TOLERANCE),
            status="pass" if solution_delta <= SOLUTION_TOLERANCE else "fail",
            status_reason="project_baseline_max_abs_delta_within_tolerance"
            if solution_delta <= SOLUTION_TOLERANCE
            else "project_baseline_max_abs_delta_tolerance_miss",
            caveat=caveat,
        ),
    ]
    return rows


def validate_selected_study_rows(rows: list[dict[str, str]]) -> None:
    expected_ids = {
        f"comparison_{FIXTURE_KEY}_project_status_v1",
        f"comparison_{FIXTURE_KEY}_baseline_status_v1",
        f"comparison_{FIXTURE_KEY}_residual_norm_v1",
        f"comparison_{FIXTURE_KEY}_solution_norm_v1",
        f"comparison_{FIXTURE_KEY}_solution_values_v1",
        f"comparison_{FIXTURE_KEY}_project_vs_baseline_max_abs_delta_v1",
    }
    counts: dict[str, int] = {}
    for row in rows:
        row_id = row["comparison_row_id"]
        counts[row_id] = counts.get(row_id, 0) + 1
    missing = sorted(expected_ids - set(counts))
    duplicates = sorted(row_id for row_id, count in counts.items() if count > 1)
    if missing:
        raise ComparisonError("missing_selected_row", ", ".join(missing))
    if duplicates:
        raise ComparisonError("duplicate_selected_row", ", ".join(duplicates))
    failures = [
        f"{row['comparison_row_id']}={row['status_reason']}"
        for row in rows
        if row["comparison_row_id"] in expected_ids and row["status"] != "pass"
    ]
    if failures:
        raise ComparisonError("metric_tolerance_miss", ", ".join(failures))


def expected_study_row_ids() -> list[str]:
    return [
        f"comparison_{FIXTURE_KEY}_project_status_v1",
        f"comparison_{FIXTURE_KEY}_baseline_status_v1",
        f"comparison_{FIXTURE_KEY}_residual_norm_v1",
        f"comparison_{FIXTURE_KEY}_solution_norm_v1",
        f"comparison_{FIXTURE_KEY}_solution_values_v1",
        f"comparison_{FIXTURE_KEY}_project_vs_baseline_max_abs_delta_v1",
    ]


def assert_comparison_error(expected: str, fn) -> None:
    try:
        fn()
    except ComparisonError as exc:
        if exc.failure_class == expected:
            return
        raise AssertionError(f"expected {expected}, got {exc.failure_class}") from exc
    raise AssertionError(f"expected {expected}, got success")


def run_self_check(root: Path) -> int:
    passing_rows = [
        {"comparison_row_id": row_id, "status": "pass", "status_reason": "self_check"}
        for row_id in expected_study_row_ids()
    ]
    validate_selected_study_rows(passing_rows)
    assert_comparison_error(
        "missing_selected_row",
        lambda: validate_selected_study_rows(passing_rows[:-1]),
    )
    assert_comparison_error(
        "duplicate_selected_row",
        lambda: validate_selected_study_rows([*passing_rows, passing_rows[0]]),
    )
    failing_rows = [dict(row) for row in passing_rows]
    failing_rows[0]["status"] = "fail"
    failing_rows[0]["status_reason"] = "self_check_failure"
    assert_comparison_error(
        "metric_tolerance_miss",
        lambda: validate_selected_study_rows(failing_rows),
    )
    assert_comparison_error(
        "metric_comparison_malformed",
        lambda: max_abs_delta([1.0], [1.0, 2.0], "metric_comparison_malformed"),
    )
    assert_comparison_error(
        "project_probe_failed",
        lambda: project_observation_rows(
            {
                "status": "SPARSE_SUCCESS",
                "residual_norm": "0",
                "solution_norm": "1",
                "solution_values": "0.5,0.5,0.5",
            }
        ),
    )
    dependency_rows = dependency_status_rows(root)
    deferred = {
        row["dependency"]: row["status"]
        for row in dependency_rows
        if row["dependency"] in {"numpy", "scipy"}
    }
    if deferred != {"numpy": "defer", "scipy": "defer"}:
        raise ComparisonError("self_check_failed", "optional package rows are not deferred")
    print("external-comparison: self-check passed")
    return 0


def write_summary(path: Path, rows: list[dict[str, str]], manifest: dict[str, str]) -> None:
    pass_rows = [row for row in rows if row["status"] == "pass"]
    non_pass_rows = [row for row in rows if row["status"] != "pass"]
    metric_lines = "\n".join(
        f"| `{row['metric']}` | `{row['project_value']}` | `{row['baseline_value']}` | "
        f"`{row['delta_value']}` | `{row['status']}` |"
        for row in rows
    )
    path.write_text(
        "\n".join(
            [
                "# QR Minimum-Norm External Comparison Study",
                "",
                "## Scope",
                "",
                "This local generated study compares one fixture-local QR "
                "minimum-norm solve against the source-controlled dense "
                "reference helper.",
                "",
                "It does not claim broad QR parity, NumPy/SciPy parity, "
                "external-library ecosystem parity, package-manager support, "
                "performance superiority, hosted CI proof, shared-library ABI "
                "support, or state-of-the-art status.",
                "",
                "## Provenance",
                "",
                f"- Target: `{manifest['target']}`",
                f"- Fixture: `{manifest['fixture_key']}`",
                f"- Source commit: `{manifest['source_commit']}`",
                f"- Source branch: `{manifest['source_branch']}`",
                f"- Worktree state: `{manifest['worktree_state']}`",
                f"- Platform: `{manifest['platform']}`",
                f"- Compiler: `{manifest['compiler']}`",
                f"- Baseline command: `{manifest['baseline_command']}`",
                f"- Project command: `{manifest['project_probe_command']}`",
                "",
                "## Rows",
                "",
                "| Metric | Project value | Baseline value | Delta | Status |",
                "| --- | --- | --- | --- | --- |",
                metric_lines,
                "",
                "## Result",
                "",
                f"- Passing selected rows: `{len(pass_rows)}`",
                f"- Non-passing selected rows: `{len(non_pass_rows)}`",
                "- Proof scope: fixture-local only when every selected row is `pass`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def reset_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(path: Path, manifest: dict[str, str]) -> None:
    rows = [{"key": key, "value": value} for key, value in sorted(manifest.items())]
    write_tsv(path, MANIFEST_FIELDS, rows)


def run(args: argparse.Namespace) -> int:
    if args.target != TARGET:
        raise ComparisonError("unsupported_target", f"unsupported target {args.target!r}")

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    library = args.library.resolve()
    reset_output_dir(output_dir)

    generated_at = utc_timestamp()
    observations, compiler = run_project_probe(root, library, args.keep_temp)
    baseline_observations = run_baseline_reference(root)
    observation_rows = project_observation_rows(observations)
    baseline_rows = baseline_observation_rows(baseline_observations)
    dependency_rows = dependency_status_rows(root)

    project_observations_path = output_dir / "project_observations.tsv"
    baseline_observations_path = output_dir / "baseline_observations.tsv"
    dependency_status_path = output_dir / "dependency_status.tsv"
    study_path = output_dir / "study.tsv"
    summary_path = output_dir / "summary.md"
    manifest_path = output_dir / "manifest.tsv"
    manifest = {
        "target": args.target,
        "fixture_key": FIXTURE_KEY,
        "generated_at_utc": generated_at,
        "source_commit": current_commit(root),
        "source_branch": current_branch(root),
        "worktree_state": worktree_state(root),
        "project_version": project_version(root),
        "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        "compiler": compiler,
        "configuration": (
            "stage=day9_comparison_logic;"
            "baseline_status=integrated_and_compared;support_tier=local_only"
        ),
        "baseline_name": "source-controlled-dense-qr-reference",
        "baseline_type": "external-process-source-controlled-helper",
        "baseline_version": "qr_external_dense_reference.py",
        "baseline_command": baseline_observations["baseline_command"],
        "baseline_helper_path": baseline_observations["baseline_helper_path"],
        "baseline_python_executable": baseline_observations["baseline_python_executable"],
        "baseline_python_version": baseline_observations["baseline_python_version"],
        "project_probe_command": observations["project_probe_command"],
        "project_observations_path": str(project_observations_path.relative_to(root))
        if project_observations_path.is_relative_to(root)
        else str(project_observations_path),
        "baseline_observations_path": str(baseline_observations_path.relative_to(root))
        if baseline_observations_path.is_relative_to(root)
        else str(baseline_observations_path),
        "dependency_status_path": str(dependency_status_path.relative_to(root))
        if dependency_status_path.is_relative_to(root)
        else str(dependency_status_path),
        "study_path": str(study_path.relative_to(root))
        if study_path.is_relative_to(root)
        else str(study_path),
        "summary_path": str(summary_path.relative_to(root))
        if summary_path.is_relative_to(root)
        else str(summary_path),
    }

    study_rows = comparison_study_rows(
        artifact_path=manifest["study_path"],
        baseline_observations=baseline_observations,
        compiler=compiler,
        generated_at=generated_at,
        manifest=manifest,
        observations=observations,
    )
    validate_selected_study_rows(study_rows)

    write_tsv(project_observations_path, PROJECT_OBSERVATION_FIELDS, observation_rows)
    write_tsv(baseline_observations_path, BASELINE_OBSERVATION_FIELDS, baseline_rows)
    write_tsv(dependency_status_path, DEPENDENCY_STATUS_FIELDS, dependency_rows)
    write_tsv(study_path, STUDY_FIELDS, study_rows)
    write_summary(summary_path, study_rows, manifest)
    write_manifest(manifest_path, manifest)

    print(f"external-comparison: wrote {project_observations_path}")
    print(f"external-comparison: wrote {baseline_observations_path}")
    print(f"external-comparison: wrote {dependency_status_path}")
    print(f"external-comparison: wrote {study_path}")
    print(f"external-comparison: wrote {summary_path}")
    print(f"external-comparison: wrote {manifest_path}")
    print("external-comparison: qr-minnorm project-vs-baseline comparison passed")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="comparison target; only qr-minnorm is supported")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_check and not args.target:
        parser.error("--target is required unless --self-check is set")
    return args


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        if args.self_check:
            return run_self_check(args.root.resolve())
        return run(args)
    except ComparisonError as exc:
        print(f"ERROR {exc.failure_class}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
