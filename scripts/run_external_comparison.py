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
DEFAULT_LIBRARY = REPO_ROOT / "build" / "libsparse_lu_ortho.a"
DEFAULT_CMAKE_CONFIG = "Release"

RESIDUAL_TOLERANCE_DEFAULT = 1e-10
SOLUTION_TOLERANCE_DEFAULT = 1e-10
SINGULAR_VALUE_TOLERANCE_DEFAULT = 1e-10
ORTHOGONALITY_TOLERANCE_DEFAULT = 1e-10
PROJECTOR_TOLERANCE_DEFAULT = 1e-10

QR_COMPATIBLE_LS_ENTRIES = [
    (0, 0, 1.0),
    (0, 2, 2.0),
    (1, 1, 1.0),
    (1, 2, -1.0),
    (2, 0, 2.0),
    (2, 1, -1.0),
    (3, 0, 1.0),
    (3, 1, 1.0),
    (3, 2, 1.0),
    (4, 0, 3.0),
    (4, 2, -2.0),
]

LU_NONSYM_SQUARE_5_ENTRIES = [
    (0, 0, 4.0),
    (0, 1, -1.0),
    (0, 3, 2.0),
    (0, 4, 0.5),
    (1, 0, 1.5),
    (1, 1, 5.0),
    (1, 2, -2.0),
    (1, 4, 1.0),
    (2, 1, 2.0),
    (2, 2, 6.0),
    (2, 3, -1.0),
    (3, 0, 3.0),
    (3, 2, 1.0),
    (3, 3, 7.0),
    (3, 4, -2.0),
    (4, 0, -1.0),
    (4, 1, 0.5),
    (4, 3, 2.0),
    (4, 4, 8.0),
]

CHOLESKY_SPD_TRIDIAG_5_ENTRIES = [
    (0, 0, 4.0),
    (0, 1, -1.0),
    (1, 0, -1.0),
    (1, 1, 4.0),
    (1, 2, -1.0),
    (2, 1, -1.0),
    (2, 2, 4.0),
    (2, 3, -1.0),
    (3, 2, -1.0),
    (3, 3, 4.0),
    (3, 4, -1.0),
    (4, 3, -1.0),
    (4, 4, 4.0),
]

TARGETS = {
    "qr-minnorm": {
        "comparison_kind": "qr",
        "fixture_key": "qr_underdetermined_minnorm_2x4",
        "generator_key": "qr_underdetermined_minnorm_2x4_generator_v1",
        "subfamily": "qr_minnorm",
        "operation": "minnorm_solve",
        "output_dir": REPO_ROOT / "build" / "comparison" / "qr_minnorm",
        "rhs": [1.0, 1.0],
        "expected_solution": [0.5, 0.5, 0.5, 0.5],
        "expected_solution_norm": 1.0,
        "residual_tolerance": RESIDUAL_TOLERANCE_DEFAULT,
        "solution_tolerance": SOLUTION_TOLERANCE_DEFAULT,
        "baseline_value_count": 6,
        "solve_mode": "minnorm",
        "claim_scope": "fixture-local qr minimum-norm comparison only",
        "summary_title": "QR Minimum-Norm External Comparison Study",
        "summary_scope": (
            "This local generated study compares one fixture-local QR "
            "minimum-norm solve against the source-controlled dense "
            "reference helper."
        ),
        "success_message": "external-comparison: qr-minnorm project-vs-baseline comparison passed",
    },
    "qr-compatible-ls": {
        "comparison_kind": "qr",
        "fixture_key": "qr_overdetermined_compatible_5x3",
        "entries": QR_COMPATIBLE_LS_ENTRIES,
        "rows": 5,
        "cols": 3,
        "subfamily": "qr_compatible_ls",
        "operation": "least_squares_solve",
        "output_dir": REPO_ROOT / "build" / "comparison" / "qr_compatible_ls",
        "rhs": [2.0, -2.5, 4.0, -0.5, 2.0],
        "expected_solution": [1.0, -2.0, 0.5],
        "expected_solution_norm": 2.2912878474779199,
        "residual_tolerance": RESIDUAL_TOLERANCE_DEFAULT,
        "solution_tolerance": SOLUTION_TOLERANCE_DEFAULT,
        "baseline_value_count": 4,
        "solve_mode": "least_squares",
        "claim_scope": "fixture-local qr compatible least-squares comparison only",
        "summary_title": "QR Compatible Least-Squares External Comparison Study",
        "summary_scope": (
            "This local generated study compares one fixture-local QR "
            "compatible least-squares solve against the source-controlled "
            "dense reference helper."
        ),
        "success_message": (
            "external-comparison: qr-compatible-ls project-vs-baseline comparison passed"
        ),
    },
    "partial-svd-diag6-k2": {
        "comparison_kind": "partial_svd",
        "fixture_key": "partial_svd_diag6_k2",
        "subfamily": "partial_svd_diag6_k2",
        "operation": "partial_svd",
        "output_dir": REPO_ROOT / "build" / "comparison" / "partial_svd_diag6_k2",
        "rows": 6,
        "cols": 6,
        "rank": 2,
        "diag_values": [9.0, 6.0, 3.0, 1.0, 0.5, 0.25],
        "expected_singular_values": [9.0, 6.0],
        "baseline_value_count": 2,
        "singular_value_tolerance": SINGULAR_VALUE_TOLERANCE_DEFAULT,
        "residual_tolerance": RESIDUAL_TOLERANCE_DEFAULT,
        "orthogonality_tolerance": ORTHOGONALITY_TOLERANCE_DEFAULT,
        "projector_tolerance": PROJECTOR_TOLERANCE_DEFAULT,
        "claim_scope": "fixture-local partial-SVD diagonal top-k comparison only",
        "summary_title": "Partial-SVD Diagonal Top-K External Comparison Study",
        "summary_scope": (
            "This local generated study compares one fixture-local partial-SVD "
            "diagonal top-k result against the source-controlled dense "
            "singular-value reference helper."
        ),
        "non_claims": (
            "no broad SVD correctness;no broad partial-SVD correctness;"
            "no raw singular-vector identity;no vector sign or orientation identity;"
            "no repeated-spectrum ordering claim;no NumPy parity;no SciPy parity;"
            "no LAPACK parity;no SuiteSparse parity;no Eigen parity;"
            "no external-library ecosystem parity;no performance claim;"
            "no package-manager proof;no hosted CI proof;no release proof;"
            "no platform portability proof;no shared-library ABI proof;"
            "no state-of-the-art claim"
        ),
        "success_message": (
            "external-comparison: partial-svd-diag6-k2 project-vs-baseline comparison passed"
        ),
    },
    "lu-nonsym-square-5": {
        "comparison_kind": "lu",
        "fixture_key": "lu_nonsym_square_5",
        "entries": LU_NONSYM_SQUARE_5_ENTRIES,
        "rows": 5,
        "cols": 5,
        "subfamily": "lu_nonsym_square_5",
        "operation": "square_solve",
        "output_dir": REPO_ROOT / "build" / "comparison" / "lu_nonsym_square_5",
        "rhs": [12.5, 10.5, 18.0, 24.0, 48.0],
        "expected_solution": [1.0, 2.0, 3.0, 4.0, 5.0],
        "expected_solution_norm": 7.416198487095663,
        "residual_tolerance": RESIDUAL_TOLERANCE_DEFAULT,
        "solution_tolerance": SOLUTION_TOLERANCE_DEFAULT,
        "baseline_value_count": 5,
        "solve_mode": "lu_square_solve",
        "claim_scope": "fixture-local linked-list LU square-solve comparison only",
        "summary_title": "Linked-List LU External Comparison Study",
        "summary_scope": (
            "This local generated study compares one fixture-local linked-list "
            "LU square solve against the source-controlled dense reference helper."
        ),
        "non_claims": (
            "no broad LU correctness;no broad nonsymmetric solve parity;no LU CSR parity;"
            "no sparse-direct solver parity;no pivoting superiority;no factor-layout identity;"
            "no NumPy parity;no SciPy parity;no LAPACK parity;no SuiteSparse parity;"
            "no Eigen parity;no external-library ecosystem parity;no hosted CI proof;"
            "no release proof;no platform portability proof;no package-manager proof;"
            "no shared-library ABI proof;no performance superiority;no state-of-the-art claim"
        ),
        "success_message": (
            "external-comparison: lu-nonsym-square-5 project-vs-baseline comparison passed"
        ),
    },
    "cholesky-spd-tridiag-5": {
        "comparison_kind": "cholesky",
        "fixture_key": "cholesky_spd_tridiag_5",
        "entries": CHOLESKY_SPD_TRIDIAG_5_ENTRIES,
        "rows": 5,
        "cols": 5,
        "subfamily": "cholesky_spd_tridiag_5",
        "operation": "cholesky_spd_solve",
        "output_dir": REPO_ROOT / "build" / "comparison" / "cholesky_spd_tridiag_5",
        "rhs": [2.0, 4.0, 6.0, 8.0, 16.0],
        "expected_solution": [1.0, 2.0, 3.0, 4.0, 5.0],
        "expected_solution_norm": 7.416198487095663,
        "residual_tolerance": RESIDUAL_TOLERANCE_DEFAULT,
        "solution_tolerance": SOLUTION_TOLERANCE_DEFAULT,
        "baseline_value_count": 5,
        "solve_mode": "cholesky_spd_solve",
        "claim_scope": "fixture-local Cholesky SPD tridiagonal solve comparison only",
        "summary_title": "Cholesky SPD Tridiagonal External Comparison Study",
        "summary_scope": (
            "This local generated study compares one fixture-local Cholesky "
            "SPD tridiagonal solve against the source-controlled dense "
            "Cholesky reference helper."
        ),
        "non_claims": (
            "no broad Cholesky correctness;no broad SPD coverage;"
            "no broad reordering coverage;no CSC-vs-linked-list parity;"
            "no factor-layout identity;no fill superiority;no NumPy parity;"
            "no SciPy parity;no LAPACK parity;no SuiteSparse parity;"
            "no Eigen parity;no external-library ecosystem parity;"
            "no hosted CI proof;no release proof;no platform portability proof;"
            "no Windows report freshness;no package-manager proof;"
            "no shared-library ABI proof;no performance superiority;"
            "no state-of-the-art claim"
        ),
        "success_message": (
            "external-comparison: cholesky-spd-tridiag-5 "
            "project-vs-baseline comparison passed"
        ),
    },
}

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
    if library.resolve() != DEFAULT_LIBRARY.resolve():
        raise ComparisonError(
            "project_build_failed",
            f"required static library is missing: {library}",
        )
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


def run_cmake_project_probe(
    *,
    root: Path,
    source: Path,
    binary_name: str,
    library: Path,
    generator: str | None,
    arch: str | None,
    config: str,
) -> tuple[str, str, str]:
    build_dir = source.parent / "cmake-build"
    cmake_lists = source.parent / "CMakeLists.txt"
    cmake_lists.write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.14)",
                "project(sparse_external_comparison_probe C)",
                "set(CMAKE_C_STANDARD 99)",
                "set(CMAKE_C_STANDARD_REQUIRED ON)",
                f"add_executable({binary_name} {source.name})",
                f"target_include_directories({binary_name} PRIVATE",
                f'  "{root / "include"}"',
                f'  "{root / "build" / "include"}"',
                ")",
                "add_library(sparse_lu_ortho STATIC IMPORTED GLOBAL)",
                "set_target_properties(sparse_lu_ortho PROPERTIES",
                f'  IMPORTED_LOCATION "{library}"',
                ")",
                f"target_link_libraries({binary_name} PRIVATE sparse_lu_ortho)",
                "if(NOT MSVC)",
                f"  target_link_libraries({binary_name} PRIVATE m)",
                "endif()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    configure_cmd = ["cmake", "-S", str(source.parent), "-B", str(build_dir)]
    if generator:
        configure_cmd.extend(["-G", generator])
    if arch:
        configure_cmd.extend(["-A", arch])
    run_capture(configure_cmd, cwd=root, failure_class="project_build_failed")
    build_cmd = ["cmake", "--build", str(build_dir), "--config", config]
    run_capture(build_cmd, cwd=root, failure_class="project_build_failed")
    binary_candidates = [
        build_dir / config / f"{binary_name}.exe",
        build_dir / config / binary_name,
        build_dir / f"{binary_name}.exe",
        build_dir / binary_name,
    ]
    binary = next((candidate for candidate in binary_candidates if candidate.is_file()), None)
    if binary is None:
        raise ComparisonError(
            "project_build_failed",
            f"CMake probe build did not produce {binary_name}",
        )
    output = run_capture([str(binary)], cwd=root)
    command = shlex.join(configure_cmd) + " && " + shlex.join(build_cmd) + " && " + str(binary)
    compiler = f"cmake-probe:{generator or 'default'}:{config}"
    return output, compiler, command


def run_compiler_project_probe(
    *,
    root: Path,
    source: Path,
    binary: Path,
    library: Path,
) -> tuple[str, str, str]:
    cc = compiler_argv()
    compiler = compiler_identity(cc)
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
    return output, compiler, shlex.join(compile_cmd) + " && " + str(binary)


def c_literal_for_entries(entries: list[tuple[int, int, float]]) -> str:
    return "\n".join(
        f"    {{{row}, {col}, {value:.17g}}}," for row, col, value in entries
    )


def c_literal_for_values(values: list[float]) -> str:
    return ", ".join(f"{value:.17g}" for value in values)


def partial_svd_project_probe_source(
    diag_values: list[float],
    rows: int,
    cols: int,
    rank: int,
) -> str:
    return f"""#include \"sparse_matrix.h\"
#include \"sparse_svd.h\"
#include \"sparse_types.h\"
#include <math.h>
#include <stdio.h>

static const double diag_values[{cols}] = {{{c_literal_for_values(diag_values)}}};

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

static double dense_entry(idx_t row, idx_t col) {{
    if (row == col && col < {cols})
        return diag_values[col];
    return 0.0;
}}

static double max_triplet_residual(const sparse_svd_t *svd) {{
    double max_residual = 0.0;
    for (idx_t comp = 0; comp < {rank}; ++comp) {{
        const double sigma = svd->sigma[comp];
        double av_sq = 0.0;
        for (idx_t row = 0; row < {rows}; ++row) {{
            double av = 0.0;
            for (idx_t col = 0; col < {cols}; ++col)
                av += dense_entry(row, col) * svd->Vt[comp + col * {rank}];
            const double diff = av - sigma * svd->U[row + comp * {rows}];
            av_sq += diff * diff;
        }}
        double atu_sq = 0.0;
        for (idx_t col = 0; col < {cols}; ++col) {{
            double atu = 0.0;
            for (idx_t row = 0; row < {rows}; ++row)
                atu += dense_entry(row, col) * svd->U[row + comp * {rows}];
            const double diff = atu - sigma * svd->Vt[comp + col * {rank}];
            atu_sq += diff * diff;
        }}
        const double av_residual = sqrt(av_sq);
        const double atu_residual = sqrt(atu_sq);
        if (av_residual > max_residual)
            max_residual = av_residual;
        if (atu_residual > max_residual)
            max_residual = atu_residual;
    }}
    return max_residual;
}}

static double u_orthogonality(const sparse_svd_t *svd) {{
    double max_error = 0.0;
    for (idx_t i = 0; i < {rank}; ++i) {{
        for (idx_t j = i; j < {rank}; ++j) {{
            double dot = 0.0;
            for (idx_t row = 0; row < {rows}; ++row)
                dot += svd->U[row + i * {rows}] * svd->U[row + j * {rows}];
            const double expected = (i == j) ? 1.0 : 0.0;
            const double error = fabs(dot - expected);
            if (error > max_error)
                max_error = error;
        }}
    }}
    return max_error;
}}

static double v_orthogonality(const sparse_svd_t *svd) {{
    double max_error = 0.0;
    for (idx_t i = 0; i < {rank}; ++i) {{
        for (idx_t j = i; j < {rank}; ++j) {{
            double dot = 0.0;
            for (idx_t col = 0; col < {cols}; ++col)
                dot += svd->Vt[i + col * {rank}] * svd->Vt[j + col * {rank}];
            const double expected = (i == j) ? 1.0 : 0.0;
            const double error = fabs(dot - expected);
            if (error > max_error)
                max_error = error;
        }}
    }}
    return max_error;
}}

static double u_projector_diag(const sparse_svd_t *svd) {{
    double max_error = 0.0;
    for (idx_t row = 0; row < {rows}; ++row) {{
        double value = 0.0;
        for (idx_t comp = 0; comp < {rank}; ++comp) {{
            const double u = svd->U[row + comp * {rows}];
            value += u * u;
        }}
        const double expected = row < {rank} ? 1.0 : 0.0;
        const double error = fabs(value - expected);
        if (error > max_error)
            max_error = error;
    }}
    return max_error;
}}

static double v_projector_diag(const sparse_svd_t *svd) {{
    double max_error = 0.0;
    for (idx_t col = 0; col < {cols}; ++col) {{
        double value = 0.0;
        for (idx_t comp = 0; comp < {rank}; ++comp) {{
            const double v = svd->Vt[comp + col * {rank}];
            value += v * v;
        }}
        const double expected = col < {rank} ? 1.0 : 0.0;
        const double error = fabs(value - expected);
        if (error > max_error)
            max_error = error;
    }}
    return max_error;
}}

int main(void) {{
    SparseMatrix *A = sparse_create({rows}, {cols});
    if (!A) {{
        fprintf(stderr, \"sparse_create failed\\n\");
        return 2;
    }}
    for (idx_t i = 0; i < {cols}; ++i) {{
        if (sparse_insert(A, i, i, diag_values[i]) != SPARSE_OK) {{
            fprintf(stderr, \"sparse_insert failed at %d\\n\", (int)i);
            sparse_free(A);
            return 3;
        }}
    }}

    sparse_svd_opts_t opts = {{.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0}};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_partial(A, {rank}, &opts, &svd);

    printf(\"status=%s\\n\", status_name(err));
    if (err == SPARSE_OK && svd.sigma && svd.U && svd.Vt && svd.k >= {rank}) {{
        double max_sigma_delta = 0.0;
        for (idx_t i = 0; i < {rank}; ++i) {{
            const double delta = fabs(svd.sigma[i] - diag_values[i]);
            if (delta > max_sigma_delta)
                max_sigma_delta = delta;
            printf(\"singular_value_%d=%.17g\\n\", (int)i, svd.sigma[i]);
        }}
        printf(\"singular_values_max_abs_delta=%.17g\\n\", max_sigma_delta);
        printf(\"residual_norm=%.17g\\n\", max_triplet_residual(&svd));
        printf(\"u_orthogonality=%.17g\\n\", u_orthogonality(&svd));
        printf(\"v_orthogonality=%.17g\\n\", v_orthogonality(&svd));
        printf(\"u_projector_diag=%.17g\\n\", u_projector_diag(&svd));
        printf(\"v_projector_diag=%.17g\\n\", v_projector_diag(&svd));
    }} else {{
        for (idx_t i = 0; i < {rank}; ++i)
            printf(\"singular_value_%d=inf\\n\", (int)i);
        printf(\"singular_values_max_abs_delta=inf\\n\");
        printf(\"residual_norm=inf\\n\");
        printf(\"u_orthogonality=inf\\n\");
        printf(\"v_orthogonality=inf\\n\");
        printf(\"u_projector_diag=inf\\n\");
        printf(\"v_projector_diag=inf\\n\");
    }}

    sparse_svd_free(&svd);
    sparse_free(A);
    return 0;
}}
"""


def project_probe_source(
    entries: list[tuple[int, int, float]],
    rows: int,
    cols: int,
    rhs: list[float],
    solve_mode: str,
) -> str:
    if solve_mode == "minnorm":
        solve_block = "    sparse_err_t err = sparse_qr_solve_minnorm(A, rhs, x, NULL);\n"
        cleanup_block = ""
    elif solve_mode == "least_squares":
        solve_block = """    sparse_qr_t qr;
    int qr_factored = 0;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    if (err == SPARSE_OK) {
        qr_factored = 1;
        err = sparse_qr_solve(&qr, rhs, x, NULL);
    }
"""
        cleanup_block = """    if (qr_factored)
        sparse_qr_free(&qr);
"""
    elif solve_mode == "lu_square_solve":
        solve_block = """    sparse_err_t err = sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12);
    if (err == SPARSE_OK)
        err = sparse_lu_solve(A, rhs, x);
"""
        cleanup_block = ""
    elif solve_mode == "cholesky_spd_solve":
        solve_block = """    sparse_err_t err = sparse_cholesky_factor(A);
    if (err == SPARSE_OK)
        err = sparse_cholesky_solve(A, rhs, x);
"""
        cleanup_block = ""
    else:
        raise ComparisonError("unsupported_target", f"unsupported project solve mode {solve_mode!r}")
    return f"""#include \"sparse_matrix.h\"
#include \"sparse_cholesky.h\"
#include \"sparse_lu.h\"
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

static const double rhs[{rows}] = {{{c_literal_for_values(rhs)}}};

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
{solve_block}

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

{cleanup_block}
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


def descriptor_entries(target: dict[str, object]) -> tuple[list[tuple[int, int, float]], int, int]:
    if "generator_key" in target:
        fixture = GENERATED_FIXTURES[str(target["generator_key"])]
        entries = fixture["entries"]()
        rows = int(fixture["rows"])
        cols = int(fixture["cols"])
        return entries, rows, cols
    if {"entries", "rows", "cols"} <= set(target):
        return (
            list(target["entries"]),  # type: ignore[arg-type]
            int(target["rows"]),
            int(target["cols"]),
        )
    raise ComparisonError(
        "missing_fixture_metadata",
        f"missing fixture metadata for {target['fixture_key']}",
    )


def run_project_probe(
    root: Path,
    library: Path,
    keep_temp: bool,
    target: dict[str, object],
    *,
    probe_build_system: str,
    cmake_generator: str | None,
    cmake_arch: str | None,
    cmake_config: str,
) -> tuple[dict[str, str], str]:
    ensure_library(root, library)

    temp_dir = Path(tempfile.mkdtemp(prefix="sparse-comparison-"))
    try:
        source = temp_dir / f"{str(target['subfamily'])}_probe.c"
        binary = temp_dir / f"{str(target['subfamily'])}_probe"
        if target.get("comparison_kind") == "partial_svd":
            source.write_text(
                partial_svd_project_probe_source(
                    list(target["diag_values"]),  # type: ignore[arg-type]
                    int(target["rows"]),
                    int(target["cols"]),
                    int(target["rank"]),
                ),
                encoding="utf-8",
            )
            required_fields = {
                "status",
                "singular_value_0",
                "singular_value_1",
                "singular_values_max_abs_delta",
                "residual_norm",
                "u_orthogonality",
                "v_orthogonality",
                "u_projector_diag",
                "v_projector_diag",
            }
        else:
            entries, rows, cols = descriptor_entries(target)
            source.write_text(
                project_probe_source(
                    entries,
                    rows,
                    cols,
                    list(target["rhs"]),  # type: ignore[arg-type]
                    str(target["solve_mode"]),
                ),
                encoding="utf-8",
            )
            required_fields = {"status", "residual_norm", "solution_norm", "solution_values"}
        if probe_build_system == "auto":
            selected_build_system = (
                "cmake"
                if library.suffix.lower() == ".lib" or platform.system().lower() == "windows"
                else "compiler"
            )
        else:
            selected_build_system = probe_build_system
        if selected_build_system == "cmake":
            output, compiler, project_probe_command = run_cmake_project_probe(
                root=root,
                source=source,
                binary_name=str(target["subfamily"]) + "_probe",
                library=library,
                generator=cmake_generator,
                arch=cmake_arch,
                config=cmake_config,
            )
        else:
            output, compiler, project_probe_command = run_compiler_project_probe(
                root=root,
                source=source,
                binary=binary,
                library=library,
            )
        parsed = parse_key_values(output, required_fields)
        parsed["project_probe_command"] = project_probe_command
        return parsed, compiler
    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def python_version() -> str:
    return " ".join(sys.version.split())


def baseline_name(target: dict[str, object]) -> str:
    if target.get("comparison_kind") == "partial_svd":
        return "source-controlled-dense-svd-reference"
    if target.get("comparison_kind") == "lu":
        return "source-controlled-dense-lu-reference"
    if target.get("comparison_kind") == "cholesky":
        return "source-controlled-dense-cholesky-reference"
    return "source-controlled-dense-qr-reference"


def baseline_version(target: dict[str, object]) -> str:
    if target.get("comparison_kind") == "partial_svd":
        return "svd_external_dense_reference.py"
    if target.get("comparison_kind") == "lu":
        return "lu_external_dense_reference.py"
    if target.get("comparison_kind") == "cholesky":
        return "chol_external_dense_reference.py"
    return "qr_external_dense_reference.py"


def comparison_configuration(target: dict[str, object]) -> str:
    if target.get("comparison_kind") == "partial_svd":
        stage = "sprint161_day5_comparison_logic"
    elif target.get("comparison_kind") == "lu":
        stage = "sprint174_day8_comparison_logic"
    elif target.get("comparison_kind") == "cholesky":
        stage = "sprint183_day8_comparison_logic"
    else:
        stage = "sprint160_day5_comparison_logic"
    return f"stage={stage};baseline_status=integrated_and_compared;support_tier=local_only"


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


def vector_norm(values: list[float]) -> float:
    return sum(value * value for value in values) ** 0.5


def residual_norm_from_entries(
    entries: list[tuple[int, int, float]],
    rows: int,
    rhs: list[float],
    solution: list[float],
) -> float:
    residual_sq = 0.0
    for row in range(rows):
        accum = 0.0
        for entry_row, entry_col, value in entries:
            if entry_row == row:
                accum += value * solution[entry_col]
        diff = accum - rhs[row]
        residual_sq += diff * diff
    return residual_sq**0.5


def run_solve_baseline_reference(
    root: Path, target: dict[str, object], helper_name: str
) -> dict[str, str]:
    helper = root / helper_name
    if not helper.is_file():
        raise ComparisonError(
            "missing_baseline_helper",
            f"selected baseline helper is missing: {helper}",
        )

    fixture_key = str(target["fixture_key"])
    baseline_value_count = int(target["baseline_value_count"])
    command = [sys.executable, str(helper), fixture_key]
    output = run_capture(command, cwd=root, failure_class="baseline_command_failed")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ComparisonError("baseline_malformed_output", "baseline emitted no output")

    header = lines[0].split()
    if len(header) != 2 or header[0] != "OK":
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline first line must be 'OK {baseline_value_count}', got {lines[0]!r}",
        )
    try:
        value_count = int(header[1])
    except ValueError as exc:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline value count is not an integer: {header[1]!r}",
        ) from exc
    if value_count != baseline_value_count:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline value count must be {baseline_value_count} for {fixture_key}, got {value_count}",
        )

    value_lines = lines[1:]
    if len(value_lines) != value_count:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline emitted {len(value_lines)} values, expected {value_count}",
        )
    try:
        solution = [float(value) for value in value_lines]
    except ValueError as exc:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline emitted non-numeric values: {', '.join(value_lines)}",
        ) from exc

    entries, rows, _ = descriptor_entries(target)
    residual = residual_norm_from_entries(
        entries,
        rows,
        list(target["rhs"]),  # type: ignore[arg-type]
        solution,
    )
    return {
        "status": "success",
        "solution_values": ",".join(f"{value:.17g}" for value in solution),
        "residual_norm": f"{residual:.17g}",
        "solution_norm": f"{vector_norm(solution):.17g}",
        "baseline_command": shlex.join(command),
        "baseline_helper_path": str(helper.relative_to(root)),
        "baseline_python_executable": sys.executable,
        "baseline_python_version": python_version(),
    }


def run_baseline_reference(root: Path, target: dict[str, object]) -> dict[str, str]:
    if target.get("comparison_kind") == "partial_svd":
        return run_partial_svd_baseline_reference(root, target)
    if target.get("comparison_kind") == "lu":
        return run_solve_baseline_reference(root, target, "tests/lu_external_dense_reference.py")
    if target.get("comparison_kind") == "cholesky":
        return run_solve_baseline_reference(
            root, target, "tests/chol_external_dense_reference.py"
        )
    helper = root / "tests" / "qr_external_dense_reference.py"
    if not helper.is_file():
        raise ComparisonError(
            "missing_baseline_helper",
            f"selected baseline helper is missing: {helper}",
        )

    fixture_key = str(target["fixture_key"])
    baseline_value_count = int(target["baseline_value_count"])
    expected_solution = list(target["expected_solution"])  # type: ignore[arg-type]
    command = [sys.executable, str(helper), fixture_key]
    output = run_capture(command, cwd=root, failure_class="baseline_command_failed")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ComparisonError("baseline_malformed_output", "baseline emitted no output")

    header = lines[0].split()
    if len(header) != 2 or header[0] != "OK":
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline first line must be 'OK {baseline_value_count}', got {lines[0]!r}",
        )
    try:
        value_count = int(header[1])
    except ValueError as exc:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline value count is not an integer: {header[1]!r}",
        ) from exc
    if value_count != baseline_value_count:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline value count must be {baseline_value_count} for {fixture_key}, got {value_count}",
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

    solution_count = len(expected_solution)
    if value_count < solution_count + 1:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline value count must include solution and residual for {fixture_key}",
        )
    solution = values[:solution_count]
    residual = values[solution_count]
    if value_count > solution_count + 1:
        solution_norm = values[solution_count + 1]
    else:
        solution_norm = sum(value * value for value in solution) ** 0.5
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


def run_partial_svd_baseline_reference(root: Path, target: dict[str, object]) -> dict[str, str]:
    helper = root / "tests" / "svd_external_dense_reference.py"
    if not helper.is_file():
        raise ComparisonError(
            "missing_baseline_helper",
            f"selected baseline helper is missing: {helper}",
        )

    fixture_key = str(target["fixture_key"])
    baseline_value_count = int(target["baseline_value_count"])
    command = [sys.executable, str(helper), fixture_key]
    output = run_capture(command, cwd=root, failure_class="baseline_command_failed")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ComparisonError("baseline_malformed_output", "baseline emitted no output")

    header = lines[0].split()
    if len(header) != 2 or header[0] != "OK":
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline first line must be 'OK {baseline_value_count}', got {lines[0]!r}",
        )
    try:
        value_count = int(header[1])
    except ValueError as exc:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline value count is not an integer: {header[1]!r}",
        ) from exc
    if value_count != baseline_value_count:
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline value count must be {baseline_value_count} for {fixture_key}, got {value_count}",
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

    observations = {
        "status": "success",
        "baseline_command": shlex.join(command),
        "baseline_helper_path": str(helper.relative_to(root)),
        "baseline_python_executable": sys.executable,
        "baseline_python_version": python_version(),
    }
    for index, value in enumerate(values):
        observations[f"singular_value_{index}"] = f"{value:.17g}"
    observations["singular_values"] = ",".join(f"{value:.17g}" for value in values)
    return observations


def project_observation_rows(
    observations: dict[str, str], target: dict[str, object]
) -> list[dict[str, str]]:
    if target.get("comparison_kind") == "partial_svd":
        return partial_svd_project_observation_rows(observations, target)
    values = parse_vector(observations["solution_values"])
    expected_solution = list(target["expected_solution"])  # type: ignore[arg-type]
    solution_tolerance = float(target["solution_tolerance"])
    residual_tolerance = float(target["residual_tolerance"])
    if len(values) != len(expected_solution):
        raise ComparisonError(
            "project_probe_failed",
            f"project solution has {len(values)} values, expected {len(expected_solution)}",
        )
    max_abs_delta = max(
        abs(expected - observed) for expected, observed in zip(expected_solution, values)
    )
    rows = [
        {
            "fixture_key": str(target["fixture_key"]),
            "metric": "project_status",
            "value": observations["status"],
            "status": "pass" if observations["status"] == "SPARSE_SUCCESS" else "fail",
            "status_reason": "project_status_match"
            if observations["status"] == "SPARSE_SUCCESS"
            else "project_status_mismatch",
        },
        {
            "fixture_key": str(target["fixture_key"]),
            "metric": "residual_norm",
            "value": observations["residual_norm"],
            "status": "pass"
            if float(observations["residual_norm"]) <= residual_tolerance
            else "fail",
            "status_reason": "project_residual_within_tolerance"
            if float(observations["residual_norm"]) <= residual_tolerance
            else "project_residual_tolerance_miss",
        },
        {
            "fixture_key": str(target["fixture_key"]),
            "metric": "solution_norm",
            "value": observations["solution_norm"],
            "status": "pass"
            if abs(float(observations["solution_norm"]) - float(target["expected_solution_norm"]))
            <= solution_tolerance
            else "fail",
            "status_reason": "project_solution_norm_within_tolerance"
            if abs(float(observations["solution_norm"]) - float(target["expected_solution_norm"]))
            <= solution_tolerance
            else "project_solution_norm_tolerance_miss",
        },
        {
            "fixture_key": str(target["fixture_key"]),
            "metric": "solution_values",
            "value": observations["solution_values"],
            "status": "pass" if max_abs_delta <= solution_tolerance else "fail",
            "status_reason": "project_solution_values_within_tolerance"
            if max_abs_delta <= solution_tolerance
            else "project_solution_values_tolerance_miss",
        },
    ]
    return rows


def metric_status(value: float, tolerance: float) -> tuple[str, str]:
    if value <= tolerance:
        return "pass", "project_metric_within_tolerance"
    return "fail", "project_metric_tolerance_miss"


def partial_svd_project_observation_rows(
    observations: dict[str, str], target: dict[str, object]
) -> list[dict[str, str]]:
    rows = [
        {
            "fixture_key": str(target["fixture_key"]),
            "metric": "project_status",
            "value": observations["status"],
            "status": "pass" if observations["status"] == "SPARSE_SUCCESS" else "fail",
            "status_reason": "project_status_match"
            if observations["status"] == "SPARSE_SUCCESS"
            else "project_status_mismatch",
        }
    ]
    singular_tolerance = float(target["singular_value_tolerance"])
    expected_singular_values = list(target["expected_singular_values"])  # type: ignore[arg-type]
    for index, expected in enumerate(expected_singular_values):
        metric = f"singular_value_{index}"
        value = float(observations[metric])
        delta = abs(value - float(expected))
        rows.append(
            {
                "fixture_key": str(target["fixture_key"]),
                "metric": metric,
                "value": observations[metric],
                "status": "pass" if delta <= singular_tolerance else "fail",
                "status_reason": "project_singular_value_within_tolerance"
                if delta <= singular_tolerance
                else "project_singular_value_tolerance_miss",
            }
        )
    metric_tolerances = {
        "singular_values_max_abs_delta": float(target["singular_value_tolerance"]),
        "residual_norm": float(target["residual_tolerance"]),
        "u_orthogonality": float(target["orthogonality_tolerance"]),
        "v_orthogonality": float(target["orthogonality_tolerance"]),
        "u_projector_diag": float(target["projector_tolerance"]),
        "v_projector_diag": float(target["projector_tolerance"]),
    }
    for metric, tolerance in metric_tolerances.items():
        status, reason = metric_status(float(observations[metric]), tolerance)
        rows.append(
            {
                "fixture_key": str(target["fixture_key"]),
                "metric": metric,
                "value": observations[metric],
                "status": status,
                "status_reason": reason,
            }
        )
    return rows


def baseline_observation_rows(
    observations: dict[str, str], target: dict[str, object]
) -> list[dict[str, str]]:
    if target.get("comparison_kind") == "partial_svd":
        return partial_svd_baseline_observation_rows(observations, target)
    values = parse_baseline_vector(observations["solution_values"])
    expected_solution = list(target["expected_solution"])  # type: ignore[arg-type]
    solution_tolerance = float(target["solution_tolerance"])
    residual_tolerance = float(target["residual_tolerance"])
    if len(values) != len(expected_solution):
        raise ComparisonError(
            "baseline_malformed_output",
            f"baseline solution has {len(values)} values, expected {len(expected_solution)}",
        )
    max_abs_delta = max(
        abs(expected - observed) for expected, observed in zip(expected_solution, values)
    )
    residual = float(observations["residual_norm"])
    solution_norm = float(observations["solution_norm"])
    rows = [
        {
            "fixture_key": str(target["fixture_key"]),
            "metric": "baseline_status",
            "value": observations["status"],
            "status": "pass" if observations["status"] == "success" else "fail",
            "status_reason": "baseline_status_success"
            if observations["status"] == "success"
            else "baseline_status_mismatch",
        },
        {
            "fixture_key": str(target["fixture_key"]),
            "metric": "baseline_residual_norm",
            "value": observations["residual_norm"],
            "status": "pass" if residual <= residual_tolerance else "fail",
            "status_reason": "baseline_residual_within_tolerance"
            if residual <= residual_tolerance
            else "baseline_residual_tolerance_miss",
        },
        {
            "fixture_key": str(target["fixture_key"]),
            "metric": "baseline_solution_norm",
            "value": observations["solution_norm"],
            "status": "pass"
            if abs(solution_norm - float(target["expected_solution_norm"])) <= solution_tolerance
            else "fail",
            "status_reason": "baseline_solution_norm_within_tolerance"
            if abs(solution_norm - float(target["expected_solution_norm"])) <= solution_tolerance
            else "baseline_solution_norm_tolerance_miss",
        },
        {
            "fixture_key": str(target["fixture_key"]),
            "metric": "baseline_solution_values",
            "value": observations["solution_values"],
            "status": "pass" if max_abs_delta <= solution_tolerance else "fail",
            "status_reason": "baseline_solution_values_within_tolerance"
            if max_abs_delta <= solution_tolerance
            else "baseline_solution_values_tolerance_miss",
        },
    ]
    return rows


def partial_svd_baseline_observation_rows(
    observations: dict[str, str], target: dict[str, object]
) -> list[dict[str, str]]:
    rows = [
        {
            "fixture_key": str(target["fixture_key"]),
            "metric": "baseline_status",
            "value": observations["status"],
            "status": "pass" if observations["status"] == "success" else "fail",
            "status_reason": "baseline_status_success"
            if observations["status"] == "success"
            else "baseline_status_mismatch",
        }
    ]
    singular_tolerance = float(target["singular_value_tolerance"])
    expected_singular_values = list(target["expected_singular_values"])  # type: ignore[arg-type]
    for index, expected in enumerate(expected_singular_values):
        metric = f"baseline_singular_value_{index}"
        observation_key = f"singular_value_{index}"
        value = float(observations[observation_key])
        delta = abs(value - float(expected))
        rows.append(
            {
                "fixture_key": str(target["fixture_key"]),
                "metric": metric,
                "value": observations[observation_key],
                "status": "pass" if delta <= singular_tolerance else "fail",
                "status_reason": "baseline_singular_value_within_tolerance"
                if delta <= singular_tolerance
                else "baseline_singular_value_tolerance_miss",
            }
        )
    return rows


def dependency_status_rows(root: Path, target: dict[str, object]) -> list[dict[str, str]]:
    if target.get("comparison_kind") == "partial_svd":
        helper_name = "tests/svd_external_dense_reference.py"
    elif target.get("comparison_kind") == "lu":
        helper_name = "tests/lu_external_dense_reference.py"
    elif target.get("comparison_kind") == "cholesky":
        helper_name = "tests/chol_external_dense_reference.py"
    else:
        helper_name = "tests/qr_external_dense_reference.py"
    helper = root / helper_name
    return [
        {
            "dependency": "python3",
            "status": "pass",
            "status_reason": "selected_interpreter_available",
            "required": "yes",
            "caveat": "current Python executable only; no package-manager inference",
        },
        {
            "dependency": helper_name,
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
    target: dict[str, object],
) -> dict[str, str]:
    return {
        "report_family": "comparison",
        "subfamily": str(target["subfamily"]),
        "fixture_key": str(target["fixture_key"]),
        "operation": str(target["operation"]),
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
        "claim_scope": str(target["claim_scope"]),
        "non_claims": str(target.get("non_claims", NON_CLAIMS)),
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
    target: dict[str, object],
) -> list[dict[str, str]]:
    if target.get("comparison_kind") == "partial_svd":
        return partial_svd_comparison_study_rows(
            artifact_path=artifact_path,
            baseline_observations=baseline_observations,
            compiler=compiler,
            generated_at=generated_at,
            manifest=manifest,
            observations=observations,
            target=target,
        )
    context = build_study_context(
        artifact_path=artifact_path,
        baseline_observations=baseline_observations,
        compiler=compiler,
        generated_at=generated_at,
        manifest=manifest,
        observations=observations,
        target=target,
    )
    fixture_key = str(target["fixture_key"])
    expected_solution_values = list(target["expected_solution"])  # type: ignore[arg-type]
    residual_tolerance = float(target["residual_tolerance"])
    solution_tolerance = float(target["solution_tolerance"])
    project_solution = parse_vector(observations["solution_values"])
    baseline_solution = parse_baseline_vector(baseline_observations["solution_values"])
    expected_solution = ",".join(format_float(value) for value in expected_solution_values)
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
            comparison_row_id=f"comparison_{fixture_key}_project_status_v1",
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
            comparison_row_id=f"comparison_{fixture_key}_baseline_status_v1",
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
            comparison_row_id=f"comparison_{fixture_key}_residual_norm_v1",
            row_kind="metric_comparison",
            metric="residual_norm",
            expected_value=f"<={format_float(residual_tolerance)}",
            project_value=observations["residual_norm"],
            baseline_value=baseline_observations["residual_norm"],
            delta_value=format_float(residual_delta),
            tolerance_kind="absolute",
            tolerance_value=format_float(residual_tolerance),
            status="pass" if residual_delta <= residual_tolerance else "fail",
            status_reason="project_baseline_residual_delta_within_tolerance"
            if residual_delta <= residual_tolerance
            else "project_baseline_residual_delta_tolerance_miss",
            caveat=caveat,
        ),
        study_row(
            context,
            comparison_row_id=f"comparison_{fixture_key}_solution_norm_v1",
            row_kind="metric_comparison",
            metric="solution_norm",
            expected_value=format_float(float(target["expected_solution_norm"])),
            project_value=observations["solution_norm"],
            baseline_value=baseline_observations["solution_norm"],
            delta_value=format_float(solution_norm_delta),
            tolerance_kind="absolute",
            tolerance_value=format_float(solution_tolerance),
            status="pass" if solution_norm_delta <= solution_tolerance else "fail",
            status_reason="project_baseline_solution_norm_delta_within_tolerance"
            if solution_norm_delta <= solution_tolerance
            else "project_baseline_solution_norm_delta_tolerance_miss",
            caveat=caveat,
        ),
        study_row(
            context,
            comparison_row_id=f"comparison_{fixture_key}_solution_values_v1",
            row_kind="metric_comparison",
            metric="solution_values",
            expected_value=expected_solution,
            project_value=observations["solution_values"],
            baseline_value=baseline_observations["solution_values"],
            delta_value=format_float(solution_delta),
            tolerance_kind="absolute_per_component",
            tolerance_value=format_float(solution_tolerance),
            status="pass" if solution_delta <= solution_tolerance else "fail",
            status_reason="project_baseline_solution_values_delta_within_tolerance"
            if solution_delta <= solution_tolerance
            else "project_baseline_solution_values_delta_tolerance_miss",
            caveat=caveat,
        ),
        study_row(
            context,
            comparison_row_id=f"comparison_{fixture_key}_project_vs_baseline_max_abs_delta_v1",
            row_kind="metric_comparison",
            metric="project_vs_baseline_max_abs_delta",
            expected_value=f"<={format_float(solution_tolerance)}",
            project_value=observations["solution_values"],
            baseline_value=baseline_observations["solution_values"],
            delta_value=format_float(solution_delta),
            tolerance_kind="absolute",
            tolerance_value=format_float(solution_tolerance),
            status="pass" if solution_delta <= solution_tolerance else "fail",
            status_reason="project_baseline_max_abs_delta_within_tolerance"
            if solution_delta <= solution_tolerance
            else "project_baseline_max_abs_delta_tolerance_miss",
            caveat=caveat,
        ),
    ]
    return rows


def partial_svd_comparison_study_rows(
    *,
    artifact_path: str,
    baseline_observations: dict[str, str],
    compiler: str,
    generated_at: str,
    manifest: dict[str, str],
    observations: dict[str, str],
    target: dict[str, object],
) -> list[dict[str, str]]:
    context = build_study_context(
        artifact_path=artifact_path,
        baseline_observations=baseline_observations,
        compiler=compiler,
        generated_at=generated_at,
        manifest=manifest,
        observations=observations,
        target=target,
    )
    fixture_key = str(target["fixture_key"])
    singular_tolerance = float(target["singular_value_tolerance"])
    residual_tolerance = float(target["residual_tolerance"])
    orthogonality_tolerance = float(target["orthogonality_tolerance"])
    projector_tolerance = float(target["projector_tolerance"])
    expected_singular_values = list(target["expected_singular_values"])  # type: ignore[arg-type]
    caveat = (
        "local generated artifact; dirty worktree allowed only as explicit provenance"
        if manifest["worktree_state"] == "dirty"
        else "local generated artifact"
    )

    rows = [
        study_row(
            context,
            comparison_row_id=f"comparison_{fixture_key}_project_status_v1",
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
            comparison_row_id=f"comparison_{fixture_key}_baseline_status_v1",
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
            caveat="required source-controlled dense singular-value reference; not an external package",
        ),
    ]
    max_delta = 0.0
    for index, expected in enumerate(expected_singular_values):
        metric = f"singular_value_{index}"
        project_value = float(observations[metric])
        baseline_value = float(baseline_observations[metric])
        delta = abs(project_value - baseline_value)
        max_delta = max(max_delta, delta)
        rows.append(
            study_row(
                context,
                comparison_row_id=f"comparison_{fixture_key}_{metric}_v1",
                row_kind="metric_comparison",
                metric=metric,
                expected_value=format_float(float(expected)),
                project_value=observations[metric],
                baseline_value=baseline_observations[metric],
                delta_value=format_float(delta),
                tolerance_kind="absolute",
                tolerance_value=format_float(singular_tolerance),
                status="pass" if delta <= singular_tolerance else "fail",
                status_reason="project_baseline_singular_value_delta_within_tolerance"
                if delta <= singular_tolerance
                else "project_baseline_singular_value_delta_tolerance_miss",
                caveat=caveat,
            )
        )
    rows.append(
        study_row(
            context,
            comparison_row_id=f"comparison_{fixture_key}_singular_values_max_abs_delta_v1",
            row_kind="metric_comparison",
            metric="singular_values_max_abs_delta",
            expected_value=f"<={format_float(singular_tolerance)}",
            project_value=observations["singular_values_max_abs_delta"],
            baseline_value="0",
            delta_value=format_float(max_delta),
            tolerance_kind="absolute",
            tolerance_value=format_float(singular_tolerance),
            status="pass" if max_delta <= singular_tolerance else "fail",
            status_reason="project_baseline_singular_values_delta_within_tolerance"
            if max_delta <= singular_tolerance
            else "project_baseline_singular_values_delta_tolerance_miss",
            caveat=caveat,
        )
    )
    for metric, tolerance in (
        ("residual_norm", residual_tolerance),
        ("u_orthogonality", orthogonality_tolerance),
        ("v_orthogonality", orthogonality_tolerance),
        ("u_projector_diag", projector_tolerance),
        ("v_projector_diag", projector_tolerance),
    ):
        project_value = float(observations[metric])
        rows.append(
            study_row(
                context,
                comparison_row_id=f"comparison_{fixture_key}_{metric}_v1",
                row_kind="metric_comparison",
                metric=metric,
                expected_value=f"<={format_float(tolerance)}",
                project_value=observations[metric],
                baseline_value="0",
                delta_value=format_float(project_value),
                tolerance_kind="upper_bound",
                tolerance_value=format_float(tolerance),
                status="pass" if project_value <= tolerance else "fail",
                status_reason=f"project_{metric}_within_tolerance"
                if project_value <= tolerance
                else f"project_{metric}_tolerance_miss",
                caveat=caveat,
            )
        )
    return rows


def validate_selected_study_rows(rows: list[dict[str, str]], target: dict[str, object]) -> None:
    expected_ids = set(expected_study_row_ids(target))
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


def expected_study_row_ids(target: dict[str, object]) -> list[str]:
    fixture_key = str(target["fixture_key"])
    if target.get("comparison_kind") == "partial_svd":
        return [
            f"comparison_{fixture_key}_project_status_v1",
            f"comparison_{fixture_key}_baseline_status_v1",
            f"comparison_{fixture_key}_singular_value_0_v1",
            f"comparison_{fixture_key}_singular_value_1_v1",
            f"comparison_{fixture_key}_singular_values_max_abs_delta_v1",
            f"comparison_{fixture_key}_residual_norm_v1",
            f"comparison_{fixture_key}_u_orthogonality_v1",
            f"comparison_{fixture_key}_v_orthogonality_v1",
            f"comparison_{fixture_key}_u_projector_diag_v1",
            f"comparison_{fixture_key}_v_projector_diag_v1",
        ]
    return [
        f"comparison_{fixture_key}_project_status_v1",
        f"comparison_{fixture_key}_baseline_status_v1",
        f"comparison_{fixture_key}_residual_norm_v1",
        f"comparison_{fixture_key}_solution_norm_v1",
        f"comparison_{fixture_key}_solution_values_v1",
        f"comparison_{fixture_key}_project_vs_baseline_max_abs_delta_v1",
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
    for target in TARGETS.values():
        passing_rows = [
            {"comparison_row_id": row_id, "status": "pass", "status_reason": "self_check"}
            for row_id in expected_study_row_ids(target)
        ]
        validate_selected_study_rows(passing_rows, target)
        assert_comparison_error(
            "missing_selected_row",
            lambda target=target, passing_rows=passing_rows: validate_selected_study_rows(
                passing_rows[:-1], target
            ),
        )
        assert_comparison_error(
            "duplicate_selected_row",
            lambda target=target, passing_rows=passing_rows: validate_selected_study_rows(
                [*passing_rows, passing_rows[0]], target
            ),
        )
        failing_rows = [dict(row) for row in passing_rows]
        failing_rows[0]["status"] = "fail"
        failing_rows[0]["status_reason"] = "self_check_failure"
        assert_comparison_error(
            "metric_tolerance_miss",
            lambda target=target, failing_rows=failing_rows: validate_selected_study_rows(
                failing_rows, target
            ),
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
            },
            TARGETS["qr-minnorm"],
        ),
    )
    dependency_rows = dependency_status_rows(root, TARGETS["qr-minnorm"])
    deferred = {
        row["dependency"]: row["status"]
        for row in dependency_rows
        if row["dependency"] in {"numpy", "scipy"}
    }
    if deferred != {"numpy": "defer", "scipy": "defer"}:
        raise ComparisonError("self_check_failed", "optional package rows are not deferred")
    print("external-comparison: self-check passed")
    return 0


def write_summary(
    path: Path, rows: list[dict[str, str]], manifest: dict[str, str], target: dict[str, object]
) -> None:
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
                f"# {target['summary_title']}",
                "",
                "## Scope",
                "",
                str(target["summary_scope"]),
                "",
                "Non-claims: " + str(target.get("non_claims", NON_CLAIMS)),
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
    if args.target not in TARGETS:
        supported = ", ".join(sorted(TARGETS))
        raise ComparisonError(
            "unsupported_target",
            f"unsupported target {args.target!r}; supported targets: {supported}",
        )

    target = TARGETS[args.target]
    root = args.root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else Path(target["output_dir"]).resolve()
    library = args.library.resolve()
    reset_output_dir(output_dir)

    generated_at = utc_timestamp()
    observations, compiler = run_project_probe(
        root,
        library,
        args.keep_temp,
        target,
        probe_build_system=args.probe_build_system,
        cmake_generator=args.cmake_generator,
        cmake_arch=args.cmake_arch,
        cmake_config=args.cmake_config,
    )
    baseline_observations = run_baseline_reference(root, target)
    observation_rows = project_observation_rows(observations, target)
    baseline_rows = baseline_observation_rows(baseline_observations, target)
    dependency_rows = dependency_status_rows(root, target)

    project_observations_path = output_dir / "project_observations.tsv"
    baseline_observations_path = output_dir / "baseline_observations.tsv"
    dependency_status_path = output_dir / "dependency_status.tsv"
    study_path = output_dir / "study.tsv"
    summary_path = output_dir / "summary.md"
    manifest_path = output_dir / "manifest.tsv"
    manifest = {
        "target": args.target,
        "fixture_key": str(target["fixture_key"]),
        "generated_at_utc": generated_at,
        "source_commit": current_commit(root),
        "source_branch": current_branch(root),
        "worktree_state": worktree_state(root),
        "project_version": project_version(root),
        "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        "compiler": compiler,
        "configuration": comparison_configuration(target),
        "baseline_name": baseline_name(target),
        "baseline_type": "external-process-source-controlled-helper",
        "baseline_version": baseline_version(target),
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
        target=target,
    )
    validate_selected_study_rows(study_rows, target)

    write_tsv(project_observations_path, PROJECT_OBSERVATION_FIELDS, observation_rows)
    write_tsv(baseline_observations_path, BASELINE_OBSERVATION_FIELDS, baseline_rows)
    write_tsv(dependency_status_path, DEPENDENCY_STATUS_FIELDS, dependency_rows)
    write_tsv(study_path, STUDY_FIELDS, study_rows)
    write_summary(summary_path, study_rows, manifest, target)
    write_manifest(manifest_path, manifest)

    print(f"external-comparison: wrote {project_observations_path}")
    print(f"external-comparison: wrote {baseline_observations_path}")
    print(f"external-comparison: wrote {dependency_status_path}")
    print(f"external-comparison: wrote {study_path}")
    print(f"external-comparison: wrote {summary_path}")
    print(f"external-comparison: wrote {manifest_path}")
    print(str(target["success_message"]))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        help=f"comparison target; supported: {', '.join(sorted(TARGETS))}",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--probe-build-system",
        choices=("auto", "compiler", "cmake"),
        default="auto",
        help="build temporary project probe directly with CC or through CMake",
    )
    parser.add_argument("--cmake-generator", default=None)
    parser.add_argument("--cmake-arch", default=None)
    parser.add_argument("--cmake-config", default=DEFAULT_CMAKE_CONFIG)
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
