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
PARTIAL_SVD_FIXTURE_KEY = "partial_svd_clustered_repeated_diag8x6_k3_v1"
PARTIAL_SVD_GENERATOR_KEY = "partial_svd_clustered_repeated_diag8x6_generator_v1"
FIRST_LANE_ORACLE_ROW_IDS = {
    f"{FIXTURE_KEY}_rank",
    f"{FIXTURE_KEY}_nullity",
    f"{FIXTURE_KEY}_projector_residual",
}
PARTIAL_SVD_ORACLE_ROW_IDS = {
    f"{PARTIAL_SVD_FIXTURE_KEY}_singular_values",
    f"{PARTIAL_SVD_FIXTURE_KEY}_left_subspace",
    f"{PARTIAL_SVD_FIXTURE_KEY}_right_subspace",
    f"{PARTIAL_SVD_FIXTURE_KEY}_vector_residual",
    f"{PARTIAL_SVD_FIXTURE_KEY}_orthogonality",
    f"{PARTIAL_SVD_FIXTURE_KEY}_default_status",
    f"{PARTIAL_SVD_FIXTURE_KEY}_tight_budget_status",
    f"{PARTIAL_SVD_FIXTURE_KEY}_tight_budget_no_partial_arrays",
}
PARTIAL_SVD_GENERATED_FIXTURES = {
    PARTIAL_SVD_FIXTURE_KEY: {
        "generator_key": PARTIAL_SVD_GENERATOR_KEY,
        "fixture_label": "clustered_repeated_diag8x6_k3",
        "value_tolerance": "1e-8",
        "projector_tolerance": "1e-8",
        "residual_tolerance": "1e-8",
        "oracle_row_ids": PARTIAL_SVD_ORACLE_ROW_IDS,
        "observations": {
            f"{PARTIAL_SVD_FIXTURE_KEY}_singular_values": (
                "top_k=10,10,9.999999;max_abs_error=0"
            ),
            f"{PARTIAL_SVD_FIXTURE_KEY}_left_subspace": "left_projector_distance=0",
            f"{PARTIAL_SVD_FIXTURE_KEY}_right_subspace": "right_projector_distance=0",
            f"{PARTIAL_SVD_FIXTURE_KEY}_vector_residual": "max_triplet_residual=0",
            f"{PARTIAL_SVD_FIXTURE_KEY}_orthogonality": "max_orthogonality_residual=0",
            f"{PARTIAL_SVD_FIXTURE_KEY}_default_status": "SPARSE_SUCCESS",
            f"{PARTIAL_SVD_FIXTURE_KEY}_tight_budget_status": "SPARSE_ERR_NOT_CONVERGED",
            f"{PARTIAL_SVD_FIXTURE_KEY}_tight_budget_no_partial_arrays": (
                "no_partial_sigma_u_vt_on_failure"
            ),
        },
    },
    "partial_svd_rankdef_diag6x4_k2_range_projector_v1": {
        "generator_key": "partial_svd_rankdef_diag6x4_k2_range_projector_generator_v1",
        "fixture_label": "rankdef_diag6x4_k2_range_projector",
        "value_tolerance": "1e-8",
        "projector_tolerance": "1e-8",
        "residual_tolerance": "1e-8",
        "oracle_row_ids": {
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_default_status",
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_singular_values",
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_rank",
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_left_subspace",
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_right_subspace",
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_vector_residuals",
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_orthogonality",
        },
        "observations": {
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_default_status": (
                "SPARSE_SUCCESS"
            ),
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_singular_values": (
                "top_k=9,6;max_abs_error=0"
            ),
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_rank": "2",
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_left_subspace": (
                "left_projector_distance=0"
            ),
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_right_subspace": (
                "right_projector_distance=0"
            ),
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_vector_residuals": (
                "max_triplet_residual=0"
            ),
            "partial_svd_rankdef_diag6x4_k2_range_projector_v1_orthogonality": (
                "max_orthogonality_residual=0"
            ),
        },
    },
    "partial_svd_lowrank_rect5x7_k3_sparse_output_v1": {
        "generator_key": "partial_svd_lowrank_rect5x7_k3_sparse_output_generator_v1",
        "fixture_label": "lowrank_rect5x7_k3_sparse_output",
        "value_tolerance": "1e-10",
        "projector_tolerance": "not_applicable",
        "residual_tolerance": "1e-10",
        "oracle_row_ids": {
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_status",
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_shape",
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_nnz",
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_selected_values",
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_dense_frobenius_error",
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_dense_frobenius_diff",
        },
        "observations": {
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_status": (
                "SPARSE_SUCCESS"
            ),
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_shape": "shape=5x7",
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_nnz": "3",
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_selected_values": (
                "selected_values=8,4,2,0;max_abs_error=0"
            ),
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_dense_frobenius_error": (
                "dense_frobenius_abs_error=0"
            ),
            "partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_dense_frobenius_diff": (
                "sparse_dense_frobenius_diff=0"
            ),
        },
    },
    "partial_svd_fail_closed_diag6_k2_v1": {
        "generator_key": "partial_svd_fail_closed_diag6_k2_generator_v1",
        "fixture_label": "fail_closed_diag6_k2",
        "value_tolerance": "1e-8",
        "projector_tolerance": "not_applicable",
        "residual_tolerance": "1e-8",
        "oracle_row_ids": {
            "partial_svd_fail_closed_diag6_k2_v1_tight_budget_status",
            "partial_svd_fail_closed_diag6_k2_v1_tight_budget_no_partial_arrays",
            "partial_svd_fail_closed_diag6_k2_v1_recovery_status",
            "partial_svd_fail_closed_diag6_k2_v1_default_singular_values",
            "partial_svd_fail_closed_diag6_k2_v1_default_vector_residuals",
        },
        "observations": {
            "partial_svd_fail_closed_diag6_k2_v1_tight_budget_status": (
                "SPARSE_ERR_NOT_CONVERGED"
            ),
            "partial_svd_fail_closed_diag6_k2_v1_tight_budget_no_partial_arrays": (
                "no_partial_sigma_u_vt_on_failure"
            ),
            "partial_svd_fail_closed_diag6_k2_v1_recovery_status": "SPARSE_SUCCESS",
            "partial_svd_fail_closed_diag6_k2_v1_default_singular_values": (
                "top_k=9,6;max_abs_error=0"
            ),
            "partial_svd_fail_closed_diag6_k2_v1_default_vector_residuals": (
                "max_triplet_residual=0"
            ),
        },
    },
}
SOLVER_QR_ORACLE_ROW_IDS = {
    f"{FIXTURE_KEY}_qr_rank": f"{FIXTURE_KEY}_rank",
    f"{FIXTURE_KEY}_qr_nullity": f"{FIXTURE_KEY}_nullity",
    f"{FIXTURE_KEY}_qr_nullspace_residual": f"{FIXTURE_KEY}_projector_residual",
}
SPRINT150_RANKDEF_QR_FIXTURES = {
    "qr_rankdef_duplicate_5x4_v1": {
        "generator_key": "qr_rankdef_duplicate_5x4_generator_v1",
        "expected_row_ids": {
            "rank": "qr_rankdef_duplicate_5x4_v1_rank",
            "nullity": "qr_rankdef_duplicate_5x4_v1_nullity",
            "nullspace_residual": "qr_rankdef_duplicate_5x4_v1_nullspace_residual",
            "nullspace_subspace": "qr_rankdef_duplicate_5x4_v1_nullspace_subspace",
        },
        "reference_null_vector": [0.0, -1.0, 0.0, 1.0],
    },
    "qr_rankdef_dependent_row_4x3_v1": {
        "generator_key": "qr_rankdef_dependent_row_4x3_generator_v1",
        "expected_row_ids": {
            "rank": "qr_rankdef_dependent_row_4x3_v1_rank",
            "nullity": "qr_rankdef_dependent_row_4x3_v1_nullity",
            "nullspace_residual": "qr_rankdef_dependent_row_4x3_v1_nullspace_residual",
            "nullspace_subspace": "qr_rankdef_dependent_row_4x3_v1_nullspace_subspace",
        },
        "reference_null_vector": [-1.0, -2.0, 1.0],
    },
}
SPRINT150_MINNORM_QR_FIXTURES = {
    "qr_underdetermined_minnorm_2x4": {
        "generator_key": "qr_underdetermined_minnorm_2x4_generator_v1",
        "expected_row_ids": {
            "status": "qr_underdetermined_minnorm_2x4_status",
            "residual": "qr_underdetermined_minnorm_2x4_residual",
            "solution_norm": "qr_underdetermined_minnorm_2x4_solution_norm",
            "solution_values": "qr_underdetermined_minnorm_2x4_solution_values",
        },
        "rhs": [1.0, 1.0],
    },
    "qr_minnorm_3x6_exact_values": {
        "generator_key": "qr_minnorm_3x6_exact_values_generator_v1",
        "expected_row_ids": {
            "status": "qr_minnorm_3x6_exact_values_status",
            "residual": "qr_minnorm_3x6_exact_values_residual",
            "solution_norm": "qr_minnorm_3x6_exact_values_solution_norm",
            "solution_values": "qr_minnorm_3x6_exact_values_solution_values",
        },
        "rhs": [3.0, 4.0, 5.0],
    },
    "qr_minnorm_5x10_exact_values": {
        "generator_key": "qr_minnorm_5x10_exact_values_generator_v1",
        "expected_row_ids": {
            "status": "qr_minnorm_5x10_exact_values_status",
            "residual": "qr_minnorm_5x10_exact_values_residual",
            "solution_norm": "qr_minnorm_5x10_exact_values_solution_norm",
            "solution_values": "qr_minnorm_5x10_exact_values_solution_values",
        },
        "rhs": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
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


def load_expected_rows(
    root: Path, fixture_key: str, required_oracle_row_ids: set[str]
) -> dict[str, dict[str, str]]:
    path = root / "expected" / f"{fixture_key}.tsv"
    rows = read_tsv(path)
    expected_by_id: dict[str, dict[str, str]] = {}
    for line, row in enumerate(rows, start=2):
        oracle_row_id = row["oracle_row_id"]
        if oracle_row_id in expected_by_id:
            raise CorpusValidationError(
                f"{path}:{line}: duplicate oracle_row_id {oracle_row_id!r}"
            )
        if oracle_row_id in required_oracle_row_ids and row["status"] != "ready_for_oracle":
            raise CorpusValidationError(
                f"{path}:{line}: required oracle row {oracle_row_id!r} "
                f"must have status 'ready_for_oracle', got {row['status']!r}"
            )
        expected_by_id[oracle_row_id] = row
    missing = sorted(required_oracle_row_ids - set(expected_by_id))
    if missing:
        raise CorpusValidationError(
            f"{path}: missing required expected oracle rows: {', '.join(missing)}"
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


def c_literal_for_values(values: list[float]) -> str:
    return ", ".join(f"{value:.17g}" for value in values)


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


def qr_rankdef_probe_source(
    entries: list[tuple[int, int, float]], rows: int, cols: int, reference: list[float]
) -> str:
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

static const double reference_null_vector[{cols}] = {{{c_literal_for_values(reference)}}};

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

    double basis[{cols}] = {{0.0}};
    double normalized_residual = INFINITY;
    double projector_distance = INFINITY;
    if (nullity == 1) {{
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

        double basis_norm_sq = 0.0;
        double ref_norm_sq = 0.0;
        for (idx_t col = 0; col < {cols}; ++col) {{
            basis_norm_sq += basis[col] * basis[col];
            ref_norm_sq += reference_null_vector[col] * reference_null_vector[col];
        }}
        if (basis_norm_sq > 0.0 && ref_norm_sq > 0.0) {{
            normalized_residual = sqrt(residual_sq) / sqrt(basis_norm_sq);
            for (idx_t row = 0; row < {cols}; ++row) {{
                for (idx_t col = 0; col < {cols}; ++col) {{
                    double observed = basis[row] * basis[col] / basis_norm_sq;
                    double expected =
                        reference_null_vector[row] * reference_null_vector[col] / ref_norm_sq;
                    double diff = fabs(observed - expected);
                    if (diff > projector_distance || !isfinite(projector_distance))
                        projector_distance = diff;
                }}
            }}
        }}
    }}

    printf(\"rank=%d\\n\", (int)rank);
    printf(\"nullity=%d\\n\", (int)nullity);
    printf(\"normalized_null_vector_residual=%.17g\\n\", normalized_residual);
    printf(\"projector_distance=%.17g\\n\", projector_distance);

    sparse_qr_free(&qr);
    sparse_free(A);
    return 0;
}}
"""


def qr_minnorm_probe_source(
    entries: list[tuple[int, int, float]], rows: int, cols: int, rhs: list[float]
) -> str:
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


def parse_probe_output(output: str, required: set[str] | None = None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    required = required or {"rank", "nullity", "normalized_residual"}
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


def run_solver_probe_source(
    source_text: str, library: Path, required_fields: set[str]
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
        source.write_text(source_text)
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
        return parse_probe_output(completed.stdout, required_fields), compiler_identity(cc)


def parse_key_values(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise CorpusValidationError(f"malformed key/value observation {text!r}")
        key, value = chunk.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_float(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise CorpusValidationError(f"{context}: expected floating-point value, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise CorpusValidationError(f"{context}: expected finite value, got {value!r}")
    return parsed


def parse_vector(value: str, context: str) -> list[float]:
    if value == "":
        raise CorpusValidationError(f"{context}: empty vector")
    return [parse_float(part.strip(), context) for part in value.split(",")]


def scalar_from_observed(observed: str, expected_prefix: str) -> float:
    if "=" not in observed:
        return parse_float(observed, "observed scalar")
    parsed = parse_key_values(observed)
    if expected_prefix in parsed:
        return parse_float(parsed[expected_prefix], expected_prefix)
    if len(parsed) == 1:
        key, value = next(iter(parsed.items()))
        return parse_float(value, key)
    raise CorpusValidationError(
        f"observed result {observed!r} does not contain expected key {expected_prefix!r}"
    )


def compare(expected: dict[str, str], observed: str) -> tuple[str, str]:
    kind = expected["comparison_kind"]
    if kind in {"rank", "nullity"}:
        if expected["tolerance_kind"] != "exact":
            raise CorpusValidationError(
                f"{expected['oracle_row_id']}: {kind} comparison requires tolerance_kind='exact'"
            )
        passed = int(observed) == int(expected["expected_result"])
    elif kind == "value":
        if expected["tolerance_kind"] != "absolute":
            raise CorpusValidationError(
                f"{expected['oracle_row_id']}: value comparison requires "
                "tolerance_kind='absolute'"
            )
        tolerance = parse_float(expected["tolerance_value"], "value tolerance")
        expected_values = parse_key_values(expected["expected_result"])
        observed_values = parse_key_values(observed)
        if "top_k" in expected_values or "top_k" in observed_values:
            if "top_k" not in expected_values or "top_k" not in observed_values:
                raise CorpusValidationError(
                    f"{expected['oracle_row_id']}: value comparison requires top_k fields"
                )
            expected_top_k = sorted(
                parse_vector(expected_values["top_k"], "expected top_k"), reverse=True
            )
            observed_top_k = sorted(
                parse_vector(observed_values["top_k"], "observed top_k"), reverse=True
            )
            if len(expected_top_k) != len(observed_top_k):
                raise CorpusValidationError(
                    f"{expected['oracle_row_id']}: observed top_k length mismatch"
                )
            max_abs_error = max(
                abs(expected_value - observed_value)
                for expected_value, observed_value in zip(expected_top_k, observed_top_k)
            )
            passed = max_abs_error <= tolerance
            if "max_abs_error" in observed_values:
                reported_error = parse_float(observed_values["max_abs_error"], "max_abs_error")
                if abs(reported_error - max_abs_error) > max(1e-15, tolerance * 1e-6):
                    raise CorpusValidationError(
                        f"{expected['oracle_row_id']}: reported max_abs_error mismatch"
                    )
        elif "solution_norm" in expected_values or "solution_norm" in observed_values:
            if "solution_norm" not in expected_values or "solution_norm" not in observed_values:
                raise CorpusValidationError(
                    f"{expected['oracle_row_id']}: value comparison requires solution_norm fields"
                )
            expected_norm = parse_float(expected_values["solution_norm"], "expected solution_norm")
            observed_norm = parse_float(observed_values["solution_norm"], "observed solution_norm")
            passed = abs(expected_norm - observed_norm) <= tolerance
        elif (
            "solution_values" in expected_values
            or "solution_values" in observed_values
            or "selected_values" in expected_values
            or "selected_values" in observed_values
        ):
            vector_field = (
                "selected_values"
                if "selected_values" in expected_values or "selected_values" in observed_values
                else "solution_values"
            )
            if vector_field not in expected_values or vector_field not in observed_values:
                raise CorpusValidationError(
                    f"{expected['oracle_row_id']}: value comparison requires {vector_field} fields"
                )
            expected_solution = parse_vector(
                expected_values[vector_field], f"expected {vector_field}"
            )
            observed_solution = parse_vector(
                observed_values[vector_field], f"observed {vector_field}"
            )
            if len(expected_solution) != len(observed_solution):
                raise CorpusValidationError(
                    f"{expected['oracle_row_id']}: observed {vector_field} length mismatch"
                )
            max_abs_error = max(
                abs(expected_value - observed_value)
                for expected_value, observed_value in zip(expected_solution, observed_solution)
            )
            passed = max_abs_error <= tolerance
            if "max_abs_error" in observed_values:
                reported_error = parse_float(observed_values["max_abs_error"], "max_abs_error")
                if abs(reported_error - max_abs_error) > max(1e-15, tolerance * 1e-6):
                    raise CorpusValidationError(
                        f"{expected['oracle_row_id']}: reported max_abs_error mismatch"
                    )
        else:
            raise CorpusValidationError(
                f"{expected['oracle_row_id']}: unsupported value comparison fields"
            )
    elif kind == "subspace_distance":
        if expected["tolerance_kind"] != "projector":
            raise CorpusValidationError(
                f"{expected['oracle_row_id']}: subspace_distance comparison requires "
                "tolerance_kind='projector'"
            )
        tolerance = parse_float(expected["tolerance_value"], "subspace tolerance")
        expected_metric = expected["expected_result"].split("<=", 1)[0]
        passed = scalar_from_observed(observed, expected_metric) <= tolerance
    elif kind == "residual_norm":
        if expected["tolerance_kind"] != "absolute":
            raise CorpusValidationError(
                f"{expected['oracle_row_id']}: residual_norm comparison requires "
                "tolerance_kind='absolute'"
            )
        tolerance = parse_float(expected["tolerance_value"], "residual tolerance")
        expected_metric = expected["expected_result"].split("<=", 1)[0]
        passed = scalar_from_observed(observed, expected_metric) <= tolerance
    elif kind == "status":
        if expected["tolerance_kind"] != "status_only" or expected["tolerance_value"] != "":
            raise CorpusValidationError(
                f"{expected['oracle_row_id']}: status comparison requires "
                "tolerance_kind='status_only' and empty tolerance_value"
            )
        passed = observed == expected["expected_result"]
    elif kind == "diagnostic":
        if expected["tolerance_kind"] != "not_applicable" or expected["tolerance_value"] != "":
            raise CorpusValidationError(
                f"{expected['oracle_row_id']}: diagnostic comparison requires "
                "tolerance_kind='not_applicable' and empty tolerance_value"
            )
        passed = observed == expected["expected_result"]
    else:
        raise CorpusValidationError(f"unsupported comparison kind: {kind}")
    return ("pass", "") if passed else ("fail", "fail_oracle_mismatch")


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reset_generated_outputs(oracle_dir: Path, report_dir: Path) -> None:
    oracle_dir.mkdir(parents=True, exist_ok=True)
    for path in oracle_dir.glob("*.tsv"):
        path.unlink()
    for name in ("index.tsv", "skips.tsv", "manifest.txt"):
        path = report_dir / name
        if path.exists():
            path.unlink()


def build_oracle_rows(root: Path, command: str) -> list[dict[str, str]]:
    validate(root)
    fixture = GENERATED_FIXTURES[GENERATOR_KEY]
    entries = fixture["entries"]()
    structure_hash = sha256_text(
        canonical_structure_text(fixture["rows"], fixture["cols"], entries)
    )
    value_hash = sha256_text(canonical_value_text(fixture["rows"], fixture["cols"], entries))
    expected = load_expected_rows(root, FIXTURE_KEY, FIRST_LANE_ORACLE_ROW_IDS)
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


def partial_svd_configuration(
    fixture_key: str,
    generator_key: str,
    fixture_info: dict[str, object],
) -> str:
    fixture = GENERATED_FIXTURES[generator_key]
    entries = fixture["entries"]()
    structure_hash = sha256_text(
        canonical_structure_text(fixture["rows"], fixture["cols"], entries)
    )
    value_hash = sha256_text(canonical_value_text(fixture["rows"], fixture["cols"], entries))
    return (
        "build_profile=static_default;optional_data_policy=disabled;"
        "proof_owner=generated_partial_svd_reference;solver_execution=none;"
        f"partial_svd_fixture={fixture_info['fixture_label']};fixture_key={fixture_key};"
        f"structure_hash={structure_hash};value_hash={value_hash};"
        f"value_tolerance={fixture_info['value_tolerance']};"
        f"projector_tolerance={fixture_info['projector_tolerance']};"
        f"residual_tolerance={fixture_info['residual_tolerance']}"
    )


def build_partial_svd_oracle_rows(
    root: Path, command: str, *, validate_root: bool = True
) -> list[dict[str, str]]:
    if validate_root:
        validate(root)
    now = utc_timestamp()
    commit = run_text(["git", "rev-parse", "HEAD"])
    branch = current_source_branch()
    platform_name = f"{platform.system().lower()}-{platform.machine().lower()}"
    rows: list[dict[str, str]] = []
    for fixture_key, fixture_info in PARTIAL_SVD_GENERATED_FIXTURES.items():
        generator_key = str(fixture_info["generator_key"])
        oracle_row_ids = set(fixture_info["oracle_row_ids"])
        observations = dict(fixture_info["observations"])
        observation_ids = set(observations)
        missing_observations = sorted(oracle_row_ids - observation_ids)
        extra_observations = sorted(observation_ids - oracle_row_ids)
        if missing_observations or extra_observations:
            details = []
            if missing_observations:
                details.append(f"missing observations: {', '.join(missing_observations)}")
            if extra_observations:
                details.append(f"extra observations: {', '.join(extra_observations)}")
            raise CorpusValidationError(
                f"partial-SVD oracle row mismatch for {fixture_key!r}: {'; '.join(details)}"
            )
        expected = load_expected_rows(root, fixture_key, oracle_row_ids)
        configuration = partial_svd_configuration(fixture_key, generator_key, fixture_info)
        for oracle_row_id in sorted(oracle_row_ids):
            if oracle_row_id not in expected:
                raise CorpusValidationError(
                    f"missing expected result for partial-SVD oracle row {oracle_row_id!r}"
                )
            expected_row = expected[oracle_row_id]
            status, failure_class = compare(expected_row, observations[oracle_row_id])
            rows.append(
                {
                    "oracle_row_id": oracle_row_id,
                    "fixture_key": fixture_key,
                    "solver_family": "partial_svd",
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
    expected = load_expected_rows(root, FIXTURE_KEY, FIRST_LANE_ORACLE_ROW_IDS)
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


def row_metadata_for_fixture(
    fixture_key: str,
    generator_key: str,
    command: str,
    compiler: str,
    operation_family: str,
) -> dict[str, str]:
    fixture = GENERATED_FIXTURES[generator_key]
    entries = fixture["entries"]()
    structure_hash = sha256_text(
        canonical_structure_text(fixture["rows"], fixture["cols"], entries)
    )
    value_hash = sha256_text(canonical_value_text(fixture["rows"], fixture["cols"], entries))
    return {
        "fixture_key": fixture_key,
        "command": command,
        "source_commit": run_text(["git", "rev-parse", "HEAD"]),
        "source_branch": current_source_branch(),
        "generated_at_utc": utc_timestamp(),
        "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        "compiler": compiler,
        "configuration": (
            "build_profile=static_default;optional_data_policy=disabled;"
            f"proof_owner=runtime_qr_probe;operation_family={operation_family};"
            f"structure_hash={structure_hash};value_hash={value_hash};qr_tolerance=1e-10"
        ),
        "support_tier": "local_only",
    }


def build_sprint150_rankdef_qr_rows(root: Path, command: str, library: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fixture_key, fixture_info in SPRINT150_RANKDEF_QR_FIXTURES.items():
        generator_key = fixture_info["generator_key"]
        fixture = GENERATED_FIXTURES[generator_key]
        entries = fixture["entries"]()
        expected = load_expected_rows(
            root, fixture_key, set(fixture_info["expected_row_ids"].values())
        )
        observations, compiler = run_solver_probe_source(
            qr_rankdef_probe_source(
                entries,
                fixture["rows"],
                fixture["cols"],
                fixture_info["reference_null_vector"],
            ),
            library,
            {"rank", "nullity", "normalized_null_vector_residual", "projector_distance"},
        )
        metadata = row_metadata_for_fixture(
            fixture_key, generator_key, command, compiler, "rankdef_nullspace"
        )
        observation_by_key = {
            "rank": observations["rank"],
            "nullity": observations["nullity"],
            "nullspace_residual": (
                f"normalized_null_vector_residual={observations['normalized_null_vector_residual']}"
            ),
            "nullspace_subspace": f"projector_distance={observations['projector_distance']}",
        }
        for suffix, observed in sorted(observation_by_key.items()):
            expected_row_id = fixture_info["expected_row_ids"][suffix]
            expected_row = expected[expected_row_id]
            status, failure_class = compare(expected_row, observed)
            rows.append(
                {
                    "oracle_row_id": f"{fixture_key}_qr_{suffix}",
                    "fixture_key": fixture_key,
                    "solver_family": "qr",
                    "operation": expected_row["operation"],
                    "comparison_kind": expected_row["comparison_kind"],
                    "expected_result_kind": expected_row["expected_result_kind"],
                    "expected_result": expected_row["expected_result"],
                    "observed_result": observed,
                    "tolerance_kind": expected_row["tolerance_kind"],
                    "tolerance_value": expected_row["tolerance_value"],
                    "comparison_status": status,
                    "failure_class": failure_class,
                    "skip_or_defer_reason": "",
                    "claim_scope": (
                        "Fixture-local solver-backed rank-deficient rectangular QR evidence."
                    ),
                    "non_claims": (
                        "no broad QR correctness; no raw-basis parity; "
                        "no sign/orientation/column-order parity; no global rank-threshold policy; "
                        "no external-library parity; no platform/package/ABI/performance or "
                        "state-of-the-art claim"
                    ),
                    **metadata,
                }
            )
    return rows


def build_sprint150_minnorm_qr_rows(root: Path, command: str, library: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fixture_key, fixture_info in SPRINT150_MINNORM_QR_FIXTURES.items():
        generator_key = fixture_info["generator_key"]
        fixture = GENERATED_FIXTURES[generator_key]
        entries = fixture["entries"]()
        expected = load_expected_rows(
            root, fixture_key, set(fixture_info["expected_row_ids"].values())
        )
        observations, compiler = run_solver_probe_source(
            qr_minnorm_probe_source(
                entries,
                fixture["rows"],
                fixture["cols"],
                fixture_info["rhs"],
            ),
            library,
            {"status", "residual_norm", "solution_norm", "solution_values"},
        )
        metadata = row_metadata_for_fixture(
            fixture_key, generator_key, command, compiler, "minnorm_solve"
        )
        solution_values = observations["solution_values"]
        observation_by_key = {
            "residual": f"residual_norm={observations['residual_norm']}",
            "solution_norm": f"solution_norm={observations['solution_norm']}",
            "solution_values": f"solution_values={solution_values}",
            "status": observations["status"],
        }
        expected_values_row = expected[fixture_info["expected_row_ids"]["solution_values"]]
        expected_solution = parse_key_values(expected_values_row["expected_result"])[
            "solution_values"
        ]
        max_abs_error = max(
            abs(expected_value - observed_value)
            for expected_value, observed_value in zip(
                parse_vector(expected_solution, f"{fixture_key} expected solution_values"),
                parse_vector(solution_values, f"{fixture_key} observed solution_values"),
            )
        )
        observation_by_key["solution_values"] = (
            f"solution_values={solution_values};max_abs_error={max_abs_error:.17g}"
        )
        for suffix, observed in sorted(observation_by_key.items()):
            expected_row_id = fixture_info["expected_row_ids"][suffix]
            expected_row = expected[expected_row_id]
            status, failure_class = compare(expected_row, observed)
            rows.append(
                {
                    "oracle_row_id": f"{fixture_key}_qr_{suffix}",
                    "fixture_key": fixture_key,
                    "solver_family": "qr",
                    "operation": expected_row["operation"],
                    "comparison_kind": expected_row["comparison_kind"],
                    "expected_result_kind": expected_row["expected_result_kind"],
                    "expected_result": expected_row["expected_result"],
                    "observed_result": observed,
                    "tolerance_kind": expected_row["tolerance_kind"],
                    "tolerance_value": expected_row["tolerance_value"],
                    "comparison_status": status,
                    "failure_class": failure_class,
                    "skip_or_defer_reason": "",
                    "claim_scope": (
                        "Fixture-local solver-backed underdetermined minimum-norm QR evidence."
                    ),
                    "non_claims": (
                        "no global minimum-norm guarantee; no SVD pseudoinverse global-oracle "
                        "claim; no broad rank-deficient recovery claim; no broad inconsistent-"
                        "system behavior claim; no external-library parity; no platform/package/"
                        "ABI/performance or state-of-the-art claim"
                    ),
                    **metadata,
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
    partial_svd_row_count = sum(
        1 for row in oracle_rows if row["solver_family"] == "partial_svd"
    )
    fixture_keys = ",".join(sorted({row["fixture_key"] for row in oracle_rows}))
    configurations = " | ".join(sorted({row["configuration"] for row in oracle_rows}))
    compilers = ",".join(sorted({row["compiler"] for row in oracle_rows}))
    path.write_text(
        "\n".join(
            [
                "corpus-oracle-report",
                f"generated_at_utc={first['generated_at_utc']}",
                f"source_commit={first['source_commit']}",
                f"source_branch={first['source_branch']}",
                f"platform={first['platform']}",
                f"compiler={compilers}",
                f"configuration={configurations}",
                f"oracle_row_count={len(oracle_rows)}",
                f"solver_families={solver_families}",
                f"solver_qr_row_count={solver_qr_row_count}",
                f"partial_svd_row_count={partial_svd_row_count}",
                f"command={command}",
                f"fixture_keys={fixture_keys}",
                "support_tier=local_only",
                "claim_boundary=fixture-local corpus/oracle evidence only",
                "non_claims=no broad QR correctness; no broad partial-SVD correctness; "
                "no raw singular-vector identity; no SuiteSparse parity; "
                "no external-library parity; no broad corpus completeness; "
                "no performance or state-of-the-art claim",
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
        "--include-partial-svd",
        action="store_true",
        help="append generated-reference partial-SVD oracle rows for maintained fixtures",
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
        oracle_rows.extend(
            build_sprint150_rankdef_qr_rows(args.root, command, args.solver_library)
        )
        oracle_rows.extend(
            build_sprint150_minnorm_qr_rows(args.root, command, args.solver_library)
        )
    if args.include_partial_svd:
        oracle_rows.extend(
            build_partial_svd_oracle_rows(args.root, command, validate_root=False)
        )
    skip_rows = build_skip_rows(args.root)
    oracle_name = "corpus.oracle.tsv" if args.include_partial_svd else f"{FIXTURE_KEY}.oracle.tsv"
    oracle_path = args.oracle_dir / oracle_name
    report_path = args.report_dir / "index.tsv"
    manifest_path = args.report_dir / "manifest.txt"
    skip_path = args.report_dir / "skips.tsv"
    reset_generated_outputs(args.oracle_dir, args.report_dir)
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
