/*
 * Sprint 17 tests for the CSC working format for LDL^T.
 *
 * Covers the `LdltCsc` struct, alloc/free helpers, sparse conversion
 * and round-trip routines, and the Day 8-9 Bunch-Kaufman elimination
 * and native-CSC solve paths built on top of the CSC working format
 * (including linked-list cross-checks).
 */

#include "sparse_chol_csc_internal.h"
#include "sparse_ldlt.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Test helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compare two lower-triangular SparseMatrices entry-by-entry.  Only
 * (i, j) with i >= j are inspected.  Shape mismatch is fatal (return
 * early) so a mismatched test stops at the root cause instead of
 * cascading into thousands of follow-on failures. */
static void assert_lower_triangle_equal(const SparseMatrix *A, const SparseMatrix *B, double tol) {
    idx_t n = sparse_rows(A);
    if (sparse_rows(A) != sparse_rows(B) || sparse_cols(A) != sparse_cols(B)) {
        TF_FAIL_("Shape mismatch: A is %dx%d, B is %dx%d", (int)sparse_rows(A), (int)sparse_cols(A),
                 (int)sparse_rows(B), (int)sparse_cols(B));
        return;
    }
    tf_asserts += 2; /* account for the shape checks we'd have done */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double a = sparse_get(A, i, j);
            double b = sparse_get(B, i, j);
            if (fabs(a - b) > tol) {
                TF_FAIL_("Lower-tri (%d,%d): A=%.15g B=%.15g diff=%.3e > tol=%.3e", (int)i, (int)j,
                         a, b, fabs(a - b), tol);
            }
            tf_asserts++;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * alloc / free
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_alloc_null_out(void) { ASSERT_ERR(ldlt_csc_alloc(3, 4, NULL), SPARSE_ERR_NULL); }

static void test_alloc_negative_n(void) {
    LdltCsc *m = NULL;
    ASSERT_ERR(ldlt_csc_alloc(-1, 4, &m), SPARSE_ERR_BADARG);
    ASSERT_NULL(m);
}

static void test_alloc_basic(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(5, 10, &m));
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->n, 5);
    ASSERT_NOT_NULL(m->L);
    ASSERT_EQ(m->L->n, 5);
    ASSERT_NOT_NULL(m->D);
    ASSERT_NOT_NULL(m->D_offdiag);
    ASSERT_NOT_NULL(m->pivot_size);
    ASSERT_NOT_NULL(m->perm);

    /* Defaults: D/D_offdiag zero, pivot_size = 1 everywhere, perm = identity. */
    for (idx_t i = 0; i < 5; i++) {
        ASSERT_NEAR(m->D[i], 0.0, 0.0);
        ASSERT_NEAR(m->D_offdiag[i], 0.0, 0.0);
        ASSERT_EQ(m->pivot_size[i], 1);
        ASSERT_EQ(m->perm[i], i);
    }
    ldlt_csc_free(m);
}

static void test_alloc_zero_n(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(0, 1, &m));
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->n, 0);
    ldlt_csc_free(m);
}

static void test_free_null(void) {
    ldlt_csc_free(NULL); /* must not crash */
    ASSERT_TRUE(1);
}

/* ═══════════════════════════════════════════════════════════════════════
 * from_sparse: null / shape / conversion
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_from_sparse_null_args(void) {
    LdltCsc *m = NULL;
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);

    ASSERT_ERR(ldlt_csc_from_sparse(NULL, NULL, 2.0, &m), SPARSE_ERR_NULL);
    ASSERT_NULL(m);
    ASSERT_ERR(ldlt_csc_from_sparse(A, NULL, 2.0, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

static void test_from_sparse_shape(void) {
    LdltCsc *m = NULL;
    SparseMatrix *rect = sparse_create(3, 5);
    ASSERT_ERR(ldlt_csc_from_sparse(rect, NULL, 2.0, &m), SPARSE_ERR_SHAPE);
    ASSERT_NULL(m);
    sparse_free(rect);
}

/* Identity matrix → round-trip preserves lower-triangle values exactly
 * (diagonal only). */
static void test_roundtrip_identity(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &m));
    ASSERT_NOT_NULL(m);
    REQUIRE_OK(ldlt_csc_validate(m));

    /* L's CSC has one entry per column = the diagonal value from A. */
    ASSERT_EQ(m->L->nnz, n);
    for (idx_t j = 0; j < n; j++) {
        ASSERT_EQ(m->L->col_ptr[j + 1] - m->L->col_ptr[j], 1);
        ASSERT_EQ(m->L->row_idx[m->L->col_ptr[j]], j);
        ASSERT_NEAR(m->L->values[m->L->col_ptr[j]], 1.0, 0.0);
    }
    /* D / pivot_size defaults untouched by conversion. */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(m->D[i], 0.0, 0.0);
        ASSERT_EQ(m->pivot_size[i], 1);
        ASSERT_EQ(m->perm[i], i);
    }

    SparseMatrix *B = NULL;
    REQUIRE_OK(ldlt_csc_to_sparse(m, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    ldlt_csc_free(m);
}

/* Diagonal indefinite matrix (mix of positive/negative diagonal entries):
 * conversion must preserve every value verbatim including the signs. */
static void test_roundtrip_diagonal_indefinite(void) {
    idx_t n = 5;
    double diag[5] = {3.0, -2.0, 1.0, -4.5, 0.25};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);

    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &m));
    REQUIRE_OK(ldlt_csc_validate(m));

    /* Each column has exactly the one diagonal entry, value preserved. */
    ASSERT_EQ(m->L->nnz, n);
    for (idx_t j = 0; j < n; j++) {
        ASSERT_EQ(m->L->row_idx[m->L->col_ptr[j]], j);
        ASSERT_NEAR(m->L->values[m->L->col_ptr[j]], diag[j], 0.0);
    }

    SparseMatrix *B = NULL;
    REQUIRE_OK(ldlt_csc_to_sparse(m, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    ldlt_csc_free(m);
}

/* Symmetric indefinite matrix with off-diagonals.  Round-trip recovers
 * the lower triangle (mirror entries in the upper half are stripped by
 * conversion, as they are for the Cholesky CSC path). */
static void test_roundtrip_symmetric_indefinite(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    /* A = [[1, 2, 0, 0], [2, -1, 3, 0], [0, 3, 2, -1], [0, 0, -1, 1]] */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, -1.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 3, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 2, 1, 3.0);
    sparse_insert(A, 1, 2, 3.0);
    sparse_insert(A, 3, 2, -1.0);
    sparse_insert(A, 2, 3, -1.0);

    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &m));
    REQUIRE_OK(ldlt_csc_validate(m));

    /* Lower-triangle nnz count: 4 diagonals + 3 off-diagonals (below). */
    ASSERT_EQ(m->L->nnz, 7);

    SparseMatrix *B = NULL;
    REQUIRE_OK(ldlt_csc_to_sparse(m, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    ldlt_csc_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * from_sparse with permutation
 * ═══════════════════════════════════════════════════════════════════════ */

/* The caller-supplied fill-reducing permutation is copied verbatim into
 * `ldlt->perm` — Day 8 will compose Bunch-Kaufman swaps on top. */
static void test_from_sparse_stores_perm(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0 + (double)i);

    idx_t perm[4] = {3, 2, 1, 0}; /* reverse */
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, perm, 2.0, &m));
    REQUIRE_OK(ldlt_csc_validate(m));
    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(m->perm[i], perm[i]);

    ldlt_csc_free(m);
    sparse_free(A);
}

/* An invalid permutation (out-of-range entry) is rejected by the
 * embedded Cholesky converter; the error propagates through. */
static void test_from_sparse_invalid_perm(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    idx_t bad[3] = {0, 5, 2};
    LdltCsc *m = NULL;
    ASSERT_ERR(ldlt_csc_from_sparse(A, bad, 2.0, &m), SPARSE_ERR_BADARG);
    ASSERT_NULL(m);

    sparse_free(A);
}

/* Reverse permutation on a dense symmetric matrix: conversion keeps
 * only one of each symmetric pair, so the count is n*(n+1)/2 and the
 * retained entries can be checked by mapping CSC entries back to A. */
static void test_reverse_perm_symmetric(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 5.0 + (double)i);
        for (idx_t j = i + 1; j < n; j++) {
            double v = 1.0 + (double)(i * n + j);
            sparse_insert(A, i, j, v);
            sparse_insert(A, j, i, v);
        }
    }

    idx_t perm[4] = {3, 2, 1, 0};
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, perm, 2.0, &m));
    REQUIRE_OK(ldlt_csc_validate(m));
    ASSERT_EQ(m->L->nnz, n * (n + 1) / 2);

    /* Each stored (new_r, new_c) value in L should match A[perm[new_r], perm[new_c]]. */
    for (idx_t j = 0; j < m->L->n; j++) {
        for (idx_t p = m->L->col_ptr[j]; p < m->L->col_ptr[j + 1]; p++) {
            idx_t new_r = m->L->row_idx[p];
            double v = m->L->values[p];
            double a = sparse_get(A, perm[new_r], perm[j]);
            if (v != a)
                TF_FAIL_("L(new %d,%d)->(orig %d,%d): csc=%.15g A=%.15g", (int)new_r, (int)j,
                         (int)perm[new_r], (int)perm[j], v, a);
            tf_asserts++;
        }
    }

    ldlt_csc_free(m);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Validate: positive and catches broken state
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_validate_null(void) { ASSERT_ERR(ldlt_csc_validate(NULL), SPARSE_ERR_NULL); }

static void test_validate_fresh_alloc_is_valid(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 4, &m));
    REQUIRE_OK(ldlt_csc_validate(m));
    ldlt_csc_free(m);
}

/* pivot_size = 3 is invalid (must be 1 or 2). */
static void test_validate_catches_bad_pivot_size(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 4, &m));
    m->pivot_size[2] = 3;
    ASSERT_ERR(ldlt_csc_validate(m), SPARSE_ERR_BADARG);
    ldlt_csc_free(m);
}

/* 2x2 pivot marked on index i with pivot_size[i+1] still == 1 is
 * inconsistent and must be rejected. */
static void test_validate_catches_half_2x2(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 4, &m));
    m->pivot_size[1] = 2;
    /* m->pivot_size[2] left at 1 — mismatch. */
    ASSERT_ERR(ldlt_csc_validate(m), SPARSE_ERR_BADARG);
    ldlt_csc_free(m);
}

/* A 2x2 pivot starting at the last index (no i+1) is invalid. */
static void test_validate_catches_trailing_2x2(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(3, 3, &m));
    m->pivot_size[2] = 2;
    ASSERT_ERR(ldlt_csc_validate(m), SPARSE_ERR_BADARG);
    ldlt_csc_free(m);
}

/* perm with a duplicate index is not a permutation. */
static void test_validate_catches_bad_perm(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(3, 3, &m));
    m->perm[0] = 1;
    m->perm[1] = 1; /* duplicate */
    m->perm[2] = 2;
    ASSERT_ERR(ldlt_csc_validate(m), SPARSE_ERR_BADARG);
    ldlt_csc_free(m);
}

/* Well-formed 2x2 pivot block spanning two consecutive indices — valid. */
static void test_validate_accepts_valid_2x2(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 4, &m));
    m->pivot_size[1] = 2;
    m->pivot_size[2] = 2;
    REQUIRE_OK(ldlt_csc_validate(m));
    ldlt_csc_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 8: Bunch-Kaufman elimination
 * ═══════════════════════════════════════════════════════════════════════ */

/* Retrieve L[r,c] from a factored LdltCsc by linear scan of column c. */
static double ldlt_csc_get(const LdltCsc *F, idx_t r, idx_t c) {
    for (idx_t p = F->L->col_ptr[c]; p < F->L->col_ptr[c + 1]; p++) {
        if (F->L->row_idx[p] == r)
            return F->L->values[p];
    }
    return 0.0;
}

/* Copy an insertion pattern (lower + mirror upper) of a dense symmetric
 * 2D array into a fresh SparseMatrix.  The helper lets us build A twice
 * for the CSC path (via ldlt_csc_from_sparse) and the linked-list
 * comparison (via sparse_ldlt_factor) from the same values. */
static SparseMatrix *build_symmetric(idx_t n, const double *vals_lower) {
    /* vals_lower[i * n + j] holds A[i,j] for i >= j; mirror sets A[j,i]. */
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double v = vals_lower[(size_t)i * (size_t)n + (size_t)j];
            if (v == 0.0)
                continue;
            sparse_insert(A, i, j, v);
            if (i != j)
                sparse_insert(A, j, i, v);
        }
    }
    return A;
}

/* ─── Null-arg ──────────────────────────────────────────────────── */

static void test_eliminate_null(void) { ASSERT_ERR(ldlt_csc_eliminate(NULL), SPARSE_ERR_NULL); }

/* ─── Pure 1x1 pivots: well-conditioned diagonal ────────────────── */

/* Diagonal-only matrix → every pivot is 1x1, D[k] matches the diagonal
 * values exactly, and L is the identity (only unit diagonals). */
static void test_eliminate_all_1x1_pivots(void) {
    idx_t n = 4;
    double diag[4] = {3.0, -2.0, 5.0, 1.5};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    for (idx_t k = 0; k < n; k++) {
        ASSERT_EQ(F->pivot_size[k], 1);
        ASSERT_NEAR(F->D[k], diag[k], 1e-12);
        ASSERT_NEAR(F->D_offdiag[k], 0.0, 0.0);
        /* L[k,k] = 1.0 (unit diagonal stored for uniformity). */
        ASSERT_NEAR(ldlt_csc_get(F, k, k), 1.0, 1e-12);
    }

    ldlt_csc_free(F);
    sparse_free(A);
}

/* Diagonally dominant tridiagonal — all pivots should be 1x1, and L
 * reconstructs A via L * D * L^T to tolerance. */
static void test_eliminate_tridiagonal_all_1x1(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    for (idx_t k = 0; k < n; k++)
        ASSERT_EQ(F->pivot_size[k], 1);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Forced 2x2 pivot: zero diagonal at (0,0) ──────────────────── */

/* A = [[0.1, 1], [1, 0.3]]: both diagonals are small relative to the
 * off-diagonal so Bunch-Kaufman's Criterion 4 fires and a 2x2 pivot is
 * chosen.  (A 2x2 where either diagonal dominates — e.g. c >= alpha*b
 * — is resolved by a 1x1 pivot with a row swap instead of a 2x2.) */
static void test_eliminate_forced_2x2(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    ASSERT_EQ(F->pivot_size[0], 2);
    ASSERT_EQ(F->pivot_size[1], 2);
    /* D[0]=0.1, D[1]=0.3, D_offdiag[0]=1.0 — the 2x2 pivot block. */
    ASSERT_NEAR(F->D[0], 0.1, 1e-12);
    ASSERT_NEAR(F->D_offdiag[0], 1.0, 1e-12);
    ASSERT_NEAR(F->D[1], 0.3, 1e-12);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Mixed 1x1 + 2x2 pivots ────────────────────────────────────── */

/* Block-diagonal with one 2x2 saddle block and a 1x1 well-conditioned
 * block in the trailing rows.  The first 2x2 block uses small diagonals
 * (both < alpha_bk * |off-diag|) to force Criterion 4 → 2x2 pivot. */
static void test_eliminate_mixed_pivots(void) {
    idx_t n = 4;
    double vals[4][4] = {
        {0.1, 1.0, 0.0, 0.0}, {1.0, 0.3, 0.0, 0.0}, {0.0, 0.0, 5.0, 0.0}, {0.0, 0.0, 0.0, 3.0}};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            if (vals[i][j] != 0.0)
                sparse_insert(A, i, j, vals[i][j]);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    ASSERT_EQ(F->pivot_size[0], 2);
    ASSERT_EQ(F->pivot_size[1], 2);
    ASSERT_EQ(F->pivot_size[2], 1);
    ASSERT_EQ(F->pivot_size[3], 1);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Element-by-element comparison vs linked-list ─────────────── */

/* Factor the same matrix via CSC and via linked-list sparse_ldlt_factor.
 * The CSC wrapper delegates to the linked-list kernel internally, so
 * results must match byte-for-byte on D, D_offdiag, pivot_size, perm,
 * and L's entries. */
static void test_eliminate_matches_linked_list_indefinite(void) {
    idx_t n = 4;
    double lower[16] = {
        /* Symmetric indefinite: mixed signs, off-diagonals present. */
        2.0,  0.0,  0.0,  0.0, /* row 0 */
        -1.0, 3.0,  0.0,  0.0, /* row 1 */
        0.5,  -2.0, -1.0, 0.0, /* row 2 */
        0.0,  1.0,  0.5,  4.0, /* row 3 */
    };
    SparseMatrix *A1 = build_symmetric(n, lower);
    SparseMatrix *A2 = build_symmetric(n, lower);

    /* CSC path */
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A1, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    /* Linked-list reference */
    sparse_ldlt_t ll = {0};
    REQUIRE_OK(sparse_ldlt_factor(A2, &ll));

    /* Compare D / D_offdiag / pivot_size / perm. */
    for (idx_t k = 0; k < n; k++) {
        ASSERT_NEAR(F->D[k], ll.D[k], 1e-12);
        ASSERT_NEAR(F->D_offdiag[k], ll.D_offdiag[k], 1e-12);
        ASSERT_EQ(F->pivot_size[k], (idx_t)ll.pivot_size[k]);
        ASSERT_EQ(F->perm[k], ll.perm[k]);
    }

    /* Compare L entries: ll.L holds below-diagonal L multipliers; the
     * CSC adds a unit 1.0 on the diagonal for storage uniformity. */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            double csc = ldlt_csc_get(F, i, j);
            double ref = sparse_get(ll.L, i, j);
            if (i == j)
                ASSERT_NEAR(csc, 1.0, 1e-12);
            else
                ASSERT_NEAR(csc, ref, 1e-12);
        }
    }

    ldlt_csc_free(F);
    sparse_ldlt_free(&ll);
    sparse_free(A1);
    sparse_free(A2);
}

/* ─── Perm composition with fill-reducing input ─────────────────── */

/* When ldlt_csc_from_sparse applies a non-identity `perm_in`, the
 * subsequent eliminate should store perm[k] = perm_in[ll_perm[k]] —
 * the composition of the input perm with the Bunch-Kaufman perm.
 *
 * We verify the composition explicitly by independently factoring the
 * pre-permuted matrix P·A·P^T with the linked-list kernel, so
 * `ll_ref.perm` is the Bunch-Kaufman pivot perm on the same input the
 * CSC wrapper sees internally.  The composition rule then predicts
 * each entry of `F->perm` exactly. */
static void test_eliminate_composes_perm(void) {
    idx_t n = 4;
    /* Symmetric indefinite values so Bunch-Kaufman has something to
     * decide (mix of signs + off-diagonals). */
    double lower[16] = {
        2.0,  0.0, 0.0,  0.0, /* row 0 */
        -1.0, 3.0, 0.0,  0.0, /* row 1 */
        0.5,  0.2, -1.0, 0.0, /* row 2 */
        0.0,  1.0, 0.5,  4.0, /* row 3 */
    };
    SparseMatrix *A = build_symmetric(n, lower);
    /* Use a non-trivial input perm (reverse order). */
    idx_t perm_in[4] = {3, 2, 1, 0};

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, perm_in, 2.0, &F));
    /* Before elimination, F->perm == perm_in verbatim. */
    for (idx_t k = 0; k < n; k++)
        ASSERT_EQ(F->perm[k], perm_in[k]);

    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    /* Build P·A·P^T and factor with the linked-list kernel directly so
     * ll_ref.perm is the BK pivot perm in the pre-permuted space — the
     * same space the CSC wrapper runs BK in internally. */
    SparseMatrix *A_perm = NULL;
    REQUIRE_OK(sparse_permute(A, perm_in, perm_in, &A_perm));
    sparse_ldlt_t ll_ref = {0};
    REQUIRE_OK(sparse_ldlt_factor(A_perm, &ll_ref));

    /* Composition rule: F->perm[k] == perm_in[ll_ref.perm[k]]. */
    for (idx_t k = 0; k < n; k++)
        ASSERT_EQ(F->perm[k], perm_in[ll_ref.perm[k]]);

    sparse_ldlt_free(&ll_ref);
    sparse_free(A_perm);
    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Inertia: sign accounting from D matches linked-list ─────── */

static void test_eliminate_inertia(void) {
    /* Diagonal with 2 positive, 2 negative entries. */
    idx_t n = 4;
    double diag[4] = {4.0, -1.0, 3.0, -2.5};
    SparseMatrix *A1 = sparse_create(n, n);
    SparseMatrix *A2 = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A1, i, i, diag[i]);
        sparse_insert(A2, i, i, diag[i]);
    }

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A1, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    sparse_ldlt_t ll = {0};
    REQUIRE_OK(sparse_ldlt_factor(A2, &ll));

    idx_t pos_ll, neg_ll, zero_ll;
    REQUIRE_OK(sparse_ldlt_inertia(&ll, &pos_ll, &neg_ll, &zero_ll));

    /* Count inertia from F's D/pivot_size directly (1x1 blocks only here). */
    idx_t pos = 0, neg = 0, zero = 0;
    for (idx_t k = 0; k < n; k++) {
        if (F->D[k] > 0)
            pos++;
        else if (F->D[k] < 0)
            neg++;
        else
            zero++;
    }
    ASSERT_EQ(pos, pos_ll);
    ASSERT_EQ(neg, neg_ll);
    ASSERT_EQ(zero, zero_ll);

    ldlt_csc_free(F);
    sparse_ldlt_free(&ll);
    sparse_free(A1);
    sparse_free(A2);
}

/* ─── Singular matrix → NOT_SPD/SINGULAR ───────────────────────── */

/* Zero matrix is rank-deficient — the first pivot fails singularity check. */
static void test_eliminate_singular_zero(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    /* Populate at least one nonzero so symmetry check passes, then
     * build a singular structure. */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    /* Column/row 1 is entirely zero → singular. */

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    sparse_err_t err = ldlt_csc_eliminate(F);
    ASSERT_TRUE(err == SPARSE_ERR_SINGULAR || err == SPARSE_ERR_NOT_SPD);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 9: Triangular + block-diagonal solve
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compute ||A*x - b||_inf / ||b||_inf.  Returns NaN on allocation
 * failure; callers compare the residual against a tolerance with
 * ASSERT_TRUE(rr < tol), which treats NaN as out-of-range. */
static double rel_residual(const SparseMatrix *A, const double *x, const double *b) {
    idx_t n = sparse_rows(A);
    double *Ax = malloc((size_t)n * sizeof(double));
    if (!Ax)
        return (double)NAN;
    sparse_matvec(A, x, Ax);
    double rmax = 0.0, bmax = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double r = fabs(Ax[i] - b[i]);
        double bi = fabs(b[i]);
        if (r > rmax)
            rmax = r;
        if (bi > bmax)
            bmax = bi;
    }
    free(Ax);
    return bmax > 0.0 ? rmax / bmax : rmax;
}

/* ─── Null-arg + missing-field handling ────────────────────────── */

static void test_solve_null_args(void) {
    /* Build a trivial factored LdltCsc.  Use REQUIRE_OK on the setup
     * so a silent regression in ldlt_csc_from_sparse/ldlt_csc_eliminate
     * can't leave F == NULL and trick the null-arg checks below into
     * succeeding for the wrong reason. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, -1.0);
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    ASSERT_NOT_NULL(F);
    REQUIRE_OK(ldlt_csc_eliminate(F));

    double b[2] = {1.0, 1.0};
    double x[2] = {0};
    ASSERT_ERR(ldlt_csc_solve(NULL, b, x), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_solve(F, NULL, x), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_solve(F, b, NULL), SPARSE_ERR_NULL);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Simple solve tests ───────────────────────────────────────── */

static void test_solve_identity(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    double b[4] = {1.5, -2.0, 3.25, 7.0};
    double x[4] = {0};
    REQUIRE_OK(ldlt_csc_solve(F, b, x));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i], 1e-12);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* Diagonal indefinite: solves collapse to element-wise b[i] / D[i]. */
static void test_solve_diagonal_indefinite(void) {
    idx_t n = 4;
    double diag[4] = {4.0, -2.0, 5.0, -1.0};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    double x_true[4] = {1.0, -1.0, 2.0, -2.0};
    double b[4];
    for (idx_t i = 0; i < n; i++)
        b[i] = diag[i] * x_true[i];

    double x[4] = {0};
    REQUIRE_OK(ldlt_csc_solve(F, b, x));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_true[i], 1e-12);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* Forced 2x2 pivot: solves exercise the 2x2 block diagonal path. */
static void test_solve_forced_2x2(void) {
    /* A = [[0.1, 1], [1, 0.3]] — same matrix that forces a 2x2 pivot
     * in the Day 8 test.  Pick a known x and verify recovery. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    ASSERT_EQ(F->pivot_size[0], 2); /* sanity: we are exercising the 2x2 path */

    double x_true[2] = {1.0, 2.0};
    double b[2];
    sparse_matvec(A, x_true, b);
    double x[2] = {0};
    REQUIRE_OK(ldlt_csc_solve(F, b, x));
    ASSERT_NEAR(x[0], x_true[0], 1e-12);
    ASSERT_NEAR(x[1], x_true[1], 1e-12);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* Tridiagonal symmetric indefinite: mix of positive and negative
 * diagonal entries with off-diagonals.  Verify residual < 1e-10. */
static void test_solve_tridiagonal_indefinite(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        /* Alternating signs on the diagonal: 4, -3, 4, -3, ... */
        sparse_insert(A, i, i, (i % 2 == 0) ? 4.0 : -3.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_true[i] = 1.0 + 0.5 * (double)i;
    sparse_matvec(A, x_true, b);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_solve(F, b, x));

    double rr = rel_residual(A, x, b);
    printf("    tridiag indefinite n=10: rel_res = %.3e\n", rr);
    ASSERT_TRUE(rr < 1e-10);

    free(x_true);
    free(b);
    free(x);
    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Comparison against linked-list sparse_ldlt_solve ───────── */

/* A symmetric indefinite matrix with a mix of 1x1 and 2x2 pivots.
 * Factor+solve the same matrix via CSC and via linked-list, compare
 * solution vectors element-wise. */
static void test_solve_matches_linked_list(void) {
    idx_t n = 4;
    double lower[16] = {
        2.0,  0.0, 0.0,  0.0, /* row 0 */
        -1.0, 3.0, 0.0,  0.0, /* row 1 */
        0.5,  0.2, -1.0, 0.0, /* row 2 */
        0.0,  1.0, 0.5,  4.0, /* row 3 */
    };
    SparseMatrix *A1 = build_symmetric(n, lower);
    SparseMatrix *A2 = build_symmetric(n, lower);

    double b[4] = {1.0, -2.0, 3.0, -4.0};
    double x_csc[4] = {0};
    double x_ll[4] = {0};

    /* CSC path. */
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A1, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_solve(F, b, x_csc));

    /* Linked-list path. */
    sparse_ldlt_t ll = {0};
    REQUIRE_OK(sparse_ldlt_factor(A2, &ll));
    REQUIRE_OK(sparse_ldlt_solve(&ll, b, x_ll));

    /* The two solutions should agree to double-precision round-off. */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_csc[i], x_ll[i], 1e-12);

    /* And residual should be tight against both. */
    double rr = rel_residual(A1, x_csc, b);
    ASSERT_TRUE(rr < 1e-10);

    ldlt_csc_free(F);
    sparse_ldlt_free(&ll);
    sparse_free(A1);
    sparse_free(A2);
}

/* ─── In-place solve (b == x) ──────────────────────────────────── */

static void test_solve_in_place(void) {
    idx_t n = 4;
    double diag[4] = {3.0, -2.0, 5.0, -1.0};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    double b_copy[4] = {3.0, -4.0, 10.0, -1.0};
    double xbuf[4];
    memcpy(xbuf, b_copy, sizeof(b_copy));
    REQUIRE_OK(ldlt_csc_solve(F, xbuf, xbuf)); /* b == x */

    double x_ref[4] = {0};
    REQUIRE_OK(ldlt_csc_solve(F, b_copy, x_ref));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(xbuf[i], x_ref[i], 1e-12);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── AMD-reordered factor + solve on a larger indefinite matrix ─ */

static void test_solve_with_amd_perm(void) {
    /* Arrow-style indefinite 6x6: diag has mixed signs, last row/col
     * dense enough to benefit from AMD reordering. */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    double dvals[6] = {4.0, -3.0, 2.0, -2.5, 3.0, -1.0};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, dvals[i]);
    /* Arrow: last column/row fills everything. */
    for (idx_t i = 0; i < n - 1; i++) {
        sparse_insert(A, n - 1, i, 0.5);
        sparse_insert(A, i, n - 1, 0.5);
    }

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_amd(A, perm));

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_true[i] = (double)(i + 1);
    sparse_matvec(A, x_true, b);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, perm, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_solve(F, b, x));

    double rr = rel_residual(A, x, b);
    printf("    arrow 6x6 indefinite (AMD): rel_res = %.3e\n", rr);
    ASSERT_TRUE(rr < 1e-10);

    free(perm);
    free(x_true);
    free(b);
    free(x);
    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Inertia consistency vs linked-list ──────────────────────── */

static void test_inertia_matches_linked_list(void) {
    /* Symmetric indefinite with known inertia (2 positive, 2 negative). */
    idx_t n = 4;
    double lower[16] = {
        3.0, 0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 0.0, 0.5, -0.5, 2.0, 0.0, 0.0, 1.0, -1.0, -3.0,
    };
    SparseMatrix *A1 = build_symmetric(n, lower);
    SparseMatrix *A2 = build_symmetric(n, lower);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A1, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    sparse_ldlt_t ll = {0};
    REQUIRE_OK(sparse_ldlt_factor(A2, &ll));

    idx_t pos_ll, neg_ll, zero_ll;
    REQUIRE_OK(sparse_ldlt_inertia(&ll, &pos_ll, &neg_ll, &zero_ll));

    /* Compute inertia from F's D / pivot_size directly (matches the
     * Sylvester law of inertia the linked-list path uses). */
    idx_t pos = 0, neg = 0, zero = 0;
    for (idx_t k = 0; k < n;) {
        if (F->pivot_size[k] == 1) {
            if (F->D[k] > 0)
                pos++;
            else if (F->D[k] < 0)
                neg++;
            else
                zero++;
            k++;
        } else {
            double d11 = F->D[k];
            double d22 = F->D[k + 1];
            double d21 = F->D_offdiag[k];
            double det = d11 * d22 - d21 * d21;
            /* A 2x2 block always contributes exactly one + and one - when
             * det < 0; two same-sign eigenvalues when det > 0. */
            if (det < 0) {
                pos++;
                neg++;
            } else if (det > 0) {
                if (d11 + d22 > 0) {
                    pos += 2;
                } else {
                    neg += 2;
                }
            } else {
                zero += 2;
            }
            k += 2;
        }
    }

    ASSERT_EQ(pos, pos_ll);
    ASSERT_EQ(neg, neg_ll);
    ASSERT_EQ(zero, zero_ll);

    ldlt_csc_free(F);
    sparse_ldlt_free(&ll);
    sparse_free(A1);
    sparse_free(A2);
}

/* ─── Singular detection ──────────────────────────────────────── */

/* Hand-craft a factored LdltCsc with a near-zero 1x1 pivot and verify
 * solve returns SPARSE_ERR_SINGULAR. */
static void test_solve_detects_tiny_1x1_pivot(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(2, 4, &F));
    /* L is identity. */
    F->L->col_ptr[0] = 0;
    F->L->col_ptr[1] = 1;
    F->L->col_ptr[2] = 2;
    F->L->row_idx[0] = 0;
    F->L->row_idx[1] = 1;
    F->L->values[0] = 1.0;
    F->L->values[1] = 1.0;
    F->L->nnz = 2;
    F->D[0] = 1e-20; /* below the relative tolerance */
    F->D[1] = 1.0;
    F->pivot_size[0] = 1;
    F->pivot_size[1] = 1;
    F->perm[0] = 0;
    F->perm[1] = 1;
    F->factor_norm = 1.0;

    double b[2] = {1.0, 1.0}, x[2] = {0};
    ASSERT_ERR(ldlt_csc_solve(F, b, x), SPARSE_ERR_SINGULAR);

    ldlt_csc_free(F);
}

/* Hand-craft a factored LdltCsc with a near-singular 2x2 block and
 * verify solve returns SPARSE_ERR_SINGULAR. */
static void test_solve_detects_singular_2x2_block(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(2, 4, &F));
    F->L->col_ptr[0] = 0;
    F->L->col_ptr[1] = 1;
    F->L->col_ptr[2] = 2;
    F->L->row_idx[0] = 0;
    F->L->row_idx[1] = 1;
    F->L->values[0] = 1.0;
    F->L->values[1] = 1.0;
    F->L->nnz = 2;
    /* 2x2 block [[1, 1], [1, 1]] — determinant 0, singular. */
    F->D[0] = 1.0;
    F->D[1] = 1.0;
    F->D_offdiag[0] = 1.0;
    F->pivot_size[0] = 2;
    F->pivot_size[1] = 2;
    F->perm[0] = 0;
    F->perm[1] = 1;
    F->factor_norm = 1.0;

    double b[2] = {1.0, 1.0}, x[2] = {0};
    ASSERT_ERR(ldlt_csc_solve(F, b, x), SPARSE_ERR_SINGULAR);

    ldlt_csc_free(F);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("ldlt_csc (Sprint 17 Day 7)");

    /* alloc / free */
    RUN_TEST(test_alloc_null_out);
    RUN_TEST(test_alloc_negative_n);
    RUN_TEST(test_alloc_basic);
    RUN_TEST(test_alloc_zero_n);
    RUN_TEST(test_free_null);

    /* from_sparse / to_sparse round-trips */
    RUN_TEST(test_from_sparse_null_args);
    RUN_TEST(test_from_sparse_shape);
    RUN_TEST(test_roundtrip_identity);
    RUN_TEST(test_roundtrip_diagonal_indefinite);
    RUN_TEST(test_roundtrip_symmetric_indefinite);

    /* permutation */
    RUN_TEST(test_from_sparse_stores_perm);
    RUN_TEST(test_from_sparse_invalid_perm);
    RUN_TEST(test_reverse_perm_symmetric);

    /* validate */
    RUN_TEST(test_validate_null);
    RUN_TEST(test_validate_fresh_alloc_is_valid);
    RUN_TEST(test_validate_catches_bad_pivot_size);
    RUN_TEST(test_validate_catches_half_2x2);
    RUN_TEST(test_validate_catches_trailing_2x2);
    RUN_TEST(test_validate_catches_bad_perm);
    RUN_TEST(test_validate_accepts_valid_2x2);

    /* Day 8 — Bunch-Kaufman elimination */
    RUN_TEST(test_eliminate_null);
    RUN_TEST(test_eliminate_all_1x1_pivots);
    RUN_TEST(test_eliminate_tridiagonal_all_1x1);
    RUN_TEST(test_eliminate_forced_2x2);
    RUN_TEST(test_eliminate_mixed_pivots);
    RUN_TEST(test_eliminate_matches_linked_list_indefinite);
    RUN_TEST(test_eliminate_composes_perm);
    RUN_TEST(test_eliminate_inertia);
    RUN_TEST(test_eliminate_singular_zero);

    /* Day 9 — Triangular + block-diagonal solve */
    RUN_TEST(test_solve_null_args);
    RUN_TEST(test_solve_identity);
    RUN_TEST(test_solve_diagonal_indefinite);
    RUN_TEST(test_solve_forced_2x2);
    RUN_TEST(test_solve_tridiagonal_indefinite);
    RUN_TEST(test_solve_matches_linked_list);
    RUN_TEST(test_solve_in_place);
    RUN_TEST(test_solve_with_amd_perm);
    RUN_TEST(test_inertia_matches_linked_list);
    RUN_TEST(test_solve_detects_tiny_1x1_pivot);
    RUN_TEST(test_solve_detects_singular_2x2_block);

    TEST_SUITE_END();
}
