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
/* Simple seeded LCG so the random cross-check is deterministic across
 * runs (same RNG style used by the Day 2 stress test). */
static unsigned int xc_rng_state = 0u;
static unsigned int xc_rng_next(void) {
    xc_rng_state = xc_rng_state * 1664525u + 1013904223u;
    return xc_rng_state;
}
static double xc_rng_uniform(double lo, double hi) {
    double u = (double)(xc_rng_next() & 0x7fffffff) / (double)0x7fffffff;
    return lo + (hi - lo) * u;
}

/* Build a random n×n symmetric indefinite matrix.  The diagonal is
 * uniform on [-3, 3] and a random subset of off-diagonals in [-2, 2]
 * is inserted (mirrored).  The seed is baked in by the caller so each
 * iteration of the test sweep visits a reproducible fixture. */
static SparseMatrix *build_random_symmetric(idx_t n, unsigned int seed) {
    xc_rng_state = seed;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        double d = xc_rng_uniform(-3.0, 3.0);
        if (fabs(d) < 0.3)
            d = (d < 0 ? -0.3 : 0.3);
        sparse_insert(A, i, i, d);
    }
    for (idx_t i = 1; i < n; i++) {
        for (idx_t j = 0; j < i; j++) {
            /* ~35% off-diagonal density — enough to trigger BK cross-
             * column cmod without saturating to a dense factor. */
            if ((xc_rng_next() % 100) < 35) {
                double v = xc_rng_uniform(-2.0, 2.0);
                sparse_insert(A, i, j, v);
                sparse_insert(A, j, i, v);
            }
        }
    }
    return A;
}

/* Compare a factored `LdltCsc` against a reference `sparse_ldlt_t` on
 * the same input (via `sparse_ldlt_factor` on a separate `SparseMatrix`
 * with identical values).  Tolerance checks match the single-matrix
 * case — pivot_size and perm must agree exactly, D / D_offdiag / L
 * entries within 1e-10 absolute. */
static int ldlt_csc_matches_linked_list(const LdltCsc *F, const sparse_ldlt_t *ll, idx_t n,
                                        double tol) {
    for (idx_t k = 0; k < n; k++) {
        if (F->pivot_size[k] != (idx_t)ll->pivot_size[k])
            return 0;
        if (F->perm[k] != ll->perm[k])
            return 0;
        if (fabs(F->D[k] - ll->D[k]) > tol)
            return 0;
        if (fabs(F->D_offdiag[k] - ll->D_offdiag[k]) > tol)
            return 0;
    }
    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < n; i++) {
            double csc = ldlt_csc_get(F, i, j);
            double ref = sparse_get(ll->L, i, j);
            if (i == j) {
                if (fabs(csc - 1.0) > tol)
                    return 0;
            } else if (fabs(csc - ref) > tol) {
                return 0;
            }
        }
    }
    return 1;
}

static void test_eliminate_matches_linked_list_indefinite(void) {
    /* Original fixed 4×4 matrix — keeps the legacy single-case coverage
     * alongside the random sweep below so a targeted regression is
     * still easy to inspect. */
    idx_t n = 4;
    double lower[16] = {
        2.0,  0.0,  0.0,  0.0, /* row 0 */
        -1.0, 3.0,  0.0,  0.0, /* row 1 */
        0.5,  -2.0, -1.0, 0.0, /* row 2 */
        0.0,  1.0,  0.5,  4.0, /* row 3 */
    };
    SparseMatrix *A1 = build_symmetric(n, lower);
    SparseMatrix *A2 = build_symmetric(n, lower);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A1, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    sparse_ldlt_t ll = {0};
    REQUIRE_OK(sparse_ldlt_factor(A2, &ll));

    ASSERT_TRUE(ldlt_csc_matches_linked_list(F, &ll, n, 1e-12));

    ldlt_csc_free(F);
    sparse_ldlt_free(&ll);
    sparse_free(A1);
    sparse_free(A2);
}

/* Sprint 18 Day 5 cross-check: sweep 20 random symmetric indefinite
 * matrices of varied size and assert the native CSC kernel's output
 * matches the linked-list `sparse_ldlt_factor` reference within
 * floating-point round-off on every one.  Seeds are fixed so failure
 * can be reproduced by rerunning the test. */
static void test_eliminate_matches_linked_list_random_indefinite(void) {
    /* Sizes chosen to cover small (where cmod contributions are
     * minimal) and moderate (where mixed 1×1/2×2 pivots and swaps are
     * likely).  Each seed is a distinct 32-bit constant — no aliasing. */
    struct {
        idx_t n;
        unsigned int seed;
    } cases[] = {
        {3, 0x1a2b3c4du},  {3, 0x5e6f7081u},  {4, 0x92a3b4c5u},  {4, 0xd6e7f809u},
        {5, 0x10123456u},  {5, 0x789abcdeu},  {6, 0xf0123456u},  {6, 0x22222222u},
        {7, 0x33333333u},  {7, 0x44444444u},  {8, 0xaaaaaaaau},  {8, 0xbbbbbbbbu},
        {9, 0xcccccccdu},  {10, 0xdeadbeefu}, {10, 0xfeedfaceu}, {11, 0xcafebabeu},
        {12, 0xbadc0ffeu}, {12, 0x13579bdfu}, {15, 0x2468aceeu}, {18, 0x01020304u},
    };
    const size_t ncases = sizeof(cases) / sizeof(cases[0]);

    int divergences = 0;
    for (size_t idx = 0; idx < ncases; idx++) {
        idx_t n = cases[idx].n;
        SparseMatrix *A1 = build_random_symmetric(n, cases[idx].seed);
        SparseMatrix *A2 = build_random_symmetric(n, cases[idx].seed);

        LdltCsc *F = NULL;
        sparse_err_t err_csc = ldlt_csc_from_sparse(A1, NULL, 2.0, &F);
        sparse_ldlt_t ll = {0};
        sparse_err_t err_ll = sparse_ldlt_factor(A2, &ll);

        /* Both paths must succeed, or both must fail with the same
         * error.  Random matrices sometimes produce singular pivots;
         * BK's decisions are identical across paths so the failures
         * should line up. */
        if (err_csc != SPARSE_OK || err_ll != SPARSE_OK) {
            ASSERT_EQ(err_csc, err_ll);
            if (F)
                ldlt_csc_free(F);
            sparse_ldlt_free(&ll);
            sparse_free(A1);
            sparse_free(A2);
            continue;
        }

        sparse_err_t err_elim = ldlt_csc_eliminate(F);
        if (err_elim != SPARSE_OK) {
            divergences++;
            ldlt_csc_free(F);
            sparse_ldlt_free(&ll);
            sparse_free(A1);
            sparse_free(A2);
            continue;
        }

        REQUIRE_OK(ldlt_csc_validate(F));
        if (!ldlt_csc_matches_linked_list(F, &ll, n, 1e-10))
            divergences++;

        ldlt_csc_free(F);
        sparse_ldlt_free(&ll);
        sparse_free(A1);
        sparse_free(A2);
    }

    ASSERT_EQ(divergences, 0);
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
 * Sprint 18 Day 1: Native-kernel scaffolding
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Verify the Day 1 skeleton: the runtime override reaches the native
 * kernel, the native kernel validates inputs the same way the wrapper
 * does, and the still-stub column loop returns SPARSE_ERR_BADARG so
 * tests and benchmarks get a clean signal while the 1x1/2x2 branches
 * land across Days 3-4.  These tests restore the default override in
 * teardown so subsequent tests still exercise the wrapper path. */

/* ─── Override getter/setter round-trip ─────────────────────────── */

static void test_native_override_roundtrip(void) {
    ASSERT_EQ(ldlt_csc_get_kernel_override(), LDLT_CSC_KERNEL_DEFAULT);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    ASSERT_EQ(ldlt_csc_get_kernel_override(), LDLT_CSC_KERNEL_NATIVE);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_WRAPPER);
    ASSERT_EQ(ldlt_csc_get_kernel_override(), LDLT_CSC_KERNEL_WRAPPER);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);
    ASSERT_EQ(ldlt_csc_get_kernel_override(), LDLT_CSC_KERNEL_DEFAULT);
}

/* ─── Placeholder preserved for the test registration order; real
 * 2×2 coverage lives in the Sprint 18 Day 4 block below, which
 * verifies that the forced-2×2 matrix factors correctly under the
 * native kernel. */

/* ─── Native kernel handles n == 0 trivially ───────────────────── */

static void test_native_empty_matrix_ok(void) {
    /* A 0x0 LdltCsc via direct alloc (ldlt_csc_from_sparse rejects
     * zero-dim input, matching sparse_ldlt_factor).  The native
     * kernel's early-out for n <= 0 returns SPARSE_OK. */
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(0, 1, &F));

    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    ASSERT_ERR(ldlt_csc_eliminate(F), SPARSE_OK);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    ldlt_csc_free(F);
}

/* ─── Native kernel input validation matches the wrapper ───────── */

static void test_native_rejects_null(void) {
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    ASSERT_ERR(ldlt_csc_eliminate(NULL), SPARSE_ERR_NULL);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);
}

/* ─── Wrapper override still reaches the wrapper ───────────────── */

static void test_wrapper_override_still_factors(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, -3.0);
    sparse_insert(A, 2, 2, 4.0);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));

    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_WRAPPER);
    REQUIRE_OK(ldlt_csc_eliminate(F));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    REQUIRE_OK(ldlt_csc_validate(F));
    ASSERT_EQ(F->pivot_size[0], 1);
    ASSERT_EQ(F->pivot_size[1], 1);
    ASSERT_EQ(F->pivot_size[2], 1);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Workspace lifecycle ──────────────────────────────────────── */

static void test_workspace_alloc_free(void) {
    LdltCscWorkspace *ws = NULL;
    REQUIRE_OK(ldlt_csc_workspace_alloc(5, &ws));
    ASSERT_TRUE(ws != NULL);
    ASSERT_EQ(ws->n, 5);
    ASSERT_EQ(ws->pattern_count, 0);
    ASSERT_EQ(ws->pattern_count_r, 0);
    ASSERT_TRUE(ws->dense_col != NULL);
    ASSERT_TRUE(ws->dense_col_r != NULL);
    ldlt_csc_workspace_free(ws);

    /* free(NULL) is a no-op. */
    ldlt_csc_workspace_free(NULL);

    /* n == 0 still yields a valid allocation (all pointers non-NULL
     * so callers don't have to special-case the empty matrix). */
    LdltCscWorkspace *ws0 = NULL;
    REQUIRE_OK(ldlt_csc_workspace_alloc(0, &ws0));
    ASSERT_TRUE(ws0->dense_col != NULL);
    ldlt_csc_workspace_free(ws0);
}

static void test_workspace_alloc_rejects_negative_n(void) {
    LdltCscWorkspace *ws = (LdltCscWorkspace *)0xdead;
    ASSERT_ERR(ldlt_csc_workspace_alloc(-1, &ws), SPARSE_ERR_BADARG);
    ASSERT_TRUE(ws == NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 2: In-place symmetric swap primitive
 * ═══════════════════════════════════════════════════════════════════════ */

/* ─── Dense helpers for cross-checking the swap ─────────────────── */

/* Copy the stored lower triangle of F->L into a dense n*n row-major
 * buffer.  Zero everywhere F->L has no entry — this matches the
 * sparse convention (structurally absent == numerically zero).  The
 * upper triangle is left zero so the oracle reads only the lower
 * triangle. */
static void ldlt_lower_to_dense(const LdltCsc *F, double *dense) {
    idx_t n = F->n;
    for (idx_t p = 0; p < n * n; p++)
        dense[p] = 0.0;
    for (idx_t c = 0; c < n; c++) {
        idx_t start = F->L->col_ptr[c];
        idx_t end = F->L->col_ptr[c + 1];
        for (idx_t p = start; p < end; p++) {
            idx_t r = F->L->row_idx[p];
            dense[r * n + c] = F->L->values[p];
        }
    }
}

/* Apply the symmetric permutation σ = (i ↔ j) to a dense lower
 * triangle.  For each stored entry (r, c) with r >= c, map to
 * (σ(r), σ(c)), reflect to lower-triangle if σ(r) < σ(c), write to
 * `dst`.  `dst` must be a separate buffer (caller zero-fills). */
static void dense_sym_swap(const double *src, double *dst, idx_t n, idx_t i, idx_t j) {
    for (idx_t p = 0; p < n * n; p++)
        dst[p] = 0.0;
    for (idx_t c = 0; c < n; c++) {
        for (idx_t r = c; r < n; r++) {
            double v = src[r * n + c];
            if (v == 0.0)
                continue;
            idx_t rn = (r == i) ? j : ((r == j) ? i : r);
            idx_t cn = (c == i) ? j : ((c == j) ? i : c);
            if (rn < cn) {
                idx_t t = rn;
                rn = cn;
                cn = t;
            }
            dst[rn * n + cn] = v;
        }
    }
}

/* Element-wise compare two dense lower triangles.  Returns 1 if all
 * corresponding entries match to tol, 0 otherwise. */
static int dense_lower_equal(const double *a, const double *b, idx_t n, double tol) {
    for (idx_t r = 0; r < n; r++) {
        for (idx_t c = 0; c <= r; c++) {
            double diff = fabs(a[r * n + c] - b[r * n + c]);
            if (diff > tol)
                return 0;
        }
    }
    return 1;
}

/* Build an LdltCsc from a list of (row, col, value) triples
 * representing the lower triangle of a symmetric matrix of dimension
 * n.  Inserts mirrored entries so `ldlt_csc_from_sparse`'s symmetry
 * check passes.  Each column's diagonal is inserted explicitly (add
 * `diag_fill[c]` for the diagonal value) so `chol_csc_validate`'s
 * diagonal-first invariant holds after the swap. */
static LdltCsc *build_ldlt_from_triples(idx_t n, const double *diag, const idx_t *rows,
                                        const idx_t *cols, const double *vals, idx_t nnz_offdiag) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t c = 0; c < n; c++)
        sparse_insert(A, c, c, diag[c]);
    for (idx_t k = 0; k < nnz_offdiag; k++) {
        sparse_insert(A, rows[k], cols[k], vals[k]);
        sparse_insert(A, cols[k], rows[k], vals[k]); /* mirror */
    }
    LdltCsc *F = NULL;
    if (ldlt_csc_from_sparse(A, NULL, 2.0, &F) != SPARSE_OK) {
        sparse_free(A);
        return NULL;
    }
    sparse_free(A);
    return F;
}

/* ─── Error-path tests ──────────────────────────────────────────── */

static void test_symmetric_swap_null(void) {
    ASSERT_ERR(ldlt_csc_symmetric_swap(NULL, 0, 1), SPARSE_ERR_NULL);
}

static void test_symmetric_swap_out_of_range(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 10, &F));
    ASSERT_ERR(ldlt_csc_symmetric_swap(F, -1, 2), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_symmetric_swap(F, 2, 4), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_symmetric_swap(F, 4, 2), SPARSE_ERR_BADARG);
    ldlt_csc_free(F);
}

static void test_symmetric_swap_noop_i_equals_j(void) {
    /* Dense 3×3 SPD-ish matrix; swap(1, 1) must leave storage
     * byte-identical. */
    idx_t rows[] = {1, 2, 2};
    idx_t cols[] = {0, 0, 1};
    double vals[] = {0.5, 0.3, 0.7};
    double diag[] = {2.0, 3.0, 4.0};
    LdltCsc *F = build_ldlt_from_triples(3, diag, rows, cols, vals, 3);
    REQUIRE_OK(ldlt_csc_validate(F));

    idx_t nnz_before = F->L->col_ptr[F->n];
    double *dense_before = calloc((size_t)(F->n * F->n), sizeof(double));
    double *dense_after = calloc((size_t)(F->n * F->n), sizeof(double));
    ldlt_lower_to_dense(F, dense_before);

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 1, 1));
    ldlt_lower_to_dense(F, dense_after);

    ASSERT_EQ(F->L->col_ptr[F->n], nnz_before);
    ASSERT_TRUE(dense_lower_equal(dense_before, dense_after, F->n, 0.0));
    REQUIRE_OK(ldlt_csc_validate(F));

    free(dense_before);
    free(dense_after);
    ldlt_csc_free(F);
}

/* ─── Adjacent swap on a dense 5×5 ───────────────────────────────── */

static void test_symmetric_swap_adjacent_dense(void) {
    /* Build a dense-ish 5×5 symmetric matrix with unique values so
     * any positional mistake after the swap is easy to spot. */
    idx_t n = 5;
    double diag[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    /* Offdiagonals (row > col): unique values so position permutation
     * is verifiable. */
    idx_t rows[] = {1, 2, 3, 4, 2, 3, 4, 3, 4, 4};
    idx_t cols[] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3};
    double vals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0};
    LdltCsc *F = build_ldlt_from_triples(n, diag, rows, cols, vals, 10);
    REQUIRE_OK(ldlt_csc_validate(F));

    double *dense_src = calloc((size_t)(n * n), sizeof(double));
    double *dense_expected = calloc((size_t)(n * n), sizeof(double));
    double *dense_actual = calloc((size_t)(n * n), sizeof(double));
    ldlt_lower_to_dense(F, dense_src);
    dense_sym_swap(dense_src, dense_expected, n, 1, 2);

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 1, 2));
    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_lower_to_dense(F, dense_actual);

    ASSERT_TRUE(dense_lower_equal(dense_expected, dense_actual, n, 1e-15));

    free(dense_src);
    free(dense_expected);
    free(dense_actual);
    ldlt_csc_free(F);
}

/* ─── Non-adjacent swap on dense 5×5 ───────────────────────────── */

static void test_symmetric_swap_non_adjacent(void) {
    idx_t n = 5;
    double diag[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    idx_t rows[] = {1, 2, 3, 4, 2, 3, 4, 3, 4, 4};
    idx_t cols[] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3};
    double vals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0};
    LdltCsc *F = build_ldlt_from_triples(n, diag, rows, cols, vals, 10);
    REQUIRE_OK(ldlt_csc_validate(F));

    double *dense_src = calloc((size_t)(n * n), sizeof(double));
    double *dense_expected = calloc((size_t)(n * n), sizeof(double));
    double *dense_actual = calloc((size_t)(n * n), sizeof(double));
    ldlt_lower_to_dense(F, dense_src);
    /* Swap i=1, j=4 — touches left block (col 0), middle block
     * (cols 1..4), and the below-j range (empty for n=5). */
    dense_sym_swap(dense_src, dense_expected, n, 1, 4);

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 1, 4));
    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_lower_to_dense(F, dense_actual);

    ASSERT_TRUE(dense_lower_equal(dense_expected, dense_actual, n, 1e-15));

    free(dense_src);
    free(dense_expected);
    free(dense_actual);
    ldlt_csc_free(F);
}

/* ─── Swap on a tridiagonal pattern ─────────────────────────────── */

static void test_symmetric_swap_tridiagonal(void) {
    idx_t n = 6;
    double diag[] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    idx_t rows[] = {1, 2, 3, 4, 5};
    idx_t cols[] = {0, 1, 2, 3, 4};
    double vals[] = {-1.0, -1.0, -1.0, -1.0, -1.0};
    LdltCsc *F = build_ldlt_from_triples(n, diag, rows, cols, vals, 5);
    REQUIRE_OK(ldlt_csc_validate(F));

    double *dense_src = calloc((size_t)(n * n), sizeof(double));
    double *dense_expected = calloc((size_t)(n * n), sizeof(double));
    double *dense_actual = calloc((size_t)(n * n), sizeof(double));
    ldlt_lower_to_dense(F, dense_src);
    /* Swap i=1, j=4 — leaves many entries structurally zero, so phase
     * A's "only i" and "only j" branches both get exercised. */
    dense_sym_swap(dense_src, dense_expected, n, 1, 4);

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 1, 4));
    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_lower_to_dense(F, dense_actual);

    ASSERT_TRUE(dense_lower_equal(dense_expected, dense_actual, n, 1e-15));

    free(dense_src);
    free(dense_expected);
    free(dense_actual);
    ldlt_csc_free(F);
}

/* ─── Swap that moves a tiny diagonal onto the pivot seat ───────── */

/* The BK use case: A has a tiny/structurally-zero diagonal at row 0
 * so the pivot selector picks another row r and swaps r ↔ 0 to put
 * the larger off-diagonal onto the pivot seat.  Verify the swap
 * correctly relocates the tiny diagonal. */
static void test_symmetric_swap_moves_tiny_diag(void) {
    idx_t n = 4;
    /* Diag: tiny at 0, normal elsewhere; large off-diagonal (0, 2). */
    double diag[] = {1e-12, 5.0, 3.0, 4.0};
    idx_t rows[] = {2};
    idx_t cols[] = {0};
    double vals[] = {9.0};
    LdltCsc *F = build_ldlt_from_triples(n, diag, rows, cols, vals, 1);
    REQUIRE_OK(ldlt_csc_validate(F));

    double *dense_src = calloc((size_t)(n * n), sizeof(double));
    double *dense_expected = calloc((size_t)(n * n), sizeof(double));
    double *dense_actual = calloc((size_t)(n * n), sizeof(double));
    ldlt_lower_to_dense(F, dense_src);
    dense_sym_swap(dense_src, dense_expected, n, 0, 2);

    /* Also check F->perm reflects the swap. */
    idx_t perm0_before = F->perm[0];
    idx_t perm2_before = F->perm[2];

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 0, 2));
    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_lower_to_dense(F, dense_actual);

    ASSERT_EQ(F->perm[0], perm2_before);
    ASSERT_EQ(F->perm[2], perm0_before);
    /* Tiny diag should now be at (2, 2). */
    ASSERT_TRUE(fabs(dense_actual[2 * n + 2] - 1e-12) < 1e-18);
    /* Old diag at (2, 2) = 3.0 should now be at (0, 0). */
    ASSERT_TRUE(fabs(dense_actual[0] - 3.0) < 1e-15);
    ASSERT_TRUE(dense_lower_equal(dense_expected, dense_actual, n, 1e-15));

    free(dense_src);
    free(dense_expected);
    free(dense_actual);
    ldlt_csc_free(F);
}

/* ─── Aux-array swap: D / D_offdiag / pivot_size ────────────────── */

static void test_symmetric_swap_aux_arrays(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 1, &F));
    /* Simulate a partially-populated state: manually set aux fields. */
    F->D[0] = 10.0;
    F->D[1] = 20.0;
    F->D[2] = 30.0;
    F->D[3] = 40.0;
    F->D_offdiag[0] = 0.0;
    F->D_offdiag[1] = 0.0;
    F->D_offdiag[2] = 7.0;
    F->D_offdiag[3] = 0.0;
    F->pivot_size[0] = 1;
    F->pivot_size[1] = 1;
    F->pivot_size[2] = 2;
    F->pivot_size[3] = 2;
    F->perm[0] = 100;
    F->perm[1] = 101;
    F->perm[2] = 102;
    F->perm[3] = 103;

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 0, 2));

    ASSERT_EQ((int)(F->D[0] * 1000), 30000);
    ASSERT_EQ((int)(F->D[2] * 1000), 10000);
    ASSERT_EQ((int)(F->D_offdiag[0] * 1000), 7000);
    ASSERT_EQ((int)(F->D_offdiag[2] * 1000), 0);
    ASSERT_EQ(F->pivot_size[0], 2);
    ASSERT_EQ(F->pivot_size[2], 1);
    ASSERT_EQ(F->perm[0], 102);
    ASSERT_EQ(F->perm[2], 100);
    /* Unswapped positions unchanged. */
    ASSERT_EQ((int)(F->D[1] * 1000), 20000);
    ASSERT_EQ((int)(F->D[3] * 1000), 40000);

    ldlt_csc_free(F);
}

/* ─── Stress test: random swaps cross-checked against dense oracle ─ */

/* LCG so tests are deterministic and portable (no rand()/time()
 * dependence).  Same style as other random tests in this repo. */
static unsigned int swap_rng_state = 0xB0F17E84u;
static unsigned int swap_rng_next(void) {
    swap_rng_state = swap_rng_state * 1664525u + 1013904223u;
    return swap_rng_state;
}

static void test_symmetric_swap_stress_random(void) {
    idx_t n = 20;
    swap_rng_state = 0xB0F17E84u;

    /* Build a random symmetric matrix with explicit diagonals. */
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t c = 0; c < n; c++)
        sparse_insert(A, c, c, 1.0 + (double)(swap_rng_next() % 100));
    /* Add ~30 random off-diagonals (mirrored). */
    for (int k = 0; k < 30; k++) {
        idx_t r = (idx_t)(swap_rng_next() % (unsigned)n);
        idx_t c = (idx_t)(swap_rng_next() % (unsigned)n);
        if (r == c)
            continue;
        if (r < c) {
            idx_t t = r;
            r = c;
            c = t;
        }
        double v = (double)((swap_rng_next() % 100) + 1) / 10.0;
        sparse_insert(A, r, c, v);
        sparse_insert(A, c, r, v);
    }
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    sparse_free(A);

    double *dense_a = calloc((size_t)(n * n), sizeof(double));
    double *dense_b = calloc((size_t)(n * n), sizeof(double));
    ldlt_lower_to_dense(F, dense_a);

    /* 50 random swaps; after each, dense oracle swap and compare. */
    for (int step = 0; step < 50; step++) {
        idx_t i = (idx_t)(swap_rng_next() % (unsigned)n);
        idx_t j = (idx_t)(swap_rng_next() % (unsigned)n);
        /* Oracle swap from dense_a → dense_b, then copy dense_b back
         * to dense_a for the next iteration. */
        dense_sym_swap(dense_a, dense_b, n, i, j);
        REQUIRE_OK(ldlt_csc_symmetric_swap(F, i, j));
        REQUIRE_OK(ldlt_csc_validate(F));
        double *tmp = dense_a;
        dense_a = dense_b;
        dense_b = tmp;
        /* dense_a now holds the oracle's current state; check CSC
         * matches. */
        double *dense_csc = calloc((size_t)(n * n), sizeof(double));
        ldlt_lower_to_dense(F, dense_csc);
        ASSERT_TRUE(dense_lower_equal(dense_a, dense_csc, n, 1e-15));
        free(dense_csc);
    }

    free(dense_a);
    free(dense_b);
    ldlt_csc_free(F);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 3: 1×1 Bunch-Kaufman column loop
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compare two factored LdltCsc values field-by-field.  Returns 1 iff
 * pivot_size, perm, col_ptr, row_idx all match exactly and D, D_offdiag,
 * L->values agree within `tol` absolute.  Use a tight `tol` (1e-12) to
 * flag any computational divergence between wrapper and native; the
 * floating-point ordering of the cmod loop is the same on both paths
 * (both subtract contributions kp = 0, 1, ..., k-1 in order), so values
 * should match to round-off. */
static int ldlt_factorizations_match(const LdltCsc *A, const LdltCsc *B, double tol) {
    if (A->n != B->n)
        return 0;
    idx_t n = A->n;
    for (idx_t i = 0; i < n; i++) {
        if (A->pivot_size[i] != B->pivot_size[i])
            return 0;
        if (A->perm[i] != B->perm[i])
            return 0;
        if (fabs(A->D[i] - B->D[i]) > tol)
            return 0;
        if (fabs(A->D_offdiag[i] - B->D_offdiag[i]) > tol)
            return 0;
    }
    if (A->L->nnz != B->L->nnz)
        return 0;
    for (idx_t j = 0; j <= n; j++) {
        if (A->L->col_ptr[j] != B->L->col_ptr[j])
            return 0;
    }
    idx_t nnz = A->L->col_ptr[n];
    for (idx_t p = 0; p < nnz; p++) {
        if (A->L->row_idx[p] != B->L->row_idx[p])
            return 0;
        if (fabs(A->L->values[p] - B->L->values[p]) > tol)
            return 0;
    }
    return 1;
}

/* Factor the matrix with both the wrapper and the native kernel, then
 * assert the resulting LdltCsc structures agree.  Asserts via the
 * framework helpers so failure points at the calling test name. */
static void check_native_matches_wrapper(const SparseMatrix *A, double tol) {
    LdltCsc *Fw = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &Fw));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_WRAPPER);
    REQUIRE_OK(ldlt_csc_eliminate(Fw));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);
    REQUIRE_OK(ldlt_csc_validate(Fw));

    LdltCsc *Fn = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    REQUIRE_OK(ldlt_csc_eliminate(Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);
    REQUIRE_OK(ldlt_csc_validate(Fn));

    ASSERT_TRUE(ldlt_factorizations_match(Fw, Fn, tol));

    ldlt_csc_free(Fw);
    ldlt_csc_free(Fn);
}

/* ─── Pure-diagonal indefinite: no cmod, no swap, all criterion-1 1×1 ─ */

static void test_native_1x1_diagonal_matches_wrapper(void) {
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, -3.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 3, -5.0);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── Tridiagonal SPD: exercises the cmod loop, all criterion-1 1×1 ─ */

static void test_native_1x1_tridiagonal_matches_wrapper(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── Mixed indefinite diagonal with weak off-diagonals: all 1×1 ─── */

static void test_native_1x1_mixed_indefinite_matches_wrapper(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    double diag[] = {3.0, -2.5, 4.0, -1.5, 2.0};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);
    /* Off-diagonals small enough that BK criterion 1 fires on every
     * column — |diag| > alpha * |offdiag| at every step. */
    sparse_insert(A, 1, 0, 0.2);
    sparse_insert(A, 0, 1, 0.2);
    sparse_insert(A, 2, 1, 0.3);
    sparse_insert(A, 1, 2, 0.3);
    sparse_insert(A, 3, 2, 0.4);
    sparse_insert(A, 2, 3, 0.4);
    sparse_insert(A, 4, 3, 0.5);
    sparse_insert(A, 3, 4, 0.5);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── Criterion 3 (1×1 with symmetric row swap) ───────────────────── */

/* A = [[0.1, 0.5, 0], [0.5, 2, 0], [0, 0, 3]].  At k=0:
 *   diag = 0.1, max_offdiag = 0.5 (row 1), alpha*max_offdiag ≈ 0.32
 *   → phase 2 triggered.  Partner col 1 has dense_col_r = {0:0.5, 1:2}.
 *   sigma_r (max over i != 1) = 0.5.
 *   Criterion 2: 0.1*0.5 = 0.05 < 0.64*0.25 = 0.16.  Fails.
 *   Criterion 3: |2| >= 0.64*0.5 = 0.32.  Passes → swap 0↔1. */
static void test_native_1x1_with_swap_matches_wrapper(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 0.5);
    sparse_insert(A, 1, 0, 0.5);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── Larger tridiagonal indefinite: stress the column loop at n=20 ─ */

static void test_native_1x1_tridiag_large_matches_wrapper(void) {
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    /* Diagonals alternating sign, off-diagonals small enough for BK
     * to pick criterion-1 1×1 pivots throughout. */
    for (idx_t i = 0; i < n; i++) {
        double d = (i % 2 == 0) ? 5.0 : -5.0;
        sparse_insert(A, i, i, d);
        if (i > 0) {
            sparse_insert(A, i, i - 1, 0.7);
            sparse_insert(A, i - 1, i, 0.7);
        }
    }
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── Zero pivot detection: sing_tol guards against near-zero 1×1 ── */

static void test_native_detects_near_zero_1x1_pivot(void) {
    /* A = diag(1e-20, 1.0) — column 0's diagonal is below sing_tol
     * (~ DROP_TOL * ||A||_inf ≈ 1e-14) and max_offdiag is 0, so BK
     * fires criterion 1 (no swap possible without off-diagonals).
     * The 1×1 singularity check must reject this. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1e-20);
    sparse_insert(A, 1, 1, 1.0);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    ASSERT_ERR(ldlt_csc_eliminate(F), SPARSE_ERR_SINGULAR);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Identity: trivial pass-through, perm stays identity ──────── */

static void test_native_1x1_identity_matches_wrapper(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 4: 2×2 Bunch-Kaufman block pivots
 * ═══════════════════════════════════════════════════════════════════════ */

/* Forward-declared helper defined in the Day 9 test block below; used
 * by the Day 4 end-to-end solve test.  Relative residual of A*x - b. */
static double rel_residual(const SparseMatrix *A, const double *x, const double *b);

/* ─── Forced 2×2 at (0, 1): the canonical BK criterion-4 matrix ─── */

static void test_native_2x2_forced_matches_wrapper(void) {
    /* A = [[0.1, 1], [1, 0.3]] — both diagonals small vs the
     * off-diagonal, BK picks a 2×2 block at (0, 1).  Matches
     * test_eliminate_forced_2x2's setup; the native kernel must now
     * produce the same L / D / D_offdiag / pivot_size / perm as the
     * wrapper. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── 2×2 pivot requiring r↔k+1 swap (partner not adjacent) ─────── */

static void test_native_2x2_nonadjacent_partner_matches_wrapper(void) {
    /* 4×4 matrix where the BK partner for column 0 is row 2, not
     * row 1.  That forces the native kernel's r ↔ k+1 swap branch.
     *
     * Construction: at k=0, we want max_offdiag at row 2 (not 1).
     * Put a tiny diag at 0, small off-diag (0,1), and large off-diag
     * (0,2); row 2's diagonal is small enough that criterion 3 fails
     * and criterion 4 fires. */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 0.05);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 2, 0.05);
    sparse_insert(A, 3, 3, 4.0);
    sparse_insert(A, 1, 0, 0.2);
    sparse_insert(A, 0, 1, 0.2);
    sparse_insert(A, 2, 0, 1.5);
    sparse_insert(A, 0, 2, 1.5);
    sparse_insert(A, 3, 2, 0.3);
    sparse_insert(A, 2, 3, 0.3);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── Mixed 1×1 + 2×2 pivots with subsequent column cmod ─────── */

static void test_native_mixed_pivots_matches_wrapper(void) {
    /* 4×4 matrix where col 0 takes a 2×2 block at (0, 1) and cols 2
     * and 3 then use 1×1 pivots whose Schur complement receives a
     * cross-term contribution from the 2×2 at (0, 1).  Verifies
     * that `ldlt_csc_cmod_unified`'s Phase B correctly accumulates
     * the `L[:, 0] * d_off * L[col, 1] + L[:, 1] * d_off * L[col, 0]`
     * term when computing dense_col for col 2 and col 3. */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 1, 1, 0.3);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 3, 5.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 2, 0, 0.4);
    sparse_insert(A, 0, 2, 0.4);
    sparse_insert(A, 2, 1, 0.5);
    sparse_insert(A, 1, 2, 0.5);
    sparse_insert(A, 3, 1, 0.3);
    sparse_insert(A, 1, 3, 0.3);
    sparse_insert(A, 3, 2, 0.2);
    sparse_insert(A, 2, 3, 0.2);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── The existing SuiteSparse-style indefinite cross-check fixtures
 *     (ported from test_eliminate_mixed_pivots in spirit) ─────── */

static void test_native_mixed_pivots_larger_matches_wrapper(void) {
    /* 6×6 matrix with a 2×2 pivot near the start and additional
     * structure that stresses cmod cross-terms for later 1×1 and 2×2
     * columns.  Adapted from the existing mixed-pivot test pattern
     * so wrapper and native can be directly compared. */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    double diag[] = {0.05, 0.2, 3.0, -4.0, 2.0, -2.5};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);
    struct {
        idx_t r;
        idx_t c;
        double v;
    } off[] = {{1, 0, 0.9}, {2, 0, 0.3}, {3, 0, 0.2}, {2, 1, 0.4}, {3, 1, 0.1},
               {3, 2, 0.6}, {4, 2, 0.5}, {4, 3, 0.3}, {5, 3, 0.4}, {5, 4, 0.2}};
    for (size_t k = 0; k < sizeof(off) / sizeof(off[0]); k++) {
        sparse_insert(A, off[k].r, off[k].c, off[k].v);
        sparse_insert(A, off[k].c, off[k].r, off[k].v);
    }
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── Solve end-to-end under native kernel ─────────────────────── */

static void test_native_2x2_solve_matches_linked_list(void) {
    /* Factor the forced-2×2 matrix via the native kernel, solve
     * A*x = b with a known b, and assert the residual matches the
     * wrapper's solve to round-off.  This exercises the full
     * factor → solve pipeline through the native path. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);

    LdltCsc *Fn = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    REQUIRE_OK(ldlt_csc_eliminate(Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    double b[] = {1.0, -0.5};
    double x[2] = {0};
    REQUIRE_OK(ldlt_csc_solve(Fn, b, x));
    ASSERT_TRUE(rel_residual(A, x, b) < 1e-12);

    ldlt_csc_free(Fn);
    sparse_free(A);
}

/* ─── Inertia preserved: #pos/#neg entries of D match the wrapper ─ */

static void test_native_2x2_inertia_matches_wrapper(void) {
    /* Use the forced-2×2 matrix [[0.1, 1], [1, 0.3]].  Wrapper and
     * native must produce the same D / D_offdiag, so their 2×2 block
     * eigenvalue sign decomposition (= inertia contribution) is
     * identical.  We don't compute eigenvalues here; instead we
     * assert D and D_offdiag match which is sufficient (the block
     * eigenvalues are a deterministic function of the entries). */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);

    LdltCsc *Fw = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &Fw));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_WRAPPER);
    REQUIRE_OK(ldlt_csc_eliminate(Fw));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    LdltCsc *Fn = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    REQUIRE_OK(ldlt_csc_eliminate(Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    for (idx_t i = 0; i < 2; i++) {
        ASSERT_EQ(Fw->pivot_size[i], Fn->pivot_size[i]);
        ASSERT_TRUE(fabs(Fw->D[i] - Fn->D[i]) < 1e-12);
        ASSERT_TRUE(fabs(Fw->D_offdiag[i] - Fn->D_offdiag[i]) < 1e-12);
    }

    ldlt_csc_free(Fw);
    ldlt_csc_free(Fn);
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
    TEST_SUITE_BEGIN("ldlt_csc (Sprint 17 Days 7-9 + Sprint 18 Days 1-5)");

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
    RUN_TEST(test_eliminate_matches_linked_list_random_indefinite);
    RUN_TEST(test_eliminate_composes_perm);
    RUN_TEST(test_eliminate_inertia);
    RUN_TEST(test_eliminate_singular_zero);

    /* Sprint 18 Day 1 — native-kernel scaffolding */
    RUN_TEST(test_native_override_roundtrip);
    RUN_TEST(test_native_empty_matrix_ok);
    RUN_TEST(test_native_rejects_null);
    RUN_TEST(test_wrapper_override_still_factors);
    RUN_TEST(test_workspace_alloc_free);
    RUN_TEST(test_workspace_alloc_rejects_negative_n);

    /* Sprint 18 Day 2 — in-place symmetric swap */
    RUN_TEST(test_symmetric_swap_null);
    RUN_TEST(test_symmetric_swap_out_of_range);
    RUN_TEST(test_symmetric_swap_noop_i_equals_j);
    RUN_TEST(test_symmetric_swap_adjacent_dense);
    RUN_TEST(test_symmetric_swap_non_adjacent);
    RUN_TEST(test_symmetric_swap_tridiagonal);
    RUN_TEST(test_symmetric_swap_moves_tiny_diag);
    RUN_TEST(test_symmetric_swap_aux_arrays);
    RUN_TEST(test_symmetric_swap_stress_random);

    /* Sprint 18 Day 3 — 1×1 Bunch-Kaufman native column loop */
    RUN_TEST(test_native_1x1_diagonal_matches_wrapper);
    RUN_TEST(test_native_1x1_tridiagonal_matches_wrapper);
    RUN_TEST(test_native_1x1_mixed_indefinite_matches_wrapper);
    RUN_TEST(test_native_1x1_with_swap_matches_wrapper);
    RUN_TEST(test_native_1x1_tridiag_large_matches_wrapper);
    RUN_TEST(test_native_detects_near_zero_1x1_pivot);
    RUN_TEST(test_native_1x1_identity_matches_wrapper);

    /* Sprint 18 Day 4 — 2×2 Bunch-Kaufman block pivots */
    RUN_TEST(test_native_2x2_forced_matches_wrapper);
    RUN_TEST(test_native_2x2_nonadjacent_partner_matches_wrapper);
    RUN_TEST(test_native_mixed_pivots_matches_wrapper);
    RUN_TEST(test_native_mixed_pivots_larger_matches_wrapper);
    RUN_TEST(test_native_2x2_solve_matches_linked_list);
    RUN_TEST(test_native_2x2_inertia_matches_wrapper);

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
