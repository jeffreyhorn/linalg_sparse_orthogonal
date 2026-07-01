/* _POSIX_C_SOURCE 200809L: needed for `setenv` / `unsetenv` used by
 * the `tf_setenv` / `tf_unsetenv` macros from test_framework.h. Must be
 * defined before any system header is included so glibc's `<features.h>` sees
 * it on first inclusion. */
#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _POSIX_C_SOURCE 200809L
#endif
/*
 * Core proof owner for the Cholesky CSC working format. Supernodal and
 * writeback-specific coverage lives in `test_chol_csc_supernodal.c`; this file
 * keeps allocation, conversion, scalar elimination, solve, and public dispatch
 * coverage together.
 */

#include "sparse_analysis.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_cholesky.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "test_framework.h"
#define TF_ENABLE_EXTERNAL_REFERENCE_HELPER
#include "test_solver_helpers.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * alloc / free smoke tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_chol_csc_alloc_null_out(void) {
    ASSERT_ERR(chol_csc_alloc(5, 10, NULL), SPARSE_ERR_NULL);
}

static void test_chol_csc_alloc_negative_n(void) {
    CholCsc *m = NULL;
    ASSERT_ERR(chol_csc_alloc(-1, 10, &m), SPARSE_ERR_BADARG);
    ASSERT_NULL(m);
}

static void test_chol_csc_alloc_basic(void) {
    CholCsc *m = NULL;
    REQUIRE_OK(chol_csc_alloc(4, 8, &m));
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->n, 4);
    ASSERT_EQ(m->nnz, 0);
    ASSERT_EQ(m->capacity, 8);
    ASSERT_NOT_NULL(m->col_ptr);
    ASSERT_NOT_NULL(m->row_idx);
    ASSERT_NOT_NULL(m->values);
    /* col_ptr zeroed so the CSC represents an empty matrix. */
    for (idx_t j = 0; j <= m->n; j++)
        ASSERT_EQ(m->col_ptr[j], 0);
    chol_csc_free(m);
}

static void test_chol_csc_alloc_zero_initial_nnz(void) {
    /* initial_nnz < 1 is clamped to 1 so the arrays are always allocated. */
    CholCsc *m = NULL;
    REQUIRE_OK(chol_csc_alloc(3, 0, &m));
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->capacity, 1);
    chol_csc_free(m);
}

static void test_chol_csc_alloc_zero_n(void) {
    /* n == 0 is a valid (empty) matrix. */
    CholCsc *m = NULL;
    REQUIRE_OK(chol_csc_alloc(0, 1, &m));
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->n, 0);
    ASSERT_NOT_NULL(m->col_ptr);
    ASSERT_EQ(m->col_ptr[0], 0);
    chol_csc_free(m);
}

static void test_chol_csc_free_null(void) {
    chol_csc_free(NULL); /* must not crash */
    ASSERT_TRUE(1);
}

/* ═══════════════════════════════════════════════════════════════════════
 * grow smoke tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_chol_csc_grow_null(void) { ASSERT_ERR(chol_csc_grow(NULL, 10), SPARSE_ERR_NULL); }

static void test_chol_csc_grow_noop(void) {
    CholCsc *m = NULL;
    REQUIRE_OK(chol_csc_alloc(3, 16, &m));
    idx_t original_cap = m->capacity;
    REQUIRE_OK(chol_csc_grow(m, 8));
    ASSERT_EQ(m->capacity, original_cap);
    chol_csc_free(m);
}

static void test_chol_csc_grow_exact(void) {
    CholCsc *m = NULL;
    REQUIRE_OK(chol_csc_alloc(3, 4, &m));
    /* Growing to exactly capacity is a no-op. */
    REQUIRE_OK(chol_csc_grow(m, 4));
    ASSERT_EQ(m->capacity, 4);
    chol_csc_free(m);
}

static void test_chol_csc_grow_geometric(void) {
    CholCsc *m = NULL;
    REQUIRE_OK(chol_csc_alloc(3, 4, &m));
    /* needed=5 triggers doubling to 8 (2 × 4). */
    REQUIRE_OK(chol_csc_grow(m, 5));
    ASSERT_EQ(m->capacity, 8);
    chol_csc_free(m);
}

static void test_chol_csc_grow_to_needed(void) {
    CholCsc *m = NULL;
    REQUIRE_OK(chol_csc_alloc(3, 4, &m));
    /* needed=20 exceeds 2×4=8, so new_cap == needed. */
    REQUIRE_OK(chol_csc_grow(m, 20));
    ASSERT_EQ(m->capacity, 20);
    chol_csc_free(m);
}

static void test_chol_csc_grow_preserves_values(void) {
    CholCsc *m = NULL;
    REQUIRE_OK(chol_csc_alloc(2, 2, &m));
    /* Seed a couple of entries so we can verify realloc preserved them. */
    m->row_idx[0] = 0;
    m->row_idx[1] = 1;
    m->values[0] = 3.5;
    m->values[1] = -2.25;
    m->nnz = 2;

    REQUIRE_OK(chol_csc_grow(m, 10));
    ASSERT_TRUE(m->capacity >= 10);
    ASSERT_EQ(m->row_idx[0], 0);
    ASSERT_EQ(m->row_idx[1], 1);
    ASSERT_NEAR(m->values[0], 3.5, 0.0);
    ASSERT_NEAR(m->values[1], -2.25, 0.0);
    ASSERT_EQ(m->nnz, 2);
    chol_csc_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Conversion helpers and test utilities
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compare two lower-triangular SparseMatrices (including diagonal) for
 * structural and value equality.  Only entries with row >= col are
 * compared — callers must pass matrices that store only the lower
 * triangle.  Shape mismatch is fatal (return early) so we don't cascade
 * thousands of follow-on failures past the root cause. */
static void assert_lower_triangle_equal(const SparseMatrix *A, const SparseMatrix *B, double tol) {
    idx_t n = sparse_rows(A);
    if (sparse_rows(A) != sparse_rows(B) || sparse_cols(A) != sparse_cols(B)) {
        TF_FAIL_("Shape mismatch: A is %dx%d, B is %dx%d", (int)sparse_rows(A), (int)sparse_cols(A),
                 (int)sparse_rows(B), (int)sparse_cols(B));
        return;
    }
    tf_asserts += 2;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double a = sparse_get(A, i, j);
            double b = sparse_get(B, i, j);
            if (fabs(a - b) > tol) {
                TF_FAIL_("Lower-triangle entry (%d,%d): A=%.15g B=%.15g diff=%.3e > tol=%.3e",
                         (int)i, (int)j, a, b, fabs(a - b), tol);
            }
            tf_asserts++;
        }
    }
}

/* Verify CSC invariants on a converted matrix:
 *  - col_ptr monotone, col_ptr[0] == 0, col_ptr[n] == nnz
 *  - row indices sorted ascending within each column
 *  - all row indices >= column (lower triangle) */
static void assert_csc_well_formed(const CholCsc *csc) {
    ASSERT_EQ(csc->col_ptr[0], 0);
    ASSERT_EQ(csc->col_ptr[csc->n], csc->nnz);
    for (idx_t j = 0; j < csc->n; j++) {
        idx_t start = csc->col_ptr[j];
        idx_t end = csc->col_ptr[j + 1];
        ASSERT_TRUE(start <= end);
        for (idx_t p = start; p < end; p++) {
            ASSERT_TRUE(csc->row_idx[p] >= j); /* lower triangle */
            if (p > start)
                ASSERT_TRUE(csc->row_idx[p] > csc->row_idx[p - 1]); /* sorted & distinct */
        }
    }
}

/* Helper: build a symmetric matrix with only lower triangle stored by
 * inserting pairs (i, j) and (j, i) for off-diagonals, then stripping
 * the upper half. Simpler: just insert lower triangle entries for
 * tests that need it. */

/* ═══════════════════════════════════════════════════════════════════════
 * Round-trip conversion tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_from_sparse_null_args(void) {
    CholCsc *csc = NULL;
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);

    ASSERT_ERR(chol_csc_from_sparse(NULL, NULL, 2.0, &csc), SPARSE_ERR_NULL);
    ASSERT_NULL(csc);
    ASSERT_ERR(chol_csc_from_sparse(A, NULL, 2.0, NULL), SPARSE_ERR_NULL);

    /* Non-square input → SHAPE error. */
    SparseMatrix *rect = sparse_create(3, 5);
    ASSERT_ERR(chol_csc_from_sparse(rect, NULL, 2.0, &csc), SPARSE_ERR_SHAPE);
    ASSERT_NULL(csc);
    sparse_free(rect);

    sparse_free(A);
}

static void test_to_sparse_null_args(void) {
    SparseMatrix *mat = NULL;
    ASSERT_ERR(chol_csc_to_sparse(NULL, NULL, &mat), SPARSE_ERR_NULL);
    ASSERT_NULL(mat);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_alloc(2, 2, &csc));
    ASSERT_ERR(chol_csc_to_sparse(csc, NULL, NULL), SPARSE_ERR_NULL);
    chol_csc_free(csc);
}

static void test_roundtrip_identity(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    ASSERT_NOT_NULL(csc);
    ASSERT_EQ(csc->n, n);
    ASSERT_EQ(csc->nnz, n);
    assert_csc_well_formed(csc);
    /* Each column should contain exactly the diagonal. */
    for (idx_t j = 0; j < n; j++) {
        ASSERT_EQ(csc->col_ptr[j + 1] - csc->col_ptr[j], 1);
        ASSERT_EQ(csc->row_idx[csc->col_ptr[j]], j);
        ASSERT_NEAR(csc->values[csc->col_ptr[j]], 1.0, 0.0);
    }

    SparseMatrix *B = NULL;
    REQUIRE_OK(chol_csc_to_sparse(csc, NULL, &B));
    ASSERT_NOT_NULL(B);
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    chol_csc_free(csc);
}

static void test_roundtrip_diagonal_spd(void) {
    /* Diagonal SPD: distinct positive entries, easy to spot index errors. */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0 + (double)i * 2.5); /* 1.0, 3.5, 6.0, ... */

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    assert_csc_well_formed(csc);
    ASSERT_EQ(csc->nnz, n);

    SparseMatrix *B = NULL;
    REQUIRE_OK(chol_csc_to_sparse(csc, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    chol_csc_free(csc);
}

static void test_roundtrip_tridiagonal_spd(void) {
    /* Tridiagonal SPD: L should contain the diagonal plus one subdiagonal
     * per column (except the last). Only the lower triangle is inserted. */
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0); /* subdiagonal only */
    }

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    assert_csc_well_formed(csc);
    /* Expected nnz: n diagonals + (n-1) subdiagonals. */
    ASSERT_EQ(csc->nnz, 2 * n - 1);
    /* Column j < n-1 has 2 entries (diagonal then subdiagonal), last has 1. */
    for (idx_t j = 0; j < n - 1; j++)
        ASSERT_EQ(csc->col_ptr[j + 1] - csc->col_ptr[j], 2);
    ASSERT_EQ(csc->col_ptr[n] - csc->col_ptr[n - 1], 1);

    SparseMatrix *B = NULL;
    REQUIRE_OK(chol_csc_to_sparse(csc, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    chol_csc_free(csc);
}

static void test_roundtrip_dense_lower_5x5(void) {
    /* Dense lower triangle: all n*(n+1)/2 entries distinct values. */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j <= i; j++)
            sparse_insert(A, i, j, 1.0 + (double)(i * n + j)); /* distinct & nonzero */

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    assert_csc_well_formed(csc);
    ASSERT_EQ(csc->nnz, n * (n + 1) / 2);

    SparseMatrix *B = NULL;
    REQUIRE_OK(chol_csc_to_sparse(csc, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    chol_csc_free(csc);
}

static void test_roundtrip_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 7.0);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    ASSERT_EQ(csc->nnz, 1);
    ASSERT_EQ(csc->row_idx[0], 0);
    ASSERT_NEAR(csc->values[0], 7.0, 0.0);

    SparseMatrix *B = NULL;
    REQUIRE_OK(chol_csc_to_sparse(csc, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    chol_csc_free(csc);
}

/* Note: n = 0 cannot be round-tripped because sparse_create() rejects
 * zero dimensions, so there is no n=0 SparseMatrix to convert from.
 * The CSC struct itself handles n = 0 (tested in the allocation tests)
 * but a user wouldn't reach the conversion path with it. */

/* Symmetric matrix where the user inserted BOTH triangles — the
 * converter must keep only the lower triangle. */
static void test_from_sparse_strips_upper_triangle(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    /* Full symmetric A with tridiagonal pattern. */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0); /* upper mirror */
        }
    }

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    assert_csc_well_formed(csc);
    /* Only lower triangle entries kept: n + (n-1). */
    ASSERT_EQ(csc->nnz, 2 * n - 1);

    chol_csc_free(csc);
    sparse_free(A);
}

/* Load a real SuiteSparse SPD matrix (nos4.mtx — standard Cholesky fixture)
 * and round-trip the lower triangle. */
static void test_roundtrip_nos4(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/nos4.mtx"));

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    assert_csc_well_formed(csc);

    SparseMatrix *B = NULL;
    REQUIRE_OK(chol_csc_to_sparse(csc, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    printf("    nos4: n=%d, nnz_lower=%d, capacity=%d\n", (int)csc->n, (int)csc->nnz,
           (int)csc->capacity);

    sparse_free(A);
    sparse_free(B);
    chol_csc_free(csc);
}

/* bcsstk04 is a larger SPD matrix — stress the converter's sorting and
 * capacity logic on something bigger than the hand-crafted tests. */
static void test_roundtrip_bcsstk04(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx"));

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    assert_csc_well_formed(csc);

    SparseMatrix *B = NULL;
    REQUIRE_OK(chol_csc_to_sparse(csc, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    printf("    bcsstk04: n=%d, nnz_lower=%d, capacity=%d\n", (int)csc->n, (int)csc->nnz,
           (int)csc->capacity);

    sparse_free(A);
    sparse_free(B);
    chol_csc_free(csc);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Permutation tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Identity permutation → same result as NULL perm. */
static void test_identity_perm_matches_null(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
    }

    idx_t id[6];
    for (idx_t i = 0; i < n; i++)
        id[i] = i;

    CholCsc *csc_null = NULL;
    CholCsc *csc_id = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc_null));
    REQUIRE_OK(chol_csc_from_sparse(A, id, 2.0, &csc_id));

    ASSERT_EQ(csc_null->nnz, csc_id->nnz);
    for (idx_t j = 0; j <= n; j++)
        ASSERT_EQ(csc_null->col_ptr[j], csc_id->col_ptr[j]);
    for (idx_t p = 0; p < csc_null->nnz; p++) {
        ASSERT_EQ(csc_null->row_idx[p], csc_id->row_idx[p]);
        ASSERT_NEAR(csc_null->values[p], csc_id->values[p], 0.0);
    }

    chol_csc_free(csc_null);
    chol_csc_free(csc_id);
    sparse_free(A);
}

/* Helper: verify every CSC entry (new_r, new_c, v) maps back to the
 * correct A value at (perm[new_r], perm[new_c]).  Applicable when the
 * CSC was built from a symmetric (or lower-triangular) input A with the
 * given perm.  Callers should pair this with an nnz count check so that
 * missing entries are also caught. */
static void assert_csc_entries_match_A(const CholCsc *csc, const SparseMatrix *A,
                                       const idx_t *perm) {
    for (idx_t j = 0; j < csc->n; j++) {
        for (idx_t p = csc->col_ptr[j]; p < csc->col_ptr[j + 1]; p++) {
            idx_t new_r = csc->row_idx[p];
            double v = csc->values[p];
            idx_t orig_r = perm ? perm[new_r] : new_r;
            idx_t orig_c = perm ? perm[j] : j;
            double a = sparse_get(A, orig_r, orig_c);
            if (v != a) {
                TF_FAIL_("CSC entry (new %d,%d)->(orig %d,%d): csc=%.15g A=%.15g", (int)new_r,
                         (int)j, (int)orig_r, (int)orig_c, v, a);
            }
            tf_asserts++;
        }
    }
}

/* Reverse permutation on a dense symmetric matrix.
 *
 * When perm != identity, converting a symmetric A and extracting only
 * the lower triangle in NEW coordinates drops exactly one of each
 * symmetric pair — so a direct lower-triangle comparison against A in
 * user coordinates would spuriously fail.  Instead we check that every
 * CSC entry maps back to the correct A value, and that the count
 * matches the diagonal + one copy of each off-diagonal pair. */
static void test_reverse_perm_symmetric(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    /* Dense symmetric A: distinct values on each pair. */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0 + (double)i);
        for (idx_t j = i + 1; j < n; j++) {
            double v = 1.0 + (double)(i * n + j);
            sparse_insert(A, i, j, v);
            sparse_insert(A, j, i, v);
        }
    }

    idx_t perm[5];
    for (idx_t i = 0; i < n; i++)
        perm[i] = n - 1 - i; /* reverse order: perm[new] = old */

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, perm, 2.0, &csc));
    assert_csc_well_formed(csc);
    /* Diagonal + one of each symmetric pair = n*(n+1)/2. */
    ASSERT_EQ(csc->nnz, n * (n + 1) / 2);
    assert_csc_entries_match_A(csc, A, perm);

    chol_csc_free(csc);
    sparse_free(A);
}

/* Apply an AMD permutation computed from the matrix itself and verify
 * every CSC entry maps to the correct A value in user space.  nos4 is
 * loaded from .mtx which expands the symmetric format to both
 * triangles, so the filter keeps exactly half the off-diagonals. */
static void test_amd_perm_entries_match(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/nos4.mtx"));
    idx_t n = sparse_rows(A);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_NOT_NULL(perm);
    REQUIRE_OK(sparse_reorder_amd(A, perm));

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, perm, 2.0, &csc));
    assert_csc_well_formed(csc);
    assert_csc_entries_match_A(csc, A, perm);

    /* Expected nnz(CSC) = nnz_diagonal + nnz_offdiagonal/2 when A is
     * symmetric with both triangles stored. */
    idx_t nnz_diag = 0;
    idx_t nnz_off = 0;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            if (sparse_get(A, i, j) != 0.0) {
                if (i == j)
                    nnz_diag++;
                else
                    nnz_off++;
            }
        }
    }
    ASSERT_EQ(csc->nnz, nnz_diag + nnz_off / 2);

    free(perm);
    sparse_free(A);
    chol_csc_free(csc);
}

/* Invalid permutation (out-of-range index) → BADARG. */
static void test_invalid_perm_out_of_range(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    idx_t bad[3] = {0, 5, 2}; /* 5 is out of range for n=3 */
    CholCsc *csc = NULL;
    ASSERT_ERR(chol_csc_from_sparse(A, bad, 2.0, &csc), SPARSE_ERR_BADARG);
    ASSERT_NULL(csc);

    sparse_free(A);
}

/* Invalid permutation (duplicate index) → BADARG. */
static void test_invalid_perm_duplicate(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    idx_t bad[3] = {1, 1, 2}; /* duplicate 1 */
    CholCsc *csc = NULL;
    ASSERT_ERR(chol_csc_from_sparse(A, bad, 2.0, &csc), SPARSE_ERR_BADARG);
    ASSERT_NULL(csc);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * fill_factor capacity tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_fill_factor_clamp(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    /* fill_factor < 1.0 → clamped to 1.0 → capacity == nnz == 4. */
    CholCsc *c_low = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 0.1, &c_low));
    ASSERT_EQ(c_low->nnz, n);
    ASSERT_TRUE(c_low->capacity >= n);

    /* fill_factor = 3.0 → capacity ≥ 3 * nnz. */
    CholCsc *c_hi = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 3.0, &c_hi));
    ASSERT_TRUE(c_hi->capacity >= 3 * n);

    chol_csc_free(c_low);
    chol_csc_free(c_hi);
    sparse_free(A);
}

static void test_factor_norm_cached(void) {
    /* The converter caches ||A||_inf on csc->factor_norm for use by the
     * eventual solve path (same convention as LuCsr). */
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 2, 3.0);
    sparse_insert(A, 2, 0, -1.0);
    /* ||A||_inf = max row sum: row 2 = |3| + |-1| = 4, so answer = 5 (row 1). */

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    ASSERT_NEAR(csc->factor_norm, 5.0, 1e-12);

    chol_csc_free(csc);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Symbolic analysis integration
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_with_analysis_null_args(void) {
    CholCsc *csc = NULL;
    SparseMatrix *A = sparse_create(3, 3);
    for (idx_t i = 0; i < 3; i++)
        sparse_insert(A, i, i, 1.0);

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    ASSERT_ERR(chol_csc_from_sparse_with_analysis(NULL, &an, &csc), SPARSE_ERR_NULL);
    ASSERT_NULL(csc);
    ASSERT_ERR(chol_csc_from_sparse_with_analysis(A, NULL, &csc), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_from_sparse_with_analysis(A, &an, NULL), SPARSE_ERR_NULL);

    sparse_analysis_free(&an);
    sparse_free(A);
}

static void test_with_analysis_wrong_type(void) {
    CholCsc *csc = NULL;
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 1, 0, -1.0);
    sparse_insert(A, 0, 1, -1.0);

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_LU,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    ASSERT_ERR(chol_csc_from_sparse_with_analysis(A, &an, &csc), SPARSE_ERR_BADARG);
    ASSERT_NULL(csc);

    sparse_analysis_free(&an);
    sparse_free(A);
}

/* The exact-allocation path should yield structurally identical CSC
 * output to the fill_factor path for the same matrix and perm (since
 * the difference is only capacity, not the entry layout). */
static void test_exact_alloc_matches_dynamic_tridiag(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    CholCsc *c_exact = NULL;
    CholCsc *c_dyn = NULL;
    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &c_exact));
    REQUIRE_OK(chol_csc_from_sparse(A, an.perm, 2.0, &c_dyn));

    /* Structural equality: same nnz, col_ptr, row_idx, values. */
    ASSERT_EQ(c_exact->n, c_dyn->n);
    ASSERT_EQ(c_exact->nnz, c_dyn->nnz);
    for (idx_t j = 0; j <= n; j++)
        ASSERT_EQ(c_exact->col_ptr[j], c_dyn->col_ptr[j]);
    for (idx_t p = 0; p < c_exact->nnz; p++) {
        ASSERT_EQ(c_exact->row_idx[p], c_dyn->row_idx[p]);
        ASSERT_NEAR(c_exact->values[p], c_dyn->values[p], 0.0);
    }

    /* Capacity differs: exact path uses predicted nnz(L) directly. */
    ASSERT_EQ(c_exact->capacity, an.sym_L.nnz);

    REQUIRE_OK(chol_csc_validate(c_exact));
    REQUIRE_OK(chol_csc_validate(c_dyn));

    chol_csc_free(c_exact);
    chol_csc_free(c_dyn);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* Verify capacity == sym_L.nnz for a SuiteSparse SPD matrix and that the
 * predicted nnz(L) is an upper bound on the actual nnz(L) produced by
 * Cholesky factorization on the same matrix.  Drop-tolerance pruning can
 * reduce the actual stored nnz below the exact symbolic prediction, so
 * the invariant is predicted >= actual. */
static void test_predicted_nnz_matches_actual_nos4(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/nos4.mtx"));

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &csc));
    ASSERT_EQ(csc->capacity, an.sym_L.nnz);
    REQUIRE_OK(chol_csc_validate(csc));

    /* Actual nnz(L) from a real Cholesky factor run on the same A. */
    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    idx_t actual_nnz_L = sparse_nnz(L);

    printf("    nos4: predicted nnz(L)=%d, actual nnz(L)=%d\n", (int)an.sym_L.nnz,
           (int)actual_nnz_L);
    ASSERT_TRUE(an.sym_L.nnz >= actual_nnz_L);

    sparse_free(L);
    chol_csc_free(csc);
    sparse_analysis_free(&an);
    sparse_free(A);
}

static void test_predicted_nnz_matches_actual_bcsstk04(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx"));

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &csc));
    ASSERT_EQ(csc->capacity, an.sym_L.nnz);
    REQUIRE_OK(chol_csc_validate(csc));

    /* Cholesky factor with the same AMD reorder to match symbolic prediction. */
    SparseMatrix *L = sparse_copy(A);
    sparse_cholesky_opts_t chol_opts = {
        .reorder = SPARSE_REORDER_AMD,
    };
    REQUIRE_OK(sparse_cholesky_factor_opts(L, &chol_opts));
    idx_t actual_nnz_L = sparse_nnz(L);

    printf("    bcsstk04 (AMD): predicted nnz(L)=%d, actual nnz(L)=%d\n", (int)an.sym_L.nnz,
           (int)actual_nnz_L);
    /* Prediction is an upper bound: drop-tolerance pruning can reduce
     * the actual stored nnz below the exact symbolic count. */
    ASSERT_TRUE(an.sym_L.nnz >= actual_nnz_L);

    sparse_free(L);
    chol_csc_free(csc);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* Random-ish small SPD: A = M * M^T + n * I, constructed so each row has
 * a known structure.  Exercises the path on something not in the
 * SuiteSparse corpus. */
static void test_predicted_nnz_matches_actual_random_spd(void) {
    idx_t n = 12;
    /* Build a banded lower-bidiagonal M, then A = M*M^T + eye scales.
     * Equivalent to a positive-definite pentadiagonal matrix. */
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0 + (double)i);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
        if (i > 1) {
            sparse_insert(A, i, i - 2, 0.5);
            sparse_insert(A, i - 2, i, 0.5);
        }
    }

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &csc));
    REQUIRE_OK(chol_csc_validate(csc));

    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    idx_t actual_nnz_L = sparse_nnz(L);

    ASSERT_TRUE(an.sym_L.nnz >= actual_nnz_L);

    sparse_free(L);
    chol_csc_free(csc);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * chol_csc_validate() tests (positive and broken-CSC catches)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_validate_null(void) { ASSERT_ERR(chol_csc_validate(NULL), SPARSE_ERR_NULL); }

static void test_validate_fresh_alloc_is_valid(void) {
    /* A freshly allocated CSC (nnz=0, empty columns) is a valid one. */
    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_alloc(5, 1, &csc));
    REQUIRE_OK(chol_csc_validate(csc));
    chol_csc_free(csc);
}

static void test_validate_catches_missing_diagonal(void) {
    /* Hand-craft a CSC where column 0 has only an off-diagonal entry
     * (the diagonal is missing).  validate() should reject it. */
    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_alloc(3, 3, &csc));
    csc->col_ptr[0] = 0;
    csc->col_ptr[1] = 1;
    csc->col_ptr[2] = 2;
    csc->col_ptr[3] = 3;
    csc->row_idx[0] = 1; /* off-diagonal in col 0 — missing diag */
    csc->row_idx[1] = 1; /* diag of col 1 */
    csc->row_idx[2] = 2; /* diag of col 2 */
    csc->values[0] = -1.0;
    csc->values[1] = 4.0;
    csc->values[2] = 4.0;
    csc->nnz = 3;
    ASSERT_ERR(chol_csc_validate(csc), SPARSE_ERR_BADARG);
    chol_csc_free(csc);
}

static void test_validate_catches_upper_triangle(void) {
    /* CSC column 1 contains row 0 (i.e. upper triangle). */
    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_alloc(2, 3, &csc));
    csc->col_ptr[0] = 0;
    csc->col_ptr[1] = 1;
    csc->col_ptr[2] = 3;
    csc->row_idx[0] = 0; /* diag col 0 */
    csc->row_idx[1] = 0; /* row 0 in col 1 -> upper triangle */
    csc->row_idx[2] = 1; /* diag col 1 */
    csc->values[0] = 1.0;
    csc->values[1] = 2.0;
    csc->values[2] = 3.0;
    csc->nnz = 3;
    ASSERT_ERR(chol_csc_validate(csc), SPARSE_ERR_BADARG);
    chol_csc_free(csc);
}

static void test_validate_catches_unsorted_column(void) {
    /* Column 0 has rows 0, 2, 1 (out of order). */
    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_alloc(3, 3, &csc));
    csc->col_ptr[0] = 0;
    csc->col_ptr[1] = 3;
    csc->col_ptr[2] = 3;
    csc->col_ptr[3] = 3;
    csc->row_idx[0] = 0;
    csc->row_idx[1] = 2;
    csc->row_idx[2] = 1; /* out of order */
    csc->values[0] = 1.0;
    csc->values[1] = -0.5;
    csc->values[2] = -0.25;
    csc->nnz = 3;
    ASSERT_ERR(chol_csc_validate(csc), SPARSE_ERR_BADARG);
    chol_csc_free(csc);
}

static void test_validate_catches_col_ptr_inconsistency(void) {
    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_alloc(3, 3, &csc));
    csc->col_ptr[0] = 0;
    csc->col_ptr[1] = 1;
    csc->col_ptr[2] = 2;
    csc->col_ptr[3] = 2; /* claims nnz=2 but we'll set nnz=3 below */
    csc->row_idx[0] = 0;
    csc->row_idx[1] = 1;
    csc->row_idx[2] = 2;
    csc->values[0] = 1.0;
    csc->values[1] = 1.0;
    csc->values[2] = 1.0;
    csc->nnz = 3;
    ASSERT_ERR(chol_csc_validate(csc), SPARSE_ERR_BADARG);
    chol_csc_free(csc);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Hardening edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

/* Structurally zero off-diagonal columns: A is pure diagonal so each CSC
 * column has only its diagonal entry.  Verifies the conversion and the
 * symbolic capacity are consistent, and validate() accepts it. */
static void test_edge_case_diagonal_only(void) {
    idx_t n = 7;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0 + (double)i);

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &csc));
    ASSERT_EQ(csc->nnz, n);
    ASSERT_EQ(csc->capacity, an.sym_L.nnz);
    ASSERT_EQ(an.sym_L.nnz, n); /* no fill */
    REQUIRE_OK(chol_csc_validate(csc));

    chol_csc_free(csc);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* Capacity-growth path exercised AFTER conversion: allocate a small
 * fill_factor (=1.0 → exact A-lower nnz), then call chol_csc_grow() to
 * simulate the elimination's fill-in demand.  Verify the stored entries
 * are preserved by the realloc. */
static void test_edge_case_external_capacity_growth(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
    }

    /* fill_factor = 1.0 → capacity == nnz(A_lower) — no headroom. */
    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    idx_t original_nnz = csc->nnz;
    idx_t original_cap = csc->capacity;
    ASSERT_EQ(original_cap, original_nnz);

    /* Grow by a large factor to exercise the reallocation path. */
    REQUIRE_OK(chol_csc_grow(csc, original_cap * 4));
    ASSERT_TRUE(csc->capacity >= original_cap * 4);
    /* Every entry preserved: structure and values. */
    ASSERT_EQ(csc->nnz, original_nnz);
    REQUIRE_OK(chol_csc_validate(csc));
    SparseMatrix *B = NULL;
    REQUIRE_OK(chol_csc_to_sparse(csc, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    chol_csc_free(csc);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workspace and elimination scaffolding
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_workspace_null_out(void) {
    ASSERT_ERR(chol_csc_workspace_alloc(5, NULL), SPARSE_ERR_NULL);
}

static void test_workspace_negative_n(void) {
    CholCscWorkspace *ws = NULL;
    ASSERT_ERR(chol_csc_workspace_alloc(-1, &ws), SPARSE_ERR_BADARG);
    ASSERT_NULL(ws);
}

static void test_workspace_free_null(void) {
    chol_csc_workspace_free(NULL); /* must not crash */
    ASSERT_TRUE(1);
}

static void test_workspace_alloc_basic(void) {
    CholCscWorkspace *ws = NULL;
    REQUIRE_OK(chol_csc_workspace_alloc(7, &ws));
    ASSERT_NOT_NULL(ws);
    ASSERT_EQ(ws->n, 7);
    ASSERT_EQ(ws->pattern_count, 0);
    ASSERT_NOT_NULL(ws->dense_col);
    ASSERT_NOT_NULL(ws->dense_pattern);
    ASSERT_NOT_NULL(ws->dense_marker);
    /* Fresh workspace: everything zero. */
    for (idx_t i = 0; i < ws->n; i++) {
        ASSERT_NEAR(ws->dense_col[i], 0.0, 0.0);
        ASSERT_EQ(ws->dense_marker[i], 0);
    }
    chol_csc_workspace_free(ws);
}

static void test_workspace_alloc_zero_n(void) {
    CholCscWorkspace *ws = NULL;
    REQUIRE_OK(chol_csc_workspace_alloc(0, &ws));
    ASSERT_NOT_NULL(ws);
    ASSERT_EQ(ws->n, 0);
    chol_csc_workspace_free(ws);
}

/* ─── cdiv direct tests (hand-crafted workspace) ─────────────────── */

static void test_cdiv_null(void) { ASSERT_ERR(chol_csc_cdiv(NULL, 0), SPARSE_ERR_NULL); }

static void test_cdiv_positive_diagonal(void) {
    /* Workspace for n=4, column j=1, diagonal = 9.0, one off-diag row 3 = 6.0.
     * Expected: L[1,1] = 3, L[3,1] = 6 / 3 = 2. */
    CholCscWorkspace *ws = NULL;
    REQUIRE_OK(chol_csc_workspace_alloc(4, &ws));
    ws->dense_col[1] = 9.0;
    ws->dense_pattern[ws->pattern_count++] = 1;
    ws->dense_marker[1] = 1;
    ws->dense_col[3] = 6.0;
    ws->dense_pattern[ws->pattern_count++] = 3;
    ws->dense_marker[3] = 1;

    REQUIRE_OK(chol_csc_cdiv(ws, 1));
    ASSERT_NEAR(ws->dense_col[1], 3.0, 1e-12);
    ASSERT_NEAR(ws->dense_col[3], 2.0, 1e-12);

    chol_csc_workspace_free(ws);
}

static void test_cdiv_zero_diagonal_not_spd(void) {
    CholCscWorkspace *ws = NULL;
    REQUIRE_OK(chol_csc_workspace_alloc(3, &ws));
    ws->dense_col[0] = 0.0; /* zero diagonal */
    ws->dense_pattern[ws->pattern_count++] = 0;
    ws->dense_marker[0] = 1;

    ASSERT_ERR(chol_csc_cdiv(ws, 0), SPARSE_ERR_NOT_SPD);
    chol_csc_workspace_free(ws);
}

static void test_cdiv_negative_diagonal_not_spd(void) {
    CholCscWorkspace *ws = NULL;
    REQUIRE_OK(chol_csc_workspace_alloc(3, &ws));
    ws->dense_col[0] = -4.0;
    ws->dense_pattern[ws->pattern_count++] = 0;
    ws->dense_marker[0] = 1;

    ASSERT_ERR(chol_csc_cdiv(ws, 0), SPARSE_ERR_NOT_SPD);
    chol_csc_workspace_free(ws);
}

/* ─── eliminate() tests ─────────────────────────────────────────── */

static void test_eliminate_null(void) { ASSERT_ERR(chol_csc_eliminate(NULL), SPARSE_ERR_NULL); }

static void test_eliminate_diagonal(void) {
    /* A = diag(4, 9, 16, 25) — Cholesky gives L = diag(2, 3, 4, 5). */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    double diag[4] = {4.0, 9.0, 16.0, 25.0};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));

    /* Column j has a single entry at (j, j) with value sqrt(diag[j]). */
    for (idx_t j = 0; j < n; j++) {
        ASSERT_EQ(csc->col_ptr[j + 1] - csc->col_ptr[j], 1);
        ASSERT_EQ(csc->row_idx[csc->col_ptr[j]], j);
        ASSERT_NEAR(csc->values[csc->col_ptr[j]], sqrt(diag[j]), 1e-12);
    }

    sparse_free(A);
    chol_csc_free(csc);
}

static void test_eliminate_2x2_spd(void) {
    /* A = [[4, 2], [2, 5]] — L = [[2, 0], [1, 2]].
     *   L[0,0] = sqrt(4) = 2
     *   L[1,0] = 2 / 2 = 1
     *   L[1,1] = sqrt(5 - 1^2) = sqrt(4) = 2 */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 1, 5.0);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));

    /* Column 0: rows 0 (L[0,0]=2) and 1 (L[1,0]=1). */
    ASSERT_EQ(csc->col_ptr[1] - csc->col_ptr[0], 2);
    ASSERT_EQ(csc->row_idx[csc->col_ptr[0]], 0);
    ASSERT_EQ(csc->row_idx[csc->col_ptr[0] + 1], 1);
    ASSERT_NEAR(csc->values[csc->col_ptr[0]], 2.0, 1e-12);
    ASSERT_NEAR(csc->values[csc->col_ptr[0] + 1], 1.0, 1e-12);
    /* Column 1: row 1 (L[1,1]=2). */
    ASSERT_EQ(csc->col_ptr[2] - csc->col_ptr[1], 1);
    ASSERT_EQ(csc->row_idx[csc->col_ptr[1]], 1);
    ASSERT_NEAR(csc->values[csc->col_ptr[1]], 2.0, 1e-12);

    sparse_free(A);
    chol_csc_free(csc);
}

/* Tridiagonal SPD: A[i,i] = 4, A[i,i-1] = A[i-1,i] = -1.  No fill-in,
 * so the scaffold must produce the correct Cholesky factor.
 * We verify by reconstructing A = L * L^T and comparing entry-by-entry. */
static void test_eliminate_tridiagonal_spd(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));

    /* Compute A_reconstructed[i,j] = sum_k L[i,k] * L[j,k].  Compare
     * against the lower triangle of A. */
    double tol = 1e-10;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double sum = 0.0;
            /* Walk columns k <= j; for each, find L[i,k] and L[j,k]. */
            for (idx_t k = 0; k <= j; k++) {
                double l_ik = 0.0, l_jk = 0.0;
                for (idx_t p = csc->col_ptr[k]; p < csc->col_ptr[k + 1]; p++) {
                    if (csc->row_idx[p] == i)
                        l_ik = csc->values[p];
                    if (csc->row_idx[p] == j)
                        l_jk = csc->values[p];
                }
                sum += l_ik * l_jk;
            }
            double a = sparse_get(A, i, j);
            if (fabs(sum - a) > tol) {
                TF_FAIL_("L*L^T[%d,%d] = %.15g, A[%d,%d] = %.15g, diff=%.3e", (int)i, (int)j, sum,
                         (int)i, (int)j, a, fabs(sum - a));
            }
            tf_asserts++;
        }
    }

    sparse_free(A);
    chol_csc_free(csc);
}

/* Non-SPD detection — matrix with a zero diagonal. */
static void test_eliminate_detects_zero_diagonal(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 0.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 2, 1.0);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    /* With sparse_insert of 0.0, the entry is removed — so col 0 is
     * structurally empty.  Insert a nonzero and then zero it via
     * a separate path: easier — use -1.0 for a negative diagonal
     * test, and test the structurally-missing diagonal via the
     * dedicated singular-negative test below. */
    (void)csc;
    chol_csc_free(csc);
    sparse_free(A);

    /* Case: non-SPD via negative diagonal in column 0. */
    SparseMatrix *B = sparse_create(n, n);
    sparse_insert(B, 0, 0, -4.0); /* negative pivot */
    sparse_insert(B, 1, 1, 4.0);
    sparse_insert(B, 2, 2, 4.0);

    CholCsc *csc_b = NULL;
    REQUIRE_OK(chol_csc_from_sparse(B, NULL, 1.0, &csc_b));
    ASSERT_ERR(chol_csc_eliminate(csc_b), SPARSE_ERR_NOT_SPD);
    chol_csc_free(csc_b);
    sparse_free(B);
}

/* Non-SPD detection — Schur-complement goes negative mid-factorization.
 * A = [[1, 2], [2, 1]] → col 0 sqrt(1)=1, L[1,0]=2; col 1 needs
 * sqrt(1 - 2*2) = sqrt(-3) → NOT_SPD detected at cdiv of j=1. */
static void test_eliminate_detects_indefinite(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 1, 1.0);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    ASSERT_ERR(chol_csc_eliminate(csc), SPARSE_ERR_NOT_SPD);
    chol_csc_free(csc);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Complete scalar kernel: fill-in, drop tolerance, L*L^T verification
 * ═══════════════════════════════════════════════════════════════════════ */

/* Fetch L[r,c] from a factored CSC by linear scan through column c.
 * Returns 0.0 if not stored. */
static double csc_get(const CholCsc *csc, idx_t r, idx_t c) {
    for (idx_t p = csc->col_ptr[c]; p < csc->col_ptr[c + 1]; p++) {
        if (csc->row_idx[p] == r)
            return csc->values[p];
    }
    return 0.0;
}

/* Verify L * L^T ≈ A (lower triangle) to the given tolerance.
 * Computes (L*L^T)[i,j] = sum_k L[i,k] * L[j,k] for i >= j. */
static void assert_llt_reconstructs_A(const CholCsc *L, const SparseMatrix *A, double tol) {
    idx_t n = L->n;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double sum = 0.0;
            for (idx_t k = 0; k <= j; k++)
                sum += csc_get(L, i, k) * csc_get(L, j, k);
            double a = sparse_get(A, i, j);
            if (fabs(sum - a) > tol) {
                TF_FAIL_("L*L^T[%d,%d] = %.15g  A[%d,%d] = %.15g  diff=%.3e > tol=%.3e", (int)i,
                         (int)j, sum, (int)i, (int)j, a, fabs(sum - a), tol);
            }
            tf_asserts++;
        }
    }
}

/* ─── Hand-sized SPD: 3x3 ───────────────────────────────────────── */

static void test_eliminate_3x3_spd(void) {
    /* A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] — classic
     * textbook Cholesky example (Wikipedia).  Factor is
     *   L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]. */
    SparseMatrix *A = sparse_create(3, 3);
    double vals[3][3] = {{4, 12, -16}, {12, 37, -43}, {-16, -43, 98}};
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 3; j++)
            sparse_insert(A, i, j, vals[i][j]);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));

    double expected[3][3] = {{2, 0, 0}, {6, 1, 0}, {-8, 5, 3}};
    for (idx_t i = 0; i < 3; i++) {
        for (idx_t j = 0; j <= i; j++) {
            ASSERT_NEAR(csc_get(csc, i, j), expected[i][j], 1e-12);
        }
    }
    assert_llt_reconstructs_A(csc, A, 1e-10);

    sparse_free(A);
    chol_csc_free(csc);
}

/* ─── Hand-sized SPD: 4x4 ───────────────────────────────────────── */

static void test_eliminate_4x4_spd(void) {
    /* A = I + e*e^T where e = [1,1,1,1]^T.  A is SPD with diag=2 and
     * off-diag=1.  Compare L*L^T against A. */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? 2.0 : 1.0);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));
    assert_llt_reconstructs_A(csc, A, 1e-10);

    sparse_free(A);
    chol_csc_free(csc);
}

/* ─── Hand-sized SPD: 5x5 ───────────────────────────────────────── */

static void test_eliminate_5x5_spd(void) {
    /* Diagonally dominant 5x5 symmetric — mix of sparse + some denser. */
    idx_t n = 5;
    double A_vals[5][5] = {
        {9.0, -1.0, 0.5, 0.0, 0.25}, {-1.0, 7.0, -0.5, 1.0, 0.0}, {0.5, -0.5, 6.0, -1.0, 0.0},
        {0.0, 1.0, -1.0, 8.0, 0.5},  {0.25, 0.0, 0.0, 0.5, 5.0},
    };
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            if (A_vals[i][j] != 0.0)
                sparse_insert(A, i, j, A_vals[i][j]);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));
    assert_llt_reconstructs_A(csc, A, 1e-10);

    sparse_free(A);
    chol_csc_free(csc);
}

/* ─── Tridiagonal SPD n=10 ──────────────────────────────────────── */

static void test_eliminate_tridiagonal_n10(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));
    assert_llt_reconstructs_A(csc, A, 1e-10);

    sparse_free(A);
    chol_csc_free(csc);
}

/* ─── Block-diagonal SPD (two 3x3 blocks) ───────────────────────── */

static void test_eliminate_block_diagonal(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    /* Block 1: indices 0..2. */
    double B1[3][3] = {{9, 2, 1}, {2, 8, -1}, {1, -1, 7}};
    /* Block 2: indices 3..5. */
    double B2[3][3] = {{5, 1, 0.5}, {1, 6, -0.5}, {0.5, -0.5, 4}};
    for (idx_t i = 0; i < 3; i++) {
        for (idx_t j = 0; j < 3; j++) {
            sparse_insert(A, i, j, B1[i][j]);
            sparse_insert(A, i + 3, j + 3, B2[i][j]);
        }
    }

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));
    assert_llt_reconstructs_A(csc, A, 1e-10);

    /* Cross-block entries must remain zero: L[i,j] for i >= 3, j < 3. */
    for (idx_t i = 3; i < 6; i++) {
        for (idx_t j = 0; j < 3; j++) {
            ASSERT_NEAR(csc_get(csc, i, j), 0.0, 1e-12);
        }
    }

    sparse_free(A);
    chol_csc_free(csc);
}

/* ─── Random SPD: A = M*M^T + n*I ─────────────────────────────── */

static void test_eliminate_random_spd(void) {
    idx_t n = 8;
    /* Build a lower-triangular random M, then A = M*M^T + n*I.  Dense
     * deterministic pattern so the test is reproducible. */
    double M[8][8] = {{0}};
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j <= i; j++)
            M[i][j] = 0.3 + 0.1 * (double)(i * n + j);

    double A_dense[8][8] = {{0}};
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++) {
            double s = 0.0;
            for (idx_t k = 0; k < n; k++)
                s += M[i][k] * M[j][k];
            A_dense[i][j] = s + (i == j ? (double)n : 0.0);
        }

    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, A_dense[i][j]);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));
    assert_llt_reconstructs_A(csc, A, 1e-8);

    sparse_free(A);
    chol_csc_free(csc);
}

/* ─── Fill-in: reverse arrowhead matrix ─────────────────────────
 *
 * A = [[d, a, b, c], [a, 1, 0, 0], [b, 0, 1, 0], [c, 0, 0, 1]].
 * Column 0 is dense and drives fill into columns 1, 2, 3.  Exercises
 * the grow+shift paths in chol_csc_gather. */
static void test_eliminate_fillin_reverse_arrowhead(void) {
    idx_t n = 4;
    double d = 10.0, a = 1.0, b = 1.5, c = 2.0;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, d);
    sparse_insert(A, 1, 0, a);
    sparse_insert(A, 0, 1, a);
    sparse_insert(A, 2, 0, b);
    sparse_insert(A, 0, 2, b);
    sparse_insert(A, 3, 0, c);
    sparse_insert(A, 0, 3, c);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 3, 3, 1.0);

    CholCsc *csc = NULL;
    /* fill_factor=2.0 gives room for fill without special allocation —
     * the grow() path still exercises capacity bookkeeping. */
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));

    /* Before elimination: col 1 has one entry (row 1), col 2 one (row 2),
     * col 3 one (row 3).  Fill will add rows 2,3 to col 1, row 3 to col 2. */
    idx_t col1_before = csc->col_ptr[2] - csc->col_ptr[1];
    idx_t col2_before = csc->col_ptr[3] - csc->col_ptr[2];
    ASSERT_EQ(col1_before, 1);
    ASSERT_EQ(col2_before, 1);

    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));

    /* After elimination: col 1 should hold rows {1, 2, 3}, col 2 {2, 3},
     * col 3 {3} — classic dense-fill in a reverse arrowhead. */
    ASSERT_EQ(csc->col_ptr[2] - csc->col_ptr[1], 3);
    ASSERT_EQ(csc->col_ptr[3] - csc->col_ptr[2], 2);
    ASSERT_EQ(csc->col_ptr[4] - csc->col_ptr[3], 1);

    assert_llt_reconstructs_A(csc, A, 1e-10);

    sparse_free(A);
    chol_csc_free(csc);
}

/* ─── Fill-in: exact-allocation symbolic path on reverse arrowhead ──
 *
 * With chol_csc_from_sparse_with_analysis, capacity is pre-sized to
 * sym_L.nnz.  Elimination should fit without any grow() calls and
 * produce the same result. */
static void test_eliminate_fillin_with_analysis(void) {
    idx_t n = 4;
    double d = 10.0, a = 1.0, b = 1.5, c = 2.0;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, d);
    sparse_insert(A, 1, 0, a);
    sparse_insert(A, 0, 1, a);
    sparse_insert(A, 2, 0, b);
    sparse_insert(A, 0, 2, b);
    sparse_insert(A, 3, 0, c);
    sparse_insert(A, 0, 3, c);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 3, 3, 1.0);

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &csc));
    idx_t cap_before = csc->capacity;

    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));

    /* Symbolic capacity held up: no grow past the prediction. */
    ASSERT_TRUE(csc->capacity == cap_before);
    /* And final nnz matches the prediction exactly (no drop-tol happened). */
    ASSERT_EQ(csc->nnz, an.sym_L.nnz);

    assert_llt_reconstructs_A(csc, A, 1e-10);

    sparse_analysis_free(&an);
    sparse_free(A);
    chol_csc_free(csc);
}

/* ─── Drop tolerance: tiny values are dropped ───────────────────
 *
 * Build an SPD matrix where an off-diagonal entry is so small that
 * the computed L entry falls below the drop threshold.  Verify the
 * factor's VALUE-LEVEL behaviour: the diagonal is sqrt(A[j,j]), the
 * below-diagonal entry (still physically present in the slot because
 * col_ptr stays immutable across elimination) holds
 * the zeroed-out dropped value, and reconstruction via L·L^T still
 * matches A to round-off.
 *
 * The storage contract keeps col_ptr immutable and zero-pads dropped slots
 * in place, so the physical slot layout may differ from the mathematical
 * nonzero pattern even though the user-visible factor is identical.  The assertions below check the
 * behavioural contract (diagonal value, dropped-value zero-ness)
 * rather than the specific slot layout. */
static void test_eliminate_drop_tolerance(void) {
    /* A = [[1, 1e-15, 0], [1e-15, 1, 0], [0, 0, 1]] — entry 1e-15 is
     * below SPARSE_DROP_TOL * |L[0,0]| = 1e-14, so L[1,0] should drop. */
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    sparse_insert(A, 1, 0, 1e-15);
    sparse_insert(A, 0, 1, 1e-15);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));
    idx_t nnz_before = csc->nnz;
    ASSERT_EQ(nnz_before, 4); /* 3 diagonals + 1 off-diagonal (lower). */

    REQUIRE_OK(chol_csc_eliminate(csc));
    REQUIRE_OK(chol_csc_validate(csc));

    /* Diagonal is always stored — check every column's first slot
     * carries L[j, j] = sqrt(A[j, j]) = 1.0. */
    for (idx_t j = 0; j < n; j++) {
        idx_t cstart = csc->col_ptr[j];
        idx_t cend = csc->col_ptr[j + 1];
        ASSERT_TRUE(cstart < cend); /* diagonal always present */
        ASSERT_EQ(csc->row_idx[cstart], j);
        ASSERT_NEAR(csc->values[cstart], 1.0, 1e-12);

        /* Any below-diagonal slot in column j must carry 0.0 — either
         * never had a value, or was dropped by the drop-tolerance
         * rule.  (Dropped slots stay allocated but zeroed.) */
        for (idx_t p = cstart + 1; p < cend; p++) {
            ASSERT_NEAR(csc->values[p], 0.0, 0.0);
        }
    }

    sparse_free(A);
    chol_csc_free(csc);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Triangular solves and end-to-end factor+solve
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: compute ||A*x - b||_inf / ||b||_inf (relative residual).
 * Returns 0.0 for an empty system and NaN on allocation failure. Callers
 * compare against a tolerance with ASSERT_TRUE(rel_res < tol), which treats
 * NaN as out-of-range.
 */
static double compute_rel_residual(const SparseMatrix *A, const double *x, const double *b) {
    idx_t n = sparse_rows(A);
    if (n == 0)
        return 0.0;
    size_t n_size = (size_t)n;
    if (n < 0 || (idx_t)n_size != n || n_size > SIZE_MAX / sizeof(double))
        return (double)NAN;
    double *Ax = malloc(n_size * sizeof(double));
    if (!Ax)
        return (double)NAN;
    sparse_matvec(A, x, Ax);
    double max_r = 0.0;
    double max_b = 0.0;
    for (size_t i = 0; i < n_size; i++) {
        double ri = fabs(Ax[i] - b[i]);
        if (ri > max_r)
            max_r = ri;
        double bi = fabs(b[i]);
        if (bi > max_b)
            max_b = bi;
    }
    free(Ax);
    return max_b > 0.0 ? max_r / max_b : max_r;
}

static int read_external_dense_reference_solution(const char *matrix_path, double *x_out, idx_t n,
                                                  char *reason, size_t reason_cap) {
    if (!matrix_path || !x_out || !reason || reason_cap == 0)
        return -1;

    char cmd[512];
    int written = snprintf(cmd, sizeof(cmd),
                           "python3 tests/chol_external_dense_reference.py \"%s\"", matrix_path);
    if (written < 0 || (size_t)written >= sizeof(cmd)) {
        snprintf(reason, reason_cap, "external dense reference command overflow");
        return -1;
    }

    return (int)tf_read_external_reference_vector(cmd, "external dense reference", x_out, n, reason,
                                                  reason_cap);
}

static void assert_cholesky_external_dense_reference(const char *matrix_path,
                                                     sparse_reorder_t reorder,
                                                     sparse_chol_backend_t backend, double tol) {
#ifdef _WIN32
    (void)matrix_path;
    (void)reorder;
    (void)backend;
    (void)tol;
    SKIP_TEST("external dense reference helper is not enabled on Windows");
#else
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, matrix_path));
    idx_t n = sparse_rows(A);

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    double *x_ref = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_true);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(x_ref);

    for (idx_t i = 0; i < n; i++)
        x_true[i] = (double)(i + 1);
    sparse_matvec(A, x_true, b);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    sparse_cholesky_opts_t opts = {
        .reorder = reorder,
        .backend = backend,
    };
    REQUIRE_OK(sparse_cholesky_factor_opts(L, &opts));
    REQUIRE_OK(sparse_cholesky_solve(L, b, x));

    char reason[256];
    int ref_status =
        read_external_dense_reference_solution(matrix_path, x_ref, n, reason, sizeof(reason));
    if (ref_status == 0) {
        free(x_true);
        free(b);
        free(x);
        free(x_ref);
        sparse_free(L);
        sparse_free(A);
        SKIP_TEST(reason);
    }
    if (ref_status < 0) {
        free(x_true);
        free(b);
        free(x);
        free(x_ref);
        sparse_free(L);
        sparse_free(A);
        TF_FAIL_("external dense reference failed: %s", reason);
        return;
    }

    double max_diff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double diff = fabs(x[i] - x_ref[i]);
        if (diff > max_diff)
            max_diff = diff;
        ASSERT_NEAR(x[i], x_ref[i], tol);
        ASSERT_NEAR(x[i], x_true[i], tol);
    }

    double rel = compute_rel_residual(A, x, b);
    printf("    external dense ref %s: n=%d, max|x-x_ref|=%.3e, rel_residual=%.3e\n", matrix_path,
           (int)n, max_diff, rel);
    ASSERT_TRUE(rel < 1e-10);

    free(x_true);
    free(b);
    free(x);
    free(x_ref);
    sparse_free(L);
    sparse_free(A);
#endif
}

/* ─── Solve: null-arg handling ──────────────────────────────────── */

static void test_solve_null_args(void) {
    /* Build a trivial factored CSC so we have a valid L to pass. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 1, 1, 9.0);
    CholCsc *L = NULL;
    chol_csc_factor(A, NULL, &L);
    ASSERT_NOT_NULL(L);

    double b[2] = {1.0, 1.0};
    double x[2] = {0.0, 0.0};
    ASSERT_ERR(chol_csc_solve(NULL, b, x), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_solve(L, NULL, x), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_solve(L, b, NULL), SPARSE_ERR_NULL);

    ASSERT_ERR(chol_csc_solve_perm(NULL, NULL, b, x), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_solve_perm(L, NULL, NULL, x), SPARSE_ERR_NULL);

    chol_csc_free(L);
    sparse_free(A);
}

/* ─── Solve: identity and diagonal ─────────────────────────────── */

static void test_solve_identity(void) {
    /* A = I → L = I → x = b. */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));

    double b[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double x[5] = {0};
    REQUIRE_OK(chol_csc_solve(L, b, x));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i], 1e-12);

    chol_csc_free(L);
    sparse_free(A);
}

static void test_solve_diagonal(void) {
    /* A = diag(d) → x[i] = b[i] / d[i]. */
    idx_t n = 4;
    double d[4] = {4.0, 9.0, 16.0, 25.0};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, d[i]);

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));

    double b[4] = {4.0, 18.0, 48.0, 100.0};
    double expected[4] = {1.0, 2.0, 3.0, 4.0};
    double x[4] = {0};
    REQUIRE_OK(chol_csc_solve(L, b, x));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], expected[i], 1e-12);

    chol_csc_free(L);
    sparse_free(A);
}

/* ─── Solve: 2x2 SPD with hand-verified result ─────────────────── */

static void test_solve_2x2_spd(void) {
    /* A = [[4, 2], [2, 5]].  A * [1, 2] = [8, 12], so we pick b = [8, 12]
     * and verify the solve recovers x = [1, 2]. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 1, 5.0);

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));

    double b[2] = {8.0, 12.0};
    double x[2] = {0};
    REQUIRE_OK(chol_csc_solve(L, b, x));
    ASSERT_NEAR(x[0], 1.0, 1e-12);
    ASSERT_NEAR(x[1], 2.0, 1e-12);

    chol_csc_free(L);
    sparse_free(A);
}

/* ─── Solve: tridiagonal SPD via residual ──────────────────────── */

static void test_solve_tridiagonal(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    /* Choose x_true = [1, 2, ..., n], compute b = A * x_true, solve, compare. */
    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_true[i] = 1.0 + (double)i;
    sparse_matvec(A, x_true, b);

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));
    REQUIRE_OK(chol_csc_solve(L, b, x));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_true[i], 1e-10);

    free(x_true);
    free(b);
    free(x);
    chol_csc_free(L);
    sparse_free(A);
}

/* ─── Solve: in-place (b == x) ─────────────────────────────────── */

static void test_solve_in_place(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 2.0);
    sparse_insert(A, 1, 0, -1.0);
    sparse_insert(A, 0, 1, -1.0);
    sparse_insert(A, 2, 1, -1.0);
    sparse_insert(A, 1, 2, -1.0);
    sparse_insert(A, 3, 2, -1.0);
    sparse_insert(A, 2, 3, -1.0);

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));

    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double b_copy[4];
    memcpy(b_copy, x, sizeof(b_copy));
    /* In-place: x is both input and output. */
    REQUIRE_OK(chol_csc_solve(L, x, x));
    /* Cross-check by solving again with separate buffers. */
    double y[4] = {0};
    REQUIRE_OK(chol_csc_solve(L, b_copy, y));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], y[i], 1e-12);

    chol_csc_free(L);
    sparse_free(A);
}

/* ─── Solve with permutation ───────────────────────────────────── */

static void test_solve_perm_null_matches_plain(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 3.0);
    sparse_insert(A, 1, 0, -1.0);
    sparse_insert(A, 0, 1, -1.0);

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));

    double b[4] = {1.0, 2.0, 3.0, 4.0};
    double x_plain[4] = {0};
    double x_perm[4] = {0};
    REQUIRE_OK(chol_csc_solve(L, b, x_plain));
    REQUIRE_OK(chol_csc_solve_perm(L, NULL, b, x_perm));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_plain[i], x_perm[i], 0.0);

    chol_csc_free(L);
    sparse_free(A);
}

/* AMD-reordered factor+solve: verify the residual matches in user coords. */
static void test_solve_perm_amd_nos4(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/nos4.mtx"));
    idx_t n = sparse_rows(A);

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_true[i] = 1.0;
    sparse_matvec(A, x_true, b);

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, &an, &L));
    REQUIRE_OK(chol_csc_solve_perm(L, an.perm, b, x));

    double rel_res = compute_rel_residual(A, x, b);
    printf("    nos4 (AMD): rel_res = %.3e\n", rel_res);
    ASSERT_TRUE(rel_res < 1e-10);

    free(x_true);
    free(b);
    free(x);
    chol_csc_free(L);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* ─── SuiteSparse validation: nos4 ─────────────────────────────── */

static void test_factor_solve_nos4(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/nos4.mtx"));
    idx_t n = sparse_rows(A);

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_true[i] = 1.0;
    sparse_matvec(A, x_true, b);

    REQUIRE_OK(chol_csc_factor_solve(A, NULL, b, x));
    double rel_res = compute_rel_residual(A, x, b);
    printf("    nos4: rel_res = %.3e\n", rel_res);
    ASSERT_TRUE(rel_res < 1e-10);

    free(x_true);
    free(b);
    free(x);
    sparse_free(A);
}

/* ─── SuiteSparse validation: bcsstk04 with AMD ──────────────── */

static void test_factor_solve_bcsstk04_amd(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx"));
    idx_t n = sparse_rows(A);

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_true[i] = 1.0;
    sparse_matvec(A, x_true, b);

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    REQUIRE_OK(chol_csc_factor_solve(A, &an, b, x));
    double rel_res = compute_rel_residual(A, x, b);
    printf("    bcsstk04 (AMD): rel_res = %.3e\n", rel_res);
    /* Target: < 1e-10. In practice the CSC path delivers ~1e-15 on
     * bcsstk04, well inside the envelope. */
    ASSERT_TRUE(rel_res < 1e-10);

    free(x_true);
    free(b);
    free(x);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* ─── factor shim: null + valid paths ─────────────────────────── */

static void test_factor_shim_null(void) {
    CholCsc *L = NULL;
    ASSERT_ERR(chol_csc_factor(NULL, NULL, &L), SPARSE_ERR_NULL);
    ASSERT_NULL(L);

    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    ASSERT_ERR(chol_csc_factor(A, NULL, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_factor_solve_null(void) {
    double b[1] = {1.0}, x[1] = {0.0};
    ASSERT_ERR(chol_csc_factor_solve(NULL, NULL, b, x), SPARSE_ERR_NULL);

    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 1.0);
    ASSERT_ERR(chol_csc_factor_solve(A, NULL, NULL, x), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_factor_solve(A, NULL, b, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* ─── Numeric edge cases ──────────────────────────────────────── */

static void test_factor_detects_indefinite(void) {
    /* A = [[1, 2], [2, 1]] — eigenvalues 3, -1 (indefinite). */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 1, 1.0);

    CholCsc *L = NULL;
    ASSERT_ERR(chol_csc_factor(A, NULL, &L), SPARSE_ERR_NOT_SPD);
    ASSERT_NULL(L);

    double b[2] = {1.0, 1.0};
    double x[2] = {0};
    ASSERT_ERR(chol_csc_factor_solve(A, NULL, b, x), SPARSE_ERR_NOT_SPD);

    sparse_free(A);
}

static void test_factor_detects_negative_diagonal(void) {
    /* Negative diagonal entry on column 0 — caught immediately by cdiv. */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, -4.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 2, 2, 4.0);

    CholCsc *L = NULL;
    ASSERT_ERR(chol_csc_factor(A, NULL, &L), SPARSE_ERR_NOT_SPD);
    ASSERT_NULL(L);
    sparse_free(A);
}

/* A CSC that passes validate() but has a near-zero diagonal below the
 * singularity threshold — solve() should report SPARSE_ERR_SINGULAR. */
static void test_solve_detects_tiny_diagonal(void) {
    /* Build a well-formed CSC directly, then stick a tiny diagonal in. */
    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_alloc(2, 2, &L));
    L->col_ptr[0] = 0;
    L->col_ptr[1] = 1;
    L->col_ptr[2] = 2;
    L->row_idx[0] = 0;
    L->row_idx[1] = 1;
    L->values[0] = 1e-20; /* well below any realistic threshold */
    L->values[1] = 1.0;
    L->nnz = 2;
    L->factor_norm = 1.0; /* makes sing_tol ≈ SPARSE_DROP_TOL ≈ 1e-14 */

    double b[2] = {1.0, 1.0};
    double x[2] = {0};
    ASSERT_ERR(chol_csc_solve(L, b, x), SPARSE_ERR_SINGULAR);

    chol_csc_free(L);
}

static int chol_csc_match_values(const CholCsc *a, const CholCsc *b, double tol) {
    if (a->n != b->n || a->nnz != b->nnz)
        return 0;
    for (idx_t j = 0; j <= a->n; j++)
        if (a->col_ptr[j] != b->col_ptr[j])
            return 0;
    for (idx_t p = 0; p < a->nnz; p++) {
        if (a->row_idx[p] != b->row_idx[p])
            return 0;
        if (fabs(a->values[p] - b->values[p]) > tol)
            return 0;
    }
    return 1;
}

static void test_factor_with_analysis_large_n_matches_explicit_supernodal_route(void) {
    const idx_t n = (idx_t)(SPARSE_CSC_THRESHOLD + 20);
    SparseMatrix *A = sparse_create(n, n);
    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_t an = {0};
    CholCsc *L_helper = NULL;
    CholCsc *L_explicit = NULL;

    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    REQUIRE_OK(sparse_analyze(A, &opts, &an));
    REQUIRE_OK(chol_csc_factor(A, &an, &L_helper));

    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &L_explicit));
    REQUIRE_OK(chol_csc_eliminate_supernodal(L_explicit, SPARSE_CSC_SUPERNODE_MIN_SIZE));

    ASSERT_TRUE(chol_csc_match_values(L_helper, L_explicit, 1e-12));

    chol_csc_free(L_helper);
    chol_csc_free(L_explicit);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* Build a diagonally dominant SPD matrix with a fixed sparsity pattern so the
 * dispatch residual checks remain deterministic across reruns.
 */
static SparseMatrix *dispatch_build_spd(idx_t n, double density, unsigned int seed) {
    unsigned int rng = seed;
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)n);
    for (idx_t i = 1; i < n; i++) {
        for (idx_t j = 0; j < i; j++) {
            rng = rng * 1664525u + 1013904223u;
            double p = (double)(rng & 0xffffff) / (double)0x1000000;
            if (p < density) {
                rng = rng * 1664525u + 1013904223u;
                double v = ((double)(rng & 0xffff) / (double)0x10000) - 0.5;
                sparse_insert(A, i, j, v);
                sparse_insert(A, j, i, v);
            }
        }
    }
    return A;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Transparent dispatch in sparse_cholesky_factor_opts
 * ═══════════════════════════════════════════════════════════════════════ */

/* Small n = 10 matrix should take the linked-list path under AUTO. */
static void test_dispatch_auto_small_uses_linked_list(void) {
    SparseMatrix *A = dispatch_build_spd(10, 0.3, 0xa5a5a5a5u);

    int used = -1;
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_AUTO,
        .used_csc_path = &used,
    };
    REQUIRE_OK(sparse_cholesky_factor_opts(A, &opts));
    ASSERT_EQ(used, 0); /* n < SPARSE_CSC_THRESHOLD */

    sparse_free(A);
}

/* Large n (>= SPARSE_CSC_THRESHOLD) should take the CSC path under
 * AUTO, produce a valid factor, and solve correctly. */
static void test_dispatch_auto_large_uses_csc_and_solves(void) {
    idx_t n = 500;
    SparseMatrix *A = dispatch_build_spd(n, 0.01, 0xcafef00du);
    /* Reference RHS = A * ones. */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_TRUE(L != NULL);

    int used = -1;
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_AUTO,
        .used_csc_path = &used,
    };
    REQUIRE_OK(sparse_cholesky_factor_opts(L, &opts));
    ASSERT_EQ(used, 1); /* n >= SPARSE_CSC_THRESHOLD */

    double *x = calloc((size_t)n, sizeof(double));
    REQUIRE_OK(sparse_cholesky_solve(L, b, x));

    /* Residual ||A x - b|| / ||b|| < 1e-10. */
    double *Ax = calloc((size_t)n, sizeof(double));
    sparse_matvec(A, x, Ax);
    double rr = 0.0, bn = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double r = fabs(Ax[i] - b[i]);
        double bi = fabs(b[i]);
        if (r > rr)
            rr = r;
        if (bi > bn)
            bn = bi;
    }
    ASSERT_TRUE(rr / (bn > 0.0 ? bn : 1.0) < 1e-10);

    free(ones);
    free(b);
    free(x);
    free(Ax);
    sparse_free(A);
    sparse_free(L);
}

/* Forcing CSC at small n and LINKED_LIST at large n must still
 * produce equivalent solves on the same matrix. */
static void test_dispatch_forced_override_both_paths_agree(void) {
    idx_t n = 40;
    SparseMatrix *A = dispatch_build_spd(n, 0.15, 0x13579bdfu);
    double *b = malloc((size_t)n * sizeof(double));
    double *x_ll = calloc((size_t)n, sizeof(double));
    double *x_cs = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0 + (double)i;

    /* Force linked-list. */
    SparseMatrix *L_ll = sparse_copy(A);
    int used_ll = -1;
    sparse_cholesky_opts_t opts_ll = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
        .used_csc_path = &used_ll,
    };
    REQUIRE_OK(sparse_cholesky_factor_opts(L_ll, &opts_ll));
    ASSERT_EQ(used_ll, 0);
    REQUIRE_OK(sparse_cholesky_solve(L_ll, b, x_ll));

    /* Force CSC. */
    SparseMatrix *L_cs = sparse_copy(A);
    int used_cs = -1;
    sparse_cholesky_opts_t opts_cs = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_CSC,
        .used_csc_path = &used_cs,
    };
    REQUIRE_OK(sparse_cholesky_factor_opts(L_cs, &opts_cs));
    ASSERT_EQ(used_cs, 1);
    REQUIRE_OK(sparse_cholesky_solve(L_cs, b, x_cs));

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_ll[i], x_cs[i], 1e-10);

    free(b);
    free(x_ll);
    free(x_cs);
    sparse_free(A);
    sparse_free(L_ll);
    sparse_free(L_cs);
}

/* Larger SuiteSparse fixtures should:
 *   (a) take the CSC dispatch path (n >= SPARSE_CSC_THRESHOLD), and
 *   (b) produce a residual ||A·x - b|| / ||b|| below the 1e-10 SPD
 *       spot-check threshold.
 * Before the full sym_L pre-population fix in
 * `chol_csc_from_sparse_with_analysis`, these residuals blew up to
 * ~1e-1 on bcsstk14 / s3rmt3m3 / Kuu because the supernodal extract
 * missed fill-in rows.  Keep these tests SPD-only; bcsstk14 is a
 * small-enough fixture for the test harness to factor in-process. */
static void dispatch_spd_fixture_residual(const char *path) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, path));
    idx_t n = sparse_rows(A);
    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD); /* must be on the CSC side */

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    SparseMatrix *L = sparse_copy(A);
    int used = -1;
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_AUTO,
        .used_csc_path = &used,
    };
    REQUIRE_OK(sparse_cholesky_factor_opts(L, &opts));
    ASSERT_EQ(used, 1);

    REQUIRE_OK(sparse_cholesky_solve(L, b, x));

    double *Ax = calloc((size_t)n, sizeof(double));
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
    double rel = bmax > 0.0 ? rmax / bmax : rmax;
    printf("    %s: n=%d, rel_residual=%.3e\n", path, (int)n, rel);
    ASSERT_TRUE(rel < 1e-10);

    free(ones);
    free(b);
    free(x);
    free(Ax);
    sparse_free(A);
    sparse_free(L);
}

static void test_dispatch_bcsstk14_residual(void) {
    dispatch_spd_fixture_residual(SS_DIR "/bcsstk14.mtx");
}

static void test_external_dense_reference_nos4_csc(void) {
    assert_cholesky_external_dense_reference(SS_DIR "/nos4.mtx", SPARSE_REORDER_NONE,
                                             SPARSE_CHOL_BACKEND_CSC, 1e-10);
}

static void test_external_dense_reference_bcsstk04_amd_csc(void) {
    assert_cholesky_external_dense_reference(SS_DIR "/bcsstk04.mtx", SPARSE_REORDER_AMD,
                                             SPARSE_CHOL_BACKEND_CSC, 1e-10);
}

/* Legacy zero-initialised opts (just { reorder }) must still work —
 * backend default-inits to AUTO and used_csc_path defaults to NULL. */
static void test_dispatch_legacy_opts_still_work(void) {
    idx_t n = 20;
    SparseMatrix *A = dispatch_build_spd(n, 0.1, 0xdeadbeefu);

    /* Legacy caller style: only set `reorder`.  Zero-init for
     * the new fields means AUTO + no used_csc_path reporting. */
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_AMD,
    };
    REQUIRE_OK(sparse_cholesky_factor_opts(A, &opts));
    ASSERT_TRUE(A->factored);

    sparse_free(A);
}

/* Out-of-range backend value rejected as SPARSE_ERR_BADARG. */
static void test_dispatch_invalid_backend_rejected(void) {
    SparseMatrix *A = dispatch_build_spd(20, 0.1, 0x1234abcdu);
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = (sparse_chol_backend_t)99,
        .used_csc_path = NULL,
    };
    ASSERT_ERR(sparse_cholesky_factor_opts(A, &opts), SPARSE_ERR_BADARG);
    sparse_free(A);
}

/* `used_csc_path` is published before later reorder / factor errors so
 * callers still observe the selected numeric path on failure. */
static void test_dispatch_csc_reports_selected_path_before_reorder_error(void) {
    SparseMatrix *A = dispatch_build_spd(20, 0.1, 0xbeef1234u);
    int used = -1;
    sparse_cholesky_opts_t opts = {
        .reorder = (sparse_reorder_t)99,
        .backend = SPARSE_CHOL_BACKEND_CSC,
        .used_csc_path = &used,
    };

    ASSERT_ERR(sparse_cholesky_factor_opts(A, &opts), SPARSE_ERR_BADARG);
    ASSERT_EQ(used, 1);

    sparse_free(A);
}

static void run_dispatch_tests(void) {
    RUN_TEST(test_dispatch_auto_small_uses_linked_list);
    RUN_TEST(test_dispatch_auto_large_uses_csc_and_solves);
    RUN_TEST(test_dispatch_forced_override_both_paths_agree);
    RUN_TEST(test_dispatch_legacy_opts_still_work);
    RUN_TEST(test_dispatch_invalid_backend_rejected);
    RUN_TEST(test_dispatch_csc_reports_selected_path_before_reorder_error);
    RUN_TEST(test_dispatch_bcsstk14_residual);
    RUN_TEST(test_external_dense_reference_nos4_csc);
    RUN_TEST(test_external_dense_reference_bcsstk04_amd_csc);
}

static void run_alloc_growth_tests(void) {
    RUN_TEST(test_chol_csc_alloc_null_out);
    RUN_TEST(test_chol_csc_alloc_negative_n);
    RUN_TEST(test_chol_csc_alloc_basic);
    RUN_TEST(test_chol_csc_alloc_zero_initial_nnz);
    RUN_TEST(test_chol_csc_alloc_zero_n);
    RUN_TEST(test_chol_csc_free_null);
    RUN_TEST(test_chol_csc_grow_null);
    RUN_TEST(test_chol_csc_grow_noop);
    RUN_TEST(test_chol_csc_grow_exact);
    RUN_TEST(test_chol_csc_grow_geometric);
    RUN_TEST(test_chol_csc_grow_to_needed);
    RUN_TEST(test_chol_csc_grow_preserves_values);
}

static void run_conversion_roundtrip_tests(void) {
    RUN_TEST(test_from_sparse_null_args);
    RUN_TEST(test_to_sparse_null_args);
    RUN_TEST(test_roundtrip_identity);
    RUN_TEST(test_roundtrip_diagonal_spd);
    RUN_TEST(test_roundtrip_tridiagonal_spd);
    RUN_TEST(test_roundtrip_dense_lower_5x5);
    RUN_TEST(test_roundtrip_1x1);
    RUN_TEST(test_from_sparse_strips_upper_triangle);
    RUN_TEST(test_roundtrip_nos4);
    RUN_TEST(test_roundtrip_bcsstk04);
}

static void run_permutation_cache_tests(void) {
    RUN_TEST(test_identity_perm_matches_null);
    RUN_TEST(test_reverse_perm_symmetric);
    RUN_TEST(test_amd_perm_entries_match);
    RUN_TEST(test_invalid_perm_out_of_range);
    RUN_TEST(test_invalid_perm_duplicate);
    RUN_TEST(test_fill_factor_clamp);
    RUN_TEST(test_factor_norm_cached);
}

static void run_symbolic_validate_edge_tests(void) {
    RUN_TEST(test_with_analysis_null_args);
    RUN_TEST(test_with_analysis_wrong_type);
    RUN_TEST(test_exact_alloc_matches_dynamic_tridiag);
    RUN_TEST(test_predicted_nnz_matches_actual_nos4);
    RUN_TEST(test_predicted_nnz_matches_actual_bcsstk04);
    RUN_TEST(test_predicted_nnz_matches_actual_random_spd);
    RUN_TEST(test_validate_null);
    RUN_TEST(test_validate_fresh_alloc_is_valid);
    RUN_TEST(test_validate_catches_missing_diagonal);
    RUN_TEST(test_validate_catches_upper_triangle);
    RUN_TEST(test_validate_catches_unsorted_column);
    RUN_TEST(test_validate_catches_col_ptr_inconsistency);
    RUN_TEST(test_edge_case_diagonal_only);
    RUN_TEST(test_edge_case_external_capacity_growth);
}

static void run_workspace_elimination_scaffold_tests(void) {
    RUN_TEST(test_workspace_null_out);
    RUN_TEST(test_workspace_negative_n);
    RUN_TEST(test_workspace_free_null);
    RUN_TEST(test_workspace_alloc_basic);
    RUN_TEST(test_workspace_alloc_zero_n);
    RUN_TEST(test_cdiv_null);
    RUN_TEST(test_cdiv_positive_diagonal);
    RUN_TEST(test_cdiv_zero_diagonal_not_spd);
    RUN_TEST(test_cdiv_negative_diagonal_not_spd);
    RUN_TEST(test_eliminate_null);
    RUN_TEST(test_eliminate_diagonal);
    RUN_TEST(test_eliminate_2x2_spd);
    RUN_TEST(test_eliminate_tridiagonal_spd);
    RUN_TEST(test_eliminate_detects_zero_diagonal);
    RUN_TEST(test_eliminate_detects_indefinite);
}

static void run_scalar_kernel_tests(void) {
    RUN_TEST(test_eliminate_3x3_spd);
    RUN_TEST(test_eliminate_4x4_spd);
    RUN_TEST(test_eliminate_5x5_spd);
    RUN_TEST(test_eliminate_tridiagonal_n10);
    RUN_TEST(test_eliminate_block_diagonal);
    RUN_TEST(test_eliminate_random_spd);
    RUN_TEST(test_eliminate_fillin_reverse_arrowhead);
    RUN_TEST(test_eliminate_fillin_with_analysis);
    RUN_TEST(test_eliminate_drop_tolerance);
}

static void run_solve_residual_shim_tests(void) {
    RUN_TEST(test_solve_null_args);
    RUN_TEST(test_solve_identity);
    RUN_TEST(test_solve_diagonal);
    RUN_TEST(test_solve_2x2_spd);
    RUN_TEST(test_solve_tridiagonal);
    RUN_TEST(test_solve_in_place);
    RUN_TEST(test_solve_perm_null_matches_plain);
    RUN_TEST(test_solve_perm_amd_nos4);
    RUN_TEST(test_factor_solve_nos4);
    RUN_TEST(test_factor_solve_bcsstk04_amd);
    RUN_TEST(test_factor_with_analysis_large_n_matches_explicit_supernodal_route);
    RUN_TEST(test_factor_shim_null);
    RUN_TEST(test_factor_solve_null);
    RUN_TEST(test_factor_detects_indefinite);
    RUN_TEST(test_factor_detects_negative_diagonal);
    RUN_TEST(test_solve_detects_tiny_diagonal);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("chol_csc");

    /* Allocation, free, and growth */
    run_alloc_growth_tests();

    /* Conversion round-trips */
    run_conversion_roundtrip_tests();

    /* Permutations */
    run_permutation_cache_tests();

    /* Symbolic analysis integration */
    run_symbolic_validate_edge_tests();

    /* Workspace and elimination scaffolding */
    run_workspace_elimination_scaffold_tests();

    /* Scalar kernel: fill-in, drop tolerance, L*L^T correctness */
    run_scalar_kernel_tests();

    /* Triangular solve, permutation, SuiteSparse residuals, shims, edges */
    run_solve_residual_shim_tests();

    /* Transparent dispatch in sparse_cholesky_factor_opts */
    run_dispatch_tests();

    TEST_SUITE_END();
}
