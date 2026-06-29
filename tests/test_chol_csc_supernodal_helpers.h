#ifndef TEST_CHOL_CSC_SUPERNODAL_HELPERS_H
#define TEST_CHOL_CSC_SUPERNODAL_HELPERS_H

#include "sparse_chol_csc_internal.h"
#include "sparse_cholesky.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "test_framework.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Family-local helper seam for the Cholesky CSC supernodal/writeback proof
 * owner. Keep this narrow and specific instead of widening the shared solver
 * test helper layer with CSC-family details.
 */

/* Linear scan helper: return L[i, j] from a factored CholCsc, or 0.0
 * if not stored. Used by the supernode diagonal-block reference checks.
 */
static double chol_csc_value_at(const CholCsc *csc, idx_t i, idx_t j) {
    if (j < 0 || j >= csc->n)
        return 0.0;
    for (idx_t p = csc->col_ptr[j]; p < csc->col_ptr[j + 1]; p++) {
        if (csc->row_idx[p] == i)
            return csc->values[p];
    }
    return 0.0;
}

/* Compare two factored CholCsc values structurally and numerically.
 * Returns 1 on match, 0 on divergence.
 */
static int chol_csc_values_match(const CholCsc *a, const CholCsc *b, double tol) {
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

/* Helper: compute ||A*x - b||_inf / ||b||_inf (relative residual).
 * Returns the residual norm, or NaN on allocation failure. Callers compare
 * against a tolerance with ASSERT_TRUE(rel_res < tol), which treats NaN as
 * out-of-range.
 */
static double compute_rel_residual(const SparseMatrix *A, const double *x, const double *b) {
    idx_t n = sparse_rows(A);
    double *Ax = malloc((size_t)n * sizeof(double));
    if (!Ax)
        return (double)NAN;
    sparse_matvec(A, x, Ax);
    double max_r = 0.0;
    double max_b = 0.0;
    for (idx_t i = 0; i < n; i++) {
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

static void detect_supernodes_alloc(const CholCsc *L, idx_t min_size, idx_t **starts_out,
                                    idx_t **sizes_out, idx_t *count_out) {
    idx_t n = L->n;
    *starts_out = NULL;
    *sizes_out = NULL;
    *count_out = 0;
    idx_t *starts = malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
    idx_t *sizes = malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
    if (!starts || !sizes) {
        free(starts);
        free(sizes);
        REQUIRE_OK(SPARSE_ERR_ALLOC);
    }
    idx_t count = 0;
    REQUIRE_OK(chol_csc_detect_supernodes(L, min_size, starts, sizes, &count));
    *starts_out = starts;
    *sizes_out = sizes;
    *count_out = count;
}

/* Count the grouped supernode columns for size >= min_size by converting a
 * sparse factored `L` back to `CholCsc` and rerunning supernode detection.
 * Return -1 on failure so callers can skip the derived check cleanly.
 */
static idx_t count_grouped_supernode_columns(const SparseMatrix *L_sparse, idx_t min_size,
                                             idx_t *count_out) {
    CholCsc *L = NULL;
    if (chol_csc_from_sparse(L_sparse, NULL, 2.0, &L) != SPARSE_OK)
        return -1;
    idx_t n = L->n;
    idx_t *starts = malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
    idx_t *sizes = malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
    idx_t count = 0;
    if (!starts || !sizes ||
        chol_csc_detect_supernodes(L, min_size, starts, sizes, &count) != SPARSE_OK) {
        free(starts);
        free(sizes);
        chol_csc_free(L);
        return -1;
    }
    idx_t total = 0;
    for (idx_t i = 0; i < count; i++)
        total += sizes[i];
    free(starts);
    free(sizes);
    chol_csc_free(L);
    *count_out = count;
    return total;
}

/* Factor `A` through the scalar and supernodal paths and assert the two
 * factored CSC values still match within `tol`.
 */
static void assert_supernodal_matches_scalar(const SparseMatrix *A, const idx_t *perm,
                                             idx_t min_size, double tol, const char *label) {
    (void)label; /* reserved for future diagnostic messages */
    CholCsc *Ls = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, perm, 2.0, &Ls));
    REQUIRE_OK(chol_csc_eliminate(Ls));

    CholCsc *Ln = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, perm, 2.0, &Ln));
    REQUIRE_OK(chol_csc_eliminate_supernodal(Ln, min_size));
    REQUIRE_OK(chol_csc_validate(Ln));

    ASSERT_TRUE(chol_csc_values_match(Ls, Ln, tol));

    chol_csc_free(Ls);
    chol_csc_free(Ln);
}

/* Field-by-field comparison helper. Returns 1 iff every checked invariant on
 * the scalar-factored `ref` matches the writeback-factored `got`.
 */
static int factored_sparse_matches(const SparseMatrix *ref, const SparseMatrix *got, double tol) {
    idx_t n = sparse_rows(ref);
    if (sparse_rows(got) != n || sparse_cols(got) != sparse_cols(ref)) {
        fprintf(stderr, "writeback: shape mismatch\n");
        return 0;
    }
    if (!ref->factored || !got->factored) {
        fprintf(stderr, "writeback: factored flag mismatch ref=%d got=%d\n", ref->factored,
                got->factored);
        return 0;
    }
    {
        double diff = fabs(ref->factor_norm - got->factor_norm);
        double scale = fabs(ref->factor_norm) > 1.0 ? fabs(ref->factor_norm) : 1.0;
        if (diff > 1e-12 * scale) {
            fprintf(stderr, "writeback: factor_norm mismatch ref=%.17g got=%.17g (rel %.3e)\n",
                    ref->factor_norm, got->factor_norm, diff / scale);
            return 0;
        }
    }
    if ((ref->reorder_perm == NULL) != (got->reorder_perm == NULL)) {
        fprintf(stderr, "writeback: reorder_perm NULLness mismatch ref=%p got=%p\n",
                (void *)ref->reorder_perm, (void *)got->reorder_perm);
        return 0;
    }
    if (ref->reorder_perm) {
        for (idx_t i = 0; i < n; i++) {
            if (ref->reorder_perm[i] != got->reorder_perm[i]) {
                fprintf(stderr, "writeback: reorder_perm[%d] mismatch ref=%d got=%d\n", (int)i,
                        (int)ref->reorder_perm[i], (int)got->reorder_perm[i]);
                return 0;
            }
        }
    }
    for (idx_t i = 0; i < n; i++) {
        if (ref->row_perm[i] != i || ref->col_perm[i] != i || ref->inv_row_perm[i] != i ||
            ref->inv_col_perm[i] != i) {
            fprintf(stderr, "writeback: ref internal perm not identity at i=%d\n", (int)i);
            return 0;
        }
        if (got->row_perm[i] != i || got->col_perm[i] != i || got->inv_row_perm[i] != i ||
            got->inv_col_perm[i] != i) {
            fprintf(stderr, "writeback: got internal perm not identity at i=%d\n", (int)i);
            return 0;
        }
    }
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            double a = sparse_get(ref, i, j);
            double b = sparse_get(got, i, j);
            if (fabs(a - b) > tol) {
                fprintf(stderr, "writeback: value mismatch at (%d,%d) ref=%.17g got=%.17g\n",
                        (int)i, (int)j, a, b);
                return 0;
            }
        }
    }
    return 1;
}

/* Build a test matrix A, factor it via both paths, and assert the writeback
 * matches the scalar reference field-by-field.
 */
static void writeback_roundtrip_check(SparseMatrix *A, int use_amd, double tol) {
    idx_t n = sparse_rows(A);

    SparseMatrix *ref = sparse_copy(A);
    ASSERT_TRUE(ref != NULL);
    sparse_cholesky_opts_t opts = {
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
        .reorder = use_amd ? SPARSE_REORDER_AMD : SPARSE_REORDER_NONE,
    };
    REQUIRE_OK(sparse_cholesky_factor_opts(ref, &opts));

    idx_t *perm = NULL;
    if (use_amd && n > 1) {
        perm = malloc((size_t)n * sizeof(idx_t));
        REQUIRE_OK(sparse_reorder_amd(A, perm));
    }
    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, perm, 2.0, &L));
    REQUIRE_OK(chol_csc_eliminate(L));

    SparseMatrix *got = sparse_copy(A);
    ASSERT_TRUE(got != NULL);
    REQUIRE_OK(chol_csc_writeback_to_sparse(L, got, perm));

    ASSERT_TRUE(factored_sparse_matches(ref, got, tol));

    free(perm);
    chol_csc_free(L);
    sparse_free(ref);
    sparse_free(got);
}

#endif
