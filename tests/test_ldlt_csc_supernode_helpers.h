#ifndef TEST_LDLT_CSC_SUPERNODE_HELPERS_H
#define TEST_LDLT_CSC_SUPERNODE_HELPERS_H

#include "sparse_chol_csc_internal.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "test_framework.h"

#include <math.h>
#include <stdlib.h>

/* Family-local helper seam for the LDLT CSC supernode proof surface. Keep this
 * header included by test_ldlt_csc.c so the existing test binary remains the
 * only proof owner.
 */

/* Helper: build an LdltCsc whose embedded L is "fully dense lower triangular"
 * (every row `i >= j` stored in column `j`) and whose pivot_size array is the
 * caller-supplied pattern. Lets detection tests focus on boundary behavior
 * without running a real factor.
 */
static LdltCsc *build_dense_ldlt_with_pivots(idx_t n, const idx_t *pivot_size) {
    LdltCsc *F = NULL;
    if (ldlt_csc_alloc(n, n * (n + 1) / 2, &F) != SPARSE_OK)
        return NULL;
    CholCsc *L = F->L;
    idx_t p = 0;
    for (idx_t j = 0; j < n; j++) {
        L->col_ptr[j] = p;
        for (idx_t i = j; i < n; i++) {
            L->row_idx[p] = i;
            L->values[p] = (i == j) ? 1.0 : 0.1;
            p++;
        }
    }
    L->col_ptr[n] = p;
    L->nnz = p;
    for (idx_t k = 0; k < n; k++)
        F->pivot_size[k] = pivot_size[k];
    return F;
}

/* Compute row by column to linear index for a column-major buffer with leading
 * dimension `lda`.
 */
static inline idx_t cm_idx(idx_t row, idx_t col, idx_t lda) { return row + col * lda; }

/* Snapshot helper: copy `F->L->values`, `F->D`, `F->D_offdiag`,
 * `F->pivot_size` for the supernode column range so we can verify the
 * round-trip is the identity.
 */
static void snapshot_supernode_state(const LdltCsc *F, idx_t s_start, idx_t s_size,
                                     double *L_values_copy, idx_t *L_nnz_in_block, double *D_copy,
                                     double *D_offdiag_copy, idx_t *pivot_size_copy) {
    idx_t cstart = F->L->col_ptr[s_start];
    idx_t cend = F->L->col_ptr[s_start + s_size];
    *L_nnz_in_block = cend - cstart;
    for (idx_t p = 0; p < cend - cstart; p++)
        L_values_copy[p] = F->L->values[cstart + p];
    for (idx_t j = 0; j < s_size; j++) {
        D_copy[j] = F->D[s_start + j];
        D_offdiag_copy[j] = F->D_offdiag[s_start + j];
        pivot_size_copy[j] = F->pivot_size[s_start + j];
    }
}

/* Compare two factored LdltCscs entry-by-entry on L, D, D_offdiag, and
 * pivot_size. Returns 1 on full match, 0 on first mismatch.
 */
static int ldlt_csc_factor_state_matches(const LdltCsc *A, const LdltCsc *B, double tol) {
    if (A->n != B->n) {
        TF_FAIL_("n mismatch: A=%d B=%d", (int)A->n, (int)B->n);
        return 0;
    }
    idx_t n = A->n;
    if (A->L->col_ptr[n] != B->L->col_ptr[n]) {
        TF_FAIL_("L nnz mismatch: A=%d B=%d", (int)A->L->col_ptr[n], (int)B->L->col_ptr[n]);
        return 0;
    }
    for (idx_t j = 0; j < n; j++) {
        if (A->L->col_ptr[j] != B->L->col_ptr[j]) {
            TF_FAIL_("col_ptr[%d] mismatch: A=%d B=%d", (int)j, (int)A->L->col_ptr[j],
                     (int)B->L->col_ptr[j]);
            return 0;
        }
    }
    idx_t total = A->L->col_ptr[n];
    for (idx_t p = 0; p < total; p++) {
        if (A->L->row_idx[p] != B->L->row_idx[p]) {
            TF_FAIL_("row_idx[%d] mismatch: A=%d B=%d", (int)p, (int)A->L->row_idx[p],
                     (int)B->L->row_idx[p]);
            return 0;
        }
        if (fabs(A->L->values[p] - B->L->values[p]) > tol) {
            TF_FAIL_("L.values[%d] mismatch: A=%.15g B=%.15g diff=%.3e", (int)p, A->L->values[p],
                     B->L->values[p], fabs(A->L->values[p] - B->L->values[p]));
            return 0;
        }
    }
    for (idx_t k = 0; k < n; k++) {
        if (fabs(A->D[k] - B->D[k]) > tol) {
            TF_FAIL_("D[%d] mismatch: A=%.15g B=%.15g", (int)k, A->D[k], B->D[k]);
            return 0;
        }
        if (fabs(A->D_offdiag[k] - B->D_offdiag[k]) > tol) {
            TF_FAIL_("D_offdiag[%d] mismatch: A=%.15g B=%.15g", (int)k, A->D_offdiag[k],
                     B->D_offdiag[k]);
            return 0;
        }
        if (A->pivot_size[k] != B->pivot_size[k]) {
            TF_FAIL_("pivot_size[%d] mismatch: A=%d B=%d", (int)k, (int)A->pivot_size[k],
                     (int)B->pivot_size[k]);
            return 0;
        }
    }
    return 1;
}

/* Build a moderately-conditioned dense SPD matrix of size n with entries in
 * [-1, 1] off-diagonal and a strong diagonal so BK picks 1x1 pivots
 * throughout.
 */
static SparseMatrix *build_dense_spd(idx_t n, unsigned int seed) {
    srand(seed);
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            if (i == j) {
                sparse_insert(A, i, i, (double)(2 * n));
            } else if (j < i) {
                double v = ((double)rand() / (double)RAND_MAX) * 0.5 - 0.25;
                sparse_insert(A, i, j, v);
                sparse_insert(A, j, i, v);
            }
        }
    }
    return A;
}

#endif
