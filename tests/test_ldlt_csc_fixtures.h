#ifndef TEST_LDLT_CSC_FIXTURES_H
#define TEST_LDLT_CSC_FIXTURES_H

#include "sparse_analysis.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"

/* Family-local fixture seam for LDLT CSC KKT and analysis-backed supernodal
 * tests. Keep these helpers included by test_ldlt_csc.c so the existing test
 * binary remains the only proof owner.
 */

/* Small 5x5 KKT saddle-point fixture. SPD 3x3 top block + 2x2 zero bottom
 * block + 2-row off-diagonal coupling.
 */
static SparseMatrix *build_kkt_5x5(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < 3; i++)
        sparse_insert(A, i, i, 4.0);
    sparse_insert(A, 0, 3, 1.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 1, 4, 1.0);
    sparse_insert(A, 4, 1, 1.0);
    return A;
}

/* Larger 10x10 KKT saddle-point: tridiagonal SPD top block (rows 0..5, diag 6,
 * off-diag -1) + 4x4 zero bottom block (rows 6..9) + 4x6 full-rank coupling.
 */
static SparseMatrix *build_kkt_10x10(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < 6; i++) {
        sparse_insert(A, i, i, 6.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    for (idx_t j = 0; j < 4; j++) {
        sparse_insert(A, 6 + j, j, 1.0);
        sparse_insert(A, j, 6 + j, 1.0);
    }
    return A;
}

/* Scaled 10x10 KKT fixture for the Sprint 102 external dense-reference lane. */
static SparseMatrix *build_kkt_scaled_10x10(void) {
    const idx_t n = 10;
    const double diag[6] = {8.0, 10.0, 12.0, 14.0, 16.0, 18.0};
    const double offdiag[5] = {-1.0, -1.25, -1.5, -1.75, -2.0};
    const idx_t coupling_rows[8] = {6, 6, 7, 7, 8, 8, 9, 9};
    const idx_t coupling_cols[8] = {0, 4, 1, 5, 2, 4, 3, 5};
    const double coupling_vals[8] = {1.0, 0.125, -2.0, 0.25, 0.5, -0.375, 3.0, 0.5};

    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < 6; i++) {
        sparse_insert(A, i, i, diag[i]);
        if (i > 0) {
            sparse_insert(A, i, i - 1, offdiag[i - 1]);
            sparse_insert(A, i - 1, i, offdiag[i - 1]);
        }
    }
    for (idx_t k = 0; k < 8; k++) {
        sparse_insert(A, coupling_rows[k], coupling_cols[k], coupling_vals[k]);
        sparse_insert(A, coupling_cols[k], coupling_rows[k], coupling_vals[k]);
    }
    return A;
}

/* Run the Option D two-pass factor on `A` and populate *F1_out (scalar
 * reference) and *F2_out (batched via the analysis-backed shim). Caller owns
 * F1, F2, and A_perm on success.
 */
static int s20_two_pass_indefinite_factor(const SparseMatrix *A, LdltCsc **F1_out, LdltCsc **F2_out,
                                          SparseMatrix **A_perm_out) {
    *F1_out = NULL;
    *F2_out = NULL;
    *A_perm_out = NULL;
    idx_t n = sparse_rows(A);

    LdltCsc *F1 = NULL;
    if (ldlt_csc_from_sparse(A, NULL, 2.0, &F1) != SPARSE_OK)
        return 0;
    if (ldlt_csc_eliminate_native(F1) != SPARSE_OK) {
        ldlt_csc_free(F1);
        return 0;
    }

    SparseMatrix *A_perm = sparse_create(n, n);
    if (!A_perm) {
        ldlt_csc_free(F1);
        return 0;
    }
    for (idx_t i_new = 0; i_new < n; i_new++) {
        for (idx_t j_new = 0; j_new < n; j_new++) {
            idx_t i_old = F1->perm[i_new];
            idx_t j_old = F1->perm[j_new];
            double v = sparse_get(A, i_old, j_old);
            if (v != 0.0)
                sparse_insert(A_perm, i_new, j_new, v);
        }
    }

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_LDLT,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_t an = {0};
    if (sparse_analyze(A_perm, &opts, &an) != SPARSE_OK) {
        ldlt_csc_free(F1);
        sparse_free(A_perm);
        return 0;
    }

    LdltCsc *F2 = NULL;
    if (ldlt_csc_from_sparse_with_analysis(A_perm, &an, &F2) != SPARSE_OK) {
        ldlt_csc_free(F1);
        sparse_analysis_free(&an);
        sparse_free(A_perm);
        return 0;
    }

    for (idx_t k = 0; k < n; k++)
        F2->pivot_size[k] = F1->pivot_size[k];

    if (ldlt_csc_eliminate_supernodal(F2, /*min_size=*/2) != SPARSE_OK) {
        ldlt_csc_free(F1);
        ldlt_csc_free(F2);
        sparse_analysis_free(&an);
        sparse_free(A_perm);
        return 0;
    }

    sparse_analysis_free(&an);
    *F1_out = F1;
    *F2_out = F2;
    *A_perm_out = A_perm;
    return 1;
}

#endif
