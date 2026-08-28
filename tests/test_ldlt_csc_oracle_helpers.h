#ifndef TEST_LDLT_CSC_ORACLE_HELPERS_H
#define TEST_LDLT_CSC_ORACLE_HELPERS_H

#include "sparse_chol_csc_internal.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "test_framework.h"

#include <math.h>

/* Family-local dense-oracle and native-wrapper comparison helpers for
 * test_ldlt_csc.c. The test bodies remain in the proof-owner file.
 */

/* Copy the stored lower triangle of F->L into a dense n*n row-major buffer. */
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

/* Apply the symmetric permutation sigma = (i <-> j) to a dense lower triangle. */
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

/* Element-wise compare two dense lower triangles. */
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

/* Build an LdltCsc from lower-triangle triples plus explicit diagonals. */
static LdltCsc *build_ldlt_from_triples(idx_t n, const double *diag, const idx_t *rows,
                                        const idx_t *cols, const double *vals, idx_t nnz_offdiag) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t c = 0; c < n; c++)
        sparse_insert(A, c, c, diag[c]);
    for (idx_t k = 0; k < nnz_offdiag; k++) {
        sparse_insert(A, rows[k], cols[k], vals[k]);
        sparse_insert(A, cols[k], rows[k], vals[k]);
    }
    LdltCsc *F = NULL;
    if (ldlt_csc_from_sparse(A, NULL, 2.0, &F) != SPARSE_OK) {
        sparse_free(A);
        return NULL;
    }
    sparse_free(A);
    return F;
}

static int ldlt_column_nonzeros_match(const CholCsc *A, const CholCsc *B, idx_t j, double tol) {
    idx_t ap = A->col_ptr[j];
    idx_t ae = A->col_ptr[j + 1];
    idx_t bp = B->col_ptr[j];
    idx_t be = B->col_ptr[j + 1];
    while (ap < ae || bp < be) {
        while (ap < ae && fabs(A->values[ap]) < tol && A->row_idx[ap] != j)
            ap++;
        while (bp < be && fabs(B->values[bp]) < tol && B->row_idx[bp] != j)
            bp++;
        if (ap == ae && bp == be)
            return 1;
        if (ap == ae || bp == be)
            return 0;
        if (A->row_idx[ap] != B->row_idx[bp])
            return 0;
        if (fabs(A->values[ap] - B->values[bp]) > tol)
            return 0;
        ap++;
        bp++;
    }
    return 1;
}

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
    for (idx_t j = 0; j < n; j++) {
        if (!ldlt_column_nonzeros_match(A->L, B->L, j, tol))
            return 0;
    }
    return 1;
}

/* Factor with both the wrapper and native kernel, then compare results. */
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

#endif
