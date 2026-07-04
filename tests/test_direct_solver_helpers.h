#ifndef TEST_DIRECT_SOLVER_HELPERS_H
#define TEST_DIRECT_SOLVER_HELPERS_H

#include "sparse_lu_csr.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <math.h>
#include <stdlib.h>

static inline void tf_assert_sparse_matrices_equal(const SparseMatrix *A, const SparseMatrix *B,
                                                   double tol) {
    ASSERT_EQ(A->rows, B->rows);
    ASSERT_EQ(A->cols, B->cols);
    for (idx_t i = 0; i < A->rows; i++) {
        for (idx_t j = 0; j < A->cols; j++) {
            double a = sparse_get(A, i, j);
            double b = sparse_get(B, i, j);
            if (fabs(a - b) > tol) {
                TF_FAIL_("Entry (%d,%d): %.15g vs %.15g, diff=%.3e > tol=%.3e", (int)i, (int)j, a,
                         b, fabs(a - b), tol);
            }
            tf_asserts++;
        }
    }
}

static inline void tf_verify_lu_csr_factorization(const SparseMatrix *A_orig, const LuCsr *lu,
                                                  const idx_t *piv, double tol_check) {
    idx_t n = lu->n;

    double *L = calloc((size_t)n * (size_t)n, sizeof(double));
    double *U = calloc((size_t)n * (size_t)n, sizeof(double));
    ASSERT_NOT_NULL(L);
    ASSERT_NOT_NULL(U);

    for (idx_t i = 0; i < n; i++)
        L[i * n + i] = 1.0;

    for (idx_t i = 0; i < n; i++) {
        for (idx_t p = lu->row_ptr[i]; p < lu->row_ptr[i + 1]; p++) {
            idx_t j = lu->col_idx[p];
            double v = lu->values[p];
            if (j < i)
                L[i * n + j] = v;
            else
                U[i * n + j] = v;
        }
    }

    double *LU = calloc((size_t)n * (size_t)n, sizeof(double));
    ASSERT_NOT_NULL(LU);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            for (idx_t k = 0; k < n; k++)
                LU[i * n + j] += L[i * n + k] * U[k * n + j];

    for (idx_t i = 0; i < n; i++) {
        idx_t orig_row = piv[i];
        for (idx_t j = 0; j < n; j++) {
            double a_val = sparse_get(A_orig, orig_row, j);
            double lu_val = LU[i * n + j];
            if (fabs(a_val - lu_val) > tol_check) {
                TF_FAIL_("P*A vs L*U mismatch at (%d,%d): P*A=%.15g, L*U=%.15g, diff=%.3e", (int)i,
                         (int)j, a_val, lu_val, fabs(a_val - lu_val));
            }
            tf_asserts++;
        }
    }

    free(L);
    free(U);
    free(LU);
}

static inline double tf_sparse_residual_norminf(const SparseMatrix *A, const double *x,
                                                const double *b, idx_t n) {
    double *r = malloc((size_t)n * sizeof(double));
    if (!r)
        return HUGE_VAL;
    sparse_matvec(A, x, r);
    double mx = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(r[i] - b[i]);
        if (d > mx)
            mx = d;
    }
    free(r);
    return mx;
}

#endif
