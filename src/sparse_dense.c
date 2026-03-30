#include "sparse_dense.h"
#include <stdlib.h>
#include <string.h>

dense_matrix_t *dense_create(idx_t rows, idx_t cols) {
    if (rows < 0 || cols < 0)
        return NULL;

    dense_matrix_t *M = malloc(sizeof(dense_matrix_t));
    if (!M)
        return NULL;

    M->rows = rows;
    M->cols = cols;

    if (rows == 0 || cols == 0) {
        M->data = NULL;
        return M;
    }

    /* Overflow check */
    size_t n = (size_t)rows * (size_t)cols;
    if (cols > 0 && n / (size_t)cols != (size_t)rows) {
        free(M);
        return NULL;
    }

    M->data = calloc(n, sizeof(double));
    if (!M->data) {
        free(M);
        return NULL;
    }

    return M;
}

void dense_free(dense_matrix_t *M) {
    if (!M)
        return;
    free(M->data);
    free(M);
}

sparse_err_t dense_gemm(const dense_matrix_t *A, const dense_matrix_t *B,
                         dense_matrix_t *C) {
    if (!A || !B || !C)
        return SPARSE_ERR_NULL;
    if (!A->data || !B->data || !C->data)
        return SPARSE_ERR_NULL;
    if (A->cols != B->rows)
        return SPARSE_ERR_SHAPE;
    if (C->rows != A->rows || C->cols != B->cols)
        return SPARSE_ERR_SHAPE;

    idx_t m = A->rows;
    idx_t k = A->cols;
    idx_t n = B->cols;

    /* Zero C */
    memset(C->data, 0, (size_t)m * (size_t)n * sizeof(double));

    /* C(i,j) = sum_p A(i,p) * B(p,j)
     * Column-major: loop over j (output column), then p, then i for cache. */
    for (idx_t j = 0; j < n; j++) {
        for (idx_t p = 0; p < k; p++) {
            double b_pj = DENSE_AT(B, p, j);
            if (b_pj == 0.0)
                continue;
            for (idx_t i = 0; i < m; i++) {
                DENSE_AT(C, i, j) += DENSE_AT(A, i, p) * b_pj;
            }
        }
    }

    return SPARSE_OK;
}

sparse_err_t dense_gemv(const dense_matrix_t *A, const double *x, double *y) {
    if (!A || !x || !y)
        return SPARSE_ERR_NULL;

    idx_t m = A->rows;
    idx_t n = A->cols;

    if (m == 0 || n == 0)
        return SPARSE_OK;

    if (!A->data)
        return SPARSE_ERR_NULL;

    /* y = 0 */
    memset(y, 0, (size_t)m * sizeof(double));

    /* y(i) = sum_j A(i,j) * x(j)
     * Column-major: loop over j (column), then i for cache. */
    for (idx_t j = 0; j < n; j++) {
        double xj = x[j];
        if (xj == 0.0)
            continue;
        for (idx_t i = 0; i < m; i++) {
            y[i] += DENSE_AT(A, i, j) * xj;
        }
    }

    return SPARSE_OK;
}
