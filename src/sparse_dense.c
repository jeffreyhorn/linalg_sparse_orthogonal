#include "sparse_dense.h"
#include <math.h>
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

/* ═══════════════════════════════════════════════════════════════════════
 * Givens rotations
 * ═══════════════════════════════════════════════════════════════════════ */

void givens_compute(double a, double b, double *c, double *s) {
    if (b == 0.0) {
        *c = 1.0;
        *s = 0.0;
    } else if (a == 0.0) {
        *c = 0.0;
        *s = (b > 0.0) ? 1.0 : -1.0;
    } else {
        double r = hypot(a, b);
        *c = a / r;
        *s = b / r;
    }
}

void givens_apply_left(double c, double s, double *x, double *y, idx_t n) {
    for (idx_t k = 0; k < n; k++) {
        double xk = x[k];
        double yk = y[k];
        x[k] = c * xk + s * yk;
        y[k] = -s * xk + c * yk;
    }
}

void givens_apply_right(double c, double s, double *x, double *y, idx_t n) {
    for (idx_t k = 0; k < n; k++) {
        double xk = x[k];
        double yk = y[k];
        x[k] = c * xk + s * yk;
        y[k] = -s * xk + c * yk;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * 2×2 symmetric eigenvalue solver
 * ═══════════════════════════════════════════════════════════════════════ */

void eigen2x2(double a, double b, double d, double *lambda1, double *lambda2) {
    /* Eigenvalues of [[a, b], [b, d]]:
     * lambda = (a+d)/2 ± sqrt(((a-d)/2)^2 + b^2)
     * Use the numerically stable form to avoid catastrophic cancellation. */
    double trace = a + d;
    double half_diff = (a - d) * 0.5;
    double disc = sqrt(half_diff * half_diff + b * b);

    /* Return in ascending order */
    *lambda1 = trace * 0.5 - disc;
    *lambda2 = trace * 0.5 + disc;
}
