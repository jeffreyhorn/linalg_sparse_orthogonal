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

    /* Overflow check: rows*cols and rows*cols*sizeof(double) */
    size_t n = (size_t)rows * (size_t)cols;
    if (cols > 0 && n / (size_t)cols != (size_t)rows) {
        free(M);
        return NULL;
    }
    if (n > SIZE_MAX / sizeof(double)) {
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

sparse_err_t dense_gemm(const dense_matrix_t *A, const dense_matrix_t *B, dense_matrix_t *C) {
    if (!A || !B || !C)
        return SPARSE_ERR_NULL;
    if (A->cols != B->rows)
        return SPARSE_ERR_SHAPE;
    if (C->rows != A->rows || C->cols != B->cols)
        return SPARSE_ERR_SHAPE;

    idx_t m = A->rows;
    idx_t k = A->cols;
    idx_t n = B->cols;

    /* Zero-sized matrices: C = 0 (any zero dimension means empty product) */
    if (m == 0 || k == 0 || n == 0) {
        if (C->data && m > 0 && n > 0) {
            size_t mn = (size_t)m * (size_t)n;
            if (mn / (size_t)n != (size_t)m)
                return SPARSE_ERR_ALLOC;
            if (mn > SIZE_MAX / sizeof(double))
                return SPARSE_ERR_ALLOC;
            memset(C->data, 0, mn * sizeof(double));
        }
        return SPARSE_OK;
    }

    if (!A->data || !B->data || !C->data)
        return SPARSE_ERR_NULL;

    /* Overflow-safe byte count for C */
    size_t mn = (size_t)m * (size_t)n;
    if (n > 0 && mn / (size_t)n != (size_t)m)
        return SPARSE_ERR_ALLOC;
    if (mn > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

    /* Zero C */
    memset(C->data, 0, mn * sizeof(double));

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

    if (m == 0)
        return SPARSE_OK;

    if (n == 0) {
        /* A is m×0: y should be the zero vector */
        memset(y, 0, (size_t)m * sizeof(double));
        return SPARSE_OK;
    }

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
        x[k] = c * xk - s * yk;
        y[k] = s * xk + c * yk;
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

/* ═══════════════════════════════════════════════════════════════════════
 * Symmetric tridiagonal eigenvalue solver (implicit QR with Wilkinson shift)
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * One implicit QR step on the unreduced tridiagonal block diag[lo..hi],
 * subdiag[lo..hi-1] using a Wilkinson shift.
 */
static void tridiag_qr_step(double *diag, double *subdiag, idx_t lo, idx_t hi) {
    /* Wilkinson shift: eigenvalue of trailing 2×2 closer to diag[hi] */
    double l1, l2;
    eigen2x2(diag[hi - 1], subdiag[hi - 1], diag[hi], &l1, &l2);
    double shift = (fabs(l1 - diag[hi]) < fabs(l2 - diag[hi])) ? l1 : l2;

    /* Initial bulge: Givens to zero subdiag[lo] in (T - shift*I) */
    double x = diag[lo] - shift;
    double z = subdiag[lo];
    double c, s;

    for (idx_t k = lo; k < hi; k++) {
        givens_compute(x, z, &c, &s);

        /* Apply Givens rotation G(k, k+1) from both sides to T.
         * This is the implicit symmetric tridiagonal QR step.
         * Only affects rows/cols k, k+1 (and neighbours). */

        if (k > lo) {
            subdiag[k - 1] = c * subdiag[k - 1] + s * z;
            /* The bulge at (k-1, k+1) is zeroed by this rotation */
        }

        double dk = diag[k];
        double dk1 = diag[k + 1];
        double ek = subdiag[k];

        /* Update 2×2 block [dk, ek; ek, dk1] under similarity transform */
        diag[k] = c * c * dk + 2.0 * c * s * ek + s * s * dk1;
        diag[k + 1] = s * s * dk - 2.0 * c * s * ek + c * c * dk1;
        subdiag[k] = c * s * (dk1 - dk) + (c * c - s * s) * ek;

        /* Prepare bulge for next iteration */
        if (k + 1 < hi) {
            z = s * subdiag[k + 1];
            subdiag[k + 1] = c * subdiag[k + 1];
            x = subdiag[k];
        }
    }
}

sparse_err_t tridiag_qr_eigenvalues(double *diag, double *subdiag, idx_t n, idx_t max_iter) {
    if (n <= 0)
        return SPARSE_OK;
    if (n == 1)
        return SPARSE_OK;
    if (!diag || !subdiag)
        return SPARSE_ERR_NULL;

    if (max_iter <= 0)
        max_iter = 30 * n;

    /* Deflation tolerance */
    double tol = 1e-14;

    idx_t total_iter = 0;
    idx_t hi = n - 1; /* top of active block */

    while (hi > 0 && total_iter < max_iter) {
        /* Check for deflation at the bottom */
        double off = fabs(subdiag[hi - 1]);
        double diag_sum = fabs(diag[hi - 1]) + fabs(diag[hi]);
        if (off <= tol * diag_sum || off < 1e-30) {
            subdiag[hi - 1] = 0.0;
            hi--;
            continue;
        }

        /* Find the start of the unreduced block */
        idx_t lo = hi - 1;
        while (lo > 0) {
            double off_lo = fabs(subdiag[lo - 1]);
            double ds_lo = fabs(diag[lo - 1]) + fabs(diag[lo]);
            if (off_lo <= tol * ds_lo || off_lo < 1e-30) {
                subdiag[lo - 1] = 0.0;
                break;
            }
            lo--;
        }

        /* One QR step on block [lo..hi] */
        tridiag_qr_step(diag, subdiag, lo, hi);
        total_iter++;
    }

    if (hi > 0)
        return SPARSE_ERR_NOT_CONVERGED;

    /* Sort eigenvalues in ascending order (insertion sort — n is small) */
    for (idx_t i = 1; i < n; i++) {
        double key = diag[i];
        idx_t j = i - 1;
        while (j >= 0 && diag[j] > key) {
            diag[j + 1] = diag[j];
            j--;
        }
        diag[j + 1] = key;
    }

    return SPARSE_OK;
}
