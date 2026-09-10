#ifndef TEST_SVD_HELPERS_H
#define TEST_SVD_HELPERS_H

#include "sparse_matrix.h"
#include "sparse_svd.h"
#include "test_framework.h"

#include <math.h>
#include <stdlib.h>

static inline int tf_svd_insert_or_free(SparseMatrix **A, idx_t row, idx_t col, double value) {
    sparse_err_t err = sparse_insert(*A, row, col, value);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(*A);
        *A = NULL;
        return 0;
    }
    return 1;
}

static inline SparseMatrix *tf_svd_make_diag_matrix(idx_t rows, idx_t cols, const double *diag,
                                                    idx_t diag_len) {
    SparseMatrix *A = sparse_create(rows, cols);
    if (!A)
        return NULL;

    idx_t limit = rows < cols ? rows : cols;
    if (diag_len < limit)
        limit = diag_len;
    for (idx_t i = 0; i < limit; i++) {
        if (diag[i] != 0.0 && !tf_svd_insert_or_free(&A, i, i, diag[i]))
            return NULL;
    }
    return A;
}

static inline SparseMatrix *tf_svd_make_rank1_row_progression(idx_t rows, idx_t cols) {
    SparseMatrix *A = sparse_create(rows, cols);
    if (!A)
        return NULL;

    for (idx_t i = 0; i < rows; i++)
        for (idx_t j = 0; j < cols; j++)
            if (!tf_svd_insert_or_free(&A, i, j, (double)(i + 1)))
                return NULL;
    return A;
}

static inline SparseMatrix *tf_svd_make_rank_deficient_colpair_5x4(void) {
    SparseMatrix *A = sparse_create(5, 4);
    if (!A)
        return NULL;

    for (idx_t i = 0; i < 5; i++) {
        if (!tf_svd_insert_or_free(&A, i, 0, (double)(i + 1)) ||
            !tf_svd_insert_or_free(&A, i, 1, (double)(i + 1)) ||
            !tf_svd_insert_or_free(&A, i, 2, (double)(i * 2 + 1)) ||
            !tf_svd_insert_or_free(&A, i, 3, (double)(i * 2 + 1)))
            return NULL;
    }
    return A;
}

static inline SparseMatrix *tf_svd_make_dependent_row_4x3(void) {
    SparseMatrix *A = sparse_create(4, 3);
    if (!A)
        return NULL;

    /* Rows are combinations of [1,0,1] and [0,1,2], so rank is exactly 2. */
    if (!tf_svd_insert_or_free(&A, 0, 0, 1.0) || !tf_svd_insert_or_free(&A, 0, 2, 1.0) ||
        !tf_svd_insert_or_free(&A, 1, 1, 1.0) || !tf_svd_insert_or_free(&A, 1, 2, 2.0) ||
        !tf_svd_insert_or_free(&A, 2, 0, 1.0) || !tf_svd_insert_or_free(&A, 2, 1, 1.0) ||
        !tf_svd_insert_or_free(&A, 2, 2, 3.0) || !tf_svd_insert_or_free(&A, 3, 0, 2.0) ||
        !tf_svd_insert_or_free(&A, 3, 1, -1.0))
        return NULL;
    return A;
}

static inline SparseMatrix *tf_svd_make_full_uv_fixture_16x8(void) {
    const idx_t m = 16, n_cols = 8;
    SparseMatrix *A = sparse_create(m, n_cols);
    if (!A)
        return NULL;

    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < n_cols; j++) {
            double v = (double)(i + 1) * 0.7 - (double)(j + 1) * 1.3 +
                       (((i + j) % 3) ? 1.0 : -1.0) * (double)((i * 11 + j * 7) % 13);
            if (!tf_svd_insert_or_free(&A, i, j, v))
                return NULL;
        }
    }
    return A;
}

static inline double tf_dense_column_orthogonality_error(const double *Q, idx_t rows, idx_t cols) {
    double maxerr = 0.0;
    for (idx_t i = 0; i < cols; i++) {
        for (idx_t j = 0; j < cols; j++) {
            double dot = 0.0;
            for (idx_t p = 0; p < rows; p++)
                dot += Q[(size_t)i * (size_t)rows + (size_t)p] *
                       Q[(size_t)j * (size_t)rows + (size_t)p];
            double expected = (i == j) ? 1.0 : 0.0;
            double e = fabs(dot - expected);
            if (e > maxerr)
                maxerr = e;
        }
    }
    return maxerr;
}

static inline double tf_svd_vt_row_orthogonality_error(const double *Vt, idx_t rows, idx_t cols,
                                                       idx_t ld) {
    double frob_err_sq = 0.0;
    for (idx_t i = 0; i < rows; i++) {
        for (idx_t j = 0; j < rows; j++) {
            double dot = 0.0;
            for (idx_t c = 0; c < cols; c++)
                dot +=
                    Vt[(size_t)c * (size_t)ld + (size_t)i] * Vt[(size_t)c * (size_t)ld + (size_t)j];
            double target = (i == j) ? 1.0 : 0.0;
            double d = dot - target;
            frob_err_sq += d * d;
        }
    }
    return sqrt(frob_err_sq);
}

static inline double tf_svd_reconstruction_max_error(const SparseMatrix *A, const sparse_svd_t *svd,
                                                     idx_t u_ld, idx_t vt_ld) {
    double max_err = 0.0;
    for (idx_t i = 0; i < sparse_rows(A); i++) {
        for (idx_t j = 0; j < sparse_cols(A); j++) {
            double recon = 0.0;
            for (idx_t s = 0; s < svd->k; s++)
                recon += svd->sigma[s] * svd->U[(size_t)s * (size_t)u_ld + (size_t)i] *
                         svd->Vt[(size_t)j * (size_t)vt_ld + (size_t)s];
            double e = fabs(sparse_get(A, i, j) - recon);
            if (e > max_err)
                max_err = e;
        }
    }
    return max_err;
}

static inline double tf_svd_reconstruction_rel_frobenius(const SparseMatrix *A,
                                                         const sparse_svd_t *svd, idx_t u_ld,
                                                         idx_t vt_ld) {
    double frob_resid_sq = 0.0;
    double frob_a_sq = 0.0;
    for (idx_t i = 0; i < sparse_rows(A); i++) {
        for (idx_t j = 0; j < sparse_cols(A); j++) {
            double recon = 0.0;
            for (idx_t s = 0; s < svd->k; s++)
                recon += svd->sigma[s] * svd->U[(size_t)s * (size_t)u_ld + (size_t)i] *
                         svd->Vt[(size_t)j * (size_t)vt_ld + (size_t)s];
            double a_ij = sparse_get(A, i, j);
            double d = a_ij - recon;
            frob_resid_sq += d * d;
            frob_a_sq += a_ij * a_ij;
        }
    }
    return sqrt(frob_a_sq) > 0.0 ? sqrt(frob_resid_sq) / sqrt(frob_a_sq) : sqrt(frob_resid_sq);
}

static inline double tf_svd_pinv_first_moore_penrose_error(const SparseMatrix *A,
                                                           const double *pinv, idx_t m,
                                                           idx_t n_cols) {
    double *B = calloc((size_t)m * (size_t)m, sizeof(double));
    if (!B)
        return HUGE_VAL;

    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < m; j++) {
            double sum = 0.0;
            for (idx_t p = 0; p < n_cols; p++)
                sum += sparse_get(A, i, p) * pinv[(size_t)j * (size_t)n_cols + (size_t)p];
            B[(size_t)j * (size_t)m + (size_t)i] = sum;
        }
    }

    double max_err = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < n_cols; j++) {
            double sum = 0.0;
            for (idx_t p = 0; p < m; p++)
                sum += B[(size_t)p * (size_t)m + (size_t)i] * sparse_get(A, p, j);
            double e = fabs(sum - sparse_get(A, i, j));
            if (e > max_err)
                max_err = e;
        }
    }

    free(B);
    return max_err;
}

static inline double tf_svd_dense_lowrank_frobenius_error(const SparseMatrix *A,
                                                          const double *dense, idx_t rows,
                                                          idx_t cols, idx_t dense_ld) {
    double frob_sq = 0.0;
    for (idx_t i = 0; i < rows; i++) {
        for (idx_t j = 0; j < cols; j++) {
            double diff = sparse_get(A, i, j) - dense[(size_t)j * (size_t)dense_ld + (size_t)i];
            frob_sq += diff * diff;
        }
    }
    return sqrt(frob_sq);
}

static inline double tf_svd_sparse_dense_frobenius_diff(const SparseMatrix *sp, const double *dense,
                                                        idx_t rows, idx_t cols, idx_t dense_ld) {
    double frob_sq = 0.0;
    for (idx_t i = 0; i < rows; i++) {
        for (idx_t j = 0; j < cols; j++) {
            double d_val = dense[(size_t)j * (size_t)dense_ld + (size_t)i];
            double diff = d_val - sparse_get(sp, i, j);
            frob_sq += diff * diff;
        }
    }
    return sqrt(frob_sq);
}

static inline double tf_svd_sparse_dense_max_abs_diff(const SparseMatrix *sp, const double *dense,
                                                      idx_t rows, idx_t cols, idx_t dense_ld) {
    double max_diff = 0.0;
    for (idx_t i = 0; i < rows; i++) {
        for (idx_t j = 0; j < cols; j++) {
            double d_val = dense[(size_t)j * (size_t)dense_ld + (size_t)i];
            double diff = fabs(d_val - sparse_get(sp, i, j));
            if (diff > max_diff)
                max_diff = diff;
        }
    }
    return max_diff;
}

static inline double tf_svd_sparse_sparse_rel_frobenius_diff(const SparseMatrix *baseline,
                                                             const SparseMatrix *candidate,
                                                             idx_t rows, idx_t cols) {
    double frob_diff_sq = 0.0;
    double frob_base_sq = 0.0;
    for (idx_t i = 0; i < rows; i++) {
        for (idx_t j = 0; j < cols; j++) {
            double base_val = sparse_get(baseline, i, j);
            double candidate_val = sparse_get(candidate, i, j);
            double diff = base_val - candidate_val;
            frob_diff_sq += diff * diff;
            frob_base_sq += base_val * base_val;
        }
    }
    return sqrt(frob_base_sq) > 0.0 ? sqrt(frob_diff_sq) / sqrt(frob_base_sq) : sqrt(frob_diff_sq);
}

#endif /* TEST_SVD_HELPERS_H */
