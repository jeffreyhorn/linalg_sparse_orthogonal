#ifndef TEST_QR_HELPERS_H
#define TEST_QR_HELPERS_H

#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_vector.h"
#include "test_framework.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static inline int tf_qr_idx_count_bytes(idx_t count, size_t elem_size, size_t *bytes) {
    if (!bytes || count < 0 || elem_size == 0)
        return 0;
    if ((uintmax_t)count > (uintmax_t)SIZE_MAX)
        return 0;

    size_t count_size = (size_t)count;
    if (count_size > SIZE_MAX / elem_size)
        return 0;

    *bytes = count_size * elem_size;
    return 1;
}

static inline int tf_qr_make_exact_rhs(const SparseMatrix *A, idx_t x_len, idx_t b_len,
                                       double **x_exact_out, double **b_out) {
    if (!x_exact_out || !b_out)
        return 0;
    *x_exact_out = NULL;
    *b_out = NULL;
    if (!A) {
        ASSERT_NOT_NULL(A);
        return 0;
    }
    if (x_len != sparse_cols(A)) {
        ASSERT_EQ(x_len, sparse_cols(A));
        return 0;
    }
    if (b_len != sparse_rows(A)) {
        ASSERT_EQ(b_len, sparse_rows(A));
        return 0;
    }

    size_t x_bytes = 0;
    size_t b_bytes = 0;
    if (!tf_qr_idx_count_bytes(x_len, sizeof(double), &x_bytes) ||
        !tf_qr_idx_count_bytes(b_len, sizeof(double), &b_bytes)) {
        ASSERT_TRUE(0);
        return 0;
    }

    double *x_exact = malloc(x_bytes);
    double *b = malloc(b_bytes);
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        return 0;
    }

    for (idx_t i = 0; i < x_len; i++)
        x_exact[i] = (double)(i + 1);
    sparse_err_t mv_err = sparse_matvec(A, x_exact, b);
    ASSERT_ERR(mv_err, SPARSE_OK);
    if (mv_err != SPARSE_OK) {
        free(x_exact);
        free(b);
        return 0;
    }

    *x_exact_out = x_exact;
    *b_out = b;
    return 1;
}

static inline int tf_qr_insert_or_free(SparseMatrix **A, idx_t row, idx_t col, double value) {
    sparse_err_t err = sparse_insert(*A, row, col, value);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(*A);
        *A = NULL;
        return 0;
    }
    return 1;
}

static inline SparseMatrix *tf_qr_make_small_banded_4x3(int include_tail) {
    SparseMatrix *A = sparse_create(4, 3);
    if (!A)
        return NULL;
    if (!tf_qr_insert_or_free(&A, 0, 0, 2.0) || !tf_qr_insert_or_free(&A, 0, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 1, 0, 1.0) || !tf_qr_insert_or_free(&A, 1, 1, 3.0) ||
        !tf_qr_insert_or_free(&A, 1, 2, 1.0) || !tf_qr_insert_or_free(&A, 2, 2, 4.0) ||
        !tf_qr_insert_or_free(&A, 3, 0, 1.0))
        return NULL;
    if (include_tail && !tf_qr_insert_or_free(&A, 3, 2, 2.0))
        return NULL;
    return A;
}

static inline SparseMatrix *tf_qr_make_duplicate_column_4x3(double duplicate_scale) {
    SparseMatrix *A = sparse_create(4, 3);
    if (!A)
        return NULL;
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 1, 0, 2.0) ||
        !tf_qr_insert_or_free(&A, 2, 0, 3.0) || !tf_qr_insert_or_free(&A, 3, 0, 4.0) ||
        !tf_qr_insert_or_free(&A, 0, 1, 5.0) || !tf_qr_insert_or_free(&A, 1, 1, 6.0) ||
        !tf_qr_insert_or_free(&A, 2, 1, 7.0) || !tf_qr_insert_or_free(&A, 3, 1, 8.0) ||
        !tf_qr_insert_or_free(&A, 0, 2, duplicate_scale) ||
        !tf_qr_insert_or_free(&A, 1, 2, duplicate_scale * 2.0) ||
        !tf_qr_insert_or_free(&A, 2, 2, duplicate_scale * 3.0) ||
        !tf_qr_insert_or_free(&A, 3, 2, duplicate_scale * 4.0))
        return NULL;
    return A;
}

static inline SparseMatrix *tf_qr_make_near_duplicate_4x3(double perturbation) {
    SparseMatrix *A = sparse_create(4, 3);
    if (!A)
        return NULL;
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 1, 0, 2.0) ||
        !tf_qr_insert_or_free(&A, 2, 0, 3.0) || !tf_qr_insert_or_free(&A, 3, 0, 4.0) ||
        !tf_qr_insert_or_free(&A, 0, 1, 5.0) || !tf_qr_insert_or_free(&A, 1, 1, 6.0) ||
        !tf_qr_insert_or_free(&A, 2, 1, 7.0) || !tf_qr_insert_or_free(&A, 3, 1, 8.0) ||
        !tf_qr_insert_or_free(&A, 0, 2, 1.0 + perturbation) ||
        !tf_qr_insert_or_free(&A, 1, 2, 2.0 + perturbation) ||
        !tf_qr_insert_or_free(&A, 2, 2, 3.0 + perturbation) ||
        !tf_qr_insert_or_free(&A, 3, 2, 4.0 + perturbation))
        return NULL;
    return A;
}

static inline SparseMatrix *tf_qr_make_dependent_row_4x3(void) {
    SparseMatrix *A = sparse_create(4, 3);
    if (!A)
        return NULL;

    /* Rows are combinations of [1,0,1] and [0,1,2], so rank is exactly 2. */
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 0, 2, 1.0) ||
        !tf_qr_insert_or_free(&A, 1, 1, 1.0) || !tf_qr_insert_or_free(&A, 1, 2, 2.0) ||
        !tf_qr_insert_or_free(&A, 2, 0, 1.0) || !tf_qr_insert_or_free(&A, 2, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 2, 2, 3.0) || !tf_qr_insert_or_free(&A, 3, 0, 2.0) ||
        !tf_qr_insert_or_free(&A, 3, 1, -1.0))
        return NULL;
    return A;
}

static inline SparseMatrix *tf_qr_make_diag_matrix(idx_t rows, idx_t cols, const double *diag,
                                                   idx_t diag_len) {
    SparseMatrix *A = sparse_create(rows, cols);
    if (!A)
        return NULL;

    idx_t limit = rows < cols ? rows : cols;
    if (diag_len < limit)
        limit = diag_len;
    for (idx_t i = 0; i < limit; i++) {
        if (diag[i] != 0.0 && !tf_qr_insert_or_free(&A, i, i, diag[i]))
            return NULL;
    }
    return A;
}

static inline SparseMatrix *tf_qr_make_tall_diagonal_dominant(idx_t m, idx_t n_cols,
                                                              double diag_value,
                                                              double offdiag_value,
                                                              int include_lower_neighbor) {
    SparseMatrix *A = sparse_create(m, n_cols);
    if (!A)
        return NULL;

    idx_t band_rows = (m < n_cols) ? m : n_cols;
    for (idx_t i = 0; i < band_rows; i++) {
        if (!tf_qr_insert_or_free(&A, i, i, diag_value))
            return NULL;
        if (i + 1 < n_cols && !tf_qr_insert_or_free(&A, i, i + 1, offdiag_value))
            return NULL;
        if (include_lower_neighbor && i > 0 && !tf_qr_insert_or_free(&A, i, i - 1, offdiag_value))
            return NULL;
    }

    return A;
}

static inline double tf_qr_reconstruction_max_error(const SparseMatrix *A, const sparse_qr_t *qr) {
    idx_t m = qr->m;
    idx_t n_cols = qr->n;
    size_t m_size = 0;
    if (m < 0 || (uintmax_t)m > (uintmax_t)SIZE_MAX) {
        ASSERT_TRUE(0);
        return HUGE_VAL;
    }
    m_size = (size_t)m;
    if (m_size != 0 && m_size > SIZE_MAX / m_size) {
        ASSERT_TRUE(0);
        return HUGE_VAL;
    }

    size_t q_count = m_size * m_size;
    size_t q_bytes = 0;
    if (q_count > SIZE_MAX / sizeof(double)) {
        ASSERT_TRUE(0);
        return HUGE_VAL;
    }
    q_bytes = q_count * sizeof(double);

    double *Q = malloc(q_bytes);
    if (!Q)
        return HUGE_VAL;
    sparse_err_t q_err = sparse_qr_form_q(qr, Q);
    ASSERT_ERR(q_err, SPARSE_OK);
    if (q_err != SPARSE_OK) {
        free(Q);
        return HUGE_VAL;
    }

    idx_t rrows = sparse_rows(qr->R);
    double maxerr = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t jp = 0; jp < n_cols; jp++) {
            double qr_val = 0.0;
            for (idx_t kk = 0; kk < rrows; kk++) {
                double q_ik = Q[(size_t)kk * (size_t)m + (size_t)i];
                double r_kj = sparse_get_phys(qr->R, kk, jp);
                qr_val += q_ik * r_kj;
            }
            idx_t orig_col = qr->col_perm[jp];
            double a_val = sparse_get_phys(A, i, orig_col);
            double diff = fabs(qr_val - a_val);
            if (diff > maxerr)
                maxerr = diff;
        }
    }
    free(Q);
    return maxerr;
}

static inline double tf_qr_relative_residual_l2(const SparseMatrix *A, const double *b,
                                                const double *x, idx_t m) {
    size_t r_bytes = 0;
    if (!tf_qr_idx_count_bytes(m, sizeof(double), &r_bytes)) {
        ASSERT_TRUE(0);
        return HUGE_VAL;
    }

    double *r = malloc(r_bytes);
    if (!r)
        return HUGE_VAL;
    sparse_err_t mv_err = sparse_matvec(A, x, r);
    ASSERT_ERR(mv_err, SPARSE_OK);
    if (mv_err != SPARSE_OK) {
        free(r);
        return HUGE_VAL;
    }
    for (idx_t i = 0; i < m; i++)
        r[i] = b[i] - r[i];
    double rnorm = vec_norm2(r, m);
    double bnorm = vec_norm2(b, m);
    free(r);
    return (bnorm > 0.0) ? rnorm / bnorm : 0.0;
}

#endif /* TEST_QR_HELPERS_H */
