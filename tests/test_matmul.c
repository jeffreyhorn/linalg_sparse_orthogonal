#include "sparse_alloc_internal.h"
#include "sparse_cholesky.h"
#include "sparse_csr.h"
#include "sparse_matrix.h"
#include "sparse_matrix_internal.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * SpMM tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Known 2x2: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]] */
static void test_matmul_2x2(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);

    SparseMatrix *B = sparse_create(2, 2);
    sparse_insert(B, 0, 0, 5.0);
    sparse_insert(B, 0, 1, 6.0);
    sparse_insert(B, 1, 0, 7.0);
    sparse_insert(B, 1, 1, 8.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);

    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 19.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 1), 22.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 43.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 50.0, 0.0);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

/* I * A = A */
static void test_matmul_identity_left(void) {
    idx_t n = 4;
    SparseMatrix *I = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(I, i, i, 1.0);

    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 3, 5.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 3, 7.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(I, A, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), sparse_nnz(A));

    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 3), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 2), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 3, 3), 7.0, 0.0);

    sparse_free(I);
    sparse_free(A);
    sparse_free(C);
}

/* A * I = A */
static void test_matmul_identity_right(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);

    SparseMatrix *I = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(I, i, i, 1.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, I, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), sparse_nnz(A));

    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 2), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 0), 7.0, 0.0);

    sparse_free(A);
    sparse_free(I);
    sparse_free(C);
}

/* Diagonal * A = scaled rows */
static void test_matmul_diag_left(void) {
    SparseMatrix *D = sparse_create(2, 2);
    sparse_insert(D, 0, 0, 2.0);
    sparse_insert(D, 1, 1, 3.0);

    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 4.0);
    sparse_insert(A, 1, 0, 5.0);
    sparse_insert(A, 1, 1, 6.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(D, A, &C), SPARSE_OK);

    /* Row 0 scaled by 2, row 1 scaled by 3 */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 1), 8.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 15.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 18.0, 0.0);

    sparse_free(D);
    sparse_free(A);
    sparse_free(C);
}

/* A * Diagonal = scaled columns */
static void test_matmul_diag_right(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 4.0);
    sparse_insert(A, 1, 0, 5.0);
    sparse_insert(A, 1, 1, 6.0);

    SparseMatrix *D = sparse_create(2, 2);
    sparse_insert(D, 0, 0, 2.0);
    sparse_insert(D, 1, 1, 3.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, D, &C), SPARSE_OK);

    /* Col 0 scaled by 2, col 1 scaled by 3 */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 1), 12.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 10.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 18.0, 0.0);

    sparse_free(A);
    sparse_free(D);
    sparse_free(C);
}

/* Dimension mismatch → SPARSE_ERR_SHAPE */
static void test_matmul_shape_mismatch(void) {
    SparseMatrix *A = sparse_create(2, 3);
    SparseMatrix *B = sparse_create(4, 2); /* 3 != 4 */
    SparseMatrix *C;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_SHAPE);
    sparse_free(A);
    sparse_free(B);
}

/* NULL inputs → SPARSE_ERR_NULL */
static void test_matmul_null(void) {
    SparseMatrix *A = sparse_create(2, 2);
    SparseMatrix *C;
    ASSERT_ERR(sparse_matmul(NULL, A, &C), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_matmul(A, NULL, &C), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_matmul(A, A, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* Rectangular: (3x2) * (2x4) → (3x4) */
static void test_matmul_rectangular(void) {
    SparseMatrix *A = sparse_create(3, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 0, 3.0);
    sparse_insert(A, 2, 1, 4.0);

    SparseMatrix *B = sparse_create(2, 4);
    sparse_insert(B, 0, 0, 1.0);
    sparse_insert(B, 0, 2, 2.0);
    sparse_insert(B, 1, 1, 3.0);
    sparse_insert(B, 1, 3, 4.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_rows(C), 3);
    ASSERT_EQ(sparse_cols(C), 4);

    /* C(0,:) = 1*row0(B) = [1, 0, 2, 0] */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 2), 2.0, 0.0);
    /* C(1,:) = 2*row1(B) = [0, 6, 0, 8] */
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 6.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 3), 8.0, 0.0);
    /* C(2,:) = 3*row0(B) + 4*row1(B) = [3, 12, 6, 16] */
    ASSERT_NEAR(sparse_get_phys(C, 2, 0), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 1), 12.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 2), 6.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 3), 16.0, 0.0);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

/* Cancellation: A * B where entries cancel to zero → no spurious nnz */
static void test_matmul_cancellation(void) {
    /* A = [[1, 1]], B = [[1], [-1]] → C = [[0]] */
    SparseMatrix *A = sparse_create(1, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);

    SparseMatrix *B = sparse_create(2, 1);
    sparse_insert(B, 0, 0, 1.0);
    sparse_insert(B, 1, 0, -1.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), 0); /* cancellation → no entry */

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_matmul_wide_row_sorted_after_reverse_touches(void) {
    const idx_t n = 1024;
    SparseMatrix *A = sparse_create(1, n);
    SparseMatrix *B = sparse_create(n, n);
    SparseMatrix *C = NULL;
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);

    for (idx_t j = 0; j < n; j++) {
        ASSERT_ERR(sparse_insert(A, 0, j, 1.0), SPARSE_OK);
        ASSERT_ERR(sparse_insert(B, j, n - 1 - j, 1.0), SPARSE_OK);
    }

    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_nnz(C), n);

    Node *node = C->row_headers[0];
    for (idx_t j = 0; j < n; j++) {
        ASSERT_NOT_NULL(node);
        ASSERT_EQ(node->row, 0);
        ASSERT_EQ(node->col, j);
        ASSERT_NEAR(node->value, 1.0, 0.0);
        node = node->right;
    }
    ASSERT_NULL(node);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_matmul_row_append_duplicate_cancellation_and_empty_row(void) {
    SparseMatrix *A = sparse_create(3, 3);
    SparseMatrix *B = sparse_create(3, 4);
    SparseMatrix *C = NULL;
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);

    ASSERT_ERR(sparse_insert(A, 0, 0, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(A, 0, 1, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(A, 2, 2, 2.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 0, 0, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 0, 1, 2.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 0, 3, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 1, 0, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 1, 1, 3.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 1, 3, -1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 2, 2, 2.0), SPARSE_OK);

    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_nnz(C), 3);
    ASSERT_NULL(C->row_headers[1]);

    Node *node = C->row_headers[0];
    ASSERT_NOT_NULL(node);
    ASSERT_EQ(node->col, 0);
    ASSERT_NEAR(node->value, 2.0, 0.0);
    node = node->right;
    ASSERT_NOT_NULL(node);
    ASSERT_EQ(node->col, 1);
    ASSERT_NEAR(node->value, 5.0, 0.0);
    ASSERT_NULL(node->right);

    node = C->row_headers[2];
    ASSERT_NOT_NULL(node);
    ASSERT_EQ(node->col, 2);
    ASSERT_NEAR(node->value, 4.0, 0.0);
    ASSERT_NULL(node->right);
    ASSERT_NEAR(sparse_get_phys(C, 0, 3), 0.0, 0.0);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_matmul_nan_drops_and_infinity_is_retained(void) {
    SparseMatrix *A = sparse_create(1, 1);
    SparseMatrix *B = sparse_create(1, 2);
    SparseMatrix *C = NULL;
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_ERR(sparse_insert(A, 0, 0, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 0, 0, nan("")), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 0, 1, HUGE_VAL), SPARSE_OK);

    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_nnz(C), 1);
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 0.0, 0.0);
    ASSERT_TRUE(isinf(sparse_get_phys(C, 0, 1)));

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_matmul_column_links_and_consumers(void) {
    SparseMatrix *A = sparse_create(4, 3);
    SparseMatrix *B = sparse_create(3, 5);
    SparseMatrix *C = NULL;
    SparseCsr *csr = NULL;
    SparseCsc *csc = NULL;
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_ERR(sparse_insert(A, 0, 0, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(A, 2, 1, 2.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(A, 3, 2, 3.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 0, 0, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 0, 2, 2.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 1, 0, 4.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 1, 2, 5.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 2, 2, 6.0), SPARSE_OK);

    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(C->rows, 4);
    ASSERT_EQ(C->cols, 5);
    ASSERT_EQ(C->nnz, 5);
    ASSERT_NULL(C->row_headers[1]);
    ASSERT_NULL(C->col_headers[1]);
    ASSERT_NULL(C->col_headers[3]);
    ASSERT_NULL(C->col_headers[4]);
    ASSERT_EQ(C->factored, 0);
    ASSERT_NEAR(C->cached_norm, -1.0, 0.0);
    for (idx_t i = 0; i < C->rows; i++) {
        ASSERT_EQ(C->row_perm[i], i);
        ASSERT_EQ(C->inv_row_perm[i], i);
    }
    for (idx_t j = 0; j < C->cols; j++) {
        ASSERT_EQ(C->col_perm[j], j);
        ASSERT_EQ(C->inv_col_perm[j], j);
    }

    Node *r00 = C->row_headers[0];
    Node *r02 = r00->right;
    Node *r20 = C->row_headers[2];
    Node *r22 = r20->right;
    Node *r32 = C->row_headers[3];
    ASSERT_EQ(r00->col, 0);
    ASSERT_EQ(r02->col, 2);
    ASSERT_EQ(r20->col, 0);
    ASSERT_EQ(r22->col, 2);
    ASSERT_EQ(r32->col, 2);
    ASSERT_NULL(r02->right);
    ASSERT_NULL(r22->right);
    ASSERT_NULL(r32->right);
    ASSERT_TRUE(C->col_headers[0] == r00);
    ASSERT_TRUE(r00->down == r20);
    ASSERT_NULL(r20->down);
    ASSERT_TRUE(C->col_headers[2] == r02);
    ASSERT_TRUE(r02->down == r22);
    ASSERT_TRUE(r22->down == r32);
    ASSERT_NULL(r32->down);
    ASSERT_NEAR(sparse_get(C, 2, 2), 10.0, 0.0);

    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[4] = {0.0};
    ASSERT_ERR(sparse_matvec(C, x, y), SPARSE_OK);
    ASSERT_NEAR(y[0], 7.0, 0.0);
    ASSERT_NEAR(y[1], 0.0, 0.0);
    ASSERT_NEAR(y[2], 38.0, 0.0);
    ASSERT_NEAR(y[3], 54.0, 0.0);

    ASSERT_ERR(sparse_to_csr(C, &csr), SPARSE_OK);
    ASSERT_ERR(sparse_to_csc(C, &csc), SPARSE_OK);
    ASSERT_EQ(csr->nnz, 5);
    ASSERT_EQ(csc->nnz, 5);
    const idx_t expected_row_ptr[5] = {0, 2, 2, 4, 5};
    const idx_t expected_col_ptr[6] = {0, 2, 2, 5, 5, 5};
    const idx_t expected_csr_cols[5] = {0, 2, 0, 2, 2};
    const idx_t expected_csc_rows[5] = {0, 2, 0, 2, 3};
    for (idx_t i = 0; i < 5; i++) {
        ASSERT_EQ(csr->row_ptr[i], expected_row_ptr[i]);
        ASSERT_EQ(csr->col_idx[i], expected_csr_cols[i]);
        ASSERT_EQ(csc->row_idx[i], expected_csc_rows[i]);
    }
    for (idx_t j = 0; j < 6; j++)
        ASSERT_EQ(csc->col_ptr[j], expected_col_ptr[j]);

    sparse_csr_free(csr);
    sparse_csc_free(csc);
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

#define MATMUL_ORACLE_MAX_DIM 6

static void assert_matmul_matches_dense_oracle_with_support(const char *name, idx_t rows,
                                                            idx_t inner, idx_t cols,
                                                            const double *a, const double *b,
                                                            const unsigned char *support,
                                                            double tolerance) {
    double expected[MATMUL_ORACLE_MAX_DIM * MATMUL_ORACLE_MAX_DIM] = {0.0};
    Node *row_nodes[MATMUL_ORACLE_MAX_DIM * MATMUL_ORACLE_MAX_DIM] = {NULL};
    SparseMatrix *A = NULL;
    SparseMatrix *B = NULL;
    SparseMatrix *C = NULL;
    SparseCsr *csr = NULL;
    SparseCsc *csc = NULL;
    idx_t expected_nnz = 0;
    idx_t row_seen = 0;
    idx_t col_seen = 0;

    printf("    dense oracle: %s\n", name);
    ASSERT_TRUE(rows > 0 && rows <= MATMUL_ORACLE_MAX_DIM);
    ASSERT_TRUE(inner > 0 && inner <= MATMUL_ORACLE_MAX_DIM);
    ASSERT_TRUE(cols > 0 && cols <= MATMUL_ORACLE_MAX_DIM);
    if (rows <= 0 || rows > MATMUL_ORACLE_MAX_DIM || inner <= 0 || inner > MATMUL_ORACLE_MAX_DIM ||
        cols <= 0 || cols > MATMUL_ORACLE_MAX_DIM)
        return;

    for (idx_t i = 0; i < rows; i++) {
        for (idx_t p = 0; p < inner; p++) {
            for (idx_t j = 0; j < cols; j++)
                expected[i * cols + j] += a[i * inner + p] * b[p * cols + j];
        }
    }
    for (idx_t i = 0; i < rows * cols; i++)
        expected_nnz += support ? support[i] != 0 : expected[i] != 0.0;

    A = sparse_create(rows, inner);
    B = sparse_create(inner, cols);
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    for (idx_t i = 0; i < rows; i++) {
        for (idx_t p = 0; p < inner; p++) {
            double value = a[i * inner + p];
            if (value != 0.0) {
                sparse_err_t err = sparse_insert(A, i, p, value);
                ASSERT_ERR(err, SPARSE_OK);
                if (err != SPARSE_OK)
                    goto cleanup;
            }
        }
    }
    for (idx_t p = 0; p < inner; p++) {
        for (idx_t j = 0; j < cols; j++) {
            double value = b[p * cols + j];
            if (value != 0.0) {
                sparse_err_t err = sparse_insert(B, p, j, value);
                ASSERT_ERR(err, SPARSE_OK);
                if (err != SPARSE_OK)
                    goto cleanup;
            }
        }
    }

    sparse_err_t err = sparse_matmul(A, B, &C);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_NOT_NULL(C);
    if (err != SPARSE_OK || !C)
        goto cleanup;
    ASSERT_EQ(C->rows, rows);
    ASSERT_EQ(C->cols, cols);
    if (C->rows != rows || C->cols != cols)
        goto cleanup;
    ASSERT_EQ(C->nnz, expected_nnz);

    for (idx_t i = 0; i < rows; i++) {
        idx_t prev_col = -1;
        for (Node *node = C->row_headers[i]; node; node = node->right) {
            if (row_seen >= rows * cols || node->col < 0 || node->col >= cols ||
                node->col <= prev_col) {
                ASSERT_TRUE(row_seen < rows * cols && node->col >= 0 && node->col < cols &&
                            node->col > prev_col);
                goto cleanup;
            }
            idx_t pos = i * cols + node->col;
            ASSERT_EQ(node->row, i);
            ASSERT_TRUE(isfinite(node->value));
            ASSERT_TRUE(support ? support[pos] != 0 : expected[pos] != 0.0);
            ASSERT_NEAR(node->value, expected[pos], tolerance);
            row_nodes[pos] = node;
            prev_col = node->col;
            row_seen++;
        }
    }
    ASSERT_EQ(row_seen, expected_nnz);
    for (idx_t pos = 0; pos < rows * cols; pos++)
        ASSERT_EQ(row_nodes[pos] != NULL, support ? support[pos] != 0 : expected[pos] != 0.0);

    for (idx_t j = 0; j < cols; j++) {
        idx_t prev_row = -1;
        for (Node *node = C->col_headers[j]; node; node = node->down) {
            if (col_seen >= rows * cols || node->row < 0 || node->row >= rows ||
                node->row <= prev_row) {
                ASSERT_TRUE(col_seen < rows * cols && node->row >= 0 && node->row < rows &&
                            node->row > prev_row);
                goto cleanup;
            }
            ASSERT_EQ(node->col, j);
            ASSERT_TRUE(row_nodes[node->row * cols + j] == node);
            prev_row = node->row;
            col_seen++;
        }
    }
    ASSERT_EQ(col_seen, expected_nnz);

    for (idx_t i = 0; i < rows; i++) {
        for (idx_t p = 0; p < inner; p++) {
            double value = sparse_get(A, i, p);
            ASSERT_TRUE(isfinite(value));
            ASSERT_NEAR(value, a[i * inner + p], 0.0);
        }
    }
    for (idx_t p = 0; p < inner; p++) {
        for (idx_t j = 0; j < cols; j++) {
            double value = sparse_get(B, p, j);
            ASSERT_TRUE(isfinite(value));
            ASSERT_NEAR(value, b[p * cols + j], 0.0);
        }
    }

    double x[MATMUL_ORACLE_MAX_DIM] = {0.0};
    double y[MATMUL_ORACLE_MAX_DIM] = {0.0};
    for (idx_t j = 0; j < cols; j++)
        x[j] = (double)(j + 1);
    ASSERT_ERR(sparse_matvec(C, x, y), SPARSE_OK);
    for (idx_t i = 0; i < rows; i++) {
        double expected_y = 0.0;
        for (idx_t j = 0; j < cols; j++) {
            idx_t pos = i * cols + j;
            int present = support ? support[pos] != 0 : expected[pos] != 0.0;
            double value = present ? expected[pos] : 0.0;
            double actual = sparse_get(C, i, j);
            ASSERT_TRUE(isfinite(actual));
            ASSERT_NEAR(actual, value, tolerance);
            expected_y += value * x[j];
        }
        ASSERT_TRUE(isfinite(y[i]));
        ASSERT_NEAR(y[i], expected_y, tolerance * (double)cols * (double)cols);
    }

    sparse_err_t csr_err = sparse_to_csr(C, &csr);
    sparse_err_t csc_err = sparse_to_csc(C, &csc);
    ASSERT_ERR(csr_err, SPARSE_OK);
    ASSERT_ERR(csc_err, SPARSE_OK);
    ASSERT_NOT_NULL(csr);
    ASSERT_NOT_NULL(csc);
    if (csr_err != SPARSE_OK || csc_err != SPARSE_OK || !csr || !csc)
        goto cleanup;
    ASSERT_EQ(csr->rows, rows);
    ASSERT_EQ(csr->cols, cols);
    ASSERT_EQ(csr->nnz, expected_nnz);
    ASSERT_EQ(csc->rows, rows);
    ASSERT_EQ(csc->cols, cols);
    ASSERT_EQ(csc->nnz, expected_nnz);

    idx_t k = 0;
    for (idx_t i = 0; i < rows; i++) {
        ASSERT_EQ(csr->row_ptr[i], k);
        for (idx_t j = 0; j < cols; j++) {
            idx_t pos = i * cols + j;
            if (support ? support[pos] != 0 : expected[pos] != 0.0) {
                ASSERT_TRUE(k < csr->nnz);
                if (k >= csr->nnz)
                    goto cleanup;
                ASSERT_EQ(csr->col_idx[k], j);
                ASSERT_TRUE(isfinite(csr->values[k]));
                ASSERT_NEAR(csr->values[k], expected[pos], tolerance);
                k++;
            }
        }
    }
    ASSERT_EQ(csr->row_ptr[rows], k);
    ASSERT_EQ(k, expected_nnz);

    k = 0;
    for (idx_t j = 0; j < cols; j++) {
        ASSERT_EQ(csc->col_ptr[j], k);
        for (idx_t i = 0; i < rows; i++) {
            idx_t pos = i * cols + j;
            if (support ? support[pos] != 0 : expected[pos] != 0.0) {
                ASSERT_TRUE(k < csc->nnz);
                if (k >= csc->nnz)
                    goto cleanup;
                ASSERT_EQ(csc->row_idx[k], i);
                ASSERT_TRUE(isfinite(csc->values[k]));
                ASSERT_NEAR(csc->values[k], expected[pos], tolerance);
                k++;
            }
        }
    }
    ASSERT_EQ(csc->col_ptr[cols], k);
    ASSERT_EQ(k, expected_nnz);

cleanup:
    sparse_csr_free(csr);
    sparse_csc_free(csc);
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void assert_matmul_matches_dense_oracle(const char *name, idx_t rows, idx_t inner,
                                               idx_t cols, const double *a, const double *b) {
    assert_matmul_matches_dense_oracle_with_support(name, rows, inner, cols, a, b, NULL, 0.0);
}

static void test_matmul_dense_oracle_rectangular(void) {
    const double a[6] = {1.0, 0.0, 0.0, 2.0, 3.0, 4.0};
    const double b[8] = {1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0};
    assert_matmul_matches_dense_oracle("rectangular 3x2 * 2x4", 3, 2, 4, a, b);
}

static void test_matmul_dense_oracle_identity(void) {
    const double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double values[9] = {2.0, 0.0, 5.0, 0.0, -3.0, 0.0, 7.0, 0.0, 1.0};
    assert_matmul_matches_dense_oracle("left identity", 3, 3, 3, identity, values);
    assert_matmul_matches_dense_oracle("right identity", 3, 3, 3, values, identity);
}

static void test_matmul_dense_oracle_empty(void) {
    const double zero_a[6] = {0.0};
    const double zero_b[6] = {0.0};
    const double nonzero_a[6] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    const double nonzero_b[6] = {1.0, 0.0, 2.0, 3.0, 0.0, 4.0};
    assert_matmul_matches_dense_oracle("empty left operand", 2, 3, 2, zero_a, nonzero_b);
    assert_matmul_matches_dense_oracle("empty right operand", 2, 3, 2, nonzero_a, zero_b);
}

static void test_matmul_dense_oracle_empty_rows_and_columns(void) {
    const double a[12] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};
    const double b[15] = {1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 5.0,
                          0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0};
    assert_matmul_matches_dense_oracle("empty output row and columns", 4, 3, 5, a, b);
}

static void test_matmul_dense_oracle_dense_result(void) {
    const double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double b[6] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    assert_matmul_matches_dense_oracle("dense 2x3 * 3x2", 2, 3, 2, a, b);
}

static void test_matmul_dense_oracle_duplicate_and_cancellation_support(void) {
    const double a[2] = {1.0, 1.0};
    const double b[12] = {2.0,
                          1.0,
                          1.0,
                          1.0,
                          -1.0,
                          -1.0,
                          3.0,
                          -1.0,
                          -(1.0 - 0x1p-50),
                          -(1.0 - 0x1p-49),
                          1.0 - 0x1p-50,
                          1.0 - 0x1p-49};
    const unsigned char support[6] = {1, 0, 0, 1, 0, 1};
    assert_matmul_matches_dense_oracle_with_support(
        "duplicates, cancellation, and signed near-threshold residuals", 1, 2, 6, a, b, support,
        0.0);
}

static void test_matmul_dense_oracle_threshold_boundary(void) {
    const double a[1] = {1.0};
    const double b[3] = {nextafter(1e-15, 0.0), 1e-15, nextafter(1e-15, HUGE_VAL)};
    const unsigned char support[3] = {0, 1, 1};
    assert_matmul_matches_dense_oracle_with_support("strictly below, equal, and above threshold", 1,
                                                    1, 3, a, b, support, 0.0);
}

static void test_matmul_dense_oracle_fractional_duplicate(void) {
    const double a[3] = {1.0, 1.0, 1.0};
    const double b[3] = {0.1, 0.2, 0.3};
    const unsigned char support[1] = {1};
    assert_matmul_matches_dense_oracle_with_support("fractional duplicate contributions", 1, 3, 1,
                                                    a, b, support, 8.0 * DBL_EPSILON);
}

static void test_matmul_wide_rows_repeat_deterministically(void) {
    const idx_t rows = 3;
    const idx_t n = 257;
    SparseMatrix *A = sparse_create(rows, n);
    SparseMatrix *B = sparse_create(n, n);
    SparseMatrix *C = NULL;
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;

    for (idx_t i = 0; i < rows; i++) {
        for (idx_t j = 0; j < n; j++) {
            sparse_err_t err = sparse_insert(A, i, j, (double)(i + 1));
            ASSERT_ERR(err, SPARSE_OK);
            if (err != SPARSE_OK)
                goto cleanup;
        }
    }
    for (idx_t j = 0; j < n; j++) {
        sparse_err_t err = sparse_insert(B, j, n - 1 - j, 1.0);
        ASSERT_ERR(err, SPARSE_OK);
        if (err != SPARSE_OK)
            goto cleanup;
    }

    for (int run = 0; run < 3; run++) {
        sparse_err_t err = sparse_matmul(A, B, &C);
        ASSERT_ERR(err, SPARSE_OK);
        ASSERT_NOT_NULL(C);
        if (err != SPARSE_OK || !C)
            goto cleanup;
        ASSERT_EQ(C->nnz, rows * n);

        for (idx_t i = 0; i < rows; i++) {
            Node *node = C->row_headers[i];
            for (idx_t j = 0; j < n; j++) {
                ASSERT_NOT_NULL(node);
                if (!node)
                    goto cleanup;
                ASSERT_EQ(node->row, i);
                ASSERT_EQ(node->col, j);
                ASSERT_NEAR(node->value, (double)(i + 1), 0.0);
                node = node->right;
            }
            ASSERT_NULL(node);
        }
        for (idx_t j = 0; j < n; j++) {
            Node *node = C->col_headers[j];
            for (idx_t i = 0; i < rows; i++) {
                ASSERT_NOT_NULL(node);
                if (!node)
                    goto cleanup;
                ASSERT_EQ(node->row, i);
                ASSERT_EQ(node->col, j);
                ASSERT_NEAR(node->value, (double)(i + 1), 0.0);
                node = node->down;
            }
            ASSERT_NULL(node);
        }

        sparse_free(C);
        C = NULL;
    }

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static uint32_t matmul_seeded_next(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static void matmul_seeded_fill(double *values, idx_t rows, idx_t cols, uint32_t *state,
                               int sparse) {
    for (idx_t i = 0; i < rows * cols; i++) {
        uint32_t word = matmul_seeded_next(state);
        values[i] = sparse && ((word >> 16) % 4u == 0u) ? 0.0 : (double)(1u + (word >> 8) % 4u);
    }
}

static void test_matmul_seeded_tall_sparse_with_empty_intermediate(void) {
    uint32_t state = UINT32_C(0x2180801);
    double a[5 * 3];
    double b[3 * 4];
    matmul_seeded_fill(a, 5, 3, &state, 1);
    matmul_seeded_fill(b, 3, 4, &state, 1);
    for (idx_t p = 0; p < 3; p++)
        b[p * 4 + 3] = 0.0;
    for (idx_t j = 0; j < 4; j++)
        b[1 * 4 + j] = 0.0;
    for (idx_t p = 0; p < 3; p++)
        a[2 * 3 + p] = 0.0;
    a[0] = 1.0;
    a[1] = 3.0;
    b[0] = 2.0;
    const unsigned char support[20] = {1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0};
    assert_matmul_matches_dense_oracle_with_support("seeded tall sparse, empty intermediate row", 5,
                                                    3, 4, a, b, support, 0.0);
}

static void test_matmul_seeded_tall_dense(void) {
    uint32_t state = UINT32_C(0x2180802);
    double a[5 * 2];
    double b[2 * 3];
    matmul_seeded_fill(a, 5, 2, &state, 0);
    matmul_seeded_fill(b, 2, 3, &state, 0);
    const unsigned char support[15] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    assert_matmul_matches_dense_oracle_with_support("seeded tall dense result", 5, 2, 3, a, b,
                                                    support, 0.0);
}

static void test_matmul_seeded_wide_sparse(void) {
    uint32_t state = UINT32_C(0x2180803);
    double a[2 * 5];
    double b[5 * 6];
    matmul_seeded_fill(a, 2, 5, &state, 0);
    matmul_seeded_fill(b, 5, 6, &state, 1);
    for (idx_t p = 0; p < 5; p++) {
        b[p * 6 + 1] = 0.0;
        b[p * 6 + 3] = 0.0;
        b[p * 6 + 5] = 0.0;
    }
    b[0] = 2.0;
    const unsigned char support[12] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    assert_matmul_matches_dense_oracle_with_support("seeded wide sparse result", 2, 5, 6, a, b,
                                                    support, 0.0);
}

static void test_matmul_seeded_wide_dense(void) {
    uint32_t state = UINT32_C(0x2180804);
    double a[2 * 5];
    double b[5 * 6];
    matmul_seeded_fill(a, 2, 5, &state, 0);
    matmul_seeded_fill(b, 5, 6, &state, 0);
    const unsigned char support[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    assert_matmul_matches_dense_oracle_with_support("seeded wide dense result", 2, 5, 6, a, b,
                                                    support, 0.0);
}

static void test_matmul_dense_oracle_large_finite(void) {
    const double a[2] = {1e150, 1e150};
    const double b[4] = {1e150, 0.0, 0.0, 1e150};
    assert_matmul_matches_dense_oracle("large finite products", 1, 2, 2, a, b);
}

static void test_matmul_nonfinite_accumulation_controls(void) {
    SparseMatrix *A = sparse_create(1, 2);
    SparseMatrix *B = sparse_create(2, 3);
    SparseMatrix *C = NULL;
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    ASSERT_ERR(sparse_insert(A, 0, 0, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(A, 0, 1, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 0, 0, HUGE_VAL), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 1, 0, -HUGE_VAL), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 0, 1, DBL_MAX), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 1, 1, DBL_MAX), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 0, 2, 1e150), SPARSE_OK);
    ASSERT_ERR(sparse_insert(B, 1, 2, 1e150), SPARSE_OK);

    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    if (!C)
        goto cleanup;
    ASSERT_EQ(C->nnz, 2);
    Node *node = C->row_headers[0];
    ASSERT_NOT_NULL(node);
    if (!node)
        goto cleanup;
    ASSERT_EQ(node->col, 1);
    ASSERT_TRUE(isinf(node->value) && node->value > 0.0);
    node = node->right;
    ASSERT_NOT_NULL(node);
    if (!node)
        goto cleanup;
    ASSERT_EQ(node->col, 2);
    ASSERT_TRUE(isfinite(node->value));
    ASSERT_NEAR(node->value, 2e150, 0.0);
    ASSERT_NULL(node->right);
    ASSERT_NEAR(sparse_get(C, 0, 0), 0.0, 0.0);
    ASSERT_TRUE(isinf(sparse_get(B, 0, 0)));
    ASSERT_TRUE(isinf(sparse_get(B, 1, 0)));
    ASSERT_TRUE(sparse_get(B, 0, 1) == DBL_MAX);
    ASSERT_TRUE(sparse_get(B, 1, 1) == DBL_MAX);

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

/* A * empty = empty (nnz=0) */
static void test_matmul_empty(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);

    SparseMatrix *B = sparse_create(3, 3); /* all zeros */

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), 0);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

/* Single-row * single-column → 1×1 (dot product) */
static void test_matmul_row_col(void) {
    SparseMatrix *A = sparse_create(1, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 0, 2, 3.0);

    SparseMatrix *B = sparse_create(3, 1);
    sparse_insert(B, 0, 0, 4.0);
    sparse_insert(B, 1, 0, 5.0);
    sparse_insert(B, 2, 0, 6.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_EQ(sparse_rows(C), 1);
    ASSERT_EQ(sparse_cols(C), 1);
    /* 1*4 + 2*5 + 3*6 = 32 */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 32.0, 0.0);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

/* Very sparse: single nnz each */
static void test_matmul_single_nnz(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 1, 2, 5.0); /* only entry */

    SparseMatrix *B = sparse_create(3, 3);
    sparse_insert(B, 2, 0, 7.0); /* only entry */

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), 1);
    /* C(1,0) = A(1,2)*B(2,0) = 35 */
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 35.0, 0.0);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Allocation-failure harness
 * ═══════════════════════════════════════════════════════════════════════ */

#define MATMUL_FAIL_AFTER_ACC 7L
#define MATMUL_FAIL_AFTER_NZ_FLAG 8L
#define MATMUL_FAIL_AFTER_TOUCHED 9L
#define MATMUL_FAIL_AFTER_COL_TAILS 10L
#define MATMUL_FAIL_AFTER_FIRST_SLAB 11L

typedef struct {
    const char *name;
    long fail_after;
} MatmulAllocationFailureCase;

static SparseMatrix *build_matmul_failure_A(void) {
    SparseMatrix *A = sparse_create(2, 3);
    if (!A)
        return NULL;

    if (sparse_insert(A, 0, 0, 1.0) != SPARSE_OK || sparse_insert(A, 0, 2, 2.0) != SPARSE_OK ||
        sparse_insert(A, 1, 1, 3.0) != SPARSE_OK) {
        sparse_free(A);
        return NULL;
    }

    return A;
}

static SparseMatrix *build_matmul_failure_B(void) {
    SparseMatrix *B = sparse_create(3, 2);
    if (!B)
        return NULL;

    if (sparse_insert(B, 0, 0, 4.0) != SPARSE_OK || sparse_insert(B, 2, 1, 5.0) != SPARSE_OK ||
        sparse_insert(B, 1, 0, 6.0) != SPARSE_OK) {
        sparse_free(B);
        return NULL;
    }

    return B;
}

static void assert_matmul_failure_retry_product(const SparseMatrix *C) {
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_rows(C), 2);
    ASSERT_EQ(sparse_cols(C), 2);
    ASSERT_EQ(sparse_nnz(C), 3);
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 1), 10.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 18.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 0.0, 0.0);
}

static void assert_matmul_failure_inputs_unchanged(const SparseMatrix *A, const SparseMatrix *B) {
    const double a[6] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    const double b[6] = {4.0, 0.0, 6.0, 0.0, 0.0, 5.0};
    ASSERT_EQ(sparse_nnz(A), 3);
    ASSERT_EQ(sparse_nnz(B), 3);
    for (idx_t i = 0; i < 2; i++) {
        for (idx_t j = 0; j < 3; j++)
            ASSERT_NEAR(sparse_get_phys(A, i, j), a[i * 3 + j], 0.0);
    }
    for (idx_t i = 0; i < 3; i++) {
        for (idx_t j = 0; j < 2; j++)
            ASSERT_NEAR(sparse_get_phys(B, i, j), b[i * 2 + j], 0.0);
    }
}

static void expect_matmul_allocation_failure(const MatmulAllocationFailureCase *failure_case) {
    SparseMatrix *A = build_matmul_failure_A();
    SparseMatrix *B = build_matmul_failure_B();
    SparseMatrix *C = NULL;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);

    sparse_alloc_test_reset();
    sparse_alloc_test_fail_after(failure_case->fail_after);
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_ALLOC);
    ASSERT_NULL(C);
    assert_matmul_failure_inputs_unchanged(A, B);

    sparse_alloc_test_reset();
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    assert_matmul_failure_retry_product(C);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
    sparse_alloc_test_reset();
}

static void test_matmul_workspace_allocation_failure_recovers(void) {
    static const MatmulAllocationFailureCase cases[] = {
        {"accumulator workspace", MATMUL_FAIL_AFTER_ACC},
        {"nonzero flag workspace", MATMUL_FAIL_AFTER_NZ_FLAG},
        {"touched column workspace", MATMUL_FAIL_AFTER_TOUCHED},
        {"column tail workspace", MATMUL_FAIL_AFTER_COL_TAILS},
    };
    const size_t case_count = sizeof(cases) / sizeof(cases[0]);

    for (size_t i = 0; i < case_count; ++i) {
        printf("    allocation-failure site: %s\n", cases[i].name);
        expect_matmul_allocation_failure(&cases[i]);
    }
}

static void expect_matmul_allocation_failure_clears_stale_output(
    const MatmulAllocationFailureCase *failure_case) {
    SparseMatrix *A = build_matmul_failure_A();
    SparseMatrix *B = build_matmul_failure_B();
    SparseMatrix *stale = sparse_create(1, 1);
    SparseMatrix *C = stale;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(stale);
    ASSERT_ERR(sparse_insert(stale, 0, 0, 42.0), SPARSE_OK);

    sparse_alloc_test_reset();
    sparse_alloc_test_fail_after(failure_case->fail_after);
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_ALLOC);
    ASSERT_NULL(C);
    ASSERT_NEAR(sparse_get_phys(stale, 0, 0), 42.0, 0.0);
    assert_matmul_failure_inputs_unchanged(A, B);

    sparse_alloc_test_reset();
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    assert_matmul_failure_retry_product(C);

    sparse_free(A);
    sparse_free(B);
    sparse_free(stale);
    sparse_free(C);
    sparse_alloc_test_reset();
}

static void test_matmul_acc_allocation_failure_clears_stale_output(void) {
    static const MatmulAllocationFailureCase failure_case = {"accumulator workspace",
                                                             MATMUL_FAIL_AFTER_ACC};

    expect_matmul_allocation_failure_clears_stale_output(&failure_case);
}

static void test_matmul_remaining_workspace_allocation_failures_clear_stale_output(void) {
    static const MatmulAllocationFailureCase cases[] = {
        {"nonzero flag workspace", MATMUL_FAIL_AFTER_NZ_FLAG},
        {"touched column workspace", MATMUL_FAIL_AFTER_TOUCHED},
        {"column tail workspace", MATMUL_FAIL_AFTER_COL_TAILS},
    };
    const size_t case_count = sizeof(cases) / sizeof(cases[0]);

    for (size_t i = 0; i < case_count; ++i) {
        printf("    stale-output allocation-failure site: %s\n", cases[i].name);
        expect_matmul_allocation_failure_clears_stale_output(&cases[i]);
    }
}

static void test_matmul_result_shell_and_first_node_allocation_failures(void) {
    static const MatmulAllocationFailureCase cases[] = {
        {"result shell", 0L},
        {"result row headers", 1L},
        {"result column headers", 2L},
        {"result row permutation", 3L},
        {"result inverse row permutation", 4L},
        {"result column permutation", 5L},
        {"result inverse column permutation", 6L},
        {"first result node slab", MATMUL_FAIL_AFTER_FIRST_SLAB},
    };
    const size_t case_count = sizeof(cases) / sizeof(cases[0]);

    for (size_t i = 0; i < case_count; i++) {
        printf("    result allocation-failure site: %s\n", cases[i].name);
        expect_matmul_allocation_failure_clears_stale_output(&cases[i]);
    }
}

static void test_matmul_later_node_slab_failure_after_partial_links(void) {
    const idx_t slab_size = NODES_PER_SLAB;
    const idx_t cols = slab_size >= 5   ? slab_size / 2 + 2
                       : slab_size == 4 ? 3
                       : slab_size == 2 ? 3
                                        : 2;
    const idx_t slab_number = (cols + slab_size) / slab_size;
    const idx_t failed_node_index = slab_number * slab_size;
    const long fail_after = MATMUL_FAIL_AFTER_FIRST_SLAB + (long)slab_number;
    SparseMatrix *A = sparse_create(2, 1);
    SparseMatrix *B = sparse_create(1, cols);
    SparseMatrix *stale = sparse_create(1, 1);
    SparseMatrix *C = stale;
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(stale);
    if (!A || !B || !stale)
        goto cleanup;
    ASSERT_TRUE(failed_node_index >= cols + 1 && failed_node_index < 2 * cols);

    ASSERT_ERR(sparse_insert(A, 0, 0, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(A, 1, 0, 2.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(stale, 0, 0, 42.0), SPARSE_OK);
    for (idx_t j = 0; j < cols; j++) {
        sparse_err_t err = sparse_insert(B, 0, j, 1.0);
        ASSERT_ERR(err, SPARSE_OK);
        if (err != SPARSE_OK)
            goto cleanup;
    }

    sparse_alloc_test_reset();
    sparse_alloc_test_fail_after(fail_after);
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_ALLOC);
    ASSERT_NULL(C);
    ASSERT_NEAR(sparse_get_phys(stale, 0, 0), 42.0, 0.0);
    ASSERT_EQ(sparse_nnz(A), 2);
    ASSERT_EQ(sparse_nnz(B), cols);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 0), 2.0, 0.0);
    for (idx_t j = 0; j < cols; j++)
        ASSERT_NEAR(sparse_get_phys(B, 0, j), 1.0, 0.0);

    sparse_alloc_test_reset();
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    if (!C)
        goto cleanup;
    ASSERT_EQ(C->nnz, 2 * cols);
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, cols - 1), 2.0, 0.0);
    for (idx_t j = 0; j < cols; j++) {
        Node *top = C->col_headers[j];
        ASSERT_NOT_NULL(top);
        if (!top)
            goto cleanup;
        ASSERT_EQ(top->row, 0);
        ASSERT_EQ(top->col, j);
        ASSERT_NOT_NULL(top->down);
        if (!top->down)
            goto cleanup;
        ASSERT_EQ(top->down->row, 1);
        ASSERT_EQ(top->down->col, j);
        ASSERT_NULL(top->down->down);
    }

cleanup:
    sparse_alloc_test_reset();
    sparse_free(A);
    sparse_free(B);
    sparse_free(stale);
    sparse_free(C == stale ? NULL : C);
}

static void assert_matmul_rollback_retry_product(const SparseMatrix *C, idx_t cols) {
    ASSERT_NOT_NULL(C);
    if (!C)
        return;
    ASSERT_EQ(C->rows, 3);
    ASSERT_EQ(C->cols, cols);
    ASSERT_EQ(C->nnz, 3 * cols);
    for (idx_t i = 0; i < 3; i++) {
        Node *node = C->row_headers[i];
        for (idx_t j = 0; j < cols; j++) {
            ASSERT_NOT_NULL(node);
            if (!node)
                return;
            ASSERT_EQ(node->row, i);
            ASSERT_EQ(node->col, j);
            ASSERT_NEAR(node->value, (double)(i + 1) * (double)(j % 7 + 1), 0.0);
            node = node->right;
        }
        ASSERT_NULL(node);
    }
    for (idx_t j = 0; j < cols; j++) {
        Node *node = C->col_headers[j];
        for (idx_t i = 0; i < 3; i++) {
            ASSERT_NOT_NULL(node);
            if (!node)
                return;
            ASSERT_EQ(node->row, i);
            ASSERT_EQ(node->col, j);
            ASSERT_NEAR(node->value, (double)(i + 1) * (double)(j % 7 + 1), 0.0);
            node = node->down;
        }
        ASSERT_NULL(node);
    }
}

static void test_matmul_node_slab_failure_sweep_rollback_and_retry(void) {
    const idx_t cols = NODES_PER_SLAB + 2;
    const idx_t total_nodes = 3 * cols;
    SparseMatrix *A = sparse_create(3, 1);
    SparseMatrix *B = sparse_create(1, cols);
    SparseMatrix *stale = sparse_create(1, 1);
    SparseMatrix *C = NULL;
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(stale);
    if (!A || !B || !stale)
        goto cleanup;
    ASSERT_ERR(sparse_insert(stale, 0, 0, 42.0), SPARSE_OK);
    for (idx_t i = 0; i < 3; i++) {
        sparse_err_t err = sparse_insert(A, i, 0, (double)(i + 1));
        ASSERT_ERR(err, SPARSE_OK);
        if (err != SPARSE_OK)
            goto cleanup;
    }
    for (idx_t j = 0; j < cols; j++) {
        sparse_err_t err = sparse_insert(B, 0, j, (double)(j % 7 + 1));
        ASSERT_ERR(err, SPARSE_OK);
        if (err != SPARSE_OK)
            goto cleanup;
    }

    for (idx_t slab = 1; slab * NODES_PER_SLAB < total_nodes; slab++) {
        idx_t failed_node = slab * NODES_PER_SLAB;
        sparse_err_t err;
        printf("    rollback at node %lld (row %lld, column %lld)\n", (long long)failed_node,
               (long long)(failed_node / cols), (long long)(failed_node % cols));
        C = stale;
        sparse_alloc_test_reset();
        sparse_alloc_test_fail_after(MATMUL_FAIL_AFTER_FIRST_SLAB + (long)slab);
        err = sparse_matmul(A, B, &C);
        ASSERT_ERR(err, SPARSE_ERR_ALLOC);
        ASSERT_NULL(C);
        if (err != SPARSE_ERR_ALLOC || C)
            goto cleanup;
        ASSERT_NEAR(sparse_get_phys(stale, 0, 0), 42.0, 0.0);
        sparse_alloc_test_reset();

        ASSERT_EQ(sparse_nnz(A), 3);
        ASSERT_EQ(sparse_nnz(B), cols);
        for (idx_t i = 0; i < 3; i++)
            ASSERT_NEAR(sparse_get_phys(A, i, 0), (double)(i + 1), 0.0);
        for (idx_t j = 0; j < cols; j++)
            ASSERT_NEAR(sparse_get_phys(B, 0, j), (double)(j % 7 + 1), 0.0);

        err = sparse_matmul(A, B, &C);
        ASSERT_ERR(err, SPARSE_OK);
        ASSERT_NOT_NULL(C);
        if (err != SPARSE_OK || !C)
            goto cleanup;
        assert_matmul_rollback_retry_product(C, cols);
        sparse_free(C);
        C = NULL;
    }

cleanup:
    sparse_alloc_test_reset();
    sparse_free(A);
    sparse_free(B);
    sparse_free(stale);
    sparse_free(C == stale ? NULL : C);
}

static void test_matmul_error_precedence_clears_stale_output(void) {
    SparseMatrix *A = build_matmul_failure_A();
    SparseMatrix *B = build_matmul_failure_B();
    SparseMatrix *shape_bad = sparse_create(4, 2);
    SparseMatrix *stale = sparse_create(1, 1);
    SparseMatrix *C = stale;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(shape_bad);
    ASSERT_NOT_NULL(stale);
    ASSERT_ERR(sparse_insert(stale, 0, 0, 42.0), SPARSE_OK);

    ASSERT_ERR(sparse_matmul(NULL, NULL, NULL), SPARSE_ERR_NULL);

    ASSERT_ERR(sparse_matmul(NULL, NULL, &C), SPARSE_ERR_NULL);
    ASSERT_NULL(C);

    C = stale;
    ASSERT_ERR(sparse_matmul(NULL, B, &C), SPARSE_ERR_NULL);
    ASSERT_NULL(C);
    ASSERT_NEAR(sparse_get_phys(stale, 0, 0), 42.0, 0.0);

    C = stale;
    ASSERT_ERR(sparse_matmul(A, NULL, &C), SPARSE_ERR_NULL);
    ASSERT_NULL(C);
    ASSERT_NEAR(sparse_get_phys(stale, 0, 0), 42.0, 0.0);

    C = stale;
    ASSERT_ERR(sparse_matmul(A, shape_bad, &C), SPARSE_ERR_SHAPE);
    ASSERT_NULL(C);
    ASSERT_NEAR(sparse_get_phys(stale, 0, 0), 42.0, 0.0);

    ASSERT_ERR(sparse_matmul(A, shape_bad, NULL), SPARSE_ERR_NULL);

    C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    assert_matmul_failure_retry_product(C);

    sparse_free(A);
    sparse_free(B);
    sparse_free(shape_bad);
    sparse_free(stale);
    sparse_free(C);
}

static void test_matmul_zero_dimensions_rejected_by_constructor(void) {
    ASSERT_NULL(sparse_create(0, 1));
    ASSERT_NULL(sparse_create(1, 0));
    ASSERT_NULL(sparse_create(0, 0));
}

/* ═══════════════════════════════════════════════════════════════════════
 * SuiteSparse validation
 * ═══════════════════════════════════════════════════════════════════════ */

/* Associativity: (A*B)*x = A*(B*x) on nos4 */
static void test_matmul_associativity(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/nos4.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);

    /* Use A as both A and B (A*A) */
    SparseMatrix *AA = NULL;
    ASSERT_ERR(sparse_matmul(A, A, &AA), SPARSE_OK);

    /* x = [1, 2, 3, ...] */
    double *x = malloc((size_t)n * sizeof(double));
    double *Ax = malloc((size_t)n * sizeof(double));
    double *AAx_direct = malloc((size_t)n * sizeof(double));
    double *A_Ax = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(Ax);
    ASSERT_NOT_NULL(AAx_direct);
    ASSERT_NOT_NULL(A_Ax);
    for (idx_t i = 0; i < n; i++)
        x[i] = (double)(i + 1);

    /* (A*A)*x */
    sparse_matvec(AA, x, AAx_direct);

    /* A*(A*x) */
    sparse_matvec(A, x, Ax);
    sparse_matvec(A, Ax, A_Ax);

    /* Should match */
    double maxdiff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(AAx_direct[i] - A_Ax[i]);
        if (d > maxdiff)
            maxdiff = d;
    }
    printf("    nos4: (A*A)*x vs A*(A*x) maxdiff = %.2e\n", maxdiff);
    ASSERT_TRUE(maxdiff < 1e-8);

    free(x);
    free(Ax);
    free(AAx_direct);
    free(A_Ax);
    sparse_free(A);
    sparse_free(AA);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cholesky integration: L * L^T should reconstruct A
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build L^T from L by manually constructing the transpose */
static SparseMatrix *build_transpose(const SparseMatrix *L, idx_t n) {
    SparseMatrix *LT = sparse_create(n, n);
    if (!LT)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double val = sparse_get_phys(L, i, j);
            if (val != 0.0)
                sparse_insert(LT, j, i, val);
        }
    }
    return LT;
}

static void test_matmul_cholesky_reconstruct(void) {
    /* A = [[10,1,2],[1,10,1],[2,1,10]] — SPD */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 10.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 2.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 10.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 2.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 10.0);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    /* Build L^T */
    SparseMatrix *LT = build_transpose(L, 3);
    ASSERT_NOT_NULL(LT);

    /* C = L * L^T should equal A */
    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(L, LT, &C), SPARSE_OK);

    /* Verify C ≈ A */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double expected = sparse_get_phys(A, (idx_t)i, (idx_t)j);
            double got = sparse_get_phys(C, (idx_t)i, (idx_t)j);
            ASSERT_NEAR(got, expected, 1e-12);
        }
    }

    sparse_free(A);
    sparse_free(L);
    sparse_free(LT);
    sparse_free(C);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sparse Matrix-Matrix Multiply Tests");

    RUN_TEST(test_matmul_2x2);
    RUN_TEST(test_matmul_identity_left);
    RUN_TEST(test_matmul_identity_right);
    RUN_TEST(test_matmul_diag_left);
    RUN_TEST(test_matmul_diag_right);
    RUN_TEST(test_matmul_shape_mismatch);
    RUN_TEST(test_matmul_null);
    RUN_TEST(test_matmul_rectangular);
    RUN_TEST(test_matmul_cancellation);
    RUN_TEST(test_matmul_wide_row_sorted_after_reverse_touches);
    RUN_TEST(test_matmul_row_append_duplicate_cancellation_and_empty_row);
    RUN_TEST(test_matmul_nan_drops_and_infinity_is_retained);
    RUN_TEST(test_matmul_column_links_and_consumers);
    RUN_TEST(test_matmul_dense_oracle_rectangular);
    RUN_TEST(test_matmul_dense_oracle_identity);
    RUN_TEST(test_matmul_dense_oracle_empty);
    RUN_TEST(test_matmul_dense_oracle_empty_rows_and_columns);
    RUN_TEST(test_matmul_dense_oracle_dense_result);
    RUN_TEST(test_matmul_dense_oracle_duplicate_and_cancellation_support);
    RUN_TEST(test_matmul_dense_oracle_threshold_boundary);
    RUN_TEST(test_matmul_dense_oracle_fractional_duplicate);
    RUN_TEST(test_matmul_wide_rows_repeat_deterministically);
    RUN_TEST(test_matmul_seeded_tall_sparse_with_empty_intermediate);
    RUN_TEST(test_matmul_seeded_tall_dense);
    RUN_TEST(test_matmul_seeded_wide_sparse);
    RUN_TEST(test_matmul_seeded_wide_dense);
    RUN_TEST(test_matmul_dense_oracle_large_finite);
    RUN_TEST(test_matmul_nonfinite_accumulation_controls);

    /* Edge cases */
    RUN_TEST(test_matmul_empty);
    RUN_TEST(test_matmul_row_col);
    RUN_TEST(test_matmul_single_nnz);

    /* Allocation-failure harness */
    RUN_TEST(test_matmul_error_precedence_clears_stale_output);
    RUN_TEST(test_matmul_acc_allocation_failure_clears_stale_output);
    RUN_TEST(test_matmul_remaining_workspace_allocation_failures_clear_stale_output);
    RUN_TEST(test_matmul_workspace_allocation_failure_recovers);
    RUN_TEST(test_matmul_result_shell_and_first_node_allocation_failures);
    RUN_TEST(test_matmul_later_node_slab_failure_after_partial_links);
    RUN_TEST(test_matmul_node_slab_failure_sweep_rollback_and_retry);
    RUN_TEST(test_matmul_zero_dimensions_rejected_by_constructor);

    /* SuiteSparse */
    RUN_TEST(test_matmul_associativity);

    /* Cholesky integration */
    RUN_TEST(test_matmul_cholesky_reconstruct);

    TEST_SUITE_END();
}
