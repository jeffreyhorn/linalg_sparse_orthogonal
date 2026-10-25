#include "sparse_csr.h"
#include "sparse_lu_csr.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static SparseMatrix *make_tridiagonal(void) {
    SparseMatrix *mat = sparse_create(3, 3);
    if (!mat)
        return NULL;
    for (idx_t i = 0; i < 3; i++) {
        if (sparse_insert(mat, i, i, 4.0) != SPARSE_OK ||
            (i > 0 && sparse_insert(mat, i, i - 1, 1.0) != SPARSE_OK) ||
            (i < 2 && sparse_insert(mat, i, i + 1, 1.0) != SPARSE_OK)) {
            sparse_free(mat);
            return NULL;
        }
    }
    return mat;
}

static void check_tridiagonal(const SparseMatrix *mat) {
    ASSERT_EQ(sparse_rows(mat), 3);
    ASSERT_EQ(sparse_cols(mat), 3);
    ASSERT_EQ(sparse_nnz(mat), 7);
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 3; j++) {
            double expected = i == j ? 4.0 : (i + 1 == j || j + 1 == i ? 1.0 : 0.0);
            ASSERT_NEAR(sparse_get_phys(mat, i, j), expected, 0.0);
        }
}

static void test_width_identity(void) {
    size_t typedef_bits = sizeof(idx_t) * CHAR_BIT;
    size_t header_bits = SPARSE_BUILT_IDX_BITS;
    size_t archive_bits = sparse_idx_bits();
    ASSERT_EQ(typedef_bits, header_bits);
    ASSERT_EQ(archive_bits, header_bits);
    printf("    index width: typedef=%zu header=%zu archive=%zu\n", typedef_bits, header_bits,
           archive_bits);
}

static void test_matrix_insert_and_lu(void) {
    SparseMatrix *mat = make_tridiagonal();
    ASSERT_NOT_NULL(mat);
    if (!mat)
        return;
    check_tridiagonal(mat);

    double b[3] = {6.0, 12.0, 14.0};
    double x[3] = {0.0, 0.0, 0.0};
    ASSERT_ERR(lu_csr_factor_solve(mat, b, x, 1e-12), SPARSE_OK);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], (double)(i + 1), 1e-10);
    sparse_free(mat);
}

static void test_matrix_market_roundtrip(void) {
    SparseMatrix *mat = make_tridiagonal();
    ASSERT_NOT_NULL(mat);
    if (!mat)
        return;
    const char *path = tf_tmp("s219_idx_width_functional.mtx");
    ASSERT_ERR(sparse_save_mm(mat, path), SPARSE_OK);

    SparseMatrix *loaded = NULL;
    ASSERT_ERR(sparse_load_mm(&loaded, path), SPARSE_OK);
    ASSERT_NOT_NULL(loaded);
    if (loaded) {
        check_tridiagonal(loaded);
        sparse_free(loaded);
    }
    ASSERT_EQ(remove(path), 0);
    sparse_free(mat);
}

static void test_csr_csc_roundtrip(void) {
    SparseMatrix *mat = make_tridiagonal();
    ASSERT_NOT_NULL(mat);
    if (!mat)
        return;
    const idx_t expected_ptr[4] = {0, 2, 5, 7};
    const idx_t expected_idx[7] = {0, 1, 0, 1, 2, 1, 2};
    const double expected_values[7] = {4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0};

    SparseCsr *csr = NULL;
    ASSERT_ERR(sparse_to_csr(mat, &csr), SPARSE_OK);
    ASSERT_NOT_NULL(csr);
    if (csr) {
        ASSERT_EQ(csr->nnz, 7);
        for (int i = 0; i < 4; i++)
            ASSERT_EQ(csr->row_ptr[i], expected_ptr[i]);
        for (int i = 0; i < 7; i++) {
            ASSERT_EQ(csr->col_idx[i], expected_idx[i]);
            ASSERT_NEAR(csr->values[i], expected_values[i], 0.0);
        }
        SparseMatrix *from_csr = NULL;
        ASSERT_ERR(sparse_from_csr(csr, &from_csr), SPARSE_OK);
        ASSERT_NOT_NULL(from_csr);
        if (from_csr) {
            check_tridiagonal(from_csr);
            sparse_free(from_csr);
        }
        sparse_csr_free(csr);
    }

    SparseCsc *csc = NULL;
    ASSERT_ERR(sparse_to_csc(mat, &csc), SPARSE_OK);
    ASSERT_NOT_NULL(csc);
    if (csc) {
        ASSERT_EQ(csc->nnz, 7);
        for (int i = 0; i < 4; i++)
            ASSERT_EQ(csc->col_ptr[i], expected_ptr[i]);
        for (int i = 0; i < 7; i++) {
            ASSERT_EQ(csc->row_idx[i], expected_idx[i]);
            ASSERT_NEAR(csc->values[i], expected_values[i], 0.0);
        }
        SparseMatrix *from_csc = NULL;
        ASSERT_ERR(sparse_from_csc(csc, &from_csc), SPARSE_OK);
        ASSERT_NOT_NULL(from_csc);
        if (from_csc) {
            check_tridiagonal(from_csc);
            sparse_free(from_csc);
        }
        sparse_csc_free(csc);
    }
    sparse_free(mat);
}

static void check_index_text(idx_t value, const char *expected) {
    char text[64];
    int length = snprintf(text, sizeof(text), "%" SPARSE_PRIDX, value);
    ASSERT_TRUE(length > 0 && (size_t)length < sizeof(text));
    if (length <= 0 || (size_t)length >= sizeof(text))
        return;
    ASSERT_TRUE(strcmp(text, expected) == 0);

    idx_t parsed = 0;
    ASSERT_EQ(sscanf(text, "%" SPARSE_SCNIDX, &parsed), 1);
    ASSERT_EQ(parsed, value);
}

static void test_index_text_boundary(void) {
#if SPARSE_IDX_BITS == 64
    check_index_text((idx_t)INT32_MAX + 1, "2147483648");
    check_index_text(IDX_MAX, "9223372036854775807");
#else
    check_index_text(IDX_MAX, "2147483647");
#endif
}

int main(void) {
    TEST_SUITE_BEGIN("Selected Index-Width Functional Lane");
    RUN_TEST(test_width_identity);
    RUN_TEST(test_matrix_insert_and_lu);
    RUN_TEST(test_matrix_market_roundtrip);
    RUN_TEST(test_csr_csc_roundtrip);
    RUN_TEST(test_index_text_boundary);
    TEST_SUITE_END();
}
