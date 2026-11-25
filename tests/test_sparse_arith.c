#include "arithmetic_contract_fixture.h"
#include "sparse_alloc_internal.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 227 arithmetic-contract fixtures
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s227_permutation_fixture_helpers(void) {
    const double physical[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const idx_t row_swap[2] = {1, 0};
    const idx_t col_cycle[3] = {2, 0, 1};
    const double row_expected[6] = {4.0, 5.0, 6.0, 1.0, 2.0, 3.0};
    const double col_expected[6] = {3.0, 1.0, 2.0, 6.0, 4.0, 5.0};
    const double both_expected[6] = {6.0, 4.0, 5.0, 3.0, 1.0, 2.0};
    const double *expected[3] = {row_expected, col_expected, both_expected};

    for (int state = 0; state < 3; state++) {
        SparseMatrix *matrix = s227_matrix_from_dense(2, 3, physical);
        double logical[6] = {0.0};
        ASSERT_NOT_NULL(matrix);
        if (!matrix)
            continue;
        if (state != 1)
            ASSERT_TRUE(s227_set_row_permutation(matrix, row_swap));
        if (state != 0)
            ASSERT_TRUE(s227_set_col_permutation(matrix, col_cycle));
        ASSERT_TRUE(s227_maps_are_valid(matrix));
        ASSERT_TRUE(s227_structure_is_valid(matrix));
        s227_copy_logical_dense(matrix, logical);
        for (idx_t i = 0; i < 6; i++)
            ASSERT_NEAR(logical[i], expected[state][i], 0.0);
        sparse_free(matrix);
    }
}

static void test_s227_independent_dense_oracles(void) {
    const double a_add[6] = {1.0, 0.0, -2.0, 3.0, 4.0, 0.5};
    const double b_add[6] = {0.0, 5.0, 2.0, -1.0, 0.0, 1.5};
    const double add_expected[6] = {2.0, -5.0, -6.0, 7.0, 8.0, -0.5};
    const double a_mul[6] = {1.0, 2.0, 0.0, -1.0, 3.0, 4.0};
    const double b_mul[6] = {2.0, 1.0, 0.0, 3.0, -2.0, 5.0};
    const double mul_expected[4] = {2.0, 7.0, -10.0, 28.0};
    double actual[6] = {0.0};

    s227_dense_add(2, 3, a_add, b_add, 2.0, -1.0, actual);
    for (idx_t i = 0; i < 6; i++)
        ASSERT_NEAR(actual[i], add_expected[i], 0.0);

    s227_dense_matmul(2, 3, 2, a_mul, b_mul, actual);
    for (idx_t i = 0; i < 4; i++)
        ASSERT_NEAR(actual[i], mul_expected[i], 0.0);
}

static void test_s227_snapshot_and_structure_helpers(void) {
    const double dense[4] = {2.0, 0.0, -1.0, 3.0};
    const idx_t swap[2] = {1, 0};
    SparseMatrix *matrix = s227_matrix_from_dense(2, 2, dense);
    s227_matrix_snapshot_t snapshot;

    ASSERT_NOT_NULL(matrix);
    if (!matrix)
        return;
    ASSERT_TRUE(s227_set_row_permutation(matrix, swap));
    ASSERT_TRUE(s227_set_col_permutation(matrix, swap));
    ASSERT_TRUE(s227_take_snapshot(matrix, &snapshot));
    ASSERT_TRUE(s227_matches_snapshot(matrix, &snapshot));
    ASSERT_ERR(sparse_insert(matrix, 0, 0, 9.0), SPARSE_OK);
    ASSERT_FALSE(s227_matches_snapshot(matrix, &snapshot));
    ASSERT_TRUE(s227_structure_is_valid(matrix));
    sparse_free(matrix);
}

static void s227_apply_permutation_kind(SparseMatrix *matrix, const idx_t *row_permutation,
                                        const idx_t *col_permutation, int permutation_kind) {
    if (permutation_kind != 1)
        ASSERT_TRUE(s227_set_row_permutation(matrix, row_permutation));
    if (permutation_kind != 0)
        ASSERT_TRUE(s227_set_col_permutation(matrix, col_permutation));
}

static void s227_assert_original_state_rejected(int api, int operand_mode, int permutation_kind) {
    const double a_dense[6] = {1.0, 0.0, 2.0, 0.0, 3.0, 4.0};
    const double b_dense[6] = {2.0, 1.0, 0.0, -1.0, 5.0, 0.0};
    const idx_t row_swap[2] = {1, 0};
    const idx_t col_cycle[3] = {2, 0, 1};
    const idx_t matmul_b_row_cycle[3] = {1, 2, 0};
    const idx_t matmul_b_col_swap[2] = {1, 0};
    SparseMatrix *A = s227_matrix_from_dense(2, 3, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(api == 2 ? 3 : 2, api == 2 ? 2 : 3, b_dense);
    SparseMatrix *C = NULL;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;
    sparse_err_t err;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    if (operand_mode != 1)
        s227_apply_permutation_kind(A, row_swap, col_cycle, permutation_kind);
    if (operand_mode != 0) {
        s227_apply_permutation_kind(B, api == 2 ? matmul_b_row_cycle : row_swap,
                                    api == 2 ? matmul_b_col_swap : col_cycle, permutation_kind);
    }
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));

    if (api == 0)
        err = sparse_add(A, B, 1.0, 1.0, &C);
    else if (api == 1)
        err = sparse_add_inplace(A, B, 1.0, 1.0);
    else
        err = sparse_matmul(A, B, &C);

    ASSERT_ERR(err, SPARSE_ERR_BADARG);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_add_original_state_rejection_contract(void) {
    for (int api = 0; api < 2; api++)
        for (int operand_mode = 0; operand_mode < 3; operand_mode++)
            for (int state = 0; state < 3; state++)
                s227_assert_original_state_rejected(api, operand_mode, state);
}

static void test_s227_matmul_original_state_rejection_contract(void) {
    for (int operand_mode = 0; operand_mode < 3; operand_mode++)
        for (int state = 0; state < 3; state++)
            s227_assert_original_state_rejected(2, operand_mode, state);
}

static void test_s227_matmul_inverse_only_state_rejected(void) {
    const double a_dense[6] = {1.0, 0.0, 2.0, 0.0, 3.0, 4.0};
    const double b_dense[6] = {2.0, 1.0, 0.0, -1.0, 5.0, 0.0};
    SparseMatrix *A = s227_matrix_from_dense(2, 3, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(3, 2, b_dense);
    SparseMatrix *C = NULL;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;

    A->inv_row_perm[0] = 1;
    A->inv_row_perm[1] = 0;
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_BADARG);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

    A->inv_row_perm[0] = 0;
    A->inv_row_perm[1] = 1;
    B->inv_col_perm[0] = 1;
    B->inv_col_perm[1] = 0;
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_BADARG);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_matmul_factored_state_rejected(void) {
    const double a_dense[4] = {2.0, 0.0, 1.0, 3.0};
    const double b_dense[4] = {4.0, 5.0, 0.0, 6.0};
    SparseMatrix *A = s227_matrix_from_dense(2, 2, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(2, 2, b_dense);
    SparseMatrix *C = NULL;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;

    ASSERT_ERR(sparse_mark_factored(A), SPARSE_OK);
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_BADARG);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

    A->factored = 0;
    ASSERT_ERR(sparse_mark_factored(B), SPARSE_OK);
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_BADARG);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_matmul_validation_precedence_and_output(void) {
    const idx_t swap[2] = {1, 0};
    SparseMatrix *A = sparse_create(2, 2);
    SparseMatrix *B = sparse_create(3, 2);
    SparseMatrix *owner = sparse_create(1, 1);
    SparseMatrix *C = owner;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(owner);
    if (!A || !B || !owner)
        goto cleanup;
    ASSERT_TRUE(s227_set_row_permutation(A, swap));
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_SHAPE);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

    C = owner;
    ASSERT_ERR(sparse_matmul(NULL, B, &C), SPARSE_ERR_NULL);
    ASSERT_NULL(C);
    ASSERT_EQ(sparse_rows(owner), 1);
    ASSERT_EQ(sparse_cols(owner), 1);

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(owner);
}

static void test_s227_matmul_state_rejection_precedes_allocation(void) {
    const double dense[4] = {1.0, 0.0, 0.0, 2.0};
    const idx_t swap[2] = {1, 0};
    SparseMatrix *A = s227_matrix_from_dense(2, 2, dense);
    SparseMatrix *B = s227_matrix_from_dense(2, 2, dense);
    SparseMatrix *C = NULL;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    ASSERT_TRUE(s227_set_col_permutation(B, swap));

    sparse_alloc_test_fail_after(0);
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_BADARG);
    ASSERT_NULL(C);

    B->col_perm[0] = 0;
    B->col_perm[1] = 1;
    B->inv_col_perm[0] = 0;
    B->inv_col_perm[1] = 1;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_ALLOC);
    ASSERT_NULL(C);

    sparse_alloc_test_reset();
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    if (C) {
        ASSERT_EQ(sparse_nnz(C), 2);
        ASSERT_NEAR(sparse_get(C, 0, 0), 1.0, 0.0);
        ASSERT_NEAR(sparse_get(C, 1, 1), 4.0, 0.0);
        ASSERT_TRUE(s227_structure_is_valid(C));
    }

cleanup:
    sparse_alloc_test_reset();
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_add_inverse_only_state_rejected(void) {
    const double a_dense[4] = {1.0, 2.0, 0.0, 3.0};
    const double b_dense[4] = {4.0, 0.0, 5.0, 6.0};
    SparseMatrix *A = s227_matrix_from_dense(2, 2, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(2, 2, b_dense);
    SparseMatrix *C = NULL;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;

    A->inv_row_perm[0] = 1;
    A->inv_row_perm[1] = 0;
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_BADARG);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

    A->inv_row_perm[0] = 0;
    A->inv_row_perm[1] = 1;
    B->inv_col_perm[0] = 1;
    B->inv_col_perm[1] = 0;
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_add_inplace(A, B, 1.0, 1.0), SPARSE_ERR_BADARG);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_add_factored_state_rejected(void) {
    const double a_dense[4] = {2.0, 0.0, 1.0, 3.0};
    const double b_dense[4] = {4.0, 5.0, 0.0, 6.0};
    SparseMatrix *A = s227_matrix_from_dense(2, 2, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(2, 2, b_dense);
    SparseMatrix *C = NULL;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;

    ASSERT_ERR(sparse_mark_factored(A), SPARSE_OK);
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_BADARG);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

    A->factored = 0;
    ASSERT_ERR(sparse_mark_factored(B), SPARSE_OK);
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_add_inplace(A, B, 1.0, 1.0), SPARSE_ERR_BADARG);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_add_shape_precedes_state_validation(void) {
    const idx_t swap[2] = {1, 0};
    SparseMatrix *A = sparse_create(2, 2);
    SparseMatrix *B = sparse_create(3, 3);
    SparseMatrix *C = NULL;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    ASSERT_TRUE(s227_set_row_permutation(A, swap));
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_SHAPE);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));
    ASSERT_ERR(sparse_add_inplace(A, B, 1.0, 1.0), SPARSE_ERR_SHAPE);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_add_inplace_self_alias_rejected_unchanged(void) {
    const double dense[4] = {1.0, 2.0, 0.0, 3.0};
    SparseMatrix *A = s227_matrix_from_dense(2, 2, dense);
    s227_matrix_snapshot_t before;
    double norm = 0.0;

    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    ASSERT_ERR(sparse_norminf(A, &norm), SPARSE_OK);
    ASSERT_NEAR(norm, 3.0, 0.0);
    ASSERT_TRUE(s227_take_snapshot(A, &before));
    ASSERT_ERR(sparse_add_inplace(A, A, 2.0, 3.0), SPARSE_ERR_BADARG);
    ASSERT_TRUE(s227_matches_snapshot(A, &before));
    sparse_free(A);
}

static void test_s227_add_state_rejection_precedes_allocation(void) {
    const double dense[4] = {1.0, 0.0, 0.0, 2.0};
    const idx_t swap[2] = {1, 0};
    SparseMatrix *A = s227_matrix_from_dense(2, 2, dense);
    SparseMatrix *B = s227_matrix_from_dense(2, 2, dense);
    SparseMatrix *C = NULL;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    ASSERT_TRUE(s227_set_row_permutation(A, swap));

    sparse_alloc_test_fail_after(0);
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_BADARG);
    ASSERT_NULL(C);

    A->row_perm[0] = 0;
    A->row_perm[1] = 1;
    A->inv_row_perm[0] = 0;
    A->inv_row_perm[1] = 1;
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_ALLOC);
    ASSERT_NULL(C);

    sparse_alloc_test_reset();
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    if (C) {
        ASSERT_EQ(sparse_nnz(C), 2);
        ASSERT_NEAR(sparse_get(C, 0, 0), 2.0, 0.0);
        ASSERT_NEAR(sparse_get(C, 1, 1), 4.0, 0.0);
    }

cleanup:
    sparse_alloc_test_reset();
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_sorted_add_builder_representative_rows(void) {
    const double a_dense[20] = {
        1.0, 0.0, 3.0,  0.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0,
        4.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    const double b_dense[20] = {
        0.0,  2.0, -3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 3.0, 0.0,
        -8.0, 0.0, 4.0,  0.0, 0.0, 0.0, 0.0, 7.0, 0.0, -1.0,
    };
    const double expected[20] = {
        2.0, 2.0, 3.0, 0.0, 11.0, 1.0, 4.0, 0.0, 3.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 7.0, 0.0, -1.0,
    };
    const idx_t expected_row_nnz[4] = {4, 3, 0, 2};
    SparseMatrix *A = s227_matrix_from_dense(4, 5, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(4, 5, b_dense);
    SparseMatrix *C = NULL;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_add(A, B, 2.0, 1.0, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));
    if (!C)
        goto cleanup;

    ASSERT_EQ(sparse_nnz(C), 9);
    ASSERT_TRUE(s227_structure_is_valid(C));
    ASSERT_NEAR(C->cached_norm, -1.0, 0.0);
    ASSERT_NEAR(C->factor_norm, -1.0, 0.0);
    ASSERT_EQ(C->factored, 0);
    for (idx_t row = 0; row < C->rows; row++) {
        idx_t row_nnz = 0;
        idx_t previous_col = -1;
        for (const Node *node = C->row_headers[row]; node; node = node->right) {
            ASSERT_TRUE(node->col > previous_col);
            previous_col = node->col;
            row_nnz++;
        }
        ASSERT_EQ(row_nnz, expected_row_nnz[row]);
        for (idx_t col = 0; col < C->cols; col++)
            ASSERT_NEAR(sparse_get(C, row, col), expected[row * C->cols + col], 0.0);
    }

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_sorted_add_builder_allocation_cleanup(void) {
    const double a_dense[4] = {1.0, 0.0, 0.0, 2.0};
    const double b_dense[4] = {0.0, 3.0, 4.0, 0.0};
    SparseMatrix *A = s227_matrix_from_dense(2, 2, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(2, 2, b_dense);
    SparseMatrix *C = NULL;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));

    sparse_alloc_test_fail_after(7);
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_ALLOC);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

    sparse_alloc_test_fail_after(8);
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_ALLOC);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

    sparse_alloc_test_reset();
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    if (C) {
        ASSERT_EQ(sparse_nnz(C), 4);
        ASSERT_TRUE(s227_structure_is_valid(C));
    }

cleanup:
    sparse_alloc_test_reset();
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

enum {
    S227_ALLOCATION_SWEEP_ROWS = NODES_PER_SLAB + 1,
    S227_ALLOCATION_SWEEP_COLS = 1,
    S227_ADD_LARGE_RESULT_ALLOCATIONS = 10,
};

static double s227_allocation_sweep_value(idx_t row, idx_t col) {
    double magnitude = (double)(row + col + 1);
    return (row + col) % 2 == 0 ? magnitude : -magnitude;
}

static int s227_allocation_sweep_input_is_unchanged(const SparseMatrix *matrix, int parity) {
    idx_t expected_nnz = 0;
    const Node *column_node = NULL;
    if (!matrix || matrix->rows != S227_ALLOCATION_SWEEP_ROWS ||
        matrix->cols != S227_ALLOCATION_SWEEP_COLS || matrix->col_perm[0] != 0 ||
        matrix->inv_col_perm[0] != 0)
        return 0;

    column_node = matrix->col_headers[0];
    for (idx_t row = 0; row < matrix->rows; row++) {
        const Node *row_node = matrix->row_headers[row];
        if (matrix->row_perm[row] != row || matrix->inv_row_perm[row] != row)
            return 0;
        if ((int)(row % 2) != parity) {
            if (row_node)
                return 0;
            continue;
        }
        if (!row_node || row_node != column_node || row_node->row != row || row_node->col != 0 ||
            row_node->right || row_node->value != s227_allocation_sweep_value(row, 0))
            return 0;
        column_node = column_node->down;
        expected_nnz++;
    }
    return !column_node && matrix->nnz == expected_nnz;
}

static int s227_allocation_sweep_output_is_valid(const SparseMatrix *matrix) {
    const Node *column_node = NULL;
    if (!matrix || matrix->rows != S227_ALLOCATION_SWEEP_ROWS ||
        matrix->cols != S227_ALLOCATION_SWEEP_COLS || matrix->nnz != S227_ALLOCATION_SWEEP_ROWS ||
        matrix->pool.num_slabs != 2 || matrix->col_perm[0] != 0 || matrix->inv_col_perm[0] != 0)
        return 0;

    column_node = matrix->col_headers[0];
    for (idx_t row = 0; row < matrix->rows; row++) {
        const Node *row_node = matrix->row_headers[row];
        if (matrix->row_perm[row] != row || matrix->inv_row_perm[row] != row || !row_node ||
            row_node != column_node || row_node->row != row || row_node->col != 0 ||
            row_node->right || row_node->value != s227_allocation_sweep_value(row, 0))
            return 0;
        column_node = column_node->down;
    }
    return !column_node;
}

static void test_s227_sorted_add_builder_exhaustive_allocation_sweep(void) {
    SparseMatrix *A = sparse_create(S227_ALLOCATION_SWEEP_ROWS, S227_ALLOCATION_SWEEP_COLS);
    SparseMatrix *B = sparse_create(S227_ALLOCATION_SWEEP_ROWS, S227_ALLOCATION_SWEEP_COLS);
    SparseMatrix *C = NULL;

    ASSERT_TRUE(S227_ALLOCATION_SWEEP_ROWS > NODES_PER_SLAB);
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;

    for (idx_t row = 0; row < S227_ALLOCATION_SWEEP_ROWS; row++) {
        SparseMatrix *target = row % 2 == 0 ? A : B;
        ASSERT_ERR(sparse_insert(target, row, 0, s227_allocation_sweep_value(row, 0)), SPARSE_OK);
    }
    ASSERT_TRUE(s227_allocation_sweep_input_is_unchanged(A, 0));
    ASSERT_TRUE(s227_allocation_sweep_input_is_unchanged(B, 1));

    for (long fail_after = 0; fail_after <= S227_ADD_LARGE_RESULT_ALLOCATIONS + 1; fail_after++) {
        sparse_alloc_test_fail_after(fail_after);
        sparse_err_t err = sparse_add(A, B, 1.0, 1.0, &C);
        if (fail_after < S227_ADD_LARGE_RESULT_ALLOCATIONS) {
            ASSERT_ERR(err, SPARSE_ERR_ALLOC);
            ASSERT_NULL(C);
            ASSERT_TRUE(s227_allocation_sweep_input_is_unchanged(A, 0));
            ASSERT_TRUE(s227_allocation_sweep_input_is_unchanged(B, 1));

            sparse_alloc_test_reset();
            ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_OK);
            ASSERT_TRUE(s227_allocation_sweep_output_is_valid(C));
            sparse_free(C);
            C = NULL;
        } else {
            ASSERT_ERR(err, SPARSE_OK);
            ASSERT_TRUE(s227_allocation_sweep_output_is_valid(C));
            sparse_free(C);
            C = NULL;
        }
    }

cleanup:
    sparse_alloc_test_reset();
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_add_failure_output_ownership_matrix(void) {
    const double a_dense[6] = {1.0, 0.0, 2.0, 0.0, 3.0, 4.0};
    const double b_dense[6] = {0.0, 5.0, -2.0, 6.0, 0.0, 7.0};
    const idx_t row_swap[2] = {1, 0};
    SparseMatrix *A = s227_matrix_from_dense(2, 3, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(2, 3, b_dense);
    SparseMatrix *wrong_shape = sparse_create(3, 3);
    SparseMatrix *stale_owner = sparse_create(1, 1);
    SparseMatrix *C = stale_owner;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;
    s227_matrix_snapshot_t wrong_before;
    s227_matrix_snapshot_t stale_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(wrong_shape);
    ASSERT_NOT_NULL(stale_owner);
    if (!A || !B || !wrong_shape || !stale_owner)
        goto cleanup;
    ASSERT_ERR(sparse_insert(stale_owner, 0, 0, 19.0), SPARSE_OK);
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_TRUE(s227_take_snapshot(wrong_shape, &wrong_before));
    ASSERT_TRUE(s227_take_snapshot(stale_owner, &stale_before));

    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, NULL), SPARSE_ERR_NULL);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

    ASSERT_ERR(sparse_add(NULL, B, 1.0, 1.0, &C), SPARSE_ERR_NULL);
    ASSERT_TRUE(C == stale_owner);
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));
    ASSERT_TRUE(s227_matches_snapshot(stale_owner, &stale_before));
    ASSERT_ERR(sparse_add(A, NULL, 1.0, 1.0, &C), SPARSE_ERR_NULL);
    ASSERT_TRUE(C == stale_owner);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(stale_owner, &stale_before));

    ASSERT_ERR(sparse_add(A, wrong_shape, 1.0, 1.0, &C), SPARSE_ERR_SHAPE);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(wrong_shape, &wrong_before));
    ASSERT_TRUE(s227_matches_snapshot(stale_owner, &stale_before));

    C = stale_owner;
    sparse_alloc_test_fail_after(0);
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_ALLOC);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));
    ASSERT_TRUE(s227_matches_snapshot(stale_owner, &stale_before));
    sparse_alloc_test_reset();

    ASSERT_TRUE(s227_set_row_permutation(A, row_swap));
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    C = stale_owner;
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_BADARG);
    ASSERT_NULL(C);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));
    ASSERT_TRUE(s227_matches_snapshot(stale_owner, &stale_before));

cleanup:
    sparse_alloc_test_reset();
    sparse_free(A);
    sparse_free(B);
    sparse_free(wrong_shape);
    sparse_free(stale_owner);
}

static void test_s227_add_cancellation_allocation_sweep(void) {
    double a_dense[36];
    double b_dense[36];
    SparseMatrix *A = NULL;
    SparseMatrix *B = NULL;
    SparseMatrix *C = NULL;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;

    for (idx_t i = 0; i < 36; i++) {
        a_dense[i] = (double)(i + 1);
        b_dense[i] = -a_dense[i];
    }
    A = s227_matrix_from_dense(6, 6, a_dense);
    B = s227_matrix_from_dense(6, 6, b_dense);
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));

    for (long fail_after = 0; fail_after <= 9; fail_after++) {
        sparse_alloc_test_fail_after(fail_after);
        sparse_err_t err = sparse_add(A, B, 1.0, 1.0, &C);
        if (fail_after < 8) {
            ASSERT_ERR(err, SPARSE_ERR_ALLOC);
            ASSERT_NULL(C);
            ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
            ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

            sparse_alloc_test_reset();
            ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_OK);
        } else {
            ASSERT_ERR(err, SPARSE_OK);
        }
        ASSERT_NOT_NULL(C);
        if (C) {
            ASSERT_EQ(C->nnz, 0);
            ASSERT_EQ(C->pool.num_slabs, 0);
            ASSERT_TRUE(s227_structure_is_valid(C));
        }
        sparse_free(C);
        C = NULL;
    }

cleanup:
    sparse_alloc_test_reset();
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_s227_add_output_variable_alias_and_repeated_calls(void) {
    const double a_dense[4] = {1.0, 2.0, 0.0, 3.0};
    const double b_dense[4] = {4.0, 0.0, 5.0, -1.0};
    SparseMatrix *A = s227_matrix_from_dense(2, 2, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(2, 2, b_dense);
    SparseMatrix *slot = NULL;
    SparseMatrix *first_result = NULL;
    SparseMatrix *second_result = NULL;
    SparseMatrix *alias_result = NULL;
    SparseMatrix *same_input_result = NULL;
    s227_matrix_snapshot_t a_before;
    s227_matrix_snapshot_t b_before;
    s227_matrix_snapshot_t first_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    ASSERT_TRUE(s227_take_snapshot(A, &a_before));
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));

    slot = A;
    ASSERT_ERR(sparse_add(slot, B, 2.0, -1.0, &slot), SPARSE_OK);
    first_result = slot;
    ASSERT_NOT_NULL(first_result);
    ASSERT_TRUE(first_result != A);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));
    ASSERT_NEAR(sparse_get(first_result, 0, 0), -2.0, 0.0);
    ASSERT_NEAR(sparse_get(first_result, 0, 1), 4.0, 0.0);
    ASSERT_NEAR(sparse_get(first_result, 1, 0), -5.0, 0.0);
    ASSERT_NEAR(sparse_get(first_result, 1, 1), 7.0, 0.0);
    ASSERT_TRUE(s227_take_snapshot(first_result, &first_before));

    slot = first_result;
    ASSERT_ERR(sparse_add(A, B, -1.0, 2.0, &slot), SPARSE_OK);
    second_result = slot;
    ASSERT_NOT_NULL(second_result);
    ASSERT_TRUE(second_result != first_result);
    ASSERT_TRUE(s227_matches_snapshot(first_result, &first_before));
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

    slot = A;
    sparse_alloc_test_fail_after(0);
    ASSERT_ERR(sparse_add(slot, B, 1.0, 1.0, &slot), SPARSE_ERR_ALLOC);
    ASSERT_NULL(slot);
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));
    sparse_alloc_test_reset();
    slot = A;
    ASSERT_ERR(sparse_add(slot, B, 1.0, 1.0, &slot), SPARSE_OK);
    alias_result = slot;
    ASSERT_TRUE(s227_structure_is_valid(alias_result));

    slot = A;
    ASSERT_ERR(sparse_add(slot, slot, 2.0, -1.0, &slot), SPARSE_OK);
    same_input_result = slot;
    ASSERT_TRUE(s227_structure_is_valid(same_input_result));
    ASSERT_TRUE(s227_matches_snapshot(A, &a_before));
    for (idx_t row = 0; row < 2; row++)
        for (idx_t col = 0; col < 2; col++)
            ASSERT_NEAR(sparse_get(same_input_result, row, col), a_dense[row * 2 + col], 0.0);

    slot = B;
    ASSERT_ERR(sparse_add(NULL, slot, 1.0, 1.0, &slot), SPARSE_ERR_NULL);
    ASSERT_TRUE(slot == B);
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

cleanup:
    sparse_alloc_test_reset();
    sparse_free(A);
    sparse_free(B);
    sparse_free(first_result);
    sparse_free(second_result);
    sparse_free(alias_result);
    sparse_free(same_input_result);
}

static void test_s227_add_inplace_adversarial_success_contract(void) {
    const double a_dense[20] = {1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 4.0, 0.0, 5.0, 7.0, 0.0, 0.0, 8.0, 0.0};
    const double b_dense[20] = {2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 8.0, 6.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0};
    double expected[20];
    SparseMatrix *A = s227_matrix_from_dense(4, 5, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(4, 5, b_dense);
    SparseMatrix *copy = NULL;
    s227_matrix_snapshot_t b_before;
    idx_t expected_nnz = 0;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    s227_dense_add(4, 5, a_dense, b_dense, 2.0, -1.0, expected);
    for (idx_t i = 0; i < 20; i++) {
        if (fabs(expected[i]) < 1e-15)
            expected[i] = 0.0;
        else
            expected_nnz++;
    }
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_add_inplace(A, B, 2.0, -1.0), SPARSE_OK);
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));
    ASSERT_TRUE(s227_structure_is_valid(A));
    ASSERT_TRUE(s227_maps_are_valid(A));
    ASSERT_EQ(sparse_nnz(A), expected_nnz);
    ASSERT_NULL(A->row_headers[1]);
    ASSERT_EQ(A->factored, 0);
    ASSERT_NEAR(A->factor_norm, -1.0, 0.0);
    ASSERT_NULL(A->factor_state);
    for (idx_t row = 0; row < 4; row++)
        for (idx_t col = 0; col < 5; col++)
            ASSERT_NEAR(sparse_get(A, row, col), expected[row * 5 + col], 0.0);

    copy = sparse_copy(A);
    ASSERT_NOT_NULL(copy);
    ASSERT_TRUE(s227_structure_is_valid(copy));
    for (idx_t row = 0; row < 4; row++)
        for (idx_t col = 0; col < 5; col++)
            ASSERT_NEAR(sparse_get(copy, row, col), expected[row * 5 + col], 0.0);

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(copy);
}

static void test_s227_add_inplace_zero_scale_reuses_removed_nodes(void) {
    const double a_dense[9] = {1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0};
    const double b_dense[9] = {0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0, 0.0};
    SparseMatrix *A = s227_matrix_from_dense(3, 3, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(3, 3, b_dense);
    s227_matrix_snapshot_t b_before;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    ASSERT_TRUE(s227_take_snapshot(B, &b_before));
    ASSERT_ERR(sparse_add_inplace(A, B, 0.0, -2.0), SPARSE_OK);
    ASSERT_TRUE(s227_matches_snapshot(B, &b_before));
    ASSERT_TRUE(s227_structure_is_valid(A));
    ASSERT_EQ(sparse_nnz(A), 4);
    for (idx_t row = 0; row < 3; row++) {
        for (idx_t col = 0; col < 3; col++)
            ASSERT_NEAR(sparse_get(A, row, col), -2.0 * b_dense[row * 3 + col], 0.0);
    }

cleanup:
    sparse_free(A);
    sparse_free(B);
}

static void test_s227_add_inplace_allocation_failure_partial_contract(void) {
    const double a_dense[4] = {1.0, 0.0, 0.0, 3.0};
    const double b_dense[4] = {4.0, 5.0, 0.0, 0.0};

    for (long fail_after = 0; fail_after <= 2; fail_after++) {
        SparseMatrix *A = s227_matrix_from_dense(2, 2, a_dense);
        SparseMatrix *B = s227_matrix_from_dense(2, 2, b_dense);
        SparseMatrix *copy = NULL;
        SparseMatrix *retry = NULL;
        SparseMatrix *zero = sparse_create(2, 2);
        SparseMatrix *out = NULL;
        s227_matrix_snapshot_t b_before;
        Node *a00 = NULL;
        Node *a11 = NULL;
        NodeSlab *initial_slab = NULL;
        double cached_norm = 0.0;

        ASSERT_NOT_NULL(A);
        ASSERT_NOT_NULL(B);
        ASSERT_NOT_NULL(zero);
        if (!A || !B || !zero)
            goto iteration_cleanup;
        ASSERT_ERR(sparse_norminf(A, &cached_norm), SPARSE_OK);
        ASSERT_NEAR(cached_norm, 3.0, 0.0);
        ASSERT_TRUE(s227_take_snapshot(B, &b_before));
        retry = sparse_copy(A);
        ASSERT_NOT_NULL(retry);
        if (!retry)
            goto iteration_cleanup;
        a00 = A->row_headers[0];
        a11 = A->row_headers[1];
        initial_slab = A->pool.current;
        ASSERT_NOT_NULL(initial_slab);
        if (!initial_slab)
            goto iteration_cleanup;

        /* Exhaust the current slab without changing the active row/column
         * topology so the B-only entry reaches the injectable slab boundary. */
        initial_slab->used = NODES_PER_SLAB;
        ASSERT_NOT_NULL(retry->pool.current);
        if (!retry->pool.current)
            goto iteration_cleanup;
        retry->pool.current->used = NODES_PER_SLAB;
        sparse_alloc_test_fail_after(fail_after);
        sparse_err_t err = sparse_add_inplace(A, B, 2.0, 3.0);
        if (fail_after == 0) {
            ASSERT_ERR(err, SPARSE_ERR_ALLOC);
            ASSERT_EQ(sparse_nnz(A), 2);
            ASSERT_TRUE(A->row_headers[0] == a00);
            ASSERT_TRUE(A->row_headers[1] == a11);
            ASSERT_TRUE(A->pool.head == initial_slab);
            ASSERT_TRUE(A->pool.current == initial_slab);
            ASSERT_EQ(A->pool.num_slabs, 1);
            ASSERT_NEAR(sparse_get(A, 0, 0), 14.0, 0.0);
            ASSERT_NEAR(sparse_get(A, 0, 1), 0.0, 0.0);
            ASSERT_NEAR(sparse_get(A, 1, 1), 6.0, 0.0);

            sparse_alloc_test_reset();
            ASSERT_ERR(sparse_add_inplace(retry, B, 2.0, 3.0), SPARSE_OK);
            ASSERT_TRUE(s227_structure_is_valid(retry));
            ASSERT_NEAR(sparse_get(retry, 0, 0), 14.0, 0.0);
            ASSERT_NEAR(sparse_get(retry, 0, 1), 15.0, 0.0);
            ASSERT_NEAR(sparse_get(retry, 1, 1), 6.0, 0.0);
        } else {
            ASSERT_ERR(err, SPARSE_OK);
            ASSERT_EQ(sparse_nnz(A), 3);
            ASSERT_TRUE(A->row_headers[0] == a00);
            ASSERT_TRUE(A->row_headers[1] == a11);
            ASSERT_TRUE(A->pool.head == initial_slab);
            ASSERT_TRUE(A->pool.current != initial_slab);
            ASSERT_EQ(A->pool.num_slabs, 2);
            ASSERT_NEAR(sparse_get(A, 0, 0), 14.0, 0.0);
            ASSERT_NEAR(sparse_get(A, 0, 1), 15.0, 0.0);
            ASSERT_NEAR(sparse_get(A, 1, 1), 6.0, 0.0);
        }
        ASSERT_TRUE(s227_matches_snapshot(B, &b_before));
        ASSERT_TRUE(s227_structure_is_valid(A));
        ASSERT_TRUE(s227_maps_are_valid(A));
        ASSERT_NEAR(A->cached_norm, -1.0, 0.0);
        ASSERT_NEAR(A->factor_norm, -1.0, 0.0);
        ASSERT_EQ(A->factored, 0);
        ASSERT_NULL(A->factor_state);
        ASSERT_NULL(A->reorder_perm);

        sparse_alloc_test_reset();
        copy = sparse_copy(A);
        ASSERT_NOT_NULL(copy);
        ASSERT_TRUE(s227_structure_is_valid(copy));
        ASSERT_ERR(sparse_add(A, zero, 1.0, 0.0, &out), SPARSE_OK);
        ASSERT_TRUE(s227_structure_is_valid(out));
        ASSERT_EQ(sparse_nnz(out), sparse_nnz(A));
        ASSERT_ERR(sparse_add_inplace(A, zero, 1.0, 1.0), SPARSE_OK);
        ASSERT_TRUE(s227_structure_is_valid(A));
        ASSERT_TRUE(s227_matches_snapshot(B, &b_before));

    iteration_cleanup:
        sparse_alloc_test_reset();
        sparse_free(A);
        sparse_free(B);
        sparse_free(copy);
        sparse_free(retry);
        sparse_free(zero);
        sparse_free(out);
    }
}

static void s227_assert_sorted_add_case(idx_t rows, idx_t cols, const double *a_dense,
                                        const double *b_dense, double alpha, double beta,
                                        const double *expected, idx_t expected_nnz) {
    SparseMatrix *A = s227_matrix_from_dense(rows, cols, a_dense);
    SparseMatrix *B = s227_matrix_from_dense(rows, cols, b_dense);
    SparseMatrix *C = NULL;
    SparseMatrix *copy = NULL;

    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    if (!A || !B)
        goto cleanup;
    ASSERT_ERR(sparse_add(A, B, alpha, beta, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    if (!C)
        goto cleanup;

    ASSERT_EQ(sparse_nnz(C), expected_nnz);
    ASSERT_TRUE(s227_structure_is_valid(C));
    for (idx_t row = 0; row < rows; row++) {
        for (idx_t col = 0; col < cols; col++) {
            ASSERT_NEAR(sparse_get(C, row, col), expected[row * cols + col], 0.0);
            ASSERT_NEAR(sparse_get(A, row, col), a_dense[row * cols + col], 0.0);
            ASSERT_NEAR(sparse_get(B, row, col), b_dense[row * cols + col], 0.0);
        }
    }

    copy = sparse_copy(C);
    ASSERT_NOT_NULL(copy);
    if (copy) {
        ASSERT_EQ(sparse_nnz(copy), expected_nnz);
        ASSERT_TRUE(s227_structure_is_valid(copy));
        for (idx_t row = 0; row < rows; row++) {
            for (idx_t col = 0; col < cols; col++)
                ASSERT_NEAR(sparse_get(copy, row, col), expected[row * cols + col], 0.0);
        }
    }

cleanup:
    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
    sparse_free(copy);
}

static void test_s227_sorted_add_builder_edge_corpus(void) {
    {
        const double zeros[12] = {0.0};
        s227_assert_sorted_add_case(3, 4, zeros, zeros, 1.0, 1.0, zeros, 0);
    }
    {
        const double a[14] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        const double b[14] = {0.0, 5.0, 0.0, 6.0, 0.0,  7.0, 0.0,
                              8.0, 0.0, 9.0, 0.0, 10.0, 0.0, 11.0};
        const double expected[14] = {1.0, 5.0, 2.0, 6.0, 3.0,  7.0, 4.0,
                                     8.0, 0.0, 9.0, 0.0, 10.0, 0.0, 11.0};
        s227_assert_sorted_add_case(2, 7, a, b, 1.0, 1.0, expected, 11);
    }
    {
        const double a[15] = {0.0, 2.0, 0.0, -4.0, 0.0, 0.0, 3.0, 0.0,
                              5.0, 0.0, 0.0, 6.0,  0.0, 7.0, 0.0};
        const double b[15] = {0.0,  -2.0, 0.0, 4.0,  0.0, 0.0,  -3.0, 0.0,
                              -5.0, 0.0,  0.0, -6.0, 0.0, -7.0, 0.0};
        const double zeros[15] = {0.0};
        s227_assert_sorted_add_case(3, 5, a, b, 1.0, 1.0, zeros, 0);
        s227_assert_sorted_add_case(3, 5, a, b, 0.0, 0.0, zeros, 0);
        s227_assert_sorted_add_case(3, 5, a, b, 0.0, -1.0, a, 6);
    }
    {
        double a[192] = {0.0};
        double b[192] = {0.0};
        double expected[192] = {0.0};
        idx_t expected_nnz = 0;
        for (idx_t col = 0; col < 64; col++) {
            a[col] = (double)(col + 1);
            b[col] = col % 3 == 0 ? -a[col] : (double)(2 * col + 1);
            expected[col] = a[col] + b[col];
            if (expected[col] != 0.0)
                expected_nnz++;
        }
        a[2 * 64 + 63] = 9.0;
        b[2 * 64] = -3.0;
        expected[2 * 64] = -3.0;
        expected[2 * 64 + 63] = 9.0;
        expected_nnz += 2;
        s227_assert_sorted_add_case(3, 64, a, b, 1.0, 1.0, expected, expected_nnz);
    }
}

static uint32_t s227_random_next(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static void test_s227_sorted_add_builder_randomized_oracle(void) {
    uint32_t state = UINT32_C(0x2270add7);
    double a[36];
    double b[36];
    double expected[36];

    for (int trial = 0; trial < 96; trial++) {
        idx_t rows = (idx_t)(s227_random_next(&state) % 6U) + 1;
        idx_t cols = (idx_t)(s227_random_next(&state) % 6U) + 1;
        double alpha = (double)((int)(s227_random_next(&state) % 5U) - 2);
        double beta = (double)((int)(s227_random_next(&state) % 5U) - 2);
        idx_t expected_nnz = 0;

        for (idx_t i = 0; i < rows * cols; i++) {
            uint32_t a_word = s227_random_next(&state);
            uint32_t b_word = s227_random_next(&state);
            a[i] = a_word % 4U == 0U ? 0.0 : (double)((int)(a_word % 13U) - 6);
            b[i] = b_word % 4U == 0U ? 0.0 : (double)((int)(b_word % 13U) - 6);
            expected[i] = alpha * a[i] + beta * b[i];
            if (fabs(expected[i]) < 1e-15)
                expected[i] = 0.0;
            else
                expected_nnz++;
        }

        s227_assert_sorted_add_case(rows, cols, a, b, alpha, beta, expected, expected_nnz);
    }
}

static void test_s227_add_randomized_wide_rectangular_oracle(void) {
    uint32_t state = UINT32_C(0x22709add);
    double a[8 * 97];
    double b[8 * 97];
    double expected[8 * 97];

    for (int trial = 0; trial < 48; trial++) {
        idx_t rows = (idx_t)(s227_random_next(&state) % 7U) + 2;
        idx_t cols = (idx_t)(s227_random_next(&state) % 65U) + 33;
        double alpha = (double)((int)(s227_random_next(&state) % 7U) - 3);
        double beta = (double)((int)(s227_random_next(&state) % 7U) - 3);
        idx_t expected_nnz = 0;

        for (idx_t i = 0; i < rows * cols; i++) {
            uint32_t a_word = s227_random_next(&state);
            uint32_t b_word = s227_random_next(&state);
            a[i] = a_word % 5U == 0U ? (double)((int)(a_word % 17U) - 8) : 0.0;
            b[i] = b_word % 5U == 0U ? (double)((int)(b_word % 17U) - 8) : 0.0;
            expected[i] = alpha * a[i] + beta * b[i];
            if (fabs(expected[i]) < 1e-15)
                expected[i] = 0.0;
            else
                expected_nnz++;
        }
        s227_assert_sorted_add_case(rows, cols, a, b, alpha, beta, expected, expected_nnz);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_scale tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_scale_identity_by_3(void) {
    SparseMatrix *A = sparse_create(4, 4);
    for (idx_t i = 0; i < 4; i++)
        sparse_insert(A, i, i, 1.0);

    ASSERT_ERR(sparse_scale(A, 3.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 4);
    for (idx_t i = 0; i < 4; i++)
        ASSERT_NEAR(sparse_get_phys(A, i, i), 3.0, 0.0);

    sparse_free(A);
}

static void test_scale_by_1(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 5.0);
    sparse_insert(A, 1, 2, 7.0);
    sparse_insert(A, 2, 1, -3.0);

    ASSERT_ERR(sparse_scale(A, 1.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 3);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 2), 7.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 2, 1), -3.0, 0.0);

    sparse_free(A);
}

static void test_scale_by_0(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);
    ASSERT_EQ(sparse_nnz(A), 3);

    ASSERT_ERR(sparse_scale(A, 0.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 0);

    /* All entries should be gone */
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 3; j++)
            ASSERT_NEAR(sparse_get_phys(A, i, j), 0.0, 0.0);

    sparse_free(A);
}

static void test_scale_by_negative_1(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, -2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 1.0);

    ASSERT_ERR(sparse_scale(A, -1.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 4);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), -4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 0), -3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), -1.0, 0.0);

    sparse_free(A);
}

static void test_scale_rectangular(void) {
    SparseMatrix *A = sparse_create(2, 5);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 4, 2.0);
    sparse_insert(A, 1, 2, 3.0);

    ASSERT_ERR(sparse_scale(A, 0.5), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 3);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 0.5, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 4), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 2), 1.5, 0.0);

    sparse_free(A);
}

static void test_scale_null(void) { ASSERT_ERR(sparse_scale(NULL, 2.0), SPARSE_ERR_NULL); }

static void test_scale_empty_matrix(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_ERR(sparse_scale(A, 5.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 0);
    sparse_free(A);
}

static void test_scale_invalidates_norm(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);

    double norm;
    ASSERT_ERR(sparse_norminf(A, &norm), SPARSE_OK);
    ASSERT_NEAR(norm, 2.0, 0.0);

    ASSERT_ERR(sparse_scale(A, 3.0), SPARSE_OK);

    ASSERT_ERR(sparse_norminf(A, &norm), SPARSE_OK);
    ASSERT_NEAR(norm, 6.0, 0.0);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_add tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_add_a_plus_zero(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);

    SparseMatrix *Z = sparse_create(3, 3); /* zero matrix */

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, Z, 1.0, 1.0, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_nnz(C), 3);
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 2), 3.0, 0.0);

    sparse_free(A);
    sparse_free(Z);
    sparse_free(C);
}

static void test_add_a_minus_a(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 5.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 7.0);
    sparse_insert(A, 2, 0, -1.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, A, 1.0, -1.0, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_nnz(C), 0); /* everything cancels */

    sparse_free(A);
    sparse_free(C);
}

static void test_add_disjoint_patterns(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 2, 2, 3.0);

    SparseMatrix *B = sparse_create(3, 3);
    sparse_insert(B, 0, 2, 4.0);
    sparse_insert(B, 1, 1, 5.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), 4);
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 2), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 2), 3.0, 0.0);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_add_overlapping_entries(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);

    SparseMatrix *B = sparse_create(2, 2);
    sparse_insert(B, 0, 0, 7.0);
    sparse_insert(B, 1, 1, 6.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, B, 2.0, 0.5, &C), SPARSE_OK);
    /* C = 2*A + 0.5*B = (6+3.5, 0; 0, 8+3) = (9.5, 0; 0, 11) */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 9.5, 1e-14);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 11.0, 1e-14);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_add_dimension_mismatch(void) {
    SparseMatrix *A = sparse_create(3, 3);
    SparseMatrix *B = sparse_create(3, 4);
    SparseMatrix *C = NULL;

    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_SHAPE);
    ASSERT_NULL(C);

    sparse_free(A);
    sparse_free(B);
}

static void test_add_rectangular(void) {
    SparseMatrix *A = sparse_create(2, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 3, 2.0);

    SparseMatrix *B = sparse_create(2, 4);
    sparse_insert(B, 0, 0, 3.0);
    sparse_insert(B, 0, 2, 4.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_OK);
    ASSERT_EQ(sparse_rows(C), 2);
    ASSERT_EQ(sparse_cols(C), 4);
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 2), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 3), 2.0, 0.0);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_add_null_args(void) {
    SparseMatrix *A = sparse_create(2, 2);
    SparseMatrix *owner = sparse_create(1, 1);
    SparseMatrix *C = owner;

    ASSERT_ERR(sparse_add(NULL, A, 1.0, 1.0, &C), SPARSE_ERR_NULL);
    ASSERT_TRUE(C == owner);
    ASSERT_ERR(sparse_add(A, NULL, 1.0, 1.0, &C), SPARSE_ERR_NULL);
    ASSERT_TRUE(C == owner);
    ASSERT_ERR(sparse_add(A, A, 1.0, 1.0, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
    sparse_free(owner);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_add_inplace tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_add_inplace_basic(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);

    SparseMatrix *B = sparse_create(3, 3);
    sparse_insert(B, 0, 0, 3.0);
    sparse_insert(B, 2, 2, 4.0);

    ASSERT_ERR(sparse_add_inplace(A, B, 1.0, 1.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 3);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 2, 2), 4.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

static void test_add_inplace_with_scaling(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 10.0);
    sparse_insert(A, 1, 1, 20.0);

    SparseMatrix *B = sparse_create(2, 2);
    sparse_insert(B, 0, 0, 1.0);
    sparse_insert(B, 1, 1, 2.0);

    /* A = 0.5*A + 3*B = (5+3, 0; 0, 10+6) = (8, 0; 0, 16) */
    ASSERT_ERR(sparse_add_inplace(A, B, 0.5, 3.0), SPARSE_OK);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 8.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), 16.0, 1e-14);

    sparse_free(A);
    sparse_free(B);
}

static void test_add_inplace_cancellation(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 5.0);
    sparse_insert(A, 1, 1, 3.0);

    SparseMatrix *B = sparse_create(2, 2);
    sparse_insert(B, 0, 0, 5.0);
    sparse_insert(B, 1, 1, 3.0);

    ASSERT_ERR(sparse_add_inplace(A, B, 1.0, -1.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 0);

    sparse_free(A);
    sparse_free(B);
}

static void test_add_inplace_null(void) {
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_ERR(sparse_add_inplace(NULL, A, 1.0, 1.0), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_add_inplace(A, NULL, 1.0, 1.0), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_add_inplace_shape_mismatch(void) {
    SparseMatrix *A = sparse_create(2, 2);
    SparseMatrix *B = sparse_create(3, 3);
    ASSERT_ERR(sparse_add_inplace(A, B, 1.0, 1.0), SPARSE_ERR_SHAPE);
    sparse_free(A);
    sparse_free(B);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Integration: arithmetic + solver
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_add_then_factor_solve(void) {
    /*
     * Construct A = 4*I + T where T is tridiagonal (-1, 0, -1).
     * This gives a diagonally dominant 5x5 tridiagonal.
     */
    idx_t n = 5;

    /* I: identity */
    SparseMatrix *I = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(I, i, i, 1.0);

    /* T: off-diagonal tridiagonal */
    SparseMatrix *T = sparse_create(n, n);
    for (idx_t i = 0; i < n - 1; i++) {
        sparse_insert(T, i, i + 1, -1.0);
        sparse_insert(T, i + 1, i, -1.0);
    }

    /* A = 4*I + 1*T */
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_add(I, T, 4.0, 1.0, &A), SPARSE_OK);

    /* Verify: diagonal = 4, off-diag = -1 */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 1), -1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 2, 2), 4.0, 0.0);

    /* Copy for residual, factor, solve */
    SparseMatrix *A_orig = sparse_copy(A);
    ASSERT_NOT_NULL(A_orig);
    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    double b[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double x[5];
    ASSERT_ERR(sparse_lu_solve(A, b, x), SPARSE_OK);

    /* Check residual ||b - A_orig*x|| */
    double r[5];
    sparse_matvec(A_orig, x, r);
    double max_res = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ri = fabs(b[i] - r[i]);
        if (ri > max_res)
            max_res = ri;
    }
    ASSERT_TRUE(max_res < 1e-12);

    sparse_free(I);
    sparse_free(T);
    sparse_free(A);
    sparse_free(A_orig);
}

static void test_scale_then_factor_solve(void) {
    /* Build 3x3 matrix, scale by 0.001, factor and solve */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4000.0);
    sparse_insert(A, 0, 1, 1000.0);
    sparse_insert(A, 1, 0, 1000.0);
    sparse_insert(A, 1, 1, 3000.0);
    sparse_insert(A, 1, 2, 1000.0);
    sparse_insert(A, 2, 1, 1000.0);
    sparse_insert(A, 2, 2, 2000.0);

    ASSERT_ERR(sparse_scale(A, 0.001), SPARSE_OK);

    /* Now entries are 4, 1, 1, 3, 1, 1, 2 */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 4.0, 1e-14);

    SparseMatrix *A_orig = sparse_copy(A);
    ASSERT_NOT_NULL(A_orig);
    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    double b[] = {1.0, 1.0, 1.0};
    double x[3];
    ASSERT_ERR(sparse_lu_solve(A, b, x), SPARSE_OK);

    /* Residual check */
    double r[3];
    sparse_matvec(A_orig, x, r);
    for (idx_t i = 0; i < 3; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    sparse_free(A);
    sparse_free(A_orig);
}

static void test_norminf_after_add(void) {
    /* Verify norminf works correctly on a matrix produced by sparse_add */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 1, 3.0);

    SparseMatrix *B = sparse_create(3, 3);
    sparse_insert(B, 0, 0, 4.0);
    sparse_insert(B, 2, 2, 5.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_OK);

    /* C: row 0 = [5, 2, 0] sum=7, row 1 = [0, 3, 0] sum=3, row 2 = [0, 0, 5] sum=5 */
    double norm;
    ASSERT_ERR(sparse_norminf(C, &norm), SPARSE_OK);
    ASSERT_NEAR(norm, 7.0, 1e-14);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sparse Matrix Arithmetic Tests");

    /* Sprint 227 fixture infrastructure and the complete arithmetic-state
     * contract are always enforced. */
    RUN_TEST(test_s227_permutation_fixture_helpers);
    RUN_TEST(test_s227_independent_dense_oracles);
    RUN_TEST(test_s227_snapshot_and_structure_helpers);
    RUN_TEST(test_s227_add_original_state_rejection_contract);
    RUN_TEST(test_s227_add_inverse_only_state_rejected);
    RUN_TEST(test_s227_add_factored_state_rejected);
    RUN_TEST(test_s227_add_shape_precedes_state_validation);
    RUN_TEST(test_s227_add_inplace_self_alias_rejected_unchanged);
    RUN_TEST(test_s227_add_state_rejection_precedes_allocation);
    RUN_TEST(test_s227_sorted_add_builder_representative_rows);
    RUN_TEST(test_s227_sorted_add_builder_allocation_cleanup);
    RUN_TEST(test_s227_sorted_add_builder_exhaustive_allocation_sweep);
    RUN_TEST(test_s227_add_failure_output_ownership_matrix);
    RUN_TEST(test_s227_add_cancellation_allocation_sweep);
    RUN_TEST(test_s227_add_output_variable_alias_and_repeated_calls);
    RUN_TEST(test_s227_add_inplace_adversarial_success_contract);
    RUN_TEST(test_s227_add_inplace_zero_scale_reuses_removed_nodes);
    RUN_TEST(test_s227_add_inplace_allocation_failure_partial_contract);
    RUN_TEST(test_s227_sorted_add_builder_edge_corpus);
    RUN_TEST(test_s227_sorted_add_builder_randomized_oracle);
    RUN_TEST(test_s227_add_randomized_wide_rectangular_oracle);
    RUN_TEST(test_s227_matmul_original_state_rejection_contract);
    RUN_TEST(test_s227_matmul_inverse_only_state_rejected);
    RUN_TEST(test_s227_matmul_factored_state_rejected);
    RUN_TEST(test_s227_matmul_validation_precedence_and_output);
    RUN_TEST(test_s227_matmul_state_rejection_precedes_allocation);

    /* Scale */
    RUN_TEST(test_scale_identity_by_3);
    RUN_TEST(test_scale_by_1);
    RUN_TEST(test_scale_by_0);
    RUN_TEST(test_scale_by_negative_1);
    RUN_TEST(test_scale_rectangular);
    RUN_TEST(test_scale_null);
    RUN_TEST(test_scale_empty_matrix);
    RUN_TEST(test_scale_invalidates_norm);

    /* Add (out-of-place) */
    RUN_TEST(test_add_a_plus_zero);
    RUN_TEST(test_add_a_minus_a);
    RUN_TEST(test_add_disjoint_patterns);
    RUN_TEST(test_add_overlapping_entries);
    RUN_TEST(test_add_dimension_mismatch);
    RUN_TEST(test_add_rectangular);
    RUN_TEST(test_add_null_args);

    /* Add (in-place) */
    RUN_TEST(test_add_inplace_basic);
    RUN_TEST(test_add_inplace_with_scaling);
    RUN_TEST(test_add_inplace_cancellation);
    RUN_TEST(test_add_inplace_null);
    RUN_TEST(test_add_inplace_shape_mismatch);

    /* Integration: arithmetic + solver */
    RUN_TEST(test_add_then_factor_solve);
    RUN_TEST(test_scale_then_factor_solve);
    RUN_TEST(test_norminf_after_add);

    TEST_SUITE_END();
}
