/*
 * Sprint 22 Day 10 — quotient-graph AMD stub contract.
 *
 * The Day 10 commit ships only the design block + stub.  Day 11
 * replaces the body with the full elimination loop and at that
 * point this file gains the parallel-comparison tests against the
 * (still-production) bitset AMD on nos4 / bcsstk04 / bcsstk14.
 * Until then we just lock in:
 *   - The stub returns SPARSE_ERR_BADARG on a valid square input
 *     (locks the "stub in progress" signal so a future commit
 *     can't silently change the contract).
 *   - NULL / non-square argument validation runs before the
 *     stub body, matching the Sprint-22 stub pattern from
 *     `sparse_graph_partition` and `sparse_graph_subgraph`.
 */

#include "sparse_matrix.h"
#include "sparse_reorder_amd_qg_internal.h"
#include "sparse_types.h"
#include "test_framework.h"

static void test_amd_qg_is_stub(void) {
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_NOT_NULL(A);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);

    idx_t perm[2] = {0};
    /* Day 10: stub returns BADARG.  Day 11 will replace this with a
     * proper success expectation. */
    ASSERT_ERR(sparse_reorder_amd_qg(A, perm), SPARSE_ERR_BADARG);

    sparse_free(A);
}

static void test_amd_qg_null_args(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    idx_t perm[1] = {0};
    ASSERT_ERR(sparse_reorder_amd_qg(NULL, perm), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_reorder_amd_qg(A, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_amd_qg_rejects_rectangular(void) {
    SparseMatrix *A = sparse_create(3, 5);
    ASSERT_NOT_NULL(A);
    idx_t perm[3] = {0};
    ASSERT_ERR(sparse_reorder_amd_qg(A, perm), SPARSE_ERR_SHAPE);
    sparse_free(A);
}

int main(void) {
    TEST_SUITE_BEGIN("Sprint 22 Day 10: quotient-graph AMD stub contract");
    RUN_TEST(test_amd_qg_is_stub);
    RUN_TEST(test_amd_qg_null_args);
    RUN_TEST(test_amd_qg_rejects_rectangular);
    TEST_SUITE_END();
}
