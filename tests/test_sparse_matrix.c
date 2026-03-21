#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"

/* ═══════════════════════════════════════════════════════════════════════
 * Creation tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_create_basic(void)
{
    SparseMatrix *m = sparse_create(5, 5);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(sparse_rows(m), 5);
    ASSERT_EQ(sparse_cols(m), 5);
    ASSERT_EQ(sparse_nnz(m), 0);
    sparse_free(m);
}

static void test_create_rectangular(void)
{
    SparseMatrix *m = sparse_create(3, 7);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(sparse_rows(m), 3);
    ASSERT_EQ(sparse_cols(m), 7);
    sparse_free(m);
}

static void test_create_1x1(void)
{
    SparseMatrix *m = sparse_create(1, 1);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(sparse_rows(m), 1);
    ASSERT_EQ(sparse_cols(m), 1);
    sparse_free(m);
}

static void test_create_invalid(void)
{
    ASSERT_NULL(sparse_create(0, 5));
    ASSERT_NULL(sparse_create(5, 0));
    ASSERT_NULL(sparse_create(-1, 5));
    ASSERT_NULL(sparse_create(5, -1));
    ASSERT_NULL(sparse_create(0, 0));
}

static void test_free_null(void)
{
    /* Should not crash */
    sparse_free(NULL);
    ASSERT_TRUE(1);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Insert / Get tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_insert_get_single(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    ASSERT_ERR(sparse_insert(m, 1, 2, 42.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(m), 1);
    ASSERT_NEAR(sparse_get_phys(m, 1, 2), 42.0, 0.0);
    /* Other positions are zero */
    ASSERT_NEAR(sparse_get_phys(m, 0, 0), 0.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 1, 1), 0.0, 0.0);
    sparse_free(m);
}

static void test_insert_get_multiple(void)
{
    SparseMatrix *m = sparse_create(4, 4);
    sparse_insert(m, 0, 0, 1.0);
    sparse_insert(m, 1, 2, 2.0);
    sparse_insert(m, 2, 1, 3.0);
    sparse_insert(m, 3, 3, 4.0);
    ASSERT_EQ(sparse_nnz(m), 4);

    ASSERT_NEAR(sparse_get_phys(m, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 1, 2), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 2, 1), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 3, 3), 4.0, 0.0);
    /* Non-stored positions */
    ASSERT_NEAR(sparse_get_phys(m, 0, 1), 0.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 3, 0), 0.0, 0.0);
    sparse_free(m);
}

static void test_insert_overwrite(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    sparse_insert(m, 1, 1, 10.0);
    ASSERT_NEAR(sparse_get_phys(m, 1, 1), 10.0, 0.0);
    ASSERT_EQ(sparse_nnz(m), 1);

    /* Overwrite with new value */
    sparse_insert(m, 1, 1, 99.0);
    ASSERT_NEAR(sparse_get_phys(m, 1, 1), 99.0, 0.0);
    ASSERT_EQ(sparse_nnz(m), 1);  /* still 1, not 2 */
    sparse_free(m);
}

static void test_insert_zero_removes(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    sparse_insert(m, 0, 0, 5.0);
    ASSERT_EQ(sparse_nnz(m), 1);

    /* Insert 0.0 should remove */
    sparse_insert(m, 0, 0, 0.0);
    ASSERT_EQ(sparse_nnz(m), 0);
    ASSERT_NEAR(sparse_get_phys(m, 0, 0), 0.0, 0.0);
    sparse_free(m);
}

static void test_get_empty(void)
{
    SparseMatrix *m = sparse_create(5, 5);
    for (idx_t i = 0; i < 5; i++)
        for (idx_t j = 0; j < 5; j++)
            ASSERT_NEAR(sparse_get_phys(m, i, j), 0.0, 0.0);
    sparse_free(m);
}

static void test_insert_multiple_same_row(void)
{
    /* Verify column ordering within a row */
    SparseMatrix *m = sparse_create(3, 5);
    sparse_insert(m, 1, 4, 40.0);
    sparse_insert(m, 1, 0, 10.0);
    sparse_insert(m, 1, 2, 20.0);
    ASSERT_EQ(sparse_nnz(m), 3);
    ASSERT_NEAR(sparse_get_phys(m, 1, 0), 10.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 1, 2), 20.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 1, 4), 40.0, 0.0);
    sparse_free(m);
}

static void test_insert_multiple_same_col(void)
{
    /* Verify row ordering within a column */
    SparseMatrix *m = sparse_create(5, 3);
    sparse_insert(m, 3, 1, 30.0);
    sparse_insert(m, 0, 1, 10.0);
    sparse_insert(m, 4, 1, 40.0);
    ASSERT_EQ(sparse_nnz(m), 3);
    ASSERT_NEAR(sparse_get_phys(m, 0, 1), 10.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 3, 1), 30.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 4, 1), 40.0, 0.0);
    sparse_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Logical get/set (through permutations)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_logical_get_set(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    /* With identity perms, logical == physical */
    ASSERT_ERR(sparse_set(m, 1, 2, 7.0), SPARSE_OK);
    ASSERT_NEAR(sparse_get(m, 1, 2), 7.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 1, 2), 7.0, 0.0);
    sparse_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Remove tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_remove_existing(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    sparse_insert(m, 1, 1, 5.0);
    ASSERT_EQ(sparse_nnz(m), 1);

    ASSERT_ERR(sparse_remove(m, 1, 1), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(m), 0);
    ASSERT_NEAR(sparse_get_phys(m, 1, 1), 0.0, 0.0);
    sparse_free(m);
}

static void test_remove_nonexistent(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    /* Remove from empty matrix — should be no-op */
    ASSERT_ERR(sparse_remove(m, 1, 1), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(m), 0);
    sparse_free(m);
}

static void test_remove_middle_of_row(void)
{
    SparseMatrix *m = sparse_create(3, 5);
    sparse_insert(m, 0, 0, 1.0);
    sparse_insert(m, 0, 2, 2.0);
    sparse_insert(m, 0, 4, 3.0);
    ASSERT_EQ(sparse_nnz(m), 3);

    /* Remove middle element */
    sparse_remove(m, 0, 2);
    ASSERT_EQ(sparse_nnz(m), 2);
    ASSERT_NEAR(sparse_get_phys(m, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 0, 2), 0.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 0, 4), 3.0, 0.0);
    sparse_free(m);
}

static void test_nnz_tracking(void)
{
    SparseMatrix *m = sparse_create(4, 4);
    ASSERT_EQ(sparse_nnz(m), 0);

    sparse_insert(m, 0, 0, 1.0);
    ASSERT_EQ(sparse_nnz(m), 1);
    sparse_insert(m, 1, 1, 2.0);
    ASSERT_EQ(sparse_nnz(m), 2);
    sparse_insert(m, 2, 2, 3.0);
    ASSERT_EQ(sparse_nnz(m), 3);

    /* Overwrite — nnz stays same */
    sparse_insert(m, 1, 1, 99.0);
    ASSERT_EQ(sparse_nnz(m), 3);

    /* Remove */
    sparse_remove(m, 1, 1);
    ASSERT_EQ(sparse_nnz(m), 2);

    /* Insert zero (remove) */
    sparse_insert(m, 0, 0, 0.0);
    ASSERT_EQ(sparse_nnz(m), 1);

    /* Remove nonexistent */
    sparse_remove(m, 3, 3);
    ASSERT_EQ(sparse_nnz(m), 1);

    sparse_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Bounds checking
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_insert_out_of_bounds(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    ASSERT_ERR(sparse_insert(m, 3, 0, 1.0), SPARSE_ERR_BOUNDS);
    ASSERT_ERR(sparse_insert(m, 0, 3, 1.0), SPARSE_ERR_BOUNDS);
    ASSERT_ERR(sparse_insert(m, -1, 0, 1.0), SPARSE_ERR_BOUNDS);
    ASSERT_ERR(sparse_insert(m, 0, -1, 1.0), SPARSE_ERR_BOUNDS);
    ASSERT_EQ(sparse_nnz(m), 0);
    sparse_free(m);
}

static void test_remove_out_of_bounds(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    ASSERT_ERR(sparse_remove(m, 5, 0), SPARSE_ERR_BOUNDS);
    ASSERT_ERR(sparse_remove(m, 0, 5), SPARSE_ERR_BOUNDS);
    sparse_free(m);
}

static void test_set_out_of_bounds(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    ASSERT_ERR(sparse_set(m, 3, 0, 1.0), SPARSE_ERR_BOUNDS);
    ASSERT_ERR(sparse_set(m, 0, 3, 1.0), SPARSE_ERR_BOUNDS);
    sparse_free(m);
}

static void test_get_out_of_bounds(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    /* get returns 0.0 for out-of-bounds (no error code available) */
    ASSERT_NEAR(sparse_get(m, 5, 0), 0.0, 0.0);
    ASSERT_NEAR(sparse_get(m, 0, 5), 0.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, -1, 0), 0.0, 0.0);
    sparse_free(m);
}

static void test_null_pointer_args(void)
{
    ASSERT_ERR(sparse_insert(NULL, 0, 0, 1.0), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_remove(NULL, 0, 0), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_set(NULL, 0, 0, 1.0), SPARSE_ERR_NULL);
    ASSERT_NEAR(sparse_get(NULL, 0, 0), 0.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(NULL, 0, 0), 0.0, 0.0);
    ASSERT_EQ(sparse_rows(NULL), 0);
    ASSERT_EQ(sparse_cols(NULL), 0);
    ASSERT_EQ(sparse_nnz(NULL), 0);
    ASSERT_NULL(sparse_row_perm(NULL));
    ASSERT_ERR(sparse_reset_perms(NULL), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Copy tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_copy_basic(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    sparse_insert(m, 0, 0, 1.0);
    sparse_insert(m, 0, 2, 3.0);
    sparse_insert(m, 1, 1, 5.0);
    sparse_insert(m, 2, 0, 7.0);
    sparse_insert(m, 2, 2, 9.0);

    SparseMatrix *c = sparse_copy(m);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(sparse_rows(c), 3);
    ASSERT_EQ(sparse_cols(c), 3);
    ASSERT_EQ(sparse_nnz(c), 5);

    /* Verify all elements match */
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 3; j++)
            ASSERT_NEAR(sparse_get_phys(c, i, j),
                        sparse_get_phys(m, i, j), 0.0);

    sparse_free(c);
    sparse_free(m);
}

static void test_copy_independent(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    sparse_insert(m, 0, 0, 1.0);
    sparse_insert(m, 1, 1, 2.0);

    SparseMatrix *c = sparse_copy(m);

    /* Modify copy */
    sparse_insert(c, 0, 0, 99.0);
    sparse_insert(c, 2, 2, 88.0);

    /* Original unchanged */
    ASSERT_NEAR(sparse_get_phys(m, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(m, 2, 2), 0.0, 0.0);
    ASSERT_EQ(sparse_nnz(m), 2);

    /* Copy has changes */
    ASSERT_NEAR(sparse_get_phys(c, 0, 0), 99.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(c, 2, 2), 88.0, 0.0);
    ASSERT_EQ(sparse_nnz(c), 3);

    sparse_free(c);
    sparse_free(m);
}

static void test_copy_null(void)
{
    ASSERT_NULL(sparse_copy(NULL));
}

/* ═══════════════════════════════════════════════════════════════════════
 * Matrix-vector product tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_matvec_identity(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    sparse_insert(m, 0, 0, 1.0);
    sparse_insert(m, 1, 1, 1.0);
    sparse_insert(m, 2, 2, 1.0);

    double x[3] = {2.0, 3.0, 4.0};
    double y[3] = {0};
    ASSERT_ERR(sparse_matvec(m, x, y), SPARSE_OK);
    ASSERT_NEAR(y[0], 2.0, 1e-15);
    ASSERT_NEAR(y[1], 3.0, 1e-15);
    ASSERT_NEAR(y[2], 4.0, 1e-15);
    sparse_free(m);
}

static void test_matvec_general(void)
{
    /* [1 0 3] [1]   [1 + 9]   [10]
     * [0 5 0] [2] = [10]    = [10]
     * [7 0 9] [3]   [7 + 27]  [34]
     */
    SparseMatrix *m = sparse_create(3, 3);
    sparse_insert(m, 0, 0, 1.0);
    sparse_insert(m, 0, 2, 3.0);
    sparse_insert(m, 1, 1, 5.0);
    sparse_insert(m, 2, 0, 7.0);
    sparse_insert(m, 2, 2, 9.0);

    double x[3] = {1.0, 2.0, 3.0};
    double y[3] = {0};
    ASSERT_ERR(sparse_matvec(m, x, y), SPARSE_OK);
    ASSERT_NEAR(y[0], 10.0, 1e-14);
    ASSERT_NEAR(y[1], 10.0, 1e-14);
    ASSERT_NEAR(y[2], 34.0, 1e-14);
    sparse_free(m);
}

static void test_matvec_null(void)
{
    SparseMatrix *m = sparse_create(2, 2);
    double x[2], y[2];
    ASSERT_ERR(sparse_matvec(NULL, x, y), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_matvec(m, NULL, y), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_matvec(m, x, NULL), SPARSE_ERR_NULL);
    sparse_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Permutation tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_perms_initial_identity(void)
{
    SparseMatrix *m = sparse_create(4, 4);
    const idx_t *rp = sparse_row_perm(m);
    const idx_t *cp = sparse_col_perm(m);
    const idx_t *irp = sparse_inv_row_perm(m);
    const idx_t *icp = sparse_inv_col_perm(m);

    for (idx_t i = 0; i < 4; i++) {
        ASSERT_EQ(rp[i], i);
        ASSERT_EQ(cp[i], i);
        ASSERT_EQ(irp[i], i);
        ASSERT_EQ(icp[i], i);
    }
    sparse_free(m);
}

static void test_perms_reset(void)
{
    SparseMatrix *m = sparse_create(3, 3);
    /* Insert and do a factorization to scramble perms */
    sparse_insert(m, 0, 0, 1.0);
    sparse_insert(m, 1, 1, 1.0);
    sparse_insert(m, 2, 2, 1.0);

    ASSERT_ERR(sparse_reset_perms(m), SPARSE_OK);

    const idx_t *rp = sparse_row_perm(m);
    for (idx_t i = 0; i < 3; i++)
        ASSERT_EQ(rp[i], i);
    sparse_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Memory info tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_memory_usage(void)
{
    SparseMatrix *m = sparse_create(10, 10);
    size_t mem0 = sparse_memory_usage(m);
    ASSERT_TRUE(mem0 > 0);

    /* Insert enough to trigger a slab allocation */
    for (idx_t i = 0; i < 10; i++)
        sparse_insert(m, i, i, (double)(i + 1));

    size_t mem1 = sparse_memory_usage(m);
    ASSERT_TRUE(mem1 >= mem0);

    ASSERT_EQ(sparse_memory_usage(NULL), 0);
    sparse_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Sparse Matrix Data Structure Tests");

    /* Creation */
    RUN_TEST(test_create_basic);
    RUN_TEST(test_create_rectangular);
    RUN_TEST(test_create_1x1);
    RUN_TEST(test_create_invalid);
    RUN_TEST(test_free_null);

    /* Insert / Get */
    RUN_TEST(test_insert_get_single);
    RUN_TEST(test_insert_get_multiple);
    RUN_TEST(test_insert_overwrite);
    RUN_TEST(test_insert_zero_removes);
    RUN_TEST(test_get_empty);
    RUN_TEST(test_insert_multiple_same_row);
    RUN_TEST(test_insert_multiple_same_col);

    /* Logical get/set */
    RUN_TEST(test_logical_get_set);

    /* Remove */
    RUN_TEST(test_remove_existing);
    RUN_TEST(test_remove_nonexistent);
    RUN_TEST(test_remove_middle_of_row);
    RUN_TEST(test_nnz_tracking);

    /* Bounds checking */
    RUN_TEST(test_insert_out_of_bounds);
    RUN_TEST(test_remove_out_of_bounds);
    RUN_TEST(test_set_out_of_bounds);
    RUN_TEST(test_get_out_of_bounds);
    RUN_TEST(test_null_pointer_args);

    /* Copy */
    RUN_TEST(test_copy_basic);
    RUN_TEST(test_copy_independent);
    RUN_TEST(test_copy_null);

    /* Matrix-vector product */
    RUN_TEST(test_matvec_identity);
    RUN_TEST(test_matvec_general);
    RUN_TEST(test_matvec_null);

    /* Permutations */
    RUN_TEST(test_perms_initial_identity);
    RUN_TEST(test_perms_reset);

    /* Memory info */
    RUN_TEST(test_memory_usage);

    TEST_SUITE_END();
}
