#ifndef TEST_ITERATIVE_HANDLE_HELPERS_H
#define TEST_ITERATIVE_HANDLE_HELPERS_H

/* Public repeated-run CG handle: NULL validation, explicit prepare + reuse,
 * and zero-init on-demand growth should all follow the final public contract. */
static void test_cg_public_handle_validation_reuse_and_on_demand(void) {
    SparseMatrix *A = build_spd_tridiag(8, 4.0, -1.0);
    ASSERT_NOT_NULL(A);

    double x_exact[8];
    double b[8];
    double x1[8] = {0};
    double x2[8] = {0};
    double x3[8] = {0};
    for (idx_t i = 0; i < 8; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t r1 = {0}, r2 = {0}, r3 = {0};
    sparse_iter_handle_t handle = {0};
    sparse_iter_handle_t zero_handle = {0};

    ASSERT_ERR(sparse_iter_handle_prepare_cg(NULL, 8), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_solve_cg_with_handle(A, b, x1, &opts, NULL, NULL, &r1, NULL),
               SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_iter_handle_prepare_cg(&handle, 8), SPARSE_OK);
    ASSERT_ERR(sparse_solve_cg_with_handle(A, b, x1, &opts, NULL, NULL, &r1, &handle), SPARSE_OK);
    ASSERT_ERR(sparse_solve_cg_with_handle(A, b, x2, &opts, NULL, NULL, &r2, &handle), SPARSE_OK);
    ASSERT_ERR(sparse_solve_cg_with_handle(A, b, x3, &opts, NULL, NULL, &r3, &zero_handle),
               SPARSE_OK);

    ASSERT_TRUE(r1.converged);
    ASSERT_TRUE(r2.converged);
    ASSERT_TRUE(r3.converged);
    ASSERT_EQ(r1.iterations, r2.iterations);
    ASSERT_EQ(r1.iterations, r3.iterations);
    for (idx_t i = 0; i < 8; i++) {
        ASSERT_NEAR(x1[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x2[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x3[i], x_exact[i], 1e-10);
    }

    sparse_iter_handle_free(&handle);
    sparse_iter_handle_free(&zero_handle);
    sparse_free(A);
}

/* Public repeated-run GMRES handle: NULL validation, explicit prepare + reuse,
 * and underprepared on-demand growth should all work under the final public
 * contract. */
static void test_gmres_public_handle_prepare_reuse_and_growth(void) {
    SparseMatrix *A_small = build_unsym_tridiag(8, 4.0, -0.5, -1.0);
    SparseMatrix *A_large = build_unsym_tridiag(12, 4.0, -0.5, -1.0);
    ASSERT_NOT_NULL(A_small);
    ASSERT_NOT_NULL(A_large);

    double x_exact_small[8];
    double b_small[8];
    double x1[8] = {0};
    double x2[8] = {0};
    for (idx_t i = 0; i < 8; i++)
        x_exact_small[i] = (double)(i + 1);
    compute_rhs(A_small, x_exact_small, b_small);

    double x_exact_large[12];
    double b_large[12];
    double x3[12] = {0};
    double x4[12] = {0};
    for (idx_t i = 0; i < 12; i++)
        x_exact_large[i] = (double)(i + 1);
    compute_rhs(A_large, x_exact_large, b_large);

    sparse_gmres_opts_t opts_small = {.max_iter = 200, .restart = 6, .tol = 1e-10, .verbose = 0};
    sparse_gmres_opts_t opts_large = {.max_iter = 200, .restart = 12, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t r1 = {0}, r2 = {0}, r3 = {0}, r4 = {0};
    sparse_iter_handle_t handle = {0};
    sparse_iter_handle_t zero_handle = {0};

    ASSERT_ERR(sparse_iter_handle_prepare_gmres(NULL, 8, opts_small.restart), SPARSE_ERR_NULL);
    ASSERT_ERR(
        sparse_solve_gmres_with_handle(A_small, b_small, x1, &opts_small, NULL, NULL, &r1, NULL),
        SPARSE_ERR_NULL);

    ASSERT_ERR(sparse_iter_handle_prepare_gmres(&handle, 8, opts_small.restart), SPARSE_OK);
    ASSERT_ERR(
        sparse_solve_gmres_with_handle(A_small, b_small, x1, &opts_small, NULL, NULL, &r1, &handle),
        SPARSE_OK);
    ASSERT_ERR(
        sparse_solve_gmres_with_handle(A_small, b_small, x2, &opts_small, NULL, NULL, &r2, &handle),
        SPARSE_OK);

    /* The same handle should grow on demand for a larger dimension / restart. */
    ASSERT_ERR(
        sparse_solve_gmres_with_handle(A_large, b_large, x3, &opts_large, NULL, NULL, &r3, &handle),
        SPARSE_OK);

    /* A zero-init handle should also grow on demand on first use. */
    ASSERT_ERR(sparse_solve_gmres_with_handle(A_large, b_large, x4, &opts_large, NULL, NULL, &r4,
                                              &zero_handle),
               SPARSE_OK);

    ASSERT_TRUE(r1.converged);
    ASSERT_TRUE(r2.converged);
    ASSERT_TRUE(r3.converged);
    ASSERT_TRUE(r4.converged);
    ASSERT_EQ(r1.iterations, r2.iterations);
    ASSERT_EQ(r3.iterations, r4.iterations);
    for (idx_t i = 0; i < 8; i++) {
        ASSERT_NEAR(x1[i], x_exact_small[i], 1e-8);
        ASSERT_NEAR(x2[i], x_exact_small[i], 1e-8);
    }
    for (idx_t i = 0; i < 12; i++) {
        ASSERT_NEAR(x3[i], x_exact_large[i], 1e-8);
        ASSERT_NEAR(x4[i], x_exact_large[i], 1e-8);
    }

    sparse_iter_handle_free(&handle);
    sparse_iter_handle_free(&zero_handle);
    sparse_free(A_small);
    sparse_free(A_large);
}

/* Public repeated-run MINRES handle: explicit prepare + reuse should preserve
 * the same numerical behavior as the one-shot path, and an underprepared
 * handle should still grow on demand for a larger solve. */
static void test_minres_public_handle_prepare_reuse_and_growth(void) {
    SparseMatrix *A_small = build_spd_tridiag(8, 4.0, -1.0);
    SparseMatrix *A_large = build_spd_tridiag(12, 4.0, -1.0);
    ASSERT_NOT_NULL(A_small);
    ASSERT_NOT_NULL(A_large);

    double x_exact_small[8];
    double b_small[8];
    double x1[8] = {0};
    double x2[8] = {0};
    for (idx_t i = 0; i < 8; i++)
        x_exact_small[i] = (double)(i + 1);
    compute_rhs(A_small, x_exact_small, b_small);

    double x_exact_large[12];
    double b_large[12];
    double x3[12] = {0};
    double x4[12] = {0};
    for (idx_t i = 0; i < 12; i++)
        x_exact_large[i] = (double)(i + 1);
    compute_rhs(A_large, x_exact_large, b_large);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t r1 = {0}, r2 = {0}, r3 = {0}, r4 = {0};
    sparse_iter_handle_t handle = {0};
    sparse_iter_handle_t zero_handle = {0};

    ASSERT_ERR(sparse_iter_handle_prepare_minres(NULL, 8), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_solve_minres_with_handle(A_small, b_small, x1, &opts, NULL, NULL, &r1, NULL),
               SPARSE_ERR_NULL);

    ASSERT_ERR(sparse_iter_handle_prepare_minres(&handle, 8), SPARSE_OK);
    ASSERT_ERR(
        sparse_solve_minres_with_handle(A_small, b_small, x1, &opts, NULL, NULL, &r1, &handle),
        SPARSE_OK);
    ASSERT_ERR(
        sparse_solve_minres_with_handle(A_small, b_small, x2, &opts, NULL, NULL, &r2, &handle),
        SPARSE_OK);

    /* The same handle should grow on demand when the later solve is larger. */
    ASSERT_ERR(
        sparse_solve_minres_with_handle(A_large, b_large, x3, &opts, NULL, NULL, &r3, &handle),
        SPARSE_OK);

    /* A zero-init handle should also grow on demand on first use. */
    ASSERT_ERR(
        sparse_solve_minres_with_handle(A_large, b_large, x4, &opts, NULL, NULL, &r4, &zero_handle),
        SPARSE_OK);

    ASSERT_TRUE(r1.converged);
    ASSERT_TRUE(r2.converged);
    ASSERT_TRUE(r3.converged);
    ASSERT_TRUE(r4.converged);
    ASSERT_EQ(r1.iterations, r2.iterations);
    ASSERT_EQ(r3.iterations, r4.iterations);
    for (idx_t i = 0; i < 8; i++) {
        ASSERT_NEAR(x1[i], x_exact_small[i], 1e-10);
        ASSERT_NEAR(x2[i], x_exact_small[i], 1e-10);
    }
    for (idx_t i = 0; i < 12; i++) {
        ASSERT_NEAR(x3[i], x_exact_large[i], 1e-10);
        ASSERT_NEAR(x4[i], x_exact_large[i], 1e-10);
    }

    sparse_iter_handle_free(&handle);
    sparse_iter_handle_free(&zero_handle);
    sparse_free(A_small);
    sparse_free(A_large);
}

#endif
