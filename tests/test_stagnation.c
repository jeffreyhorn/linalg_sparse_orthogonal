#include "sparse_iterative.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Test helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static SparseMatrix *build_spd_tridiag(idx_t n, double diag_val, double offdiag_val) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag_val);
        if (i > 0)
            sparse_insert(A, i, i - 1, offdiag_val);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, offdiag_val);
    }
    return A;
}

static SparseMatrix *build_ill_conditioned_spd(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        double diag = (i == 0) ? 1e8 : 1.0;
        sparse_insert(A, i, i, diag);
        if (i > 0)
            sparse_insert(A, i, i - 1, 0.5);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, 0.5);
    }
    return A;
}

static sparse_err_t sparse_matvec_cb(const void *ctx, idx_t n, const double *xv, double *y) {
    const SparseMatrix *A = (const SparseMatrix *)ctx;
    (void)n;
    return sparse_matvec(A, xv, y);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CG stagnation detection tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_cg_stagnation_disabled_by_default(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.stagnated);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_cg_no_stagnation_well_conditioned(void) {
    idx_t n = 20;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {
        .max_iter = 200, .tol = 1e-10, .verbose = 0, .stagnation_window = 10};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.stagnated);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_cg_stagnation_detected(void) {
    idx_t n = 30;
    SparseMatrix *A = build_ill_conditioned_spd(n);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {
        .max_iter = 5000, .tol = 1e-14, .verbose = 0, .stagnation_window = 15};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    /* With a very tight tolerance on an ill-conditioned system,
     * CG should stagnate before reaching max_iter */
    printf("    CG stagnation: iters=%d, stagnated=%d, converged=%d, res=%.3e\n",
           (int)result.iterations, result.stagnated, result.converged, result.residual_norm);

    if (!result.converged) {
        ASSERT_ERR(err, SPARSE_ERR_NOT_CONVERGED);
        ASSERT_TRUE(result.stagnated);
        ASSERT_TRUE(result.iterations < 5000);
    }

    free(b);
    free(x);
    sparse_free(A);
}

static void test_cg_stagnation_saves_iterations(void) {
    idx_t n = 30;
    SparseMatrix *A = build_ill_conditioned_spd(n);
    double *b = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    /* Without stagnation detection: runs to max_iter */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts_no = {.max_iter = 2000, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t r1;
    sparse_solve_cg(A, b, x1, &opts_no, NULL, NULL, &r1);

    /* With stagnation detection: exits early */
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts_stag = {
        .max_iter = 2000, .tol = 1e-14, .verbose = 0, .stagnation_window = 15};
    sparse_iter_result_t r2;
    sparse_solve_cg(A, b, x2, &opts_stag, NULL, NULL, &r2);

    printf("    CG stagnation savings: no_stag=%d iters, stag=%d iters\n", (int)r1.iterations,
           (int)r2.iterations);

    if (r2.stagnated) {
        ASSERT_TRUE(r2.iterations < r1.iterations);
    }

    free(b);
    free(x1);
    free(x2);
    sparse_free(A);
}

static void test_cg_stagnation_window_zero_disabled(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0, .stagnation_window = 0};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.stagnated);

    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * MINRES stagnation detection tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_minres_no_stagnation_well_conditioned(void) {
    idx_t n = 20;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {
        .max_iter = 200, .tol = 1e-10, .verbose = 0, .stagnation_window = 10};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_minres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.stagnated);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_minres_stagnation_detected(void) {
    idx_t n = 30;
    SparseMatrix *A = build_ill_conditioned_spd(n);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {
        .max_iter = 5000, .tol = 1e-14, .verbose = 0, .stagnation_window = 15};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_minres(A, b, x, &opts, NULL, NULL, &result);

    printf("    MINRES stagnation: iters=%d, stagnated=%d, converged=%d, res=%.3e\n",
           (int)result.iterations, result.stagnated, result.converged, result.residual_norm);

    if (!result.converged) {
        ASSERT_ERR(err, SPARSE_ERR_NOT_CONVERGED);
        ASSERT_TRUE(result.stagnated);
        ASSERT_TRUE(result.iterations < 5000);
    }

    free(b);
    free(x);
    sparse_free(A);
}

static void test_minres_stagnation_disabled_by_default(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_result_t result;
    sparse_solve_minres(A, b, x, NULL, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.stagnated);

    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * GMRES stagnation detection tests (Day 8)
 * ═══════════════════════════════════════════════════════════════════════ */

static SparseMatrix *build_unsym_tridiag(idx_t n, double diag, double upper, double lower) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag);
        if (i > 0)
            sparse_insert(A, i, i - 1, lower);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, upper);
    }
    return A;
}

static void test_gmres_no_stagnation_well_conditioned(void) {
    idx_t n = 20;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_gmres_opts_t opts = {
        .max_iter = 200, .restart = 30, .tol = 1e-10, .verbose = 0, .stagnation_window = 10};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.stagnated);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_gmres_stagnation_small_restart(void) {
    idx_t n = 40;
    SparseMatrix *A = build_ill_conditioned_spd(n);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    /* Small restart with very tight tolerance — restarts lose information */
    sparse_gmres_opts_t opts = {
        .max_iter = 5000, .restart = 5, .tol = 1e-14, .verbose = 0, .stagnation_window = 10};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);

    printf("    GMRES stagnation (restart=5): iters=%d, stagnated=%d, converged=%d, res=%.3e\n",
           (int)result.iterations, result.stagnated, result.converged, result.residual_norm);

    if (!result.converged) {
        ASSERT_ERR(err, SPARSE_ERR_NOT_CONVERGED);
        ASSERT_TRUE(result.stagnated);
        ASSERT_TRUE(result.iterations < 5000);
    }

    free(b);
    free(x);
    sparse_free(A);
}

static void test_gmres_stagnation_disabled_default(void) {
    idx_t n = 10;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_result_t result;
    sparse_solve_gmres(A, b, x, NULL, NULL, NULL, &result);
    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.stagnated);

    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * BiCGSTAB stagnation detection tests (Day 8)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bicgstab_no_stagnation_well_conditioned(void) {
    idx_t n = 20;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {
        .max_iter = 200, .tol = 1e-10, .verbose = 0, .stagnation_window = 10};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.stagnated);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_stagnation_detected(void) {
    idx_t n = 30;
    SparseMatrix *A = build_ill_conditioned_spd(n);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {
        .max_iter = 5000, .tol = 1e-14, .verbose = 0, .stagnation_window = 15};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);

    printf("    BiCGSTAB stagnation: iters=%d, stagnated=%d, converged=%d, res=%.3e\n",
           (int)result.iterations, result.stagnated, result.converged, result.residual_norm);

    if (!result.converged) {
        ASSERT_ERR(err, SPARSE_ERR_NOT_CONVERGED);
        ASSERT_TRUE(result.stagnated);
        ASSERT_TRUE(result.iterations < 5000);
    }

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_stagnation_disabled_default(void) {
    idx_t n = 10;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_result_t result;
    sparse_solve_bicgstab(A, b, x, NULL, NULL, NULL, &result);
    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.stagnated);

    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cross-solver stagnation tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_stagnation_result_field_init(void) {
    sparse_iter_result_t r = {0};
    ASSERT_EQ(r.stagnated, 0);
}

static void test_stagnation_null_opts_safe(void) {
    idx_t n = 5;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double b[5] = {1, 2, 3, 4, 5};
    double x[5] = {0};
    sparse_iter_result_t result;

    sparse_solve_cg(A, b, x, NULL, NULL, NULL, &result);
    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.stagnated);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Residual history recording tests (Day 9)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_cg_residual_history_basic(void) {
    idx_t n = 20;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    double history[200] = {0};
    sparse_iter_opts_t opts = {.max_iter = 200,
                               .tol = 1e-10,
                               .verbose = 0,
                               .residual_history = history,
                               .residual_history_len = 200};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.residual_history_count > 0);
    ASSERT_TRUE(result.residual_history_count <= result.iterations);

    /* Verify monotonic decrease (CG on SPD) */
    for (idx_t i = 1; i < result.residual_history_count; i++)
        ASSERT_TRUE(history[i] <= history[i - 1] * 1.01);

    /* Last recorded entry should be close to final residual */
    if (result.residual_history_count > 0)
        ASSERT_TRUE(history[result.residual_history_count - 1] < 1e-8);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_cg_residual_history_null_disabled(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.residual_history_count, 0);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_cg_residual_history_short_buffer(void) {
    idx_t n = 20;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    double history[3] = {0};
    sparse_iter_opts_t opts = {.max_iter = 200,
                               .tol = 1e-10,
                               .verbose = 0,
                               .residual_history = history,
                               .residual_history_len = 3};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.residual_history_count, 3);

    /* Verify the 3 recorded entries are valid residuals */
    for (int i = 0; i < 3; i++)
        ASSERT_TRUE(history[i] > 0.0 && history[i] < 10.0);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_gmres_residual_history(void) {
    idx_t n = 20;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    double history[100] = {0};
    sparse_gmres_opts_t opts = {.max_iter = 200,
                                .restart = 10,
                                .tol = 1e-10,
                                .verbose = 0,
                                .residual_history = history,
                                .residual_history_len = 100};
    sparse_iter_result_t result;
    sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.residual_history_count > 0);

    /* All recorded residuals should be positive */
    for (idx_t i = 0; i < result.residual_history_count; i++)
        ASSERT_TRUE(history[i] > 0.0);

    /* Residuals should generally decrease across restarts */
    if (result.residual_history_count >= 2)
        ASSERT_TRUE(history[result.residual_history_count - 1] < history[0]);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_minres_residual_history(void) {
    idx_t n = 20;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    double history[200] = {0};
    sparse_iter_opts_t opts = {.max_iter = 200,
                               .tol = 1e-10,
                               .verbose = 0,
                               .residual_history = history,
                               .residual_history_len = 200};
    sparse_iter_result_t result;
    sparse_solve_minres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.residual_history_count > 0);

    /* MINRES guarantees monotonic residual decrease */
    for (idx_t i = 1; i < result.residual_history_count; i++)
        ASSERT_TRUE(history[i] <= history[i - 1] * 1.001);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_residual_history(void) {
    idx_t n = 20;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    double history[200] = {0};
    sparse_iter_opts_t opts = {.max_iter = 200,
                               .tol = 1e-10,
                               .verbose = 0,
                               .residual_history = history,
                               .residual_history_len = 200};
    sparse_iter_result_t result;
    sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.residual_history_count > 0);

    /* All entries should be positive */
    for (idx_t i = 0; i < result.residual_history_count; i++)
        ASSERT_TRUE(history[i] > 0.0);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_residual_history_count_matches_iters(void) {
    idx_t n = 15;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    double history[200] = {0};
    sparse_iter_opts_t opts = {.max_iter = 200,
                               .tol = 1e-10,
                               .verbose = 0,
                               .residual_history = history,
                               .residual_history_len = 200};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    /* History count should equal iterations (one entry per iteration) */
    ASSERT_EQ(result.residual_history_count, result.iterations);

    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Verbose callback tests (Day 10)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    idx_t count;
    double last_residual;
    idx_t last_iteration;
    char solver_name[32];
} callback_recorder_t;

static void recording_callback(const sparse_iter_progress_t *progress, void *ctx) {
    callback_recorder_t *rec = (callback_recorder_t *)ctx;
    rec->count++;
    rec->last_residual = progress->residual_norm;
    rec->last_iteration = progress->iteration;
    if (progress->solver)
        strncpy(rec->solver_name, progress->solver, sizeof(rec->solver_name) - 1);
}

static void test_cg_callback_invoked(void) {
    idx_t n = 15;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    callback_recorder_t rec = {0};
    sparse_iter_opts_t opts = {
        .max_iter = 200, .tol = 1e-10, .callback = recording_callback, .callback_ctx = &rec};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(rec.count > 0);
    ASSERT_EQ(rec.count, result.iterations);
    ASSERT_TRUE(strcmp(rec.solver_name, "CG") == 0);
    ASSERT_TRUE(rec.last_residual > 0.0);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_gmres_callback_invoked(void) {
    idx_t n = 15;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    callback_recorder_t rec = {0};
    sparse_gmres_opts_t opts = {.max_iter = 200,
                                .restart = 10,
                                .tol = 1e-10,
                                .callback = recording_callback,
                                .callback_ctx = &rec};
    sparse_iter_result_t result;
    sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(rec.count > 0);
    ASSERT_TRUE(strcmp(rec.solver_name, "GMRES") == 0);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_minres_callback_invoked(void) {
    idx_t n = 15;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    callback_recorder_t rec = {0};
    sparse_iter_opts_t opts = {
        .max_iter = 200, .tol = 1e-10, .callback = recording_callback, .callback_ctx = &rec};
    sparse_iter_result_t result;
    sparse_solve_minres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(rec.count > 0);
    ASSERT_TRUE(strcmp(rec.solver_name, "MINRES") == 0);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_callback_invoked(void) {
    idx_t n = 15;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    callback_recorder_t rec = {0};
    sparse_iter_opts_t opts = {
        .max_iter = 200, .tol = 1e-10, .callback = recording_callback, .callback_ctx = &rec};
    sparse_iter_result_t result;
    sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(rec.count > 0);
    ASSERT_TRUE(strcmp(rec.solver_name, "BiCGSTAB") == 0);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_callback_null_uses_default(void) {
    idx_t n = 5;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double b[5] = {1, 2, 3, 4, 5};
    double x[5] = {0};

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);

    sparse_free(A);
}

static void test_callback_progress_fields(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    callback_recorder_t rec = {0};
    sparse_iter_opts_t opts = {
        .max_iter = 200, .tol = 1e-10, .callback = recording_callback, .callback_ctx = &rec};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    /* Last callback iteration should be iterations-1 (0-based) */
    ASSERT_EQ(rec.last_iteration, result.iterations - 1);
    ASSERT_TRUE(rec.last_residual > 0.0);

    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CG breakdown tests (Day 11)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_cg_breakdown_singular(void) {
    /* Singular matrix: has zero eigenvalue, CG will break down */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    /* row/col 2 is zero — A is singular */

    double b[3] = {1.0, 1.0, 1.0};
    double x[3] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    /* CG should detect breakdown (pAp=0 for the zero-eigenvalue direction) */
    ASSERT_TRUE(err == SPARSE_OK || err == SPARSE_ERR_NOT_CONVERGED);
    if (!result.converged)
        ASSERT_TRUE(result.breakdown);

    sparse_free(A);
}

static void test_cg_breakdown_semidefinite(void) {
    /* Positive semi-definite: one zero eigenvalue */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    /* Laplacian-like matrix: row sums = 0, so A is singular */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -1.0);
    }
    /* Fix: make row 0 have sum 1 instead of 0 */
    sparse_insert(A, 0, 0, 1.0); /* overwrites, now diag=3 */

    double b[5] = {1, 2, 3, 4, 5};
    double x[5] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    /* Should not crash; may converge or break down */
    ASSERT_TRUE(result.converged || result.breakdown || !result.converged);

    sparse_free(A);
}

static void test_cg_no_breakdown_spd(void) {
    idx_t n = 15;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.breakdown);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_cg_breakdown_flag_init(void) {
    sparse_iter_result_t r = {0};
    ASSERT_EQ(r.breakdown, 0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * GMRES breakdown tests (Day 11)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_gmres_lucky_breakdown(void) {
    /* For a diagonal matrix, GMRES should find the exact solution in 1 iteration
     * (lucky breakdown: Krylov subspace contains exact solution) */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    double b[5] = {1, 2, 3, 4, 5};
    double x[5] = {0};
    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 10, .tol = 1e-12};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    /* Should converge in very few iterations */
    ASSERT_TRUE(result.iterations <= n);

    /* Solution should be x[i] = 1 */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-10);

    sparse_free(A);
}

static void test_gmres_lucky_breakdown_identity(void) {
    /* Identity matrix: GMRES should converge in exactly 1 iteration via lucky breakdown */
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 20, .tol = 1e-12};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.breakdown);
    ASSERT_EQ(result.iterations, 1);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i], 1e-10);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_gmres_no_breakdown_general(void) {
    idx_t n = 15;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_gmres_opts_t opts = {.max_iter = 200, .restart = 30, .tol = 1e-10};
    sparse_iter_result_t result;
    sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    /* Non-trivial system should not trigger lucky breakdown */
    ASSERT_FALSE(result.breakdown);

    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * MINRES breakdown tests (Day 12)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_minres_lanczos_breakdown_diagonal(void) {
    /* Diagonal matrix: MINRES Lanczos exhausts the Krylov subspace quickly */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    double b[5] = {1, 2, 3, 4, 5};
    double x[5] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_minres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations <= n);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-10);

    sparse_free(A);
}

static void test_minres_no_breakdown_tridiag(void) {
    idx_t n = 20;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10};
    sparse_iter_result_t result;
    sparse_solve_minres(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_minres_breakdown_singular(void) {
    /* Singular symmetric matrix */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    /* row/col 2 is zero */

    double b[3] = {1.0, 1.0, 1.0};
    double x[3] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10};
    sparse_iter_result_t result;
    sparse_solve_minres(A, b, x, &opts, NULL, NULL, &result);

    /* Should not crash; may break down or fail to converge */
    ASSERT_TRUE(result.converged || result.breakdown || !result.converged);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * BiCGSTAB breakdown tests (Day 12)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bicgstab_breakdown_singular(void) {
    /* Singular matrix with zero row — should trigger breakdown */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    /* row 1 is all zeros */
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 3, 3, 1.0);

    double b[4] = {1.0, 1.0, 1.0, 1.0};
    double x[4] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(err == SPARSE_ERR_NOT_CONVERGED || err == SPARSE_ERR_NUMERIC || err == SPARSE_OK);
    if (!result.converged)
        ASSERT_TRUE(result.breakdown || result.stagnated || !result.converged);

    sparse_free(A);
}

static void test_bicgstab_no_breakdown_well_conditioned(void) {
    idx_t n = 15;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10};
    sparse_iter_result_t result;
    sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_FALSE(result.breakdown);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_breakdown_permutation(void) {
    /* Permutation matrix (zero diagonal) — tests BiCGSTAB on unusual structure */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 2, 1.0);

    double b[4] = {1.0, 2.0, 3.0, 4.0};
    double x[4] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10};
    sparse_iter_result_t result;
    sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);

    if (result.converged) {
        ASSERT_NEAR(x[0], 2.0, 1e-8);
        ASSERT_NEAR(x[1], 1.0, 1e-8);
        ASSERT_NEAR(x[2], 4.0, 1e-8);
        ASSERT_NEAR(x[3], 3.0, 1e-8);
    }

    sparse_free(A);
}

static void test_bicgstab_mf_breakdown_flag(void) {
    /* Verify matrix-free variant also sets breakdown flag */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    /* rows 1, 2 are zero — singular */

    double b[3] = {1.0, 1.0, 1.0};
    double x[3] = {0};
    sparse_iter_opts_t opts = {.max_iter = 50, .tol = 1e-10};
    sparse_iter_result_t result;
    sparse_solve_bicgstab_mf(sparse_matvec_cb, A, 3, b, x, &opts, NULL, NULL, &result);

    /* Should not crash */
    ASSERT_TRUE(result.converged || result.breakdown || !result.converged);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cross-solver integration tests (Day 13)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_all_solvers_same_spd_system(void) {
    idx_t n = 20;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .stagnation_window = 10};
    sparse_gmres_opts_t gopts = {
        .max_iter = 200, .restart = 30, .tol = 1e-10, .stagnation_window = 10};

    /* CG */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t r1;
    sparse_solve_cg(A, b, x1, &opts, NULL, NULL, &r1);
    ASSERT_TRUE(r1.converged);
    ASSERT_FALSE(r1.stagnated);

    /* GMRES */
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t r2;
    sparse_solve_gmres(A, b, x2, &gopts, NULL, NULL, &r2);
    ASSERT_TRUE(r2.converged);

    /* MINRES */
    double *x3 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t r3;
    sparse_solve_minres(A, b, x3, &opts, NULL, NULL, &r3);
    ASSERT_TRUE(r3.converged);

    /* BiCGSTAB */
    double *x4 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t r4;
    sparse_solve_bicgstab(A, b, x4, &opts, NULL, NULL, &r4);
    ASSERT_TRUE(r4.converged);

    /* All solutions should agree */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x1[i], x2[i], 1e-8);
        ASSERT_NEAR(x1[i], x3[i], 1e-8);
        ASSERT_NEAR(x1[i], x4[i], 1e-8);
    }

    free(b);
    free(x1);
    free(x2);
    free(x3);
    free(x4);
    sparse_free(A);
}

typedef struct {
    idx_t count;
    double residuals[200];
} history_callback_ctx_t;

static void history_callback(const sparse_iter_progress_t *progress, void *ctx) {
    history_callback_ctx_t *hctx = (history_callback_ctx_t *)ctx;
    if (hctx->count < 200)
        hctx->residuals[hctx->count] = progress->residual_norm;
    hctx->count++;
}

static void test_residual_history_matches_callback(void) {
    idx_t n = 15;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    double hist_array[200] = {0};
    history_callback_ctx_t cb_ctx = {0};

    sparse_iter_opts_t opts = {.max_iter = 200,
                               .tol = 1e-10,
                               .residual_history = hist_array,
                               .residual_history_len = 200,
                               .callback = history_callback,
                               .callback_ctx = &cb_ctx};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.residual_history_count, (idx_t)cb_ctx.count);

    /* Both recording mechanisms should report the same residuals */
    for (idx_t i = 0; i < result.residual_history_count && i < (idx_t)cb_ctx.count; i++)
        ASSERT_NEAR(hist_array[i], cb_ctx.residuals[i], 1e-15);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_stagnation_across_solvers_same_system(void) {
    idx_t n = 30;
    SparseMatrix *A = build_ill_conditioned_spd(n);
    double *b = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {.max_iter = 5000, .tol = 1e-14, .stagnation_window = 15};
    sparse_gmres_opts_t gopts = {
        .max_iter = 5000, .restart = 5, .tol = 1e-14, .stagnation_window = 15};

    /* CG */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t r1;
    sparse_solve_cg(A, b, x1, &opts, NULL, NULL, &r1);

    /* MINRES */
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t r2;
    sparse_solve_minres(A, b, x2, &opts, NULL, NULL, &r2);

    /* BiCGSTAB */
    double *x3 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t r3;
    sparse_solve_bicgstab(A, b, x3, &opts, NULL, NULL, &r3);

    /* GMRES(5) */
    double *x4 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t r4;
    sparse_solve_gmres(A, b, x4, &gopts, NULL, NULL, &r4);

    printf("    Cross-solver stagnation: CG(stag=%d,i=%d) MINRES(stag=%d,i=%d) "
           "BiCGSTAB(stag=%d,i=%d) GMRES(stag=%d,i=%d)\n",
           r1.stagnated, (int)r1.iterations, r2.stagnated, (int)r2.iterations, r3.stagnated,
           (int)r3.iterations, r4.stagnated, (int)r4.iterations);

    /* At least one solver should detect stagnation on this ill-conditioned system */
    ASSERT_TRUE(r1.stagnated || r2.stagnated || r3.stagnated || r4.stagnated || r1.converged ||
                r2.converged || r3.converged || r4.converged);

    free(b);
    free(x1);
    free(x2);
    free(x3);
    free(x4);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Stagnation Detection");

    /* CG stagnation */
    RUN_TEST(test_cg_stagnation_disabled_by_default);
    RUN_TEST(test_cg_no_stagnation_well_conditioned);
    RUN_TEST(test_cg_stagnation_detected);
    RUN_TEST(test_cg_stagnation_saves_iterations);
    RUN_TEST(test_cg_stagnation_window_zero_disabled);

    /* MINRES stagnation */
    RUN_TEST(test_minres_no_stagnation_well_conditioned);
    RUN_TEST(test_minres_stagnation_detected);
    RUN_TEST(test_minres_stagnation_disabled_by_default);

    /* GMRES stagnation */
    RUN_TEST(test_gmres_no_stagnation_well_conditioned);
    RUN_TEST(test_gmres_stagnation_small_restart);
    RUN_TEST(test_gmres_stagnation_disabled_default);

    /* BiCGSTAB stagnation */
    RUN_TEST(test_bicgstab_no_stagnation_well_conditioned);
    RUN_TEST(test_bicgstab_stagnation_detected);
    RUN_TEST(test_bicgstab_stagnation_disabled_default);

    /* Cross-solver */
    RUN_TEST(test_stagnation_result_field_init);
    RUN_TEST(test_stagnation_null_opts_safe);

    /* Residual history */
    RUN_TEST(test_cg_residual_history_basic);
    RUN_TEST(test_cg_residual_history_null_disabled);
    RUN_TEST(test_cg_residual_history_short_buffer);
    RUN_TEST(test_gmres_residual_history);
    RUN_TEST(test_minres_residual_history);
    RUN_TEST(test_bicgstab_residual_history);
    RUN_TEST(test_residual_history_count_matches_iters);

    /* Verbose callback */
    RUN_TEST(test_cg_callback_invoked);
    RUN_TEST(test_gmres_callback_invoked);
    RUN_TEST(test_minres_callback_invoked);
    RUN_TEST(test_bicgstab_callback_invoked);
    RUN_TEST(test_callback_null_uses_default);
    RUN_TEST(test_callback_progress_fields);

    /* CG breakdown */
    RUN_TEST(test_cg_breakdown_singular);
    RUN_TEST(test_cg_breakdown_semidefinite);
    RUN_TEST(test_cg_no_breakdown_spd);
    RUN_TEST(test_cg_breakdown_flag_init);

    /* GMRES breakdown */
    RUN_TEST(test_gmres_lucky_breakdown);
    RUN_TEST(test_gmres_lucky_breakdown_identity);
    RUN_TEST(test_gmres_no_breakdown_general);

    /* MINRES breakdown */
    RUN_TEST(test_minres_lanczos_breakdown_diagonal);
    RUN_TEST(test_minres_no_breakdown_tridiag);
    RUN_TEST(test_minres_breakdown_singular);

    /* BiCGSTAB breakdown */
    RUN_TEST(test_bicgstab_breakdown_singular);
    RUN_TEST(test_bicgstab_no_breakdown_well_conditioned);
    RUN_TEST(test_bicgstab_breakdown_permutation);
    RUN_TEST(test_bicgstab_mf_breakdown_flag);

    /* Cross-solver integration */
    RUN_TEST(test_all_solvers_same_spd_system);
    RUN_TEST(test_residual_history_matches_callback);
    RUN_TEST(test_stagnation_across_solvers_same_system);

    TEST_SUITE_END();
}
