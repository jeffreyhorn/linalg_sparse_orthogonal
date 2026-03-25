#include "sparse_matrix.h"
#include "sparse_lu.h"
#include "sparse_cholesky.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <pthread.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Thread helpers
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int thread_id;
    int success;       /* 1 = pass, 0 = fail */
    double max_error;
} thread_result_t;

/* ═══════════════════════════════════════════════════════════════════════
 * Test: independent LU factor+solve on separate matrices per thread
 * ═══════════════════════════════════════════════════════════════════════ */

static void *thread_independent_lu(void *arg)
{
    thread_result_t *res = (thread_result_t *)arg;
    res->success = 0;
    res->max_error = 0.0;

    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;

    /* Build a diag-dominant SPD tridiagonal — different per thread */
    double diag_val = 4.0 + (double)res->thread_id;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag_val);
        if (i > 0) { sparse_insert(A, i, i-1, -1.0); sparse_insert(A, i-1, i, -1.0); }
    }

    /* b = A * ones */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b    = malloc((size_t)n * sizeof(double));
    double *x    = malloc((size_t)n * sizeof(double));
    if (!ones || !b || !x) { free(ones); free(b); free(x); sparse_free(A); return NULL; }
    for (idx_t i = 0; i < n; i++) ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* Factor and solve */
    SparseMatrix *LU = sparse_copy(A);
    if (!LU) { free(ones); free(b); free(x); sparse_free(A); return NULL; }

    if (sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12) != SPARSE_OK) {
        sparse_free(LU); free(ones); free(b); free(x); sparse_free(A);
        return NULL;
    }

    if (sparse_lu_solve(LU, b, x) != SPARSE_OK) {
        sparse_free(LU); free(ones); free(b); free(x); sparse_free(A);
        return NULL;
    }

    /* Verify x ≈ ones */
    for (idx_t i = 0; i < n; i++) {
        double err = fabs(x[i] - 1.0);
        if (err > res->max_error) res->max_error = err;
    }

    res->success = (res->max_error < 1e-10) ? 1 : 0;

    sparse_free(LU); free(ones); free(b); free(x); sparse_free(A);
    return NULL;
}

static void test_independent_lu_threads(void)
{
    int nthreads = 4;
    pthread_t threads[4];
    thread_result_t results[4];

    for (int t = 0; t < nthreads; t++) {
        results[t].thread_id = t;
        int rc = pthread_create(&threads[t], NULL, thread_independent_lu, &results[t]);
        ASSERT_EQ(rc, 0);
    }

    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        printf("    thread %d: success=%d max_error=%.2e\n",
               t, results[t].success, results[t].max_error);
        ASSERT_TRUE(results[t].success);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: concurrent solves on the same factored matrix
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    const SparseMatrix *LU;
    const SparseMatrix *A;
    int thread_id;
    int success;
    double max_error;
    int iterations;
} shared_solve_arg_t;

static void *thread_concurrent_solve(void *arg)
{
    shared_solve_arg_t *sa = (shared_solve_arg_t *)arg;
    sa->success = 0;
    sa->max_error = 0.0;

    idx_t n = sparse_rows(sa->A);

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    if (!b || !x || !r) { free(b); free(x); free(r); return NULL; }

    /* Use a different RHS for each thread */
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1 + sa->thread_id * 100);

    for (int iter = 0; iter < sa->iterations; iter++) {
        if (sparse_lu_solve(sa->LU, b, x) != SPARSE_OK) {
            free(b); free(x); free(r);
            return NULL;
        }

        /* Verify residual */
        sparse_matvec(sa->A, x, r);
        for (idx_t i = 0; i < n; i++) {
            double err = fabs(r[i] - b[i]);
            if (err > sa->max_error) sa->max_error = err;
        }
    }

    sa->success = (sa->max_error < 1e-8) ? 1 : 0;

    free(b); free(x); free(r);
    return NULL;
}

static void test_concurrent_solve_shared(void)
{
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0) { sparse_insert(A, i, i-1, -1.0); sparse_insert(A, i-1, i, -1.0); }
    }

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    int nthreads = 4;
    pthread_t threads[4];
    shared_solve_arg_t args[4];

    for (int t = 0; t < nthreads; t++) {
        args[t].LU = LU;
        args[t].A = A;
        args[t].thread_id = t;
        args[t].iterations = 100;
        int rc = pthread_create(&threads[t], NULL, thread_concurrent_solve, &args[t]);
        ASSERT_EQ(rc, 0);
    }

    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        printf("    thread %d: %d iters, success=%d max_error=%.2e\n",
               t, args[t].iterations, args[t].success, args[t].max_error);
        ASSERT_TRUE(args[t].success);
    }

    sparse_free(A);
    sparse_free(LU);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: concurrent Cholesky solves on shared factored matrix
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    const SparseMatrix *L;
    const SparseMatrix *A;
    int thread_id;
    int success;
    double max_error;
    int iterations;
} chol_solve_arg_t;

static void *thread_concurrent_cholesky_solve(void *arg)
{
    chol_solve_arg_t *sa = (chol_solve_arg_t *)arg;
    sa->success = 0;
    sa->max_error = 0.0;

    idx_t n = sparse_rows(sa->A);

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    if (!b || !x || !r) { free(b); free(x); free(r); return NULL; }

    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1 + sa->thread_id * 50);

    for (int iter = 0; iter < sa->iterations; iter++) {
        if (sparse_cholesky_solve(sa->L, b, x) != SPARSE_OK) {
            free(b); free(x); free(r);
            return NULL;
        }

        sparse_matvec(sa->A, x, r);
        for (idx_t i = 0; i < n; i++) {
            double err = fabs(r[i] - b[i]);
            if (err > sa->max_error) sa->max_error = err;
        }
    }

    sa->success = (sa->max_error < 1e-8) ? 1 : 0;

    free(b); free(x); free(r);
    return NULL;
}

static void test_concurrent_cholesky_solve(void)
{
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0) { sparse_insert(A, i, i-1, -1.0); sparse_insert(A, i-1, i, -1.0); }
    }

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    int nthreads = 4;
    pthread_t threads[4];
    chol_solve_arg_t args[4];

    for (int t = 0; t < nthreads; t++) {
        args[t].L = L;
        args[t].A = A;
        args[t].thread_id = t;
        args[t].iterations = 100;
        int rc = pthread_create(&threads[t], NULL, thread_concurrent_cholesky_solve, &args[t]);
        ASSERT_EQ(rc, 0);
    }

    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        printf("    thread %d: %d iters, success=%d max_error=%.2e\n",
               t, args[t].iterations, args[t].success, args[t].max_error);
        ASSERT_TRUE(args[t].success);
    }

    sparse_free(A);
    sparse_free(L);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Thread Safety Tests");

    RUN_TEST(test_independent_lu_threads);
    RUN_TEST(test_concurrent_solve_shared);
    RUN_TEST(test_concurrent_cholesky_solve);

    TEST_SUITE_END();
}
