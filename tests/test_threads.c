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
 * Stress tests: more threads, more iterations
 * ═══════════════════════════════════════════════════════════════════════ */

#define STRESS_THREADS 8
#define STRESS_ITERS   1000

/* 8 threads × 1000 iterations on shared LU */
static void test_lu_solve_stress(void)
{
    idx_t n = 30;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0) { sparse_insert(A, i, i-1, -1.0); sparse_insert(A, i-1, i, -1.0); }
    }

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    pthread_t threads[STRESS_THREADS];
    shared_solve_arg_t args[STRESS_THREADS];

    for (int t = 0; t < STRESS_THREADS; t++) {
        args[t].LU = LU;
        args[t].A = A;
        args[t].thread_id = t;
        args[t].iterations = STRESS_ITERS;
        int rc = pthread_create(&threads[t], NULL, thread_concurrent_solve, &args[t]);
        ASSERT_EQ(rc, 0);
    }

    int all_pass = 1;
    for (int t = 0; t < STRESS_THREADS; t++) {
        pthread_join(threads[t], NULL);
        if (!args[t].success) all_pass = 0;
    }
    printf("    LU stress: %d threads × %d iters, all_pass=%d\n",
           STRESS_THREADS, STRESS_ITERS, all_pass);
    ASSERT_TRUE(all_pass);

    sparse_free(A);
    sparse_free(LU);
}

/* 8 threads × 1000 iterations on shared Cholesky */
static void test_cholesky_solve_stress(void)
{
    idx_t n = 30;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0) { sparse_insert(A, i, i-1, -1.0); sparse_insert(A, i-1, i, -1.0); }
    }

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    pthread_t threads[STRESS_THREADS];
    chol_solve_arg_t args[STRESS_THREADS];

    for (int t = 0; t < STRESS_THREADS; t++) {
        args[t].L = L;
        args[t].A = A;
        args[t].thread_id = t;
        args[t].iterations = STRESS_ITERS;
        int rc = pthread_create(&threads[t], NULL, thread_concurrent_cholesky_solve, &args[t]);
        ASSERT_EQ(rc, 0);
    }

    int all_pass = 1;
    for (int t = 0; t < STRESS_THREADS; t++) {
        pthread_join(threads[t], NULL);
        if (!args[t].success) all_pass = 0;
    }
    printf("    Cholesky stress: %d threads × %d iters, all_pass=%d\n",
           STRESS_THREADS, STRESS_ITERS, all_pass);
    ASSERT_TRUE(all_pass);

    sparse_free(A);
    sparse_free(L);
}

/* Concurrent independent create+factor+solve (8 threads) */
static void test_independent_stress(void)
{
    pthread_t threads[STRESS_THREADS];
    thread_result_t results[STRESS_THREADS];

    for (int t = 0; t < STRESS_THREADS; t++) {
        results[t].thread_id = t;
        int rc = pthread_create(&threads[t], NULL, thread_independent_lu, &results[t]);
        ASSERT_EQ(rc, 0);
    }

    int all_pass = 1;
    for (int t = 0; t < STRESS_THREADS; t++) {
        pthread_join(threads[t], NULL);
        if (!results[t].success) all_pass = 0;
    }
    printf("    Independent stress: %d threads, all_pass=%d\n",
           STRESS_THREADS, all_pass);
    ASSERT_TRUE(all_pass);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Concurrent insert test (exercises SPARSE_MUTEX when enabled)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    SparseMatrix *mat;
    int thread_id;
    idx_t start_row;
    idx_t end_row;
    int success;
} insert_arg_t;

static void *thread_concurrent_insert(void *arg)
{
    insert_arg_t *ia = (insert_arg_t *)arg;
    ia->success = 1;

    /* Each thread inserts entries into its own row AND column range so that
     * no two threads touch the same row-header or column-header list.
     * This makes the test safe even without SPARSE_MUTEX. */
    for (idx_t i = ia->start_row; i < ia->end_row; i++) {
        sparse_err_t err = sparse_insert(ia->mat, i, ia->start_row,
                                         (double)(i + 1));
        if (err != SPARSE_OK) { ia->success = 0; return NULL; }
        err = sparse_insert(ia->mat, i, i, (double)(i + 1) * 10.0);
        if (err != SPARSE_OK) { ia->success = 0; return NULL; }
    }
    return NULL;
}

static void test_concurrent_insert(void)
{
    idx_t n = 40;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);

    int nthreads = 4;
    pthread_t threads[4];
    insert_arg_t args[4];
    idx_t rows_per_thread = n / nthreads;

    for (int t = 0; t < nthreads; t++) {
        args[t].mat = A;
        args[t].thread_id = t;
        args[t].start_row = (idx_t)t * rows_per_thread;
        args[t].end_row = (t == nthreads - 1) ? n : args[t].start_row + rows_per_thread;
        int rc = pthread_create(&threads[t], NULL, thread_concurrent_insert, &args[t]);
        ASSERT_EQ(rc, 0);
    }

    int all_pass = 1;
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        if (!args[t].success) all_pass = 0;
    }
    ASSERT_TRUE(all_pass);

    /* Verify all entries were inserted correctly.
     * Each thread inserted into column start_row (its first row) and the
     * diagonal.  Row 0: col 0 is both — diagonal wins (last write). */
    for (int t = 0; t < nthreads; t++) {
        idx_t sr = args[t].start_row;
        idx_t er = args[t].end_row;
        for (idx_t i = sr; i < er; i++) {
            if (i == sr) {
                /* col sr overlaps with diagonal when i == sr */
                ASSERT_NEAR(sparse_get_phys(A, i, i),
                            (double)(i + 1) * 10.0, 0.0);
            } else {
                ASSERT_NEAR(sparse_get_phys(A, i, sr),
                            (double)(i + 1), 0.0);
                ASSERT_NEAR(sparse_get_phys(A, i, i),
                            (double)(i + 1) * 10.0, 0.0);
            }
        }
    }

    printf("    concurrent insert: %d threads, %d rows, nnz=%d, all correct\n",
           nthreads, (int)n, (int)sparse_nnz(A));

    sparse_free(A);
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

    /* Stress tests */
    RUN_TEST(test_lu_solve_stress);
    RUN_TEST(test_cholesky_solve_stress);
    RUN_TEST(test_independent_stress);

    /* Concurrent insert (non-overlapping rows — safe even without mutex) */
    RUN_TEST(test_concurrent_insert);

    TEST_SUITE_END();
}
