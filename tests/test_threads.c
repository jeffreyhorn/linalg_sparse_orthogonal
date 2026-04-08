#include "sparse_cholesky.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Thread helpers
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int thread_id;
    int success; /* 1 = pass, 0 = fail */
    double max_error;
} thread_result_t;

/* ═══════════════════════════════════════════════════════════════════════
 * Test: independent LU factor+solve on separate matrices per thread
 * ═══════════════════════════════════════════════════════════════════════ */

static void *thread_independent_lu(void *arg) {
    thread_result_t *res = (thread_result_t *)arg;
    res->success = 0;
    res->max_error = 0.0;

    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;

    /* Build a diag-dominant SPD tridiagonal — different per thread */
    double diag_val = 4.0 + (double)res->thread_id;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag_val);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    /* b = A * ones */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    if (!ones || !b || !x) {
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return NULL;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* Factor and solve */
    SparseMatrix *LU = sparse_copy(A);
    if (!LU) {
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return NULL;
    }

    if (sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12) != SPARSE_OK) {
        sparse_free(LU);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return NULL;
    }

    if (sparse_lu_solve(LU, b, x) != SPARSE_OK) {
        sparse_free(LU);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return NULL;
    }

    /* Verify x ≈ ones */
    for (idx_t i = 0; i < n; i++) {
        double err = fabs(x[i] - 1.0);
        if (err > res->max_error)
            res->max_error = err;
    }

    res->success = (res->max_error < 1e-10) ? 1 : 0;

    sparse_free(LU);
    free(ones);
    free(b);
    free(x);
    sparse_free(A);
    return NULL;
}

static void test_independent_lu_threads(void) {
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
        printf("    thread %d: success=%d max_error=%.2e\n", t, results[t].success,
               results[t].max_error);
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

static void *thread_concurrent_solve(void *arg) {
    shared_solve_arg_t *sa = (shared_solve_arg_t *)arg;
    sa->success = 0;
    sa->max_error = 0.0;

    idx_t n = sparse_rows(sa->A);

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    if (!b || !x || !r) {
        free(b);
        free(x);
        free(r);
        return NULL;
    }

    /* Use a different RHS for each thread */
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1 + sa->thread_id * 100);

    for (int iter = 0; iter < sa->iterations; iter++) {
        if (sparse_lu_solve(sa->LU, b, x) != SPARSE_OK) {
            free(b);
            free(x);
            free(r);
            return NULL;
        }

        /* Verify residual */
        sparse_matvec(sa->A, x, r);
        for (idx_t i = 0; i < n; i++) {
            double err = fabs(r[i] - b[i]);
            if (err > sa->max_error)
                sa->max_error = err;
        }
    }

    sa->success = (sa->max_error < 1e-8) ? 1 : 0;

    free(b);
    free(x);
    free(r);
    return NULL;
}

static void test_concurrent_solve_shared(void) {
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
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
        printf("    thread %d: %d iters, success=%d max_error=%.2e\n", t, args[t].iterations,
               args[t].success, args[t].max_error);
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

static void *thread_concurrent_cholesky_solve(void *arg) {
    chol_solve_arg_t *sa = (chol_solve_arg_t *)arg;
    sa->success = 0;
    sa->max_error = 0.0;

    idx_t n = sparse_rows(sa->A);

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    if (!b || !x || !r) {
        free(b);
        free(x);
        free(r);
        return NULL;
    }

    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1 + sa->thread_id * 50);

    for (int iter = 0; iter < sa->iterations; iter++) {
        if (sparse_cholesky_solve(sa->L, b, x) != SPARSE_OK) {
            free(b);
            free(x);
            free(r);
            return NULL;
        }

        sparse_matvec(sa->A, x, r);
        for (idx_t i = 0; i < n; i++) {
            double err = fabs(r[i] - b[i]);
            if (err > sa->max_error)
                sa->max_error = err;
        }
    }

    sa->success = (sa->max_error < 1e-8) ? 1 : 0;

    free(b);
    free(x);
    free(r);
    return NULL;
}

static void test_concurrent_cholesky_solve(void) {
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
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
        printf("    thread %d: %d iters, success=%d max_error=%.2e\n", t, args[t].iterations,
               args[t].success, args[t].max_error);
        ASSERT_TRUE(args[t].success);
    }

    sparse_free(A);
    sparse_free(L);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Stress tests: more threads, more iterations
 * ═══════════════════════════════════════════════════════════════════════ */

#define STRESS_THREADS 8
#define STRESS_ITERS 1000

/* 8 threads × 1000 iterations on shared LU */
static void test_lu_solve_stress(void) {
    idx_t n = 30;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
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
        if (!args[t].success)
            all_pass = 0;
    }
    printf("    LU stress: %d threads × %d iters, all_pass=%d\n", STRESS_THREADS, STRESS_ITERS,
           all_pass);
    ASSERT_TRUE(all_pass);

    sparse_free(A);
    sparse_free(LU);
}

/* 8 threads × 1000 iterations on shared Cholesky */
static void test_cholesky_solve_stress(void) {
    idx_t n = 30;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
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
        if (!args[t].success)
            all_pass = 0;
    }
    printf("    Cholesky stress: %d threads × %d iters, all_pass=%d\n", STRESS_THREADS,
           STRESS_ITERS, all_pass);
    ASSERT_TRUE(all_pass);

    sparse_free(A);
    sparse_free(L);
}

/* Concurrent independent create+factor+solve (8 threads) */
static void test_independent_stress(void) {
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
        if (!results[t].success)
            all_pass = 0;
    }
    printf("    Independent stress: %d threads, all_pass=%d\n", STRESS_THREADS, all_pass);
    ASSERT_TRUE(all_pass);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: concurrent norminf on the same matrix (Sprint 11 Day 6)
 *
 * Multiple threads call sparse_norminf() on the same unfactored matrix.
 * With _Atomic cached_norm, this should be TSan-clean.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    SparseMatrix *mat;
    int thread_id;
    int success;
    double norm_result;
    int iterations;
} norminf_arg_t;

static void *thread_concurrent_norminf(void *arg) {
    norminf_arg_t *na = (norminf_arg_t *)arg;
    na->success = 1;
    na->norm_result = 0.0;

    for (int iter = 0; iter < na->iterations; iter++) {
        double norm;
        sparse_err_t err = sparse_norminf(na->mat, &norm);
        if (err != SPARSE_OK) {
            na->success = 0;
            return NULL;
        }
        na->norm_result = norm;
    }
    return NULL;
}

static void test_concurrent_norminf(void) {
    /* Build a 20×20 tridiagonal matrix */
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    /* Expected norm: max row sum = 12.0 (interior row: 1+10+1) */
    double expected_norm;
    ASSERT_ERR(sparse_norminf(A, &expected_norm), SPARSE_OK);
    ASSERT_NEAR(expected_norm, 12.0, 1e-14);

    /* Invalidate cache so threads will race to compute it.
     * sparse_copy preserves cached_norm, so we insert a dummy entry
     * (which invalidates the cache) and then remove it to restore
     * the original matrix structure. */
    SparseMatrix *A2 = sparse_copy(A);
    ASSERT_NOT_NULL(A2);
    sparse_insert(A2, 0, (idx_t)(n - 1), 0.001);
    sparse_remove(A2, 0, (idx_t)(n - 1));

    int nthreads = 4;
    pthread_t threads[4];
    norminf_arg_t args[4];

    for (int t = 0; t < nthreads; t++) {
        args[t].mat = A2;
        args[t].thread_id = t;
        args[t].iterations = 1000;
        int rc = pthread_create(&threads[t], NULL, thread_concurrent_norminf, &args[t]);
        ASSERT_EQ(rc, 0);
    }

    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        ASSERT_TRUE(args[t].success);
        ASSERT_NEAR(args[t].norm_result, expected_norm, 1e-14);
    }
    printf("    concurrent norminf: %d threads × %d iters, all agree on norm=%.1f\n", nthreads,
           args[0].iterations, expected_norm);

    sparse_free(A);
    sparse_free(A2);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: concurrent norminf + solve (Sprint 11 Day 6)
 *
 * One thread repeatedly calls sparse_norminf() on a factored matrix,
 * while other threads solve concurrently. Tests that cached_norm writes
 * during norminf don't race with solve's read of factor_norm.
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_concurrent_norminf_and_solve(void) {
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    /* Thread 0: concurrent norminf on the factored matrix */
    norminf_arg_t norm_arg;
    norm_arg.mat = LU;
    norm_arg.thread_id = 0;
    norm_arg.iterations = 500;

    /* Threads 1-3: concurrent solves */
    shared_solve_arg_t solve_args[3];
    pthread_t threads[4];

    int rc = pthread_create(&threads[0], NULL, thread_concurrent_norminf, &norm_arg);
    ASSERT_EQ(rc, 0);

    for (int t = 0; t < 3; t++) {
        solve_args[t].LU = LU;
        solve_args[t].A = A;
        solve_args[t].thread_id = t + 1;
        solve_args[t].iterations = 500;
        rc = pthread_create(&threads[t + 1], NULL, thread_concurrent_solve, &solve_args[t]);
        ASSERT_EQ(rc, 0);
    }

    for (int t = 0; t < 4; t++)
        pthread_join(threads[t], NULL);

    ASSERT_TRUE(norm_arg.success);
    for (int t = 0; t < 3; t++)
        ASSERT_TRUE(solve_args[t].success);

    printf("    norminf+solve: 1 norminf thread + 3 solve threads, all clean\n");

    sparse_free(A);
    sparse_free(LU);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Concurrent insert test (only safe with SPARSE_MUTEX — pool is shared)
 * ═══════════════════════════════════════════════════════════════════════ */

#ifdef SPARSE_MUTEX

typedef struct {
    SparseMatrix *mat;
    int thread_id;
    idx_t start_row;
    idx_t end_row;
    int success;
} insert_arg_t;

static void *thread_concurrent_insert(void *arg) {
    insert_arg_t *ia = (insert_arg_t *)arg;
    ia->success = 1;

    /* Each thread inserts entries into its own row AND column range to
     * avoid logical conflicts. However, the pool allocator and mat->nnz
     * are shared, so SPARSE_MUTEX is required for thread safety. */
    for (idx_t i = ia->start_row; i < ia->end_row; i++) {
        sparse_err_t err = sparse_insert(ia->mat, i, ia->start_row, (double)(i + 1));
        if (err != SPARSE_OK) {
            ia->success = 0;
            return NULL;
        }
        err = sparse_insert(ia->mat, i, i, (double)(i + 1) * 10.0);
        if (err != SPARSE_OK) {
            ia->success = 0;
            return NULL;
        }
    }
    return NULL;
}

static void test_concurrent_insert(void) {
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
        if (!args[t].success)
            all_pass = 0;
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
                ASSERT_NEAR(sparse_get_phys(A, i, i), (double)(i + 1) * 10.0, 0.0);
            } else {
                ASSERT_NEAR(sparse_get_phys(A, i, sr), (double)(i + 1), 0.0);
                ASSERT_NEAR(sparse_get_phys(A, i, i), (double)(i + 1) * 10.0, 0.0);
            }
        }
    }

    printf("    concurrent insert: %d threads, %d rows, nnz=%d, all correct\n", nthreads, (int)n,
           (int)sparse_nnz(A));

    sparse_free(A);
}

#endif /* SPARSE_MUTEX */

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Thread Safety Tests");

    RUN_TEST(test_independent_lu_threads);
    RUN_TEST(test_concurrent_solve_shared);
    RUN_TEST(test_concurrent_cholesky_solve);

    /* Stress tests */
    RUN_TEST(test_lu_solve_stress);
    RUN_TEST(test_cholesky_solve_stress);
    RUN_TEST(test_independent_stress);

    /* Concurrent norminf (Sprint 11 Day 6) */
    RUN_TEST(test_concurrent_norminf);
    RUN_TEST(test_concurrent_norminf_and_solve);

    /* Concurrent insert — only safe with SPARSE_MUTEX (pool/nnz are shared state) */
#ifdef SPARSE_MUTEX
    RUN_TEST(test_concurrent_insert);
#else
    printf("  [SKIP] test_concurrent_insert (requires -DSPARSE_MUTEX)\n");
#endif

    TEST_SUITE_END();
}
