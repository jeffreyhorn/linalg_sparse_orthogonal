/*
 * bench_scaling.c — Measure factorization and solve scaling
 *
 * Generates tridiagonal matrices of increasing size and reports
 * timing, nnz, and memory for each. Tridiagonal should show
 * near-linear scaling since there is no fill-in.
 */
#include "sparse_matrix.h"
#include "sparse_lu.h"
#include "sparse_vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static SparseMatrix *make_tridiag(idx_t n)
{
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0);
        if (i > 0)     sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1) sparse_insert(A, i, i + 1, -1.0);
    }
    return A;
}

static SparseMatrix *make_dense_random(idx_t n, unsigned seed)
{
    srand(seed);
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            double v = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
            if (i == j) v += (double)n;  /* diagonal dominance */
            sparse_insert(A, i, j, v);
        }
    }
    return A;
}

static void run_scaling(const char *label,
                        SparseMatrix *(*gen)(idx_t),
                        const idx_t *sizes, int nsizes,
                        sparse_pivot_t pivot, int repeats)
{
    printf("=== %s (pivot=%s, repeats=%d) ===\n", label,
           pivot == SPARSE_PIVOT_COMPLETE ? "complete" : "partial", repeats);
    printf("%-8s %-8s %-10s %-12s %-12s %-12s %-10s %-10s\n",
           "n", "nnz", "nnz_LU", "factor(ms)", "solve(ms)", "spmv(ms)",
           "memory", "residual");
    printf("-------- -------- ---------- ------------ ------------ "
           "------------ ---------- ----------\n");

    for (int s = 0; s < nsizes; s++) {
        idx_t n = sizes[s];
        SparseMatrix *A = gen(n);
        if (!A) { printf("n=%d: alloc failed\n", (int)n); continue; }

        idx_t nnz_orig = sparse_nnz(A);

        /* RHS */
        double *b = malloc((size_t)n * sizeof(double));
        double *x = malloc((size_t)n * sizeof(double));
        double *r = malloc((size_t)n * sizeof(double));
        for (idx_t i = 0; i < n; i++) b[i] = 1.0;

        /* SpMV timing */
        double t0 = wall_time();
        for (int rep = 0; rep < repeats; rep++)
            sparse_matvec(A, b, r);
        double t_spmv = (wall_time() - t0) / repeats * 1000.0;

        /* Factor + solve timing */
        double t_factor = 0, t_solve = 0;
        idx_t nnz_lu = 0;
        double residual = 0;

        for (int rep = 0; rep < repeats; rep++) {
            SparseMatrix *LU = sparse_copy(A);
            t0 = wall_time();
            sparse_lu_factor(LU, pivot, 1e-12);
            t_factor += (wall_time() - t0) * 1000.0;

            if (rep == 0) nnz_lu = sparse_nnz(LU);

            t0 = wall_time();
            sparse_lu_solve(LU, b, x);
            t_solve += (wall_time() - t0) * 1000.0;

            if (rep == repeats - 1) {
                sparse_matvec(A, x, r);
                for (idx_t i = 0; i < n; i++) r[i] -= b[i];
                residual = vec_norminf(r, n);
            }
            sparse_free(LU);
        }

        printf("%-8d %-8d %-10d %-12.3f %-12.3f %-12.3f %-10zu %.3e\n",
               (int)n, (int)nnz_orig, (int)nnz_lu,
               t_factor / repeats, t_solve / repeats, t_spmv,
               sparse_memory_usage(A), residual);

        free(b); free(x); free(r);
        sparse_free(A);
    }
    printf("\n");
}

/* Wrapper matching the gen signature for dense */
static idx_t dense_seed_n;
static SparseMatrix *make_dense_wrapper(idx_t n)
{
    dense_seed_n = n;
    return make_dense_random(n, 42);
}

int main(void)
{
    printf("Sparse LU Scaling Benchmarks\n");
    printf("============================\n\n");

    /* Tridiagonal scaling */
    idx_t tri_sizes[] = {100, 500, 1000, 2000, 5000};
    int ntri = (int)(sizeof(tri_sizes) / sizeof(tri_sizes[0]));

    run_scaling("Tridiagonal (-1, 2, -1)", make_tridiag,
                tri_sizes, ntri, SPARSE_PIVOT_PARTIAL, 3);

    run_scaling("Tridiagonal (-1, 2, -1)", make_tridiag,
                tri_sizes, ntri, SPARSE_PIVOT_COMPLETE, 3);

    /* Dense scaling (smaller sizes) */
    idx_t dense_sizes[] = {10, 20, 50, 100, 200};
    int ndense = (int)(sizeof(dense_sizes) / sizeof(dense_sizes[0]));

    run_scaling("Dense random (diag-dominant)", make_dense_wrapper,
                dense_sizes, ndense, SPARSE_PIVOT_COMPLETE, 3);

    return 0;
}
