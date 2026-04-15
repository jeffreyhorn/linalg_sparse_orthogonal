/*
 * bench_bicgstab.c — BiCGSTAB vs GMRES comparison benchmark
 *
 * Usage:  ./bench_bicgstab
 *
 * Compares BiCGSTAB and GMRES on SuiteSparse nonsymmetric matrices,
 * with and without ILU(0) preconditioning. Reports iteration count,
 * time, and final residual for each combination.
 */
#define _POSIX_C_SOURCE 199309L
#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_matrix.h"
#include "sparse_vector.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double compute_rel_residual(const SparseMatrix *A, const double *b, const double *x,
                                   idx_t n) {
    double *Ax = calloc((size_t)n, sizeof(double));
    if (!Ax)
        return -1.0;
    sparse_matvec(A, x, Ax);
    double rnorm = 0.0, bnorm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ri = b[i] - Ax[i];
        rnorm += ri * ri;
        bnorm += b[i] * b[i];
    }
    free(Ax);
    return (bnorm > 0.0) ? sqrt(rnorm / bnorm) : sqrt(rnorm);
}

typedef struct {
    const char *name;
    const char *path;
} matrix_entry_t;

static void run_comparison(const char *name, SparseMatrix *A) {
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    printf("  %-12s (n=%d)\n", name, (int)n);
    printf("  %-30s %8s %10s %12s %8s\n", "Config", "Iters", "Time(ms)", "Residual", "Conv");

    /* --- Unpreconditioned --- */

    /* BiCGSTAB */
    {
        double *x = calloc((size_t)n, sizeof(double));
        sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-8};
        sparse_iter_result_t result;
        double t0 = wall_time();
        sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
        double dt = (wall_time() - t0) * 1000.0;
        double res = compute_rel_residual(A, b, x, n);
        printf("  %-30s %8d %10.2f %12.3e %8s\n", "BiCGSTAB (no precond)", (int)result.iterations,
               dt, res, result.converged ? "yes" : "no");
        free(x);
    }

    /* GMRES(30) */
    {
        double *x = calloc((size_t)n, sizeof(double));
        sparse_gmres_opts_t opts = {.max_iter = 2000, .restart = 30, .tol = 1e-8};
        sparse_iter_result_t result;
        double t0 = wall_time();
        sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);
        double dt = (wall_time() - t0) * 1000.0;
        double res = compute_rel_residual(A, b, x, n);
        printf("  %-30s %8d %10.2f %12.3e %8s\n", "GMRES(30) (no precond)", (int)result.iterations,
               dt, res, result.converged ? "yes" : "no");
        free(x);
    }

    /* --- ILU(0) preconditioned --- */
    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
    if (ferr == SPARSE_OK) {
        /* BiCGSTAB + ILU */
        {
            double *x = calloc((size_t)n, sizeof(double));
            sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-8};
            sparse_iter_result_t result;
            double t0 = wall_time();
            sparse_solve_bicgstab(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);
            double dt = (wall_time() - t0) * 1000.0;
            double res = compute_rel_residual(A, b, x, n);
            printf("  %-30s %8d %10.2f %12.3e %8s\n", "BiCGSTAB + ILU(0)", (int)result.iterations,
                   dt, res, result.converged ? "yes" : "no");
            free(x);
        }

        /* GMRES(30) + ILU */
        {
            double *x = calloc((size_t)n, sizeof(double));
            sparse_gmres_opts_t opts = {.max_iter = 2000, .restart = 30, .tol = 1e-8};
            sparse_iter_result_t result;
            double t0 = wall_time();
            sparse_solve_gmres(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);
            double dt = (wall_time() - t0) * 1000.0;
            double res = compute_rel_residual(A, b, x, n);
            printf("  %-30s %8d %10.2f %12.3e %8s\n", "GMRES(30) + ILU(0)", (int)result.iterations,
                   dt, res, result.converged ? "yes" : "no");
            free(x);
        }
        sparse_ilu_free(&ilu);
    } else {
        printf("  %-30s  (ILU(0) factor failed — skipped)\n", "ILU(0) precond");
    }

    printf("\n");
    free(x_exact);
    free(b);
}

int main(void) {
    printf("=== BiCGSTAB vs GMRES Comparison Benchmark ===\n\n");

    matrix_entry_t matrices[] = {
        {"steam1", SS_DIR "/steam1.mtx"},
        {"orsirr_1", SS_DIR "/orsirr_1.mtx"},
    };
    int nmat = (int)(sizeof(matrices) / sizeof(matrices[0]));

    for (int m = 0; m < nmat; m++) {
        SparseMatrix *A = NULL;
        sparse_err_t err = sparse_load_mm(&A, matrices[m].path);
        if (err != SPARSE_OK) {
            printf("  [SKIP] %s: load failed\n\n", matrices[m].name);
            continue;
        }
        run_comparison(matrices[m].name, A);
        sparse_free(A);
    }

    /* Synthetic tridiagonal */
    {
        idx_t n = 200;
        SparseMatrix *A = sparse_create(n, n);
        for (idx_t i = 0; i < n; i++) {
            sparse_insert(A, i, i, 4.0);
            if (i > 0)
                sparse_insert(A, i, i - 1, -2.0);
            if (i < n - 1)
                sparse_insert(A, i, i + 1, -1.0);
        }
        run_comparison("tridiag200", A);
        sparse_free(A);
    }

    return 0;
}
