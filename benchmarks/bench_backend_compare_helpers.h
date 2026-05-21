#ifndef BENCH_BACKEND_COMPARE_HELPERS_H
#define BENCH_BACKEND_COMPARE_HELPERS_H

#include "sparse_matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Pair-scoped helper layer for the benchmark backend-comparison family.
 *
 * Sprint 37 Day 6 consolidates the repeated timer / residual /
 * matrix-load / unit-RHS setup logic shared by `bench_chol_csc.c`
 * and `bench_ldlt_csc.c`.  Keep this narrow and explicit instead of
 * growing a broad benchmark framework.
 */

typedef struct {
    double factor_ms;
    double solve_ms;
    double residual;
    int ok;
} bench_backend_result_t;

static inline double bench_backend_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline double bench_backend_rel_residual_max(const char *owner, const SparseMatrix *A,
                                                    const double *x, const double *b) {
    idx_t n = sparse_rows(A);
    double *Ax = malloc((size_t)n * sizeof(double));
    if (!Ax) {
        fprintf(stderr, "%s: malloc failed in rel_residual (n=%d)\n", owner, (int)n);
        return nan("");
    }

    sparse_matvec(A, x, Ax);
    double rmax = 0.0;
    double bmax = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double r = fabs(Ax[i] - b[i]);
        double bi = fabs(b[i]);
        if (r > rmax)
            rmax = r;
        if (bi > bmax)
            bmax = bi;
    }

    free(Ax);
    return (bmax > 0.0) ? rmax / bmax : rmax;
}

static inline int bench_backend_load_matrix(const char *owner, const char *path, SparseMatrix **A_out,
                                            const char **label_out) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, path) != SPARSE_OK) {
        fprintf(stderr, "%s: failed to load %s\n", owner, path);
        return 1;
    }

    const char *base = strrchr(path, '/');
    *A_out = A;
    *label_out = base ? base + 1 : path;
    return 0;
}

static inline int bench_backend_make_unit_rhs(const char *owner, const SparseMatrix *A,
                                              double **b_out, double **x_out) {
    idx_t n = sparse_rows(A);
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    if (!ones || !b || !x) {
        fprintf(stderr, "%s: malloc failed in unit RHS setup (n=%d)\n", owner, (int)n);
        free(ones);
        free(b);
        free(x);
        return 1;
    }

    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    free(ones);
    *b_out = b;
    *x_out = x;
    return 0;
}

#endif
