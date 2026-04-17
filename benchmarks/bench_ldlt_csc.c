/*
 * bench_ldlt_csc.c — LDL^T backend comparison: linked-list vs CSC
 *
 * Compares wall-clock factor + solve time between:
 *   1. Linked-list LDL^T (sparse_ldlt_factor_opts with AMD reordering
 *      + sparse_ldlt_solve)
 *   2. CSC LDL^T        (ldlt_csc_from_sparse + ldlt_csc_eliminate +
 *      ldlt_csc_solve, using a precomputed AMD permutation so both
 *      paths see the same fill-reducing ordering)
 *
 * Day 8's CSC LDL^T delegates to the linked-list kernel after
 * expanding the lower triangle to full symmetric — so the CSC path is
 * expected to run slightly slower than linked-list on today's
 * implementation.  This benchmark quantifies that overhead so a later
 * sprint (native CSC LDL^T kernel) can measure its speedup against a
 * known baseline.
 *
 * Output is CSV on stdout:
 *   matrix, n, nnz, factor_ll, factor_csc, solve_ll, solve_csc,
 *   speedup_csc, res_ll, res_csc
 * Times are milliseconds, averaged across --repeat runs.
 */
#define _POSIX_C_SOURCE 199309L

#include "sparse_ldlt.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_vector.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double rel_residual(const SparseMatrix *A, const double *x, const double *b) {
    idx_t n = sparse_rows(A);
    double *Ax = malloc((size_t)n * sizeof(double));
    if (!Ax) {
        /* Sentinel value: NaN signals an unmeasurable residual due to
         * allocation failure.  Callers print the raw value in the CSV
         * row, so "nan" in the output column is the visible cue. */
        fprintf(stderr, "bench_ldlt_csc: malloc failed in rel_residual (n=%d)\n", (int)n);
        return (double)NAN;
    }
    sparse_matvec(A, x, Ax);
    double rmax = 0.0, bmax = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double r = fabs(Ax[i] - b[i]);
        double bi = fabs(b[i]);
        if (r > rmax)
            rmax = r;
        if (bi > bmax)
            bmax = bi;
    }
    free(Ax);
    return bmax > 0.0 ? rmax / bmax : rmax;
}

typedef struct {
    double factor_ms;
    double solve_ms;
    double residual;
    int ok;
} bench_result_t;

static bench_result_t bench_linked_list(const SparseMatrix *A, const double *b, double *x,
                                        int repeat) {
    bench_result_t r = {0, 0, 0, 1};
    double factor_total = 0.0, solve_total = 0.0;

    for (int rep = 0; rep < repeat; rep++) {
        sparse_ldlt_t ldlt = {0};
        sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
        double t0 = wall_time();
        if (sparse_ldlt_factor_opts(A, &opts, &ldlt) != SPARSE_OK) {
            r.ok = 0;
            return r;
        }
        factor_total += wall_time() - t0;

        t0 = wall_time();
        if (sparse_ldlt_solve(&ldlt, b, x) != SPARSE_OK) {
            sparse_ldlt_free(&ldlt);
            r.ok = 0;
            return r;
        }
        solve_total += wall_time() - t0;

        if (rep == repeat - 1)
            r.residual = rel_residual(A, x, b);
        sparse_ldlt_free(&ldlt);
    }
    r.factor_ms = factor_total * 1000.0 / (double)repeat;
    r.solve_ms = solve_total * 1000.0 / (double)repeat;
    return r;
}

static bench_result_t bench_csc_path(const SparseMatrix *A, const idx_t *amd_perm, const double *b,
                                     double *x, int repeat) {
    bench_result_t r = {0, 0, 0, 1};
    double factor_total = 0.0, solve_total = 0.0;

    for (int rep = 0; rep < repeat; rep++) {
        LdltCsc *F = NULL;
        if (ldlt_csc_from_sparse(A, amd_perm, 2.0, &F) != SPARSE_OK) {
            r.ok = 0;
            break;
        }
        double t0 = wall_time();
        if (ldlt_csc_eliminate(F) != SPARSE_OK) {
            ldlt_csc_free(F);
            r.ok = 0;
            break;
        }
        factor_total += wall_time() - t0;

        t0 = wall_time();
        if (ldlt_csc_solve(F, b, x) != SPARSE_OK) {
            ldlt_csc_free(F);
            r.ok = 0;
            break;
        }
        solve_total += wall_time() - t0;

        if (rep == repeat - 1)
            r.residual = rel_residual(A, x, b);
        ldlt_csc_free(F);
    }

    if (r.ok) {
        r.factor_ms = factor_total * 1000.0 / (double)repeat;
        r.solve_ms = solve_total * 1000.0 / (double)repeat;
    }
    return r;
}

static int bench_matrix(const char *path, int repeat) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, path) != SPARSE_OK) {
        fprintf(stderr, "bench_ldlt_csc: failed to load %s\n", path);
        return 1;
    }
    idx_t n = sparse_rows(A);
    idx_t nnz = sparse_nnz(A);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* Precompute AMD perm for the CSC path (the linked-list path uses
     * its own internal AMD via opts.reorder). */
    idx_t *amd_perm = malloc((size_t)n * sizeof(idx_t));
    if (sparse_reorder_amd(A, amd_perm) != SPARSE_OK) {
        fprintf(stderr, "bench_ldlt_csc: AMD failed on %s\n", path);
        free(ones);
        free(b);
        free(x);
        free(amd_perm);
        sparse_free(A);
        return 1;
    }

    bench_result_t rl = bench_linked_list(A, b, x, repeat);
    bench_result_t rc = bench_csc_path(A, amd_perm, b, x, repeat);

    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;

    if (!rl.ok || !rc.ok) {
        fprintf(stderr, "bench_ldlt_csc: %s — one or more paths failed\n", base);
        free(ones);
        free(b);
        free(x);
        free(amd_perm);
        sparse_free(A);
        return 1;
    }

    double sp = rl.factor_ms / rc.factor_ms;

    printf("%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2e,%.2e\n", base, (int)n, (int)nnz, rl.factor_ms,
           rc.factor_ms, rl.solve_ms, rc.solve_ms, sp, rl.residual, rc.residual);

    free(ones);
    free(b);
    free(x);
    free(amd_perm);
    sparse_free(A);
    return 0;
}

static const char *default_matrices[] = {
    "tests/data/suitesparse/nos4.mtx",
    "tests/data/suitesparse/bcsstk04.mtx",
};
static const int default_matrix_count =
    (int)(sizeof(default_matrices) / sizeof(default_matrices[0]));

int main(int argc, char **argv) {
    int repeat = 3;
    const char *single_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--repeat") && i + 1 < argc) {
            repeat = atoi(argv[++i]);
            if (repeat < 1)
                repeat = 1;
        } else if (argv[i][0] != '-') {
            single_path = argv[i];
        }
    }

    printf("matrix,n,nnz,factor_ll_ms,factor_csc_ms,solve_ll_ms,solve_csc_ms,"
           "speedup_csc,res_ll,res_csc\n");

    int rc = 0;
    if (single_path) {
        rc |= bench_matrix(single_path, repeat);
    } else {
        for (int i = 0; i < default_matrix_count; i++)
            rc |= bench_matrix(default_matrices[i], repeat);
    }
    return rc;
}
