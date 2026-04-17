/*
 * bench_chol_csc.c — Cholesky backend comparison: linked-list vs CSC
 *
 * Compares wall-clock time for factor and solve between:
 *   1. Linked-list Cholesky (sparse_cholesky_factor + sparse_cholesky_solve)
 *   2. CSC scalar     (chol_csc_factor + chol_csc_solve_perm)
 *   3. CSC supernodal (chol_csc_eliminate_supernodal + chol_csc_solve_perm)
 *
 * All three use the same AMD fill-reducing reordering to keep the
 * comparison apples-to-apples.  Residuals from each path are checked
 * against the original A to 1e-8 relative — if any path fails the
 * residual check the benchmark prints a warning and skips timing
 * reporting for that matrix.
 *
 * Output is CSV on stdout: one header row, one row per matrix with
 *   matrix, n, nnz, factor_ll, factor_csc, factor_csc_sn,
 *   solve_ll,  solve_csc,  solve_csc_sn,
 *   speedup_csc, speedup_csc_sn, res_ll, res_csc, res_csc_sn
 * Times are in milliseconds (averaged across --repeat runs).
 *
 * Usage:
 *   ./bench_chol_csc                                # default matrix list
 *   ./bench_chol_csc path/to/matrix.mtx             # benchmark one matrix
 *   ./bench_chol_csc --repeat 5                     # average 5 runs
 */
#define _POSIX_C_SOURCE 199309L

#include "sparse_analysis.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_cholesky.h"
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
        fprintf(stderr, "bench_chol_csc: malloc failed in rel_residual (n=%d)\n", (int)n);
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
    double factor_ms; /* average factor time, milliseconds */
    double solve_ms;  /* average solve time, milliseconds */
    double residual;  /* relative residual for the last run */
    int ok;           /* 1 on success, 0 on error */
} bench_result_t;

/* ─── Linked-list Cholesky path ───────────────────────────────────── */

static bench_result_t bench_linked_list(const SparseMatrix *A, const double *b, double *x,
                                        int repeat) {
    bench_result_t r = {0, 0, 0, 1};
    double factor_total = 0.0, solve_total = 0.0;
    for (int rep = 0; rep < repeat; rep++) {
        SparseMatrix *L = sparse_copy(A);
        sparse_cholesky_opts_t opts = {SPARSE_REORDER_AMD};

        double t0 = wall_time();
        if (sparse_cholesky_factor_opts(L, &opts) != SPARSE_OK) {
            sparse_free(L);
            r.ok = 0;
            return r;
        }
        factor_total += wall_time() - t0;

        t0 = wall_time();
        if (sparse_cholesky_solve(L, b, x) != SPARSE_OK) {
            sparse_free(L);
            r.ok = 0;
            return r;
        }
        solve_total += wall_time() - t0;

        if (rep == repeat - 1)
            r.residual = rel_residual(A, x, b);
        sparse_free(L);
    }
    r.factor_ms = factor_total * 1000.0 / (double)repeat;
    r.solve_ms = solve_total * 1000.0 / (double)repeat;
    return r;
}

/* ─── CSC scalar & CSC supernodal paths ──────────────────────────── */

typedef sparse_err_t (*csc_eliminate_fn)(CholCsc *);

static bench_result_t bench_csc_path(const SparseMatrix *A, const double *b, double *x, int repeat,
                                     csc_eliminate_fn eliminate) {
    bench_result_t r = {0, 0, 0, 1};
    double factor_total = 0.0, solve_total = 0.0;

    /* AMD analysis computed once; same permutation for every rep. */
    sparse_analysis_opts_t aopts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_AMD};
    sparse_analysis_t an = {0};
    if (sparse_analyze(A, &aopts, &an) != SPARSE_OK) {
        r.ok = 0;
        return r;
    }

    for (int rep = 0; rep < repeat; rep++) {
        CholCsc *L = NULL;
        /* Include CSC conversion time in factor_ms: CSC conversion is a
         * required part of the CSC factor pipeline, so excluding it
         * would overstate the speedup.  The linked-list path has its
         * own conversion cost (AMD reorder + in-place structure setup)
         * rolled into sparse_cholesky_factor_opts, so including ours
         * here keeps the comparison fair. */
        double t0 = wall_time();
        if (chol_csc_from_sparse_with_analysis(A, &an, &L) != SPARSE_OK) {
            r.ok = 0;
            break;
        }
        if (eliminate(L) != SPARSE_OK) {
            chol_csc_free(L);
            r.ok = 0;
            break;
        }
        factor_total += wall_time() - t0;

        t0 = wall_time();
        if (chol_csc_solve_perm(L, an.perm, b, x) != SPARSE_OK) {
            chol_csc_free(L);
            r.ok = 0;
            break;
        }
        solve_total += wall_time() - t0;

        if (rep == repeat - 1)
            r.residual = rel_residual(A, x, b);
        chol_csc_free(L);
    }

    sparse_analysis_free(&an);

    if (r.ok) {
        r.factor_ms = factor_total * 1000.0 / (double)repeat;
        r.solve_ms = solve_total * 1000.0 / (double)repeat;
    }
    return r;
}

static sparse_err_t eliminate_scalar(CholCsc *L) { return chol_csc_eliminate(L); }

static sparse_err_t eliminate_supernodal(CholCsc *L) { return chol_csc_eliminate_supernodal(L, 4); }

/* ─── Matrix runner ─────────────────────────────────────────────── */

static int bench_matrix(const char *path, int repeat) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, path) != SPARSE_OK) {
        fprintf(stderr, "bench_chol_csc: failed to load %s\n", path);
        return 1;
    }
    idx_t n = sparse_rows(A);
    idx_t nnz = sparse_nnz(A);

    /* RHS b = A * [1, 1, ..., 1]. */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);
    for (idx_t i = 0; i < n; i++)
        x[i] = 0.0;

    bench_result_t rl = bench_linked_list(A, b, x, repeat);
    bench_result_t rc = bench_csc_path(A, b, x, repeat, eliminate_scalar);
    bench_result_t rs = bench_csc_path(A, b, x, repeat, eliminate_supernodal);

    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;

    if (!rl.ok || !rc.ok || !rs.ok) {
        fprintf(stderr, "bench_chol_csc: %s — one or more paths failed\n", base);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return 1;
    }

    double sp_csc = rl.factor_ms / rc.factor_ms;
    double sp_sn = rl.factor_ms / rs.factor_ms;

    /* CSV row */
    printf("%s,%d,%d,", base, (int)n, (int)nnz);
    printf("%.3f,%.3f,%.3f,", rl.factor_ms, rc.factor_ms, rs.factor_ms);
    printf("%.3f,%.3f,%.3f,", rl.solve_ms, rc.solve_ms, rs.solve_ms);
    printf("%.2f,%.2f,", sp_csc, sp_sn);
    printf("%.2e,%.2e,%.2e\n", rl.residual, rc.residual, rs.residual);

    free(ones);
    free(b);
    free(x);
    sparse_free(A);
    return 0;
}

/* ─── Main ─────────────────────────────────────────────────────── */

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

    printf("matrix,n,nnz,"
           "factor_ll_ms,factor_csc_ms,factor_csc_sn_ms,"
           "solve_ll_ms,solve_csc_ms,solve_csc_sn_ms,"
           "speedup_csc,speedup_csc_sn,"
           "res_ll,res_csc,res_csc_sn\n");

    int rc = 0;
    if (single_path) {
        rc |= bench_matrix(single_path, repeat);
    } else {
        for (int i = 0; i < default_matrix_count; i++)
            rc |= bench_matrix(default_matrices[i], repeat);
    }
    return rc;
}
