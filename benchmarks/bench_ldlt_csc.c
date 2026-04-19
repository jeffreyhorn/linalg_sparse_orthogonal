/*
 * bench_ldlt_csc.c — LDL^T backend comparison: linked-list vs CSC
 *
 * Reports wall-clock factor + solve time across three paths:
 *   1. Linked-list LDL^T  (`sparse_ldlt_factor_opts` with AMD
 *      reordering + `sparse_ldlt_solve`).
 *   2. CSC LDL^T — native Bunch-Kaufman kernel (default since Sprint
 *      18 Day 5): `ldlt_csc_from_sparse` + `ldlt_csc_eliminate` +
 *      `ldlt_csc_solve`, using a precomputed AMD permutation so all
 *      three paths see the same fill-reducing ordering.
 *   3. CSC LDL^T — Sprint 17 wrapper (expand to full symmetric
 *      SparseMatrix → call `sparse_ldlt_factor` → unpack).  Reached
 *      via the runtime override `ldlt_csc_set_kernel_override` so
 *      this benchmark can measure the native-vs-wrapper delta without
 *      recompiling.
 *
 * Output is CSV on stdout:
 *   matrix, n, nnz,
 *   factor_ll_ms, factor_csc_native_ms, factor_csc_wrapper_ms,
 *   solve_ll_ms, solve_csc_native_ms,
 *   speedup_csc_native, speedup_csc_wrapper,
 *   res_ll, res_csc_native
 * Times are milliseconds, averaged across --repeat runs.  The
 * "speedup_csc_*" columns report `factor_ll / factor_csc_*` — higher
 * means the CSC path is faster than linked-list.
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

/* Time the CSC path with an explicit kernel override.  Pass
 * LDLT_CSC_KERNEL_NATIVE or LDLT_CSC_KERNEL_WRAPPER to pin which
 * elimination body the benchmark exercises — both invocations run
 * the same scatter/cmod setup around `ldlt_csc_eliminate`, so the
 * only difference in factor_ms is the kernel body itself. */
static bench_result_t bench_csc_path(const SparseMatrix *A, const double *b, double *x, int repeat,
                                     LdltCscKernelOverride kernel) {
    bench_result_t r = {0, 0, 0, 1};
    double factor_total = 0.0, solve_total = 0.0;
    idx_t n = sparse_rows(A);

    ldlt_csc_set_kernel_override(kernel);

    for (int rep = 0; rep < repeat; rep++) {
        /* Fair comparison: the linked-list path's
         * sparse_ldlt_factor_opts(..., AMD) re-runs AMD every
         * iteration, so do the same here — include AMD + CSC
         * conversion + elimination in factor_ms. */
        idx_t *amd_perm = malloc((size_t)n * sizeof(idx_t));
        if (!amd_perm) {
            r.ok = 0;
            break;
        }
        LdltCsc *F = NULL;
        double t0 = wall_time();
        if (sparse_reorder_amd(A, amd_perm) != SPARSE_OK) {
            free(amd_perm);
            r.ok = 0;
            break;
        }
        if (ldlt_csc_from_sparse(A, amd_perm, 2.0, &F) != SPARSE_OK) {
            free(amd_perm);
            r.ok = 0;
            break;
        }
        if (ldlt_csc_eliminate(F) != SPARSE_OK) {
            ldlt_csc_free(F);
            free(amd_perm);
            r.ok = 0;
            break;
        }
        factor_total += wall_time() - t0;

        t0 = wall_time();
        if (ldlt_csc_solve(F, b, x) != SPARSE_OK) {
            ldlt_csc_free(F);
            free(amd_perm);
            r.ok = 0;
            break;
        }
        solve_total += wall_time() - t0;

        if (rep == repeat - 1)
            r.residual = rel_residual(A, x, b);
        ldlt_csc_free(F);
        free(amd_perm);
    }

    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

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
    if (!ones || !b || !x) {
        fprintf(stderr, "bench_ldlt_csc: malloc failed in bench_matrix (n=%d)\n", (int)n);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return 1;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    bench_result_t rl = bench_linked_list(A, b, x, repeat);
    bench_result_t rn = bench_csc_path(A, b, x, repeat, LDLT_CSC_KERNEL_NATIVE);
    bench_result_t rw = bench_csc_path(A, b, x, repeat, LDLT_CSC_KERNEL_WRAPPER);

    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;

    if (!rl.ok || !rn.ok || !rw.ok) {
        fprintf(stderr, "bench_ldlt_csc: %s — one or more paths failed\n", base);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return 1;
    }

    double sp_native = rl.factor_ms / rn.factor_ms;
    double sp_wrapper = rl.factor_ms / rw.factor_ms;

    /* Columns: matrix, n, nnz, factor_ll, factor_csc_native,
     * factor_csc_wrapper, solve_ll, solve_csc_native, speedup_native,
     * speedup_wrapper, res_ll, res_csc_native. */
    printf("%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2e,%.2e\n", base, (int)n, (int)nnz,
           rl.factor_ms, rn.factor_ms, rw.factor_ms, rl.solve_ms, rn.solve_ms, sp_native,
           sp_wrapper, rl.residual, rn.residual);

    free(ones);
    free(b);
    free(x);
    sparse_free(A);
    return 0;
}

/* Sprint 18 Day 12 corpus selection rationale:
 *   SPD: nos4, bcsstk04, bcsstk14 kept as default.  s3rmt3m3 (n=5357)
 *   is added for scaling visibility without blowing the bench runtime.
 *   Kuu/Pres_Poisson are SPD but large enough that the wrapper path's
 *   full-symmetric expansion dominates; they are available via the
 *   single-matrix CLI (argv[1]) for operators who want the numbers.
 *   GHS_indef/bloweybq is reported SINGULAR by the linked-list
 *   baseline, so it is not a fair CSC-vs-LL comparison target.
 *   Hollinger/tuma1 (n=22967) runs longer than 3 minutes per factor
 *   even on the fastest path and is excluded from the default list. */
static const char *default_matrices[] = {
    "tests/data/suitesparse/nos4.mtx",
    "tests/data/suitesparse/bcsstk04.mtx",
    "tests/data/suitesparse/bcsstk14.mtx",
    "tests/data/suitesparse/s3rmt3m3.mtx",
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
           "factor_ll_ms,factor_csc_native_ms,factor_csc_wrapper_ms,"
           "solve_ll_ms,solve_csc_native_ms,"
           "speedup_csc_native,speedup_csc_wrapper,"
           "res_ll,res_csc_native\n");

    int rc = 0;
    if (single_path) {
        rc |= bench_matrix(single_path, repeat);
    } else {
        for (int i = 0; i < default_matrix_count; i++)
            rc |= bench_matrix(default_matrices[i], repeat);
    }
    return rc;
}
