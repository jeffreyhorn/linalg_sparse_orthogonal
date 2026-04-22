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
        sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0, SPARSE_LDLT_BACKEND_LINKED_LIST, NULL};
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

/* Sprint 19 Day 14: time the batched supernodal LDL^T path.
 *
 * Two-pass model — see the design block in
 * `src/sparse_ldlt_csc_internal.h`: a scalar pre-pass resolves BK
 * swaps and bakes the resulting permutation into `F->perm`; a
 * subsequent `ldlt_csc_from_sparse(A_perm, ...)` + `pivot_size`
 * carry-over + `ldlt_csc_eliminate_supernodal` runs the batched
 * factor on a pre-permuted view where BK chooses the same pivots
 * without further swaps.
 *
 * The scalar pre-pass and AMD analysis are computed once up front
 * and are NOT timed.  Each timed repetition reuses the cached
 * permutation / pivot decisions and measures only the pre-permuted
 * CSC path (conversion/build + permute + supernodal factor, plus
 * solve).
 *
 * This therefore benchmarks an analyze-once / factor-many workflow.
 * Its `factor_ms` is not directly comparable to `bench_linked_list()`
 * or `bench_csc_path()`, both of which re-run AMD inside every timed
 * iteration — `speedup_csc_sn` will be correspondingly inflated
 * relative to `speedup_csc`.  Interpret `factor_csc_sn_ms` as the
 * steady-state cost of a fresh numeric factor on a previously-
 * analysed matrix, not as the end-to-end one-shot cost. */
static bench_result_t bench_csc_supernodal(const SparseMatrix *A, const double *b, double *x,
                                           int repeat) {
    bench_result_t r = {0, 0, 0, 1};
    double factor_total = 0.0, solve_total = 0.0;
    idx_t n = sparse_rows(A);

    /* Pre-pass: factor scalar once to get F1->perm and F1->pivot_size.
     * Failure here aborts the bench for this matrix. */
    idx_t *amd_perm = malloc((size_t)n * sizeof(idx_t));
    if (!amd_perm) {
        r.ok = 0;
        return r;
    }
    if (sparse_reorder_amd(A, amd_perm) != SPARSE_OK) {
        free(amd_perm);
        r.ok = 0;
        return r;
    }
    LdltCsc *F1 = NULL;
    if (ldlt_csc_from_sparse(A, amd_perm, 2.0, &F1) != SPARSE_OK) {
        free(amd_perm);
        r.ok = 0;
        return r;
    }
    if (ldlt_csc_eliminate(F1) != SPARSE_OK) {
        ldlt_csc_free(F1);
        free(amd_perm);
        r.ok = 0;
        return r;
    }

    /* Cache F1's composed permutation + pivot pattern.  Subsequent
     * batched timed iterations reuse these. */
    idx_t *cached_perm = malloc((size_t)n * sizeof(idx_t));
    idx_t *cached_pivot_size = malloc((size_t)n * sizeof(idx_t));
    if (!cached_perm || !cached_pivot_size) {
        free(cached_perm);
        free(cached_pivot_size);
        ldlt_csc_free(F1);
        free(amd_perm);
        r.ok = 0;
        return r;
    }
    for (idx_t i = 0; i < n; i++) {
        cached_perm[i] = F1->perm[i];
        cached_pivot_size[i] = F1->pivot_size[i];
    }
    ldlt_csc_free(F1);
    free(amd_perm);

    for (int rep = 0; rep < repeat; rep++) {
        LdltCsc *F2 = NULL;
        double t0 = wall_time();
        if (ldlt_csc_from_sparse(A, cached_perm, 2.0, &F2) != SPARSE_OK) {
            r.ok = 0;
            break;
        }
        for (idx_t k = 0; k < n; k++)
            F2->pivot_size[k] = cached_pivot_size[k];
        sparse_err_t e_sn = ldlt_csc_eliminate_supernodal(F2, /*min_size=*/2);
        if (e_sn != SPARSE_OK) {
            ldlt_csc_free(F2);
            r.ok = 0;
            break;
        }
        factor_total += wall_time() - t0;

        t0 = wall_time();
        if (ldlt_csc_solve(F2, b, x) != SPARSE_OK) {
            ldlt_csc_free(F2);
            r.ok = 0;
            break;
        }
        solve_total += wall_time() - t0;

        if (rep == repeat - 1)
            r.residual = rel_residual(A, x, b);
        ldlt_csc_free(F2);
    }

    free(cached_perm);
    free(cached_pivot_size);

    if (r.ok) {
        r.factor_ms = factor_total * 1000.0 / (double)repeat;
        r.solve_ms = solve_total * 1000.0 / (double)repeat;
    }
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

static int bench_matrix(const char *path, int repeat, int supernodal) {
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
    bench_result_t rs = {0, 0, 0, 1};
    if (supernodal)
        rs = bench_csc_supernodal(A, b, x, repeat);

    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;

    if (!rl.ok || !rn.ok || !rw.ok || (supernodal && !rs.ok)) {
        fprintf(stderr, "bench_ldlt_csc: %s — one or more paths failed\n", base);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return 1;
    }

    double sp_native = rl.factor_ms / rn.factor_ms;
    double sp_wrapper = rl.factor_ms / rw.factor_ms;

    if (supernodal) {
        /* Extra columns: factor_csc_sn_ms, solve_csc_sn_ms,
         * speedup_csc_sn (vs LL), res_csc_sn. */
        double sp_sn = rl.factor_ms / rs.factor_ms;
        printf("%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2e,%.2e,%.2e\n", base,
               (int)n, (int)nnz, rl.factor_ms, rn.factor_ms, rw.factor_ms, rs.factor_ms,
               rl.solve_ms, rn.solve_ms, rs.solve_ms, sp_native, sp_wrapper, sp_sn, rl.residual,
               rn.residual, rs.residual);
    } else {
        printf("%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2e,%.2e\n", base, (int)n, (int)nnz,
               rl.factor_ms, rn.factor_ms, rw.factor_ms, rl.solve_ms, rn.solve_ms, sp_native,
               sp_wrapper, rl.residual, rn.residual);
    }

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

/* Sprint 20 Day 6: --dispatch mode.  For each matrix, factor twice
 * — once with `SPARSE_LDLT_BACKEND_AUTO` (the transparent Day 5
 * dispatch), once with `SPARSE_LDLT_BACKEND_LINKED_LIST` forced —
 * and report which backend AUTO selected, its factor / solve
 * timings, and the speedup over the forced linked-list baseline.
 * Exercises the public `sparse_ldlt_factor_opts` transparent
 * dispatch end-to-end at the bench level.
 *
 * CSV columns:
 *   matrix, n, nnz, backend, factor_auto_ms, factor_ll_ms,
 *   solve_auto_ms, solve_ll_ms, speedup_auto_vs_ll, res_auto, res_ll
 *
 * The `backend` column is a literal "linked-list" or "csc" string
 * — the value AUTO selected based on `n >= SPARSE_CSC_THRESHOLD`. */
typedef struct {
    double factor_ms;
    double solve_ms;
    double residual;
    int used_csc; /* Populated only by the AUTO run; 0/1. */
    int ok;
} dispatch_result_t;

static dispatch_result_t bench_dispatch_path(const SparseMatrix *A, const double *b, double *x,
                                             int repeat, sparse_ldlt_backend_t backend) {
    dispatch_result_t r = {0, 0, 0, 0, 1};
    double factor_total = 0.0, solve_total = 0.0;
    int used_csc = 0;

    for (int rep = 0; rep < repeat; rep++) {
        sparse_ldlt_t ldlt = {0};
        sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0, backend, &used_csc};
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
    r.used_csc = used_csc;
    return r;
}

static int bench_dispatch_for_matrix(const char *label, SparseMatrix *A, int repeat) {
    idx_t n = sparse_rows(A);
    idx_t nnz = sparse_nnz(A);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    if (!ones || !b || !x) {
        fprintf(stderr, "bench_ldlt_csc --dispatch: malloc failed (n=%d)\n", (int)n);
        free(ones);
        free(b);
        free(x);
        return 1;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    dispatch_result_t r_auto = bench_dispatch_path(A, b, x, repeat, SPARSE_LDLT_BACKEND_AUTO);
    dispatch_result_t r_ll = bench_dispatch_path(A, b, x, repeat, SPARSE_LDLT_BACKEND_LINKED_LIST);

    if (!r_auto.ok || !r_ll.ok) {
        fprintf(stderr, "bench_ldlt_csc --dispatch: %s — one or more factor runs failed\n", label);
        free(ones);
        free(b);
        free(x);
        return 1;
    }

    const char *backend_str = r_auto.used_csc ? "csc" : "linked-list";
    double speedup = r_ll.factor_ms / r_auto.factor_ms;
    printf("%s,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.2f,%.2e,%.2e\n", label, (int)n, (int)nnz, backend_str,
           r_auto.factor_ms, r_ll.factor_ms, r_auto.solve_ms, r_ll.solve_ms, speedup,
           r_auto.residual, r_ll.residual);

    free(ones);
    free(b);
    free(x);
    return 0;
}

/* Build an in-memory KKT saddle point for the --dispatch corpus so
 * the bench can report the indefinite above-threshold AUTO path
 * without depending on an on-disk .mtx fixture.  Size 150 (top
 * 140×140 tridiagonal SPD, bottom 10×10 zero, identity-pattern
 * coupling) — matches `test_sprint20_integration.c`'s KKT fixture. */
static SparseMatrix *bench_build_kkt_150(void) {
    idx_t n_top = 140, n_bot = 10;
    idx_t n = n_top + n_bot;
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n_top; i++) {
        sparse_insert(A, i, i, 6.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    for (idx_t j = 0; j < n_bot; j++) {
        sparse_insert(A, n_top + j, j, 1.0);
        sparse_insert(A, j, n_top + j, 1.0);
    }
    return A;
}

static int bench_dispatch_mode(int repeat) {
    printf("matrix,n,nnz,backend,"
           "factor_auto_ms,factor_ll_ms,solve_auto_ms,solve_ll_ms,"
           "speedup_auto_vs_ll,res_auto,res_ll\n");

    int rc = 0;
    for (int i = 0; i < default_matrix_count; i++) {
        SparseMatrix *A = NULL;
        if (sparse_load_mm(&A, default_matrices[i]) != SPARSE_OK) {
            fprintf(stderr, "bench_ldlt_csc --dispatch: failed to load %s\n", default_matrices[i]);
            rc = 1;
            continue;
        }
        const char *base = strrchr(default_matrices[i], '/');
        base = base ? base + 1 : default_matrices[i];
        rc |= bench_dispatch_for_matrix(base, A, repeat);
        sparse_free(A);
    }

    /* Synthetic KKT fixture to exercise the indefinite above-
     * threshold AUTO path alongside the SuiteSparse corpus. */
    SparseMatrix *K = bench_build_kkt_150();
    if (K) {
        rc |= bench_dispatch_for_matrix("kkt-150", K, repeat);
        sparse_free(K);
    }
    return rc;
}

int main(int argc, char **argv) {
    int repeat = 3;
    int supernodal = 0;
    int dispatch = 0;
    const char *single_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--repeat") && i + 1 < argc) {
            repeat = atoi(argv[++i]);
            if (repeat < 1)
                repeat = 1;
        } else if (!strcmp(argv[i], "--supernodal")) {
            supernodal = 1;
        } else if (!strcmp(argv[i], "--dispatch")) {
            dispatch = 1;
        } else if (argv[i][0] != '-') {
            single_path = argv[i];
        }
    }

    if (dispatch)
        return bench_dispatch_mode(repeat);

    if (supernodal) {
        printf("matrix,n,nnz,"
               "factor_ll_ms,factor_csc_native_ms,factor_csc_wrapper_ms,factor_csc_sn_ms,"
               "solve_ll_ms,solve_csc_native_ms,solve_csc_sn_ms,"
               "speedup_csc_native,speedup_csc_wrapper,speedup_csc_sn,"
               "res_ll,res_csc_native,res_csc_sn\n");
    } else {
        printf("matrix,n,nnz,"
               "factor_ll_ms,factor_csc_native_ms,factor_csc_wrapper_ms,"
               "solve_ll_ms,solve_csc_native_ms,"
               "speedup_csc_native,speedup_csc_wrapper,"
               "res_ll,res_csc_native\n");
    }

    int rc = 0;
    if (single_path) {
        rc |= bench_matrix(single_path, repeat, supernodal);
    } else {
        for (int i = 0; i < default_matrix_count; i++)
            rc |= bench_matrix(default_matrices[i], repeat, supernodal);
    }
    return rc;
}
