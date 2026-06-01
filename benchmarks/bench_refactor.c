/*
 * bench_refactor.c — Benchmark: one-shot vs analyze-once / factor-many.
 *
 * Measures the strongest public repeated-run direct workflow on a moderate
 * Cholesky corpus:
 *
 *   (A) one-shot:
 *       sparse_copy(A_base) + sparse_cholesky_factor() on each iteration
 *
 *   (B) analyze-once / factor-many:
 *       sparse_analyze(A_base) once
 *       sparse_factor_numeric(A_base, &analysis, &factors) once
 *       sparse_refactor_numeric(A_perturb, &analysis, &factors) for later
 *       same-pattern value changes
 *
 * The repeated-run path now perturbs values across iterations instead of
 * repeatedly refactoring the identical matrix, so the benchmark reflects the
 * actual "same sparsity pattern, different numeric values" contract.
 *
 * Output stays human-readable rather than CSV:
 *   oneshot      = average one-shot wall time per iteration
 *   analyze_once = one-time symbolic analysis cost
 *   initial      = first numeric factorization after analysis
 *   refactor_avg = average later sparse_refactor_numeric cost
 *   repeated_avg = (analyze_once + initial + all later refactors) / reps
 *   speedup      = oneshot / repeated_avg
 *   residual     = final relative residual on the last perturbed matrix
 */
#define _POSIX_C_SOURCE 200809L
#include "sparse_analysis.h"
#include "sparse_cholesky.h"
#include "sparse_matrix.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __MACH__
#include <mach/mach_time.h>
static double wall_time(void) {
    static mach_timebase_info_data_t info;
    if (info.denom == 0)
        mach_timebase_info(&info);
    return (double)mach_absolute_time() * (double)info.numer / (double)info.denom / 1e9;
}
#else
static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
#endif

static double rel_residual(const SparseMatrix *A, const double *x, const double *b) {
    idx_t n = sparse_rows(A);
    double *Ax = malloc((size_t)n * sizeof(double));
    if (!Ax)
        return (double)INFINITY;

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
    return bmax > 0.0 ? rmax / bmax : rmax;
}

static double symmetric_noise(idx_t i, idx_t j, idx_t n, uint64_t seed) {
    idx_t a = (i < j) ? i : j;
    idx_t b = (i < j) ? j : i;
    uint64_t key = (uint64_t)a * (uint64_t)n + (uint64_t)b + seed;
    uint64_t h = key * 0x9e3779b97f4a7c15ULL;
    return (double)(h >> 32) / (double)(1ULL << 32) - 0.5;
}

static void perturb_values_in_place(SparseMatrix *A, double eps, uint64_t seed) {
    idx_t rows = sparse_rows(A);
    idx_t cols = sparse_cols(A);
    for (idx_t i = 0; i < rows; i++) {
        for (idx_t j = 0; j < cols; j++) {
            double val = sparse_get(A, i, j);
            if (val != 0.0) {
                double rnd = symmetric_noise(i, j, rows, seed);
                (void)sparse_set(A, i, j, val * (1.0 + eps * rnd));
            }
        }
    }
}

/* Build n×n SPD tridiagonal with parameterized diagonal */
static SparseMatrix *make_tridiag(idx_t n, double diag) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        if (sparse_insert(A, i, i, diag) != SPARSE_OK)
            goto fail;
        if (i > 0) {
            if (sparse_insert(A, i, i - 1, -1.0) != SPARSE_OK)
                goto fail;
            if (sparse_insert(A, i - 1, i, -1.0) != SPARSE_OK)
                goto fail;
        }
    }
    return A;
fail:
    sparse_free(A);
    return NULL;
}

static void bench_matrix(const char *name, SparseMatrix *A, int reps) {
    if (!A) {
        printf("  %-20s  [SKIP] matrix construction failed\n", name);
        return;
    }
    idx_t n = sparse_rows(A);
    printf("  %-20s  n=%-5d nnz=%-6d reps=%-4d", name, (int)n, (int)sparse_nnz(A), reps);
    fflush(stdout);

    /* ── Approach A: one-shot repeated on value-perturbed copies ─────────── */
    double t0 = wall_time();
    int ok = 1;
    for (int r = 0; r < reps && ok; r++) {
        SparseMatrix *L = sparse_copy(A);
        if (!L) {
            ok = 0;
            break;
        }
        perturb_values_in_place(L, 1e-9, (uint64_t)r * 0xcafef00dULL);
        if (sparse_cholesky_factor(L) != SPARSE_OK) {
            ok = 0;
            sparse_free(L);
            break;
        }
        sparse_free(L);
    }
    double t_oneshot_total = wall_time() - t0;

    if (!ok) {
        printf("  [SKIP] one-shot factorization failed\n");
        return;
    }

    /* ── Approach B: analyze once, factor once, then refactor perturbed values ─── */
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double analyze_s = 0.0;
    double initial_factor_s = 0.0;
    double refactor_total_s = 0.0;
    double final_residual = 0.0;

    if (!ones || !b || !x) {
        printf("  [SKIP] benchmark workspace allocation failed\n");
        free(ones);
        free(b);
        free(x);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;

    t0 = wall_time();
    if (sparse_analyze(A, NULL, &analysis) != SPARSE_OK) {
        printf("  [SKIP] analyze+factor failed\n");
        sparse_factor_free(&factors);
        sparse_analysis_free(&analysis);
        free(ones);
        free(b);
        free(x);
        return;
    }
    analyze_s = wall_time() - t0;

    t0 = wall_time();
    if (sparse_factor_numeric(A, &analysis, &factors) != SPARSE_OK) {
        printf("  [SKIP] analyze+factor failed\n");
        sparse_factor_free(&factors);
        sparse_analysis_free(&analysis);
        free(ones);
        free(b);
        free(x);
        return;
    }
    initial_factor_s = wall_time() - t0;

    int completed = 1;
    for (int r = 1; r < reps; r++) {
        SparseMatrix *A_perturb = sparse_copy(A);
        if (!A_perturb) {
            printf("  [SKIP] perturb-copy failed at iter %d\n", r);
            sparse_factor_free(&factors);
            sparse_analysis_free(&analysis);
            free(ones);
            free(b);
            free(x);
            return;
        }
        perturb_values_in_place(A_perturb, 1e-9, (uint64_t)r * 0xcafef00dULL);

        t0 = wall_time();
        if (sparse_refactor_numeric(A_perturb, &analysis, &factors) != SPARSE_OK) {
            printf("  [SKIP] refactor failed at iter %d\n", r);
            sparse_free(A_perturb);
            sparse_factor_free(&factors);
            sparse_analysis_free(&analysis);
            free(ones);
            free(b);
            free(x);
            return;
        }
        refactor_total_s += wall_time() - t0;

        if (r == reps - 1) {
            sparse_matvec(A_perturb, ones, b);
            if (sparse_factor_solve(&factors, &analysis, b, x) != SPARSE_OK) {
                printf("  [SKIP] final solve failed\n");
                sparse_free(A_perturb);
                sparse_factor_free(&factors);
                sparse_analysis_free(&analysis);
                free(ones);
                free(b);
                free(x);
                return;
            }
            final_residual = rel_residual(A_perturb, x, b);
        }

        sparse_free(A_perturb);
        completed++;
    }

    double oneshot_avg_s = t_oneshot_total / (double)reps;
    double repeated_avg_s = (analyze_s + initial_factor_s + refactor_total_s) / (double)reps;
    double refactor_avg_s = (reps > 1) ? refactor_total_s / (double)(reps - 1) : 0.0;
    double speedup = oneshot_avg_s / repeated_avg_s;

    printf("  oneshot=%.4fs  analyze_once=%.4fs  initial=%.4fs  refactor_avg=%.4fs"
           "  repeated_avg=%.4fs  speedup=%.2fx  residual=%.2e (%d iters)\n",
           oneshot_avg_s, analyze_s, initial_factor_s, refactor_avg_s, repeated_avg_s, speedup,
           final_residual, completed);

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    free(ones);
    free(b);
    free(x);
}

int main(void) {
    printf("=== Refactorization Benchmark: One-Shot vs Analyze-Once ===\n\n");

    int reps = 100;

    /* Small tridiagonal */
    SparseMatrix *A1 = make_tridiag(50, 4.0);
    bench_matrix("tridiag-50", A1, reps);
    sparse_free(A1);

    /* Medium tridiagonal */
    SparseMatrix *A2 = make_tridiag(200, 4.0);
    bench_matrix("tridiag-200", A2, reps);
    sparse_free(A2);

    /* Large tridiagonal */
    SparseMatrix *A3 = make_tridiag(500, 4.0);
    bench_matrix("tridiag-500", A3, reps);
    sparse_free(A3);

    /* SuiteSparse bcsstk04 */
    SparseMatrix *A4 = NULL;
    if (sparse_load_mm(&A4, "tests/data/suitesparse/bcsstk04.mtx") == SPARSE_OK) {
        bench_matrix("bcsstk04", A4, reps);
        sparse_free(A4);
    } else {
        printf("  bcsstk04              [SKIP] not found\n");
    }

    /* SuiteSparse nos4 */
    SparseMatrix *A5 = NULL;
    if (sparse_load_mm(&A5, "tests/data/suitesparse/nos4.mtx") == SPARSE_OK) {
        bench_matrix("nos4", A5, reps);
        sparse_free(A5);
    } else {
        printf("  nos4                  [SKIP] not found\n");
    }

    printf("\nDone.\n");
    return 0;
}
