/*
 * bench_refactor.c — Benchmark: symbolic-once vs repeated full factorization.
 *
 * Compares two approaches for factoring the same sparsity pattern N times:
 *   (A) One-shot: call sparse_cholesky_factor() N times (each redo symbolic work)
 *   (B) Analyze-once: call sparse_analyze() once, then sparse_refactor_numeric() N-1 times
 *
 * Reports wall-clock times and speedup ratio.
 *
 * Build:
 *   cc -O2 -Iinclude examples/bench_refactor.c -Lbuild -lsparse_lu_ortho -lm
 */
#define _POSIX_C_SOURCE 199309L
#include "sparse_analysis.h"
#include "sparse_cholesky.h"
#include "sparse_matrix.h"
#include <math.h>
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

/* Build n×n SPD tridiagonal with parameterized diagonal */
static SparseMatrix *make_tridiag(idx_t n, double diag) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    return A;
}

static void bench_matrix(const char *name, SparseMatrix *A, int reps) {
    idx_t n = sparse_rows(A);
    printf("  %-20s  n=%-5d nnz=%-6d reps=%-4d", name, (int)n, (int)sparse_nnz(A), reps);
    fflush(stdout);

    /* ── Approach A: one-shot repeated ─────────────────────────── */
    double t0 = wall_time();
    for (int r = 0; r < reps; r++) {
        SparseMatrix *L = sparse_copy(A);
        sparse_cholesky_factor(L);
        sparse_free(L);
    }
    double t_oneshot = wall_time() - t0;

    /* ── Approach B: analyze once, refactor N-1 times ──────────── */
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    t0 = wall_time();
    sparse_analyze(A, NULL, &analysis);
    sparse_factor_numeric(A, &analysis, &factors);
    for (int r = 1; r < reps; r++) {
        sparse_refactor_numeric(A, &analysis, &factors);
    }
    double t_analyze = wall_time() - t0;

    double speedup = t_oneshot / t_analyze;
    printf("  oneshot=%.4fs  analyze=%.4fs  speedup=%.2fx\n", t_oneshot, t_analyze, speedup);

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
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
