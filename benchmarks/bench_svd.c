/*
 * bench_svd.c — SVD performance profiling harness
 *
 * Usage:
 *   ./bench_svd                     Profile all available SuiteSparse matrices
 *   ./bench_svd <matrix.mtx>        Profile a single matrix
 *
 * Reports wall-clock time for each SVD phase:
 *   - Bidiagonalization
 *   - QR iteration (singular values only)
 *   - Full SVD (with U/V extraction)
 *   - Partial SVD (Lanczos, k=5)
 *   - Partial SVD with vectors (k=5)
 */
#define _POSIX_C_SOURCE 199309L
#include "sparse_bidiag.h"
#include "sparse_matrix.h"
#include "sparse_svd.h"
#include "sparse_svd_internal.h"
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

static void profile_matrix(const char *name, const char *path) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, path);
    if (err != SPARSE_OK || !A) {
        printf("  SKIP %s: load failed\n", name);
        return;
    }

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);
    idx_t nnz = sparse_nnz(A);
    idx_t k = (m < n) ? m : n;
    printf("\n=== %s (%d×%d, nnz=%d) ===\n", name, (int)m, (int)n, (int)nnz);

    /* Phase 1: Bidiagonalization only */
    double t0 = wall_time();
    sparse_bidiag_t bd;
    err = sparse_bidiag_factor(A, &bd);
    double t_bidiag = wall_time() - t0;
    if (err != SPARSE_OK) {
        printf("  bidiag failed: %d\n", (int)err);
        sparse_free(A);
        return;
    }
    printf("  Bidiagonalization:       %8.3f ms\n", t_bidiag * 1000.0);

    /* Phase 2: QR iteration on bidiagonal (singular values only) */
    double *diag_copy = malloc((size_t)k * sizeof(double));
    double *super_copy = (k > 1) ? malloc((size_t)(k - 1) * sizeof(double)) : NULL;
    if (!diag_copy || (k > 1 && !super_copy)) {
        printf("  alloc failed\n");
        free(diag_copy);
        free(super_copy);
        sparse_bidiag_free(&bd);
        sparse_free(A);
        return;
    }
    memcpy(diag_copy, bd.diag, (size_t)k * sizeof(double));
    if (k > 1)
        memcpy(super_copy, bd.superdiag, (size_t)(k - 1) * sizeof(double));

    t0 = wall_time();
    err = bidiag_svd_iterate(diag_copy, super_copy, k, NULL, 0, NULL, 0, 0, 0.0);
    double t_qr = wall_time() - t0;
    printf("  QR iteration (σ only):   %8.3f ms  (err=%d)\n", t_qr * 1000.0, (int)err);
    free(diag_copy);
    free(super_copy);
    sparse_bidiag_free(&bd);

    /* Phase 3: Full SVD (singular values only) */
    t0 = wall_time();
    sparse_svd_t svd;
    err = sparse_svd_compute(A, NULL, &svd);
    double t_full_sigma = wall_time() - t0;
    if (err == SPARSE_OK) {
        printf("  Full SVD (σ only):       %8.3f ms  σ_max=%.4e σ_min=%.4e\n",
               t_full_sigma * 1000.0, svd.sigma[0], svd.sigma[k - 1]);
        sparse_svd_free(&svd);
    } else {
        printf("  Full SVD (σ only):       %8.3f ms  FAILED(%d)\n", t_full_sigma * 1000.0,
               (int)err);
    }

    /* Phase 4: Full SVD with UV */
    sparse_svd_opts_t opts_uv = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    t0 = wall_time();
    err = sparse_svd_compute(A, &opts_uv, &svd);
    double t_full_uv = wall_time() - t0;
    if (err == SPARSE_OK) {
        printf("  Full SVD (with U/V):     %8.3f ms\n", t_full_uv * 1000.0);
        sparse_svd_free(&svd);
    } else {
        printf("  Full SVD (with U/V):     %8.3f ms  FAILED(%d)\n", t_full_uv * 1000.0, (int)err);
    }

    /* Phase 5: Partial SVD (k=5, sigma only) */
    idx_t pk = 5;
    if (pk > k)
        pk = k;
    t0 = wall_time();
    err = sparse_svd_partial(A, pk, NULL, &svd);
    double t_partial = wall_time() - t0;
    if (err == SPARSE_OK) {
        printf("  Partial SVD (k=%d, σ):   %8.3f ms\n", (int)pk, t_partial * 1000.0);
        sparse_svd_free(&svd);
    } else {
        printf("  Partial SVD (k=%d, σ):   %8.3f ms  FAILED(%d)\n", (int)pk, t_partial * 1000.0,
               (int)err);
    }

    /* Phase 6: Partial SVD with vectors (k=5) */
    t0 = wall_time();
    err = sparse_svd_partial(A, pk, &opts_uv, &svd);
    double t_partial_uv = wall_time() - t0;
    if (err == SPARSE_OK) {
        printf("  Partial SVD (k=%d, U/V): %8.3f ms\n", (int)pk, t_partial_uv * 1000.0);
        sparse_svd_free(&svd);
    } else {
        printf("  Partial SVD (k=%d, U/V): %8.3f ms  FAILED(%d)\n", (int)pk, t_partial_uv * 1000.0,
               (int)err);
    }

    /* Summary — only print ratios when full SVD succeeded and has meaningful timing */
    printf("  ---\n");
    if (t_full_sigma > 1e-9) {
        printf("  Bidiag / Full(σ):  %.0f%%\n", t_bidiag / t_full_sigma * 100.0);
        printf("  QR / Full(σ):      %.0f%%\n", t_qr / t_full_sigma * 100.0);
        if (t_full_uv > 0)
            printf("  UV overhead:       %.0f%%\n",
                   (t_full_uv - t_full_sigma) / t_full_sigma * 100.0);
        if (t_partial > 1e-9)
            printf("  Partial/Full:      %.1fx speedup\n", t_full_sigma / t_partial);
    } else {
        printf("  (ratios not computed — full SVD too fast or failed)\n");
    }

    sparse_free(A);
}

int main(int argc, char **argv) {
    printf("SVD Performance Profiling\n");
    printf("=========================\n");

    if (argc > 1) {
        /* Single matrix */
        profile_matrix(argv[1], argv[1]);
    } else {
        /* Profile all SuiteSparse matrices */
        const char *matrices[][2] = {
            {"nos4", "tests/data/suitesparse/nos4.mtx"},
            {"west0067", "tests/data/suitesparse/west0067.mtx"},
            {"bcsstk04", "tests/data/suitesparse/bcsstk04.mtx"},
            {"steam1", "tests/data/suitesparse/steam1.mtx"},
            {"orsirr_1", "tests/data/suitesparse/orsirr_1.mtx"},
        };
        size_t nmat = sizeof(matrices) / sizeof(matrices[0]);
        for (size_t i = 0; i < nmat; i++)
            profile_matrix(matrices[i][0], matrices[i][1]);
    }

    printf("\nDone.\n");
    return 0;
}
