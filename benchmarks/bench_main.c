/*
 * bench_main.c — Benchmark harness for sparse LU library
 *
 * Usage:
 *   ./bench_main [matrix.mtx]              Load from file
 *   ./bench_main --size N                  Generate NxN diag-dominant random sparse
 *   ./bench_main --size N --repeat R       Average over R repetitions
 *   ./bench_main --dir PATH                Benchmark all .mtx files in directory
 *   ./bench_main --dir PATH --pivot partial  Use partial pivoting (default: complete)
 *
 * Reports wall-clock time for: construction, factorization, solve, SpMV.
 * Also reports nnz, fill-in, memory, and residual norm.
 */
#define _POSIX_C_SOURCE 199309L
#include "sparse_matrix.h"
#include "sparse_lu.h"
#include "sparse_vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <math.h>

/* ─── Portable wall-clock timer ─────────────────────────────────────── */

static double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─── Generate a diag-dominant sparse matrix ────────────────────────── */

static SparseMatrix *generate_sparse(idx_t n, unsigned seed)
{
    srand(seed);
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;

    /* Insert ~5 off-diagonal entries per row + diagonal */
    for (idx_t i = 0; i < n; i++) {
        double diag_sum = 0.0;
        int entries = (n < 10) ? n - 1 : 5;
        for (int k = 0; k < entries; k++) {
            idx_t j = (idx_t)(rand() % n);
            if (j == i) continue;
            double v = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
            sparse_insert(A, i, j, v);
            diag_sum += (v > 0 ? v : -v);
        }
        /* Ensure diagonal dominance */
        sparse_insert(A, i, i, diag_sum + 1.0);
    }

    return A;
}

/* ─── Benchmark a single matrix ─────────────────────────────────────── */

static void benchmark(SparseMatrix *A, int repeats, sparse_pivot_t pivot)
{
    idx_t n = sparse_rows(A);
    idx_t nnz_orig = sparse_nnz(A);

    printf("Matrix: %d x %d, nnz = %d\n", (int)n, (int)n, (int)nnz_orig);
    printf("Memory: %zu bytes\n", sparse_memory_usage(A));
    printf("Pivot:  %s\n", pivot == SPARSE_PIVOT_PARTIAL ? "partial" : "complete");
    printf("Repeats: %d\n\n", repeats);

    /* Prepare RHS: b = A * [1,1,...,1] */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b    = malloc((size_t)n * sizeof(double));
    double *x    = malloc((size_t)n * sizeof(double));
    double *r    = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* --- SpMV timing --- */
    double t0 = wall_time();
    for (int rep = 0; rep < repeats; rep++)
        sparse_matvec(A, ones, r);
    double t_spmv = (wall_time() - t0) / repeats;
    printf("SpMV:          %.6f s\n", t_spmv);

    /* --- LU factorization timing --- */
    double t_factor_total = 0.0;
    double t_solve_total = 0.0;
    idx_t nnz_after = 0;
    double residual = 0.0;

    for (int rep = 0; rep < repeats; rep++) {
        SparseMatrix *LU = sparse_copy(A);

        t0 = wall_time();
        sparse_err_t err = sparse_lu_factor(LU, pivot, 1e-12);
        t_factor_total += wall_time() - t0;

        if (err != SPARSE_OK) {
            printf("LU factor failed: %s\n", sparse_strerror(err));
            sparse_free(LU);
            break;
        }

        if (rep == 0)
            nnz_after = sparse_nnz(LU);

        /* --- Solve timing --- */
        t0 = wall_time();
        sparse_lu_solve(LU, b, x);
        t_solve_total += wall_time() - t0;

        /* Residual (on last rep) */
        if (rep == repeats - 1) {
            sparse_matvec(A, x, r);
            for (idx_t i = 0; i < n; i++) r[i] -= b[i];
            residual = vec_norminf(r, n);
        }

        sparse_free(LU);
    }

    printf("LU factor:     %.6f s\n", t_factor_total / repeats);
    printf("LU solve:      %.6f s\n", t_solve_total / repeats);
    printf("nnz after LU:  %d (fill-in ratio: %.2f)\n",
           (int)nnz_after, (double)nnz_after / (double)nnz_orig);
    printf("Residual:      %.3e\n", residual);

    free(ones); free(b); free(x); free(r);
}

/* ─── Tabular benchmark for directory mode ─────────────────────────── */

static void benchmark_tabular(const char *name, SparseMatrix *A,
                               int repeats, sparse_pivot_t pivot)
{
    idx_t n = sparse_rows(A);
    idx_t nnz_orig = sparse_nnz(A);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b    = malloc((size_t)n * sizeof(double));
    double *x    = malloc((size_t)n * sizeof(double));
    double *r    = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* SpMV */
    double t0 = wall_time();
    for (int rep = 0; rep < repeats; rep++)
        sparse_matvec(A, ones, r);
    double t_spmv = (wall_time() - t0) / repeats * 1000.0;

    /* Factor + solve */
    double t_factor_total = 0.0, t_solve_total = 0.0;
    idx_t nnz_after = 0;
    double residual = 0.0;
    int ok = 1;

    for (int rep = 0; rep < repeats; rep++) {
        SparseMatrix *LU = sparse_copy(A);
        t0 = wall_time();
        sparse_err_t err = sparse_lu_factor(LU, pivot, 1e-12);
        t_factor_total += wall_time() - t0;
        if (err != SPARSE_OK) {
            printf("%-20s %5d %7d   SINGULAR\n", name, (int)n, (int)nnz_orig);
            sparse_free(LU);
            ok = 0;
            break;
        }
        if (rep == 0) nnz_after = sparse_nnz(LU);
        t0 = wall_time();
        sparse_lu_solve(LU, b, x);
        t_solve_total += wall_time() - t0;
        if (rep == repeats - 1) {
            sparse_matvec(A, x, r);
            for (idx_t i = 0; i < n; i++) r[i] -= b[i];
            residual = vec_norminf(r, n);
        }
        sparse_free(LU);
    }

    if (ok) {
        printf("%-20s %5d %7d %9d %6.2f  %10.3f %10.3f %10.3f %10zu  %.3e\n",
               name, (int)n, (int)nnz_orig, (int)nnz_after,
               (double)nnz_after / (double)nnz_orig,
               t_factor_total / repeats * 1000.0,
               t_solve_total / repeats * 1000.0,
               t_spmv,
               sparse_memory_usage(A),
               residual);
    }

    free(ones); free(b); free(x); free(r);
}

/* ─── Benchmark all .mtx files in a directory ──────────────────────── */

static int ends_with(const char *s, const char *suffix)
{
    size_t slen = strlen(s), suflen = strlen(suffix);
    if (slen < suflen) return 0;
    return strcmp(s + slen - suflen, suffix) == 0;
}

static void benchmark_dir(const char *dirpath, int repeats, sparse_pivot_t pivot)
{
    DIR *d = opendir(dirpath);
    if (!d) {
        fprintf(stderr, "Cannot open directory: %s\n", dirpath);
        return;
    }

    const char *piv_name = (pivot == SPARSE_PIVOT_PARTIAL) ? "partial" : "complete";
    printf("=== Benchmarking %s (pivot=%s, repeats=%d) ===\n\n", dirpath, piv_name, repeats);
    printf("%-20s %5s %7s %9s %6s  %10s %10s %10s %10s  %s\n",
           "name", "n", "nnz", "nnz_LU", "fill",
           "factor(ms)", "solve(ms)", "spmv(ms)", "memory", "residual");
    printf("%-20s %5s %7s %9s %6s  %10s %10s %10s %10s  %s\n",
           "----", "-----", "-------", "---------", "------",
           "----------", "----------", "----------", "----------", "----------");

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!ends_with(ent->d_name, ".mtx")) continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dirpath, ent->d_name);

        SparseMatrix *A = NULL;
        sparse_err_t err = sparse_load_mm(&A, path);
        if (err != SPARSE_OK) {
            printf("%-20s  LOAD FAILED: %s\n", ent->d_name, sparse_strerror(err));
            continue;
        }

        if (sparse_rows(A) != sparse_cols(A)) {
            printf("%-20s  non-square, skipping\n", ent->d_name);
            sparse_free(A);
            continue;
        }

        /* Strip .mtx extension for display */
        char name[256];
        snprintf(name, sizeof(name), "%s", ent->d_name);
        char *dot = strrchr(name, '.');
        if (dot) *dot = '\0';

        benchmark_tabular(name, A, repeats, pivot);
        sparse_free(A);
    }

    closedir(d);
    printf("\n");
}

/* ─── Main ──────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    const char *filename = NULL;
    const char *dirpath = NULL;
    idx_t size = 100;
    int repeats = 3;
    sparse_pivot_t pivot = SPARSE_PIVOT_COMPLETE;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = (idx_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) {
            dirpath = argv[++i];
        } else if (strcmp(argv[i], "--pivot") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "partial") == 0)
                pivot = SPARSE_PIVOT_PARTIAL;
            else
                pivot = SPARSE_PIVOT_COMPLETE;
        } else if (argv[i][0] != '-') {
            filename = argv[i];
        }
    }

    if (dirpath) {
        benchmark_dir(dirpath, repeats, pivot);
        return 0;
    }

    SparseMatrix *A = NULL;

    if (filename) {
        printf("Loading %s...\n", filename);
        sparse_err_t err = sparse_load_mm(&A, filename);
        if (err != SPARSE_OK) {
            fprintf(stderr, "Failed to load %s: %s\n",
                    filename, sparse_strerror(err));
            return 1;
        }
    } else {
        printf("Generating %dx%d sparse matrix...\n", (int)size, (int)size);
        A = generate_sparse(size, 42);
        if (!A) {
            fprintf(stderr, "Failed to generate matrix\n");
            return 1;
        }
    }

    benchmark(A, repeats, pivot);
    sparse_free(A);
    return 0;
}
