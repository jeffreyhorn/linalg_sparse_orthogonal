/*
 * bench_main.c — Benchmark harness for sparse LU library
 *
 * Usage:
 *   ./bench_main [matrix.mtx]              Load from file
 *   ./bench_main --size N                  Generate NxN diag-dominant random sparse
 *   ./bench_main --size N --repeat R       Average over R repetitions
 *   ./bench_main --dir PATH                Benchmark all .mtx files in directory
 *   ./bench_main --dir PATH --pivot partial  Use partial pivoting (default: complete)
 *   ./bench_main --reorder rcm|amd|none      Apply fill-reducing reordering
 *   ./bench_main --cholesky                  Use Cholesky instead of LU (SPD matrices only)
 *   ./bench_main --spmv [matrix.mtx|--dir PATH]  SpMV-only benchmark (throughput, MFLOP/s)
 *   ./bench_main --iterative [matrix.mtx|--dir PATH]  Iterative solver benchmark
 *
 * Reports wall-clock time for: construction, factorization, solve, SpMV.
 * Also reports nnz, fill-in, memory, and residual norm.
 */
#define _POSIX_C_SOURCE 199309L
#include "sparse_matrix.h"
#include "sparse_lu.h"
#include "sparse_cholesky.h"
#include "sparse_reorder.h"
#include "sparse_iterative.h"
#include "sparse_ilu.h"
#include "sparse_vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <math.h>

#ifdef SPARSE_OPENMP
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <omp.h>
#pragma GCC diagnostic pop
#endif

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

static const char *reorder_name(sparse_reorder_t r)
{
    switch (r) {
    case SPARSE_REORDER_NONE: return "none";
    case SPARSE_REORDER_RCM:  return "rcm";
    case SPARSE_REORDER_AMD:  return "amd";
    }
    return "unknown";
}

static void benchmark(SparseMatrix *A, int repeats, sparse_pivot_t pivot,
                      sparse_reorder_t reorder, int use_cholesky)
{
    idx_t n = sparse_rows(A);
    idx_t nnz_orig = sparse_nnz(A);

    printf("Matrix:  %d x %d, nnz = %d\n", (int)n, (int)n, (int)nnz_orig);
    printf("Memory:  %zu bytes\n", sparse_memory_usage(A));
    printf("Solver:  %s\n", use_cholesky ? "Cholesky" : "LU");
    if (!use_cholesky)
        printf("Pivot:   %s\n", pivot == SPARSE_PIVOT_PARTIAL ? "partial" : "complete");
    printf("Reorder: %s\n", reorder_name(reorder));
    printf("Repeats: %d\n", repeats);
    printf("Bandwidth: %d\n\n", (int)sparse_bandwidth(A));

    /* Prepare RHS: b = A * [1,1,...,1] */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b    = malloc((size_t)n * sizeof(double));
    double *x    = malloc((size_t)n * sizeof(double));
    double *r    = malloc((size_t)n * sizeof(double));
    if (!ones || !b || !x || !r) {
        fprintf(stderr, "Allocation failed\n");
        free(ones); free(b); free(x); free(r);
        return;
    }
    for (idx_t i = 0; i < n; i++) ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* --- SpMV timing --- */
    double t0 = wall_time();
    for (int rep = 0; rep < repeats; rep++)
        sparse_matvec(A, ones, r);
    double t_spmv = (wall_time() - t0) / repeats;
    printf("SpMV:          %.6f s\n", t_spmv);

    /* --- Factorization timing --- */
    double t_factor_total = 0.0;
    double t_solve_total = 0.0;
    idx_t nnz_after = 0;
    double residual = 0.0;
    double cond_est = 0.0;

    sparse_lu_opts_t lu_opts = { pivot, reorder, 1e-12 };
    sparse_cholesky_opts_t chol_opts = { reorder };
    int reps_done = 0;

    for (int rep = 0; rep < repeats; rep++) {
        SparseMatrix *F = sparse_copy(A);
        if (!F) {
            fprintf(stderr, "Allocation failed during benchmark\n");
            break;
        }

        t0 = wall_time();
        sparse_err_t err;
        if (use_cholesky)
            err = sparse_cholesky_factor_opts(F, &chol_opts);
        else
            err = sparse_lu_factor_opts(F, &lu_opts);
        double t_factor = wall_time() - t0;

        if (err != SPARSE_OK) {
            printf("%s factor failed: %s\n",
                   use_cholesky ? "Cholesky" : "LU", sparse_strerror(err));
            sparse_free(F);
            break;
        }
        t_factor_total += t_factor;

        if (rep == 0) {
            nnz_after = sparse_nnz(F);
            if (!use_cholesky) {
                if (sparse_lu_condest(A, F, &cond_est) != SPARSE_OK)
                    cond_est = -1.0;
            }
        }

        /* --- Solve timing --- */
        t0 = wall_time();
        sparse_err_t serr;
        if (use_cholesky)
            serr = sparse_cholesky_solve(F, b, x);
        else
            serr = sparse_lu_solve(F, b, x);
        double t_solve = wall_time() - t0;

        if (serr != SPARSE_OK) {
            printf("%s solve failed: %s\n",
                   use_cholesky ? "Cholesky" : "LU", sparse_strerror(serr));
            sparse_free(F);
            break;
        }
        t_solve_total += t_solve;

        /* Compute residual on every successful rep (last one is reported) */
        sparse_matvec(A, x, r);
        for (idx_t i = 0; i < n; i++) r[i] -= b[i];
        residual = vec_norminf(r, n);

        sparse_free(F);
        reps_done++;
    }

    if (reps_done == 0) {
        printf("No successful repetitions.\n");
        free(ones); free(b); free(x); free(r);
        return;
    }

    printf("Factor:        %.6f s\n", t_factor_total / reps_done);
    printf("Solve:         %.6f s\n", t_solve_total / reps_done);
    printf("nnz after:     %d (fill-in ratio: %.2f)\n",
           (int)nnz_after, (double)nnz_after / (double)nnz_orig);
    printf("Residual:      %.3e\n", residual);
    if (!use_cholesky)
        printf("Cond est:      %.3e\n", cond_est);
    else
        printf("Cond est:      N/A (Cholesky)\n");

    free(ones); free(b); free(x); free(r);
}

/* ─── Tabular benchmark for directory mode ─────────────────────────── */

static void benchmark_tabular(const char *name, SparseMatrix *A,
                               int repeats, sparse_pivot_t pivot,
                               sparse_reorder_t reorder)
{
    idx_t n = sparse_rows(A);
    idx_t nnz_orig = sparse_nnz(A);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b    = malloc((size_t)n * sizeof(double));
    double *x    = malloc((size_t)n * sizeof(double));
    double *r    = malloc((size_t)n * sizeof(double));
    if (!ones || !b || !x || !r) {
        fprintf(stderr, "benchmark_tabular: failed to allocate workspace\n");
        free(ones); free(b); free(x); free(r);
        return;
    }
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
    double cond_est = 0.0;
    int ok = 1;

    sparse_lu_opts_t opts = { pivot, reorder, 1e-12 };

    for (int rep = 0; rep < repeats; rep++) {
        SparseMatrix *LU = sparse_copy(A);
        t0 = wall_time();
        sparse_err_t err = sparse_lu_factor_opts(LU, &opts);
        t_factor_total += wall_time() - t0;
        if (err != SPARSE_OK) {
            printf("%-20s %5d %7d   SINGULAR\n", name, (int)n, (int)nnz_orig);
            sparse_free(LU);
            ok = 0;
            break;
        }
        if (rep == 0) {
            nnz_after = sparse_nnz(LU);
            if (sparse_lu_condest(A, LU, &cond_est) != SPARSE_OK)
                cond_est = -1.0;
        }
        t0 = wall_time();
        err = sparse_lu_solve(LU, b, x);
        t_solve_total += wall_time() - t0;
        if (err != SPARSE_OK) {
            printf("%-20s %5d %7d   SOLVE FAILED: %s\n",
                   name, (int)n, (int)nnz_orig, sparse_strerror(err));
            sparse_free(LU);
            ok = 0;
            break;
        }
        if (rep == repeats - 1) {
            sparse_matvec(A, x, r);
            for (idx_t i = 0; i < n; i++) r[i] -= b[i];
            residual = vec_norminf(r, n);
        }
        sparse_free(LU);
    }

    if (ok) {
        printf("%-20s %5d %7d %9d %6.2f  %10.3f %10.3f %10.3f %10zu  %.3e  %.3e\n",
               name, (int)n, (int)nnz_orig, (int)nnz_after,
               nnz_orig > 0 ? (double)nnz_after / (double)nnz_orig : 0.0,
               t_factor_total / repeats * 1000.0,
               t_solve_total / repeats * 1000.0,
               t_spmv,
               sparse_memory_usage(A),
               residual, cond_est);
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

static void benchmark_dir(const char *dirpath, int repeats,
                          sparse_pivot_t pivot, sparse_reorder_t reorder)
{
    DIR *d = opendir(dirpath);
    if (!d) {
        fprintf(stderr, "Cannot open directory: %s\n", dirpath);
        return;
    }

    const char *piv_name = (pivot == SPARSE_PIVOT_PARTIAL) ? "partial" : "complete";
    printf("=== Benchmarking %s (pivot=%s, reorder=%s, repeats=%d) ===\n\n",
           dirpath, piv_name, reorder_name(reorder), repeats);
    printf("%-20s %5s %7s %9s %6s  %10s %10s %10s %10s  %10s  %10s\n",
           "name", "n", "nnz", "nnz_LU", "fill",
           "factor(ms)", "solve(ms)", "spmv(ms)", "memory", "residual", "condest");
    printf("%-20s %5s %7s %9s %6s  %10s %10s %10s %10s  %10s  %10s\n",
           "----", "-----", "-------", "---------", "------",
           "----------", "----------", "----------", "----------",
           "----------", "----------");

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

        benchmark_tabular(name, A, repeats, pivot, reorder);
        sparse_free(A);
    }

    closedir(d);
    printf("\n");
}

/* ─── SpMV-only benchmark ──────────────────────────────────────────── */

static void benchmark_spmv(SparseMatrix *A, const char *name, int iters)
{
    idx_t n = sparse_rows(A);
    idx_t nnz = sparse_nnz(A);

    double *x = malloc((size_t)n * sizeof(double));
    double *y = malloc((size_t)n * sizeof(double));
    if (!x || !y) { free(x); free(y); return; }
    for (idx_t i = 0; i < n; i++) x[i] = 1.0;

    /* Warm up */
    sparse_matvec(A, x, y);

    double t0 = wall_time();
    for (int i = 0; i < iters; i++)
        sparse_matvec(A, x, y);
    double elapsed = wall_time() - t0;

    double per_spmv = elapsed / iters;
    /* 2 flops per nnz (multiply + add) */
    double mflops = 2.0 * (double)nnz / per_spmv / 1e6;

    printf("  %-20s  n=%5d  nnz=%7d  %d iters  %.6f s/iter  %.1f MFLOP/s\n",
           name, (int)n, (int)nnz, iters, per_spmv, mflops);

    free(x); free(y);
}

static void benchmark_spmv_dir(const char *dirpath, int iters)
{
#ifdef SPARSE_OPENMP
    printf("=== SpMV Benchmark (OpenMP, max_threads=%d) ===\n\n", omp_get_max_threads());
#else
    printf("=== SpMV Benchmark (serial) ===\n\n");
#endif

    DIR *d = opendir(dirpath);
    if (!d) { fprintf(stderr, "Cannot open directory: %s\n", dirpath); return; }

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!ends_with(ent->d_name, ".mtx")) continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dirpath, ent->d_name);
        SparseMatrix *A = NULL;
        if (sparse_load_mm(&A, path) != SPARSE_OK) continue;
        if (sparse_rows(A) != sparse_cols(A)) { sparse_free(A); continue; }

        char name[256];
        snprintf(name, sizeof(name), "%s", ent->d_name);
        char *dot = strrchr(name, '.'); if (dot) *dot = '\0';

        benchmark_spmv(A, name, iters);
        sparse_free(A);
    }
    closedir(d);
    printf("\n");
}

/* ─── Iterative solver benchmark ───────────────────────────────────── */

static void benchmark_iterative(SparseMatrix *A, const char *name, int is_spd)
{
    idx_t n = sparse_rows(A);
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    if (!x_exact || !b) { free(x_exact); free(b); return; }
    for (idx_t i = 0; i < n; i++) x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    printf("  %s (n=%d, nnz=%d):\n", name, (int)n, (int)sparse_nnz(A));

    /* --- CG (SPD only) --- */
    if (is_spd) {
        double *x = calloc((size_t)n, sizeof(double));
        if (!x) { fprintf(stderr, "    CG: allocation failed\n"); free(x_exact); free(b); return; }
        sparse_iter_opts_t cg_opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};
        sparse_iter_result_t res;
        double t0 = wall_time();
        sparse_solve_cg(A, b, x, &cg_opts, NULL, NULL, &res);
        double t_cg = wall_time() - t0;
        printf("    CG:          %4d iters  %.6f s  res=%.3e  conv=%d\n",
               (int)res.iterations, t_cg, res.residual_norm, res.converged);
        free(x);

        /* ILU-preconditioned CG */
        sparse_ilu_t ilu;
        if (sparse_ilu_factor(A, &ilu) == SPARSE_OK) {
            x = calloc((size_t)n, sizeof(double));
            t0 = wall_time();
            sparse_solve_cg(A, b, x, &cg_opts, sparse_ilu_precond, &ilu, &res);
            double t_pcg = wall_time() - t0;
            printf("    ILU-CG:      %4d iters  %.6f s  res=%.3e  conv=%d\n",
                   (int)res.iterations, t_pcg, res.residual_norm, res.converged);
            free(x);
            sparse_ilu_free(&ilu);
        }
    }

    /* --- GMRES --- */
    {
        double *x = calloc((size_t)n, sizeof(double));
        if (!x) { fprintf(stderr, "    GMRES: allocation failed\n"); free(x_exact); free(b); return; }
        sparse_gmres_opts_t gm_opts = {.max_iter = 2000, .restart = 50, .tol = 1e-10, .verbose = 0};
        sparse_iter_result_t res;
        double t0 = wall_time();
        sparse_solve_gmres(A, b, x, &gm_opts, NULL, NULL, &res);
        double t_gm = wall_time() - t0;
        printf("    GMRES(50):   %4d iters  %.6f s  res=%.3e  conv=%d\n",
               (int)res.iterations, t_gm, res.residual_norm, res.converged);
        free(x);

        /* ILU-preconditioned GMRES */
        sparse_ilu_t ilu;
        if (sparse_ilu_factor(A, &ilu) == SPARSE_OK) {
            x = calloc((size_t)n, sizeof(double));
            t0 = wall_time();
            sparse_solve_gmres(A, b, x, &gm_opts, sparse_ilu_precond, &ilu, &res);
            double t_pgm = wall_time() - t0;
            printf("    ILU-GMRES:   %4d iters  %.6f s  res=%.3e  conv=%d\n",
                   (int)res.iterations, t_pgm, res.residual_norm, res.converged);
            free(x);
            sparse_ilu_free(&ilu);
        } else {
            printf("    ILU-GMRES:   (ILU factor failed — zero pivot)\n");
        }
    }

    /* --- Direct (LU) for comparison --- */
    {
        SparseMatrix *LU = sparse_copy(A);
        double *x = malloc((size_t)n * sizeof(double));
        if (!LU || !x) {
            fprintf(stderr, "    LU direct: allocation failed, skipping\n");
        } else {
            double t0 = wall_time();
            sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
            double t_factor = wall_time() - t0;
            if (err == SPARSE_OK) {
                t0 = wall_time();
                sparse_lu_solve(LU, b, x);
                double t_solve = wall_time() - t0;
                printf("    LU direct:         factor=%.6f s  solve=%.6f s\n",
                       t_factor, t_solve);
            }
        }
        free(x);
        sparse_free(LU);
    }

    free(x_exact); free(b);
}

static void benchmark_iterative_dir(const char *dirpath)
{
    printf("=== Iterative Solver Benchmark ===\n\n");

    /* Known SPD matrices */
    const char *spd[] = {"nos4.mtx", "bcsstk04.mtx", NULL};

    DIR *d = opendir(dirpath);
    if (!d) { fprintf(stderr, "Cannot open directory: %s\n", dirpath); return; }

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!ends_with(ent->d_name, ".mtx")) continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dirpath, ent->d_name);
        SparseMatrix *A = NULL;
        if (sparse_load_mm(&A, path) != SPARSE_OK) continue;
        if (sparse_rows(A) != sparse_cols(A)) { sparse_free(A); continue; }

        char name[256];
        snprintf(name, sizeof(name), "%s", ent->d_name);
        char *dot = strrchr(name, '.'); if (dot) *dot = '\0';

        int is_spd = 0;
        for (int s = 0; spd[s]; s++) {
            if (strcmp(ent->d_name, spd[s]) == 0) { is_spd = 1; break; }
        }

        benchmark_iterative(A, name, is_spd);
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
    sparse_reorder_t reorder = SPARSE_REORDER_NONE;
    int use_cholesky = 0;
    int mode_spmv = 0;
    int mode_iterative = 0;
    int spmv_iters = 1000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--spmv") == 0) {
            mode_spmv = 1;
        } else if (strcmp(argv[i], "--iterative") == 0) {
            mode_iterative = 1;
        } else if (strcmp(argv[i], "--spmv-iters") == 0 && i + 1 < argc) {
            spmv_iters = atoi(argv[++i]);
            if (spmv_iters <= 0) {
                fprintf(stderr, "Error: --spmv-iters must be > 0\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = (idx_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) {
            dirpath = argv[++i];
        } else if (strcmp(argv[i], "--pivot") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "partial") == 0)
                pivot = SPARSE_PIVOT_PARTIAL;
            else if (strcmp(argv[i], "complete") == 0)
                pivot = SPARSE_PIVOT_COMPLETE;
            else {
                fprintf(stderr, "Error: unknown pivot mode '%s' (use 'partial' or 'complete')\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--reorder") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "none") == 0)
                reorder = SPARSE_REORDER_NONE;
            else if (strcmp(argv[i], "rcm") == 0)
                reorder = SPARSE_REORDER_RCM;
            else if (strcmp(argv[i], "amd") == 0)
                reorder = SPARSE_REORDER_AMD;
            else {
                fprintf(stderr, "Error: unknown reorder mode '%s' (use 'none', 'rcm', or 'amd')\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--cholesky") == 0) {
            use_cholesky = 1;
        } else if (argv[i][0] != '-') {
            filename = argv[i];
        }
    }

    if (repeats < 1) {
        fprintf(stderr, "Error: --repeat must be >= 1\n");
        return 1;
    }

    /* SpMV-only benchmark mode */
    if (mode_spmv) {
        if (dirpath) {
            benchmark_spmv_dir(dirpath, spmv_iters);
        } else if (filename) {
            SparseMatrix *A = NULL;
            if (sparse_load_mm(&A, filename) != SPARSE_OK) {
                fprintf(stderr, "Failed to load %s\n", filename);
                return 1;
            }
            if (sparse_rows(A) != sparse_cols(A)) {
                fprintf(stderr, "Error: %s is non-square (%dx%d), skipping SpMV benchmark\n",
                        filename, (int)sparse_rows(A), (int)sparse_cols(A));
                sparse_free(A);
                return 1;
            }
            benchmark_spmv(A, filename, spmv_iters);
            sparse_free(A);
        } else {
            SparseMatrix *A = generate_sparse(size, 42);
            benchmark_spmv(A, "generated", spmv_iters);
            sparse_free(A);
        }
        return 0;
    }

    /* Iterative solver benchmark mode */
    if (mode_iterative) {
        if (dirpath) {
            benchmark_iterative_dir(dirpath);
        } else if (filename) {
            SparseMatrix *A = NULL;
            if (sparse_load_mm(&A, filename) != SPARSE_OK) {
                fprintf(stderr, "Failed to load %s\n", filename);
                return 1;
            }
            if (sparse_rows(A) != sparse_cols(A)) {
                fprintf(stderr, "Error: %s is non-square (%dx%d), skipping iterative benchmark\n",
                        filename, (int)sparse_rows(A), (int)sparse_cols(A));
                sparse_free(A);
                return 1;
            }
            benchmark_iterative(A, filename, sparse_is_symmetric(A, 1e-10));
            sparse_free(A);
        } else {
            SparseMatrix *A = generate_sparse(size, 42);
            benchmark_iterative(A, "generated", 0);
            sparse_free(A);
        }
        return 0;
    }

    if (dirpath) {
        benchmark_dir(dirpath, repeats, pivot, reorder);
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

    benchmark(A, repeats, pivot, reorder, use_cholesky);
    sparse_free(A);
    return 0;
}
