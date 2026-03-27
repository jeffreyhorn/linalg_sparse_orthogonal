/*
 * bench_convergence.c — Convergence analysis for iterative solvers
 *
 * Usage:
 *   ./bench_convergence                            Run all SuiteSparse matrices
 *   ./bench_convergence --dir PATH                 Use custom matrix directory
 *   ./bench_convergence --history tests/data/suitesparse/nos4.mtx
 *                                                  Residual-vs-iteration history
 *
 * Reports: solver, preconditioner, iterations, time, residual for each matrix.
 * In --history mode, prints per-iteration residual for convergence analysis.
 */
#define _POSIX_C_SOURCE 199309L
#include "sparse_cholesky.h"
#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_vector.h"
#include <dirent.h>
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

/* Cholesky preconditioner callback */
static sparse_err_t cholesky_precond(const void *ctx, idx_t n, const double *r, double *z) {
    (void)n;
    return sparse_cholesky_solve((const SparseMatrix *)ctx, r, z);
}

static double compute_rel_residual(const SparseMatrix *A, const double *b, const double *x,
                                   idx_t n) {
    double *r = malloc((size_t)n * sizeof(double));
    if (!r)
        return NAN;
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];
    double rn = vec_norm2(r, n);
    double bn = vec_norm2(b, n);
    free(r);
    return (bn > 0.0) ? rn / bn : 0.0;
}

static int ends_with(const char *s, const char *suffix) {
    size_t slen = strlen(s), suflen = strlen(suffix);
    if (slen < suflen)
        return 0;
    return strcmp(s + slen - suflen, suffix) == 0;
}

/* Known SPD matrices */
static int is_spd_matrix(const char *name) {
    return (strstr(name, "nos4") || strstr(name, "bcsstk04"));
}

/* ─── Full convergence table for one matrix ─────────────────────────── */

static void convergence_table(const char *name, SparseMatrix *A) {
    idx_t n = sparse_rows(A);
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    if (!x_exact || !b) {
        fprintf(stderr, "convergence_table: out of memory for %s (n=%d)\n", name, (int)n);
        free(x_exact);
        free(b);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    int spd = is_spd_matrix(name);

    printf("  %s (n=%d, nnz=%d, SPD=%s):\n", name, (int)n, (int)sparse_nnz(A), spd ? "yes" : "no");
    printf("    %-20s %6s %10s %12s %6s\n", "Solver", "Iters", "Time(s)", "Residual", "Conv");
    printf("    %-20s %6s %10s %12s %6s\n", "--------------------", "------", "----------",
           "------------", "------");

    /* --- CG (SPD only) --- */
    if (spd) {
        double *x = calloc((size_t)n, sizeof(double));
        if (!x) {
            free(x_exact);
            free(b);
            return;
        }
        sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};
        sparse_iter_result_t res;
        double t0 = wall_time();
        sparse_solve_cg(A, b, x, &opts, NULL, NULL, &res);
        double t = wall_time() - t0;
        double rr = compute_rel_residual(A, b, x, n);
        printf("    %-20s %6d %10.6f %12.3e %6s\n", "CG", (int)res.iterations, t, rr,
               res.converged ? "yes" : "no");
        free(x);

        /* ILU-preconditioned CG */
        sparse_ilu_t ilu;
        if (sparse_ilu_factor(A, &ilu) == SPARSE_OK) {
            x = calloc((size_t)n, sizeof(double));
            if (x) {
                t0 = wall_time();
                sparse_solve_cg(A, b, x, &opts, sparse_ilu_precond, &ilu, &res);
                t = wall_time() - t0;
                rr = compute_rel_residual(A, b, x, n);
                printf("    %-20s %6d %10.6f %12.3e %6s\n", "ILU-CG", (int)res.iterations, t, rr,
                       res.converged ? "yes" : "no");
            }
            free(x);
            sparse_ilu_free(&ilu);
        }

        /* Cholesky-preconditioned CG */
        SparseMatrix *L = sparse_copy(A);
        if (!L) {
            fprintf(stderr, "    Cholesky-CG: sparse_copy failed\n");
        } else if (sparse_cholesky_factor(L) == SPARSE_OK) {
            x = calloc((size_t)n, sizeof(double));
            if (x) {
                t0 = wall_time();
                sparse_solve_cg(A, b, x, &opts, cholesky_precond, L, &res);
                t = wall_time() - t0;
                rr = compute_rel_residual(A, b, x, n);
                printf("    %-20s %6d %10.6f %12.3e %6s\n", "Cholesky-CG", (int)res.iterations, t,
                       rr, res.converged ? "yes" : "no");
            }
            free(x);
        }
        sparse_free(L);
    }

    /* --- GMRES --- */
    {
        double *x = calloc((size_t)n, sizeof(double));
        if (!x) {
            free(x_exact);
            free(b);
            return;
        }
        sparse_gmres_opts_t opts = {.max_iter = 2000, .restart = 50, .tol = 1e-10, .verbose = 0};
        sparse_iter_result_t res;
        double t0 = wall_time();
        sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &res);
        double t = wall_time() - t0;
        double rr = compute_rel_residual(A, b, x, n);
        printf("    %-20s %6d %10.6f %12.3e %6s\n", "GMRES(50)", (int)res.iterations, t, rr,
               res.converged ? "yes" : "no");
        free(x);
    }

    /* --- ILU-preconditioned GMRES --- */
    {
        sparse_ilu_t ilu;
        if (sparse_ilu_factor(A, &ilu) == SPARSE_OK) {
            double *x = calloc((size_t)n, sizeof(double));
            if (x) {
                sparse_gmres_opts_t opts = {
                    .max_iter = 2000, .restart = 50, .tol = 1e-10, .verbose = 0};
                sparse_iter_result_t res;
                double t0 = wall_time();
                sparse_solve_gmres(A, b, x, &opts, sparse_ilu_precond, &ilu, &res);
                double t = wall_time() - t0;
                double rr = compute_rel_residual(A, b, x, n);
                printf("    %-20s %6d %10.6f %12.3e %6s\n", "ILU-GMRES(50)", (int)res.iterations, t,
                       rr, res.converged ? "yes" : "no");
            }
            free(x);
            sparse_ilu_free(&ilu);
        } else {
            printf("    %-20s %6s %10s %12s %6s\n", "ILU-GMRES(50)", "-", "-", "(ILU fail)", "-");
        }
    }

    /* --- Direct LU --- */
    {
        SparseMatrix *LU = sparse_copy(A);
        double *x = malloc((size_t)n * sizeof(double));
        if (!LU || !x) {
            fprintf(stderr, "    LU direct: allocation failed\n");
        } else {
            double t0 = wall_time();
            sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
            if (err == SPARSE_OK) {
                sparse_err_t serr = sparse_lu_solve(LU, b, x);
                double t = wall_time() - t0;
                if (serr == SPARSE_OK) {
                    double rr = compute_rel_residual(A, b, x, n);
                    printf("    %-20s %6s %10.6f %12.3e %6s\n", "LU direct", "-", t, rr, "yes");
                } else {
                    printf("    %-20s %6s %10s %12s %6s\n", "LU direct", "-", "-", "(solve fail)",
                           "-");
                }
            }
        }
        free(x);
        sparse_free(LU);
    }

    /* --- Direct Cholesky (SPD only) --- */
    if (spd) {
        SparseMatrix *Ch = sparse_copy(A);
        double *x = malloc((size_t)n * sizeof(double));
        if (!Ch || !x) {
            fprintf(stderr, "    Cholesky direct: allocation failed\n");
        } else {
            double t0 = wall_time();
            sparse_err_t cerr = sparse_cholesky_factor(Ch);
            if (cerr == SPARSE_OK) {
                sparse_err_t serr = sparse_cholesky_solve(Ch, b, x);
                double t = wall_time() - t0;
                if (serr == SPARSE_OK) {
                    double rr = compute_rel_residual(A, b, x, n);
                    printf("    %-20s %6s %10.6f %12.3e %6s\n", "Cholesky direct", "-", t, rr,
                           "yes");
                }
            }
        }
        free(x);
        sparse_free(Ch);
    }

    printf("\n");
    free(x_exact);
    free(b);
}

/* ─── Convergence history: residual vs iteration ──────────────────── */

static void convergence_history(SparseMatrix *A, const char *name) {
    idx_t n = sparse_rows(A);
    int spd = is_spd_matrix(name);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    if (!x_exact || !b) {
        fprintf(stderr, "convergence_history: out of memory for %s (n=%d)\n", name, (int)n);
        free(x_exact);
        free(b);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    /* Record residual at increasing max_iter cutoffs */
    int cutoffs[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 0};

    if (spd) {
        printf("  %s CG convergence history:\n", name);
        printf("    %6s %12s %12s\n", "Iters", "CG", "ILU-CG");
        printf("    %6s %12s %12s\n", "------", "------------", "------------");

        sparse_ilu_t ilu;
        int have_ilu = (sparse_ilu_factor(A, &ilu) == SPARSE_OK);

        for (int c = 0; cutoffs[c] > 0; c++) {
            int mi = cutoffs[c];
            if (mi > n * 2)
                break;

            /* Plain CG */
            double *x = calloc((size_t)n, sizeof(double));
            if (!x)
                break;
            sparse_iter_opts_t opts = {.max_iter = (idx_t)mi, .tol = 1e-15, .verbose = 0};
            sparse_iter_result_t res;
            sparse_solve_cg(A, b, x, &opts, NULL, NULL, &res);
            double rr_cg = compute_rel_residual(A, b, x, n);
            free(x);

            /* ILU-CG */
            double rr_ilu = -1.0;
            if (have_ilu) {
                x = calloc((size_t)n, sizeof(double));
                if (!x)
                    break;
                sparse_solve_cg(A, b, x, &opts, sparse_ilu_precond, &ilu, &res);
                rr_ilu = compute_rel_residual(A, b, x, n);
                free(x);
            }

            if (have_ilu)
                printf("    %6d %12.3e %12.3e\n", mi, rr_cg, rr_ilu);
            else
                printf("    %6d %12.3e %12s\n", mi, rr_cg, "-");

            if (rr_cg < 1e-14 && (rr_ilu < 1e-14 || !have_ilu))
                break;
        }
        if (have_ilu)
            sparse_ilu_free(&ilu);
        printf("\n");
    }

    /* GMRES history */
    printf("  %s GMRES convergence history:\n", name);
    printf("    %6s %12s %12s\n", "Iters", "GMRES(50)", "ILU-GMRES");
    printf("    %6s %12s %12s\n", "------", "------------", "------------");

    sparse_ilu_t ilu;
    int have_ilu = (sparse_ilu_factor(A, &ilu) == SPARSE_OK);

    for (int c = 0; cutoffs[c] > 0; c++) {
        int mi = cutoffs[c];
        if (mi > n * 2)
            break;

        double *x = calloc((size_t)n, sizeof(double));
        if (!x)
            break;
        sparse_gmres_opts_t opts = {
            .max_iter = (idx_t)mi, .restart = 50, .tol = 1e-15, .verbose = 0};
        sparse_iter_result_t res;
        sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &res);
        double rr_gm = compute_rel_residual(A, b, x, n);
        free(x);

        double rr_ilu_gm = -1.0;
        if (have_ilu) {
            x = calloc((size_t)n, sizeof(double));
            if (!x)
                break;
            sparse_solve_gmres(A, b, x, &opts, sparse_ilu_precond, &ilu, &res);
            rr_ilu_gm = compute_rel_residual(A, b, x, n);
            free(x);
        }

        if (have_ilu)
            printf("    %6d %12.3e %12.3e\n", mi, rr_gm, rr_ilu_gm);
        else
            printf("    %6d %12.3e %12s\n", mi, rr_gm, "-");

        if (rr_gm < 1e-14 && (rr_ilu_gm < 1e-14 || !have_ilu))
            break;
    }
    if (have_ilu)
        sparse_ilu_free(&ilu);
    printf("\n");

    free(x_exact);
    free(b);
}

/* ─── Main ──────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *dirpath = "tests/data/suitesparse";
    const char *history_file = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc)
            dirpath = argv[++i];
        else if (strcmp(argv[i], "--history") == 0 && i + 1 < argc)
            history_file = argv[++i];
    }

    /* History mode: per-iteration convergence for one matrix */
    if (history_file) {
        SparseMatrix *A = NULL;
        if (sparse_load_mm(&A, history_file) != SPARSE_OK) {
            fprintf(stderr, "Failed to load %s\n", history_file);
            return 1;
        }
        char name[256];
        const char *base = strrchr(history_file, '/');
        snprintf(name, sizeof(name), "%s", base ? base + 1 : history_file);
        char *dot = strrchr(name, '.');
        if (dot)
            *dot = '\0';

        printf("=== Convergence History: %s ===\n\n", name);
        convergence_history(A, name);
        sparse_free(A);
        return 0;
    }

    /* Table mode: all matrices in directory */
    printf("=== Convergence Benchmark ===\n\n");

    DIR *d = opendir(dirpath);
    if (!d) {
        fprintf(stderr, "Cannot open %s\n", dirpath);
        return 1;
    }

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!ends_with(ent->d_name, ".mtx"))
            continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dirpath, ent->d_name);
        SparseMatrix *A = NULL;
        if (sparse_load_mm(&A, path) != SPARSE_OK)
            continue;
        if (sparse_rows(A) != sparse_cols(A)) {
            sparse_free(A);
            continue;
        }

        char name[256];
        snprintf(name, sizeof(name), "%s", ent->d_name);
        char *dot = strrchr(name, '.');
        if (dot)
            *dot = '\0';

        convergence_table(name, A);
        sparse_free(A);
    }
    closedir(d);

    return 0;
}
