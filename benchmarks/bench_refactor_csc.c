/*
 * bench_refactor_csc.c — analyze-once / factor-many direct benchmark
 *
 * Default mode keeps the original SPD / Cholesky repeated-run corpus:
 *
 *   public repeated-run path:
 *     sparse_analyze + sparse_factor_numeric + sparse_refactor_numeric
 *   direct CSC path:
 *     chol_csc_from_sparse_with_analysis + chol_csc_eliminate_supernodal
 *
 * Sprint 53 adds a bounded LDL^T indefinite mode:
 *
 *   ./bench_refactor_csc --indefinite-kkt [--repeat N]
 *
 * That mode builds a synthetic above-threshold KKT saddle-point matrix and
 * compares:
 *
 *   public repeated-run path:
 *     sparse_analyze + sparse_factor_numeric + sparse_refactor_numeric
 *   direct CSC completion path:
 *     ldlt_csc_prepare_resolved_analysis +
 *     ldlt_csc_factor_with_resolved_analysis
 *
 * The benchmark stays intentionally narrow:
 *
 * - one analyze call is amortized across N same-pattern numeric refreshes
 * - value perturbations happen outside the timed region
 * - solves remain timed separately from refactor work
 * - the direct CSC side measures the resolved-analysis completion seam
 *   explicitly instead of re-entering the public one-shot wrapper path
 *
 * Output is CSV on stdout:
 *   benchmark,category,matrix,scenario,n,nnz,analyze_ms,
 *   refactor_public_ms,refactor_csc_ms,
 *   solve_public_ms,solve_csc_ms,
 *   speedup_refactor,res_public,res_csc
 *
 * `speedup_refactor = refactor_public_ms / refactor_csc_ms`; > 1.0 means the
 * direct CSC completion path is faster than the public repeated-run path on
 * that workflow.
 */
#define _POSIX_C_SOURCE 200809L

#include "sparse_analysis.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "sparse_matrix_internal.h"

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
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
        /* Sentinel: return +INFINITY so the caller's residual-below-
         * tolerance check fires visibly under OOM instead of reading
         * an uninitialised buffer. */
        fprintf(stderr, "bench_refactor_csc: malloc failed in rel_residual (n=%d)\n", (int)n);
        return (double)INFINITY;
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

/* Deterministic symmetric-key noise in [-0.5, 0.5).  For a stored
 * (i, j) entry we key on `min(i, j) * n + max(i, j)` so (i, j) and
 * (j, i) get identical noise — required to preserve symmetry, which
 * `sparse_refactor_numeric` enforces via `sparse_is_symmetric` inside
 * `sparse_cholesky_factor`. */
static double symmetric_noise(idx_t i, idx_t j, idx_t n, uint64_t seed) {
    idx_t a = (i < j) ? i : j;
    idx_t b = (i < j) ? j : i;
    uint64_t key = (uint64_t)a * (uint64_t)n + (uint64_t)b + seed;
    uint64_t h = key * 0x9e3779b97f4a7c15ULL;
    return (double)(h >> 32) / (double)(1ULL << 32) - 0.5;
}

/* Walk A's row_headers and multiply every stored value by
 * 1 + eps * symmetric_noise(i, j, iter).  The noise key is
 * symmetric in (i, j) so A[i,j] and A[j,i] receive the same
 * multiplier and the matrix stays symmetric — mandatory for
 * `sparse_refactor_numeric` which calls `sparse_is_symmetric` inside
 * `sparse_cholesky_factor`.  The main loop calls this with
 * `eps = 1e-9`, which keeps the matrices in the default corpus
 * comfortably inside SPD territory. */
static void perturb_values_in_place(SparseMatrix *A, double eps, uint64_t seed) {
    idx_t n = sparse_rows(A);
    for (idx_t phys_i = 0; phys_i < n; phys_i++) {
        Node *node = A->row_headers[phys_i];
        while (node) {
            /* Fresh matrices from sparse_load_mm have identity perms,
             * so phys_i == logical_row and node->col is logical_col
             * — no permutation bookkeeping needed here. */
            double rnd = symmetric_noise(phys_i, node->col, n, seed);
            node->value *= 1.0 + eps * rnd;
            node = node->right;
        }
    }
    /* cached_norm is now stale — invalidate it. */
    atomic_store_explicit(&A->cached_norm, -1.0, memory_order_relaxed);
}

static SparseMatrix *build_kkt_150(void) {
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

static sparse_err_t perturb_kkt_values_in_place(SparseMatrix *A, idx_t n_top, idx_t n_bot,
                                                double scale) {
    for (idx_t i = 0; i < n_top; i++) {
        double diag = 6.0 + scale * (double)((i % 7) - 3);
        sparse_err_t err = sparse_set(A, i, i, diag);
        if (err != SPARSE_OK)
            return err;
        if (i > 0) {
            double offdiag = -1.0 - 0.1 * scale * (double)(i % 3);
            err = sparse_set(A, i, i - 1, offdiag);
            if (err != SPARSE_OK)
                return err;
            err = sparse_set(A, i - 1, i, offdiag);
            if (err != SPARSE_OK)
                return err;
        }
    }

    for (idx_t j = 0; j < n_bot; j++) {
        double coupling = 1.0 + 0.05 * scale * (double)((j % 5) - 2);
        sparse_err_t err = sparse_set(A, n_top + j, j, coupling);
        if (err != SPARSE_OK)
            return err;
        err = sparse_set(A, j, n_top + j, coupling);
        if (err != SPARSE_OK)
            return err;
    }

    return SPARSE_OK;
}

static void emit_csv_row(const char *matrix, const char *scenario, idx_t n, idx_t nnz,
                         double analyze_ms, double refactor_public_ms, double refactor_csc_ms,
                         double solve_public_ms, double solve_csc_ms, double speedup,
                         double res_public, double res_csc) {
    printf("bench_refactor_csc,proof,%s,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2e,%.2e\n", matrix,
           scenario, (int)n, (int)nnz, analyze_ms, refactor_public_ms, refactor_csc_ms,
           solve_public_ms, solve_csc_ms, speedup, res_public, res_csc);
}

/* SPD / Cholesky matrix runner: analyze once, refactor N times, emit one CSV row. */
static int bench_spd_matrix(const char *path, int repeat) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, path) != SPARSE_OK) {
        fprintf(stderr, "bench_refactor_csc: failed to load %s\n", path);
        return 1;
    }
    idx_t n = sparse_rows(A);
    idx_t nnz = sparse_nnz(A);

    /* RHS b = A * [1, 1, ..., 1] — same fixture as bench_chol_csc. */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x_public = calloc((size_t)n, sizeof(double));
    double *x_csc = calloc((size_t)n, sizeof(double));
    if (!ones || !b || !x_public || !x_csc) {
        fprintf(stderr, "bench_refactor_csc: malloc failed in bench_matrix (n=%d)\n", (int)n);
        free(ones);
        free(b);
        free(x_public);
        free(x_csc);
        sparse_free(A);
        return 1;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* One analyze call — outside the timed region, reused across all
     * refactor iterations on both backends.  This is the cost that
     * amortises. */
    sparse_analysis_opts_t aopts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_t an = {0};
    double t0 = wall_time();
    sparse_err_t err = sparse_analyze(A, &aopts, &an);
    double analyze_ms = (wall_time() - t0) * 1000.0;
    if (err != SPARSE_OK) {
        fprintf(stderr, "bench_refactor_csc: sparse_analyze failed on %s (err=%d)\n", path,
                (int)err);
        /* sparse_analyze may have partially populated `an` before failing;
         * sparse_analysis_free is safe on a zero-initialised struct. */
        sparse_analysis_free(&an);
        free(ones);
        free(b);
        free(x_public);
        free(x_csc);
        sparse_free(A);
        return 1;
    }

    /* Prime the LL factors via a non-timed factor_numeric — subsequent
     * refactor calls can then refactor in place.  Uses the original A
     * (unperturbed) for priming. */
    sparse_factors_t factors_public = {0};
    if (sparse_factor_numeric(A, &an, &factors_public) != SPARSE_OK) {
        fprintf(stderr, "bench_refactor_csc: priming sparse_factor_numeric failed on %s\n", path);
        sparse_analysis_free(&an);
        free(ones);
        free(b);
        free(x_public);
        free(x_csc);
        sparse_free(A);
        return 1;
    }

    /* Timed region: N refactors on both backends. */
    double refactor_public_total = 0.0, refactor_csc_total = 0.0;
    double solve_public_total = 0.0, solve_csc_total = 0.0;
    double res_public = 0.0, res_csc = 0.0;
    int ok = 1;

    for (int rep = 0; rep < repeat; rep++) {
        SparseMatrix *A_perturb = sparse_copy(A);
        if (!A_perturb) {
            ok = 0;
            break;
        }
        /* `eps = 1e-9` is small enough that poorly-conditioned SPD
         * fixtures (e.g. s3rmt3m3, where the minimum diagonal pivot
         * after Cholesky sits close to the symmetry-check tolerance)
         * stay comfortably inside SPD territory across N refactors,
         * while still producing a value change that `sparse_refactor_numeric`
         * re-flows through elimination rather than trivially returning
         * the cached factor. */
        perturb_values_in_place(A_perturb, 1e-9, (uint64_t)rep * 0xcafef00dULL);

        /* Public repeated-run refactor path. */
        double t_public0 = wall_time();
        sparse_err_t e_public = sparse_refactor_numeric(A_perturb, &an, &factors_public);
        refactor_public_total += wall_time() - t_public0;
        if (e_public != SPARSE_OK) {
            fprintf(stderr,
                    "bench_refactor_csc: sparse_refactor_numeric failed on %s (rep=%d, err=%d)\n",
                    path, rep, (int)e_public);
            sparse_free(A_perturb);
            ok = 0;
            break;
        }

        /* CSC refactor: from_sparse_with_analysis + eliminate_supernodal.
         * Freed and rebuilt per iteration — same pattern as
         * sparse_refactor_numeric internally.  Includes the CSC
         * build + symbolic materialisation, matching the LL side's
         * build_permuted_copy + factor cost structure. */
        CholCsc *L_csc = NULL;
        double t_csc0 = wall_time();
        sparse_err_t e_csc = chol_csc_from_sparse_with_analysis(A_perturb, &an, &L_csc);
        if (e_csc == SPARSE_OK)
            e_csc = chol_csc_eliminate_supernodal(L_csc, 4);
        refactor_csc_total += wall_time() - t_csc0;
        if (e_csc != SPARSE_OK) {
            fprintf(stderr, "bench_refactor_csc: CSC refactor failed on %s (rep=%d, err=%d)\n",
                    path, rep, (int)e_csc);
            chol_csc_free(L_csc);
            sparse_free(A_perturb);
            ok = 0;
            break;
        }

        /* Solve on both backends; residuals measured vs the PERTURBED
         * A so they should be within round-off on every iteration. */
        double t_spublic0 = wall_time();
        sparse_err_t e_spublic = sparse_factor_solve(&factors_public, &an, b, x_public);
        solve_public_total += wall_time() - t_spublic0;

        double t_scsc0 = wall_time();
        sparse_err_t e_scsc = chol_csc_solve_perm(L_csc, an.perm, b, x_csc);
        solve_csc_total += wall_time() - t_scsc0;

        if (e_spublic != SPARSE_OK || e_scsc != SPARSE_OK) {
            fprintf(stderr, "bench_refactor_csc: solve failed on %s (rep=%d)\n", path, rep);
            chol_csc_free(L_csc);
            sparse_free(A_perturb);
            ok = 0;
            break;
        }

        /* Last-iteration residuals vs the PERTURBED A used for this
         * iteration's refactor/solve path.  Keep `A_perturb` here to
         * report how accurately each backend solved the matrix it was
         * actually given on the final iteration — this is the honest
         * "did the numeric factorization work?" check for the
         * analyze-once / factor-many workflow.  (Residuals against the
         * original `A` would be dominated by `b = A * ones` vs
         * `A_perturb * x`, which is `1e-9`-level noise, not a
         * factorization quality signal.) */
        if (rep == repeat - 1) {
            res_public = rel_residual(A_perturb, x_public, b);
            res_csc = rel_residual(A_perturb, x_csc, b);
        }

        chol_csc_free(L_csc);
        sparse_free(A_perturb);
    }

    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;

    if (!ok) {
        fprintf(stderr, "bench_refactor_csc: %s — aborted, partial timings discarded\n", base);
        sparse_factor_free(&factors_public);
        sparse_analysis_free(&an);
        free(ones);
        free(b);
        free(x_public);
        free(x_csc);
        sparse_free(A);
        return 1;
    }

    double refactor_public_ms = refactor_public_total * 1000.0 / (double)repeat;
    double refactor_csc_ms = refactor_csc_total * 1000.0 / (double)repeat;
    double solve_public_ms = solve_public_total * 1000.0 / (double)repeat;
    double solve_csc_ms = solve_csc_total * 1000.0 / (double)repeat;
    double speedup = refactor_public_ms / refactor_csc_ms;

    emit_csv_row(base, "chol_spd", n, nnz, analyze_ms, refactor_public_ms, refactor_csc_ms,
                 solve_public_ms, solve_csc_ms, speedup, res_public, res_csc);

    sparse_factor_free(&factors_public);
    sparse_analysis_free(&an);
    free(ones);
    free(b);
    free(x_public);
    free(x_csc);
    sparse_free(A);
    return 0;
}

static int bench_indefinite_kkt(int repeat) {
    const idx_t n_top = 140;
    const idx_t n_bot = 10;
    SparseMatrix *A = build_kkt_150();
    if (!A) {
        fprintf(stderr, "bench_refactor_csc: failed to build kkt-150\n");
        return 1;
    }
    idx_t n = sparse_rows(A);
    idx_t nnz = sparse_nnz(A);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x_public = calloc((size_t)n, sizeof(double));
    double *x_csc = calloc((size_t)n, sizeof(double));
    if (!ones || !b || !x_public || !x_csc) {
        fprintf(stderr, "bench_refactor_csc: malloc failed in bench_indefinite_kkt (n=%d)\n",
                (int)n);
        free(ones);
        free(b);
        free(x_public);
        free(x_csc);
        sparse_free(A);
        return 1;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    sparse_analysis_opts_t aopts = {
        .factor_type = SPARSE_FACTOR_LDLT,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_t an = {0};
    double t0 = wall_time();
    sparse_err_t err = sparse_analyze(A, &aopts, &an);
    double analyze_ms = (wall_time() - t0) * 1000.0;
    if (err != SPARSE_OK) {
        fprintf(stderr, "bench_refactor_csc: sparse_analyze failed on kkt-150 (err=%d)\n",
                (int)err);
        sparse_analysis_free(&an);
        free(ones);
        free(b);
        free(x_public);
        free(x_csc);
        sparse_free(A);
        return 1;
    }

    sparse_factors_t factors_public = {0};
    if (sparse_factor_numeric(A, &an, &factors_public) != SPARSE_OK) {
        fprintf(stderr, "bench_refactor_csc: priming sparse_factor_numeric failed on kkt-150\n");
        sparse_analysis_free(&an);
        free(ones);
        free(b);
        free(x_public);
        free(x_csc);
        sparse_free(A);
        return 1;
    }

    double refactor_public_total = 0.0, refactor_csc_total = 0.0;
    double solve_public_total = 0.0, solve_csc_total = 0.0;
    double res_public = 0.0, res_csc = 0.0;
    int ok = 1;

    for (int rep = 0; rep < repeat; rep++) {
        SparseMatrix *A_perturb = sparse_copy(A);
        if (!A_perturb) {
            ok = 0;
            break;
        }
        sparse_err_t perturb_err =
            perturb_kkt_values_in_place(A_perturb, n_top, n_bot, 0.20 + 0.03 * (double)(rep % 5));
        if (perturb_err != SPARSE_OK) {
            fprintf(stderr,
                    "bench_refactor_csc: kkt perturbation failed on kkt-150 "
                    "(rep=%d, err=%d)\n",
                    rep, (int)perturb_err);
            sparse_free(A_perturb);
            ok = 0;
            break;
        }

        double t_public0 = wall_time();
        sparse_err_t e_public = sparse_refactor_numeric(A_perturb, &an, &factors_public);
        refactor_public_total += wall_time() - t_public0;
        if (e_public != SPARSE_OK) {
            fprintf(stderr,
                    "bench_refactor_csc: sparse_refactor_numeric failed on kkt-150 "
                    "(rep=%d, err=%d)\n",
                    rep, (int)e_public);
            sparse_free(A_perturb);
            ok = 0;
            break;
        }

        LdltCsc *pre_factor = NULL;
        SparseMatrix *A_perm = NULL;
        sparse_analysis_t derived_analysis = {0};
        const SparseMatrix *factored_mat = NULL;
        const sparse_analysis_t *resolved_analysis = NULL;
        sparse_ldlt_t ldlt_csc = {0};

        double t_csc0 = wall_time();
        sparse_err_t e_csc = ldlt_csc_prepare_resolved_analysis(A_perturb, &an, &pre_factor,
                                                                &A_perm, &derived_analysis,
                                                                &factored_mat, &resolved_analysis);
        if (e_csc == SPARSE_OK) {
            e_csc = ldlt_csc_factor_with_resolved_analysis(
                factored_mat, resolved_analysis, pre_factor, /*min_size=*/2, 0.0, &ldlt_csc);
        }
        refactor_csc_total += wall_time() - t_csc0;
        if (e_csc != SPARSE_OK) {
            fprintf(stderr,
                    "bench_refactor_csc: LDLT CSC refactor failed on kkt-150 "
                    "(rep=%d, err=%d)\n",
                    rep, (int)e_csc);
            sparse_ldlt_free(&ldlt_csc);
            ldlt_csc_free(pre_factor);
            sparse_analysis_free(&derived_analysis);
            sparse_free(A_perm);
            sparse_free(A_perturb);
            ok = 0;
            break;
        }

        double t_spublic0 = wall_time();
        sparse_err_t e_spublic = sparse_factor_solve(&factors_public, &an, b, x_public);
        solve_public_total += wall_time() - t_spublic0;

        double t_scsc0 = wall_time();
        sparse_err_t e_scsc = sparse_ldlt_solve(&ldlt_csc, b, x_csc);
        solve_csc_total += wall_time() - t_scsc0;

        if (e_spublic != SPARSE_OK || e_scsc != SPARSE_OK) {
            fprintf(stderr, "bench_refactor_csc: solve failed on kkt-150 (rep=%d)\n", rep);
            sparse_ldlt_free(&ldlt_csc);
            ldlt_csc_free(pre_factor);
            sparse_analysis_free(&derived_analysis);
            sparse_free(A_perm);
            sparse_free(A_perturb);
            ok = 0;
            break;
        }

        if (rep == repeat - 1) {
            res_public = rel_residual(A_perturb, x_public, b);
            res_csc = rel_residual(A_perturb, x_csc, b);
        }

        sparse_ldlt_free(&ldlt_csc);
        ldlt_csc_free(pre_factor);
        sparse_analysis_free(&derived_analysis);
        sparse_free(A_perm);
        sparse_free(A_perturb);
    }

    if (!ok) {
        fprintf(stderr, "bench_refactor_csc: kkt-150 — aborted, partial timings discarded\n");
        sparse_factor_free(&factors_public);
        sparse_analysis_free(&an);
        free(ones);
        free(b);
        free(x_public);
        free(x_csc);
        sparse_free(A);
        return 1;
    }

    double refactor_public_ms = refactor_public_total * 1000.0 / (double)repeat;
    double refactor_csc_ms = refactor_csc_total * 1000.0 / (double)repeat;
    double solve_public_ms = solve_public_total * 1000.0 / (double)repeat;
    double solve_csc_ms = solve_csc_total * 1000.0 / (double)repeat;
    double speedup = refactor_public_ms / refactor_csc_ms;

    emit_csv_row("kkt-150", "ldlt_kkt", n, nnz, analyze_ms, refactor_public_ms, refactor_csc_ms,
                 solve_public_ms, solve_csc_ms, speedup, res_public, res_csc);

    sparse_factor_free(&factors_public);
    sparse_analysis_free(&an);
    free(ones);
    free(b);
    free(x_public);
    free(x_csc);
    sparse_free(A);
    return 0;
}

/* ─── Main ─────────────────────────────────────────────────────── */

static const char *default_matrices[] = {
    "tests/data/suitesparse/nos4.mtx",     "tests/data/suitesparse/bcsstk04.mtx",
    "tests/data/suitesparse/bcsstk14.mtx", "tests/data/suitesparse/s3rmt3m3.mtx",
    "tests/data/suitesparse/Kuu.mtx",      "tests/data/suitesparse/Pres_Poisson.mtx",
};
static const int default_matrix_count =
    (int)(sizeof(default_matrices) / sizeof(default_matrices[0]));

int main(int argc, char **argv) {
    int repeat = 10;
    int indefinite_kkt = 0;
    const char *single_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--repeat") && i + 1 < argc) {
            repeat = atoi(argv[++i]);
            if (repeat < 1)
                repeat = 1;
        } else if (!strcmp(argv[i], "--indefinite-kkt")) {
            indefinite_kkt = 1;
        } else if (argv[i][0] != '-') {
            single_path = argv[i];
        }
    }

    printf("benchmark,category,matrix,scenario,n,nnz,analyze_ms,"
           "refactor_public_ms,refactor_csc_ms,"
           "solve_public_ms,solve_csc_ms,"
           "speedup_refactor,res_public,res_csc\n");

    int rc = 0;
    if (indefinite_kkt) {
        rc |= bench_indefinite_kkt(repeat);
    } else if (single_path) {
        rc |= bench_spd_matrix(single_path, repeat);
    } else {
        for (int i = 0; i < default_matrix_count; i++)
            rc |= bench_spd_matrix(default_matrices[i], repeat);
    }
    return rc;
}
