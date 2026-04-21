/*
 * bench_refactor_csc.c — Analyze-once / factor-many Cholesky benchmark
 *
 * Sprint 17 and Sprint 18 `PERF_NOTES.md` both hypothesise that the
 * CSC Cholesky kernel's speedup over the linked-list baseline is
 * LARGER in the analyze-once / factor-many workflow — where one
 * `sparse_analyze` amortises across N refactorizations with the
 * same sparsity pattern but perturbed values — than in the one-shot
 * factor workflow where AMD reordering runs on every call and
 * dominates the small-matrix cost.
 *
 * `benchmarks/bench_chol_csc.c` measures the one-shot case; this
 * benchmark measures the analyze-once case.  Together they let the
 * Day 2 `PERF_NOTES.md` update record both numbers side-by-side and
 * either confirm or disconfirm the hypothesis.
 *
 * ─── Workflow ──────────────────────────────────────────────────────
 *
 * Per matrix (outside the timed region):
 *   1. Load A from .mtx.
 *   2. `sparse_analyze(A, AMD, &an)` — computes AMD perm + symbolic
 *      L pattern. Cost measured once and reported in `analyze_ms`.
 *   3. One priming `sparse_factor_numeric` on the original A so the
 *      subsequent `sparse_refactor_numeric` calls have a valid
 *      starting `sparse_factors_t`.  Not timed — setup only.
 *
 * Per iteration i = 0..N-1 (timed):
 *   4. `sparse_copy(A)` → `A_perturb` with per-entry multiplicative
 *      noise `v *= 1 + 1e-9 * symmetric_noise(row, col, seed)`.
 *      `symmetric_noise` is a deterministic hash keyed on
 *      `min(i, j) * n + max(i, j) + seed` so A[i,j] and A[j,i] get
 *      the same multiplier — required to keep the matrix symmetric
 *      (`sparse_cholesky_factor` runs `sparse_is_symmetric` first
 *      and returns SPARSE_ERR_NOT_SPD on asymmetry).  Preserves the
 *      sparsity pattern (no inserts / removes) and keeps the matrix
 *      well within SPD territory for the test corpus.
 *   5. `sparse_refactor_numeric(A_perturb, &an, &factors_ll)` — the
 *      LL side. Timed into `refactor_ll`.
 *   6. `chol_csc_from_sparse_with_analysis(A_perturb, &an, &L_csc)`
 *      + `chol_csc_eliminate_supernodal(L_csc, 4)` — the CSC side.
 *      Timed into `refactor_csc`.
 *   7. `sparse_factor_solve(&factors_ll, &an, b, x_ll)` — LL solve
 *      on every iteration (timed into `solve_ll`).
 *   8. `chol_csc_solve_perm(L_csc, an.perm, b, x_csc)` — CSC solve
 *      on every iteration (timed into `solve_csc`).
 *   9. On the last iteration, compute relative residuals `res_ll`,
 *      `res_csc` against the perturbed A to confirm both backends
 *      produced equivalent solutions to round-off.
 *
 * N is 10 by default; change with the existing `--repeat` flag.
 *
 * Output is CSV on stdout: one header row, one row per matrix with
 *   matrix, n, nnz, analyze_ms,
 *   refactor_ll_ms, refactor_csc_ms,
 *   solve_ll_ms, solve_csc_ms,
 *   speedup_refactor, res_ll, res_csc
 *
 * `speedup_refactor = refactor_ll_ms / refactor_csc_ms`; > 1.0 means
 * CSC is faster in the refactor-many regime.  `analyze_ms` is
 * reported so callers can see the amortisation opportunity — with
 * N = 10 refactors, the analyze cost amortises to `analyze_ms / 10`
 * per factor, whereas the one-shot `bench_chol_csc` measures that
 * cost in every factor timing.
 *
 * Usage:
 *   ./bench_refactor_csc                              # default matrix list
 *   ./bench_refactor_csc path/to/matrix.mtx           # single matrix
 *   ./bench_refactor_csc --repeat 5                   # 5 refactors instead of 10
 */
#define _POSIX_C_SOURCE 199309L

#include "sparse_analysis.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_cholesky.h"
#include "sparse_matrix.h"
#include "sparse_matrix_internal.h"
#include "sparse_reorder.h"
#include "sparse_vector.h"

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

/* Matrix runner: analyze once, refactor N times, emit one CSV row. */
static int bench_matrix(const char *path, int repeat) {
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
    double *x_ll = calloc((size_t)n, sizeof(double));
    double *x_csc = calloc((size_t)n, sizeof(double));
    if (!ones || !b || !x_ll || !x_csc) {
        fprintf(stderr, "bench_refactor_csc: malloc failed in bench_matrix (n=%d)\n", (int)n);
        free(ones);
        free(b);
        free(x_ll);
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
    sparse_analysis_opts_t aopts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_AMD};
    sparse_analysis_t an = {0};
    double t0 = wall_time();
    sparse_err_t err = sparse_analyze(A, &aopts, &an);
    double analyze_ms = (wall_time() - t0) * 1000.0;
    if (err != SPARSE_OK) {
        fprintf(stderr, "bench_refactor_csc: sparse_analyze failed on %s (err=%d)\n", path,
                (int)err);
        free(ones);
        free(b);
        free(x_ll);
        free(x_csc);
        sparse_free(A);
        return 1;
    }

    /* Prime the LL factors via a non-timed factor_numeric — subsequent
     * refactor calls can then refactor in place.  Uses the original A
     * (unperturbed) for priming. */
    sparse_factors_t factors_ll = {0};
    if (sparse_factor_numeric(A, &an, &factors_ll) != SPARSE_OK) {
        fprintf(stderr, "bench_refactor_csc: priming sparse_factor_numeric failed on %s\n", path);
        sparse_analysis_free(&an);
        free(ones);
        free(b);
        free(x_ll);
        free(x_csc);
        sparse_free(A);
        return 1;
    }

    /* Timed region: N refactors on both backends. */
    double refactor_ll_total = 0.0, refactor_csc_total = 0.0;
    double solve_ll_total = 0.0, solve_csc_total = 0.0;
    double res_ll = 0.0, res_csc = 0.0;
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

        /* Linked-list refactor. */
        double t_ll0 = wall_time();
        sparse_err_t e_ll = sparse_refactor_numeric(A_perturb, &an, &factors_ll);
        refactor_ll_total += wall_time() - t_ll0;
        if (e_ll != SPARSE_OK) {
            fprintf(stderr,
                    "bench_refactor_csc: sparse_refactor_numeric failed on %s (rep=%d, err=%d)\n",
                    path, rep, (int)e_ll);
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
        double t_sll0 = wall_time();
        sparse_err_t e_sll = sparse_factor_solve(&factors_ll, &an, b, x_ll);
        solve_ll_total += wall_time() - t_sll0;

        double t_scsc0 = wall_time();
        sparse_err_t e_scsc = chol_csc_solve_perm(L_csc, an.perm, b, x_csc);
        solve_csc_total += wall_time() - t_scsc0;

        if (e_sll != SPARSE_OK || e_scsc != SPARSE_OK) {
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
            res_ll = rel_residual(A_perturb, x_ll, b);
            res_csc = rel_residual(A_perturb, x_csc, b);
        }

        chol_csc_free(L_csc);
        sparse_free(A_perturb);
    }

    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;

    if (!ok) {
        fprintf(stderr, "bench_refactor_csc: %s — aborted, partial timings discarded\n", base);
        sparse_factor_free(&factors_ll);
        sparse_analysis_free(&an);
        free(ones);
        free(b);
        free(x_ll);
        free(x_csc);
        sparse_free(A);
        return 1;
    }

    double refactor_ll_ms = refactor_ll_total * 1000.0 / (double)repeat;
    double refactor_csc_ms = refactor_csc_total * 1000.0 / (double)repeat;
    double solve_ll_ms = solve_ll_total * 1000.0 / (double)repeat;
    double solve_csc_ms = solve_csc_total * 1000.0 / (double)repeat;
    double speedup = refactor_ll_ms / refactor_csc_ms;

    /* CSV row:
     * matrix, n, nnz, analyze_ms, refactor_ll_ms, refactor_csc_ms,
     * solve_ll_ms, solve_csc_ms, speedup_refactor, res_ll, res_csc */
    printf("%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2e,%.2e\n", base, (int)n, (int)nnz, analyze_ms,
           refactor_ll_ms, refactor_csc_ms, solve_ll_ms, solve_csc_ms, speedup, res_ll, res_csc);

    sparse_factor_free(&factors_ll);
    sparse_analysis_free(&an);
    free(ones);
    free(b);
    free(x_ll);
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

    printf("matrix,n,nnz,analyze_ms,"
           "refactor_ll_ms,refactor_csc_ms,"
           "solve_ll_ms,solve_csc_ms,"
           "speedup_refactor,res_ll,res_csc\n");

    int rc = 0;
    if (single_path) {
        rc |= bench_matrix(single_path, repeat);
    } else {
        for (int i = 0; i < default_matrix_count; i++)
            rc |= bench_matrix(default_matrices[i], repeat);
    }
    return rc;
}
