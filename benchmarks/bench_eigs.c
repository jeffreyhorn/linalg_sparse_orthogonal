/*
 * bench_eigs.c — Sprint 21 Day 11 permanent eigensolver benchmark driver.
 *
 * Replaces the Sprint 20 Day 13 throwaway `/tmp/bench_eigs.c` driver
 * with a permanent executable that exercises all three backends
 * (`SPARSE_EIGS_BACKEND_LANCZOS` grow-m, `_LANCZOS_THICK_RESTART`,
 * `_LOBPCG`) under their three `which` modes (LARGEST / SMALLEST /
 * NEAREST_SIGMA) on the standard SuiteSparse + KKT corpus, with
 * preconditioner sweeps for LOBPCG (NONE / IC0 / LDLT).
 *
 * Modes:
 *   --sweep default       Full sweep across the default corpus —
 *                         (nos4, bcsstk04, bcsstk14) × (k: 3, 5) ×
 *                         (which: LARGEST, SMALLEST) × (3 backends)
 *                         + kkt-150 × NEAREST_SIGMA × 3 backends.
 *                         33 rows with the current corpus / loop set
 *                         (bcsstk14 is LARGEST-only; KKT contributes
 *                         3 rows).  See `bench_day14.txt` for the
 *                         committed capture.
 *   --compare             Pivoted three-backend head-to-head on a
 *                         smaller corpus; rows are (matrix, k, which),
 *                         columns are (backend × {iters, wall_ms,
 *                         residual}).
 *   --matrix <path>       Single-matrix mode: run the configured
 *                         (--k, --which, --backend, --precond)
 *                         combination on the supplied .mtx fixture.
 *
 * Output:
 *   --csv  emits one CSV row per (matrix, k, which, backend, precond)
 *          combination with the schema:
 *            matrix,n,k,which,sigma,backend,precond,
 *            iterations,peak_basis,wall_ms,residual,status
 *   no flag emits a human-readable table to stdout.
 *
 * Without `--csv`, the human-readable mode is the default for
 * `make bench-eigs`'s smoke invocation — keeps the make output
 * scannable while still flagging regressions in iteration count or
 * residual.
 *
 * The wall-clock measurement reports the median of `--repeats`
 * runs (default 3 for the smoke target, 5 for the recorded numbers
 * landed at `docs/planning/EPIC_2/SPRINT_21/bench_day14.txt`).
 *
 * Implementation note on preconditioners: IC(0) plugs in via
 * `sparse_ic_precond` directly, while LDL^T needs a small adapter
 * because `sparse_ldlt_solve`'s signature differs from
 * `sparse_precond_fn`.  The adapter is a one-liner kept local to
 * this file (the same pattern test_eigs_lobpcg.c uses).  When the
 * `--which NEAREST` shift-invert path is active, `sparse_eigs_sym`
 * already factors `(A - σI)` via LDL^T internally and hands the
 * inverse to the operator callback — applying a separate `--precond
 * LDLT` on top is well-defined but doesn't compose physically (the
 * preconditioner is M^{-1} ≈ A^{-1} but the operator is `(A − σI)^{-1}`),
 * so the bench skips precond combinations on NEAREST_SIGMA.
 */

#define _POSIX_C_SOURCE 199309L

#include "sparse_eigs.h"
#include "sparse_ic.h"
#include "sparse_ilu.h"
#include "sparse_ldlt.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ─── Wall clock helpers. ──────────────────────────────────────────── */

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y)
        return -1;
    if (x > y)
        return 1;
    return 0;
}

static double median_double(double *arr, int n) {
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    return arr[n / 2];
}

/* ─── LDL^T preconditioner adapter. ────────────────────────────────── */

static sparse_err_t bench_ldlt_precond(const void *ctx, idx_t n, const double *r, double *z) {
    (void)n;
    return sparse_ldlt_solve((const sparse_ldlt_t *)ctx, r, z);
}

/* ─── Run-config + result types. ───────────────────────────────────── */

typedef enum {
    BENCH_PRECOND_NONE = 0,
    BENCH_PRECOND_IC0 = 1,
    BENCH_PRECOND_LDLT = 2,
} bench_precond_kind_t;

typedef struct {
    sparse_eigs_backend_t backend; /* AUTO / LANCZOS / THICK_RESTART / LOBPCG */
    bench_precond_kind_t precond_kind;
    sparse_eigs_which_t which;
    double sigma;
    idx_t k;
    idx_t block_size; /* 0 = library default (= k) */
    int compute_vectors;
    double tol;
    idx_t max_iters;
    int repeats;
} run_config_t;

typedef struct {
    int ok;
    double wall_ms_median;
    idx_t iterations;
    idx_t peak_basis;
    double residual;
    sparse_eigs_backend_t backend_used;
    bench_precond_kind_t precond_used; /* effective precond after the
                                          NEAREST_SIGMA / non-LOBPCG
                                          gating in run_one() */
    sparse_err_t last_err;
} run_result_t;

/* ─── String labels for CSV / human output. ────────────────────────── */

static const char *backend_label(sparse_eigs_backend_t b) {
    switch (b) {
    case SPARSE_EIGS_BACKEND_AUTO:
        return "AUTO";
    case SPARSE_EIGS_BACKEND_LANCZOS:
        return "GROWING_M";
    case SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART:
        return "THICK_RESTART";
    case SPARSE_EIGS_BACKEND_LOBPCG:
        return "LOBPCG";
    default:
        return "UNKNOWN";
    }
}

static const char *which_label(sparse_eigs_which_t w) {
    switch (w) {
    case SPARSE_EIGS_LARGEST:
        return "LARGEST";
    case SPARSE_EIGS_SMALLEST:
        return "SMALLEST";
    case SPARSE_EIGS_NEAREST_SIGMA:
        return "NEAREST";
    default:
        return "UNKNOWN";
    }
}

static const char *precond_label(bench_precond_kind_t p) {
    switch (p) {
    case BENCH_PRECOND_NONE:
        return "NONE";
    case BENCH_PRECOND_IC0:
        return "IC0";
    case BENCH_PRECOND_LDLT:
        return "LDLT";
    default:
        return "UNKNOWN";
    }
}

static const char *err_label(sparse_err_t e) {
    if (e == SPARSE_OK)
        return "OK";
    if (e == SPARSE_ERR_NOT_CONVERGED)
        return "NOT_CONVERGED";
    if (e == SPARSE_ERR_BADARG)
        return "BADARG";
    if (e == SPARSE_ERR_ALLOC)
        return "ALLOC";
    if (e == SPARSE_ERR_NULL)
        return "NULL";
    if (e == SPARSE_ERR_SHAPE)
        return "SHAPE";
    if (e == SPARSE_ERR_SINGULAR)
        return "SINGULAR";
    if (e == SPARSE_ERR_NOT_SPD)
        return "NOT_SPD";
    return "OTHER";
}

/* ─── Single-config runner.
 *
 * Builds the preconditioner once (factor cost is amortised across the
 * `repeats` iterations — matches the bench_ldlt_csc.c convention).
 * Times only `sparse_eigs_sym`; reports the median wall-clock time.
 * The eigenvalue / eigenvector buffers are reused across repeats. */

static run_result_t run_one(const SparseMatrix *A, const run_config_t *cfg) {
    run_result_t r = {0};
    idx_t n = sparse_rows(A);
    idx_t k = cfg->k;

    /* Decide whether the requested preconditioner is actually
     * applicable to the (backend, which) combination.  Skipping
     * the factor cost for backends that ignore `opts.precond`
     * avoids both (a) wasted work and (b) spurious failures on
     * indefinite matrices when the user asked for e.g.
     * `--backend THICK_RESTART --precond IC0` (IC(0) requires
     * SPD; THICK_RESTART ignores precond entirely).
     *
     * Gating rules:
     *   - NEAREST_SIGMA already factors `(A − σI)` internally and
     *     hands the inverse to the operator callback; an outer
     *     precond doesn't compose physically (file header explains).
     *   - LANCZOS / LANCZOS_THICK_RESTART ignore opts.precond.
     *   - LOBPCG always honours it.
     *   - AUTO honours it only if the AUTO predicate would route
     *     to LOBPCG; otherwise AUTO routes to a Lanczos backend
     *     and the precond is ignored.  The AUTO → LOBPCG predicate
     *     mirrors `sparse_eigs_sym`: precond != NULL AND
     *     n >= SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD AND
     *     effective block_size >= 4 (defaults to k when 0). */
    bench_precond_kind_t effective_precond = cfg->precond_kind;
    if (cfg->which == SPARSE_EIGS_NEAREST_SIGMA || cfg->backend == SPARSE_EIGS_BACKEND_LANCZOS ||
        cfg->backend == SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART) {
        effective_precond = BENCH_PRECOND_NONE;
    } else if (cfg->backend == SPARSE_EIGS_BACKEND_AUTO) {
        idx_t bs_for_auto = (cfg->block_size > 0) ? cfg->block_size : k;
        int auto_lobpcg = (effective_precond != BENCH_PRECOND_NONE) &&
                          (n >= (idx_t)SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD) && (bs_for_auto >= 4);
        if (!auto_lobpcg)
            effective_precond = BENCH_PRECOND_NONE;
    }
    r.precond_used = effective_precond;

    /* Build the precond once. */
    sparse_ilu_t ic = {0};
    sparse_ldlt_t ldlt = {0};
    sparse_precond_fn precond = NULL;
    const void *precond_ctx = NULL;
    if (effective_precond == BENCH_PRECOND_IC0) {
        sparse_err_t e = sparse_ic_factor(A, &ic);
        if (e != SPARSE_OK) {
            r.ok = 0;
            r.last_err = e;
            return r;
        }
        precond = sparse_ic_precond;
        precond_ctx = &ic;
    } else if (effective_precond == BENCH_PRECOND_LDLT) {
        sparse_err_t e = sparse_ldlt_factor(A, &ldlt);
        if (e != SPARSE_OK) {
            r.ok = 0;
            r.last_err = e;
            return r;
        }
        precond = bench_ldlt_precond;
        precond_ctx = &ldlt;
    }

    int repeats = cfg->repeats > 0 ? cfg->repeats : 1;
    double *vals = malloc((size_t)k * sizeof(double));
    double *vecs = cfg->compute_vectors ? malloc((size_t)n * (size_t)k * sizeof(double)) : NULL;
    double *times = malloc((size_t)repeats * sizeof(double));
    if (!vals || !times || (cfg->compute_vectors && !vecs)) {
        free(vals);
        free(vecs);
        free(times);
        sparse_ic_free(&ic);
        sparse_ldlt_free(&ldlt);
        r.ok = 0;
        r.last_err = SPARSE_ERR_ALLOC;
        return r;
    }

    sparse_eigs_t res = {0};
    sparse_err_t last_err = SPARSE_OK;
    /* Reps that actually completed (OK or NOT_CONVERGED).  Hard
     * errors abort the loop; reporting the median of fast-failing
     * runs would be misleading. */
    int reps_done = 0;
    for (int rep = 0; rep < repeats; rep++) {
        memset(vals, 0, (size_t)k * sizeof(double));
        if (vecs)
            memset(vecs, 0, (size_t)n * (size_t)k * sizeof(double));
        res = (sparse_eigs_t){.eigenvalues = vals, .eigenvectors = vecs};

        sparse_eigs_opts_t opts = {
            .which = cfg->which,
            .sigma = cfg->sigma,
            .max_iterations = cfg->max_iters,
            .tol = cfg->tol > 0.0 ? cfg->tol : 1e-8,
            .reorthogonalize = 1,
            .compute_vectors = cfg->compute_vectors,
            .backend = cfg->backend,
            .block_size = cfg->block_size,
            .precond = precond,
            .precond_ctx = precond_ctx,
            .lobpcg_soft_lock = 1,
        };
        double t0 = wall_time();
        last_err = sparse_eigs_sym(A, k, &opts, &res);
        times[reps_done] = (wall_time() - t0) * 1000.0;
        if (last_err != SPARSE_OK && last_err != SPARSE_ERR_NOT_CONVERGED) {
            /* Hard error — record nothing further; the timing of a
             * fast-fail run isn't comparable to a real solve. */
            break;
        }
        reps_done++;
    }

    /* Median over completed reps only.  When every rep hard-failed
     * (reps_done == 0), report 0.0 ms so the row stays parseable;
     * the status column carries the diagnostic. */
    r.wall_ms_median = (reps_done > 0) ? median_double(times, reps_done) : 0.0;
    r.iterations = res.iterations;
    r.peak_basis = res.peak_basis_size;
    r.residual = res.residual_norm;
    r.backend_used = res.backend_used;
    r.last_err = last_err;
    /* OK or NOT_CONVERGED both count as "ran to completion" — the
     * status column reports the distinction so consumers can filter. */
    r.ok = (last_err == SPARSE_OK || last_err == SPARSE_ERR_NOT_CONVERGED);

    free(vals);
    free(vecs);
    free(times);
    sparse_ic_free(&ic);
    sparse_ldlt_free(&ldlt);
    return r;
}

/* ─── CSV / human-readable output. ─────────────────────────────────── */

static void emit_header(int csv) {
    if (csv) {
        printf("matrix,n,k,which,sigma,backend,precond,"
               "iterations,peak_basis,wall_ms,residual,status\n");
    } else {
        printf("%-12s %5s %3s %-8s %6s %-13s %5s %7s %8s %9s %10s %s\n", "matrix", "n", "k",
               "which", "sigma", "backend", "prec", "iters", "peak_b", "wall_ms", "residual",
               "status");
    }
}

static void emit_row(int csv, const char *matrix_label, idx_t n, const run_config_t *cfg,
                     const run_result_t *r) {
    /* When the caller requested AUTO, report the backend the library
     * actually picked (via result.backend_used) rather than "AUTO" —
     * makes the CSV diffable across the dispatch decision tree. */
    sparse_eigs_backend_t reported_backend =
        (cfg->backend == SPARSE_EIGS_BACKEND_AUTO) ? r->backend_used : cfg->backend;
    /* Report the precond actually applied (after the gating in
     * run_one), not what the user asked for — keeps the CSV row
     * consistent with the wall-time / residual columns. */
    if (csv) {
        printf("%s,%d,%d,%s,%.4g,%s,%s,%d,%d,%.3f,%.3e,%s\n", matrix_label, (int)n, (int)cfg->k,
               which_label(cfg->which), cfg->sigma, backend_label(reported_backend),
               precond_label(r->precond_used), (int)r->iterations, (int)r->peak_basis,
               r->wall_ms_median, r->residual, err_label(r->last_err));
    } else {
        printf("%-12s %5d %3d %-8s %6.2f %-13s %5s %7d %8d %9.3f %10.2e %s\n", matrix_label, (int)n,
               (int)cfg->k, which_label(cfg->which), cfg->sigma, backend_label(reported_backend),
               precond_label(r->precond_used), (int)r->iterations, (int)r->peak_basis,
               r->wall_ms_median, r->residual, err_label(r->last_err));
    }
}

/* ─── KKT builder (mirrors bench_ldlt_csc.c's bench_build_kkt_150). ── */

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

/* ─── Default sweep — Day 11 PLAN target.
 *
 * Three SuiteSparse fixtures × LARGEST/SMALLEST × {k=3, k=5} × three
 * backends, plus the KKT-150 NEAREST_SIGMA at σ=0 sweep.  All
 * unpreconditioned — preconditioner sweeps live in `--compare` mode
 * (where a single fixture is profiled across (vanilla / IC0 / LDLT))
 * to keep runtime tractable.
 *
 * Runtime constraint: keep `--sweep default --repeats 3` under
 * ~3 minutes on the developer machine so `make bench-eigs` stays
 * cheap to invoke.  The configurations excluded below saturate
 * the iteration cap without converging on the un-preconditioned
 * path — useful as data points but not as smoke-test targets:
 *   - bcsstk14 SMALLEST: bottom of spectrum is so clustered that
 *     vanilla iteration takes ~minutes per backend; preconditioned
 *     runs land in `--compare` where the precond sweep is the
 *     point of comparison.
 * Configs that exhaust the cap on the included entries report
 * NOT_CONVERGED in the status column — the bench is for measurement,
 * not regression-fail. */

typedef struct {
    const char *path;
    const char *label;
    int include_smallest; /* skip SMALLEST when 0 */
} corpus_entry_t;

static const corpus_entry_t bench_corpus[] = {
    {"tests/data/suitesparse/nos4.mtx", "nos4", 1},
    {"tests/data/suitesparse/bcsstk04.mtx", "bcsstk04", 1},
    {"tests/data/suitesparse/bcsstk14.mtx", "bcsstk14", 0},
};
static const int bench_corpus_count = (int)(sizeof(bench_corpus) / sizeof(bench_corpus[0]));

static int run_default_sweep(int repeats, int csv) {
    emit_header(csv);

    sparse_eigs_backend_t backends[] = {
        SPARSE_EIGS_BACKEND_LANCZOS,
        SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART,
        SPARSE_EIGS_BACKEND_LOBPCG,
    };
    int n_backends = (int)(sizeof(backends) / sizeof(backends[0]));

    sparse_eigs_which_t whichs[] = {SPARSE_EIGS_LARGEST, SPARSE_EIGS_SMALLEST};
    int n_whichs = (int)(sizeof(whichs) / sizeof(whichs[0]));

    idx_t ks[] = {3, 5};
    int n_ks = (int)(sizeof(ks) / sizeof(ks[0]));

    int rc = 0;

    for (int ci = 0; ci < bench_corpus_count; ci++) {
        SparseMatrix *A = NULL;
        if (sparse_load_mm(&A, bench_corpus[ci].path) != SPARSE_OK) {
            fprintf(stderr, "bench_eigs: skipping %s (load failed)\n", bench_corpus[ci].path);
            rc = 1;
            continue;
        }
        idx_t n = sparse_rows(A);
        for (int wi = 0; wi < n_whichs; wi++) {
            if (whichs[wi] == SPARSE_EIGS_SMALLEST && !bench_corpus[ci].include_smallest)
                continue;
            for (int ki = 0; ki < n_ks; ki++) {
                for (int bi = 0; bi < n_backends; bi++) {
                    idx_t max_iters = (whichs[wi] == SPARSE_EIGS_SMALLEST) ? 800 : 300;
                    run_config_t cfg = {
                        .backend = backends[bi],
                        .precond_kind = BENCH_PRECOND_NONE,
                        .which = whichs[wi],
                        .sigma = 0.0,
                        .k = ks[ki],
                        .block_size = 0,
                        .compute_vectors = 0,
                        .tol = 1e-8,
                        .max_iters = max_iters,
                        .repeats = repeats,
                    };
                    run_result_t r = run_one(A, &cfg);
                    emit_row(csv, bench_corpus[ci].label, n, &cfg, &r);
                }
            }
        }
        sparse_free(A);
    }

    /* KKT-150 × NEAREST_SIGMA × 3 backends. */
    SparseMatrix *kkt = bench_build_kkt_150();
    if (kkt) {
        idx_t n = sparse_rows(kkt);
        for (int bi = 0; bi < n_backends; bi++) {
            run_config_t cfg = {
                .backend = backends[bi],
                .precond_kind = BENCH_PRECOND_NONE,
                .which = SPARSE_EIGS_NEAREST_SIGMA,
                .sigma = 0.0,
                .k = 3,
                .block_size = 0,
                .compute_vectors = 0,
                .tol = 1e-8,
                .max_iters = 300,
                .repeats = repeats,
            };
            run_result_t r = run_one(kkt, &cfg);
            emit_row(csv, "kkt-150", n, &cfg, &r);
        }
        sparse_free(kkt);
    }

    return rc;
}

/* ─── --compare mode: pivoted three-backend head-to-head.
 *
 * Format: one row per (matrix, k, which, lobpcg_precond), columns =
 * three backends × {iters, wall_ms, residual}.  Makes "LOBPCG wins
 * here, thick-restart wins there" obvious at a glance.  Includes the
 * preconditioner sweep (NONE / IC0 / LDLT) so the precond comparison
 * is visible in the same table.
 *
 * The precond column is named `lobpcg_precond` because Lanczos
 * grow-m and thick-restart ignore `opts.precond` entirely (per
 * `run_one()`'s gating, the effective precond on those backends is
 * always NONE).  Identical grow-m / thick numbers therefore repeat
 * across the IC0 / LDLT rows; they're presented for at-a-glance
 * comparison against the LOBPCG arm rather than as separate runs. */

static void emit_compare_header(int csv) {
    if (csv) {
        printf("matrix,n,k,which,sigma,lobpcg_precond,"
               "growing_m_iters,growing_m_wall_ms,growing_m_residual,growing_m_status,"
               "thick_iters,thick_wall_ms,thick_residual,thick_status,"
               "lobpcg_iters,lobpcg_wall_ms,lobpcg_residual,lobpcg_status\n");
    } else {
        printf("%-12s %5s %3s %-8s %6s %7s | %23s | %23s | %23s\n", "matrix", "n", "k", "which",
               "sigma", "lb_prec", "growing_m (iters/ms/res)", "thick (iters/ms/res)",
               "lobpcg (iters/ms/res)");
    }
}

static void emit_compare_row(int csv, const char *matrix_label, idx_t n, const run_config_t *cfg,
                             const run_result_t *rgm, const run_result_t *rtr,
                             const run_result_t *rlb) {
    /* Report the precond actually used by the LOBPCG arm; grow-m and
     * thick-restart always run with NONE (see run_one's gating).
     * `run_one` writes precond_used unconditionally (early in the
     * function, before any error path), so it's always populated. */
    bench_precond_kind_t lobpcg_prec = rlb->precond_used;
    if (csv) {
        printf("%s,%d,%d,%s,%.4g,%s,"
               "%d,%.3f,%.3e,%s,"
               "%d,%.3f,%.3e,%s,"
               "%d,%.3f,%.3e,%s\n",
               matrix_label, (int)n, (int)cfg->k, which_label(cfg->which), cfg->sigma,
               precond_label(lobpcg_prec), (int)rgm->iterations, rgm->wall_ms_median, rgm->residual,
               err_label(rgm->last_err), (int)rtr->iterations, rtr->wall_ms_median, rtr->residual,
               err_label(rtr->last_err), (int)rlb->iterations, rlb->wall_ms_median, rlb->residual,
               err_label(rlb->last_err));
    } else {
        printf("%-12s %5d %3d %-8s %6.2f %7s | %5d/%6.1f/%.1e %s | %5d/%6.1f/%.1e %s | "
               "%5d/%6.1f/%.1e %s\n",
               matrix_label, (int)n, (int)cfg->k, which_label(cfg->which), cfg->sigma,
               precond_label(lobpcg_prec), (int)rgm->iterations, rgm->wall_ms_median, rgm->residual,
               err_label(rgm->last_err), (int)rtr->iterations, rtr->wall_ms_median, rtr->residual,
               err_label(rtr->last_err), (int)rlb->iterations, rlb->wall_ms_median, rlb->residual,
               err_label(rlb->last_err));
    }
}

static int run_compare_mode(int repeats, int csv) {
    emit_compare_header(csv);

    /* Compare corpus: a focused subset where the three backends
     * differ visibly.  bcsstk04 SMALLEST k=3 stresses the spectral
     * cluster; bcsstk14 LARGEST k=5 stresses the basis size. */
    typedef struct {
        const char *path;
        const char *label;
        sparse_eigs_which_t which;
        idx_t k;
    } compare_entry_t;

    const compare_entry_t entries[] = {
        {"tests/data/suitesparse/nos4.mtx", "nos4", SPARSE_EIGS_LARGEST, 5},
        {"tests/data/suitesparse/bcsstk04.mtx", "bcsstk04", SPARSE_EIGS_SMALLEST, 3},
        {"tests/data/suitesparse/bcsstk14.mtx", "bcsstk14", SPARSE_EIGS_LARGEST, 5},
    };
    const int n_entries = (int)(sizeof(entries) / sizeof(entries[0]));

    /* For each compare entry, run all three backends with NONE
     * precond, then run LOBPCG with IC0 + LDLT.  The first three rows
     * of each entry compare backends; the last two rows of each
     * entry probe LOBPCG's preconditioner spectrum. */
    bench_precond_kind_t preconds[] = {
        BENCH_PRECOND_NONE,
        BENCH_PRECOND_IC0,
        BENCH_PRECOND_LDLT,
    };
    int n_preconds = (int)(sizeof(preconds) / sizeof(preconds[0]));

    int rc = 0;
    for (int ei = 0; ei < n_entries; ei++) {
        SparseMatrix *A = NULL;
        if (sparse_load_mm(&A, entries[ei].path) != SPARSE_OK) {
            fprintf(stderr, "bench_eigs --compare: skipping %s (load failed)\n", entries[ei].path);
            rc = 1;
            continue;
        }
        idx_t n = sparse_rows(A);
        run_config_t base = {
            .backend = SPARSE_EIGS_BACKEND_LANCZOS,
            .precond_kind = BENCH_PRECOND_NONE,
            .which = entries[ei].which,
            .sigma = 0.0,
            .k = entries[ei].k,
            .block_size = 0,
            .compute_vectors = 0,
            .tol = 1e-8,
            .max_iters = (entries[ei].which == SPARSE_EIGS_SMALLEST) ? (idx_t)800 : (idx_t)300,
            .repeats = repeats,
        };
        /* The grow-m and thick-restart Lanczos backends ignore
         * `opts.precond` (`run_one`'s gating downgrades to NONE),
         * so their numbers are identical across the IC0 / LDLT
         * precond rows.  Run them once per (matrix, k, which) and
         * reuse the results for the three precond rows; only the
         * LOBPCG arm actually varies per precond. */
        run_config_t cfg_gm = base;
        cfg_gm.backend = SPARSE_EIGS_BACKEND_LANCZOS;
        run_config_t cfg_tr = base;
        cfg_tr.backend = SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART;
        run_result_t rgm = run_one(A, &cfg_gm);
        run_result_t rtr = run_one(A, &cfg_tr);
        for (int pi = 0; pi < n_preconds; pi++) {
            run_config_t cfg_lb = base;
            cfg_lb.backend = SPARSE_EIGS_BACKEND_LOBPCG;
            cfg_lb.precond_kind = preconds[pi];
            run_result_t rlb = run_one(A, &cfg_lb);
            emit_compare_row(csv, entries[ei].label, n, &cfg_lb, &rgm, &rtr, &rlb);
        }
        sparse_free(A);
    }
    return rc;
}

/* ─── Single-matrix path. ──────────────────────────────────────────── */

static int parse_which(const char *s, sparse_eigs_which_t *out) {
    if (!strcmp(s, "LARGEST"))
        *out = SPARSE_EIGS_LARGEST;
    else if (!strcmp(s, "SMALLEST"))
        *out = SPARSE_EIGS_SMALLEST;
    else if (!strcmp(s, "NEAREST"))
        *out = SPARSE_EIGS_NEAREST_SIGMA;
    else
        return 0;
    return 1;
}

static int parse_backend(const char *s, sparse_eigs_backend_t *out) {
    if (!strcmp(s, "AUTO"))
        *out = SPARSE_EIGS_BACKEND_AUTO;
    else if (!strcmp(s, "GROWING_M") || !strcmp(s, "LANCZOS"))
        *out = SPARSE_EIGS_BACKEND_LANCZOS;
    else if (!strcmp(s, "THICK_RESTART"))
        *out = SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART;
    else if (!strcmp(s, "LOBPCG"))
        *out = SPARSE_EIGS_BACKEND_LOBPCG;
    else
        return 0;
    return 1;
}

static int parse_precond(const char *s, bench_precond_kind_t *out) {
    if (!strcmp(s, "NONE"))
        *out = BENCH_PRECOND_NONE;
    else if (!strcmp(s, "IC0"))
        *out = BENCH_PRECOND_IC0;
    else if (!strcmp(s, "LDLT"))
        *out = BENCH_PRECOND_LDLT;
    else
        return 0;
    return 1;
}

/* Parse a signed-integer CLI argument into an `idx_t` with bounds.
 * Rejects non-numeric input, trailing junk, overflow, and values
 * outside [min, INT32_MAX] (idx_t is int32_t).  Logs a diagnostic
 * to stderr on failure.  Returns 1 on success, 0 on failure. */
static int parse_idx_arg(const char *flag, const char *s, long min, idx_t *out) {
    if (!s || !*s) {
        fprintf(stderr, "bench_eigs: %s requires a value\n", flag);
        return 0;
    }
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno == ERANGE || end == s || *end != '\0') {
        fprintf(stderr, "bench_eigs: %s: not a valid integer: '%s'\n", flag, s);
        return 0;
    }
    if (v < min || v > (long)INT32_MAX) {
        fprintf(stderr, "bench_eigs: %s: out of range [%ld, %ld]: %ld\n", flag, min,
                (long)INT32_MAX, v);
        return 0;
    }
    *out = (idx_t)v;
    return 1;
}

/* Parse a floating-point CLI argument with bounds.
 * Rejects non-numeric input, trailing junk, overflow, NaN.
 * Returns 1 on success, 0 on failure. */
static int parse_double_arg(const char *flag, const char *s, double min, double *out) {
    if (!s || !*s) {
        fprintf(stderr, "bench_eigs: %s requires a value\n", flag);
        return 0;
    }
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno == ERANGE || end == s || *end != '\0' || v != v /* NaN */) {
        fprintf(stderr, "bench_eigs: %s: not a valid number: '%s'\n", flag, s);
        return 0;
    }
    if (v < min) {
        fprintf(stderr, "bench_eigs: %s: must be >= %g: %g\n", flag, min, v);
        return 0;
    }
    *out = v;
    return 1;
}

static int run_single(const char *path, run_config_t cfg, int csv) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, path) != SPARSE_OK) {
        fprintf(stderr, "bench_eigs: failed to load %s\n", path);
        return 1;
    }
    idx_t n = sparse_rows(A);
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;

    emit_header(csv);
    run_result_t r = run_one(A, &cfg);
    emit_row(csv, base, n, &cfg, &r);

    sparse_free(A);
    return 0;
}

/* ─── Help. ────────────────────────────────────────────────────────── */

static void print_help(void) {
    fprintf(stderr, "Usage: bench_eigs [options]\n"
                    "\n"
                    "Modes (mutually exclusive):\n"
                    "  --sweep default      Run the default Day 11 sweep across the\n"
                    "                       SuiteSparse + KKT corpus.  Default mode\n"
                    "                       when no other mode flag is given.\n"
                    "  --compare            Pivoted three-backend × precond comparison\n"
                    "                       on a focused subset of the corpus.\n"
                    "  --matrix <path>      Run on a single .mtx fixture; honors the\n"
                    "                       --k / --which / --backend / --precond /\n"
                    "                       --sigma flags below.\n"
                    "\n"
                    "Single-matrix options (only used with --matrix):\n"
                    "  --k <N>              Number of eigenpairs (default 3).\n"
                    "  --which <MODE>       LARGEST | SMALLEST | NEAREST  (default LARGEST).\n"
                    "  --sigma <FLOAT>      Shift for NEAREST mode (default 0.0).\n"
                    "  --backend <BACKEND>  AUTO | GROWING_M | LANCZOS | THICK_RESTART | LOBPCG\n"
                    "                       (default AUTO).\n"
                    "  --precond <KIND>     NONE | IC0 | LDLT (default NONE).\n"
                    "  --block-size <N>     LOBPCG block size; 0 = library default = k.\n"
                    "  --tol <FLOAT>        Convergence tolerance (default 1e-8).\n"
                    "  --max-iters <N>      Iteration cap (default 300).\n"
                    "\n"
                    "Common options:\n"
                    "  --repeats <N>        Median over N runs (default 3 — bumped to\n"
                    "                       5 when capturing for `bench_day14.txt`).\n"
                    "  --csv                Emit CSV format instead of human-readable.\n"
                    "  --help               This message.\n"
                    "\n"
                    "CSV schema (--sweep / --matrix):\n"
                    "  matrix,n,k,which,sigma,backend,precond,iterations,peak_basis,\n"
                    "  wall_ms,residual,status\n");
}

/* ─── Main. ────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int csv = 0;
    int repeats = 3;
    int sweep = 0;
    int compare = 0;
    const char *single_matrix = NULL;
    run_config_t cfg = {
        .backend = SPARSE_EIGS_BACKEND_AUTO,
        .precond_kind = BENCH_PRECOND_NONE,
        .which = SPARSE_EIGS_LARGEST,
        .sigma = 0.0,
        .k = 3,
        .block_size = 0,
        .compute_vectors = 0,
        .tol = 1e-8,
        .max_iters = 300,
        .repeats = 3,
    };

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv")) {
            csv = 1;
        } else if (!strcmp(argv[i], "--compare")) {
            compare = 1;
        } else if (!strcmp(argv[i], "--sweep") && i + 1 < argc) {
            const char *preset = argv[++i];
            if (!strcmp(preset, "default")) {
                sweep = 1;
            } else {
                fprintf(stderr, "bench_eigs: unknown sweep preset '%s'\n", preset);
                return 2;
            }
        } else if (!strcmp(argv[i], "--matrix") && i + 1 < argc) {
            single_matrix = argv[++i];
        } else if (!strcmp(argv[i], "--k") && i + 1 < argc) {
            if (!parse_idx_arg("--k", argv[++i], 1, &cfg.k))
                return 2;
        } else if (!strcmp(argv[i], "--which") && i + 1 < argc) {
            if (!parse_which(argv[++i], &cfg.which)) {
                fprintf(stderr, "bench_eigs: unknown --which '%s'\n", argv[i]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--sigma") && i + 1 < argc) {
            /* sigma may legitimately be any finite double (including
             * negative shifts). -DBL_MAX as the floor accepts the full
             * range while still rejecting NaN / parse errors. */
            if (!parse_double_arg("--sigma", argv[++i], -1.0e308, &cfg.sigma))
                return 2;
        } else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            if (!parse_backend(argv[++i], &cfg.backend)) {
                fprintf(stderr, "bench_eigs: unknown --backend '%s'\n", argv[i]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--precond") && i + 1 < argc) {
            if (!parse_precond(argv[++i], &cfg.precond_kind)) {
                fprintf(stderr, "bench_eigs: unknown --precond '%s'\n", argv[i]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--block-size") && i + 1 < argc) {
            if (!parse_idx_arg("--block-size", argv[++i], 0, &cfg.block_size))
                return 2;
        } else if (!strcmp(argv[i], "--tol") && i + 1 < argc) {
            if (!parse_double_arg("--tol", argv[++i], 0.0, &cfg.tol))
                return 2;
        } else if (!strcmp(argv[i], "--max-iters") && i + 1 < argc) {
            if (!parse_idx_arg("--max-iters", argv[++i], 1, &cfg.max_iters))
                return 2;
        } else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) {
            idx_t r = 0;
            if (!parse_idx_arg("--repeats", argv[++i], 1, &r))
                return 2;
            repeats = (int)r;
            cfg.repeats = repeats;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_help();
            return 0;
        } else {
            fprintf(stderr, "bench_eigs: unknown option '%s' (try --help)\n", argv[i]);
            return 2;
        }
    }

    /* Default to --sweep default when no mode flag is given.
     * Matches the Makefile `bench-eigs` smoke convention. */
    if (!single_matrix && !compare && !sweep)
        sweep = 1;

    if (compare)
        return run_compare_mode(repeats, csv);
    if (single_matrix)
        return run_single(single_matrix, cfg, csv);
    return run_default_sweep(repeats, csv);
}
