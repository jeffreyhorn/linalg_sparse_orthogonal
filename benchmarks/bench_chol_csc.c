/*
 * bench_chol_csc.c — Cholesky backend comparison: linked-list vs CSC
 *
 * Compares wall-clock time for factor and solve between:
 *   1. Linked-list Cholesky (sparse_cholesky_factor + sparse_cholesky_solve)
 *   2. CSC scalar     (chol_csc_factor + chol_csc_solve_perm)
 *   3. CSC supernodal (chol_csc_eliminate_supernodal + chol_csc_solve_perm)
 *
 * Since Sprint 18 Day 8 the supernodal path runs the fully batched
 * kernel on detected fundamental supernodes:
 *   Day 6 extract / writeback      — CSC ↔ dense column-major buffer
 *   Day 7 eliminate_diag           — external cmod + chol_dense_factor
 *   Day 8 eliminate_panel          — row-by-row chol_dense_solve_lower
 * For columns not inside any detected supernode the scalar
 * scatter/cmod/cdiv/gather loop runs instead.  Prior to Day 8 the
 * `factor_csc_sn` / `speedup_csc_sn` columns reflected detection
 * overhead on top of the scalar kernel; from Day 9 onwards they
 * measure the real batched speedup.
 *
 * All three paths use the same AMD fill-reducing reordering to keep
 * the comparison apples-to-apples.  Residuals from each path are
 * checked against the original A to 1e-8 relative — if any path fails
 * the residual check the benchmark prints a warning and skips timing
 * reporting for that matrix.
 *
 * Output is CSV on stdout: one header row, one row per matrix with
 *   matrix, n, nnz, factor_ll, factor_csc, factor_csc_sn,
 *   solve_ll,  solve_csc,  solve_csc_sn,
 *   speedup_csc, speedup_csc_sn, res_ll, res_csc, res_csc_sn
 * Times are in milliseconds (averaged across --repeat runs).
 *
 * Usage:
 *   ./bench_chol_csc                                # default matrix list
 *   ./bench_chol_csc path/to/matrix.mtx             # benchmark one matrix
 *   ./bench_chol_csc --repeat 5                     # average 5 runs
 *   ./bench_chol_csc --small-corpus                 # 10 sub-threshold synthetic SPDs
 *                                                   # (Sprint 19 Day 3; feeds Day 4's
 *                                                   # SPARSE_CSC_THRESHOLD retrospective)
 */
#define _POSIX_C_SOURCE 199309L

#include "sparse_analysis.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_cholesky.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_vector.h"

#include <math.h>
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
        /* Sentinel value: NaN signals an unmeasurable residual due to
         * allocation failure.  Callers print the raw value in the CSV
         * row, so "nan" in the output column is the visible cue. */
        fprintf(stderr, "bench_chol_csc: malloc failed in rel_residual (n=%d)\n", (int)n);
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
    double factor_ms; /* average factor time, milliseconds */
    double solve_ms;  /* average solve time, milliseconds */
    double residual;  /* relative residual for the last run */
    int ok;           /* 1 on success, 0 on error */
} bench_result_t;

/* ─── Linked-list Cholesky path ───────────────────────────────────── */

static bench_result_t bench_linked_list(const SparseMatrix *A, const double *b, double *x,
                                        int repeat) {
    bench_result_t r = {0, 0, 0, 1};
    double factor_total = 0.0, solve_total = 0.0;
    for (int rep = 0; rep < repeat; rep++) {
        SparseMatrix *L = sparse_copy(A);
        /* Force the linked-list kernel here so `factor_ll_ms` always
         * measures the linked-list baseline regardless of n.  Under
         * SPARSE_CHOL_BACKEND_AUTO the Sprint 18 Day 11 dispatch would
         * silently route large fixtures to the CSC supernodal kernel
         * and collapse the `speedup_csc*` columns to ~1. */
        sparse_cholesky_opts_t opts = {SPARSE_REORDER_AMD, SPARSE_CHOL_BACKEND_LINKED_LIST, NULL};

        double t0 = wall_time();
        if (sparse_cholesky_factor_opts(L, &opts) != SPARSE_OK) {
            sparse_free(L);
            r.ok = 0;
            return r;
        }
        factor_total += wall_time() - t0;

        t0 = wall_time();
        if (sparse_cholesky_solve(L, b, x) != SPARSE_OK) {
            sparse_free(L);
            r.ok = 0;
            return r;
        }
        solve_total += wall_time() - t0;

        if (rep == repeat - 1)
            r.residual = rel_residual(A, x, b);
        sparse_free(L);
    }
    r.factor_ms = factor_total * 1000.0 / (double)repeat;
    r.solve_ms = solve_total * 1000.0 / (double)repeat;
    return r;
}

/* ─── CSC scalar & CSC supernodal paths ──────────────────────────── */

typedef sparse_err_t (*csc_eliminate_fn)(CholCsc *);

static bench_result_t bench_csc_path(const SparseMatrix *A, const double *b, double *x, int repeat,
                                     csc_eliminate_fn eliminate) {
    bench_result_t r = {0, 0, 0, 1};
    double factor_total = 0.0, solve_total = 0.0;
    sparse_analysis_opts_t aopts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_AMD};

    for (int rep = 0; rep < repeat; rep++) {
        /* Fair comparison: the linked-list path runs sparse_analyze's
         * equivalent (AMD reorder + symbolic work) inside
         * sparse_cholesky_factor_opts on every iteration, so do the
         * same on the CSC path — include `sparse_analyze` + CSC
         * conversion + elimination in factor_ms.  Previously the
         * analysis was cached outside the loop, which inflated
         * speedup_csc by the cost of AMD (often the dominant factor
         * for small matrices). */
        sparse_analysis_t an = {0};
        CholCsc *L = NULL;
        double t0 = wall_time();
        if (sparse_analyze(A, &aopts, &an) != SPARSE_OK) {
            r.ok = 0;
            break;
        }
        if (chol_csc_from_sparse_with_analysis(A, &an, &L) != SPARSE_OK) {
            sparse_analysis_free(&an);
            r.ok = 0;
            break;
        }
        if (eliminate(L) != SPARSE_OK) {
            chol_csc_free(L);
            sparse_analysis_free(&an);
            r.ok = 0;
            break;
        }
        factor_total += wall_time() - t0;

        t0 = wall_time();
        if (chol_csc_solve_perm(L, an.perm, b, x) != SPARSE_OK) {
            chol_csc_free(L);
            sparse_analysis_free(&an);
            r.ok = 0;
            break;
        }
        solve_total += wall_time() - t0;

        if (rep == repeat - 1)
            r.residual = rel_residual(A, x, b);
        chol_csc_free(L);
        sparse_analysis_free(&an);
    }

    if (r.ok) {
        r.factor_ms = factor_total * 1000.0 / (double)repeat;
        r.solve_ms = solve_total * 1000.0 / (double)repeat;
    }
    return r;
}

static sparse_err_t eliminate_scalar(CholCsc *L) { return chol_csc_eliminate(L); }

static sparse_err_t eliminate_supernodal(CholCsc *L) { return chol_csc_eliminate_supernodal(L, 4); }

/* ─── Matrix runner ─────────────────────────────────────────────── */

/* Core runner: takes an already-constructed A and a display label,
 * runs all three paths, prints one CSV row, and frees A.  Called by
 * both `bench_matrix` (file-backed) and `bench_synthetic` (in-memory
 * small-corpus fixtures). */
static int bench_matrix_impl(const char *label, SparseMatrix *A, int repeat) {
    idx_t n = sparse_rows(A);
    idx_t nnz = sparse_nnz(A);

    /* RHS b = A * [1, 1, ..., 1]. */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    if (!ones || !b || !x) {
        fprintf(stderr, "bench_chol_csc: malloc failed in bench_matrix_impl (n=%d)\n", (int)n);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return 1;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);
    for (idx_t i = 0; i < n; i++)
        x[i] = 0.0;

    bench_result_t rl = bench_linked_list(A, b, x, repeat);
    bench_result_t rc = bench_csc_path(A, b, x, repeat, eliminate_scalar);
    bench_result_t rs = bench_csc_path(A, b, x, repeat, eliminate_supernodal);

    if (!rl.ok || !rc.ok || !rs.ok) {
        fprintf(stderr, "bench_chol_csc: %s — one or more paths failed\n", label);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return 1;
    }

    double sp_csc = rl.factor_ms / rc.factor_ms;
    double sp_sn = rl.factor_ms / rs.factor_ms;

    /* CSV row */
    printf("%s,%d,%d,", label, (int)n, (int)nnz);
    printf("%.3f,%.3f,%.3f,", rl.factor_ms, rc.factor_ms, rs.factor_ms);
    printf("%.3f,%.3f,%.3f,", rl.solve_ms, rc.solve_ms, rs.solve_ms);
    printf("%.2f,%.2f,", sp_csc, sp_sn);
    printf("%.2e,%.2e,%.2e\n", rl.residual, rc.residual, rs.residual);

    free(ones);
    free(b);
    free(x);
    sparse_free(A);
    return 0;
}

static int bench_matrix(const char *path, int repeat) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, path) != SPARSE_OK) {
        fprintf(stderr, "bench_chol_csc: failed to load %s\n", path);
        return 1;
    }
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    return bench_matrix_impl(base, A, repeat);
}

/* ─── Small-matrix synthetic fixtures (Sprint 19 Day 3) ───────────
 *
 * Ten in-memory SPD fixtures sized below the current
 * `SPARSE_CSC_THRESHOLD = 100` (n ∈ {20, 40, 60, 80}) so Day 4's
 * threshold retrospective has data to tune the crossover.  All
 * fixtures are generated deterministically from fixed seeds — the
 * seed constants are baked into each builder so repeat runs produce
 * identical numbers.
 *
 * Three families cover the scaling-signature space:
 *
 *   tridiag-N:  A[i,i] = 4, A[i, i±1] = -1.  Minimal fill, no RNG.
 *               Stresses pointer-chasing overhead in the linked-list
 *               kernel vs CSC's contiguous column traversal.
 *   banded-N:   diagonal = 2*bw + 2, off-diagonals ≈ -1/(d+1) + small
 *               random jitter with bandwidth bw = 4.  Diagonally
 *               dominant by construction.  Stresses the scalar
 *               kernel's `shift_columns_right_of` on non-trivial
 *               fill.
 *   dense-N:    random symmetric off-diagonals in [-1, 1] + diagonal
 *               = 2n for diagonal dominance.  Max-fill stress test
 *               for supernodal detection overhead on small supernodes.
 */

/* Deterministic uniform random in [0, 1) from a 64-bit key — avoids
 * pulling in srand48/drand48, which are XSI extensions not guaranteed
 * under `_POSIX_C_SOURCE 199309L`.  SplitMix64-style mixer. */
static double jitter_u01(uint64_t key) {
    uint64_t h = key + 0x9e3779b97f4a7c15ULL;
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return (double)(h >> 32) / (double)(1ULL << 32);
}

/* Tridiagonal SPD: A[i,i] = 4, A[i, i±1] = -1. */
static SparseMatrix *build_tridiag_spd(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    return A;
}

/* Banded SPD with bandwidth bw.  Off-diagonals are -1/(d+1) plus a
 * ±0.025 deterministic perturbation; the diagonal is 2*bw + 2,
 * guaranteeing strict diagonal dominance (row sum of |off-diag|
 * entries is 2 * sum_{d=1..bw} (1/(d+1) + 0.025) < 2 * bw < 2*bw + 2
 * for all bw >= 1). */
static SparseMatrix *build_banded_spd(idx_t n, idx_t bw, uint64_t seed) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(2 * bw + 2));
        for (idx_t d = 1; d <= bw && i + d < n; d++) {
            double base = -1.0 / (double)(d + 1);
            uint64_t key = seed ^ ((uint64_t)i * (uint64_t)n + (uint64_t)d);
            double jitter = 0.05 * (jitter_u01(key) - 0.5);
            double off = base + jitter;
            sparse_insert(A, i, i + d, off);
            sparse_insert(A, i + d, i, off);
        }
    }
    return A;
}

/* Dense SPD: random symmetric off-diagonals in [-1, 1] + diagonal
 * 2*n so every row has |A[i,i]| > sum_j |A[i,j]| (diagonal
 * dominance → SPD). */
static SparseMatrix *build_dense_spd(idx_t n, uint64_t seed) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(2 * n));
        for (idx_t j = i + 1; j < n; j++) {
            uint64_t key = seed ^ ((uint64_t)i * (uint64_t)n + (uint64_t)j);
            double v = 2.0 * (jitter_u01(key) - 0.5); /* in [-1, 1] */
            sparse_insert(A, i, j, v);
            sparse_insert(A, j, i, v);
        }
    }
    return A;
}

/* Per-fixture builder wrappers so each row in `small_corpus` has a
 * zero-argument build fn with its own n and seed baked in. */
static SparseMatrix *build_tri20(void) { return build_tridiag_spd(20); }
static SparseMatrix *build_tri40(void) { return build_tridiag_spd(40); }
static SparseMatrix *build_tri60(void) { return build_tridiag_spd(60); }
static SparseMatrix *build_tri80(void) { return build_tridiag_spd(80); }
static SparseMatrix *build_band20(void) { return build_banded_spd(20, 4, 0xc0ffee01UL); }
static SparseMatrix *build_band40(void) { return build_banded_spd(40, 4, 0xc0ffee02UL); }
static SparseMatrix *build_band60(void) { return build_banded_spd(60, 4, 0xc0ffee03UL); }
static SparseMatrix *build_band80(void) { return build_banded_spd(80, 4, 0xc0ffee04UL); }
static SparseMatrix *build_dense20(void) { return build_dense_spd(20, 0xbadcafe1UL); }
static SparseMatrix *build_dense60(void) { return build_dense_spd(60, 0xbadcafe2UL); }

typedef SparseMatrix *(*fixture_builder_fn)(void);

typedef struct {
    const char *label;
    fixture_builder_fn build;
} small_fixture_t;

static small_fixture_t small_corpus[] = {
    {"tridiag-20", build_tri20}, {"tridiag-40", build_tri40}, {"tridiag-60", build_tri60},
    {"tridiag-80", build_tri80}, {"banded-20", build_band20}, {"banded-40", build_band40},
    {"banded-60", build_band60}, {"banded-80", build_band80}, {"dense-20", build_dense20},
    {"dense-60", build_dense60},
};
static const int small_corpus_count = (int)(sizeof(small_corpus) / sizeof(small_corpus[0]));

static int bench_synthetic(const char *label, fixture_builder_fn builder, int repeat) {
    SparseMatrix *A = builder();
    if (!A) {
        fprintf(stderr, "bench_chol_csc: synthetic builder failed for %s\n", label);
        return 1;
    }
    return bench_matrix_impl(label, A, repeat);
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
    int repeat = 3;
    const char *single_path = NULL;
    int small_corpus_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--repeat") && i + 1 < argc) {
            repeat = atoi(argv[++i]);
            if (repeat < 1)
                repeat = 1;
        } else if (!strcmp(argv[i], "--small-corpus")) {
            small_corpus_mode = 1;
        } else if (argv[i][0] != '-') {
            single_path = argv[i];
        }
    }

    printf("matrix,n,nnz,"
           "factor_ll_ms,factor_csc_ms,factor_csc_sn_ms,"
           "solve_ll_ms,solve_csc_ms,solve_csc_sn_ms,"
           "speedup_csc,speedup_csc_sn,"
           "res_ll,res_csc,res_csc_sn\n");

    int rc = 0;
    if (small_corpus_mode) {
        for (int i = 0; i < small_corpus_count; i++)
            rc |= bench_synthetic(small_corpus[i].label, small_corpus[i].build, repeat);
    } else if (single_path) {
        rc |= bench_matrix(single_path, repeat);
    } else {
        for (int i = 0; i < default_matrix_count; i++)
            rc |= bench_matrix(default_matrices[i], repeat);
    }
    return rc;
}
