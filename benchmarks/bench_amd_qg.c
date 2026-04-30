/*
 * bench_amd_qg.c — Sprint 22 Day 13 AMD bitset vs quotient-graph.
 *
 * Day 12 deleted the bitset AMD; for the Day 13 comparison this
 * file reconstitutes the pre-swap bitset implementation as a static
 * helper (`bench_amd_bitset`) that lives only in the bench binary,
 * and times it side-by-side with the production
 * `sparse_reorder_amd` (now a thin wrapper around the quotient-graph
 * implementation in `src/sparse_reorder_amd_qg.c`).  Keeping both
 * implementations in one binary avoids the `git stash` + clean-build
 * dance the Day 13 plan suggested as the AB harness.
 *
 * For each fixture (SuiteSparse + synthetic banded) we report:
 *   - wall_ms     : wall-clock time of the reorder call alone.
 *   - peak_rss_mb : max-RSS delta around the call, from `getrusage`.
 *                   macOS reports ru_maxrss in bytes, Linux in
 *                   kilobytes — the platform branch lives below.
 *   - nnz_L       : symbolic Cholesky nnz under the resulting
 *                   permutation; sanity check that both
 *                   implementations land on the same fill quality.
 *
 * Output schema: plain CSV.  Pipe through `column -t -s,` to read.
 *
 * Optional flags:
 *   --skip-bitset       Skip the bitset side (the synthetic 20 000×
 *                       20 000 row takes ~minutes under the bitset
 *                       and CSV consumers may not need it again).
 *   --only <fixture>    Run only the named fixture (substring match).
 */

#include "sparse_analysis.h"
#include "sparse_matrix.h"
#include "sparse_matrix_internal.h" /* sparse_build_adj */
#include "sparse_reorder.h"
#include "sparse_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

/* ─── Pre-Day-12 bitset AMD (verbatim copy from `git show
 * sprint-22~1:src/sparse_reorder.c`).  Deleted from production by
 * Day 12 commit `6fa7245`; lives here as a benchmarking foil. */

typedef uint64_t bench_bword_t;
#define BENCH_BWORD_BITS 64
#define BENCH_BWORD_IDX(i) ((i) / BENCH_BWORD_BITS)
#define BENCH_BWORD_BIT(i) ((bench_bword_t)1 << ((i) % BENCH_BWORD_BITS))

static inline void bench_bset(bench_bword_t *bs, idx_t i) {
    bs[BENCH_BWORD_IDX(i)] |= BENCH_BWORD_BIT(i);
}
static inline void bench_bclr(bench_bword_t *bs, idx_t i) {
    bs[BENCH_BWORD_IDX(i)] &= ~BENCH_BWORD_BIT(i);
}
static inline int bench_btest(const bench_bword_t *bs, idx_t i) {
    return (bs[BENCH_BWORD_IDX(i)] & BENCH_BWORD_BIT(i)) != 0;
}
static void bench_bunion(bench_bword_t *dst, const bench_bword_t *src, idx_t nwords) {
    for (idx_t w = 0; w < nwords; w++)
        dst[w] |= src[w];
}

static sparse_err_t bench_amd_bitset(const SparseMatrix *A, idx_t *perm) {
    if (!A || !perm)
        return SPARSE_ERR_NULL;
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;
    idx_t n = sparse_rows(A);
    if (n == 0)
        return SPARSE_OK;
    if (n == 1) {
        perm[0] = 0;
        return SPARSE_OK;
    }

    idx_t *adj_ptr = NULL;
    idx_t *adj_list = NULL;
    sparse_err_t rc = sparse_build_adj(A, &adj_ptr, &adj_list);
    if (rc != SPARSE_OK)
        return rc;

    idx_t nwords = (n + BENCH_BWORD_BITS - 1) / BENCH_BWORD_BITS;
    bench_bword_t *adj_bits = calloc((size_t)n * (size_t)nwords, sizeof(bench_bword_t));
    int *eliminated = calloc((size_t)n, sizeof(int));
    idx_t *degree = malloc((size_t)n * sizeof(idx_t));
    if (!adj_bits || !eliminated || !degree) {
        free(adj_bits);
        free(eliminated);
        free(degree);
        free(adj_ptr);
        free(adj_list);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++) {
        bench_bword_t *row = &adj_bits[(size_t)i * (size_t)nwords];
        for (idx_t k = adj_ptr[i]; k < adj_ptr[i + 1]; k++)
            bench_bset(row, adj_list[k]);
        degree[i] = adj_ptr[i + 1] - adj_ptr[i];
    }
    free(adj_ptr);
    free(adj_list);

    for (idx_t step = 0; step < n; step++) {
        idx_t best = -1;
        idx_t best_deg = n + 1;
        for (idx_t i = 0; i < n; i++) {
            if (!eliminated[i] && degree[i] < best_deg) {
                best_deg = degree[i];
                best = i;
            }
        }
        perm[step] = best;
        /* `best` is guaranteed in [0, n) — every step has at least
         * one unelim vertex with finite degree.  Static analyser
         * doesn't track this. */
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        eliminated[best] = 1;

        bench_bword_t *best_row = &adj_bits[(size_t)best * (size_t)nwords];
        for (idx_t u = 0; u < n; u++) {
            if (eliminated[u] || !bench_btest(best_row, u))
                continue;
            bench_bword_t *u_row = &adj_bits[(size_t)u * (size_t)nwords];
            bench_bunion(u_row, best_row, nwords);
            bench_bclr(u_row, best);
            bench_bclr(u_row, u);
            idx_t deg = 0;
            for (idx_t w = 0; w < nwords; w++) {
                bench_bword_t v = u_row[w];
                while (v) {
                    v &= v - 1;
                    deg++;
                }
            }
            degree[u] = deg;
        }
        for (idx_t i = 0; i < n; i++)
            bench_bclr(&adj_bits[(size_t)i * (size_t)nwords], best);
    }

    free(adj_bits);
    free(eliminated);
    free(degree);
    return SPARSE_OK;
}

/* ─── RSS measurement ────────────────────────────────────────────── */

/* Returns ru_maxrss in MB, normalised across platforms.  macOS
 * reports bytes, Linux kilobytes — the divisor adjusts. */
static double peak_rss_mb(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0)
        return 0.0;
    long bytes = ru.ru_maxrss;
#ifdef __APPLE__
    return (double)bytes / (1024.0 * 1024.0);
#else
    /* Linux / most Unixes: ru_maxrss is in KiB. */
    return (double)bytes / 1024.0;
#endif
}

static double now_ms(void) { return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC; }

/* ─── Synthetic banded matrix builder ────────────────────────────── */

static SparseMatrix *make_banded(idx_t n, idx_t bandwidth) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    /* Check sparse_insert returns: a partial fixture would surface
     * downstream as misleading bench timings or fill counts.  On any
     * failure free A and propagate NULL so the caller bails out
     * before timing an incomplete matrix. */
    for (idx_t i = 0; i < n; i++) {
        sparse_err_t rc = sparse_insert(A, i, i, 2.0);
        for (idx_t k = 1; rc == SPARSE_OK && k <= bandwidth; k++) {
            if (i + k < n) {
                rc = sparse_insert(A, i, i + k, -1.0);
                if (rc == SPARSE_OK)
                    rc = sparse_insert(A, i + k, i, -1.0);
            }
        }
        if (rc != SPARSE_OK) {
            sparse_free(A);
            return NULL;
        }
    }
    return A;
}

/* ─── Symbolic Cholesky nnz for the parity sanity check ──────────── */

static idx_t symbolic_nnz_L(const SparseMatrix *A, const idx_t *perm) {
    SparseMatrix *PA = NULL;
    if (sparse_permute(A, perm, perm, &PA) != SPARSE_OK)
        return -1;
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};
    sparse_analysis_t a = {0};
    sparse_err_t rc = sparse_analyze(PA, &opts, &a);
    idx_t nnz = (rc == SPARSE_OK) ? a.sym_L.nnz : (idx_t)-1;
    sparse_analysis_free(&a);
    sparse_free(PA);
    return nnz;
}

/* ─── One-fixture measurement ───────────────────────────────────── */

static void run_one(const char *name, SparseMatrix *A, int do_bitset) {
    if (!A) {
        fprintf(stderr, "skipped %s: A is NULL\n", name);
        return;
    }
    idx_t n = sparse_rows(A);
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    if (!perm) {
        fprintf(stderr, "skipped %s: perm alloc failed\n", name);
        return;
    }

    /* Quotient-graph (production). */
    {
        double rss_before = peak_rss_mb();
        double t0 = now_ms();
        sparse_err_t rc = sparse_reorder_amd(A, perm);
        double t = now_ms() - t0;
        double rss_delta = peak_rss_mb() - rss_before;
        if (rss_delta < 0)
            rss_delta = 0; /* getrusage is monotonic; floor at zero */
        if (rc == SPARSE_OK) {
            idx_t nnz = symbolic_nnz_L(A, perm);
            printf("%s,%d,qg,%.1f,%.2f,%d\n", name, (int)n, t, rss_delta, (int)nnz);
        } else {
            printf("%s,%d,qg,error_%d,%.2f,n/a\n", name, (int)n, (int)rc, rss_delta);
        }
        fflush(stdout);
    }

    /* Bitset (foil; deleted from production at Day 12). */
    if (do_bitset) {
        double rss_before = peak_rss_mb();
        double t0 = now_ms();
        sparse_err_t rc = bench_amd_bitset(A, perm);
        double t = now_ms() - t0;
        double rss_delta = peak_rss_mb() - rss_before;
        if (rss_delta < 0)
            rss_delta = 0;
        if (rc == SPARSE_OK) {
            idx_t nnz = symbolic_nnz_L(A, perm);
            printf("%s,%d,bitset,%.1f,%.2f,%d\n", name, (int)n, t, rss_delta, (int)nnz);
        } else {
            printf("%s,%d,bitset,error_%d,%.2f,n/a\n", name, (int)n, (int)rc, rss_delta);
        }
        fflush(stdout);
    }

    free(perm);
}

static SparseMatrix *load_or_null(const char *path) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, path) != SPARSE_OK)
        return NULL;
    return A;
}

int main(int argc, char **argv) {
    int do_bitset = 1;
    const char *only = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--skip-bitset") == 0)
            do_bitset = 0;
        else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc)
            only = argv[++i];
    }

    fprintf(stderr, "# bitset=%s\n", do_bitset ? "yes" : "no");
    printf("matrix,n,impl,reorder_ms,peak_rss_mb,nnz_L\n");

    typedef struct {
        const char *name;
        const char *path;
    } fixture_t;
    static const fixture_t kSS[] = {
        {"nos4", "tests/data/suitesparse/nos4.mtx"},
        {"bcsstk04", "tests/data/suitesparse/bcsstk04.mtx"},
        {"bcsstk14", "tests/data/suitesparse/bcsstk14.mtx"},
        {"Kuu", "tests/data/suitesparse/Kuu.mtx"},
        {"s3rmt3m3", "tests/data/suitesparse/s3rmt3m3.mtx"},
        {"Pres_Poisson", "tests/data/suitesparse/Pres_Poisson.mtx"},
    };
    for (size_t i = 0; i < sizeof(kSS) / sizeof(kSS[0]); i++) {
        if (only && !strstr(kSS[i].name, only))
            continue;
        SparseMatrix *A = load_or_null(kSS[i].path);
        run_one(kSS[i].name, A, do_bitset);
        sparse_free(A);
    }

    /* Synthetic banded — covers the n ≥ 5 000 territory where the
     * bitset's O(n²/64) memory starts to matter even though the
     * SuiteSparse corpus stops at ~15 000.  Bandwidth 5 keeps nnz
     * tractable; the bitset still allocates n²/8 bytes regardless. */
    typedef struct {
        const char *name;
        idx_t n;
        idx_t bw;
    } banded_t;
    static const banded_t kBanded[] = {
        {"banded_5000", 5000, 5},
        {"banded_10000", 10000, 5},
        {"banded_20000", 20000, 5},
    };
    for (size_t i = 0; i < sizeof(kBanded) / sizeof(kBanded[0]); i++) {
        if (only && !strstr(kBanded[i].name, only))
            continue;
        SparseMatrix *A = make_banded(kBanded[i].n, kBanded[i].bw);
        run_one(kBanded[i].name, A, do_bitset);
        sparse_free(A);
    }

    return 0;
}
