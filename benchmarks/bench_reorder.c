/*
 * bench_reorder.c — Sprint 22 Day 9 cross-ordering benchmark.
 *
 * For each fixture in the SuiteSparse corpus the eigensolver work
 * has been stressing since Sprint 20, run the four fill-reducing
 * orderings (RCM / AMD / COLAMD / ND) plus NONE as a baseline and
 * report:
 *
 *   - matrix          fixture name
 *   - n               row/col dimension
 *   - reorder         ordering name
 *   - nnz_L           nonzero count of the symbolic Cholesky factor
 *                     (from `sparse_analyze`'s sym_L; an upper
 *                     bound on what the numeric factor stores)
 *   - reorder_ms      wall-clock time of the reorder call alone
 *                     (excludes the symbolic etree / colcount work)
 *   - factor_ms       wall-clock time of `sparse_cholesky_factor`
 *                     on the permuted matrix; "skip" when the
 *                     factor is impractically slow at the bench's
 *                     time budget (Pres_Poisson under linked-list
 *                     Cholesky takes minutes)
 *
 * Output schema is plain CSV.  Pipe through `column -t -s,` to read.
 *
 * Optional flags:
 *   --nd-threshold <n>    Override `sparse_reorder_nd_base_threshold`
 *                         (Day 9 sweep: {4, 8, 16, 32, 64, 128, 200,
 *                         500}; default landed at 32 — see
 *                         `docs/planning/EPIC_2/SPRINT_22/bench_day9_nd.txt`).
 *   --skip-factor         Skip the numeric factor pass — useful for
 *                         the ND threshold sweep where only fill /
 *                         reorder time matter.
 *   --only <fixture>      Run only the named fixture (substring
 *                         match against the fixture name).
 */
#include "sparse_analysis.h"
#include "sparse_cholesky.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The Day-6 ND base-case cutoff.  Exposed by `src/sparse_reorder_nd.c`
 * as a non-`static` global so the Day 9 threshold sweep doesn't need
 * to recompile the library each time. */
extern idx_t sparse_reorder_nd_base_threshold;

typedef struct {
    const char *name;
    sparse_reorder_t value;
} reorder_entry_t;

static const reorder_entry_t kReorderings[] = {
    {"NONE", SPARSE_REORDER_NONE},     {"RCM", SPARSE_REORDER_RCM}, {"AMD", SPARSE_REORDER_AMD},
    {"COLAMD", SPARSE_REORDER_COLAMD}, {"ND", SPARSE_REORDER_ND},
};
static const size_t kReorderingsCount = sizeof(kReorderings) / sizeof(kReorderings[0]);

typedef struct {
    const char *name;
    const char *path;
} fixture_t;

static const fixture_t kFixtures[] = {
    {"nos4", "tests/data/suitesparse/nos4.mtx"},
    {"bcsstk04", "tests/data/suitesparse/bcsstk04.mtx"},
    {"Kuu", "tests/data/suitesparse/Kuu.mtx"},
    {"bcsstk14", "tests/data/suitesparse/bcsstk14.mtx"},
    {"s3rmt3m3", "tests/data/suitesparse/s3rmt3m3.mtx"},
    {"Pres_Poisson", "tests/data/suitesparse/Pres_Poisson.mtx"},
};
static const size_t kFixtureCount = sizeof(kFixtures) / sizeof(kFixtures[0]);

static double now_ms(void) { return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC; }

/* Apply a reordering to A and time it; returns the perm (caller frees)
 * or NULL when reorder == NONE. */
static sparse_err_t time_reorder(const SparseMatrix *A, sparse_reorder_t reorder, idx_t **perm_out,
                                 double *reorder_ms_out) {
    *perm_out = NULL;
    *reorder_ms_out = 0.0;
    if (reorder == SPARSE_REORDER_NONE)
        return SPARSE_OK;

    idx_t n = sparse_rows(A);
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    if (!perm)
        return SPARSE_ERR_ALLOC;

    double t0 = now_ms();
    sparse_err_t rc;
    switch (reorder) {
    case SPARSE_REORDER_RCM:
        rc = sparse_reorder_rcm(A, perm);
        break;
    case SPARSE_REORDER_AMD:
        rc = sparse_reorder_amd(A, perm);
        break;
    case SPARSE_REORDER_COLAMD:
        rc = sparse_reorder_colamd(A, perm);
        break;
    case SPARSE_REORDER_ND:
        rc = sparse_reorder_nd(A, perm);
        break;
    default:
        free(perm);
        return SPARSE_ERR_BADARG;
    }
    *reorder_ms_out = now_ms() - t0;

    if (rc != SPARSE_OK) {
        free(perm);
        return rc;
    }
    *perm_out = perm;
    return SPARSE_OK;
}

/* Compute symbolic Cholesky nnz(L) on a permuted matrix.  Caller
 * supplies the permutation (NULL ⇒ identity).  Returns -1 on
 * analysis failure (e.g. non-symmetric input). */
static idx_t symbolic_nnz_L(const SparseMatrix *A, const idx_t *perm) {
    SparseMatrix *PA = NULL;
    if (perm) {
        if (sparse_permute(A, perm, perm, &PA) != SPARSE_OK)
            return -1;
    }
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_err_t rc = sparse_analyze(perm ? PA : A, &opts, &analysis);
    idx_t nnz = (rc == SPARSE_OK) ? analysis.sym_L.nnz : (idx_t)-1;
    sparse_analysis_free(&analysis);
    sparse_free(PA);
    return nnz;
}

/* Time numeric Cholesky on a permuted copy.  Returns the wall-clock
 * ms or -1 if the factor failed (matrix not SPD, or simply slow on
 * `--skip-factor` runs). */
static double time_factor(const SparseMatrix *A, const idx_t *perm) {
    SparseMatrix *PA = NULL;
    if (perm) {
        if (sparse_permute(A, perm, perm, &PA) != SPARSE_OK)
            return -1.0;
    } else {
        PA = sparse_copy(A);
        if (!PA)
            return -1.0;
    }
    double t0 = now_ms();
    sparse_err_t rc = sparse_cholesky_factor(PA);
    double t = now_ms() - t0;
    sparse_free(PA);
    return (rc == SPARSE_OK) ? t : -1.0;
}

static void run_one(const fixture_t *fx, int do_factor) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, fx->path) != SPARSE_OK) {
        fprintf(stderr, "skipped %s: load failed\n", fx->name);
        return;
    }
    idx_t n = sparse_rows(A);

    for (size_t i = 0; i < kReorderingsCount; i++) {
        const reorder_entry_t *r = &kReorderings[i];

        idx_t *perm = NULL;
        double r_ms = 0.0;
        sparse_err_t rc = time_reorder(A, r->value, &perm, &r_ms);
        if (rc != SPARSE_OK) {
            printf("%s,%d,%s,error,%.1f,n/a\n", fx->name, (int)n, r->name, r_ms);
            continue;
        }

        idx_t nnz = symbolic_nnz_L(A, perm);
        if (nnz < 0) {
            printf("%s,%d,%s,n/a,%.1f,n/a\n", fx->name, (int)n, r->name, r_ms);
            free(perm);
            continue;
        }

        double f_ms = -1.0;
        if (do_factor)
            f_ms = time_factor(A, perm);

        if (f_ms < 0)
            printf("%s,%d,%s,%d,%.1f,skip\n", fx->name, (int)n, r->name, (int)nnz, r_ms);
        else
            printf("%s,%d,%s,%d,%.1f,%.1f\n", fx->name, (int)n, r->name, (int)nnz, r_ms, f_ms);
        fflush(stdout);

        free(perm);
    }
    sparse_free(A);
}

int main(int argc, char **argv) {
    int do_factor = 1;
    const char *only = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nd-threshold") == 0 && i + 1 < argc) {
            sparse_reorder_nd_base_threshold = (idx_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--skip-factor") == 0) {
            do_factor = 0;
        } else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            only = argv[++i];
        }
    }

    fprintf(stderr, "# nd_base_threshold=%d, factor=%s\n", (int)sparse_reorder_nd_base_threshold,
            do_factor ? "yes" : "no");
    printf("matrix,n,reorder,nnz_L,reorder_ms,factor_ms\n");

    for (size_t i = 0; i < kFixtureCount; i++) {
        if (only && !strstr(kFixtures[i].name, only))
            continue;
        run_one(&kFixtures[i], do_factor);
    }
    return 0;
}
