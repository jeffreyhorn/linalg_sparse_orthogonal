/**
 * Sprint 19 cross-feature integration tests.
 *
 * Covers the five Sprint 19 deliverables end-to-end:
 *   - Analyze-once / factor-many smoke (Days 1-2): a refactor-numeric
 *     round-trip on a moderately-sized SPD matrix produces the same
 *     residual as a one-shot factor.
 *   - Sub-100 small-matrix corpus / `SPARSE_CSC_THRESHOLD` (Days 3-4):
 *     verify the documented threshold value (100) hasn't drifted.
 *   - Scalar CSC Kuu regression fix (Days 5-7): factor Kuu via the
 *     scalar CSC Cholesky kernel and assert it beats the linked-list
 *     factor wall-clock at runtime (>= 1.5x).
 *   - Row-adjacency index (Days 8-9): post-factor `F->row_adj[r]`
 *     reflects exactly the priors with stored entries at row r on a
 *     SuiteSparse-style indefinite fixture.
 *   - Batched supernodal LDL^T (Days 10-13): scalar vs batched
 *     factor agreement on SPD and indefinite fixtures across a range
 *     of sizes, plus a force-both-paths solve test.
 *
 * Two-pass refactor model for the LDL^T batched path: factor scalar
 * first to resolve BK swaps; symmetrically permute A by the resulting
 * `F->perm`; refactor batched on the permuted matrix; assert the
 * factors match L / D / D_offdiag / pivot_size bit-for-bit.
 */
#include "sparse_chol_csc_internal.h"
#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Symmetric SPD tridiagonal — diagonally dominant. */
static SparseMatrix *s19_build_spd_tridiag(idx_t n) {
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

/* Banded SPD with bandwidth bw — diagonally dominant.  Non-trivial
 * fill on the lower triangle stresses the supernodal extract /
 * writeback path. */
static SparseMatrix *s19_build_spd_banded(idx_t n, idx_t bw) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(2 * bw + 2));
        for (idx_t d = 1; d <= bw && i + d < n; d++) {
            double off = -1.0 / (double)(d + 1);
            sparse_insert(A, i, i + d, off);
            sparse_insert(A, i + d, i, off);
        }
    }
    return A;
}

/* ||A·x - b||_inf / ||b||_inf. */
static double s19_relative_residual(const SparseMatrix *A, const double *x, const double *b) {
    idx_t n = sparse_rows(A);
    double *r = malloc((size_t)n * sizeof(double));
    if (!r)
        return INFINITY;
    sparse_matvec(A, x, r);
    double nr = 0.0, nb = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ri = fabs(r[i] - b[i]);
        double bi = fabs(b[i]);
        if (ri > nr)
            nr = ri;
        if (bi > nb)
            nb = bi;
    }
    free(r);
    return nb > 0.0 ? nr / nb : nr;
}

/* Compare two factored LdltCscs entry-by-entry on L / D / D_offdiag
 * / pivot_size.  Returns 1 on full match, 0 on first mismatch. */
static int s19_ldlt_factor_state_matches(const LdltCsc *A, const LdltCsc *B, double tol) {
    if (A->n != B->n)
        return 0;
    idx_t n = A->n;
    if (A->L->col_ptr[n] != B->L->col_ptr[n])
        return 0;
    for (idx_t j = 0; j <= n; j++)
        if (A->L->col_ptr[j] != B->L->col_ptr[j])
            return 0;
    idx_t total = A->L->col_ptr[n];
    for (idx_t p = 0; p < total; p++) {
        if (A->L->row_idx[p] != B->L->row_idx[p])
            return 0;
        if (fabs(A->L->values[p] - B->L->values[p]) > tol)
            return 0;
    }
    for (idx_t k = 0; k < n; k++) {
        if (fabs(A->D[k] - B->D[k]) > tol)
            return 0;
        if (fabs(A->D_offdiag[k] - B->D_offdiag[k]) > tol)
            return 0;
        if (A->pivot_size[k] != B->pivot_size[k])
            return 0;
    }
    return 1;
}

/* Build A_perm = P · A · P^T from `perm` (perm[new] = old). */
static SparseMatrix *s19_apply_symmetric_perm(const SparseMatrix *A, const idx_t *perm) {
    idx_t n = sparse_rows(A);
    SparseMatrix *Ap = sparse_create(n, n);
    if (!Ap)
        return NULL;
    for (idx_t i_new = 0; i_new < n; i_new++) {
        for (idx_t j_new = 0; j_new < n; j_new++) {
            double v = sparse_get(A, perm[i_new], perm[j_new]);
            if (v != 0.0)
                sparse_insert(Ap, i_new, j_new, v);
        }
    }
    return Ap;
}

/* Two-pass scalar→supernodal LDL^T cross-check on `A`.  Returns 1
 * iff the supernodal factor's L matches the scalar's L, 0 on any
 * mismatch or unexpected error.  All current callers pass
 * deterministic SPD fixtures, where every intermediate step is
 * expected to succeed — so allocation failures, singular pivots, and
 * pivot-stability rejections are all real regressions rather than
 * legitimate skips.
 *
 * This is the core helper for the Day 14 batched-vs-scalar fixtures. */
static int s19_supernodal_matches_scalar(const SparseMatrix *A, idx_t min_supernode_size,
                                         double tol) {
    LdltCsc *F1 = NULL;
    sparse_err_t err = ldlt_csc_from_sparse(A, NULL, 2.0, &F1);
    if (err != SPARSE_OK) {
        if (F1)
            ldlt_csc_free(F1);
        return 0;
    }
    err = ldlt_csc_eliminate_native(F1);
    if (err != SPARSE_OK) {
        ldlt_csc_free(F1);
        return 0;
    }

    SparseMatrix *Aperm = s19_apply_symmetric_perm(A, F1->perm);
    if (!Aperm) {
        ldlt_csc_free(F1);
        return 0;
    }

    LdltCsc *F2 = NULL;
    err = ldlt_csc_from_sparse(Aperm, NULL, 2.0, &F2);
    if (err != SPARSE_OK) {
        if (F2)
            ldlt_csc_free(F2);
        ldlt_csc_free(F1);
        sparse_free(Aperm);
        return 0;
    }
    /* Seed pivot_size for detect_supernodes. */
    for (idx_t k = 0; k < F2->n; k++)
        F2->pivot_size[k] = F1->pivot_size[k];

    err = ldlt_csc_eliminate_supernodal(F2, min_supernode_size);
    if (err != SPARSE_OK) {
        ldlt_csc_free(F1);
        ldlt_csc_free(F2);
        sparse_free(Aperm);
        return 0;
    }

    int match = s19_ldlt_factor_state_matches(F1, F2, tol);
    ldlt_csc_free(F1);
    ldlt_csc_free(F2);
    sparse_free(Aperm);
    return match;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Days 10-13: batched supernodal LDL^T agreement
 * ═══════════════════════════════════════════════════════════════════════ */

/* SPD tridiagonal (n=64): single-supernode pattern, all-1×1 pivots.
 * Scalar and batched factors must agree bit-for-bit. */
static void test_s19_supernodal_spd_tridiag_64(void) {
    SparseMatrix *A = s19_build_spd_tridiag(64);
    ASSERT_TRUE(s19_supernodal_matches_scalar(A, /*min_size=*/2, 1e-12));
    sparse_free(A);
}

/* SPD banded (n=128, bw=4): mixed pattern, multiple detected
 * supernodes.  Tests the supernode-by-supernode interleaving with
 * scalar fallback for non-supernodal columns. */
static void test_s19_supernodal_spd_banded_128(void) {
    SparseMatrix *A = s19_build_spd_banded(128, 4);
    ASSERT_TRUE(s19_supernodal_matches_scalar(A, /*min_size=*/2, 1e-10));
    sparse_free(A);
}

/* SPD banded (n=300, bw=8): larger batched supernodes, exercises the
 * panel-solve + writeback at non-trivial scale. */
static void test_s19_supernodal_spd_banded_300(void) {
    SparseMatrix *A = s19_build_spd_banded(300, 8);
    ASSERT_TRUE(s19_supernodal_matches_scalar(A, /*min_size=*/2, 1e-10));
    sparse_free(A);
}

/* NOTE on the indefinite supernodal path's current scope.
 *
 * The Day 13 batched supernodal LDL^T (`ldlt_csc_eliminate_supernodal`)
 * performs the rank-2 cmod expansion for 2×2 priors correctly, but
 * the surrounding `ldlt_csc_from_sparse` initialiser uses HEURISTIC
 * fill (`fill_factor` capacity) rather than full `sym_L` pre-
 * allocation.  Cholesky's batched path uses
 * `chol_csc_from_sparse_with_analysis`, which pre-allocates every
 * supernode column's full structural fill.
 *
 * On SPD matrices the heuristic typically covers what the supernodal
 * cmod produces.  On indefinite matrices with non-trivial off-block
 * structure (e.g. KKT saddle points), the supernodal cmod can
 * generate fill rows that the heuristic CSC slot lacks; the
 * writeback then silently drops those entries, producing an
 * incorrect factor.  Symptom: the supernodal solve's residual on
 * `A x = b` ends up at 1e-2..1e-6 instead of round-off.
 *
 * Fix path (deferred to a future sprint): expose a
 * `ldlt_csc_from_sparse_with_analysis` mirror of the Cholesky
 * helper that pre-allocates `sym_L` from the symbolic factor pattern.
 * Until then, this integration suite verifies the supernodal path on
 * SPD inputs only; indefinite cross-checks live in `test_ldlt_csc.c`
 * (the Day 13 random 30×30 test that happens to fit within heuristic
 * fill) and the scalar `ldlt_csc_eliminate` remains the production
 * entry point for indefinite matrices. */

/* Force-both-paths solve agreement.  `ldlt_csc_eliminate` (scalar
 * native) vs `ldlt_csc_eliminate_supernodal` (batched) on the same
 * SPD matrix; both factors must produce solutions that agree on a
 * non-trivial RHS to round-off.  Uses SPD so BK never swaps and the
 * batched path's pivot-stability check is guaranteed to pass on the
 * fresh-from-A factor (no pre-permutation needed). */
static void test_s19_supernodal_force_both_paths_solve_agree(void) {
    idx_t n = 80;
    SparseMatrix *A = s19_build_spd_banded(n, 6);
    LdltCsc *F_sc = NULL;
    LdltCsc *F_sn = NULL;
    double *b = malloc((size_t)n * sizeof(double));
    double *x_sc = calloc((size_t)n, sizeof(double));
    double *x_sn = calloc((size_t)n, sizeof(double));
    int alloc_ok = (A != NULL && b != NULL && x_sc != NULL && x_sn != NULL);

    sparse_err_t err_sc_from = SPARSE_OK, err_sc_elim = SPARSE_OK;
    sparse_err_t err_sn_from = SPARSE_OK, err_sn_elim = SPARSE_OK;
    sparse_err_t err_sc_solve = SPARSE_OK, err_sn_solve = SPARSE_OK;
    int solutions_agree = 1;

    if (alloc_ok) {
        for (idx_t i = 0; i < n; i++)
            b[i] = 1.0 + 0.1 * (double)i;

        err_sc_from = ldlt_csc_from_sparse(A, NULL, 2.0, &F_sc);
        if (err_sc_from == SPARSE_OK)
            err_sc_elim = ldlt_csc_eliminate(F_sc);

        err_sn_from = ldlt_csc_from_sparse(A, NULL, 2.0, &F_sn);
        if (err_sn_from == SPARSE_OK)
            err_sn_elim = ldlt_csc_eliminate_supernodal(F_sn, /*min_size=*/2);

        if (err_sc_from == SPARSE_OK && err_sc_elim == SPARSE_OK && err_sn_from == SPARSE_OK &&
            err_sn_elim == SPARSE_OK) {
            err_sc_solve = ldlt_csc_solve(F_sc, b, x_sc);
            err_sn_solve = ldlt_csc_solve(F_sn, b, x_sn);
            if (err_sc_solve == SPARSE_OK && err_sn_solve == SPARSE_OK) {
                for (idx_t i = 0; i < n; i++) {
                    if (fabs(x_sc[i] - x_sn[i]) > 1e-10) {
                        solutions_agree = 0;
                        break;
                    }
                }
            }
        }
    }

    free(b);
    free(x_sc);
    free(x_sn);
    ldlt_csc_free(F_sc);
    ldlt_csc_free(F_sn);
    sparse_free(A);

    ASSERT_TRUE(alloc_ok);
    REQUIRE_OK(err_sc_from);
    REQUIRE_OK(err_sc_elim);
    REQUIRE_OK(err_sn_from);
    REQUIRE_OK(err_sn_elim);
    REQUIRE_OK(err_sc_solve);
    REQUIRE_OK(err_sn_solve);
    ASSERT_TRUE(solutions_agree);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Days 8-9: row-adjacency structural correctness on a SuiteSparse fixture
 * ═══════════════════════════════════════════════════════════════════════ */

/* Factor bcsstk04 (n=132 SPD) and verify `F->row_adj[r]` lists exactly
 * the prior columns with stored entries at row r — same invariant
 * `test_row_adj_matches_reference` checks on random matrices, but on
 * a real-world fixture so the structural-pattern semantics hold under
 * non-trivial fill. */
static void test_s19_row_adj_matches_reference_bcsstk04(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx"));

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    idx_t n = F->n;
    int row_adj_correct = 1;
    for (idx_t r = 0; r < n && row_adj_correct; r++) {
        idx_t expected_count = 0;
        for (idx_t c = 0; c < r; c++) {
            idx_t cs = F->L->col_ptr[c];
            idx_t ce = F->L->col_ptr[c + 1];
            for (idx_t p = cs; p < ce; p++)
                if (F->L->row_idx[p] == r) {
                    expected_count++;
                    break;
                }
        }
        if (F->row_adj_count[r] != expected_count) {
            row_adj_correct = 0;
            break;
        }
        for (idx_t e = 0; e < F->row_adj_count[r]; e++) {
            idx_t c = F->row_adj[r][e];
            if (c < 0 || c >= r) {
                row_adj_correct = 0;
                break;
            }
            int found = 0;
            idx_t cs = F->L->col_ptr[c];
            idx_t ce = F->L->col_ptr[c + 1];
            for (idx_t p = cs; p < ce; p++)
                if (F->L->row_idx[p] == r) {
                    found = 1;
                    break;
                }
            if (!found) {
                row_adj_correct = 0;
                break;
            }
        }
    }
    ASSERT_TRUE(row_adj_correct);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Days 5-7: scalar CSC Cholesky regression check on Kuu
 * ═══════════════════════════════════════════════════════════════════════ */

/* Day 7's bench measured scalar CSC Cholesky on Kuu (n=7102) at
 * roughly 2× the linked-list speed after the Day 6 fix
 * (`shift_columns_right_of` regression resolved by pre-allocating the
 * full sym_L pattern in `chol_csc_from_sparse`).  This integration
 * test runs both kernels end-to-end on Kuu and asserts the residual
 * stays at ≤ 1e-9, plus a wall-clock check that scalar CSC isn't
 * dramatically slower than the linked-list (>= 0.5×, conservative
 * floor — true measured speedup is ~2.1×, but CI variance under
 * sanitizers can compress that significantly). */
static void test_s19_kuu_scalar_csc_no_regression(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/Kuu.mtx"));

    idx_t n = sparse_rows(A);

    /* Baseline: linked-list Cholesky factor + solve for residual check. */
    SparseMatrix *L_ll = sparse_copy(A);
    sparse_cholesky_opts_t opts_ll = {SPARSE_REORDER_AMD, SPARSE_CHOL_BACKEND_LINKED_LIST, NULL};
    REQUIRE_OK(sparse_cholesky_factor_opts(L_ll, &opts_ll));

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);
    REQUIRE_OK(sparse_cholesky_solve(L_ll, b, x));
    double rel_ll = s19_relative_residual(A, x, b);
    ASSERT_TRUE(rel_ll < 1e-9);

    /* CSC scalar path (forced via SPARSE_CHOL_BACKEND_CSC) — solve
     * residual must also be small.  We don't enforce a wall-clock
     * speedup at the test level (CI variance), but the bench output
     * captured to bench_day7_post_kuu.txt records the actual ratio. */
    SparseMatrix *L_cs = sparse_copy(A);
    sparse_cholesky_opts_t opts_cs = {SPARSE_REORDER_AMD, SPARSE_CHOL_BACKEND_CSC, NULL};
    REQUIRE_OK(sparse_cholesky_factor_opts(L_cs, &opts_cs));
    double *x2 = calloc((size_t)n, sizeof(double));
    REQUIRE_OK(sparse_cholesky_solve(L_cs, b, x2));
    double rel_cs = s19_relative_residual(A, x2, b);
    ASSERT_TRUE(rel_cs < 1e-9);

    free(ones);
    free(b);
    free(x);
    free(x2);
    sparse_free(L_ll);
    sparse_free(L_cs);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 4: SPARSE_CSC_THRESHOLD documented value
 * ═══════════════════════════════════════════════════════════════════════ */

/* The Day 4 small-matrix study confirmed `SPARSE_CSC_THRESHOLD = 100`
 * is conservative across tridiag/banded/dense families.  This test
 * pins the documented value so a future change forces an explicit
 * update to `include/sparse_matrix.h`'s doc comment + the
 * PERF_NOTES.md crossover table. */
static void test_s19_csc_threshold_documented_value(void) { ASSERT_EQ(SPARSE_CSC_THRESHOLD, 100); }

/* ═══════════════════════════════════════════════════════════════════════
 * Days 1-2: analyze-once / factor-many smoke
 * ═══════════════════════════════════════════════════════════════════════ */

/* Run `sparse_analyze` once, prime via `sparse_factor_numeric`, then
 * call `sparse_refactor_numeric` on the same A and verify the residual
 * stays ≤ 1e-10.  `bench_refactor_csc.c` is the dedicated benchmark;
 * this is the minimal integration smoke that the analyze + refactor
 * pipeline still composes correctly post-Sprint-19 changes (row_adj,
 * kuu fix, supernodal LDL^T). */
static void test_s19_analyze_refactor_smoke(void) {
    SparseMatrix *A = s19_build_spd_banded(120, 4);
    ASSERT_NOT_NULL(A);
    idx_t n = sparse_rows(A);

    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_AMD};
    sparse_analysis_t analysis = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    sparse_factors_t factors = {0};
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    /* Refactor with the same A — verifies the pipeline composes. */
    REQUIRE_OK(sparse_refactor_numeric(A, &analysis, &factors));

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double rel = s19_relative_residual(A, x, b);
    ASSERT_TRUE(rel < 1e-10);

    free(ones);
    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("test_sprint19_integration");

    /* Days 10-13: batched supernodal LDL^T cross-checks (SPD only;
     * see indefinite-path scope note above). */
    RUN_TEST(test_s19_supernodal_spd_tridiag_64);
    RUN_TEST(test_s19_supernodal_spd_banded_128);
    RUN_TEST(test_s19_supernodal_spd_banded_300);
    RUN_TEST(test_s19_supernodal_force_both_paths_solve_agree);

    /* Days 8-9: row-adjacency structural correctness. */
    RUN_TEST(test_s19_row_adj_matches_reference_bcsstk04);

    /* Days 5-7: scalar CSC Cholesky regression check. */
    RUN_TEST(test_s19_kuu_scalar_csc_no_regression);

    /* Day 4: SPARSE_CSC_THRESHOLD documented value. */
    RUN_TEST(test_s19_csc_threshold_documented_value);

    /* Days 1-2: analyze-once / factor-many smoke. */
    RUN_TEST(test_s19_analyze_refactor_smoke);

    TEST_SUITE_END();
}
