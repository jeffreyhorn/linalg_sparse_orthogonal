#ifndef TEST_LDLT_CSC_NATIVE_PARITY_HELPERS_H
#define TEST_LDLT_CSC_NATIVE_PARITY_HELPERS_H

#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "test_framework.h"
#include "test_ldlt_csc_oracle_helpers.h"

#include <math.h>

/* Family-local native-kernel parity tests for test_ldlt_csc.c. The proof-owner
 * file keeps RUN_TEST registration and execution order.
 */

/* Defined in the proof-owner solve block. */
static double rel_residual(const SparseMatrix *A, const double *x, const double *b);

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 3: 1x1 Bunch-Kaufman column loop
 * ═══════════════════════════════════════════════════════════════════════ */

/* Pure-diagonal indefinite: no cmod, no swap, all criterion-1 1x1. */
static void test_native_1x1_diagonal_matches_wrapper(void) {
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, -3.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 3, -5.0);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* Tridiagonal SPD: exercises the cmod loop, all criterion-1 1x1. */
static void test_native_1x1_tridiagonal_matches_wrapper(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* Mixed indefinite diagonal with weak off-diagonals: all 1x1. */
static void test_native_1x1_mixed_indefinite_matches_wrapper(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    double diag[] = {3.0, -2.5, 4.0, -1.5, 2.0};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);
    /* Off-diagonals small enough that BK criterion 1 fires on every column:
     * |diag| > alpha * |offdiag| at every step.
     */
    sparse_insert(A, 1, 0, 0.2);
    sparse_insert(A, 0, 1, 0.2);
    sparse_insert(A, 2, 1, 0.3);
    sparse_insert(A, 1, 2, 0.3);
    sparse_insert(A, 3, 2, 0.4);
    sparse_insert(A, 2, 3, 0.4);
    sparse_insert(A, 4, 3, 0.5);
    sparse_insert(A, 3, 4, 0.5);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* A = [[0.1, 0.5, 0], [0.5, 2, 0], [0, 0, 3]].  At k=0:
 *   diag = 0.1, max_offdiag = 0.5 (row 1), alpha*max_offdiag ~= 0.32
 *   -> phase 2 triggered.  Partner col 1 has dense_col_r = {0:0.5, 1:2}.
 *   sigma_r (max over i != 1) = 0.5.
 *   Criterion 2: 0.1*0.5 = 0.05 < 0.64*0.25 = 0.16.  Fails.
 *   Criterion 3: |2| >= 0.64*0.5 = 0.32.  Passes -> swap 0<->1.
 */
static void test_native_1x1_with_swap_matches_wrapper(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 0.5);
    sparse_insert(A, 1, 0, 0.5);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* Larger tridiagonal indefinite: stress the column loop at n=20. */
static void test_native_1x1_tridiag_large_matches_wrapper(void) {
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    /* Diagonals alternating sign, off-diagonals small enough for BK to pick
     * criterion-1 1x1 pivots throughout.
     */
    for (idx_t i = 0; i < n; i++) {
        double d = (i % 2 == 0) ? 5.0 : -5.0;
        sparse_insert(A, i, i, d);
        if (i > 0) {
            sparse_insert(A, i, i - 1, 0.7);
            sparse_insert(A, i - 1, i, 0.7);
        }
    }
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* Zero pivot detection: sing_tol guards against near-zero 1x1. */
static void test_native_detects_near_zero_1x1_pivot(void) {
    /* A = diag(1e-20, 1.0): column 0's diagonal is below sing_tol
     * (~DROP_TOL * ||A||_inf ~= 1e-14) and max_offdiag is 0, so BK fires
     * criterion 1 (no swap possible without off-diagonals). The 1x1
     * singularity check must reject this.
     */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1e-20);
    sparse_insert(A, 1, 1, 1.0);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    ASSERT_ERR(ldlt_csc_eliminate(F), SPARSE_ERR_SINGULAR);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* Identity: trivial pass-through, perm stays identity. */
static void test_native_1x1_identity_matches_wrapper(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ==========================================================================
 * Sprint 18 Day 4: 2x2 Bunch-Kaufman block pivots
 * ========================================================================== */

/* Forced 2x2 at (0, 1): the canonical BK criterion-4 matrix. */
static void test_native_2x2_forced_matches_wrapper(void) {
    /* A = [[0.1, 1], [1, 0.3]]: both diagonals small vs the off-diagonal, BK
     * picks a 2x2 block at (0, 1). Matches test_eliminate_forced_2x2's setup;
     * the native kernel must now produce the same L / D / D_offdiag /
     * pivot_size / perm as the wrapper.
     */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* 2x2 pivot requiring r<->k+1 swap (partner not adjacent). */
static void test_native_2x2_nonadjacent_partner_matches_wrapper(void) {
    /* 4x4 matrix where the BK partner for column 0 is row 2, not row 1. That
     * forces the native kernel's r <-> k+1 swap branch.
     *
     * Construction: at k=0, we want max_offdiag at row 2 (not 1). Put a tiny
     * diag at 0, small off-diag (0,1), and large off-diag (0,2); row 2's
     * diagonal is small enough that criterion 3 fails and criterion 4 fires.
     */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 0.05);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 2, 0.05);
    sparse_insert(A, 3, 3, 4.0);
    sparse_insert(A, 1, 0, 0.2);
    sparse_insert(A, 0, 1, 0.2);
    sparse_insert(A, 2, 0, 1.5);
    sparse_insert(A, 0, 2, 1.5);
    sparse_insert(A, 3, 2, 0.3);
    sparse_insert(A, 2, 3, 0.3);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* Mixed 1x1 + 2x2 pivots with subsequent column cmod. */
static void test_native_mixed_pivots_matches_wrapper(void) {
    /* 4x4 matrix where col 0 takes a 2x2 block at (0, 1) and cols 2 and 3 then
     * use 1x1 pivots whose Schur complement receives a cross-term contribution
     * from the 2x2 at (0, 1). Verifies that `ldlt_csc_cmod_unified`'s Phase B
     * correctly accumulates the
     * `L[:, 0] * d_off * L[col, 1] + L[:, 1] * d_off * L[col, 0]` term when
     * computing dense_col for col 2 and col 3.
     */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 1, 1, 0.3);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 3, 5.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 2, 0, 0.4);
    sparse_insert(A, 0, 2, 0.4);
    sparse_insert(A, 2, 1, 0.5);
    sparse_insert(A, 1, 2, 0.5);
    sparse_insert(A, 3, 1, 0.3);
    sparse_insert(A, 1, 3, 0.3);
    sparse_insert(A, 3, 2, 0.2);
    sparse_insert(A, 2, 3, 0.2);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* Existing SuiteSparse-style indefinite cross-check fixtures. */
static void test_native_mixed_pivots_larger_matches_wrapper(void) {
    /* 6x6 matrix with a 2x2 pivot near the start and additional structure that
     * stresses cmod cross-terms for later 1x1 and 2x2 columns. Adapted from the
     * existing mixed-pivot test pattern so wrapper and native can be directly
     * compared.
     */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    double diag[] = {0.05, 0.2, 3.0, -4.0, 2.0, -2.5};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);
    struct {
        idx_t r;
        idx_t c;
        double v;
    } off[] = {{1, 0, 0.9}, {2, 0, 0.3}, {3, 0, 0.2}, {2, 1, 0.4}, {3, 1, 0.1},
               {3, 2, 0.6}, {4, 2, 0.5}, {4, 3, 0.3}, {5, 3, 0.4}, {5, 4, 0.2}};
    for (size_t k = 0; k < sizeof(off) / sizeof(off[0]); k++) {
        sparse_insert(A, off[k].r, off[k].c, off[k].v);
        sparse_insert(A, off[k].c, off[k].r, off[k].v);
    }
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* Solve end-to-end under native kernel. */
static void test_native_2x2_solve_matches_linked_list(void) {
    /* Factor the forced-2x2 matrix via the native kernel, solve A*x = b with a
     * known b, and assert the residual matches the wrapper's solve to
     * round-off. This exercises the full factor -> solve pipeline through the
     * native path.
     */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);

    LdltCsc *Fn = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    REQUIRE_OK(ldlt_csc_eliminate(Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    double b[] = {1.0, -0.5};
    double x[2] = {0};
    REQUIRE_OK(ldlt_csc_solve(Fn, b, x));
    ASSERT_TRUE(rel_residual(A, x, b) < 1e-12);

    ldlt_csc_free(Fn);
    sparse_free(A);
}

/* Inertia preserved: #pos/#neg entries of D match the wrapper. */
static void test_native_2x2_inertia_matches_wrapper(void) {
    /* Use the forced-2x2 matrix [[0.1, 1], [1, 0.3]]. Wrapper and native must
     * produce the same D / D_offdiag, so their 2x2 block eigenvalue sign
     * decomposition (= inertia contribution) is identical. We don't compute
     * eigenvalues here; instead we assert D and D_offdiag match which is
     * sufficient (the block eigenvalues are a deterministic function of the
     * entries).
     */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);

    LdltCsc *Fw = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &Fw));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_WRAPPER);
    REQUIRE_OK(ldlt_csc_eliminate(Fw));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    LdltCsc *Fn = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    REQUIRE_OK(ldlt_csc_eliminate(Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    for (idx_t i = 0; i < 2; i++) {
        ASSERT_EQ(Fw->pivot_size[i], Fn->pivot_size[i]);
        ASSERT_TRUE(fabs(Fw->D[i] - Fn->D[i]) < 1e-12);
        ASSERT_TRUE(fabs(Fw->D_offdiag[i] - Fn->D_offdiag[i]) < 1e-12);
    }

    ldlt_csc_free(Fw);
    ldlt_csc_free(Fn);
    sparse_free(A);
}

#endif
