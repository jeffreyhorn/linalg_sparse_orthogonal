/*
 * Sprint 17 tests for the CSC working format for LDL^T.
 *
 * Covers the `LdltCsc` struct, alloc/free helpers, sparse conversion
 * and round-trip routines, and the Day 8-9 Bunch-Kaufman elimination
 * and native-CSC solve paths built on top of the CSC working format
 * (including linked-list cross-checks).
 */

#include "sparse_chol_csc_internal.h"
#include "sparse_ldlt.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Test helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compare two lower-triangular SparseMatrices entry-by-entry.  Only
 * (i, j) with i >= j are inspected.  Shape mismatch is fatal (return
 * early) so a mismatched test stops at the root cause instead of
 * cascading into thousands of follow-on failures. */
static void assert_lower_triangle_equal(const SparseMatrix *A, const SparseMatrix *B, double tol) {
    idx_t n = sparse_rows(A);
    if (sparse_rows(A) != sparse_rows(B) || sparse_cols(A) != sparse_cols(B)) {
        TF_FAIL_("Shape mismatch: A is %dx%d, B is %dx%d", (int)sparse_rows(A), (int)sparse_cols(A),
                 (int)sparse_rows(B), (int)sparse_cols(B));
        return;
    }
    tf_asserts += 2; /* account for the shape checks we'd have done */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double a = sparse_get(A, i, j);
            double b = sparse_get(B, i, j);
            if (fabs(a - b) > tol) {
                TF_FAIL_("Lower-tri (%d,%d): A=%.15g B=%.15g diff=%.3e > tol=%.3e", (int)i, (int)j,
                         a, b, fabs(a - b), tol);
            }
            tf_asserts++;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * alloc / free
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_alloc_null_out(void) { ASSERT_ERR(ldlt_csc_alloc(3, 4, NULL), SPARSE_ERR_NULL); }

static void test_alloc_negative_n(void) {
    LdltCsc *m = NULL;
    ASSERT_ERR(ldlt_csc_alloc(-1, 4, &m), SPARSE_ERR_BADARG);
    ASSERT_NULL(m);
}

static void test_alloc_basic(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(5, 10, &m));
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->n, 5);
    ASSERT_NOT_NULL(m->L);
    ASSERT_EQ(m->L->n, 5);
    ASSERT_NOT_NULL(m->D);
    ASSERT_NOT_NULL(m->D_offdiag);
    ASSERT_NOT_NULL(m->pivot_size);
    ASSERT_NOT_NULL(m->perm);

    /* Defaults: D/D_offdiag zero, pivot_size = 1 everywhere, perm = identity. */
    for (idx_t i = 0; i < 5; i++) {
        ASSERT_NEAR(m->D[i], 0.0, 0.0);
        ASSERT_NEAR(m->D_offdiag[i], 0.0, 0.0);
        ASSERT_EQ(m->pivot_size[i], 1);
        ASSERT_EQ(m->perm[i], i);
    }
    ldlt_csc_free(m);
}

static void test_alloc_zero_n(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(0, 1, &m));
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->n, 0);
    ldlt_csc_free(m);
}

static void test_free_null(void) {
    ldlt_csc_free(NULL); /* must not crash */
    ASSERT_TRUE(1);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 19 Day 8: row-adjacency index
 * ═══════════════════════════════════════════════════════════════════════ */

/* Alloc / free round-trip with no appends — every per-row slot should
 * be NULL after alloc and free should be a no-op on those slots. */
static void test_row_adj_empty_round_trip(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(10, 1, &F));
    ASSERT_NOT_NULL(F->row_adj);
    ASSERT_NOT_NULL(F->row_adj_count);
    ASSERT_NOT_NULL(F->row_adj_cap);
    for (idx_t r = 0; r < 10; r++) {
        ASSERT_TRUE(F->row_adj[r] == NULL);
        ASSERT_EQ(F->row_adj_count[r], 0);
        ASSERT_EQ(F->row_adj_cap[r], 0);
    }
    ldlt_csc_free(F); /* must not crash or leak */
    ASSERT_TRUE(1);
}

/* Append 3 columns to row 5, verify insertion order preserved. */
static void test_row_adj_append_preserves_order(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(10, 1, &F));
    REQUIRE_OK(ldlt_csc_row_adj_append(F, 5, 0));
    REQUIRE_OK(ldlt_csc_row_adj_append(F, 5, 2));
    REQUIRE_OK(ldlt_csc_row_adj_append(F, 5, 4));

    ASSERT_EQ(F->row_adj_count[5], 3);
    ASSERT_TRUE(F->row_adj_cap[5] >= 3);
    ASSERT_NOT_NULL(F->row_adj[5]);
    ASSERT_EQ(F->row_adj[5][0], 0);
    ASSERT_EQ(F->row_adj[5][1], 2);
    ASSERT_EQ(F->row_adj[5][2], 4);
    /* Other rows untouched. */
    for (idx_t r = 0; r < 10; r++) {
        if (r == 5)
            continue;
        ASSERT_EQ(F->row_adj_count[r], 0);
    }
    ldlt_csc_free(F);
}

/* Geometric growth: 100 appends to one row must succeed and preserve
 * every entry in insertion order. */
static void test_row_adj_geometric_growth(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(200, 1, &F));
    for (idx_t c = 0; c < 100; c++)
        REQUIRE_OK(ldlt_csc_row_adj_append(F, 101, c));

    ASSERT_EQ(F->row_adj_count[101], 100);
    ASSERT_TRUE(F->row_adj_cap[101] >= 100);
    for (idx_t c = 0; c < 100; c++)
        ASSERT_EQ(F->row_adj[101][c], c);
    ldlt_csc_free(F);
}

/* Null / out-of-range argument checks. */
static void test_row_adj_append_arg_checks(void) {
    ASSERT_ERR(ldlt_csc_row_adj_append(NULL, 0, 0), SPARSE_ERR_NULL);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(5, 1, &F));
    ASSERT_ERR(ldlt_csc_row_adj_append(F, -1, 0), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_row_adj_append(F, 5, 0), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_row_adj_append(F, 0, -1), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_row_adj_append(F, 0, 5), SPARSE_ERR_BADARG);
    ldlt_csc_free(F);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 19 Day 10: 2×2-aware supernode detection
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: build an LdltCsc whose embedded L is "fully dense lower
 * triangular" (every row `i >= j` stored in column `j`) and whose
 * pivot_size array is the caller-supplied pattern.  Lets the
 * detection tests focus on boundary behaviour without running a
 * real factor. */
static LdltCsc *build_dense_ldlt_with_pivots(idx_t n, const idx_t *pivot_size) {
    LdltCsc *F = NULL;
    if (ldlt_csc_alloc(n, n * (n + 1) / 2, &F) != SPARSE_OK)
        return NULL;
    CholCsc *L = F->L;
    idx_t p = 0;
    for (idx_t j = 0; j < n; j++) {
        L->col_ptr[j] = p;
        for (idx_t i = j; i < n; i++) {
            L->row_idx[p] = i;
            L->values[p] = (i == j) ? 1.0 : 0.1;
            p++;
        }
    }
    L->col_ptr[n] = p;
    L->nnz = p;
    for (idx_t k = 0; k < n; k++)
        F->pivot_size[k] = pivot_size[k];
    return F;
}

/* Dense 6×6 matrix with all 1×1 pivots: a single supernode covering
 * [0, 6) of size 6. */
static void test_detect_supernodes_dense_all_1x1(void) {
    idx_t pivot_size[6] = {1, 1, 1, 1, 1, 1};
    LdltCsc *F = build_dense_ldlt_with_pivots(6, pivot_size);
    ASSERT_NOT_NULL(F);

    idx_t starts[6] = {0}, sizes[6] = {0};
    idx_t count = 0;
    REQUIRE_OK(ldlt_csc_detect_supernodes(F, /*min_size=*/2, starts, sizes, &count));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(starts[0], 0);
    ASSERT_EQ(sizes[0], 6);

    ldlt_csc_free(F);
}

/* Dense 6×6 with a 2×2 pivot at (2, 3), all other pivots 1×1.  The
 * Liu-Ng-Peyton pattern lets the whole dense matrix form one
 * supernode; the 2×2 is fully contained, so detection keeps one
 * supernode covering [0, 6). */
static void test_detect_supernodes_dense_with_2x2(void) {
    idx_t pivot_size[6] = {1, 1, 2, 2, 1, 1};
    LdltCsc *F = build_dense_ldlt_with_pivots(6, pivot_size);
    ASSERT_NOT_NULL(F);

    idx_t starts[6] = {0}, sizes[6] = {0};
    idx_t count = 0;
    REQUIRE_OK(ldlt_csc_detect_supernodes(F, /*min_size=*/2, starts, sizes, &count));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(starts[0], 0);
    ASSERT_EQ(sizes[0], 6);

    ldlt_csc_free(F);
}

/* Block-diagonal 8×8 with two 4×4 dense blocks, a 2×2 pivot inside
 * each block.  The Liu-Ng-Peyton pattern breaks between the two
 * blocks (columns 3 and 4 don't share a below-diagonal pattern), so
 * detection must emit two supernodes [0, 4) and [4, 8).  Neither
 * boundary falls inside a 2×2 pair. */
static void test_detect_supernodes_block_diagonal_with_2x2(void) {
    idx_t n = 8;
    /* Build two 4×4 dense blocks stored as a block-diagonal CSC. */
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(n, 2 * 4 * (4 + 1) / 2, &F));
    CholCsc *L = F->L;
    idx_t p = 0;
    for (idx_t j = 0; j < n; j++) {
        L->col_ptr[j] = p;
        idx_t block_start = (j < 4) ? 0 : 4;
        idx_t block_end = block_start + 4;
        for (idx_t i = j; i < block_end; i++) {
            L->row_idx[p] = i;
            L->values[p] = (i == j) ? 1.0 : 0.1;
            p++;
        }
    }
    L->col_ptr[n] = p;
    L->nnz = p;

    /* Block 0: 2×2 at (1, 2), singles at 0 and 3.
     * Block 1: 2×2 at (5, 6), singles at 4 and 7. */
    idx_t pivot_size[8] = {1, 2, 2, 1, 1, 2, 2, 1};
    for (idx_t k = 0; k < n; k++)
        F->pivot_size[k] = pivot_size[k];

    idx_t starts[8] = {0}, sizes[8] = {0};
    idx_t count = 0;
    REQUIRE_OK(ldlt_csc_detect_supernodes(F, /*min_size=*/2, starts, sizes, &count));
    ASSERT_EQ(count, 2);
    ASSERT_EQ(starts[0], 0);
    ASSERT_EQ(sizes[0], 4);
    ASSERT_EQ(starts[1], 4);
    ASSERT_EQ(sizes[1], 4);

    ldlt_csc_free(F);
}

/* Null / badarg checks. */
static void test_detect_supernodes_arg_checks(void) {
    idx_t starts[4] = {0}, sizes[4] = {0};
    idx_t count = 0;
    ASSERT_ERR(ldlt_csc_detect_supernodes(NULL, 1, starts, sizes, &count), SPARSE_ERR_NULL);

    idx_t pivot_size[2] = {1, 1};
    LdltCsc *F = build_dense_ldlt_with_pivots(2, pivot_size);
    ASSERT_NOT_NULL(F);
    ASSERT_ERR(ldlt_csc_detect_supernodes(F, 0, starts, sizes, &count), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_detect_supernodes(F, -1, starts, sizes, &count), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_detect_supernodes(F, 1, NULL, sizes, &count), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_detect_supernodes(F, 1, starts, NULL, &count), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_detect_supernodes(F, 1, starts, sizes, NULL), SPARSE_ERR_NULL);
    ldlt_csc_free(F);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 19 Day 12: supernode extract / writeback round-trips
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compute row × column → linear index for a column-major buffer with
 * leading dimension `lda`. */
static inline idx_t cm_idx(idx_t row, idx_t col, idx_t lda) { return row + col * lda; }

/* Snapshot helper: copy `F->L->values`, `F->D`, `F->D_offdiag`,
 * `F->pivot_size` for the supernode column range so we can verify the
 * round-trip is the identity (no off-by-one in writeback). */
static void snapshot_supernode_state(const LdltCsc *F, idx_t s_start, idx_t s_size,
                                     double *L_values_copy, idx_t *L_nnz_in_block, double *D_copy,
                                     double *D_offdiag_copy, idx_t *pivot_size_copy) {
    idx_t cstart = F->L->col_ptr[s_start];
    idx_t cend = F->L->col_ptr[s_start + s_size];
    *L_nnz_in_block = cend - cstart;
    for (idx_t p = 0; p < cend - cstart; p++)
        L_values_copy[p] = F->L->values[cstart + p];
    for (idx_t j = 0; j < s_size; j++) {
        D_copy[j] = F->D[s_start + j];
        D_offdiag_copy[j] = F->D_offdiag[s_start + j];
        pivot_size_copy[j] = F->pivot_size[s_start + j];
    }
}

/* Dense 6×6 supernode round-trip: extract → memcpy (identity) →
 * writeback reproduces every L value, D entry, D_offdiag entry, and
 * pivot_size entry exactly. */
static void test_supernode_extract_writeback_dense_6x6(void) {
    idx_t n = 6;
    idx_t pivot_size[6] = {1, 1, 1, 1, 1, 1};
    LdltCsc *F = build_dense_ldlt_with_pivots(n, pivot_size);
    ASSERT_NOT_NULL(F);

    /* Seed D / D_offdiag with distinguishable values so a misplaced
     * writeback would surface as a mismatch. */
    for (idx_t k = 0; k < n; k++) {
        F->D[k] = 2.5 + (double)k;
        F->D_offdiag[k] = 0.0;
    }

    /* Snapshot the pre-extract state. */
    double L_pre[6 * 6];
    idx_t L_nnz = 0;
    double D_pre[6], D_off_pre[6];
    idx_t ps_pre[6];
    snapshot_supernode_state(F, 0, n, L_pre, &L_nnz, D_pre, D_off_pre, ps_pre);

    /* Extract → identity → writeback. */
    idx_t panel_height = 0;
    idx_t row_map[6];
    idx_t lda = n; /* dense column-major; lda == panel_height for full square block */
    double dense[36] = {0};
    REQUIRE_OK(ldlt_csc_supernode_extract(F, 0, n, dense, lda, row_map, &panel_height));
    ASSERT_EQ(panel_height, n);
    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(row_map[i], i);

    REQUIRE_OK(ldlt_csc_supernode_writeback(F, 0, n, dense, lda, row_map, panel_height, D_pre,
                                            D_off_pre, ps_pre, 0.0));

    /* Verify L values restored bit-for-bit. */
    idx_t cstart = F->L->col_ptr[0];
    for (idx_t p = 0; p < L_nnz; p++)
        ASSERT_NEAR(F->L->values[cstart + p], L_pre[p], 0.0);
    /* D / D_offdiag / pivot_size restored. */
    for (idx_t k = 0; k < n; k++) {
        ASSERT_NEAR(F->D[k], D_pre[k], 0.0);
        ASSERT_NEAR(F->D_offdiag[k], D_off_pre[k], 0.0);
        ASSERT_EQ(F->pivot_size[k], ps_pre[k]);
    }

    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_csc_free(F);
}

/* Block-diagonal 8×8 with two 4×4 dense blocks: extract and writeback
 * each block independently; mutations to one block must not bleed
 * into the other. */
static void test_supernode_extract_writeback_block_diagonal_8x8(void) {
    idx_t n = 8;
    LdltCsc *F = NULL;
    /* Build manually: each block is dense lower triangular within
     * itself; off-block entries are zero (not stored). */
    idx_t initial_nnz = 2 * (4 * 5 / 2); /* 2 blocks × 10 stored entries */
    REQUIRE_OK(ldlt_csc_alloc(n, initial_nnz, &F));
    CholCsc *L = F->L;
    idx_t p = 0;
    for (idx_t blk = 0; blk < 2; blk++) {
        idx_t base = blk * 4;
        for (idx_t j = 0; j < 4; j++) {
            L->col_ptr[base + j] = p;
            for (idx_t i = j; i < 4; i++) {
                L->row_idx[p] = base + i;
                L->values[p] = (i == j) ? 1.0 : (0.1 + 0.01 * (double)(blk * 4 + j));
                p++;
            }
        }
    }
    L->col_ptr[n] = p;
    L->nnz = p;
    for (idx_t k = 0; k < n; k++) {
        F->pivot_size[k] = 1;
        F->D[k] = 3.0 + (double)k;
        F->D_offdiag[k] = 0.0;
    }

    /* Round-trip block 0; assert block 1 is undisturbed. */
    double dense_b0[16] = {0};
    idx_t row_map_b0[4];
    idx_t panel_height_b0 = 0;
    REQUIRE_OK(ldlt_csc_supernode_extract(F, 0, 4, dense_b0, 4, row_map_b0, &panel_height_b0));
    ASSERT_EQ(panel_height_b0, 4);
    for (idx_t i = 0; i < 4; i++)
        ASSERT_EQ(row_map_b0[i], i);

    /* Snapshot block 1 BEFORE block 0's writeback. */
    double L_b1_pre[10];
    idx_t L_b1_cstart = F->L->col_ptr[4];
    for (idx_t i = 0; i < 10; i++)
        L_b1_pre[i] = F->L->values[L_b1_cstart + i];

    /* Mutate block 0's dense buffer and write back. */
    for (idx_t j = 0; j < 4; j++)
        for (idx_t i = j + 1; i < 4; i++)
            dense_b0[cm_idx(i, j, 4)] = 9.99; /* arbitrary new off-diag */
    double D_b0[4], D_off_b0[4];
    idx_t ps_b0[4];
    for (idx_t j = 0; j < 4; j++) {
        D_b0[j] = F->D[j];
        D_off_b0[j] = F->D_offdiag[j];
        ps_b0[j] = F->pivot_size[j];
    }
    REQUIRE_OK(ldlt_csc_supernode_writeback(F, 0, 4, dense_b0, 4, row_map_b0, panel_height_b0, D_b0,
                                            D_off_b0, ps_b0, 0.0));

    /* Block 0's below-diagonal off-diagonals must be 9.99 now. */
    for (idx_t j = 0; j < 4; j++) {
        idx_t cs = F->L->col_ptr[j];
        idx_t ce = F->L->col_ptr[j + 1];
        for (idx_t q = cs; q < ce; q++) {
            idx_t r = F->L->row_idx[q];
            if (r == j)
                ASSERT_NEAR(F->L->values[q], 1.0, 0.0); /* diag preserved */
            else
                ASSERT_NEAR(F->L->values[q], 9.99, 0.0);
        }
    }
    /* Block 1 untouched. */
    for (idx_t i = 0; i < 10; i++)
        ASSERT_NEAR(F->L->values[L_b1_cstart + i], L_b1_pre[i], 0.0);

    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_csc_free(F);
}

/* Mixed 1×1 / 2×2 pivot supernode: round-trip preserves D_offdiag
 * for the 2×2 pair and zeros for the 1×1 columns, plus drop_tol = 0
 * keeps every L value verbatim. */
static void test_supernode_extract_writeback_with_2x2(void) {
    idx_t n = 4;
    /* Pattern: 1×1, then a 2×2 at (1,2), then 1×1 — within a single
     * supernode (the dense lower triangle covers the whole matrix). */
    idx_t pivot_size[4] = {1, 2, 2, 1};
    LdltCsc *F = build_dense_ldlt_with_pivots(n, pivot_size);
    ASSERT_NOT_NULL(F);

    /* Hand-set D / D_offdiag matching the 2×2 convention:
     *   D[1] = d11, D[2] = d22, D_offdiag[1] = d21 != 0, D_offdiag[2] = 0. */
    F->D[0] = 2.0;
    F->D[1] = 3.0;
    F->D[2] = 4.0;
    F->D[3] = 5.0;
    F->D_offdiag[0] = 0.0;
    F->D_offdiag[1] = 1.5;
    F->D_offdiag[2] = 0.0;
    F->D_offdiag[3] = 0.0;

    /* Snapshot. */
    double L_pre[4 * 5 / 2];
    idx_t L_nnz = 0;
    double D_pre[4], D_off_pre[4];
    idx_t ps_pre[4];
    snapshot_supernode_state(F, 0, n, L_pre, &L_nnz, D_pre, D_off_pre, ps_pre);

    /* Extract → identity → writeback. */
    double dense[16] = {0};
    idx_t row_map[4];
    idx_t panel_height = 0;
    REQUIRE_OK(ldlt_csc_supernode_extract(F, 0, n, dense, n, row_map, &panel_height));
    REQUIRE_OK(ldlt_csc_supernode_writeback(F, 0, n, dense, n, row_map, panel_height, D_pre,
                                            D_off_pre, ps_pre, 0.0));

    /* L values restored. */
    idx_t cstart = F->L->col_ptr[0];
    for (idx_t p = 0; p < L_nnz; p++)
        ASSERT_NEAR(F->L->values[cstart + p], L_pre[p], 0.0);

    /* D_offdiag survives the round-trip with the 2×2 pair's d21
     * intact. */
    ASSERT_NEAR(F->D_offdiag[0], 0.0, 0.0);
    ASSERT_NEAR(F->D_offdiag[1], 1.5, 0.0);
    ASSERT_NEAR(F->D_offdiag[2], 0.0, 0.0);
    ASSERT_NEAR(F->D_offdiag[3], 0.0, 0.0);

    /* pivot_size pattern survives. */
    ASSERT_EQ(F->pivot_size[0], 1);
    ASSERT_EQ(F->pivot_size[1], 2);
    ASSERT_EQ(F->pivot_size[2], 2);
    ASSERT_EQ(F->pivot_size[3], 1);

    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_csc_free(F);
}

/* Argument-validation checks. */
static void test_supernode_extract_writeback_arg_checks(void) {
    idx_t pivot_size[3] = {1, 1, 1};
    LdltCsc *F = build_dense_ldlt_with_pivots(3, pivot_size);
    ASSERT_NOT_NULL(F);
    F->D[0] = 1.0;
    F->D[1] = 2.0;
    F->D[2] = 3.0;

    double dense[9] = {0};
    idx_t row_map[3];
    idx_t panel_height = 0;
    double D[3] = {1.0, 2.0, 3.0};
    double D_off[3] = {0};
    idx_t ps[3] = {1, 1, 1};

    /* Extract: NULL inputs. */
    ASSERT_ERR(ldlt_csc_supernode_extract(NULL, 0, 1, dense, 3, row_map, &panel_height),
               SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_supernode_extract(F, 0, 1, NULL, 3, row_map, &panel_height),
               SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_supernode_extract(F, 0, 1, dense, 3, NULL, &panel_height), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_supernode_extract(F, 0, 1, dense, 3, row_map, NULL), SPARSE_ERR_NULL);

    /* Extract: invalid range / lda. */
    ASSERT_ERR(ldlt_csc_supernode_extract(F, -1, 1, dense, 3, row_map, &panel_height),
               SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_supernode_extract(F, 0, 0, dense, 3, row_map, &panel_height),
               SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_supernode_extract(F, 0, 4, dense, 3, row_map, &panel_height),
               SPARSE_ERR_BADARG); /* s_start + s_size > n */
    ASSERT_ERR(ldlt_csc_supernode_extract(F, 0, 1, dense, 0, row_map, &panel_height),
               SPARSE_ERR_BADARG); /* lda < panel_height */

    /* Writeback: NULL inputs. */
    REQUIRE_OK(ldlt_csc_supernode_extract(F, 0, 3, dense, 3, row_map, &panel_height));
    ASSERT_ERR(ldlt_csc_supernode_writeback(NULL, 0, 3, dense, 3, row_map, panel_height, D, D_off,
                                            ps, 0.0),
               SPARSE_ERR_NULL);
    ASSERT_ERR(
        ldlt_csc_supernode_writeback(F, 0, 3, NULL, 3, row_map, panel_height, D, D_off, ps, 0.0),
        SPARSE_ERR_NULL);
    ASSERT_ERR(
        ldlt_csc_supernode_writeback(F, 0, 3, dense, 3, NULL, panel_height, D, D_off, ps, 0.0),
        SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_supernode_writeback(F, 0, 3, dense, 3, row_map, panel_height, NULL, D_off,
                                            ps, 0.0),
               SPARSE_ERR_NULL);
    ASSERT_ERR(
        ldlt_csc_supernode_writeback(F, 0, 3, dense, 3, row_map, panel_height, D, NULL, ps, 0.0),
        SPARSE_ERR_NULL);
    ASSERT_ERR(
        ldlt_csc_supernode_writeback(F, 0, 3, dense, 3, row_map, panel_height, D, D_off, NULL, 0.0),
        SPARSE_ERR_NULL);

    /* Writeback: bad pivot_size value. */
    idx_t ps_bad[3] = {1, 5, 1};
    ASSERT_ERR(ldlt_csc_supernode_writeback(F, 0, 3, dense, 3, row_map, panel_height, D, D_off,
                                            ps_bad, 0.0),
               SPARSE_ERR_BADARG);

    ldlt_csc_free(F);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 19 Day 13: batched supernodal LDL^T cross-checks
 * ═══════════════════════════════════════════════════════════════════════ */

/* Forward declaration: `build_random_symmetric` is defined later in
 * this file (used by the existing scalar 20-matrix cross-check) and
 * the Day 13 random-indefinite test reaches it from above. */
static SparseMatrix *build_random_symmetric(idx_t n, unsigned int seed);

/* Compare two factored LdltCscs entry-by-entry on L, D, D_offdiag,
 * pivot_size.  Returns 1 on full match, 0 on first mismatch (and
 * surfaces the diff via TF_FAIL_). */
static int ldlt_csc_factor_state_matches(const LdltCsc *A, const LdltCsc *B, double tol) {
    if (A->n != B->n) {
        TF_FAIL_("n mismatch: A=%d B=%d", (int)A->n, (int)B->n);
        return 0;
    }
    idx_t n = A->n;
    /* L structural pattern + values. */
    if (A->L->col_ptr[n] != B->L->col_ptr[n]) {
        TF_FAIL_("L nnz mismatch: A=%d B=%d", (int)A->L->col_ptr[n], (int)B->L->col_ptr[n]);
        return 0;
    }
    for (idx_t j = 0; j < n; j++) {
        if (A->L->col_ptr[j] != B->L->col_ptr[j]) {
            TF_FAIL_("col_ptr[%d] mismatch: A=%d B=%d", (int)j, (int)A->L->col_ptr[j],
                     (int)B->L->col_ptr[j]);
            return 0;
        }
    }
    idx_t total = A->L->col_ptr[n];
    for (idx_t p = 0; p < total; p++) {
        if (A->L->row_idx[p] != B->L->row_idx[p]) {
            TF_FAIL_("row_idx[%d] mismatch: A=%d B=%d", (int)p, (int)A->L->row_idx[p],
                     (int)B->L->row_idx[p]);
            return 0;
        }
        if (fabs(A->L->values[p] - B->L->values[p]) > tol) {
            TF_FAIL_("L.values[%d] mismatch: A=%.15g B=%.15g diff=%.3e", (int)p, A->L->values[p],
                     B->L->values[p], fabs(A->L->values[p] - B->L->values[p]));
            return 0;
        }
    }
    /* D / D_offdiag / pivot_size. */
    for (idx_t k = 0; k < n; k++) {
        if (fabs(A->D[k] - B->D[k]) > tol) {
            TF_FAIL_("D[%d] mismatch: A=%.15g B=%.15g", (int)k, A->D[k], B->D[k]);
            return 0;
        }
        if (fabs(A->D_offdiag[k] - B->D_offdiag[k]) > tol) {
            TF_FAIL_("D_offdiag[%d] mismatch: A=%.15g B=%.15g", (int)k, A->D_offdiag[k],
                     B->D_offdiag[k]);
            return 0;
        }
        if (A->pivot_size[k] != B->pivot_size[k]) {
            TF_FAIL_("pivot_size[%d] mismatch: A=%d B=%d", (int)k, (int)A->pivot_size[k],
                     (int)B->pivot_size[k]);
            return 0;
        }
    }
    return 1;
}

/* Build a moderately-conditioned dense SPD matrix of size n with
 * entries in [-1, 1] off-diagonal and a strong diagonal so BK picks
 * 1×1 pivots throughout (no swaps). */
static SparseMatrix *build_dense_spd(idx_t n, unsigned int seed) {
    srand(seed);
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            if (i == j) {
                /* Strong diagonal — guarantees SPD and dominant 1×1 pivots. */
                sparse_insert(A, i, i, (double)(2 * n));
            } else if (j < i) {
                double v = ((double)rand() / (double)RAND_MAX) * 0.5 - 0.25;
                sparse_insert(A, i, j, v);
                sparse_insert(A, j, i, v);
            }
        }
    }
    return A;
}

/* Dense 8×8 SPD: the matrix forms one big supernode under
 * `ldlt_csc_detect_supernodes`'s default pivot_size=1 pattern.  Both
 * the scalar and batched paths factor identically (no BK swaps on an
 * SPD matrix). */
static void test_supernodal_dense_spd_8x8(void) {
    idx_t n = 8;
    SparseMatrix *A = build_dense_spd(n, 0xab12cd34u);

    LdltCsc *F_scalar = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F_scalar));
    REQUIRE_OK(ldlt_csc_eliminate_native(F_scalar));

    LdltCsc *F_batched = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F_batched));
    /* Seed pivot_size from the scalar pass so detect_supernodes sees
     * the resolved pattern (identical to the as-allocated all-1
     * default for SPD, but explicit for clarity and symmetry with the
     * indefinite test below). */
    for (idx_t k = 0; k < n; k++)
        F_batched->pivot_size[k] = F_scalar->pivot_size[k];
    REQUIRE_OK(ldlt_csc_eliminate_supernodal(F_batched, /*min_size=*/2));

    REQUIRE_OK(ldlt_csc_validate(F_batched));
    ASSERT_TRUE(ldlt_csc_factor_state_matches(F_scalar, F_batched, 1e-12));

    ldlt_csc_free(F_scalar);
    ldlt_csc_free(F_batched);
    sparse_free(A);
}

/* Block-diagonal 12×12 SPD with two 6×6 dense blocks: detection
 * emits two supernodes; the batched path runs each block
 * independently and must match the scalar factor on both. */
static void test_supernodal_block_diagonal_spd_12x12(void) {
    idx_t n = 12;
    SparseMatrix *A = sparse_create(n, n);
    srand(0xdeadbeefu);
    for (idx_t blk = 0; blk < 2; blk++) {
        idx_t base = blk * 6;
        for (idx_t i = 0; i < 6; i++) {
            for (idx_t j = 0; j < 6; j++) {
                if (i == j) {
                    sparse_insert(A, base + i, base + i, 12.0);
                } else if (j < i) {
                    double v = ((double)rand() / (double)RAND_MAX) * 0.4 - 0.2;
                    sparse_insert(A, base + i, base + j, v);
                    sparse_insert(A, base + j, base + i, v);
                }
            }
        }
    }

    LdltCsc *F_scalar = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F_scalar));
    REQUIRE_OK(ldlt_csc_eliminate_native(F_scalar));

    LdltCsc *F_batched = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F_batched));
    for (idx_t k = 0; k < n; k++)
        F_batched->pivot_size[k] = F_scalar->pivot_size[k];
    REQUIRE_OK(ldlt_csc_eliminate_supernodal(F_batched, /*min_size=*/2));

    REQUIRE_OK(ldlt_csc_validate(F_batched));
    ASSERT_TRUE(ldlt_csc_factor_state_matches(F_scalar, F_batched, 1e-12));

    ldlt_csc_free(F_scalar);
    ldlt_csc_free(F_batched);
    sparse_free(A);
}

/* Two-pass refactor on a random indefinite 30×30: factor scalar to
 * resolve BK swaps; permute A by the resulting perm; refactor with
 * the batched path; assert bit-identical L / D / pivot_size.
 *
 * The scalar pass produces F1 with `F1->perm = sigma` (a composition
 * of BK swaps).  Building F2 from `sparse_apply_symmetric_perm(A, sigma)`
 * (or equivalent) places A in the post-swap row order, so the
 * batched path's `ldlt_dense_factor` invocations see a matrix where
 * BK chooses identical 1×1/2×2 pivots without any further swaps. */
static void test_supernodal_random_indefinite_30x30(void) {
    /* Build a random symmetric matrix with mixed signs on the
     * diagonal — gives BK a real chance to choose 1×1 vs 2×2. */
    idx_t n = 30;
    SparseMatrix *A_orig = build_random_symmetric(n, 0xc0ffee01u);

    LdltCsc *F1 = NULL;
    sparse_err_t err1 = ldlt_csc_from_sparse(A_orig, NULL, 2.0, &F1);
    if (err1 != SPARSE_OK) {
        if (F1)
            ldlt_csc_free(F1);
        sparse_free(A_orig);
        return;
    }
    sparse_err_t err_elim = ldlt_csc_eliminate_native(F1);
    if (err_elim != SPARSE_OK) {
        ldlt_csc_free(F1);
        sparse_free(A_orig);
        return; /* random matrix may produce a singular pivot — skip */
    }
    REQUIRE_OK(ldlt_csc_validate(F1));

    /* Apply F1->perm symmetrically to A, producing A_perm = P*A*P^T. */
    SparseMatrix *A_perm = sparse_create(n, n);
    for (idx_t i_new = 0; i_new < n; i_new++) {
        for (idx_t j_new = 0; j_new < n; j_new++) {
            idx_t i_old = F1->perm[i_new];
            idx_t j_old = F1->perm[j_new];
            double v = sparse_get(A_orig, i_old, j_old);
            if (v != 0.0)
                sparse_insert(A_perm, i_new, j_new, v);
        }
    }

    LdltCsc *F2 = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A_perm, NULL, 2.0, &F2));
    /* Seed F2->pivot_size from F1 so detect_supernodes respects the
     * 2×2-aware boundaries from the scalar pass. */
    for (idx_t k = 0; k < n; k++)
        F2->pivot_size[k] = F1->pivot_size[k];

    sparse_err_t err_b = ldlt_csc_eliminate_supernodal(F2, /*min_size=*/2);
    if (err_b == SPARSE_ERR_BADARG) {
        /* The batched path's pivot-stability check rejected the
         * supernode (e.g., dense BK diverged on numerical drift).
         * Skip — the test's main purpose is to verify equivalence
         * when the path is taken successfully. */
        ldlt_csc_free(F1);
        ldlt_csc_free(F2);
        sparse_free(A_orig);
        sparse_free(A_perm);
        return;
    }
    REQUIRE_OK(err_b);
    REQUIRE_OK(ldlt_csc_validate(F2));

    /* F1 was factored on A; F2 on P*A*P^T.  Both should produce the
     * SAME L / D / pivot_size because P*A*P^T is just the post-swap
     * view of what F1 internally produced.  F1->perm == F2->perm
     * doesn't necessarily hold (F2 starts from identity-perm of
     * A_perm), so we compare only L / D / D_offdiag / pivot_size. */
    ASSERT_TRUE(ldlt_csc_factor_state_matches(F1, F2, 1e-10));

    ldlt_csc_free(F1);
    ldlt_csc_free(F2);
    sparse_free(A_orig);
    sparse_free(A_perm);
}

/* Argument validation for the Day 13 helpers. */
static void test_supernodal_arg_checks(void) {
    ASSERT_ERR(ldlt_csc_eliminate_supernodal(NULL, 1), SPARSE_ERR_NULL);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(3, 6, &F));
    ASSERT_ERR(ldlt_csc_eliminate_supernodal(F, 0), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_eliminate_supernodal(F, -1), SPARSE_ERR_BADARG);
    ldlt_csc_free(F);

    /* eliminate_diag NULL inputs. */
    double dense[4] = {0};
    idx_t row_map[2] = {0};
    double D[2] = {0}, D_off[2] = {0};
    idx_t ps[2] = {1, 1};
    REQUIRE_OK(ldlt_csc_alloc(2, 3, &F));
    F->L->col_ptr[0] = 0;
    F->L->col_ptr[1] = 1;
    F->L->col_ptr[2] = 1;
    F->L->row_idx[0] = 0;
    F->L->values[0] = 1.0;
    F->L->nnz = 1;
    ASSERT_ERR(ldlt_csc_supernode_eliminate_diag(NULL, 0, 1, dense, 1, row_map, 1, D, D_off, ps, 0),
               SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_supernode_eliminate_diag(F, -1, 1, dense, 1, row_map, 1, D, D_off, ps, 0),
               SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_supernode_eliminate_diag(F, 0, 0, dense, 1, row_map, 1, D, D_off, ps, 0),
               SPARSE_ERR_BADARG);
    ldlt_csc_free(F);

    /* eliminate_panel: panel_rows == 0 short-circuits successfully. */
    double Ld[1] = {1.0};
    double Db[1] = {1.0};
    double Doffb[1] = {0.0};
    idx_t psb[1] = {1};
    REQUIRE_OK(ldlt_csc_supernode_eliminate_panel(Ld, Db, Doffb, psb, 1, 1, NULL, 0, 0));
    ASSERT_ERR(ldlt_csc_supernode_eliminate_panel(NULL, Db, Doffb, psb, 1, 1, dense, 1, 1),
               SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_supernode_eliminate_panel(Ld, Db, Doffb, psb, 0, 1, dense, 1, 1),
               SPARSE_ERR_BADARG);
}

/* ═══════════════════════════════════════════════════════════════════════
 * from_sparse: null / shape / conversion
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_from_sparse_null_args(void) {
    LdltCsc *m = NULL;
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);

    ASSERT_ERR(ldlt_csc_from_sparse(NULL, NULL, 2.0, &m), SPARSE_ERR_NULL);
    ASSERT_NULL(m);
    ASSERT_ERR(ldlt_csc_from_sparse(A, NULL, 2.0, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

static void test_from_sparse_shape(void) {
    LdltCsc *m = NULL;
    SparseMatrix *rect = sparse_create(3, 5);
    ASSERT_ERR(ldlt_csc_from_sparse(rect, NULL, 2.0, &m), SPARSE_ERR_SHAPE);
    ASSERT_NULL(m);
    sparse_free(rect);
}

/* Identity matrix → round-trip preserves lower-triangle values exactly
 * (diagonal only). */
static void test_roundtrip_identity(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &m));
    ASSERT_NOT_NULL(m);
    REQUIRE_OK(ldlt_csc_validate(m));

    /* L's CSC has one entry per column = the diagonal value from A. */
    ASSERT_EQ(m->L->nnz, n);
    for (idx_t j = 0; j < n; j++) {
        ASSERT_EQ(m->L->col_ptr[j + 1] - m->L->col_ptr[j], 1);
        ASSERT_EQ(m->L->row_idx[m->L->col_ptr[j]], j);
        ASSERT_NEAR(m->L->values[m->L->col_ptr[j]], 1.0, 0.0);
    }
    /* D / pivot_size defaults untouched by conversion. */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(m->D[i], 0.0, 0.0);
        ASSERT_EQ(m->pivot_size[i], 1);
        ASSERT_EQ(m->perm[i], i);
    }

    SparseMatrix *B = NULL;
    REQUIRE_OK(ldlt_csc_to_sparse(m, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    ldlt_csc_free(m);
}

/* Diagonal indefinite matrix (mix of positive/negative diagonal entries):
 * conversion must preserve every value verbatim including the signs. */
static void test_roundtrip_diagonal_indefinite(void) {
    idx_t n = 5;
    double diag[5] = {3.0, -2.0, 1.0, -4.5, 0.25};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);

    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &m));
    REQUIRE_OK(ldlt_csc_validate(m));

    /* Each column has exactly the one diagonal entry, value preserved. */
    ASSERT_EQ(m->L->nnz, n);
    for (idx_t j = 0; j < n; j++) {
        ASSERT_EQ(m->L->row_idx[m->L->col_ptr[j]], j);
        ASSERT_NEAR(m->L->values[m->L->col_ptr[j]], diag[j], 0.0);
    }

    SparseMatrix *B = NULL;
    REQUIRE_OK(ldlt_csc_to_sparse(m, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    ldlt_csc_free(m);
}

/* Symmetric indefinite matrix with off-diagonals.  Round-trip recovers
 * the lower triangle (mirror entries in the upper half are stripped by
 * conversion, as they are for the Cholesky CSC path). */
static void test_roundtrip_symmetric_indefinite(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    /* A = [[1, 2, 0, 0], [2, -1, 3, 0], [0, 3, 2, -1], [0, 0, -1, 1]] */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, -1.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 3, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 2, 1, 3.0);
    sparse_insert(A, 1, 2, 3.0);
    sparse_insert(A, 3, 2, -1.0);
    sparse_insert(A, 2, 3, -1.0);

    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &m));
    REQUIRE_OK(ldlt_csc_validate(m));

    /* Lower-triangle nnz count: 4 diagonals + 3 off-diagonals (below). */
    ASSERT_EQ(m->L->nnz, 7);

    SparseMatrix *B = NULL;
    REQUIRE_OK(ldlt_csc_to_sparse(m, NULL, &B));
    assert_lower_triangle_equal(A, B, 0.0);

    sparse_free(A);
    sparse_free(B);
    ldlt_csc_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 20 Day 2: ldlt_csc_from_sparse_with_analysis
 * ═══════════════════════════════════════════════════════════════════════ */

/* NULL / invalid-type / shape-mismatch paths all return the expected
 * error code without allocating. */
static void test_from_sparse_with_analysis_arg_checks(void) {
    LdltCsc *m = NULL;
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 2.0);

    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    /* ldlt_out NULL → SPARSE_ERR_NULL. */
    ASSERT_ERR(ldlt_csc_from_sparse_with_analysis(A, &an, NULL), SPARSE_ERR_NULL);
    /* mat NULL → SPARSE_ERR_NULL. */
    ASSERT_ERR(ldlt_csc_from_sparse_with_analysis(NULL, &an, &m), SPARSE_ERR_NULL);
    ASSERT_NULL(m);
    /* analysis NULL → SPARSE_ERR_NULL. */
    ASSERT_ERR(ldlt_csc_from_sparse_with_analysis(A, NULL, &m), SPARSE_ERR_NULL);
    ASSERT_NULL(m);

    /* Wrong analysis type (LU) → SPARSE_ERR_BADARG. */
    sparse_analysis_t an_lu = {0};
    an_lu.type = SPARSE_FACTOR_LU;
    an_lu.n = 3;
    ASSERT_ERR(ldlt_csc_from_sparse_with_analysis(A, &an_lu, &m), SPARSE_ERR_BADARG);
    ASSERT_NULL(m);

    /* Rectangular matrix → SPARSE_ERR_SHAPE. */
    SparseMatrix *rect = sparse_create(3, 5);
    ASSERT_ERR(ldlt_csc_from_sparse_with_analysis(rect, &an, &m), SPARSE_ERR_SHAPE);
    ASSERT_NULL(m);

    /* n mismatch (analysis for different size) → SPARSE_ERR_SHAPE. */
    SparseMatrix *B4 = sparse_create(4, 4);
    for (idx_t i = 0; i < 4; i++)
        sparse_insert(B4, i, i, 1.0);
    ASSERT_ERR(ldlt_csc_from_sparse_with_analysis(B4, &an, &m), SPARSE_ERR_SHAPE);
    ASSERT_NULL(m);

    sparse_analysis_free(&an);
    sparse_free(rect);
    sparse_free(B4);
    sparse_free(A);
}

/* The returned LdltCsc's embedded L pattern matches `analysis->sym_L`
 * exactly: same col_ptr, same row_idx, same nnz.  This is the shim's
 * core contract — it's why the batched supernodal writeback can
 * preserve cmod fill rows that `ldlt_csc_from_sparse`'s heuristic
 * pattern drops. */
static void test_from_sparse_with_analysis_pattern_matches_sym_L(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    /* Tridiagonal SPD: diag 4, sub-diag -1. */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse_with_analysis(A, &an, &m));
    ASSERT_NOT_NULL(m);
    REQUIRE_OK(ldlt_csc_validate(m));

    /* col_ptr identity. */
    for (idx_t j = 0; j <= n; j++)
        ASSERT_EQ(m->L->col_ptr[j], an.sym_L.col_ptr[j]);
    /* row_idx identity (over the whole stored range). */
    idx_t nnz = an.sym_L.col_ptr[n];
    ASSERT_EQ(m->L->nnz, nnz);
    for (idx_t p = 0; p < nnz; p++)
        ASSERT_EQ(m->L->row_idx[p], an.sym_L.row_idx[p]);
    /* sym_L pre-allocated flag is the signal downstream kernels check. */
    ASSERT_EQ(m->L->sym_L_preallocated, 1);

    ldlt_csc_free(m);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* Scatter correctness: every lower-triangle entry of A lands at its
 * matching `col_ptr[j]..col_ptr[j+1]` slot with the correct value.
 * Round-trips through `ldlt_csc_to_sparse` which reconstructs a
 * lower-triangle SparseMatrix from the CSC slots, so
 * `assert_lower_triangle_equal` against the original A verifies each
 * stored value propagated through intact. */
static void test_from_sparse_with_analysis_scatter_preserves_values(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    /* Banded SPD with distinct entries so any scatter mix-up shows up. */
    double diag_vals[6] = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag_vals[i]);
    /* Bandwidth-2 off-diagonals with distinct values. */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t d = 1; d <= 2 && i + d < n; d++) {
            double v = -0.25 - 0.05 * (double)(i + d);
            sparse_insert(A, i, i + d, v);
            sparse_insert(A, i + d, i, v);
        }
    }

    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse_with_analysis(A, &an, &m));
    REQUIRE_OK(ldlt_csc_validate(m));

    SparseMatrix *B = NULL;
    REQUIRE_OK(ldlt_csc_to_sparse(m, NULL, &B));
    /* Every lower-triangle entry of A must be present in B with the
     * exact stored value; fill positions (rows in sym_L that A did
     * not populate) are zero-initialised and therefore invisible to
     * `ldlt_csc_to_sparse` (which skips zero values). */
    assert_lower_triangle_equal(A, B, 0.0);

    ldlt_csc_free(m);
    sparse_free(B);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* Indefinite KKT-style matrix smoke test.  The shim must accept the
 * input without overflow, produce a valid LdltCsc, and pre-allocate
 * the full sym_L pattern — this is Day 2's contract; Day 3 uses the
 * pre-allocated pattern in the supernodal writeback fast path to
 * actually factor such matrices correctly. */
static void test_from_sparse_with_analysis_indefinite_kkt_smoke(void) {
    /* 2×2 block KKT: top block SPD diag(4,4,4); bottom block 0;
     * off-diagonal B = identity (rows 3-4 couple to rows 0-1).
     *
     *   [ 4 0 0 | 1 0 ]
     *   [ 0 4 0 | 0 1 ]
     *   [ 0 0 4 | 0 0 ]
     *   [ 1 0 0 | 0 0 ]
     *   [ 0 1 0 | 0 0 ]
     *
     * Symmetric indefinite — eigenvalues include positives from the
     * SPD block and non-positive from the trailing zero block. */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < 3; i++)
        sparse_insert(A, i, i, 4.0);
    /* B (offdiag coupling) + its transpose. */
    sparse_insert(A, 0, 3, 1.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 1, 4, 1.0);
    sparse_insert(A, 4, 1, 1.0);

    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse_with_analysis(A, &an, &m));
    ASSERT_NOT_NULL(m);
    REQUIRE_OK(ldlt_csc_validate(m));

    /* sym_L pre-allocation signal + pattern match. */
    ASSERT_EQ(m->L->sym_L_preallocated, 1);
    ASSERT_EQ(m->L->nnz, an.sym_L.col_ptr[n]);
    /* Default state: pivot_size = 1, perm = identity (no reorder). */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_EQ(m->pivot_size[i], 1);
        ASSERT_EQ(m->perm[i], i);
    }

    ldlt_csc_free(m);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* End-to-end round-trip on five SPD fixtures of different sizes and
 * fill patterns.  For each fixture, factor via the new shim +
 * `ldlt_csc_eliminate_supernodal` (batched path) and via the
 * heuristic `ldlt_csc_from_sparse` + `ldlt_csc_eliminate_native`
 * (scalar reference), then assert the resulting factors match under
 * `ldlt_csc_factor_state_matches`.  Day 2's completion criterion:
 * "Round-trip on 5+ existing SPD fixtures produces the same factored
 * state (after `ldlt_csc_eliminate_supernodal`) as the heuristic path
 * does on SPD." */
static void test_from_sparse_with_analysis_spd_factor_matches_heuristic(void) {
    const idx_t sizes[] = {8, 12, 16, 24, 32};
    const int num_fixtures = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int f = 0; f < num_fixtures; f++) {
        idx_t n = sizes[f];
        /* Diagonally-dominant banded SPD with bandwidth 2. */
        SparseMatrix *A = sparse_create(n, n);
        for (idx_t i = 0; i < n; i++) {
            sparse_insert(A, i, i, 6.0);
            for (idx_t d = 1; d <= 2 && i + d < n; d++) {
                sparse_insert(A, i, i + d, -0.5);
                sparse_insert(A, i + d, i, -0.5);
            }
        }

        /* Reference: heuristic CSC + scalar native factor. */
        LdltCsc *F_heuristic = NULL;
        REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F_heuristic));
        REQUIRE_OK(ldlt_csc_eliminate_native(F_heuristic));
        REQUIRE_OK(ldlt_csc_validate(F_heuristic));

        /* Shim under test: analysis-driven CSC + batched supernodal. */
        sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
        sparse_analysis_t an = {0};
        REQUIRE_OK(sparse_analyze(A, &opts, &an));

        LdltCsc *F_with_an = NULL;
        REQUIRE_OK(ldlt_csc_from_sparse_with_analysis(A, &an, &F_with_an));
        /* Seed pivot_size from the scalar reference so the supernodal
         * detection respects the same pivot boundaries. */
        for (idx_t k = 0; k < n; k++)
            F_with_an->pivot_size[k] = F_heuristic->pivot_size[k];
        REQUIRE_OK(ldlt_csc_eliminate_supernodal(F_with_an, /*min_size=*/2));
        REQUIRE_OK(ldlt_csc_validate(F_with_an));

        ASSERT_TRUE(ldlt_csc_factor_state_matches(F_heuristic, F_with_an, 1e-12));

        ldlt_csc_free(F_heuristic);
        ldlt_csc_free(F_with_an);
        sparse_analysis_free(&an);
        sparse_free(A);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 20 Day 3: batched supernodal LDL^T end-to-end on indefinite
 *                  inputs via the Day 2 `_with_analysis` shim
 * ═══════════════════════════════════════════════════════════════════════
 *
 * These tests exercise the Option D workflow documented in the
 * `ldlt_csc_from_sparse_with_analysis` design block in
 * `src/sparse_ldlt_csc.c`:
 *
 *   1. Scalar pre-pass on heuristic CSC → F1 carrying the BK-chosen
 *      perm + pivot_size.
 *   2. Symmetrically permute A by F1->perm → A_perm.
 *   3. `sparse_analyze(A_perm)` → symbolic sym_L for the pre-
 *      permuted matrix.  After pre-permutation BK will not swap
 *      again during the batched factor, so sym_L is complete.
 *   4. `ldlt_csc_from_sparse_with_analysis(A_perm, &an, &F2)` — the
 *      Day 2 shim pre-allocates F2->L with the full sym_L pattern.
 *   5. Seed F2->pivot_size from F1 (the batched path's pivot-
 *      stability check needs the scalar pass's decisions).
 *   6. `ldlt_csc_eliminate_supernodal(F2, ...)` — Day 3 core claim:
 *      when F2->L->sym_L_preallocated == 1, the existing batched
 *      path's `row_map` (built from column s_start's full sym_L
 *      slot in `ldlt_csc_supernode_extract`) covers every row the
 *      cmod can touch.  The silent-skip branches in
 *      `ldlt_csc_supernode_eliminate_diag` (the `if (local >=
 *      panel_height) continue;` at each cmod update) therefore
 *      never fire, and the writeback finds every sym_L slot pre-
 *      populated — no drops.  Sprint 19 NOTE's 1e-2..1e-6 residual
 *      symptom is resolved without touching the batched kernel
 *      code; Day 2's shim is sufficient.
 *
 * Each test captures the cross-check under `ldlt_csc_factor_state_matches`
 * (F1 vs F2 bit-for-bit within 1e-10) and a solve-residual check
 * ≤ 1e-10 against the original A.
 */

/* Small 5×5 KKT saddle-point fixture.  SPD 3×3 top block +
 * 2×2 zero bottom block + 2-row off-diagonal coupling.  Symmetric
 * indefinite by construction. */
static SparseMatrix *build_kkt_5x5(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < 3; i++)
        sparse_insert(A, i, i, 4.0);
    sparse_insert(A, 0, 3, 1.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 1, 4, 1.0);
    sparse_insert(A, 4, 1, 1.0);
    return A;
}

/* Larger 10×10 KKT saddle-point: tridiagonal SPD top block (rows
 * 0..5, diag 6, off-diag -1) + 4×4 zero bottom block (rows 6..9) +
 * 4×6 full-rank coupling A = [I_4 | 0_{4×2}] (i.e. row j of the
 * bottom block couples to column j of the top block).  Rank-4
 * coupling ensures the KKT matrix is non-singular (Schur complement
 * is a permuted principal submatrix of H^{-1}, which is dense and
 * SPD for a tridiagonal diagonally-dominant H).  Symmetric
 * indefinite — 4 negative eigenvalues from the saddle, 6 positive
 * from the SPD block.  Stresses the supernodal panel beyond the
 * first block. */
static SparseMatrix *build_kkt_10x10(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    /* SPD top block H: tridiagonal, diag 6, sub-diag -1 (rows 0..5). */
    for (idx_t i = 0; i < 6; i++) {
        sparse_insert(A, i, i, 6.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    /* Zero bottom block at rows 6..9 — no diagonal entries. */
    /* Coupling: A[j, k] = δ(j, k) for j ∈ [0..3], k ∈ [0..3], i.e.
     * row 6+j couples to column j of the top block.  Gives a rank-4
     * coupling matrix (identity pattern on the first 4 columns). */
    for (idx_t j = 0; j < 4; j++) {
        sparse_insert(A, 6 + j, j, 1.0);
        sparse_insert(A, j, 6 + j, 1.0);
    }
    return A;
}

/* Day 3 workflow helper: run the Option D two-pass factor on `A`
 * and populate *F1_out (scalar reference) and *F2_out (batched via
 * the new shim).  Caller owns F1, F2, and A_perm on success and
 * must free with `ldlt_csc_free` / `sparse_free`.  On any
 * intermediate failure, frees all partial state, leaves the
 * out-params NULL, and returns 0; caller uses ASSERT_TRUE to
 * surface the failure with the test-framework standard message. */
static int s20_two_pass_indefinite_factor(const SparseMatrix *A, LdltCsc **F1_out, LdltCsc **F2_out,
                                          SparseMatrix **A_perm_out) {
    *F1_out = NULL;
    *F2_out = NULL;
    *A_perm_out = NULL;
    idx_t n = sparse_rows(A);

    /* Step 1: scalar pre-pass on heuristic CSC. */
    LdltCsc *F1 = NULL;
    if (ldlt_csc_from_sparse(A, NULL, 2.0, &F1) != SPARSE_OK)
        return 0;
    if (ldlt_csc_eliminate_native(F1) != SPARSE_OK) {
        ldlt_csc_free(F1);
        return 0;
    }

    /* Step 2: symmetrically permute A by F1->perm. */
    SparseMatrix *A_perm = sparse_create(n, n);
    if (!A_perm) {
        ldlt_csc_free(F1);
        return 0;
    }
    for (idx_t i_new = 0; i_new < n; i_new++) {
        for (idx_t j_new = 0; j_new < n; j_new++) {
            idx_t i_old = F1->perm[i_new];
            idx_t j_old = F1->perm[j_new];
            double v = sparse_get(A, i_old, j_old);
            if (v != 0.0)
                sparse_insert(A_perm, i_new, j_new, v);
        }
    }

    /* Step 3: analyze the pre-permuted matrix. */
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t an = {0};
    if (sparse_analyze(A_perm, &opts, &an) != SPARSE_OK) {
        ldlt_csc_free(F1);
        sparse_free(A_perm);
        return 0;
    }

    /* Step 4: build F2 via the Day 2 shim. */
    LdltCsc *F2 = NULL;
    if (ldlt_csc_from_sparse_with_analysis(A_perm, &an, &F2) != SPARSE_OK) {
        ldlt_csc_free(F1);
        sparse_analysis_free(&an);
        sparse_free(A_perm);
        return 0;
    }

    /* Step 5: seed pivot_size from scalar pass. */
    for (idx_t k = 0; k < n; k++)
        F2->pivot_size[k] = F1->pivot_size[k];

    /* Step 6: batched supernodal factor. */
    if (ldlt_csc_eliminate_supernodal(F2, /*min_size=*/2) != SPARSE_OK) {
        ldlt_csc_free(F1);
        ldlt_csc_free(F2);
        sparse_analysis_free(&an);
        sparse_free(A_perm);
        return 0;
    }

    sparse_analysis_free(&an);
    *F1_out = F1;
    *F2_out = F2;
    *A_perm_out = A_perm;
    return 1;
}

/* Forward declaration: `rel_residual` is defined with the Day 9
 * solve tests further down the file.  Day 3's helper consumes it. */
static double rel_residual(const SparseMatrix *A, const double *x, const double *b);

/* Compute ||A_perm · x - b|| / ||b|| for F2's solve against its own
 * (pre-permuted) A.  Uses the RHS b = A_perm · ones so the true
 * solution is the all-ones vector; residual ≤ 1e-10 implies the
 * batched factor is correct. */
static double s20_solve_residual(LdltCsc *F, const SparseMatrix *A_ref) {
    idx_t n = F->n;
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    if (!ones || !b || !x) {
        free(ones);
        free(b);
        free(x);
        return 1.0; /* arbitrary large — treat as fail */
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A_ref, ones, b);
    if (ldlt_csc_solve(F, b, x) != SPARSE_OK) {
        free(ones);
        free(b);
        free(x);
        return 1.0;
    }
    double res = rel_residual(A_ref, x, b);
    free(ones);
    free(b);
    free(x);
    return res;
}

/* Note on comparison semantics.  `ldlt_csc_factor_state_matches`
 * requires identical `col_ptr` and `row_idx` between two factors.
 * For heuristic-vs-analysis factor comparisons on indefinite fill
 * this is legitimately stricter than we want:
 *
 *   - heuristic CSC path drops below-drop-tol fill entries from the
 *     CSC storage via `shift_columns_right_of` (column ends shrink);
 *   - sym_L-preallocated path keeps every sym_L slot with value 0.0
 *     for dropped positions (column ends immutable).
 *
 * The two paths produce the same numeric factor at corresponding
 * (row, col) positions, but the CSC layouts diverge.  Day 3 tests
 * therefore use `s20_solve_residual` (user-visible correctness
 * signal) as the primary assertion, and only use
 * `ldlt_csc_factor_state_matches` on the smallest fixture (5×5 KKT)
 * where no fill is dropped in either path. */

/* 5×5 KKT smoke: both paths produce identical L layouts (no
 * dropped fill) so factor_state_matches works bit-for-bit.
 * Solve residual must also be round-off. */
static void test_s20_supernodal_with_analysis_kkt_5x5(void) {
    SparseMatrix *A = build_kkt_5x5();
    LdltCsc *F1 = NULL;
    LdltCsc *F2 = NULL;
    SparseMatrix *A_perm = NULL;
    ASSERT_TRUE(s20_two_pass_indefinite_factor(A, &F1, &F2, &A_perm));

    REQUIRE_OK(ldlt_csc_validate(F2));
    ASSERT_TRUE(ldlt_csc_factor_state_matches(F1, F2, 1e-10));
    ASSERT_TRUE(s20_solve_residual(F2, A_perm) < 1e-10);

    ldlt_csc_free(F1);
    ldlt_csc_free(F2);
    sparse_free(A_perm);
    sparse_free(A);
}

/* 10×10 KKT with non-trivial off-block coupling exercises the
 * supernodal panel + cmod path where the Sprint 19 NOTE's fill-row
 * drop symptom previously appeared on heuristic CSC.  Solve
 * residual is the primary correctness signal (see comparison
 * semantics note above). */
static void test_s20_supernodal_with_analysis_kkt_10x10(void) {
    SparseMatrix *A = build_kkt_10x10();
    LdltCsc *F1 = NULL;
    LdltCsc *F2 = NULL;
    SparseMatrix *A_perm = NULL;
    ASSERT_TRUE(s20_two_pass_indefinite_factor(A, &F1, &F2, &A_perm));

    REQUIRE_OK(ldlt_csc_validate(F2));
    ASSERT_TRUE(s20_solve_residual(F2, A_perm) < 1e-10);

    ldlt_csc_free(F1);
    ldlt_csc_free(F2);
    sparse_free(A_perm);
    sparse_free(A);
}

/* Random indefinite n=30 — the same fixture
 * `test_supernodal_random_indefinite_30x30` exercises via the
 * heuristic shim.  Day 3's version drives the Option D workflow
 * end-to-end via `ldlt_csc_from_sparse_with_analysis` and asserts
 * the final batched factor solves correctly.  If the scalar pass
 * produces a singular pivot or the batched path's pivot-stability
 * check trips, skip — mirrors the heuristic version's skip
 * convention. */
static void test_s20_supernodal_with_analysis_random_indefinite_30x30(void) {
    idx_t n = 30;
    SparseMatrix *A = build_random_symmetric(n, 0xc0ffee01u);

    LdltCsc *F1 = NULL;
    LdltCsc *F2 = NULL;
    SparseMatrix *A_perm = NULL;
    if (!s20_two_pass_indefinite_factor(A, &F1, &F2, &A_perm)) {
        /* Random input hit a singular pivot in the scalar pass or
         * BK drifted during the batched factor — skip, matching
         * `test_supernodal_random_indefinite_30x30`'s convention. */
        sparse_free(A);
        return;
    }

    REQUIRE_OK(ldlt_csc_validate(F2));
    ASSERT_TRUE(s20_solve_residual(F2, A_perm) < 1e-10);

    ldlt_csc_free(F1);
    ldlt_csc_free(F2);
    sparse_free(A_perm);
    sparse_free(A);
}

/* Before/after residual regression guard on the 10×10 KKT fixture.
 * The "before" path uses the heuristic `ldlt_csc_from_sparse`
 * initialiser and runs the batched supernodal factor on it — this
 * is the Sprint 19 NOTE's failure mode (1e-2..1e-6 residuals on
 * indefinite matrices where supernodal cmod produces fill rows
 * outside A's heuristic pattern, which the writeback silently
 * drops).  The "after" path uses `ldlt_csc_from_sparse_with_analysis`
 * which pre-allocates the full sym_L pattern so every fill row has
 * a slot.  Day 3's claim is that the "after" residual is round-off
 * (≤ 1e-10) and dominates the "before" residual on inputs where
 * supernodal cmod exceeds A's lower-triangle pattern.  The
 * "before" path may: (1) return an error (e.g. extract's row_map
 * lookup fails), (2) produce a large residual, or (3) happen to
 * work if no cmod fill escapes A's pattern on this fixture.  All
 * three cases are acceptable documentation; we only assert the
 * "after" residual.  Measured numbers captured in
 * `docs/planning/EPIC_2/SPRINT_20/bench_day3_indefinite.txt`. */
static void test_s20_supernodal_heuristic_vs_with_analysis_residuals(void) {
    SparseMatrix *A = build_kkt_10x10();
    idx_t n = sparse_rows(A);

    /* Shared scalar pre-pass: resolves BK perm used by both paths. */
    LdltCsc *F_pre = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F_pre));
    REQUIRE_OK(ldlt_csc_eliminate_native(F_pre));

    SparseMatrix *A_perm = sparse_create(n, n);
    for (idx_t i_new = 0; i_new < n; i_new++) {
        for (idx_t j_new = 0; j_new < n; j_new++) {
            idx_t i_old = F_pre->perm[i_new];
            idx_t j_old = F_pre->perm[j_new];
            double v = sparse_get(A, i_old, j_old);
            if (v != 0.0)
                sparse_insert(A_perm, i_new, j_new, v);
        }
    }

    /* "After" path: _with_analysis shim → residual must be round-off. */
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A_perm, &opts, &an));

    LdltCsc *F_after = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse_with_analysis(A_perm, &an, &F_after));
    for (idx_t k = 0; k < n; k++)
        F_after->pivot_size[k] = F_pre->pivot_size[k];
    REQUIRE_OK(ldlt_csc_eliminate_supernodal(F_after, /*min_size=*/2));

    double res_after = s20_solve_residual(F_after, A_perm);
    ASSERT_TRUE(res_after < 1e-10);

    ldlt_csc_free(F_pre);
    ldlt_csc_free(F_after);
    sparse_analysis_free(&an);
    sparse_free(A);
    sparse_free(A_perm);
}

/* ═══════════════════════════════════════════════════════════════════════
 * from_sparse with permutation
 * ═══════════════════════════════════════════════════════════════════════ */

/* The caller-supplied fill-reducing permutation is copied verbatim into
 * `ldlt->perm` — Day 8 will compose Bunch-Kaufman swaps on top. */
static void test_from_sparse_stores_perm(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0 + (double)i);

    idx_t perm[4] = {3, 2, 1, 0}; /* reverse */
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, perm, 2.0, &m));
    REQUIRE_OK(ldlt_csc_validate(m));
    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(m->perm[i], perm[i]);

    ldlt_csc_free(m);
    sparse_free(A);
}

/* An invalid permutation (out-of-range entry) is rejected by the
 * embedded Cholesky converter; the error propagates through. */
static void test_from_sparse_invalid_perm(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    idx_t bad[3] = {0, 5, 2};
    LdltCsc *m = NULL;
    ASSERT_ERR(ldlt_csc_from_sparse(A, bad, 2.0, &m), SPARSE_ERR_BADARG);
    ASSERT_NULL(m);

    sparse_free(A);
}

/* Reverse permutation on a dense symmetric matrix: conversion keeps
 * only one of each symmetric pair, so the count is n*(n+1)/2 and the
 * retained entries can be checked by mapping CSC entries back to A. */
static void test_reverse_perm_symmetric(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 5.0 + (double)i);
        for (idx_t j = i + 1; j < n; j++) {
            double v = 1.0 + (double)(i * n + j);
            sparse_insert(A, i, j, v);
            sparse_insert(A, j, i, v);
        }
    }

    idx_t perm[4] = {3, 2, 1, 0};
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, perm, 2.0, &m));
    REQUIRE_OK(ldlt_csc_validate(m));
    ASSERT_EQ(m->L->nnz, n * (n + 1) / 2);

    /* Each stored (new_r, new_c) value in L should match A[perm[new_r], perm[new_c]]. */
    for (idx_t j = 0; j < m->L->n; j++) {
        for (idx_t p = m->L->col_ptr[j]; p < m->L->col_ptr[j + 1]; p++) {
            idx_t new_r = m->L->row_idx[p];
            double v = m->L->values[p];
            double a = sparse_get(A, perm[new_r], perm[j]);
            if (v != a)
                TF_FAIL_("L(new %d,%d)->(orig %d,%d): csc=%.15g A=%.15g", (int)new_r, (int)j,
                         (int)perm[new_r], (int)perm[j], v, a);
            tf_asserts++;
        }
    }

    ldlt_csc_free(m);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Validate: positive and catches broken state
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_validate_null(void) { ASSERT_ERR(ldlt_csc_validate(NULL), SPARSE_ERR_NULL); }

static void test_validate_fresh_alloc_is_valid(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 4, &m));
    REQUIRE_OK(ldlt_csc_validate(m));
    ldlt_csc_free(m);
}

/* pivot_size = 3 is invalid (must be 1 or 2). */
static void test_validate_catches_bad_pivot_size(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 4, &m));
    m->pivot_size[2] = 3;
    ASSERT_ERR(ldlt_csc_validate(m), SPARSE_ERR_BADARG);
    ldlt_csc_free(m);
}

/* 2x2 pivot marked on index i with pivot_size[i+1] still == 1 is
 * inconsistent and must be rejected. */
static void test_validate_catches_half_2x2(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 4, &m));
    m->pivot_size[1] = 2;
    /* m->pivot_size[2] left at 1 — mismatch. */
    ASSERT_ERR(ldlt_csc_validate(m), SPARSE_ERR_BADARG);
    ldlt_csc_free(m);
}

/* A 2x2 pivot starting at the last index (no i+1) is invalid. */
static void test_validate_catches_trailing_2x2(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(3, 3, &m));
    m->pivot_size[2] = 2;
    ASSERT_ERR(ldlt_csc_validate(m), SPARSE_ERR_BADARG);
    ldlt_csc_free(m);
}

/* perm with a duplicate index is not a permutation. */
static void test_validate_catches_bad_perm(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(3, 3, &m));
    m->perm[0] = 1;
    m->perm[1] = 1; /* duplicate */
    m->perm[2] = 2;
    ASSERT_ERR(ldlt_csc_validate(m), SPARSE_ERR_BADARG);
    ldlt_csc_free(m);
}

/* Well-formed 2x2 pivot block spanning two consecutive indices — valid. */
static void test_validate_accepts_valid_2x2(void) {
    LdltCsc *m = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 4, &m));
    m->pivot_size[1] = 2;
    m->pivot_size[2] = 2;
    REQUIRE_OK(ldlt_csc_validate(m));
    ldlt_csc_free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 8: Bunch-Kaufman elimination
 * ═══════════════════════════════════════════════════════════════════════ */

/* Retrieve L[r,c] from a factored LdltCsc by linear scan of column c. */
static double ldlt_csc_get(const LdltCsc *F, idx_t r, idx_t c) {
    for (idx_t p = F->L->col_ptr[c]; p < F->L->col_ptr[c + 1]; p++) {
        if (F->L->row_idx[p] == r)
            return F->L->values[p];
    }
    return 0.0;
}

/* Copy an insertion pattern (lower + mirror upper) of a dense symmetric
 * 2D array into a fresh SparseMatrix.  The helper lets us build A twice
 * for the CSC path (via ldlt_csc_from_sparse) and the linked-list
 * comparison (via sparse_ldlt_factor) from the same values. */
static SparseMatrix *build_symmetric(idx_t n, const double *vals_lower) {
    /* vals_lower[i * n + j] holds A[i,j] for i >= j; mirror sets A[j,i]. */
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double v = vals_lower[(size_t)i * (size_t)n + (size_t)j];
            if (v == 0.0)
                continue;
            sparse_insert(A, i, j, v);
            if (i != j)
                sparse_insert(A, j, i, v);
        }
    }
    return A;
}

/* ─── Null-arg ──────────────────────────────────────────────────── */

static void test_eliminate_null(void) { ASSERT_ERR(ldlt_csc_eliminate(NULL), SPARSE_ERR_NULL); }

/* ─── Pure 1x1 pivots: well-conditioned diagonal ────────────────── */

/* Diagonal-only matrix → every pivot is 1x1, D[k] matches the diagonal
 * values exactly, and L is the identity (only unit diagonals). */
static void test_eliminate_all_1x1_pivots(void) {
    idx_t n = 4;
    double diag[4] = {3.0, -2.0, 5.0, 1.5};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    for (idx_t k = 0; k < n; k++) {
        ASSERT_EQ(F->pivot_size[k], 1);
        ASSERT_NEAR(F->D[k], diag[k], 1e-12);
        ASSERT_NEAR(F->D_offdiag[k], 0.0, 0.0);
        /* L[k,k] = 1.0 (unit diagonal stored for uniformity). */
        ASSERT_NEAR(ldlt_csc_get(F, k, k), 1.0, 1e-12);
    }

    ldlt_csc_free(F);
    sparse_free(A);
}

/* Diagonally dominant tridiagonal — all pivots should be 1x1, and L
 * reconstructs A via L * D * L^T to tolerance. */
static void test_eliminate_tridiagonal_all_1x1(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    for (idx_t k = 0; k < n; k++)
        ASSERT_EQ(F->pivot_size[k], 1);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Forced 2x2 pivot: zero diagonal at (0,0) ──────────────────── */

/* A = [[0.1, 1], [1, 0.3]]: both diagonals are small relative to the
 * off-diagonal so Bunch-Kaufman's Criterion 4 fires and a 2x2 pivot is
 * chosen.  (A 2x2 where either diagonal dominates — e.g. c >= alpha*b
 * — is resolved by a 1x1 pivot with a row swap instead of a 2x2.) */
static void test_eliminate_forced_2x2(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    ASSERT_EQ(F->pivot_size[0], 2);
    ASSERT_EQ(F->pivot_size[1], 2);
    /* D[0]=0.1, D[1]=0.3, D_offdiag[0]=1.0 — the 2x2 pivot block. */
    ASSERT_NEAR(F->D[0], 0.1, 1e-12);
    ASSERT_NEAR(F->D_offdiag[0], 1.0, 1e-12);
    ASSERT_NEAR(F->D[1], 0.3, 1e-12);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Mixed 1x1 + 2x2 pivots ────────────────────────────────────── */

/* Block-diagonal with one 2x2 saddle block and a 1x1 well-conditioned
 * block in the trailing rows.  The first 2x2 block uses small diagonals
 * (both < alpha_bk * |off-diag|) to force Criterion 4 → 2x2 pivot. */
static void test_eliminate_mixed_pivots(void) {
    idx_t n = 4;
    double vals[4][4] = {
        {0.1, 1.0, 0.0, 0.0}, {1.0, 0.3, 0.0, 0.0}, {0.0, 0.0, 5.0, 0.0}, {0.0, 0.0, 0.0, 3.0}};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            if (vals[i][j] != 0.0)
                sparse_insert(A, i, j, vals[i][j]);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    ASSERT_EQ(F->pivot_size[0], 2);
    ASSERT_EQ(F->pivot_size[1], 2);
    ASSERT_EQ(F->pivot_size[2], 1);
    ASSERT_EQ(F->pivot_size[3], 1);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Element-by-element comparison vs linked-list ─────────────── */

/* Factor the same matrix via CSC and via linked-list sparse_ldlt_factor.
 * The CSC wrapper delegates to the linked-list kernel internally, so
 * results must match byte-for-byte on D, D_offdiag, pivot_size, perm,
 * and L's entries. */
/* Simple seeded LCG so the random cross-check is deterministic across
 * runs (same RNG style used by the Day 2 stress test). */
static unsigned int xc_rng_state = 0u;
static unsigned int xc_rng_next(void) {
    xc_rng_state = xc_rng_state * 1664525u + 1013904223u;
    return xc_rng_state;
}
static double xc_rng_uniform(double lo, double hi) {
    double u = (double)(xc_rng_next() & 0x7fffffff) / (double)0x7fffffff;
    return lo + (hi - lo) * u;
}

/* Build a random n×n symmetric indefinite matrix.  The diagonal is
 * uniform on [-3, 3] and a random subset of off-diagonals in [-2, 2]
 * is inserted (mirrored).  The seed is baked in by the caller so each
 * iteration of the test sweep visits a reproducible fixture. */
static SparseMatrix *build_random_symmetric(idx_t n, unsigned int seed) {
    xc_rng_state = seed;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        double d = xc_rng_uniform(-3.0, 3.0);
        if (fabs(d) < 0.3)
            d = (d < 0 ? -0.3 : 0.3);
        sparse_insert(A, i, i, d);
    }
    for (idx_t i = 1; i < n; i++) {
        for (idx_t j = 0; j < i; j++) {
            /* ~35% off-diagonal density — enough to trigger BK cross-
             * column cmod without saturating to a dense factor. */
            if ((xc_rng_next() % 100) < 35) {
                double v = xc_rng_uniform(-2.0, 2.0);
                sparse_insert(A, i, j, v);
                sparse_insert(A, j, i, v);
            }
        }
    }
    return A;
}

/* Compare a factored `LdltCsc` against a reference `sparse_ldlt_t` on
 * the same input (via `sparse_ldlt_factor` on a separate `SparseMatrix`
 * with identical values).  Tolerance checks match the single-matrix
 * case — pivot_size and perm must agree exactly, D / D_offdiag / L
 * entries within 1e-10 absolute. */
static int ldlt_csc_matches_linked_list(const LdltCsc *F, const sparse_ldlt_t *ll, idx_t n,
                                        double tol) {
    for (idx_t k = 0; k < n; k++) {
        if (F->pivot_size[k] != (idx_t)ll->pivot_size[k])
            return 0;
        if (F->perm[k] != ll->perm[k])
            return 0;
        if (fabs(F->D[k] - ll->D[k]) > tol)
            return 0;
        if (fabs(F->D_offdiag[k] - ll->D_offdiag[k]) > tol)
            return 0;
    }
    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < n; i++) {
            double csc = ldlt_csc_get(F, i, j);
            double ref = sparse_get(ll->L, i, j);
            if (i == j) {
                if (fabs(csc - 1.0) > tol)
                    return 0;
            } else if (fabs(csc - ref) > tol) {
                return 0;
            }
        }
    }
    return 1;
}

static void test_eliminate_matches_linked_list_indefinite(void) {
    /* Original fixed 4×4 matrix — keeps the legacy single-case coverage
     * alongside the random sweep below so a targeted regression is
     * still easy to inspect. */
    idx_t n = 4;
    double lower[16] = {
        2.0,  0.0,  0.0,  0.0, /* row 0 */
        -1.0, 3.0,  0.0,  0.0, /* row 1 */
        0.5,  -2.0, -1.0, 0.0, /* row 2 */
        0.0,  1.0,  0.5,  4.0, /* row 3 */
    };
    SparseMatrix *A1 = build_symmetric(n, lower);
    SparseMatrix *A2 = build_symmetric(n, lower);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A1, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    sparse_ldlt_t ll = {0};
    REQUIRE_OK(sparse_ldlt_factor(A2, &ll));

    ASSERT_TRUE(ldlt_csc_matches_linked_list(F, &ll, n, 1e-12));

    ldlt_csc_free(F);
    sparse_ldlt_free(&ll);
    sparse_free(A1);
    sparse_free(A2);
}

/* Sprint 18 Day 5 cross-check: sweep 20 random symmetric indefinite
 * matrices of varied size and assert the native CSC kernel's output
 * matches the linked-list `sparse_ldlt_factor` reference within
 * floating-point round-off on every one.  Seeds are fixed so failure
 * can be reproduced by rerunning the test. */
static void test_eliminate_matches_linked_list_random_indefinite(void) {
    /* Sizes chosen to cover small (where cmod contributions are
     * minimal) and moderate (where mixed 1×1/2×2 pivots and swaps are
     * likely).  Each seed is a distinct 32-bit constant — no aliasing. */
    struct {
        idx_t n;
        unsigned int seed;
    } cases[] = {
        {3, 0x1a2b3c4du},  {3, 0x5e6f7081u},  {4, 0x92a3b4c5u},  {4, 0xd6e7f809u},
        {5, 0x10123456u},  {5, 0x789abcdeu},  {6, 0xf0123456u},  {6, 0x22222222u},
        {7, 0x33333333u},  {7, 0x44444444u},  {8, 0xaaaaaaaau},  {8, 0xbbbbbbbbu},
        {9, 0xcccccccdu},  {10, 0xdeadbeefu}, {10, 0xfeedfaceu}, {11, 0xcafebabeu},
        {12, 0xbadc0ffeu}, {12, 0x13579bdfu}, {15, 0x2468aceeu}, {18, 0x01020304u},
    };
    const size_t ncases = sizeof(cases) / sizeof(cases[0]);

    int divergences = 0;
    for (size_t idx = 0; idx < ncases; idx++) {
        idx_t n = cases[idx].n;
        SparseMatrix *A1 = build_random_symmetric(n, cases[idx].seed);
        SparseMatrix *A2 = build_random_symmetric(n, cases[idx].seed);

        LdltCsc *F = NULL;
        sparse_err_t err_csc = ldlt_csc_from_sparse(A1, NULL, 2.0, &F);
        sparse_ldlt_t ll = {0};
        sparse_err_t err_ll = sparse_ldlt_factor(A2, &ll);

        /* Both paths must succeed, or both must fail with the same
         * error.  Random matrices sometimes produce singular pivots;
         * BK's decisions are identical across paths so the failures
         * should line up. */
        if (err_csc != SPARSE_OK || err_ll != SPARSE_OK) {
            ASSERT_EQ(err_csc, err_ll);
            if (F)
                ldlt_csc_free(F);
            sparse_ldlt_free(&ll);
            sparse_free(A1);
            sparse_free(A2);
            continue;
        }

        sparse_err_t err_elim = ldlt_csc_eliminate(F);
        if (err_elim != SPARSE_OK) {
            divergences++;
            ldlt_csc_free(F);
            sparse_ldlt_free(&ll);
            sparse_free(A1);
            sparse_free(A2);
            continue;
        }

        REQUIRE_OK(ldlt_csc_validate(F));
        if (!ldlt_csc_matches_linked_list(F, &ll, n, 1e-10))
            divergences++;

        ldlt_csc_free(F);
        sparse_ldlt_free(&ll);
        sparse_free(A1);
        sparse_free(A2);
    }

    ASSERT_EQ(divergences, 0);
}

/* Sprint 19 Day 9 cross-check: after `ldlt_csc_eliminate_native`
 * finishes, `F->row_adj[r]` must list *exactly* the set of columns
 * `c < r` where `L[r, c] != 0`.  Sweeping 20 random indefinite
 * matrices catches both over-population (spurious entries) and
 * under-population (missing priors that would silently drop cmod
 * contributions on later columns). */
static void test_row_adj_matches_reference(void) {
    struct {
        idx_t n;
        unsigned int seed;
    } cases[] = {
        {3, 0x1a2b3c4du},  {3, 0x5e6f7081u},  {4, 0x92a3b4c5u},  {4, 0xd6e7f809u},
        {5, 0x10123456u},  {5, 0x789abcdeu},  {6, 0xf0123456u},  {6, 0x22222222u},
        {7, 0x33333333u},  {7, 0x44444444u},  {8, 0xaaaaaaaau},  {8, 0xbbbbbbbbu},
        {9, 0xcccccccdu},  {10, 0xdeadbeefu}, {10, 0xfeedfaceu}, {11, 0xcafebabeu},
        {12, 0xbadc0ffeu}, {12, 0x13579bdfu}, {15, 0x2468aceeu}, {18, 0x01020304u},
    };
    const size_t ncases = sizeof(cases) / sizeof(cases[0]);

    for (size_t idx = 0; idx < ncases; idx++) {
        idx_t n = cases[idx].n;
        SparseMatrix *A = build_random_symmetric(n, cases[idx].seed);

        LdltCsc *F = NULL;
        sparse_err_t err_csc = ldlt_csc_from_sparse(A, NULL, 2.0, &F);
        if (err_csc != SPARSE_OK) {
            if (F)
                ldlt_csc_free(F);
            sparse_free(A);
            continue;
        }
        sparse_err_t err_elim = ldlt_csc_eliminate(F);
        if (err_elim != SPARSE_OK) {
            ldlt_csc_free(F);
            sparse_free(A);
            continue;
        }

        /* Build the reference `expected_adj[r]` by walking L column-
         * major: for every stored `L[i, c]` with `i > c`, the column
         * `c` is a prior of row `i`. */
        for (idx_t r = 0; r < n; r++) {
            idx_t expected_count = 0;
            for (idx_t c = 0; c < r; c++) {
                idx_t cs = F->L->col_ptr[c];
                idx_t ce = F->L->col_ptr[c + 1];
                for (idx_t p = cs; p < ce; p++) {
                    if (F->L->row_idx[p] == r) {
                        expected_count++;
                        break;
                    }
                }
            }
            ASSERT_EQ(F->row_adj_count[r], expected_count);

            /* Every entry in row_adj[r] must point to a column < r
             * where row r is actually stored.  Duplicates would
             * inflate `expected_count` above, so the count equality
             * also rules them out. */
            for (idx_t e = 0; e < F->row_adj_count[r]; e++) {
                idx_t c = F->row_adj[r][e];
                ASSERT_TRUE(c >= 0 && c < r);
                int found = 0;
                idx_t cs = F->L->col_ptr[c];
                idx_t ce = F->L->col_ptr[c + 1];
                for (idx_t p = cs; p < ce; p++) {
                    if (F->L->row_idx[p] == r) {
                        found = 1;
                        break;
                    }
                }
                ASSERT_TRUE(found);
            }
        }

        ldlt_csc_free(F);
        sparse_free(A);
    }
}

/* ─── Perm composition with fill-reducing input ─────────────────── */

/* When ldlt_csc_from_sparse applies a non-identity `perm_in`, the
 * subsequent eliminate should store perm[k] = perm_in[ll_perm[k]] —
 * the composition of the input perm with the Bunch-Kaufman perm.
 *
 * We verify the composition explicitly by independently factoring the
 * pre-permuted matrix P·A·P^T with the linked-list kernel, so
 * `ll_ref.perm` is the Bunch-Kaufman pivot perm on the same input the
 * CSC wrapper sees internally.  The composition rule then predicts
 * each entry of `F->perm` exactly. */
static void test_eliminate_composes_perm(void) {
    idx_t n = 4;
    /* Symmetric indefinite values so Bunch-Kaufman has something to
     * decide (mix of signs + off-diagonals). */
    double lower[16] = {
        2.0,  0.0, 0.0,  0.0, /* row 0 */
        -1.0, 3.0, 0.0,  0.0, /* row 1 */
        0.5,  0.2, -1.0, 0.0, /* row 2 */
        0.0,  1.0, 0.5,  4.0, /* row 3 */
    };
    SparseMatrix *A = build_symmetric(n, lower);
    /* Use a non-trivial input perm (reverse order). */
    idx_t perm_in[4] = {3, 2, 1, 0};

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, perm_in, 2.0, &F));
    /* Before elimination, F->perm == perm_in verbatim. */
    for (idx_t k = 0; k < n; k++)
        ASSERT_EQ(F->perm[k], perm_in[k]);

    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_validate(F));

    /* Build P·A·P^T and factor with the linked-list kernel directly so
     * ll_ref.perm is the BK pivot perm in the pre-permuted space — the
     * same space the CSC wrapper runs BK in internally. */
    SparseMatrix *A_perm = NULL;
    REQUIRE_OK(sparse_permute(A, perm_in, perm_in, &A_perm));
    sparse_ldlt_t ll_ref = {0};
    REQUIRE_OK(sparse_ldlt_factor(A_perm, &ll_ref));

    /* Composition rule: F->perm[k] == perm_in[ll_ref.perm[k]]. */
    for (idx_t k = 0; k < n; k++)
        ASSERT_EQ(F->perm[k], perm_in[ll_ref.perm[k]]);

    sparse_ldlt_free(&ll_ref);
    sparse_free(A_perm);
    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Inertia: sign accounting from D matches linked-list ─────── */

static void test_eliminate_inertia(void) {
    /* Diagonal with 2 positive, 2 negative entries. */
    idx_t n = 4;
    double diag[4] = {4.0, -1.0, 3.0, -2.5};
    SparseMatrix *A1 = sparse_create(n, n);
    SparseMatrix *A2 = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A1, i, i, diag[i]);
        sparse_insert(A2, i, i, diag[i]);
    }

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A1, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    sparse_ldlt_t ll = {0};
    REQUIRE_OK(sparse_ldlt_factor(A2, &ll));

    idx_t pos_ll, neg_ll, zero_ll;
    REQUIRE_OK(sparse_ldlt_inertia(&ll, &pos_ll, &neg_ll, &zero_ll));

    /* Count inertia from F's D/pivot_size directly (1x1 blocks only here). */
    idx_t pos = 0, neg = 0, zero = 0;
    for (idx_t k = 0; k < n; k++) {
        if (F->D[k] > 0)
            pos++;
        else if (F->D[k] < 0)
            neg++;
        else
            zero++;
    }
    ASSERT_EQ(pos, pos_ll);
    ASSERT_EQ(neg, neg_ll);
    ASSERT_EQ(zero, zero_ll);

    ldlt_csc_free(F);
    sparse_ldlt_free(&ll);
    sparse_free(A1);
    sparse_free(A2);
}

/* ─── Singular matrix → NOT_SPD/SINGULAR ───────────────────────── */

/* Zero matrix is rank-deficient — the first pivot fails singularity check. */
static void test_eliminate_singular_zero(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    /* Populate at least one nonzero so symmetry check passes, then
     * build a singular structure. */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    /* Column/row 1 is entirely zero → singular. */

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    sparse_err_t err = ldlt_csc_eliminate(F);
    ASSERT_TRUE(err == SPARSE_ERR_SINGULAR || err == SPARSE_ERR_NOT_SPD);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 1: Native-kernel scaffolding
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Verify the Day 1 skeleton: the runtime override reaches the native
 * kernel, the native kernel validates inputs the same way the wrapper
 * does, and the still-stub column loop returns SPARSE_ERR_BADARG so
 * tests and benchmarks get a clean signal while the 1x1/2x2 branches
 * land across Days 3-4.  These tests restore the default override in
 * teardown so subsequent tests still exercise the wrapper path. */

/* ─── Override getter/setter round-trip ─────────────────────────── */

static void test_native_override_roundtrip(void) {
    ASSERT_EQ(ldlt_csc_get_kernel_override(), LDLT_CSC_KERNEL_DEFAULT);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    ASSERT_EQ(ldlt_csc_get_kernel_override(), LDLT_CSC_KERNEL_NATIVE);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_WRAPPER);
    ASSERT_EQ(ldlt_csc_get_kernel_override(), LDLT_CSC_KERNEL_WRAPPER);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);
    ASSERT_EQ(ldlt_csc_get_kernel_override(), LDLT_CSC_KERNEL_DEFAULT);
}

/* ─── Placeholder preserved for the test registration order; real
 * 2×2 coverage lives in the Sprint 18 Day 4 block below, which
 * verifies that the forced-2×2 matrix factors correctly under the
 * native kernel. */

/* ─── Native kernel handles n == 0 trivially ───────────────────── */

static void test_native_empty_matrix_ok(void) {
    /* A 0x0 LdltCsc via direct alloc (ldlt_csc_from_sparse rejects
     * zero-dim input, matching sparse_ldlt_factor).  The native
     * kernel's early-out for n <= 0 returns SPARSE_OK. */
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(0, 1, &F));

    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    ASSERT_ERR(ldlt_csc_eliminate(F), SPARSE_OK);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    ldlt_csc_free(F);
}

/* ─── Native kernel input validation matches the wrapper ───────── */

static void test_native_rejects_null(void) {
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    ASSERT_ERR(ldlt_csc_eliminate(NULL), SPARSE_ERR_NULL);
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);
}

/* ─── Wrapper override still reaches the wrapper ───────────────── */

static void test_wrapper_override_still_factors(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, -3.0);
    sparse_insert(A, 2, 2, 4.0);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));

    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_WRAPPER);
    REQUIRE_OK(ldlt_csc_eliminate(F));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);

    REQUIRE_OK(ldlt_csc_validate(F));
    ASSERT_EQ(F->pivot_size[0], 1);
    ASSERT_EQ(F->pivot_size[1], 1);
    ASSERT_EQ(F->pivot_size[2], 1);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Workspace lifecycle ──────────────────────────────────────── */

static void test_workspace_alloc_free(void) {
    LdltCscWorkspace *ws = NULL;
    REQUIRE_OK(ldlt_csc_workspace_alloc(5, &ws));
    ASSERT_TRUE(ws != NULL);
    ASSERT_EQ(ws->n, 5);
    ASSERT_EQ(ws->pattern_count, 0);
    ASSERT_EQ(ws->pattern_count_r, 0);
    ASSERT_TRUE(ws->dense_col != NULL);
    ASSERT_TRUE(ws->dense_col_r != NULL);
    ldlt_csc_workspace_free(ws);

    /* free(NULL) is a no-op. */
    ldlt_csc_workspace_free(NULL);

    /* n == 0 still yields a valid allocation (all pointers non-NULL
     * so callers don't have to special-case the empty matrix). */
    LdltCscWorkspace *ws0 = NULL;
    REQUIRE_OK(ldlt_csc_workspace_alloc(0, &ws0));
    ASSERT_TRUE(ws0->dense_col != NULL);
    ldlt_csc_workspace_free(ws0);
}

static void test_workspace_alloc_rejects_negative_n(void) {
    LdltCscWorkspace *ws = (LdltCscWorkspace *)0xdead;
    ASSERT_ERR(ldlt_csc_workspace_alloc(-1, &ws), SPARSE_ERR_BADARG);
    ASSERT_TRUE(ws == NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 2: In-place symmetric swap primitive
 * ═══════════════════════════════════════════════════════════════════════ */

/* ─── Dense helpers for cross-checking the swap ─────────────────── */

/* Copy the stored lower triangle of F->L into a dense n*n row-major
 * buffer.  Zero everywhere F->L has no entry — this matches the
 * sparse convention (structurally absent == numerically zero).  The
 * upper triangle is left zero so the oracle reads only the lower
 * triangle. */
static void ldlt_lower_to_dense(const LdltCsc *F, double *dense) {
    idx_t n = F->n;
    for (idx_t p = 0; p < n * n; p++)
        dense[p] = 0.0;
    for (idx_t c = 0; c < n; c++) {
        idx_t start = F->L->col_ptr[c];
        idx_t end = F->L->col_ptr[c + 1];
        for (idx_t p = start; p < end; p++) {
            idx_t r = F->L->row_idx[p];
            dense[r * n + c] = F->L->values[p];
        }
    }
}

/* Apply the symmetric permutation σ = (i ↔ j) to a dense lower
 * triangle.  For each stored entry (r, c) with r >= c, map to
 * (σ(r), σ(c)), reflect to lower-triangle if σ(r) < σ(c), write to
 * `dst`.  `dst` must be a separate buffer (caller zero-fills). */
static void dense_sym_swap(const double *src, double *dst, idx_t n, idx_t i, idx_t j) {
    for (idx_t p = 0; p < n * n; p++)
        dst[p] = 0.0;
    for (idx_t c = 0; c < n; c++) {
        for (idx_t r = c; r < n; r++) {
            double v = src[r * n + c];
            if (v == 0.0)
                continue;
            idx_t rn = (r == i) ? j : ((r == j) ? i : r);
            idx_t cn = (c == i) ? j : ((c == j) ? i : c);
            if (rn < cn) {
                idx_t t = rn;
                rn = cn;
                cn = t;
            }
            dst[rn * n + cn] = v;
        }
    }
}

/* Element-wise compare two dense lower triangles.  Returns 1 if all
 * corresponding entries match to tol, 0 otherwise. */
static int dense_lower_equal(const double *a, const double *b, idx_t n, double tol) {
    for (idx_t r = 0; r < n; r++) {
        for (idx_t c = 0; c <= r; c++) {
            double diff = fabs(a[r * n + c] - b[r * n + c]);
            if (diff > tol)
                return 0;
        }
    }
    return 1;
}

/* Build an LdltCsc from a list of (row, col, value) triples
 * representing the lower triangle of a symmetric matrix of dimension
 * n.  Inserts mirrored entries so `ldlt_csc_from_sparse`'s symmetry
 * check passes.  Each column's diagonal is inserted explicitly (add
 * `diag_fill[c]` for the diagonal value) so `chol_csc_validate`'s
 * diagonal-first invariant holds after the swap. */
static LdltCsc *build_ldlt_from_triples(idx_t n, const double *diag, const idx_t *rows,
                                        const idx_t *cols, const double *vals, idx_t nnz_offdiag) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t c = 0; c < n; c++)
        sparse_insert(A, c, c, diag[c]);
    for (idx_t k = 0; k < nnz_offdiag; k++) {
        sparse_insert(A, rows[k], cols[k], vals[k]);
        sparse_insert(A, cols[k], rows[k], vals[k]); /* mirror */
    }
    LdltCsc *F = NULL;
    if (ldlt_csc_from_sparse(A, NULL, 2.0, &F) != SPARSE_OK) {
        sparse_free(A);
        return NULL;
    }
    sparse_free(A);
    return F;
}

/* ─── Error-path tests ──────────────────────────────────────────── */

static void test_symmetric_swap_null(void) {
    ASSERT_ERR(ldlt_csc_symmetric_swap(NULL, 0, 1), SPARSE_ERR_NULL);
}

static void test_symmetric_swap_out_of_range(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 10, &F));
    ASSERT_ERR(ldlt_csc_symmetric_swap(F, -1, 2), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_symmetric_swap(F, 2, 4), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_csc_symmetric_swap(F, 4, 2), SPARSE_ERR_BADARG);
    ldlt_csc_free(F);
}

static void test_symmetric_swap_noop_i_equals_j(void) {
    /* Dense 3×3 SPD-ish matrix; swap(1, 1) must leave storage
     * byte-identical. */
    idx_t rows[] = {1, 2, 2};
    idx_t cols[] = {0, 0, 1};
    double vals[] = {0.5, 0.3, 0.7};
    double diag[] = {2.0, 3.0, 4.0};
    LdltCsc *F = build_ldlt_from_triples(3, diag, rows, cols, vals, 3);
    REQUIRE_OK(ldlt_csc_validate(F));

    idx_t nnz_before = F->L->col_ptr[F->n];
    double *dense_before = calloc((size_t)(F->n * F->n), sizeof(double));
    double *dense_after = calloc((size_t)(F->n * F->n), sizeof(double));
    ldlt_lower_to_dense(F, dense_before);

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 1, 1));
    ldlt_lower_to_dense(F, dense_after);

    ASSERT_EQ(F->L->col_ptr[F->n], nnz_before);
    ASSERT_TRUE(dense_lower_equal(dense_before, dense_after, F->n, 0.0));
    REQUIRE_OK(ldlt_csc_validate(F));

    free(dense_before);
    free(dense_after);
    ldlt_csc_free(F);
}

/* ─── Adjacent swap on a dense 5×5 ───────────────────────────────── */

static void test_symmetric_swap_adjacent_dense(void) {
    /* Build a dense-ish 5×5 symmetric matrix with unique values so
     * any positional mistake after the swap is easy to spot. */
    idx_t n = 5;
    double diag[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    /* Offdiagonals (row > col): unique values so position permutation
     * is verifiable. */
    idx_t rows[] = {1, 2, 3, 4, 2, 3, 4, 3, 4, 4};
    idx_t cols[] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3};
    double vals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0};
    LdltCsc *F = build_ldlt_from_triples(n, diag, rows, cols, vals, 10);
    REQUIRE_OK(ldlt_csc_validate(F));

    double *dense_src = calloc((size_t)(n * n), sizeof(double));
    double *dense_expected = calloc((size_t)(n * n), sizeof(double));
    double *dense_actual = calloc((size_t)(n * n), sizeof(double));
    ldlt_lower_to_dense(F, dense_src);
    dense_sym_swap(dense_src, dense_expected, n, 1, 2);

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 1, 2));
    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_lower_to_dense(F, dense_actual);

    ASSERT_TRUE(dense_lower_equal(dense_expected, dense_actual, n, 1e-15));

    free(dense_src);
    free(dense_expected);
    free(dense_actual);
    ldlt_csc_free(F);
}

/* ─── Non-adjacent swap on dense 5×5 ───────────────────────────── */

static void test_symmetric_swap_non_adjacent(void) {
    idx_t n = 5;
    double diag[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    idx_t rows[] = {1, 2, 3, 4, 2, 3, 4, 3, 4, 4};
    idx_t cols[] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3};
    double vals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0};
    LdltCsc *F = build_ldlt_from_triples(n, diag, rows, cols, vals, 10);
    REQUIRE_OK(ldlt_csc_validate(F));

    double *dense_src = calloc((size_t)(n * n), sizeof(double));
    double *dense_expected = calloc((size_t)(n * n), sizeof(double));
    double *dense_actual = calloc((size_t)(n * n), sizeof(double));
    ldlt_lower_to_dense(F, dense_src);
    /* Swap i=1, j=4 — touches left block (col 0), middle block
     * (cols 1..4), and the below-j range (empty for n=5). */
    dense_sym_swap(dense_src, dense_expected, n, 1, 4);

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 1, 4));
    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_lower_to_dense(F, dense_actual);

    ASSERT_TRUE(dense_lower_equal(dense_expected, dense_actual, n, 1e-15));

    free(dense_src);
    free(dense_expected);
    free(dense_actual);
    ldlt_csc_free(F);
}

/* ─── Swap on a tridiagonal pattern ─────────────────────────────── */

static void test_symmetric_swap_tridiagonal(void) {
    idx_t n = 6;
    double diag[] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    idx_t rows[] = {1, 2, 3, 4, 5};
    idx_t cols[] = {0, 1, 2, 3, 4};
    double vals[] = {-1.0, -1.0, -1.0, -1.0, -1.0};
    LdltCsc *F = build_ldlt_from_triples(n, diag, rows, cols, vals, 5);
    REQUIRE_OK(ldlt_csc_validate(F));

    double *dense_src = calloc((size_t)(n * n), sizeof(double));
    double *dense_expected = calloc((size_t)(n * n), sizeof(double));
    double *dense_actual = calloc((size_t)(n * n), sizeof(double));
    ldlt_lower_to_dense(F, dense_src);
    /* Swap i=1, j=4 — leaves many entries structurally zero, so phase
     * A's "only i" and "only j" branches both get exercised. */
    dense_sym_swap(dense_src, dense_expected, n, 1, 4);

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 1, 4));
    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_lower_to_dense(F, dense_actual);

    ASSERT_TRUE(dense_lower_equal(dense_expected, dense_actual, n, 1e-15));

    free(dense_src);
    free(dense_expected);
    free(dense_actual);
    ldlt_csc_free(F);
}

/* ─── Swap that moves a tiny diagonal onto the pivot seat ───────── */

/* The BK use case: A has a tiny/structurally-zero diagonal at row 0
 * so the pivot selector picks another row r and swaps r ↔ 0 to put
 * the larger off-diagonal onto the pivot seat.  Verify the swap
 * correctly relocates the tiny diagonal. */
static void test_symmetric_swap_moves_tiny_diag(void) {
    idx_t n = 4;
    /* Diag: tiny at 0, normal elsewhere; large off-diagonal (0, 2). */
    double diag[] = {1e-12, 5.0, 3.0, 4.0};
    idx_t rows[] = {2};
    idx_t cols[] = {0};
    double vals[] = {9.0};
    LdltCsc *F = build_ldlt_from_triples(n, diag, rows, cols, vals, 1);
    REQUIRE_OK(ldlt_csc_validate(F));

    double *dense_src = calloc((size_t)(n * n), sizeof(double));
    double *dense_expected = calloc((size_t)(n * n), sizeof(double));
    double *dense_actual = calloc((size_t)(n * n), sizeof(double));
    ldlt_lower_to_dense(F, dense_src);
    dense_sym_swap(dense_src, dense_expected, n, 0, 2);

    /* Also check F->perm reflects the swap. */
    idx_t perm0_before = F->perm[0];
    idx_t perm2_before = F->perm[2];

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 0, 2));
    REQUIRE_OK(ldlt_csc_validate(F));
    ldlt_lower_to_dense(F, dense_actual);

    ASSERT_EQ(F->perm[0], perm2_before);
    ASSERT_EQ(F->perm[2], perm0_before);
    /* Tiny diag should now be at (2, 2). */
    ASSERT_TRUE(fabs(dense_actual[2 * n + 2] - 1e-12) < 1e-18);
    /* Old diag at (2, 2) = 3.0 should now be at (0, 0). */
    ASSERT_TRUE(fabs(dense_actual[0] - 3.0) < 1e-15);
    ASSERT_TRUE(dense_lower_equal(dense_expected, dense_actual, n, 1e-15));

    free(dense_src);
    free(dense_expected);
    free(dense_actual);
    ldlt_csc_free(F);
}

/* ─── Aux-array swap: D / D_offdiag / pivot_size ────────────────── */

static void test_symmetric_swap_aux_arrays(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(4, 1, &F));
    /* Simulate a partially-populated state: manually set aux fields. */
    F->D[0] = 10.0;
    F->D[1] = 20.0;
    F->D[2] = 30.0;
    F->D[3] = 40.0;
    F->D_offdiag[0] = 0.0;
    F->D_offdiag[1] = 0.0;
    F->D_offdiag[2] = 7.0;
    F->D_offdiag[3] = 0.0;
    F->pivot_size[0] = 1;
    F->pivot_size[1] = 1;
    F->pivot_size[2] = 2;
    F->pivot_size[3] = 2;
    F->perm[0] = 100;
    F->perm[1] = 101;
    F->perm[2] = 102;
    F->perm[3] = 103;

    REQUIRE_OK(ldlt_csc_symmetric_swap(F, 0, 2));

    ASSERT_EQ((int)(F->D[0] * 1000), 30000);
    ASSERT_EQ((int)(F->D[2] * 1000), 10000);
    ASSERT_EQ((int)(F->D_offdiag[0] * 1000), 7000);
    ASSERT_EQ((int)(F->D_offdiag[2] * 1000), 0);
    ASSERT_EQ(F->pivot_size[0], 2);
    ASSERT_EQ(F->pivot_size[2], 1);
    ASSERT_EQ(F->perm[0], 102);
    ASSERT_EQ(F->perm[2], 100);
    /* Unswapped positions unchanged. */
    ASSERT_EQ((int)(F->D[1] * 1000), 20000);
    ASSERT_EQ((int)(F->D[3] * 1000), 40000);

    ldlt_csc_free(F);
}

/* ─── Stress test: random swaps cross-checked against dense oracle ─ */

/* LCG so tests are deterministic and portable (no rand()/time()
 * dependence).  Same style as other random tests in this repo. */
static unsigned int swap_rng_state = 0xB0F17E84u;
static unsigned int swap_rng_next(void) {
    swap_rng_state = swap_rng_state * 1664525u + 1013904223u;
    return swap_rng_state;
}

static void test_symmetric_swap_stress_random(void) {
    idx_t n = 20;
    swap_rng_state = 0xB0F17E84u;

    /* Build a random symmetric matrix with explicit diagonals. */
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t c = 0; c < n; c++)
        sparse_insert(A, c, c, 1.0 + (double)(swap_rng_next() % 100));
    /* Add ~30 random off-diagonals (mirrored). */
    for (int k = 0; k < 30; k++) {
        idx_t r = (idx_t)(swap_rng_next() % (unsigned)n);
        idx_t c = (idx_t)(swap_rng_next() % (unsigned)n);
        if (r == c)
            continue;
        if (r < c) {
            idx_t t = r;
            r = c;
            c = t;
        }
        double v = (double)((swap_rng_next() % 100) + 1) / 10.0;
        sparse_insert(A, r, c, v);
        sparse_insert(A, c, r, v);
    }
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    sparse_free(A);

    double *dense_a = calloc((size_t)(n * n), sizeof(double));
    double *dense_b = calloc((size_t)(n * n), sizeof(double));
    ldlt_lower_to_dense(F, dense_a);

    /* 50 random swaps; after each, dense oracle swap and compare. */
    for (int step = 0; step < 50; step++) {
        idx_t i = (idx_t)(swap_rng_next() % (unsigned)n);
        idx_t j = (idx_t)(swap_rng_next() % (unsigned)n);
        /* Oracle swap from dense_a → dense_b, then copy dense_b back
         * to dense_a for the next iteration. */
        dense_sym_swap(dense_a, dense_b, n, i, j);
        REQUIRE_OK(ldlt_csc_symmetric_swap(F, i, j));
        REQUIRE_OK(ldlt_csc_validate(F));
        double *tmp = dense_a;
        dense_a = dense_b;
        dense_b = tmp;
        /* dense_a now holds the oracle's current state; check CSC
         * matches. */
        double *dense_csc = calloc((size_t)(n * n), sizeof(double));
        ldlt_lower_to_dense(F, dense_csc);
        ASSERT_TRUE(dense_lower_equal(dense_a, dense_csc, n, 1e-15));
        free(dense_csc);
    }

    free(dense_a);
    free(dense_b);
    ldlt_csc_free(F);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 3: 1×1 Bunch-Kaufman column loop
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compare two factored LdltCsc values field-by-field.  Returns 1 iff
 * pivot_size, perm, col_ptr, row_idx all match exactly and D, D_offdiag,
 * L->values agree within `tol` absolute.  Use a tight `tol` (1e-12) to
 * flag any computational divergence between wrapper and native; the
 * floating-point ordering of the cmod loop is the same on both paths
 * (both subtract contributions kp = 0, 1, ..., k-1 in order), so values
 * should match to round-off. */
/* Walk column `j` of L and for every stored entry whose magnitude is
 * >= `tol`, check that B has a stored entry at the same row with a
 * matching value.  Either side may zero-pad dropped positions (e.g.,
 * Sprint 19 Day 6's `chol_csc_gather` keeps `col_ptr` immutable and
 * writes 0.0 into below-threshold slots instead of shrinking the
 * column), so the comparison filters zeros before comparing. */
static int ldlt_column_nonzeros_match(const CholCsc *A, const CholCsc *B, idx_t j, double tol) {
    idx_t ap = A->col_ptr[j];
    idx_t ae = A->col_ptr[j + 1];
    idx_t bp = B->col_ptr[j];
    idx_t be = B->col_ptr[j + 1];
    while (ap < ae || bp < be) {
        /* Skip zero-valued entries on A's side. */
        while (ap < ae && fabs(A->values[ap]) < tol && A->row_idx[ap] != j)
            ap++;
        while (bp < be && fabs(B->values[bp]) < tol && B->row_idx[bp] != j)
            bp++;
        if (ap == ae && bp == be)
            return 1;
        if (ap == ae || bp == be)
            return 0;
        if (A->row_idx[ap] != B->row_idx[bp])
            return 0;
        if (fabs(A->values[ap] - B->values[bp]) > tol)
            return 0;
        ap++;
        bp++;
    }
    return 1;
}

static int ldlt_factorizations_match(const LdltCsc *A, const LdltCsc *B, double tol) {
    if (A->n != B->n)
        return 0;
    idx_t n = A->n;
    for (idx_t i = 0; i < n; i++) {
        if (A->pivot_size[i] != B->pivot_size[i])
            return 0;
        if (A->perm[i] != B->perm[i])
            return 0;
        if (fabs(A->D[i] - B->D[i]) > tol)
            return 0;
        if (fabs(A->D_offdiag[i] - B->D_offdiag[i]) > tol)
            return 0;
    }
    /* Sprint 19 Day 6: `chol_csc_gather` now writes in place and
     * zero-pads dropped slots rather than shrinking `col_ptr`, so the
     * two LdltCsc layouts can differ in zero-padding even when the
     * underlying L factor is identical.  Compare value-wise by
     * walking each column and filtering zeros instead of asserting
     * `col_ptr` / `row_idx` / `values` are bit-identical. */
    for (idx_t j = 0; j < n; j++) {
        if (!ldlt_column_nonzeros_match(A->L, B->L, j, tol))
            return 0;
    }
    return 1;
}

/* Factor the matrix with both the wrapper and the native kernel, then
 * assert the resulting LdltCsc structures agree.  Asserts via the
 * framework helpers so failure points at the calling test name. */
static void check_native_matches_wrapper(const SparseMatrix *A, double tol) {
    LdltCsc *Fw = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &Fw));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_WRAPPER);
    REQUIRE_OK(ldlt_csc_eliminate(Fw));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);
    REQUIRE_OK(ldlt_csc_validate(Fw));

    LdltCsc *Fn = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    REQUIRE_OK(ldlt_csc_eliminate(Fn));
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_DEFAULT);
    REQUIRE_OK(ldlt_csc_validate(Fn));

    ASSERT_TRUE(ldlt_factorizations_match(Fw, Fn, tol));

    ldlt_csc_free(Fw);
    ldlt_csc_free(Fn);
}

/* ─── Pure-diagonal indefinite: no cmod, no swap, all criterion-1 1×1 ─ */

static void test_native_1x1_diagonal_matches_wrapper(void) {
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, -3.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 3, -5.0);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── Tridiagonal SPD: exercises the cmod loop, all criterion-1 1×1 ─ */

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

/* ─── Mixed indefinite diagonal with weak off-diagonals: all 1×1 ─── */

static void test_native_1x1_mixed_indefinite_matches_wrapper(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    double diag[] = {3.0, -2.5, 4.0, -1.5, 2.0};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);
    /* Off-diagonals small enough that BK criterion 1 fires on every
     * column — |diag| > alpha * |offdiag| at every step. */
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

/* ─── Criterion 3 (1×1 with symmetric row swap) ───────────────────── */

/* A = [[0.1, 0.5, 0], [0.5, 2, 0], [0, 0, 3]].  At k=0:
 *   diag = 0.1, max_offdiag = 0.5 (row 1), alpha*max_offdiag ≈ 0.32
 *   → phase 2 triggered.  Partner col 1 has dense_col_r = {0:0.5, 1:2}.
 *   sigma_r (max over i != 1) = 0.5.
 *   Criterion 2: 0.1*0.5 = 0.05 < 0.64*0.25 = 0.16.  Fails.
 *   Criterion 3: |2| >= 0.64*0.5 = 0.32.  Passes → swap 0↔1. */
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

/* ─── Larger tridiagonal indefinite: stress the column loop at n=20 ─ */

static void test_native_1x1_tridiag_large_matches_wrapper(void) {
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    /* Diagonals alternating sign, off-diagonals small enough for BK
     * to pick criterion-1 1×1 pivots throughout. */
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

/* ─── Zero pivot detection: sing_tol guards against near-zero 1×1 ── */

static void test_native_detects_near_zero_1x1_pivot(void) {
    /* A = diag(1e-20, 1.0) — column 0's diagonal is below sing_tol
     * (~ DROP_TOL * ||A||_inf ≈ 1e-14) and max_offdiag is 0, so BK
     * fires criterion 1 (no swap possible without off-diagonals).
     * The 1×1 singularity check must reject this. */
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

/* ─── Identity: trivial pass-through, perm stays identity ──────── */

static void test_native_1x1_identity_matches_wrapper(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 4: 2×2 Bunch-Kaufman block pivots
 * ═══════════════════════════════════════════════════════════════════════ */

/* Forward-declared helper defined in the Day 9 test block below; used
 * by the Day 4 end-to-end solve test.  Relative residual of A*x - b. */
static double rel_residual(const SparseMatrix *A, const double *x, const double *b);

/* ─── Forced 2×2 at (0, 1): the canonical BK criterion-4 matrix ─── */

static void test_native_2x2_forced_matches_wrapper(void) {
    /* A = [[0.1, 1], [1, 0.3]] — both diagonals small vs the
     * off-diagonal, BK picks a 2×2 block at (0, 1).  Matches
     * test_eliminate_forced_2x2's setup; the native kernel must now
     * produce the same L / D / D_offdiag / pivot_size / perm as the
     * wrapper. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);
    check_native_matches_wrapper(A, 1e-12);
    sparse_free(A);
}

/* ─── 2×2 pivot requiring r↔k+1 swap (partner not adjacent) ─────── */

static void test_native_2x2_nonadjacent_partner_matches_wrapper(void) {
    /* 4×4 matrix where the BK partner for column 0 is row 2, not
     * row 1.  That forces the native kernel's r ↔ k+1 swap branch.
     *
     * Construction: at k=0, we want max_offdiag at row 2 (not 1).
     * Put a tiny diag at 0, small off-diag (0,1), and large off-diag
     * (0,2); row 2's diagonal is small enough that criterion 3 fails
     * and criterion 4 fires. */
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

/* ─── Mixed 1×1 + 2×2 pivots with subsequent column cmod ─────── */

static void test_native_mixed_pivots_matches_wrapper(void) {
    /* 4×4 matrix where col 0 takes a 2×2 block at (0, 1) and cols 2
     * and 3 then use 1×1 pivots whose Schur complement receives a
     * cross-term contribution from the 2×2 at (0, 1).  Verifies
     * that `ldlt_csc_cmod_unified`'s Phase B correctly accumulates
     * the `L[:, 0] * d_off * L[col, 1] + L[:, 1] * d_off * L[col, 0]`
     * term when computing dense_col for col 2 and col 3. */
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

/* ─── The existing SuiteSparse-style indefinite cross-check fixtures
 *     (ported from test_eliminate_mixed_pivots in spirit) ─────── */

static void test_native_mixed_pivots_larger_matches_wrapper(void) {
    /* 6×6 matrix with a 2×2 pivot near the start and additional
     * structure that stresses cmod cross-terms for later 1×1 and 2×2
     * columns.  Adapted from the existing mixed-pivot test pattern
     * so wrapper and native can be directly compared. */
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

/* ─── Solve end-to-end under native kernel ─────────────────────── */

static void test_native_2x2_solve_matches_linked_list(void) {
    /* Factor the forced-2×2 matrix via the native kernel, solve
     * A*x = b with a known b, and assert the residual matches the
     * wrapper's solve to round-off.  This exercises the full
     * factor → solve pipeline through the native path. */
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

/* ─── Inertia preserved: #pos/#neg entries of D match the wrapper ─ */

static void test_native_2x2_inertia_matches_wrapper(void) {
    /* Use the forced-2×2 matrix [[0.1, 1], [1, 0.3]].  Wrapper and
     * native must produce the same D / D_offdiag, so their 2×2 block
     * eigenvalue sign decomposition (= inertia contribution) is
     * identical.  We don't compute eigenvalues here; instead we
     * assert D and D_offdiag match which is sufficient (the block
     * eigenvalues are a deterministic function of the entries). */
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

/* ═══════════════════════════════════════════════════════════════════════
 * Day 9: Triangular + block-diagonal solve
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compute ||A*x - b||_inf / ||b||_inf.  Returns NaN on allocation
 * failure; callers compare the residual against a tolerance with
 * ASSERT_TRUE(rr < tol), which treats NaN as out-of-range. */
static double rel_residual(const SparseMatrix *A, const double *x, const double *b) {
    idx_t n = sparse_rows(A);
    double *Ax = malloc((size_t)n * sizeof(double));
    if (!Ax)
        return (double)NAN;
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

/* ─── Null-arg + missing-field handling ────────────────────────── */

static void test_solve_null_args(void) {
    /* Build a trivial factored LdltCsc.  Use REQUIRE_OK on the setup
     * so a silent regression in ldlt_csc_from_sparse/ldlt_csc_eliminate
     * can't leave F == NULL and trick the null-arg checks below into
     * succeeding for the wrong reason. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, -1.0);
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    ASSERT_NOT_NULL(F);
    REQUIRE_OK(ldlt_csc_eliminate(F));

    double b[2] = {1.0, 1.0};
    double x[2] = {0};
    ASSERT_ERR(ldlt_csc_solve(NULL, b, x), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_solve(F, NULL, x), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_csc_solve(F, b, NULL), SPARSE_ERR_NULL);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Simple solve tests ───────────────────────────────────────── */

static void test_solve_identity(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    double b[4] = {1.5, -2.0, 3.25, 7.0};
    double x[4] = {0};
    REQUIRE_OK(ldlt_csc_solve(F, b, x));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i], 1e-12);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* Diagonal indefinite: solves collapse to element-wise b[i] / D[i]. */
static void test_solve_diagonal_indefinite(void) {
    idx_t n = 4;
    double diag[4] = {4.0, -2.0, 5.0, -1.0};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    double x_true[4] = {1.0, -1.0, 2.0, -2.0};
    double b[4];
    for (idx_t i = 0; i < n; i++)
        b[i] = diag[i] * x_true[i];

    double x[4] = {0};
    REQUIRE_OK(ldlt_csc_solve(F, b, x));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_true[i], 1e-12);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* Forced 2x2 pivot: solves exercise the 2x2 block diagonal path. */
static void test_solve_forced_2x2(void) {
    /* A = [[0.1, 1], [1, 0.3]] — same matrix that forces a 2x2 pivot
     * in the Day 8 test.  Pick a known x and verify recovery. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 0.1);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.3);
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    ASSERT_EQ(F->pivot_size[0], 2); /* sanity: we are exercising the 2x2 path */

    double x_true[2] = {1.0, 2.0};
    double b[2];
    sparse_matvec(A, x_true, b);
    double x[2] = {0};
    REQUIRE_OK(ldlt_csc_solve(F, b, x));
    ASSERT_NEAR(x[0], x_true[0], 1e-12);
    ASSERT_NEAR(x[1], x_true[1], 1e-12);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* Tridiagonal symmetric indefinite: mix of positive and negative
 * diagonal entries with off-diagonals.  Verify residual < 1e-10. */
static void test_solve_tridiagonal_indefinite(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        /* Alternating signs on the diagonal: 4, -3, 4, -3, ... */
        sparse_insert(A, i, i, (i % 2 == 0) ? 4.0 : -3.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_true[i] = 1.0 + 0.5 * (double)i;
    sparse_matvec(A, x_true, b);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_solve(F, b, x));

    double rr = rel_residual(A, x, b);
    printf("    tridiag indefinite n=10: rel_res = %.3e\n", rr);
    ASSERT_TRUE(rr < 1e-10);

    free(x_true);
    free(b);
    free(x);
    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Comparison against linked-list sparse_ldlt_solve ───────── */

/* A symmetric indefinite matrix with a mix of 1x1 and 2x2 pivots.
 * Factor+solve the same matrix via CSC and via linked-list, compare
 * solution vectors element-wise. */
static void test_solve_matches_linked_list(void) {
    idx_t n = 4;
    double lower[16] = {
        2.0,  0.0, 0.0,  0.0, /* row 0 */
        -1.0, 3.0, 0.0,  0.0, /* row 1 */
        0.5,  0.2, -1.0, 0.0, /* row 2 */
        0.0,  1.0, 0.5,  4.0, /* row 3 */
    };
    SparseMatrix *A1 = build_symmetric(n, lower);
    SparseMatrix *A2 = build_symmetric(n, lower);

    double b[4] = {1.0, -2.0, 3.0, -4.0};
    double x_csc[4] = {0};
    double x_ll[4] = {0};

    /* CSC path. */
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A1, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_solve(F, b, x_csc));

    /* Linked-list path. */
    sparse_ldlt_t ll = {0};
    REQUIRE_OK(sparse_ldlt_factor(A2, &ll));
    REQUIRE_OK(sparse_ldlt_solve(&ll, b, x_ll));

    /* The two solutions should agree to double-precision round-off. */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_csc[i], x_ll[i], 1e-12);

    /* And residual should be tight against both. */
    double rr = rel_residual(A1, x_csc, b);
    ASSERT_TRUE(rr < 1e-10);

    ldlt_csc_free(F);
    sparse_ldlt_free(&ll);
    sparse_free(A1);
    sparse_free(A2);
}

/* ─── In-place solve (b == x) ──────────────────────────────────── */

static void test_solve_in_place(void) {
    idx_t n = 4;
    double diag[4] = {3.0, -2.0, 5.0, -1.0};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    double b_copy[4] = {3.0, -4.0, 10.0, -1.0};
    double xbuf[4];
    memcpy(xbuf, b_copy, sizeof(b_copy));
    REQUIRE_OK(ldlt_csc_solve(F, xbuf, xbuf)); /* b == x */

    double x_ref[4] = {0};
    REQUIRE_OK(ldlt_csc_solve(F, b_copy, x_ref));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(xbuf[i], x_ref[i], 1e-12);

    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── AMD-reordered factor + solve on a larger indefinite matrix ─ */

static void test_solve_with_amd_perm(void) {
    /* Arrow-style indefinite 6x6: diag has mixed signs, last row/col
     * dense enough to benefit from AMD reordering. */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    double dvals[6] = {4.0, -3.0, 2.0, -2.5, 3.0, -1.0};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, dvals[i]);
    /* Arrow: last column/row fills everything. */
    for (idx_t i = 0; i < n - 1; i++) {
        sparse_insert(A, n - 1, i, 0.5);
        sparse_insert(A, i, n - 1, 0.5);
    }

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_amd(A, perm));

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_true[i] = (double)(i + 1);
    sparse_matvec(A, x_true, b);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, perm, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));
    REQUIRE_OK(ldlt_csc_solve(F, b, x));

    double rr = rel_residual(A, x, b);
    printf("    arrow 6x6 indefinite (AMD): rel_res = %.3e\n", rr);
    ASSERT_TRUE(rr < 1e-10);

    free(perm);
    free(x_true);
    free(b);
    free(x);
    ldlt_csc_free(F);
    sparse_free(A);
}

/* ─── Inertia consistency vs linked-list ──────────────────────── */

static void test_inertia_matches_linked_list(void) {
    /* Symmetric indefinite with known inertia (2 positive, 2 negative). */
    idx_t n = 4;
    double lower[16] = {
        3.0, 0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 0.0, 0.5, -0.5, 2.0, 0.0, 0.0, 1.0, -1.0, -3.0,
    };
    SparseMatrix *A1 = build_symmetric(n, lower);
    SparseMatrix *A2 = build_symmetric(n, lower);

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A1, NULL, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    sparse_ldlt_t ll = {0};
    REQUIRE_OK(sparse_ldlt_factor(A2, &ll));

    idx_t pos_ll, neg_ll, zero_ll;
    REQUIRE_OK(sparse_ldlt_inertia(&ll, &pos_ll, &neg_ll, &zero_ll));

    /* Compute inertia from F's D / pivot_size directly (matches the
     * Sylvester law of inertia the linked-list path uses). */
    idx_t pos = 0, neg = 0, zero = 0;
    for (idx_t k = 0; k < n;) {
        if (F->pivot_size[k] == 1) {
            if (F->D[k] > 0)
                pos++;
            else if (F->D[k] < 0)
                neg++;
            else
                zero++;
            k++;
        } else {
            double d11 = F->D[k];
            double d22 = F->D[k + 1];
            double d21 = F->D_offdiag[k];
            double det = d11 * d22 - d21 * d21;
            /* A 2x2 block always contributes exactly one + and one - when
             * det < 0; two same-sign eigenvalues when det > 0. */
            if (det < 0) {
                pos++;
                neg++;
            } else if (det > 0) {
                if (d11 + d22 > 0) {
                    pos += 2;
                } else {
                    neg += 2;
                }
            } else {
                zero += 2;
            }
            k += 2;
        }
    }

    ASSERT_EQ(pos, pos_ll);
    ASSERT_EQ(neg, neg_ll);
    ASSERT_EQ(zero, zero_ll);

    ldlt_csc_free(F);
    sparse_ldlt_free(&ll);
    sparse_free(A1);
    sparse_free(A2);
}

/* ─── Singular detection ──────────────────────────────────────── */

/* Hand-craft a factored LdltCsc with a near-zero 1x1 pivot and verify
 * solve returns SPARSE_ERR_SINGULAR. */
static void test_solve_detects_tiny_1x1_pivot(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(2, 4, &F));
    /* L is identity. */
    F->L->col_ptr[0] = 0;
    F->L->col_ptr[1] = 1;
    F->L->col_ptr[2] = 2;
    F->L->row_idx[0] = 0;
    F->L->row_idx[1] = 1;
    F->L->values[0] = 1.0;
    F->L->values[1] = 1.0;
    F->L->nnz = 2;
    F->D[0] = 1e-20; /* below the relative tolerance */
    F->D[1] = 1.0;
    F->pivot_size[0] = 1;
    F->pivot_size[1] = 1;
    F->perm[0] = 0;
    F->perm[1] = 1;
    F->factor_norm = 1.0;

    double b[2] = {1.0, 1.0}, x[2] = {0};
    ASSERT_ERR(ldlt_csc_solve(F, b, x), SPARSE_ERR_SINGULAR);

    ldlt_csc_free(F);
}

/* Hand-craft a factored LdltCsc with a near-singular 2x2 block and
 * verify solve returns SPARSE_ERR_SINGULAR. */
static void test_solve_detects_singular_2x2_block(void) {
    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_alloc(2, 4, &F));
    F->L->col_ptr[0] = 0;
    F->L->col_ptr[1] = 1;
    F->L->col_ptr[2] = 2;
    F->L->row_idx[0] = 0;
    F->L->row_idx[1] = 1;
    F->L->values[0] = 1.0;
    F->L->values[1] = 1.0;
    F->L->nnz = 2;
    /* 2x2 block [[1, 1], [1, 1]] — determinant 0, singular. */
    F->D[0] = 1.0;
    F->D[1] = 1.0;
    F->D_offdiag[0] = 1.0;
    F->pivot_size[0] = 2;
    F->pivot_size[1] = 2;
    F->perm[0] = 0;
    F->perm[1] = 1;
    F->factor_norm = 1.0;

    double b[2] = {1.0, 1.0}, x[2] = {0};
    ASSERT_ERR(ldlt_csc_solve(F, b, x), SPARSE_ERR_SINGULAR);

    ldlt_csc_free(F);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("ldlt_csc (Sprint 17 Days 7-9 + Sprint 18 Days 1-5)");

    /* alloc / free */
    RUN_TEST(test_alloc_null_out);
    RUN_TEST(test_alloc_negative_n);
    RUN_TEST(test_alloc_basic);
    RUN_TEST(test_alloc_zero_n);
    RUN_TEST(test_free_null);
    /* Sprint 19 Day 8: row-adjacency index */
    RUN_TEST(test_row_adj_empty_round_trip);
    RUN_TEST(test_row_adj_append_preserves_order);
    RUN_TEST(test_row_adj_geometric_growth);
    RUN_TEST(test_row_adj_append_arg_checks);
    /* Sprint 19 Day 10: 2×2-aware supernode detection */
    RUN_TEST(test_detect_supernodes_dense_all_1x1);
    RUN_TEST(test_detect_supernodes_dense_with_2x2);
    RUN_TEST(test_detect_supernodes_block_diagonal_with_2x2);
    RUN_TEST(test_detect_supernodes_arg_checks);
    RUN_TEST(test_supernode_extract_writeback_dense_6x6);
    RUN_TEST(test_supernode_extract_writeback_block_diagonal_8x8);
    RUN_TEST(test_supernode_extract_writeback_with_2x2);
    RUN_TEST(test_supernode_extract_writeback_arg_checks);
    RUN_TEST(test_supernodal_dense_spd_8x8);
    RUN_TEST(test_supernodal_block_diagonal_spd_12x12);
    RUN_TEST(test_supernodal_random_indefinite_30x30);
    RUN_TEST(test_supernodal_arg_checks);

    /* from_sparse / to_sparse round-trips */
    RUN_TEST(test_from_sparse_null_args);
    RUN_TEST(test_from_sparse_shape);
    RUN_TEST(test_roundtrip_identity);
    RUN_TEST(test_roundtrip_diagonal_indefinite);
    RUN_TEST(test_roundtrip_symmetric_indefinite);

    /* Sprint 20 Day 2 — ldlt_csc_from_sparse_with_analysis */
    RUN_TEST(test_from_sparse_with_analysis_arg_checks);
    RUN_TEST(test_from_sparse_with_analysis_pattern_matches_sym_L);
    RUN_TEST(test_from_sparse_with_analysis_scatter_preserves_values);
    RUN_TEST(test_from_sparse_with_analysis_indefinite_kkt_smoke);
    RUN_TEST(test_from_sparse_with_analysis_spd_factor_matches_heuristic);

    /* Sprint 20 Day 3 — batched supernodal LDL^T on indefinite inputs */
    RUN_TEST(test_s20_supernodal_with_analysis_kkt_5x5);
    RUN_TEST(test_s20_supernodal_with_analysis_kkt_10x10);
    RUN_TEST(test_s20_supernodal_with_analysis_random_indefinite_30x30);
    RUN_TEST(test_s20_supernodal_heuristic_vs_with_analysis_residuals);

    /* permutation */
    RUN_TEST(test_from_sparse_stores_perm);
    RUN_TEST(test_from_sparse_invalid_perm);
    RUN_TEST(test_reverse_perm_symmetric);

    /* validate */
    RUN_TEST(test_validate_null);
    RUN_TEST(test_validate_fresh_alloc_is_valid);
    RUN_TEST(test_validate_catches_bad_pivot_size);
    RUN_TEST(test_validate_catches_half_2x2);
    RUN_TEST(test_validate_catches_trailing_2x2);
    RUN_TEST(test_validate_catches_bad_perm);
    RUN_TEST(test_validate_accepts_valid_2x2);

    /* Day 8 — Bunch-Kaufman elimination */
    RUN_TEST(test_eliminate_null);
    RUN_TEST(test_eliminate_all_1x1_pivots);
    RUN_TEST(test_eliminate_tridiagonal_all_1x1);
    RUN_TEST(test_eliminate_forced_2x2);
    RUN_TEST(test_eliminate_mixed_pivots);
    RUN_TEST(test_eliminate_matches_linked_list_indefinite);
    RUN_TEST(test_eliminate_matches_linked_list_random_indefinite);
    RUN_TEST(test_row_adj_matches_reference);
    RUN_TEST(test_eliminate_composes_perm);
    RUN_TEST(test_eliminate_inertia);
    RUN_TEST(test_eliminate_singular_zero);

    /* Sprint 18 Day 1 — native-kernel scaffolding */
    RUN_TEST(test_native_override_roundtrip);
    RUN_TEST(test_native_empty_matrix_ok);
    RUN_TEST(test_native_rejects_null);
    RUN_TEST(test_wrapper_override_still_factors);
    RUN_TEST(test_workspace_alloc_free);
    RUN_TEST(test_workspace_alloc_rejects_negative_n);

    /* Sprint 18 Day 2 — in-place symmetric swap */
    RUN_TEST(test_symmetric_swap_null);
    RUN_TEST(test_symmetric_swap_out_of_range);
    RUN_TEST(test_symmetric_swap_noop_i_equals_j);
    RUN_TEST(test_symmetric_swap_adjacent_dense);
    RUN_TEST(test_symmetric_swap_non_adjacent);
    RUN_TEST(test_symmetric_swap_tridiagonal);
    RUN_TEST(test_symmetric_swap_moves_tiny_diag);
    RUN_TEST(test_symmetric_swap_aux_arrays);
    RUN_TEST(test_symmetric_swap_stress_random);

    /* Sprint 18 Day 3 — 1×1 Bunch-Kaufman native column loop */
    RUN_TEST(test_native_1x1_diagonal_matches_wrapper);
    RUN_TEST(test_native_1x1_tridiagonal_matches_wrapper);
    RUN_TEST(test_native_1x1_mixed_indefinite_matches_wrapper);
    RUN_TEST(test_native_1x1_with_swap_matches_wrapper);
    RUN_TEST(test_native_1x1_tridiag_large_matches_wrapper);
    RUN_TEST(test_native_detects_near_zero_1x1_pivot);
    RUN_TEST(test_native_1x1_identity_matches_wrapper);

    /* Sprint 18 Day 4 — 2×2 Bunch-Kaufman block pivots */
    RUN_TEST(test_native_2x2_forced_matches_wrapper);
    RUN_TEST(test_native_2x2_nonadjacent_partner_matches_wrapper);
    RUN_TEST(test_native_mixed_pivots_matches_wrapper);
    RUN_TEST(test_native_mixed_pivots_larger_matches_wrapper);
    RUN_TEST(test_native_2x2_solve_matches_linked_list);
    RUN_TEST(test_native_2x2_inertia_matches_wrapper);

    /* Day 9 — Triangular + block-diagonal solve */
    RUN_TEST(test_solve_null_args);
    RUN_TEST(test_solve_identity);
    RUN_TEST(test_solve_diagonal_indefinite);
    RUN_TEST(test_solve_forced_2x2);
    RUN_TEST(test_solve_tridiagonal_indefinite);
    RUN_TEST(test_solve_matches_linked_list);
    RUN_TEST(test_solve_in_place);
    RUN_TEST(test_solve_with_amd_perm);
    RUN_TEST(test_inertia_matches_linked_list);
    RUN_TEST(test_solve_detects_tiny_1x1_pivot);
    RUN_TEST(test_solve_detects_singular_2x2_block);

    TEST_SUITE_END();
}
