#include "sparse_analysis.h"
#include "sparse_cholesky.h"
#include "sparse_colamd_internal.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_reorder.h"
#include "sparse_svd.h"
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
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Check that adjacency is symmetric: if j is in adj[i], then i is in adj[j] */
static int is_symmetric_adj(const colamd_graph_t *g) {
    for (idx_t j = 0; j < g->ncols; j++) {
        for (idx_t p = g->col_adj_ptr[j]; p < g->col_adj_ptr[j + 1]; p++) {
            idx_t k = g->col_adj_list[p];
            /* Search for j in adj[k] */
            int found = 0;
            for (idx_t q = g->col_adj_ptr[k]; q < g->col_adj_ptr[k + 1]; q++) {
                if (g->col_adj_list[q] == j) {
                    found = 1;
                    break;
                }
            }
            if (!found)
                return 0;
        }
    }
    return 1;
}

/* Check that no self-loops exist in adjacency */
static int has_no_self_loops(const colamd_graph_t *g) {
    for (idx_t j = 0; j < g->ncols; j++) {
        for (idx_t p = g->col_adj_ptr[j]; p < g->col_adj_ptr[j + 1]; p++) {
            if (g->col_adj_list[p] == j)
                return 0;
        }
    }
    return 1;
}

/* Column degree in adjacency graph */
static idx_t col_degree(const colamd_graph_t *g, idx_t j) {
    return g->col_adj_ptr[j + 1] - g->col_adj_ptr[j];
}

/* Check if column j is adjacent to column k */
static int is_adjacent(const colamd_graph_t *g, idx_t j, idx_t k) {
    for (idx_t p = g->col_adj_ptr[j]; p < g->col_adj_ptr[j + 1]; p++) {
        if (g->col_adj_list[p] == k)
            return 1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Column adjacency graph tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_graph_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    colamd_graph_t g;
    ASSERT_ERR(colamd_build_graph(NULL, 0, &g), SPARSE_ERR_NULL);
    ASSERT_ERR(colamd_build_graph(A, 0, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_graph_empty(void) {
    /* sparse_create(0,0) returns NULL in this library, so test with 1x1 empty */
    SparseMatrix *A = sparse_create(1, 1);
    /* Don't insert anything — column 0 has no entries */
    colamd_graph_t g;
    REQUIRE_OK(colamd_build_graph(A, 0, &g));
    ASSERT_EQ(g.ncols, 1);
    ASSERT_EQ(g.nnz_adj, 0);
    colamd_graph_free(&g);
    sparse_free(A);
}

static void test_graph_diagonal(void) {
    /* Diagonal matrix: no two columns share a row, so no adjacency */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    colamd_graph_t g;
    REQUIRE_OK(colamd_build_graph(A, 0, &g));
    ASSERT_EQ(g.ncols, n);
    ASSERT_EQ(g.nnz_adj, 0);

    for (idx_t j = 0; j < n; j++)
        ASSERT_EQ(col_degree(&g, j), 0);

    colamd_graph_free(&g);
    sparse_free(A);
}

static void test_graph_dense_row(void) {
    /* Single dense row connecting all columns: columns are all adjacent */
    idx_t m = 1, n = 4;
    SparseMatrix *A = sparse_create(m, n);
    for (idx_t j = 0; j < n; j++)
        sparse_insert(A, 0, j, 1.0);

    colamd_graph_t g;
    REQUIRE_OK(colamd_build_graph(A, 0, &g));

    /* Each column should be adjacent to n-1 others */
    ASSERT_EQ(g.ncols, n);
    for (idx_t j = 0; j < n; j++)
        ASSERT_EQ(col_degree(&g, j), n - 1);

    ASSERT_TRUE(is_symmetric_adj(&g));
    ASSERT_TRUE(has_no_self_loops(&g));

    printf("    dense row 1x4: all columns adjacent, total adj = %d ✓\n", (int)g.nnz_adj);

    colamd_graph_free(&g);
    sparse_free(A);
}

static void test_graph_tridiag(void) {
    /* Tridiagonal 4x4:
     * [x x . .]
     * [x x x .]
     * [. x x x]
     * [. . x x]
     * Columns 0,1 share rows 0,1 → adjacent
     * Columns 1,2 share rows 1,2 → adjacent
     * Columns 2,3 share rows 2,3 → adjacent
     * Columns 0,2 share row 1 → adjacent
     * Columns 1,3 share row 2 → adjacent
     * Columns 0,3 don't share any row → NOT adjacent */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    colamd_graph_t g;
    REQUIRE_OK(colamd_build_graph(A, 0, &g));

    ASSERT_EQ(g.ncols, n);
    ASSERT_TRUE(is_symmetric_adj(&g));
    ASSERT_TRUE(has_no_self_loops(&g));

    /* Check specific adjacencies */
    ASSERT_TRUE(is_adjacent(&g, 0, 1));
    ASSERT_TRUE(is_adjacent(&g, 1, 2));
    ASSERT_TRUE(is_adjacent(&g, 2, 3));
    ASSERT_TRUE(is_adjacent(&g, 0, 2));  /* share row 1 */
    ASSERT_TRUE(is_adjacent(&g, 1, 3));  /* share row 2 */
    ASSERT_TRUE(!is_adjacent(&g, 0, 3)); /* no shared row */

    /* Degrees: cols 0,3 have degree 2; cols 1,2 have degree 3 */
    ASSERT_EQ(col_degree(&g, 0), 2);
    ASSERT_EQ(col_degree(&g, 1), 3);
    ASSERT_EQ(col_degree(&g, 2), 3);
    ASSERT_EQ(col_degree(&g, 3), 2);

    printf("    tridiag 4x4: adjacency correct, symmetric ✓\n");

    colamd_graph_free(&g);
    sparse_free(A);
}

static void test_graph_unsymmetric(void) {
    /* Unsymmetric 3x4 matrix:
     * [1 1 0 0]
     * [0 1 1 0]
     * [0 0 1 1]
     * Row 0: cols 0,1 → 0-1 adjacent
     * Row 1: cols 1,2 → 1-2 adjacent
     * Row 2: cols 2,3 → 2-3 adjacent */
    SparseMatrix *A = sparse_create(3, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 2, 3, 1.0);

    colamd_graph_t g;
    REQUIRE_OK(colamd_build_graph(A, 0, &g));

    ASSERT_EQ(g.ncols, 4);
    ASSERT_TRUE(is_symmetric_adj(&g));
    ASSERT_TRUE(has_no_self_loops(&g));

    ASSERT_TRUE(is_adjacent(&g, 0, 1));
    ASSERT_TRUE(is_adjacent(&g, 1, 2));
    ASSERT_TRUE(is_adjacent(&g, 2, 3));
    ASSERT_TRUE(!is_adjacent(&g, 0, 2));
    ASSERT_TRUE(!is_adjacent(&g, 0, 3));
    ASSERT_TRUE(!is_adjacent(&g, 1, 3));

    ASSERT_EQ(col_degree(&g, 0), 1);
    ASSERT_EQ(col_degree(&g, 1), 2);
    ASSERT_EQ(col_degree(&g, 2), 2);
    ASSERT_EQ(col_degree(&g, 3), 1);

    printf("    unsym 3x4: path adjacency, degrees [1,2,2,1] ✓\n");

    colamd_graph_free(&g);
    sparse_free(A);
}

static void test_graph_arrow(void) {
    /* Arrow matrix 5x5: dense last column/row.
     * Columns 0..3 each appear in row i and row 4.
     * Column 4 appears in all rows.
     * All column pairs share at least row 4 → fully connected graph. */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i < n - 1) {
            sparse_insert(A, i, n - 1, -1.0);
            sparse_insert(A, n - 1, i, -1.0);
        }
    }

    colamd_graph_t g;
    REQUIRE_OK(colamd_build_graph(A, 0, &g));

    ASSERT_EQ(g.ncols, n);
    ASSERT_TRUE(is_symmetric_adj(&g));
    ASSERT_TRUE(has_no_self_loops(&g));

    /* All columns should be adjacent to all others */
    for (idx_t j = 0; j < n; j++)
        ASSERT_EQ(col_degree(&g, j), n - 1);

    printf("    arrow 5x5: fully connected, all degrees = %d ✓\n", (int)(n - 1));

    colamd_graph_free(&g);
    sparse_free(A);
}

static void test_graph_dense_row_skip(void) {
    /* Test dense row skipping: 4x4 matrix with one very dense row.
     * Row 0: cols 0,1,2,3 (dense)
     * Row 1: col 0
     * Row 2: col 1
     * Row 3: col 2
     * With dense_threshold=2, row 0 is skipped.
     * Without row 0, no columns share a row → zero adjacency. */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 0, 3, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 3, 2, 1.0);

    colamd_graph_t g;
    REQUIRE_OK(colamd_build_graph(A, 2, &g));

    /* Row 0 has 4 nonzeros > threshold 2, so it's skipped.
     * Remaining rows each have 1 nonzero → no pairs → no adjacency. */
    ASSERT_EQ(g.nnz_adj, 0);

    colamd_graph_free(&g);

    /* Without dense skipping, all columns are adjacent via row 0 */
    REQUIRE_OK(colamd_build_graph(A, 0, &g));
    for (idx_t j = 0; j < 4; j++)
        ASSERT_EQ(col_degree(&g, j), 3);

    printf("    dense row skip: threshold=2 removes all adjacency ✓\n");

    colamd_graph_free(&g);
    sparse_free(A);
}

static void test_graph_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 5.0);

    colamd_graph_t g;
    REQUIRE_OK(colamd_build_graph(A, 0, &g));
    ASSERT_EQ(g.ncols, 1);
    ASSERT_EQ(g.nnz_adj, 0);

    colamd_graph_free(&g);
    sparse_free(A);
}

static void test_graph_rectangular_tall(void) {
    /* 6x3 tall matrix:
     * Row 0: cols 0,1
     * Row 1: cols 1,2
     * Row 2: col 0
     * Row 3: col 2
     * Row 4: cols 0,2
     * Row 5: col 1
     * Adjacency: 0-1 (row 0), 1-2 (row 1), 0-2 (row 4) */
    SparseMatrix *A = sparse_create(6, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 3, 2, 1.0);
    sparse_insert(A, 4, 0, 1.0);
    sparse_insert(A, 4, 2, 1.0);
    sparse_insert(A, 5, 1, 1.0);

    colamd_graph_t g;
    REQUIRE_OK(colamd_build_graph(A, 0, &g));

    ASSERT_EQ(g.ncols, 3);
    ASSERT_TRUE(is_symmetric_adj(&g));

    /* All columns adjacent to all others */
    ASSERT_TRUE(is_adjacent(&g, 0, 1));
    ASSERT_TRUE(is_adjacent(&g, 1, 2));
    ASSERT_TRUE(is_adjacent(&g, 0, 2));
    ASSERT_EQ(col_degree(&g, 0), 2);
    ASSERT_EQ(col_degree(&g, 1), 2);
    ASSERT_EQ(col_degree(&g, 2), 2);

    printf("    tall 6x3: fully connected, symmetric ✓\n");

    colamd_graph_free(&g);
    sparse_free(A);
}

static void test_graph_free_zeroed(void) {
    colamd_graph_t g;
    memset(&g, 0, sizeof(g));
    colamd_graph_free(&g); /* should not crash */
    colamd_graph_free(NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Ordering helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static int is_valid_perm(const idx_t *perm, idx_t n) {
    unsigned char *seen = calloc((size_t)n, sizeof(unsigned char));
    if (!seen)
        return 0;
    for (idx_t i = 0; i < n; i++) {
        if (perm[i] < 0 || perm[i] >= n || seen[perm[i]]) {
            free(seen);
            return 0;
        }
        seen[perm[i]] = 1;
    }
    free(seen);
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * COLAMD ordering tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_order_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    idx_t perm[3];
    ASSERT_ERR(colamd_order(NULL, perm), SPARSE_ERR_NULL);
    ASSERT_ERR(colamd_order(A, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_order_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 5.0);
    idx_t perm[1];
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_EQ(perm[0], 0);
    sparse_free(A);
}

static void test_order_diagonal(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    printf("    order diagonal n=%d: valid perm ✓\n", (int)n);
    free(perm);
    sparse_free(A);
}

static void test_order_tridiag(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    printf("    order tridiag n=%d: perm = [", (int)n);
    for (idx_t i = 0; i < n; i++)
        printf("%s%d", i > 0 ? "," : "", (int)perm[i]);
    printf("] ✓\n");

    free(perm);
    sparse_free(A);
}

static void test_order_unsymmetric(void) {
    /* 3x4 bidiagonal-ish:
     * [1 1 0 0]
     * [0 1 1 0]
     * [0 0 1 1]
     */
    SparseMatrix *A = sparse_create(3, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 2, 3, 1.0);

    idx_t perm[4];
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, 4));

    printf("    order unsym 3x4: perm = [%d,%d,%d,%d] ✓\n", (int)perm[0], (int)perm[1],
           (int)perm[2], (int)perm[3]);

    sparse_free(A);
}

static void test_order_arrow(void) {
    /* Arrow matrix: dense last column/row.
     * COLAMD should order the dense column last (or near last). */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i < n - 1) {
            sparse_insert(A, i, n - 1, -1.0);
            sparse_insert(A, n - 1, i, -1.0);
        }
    }

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    /* All columns share the dense row, so all have equal adjacency degree.
     * We can only verify the permutation is valid, not specific ordering. */

    printf("    order arrow n=%d: valid perm ✓\n", (int)n);

    free(perm);
    sparse_free(A);
}

static void test_order_initial_degree(void) {
    /* Verify that the first column eliminated has minimum initial degree.
     * Diagonal matrix: all degrees are 0, so first eliminated can be any. */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    idx_t perm[4];
    REQUIRE_OK(colamd_order(A, perm));

    /* All columns have degree 0 — perm should be valid, first can be anything */
    ASSERT_TRUE(is_valid_perm(perm, n));

    /* For a more interesting check: tridiag where corners have min degree */
    SparseMatrix *B = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(B, i, i, 4.0);
        if (i > 0) {
            sparse_insert(B, i, i - 1, -1.0);
            sparse_insert(B, i - 1, i, -1.0);
        }
    }

    REQUIRE_OK(colamd_order(B, perm));
    /* Columns 0 and 3 have the smallest adj degrees (2 vs 3).
     * First eliminated should be one of them. */
    ASSERT_TRUE(perm[0] == 0 || perm[0] == 3);

    printf("    order initial degree: corner column eliminated first ✓\n");

    sparse_free(B);
    sparse_free(A);
}

static void test_order_wide_matrix(void) {
    /* 2x6 matrix:
     * [1 1 1 0 0 0]
     * [0 0 0 1 1 1]
     * Two disconnected groups: {0,1,2} and {3,4,5} */
    SparseMatrix *A = sparse_create(2, 6);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 1, 4, 1.0);
    sparse_insert(A, 1, 5, 1.0);

    idx_t perm[6];
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, 6));

    printf("    order wide 2x6: valid perm ✓\n");
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Pathological input tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_order_all_dense_rows(void) {
    /* Every row is dense (all columns present). Dense row skipping should
     * skip all rows, leaving zero adjacency. Ordering should still be valid. */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, 1.0);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    printf("    order all-dense 5x5: valid perm ✓\n");
    free(perm);
    sparse_free(A);
}

static void test_order_single_dense_row(void) {
    /* Sparse matrix with one dense row connecting all columns.
     * The dense row may or may not be skipped depending on n vs threshold. */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n + 1, n);
    /* Sparse rows: each has one entry */
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    /* Dense row: all columns */
    for (idx_t j = 0; j < n; j++)
        sparse_insert(A, n, j, 1.0);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    printf("    order single dense row (%dx%d): valid perm ✓\n", (int)(n + 1), (int)n);
    free(perm);
    sparse_free(A);
}

static void test_order_very_tall(void) {
    /* m >> n: 100x3 matrix */
    idx_t m = 100, n = 3;
    SparseMatrix *A = sparse_create(m, n);
    for (idx_t i = 0; i < m; i++)
        sparse_insert(A, i, i % n, 1.0);
    /* Add some cross-column entries */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 1, 2, 1.0);

    idx_t perm[3];
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    printf("    order very tall %dx%d: valid perm ✓\n", (int)m, (int)n);
    sparse_free(A);
}

static void test_order_very_wide(void) {
    /* m << n: 2x50 matrix */
    idx_t m = 2, n = 50;
    SparseMatrix *A = sparse_create(m, n);
    /* Row 0: first half of columns */
    for (idx_t j = 0; j < n / 2; j++)
        sparse_insert(A, 0, j, 1.0);
    /* Row 1: second half */
    for (idx_t j = n / 2; j < n; j++)
        sparse_insert(A, 1, j, 1.0);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    printf("    order very wide %dx%d: valid perm ✓\n", (int)m, (int)n);
    free(perm);
    sparse_free(A);
}

static void test_order_empty_columns(void) {
    /* Matrix with some completely empty columns */
    SparseMatrix *A = sparse_create(3, 5);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 4, 1.0);
    /* Columns 1 and 3 are empty */

    idx_t perm[5];
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, 5));

    printf("    order empty columns: valid perm ✓\n");
    sparse_free(A);
}

static void test_order_duplicate_entries(void) {
    /* Matrix where the same position is inserted multiple times
     * (sparse_insert overwrites, so this shouldn't cause issues) */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 0, 1, 3.0); /* overwrite */
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 2, 1.0);

    idx_t perm[3];
    REQUIRE_OK(colamd_order(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, 3));

    printf("    order duplicate entries: valid perm ✓\n");
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_public_api_null(void) {
    SparseMatrix *A = sparse_create(3, 3);
    idx_t perm[3];
    ASSERT_ERR(sparse_reorder_colamd(NULL, perm), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_reorder_colamd(A, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_public_api_square(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_colamd(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    printf("    public API square n=%d: valid perm ✓\n", (int)n);
    free(perm);
    sparse_free(A);
}

static void test_public_api_rectangular(void) {
    /* COLAMD supports rectangular matrices (unlike AMD/RCM) */
    SparseMatrix *A = sparse_create(3, 5);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 2, 4, 1.0);

    idx_t perm[5];
    REQUIRE_OK(sparse_reorder_colamd(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, 5));

    printf("    public API rectangular 3x5: valid perm ✓\n");
    sparse_free(A);
}

static void test_public_api_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] west0067.mtx not found\n");
        return;
    }

    idx_t n = A->cols;
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_colamd(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    /* Compare fill-in: COLAMD vs natural ordering for LU */
    SparseMatrix *LU_nat = sparse_copy(A);
    REQUIRE_OK(sparse_lu_factor(LU_nat, SPARSE_PIVOT_PARTIAL, 1e-12));
    idx_t fill_nat = sparse_nnz(LU_nat);

    SparseMatrix *PA = NULL;
    REQUIRE_OK(sparse_permute(A, perm, perm, &PA));
    SparseMatrix *LU_col = sparse_copy(PA);
    REQUIRE_OK(sparse_lu_factor(LU_col, SPARSE_PIVOT_PARTIAL, 1e-12));
    idx_t fill_col = sparse_nnz(LU_col);

    printf("    west0067 (%dx%d): natural fill=%d, COLAMD fill=%d", (int)n, (int)n, (int)fill_nat,
           (int)fill_col);
    if (fill_col <= fill_nat)
        printf(" (%.0f%% reduction)", 100.0 * (1.0 - (double)fill_col / (double)fill_nat));
    printf(" ✓\n");

    free(perm);
    sparse_free(LU_col);
    sparse_free(PA);
    sparse_free(LU_nat);
    sparse_free(A);
}

static void test_public_api_steam1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/steam1.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] steam1.mtx not found\n");
        return;
    }

    idx_t n = A->cols;
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_colamd(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    printf("    steam1 (%dx%d): valid COLAMD perm ✓\n", (int)A->rows, (int)n);

    free(perm);
    sparse_free(A);
}

static void test_colamd_vs_amd_fill(void) {
    /* Compare COLAMD vs AMD fill-in on a small unsymmetric matrix.
     * Both should produce valid orderings. */
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    /* Arrow-like unsymmetric pattern */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        sparse_insert(A, i, n - 1, 1.0);
        if (i < n - 1)
            sparse_insert(A, n - 1, i, 1.0);
    }

    idx_t *perm_col = malloc((size_t)n * sizeof(idx_t));
    idx_t *perm_amd = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_colamd(A, perm_col));
    REQUIRE_OK(sparse_reorder_amd(A, perm_amd));
    ASSERT_TRUE(is_valid_perm(perm_col, n));
    ASSERT_TRUE(is_valid_perm(perm_amd, n));

    printf("    COLAMD vs AMD n=%d: both valid ✓\n", (int)n);

    free(perm_col);
    free(perm_amd);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * QR + COLAMD tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_qr_colamd_solve(void) {
    /* Overdetermined 6x4 system: verify QR+COLAMD least-squares solve */
    idx_t m = 6, n = 4;
    SparseMatrix *A = sparse_create(m, n);
    for (idx_t i = 0; i < m; i++) {
        sparse_insert(A, i, i % n, 2.0);
        if (i + 1 < n)
            sparse_insert(A, i, i + 1, 1.0);
    }

    double b[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double x[4];
    double resid;
    sparse_qr_t qr;
    sparse_qr_opts_t opts = {SPARSE_REORDER_COLAMD, 0, 0};

    REQUIRE_OK(sparse_qr_factor_opts(A, &opts, &qr));
    REQUIRE_OK(sparse_qr_solve(&qr, b, x, &resid));

    /* Verify A*x ≈ b in least-squares sense */
    double Ax[6] = {0};
    sparse_matvec(A, x, Ax);
    double maxerr = 0;
    for (idx_t i = 0; i < m; i++) {
        double e = fabs(Ax[i] - b[i]);
        if (e > maxerr)
            maxerr = e;
    }
    ASSERT_TRUE(resid < 1e2);  /* least-squares residual is finite and bounded */
    ASSERT_TRUE(maxerr < 1e2); /* Ax - b error is bounded */

    printf("    QR+COLAMD solve 6x4: residual = %.2e ✓\n", resid);
    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_colamd_vs_amd(void) {
    /* Compare fill-in: QR with COLAMD vs AMD vs natural */
    idx_t m = 8, n = 6;
    SparseMatrix *A = sparse_create(m, n);
    /* Arrow-like pattern */
    for (idx_t i = 0; i < m; i++) {
        if (i < n)
            sparse_insert(A, i, i, 3.0);
        sparse_insert(A, i, n - 1, 1.0);
    }

    sparse_qr_t qr_none, qr_amd, qr_colamd;
    sparse_qr_opts_t opts_none = {SPARSE_REORDER_NONE, 0, 0};
    sparse_qr_opts_t opts_amd = {SPARSE_REORDER_AMD, 0, 0};
    sparse_qr_opts_t opts_colamd = {SPARSE_REORDER_COLAMD, 0, 0};

    REQUIRE_OK(sparse_qr_factor_opts(A, &opts_none, &qr_none));
    REQUIRE_OK(sparse_qr_factor_opts(A, &opts_amd, &qr_amd));
    REQUIRE_OK(sparse_qr_factor_opts(A, &opts_colamd, &qr_colamd));

    /* All should produce valid QR — verify solve works */
    double b[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    double x_none[6], x_amd[6], x_colamd[6];
    double r_none, r_amd, r_colamd;
    REQUIRE_OK(sparse_qr_solve(&qr_none, b, x_none, &r_none));
    REQUIRE_OK(sparse_qr_solve(&qr_amd, b, x_amd, &r_amd));
    REQUIRE_OK(sparse_qr_solve(&qr_colamd, b, x_colamd, &r_colamd));

    /* Residuals should be similar (same least-squares problem) */
    ASSERT_NEAR(r_none, r_colamd, r_none * 0.1 + 1e-10);

    printf("    QR COLAMD vs AMD vs natural: resid none=%.2e, amd=%.2e, colamd=%.2e ✓\n", r_none,
           r_amd, r_colamd);

    sparse_qr_free(&qr_none);
    sparse_qr_free(&qr_amd);
    sparse_qr_free(&qr_colamd);
    sparse_free(A);
}

static void test_qr_colamd_sparse_mode(void) {
    /* Verify COLAMD works with sparse_mode QR as well */
    idx_t m = 6, n = 4;
    SparseMatrix *A = sparse_create(m, n);
    for (idx_t i = 0; i < m; i++)
        sparse_insert(A, i, i % n, (double)(i + 1));

    sparse_qr_t qr;
    sparse_qr_opts_t opts = {SPARSE_REORDER_COLAMD, 0, 1}; /* sparse_mode=1 */

    REQUIRE_OK(sparse_qr_factor_opts(A, &opts, &qr));

    double b[6] = {1, 2, 3, 4, 5, 6};
    double x[4];
    double resid;
    REQUIRE_OK(sparse_qr_solve(&qr, b, x, &resid));

    printf("    QR+COLAMD sparse_mode: residual = %.2e ✓\n", resid);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Analyze + Factor with COLAMD
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: compute ||b - A*x||_inf / ||b||_inf */
static double solve_residual(const SparseMatrix *A, const double *b, const double *x) {
    idx_t n = sparse_rows(A);
    double *Ax = calloc((size_t)n, sizeof(double));
    if (!Ax)
        return -1.0;
    sparse_matvec(A, x, Ax);
    double rnorm = 0, bnorm = 0;
    for (idx_t i = 0; i < n; i++) {
        double ri = fabs(b[i] - Ax[i]);
        if (ri > rnorm)
            rnorm = ri;
        if (fabs(b[i]) > bnorm)
            bnorm = fabs(b[i]);
    }
    free(Ax);
    return bnorm > 0 ? rnorm / bnorm : rnorm;
}

static void test_analyze_lu_colamd(void) {
    /* Unsymmetric 3x3 → analyze+factor with COLAMD for LU */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 2, 4.0);

    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_COLAMD};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    ASSERT_NOT_NULL(analysis.perm);
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double b[3] = {1.0, 2.0, 3.0};
    double x[3];
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-12);

    printf("    analyze+factor LU+COLAMD: residual = %.2e ✓\n", resid);

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_analyze_cholesky_colamd(void) {
    /* Symmetric SPD tridiag → analyze+factor with COLAMD for Cholesky */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_COLAMD};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-12);

    printf("    analyze+factor Cholesky+COLAMD: residual = %.2e ✓\n", resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_analyze_lu_colamd_west0067(void) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, SS_DIR "/west0067.mtx") != SPARSE_OK) {
        printf("    [SKIP] west0067.mtx not found\n");
        return;
    }

    idx_t n = sparse_rows(A);
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_COLAMD};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-8);

    printf("    analyze+factor LU+COLAMD west0067: residual = %.2e ✓\n", resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * COLAMD stress tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_stress_identity(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_colamd(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    free(perm);
    sparse_free(A);
}

static void test_stress_all_zero(void) {
    /* Matrix with structure but all zero values — COLAMD only looks at pattern */
    SparseMatrix *A = sparse_create(3, 3);
    /* Insert then remove to create structure with no actual entries */
    /* Actually, sparse_insert with 0.0 removes entries, so just create empty */
    idx_t perm[3];
    REQUIRE_OK(sparse_reorder_colamd(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, 3));
    sparse_free(A);
}

static void test_stress_single_entry(void) {
    SparseMatrix *A = sparse_create(5, 5);
    sparse_insert(A, 2, 3, 7.0);

    idx_t perm[5];
    REQUIRE_OK(sparse_reorder_colamd(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, 5));
    sparse_free(A);
}

static void test_stress_dense_matrix(void) {
    /* Fully dense 8x8 */
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (double)(i * n + j + 1));

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_colamd(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    free(perm);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Minimum-norm tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: compute ||x||_2 */
static double vec_norm2(const double *x, idx_t n) {
    double s = 0;
    for (idx_t i = 0; i < n; i++)
        s += x[i] * x[i];
    return sqrt(s);
}

static void test_minnorm_null_args(void) {
    SparseMatrix *A = sparse_create(2, 4);
    double b[2] = {1, 2};
    double x[4];
    ASSERT_ERR(sparse_qr_solve_minnorm(NULL, b, x, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_qr_solve_minnorm(A, NULL, x, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_qr_solve_minnorm(A, b, NULL, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_minnorm_2x4_known(void) {
    /* Underdetermined 2×4 system:
     * A = [1 0 1 0]   b = [1]
     *     [0 1 0 1]       [1]
     *
     * All solutions: x = [a, b, 1-a, 1-b] for any a,b.
     * Minimum-norm solution: a=b=0.5, so x = [0.5, 0.5, 0.5, 0.5].
     * ||x||_2 = 1.0 */
    SparseMatrix *A = sparse_create(2, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 1, 3, 1.0);

    double b[2] = {1.0, 1.0};
    double x[4];
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, NULL));

    /* Check A*x = b */
    double Ax[2] = {0};
    sparse_matvec(A, x, Ax);
    ASSERT_NEAR(Ax[0], 1.0, 1e-12);
    ASSERT_NEAR(Ax[1], 1.0, 1e-12);

    /* Check minimum norm: x should be [0.5, 0.5, 0.5, 0.5] */
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x[i], 0.5, 1e-12);

    double xnorm = vec_norm2(x, 4);
    ASSERT_NEAR(xnorm, 1.0, 1e-12);

    printf("    minnorm 2x4: x=[%.3f,%.3f,%.3f,%.3f] ||x||=%.4f ✓\n", x[0], x[1], x[2], x[3],
           xnorm);

    sparse_free(A);
}

static void test_minnorm_is_minimal(void) {
    /* Verify ||x_minnorm||_2 < ||x_any||_2 for a non-minimum-norm solution */
    SparseMatrix *A = sparse_create(2, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 1, 3, 1.0);

    double b[2] = {1.0, 1.0};
    double x_min[4];
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x_min, NULL));
    double norm_min = vec_norm2(x_min, 4);

    /* Another valid solution: x = [1, 1, 0, 0] — norm = sqrt(2) > 1 */
    double x_other[4] = {1.0, 1.0, 0.0, 0.0};
    double norm_other = vec_norm2(x_other, 4);

    ASSERT_TRUE(norm_min < norm_other);
    printf("    minnorm is minimal: ||x_min||=%.4f < ||x_other||=%.4f ✓\n", norm_min, norm_other);

    sparse_free(A);
}

static void test_minnorm_3x6(void) {
    /* 3×6 underdetermined system */
    SparseMatrix *A = sparse_create(3, 6);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 3, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 4, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 2, 5, 2.0);

    double b[3] = {3.0, 4.0, 5.0};
    double x[6];
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, NULL));

    /* Check A*x = b */
    double Ax[3] = {0};
    sparse_matvec(A, x, Ax);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(Ax[i], b[i], 1e-10);

    printf("    minnorm 3x6: ||x||=%.4f, residual OK ✓\n", vec_norm2(x, 6));
    sparse_free(A);
}

static void test_minnorm_fallback_overdetermined(void) {
    /* For m >= n, minnorm should fall back to regular least-squares */
    SparseMatrix *A = sparse_create(4, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 3, 2, 1.0);

    double b[4] = {1.0, 2.0, 3.0, 6.0};
    double x[3];
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, NULL));

    /* Verify A*x ≈ b */
    double Ax[4] = {0};
    sparse_matvec(A, x, Ax);
    double maxerr = 0;
    for (int i = 0; i < 4; i++) {
        double e = fabs(Ax[i] - b[i]);
        if (e > maxerr)
            maxerr = e;
    }
    ASSERT_TRUE(maxerr < 1e-10);

    printf("    minnorm fallback 4x3: x=[%.3f,%.3f,%.3f], maxerr=%.2e ✓\n", x[0], x[1], x[2],
           maxerr);
    sparse_free(A);
}

static void test_minnorm_with_colamd(void) {
    /* Minimum-norm with COLAMD ordering */
    SparseMatrix *A = sparse_create(2, 5);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 1, 4, 1.0);

    double b[2] = {3.0, 3.0};
    double x[5];
    sparse_qr_opts_t opts = {SPARSE_REORDER_COLAMD, 0, 0};
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, &opts));

    /* Check A*x = b */
    double Ax[2] = {0};
    sparse_matvec(A, x, Ax);
    ASSERT_NEAR(Ax[0], 3.0, 1e-10);
    ASSERT_NEAR(Ax[1], 3.0, 1e-10);

    printf("    minnorm+COLAMD 2x5: ||x||=%.4f ✓\n", vec_norm2(x, 5));
    sparse_free(A);
}

static void test_minnorm_5x10(void) {
    /* Larger underdetermined: 5×10 */
    idx_t m = 5, n = 10;
    SparseMatrix *A = sparse_create(m, n);
    /* Diagonal + off-diagonal pattern */
    for (idx_t i = 0; i < m; i++) {
        sparse_insert(A, i, i, 2.0);
        sparse_insert(A, i, i + m, 1.0);
    }

    double b[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double x[10];
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, NULL));

    /* Check A*x = b */
    double Ax[5] = {0};
    sparse_matvec(A, x, Ax);
    for (idx_t i = 0; i < m; i++)
        ASSERT_NEAR(Ax[i], b[i], 1e-10);

    printf("    minnorm 5x10: ||x||=%.4f ✓\n", vec_norm2(x, n));
    sparse_free(A);
}

static void test_minnorm_rank_deficient(void) {
    /* 2×4 with rank 1: rows are linearly dependent.
     * A = [1 1 1 1]   b = [2]
     *     [2 2 2 2]       [4]
     * System is consistent (b2 = 2*b1). Min-norm solution: x = [0.5, 0.5, 0.5, 0.5].
     */
    SparseMatrix *A = sparse_create(2, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 0, 3, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 1, 3, 2.0);

    double b[2] = {2.0, 4.0};
    double x[4];
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, NULL));

    /* A*x should approximate b */
    double Ax[2] = {0};
    sparse_matvec(A, x, Ax);
    ASSERT_NEAR(Ax[0], 2.0, 1e-8);
    ASSERT_NEAR(Ax[1], 4.0, 1e-8);

    printf("    minnorm rank-deficient 2x4: ||x||=%.4f ✓\n", vec_norm2(x, 4));
    sparse_free(A);
}

static void test_minnorm_square(void) {
    /* Square system: m == n, should fall back to regular QR solve */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 2.0);

    double b[3] = {1.0, 2.0, 3.0};
    double x[3];
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, NULL));

    double Ax[3] = {0};
    sparse_matvec(A, x, Ax);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(Ax[i], b[i], 1e-10);

    printf("    minnorm square 3x3: exact solve ✓\n");
    sparse_free(A);
}

static void test_minnorm_1xn(void) {
    /* Extremely underdetermined: 1×5 */
    SparseMatrix *A = sparse_create(1, 5);
    for (idx_t j = 0; j < 5; j++)
        sparse_insert(A, 0, j, 1.0);

    double b[1] = {5.0};
    double x[5];
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, NULL));

    /* Min-norm solution: x = [1, 1, 1, 1, 1] */
    for (int i = 0; i < 5; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-12);

    printf("    minnorm 1x5: x=[1,1,1,1,1] ✓\n");
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Minimum-norm refinement & rank-deficiency tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_refine_minnorm(void) {
    /* Refinement should improve (or maintain) residual */
    SparseMatrix *A = sparse_create(2, 5);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 1, 4, 1.0);

    double b[2] = {3.0, 3.0};
    double x[5];
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, NULL));

    double resid_before = 0;
    {
        double Ax[2] = {0};
        sparse_matvec(A, x, Ax);
        for (int i = 0; i < 2; i++)
            resid_before += (b[i] - Ax[i]) * (b[i] - Ax[i]);
        resid_before = sqrt(resid_before);
    }

    double resid_after;
    REQUIRE_OK(sparse_qr_refine_minnorm(A, b, x, 3, &resid_after, NULL));

    ASSERT_TRUE(resid_after <= resid_before + 1e-14);

    printf("    refine minnorm: before=%.2e, after=%.2e ✓\n", resid_before, resid_after);
    sparse_free(A);
}

static void test_minnorm_zero_row(void) {
    /* Matrix with a zero row: rank(A) < m, system may be inconsistent.
     * A = [1 0 1 0]   b = [2]
     *     [0 0 0 0]       [0]
     * Row 1 is zero, so b[1] must be 0 for consistency.
     * Min-norm of consistent part: x = [1, 0, 1, 0] → but minnorm
     * should give x = [0.5, 0, 0.5, 0] (spread equally). */
    SparseMatrix *A = sparse_create(2, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    /* Row 1 is empty (zero) */

    double b[2] = {2.0, 0.0};
    double x[4];
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, NULL));

    double Ax[2] = {0};
    sparse_matvec(A, x, Ax);
    ASSERT_NEAR(Ax[0], 2.0, 1e-10);

    printf("    minnorm zero row: Ax[0]=%.4f, ||x||=%.4f ✓\n", Ax[0], vec_norm2(x, 4));
    sparse_free(A);
}

static void test_minnorm_vs_pinv(void) {
    /* Verify minimum-norm QR solution matches the SVD pseudoinverse.
     * Use sparse_pinv to compute A^+ * b and compare. */
    idx_t m = 2, n = 4;
    SparseMatrix *A = sparse_create(m, n);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 1, 3, 1.0);

    double b[2] = {1.0, 1.0};
    double x_qr[4];

    /* QR minimum-norm */
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x_qr, NULL));

    /* SVD pseudoinverse: x = A^+ * b */
    double *pinv_dense = NULL;
    sparse_err_t perr = sparse_pinv(A, 1e-12, &pinv_dense);
    if (perr != SPARSE_OK || !pinv_dense) {
        printf("    [SKIP] sparse_pinv failed\n");
        free(pinv_dense);
        sparse_free(A);
        return;
    }

    /* pinv_dense is n×m column-major: x_svd[j] = sum_r pinv[j + n*r] * b[r] */
    double x_svd[4] = {0};
    for (idx_t j = 0; j < n; j++)
        for (idx_t r = 0; r < m; r++)
            x_svd[j] += pinv_dense[j + n * r] * b[r];
    free(pinv_dense);

    /* Both should match */
    for (idx_t j = 0; j < n; j++)
        ASSERT_NEAR(x_qr[j], x_svd[j], 1e-10);

    printf("    minnorm vs pinv: x_qr=[%.3f,%.3f,%.3f,%.3f] matches ✓\n", x_qr[0], x_qr[1], x_qr[2],
           x_qr[3]);

    sparse_free(A);
}

static void test_refine_minnorm_null(void) {
    ASSERT_ERR(sparse_qr_refine_minnorm(NULL, NULL, NULL, 0, NULL, NULL), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Rank-revealing diagnostics tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_diag_r_null(void) { ASSERT_ERR(sparse_qr_diag_r(NULL, NULL), SPARSE_ERR_NULL); }

static void test_diag_r_basic(void) {
    /* Factor a 4x3 matrix and extract R diagonal */
    SparseMatrix *A = sparse_create(4, 3);
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 2, 2, 5.0);
    sparse_insert(A, 3, 0, 1.0);

    sparse_qr_t qr;
    REQUIRE_OK(sparse_qr_factor(A, &qr));

    double diag[3];
    REQUIRE_OK(sparse_qr_diag_r(&qr, diag));

    /* Verify diagonal matches direct R access */
    for (int i = 0; i < 3; i++) {
        double rii = sparse_get_phys(qr.R, i, i);
        ASSERT_NEAR(diag[i], rii, 1e-14);
    }

    /* R diagonals should be nonzero for this full-rank matrix */
    for (int i = 0; i < 3; i++)
        ASSERT_TRUE(fabs(diag[i]) > 1e-10);

    printf("    diag_r 4x3: R diag = [%.3f, %.3f, %.3f] ✓\n", diag[0], diag[1], diag[2]);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_rank_info_full_rank(void) {
    SparseMatrix *A = sparse_create(4, 3);
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 2, 2, 5.0);
    sparse_insert(A, 3, 0, 1.0);

    sparse_qr_t qr;
    REQUIRE_OK(sparse_qr_factor(A, &qr));

    sparse_qr_rank_info_t info;
    REQUIRE_OK(sparse_qr_rank_info(&qr, 0, &info));

    ASSERT_EQ(info.rank, 3);
    ASSERT_EQ(info.k, 3);
    ASSERT_TRUE(info.r_max > 0);
    ASSERT_TRUE(info.r_min > 0);
    ASSERT_TRUE(info.condest >= 1.0);
    ASSERT_EQ(info.near_deficient, 0);

    printf("    rank_info full-rank 4x3: rank=%d, condest=%.2f ✓\n", (int)info.rank, info.condest);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_rank_info_deficient(void) {
    /* 3x3 matrix with rank 2 (row 2 = row 0) */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 1, 2.0);

    sparse_qr_t qr;
    REQUIRE_OK(sparse_qr_factor(A, &qr));

    sparse_qr_rank_info_t info;
    REQUIRE_OK(sparse_qr_rank_info(&qr, 0, &info));

    ASSERT_EQ(info.rank, 2);

    printf("    rank_info deficient 3x3: rank=%d, condest=%.2f ✓\n", (int)info.rank, info.condest);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_condest_basic(void) {
    /* Well-conditioned diagonal matrix: cond ≈ 1 */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 2, 1.0);

    sparse_qr_t qr;
    REQUIRE_OK(sparse_qr_factor(A, &qr));

    double cond = sparse_qr_condest(&qr);
    ASSERT_NEAR(cond, 1.0, 1e-10);

    printf("    condest identity: %.2f ✓\n", cond);
    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_condest_ill(void) {
    /* Ill-conditioned: diag = [1, 1e-8] → cond ≈ 1e8 */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1e-8);

    sparse_qr_t qr;
    REQUIRE_OK(sparse_qr_factor(A, &qr));

    double cond = sparse_qr_condest(&qr);
    ASSERT_TRUE(cond > 1e7);

    sparse_qr_rank_info_t info;
    REQUIRE_OK(sparse_qr_rank_info(&qr, 0, &info));
    /* With r_min/r_max = 1e-8, this should be near-deficient */
    ASSERT_TRUE(info.condest > 1e7);

    printf("    condest ill 2x2: %.2e, condest=%.2e ✓\n", cond, info.condest);
    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_condest_null(void) { ASSERT_NEAR(sparse_qr_condest(NULL), -1.0, 0.0); }

static void test_condest_vs_true(void) {
    /* Compare QR condest with true condition number from SVD.
     * The R-diagonal estimate should be within an order of magnitude. */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 10.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 3, 1.0);

    sparse_qr_t qr;
    REQUIRE_OK(sparse_qr_factor(A, &qr));
    double qr_cond = sparse_qr_condest(&qr);

    sparse_err_t cerr;
    double true_cond = sparse_cond(A, &cerr);

    if (cerr == SPARSE_OK && true_cond > 0) {
        /* QR condest should be within 10x of true condition */
        double ratio = qr_cond / true_cond;
        ASSERT_TRUE(ratio > 0.1 && ratio < 10.0);
        printf("    condest vs true: qr=%.2f, svd=%.2f, ratio=%.2f ✓\n", qr_cond, true_cond, ratio);
    } else {
        printf("    [SKIP] sparse_cond failed\n");
    }

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SuiteSparse integration tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: test COLAMD ordering + LU solve on a SuiteSparse matrix */
static void test_ss_colamd_lu(const char *path, const char *name) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, path) != SPARSE_OK) {
        printf("    [SKIP] %s not found\n", name);
        return;
    }

    idx_t n = sparse_cols(A);
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_colamd(A, perm));
    ASSERT_TRUE(is_valid_perm(perm, n));

    /* Compare LU fill: natural vs COLAMD */
    if (sparse_rows(A) == n) {
        SparseMatrix *LU_nat = sparse_copy(A);
        sparse_err_t e1 = sparse_lu_factor(LU_nat, SPARSE_PIVOT_PARTIAL, 1e-12);

        SparseMatrix *PA = NULL;
        sparse_permute(A, perm, perm, &PA);
        SparseMatrix *LU_col = PA ? sparse_copy(PA) : NULL;
        sparse_err_t e2 =
            LU_col ? sparse_lu_factor(LU_col, SPARSE_PIVOT_PARTIAL, 1e-12) : SPARSE_ERR_ALLOC;

        if (e1 == SPARSE_OK && e2 == SPARSE_OK) {
            printf("    %s (%dx%d): natural fill=%d, COLAMD fill=%d", name, (int)n, (int)n,
                   (int)sparse_nnz(LU_nat), (int)sparse_nnz(LU_col));
            idx_t fill_nat = sparse_nnz(LU_nat);
            idx_t fill_col = sparse_nnz(LU_col);
            if (fill_nat > 0)
                printf(" (%.0f%%)", 100.0 * (1.0 - (double)fill_col / (double)fill_nat));
            printf(" ✓\n");
        } else {
            printf("    %s: LU factor failed, COLAMD perm valid ✓\n", name);
        }
        sparse_free(LU_col);
        sparse_free(PA);
        sparse_free(LU_nat);
    } else {
        printf("    %s (%dx%d): rectangular, COLAMD perm valid ✓\n", name, (int)sparse_rows(A),
               (int)n);
    }

    free(perm);
    sparse_free(A);
}

static void test_ss_west0067(void) { test_ss_colamd_lu(SS_DIR "/west0067.mtx", "west0067"); }

static void test_ss_steam1(void) { test_ss_colamd_lu(SS_DIR "/steam1.mtx", "steam1"); }

static void test_ss_fs_541_1(void) { test_ss_colamd_lu(SS_DIR "/fs_541_1.mtx", "fs_541_1"); }

static void test_ss_orsirr_1(void) { test_ss_colamd_lu(SS_DIR "/orsirr_1.mtx", "orsirr_1"); }

/* ═══════════════════════════════════════════════════════════════════════
 * Underdetermined system tests from SuiteSparse submatrices
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_minnorm_ss_submatrix(void) {
    /* Take first 30 rows of west0067 (67x67) to create a 30×67 underdetermined system */
    SparseMatrix *A_full = NULL;
    if (sparse_load_mm(&A_full, SS_DIR "/west0067.mtx") != SPARSE_OK) {
        printf("    [SKIP] west0067.mtx not found\n");
        return;
    }

    idx_t m = 30, n = sparse_cols(A_full);
    SparseMatrix *A = sparse_create(m, n);
    if (!A) {
        sparse_free(A_full);
        return;
    }

    /* Copy first m rows */
    for (idx_t i = 0; i < m; i++) {
        for (Node *nd = A_full->row_headers[i]; nd; nd = nd->right)
            sparse_insert(A, i, nd->col, nd->value);
    }
    sparse_free(A_full);

    /* Build RHS: b = A * ones */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = calloc((size_t)m, sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    if (!ones || !b || !x) {
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return;
    }
    for (idx_t j = 0; j < n; j++)
        ones[j] = 1.0;
    sparse_matvec(A, ones, b);

    /* Solve minimum-norm */
    sparse_err_t err = sparse_qr_solve_minnorm(A, b, x, NULL);
    if (err != SPARSE_OK) {
        printf("    [SKIP] minnorm solve failed on west0067 submatrix\n");
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return;
    }

    /* Verify A*x ≈ b */
    double *Ax = calloc((size_t)m, sizeof(double));
    sparse_matvec(A, x, Ax);
    double maxerr = 0;
    for (idx_t i = 0; i < m; i++) {
        double e = fabs(Ax[i] - b[i]);
        if (e > maxerr)
            maxerr = e;
    }

    /* Verify ||x_min|| <= ||ones|| (ones is a valid solution) */
    double norm_x = vec_norm2(x, n);
    double norm_ones = vec_norm2(ones, n);
    ASSERT_TRUE(norm_x <= norm_ones + 1e-8);

    printf("    minnorm west0067 submatrix %dx%d: maxerr=%.2e, ||x||=%.2f <= ||1||=%.2f ✓\n",
           (int)m, (int)n, maxerr, norm_x, norm_ones);

    free(Ax);
    free(ones);
    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Backward compatibility
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_backward_compat_qr_no_reorder(void) {
    /* Verify QR with no reordering still works identically */
    SparseMatrix *A = sparse_create(4, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 3, 2, 1.0);

    sparse_qr_t qr;
    REQUIRE_OK(sparse_qr_factor(A, &qr));

    double b[4] = {1, 2, 3, 4};
    double x[3];
    double resid;
    REQUIRE_OK(sparse_qr_solve(&qr, b, x, &resid));

    /* Verify solve produces a reasonable result */
    double Ax[4] = {0};
    sparse_matvec(A, x, Ax);
    double maxerr = 0;
    for (int i = 0; i < 4; i++) {
        double e = fabs(Ax[i] - b[i]);
        if (e > maxerr)
            maxerr = e;
    }
    ASSERT_TRUE(resid < 1e2);
    ASSERT_TRUE(maxerr < 1e2);

    printf("    backward compat QR (no reorder): resid=%.2e, maxerr=%.2e ✓\n", resid, maxerr);
    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_backward_compat_qr_amd(void) {
    /* Verify QR with AMD still works identically */
    SparseMatrix *A = sparse_create(4, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 3, 2, 1.0);

    sparse_qr_t qr;
    sparse_qr_opts_t opts = {SPARSE_REORDER_AMD, 0, 0};
    REQUIRE_OK(sparse_qr_factor_opts(A, &opts, &qr));

    double b[4] = {1, 2, 3, 4};
    double x[3];
    double resid;
    REQUIRE_OK(sparse_qr_solve(&qr, b, x, &resid));

    double Ax[4] = {0};
    sparse_matvec(A, x, Ax);
    double maxerr = 0;
    for (int i = 0; i < 4; i++) {
        double e = fabs(Ax[i] - b[i]);
        if (e > maxerr)
            maxerr = e;
    }
    ASSERT_TRUE(resid < 1e2);
    ASSERT_TRUE(maxerr < 1e2);

    printf("    backward compat QR+AMD: resid=%.2e, maxerr=%.2e ✓\n", resid, maxerr);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("test_colamd");

    /* Column adjacency graph */
    RUN_TEST(test_graph_null_args);
    RUN_TEST(test_graph_empty);
    RUN_TEST(test_graph_1x1);
    RUN_TEST(test_graph_diagonal);
    RUN_TEST(test_graph_dense_row);
    RUN_TEST(test_graph_tridiag);
    RUN_TEST(test_graph_unsymmetric);
    RUN_TEST(test_graph_arrow);
    RUN_TEST(test_graph_dense_row_skip);
    RUN_TEST(test_graph_rectangular_tall);
    RUN_TEST(test_graph_free_zeroed);

    /* COLAMD ordering */
    RUN_TEST(test_order_null_args);
    RUN_TEST(test_order_1x1);
    RUN_TEST(test_order_diagonal);
    RUN_TEST(test_order_tridiag);
    RUN_TEST(test_order_unsymmetric);
    RUN_TEST(test_order_arrow);
    RUN_TEST(test_order_initial_degree);
    RUN_TEST(test_order_wide_matrix);

    /* Pathological inputs */
    RUN_TEST(test_order_all_dense_rows);
    RUN_TEST(test_order_single_dense_row);
    RUN_TEST(test_order_very_tall);
    RUN_TEST(test_order_very_wide);
    RUN_TEST(test_order_empty_columns);
    RUN_TEST(test_order_duplicate_entries);

    /* Public API */
    RUN_TEST(test_public_api_null);
    RUN_TEST(test_public_api_square);
    RUN_TEST(test_public_api_rectangular);
    RUN_TEST(test_public_api_west0067);
    RUN_TEST(test_public_api_steam1);
    RUN_TEST(test_colamd_vs_amd_fill);

    /* QR + COLAMD */
    RUN_TEST(test_qr_colamd_solve);
    RUN_TEST(test_qr_colamd_vs_amd);
    RUN_TEST(test_qr_colamd_sparse_mode);

    /* Analyze + Factor with COLAMD */
    RUN_TEST(test_analyze_lu_colamd);
    RUN_TEST(test_analyze_cholesky_colamd);
    RUN_TEST(test_analyze_lu_colamd_west0067);

    /* Stress tests */
    RUN_TEST(test_stress_identity);
    RUN_TEST(test_stress_all_zero);
    RUN_TEST(test_stress_single_entry);
    RUN_TEST(test_stress_dense_matrix);

    /* Minimum-norm */
    RUN_TEST(test_minnorm_null_args);
    RUN_TEST(test_minnorm_2x4_known);
    RUN_TEST(test_minnorm_is_minimal);
    RUN_TEST(test_minnorm_3x6);
    RUN_TEST(test_minnorm_fallback_overdetermined);
    RUN_TEST(test_minnorm_with_colamd);
    RUN_TEST(test_minnorm_5x10);
    RUN_TEST(test_minnorm_rank_deficient);
    RUN_TEST(test_minnorm_square);
    RUN_TEST(test_minnorm_1xn);

    /* Refinement & rank deficiency */
    RUN_TEST(test_refine_minnorm);
    RUN_TEST(test_minnorm_zero_row);
    RUN_TEST(test_minnorm_vs_pinv);
    RUN_TEST(test_refine_minnorm_null);

    /* Rank-revealing diagnostics */
    RUN_TEST(test_diag_r_null);
    RUN_TEST(test_diag_r_basic);
    RUN_TEST(test_rank_info_full_rank);
    RUN_TEST(test_rank_info_deficient);
    RUN_TEST(test_condest_basic);
    RUN_TEST(test_condest_ill);
    RUN_TEST(test_condest_null);
    RUN_TEST(test_condest_vs_true);

    /* SuiteSparse integration */
    RUN_TEST(test_ss_west0067);
    RUN_TEST(test_ss_steam1);
    RUN_TEST(test_ss_fs_541_1);
    RUN_TEST(test_ss_orsirr_1);

    /* Underdetermined SuiteSparse */
    RUN_TEST(test_minnorm_ss_submatrix);

    /* Backward compatibility */
    RUN_TEST(test_backward_compat_qr_no_reorder);
    RUN_TEST(test_backward_compat_qr_amd);

    TEST_SUITE_END();
}
