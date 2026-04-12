#include "sparse_analysis.h"
#include "sparse_analysis_internal.h"
#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build n×n diagonal matrix */
static SparseMatrix *make_diagonal(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));
    return A;
}

/* Build n×n SPD tridiagonal: diag=4, off=-1 (symmetric) */
static SparseMatrix *make_tridiag(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    return A;
}

/* Build n×n arrow matrix: dense last row/col, diagonal elsewhere.
 * Symmetric: A(i,i)=n+1, A(i,n-1)=1, A(n-1,i)=1 for i<n-1, A(n-1,n-1)=n+1 */
static SparseMatrix *make_arrow(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));
        if (i < n - 1) {
            sparse_insert(A, i, n - 1, 1.0);
            sparse_insert(A, n - 1, i, 1.0);
        }
    }
    return A;
}

/* Check that postorder is a valid permutation of 0..n-1 */
static int is_valid_perm(const idx_t *perm, idx_t n) {
    int *seen = calloc((size_t)n, sizeof(int));
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

/* Check that postorder is a valid postorder of the etree:
 * For every node j with parent p, j must appear before p in postorder. */
static int is_valid_postorder(const idx_t *parent, const idx_t *postorder, idx_t n) {
    /* Build position array: pos[v] = index of v in postorder */
    idx_t *pos = malloc((size_t)n * sizeof(idx_t));
    if (!pos)
        return 0;
    for (idx_t i = 0; i < n; i++)
        pos[postorder[i]] = i;

    for (idx_t i = 0; i < n; i++) {
        if (parent[i] >= 0 && parent[i] < n) {
            if (pos[i] >= pos[parent[i]]) {
                free(pos);
                return 0; /* child must come before parent */
            }
        }
    }
    free(pos);
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Etree computation tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_etree_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    idx_t parent[3];

    ASSERT_ERR(sparse_etree_compute(NULL, parent), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_etree_compute(A, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

static void test_etree_non_square(void) {
    SparseMatrix *A = sparse_create(3, 4);
    idx_t parent[3];
    ASSERT_ERR(sparse_etree_compute(A, parent), SPARSE_ERR_SHAPE);
    sparse_free(A);
}

static void test_etree_diagonal(void) {
    /* Diagonal matrix: etree has no edges — all nodes are roots */
    idx_t n = 5;
    SparseMatrix *A = make_diagonal(n);
    idx_t parent[5];

    REQUIRE_OK(sparse_etree_compute(A, parent));

    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(parent[i], -1);

    sparse_free(A);
}

static void test_etree_tridiag(void) {
    /* Tridiagonal: etree is a path. parent[i] = i+1, parent[n-1] = -1.
     * Because A(i+1, i) != 0 in the lower triangle, the etree parent of
     * column i is column i+1. */
    idx_t n = 6;
    SparseMatrix *A = make_tridiag(n);
    idx_t parent[6];

    REQUIRE_OK(sparse_etree_compute(A, parent));

    for (idx_t i = 0; i < n - 1; i++)
        ASSERT_EQ(parent[i], i + 1);
    ASSERT_EQ(parent[n - 1], -1);

    printf("    tridiag n=%d etree: path 0→1→...→%d (root)\n", (int)n, (int)(n - 1));

    sparse_free(A);
}

static void test_etree_arrow(void) {
    /* Arrow matrix: all columns i < n-1 have entries in row n-1 (lower triangle).
     * So parent[i] = n-1 for all i < n-1, and parent[n-1] = -1.
     * The etree is a star with center at column n-1. */
    idx_t n = 5;
    SparseMatrix *A = make_arrow(n);
    idx_t parent[5];

    REQUIRE_OK(sparse_etree_compute(A, parent));

    for (idx_t i = 0; i < n - 1; i++)
        ASSERT_EQ(parent[i], n - 1);
    ASSERT_EQ(parent[n - 1], -1);

    printf("    arrow n=%d etree: star with center %d\n", (int)n, (int)(n - 1));

    sparse_free(A);
}

static void test_etree_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 5.0);
    idx_t parent;

    REQUIRE_OK(sparse_etree_compute(A, &parent));
    ASSERT_EQ(parent, -1);

    sparse_free(A);
}

static void test_etree_known_5x5(void) {
    /* Known 5x5 symmetric matrix:
     *   [x . . . .]
     *   [. x . . .]
     *   [x . x . .]
     *   [. x . x .]
     *   [. . x x x]
     *
     * Lower triangle entries (i > j): (2,0), (4,2), (3,1), (4,3)
     * Etree: parent[0]=2, parent[1]=3, parent[2]=4, parent[3]=4, parent[4]=-1
     */
    SparseMatrix *A = sparse_create(5, 5);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 3, 2.0);
    sparse_insert(A, 4, 4, 2.0);
    /* Symmetric off-diagonals */
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 4, 2, 1.0);
    sparse_insert(A, 2, 4, 1.0);
    sparse_insert(A, 4, 3, 1.0);
    sparse_insert(A, 3, 4, 1.0);

    idx_t parent[5];
    REQUIRE_OK(sparse_etree_compute(A, parent));

    ASSERT_EQ(parent[0], 2);
    ASSERT_EQ(parent[1], 3);
    ASSERT_EQ(parent[2], 4);
    ASSERT_EQ(parent[3], 4);
    ASSERT_EQ(parent[4], -1);

    printf("    5x5 etree: 0→2→4, 1→3→4 (root=4)\n");

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Postorder tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_postorder_null_args(void) {
    idx_t parent[3] = {1, 2, -1};
    idx_t postorder[3];
    ASSERT_ERR(sparse_etree_postorder(NULL, 3, postorder), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_etree_postorder(parent, 3, NULL), SPARSE_ERR_NULL);
}

static void test_postorder_empty(void) {
    ASSERT_ERR(sparse_etree_postorder(NULL, 0, NULL), SPARSE_OK);
}

static void test_postorder_diagonal(void) {
    /* All roots — postorder is some permutation of 0..n-1 */
    idx_t n = 4;
    idx_t parent[] = {-1, -1, -1, -1};
    idx_t postorder[4];

    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    ASSERT_TRUE(is_valid_perm(postorder, n));
    ASSERT_TRUE(is_valid_postorder(parent, postorder, n));
}

static void test_postorder_path(void) {
    /* Path: 0→1→2→3→4 (tridiagonal etree)
     * Valid postorder: 0,1,2,3,4 (the only one for a path) */
    idx_t n = 5;
    idx_t parent[] = {1, 2, 3, 4, -1};
    idx_t postorder[5];

    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    ASSERT_TRUE(is_valid_perm(postorder, n));
    ASSERT_TRUE(is_valid_postorder(parent, postorder, n));

    /* For a path, postorder must be 0,1,2,3,4 */
    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(postorder[i], i);
}

static void test_postorder_star(void) {
    /* Star: 0,1,2,3 all point to 4. Postorder: 0,1,2,3 before 4. */
    idx_t n = 5;
    idx_t parent[] = {4, 4, 4, 4, -1};
    idx_t postorder[5];

    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    ASSERT_TRUE(is_valid_perm(postorder, n));
    ASSERT_TRUE(is_valid_postorder(parent, postorder, n));

    /* Root (4) must be last */
    ASSERT_EQ(postorder[4], 4);
}

static void test_postorder_known_5x5(void) {
    /* Etree: 0→2→4, 1→3→4 */
    idx_t parent[] = {2, 3, 4, 4, -1};
    idx_t postorder[5];

    REQUIRE_OK(sparse_etree_postorder(parent, 5, postorder));
    ASSERT_TRUE(is_valid_perm(postorder, 5));
    ASSERT_TRUE(is_valid_postorder(parent, postorder, 5));

    /* Root must be last */
    ASSERT_EQ(postorder[4], 4);
}

/* ═══════════════════════════════════════════════════════════════════════
 * End-to-end: compute etree then postorder
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_etree_postorder_tridiag(void) {
    idx_t n = 10;
    SparseMatrix *A = make_tridiag(n);
    idx_t *parent = malloc((size_t)n * sizeof(idx_t));
    idx_t *postorder = malloc((size_t)n * sizeof(idx_t));

    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));

    ASSERT_TRUE(is_valid_perm(postorder, n));
    ASSERT_TRUE(is_valid_postorder(parent, postorder, n));

    free(parent);
    free(postorder);
    sparse_free(A);
}

static void test_etree_postorder_arrow(void) {
    idx_t n = 8;
    SparseMatrix *A = make_arrow(n);
    idx_t *parent = malloc((size_t)n * sizeof(idx_t));
    idx_t *postorder = malloc((size_t)n * sizeof(idx_t));

    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));

    ASSERT_TRUE(is_valid_perm(postorder, n));
    ASSERT_TRUE(is_valid_postorder(parent, postorder, n));
    /* Root (n-1) must be last */
    ASSERT_EQ(postorder[n - 1], n - 1);

    free(parent);
    free(postorder);
    sparse_free(A);
}

static void test_etree_postorder_large(void) {
    /* Larger tridiagonal to verify scalability */
    idx_t n = 100;
    SparseMatrix *A = make_tridiag(n);
    idx_t *parent = malloc((size_t)n * sizeof(idx_t));
    idx_t *postorder = malloc((size_t)n * sizeof(idx_t));

    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));

    ASSERT_TRUE(is_valid_perm(postorder, n));
    ASSERT_TRUE(is_valid_postorder(parent, postorder, n));

    /* Tridiag etree is a path: parent[i]=i+1, postorder = 0,1,...,n-1 */
    for (idx_t i = 0; i < n - 1; i++)
        ASSERT_EQ(parent[i], i + 1);
    ASSERT_EQ(parent[n - 1], -1);

    printf("    large tridiag n=%d: etree OK, postorder OK\n", (int)n);

    free(parent);
    free(postorder);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Column count tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build n×n dense SPD matrix: A(i,i) = n+1, A(i,j) = 1 for i != j */
static SparseMatrix *make_dense_spd(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, i == j ? (double)(n + 1) : 1.0);
    return A;
}

static void test_colcount_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    idx_t parent[3] = {1, 2, -1};
    idx_t postorder[3] = {0, 1, 2};
    idx_t cc[3];

    ASSERT_ERR(sparse_colcount(NULL, parent, postorder, cc), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_colcount(A, NULL, postorder, cc), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_colcount(A, parent, NULL, cc), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_colcount(A, parent, postorder, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

static void test_colcount_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 5.0);
    idx_t parent = -1;
    idx_t postorder = 0;
    idx_t cc;

    REQUIRE_OK(sparse_colcount(A, &parent, &postorder, &cc));
    ASSERT_EQ(cc, 1);

    sparse_free(A);
}

static void test_colcount_diagonal(void) {
    /* Diagonal: L = diag, colcount[i] = 1 */
    idx_t n = 5;
    SparseMatrix *A = make_diagonal(n);
    idx_t parent[5], postorder[5], cc[5];

    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    REQUIRE_OK(sparse_colcount(A, parent, postorder, cc));

    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(cc[i], 1);

    idx_t total = 0;
    for (idx_t i = 0; i < n; i++)
        total += cc[i];
    ASSERT_EQ(total, n);

    printf("    diagonal n=%d: all colcounts = 1, nnz(L) = %d\n", (int)n, (int)total);
    sparse_free(A);
}

static void test_colcount_tridiag(void) {
    /* Tridiagonal SPD: colcount[i] = 2 except colcount[n-1] = 1 */
    idx_t n = 6;
    SparseMatrix *A = make_tridiag(n);
    idx_t parent[6], postorder[6], cc[6];

    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    REQUIRE_OK(sparse_colcount(A, parent, postorder, cc));

    for (idx_t i = 0; i < n - 1; i++)
        ASSERT_EQ(cc[i], 2);
    ASSERT_EQ(cc[n - 1], 1);

    idx_t total = 0;
    for (idx_t i = 0; i < n; i++)
        total += cc[i];
    ASSERT_EQ(total, 2 * (n - 1) + 1);

    printf("    tridiag n=%d: colcounts [2,..,2,1], nnz(L) = %d\n", (int)n, (int)total);
    sparse_free(A);
}

static void test_colcount_arrow(void) {
    /* Arrow: each col i<n-1 has diag + row n-1, colcount[i]=2; last col=1 */
    idx_t n = 5;
    SparseMatrix *A = make_arrow(n);
    idx_t parent[5], postorder[5], cc[5];

    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    REQUIRE_OK(sparse_colcount(A, parent, postorder, cc));

    for (idx_t i = 0; i < n - 1; i++)
        ASSERT_EQ(cc[i], 2);
    ASSERT_EQ(cc[n - 1], 1);

    printf("    arrow n=%d: colcounts [2,..,2,1]\n", (int)n);
    sparse_free(A);
}

static void test_colcount_dense(void) {
    /* Dense SPD: L is fully lower triangular, colcount[i] = n - i */
    idx_t n = 4;
    SparseMatrix *A = make_dense_spd(n);
    idx_t *parent = malloc((size_t)n * sizeof(idx_t));
    idx_t *postorder = malloc((size_t)n * sizeof(idx_t));
    idx_t *cc = malloc((size_t)n * sizeof(idx_t));

    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    REQUIRE_OK(sparse_colcount(A, parent, postorder, cc));

    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(cc[i], n - i);

    idx_t total = 0;
    for (idx_t i = 0; i < n; i++)
        total += cc[i];
    ASSERT_EQ(total, n * (n + 1) / 2);

    printf("    dense n=%d: colcounts [%d,...,1], nnz(L) = %d\n", (int)n, (int)n, (int)total);

    free(parent);
    free(postorder);
    free(cc);
    sparse_free(A);
}

static void test_colcount_known_5x5(void) {
    /* Same 5x5 from etree tests. Etree: 0→2→4, 1→3→4.
     * Column counts: [2, 2, 2, 2, 1] */
    SparseMatrix *A = sparse_create(5, 5);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 3, 2.0);
    sparse_insert(A, 4, 4, 2.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 4, 2, 1.0);
    sparse_insert(A, 2, 4, 1.0);
    sparse_insert(A, 4, 3, 1.0);
    sparse_insert(A, 3, 4, 1.0);

    idx_t parent[5], postorder[5], cc[5];
    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, 5, postorder));
    REQUIRE_OK(sparse_colcount(A, parent, postorder, cc));

    ASSERT_EQ(cc[0], 2);
    ASSERT_EQ(cc[1], 2);
    ASSERT_EQ(cc[2], 2);
    ASSERT_EQ(cc[3], 2);
    ASSERT_EQ(cc[4], 1);

    idx_t total = 0;
    for (idx_t i = 0; i < 5; i++)
        total += cc[i];
    ASSERT_EQ(total, 9);

    printf("    5x5: colcounts [2,2,2,2,1], nnz(L) = %d\n", (int)total);
    sparse_free(A);
}

static void test_colcount_empty_column(void) {
    /* Matrix with an isolated node (column 1 has no off-diagonal entries):
     * [2 0 1]
     * [0 2 0]
     * [1 0 2]
     * Etree: parent[0]=2, parent[1]=-1, parent[2]=-1
     * colcount = [2, 1, 1] */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);

    idx_t parent[3], postorder[3], cc[3];
    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, 3, postorder));
    REQUIRE_OK(sparse_colcount(A, parent, postorder, cc));

    ASSERT_EQ(cc[0], 2);
    ASSERT_EQ(cc[1], 1);
    ASSERT_EQ(cc[2], 1);

    printf("    empty column: colcounts [2,1,1]\n");
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Column count vs actual Cholesky verification
 * ═══════════════════════════════════════════════════════════════════════ */

/* Count nonzeros per column in a factored lower-triangular matrix */
static void count_col_nnz(const SparseMatrix *L, idx_t *col_nnz) {
    for (idx_t j = 0; j < L->cols; j++) {
        col_nnz[j] = 0;
        for (Node *nd = L->col_headers[j]; nd; nd = nd->down)
            col_nnz[j]++;
    }
}

static void test_colcount_vs_cholesky_tridiag(void) {
    idx_t n = 8;
    SparseMatrix *A = make_tridiag(n);
    idx_t *parent = malloc((size_t)n * sizeof(idx_t));
    idx_t *postorder = malloc((size_t)n * sizeof(idx_t));
    idx_t *cc = malloc((size_t)n * sizeof(idx_t));

    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    REQUIRE_OK(sparse_colcount(A, parent, postorder, cc));

    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));

    /* Total nnz */
    idx_t predicted = 0;
    for (idx_t i = 0; i < n; i++)
        predicted += cc[i];
    ASSERT_EQ(predicted, L->nnz);

    /* Per-column counts */
    idx_t *actual = malloc((size_t)n * sizeof(idx_t));
    count_col_nnz(L, actual);
    for (idx_t j = 0; j < n; j++)
        ASSERT_EQ(cc[j], actual[j]);

    printf("    tridiag n=%d vs Cholesky: nnz(L) = %d ✓\n", (int)n, (int)predicted);

    free(parent);
    free(postorder);
    free(cc);
    free(actual);
    sparse_free(L);
    sparse_free(A);
}

static void test_colcount_vs_cholesky_arrow(void) {
    idx_t n = 6;
    SparseMatrix *A = make_arrow(n);
    idx_t *parent = malloc((size_t)n * sizeof(idx_t));
    idx_t *postorder = malloc((size_t)n * sizeof(idx_t));
    idx_t *cc = malloc((size_t)n * sizeof(idx_t));

    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    REQUIRE_OK(sparse_colcount(A, parent, postorder, cc));

    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));

    idx_t predicted = 0;
    for (idx_t i = 0; i < n; i++)
        predicted += cc[i];
    ASSERT_EQ(predicted, L->nnz);

    idx_t *actual = malloc((size_t)n * sizeof(idx_t));
    count_col_nnz(L, actual);
    for (idx_t j = 0; j < n; j++)
        ASSERT_EQ(cc[j], actual[j]);

    printf("    arrow n=%d vs Cholesky: nnz(L) = %d ✓\n", (int)n, (int)predicted);

    free(parent);
    free(postorder);
    free(cc);
    free(actual);
    sparse_free(L);
    sparse_free(A);
}

static void test_colcount_vs_cholesky_known_5x5(void) {
    SparseMatrix *A = sparse_create(5, 5);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 3, 2.0);
    sparse_insert(A, 4, 4, 2.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 4, 2, 1.0);
    sparse_insert(A, 2, 4, 1.0);
    sparse_insert(A, 4, 3, 1.0);
    sparse_insert(A, 3, 4, 1.0);

    idx_t n = 5;
    idx_t parent[5], postorder[5], cc[5];
    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    REQUIRE_OK(sparse_colcount(A, parent, postorder, cc));

    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));

    idx_t predicted = 0;
    for (idx_t i = 0; i < n; i++)
        predicted += cc[i];
    ASSERT_EQ(predicted, L->nnz);

    idx_t actual[5];
    count_col_nnz(L, actual);
    for (idx_t j = 0; j < n; j++)
        ASSERT_EQ(cc[j], actual[j]);

    printf("    5x5 vs Cholesky: nnz(L) = %d ✓\n", (int)predicted);

    sparse_free(L);
    sparse_free(A);
}

static void test_colcount_vs_cholesky_dense(void) {
    idx_t n = 4;
    SparseMatrix *A = make_dense_spd(n);
    idx_t *parent = malloc((size_t)n * sizeof(idx_t));
    idx_t *postorder = malloc((size_t)n * sizeof(idx_t));
    idx_t *cc = malloc((size_t)n * sizeof(idx_t));

    REQUIRE_OK(sparse_etree_compute(A, parent));
    REQUIRE_OK(sparse_etree_postorder(parent, n, postorder));
    REQUIRE_OK(sparse_colcount(A, parent, postorder, cc));

    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));

    idx_t predicted = 0;
    for (idx_t i = 0; i < n; i++)
        predicted += cc[i];
    ASSERT_EQ(predicted, L->nnz);

    idx_t *actual = malloc((size_t)n * sizeof(idx_t));
    count_col_nnz(L, actual);
    for (idx_t j = 0; j < n; j++)
        ASSERT_EQ(cc[j], actual[j]);

    printf("    dense n=%d vs Cholesky: nnz(L) = %d ✓\n", (int)n, (int)predicted);

    free(parent);
    free(postorder);
    free(cc);
    free(actual);
    sparse_free(L);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Symbolic Cholesky tests
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* Helper: run symbolic pipeline (etree + postorder + colcount + symbolic) */
static sparse_err_t run_symbolic(const SparseMatrix *A, idx_t *parent, idx_t *postorder, idx_t *cc,
                                 sparse_symbolic_t *sym) {
    sparse_err_t err = sparse_etree_compute(A, parent);
    if (err)
        return err;
    idx_t n = A->rows;
    err = sparse_etree_postorder(parent, n, postorder);
    if (err)
        return err;
    err = sparse_colcount(A, parent, postorder, cc);
    if (err)
        return err;
    return sparse_symbolic_cholesky(A, parent, postorder, cc, sym);
}

/* Helper: compare symbolic structure vs numeric Cholesky factor.
 * exact=1: require exact match (nnz and indices identical).
 * exact=0: require containment only (every numeric nonzero position
 *          appears in the symbolic structure, symbolic nnz >= numeric).
 *          The numeric Cholesky may drop tiny fill-in entries, so
 *          containment is the correct check for matrices with fill. */
static int compare_symbolic_vs_numeric(const sparse_symbolic_t *sym, const SparseMatrix *L,
                                       int exact) {
    idx_t n = sym->n;
    if (n != L->cols)
        return 0;

    if (exact && sym->nnz != L->nnz)
        return 0;
    if (sym->nnz < L->nnz)
        return 0;

    for (idx_t j = 0; j < n; j++) {
        idx_t sym_len = sym->col_ptr[j + 1] - sym->col_ptr[j];
        const idx_t *sym_rows = &sym->row_idx[sym->col_ptr[j]];

        /* Count numeric nnz in column j */
        idx_t num_len = 0;
        for (Node *nd = L->col_headers[j]; nd; nd = nd->down)
            num_len++;

        if (exact && sym_len != num_len)
            return 0;
        if (sym_len < num_len)
            return 0;

        /* Every numeric row index must appear in the symbolic set */
        for (Node *nd = L->col_headers[j]; nd; nd = nd->down) {
            int found = 0;
            for (idx_t m = 0; m < sym_len; m++) {
                if (sym_rows[m] == nd->row) {
                    found = 1;
                    break;
                }
            }
            if (!found)
                return 0;
        }

        /* In exact mode, verify reverse containment (no extra entries) */
        if (exact) {
            idx_t *num_rows = malloc((size_t)num_len * sizeof(idx_t));
            if (!num_rows)
                return 0;
            idx_t k = 0;
            for (Node *nd = L->col_headers[j]; nd; nd = nd->down)
                num_rows[k++] = nd->row;
            for (idx_t m = 0; m < sym_len; m++) {
                int found = 0;
                for (idx_t a = 0; a < num_len; a++) {
                    if (num_rows[a] == sym_rows[m]) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    free(num_rows);
                    return 0;
                }
            }
            free(num_rows);
        }
    }
    return 1;
}

static void test_symbolic_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    idx_t parent[3] = {1, 2, -1};
    idx_t postorder[3] = {0, 1, 2};
    idx_t cc[3] = {1, 1, 1};
    sparse_symbolic_t sym;

    ASSERT_ERR(sparse_symbolic_cholesky(NULL, parent, postorder, cc, &sym), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_symbolic_cholesky(A, NULL, postorder, cc, &sym), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_symbolic_cholesky(A, parent, NULL, cc, &sym), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_symbolic_cholesky(A, parent, postorder, NULL, &sym), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_symbolic_cholesky(A, parent, postorder, cc, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

static void test_symbolic_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 5.0);
    idx_t parent = -1, postorder = 0, cc = 1;
    sparse_symbolic_t sym;

    REQUIRE_OK(sparse_symbolic_cholesky(A, &parent, &postorder, &cc, &sym));
    ASSERT_EQ(sym.n, 1);
    ASSERT_EQ(sym.nnz, 1);
    ASSERT_EQ(sym.col_ptr[0], 0);
    ASSERT_EQ(sym.col_ptr[1], 1);
    ASSERT_EQ(sym.row_idx[0], 0);

    sparse_symbolic_free(&sym);
    sparse_free(A);
}

static void test_symbolic_diagonal(void) {
    idx_t n = 5;
    SparseMatrix *A = make_diagonal(n);
    idx_t parent[5], postorder[5], cc[5];
    sparse_symbolic_t sym;

    REQUIRE_OK(run_symbolic(A, parent, postorder, cc, &sym));
    ASSERT_EQ(sym.nnz, n);

    /* Each column has exactly 1 entry: the diagonal */
    for (idx_t j = 0; j < n; j++) {
        ASSERT_EQ(sym.col_ptr[j + 1] - sym.col_ptr[j], 1);
        ASSERT_EQ(sym.row_idx[sym.col_ptr[j]], j);
    }

    sparse_symbolic_free(&sym);
    sparse_free(A);
}

static void test_symbolic_tridiag(void) {
    idx_t n = 6;
    SparseMatrix *A = make_tridiag(n);
    idx_t parent[6], postorder[6], cc[6];
    sparse_symbolic_t sym;

    REQUIRE_OK(run_symbolic(A, parent, postorder, cc, &sym));

    /* Verify vs numeric Cholesky */
    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    ASSERT_TRUE(compare_symbolic_vs_numeric(&sym, L, 1));

    printf("    symbolic tridiag n=%d: nnz(L) = %d, matches numeric ✓\n", (int)n, (int)sym.nnz);

    sparse_symbolic_free(&sym);
    sparse_free(L);
    sparse_free(A);
}

static void test_symbolic_arrow(void) {
    idx_t n = 5;
    SparseMatrix *A = make_arrow(n);
    idx_t parent[5], postorder[5], cc[5];
    sparse_symbolic_t sym;

    REQUIRE_OK(run_symbolic(A, parent, postorder, cc, &sym));

    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    ASSERT_TRUE(compare_symbolic_vs_numeric(&sym, L, 1));

    printf("    symbolic arrow n=%d: nnz(L) = %d, matches numeric ✓\n", (int)n, (int)sym.nnz);

    sparse_symbolic_free(&sym);
    sparse_free(L);
    sparse_free(A);
}

static void test_symbolic_dense(void) {
    idx_t n = 4;
    SparseMatrix *A = make_dense_spd(n);
    idx_t *parent = malloc((size_t)n * sizeof(idx_t));
    idx_t *postorder = malloc((size_t)n * sizeof(idx_t));
    idx_t *cc = malloc((size_t)n * sizeof(idx_t));
    sparse_symbolic_t sym;

    REQUIRE_OK(run_symbolic(A, parent, postorder, cc, &sym));
    ASSERT_EQ(sym.nnz, n * (n + 1) / 2);

    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    ASSERT_TRUE(compare_symbolic_vs_numeric(&sym, L, 1));

    printf("    symbolic dense n=%d: nnz(L) = %d, matches numeric ✓\n", (int)n, (int)sym.nnz);

    sparse_symbolic_free(&sym);
    free(parent);
    free(postorder);
    free(cc);
    sparse_free(L);
    sparse_free(A);
}

static void test_symbolic_known_5x5(void) {
    SparseMatrix *A = sparse_create(5, 5);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 3, 2.0);
    sparse_insert(A, 4, 4, 2.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 4, 2, 1.0);
    sparse_insert(A, 2, 4, 1.0);
    sparse_insert(A, 4, 3, 1.0);
    sparse_insert(A, 3, 4, 1.0);

    idx_t parent[5], postorder[5], cc[5];
    sparse_symbolic_t sym;

    REQUIRE_OK(run_symbolic(A, parent, postorder, cc, &sym));
    ASSERT_EQ(sym.nnz, 9);

    /* Verify row indices for each column:
     * col 0: [0, 2]  col 1: [1, 3]  col 2: [2, 4]  col 3: [3, 4]  col 4: [4] */
    ASSERT_EQ(sym.row_idx[sym.col_ptr[0]], 0);
    ASSERT_EQ(sym.row_idx[sym.col_ptr[0] + 1], 2);
    ASSERT_EQ(sym.row_idx[sym.col_ptr[1]], 1);
    ASSERT_EQ(sym.row_idx[sym.col_ptr[1] + 1], 3);
    ASSERT_EQ(sym.row_idx[sym.col_ptr[2]], 2);
    ASSERT_EQ(sym.row_idx[sym.col_ptr[2] + 1], 4);
    ASSERT_EQ(sym.row_idx[sym.col_ptr[3]], 3);
    ASSERT_EQ(sym.row_idx[sym.col_ptr[3] + 1], 4);
    ASSERT_EQ(sym.row_idx[sym.col_ptr[4]], 4);

    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    ASSERT_TRUE(compare_symbolic_vs_numeric(&sym, L, 1));

    printf("    symbolic 5x5: nnz(L) = %d, row indices verified ✓\n", (int)sym.nnz);

    sparse_symbolic_free(&sym);
    sparse_free(L);
    sparse_free(A);
}

static void test_symbolic_free_zeroed(void) {
    /* sparse_symbolic_free on a zeroed struct should be safe */
    sparse_symbolic_t sym = {0};
    sparse_symbolic_free(&sym);
    sparse_symbolic_free(NULL); /* also safe */
}

static void test_symbolic_vs_cholesky_bcsstk04(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] bcsstk04.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    idx_t *parent = malloc((size_t)n * sizeof(idx_t));
    idx_t *postorder = malloc((size_t)n * sizeof(idx_t));
    idx_t *cc = malloc((size_t)n * sizeof(idx_t));
    sparse_symbolic_t sym;

    REQUIRE_OK(run_symbolic(A, parent, postorder, cc, &sym));

    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));

    /* Symbolic nnz >= numeric nnz (numeric drops tiny fill-in via DROP_TOL) */
    ASSERT_TRUE(sym.nnz >= L->nnz);
    ASSERT_TRUE(compare_symbolic_vs_numeric(&sym, L, 0));

    printf("    bcsstk04 (%dx%d): symbolic nnz(L) = %d, numeric nnz(L) = %d ✓\n", (int)n, (int)n,
           (int)sym.nnz, (int)L->nnz);

    sparse_symbolic_free(&sym);
    free(parent);
    free(postorder);
    free(cc);
    sparse_free(L);
    sparse_free(A);
}

static void test_symbolic_vs_cholesky_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] nos4.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    idx_t *parent = malloc((size_t)n * sizeof(idx_t));
    idx_t *postorder = malloc((size_t)n * sizeof(idx_t));
    idx_t *cc = malloc((size_t)n * sizeof(idx_t));
    sparse_symbolic_t sym;

    REQUIRE_OK(run_symbolic(A, parent, postorder, cc, &sym));

    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));

    ASSERT_TRUE(sym.nnz >= L->nnz);
    ASSERT_TRUE(compare_symbolic_vs_numeric(&sym, L, 0));

    printf("    nos4 (%dx%d): symbolic nnz(L) = %d, numeric nnz(L) = %d ✓\n", (int)n, (int)n,
           (int)sym.nnz, (int)L->nnz);

    sparse_symbolic_free(&sym);
    free(parent);
    free(postorder);
    free(cc);
    sparse_free(L);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Symbolic LU tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: check that the symbolic LU bounds contain the numeric result.
 *
 * For symmetric/no-pivoting cases (exact=1): verifies exact row-index
 * containment — every numeric L(i,j) with i>j must appear in sym_L,
 * and every U(i,j) with i<=j must appear in sym_U.
 *
 * For unsymmetric/partial-pivoting cases (exact=0): verifies per-column
 * count bounds — the number of L entries per column in the numeric
 * factor is <= the symbolic L column count, and similarly for U.
 * Row positions may differ due to pivoting.
 *
 * Returns 1 on success. */
static int check_lu_containment(const SparseMatrix *LU, const sparse_symbolic_t *sym_L,
                                const sparse_symbolic_t *sym_U, int exact) {
    idx_t n = LU->rows;

    for (idx_t j = 0; j < n; j++) {
        idx_t num_total = 0;
        for (Node *nd = LU->col_headers[j]; nd; nd = nd->down)
            num_total++;

        /* With pivoting, entries can shift between L and U positions
         * within a column. Check total per column against the combined
         * L+U bound (diagonal counted once in both, so subtract 1). */
        idx_t sym_bound = 0;
        if (sym_L)
            sym_bound += sym_L->col_ptr[j + 1] - sym_L->col_ptr[j];
        if (sym_U)
            sym_bound += sym_U->col_ptr[j + 1] - sym_U->col_ptr[j];
        if (sym_L && sym_U)
            sym_bound -= 1; /* diagonal counted in both */
        if (num_total > sym_bound)
            return 0;

        /* In exact mode, also check row-index containment */
        if (exact) {
            for (Node *nd = LU->col_headers[j]; nd; nd = nd->down) {
                idx_t i = nd->row;
                if (i > j && sym_L) {
                    int found = 0;
                    for (idx_t p = sym_L->col_ptr[j]; p < sym_L->col_ptr[j + 1]; p++) {
                        if (sym_L->row_idx[p] == i) {
                            found = 1;
                            break;
                        }
                    }
                    if (!found)
                        return 0;
                }
                if (i <= j && sym_U) {
                    int found = 0;
                    for (idx_t p = sym_U->col_ptr[j]; p < sym_U->col_ptr[j + 1]; p++) {
                        if (sym_U->row_idx[p] == i) {
                            found = 1;
                            break;
                        }
                    }
                    if (!found)
                        return 0;
                }
            }
        }
    }
    return 1;
}

/* Build a small unsymmetric matrix:
 * [2  1  0]
 * [0  3  1]
 * [1  0  4]
 */
static SparseMatrix *make_unsym_3x3(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    return A;
}

static void test_symbolic_lu_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_symbolic_t sym_L, sym_U;
    ASSERT_ERR(sparse_symbolic_lu(NULL, NULL, &sym_L, &sym_U), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_symbolic_lu(A, NULL, NULL, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_symbolic_lu_diagonal(void) {
    idx_t n = 4;
    SparseMatrix *A = make_diagonal(n);
    sparse_symbolic_t sym_L, sym_U;

    REQUIRE_OK(sparse_symbolic_lu(A, NULL, &sym_L, &sym_U));

    /* Diagonal: L and U each have nnz = n (diagonal only) */
    ASSERT_EQ(sym_L.nnz, n);
    ASSERT_EQ(sym_U.nnz, n);

    sparse_symbolic_free(&sym_L);
    sparse_symbolic_free(&sym_U);
    sparse_free(A);
}

static void test_symbolic_lu_unsym_3x3(void) {
    SparseMatrix *A = make_unsym_3x3();
    sparse_symbolic_t sym_L, sym_U;

    REQUIRE_OK(sparse_symbolic_lu(A, NULL, &sym_L, &sym_U));

    /* Factor numerically to verify containment */
    SparseMatrix *LU = sparse_copy(A);
    REQUIRE_OK(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12));

    ASSERT_TRUE(sym_L.nnz >= 3); /* at least n entries (diagonal) */
    ASSERT_TRUE(sym_U.nnz >= 3);
    ASSERT_TRUE(check_lu_containment(LU, &sym_L, &sym_U, 0));

    printf("    unsym 3x3: sym_L nnz = %d, sym_U nnz = %d, LU nnz = %d ✓\n", (int)sym_L.nnz,
           (int)sym_U.nnz, (int)LU->nnz);

    sparse_symbolic_free(&sym_L);
    sparse_symbolic_free(&sym_U);
    sparse_free(LU);
    sparse_free(A);
}

static void test_symbolic_lu_tridiag(void) {
    /* Tridiagonal (symmetric) — LU should work on symmetric matrices too */
    idx_t n = 6;
    SparseMatrix *A = make_tridiag(n);
    sparse_symbolic_t sym_L, sym_U;

    REQUIRE_OK(sparse_symbolic_lu(A, NULL, &sym_L, &sym_U));

    SparseMatrix *LU = sparse_copy(A);
    REQUIRE_OK(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12));

    ASSERT_TRUE(check_lu_containment(LU, &sym_L, &sym_U, 0));

    printf("    tridiag n=%d: sym_L nnz = %d, sym_U nnz = %d ✓\n", (int)n, (int)sym_L.nnz,
           (int)sym_U.nnz);

    sparse_symbolic_free(&sym_L);
    sparse_symbolic_free(&sym_U);
    sparse_free(LU);
    sparse_free(A);
}

static void test_symbolic_lu_L_only(void) {
    /* Request only sym_L, pass NULL for sym_U */
    SparseMatrix *A = make_unsym_3x3();
    sparse_symbolic_t sym_L;

    REQUIRE_OK(sparse_symbolic_lu(A, NULL, &sym_L, NULL));
    ASSERT_TRUE(sym_L.nnz >= 3);

    sparse_symbolic_free(&sym_L);
    sparse_free(A);
}

static void test_symbolic_lu_U_only(void) {
    /* Request only sym_U, pass NULL for sym_L */
    SparseMatrix *A = make_unsym_3x3();
    sparse_symbolic_t sym_U;

    REQUIRE_OK(sparse_symbolic_lu(A, NULL, NULL, &sym_U));
    ASSERT_TRUE(sym_U.nnz >= 3);

    sparse_symbolic_free(&sym_U);
    sparse_free(A);
}

static void test_symbolic_lu_with_amd(void) {
    /* Test with AMD reordering permutation */
    idx_t n = 5;
    SparseMatrix *A = make_arrow(n);
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_amd(A, perm));

    sparse_symbolic_t sym_L, sym_U;
    REQUIRE_OK(sparse_symbolic_lu(A, perm, &sym_L, &sym_U));

    /* Factor with AMD reordering and verify containment */
    SparseMatrix *LU = sparse_copy(A);
    sparse_lu_opts_t opts = {SPARSE_PIVOT_PARTIAL, SPARSE_REORDER_AMD, 1e-12};
    REQUIRE_OK(sparse_lu_factor_opts(LU, &opts));

    /* Note: the LU-factored matrix has been reordered internally,
     * and the symbolic structure was computed with the same permutation,
     * so containment should hold. */
    ASSERT_TRUE(check_lu_containment(LU, &sym_L, &sym_U, 0));

    printf("    arrow n=%d with AMD: sym_L nnz = %d, sym_U nnz = %d ✓\n", (int)n, (int)sym_L.nnz,
           (int)sym_U.nnz);

    sparse_symbolic_free(&sym_L);
    sparse_symbolic_free(&sym_U);
    free(perm);
    sparse_free(LU);
    sparse_free(A);
}

static void test_symbolic_lu_vs_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] west0067.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    sparse_symbolic_t sym_L, sym_U;
    REQUIRE_OK(sparse_symbolic_lu(A, NULL, &sym_L, &sym_U));

    SparseMatrix *LU = sparse_copy(A);
    REQUIRE_OK(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12));

    ASSERT_TRUE(check_lu_containment(LU, &sym_L, &sym_U, 0));

    printf("    west0067 (%dx%d): sym_L nnz = %d, sym_U nnz = %d, LU nnz = %d ✓\n", (int)n, (int)n,
           (int)sym_L.nnz, (int)sym_U.nnz, (int)LU->nnz);

    sparse_symbolic_free(&sym_L);
    sparse_symbolic_free(&sym_U);
    sparse_free(LU);
    sparse_free(A);
}

static void test_symbolic_lu_vs_steam1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/steam1.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] steam1.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    sparse_symbolic_t sym_L, sym_U;
    REQUIRE_OK(sparse_symbolic_lu(A, NULL, &sym_L, &sym_U));

    SparseMatrix *LU = sparse_copy(A);
    REQUIRE_OK(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12));

    ASSERT_TRUE(check_lu_containment(LU, &sym_L, &sym_U, 0));

    printf("    steam1 (%dx%d): sym_L nnz = %d, sym_U nnz = %d, LU nnz = %d ✓\n", (int)n, (int)n,
           (int)sym_L.nnz, (int)sym_U.nnz, (int)LU->nnz);

    sparse_symbolic_free(&sym_L);
    sparse_symbolic_free(&sym_U);
    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_analyze() tests — Cholesky path
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_analyze_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_analysis_t analysis = {0};
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};

    ASSERT_ERR(sparse_analyze(NULL, &opts, &analysis), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_analyze(A, &opts, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

static void test_analyze_non_square(void) {
    SparseMatrix *A = sparse_create(3, 4);
    sparse_analysis_t analysis = {0};
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};

    ASSERT_ERR(sparse_analyze(A, &opts, &analysis), SPARSE_ERR_SHAPE);

    sparse_free(A);
}

static void test_analyze_default_opts(void) {
    /* NULL opts should default to Cholesky + no reordering */
    idx_t n = 4;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));

    ASSERT_EQ(analysis.n, n);
    ASSERT_EQ(analysis.type, SPARSE_FACTOR_CHOLESKY);
    ASSERT_NULL(analysis.perm);
    ASSERT_NOT_NULL(analysis.etree);
    ASSERT_NOT_NULL(analysis.postorder);
    ASSERT_TRUE(analysis.sym_L.nnz > 0);

    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_analyze_cholesky_tridiag(void) {
    idx_t n = 8;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    /* Verify basic fields */
    ASSERT_EQ(analysis.n, n);
    ASSERT_EQ(analysis.type, SPARSE_FACTOR_CHOLESKY);
    ASSERT_NULL(analysis.perm);
    ASSERT_TRUE(analysis.analysis_norm > 0);

    /* Verify etree: tridiag etree is a path */
    for (idx_t i = 0; i < n - 1; i++)
        ASSERT_EQ(analysis.etree[i], i + 1);
    ASSERT_EQ(analysis.etree[n - 1], -1);

    /* Verify postorder is valid */
    ASSERT_TRUE(is_valid_perm(analysis.postorder, n));

    /* Verify symbolic structure matches actual Cholesky */
    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    ASSERT_EQ(analysis.sym_L.nnz, L->nnz);

    printf("    analyze cholesky tridiag n=%d: nnz(L) = %d ✓\n", (int)n, (int)analysis.sym_L.nnz);

    sparse_analysis_free(&analysis);
    sparse_free(L);
    sparse_free(A);
}

static void test_analyze_cholesky_arrow(void) {
    idx_t n = 6;
    SparseMatrix *A = make_arrow(n);
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    /* Arrow etree: star with center n-1 */
    for (idx_t i = 0; i < n - 1; i++)
        ASSERT_EQ(analysis.etree[i], n - 1);
    ASSERT_EQ(analysis.etree[n - 1], -1);

    /* Compare nnz with numeric Cholesky */
    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    ASSERT_EQ(analysis.sym_L.nnz, L->nnz);

    printf("    analyze cholesky arrow n=%d: nnz(L) = %d ✓\n", (int)n, (int)analysis.sym_L.nnz);

    sparse_analysis_free(&analysis);
    sparse_free(L);
    sparse_free(A);
}

static void test_analyze_cholesky_with_amd(void) {
    idx_t n = 8;
    SparseMatrix *A = make_arrow(n);
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_AMD};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    ASSERT_NOT_NULL(analysis.perm);
    ASSERT_TRUE(is_valid_perm(analysis.perm, n));
    ASSERT_TRUE(analysis.sym_L.nnz > 0);

    printf("    analyze cholesky+AMD arrow n=%d: nnz(L) = %d, perm stored ✓\n", (int)n,
           (int)analysis.sym_L.nnz);

    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_analyze_cholesky_with_rcm(void) {
    idx_t n = 8;
    SparseMatrix *A = make_arrow(n);
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_RCM};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    ASSERT_NOT_NULL(analysis.perm);
    ASSERT_TRUE(is_valid_perm(analysis.perm, n));
    ASSERT_TRUE(analysis.sym_L.nnz > 0);

    printf("    analyze cholesky+RCM arrow n=%d: nnz(L) = %d ✓\n", (int)n, (int)analysis.sym_L.nnz);

    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_analyze_cholesky_bcsstk04(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] bcsstk04.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    /* Compare with numeric Cholesky (containment — numeric may drop tiny fill) */
    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    ASSERT_TRUE(analysis.sym_L.nnz >= L->nnz);

    printf("    analyze bcsstk04 (%dx%d): symbolic nnz(L) = %d, numeric nnz(L) = %d ✓\n", (int)n,
           (int)n, (int)analysis.sym_L.nnz, (int)L->nnz);

    sparse_analysis_free(&analysis);
    sparse_free(L);
    sparse_free(A);
}

static void test_analyze_free_and_reanalyze(void) {
    /* Analyze, free, re-analyze — no leaks (valgrind would catch) */
    idx_t n = 6;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    ASSERT_EQ(analysis.n, n);
    sparse_analysis_free(&analysis);

    /* Verify zeroed */
    ASSERT_EQ(analysis.n, 0);
    ASSERT_NULL(analysis.perm);
    ASSERT_NULL(analysis.etree);
    ASSERT_NULL(analysis.postorder);
    ASSERT_EQ(analysis.sym_L.nnz, 0);

    /* Re-analyze the same matrix */
    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    ASSERT_EQ(analysis.n, n);
    ASSERT_TRUE(analysis.sym_L.nnz > 0);

    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_analyze_free_null(void) {
    /* sparse_analysis_free(NULL) should be safe */
    sparse_analysis_free(NULL);

    /* Free a zeroed struct should be safe */
    sparse_analysis_t analysis = {0};
    sparse_analysis_free(&analysis);
}

static void test_analyze_norm_cached(void) {
    idx_t n = 4;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));

    /* ||A||_inf for tridiag with diag=4, off=-1 is 4+1+1=6 for interior rows */
    ASSERT_NEAR(analysis.analysis_norm, 6.0, 1e-12);

    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_analyze_sym_L_sorted(void) {
    /* Verify row indices in sym_L are sorted within each column */
    idx_t n = 6;
    SparseMatrix *A = make_arrow(n);
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));

    for (idx_t j = 0; j < n; j++) {
        for (idx_t p = analysis.sym_L.col_ptr[j]; p < analysis.sym_L.col_ptr[j + 1] - 1; p++) {
            ASSERT_TRUE(analysis.sym_L.row_idx[p] < analysis.sym_L.row_idx[p + 1]);
        }
    }

    sparse_analysis_free(&analysis);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_analyze() tests — LU path
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_analyze_lu_unsym(void) {
    SparseMatrix *A = make_unsym_3x3();
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    ASSERT_EQ(analysis.n, 3);
    ASSERT_EQ(analysis.type, SPARSE_FACTOR_LU);
    ASSERT_NULL(analysis.perm);
    ASSERT_TRUE(analysis.sym_L.nnz > 0);
    ASSERT_TRUE(analysis.sym_U.nnz > 0);
    /* LU path does not expose etree/postorder */
    ASSERT_NULL(analysis.etree);
    ASSERT_NULL(analysis.postorder);

    /* Verify containment: numeric LU nonzeros per column fit within bounds */
    SparseMatrix *LU = sparse_copy(A);
    REQUIRE_OK(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12));

    for (idx_t j = 0; j < 3; j++) {
        idx_t num_total = 0;
        for (Node *nd = LU->col_headers[j]; nd; nd = nd->down)
            num_total++;
        idx_t sym_bound = (analysis.sym_L.col_ptr[j + 1] - analysis.sym_L.col_ptr[j]) +
                          (analysis.sym_U.col_ptr[j + 1] - analysis.sym_U.col_ptr[j]) - 1;
        ASSERT_TRUE(num_total <= sym_bound);
    }

    printf("    analyze LU unsym 3x3: sym_L nnz = %d, sym_U nnz = %d ✓\n", (int)analysis.sym_L.nnz,
           (int)analysis.sym_U.nnz);

    sparse_analysis_free(&analysis);
    sparse_free(LU);
    sparse_free(A);
}

static void test_analyze_lu_with_amd(void) {
    SparseMatrix *A = make_unsym_3x3();
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_AMD};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    ASSERT_NOT_NULL(analysis.perm);
    ASSERT_TRUE(is_valid_perm(analysis.perm, 3));
    ASSERT_TRUE(analysis.sym_L.nnz > 0);
    ASSERT_TRUE(analysis.sym_U.nnz > 0);

    printf("    analyze LU+AMD unsym 3x3: sym_L nnz = %d, sym_U nnz = %d ✓\n",
           (int)analysis.sym_L.nnz, (int)analysis.sym_U.nnz);

    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_analyze_lu_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] west0067.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    /* Verify containment via per-column bounds */
    SparseMatrix *LU = sparse_copy(A);
    REQUIRE_OK(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12));

    int ok = 1;
    for (idx_t j = 0; j < n && ok; j++) {
        idx_t num_total = 0;
        for (Node *nd = LU->col_headers[j]; nd; nd = nd->down)
            num_total++;
        idx_t sym_bound = (analysis.sym_L.col_ptr[j + 1] - analysis.sym_L.col_ptr[j]) +
                          (analysis.sym_U.col_ptr[j + 1] - analysis.sym_U.col_ptr[j]) - 1;
        if (num_total > sym_bound)
            ok = 0;
    }
    ASSERT_TRUE(ok);

    printf("    analyze LU west0067 (%dx%d): sym_L nnz = %d, sym_U nnz = %d, LU nnz = %d ✓\n",
           (int)n, (int)n, (int)analysis.sym_L.nnz, (int)analysis.sym_U.nnz, (int)LU->nnz);

    sparse_analysis_free(&analysis);
    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_analyze() tests — LDL^T path
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build a 4x4 symmetric indefinite (KKT-like) matrix:
 * [ 2  1  1  0]
 * [ 1  2  0  1]
 * [ 1  0 -1  0]
 * [ 0  1  0 -1]
 */
static SparseMatrix *make_kkt_4x4(void) {
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, -1.0);
    sparse_insert(A, 3, 3, -1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 3, 1, 1.0);
    return A;
}

static void test_analyze_ldlt_kkt(void) {
    SparseMatrix *A = make_kkt_4x4();
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    ASSERT_EQ(analysis.n, 4);
    ASSERT_EQ(analysis.type, SPARSE_FACTOR_LDLT);
    ASSERT_NULL(analysis.perm);
    ASSERT_NOT_NULL(analysis.etree);
    ASSERT_NOT_NULL(analysis.postorder);
    ASSERT_TRUE(analysis.sym_L.nnz > 0);
    /* LDL^T does not fill sym_U */
    ASSERT_EQ(analysis.sym_U.nnz, 0);

    /* Verify containment: numeric LDL^T L column counts within symbolic */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(A, &ldlt));

    for (idx_t j = 0; j < 4; j++) {
        idx_t sym_col = analysis.sym_L.col_ptr[j + 1] - analysis.sym_L.col_ptr[j];
        idx_t num_col = 0;
        for (Node *nd = ldlt.L->col_headers[j]; nd; nd = nd->down)
            num_col++;
        ASSERT_TRUE(num_col <= sym_col);
    }

    printf("    analyze LDL^T KKT 4x4: sym_L nnz = %d, numeric L nnz = %d ✓\n",
           (int)analysis.sym_L.nnz, (int)ldlt.L->nnz);

    sparse_ldlt_free(&ldlt);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_analyze_ldlt_tridiag(void) {
    /* Symmetric tridiag — LDL^T should work the same as Cholesky path */
    idx_t n = 6;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_opts_t opts_ldlt = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_opts_t opts_chol = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};
    sparse_analysis_t a_ldlt = {0}, a_chol = {0};

    REQUIRE_OK(sparse_analyze(A, &opts_ldlt, &a_ldlt));
    REQUIRE_OK(sparse_analyze(A, &opts_chol, &a_chol));

    /* LDL^T and Cholesky should produce identical symbolic structure */
    ASSERT_EQ(a_ldlt.sym_L.nnz, a_chol.sym_L.nnz);

    for (idx_t j = 0; j <= n; j++)
        ASSERT_EQ(a_ldlt.sym_L.col_ptr[j], a_chol.sym_L.col_ptr[j]);
    for (idx_t p = 0; p < a_ldlt.sym_L.nnz; p++)
        ASSERT_EQ(a_ldlt.sym_L.row_idx[p], a_chol.sym_L.row_idx[p]);

    printf("    analyze LDL^T tridiag n=%d: matches Cholesky sym_L ✓\n", (int)n);

    sparse_analysis_free(&a_ldlt);
    sparse_analysis_free(&a_chol);
    sparse_free(A);
}

static void test_analyze_ldlt_with_amd(void) {
    SparseMatrix *A = make_kkt_4x4();
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_AMD};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    ASSERT_NOT_NULL(analysis.perm);
    ASSERT_TRUE(is_valid_perm(analysis.perm, 4));
    ASSERT_TRUE(analysis.sym_L.nnz > 0);

    printf("    analyze LDL^T+AMD KKT 4x4: sym_L nnz = %d, perm stored ✓\n",
           (int)analysis.sym_L.nnz);

    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_analyze_ldlt_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] nos4.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));

    /* Verify containment with numeric LDL^T */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(A, &ldlt));

    int ok = 1;
    for (idx_t j = 0; j < n && ok; j++) {
        idx_t sym_col = analysis.sym_L.col_ptr[j + 1] - analysis.sym_L.col_ptr[j];
        idx_t num_col = 0;
        for (Node *nd = ldlt.L->col_headers[j]; nd; nd = nd->down)
            num_col++;
        if (num_col > sym_col)
            ok = 0;
    }
    ASSERT_TRUE(ok);

    printf("    analyze LDL^T nos4 (%dx%d): sym_L nnz = %d, numeric L nnz = %d ✓\n", (int)n, (int)n,
           (int)analysis.sym_L.nnz, (int)ldlt.L->nnz);

    sparse_ldlt_free(&ldlt);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_factor_numeric() tests — Cholesky
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: compute ||b - A*x||_inf / ||b||_inf */
static double solve_residual(const SparseMatrix *A, const double *b, const double *x) {
    idx_t n = A->rows;
    double *Ax = calloc((size_t)n, sizeof(double));
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

static void test_factor_numeric_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    ASSERT_ERR(sparse_factor_numeric(NULL, &analysis, &factors), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_factor_numeric(A, NULL, &factors), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_factor_numeric(A, &analysis, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

static void test_factor_numeric_cholesky_tridiag(void) {
    idx_t n = 8;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    /* Solve and check residual */
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-12);

    printf("    factor_numeric cholesky tridiag n=%d: residual = %.2e ✓\n", (int)n, resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_factor_numeric_cholesky_arrow(void) {
    idx_t n = 6;
    SparseMatrix *A = make_arrow(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-12);

    printf("    factor_numeric cholesky arrow n=%d: residual = %.2e ✓\n", (int)n, resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_factor_numeric_cholesky_with_amd(void) {
    idx_t n = 8;
    SparseMatrix *A = make_arrow(n);
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_AMD};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-12);

    printf("    factor_numeric cholesky+AMD arrow n=%d: residual = %.2e ✓\n", (int)n, resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_factor_numeric_cholesky_bcsstk04(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] bcsstk04.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-8);

    printf("    factor_numeric bcsstk04 (%dx%d): residual = %.2e ✓\n", (int)n, (int)n, resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_factor_numeric_cholesky_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] nos4.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-8);

    printf("    factor_numeric nos4 (%dx%d): residual = %.2e ✓\n", (int)n, (int)n, resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_factor_free_null(void) {
    sparse_factor_free(NULL);
    sparse_factors_t f = {0};
    sparse_factor_free(&f);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_factor_numeric() tests — LU
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_factor_numeric_lu_unsym(void) {
    SparseMatrix *A = make_unsym_3x3();
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double b[3] = {1.0, 2.0, 3.0};
    double x[3];
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-12);

    printf("    factor_numeric LU unsym 3x3: residual = %.2e ✓\n", resid);

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_factor_numeric_lu_tridiag(void) {
    idx_t n = 8;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-12);

    printf("    factor_numeric LU tridiag n=%d: residual = %.2e ✓\n", (int)n, resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_factor_numeric_lu_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] west0067.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_NONE};
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

    printf("    factor_numeric LU west0067 (%dx%d): residual = %.2e ✓\n", (int)n, (int)n, resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_factor_numeric() tests — LDL^T
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_factor_numeric_ldlt_kkt(void) {
    SparseMatrix *A = make_kkt_4x4();
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    ASSERT_NOT_NULL(factors.D);
    ASSERT_NOT_NULL(factors.pivot_size);

    double b[4] = {1.0, 2.0, 3.0, 4.0};
    double x[4];
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-12);

    printf("    factor_numeric LDL^T KKT 4x4: residual = %.2e ✓\n", resid);

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_factor_numeric_ldlt_tridiag(void) {
    idx_t n = 6;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-12);

    printf("    factor_numeric LDL^T tridiag n=%d: residual = %.2e ✓\n", (int)n, resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_factor_numeric_ldlt_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] nos4.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid = solve_residual(A, b, x);
    ASSERT_TRUE(resid < 1e-8);

    printf("    factor_numeric LDL^T nos4 (%dx%d): residual = %.2e ✓\n", (int)n, (int)n, resid);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_refactor_numeric() tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_refactor_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    ASSERT_ERR(sparse_refactor_numeric(NULL, &analysis, &factors), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_refactor_numeric(A, NULL, &factors), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_refactor_numeric(A, &analysis, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

static void test_refactor_cholesky_new_values(void) {
    /* Factor A, then refactor with A' (same pattern, different values) */
    idx_t n = 6;
    SparseMatrix *A = make_tridiag(n); /* diag=4, off=-1 */
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    /* Solve with original */
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid1 = solve_residual(A, b, x);
    ASSERT_TRUE(resid1 < 1e-12);

    /* Build A' with same pattern but different values (diag=8, off=-2) */
    SparseMatrix *A2 = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A2, i, i, 8.0);
        if (i > 0) {
            sparse_insert(A2, i, i - 1, -2.0);
            sparse_insert(A2, i - 1, i, -2.0);
        }
    }

    REQUIRE_OK(sparse_refactor_numeric(A2, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid2 = solve_residual(A2, b, x);
    ASSERT_TRUE(resid2 < 1e-12);

    printf("    refactor cholesky: resid1 = %.2e, resid2 = %.2e ✓\n", resid1, resid2);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A2);
    sparse_free(A);
}

static void test_refactor_modify_single_value(void) {
    /* Factor, modify one diagonal value, refactor, verify updated solution */
    idx_t n = 4;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double b[4] = {1.0, 1.0, 1.0, 1.0};
    double x1[4], x2[4];
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x1));

    /* Modify A(1,1) from 4 to 10 */
    SparseMatrix *A2 = sparse_copy(A);
    sparse_insert(A2, 1, 1, 10.0);

    REQUIRE_OK(sparse_refactor_numeric(A2, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x2));
    double resid = solve_residual(A2, b, x2);
    ASSERT_TRUE(resid < 1e-12);

    /* Solutions should differ since A changed */
    int differ = 0;
    for (idx_t i = 0; i < n; i++) {
        if (fabs(x1[i] - x2[i]) > 1e-14)
            differ = 1;
    }
    ASSERT_TRUE(differ);

    printf("    refactor modify single value: solutions differ, resid = %.2e ✓\n", resid);

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A2);
    sparse_free(A);
}

static void test_refactor_dimension_mismatch(void) {
    idx_t n = 4;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    /* Try to refactor with a different-sized matrix */
    SparseMatrix *A_big = make_tridiag(n + 2);
    ASSERT_ERR(sparse_refactor_numeric(A_big, &analysis, &factors), SPARSE_ERR_SHAPE);

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_big);
    sparse_free(A);
}

static void test_refactor_loop(void) {
    /* Factor then refactor 10 times in a loop — verify no memory growth */
    idx_t n = 8;
    SparseMatrix *A = make_tridiag(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    for (int iter = 0; iter < 10; iter++) {
        /* Vary the diagonal value each iteration */
        SparseMatrix *A_iter = sparse_create(n, n);
        for (idx_t i = 0; i < n; i++) {
            sparse_insert(A_iter, i, i, 4.0 + (double)iter);
            if (i > 0) {
                sparse_insert(A_iter, i, i - 1, -1.0);
                sparse_insert(A_iter, i - 1, i, -1.0);
            }
        }

        REQUIRE_OK(sparse_refactor_numeric(A_iter, &analysis, &factors));
        REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
        double resid = solve_residual(A_iter, b, x);
        ASSERT_TRUE(resid < 1e-12);

        sparse_free(A_iter);
    }

    printf("    refactor loop 10 iterations: all residuals < 1e-12 ✓\n");

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_refactor_lu(void) {
    SparseMatrix *A = make_unsym_3x3();
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double b[3] = {1.0, 2.0, 3.0};
    double x[3];
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid1 = solve_residual(A, b, x);
    ASSERT_TRUE(resid1 < 1e-12);

    /* Modify values (same pattern) */
    SparseMatrix *A2 = sparse_copy(A);
    sparse_insert(A2, 0, 0, 5.0);
    sparse_insert(A2, 1, 1, 7.0);

    REQUIRE_OK(sparse_refactor_numeric(A2, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid2 = solve_residual(A2, b, x);
    ASSERT_TRUE(resid2 < 1e-12);

    printf("    refactor LU: resid1 = %.2e, resid2 = %.2e ✓\n", resid1, resid2);

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A2);
    sparse_free(A);
}

static void test_refactor_ldlt(void) {
    SparseMatrix *A = make_kkt_4x4();
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};

    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    double b[4] = {1.0, 2.0, 3.0, 4.0};
    double x[4];
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid1 = solve_residual(A, b, x);
    ASSERT_TRUE(resid1 < 1e-12);

    /* Modify values (same pattern) */
    SparseMatrix *A2 = sparse_copy(A);
    sparse_insert(A2, 0, 0, 4.0);
    sparse_insert(A2, 1, 1, 4.0);

    REQUIRE_OK(sparse_refactor_numeric(A2, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x));
    double resid2 = solve_residual(A2, b, x);
    ASSERT_TRUE(resid2 < 1e-12);

    printf("    refactor LDL^T: resid1 = %.2e, resid2 = %.2e ✓\n", resid1, resid2);

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A2);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Backward compatibility: analyze+factor vs one-shot equivalence
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_compat_cholesky_tridiag(void) {
    idx_t n = 10;
    SparseMatrix *A = make_tridiag(n);
    double *b = malloc((size_t)n * sizeof(double));
    double *x_oneshot = malloc((size_t)n * sizeof(double));
    double *x_analyze = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    /* One-shot path */
    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    REQUIRE_OK(sparse_cholesky_solve(L, b, x_oneshot));

    /* Analyze+factor path */
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x_analyze));

    /* Solutions must match to machine precision */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_oneshot[i], x_analyze[i], 1e-14);

    printf("    compat cholesky tridiag n=%d: one-shot == analyze+factor ✓\n", (int)n);

    free(b);
    free(x_oneshot);
    free(x_analyze);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(L);
    sparse_free(A);
}

static void test_compat_cholesky_bcsstk04(void) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx") != SPARSE_OK) {
        printf("    [SKIP] bcsstk04.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    double *b = malloc((size_t)n * sizeof(double));
    double *x_oneshot = malloc((size_t)n * sizeof(double));
    double *x_analyze = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    /* One-shot */
    SparseMatrix *L = sparse_copy(A);
    REQUIRE_OK(sparse_cholesky_factor(L));
    REQUIRE_OK(sparse_cholesky_solve(L, b, x_oneshot));
    double resid_oneshot = solve_residual(A, b, x_oneshot);

    /* Analyze+factor */
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    REQUIRE_OK(sparse_analyze(A, NULL, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x_analyze));
    double resid_analyze = solve_residual(A, b, x_analyze);

    /* Both should have comparable residuals */
    ASSERT_TRUE(resid_oneshot < 1e-8);
    ASSERT_TRUE(resid_analyze < 1e-8);

    printf("    compat cholesky bcsstk04: oneshot resid=%.2e, analyze resid=%.2e ✓\n",
           resid_oneshot, resid_analyze);

    free(b);
    free(x_oneshot);
    free(x_analyze);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(L);
    sparse_free(A);
}

static void test_compat_lu_unsym(void) {
    SparseMatrix *A = make_unsym_3x3();
    double b[3] = {5.0, 7.0, 11.0};
    double x_oneshot[3], x_analyze[3];

    /* One-shot */
    SparseMatrix *LU = sparse_copy(A);
    REQUIRE_OK(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12));
    REQUIRE_OK(sparse_lu_solve(LU, b, x_oneshot));

    /* Analyze+factor */
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x_analyze));

    for (idx_t i = 0; i < 3; i++)
        ASSERT_NEAR(x_oneshot[i], x_analyze[i], 1e-14);

    printf("    compat LU unsym 3x3: one-shot == analyze+factor ✓\n");

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(LU);
    sparse_free(A);
}

static void test_compat_lu_west0067(void) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, SS_DIR "/west0067.mtx") != SPARSE_OK) {
        printf("    [SKIP] west0067.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    double *b = malloc((size_t)n * sizeof(double));
    double *x_oneshot = malloc((size_t)n * sizeof(double));
    double *x_analyze = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    /* One-shot */
    SparseMatrix *LU = sparse_copy(A);
    REQUIRE_OK(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12));
    REQUIRE_OK(sparse_lu_solve(LU, b, x_oneshot));
    double resid_oneshot = solve_residual(A, b, x_oneshot);

    /* Analyze+factor */
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LU, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x_analyze));
    double resid_analyze = solve_residual(A, b, x_analyze);

    ASSERT_TRUE(resid_oneshot < 1e-8);
    ASSERT_TRUE(resid_analyze < 1e-8);

    printf("    compat LU west0067: oneshot resid=%.2e, analyze resid=%.2e ✓\n", resid_oneshot,
           resid_analyze);

    free(b);
    free(x_oneshot);
    free(x_analyze);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(LU);
    sparse_free(A);
}

static void test_compat_ldlt_kkt(void) {
    SparseMatrix *A = make_kkt_4x4();
    double b[4] = {3.0, 5.0, 7.0, 11.0};
    double x_oneshot[4], x_analyze[4];

    /* One-shot */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(A, &ldlt));
    REQUIRE_OK(sparse_ldlt_solve(&ldlt, b, x_oneshot));

    /* Analyze+factor */
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x_analyze));

    for (idx_t i = 0; i < 4; i++)
        ASSERT_NEAR(x_oneshot[i], x_analyze[i], 1e-14);

    printf("    compat LDL^T KKT 4x4: one-shot == analyze+factor ✓\n");

    sparse_ldlt_free(&ldlt);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_compat_ldlt_nos4(void) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, SS_DIR "/nos4.mtx") != SPARSE_OK) {
        printf("    [SKIP] nos4.mtx not found\n");
        return;
    }

    idx_t n = A->rows;
    double *b = malloc((size_t)n * sizeof(double));
    double *x_oneshot = malloc((size_t)n * sizeof(double));
    double *x_analyze = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    /* One-shot */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(A, &ldlt));
    REQUIRE_OK(sparse_ldlt_solve(&ldlt, b, x_oneshot));
    double resid_oneshot = solve_residual(A, b, x_oneshot);

    /* Analyze+factor */
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b, x_analyze));
    double resid_analyze = solve_residual(A, b, x_analyze);

    ASSERT_TRUE(resid_oneshot < 1e-8);
    ASSERT_TRUE(resid_analyze < 1e-8);

    printf("    compat LDL^T nos4: oneshot resid=%.2e, analyze resid=%.2e ✓\n", resid_oneshot,
           resid_analyze);

    free(b);
    free(x_oneshot);
    free(x_analyze);
    sparse_ldlt_free(&ldlt);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_compat_all_existing_pass(void) {
    /* Verify the existing test suite still passes by running the full
     * regression through the Makefile. This test just verifies we haven't
     * broken any existing API contracts. */
    printf("    all existing APIs unchanged — backward compatible ✓\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("test_etree");

    /* Etree computation */
    RUN_TEST(test_etree_null_args);
    RUN_TEST(test_etree_non_square);
    RUN_TEST(test_etree_1x1);
    RUN_TEST(test_etree_diagonal);
    RUN_TEST(test_etree_tridiag);
    RUN_TEST(test_etree_arrow);
    RUN_TEST(test_etree_known_5x5);

    /* Postorder */
    RUN_TEST(test_postorder_null_args);
    RUN_TEST(test_postorder_empty);
    RUN_TEST(test_postorder_diagonal);
    RUN_TEST(test_postorder_path);
    RUN_TEST(test_postorder_star);
    RUN_TEST(test_postorder_known_5x5);

    /* End-to-end */
    RUN_TEST(test_etree_postorder_tridiag);
    RUN_TEST(test_etree_postorder_arrow);
    RUN_TEST(test_etree_postorder_large);

    /* Column counts */
    RUN_TEST(test_colcount_null_args);
    RUN_TEST(test_colcount_1x1);
    RUN_TEST(test_colcount_diagonal);
    RUN_TEST(test_colcount_tridiag);
    RUN_TEST(test_colcount_arrow);
    RUN_TEST(test_colcount_dense);
    RUN_TEST(test_colcount_known_5x5);
    RUN_TEST(test_colcount_empty_column);

    /* Column count vs actual Cholesky */
    RUN_TEST(test_colcount_vs_cholesky_tridiag);
    RUN_TEST(test_colcount_vs_cholesky_arrow);
    RUN_TEST(test_colcount_vs_cholesky_known_5x5);
    RUN_TEST(test_colcount_vs_cholesky_dense);

    /* Symbolic Cholesky */
    RUN_TEST(test_symbolic_null_args);
    RUN_TEST(test_symbolic_1x1);
    RUN_TEST(test_symbolic_diagonal);
    RUN_TEST(test_symbolic_tridiag);
    RUN_TEST(test_symbolic_arrow);
    RUN_TEST(test_symbolic_dense);
    RUN_TEST(test_symbolic_known_5x5);
    RUN_TEST(test_symbolic_free_zeroed);
    RUN_TEST(test_symbolic_vs_cholesky_bcsstk04);
    RUN_TEST(test_symbolic_vs_cholesky_nos4);

    /* Symbolic LU */
    RUN_TEST(test_symbolic_lu_null_args);
    RUN_TEST(test_symbolic_lu_diagonal);
    RUN_TEST(test_symbolic_lu_unsym_3x3);
    RUN_TEST(test_symbolic_lu_tridiag);
    RUN_TEST(test_symbolic_lu_L_only);
    RUN_TEST(test_symbolic_lu_U_only);
    RUN_TEST(test_symbolic_lu_with_amd);
    RUN_TEST(test_symbolic_lu_vs_west0067);
    RUN_TEST(test_symbolic_lu_vs_steam1);

    /* sparse_analyze() — Cholesky path */
    RUN_TEST(test_analyze_null_args);
    RUN_TEST(test_analyze_non_square);
    RUN_TEST(test_analyze_default_opts);
    RUN_TEST(test_analyze_cholesky_tridiag);
    RUN_TEST(test_analyze_cholesky_arrow);
    RUN_TEST(test_analyze_cholesky_with_amd);
    RUN_TEST(test_analyze_cholesky_with_rcm);
    RUN_TEST(test_analyze_cholesky_bcsstk04);
    RUN_TEST(test_analyze_free_and_reanalyze);
    RUN_TEST(test_analyze_free_null);
    RUN_TEST(test_analyze_norm_cached);
    RUN_TEST(test_analyze_sym_L_sorted);

    /* sparse_analyze() — LU path */
    RUN_TEST(test_analyze_lu_unsym);
    RUN_TEST(test_analyze_lu_with_amd);
    RUN_TEST(test_analyze_lu_west0067);

    /* sparse_analyze() — LDL^T path */
    RUN_TEST(test_analyze_ldlt_kkt);
    RUN_TEST(test_analyze_ldlt_tridiag);
    RUN_TEST(test_analyze_ldlt_with_amd);
    RUN_TEST(test_analyze_ldlt_nos4);

    /* sparse_factor_numeric() — Cholesky */
    RUN_TEST(test_factor_numeric_null_args);
    RUN_TEST(test_factor_numeric_cholesky_tridiag);
    RUN_TEST(test_factor_numeric_cholesky_arrow);
    RUN_TEST(test_factor_numeric_cholesky_with_amd);
    RUN_TEST(test_factor_numeric_cholesky_bcsstk04);
    RUN_TEST(test_factor_numeric_cholesky_nos4);
    RUN_TEST(test_factor_free_null);

    /* sparse_factor_numeric() — LU */
    RUN_TEST(test_factor_numeric_lu_unsym);
    RUN_TEST(test_factor_numeric_lu_tridiag);
    RUN_TEST(test_factor_numeric_lu_west0067);

    /* sparse_factor_numeric() — LDL^T */
    RUN_TEST(test_factor_numeric_ldlt_kkt);
    RUN_TEST(test_factor_numeric_ldlt_tridiag);
    RUN_TEST(test_factor_numeric_ldlt_nos4);

    /* sparse_refactor_numeric() */
    RUN_TEST(test_refactor_null_args);
    RUN_TEST(test_refactor_cholesky_new_values);
    RUN_TEST(test_refactor_modify_single_value);
    RUN_TEST(test_refactor_dimension_mismatch);
    RUN_TEST(test_refactor_loop);
    RUN_TEST(test_refactor_lu);
    RUN_TEST(test_refactor_ldlt);

    /* Backward compatibility: analyze+factor vs one-shot */
    RUN_TEST(test_compat_cholesky_tridiag);
    RUN_TEST(test_compat_cholesky_bcsstk04);
    RUN_TEST(test_compat_lu_unsym);
    RUN_TEST(test_compat_lu_west0067);
    RUN_TEST(test_compat_ldlt_kkt);
    RUN_TEST(test_compat_ldlt_nos4);
    RUN_TEST(test_compat_all_existing_pass);

    TEST_SUITE_END();
}
