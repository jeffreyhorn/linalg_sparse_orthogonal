#include "sparse_colamd_internal.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Dense row threshold scale factor. Rows with nnz > COLAMD_DENSE_SCALE * sqrt(n)
 * are excluded from the column adjacency graph to control cost. */
#define COLAMD_DENSE_SCALE 10.0

/* ═══════════════════════════════════════════════════════════════════════
 * Column adjacency graph construction
 * ═══════════════════════════════════════════════════════════════════════ */

void colamd_graph_free(colamd_graph_t *graph) {
    if (!graph)
        return;
    free(graph->col_adj_ptr);
    free(graph->col_adj_list);
    memset(graph, 0, sizeof(*graph));
}

sparse_err_t colamd_build_graph(const SparseMatrix *A, idx_t dense_threshold,
                                colamd_graph_t *graph) {
    if (!A || !graph)
        return SPARSE_ERR_NULL;

    idx_t m = A->rows;
    idx_t n = A->cols;
    memset(graph, 0, sizeof(*graph));
    graph->ncols = n;

    if (n == 0) {
        graph->col_adj_ptr = calloc(1, sizeof(idx_t));
        if (!graph->col_adj_ptr)
            return SPARSE_ERR_ALLOC;
        return SPARSE_OK;
    }

    /* Overflow guard for n-based allocations */
    if ((size_t)n > SIZE_MAX / sizeof(idx_t) || (size_t)m > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    /* For each column j, find all columns that share a row with j.
     * Walk column j's entries to get the row indices, then for each such
     * row, walk the row to find the other columns. Use marker[k] = j to
     * avoid counting duplicates per column j.
     *
     * Pass 1: count degree per column.
     * Pass 2: fill adjacency lists. */

    /* Precompute row lengths for dense-row skipping */
    idx_t *row_len = NULL;
    if (dense_threshold > 0) {
        row_len = calloc((size_t)m, sizeof(idx_t));
        if (!row_len)
            return SPARSE_ERR_ALLOC;
        for (idx_t i = 0; i < m; i++) {
            for (Node *nd = A->row_headers[i]; nd; nd = nd->right)
                row_len[i]++;
        }
    }

    idx_t *marker = malloc((size_t)n * sizeof(idx_t));
    idx_t *degree = calloc((size_t)n, sizeof(idx_t));
    if (!marker || !degree) {
        free(marker);
        free(degree);
        free(row_len);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t j = 0; j < n; j++)
        marker[j] = -1;

    /* Pass 1: count adjacency degree per column */
    for (idx_t j = 0; j < n; j++) {
        /* For each row containing column j */
        for (Node *col_nd = A->col_headers[j]; col_nd; col_nd = col_nd->down) {
            idx_t row = col_nd->row;

            /* Skip dense rows */
            if (dense_threshold > 0 && row_len[row] > dense_threshold)
                continue;

            /* Walk this row to find other columns */
            for (Node *row_nd = A->row_headers[row]; row_nd; row_nd = row_nd->right) {
                idx_t k = row_nd->col;
                if (k == j)
                    continue;
                if (marker[k] != j) {
                    marker[k] = j;
                    degree[j]++;
                }
            }
        }
    }

    /* Build col_adj_ptr from degrees */
    {
        size_t ptr_len = (size_t)n + 1;
        if (ptr_len > SIZE_MAX / sizeof(idx_t)) {
            free(marker);
            free(degree);
            free(row_len);
            return SPARSE_ERR_ALLOC;
        }
        graph->col_adj_ptr = malloc(ptr_len * sizeof(idx_t));
    }
    if (!graph->col_adj_ptr) {
        free(marker);
        free(degree);
        free(row_len);
        return SPARSE_ERR_ALLOC;
    }
    graph->col_adj_ptr[0] = 0;
    {
        size_t total = 0;
        for (idx_t j = 0; j < n; j++) {
            total += (size_t)degree[j];
            if (total > (size_t)INT32_MAX) {
                free(graph->col_adj_ptr);
                graph->col_adj_ptr = NULL;
                free(marker);
                free(degree);
                free(row_len);
                return SPARSE_ERR_ALLOC;
            }
            graph->col_adj_ptr[j + 1] = (idx_t)total;
        }
    }
    // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
    graph->nnz_adj = graph->col_adj_ptr[n];

    if (graph->nnz_adj == 0) {
        graph->col_adj_list = NULL;
        free(marker);
        free(degree);
        free(row_len);
        return SPARSE_OK;
    }

    if ((size_t)graph->nnz_adj > SIZE_MAX / sizeof(idx_t)) {
        colamd_graph_free(graph);
        free(marker);
        free(degree);
        free(row_len);
        return SPARSE_ERR_ALLOC;
    }
    graph->col_adj_list = malloc((size_t)graph->nnz_adj * sizeof(idx_t));
    if (!graph->col_adj_list) {
        colamd_graph_free(graph);
        free(marker);
        free(degree);
        free(row_len);
        return SPARSE_ERR_ALLOC;
    }

    /* Pass 2: fill adjacency lists.
     * Reset markers and reuse degree as write cursors. */
    for (idx_t j = 0; j < n; j++) {
        marker[j] = -1;
        degree[j] = 0;
    }

    for (idx_t j = 0; j < n; j++) {
        for (Node *col_nd = A->col_headers[j]; col_nd; col_nd = col_nd->down) {
            idx_t row = col_nd->row;

            if (dense_threshold > 0 && row_len[row] > dense_threshold)
                continue;

            for (Node *row_nd = A->row_headers[row]; row_nd; row_nd = row_nd->right) {
                idx_t k = row_nd->col;
                if (k == j)
                    continue;
                if (marker[k] != j) {
                    marker[k] = j;
                    idx_t pos = graph->col_adj_ptr[j] + degree[j];
                    graph->col_adj_list[pos] = k; // NOLINT(clang-analyzer-security.ArrayBound)
                    degree[j]++;
                }
            }
        }
    }

    free(marker);
    free(degree);
    free(row_len);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * COLAMD ordering via minimum degree on column adjacency graph
 *
 * Uses bitset adjacency (same approach as AMD in sparse_reorder.c) but
 * operates on the column adjacency graph rather than the symmetrized
 * matrix graph. Dense rows are skipped during graph construction via a
 * threshold of 10 * sqrt(n).
 * ═══════════════════════════════════════════════════════════════════════ */

/* Bitset helpers */
typedef uint64_t bword_t;
#define BWORD_BITS 64
#define BWORD_IDX(i) ((i) / BWORD_BITS)
#define BWORD_BIT(i) ((bword_t)1 << ((i) % BWORD_BITS))

static inline void bset(bword_t *bs, idx_t i) { bs[BWORD_IDX(i)] |= BWORD_BIT(i); }

// NOLINTNEXTLINE(clang-analyzer-core.BitwiseShift)
static inline void bclr(bword_t *bs, idx_t i) { bs[BWORD_IDX(i)] &= ~BWORD_BIT(i); }

static inline int btest(const bword_t *bs, idx_t i) {
    return (bs[BWORD_IDX(i)] & BWORD_BIT(i)) != 0;
}

static void bunion(bword_t *dst, const bword_t *src, size_t nw) {
    for (size_t w = 0; w < nw; w++)
        dst[w] |= src[w];
}

static idx_t bpopcount(const bword_t *bs, size_t nw) {
    idx_t cnt = 0;
    for (size_t w = 0; w < nw; w++) {
        bword_t v = bs[w];
        while (v) {
            v &= v - 1;
            cnt++;
        }
    }
    return cnt;
}

sparse_err_t colamd_order(const SparseMatrix *A, idx_t *perm) {
    if (!A || !perm)
        return SPARSE_ERR_NULL;

    idx_t n = A->cols;
    if (n == 0)
        return SPARSE_OK;
    if (n == 1) {
        perm[0] = 0;
        return SPARSE_OK;
    }

    /* Dense row threshold: rows with nnz > COLAMD_DENSE_SCALE * sqrt(n)
     * are skipped to avoid O(nnz_row^2) blowup in graph construction.
     * Minimum threshold of 16 to avoid skipping in tiny matrices. */
    idx_t dense_thresh = (idx_t)(COLAMD_DENSE_SCALE * sqrt((double)n));
    if (dense_thresh < 16)
        dense_thresh = 16;

    /* Build column adjacency graph */
    colamd_graph_t graph;
    sparse_err_t err = colamd_build_graph(A, dense_thresh, &graph);
    if (err != SPARSE_OK)
        return err;

    /* Convert CSR adjacency to bitset format for O(1) neighbor queries.
     * Compute nwords in size_t to avoid signed overflow for large n. */
    size_t nwords = ((size_t)n + BWORD_BITS - 1) / BWORD_BITS;

    /* Check for overflow: n * nwords * sizeof(bword_t) */
    if ((size_t)n > SIZE_MAX / (nwords * sizeof(bword_t))) {
        colamd_graph_free(&graph);
        return SPARSE_ERR_ALLOC;
    }

    bword_t *adj_bits = calloc((size_t)n * nwords, sizeof(bword_t));
    int *eliminated = calloc((size_t)n, sizeof(int));
    idx_t *deg = malloc((size_t)n * sizeof(idx_t));

    if (!adj_bits || !eliminated || !deg) {
        free(adj_bits);
        free(eliminated);
        free(deg);
        colamd_graph_free(&graph);
        return SPARSE_ERR_ALLOC;
    }

    /* Initialize bitset adjacency from CSR graph */
    for (idx_t j = 0; j < n; j++) {
        bword_t *row = &adj_bits[(size_t)j * nwords];
        for (idx_t p = graph.col_adj_ptr[j]; p < graph.col_adj_ptr[j + 1]; p++)
            // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
            bset(row, graph.col_adj_list[p]);
        deg[j] = graph.col_adj_ptr[j + 1] - graph.col_adj_ptr[j];
    }

    colamd_graph_free(&graph);

    /* Main elimination loop: minimum degree ordering */
    for (idx_t step = 0; step < n; step++) {
        /* Find uneliminated column with minimum degree */
        idx_t best = -1;
        idx_t best_deg = n + 1;
        for (idx_t j = 0; j < n; j++) {
            if (!eliminated[j] && deg[j] < best_deg) {
                best_deg = deg[j];
                best = j;
            }
        }

        perm[step] = best;
        eliminated[best] = 1;

        bword_t *best_row = &adj_bits[(size_t)best * nwords];

        /* For each uneliminated neighbor u of best:
         *   adj[u] = (adj[u] | adj[best]) \ {best, u}
         * Models fill-in from eliminating column best. */
        for (idx_t u = 0; u < n; u++) {
            if (eliminated[u] || !btest(best_row, u))
                continue;

            bword_t *u_row = &adj_bits[(size_t)u * nwords];
            bunion(u_row, best_row, nwords);
            bclr(u_row, best);
            bclr(u_row, u);

            deg[u] = bpopcount(u_row, nwords);
        }

        /* Clear best from all adjacency rows */
        for (idx_t j = 0; j < n; j++)
            bclr(&adj_bits[(size_t)j * nwords], best);
    }

    free(adj_bits);
    free(eliminated);
    free(deg);
    return SPARSE_OK;
}
