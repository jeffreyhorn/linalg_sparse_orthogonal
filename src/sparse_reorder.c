#include "sparse_reorder.h"
#include "sparse_matrix_internal.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ─── Bandwidth ──────────────────────────────────────────────────────── */

idx_t sparse_bandwidth(const SparseMatrix *A)
{
    if (!A) return 0;
    idx_t bw = 0;
    for (idx_t i = 0; i < A->rows; i++) {
        Node *node = A->row_headers[i];
        while (node) {
            idx_t diff = node->row - node->col;
            if (diff < 0) diff = -diff;
            if (diff > bw) bw = diff;
            node = node->right;
        }
    }
    return bw;
}

/* ─── Permute ────────────────────────────────────────────────────────── */

sparse_err_t sparse_permute(const SparseMatrix *A,
                            const idx_t *row_perm, const idx_t *col_perm,
                            SparseMatrix **B)
{
    if (!A || !row_perm || !col_perm || !B) return SPARSE_ERR_NULL;
    *B = NULL;

    idx_t m = A->rows;
    idx_t nc = A->cols;

    /* Build inverse column permutation: inv_col[old_j] = new_j */
    idx_t *inv_col = malloc((size_t)nc * sizeof(idx_t));
    if (!inv_col) return SPARSE_ERR_ALLOC;
    for (idx_t j = 0; j < nc; j++)
        inv_col[col_perm[j]] = j;

    SparseMatrix *out = sparse_create(m, nc);
    if (!out) { free(inv_col); return SPARSE_ERR_ALLOC; }

    /* For each new row i, walk old row row_perm[i] and insert with mapped columns */
    for (idx_t i = 0; i < m; i++) {
        idx_t old_row = row_perm[i];
        Node *node = A->row_headers[old_row];
        while (node) {
            idx_t new_col = inv_col[node->col];
            sparse_err_t err = sparse_insert(out, i, new_col, node->value);
            if (err != SPARSE_OK) {
                free(inv_col);
                sparse_free(out);
                return err;
            }
            node = node->right;
        }
    }

    free(inv_col);
    *B = out;
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Internal: CSR adjacency graph of A + A^T (symmetrized, no self-loops)
 *
 * adj_ptr[i] .. adj_ptr[i+1]-1 index into adj_list[] for neighbors of i.
 * Caller must free both adj_ptr and adj_list.
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_build_adj(const SparseMatrix *A,
                              idx_t **adj_ptr_out, idx_t **adj_list_out)
{
    idx_t n = A->rows;

    /* Pass 1: count upper-bound degree per node.
     * For each off-diagonal (i,j) in A, both i and j get a neighbor. */
    idx_t *degree = calloc((size_t)n, sizeof(idx_t));
    if (!degree) return SPARSE_ERR_ALLOC;

    for (idx_t i = 0; i < n; i++) {
        Node *node = A->row_headers[i];
        while (node) {
            idx_t j = node->col;
            if (j != i && j < n) {
                degree[i]++;
                degree[j]++;
            }
            node = node->right;
        }
    }

    /* Build initial CSR pointers (over-allocated for duplicates) */
    idx_t *adj_ptr = malloc((size_t)(n + 1) * sizeof(idx_t));
    if (!adj_ptr) { free(degree); return SPARSE_ERR_ALLOC; }

    adj_ptr[0] = 0;
    for (idx_t i = 0; i < n; i++)
        adj_ptr[i + 1] = adj_ptr[i] + degree[i];

    idx_t total = adj_ptr[n];
    idx_t *adj_list = malloc((size_t)(total > 0 ? total : 1) * sizeof(idx_t));
    if (!adj_list) { free(degree); free(adj_ptr); return SPARSE_ERR_ALLOC; }

    /* Pass 2: fill adjacency lists (may contain duplicates) */
    idx_t *cursor = calloc((size_t)n, sizeof(idx_t));
    if (!cursor) { free(degree); free(adj_ptr); free(adj_list); return SPARSE_ERR_ALLOC; }

    for (idx_t i = 0; i < n; i++) {
        Node *node = A->row_headers[i];
        while (node) {
            idx_t j = node->col;
            if (j != i && j < n) {
                adj_list[adj_ptr[i] + cursor[i]++] = j;
                adj_list[adj_ptr[j] + cursor[j]++] = i;
            }
            node = node->right;
        }
    }

    /* Sort each adjacency list and remove duplicates in-place */
    for (idx_t i = 0; i < n; i++) {
        idx_t start = adj_ptr[i];
        idx_t len = cursor[i];
        if (len <= 1) { degree[i] = len; continue; }

        /* Insertion sort */
        idx_t *list = &adj_list[start];
        for (idx_t a = 1; a < len; a++) {
            idx_t key = list[a];
            idx_t b = a - 1;
            while (b >= 0 && list[b] > key) {
                list[b + 1] = list[b];
                b--;
            }
            list[b + 1] = key;
        }

        /* Remove consecutive duplicates */
        idx_t write = 1;
        for (idx_t a = 1; a < len; a++) {
            if (list[a] != list[a - 1])
                list[write++] = list[a];
        }
        degree[i] = write;  /* compacted count */
    }

    /* Rebuild into compacted arrays */
    idx_t *final_ptr = malloc((size_t)(n + 1) * sizeof(idx_t));
    if (!final_ptr) {
        free(degree); free(cursor); free(adj_ptr); free(adj_list);
        return SPARSE_ERR_ALLOC;
    }
    final_ptr[0] = 0;
    for (idx_t i = 0; i < n; i++)
        final_ptr[i + 1] = final_ptr[i] + degree[i];

    idx_t final_total = final_ptr[n];
    idx_t *final_list = malloc((size_t)(final_total > 0 ? final_total : 1) * sizeof(idx_t));
    if (!final_list) {
        free(degree); free(cursor); free(adj_ptr); free(adj_list); free(final_ptr);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++) {
        memcpy(&final_list[final_ptr[i]],
               &adj_list[adj_ptr[i]],
               (size_t)degree[i] * sizeof(idx_t));
    }

    free(degree);
    free(cursor);
    free(adj_ptr);
    free(adj_list);

    *adj_ptr_out = final_ptr;
    *adj_list_out = final_list;
    return SPARSE_OK;
}

/* ─── RCM ────────────────────────────────────────────────────────────── */

sparse_err_t sparse_reorder_rcm(const SparseMatrix *A, idx_t *perm)
{
    if (!A || !perm) return SPARSE_ERR_NULL;
    if (A->rows != A->cols) return SPARSE_ERR_SHAPE;
    /* TODO: implement in Day 7 */
    idx_t n = A->rows;
    for (idx_t i = 0; i < n; i++) perm[i] = i;  /* identity for now */
    return SPARSE_OK;
}

/* ─── AMD ────────────────────────────────────────────────────────────── */

sparse_err_t sparse_reorder_amd(const SparseMatrix *A, idx_t *perm)
{
    if (!A || !perm) return SPARSE_ERR_NULL;
    if (A->rows != A->cols) return SPARSE_ERR_SHAPE;
    /* TODO: implement in Day 9 */
    idx_t n = A->rows;
    for (idx_t i = 0; i < n; i++) perm[i] = i;  /* identity for now */
    return SPARSE_OK;
}
