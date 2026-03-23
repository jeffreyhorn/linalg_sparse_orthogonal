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

/* Comparison function for sorting neighbor indices by degree (ascending) */
typedef struct { idx_t node; idx_t deg; } node_deg_t;

static int cmp_node_deg(const void *a, const void *b)
{
    idx_t da = ((const node_deg_t *)a)->deg;
    idx_t db = ((const node_deg_t *)b)->deg;
    return (da > db) - (da < db);
}

/* Find a pseudo-peripheral node using repeated BFS.
 * Start from node 'start', BFS to find the farthest node, repeat
 * until the eccentricity stops increasing. Returns a good starting node. */
static idx_t find_pseudo_peripheral(idx_t n, const idx_t *adj_ptr,
                                    const idx_t *adj_list, idx_t start,
                                    idx_t *visited, idx_t *queue)
{
    idx_t best = start;
    idx_t best_ecc = 0;

    for (int attempt = 0; attempt < 5; attempt++) {
        /* BFS from 'best' */
        memset(visited, -1, (size_t)n * sizeof(idx_t));
        idx_t head = 0, tail = 0;
        queue[tail++] = best;
        visited[best] = 0;
        idx_t last = best;

        while (head < tail) {
            idx_t u = queue[head++];
            last = u;
            for (idx_t k = adj_ptr[u]; k < adj_ptr[u + 1]; k++) {
                idx_t v = adj_list[k];
                if (visited[v] < 0) {
                    visited[v] = visited[u] + 1;
                    queue[tail++] = v;
                }
            }
        }

        idx_t ecc = visited[last];
        if (ecc <= best_ecc) break;  /* no improvement */
        best_ecc = ecc;
        best = last;
    }

    return best;
}

sparse_err_t sparse_reorder_rcm(const SparseMatrix *A, idx_t *perm)
{
    if (!A || !perm) return SPARSE_ERR_NULL;
    if (A->rows != A->cols) return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;
    if (n == 0) return SPARSE_OK;
    if (n == 1) { perm[0] = 0; return SPARSE_OK; }

    /* Build adjacency graph */
    idx_t *adj_ptr = NULL, *adj_list = NULL;
    sparse_err_t err = sparse_build_adj(A, &adj_ptr, &adj_list);
    if (err != SPARSE_OK) return err;

    /* Allocate workspace */
    idx_t *visited = malloc((size_t)n * sizeof(idx_t));   /* -1 = unvisited, else BFS level */
    idx_t *queue   = malloc((size_t)n * sizeof(idx_t));   /* BFS queue */
    node_deg_t *nbuf = malloc((size_t)n * sizeof(node_deg_t)); /* neighbor sort buffer */
    if (!visited || !queue || !nbuf) {
        free(visited); free(queue); free(nbuf);
        free(adj_ptr); free(adj_list);
        return SPARSE_ERR_ALLOC;
    }

    memset(visited, -1, (size_t)n * sizeof(idx_t));
    idx_t perm_pos = 0;  /* next position in Cuthill-McKee ordering */

    /* Process each connected component */
    for (idx_t s = 0; s < n; s++) {
        if (visited[s] >= 0) continue;  /* already visited */

        /* Find pseudo-peripheral starting node for this component */
        /* Use a temporary BFS for the heuristic (reuses visited/queue) */
        idx_t start = find_pseudo_peripheral(n, adj_ptr, adj_list, s,
                                             visited, queue);

        /* Reset visited for the actual Cuthill-McKee BFS */
        /* Only reset nodes that were marked in the pseudo-peripheral search */
        for (idx_t i = 0; i < n; i++) {
            if (visited[i] >= 0 && i != start)
                visited[i] = -1;
        }
        /* Also need to reset start if it was set by the heuristic
         * but we haven't done the "real" marking yet. Let's just
         * re-mark everything fresh for this component. */

        /* Actually, simpler: reset all unplaced nodes */
        for (idx_t i = 0; i < n; i++) {
            /* Keep visited[i] >= 0 only if already in perm (placed before this component) */
            if (visited[i] >= 0) {
                /* Check if it's already placed */
                int placed = 0;
                for (idx_t p = 0; p < perm_pos; p++) {
                    if (perm[p] == i) { placed = 1; break; }
                }
                if (!placed) visited[i] = -1;
            }
        }

        /* BFS from start, visiting neighbors in order of increasing degree */
        idx_t head = 0, tail = 0;
        queue[tail++] = start;
        visited[start] = 0;

        while (head < tail) {
            idx_t u = queue[head++];
            perm[perm_pos++] = u;

            /* Collect unvisited neighbors and sort by degree */
            idx_t ncount = 0;
            for (idx_t k = adj_ptr[u]; k < adj_ptr[u + 1]; k++) {
                idx_t v = adj_list[k];
                if (visited[v] < 0) {
                    nbuf[ncount].node = v;
                    nbuf[ncount].deg = adj_ptr[v + 1] - adj_ptr[v];
                    ncount++;
                    visited[v] = 0;  /* mark to avoid re-adding */
                }
            }

            if (ncount > 1)
                qsort(nbuf, (size_t)ncount, sizeof(node_deg_t), cmp_node_deg);

            for (idx_t k = 0; k < ncount; k++)
                queue[tail++] = nbuf[k].node;
        }
    }

    /* Reverse the ordering: Cuthill-McKee → Reverse Cuthill-McKee */
    for (idx_t i = 0; i < n / 2; i++) {
        idx_t tmp = perm[i];
        perm[i] = perm[n - 1 - i];
        perm[n - 1 - i] = tmp;
    }

    free(visited);
    free(queue);
    free(nbuf);
    free(adj_ptr);
    free(adj_list);
    return SPARSE_OK;
}

/* ─── AMD ────────────────────────────────────────────────────────────── */

/*
 * Minimum degree ordering using bitset adjacency.
 *
 * For each elimination step:
 *  1. Pick the uneliminated node with smallest degree
 *  2. Record it in the permutation
 *  3. For each uneliminated neighbor u, merge the eliminated node's
 *     neighbors into u's adjacency (this models fill-in)
 *  4. Remove the eliminated node from all adjacency sets
 *
 * Uses one bitset row per node (n bits each). For n=1030, total memory
 * is ~130 KB — well within budget for our target matrix sizes.
 */

/* Bitset helpers: one bit per node, packed into 64-bit words */
typedef uint64_t bword_t;
#define BWORD_BITS 64
#define BWORD_IDX(i)  ((i) / BWORD_BITS)
#define BWORD_BIT(i)  ((bword_t)1 << ((i) % BWORD_BITS))

static inline void bset(bword_t *bs, idx_t i)   { bs[BWORD_IDX(i)] |= BWORD_BIT(i); }
static inline void bclr(bword_t *bs, idx_t i)   { bs[BWORD_IDX(i)] &= ~BWORD_BIT(i); }
static inline int  btest(const bword_t *bs, idx_t i) { return (bs[BWORD_IDX(i)] & BWORD_BIT(i)) != 0; }

/* Union: dst |= src */
static void bunion(bword_t *dst, const bword_t *src, idx_t nwords)
{
    for (idx_t w = 0; w < nwords; w++)
        dst[w] |= src[w];
}

sparse_err_t sparse_reorder_amd(const SparseMatrix *A, idx_t *perm)
{
    if (!A || !perm) return SPARSE_ERR_NULL;
    if (A->rows != A->cols) return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;
    if (n == 0) return SPARSE_OK;
    if (n == 1) { perm[0] = 0; return SPARSE_OK; }

    /* Build adjacency graph */
    idx_t *adj_ptr = NULL, *adj_list = NULL;
    sparse_err_t err = sparse_build_adj(A, &adj_ptr, &adj_list);
    if (err != SPARSE_OK) return err;

    /* Allocate bitset adjacency matrix: n rows, each with nwords words */
    idx_t nwords = (n + BWORD_BITS - 1) / BWORD_BITS;
    bword_t *adj_bits = calloc((size_t)n * (size_t)nwords, sizeof(bword_t));
    int *eliminated = calloc((size_t)n, sizeof(int));
    idx_t *degree = malloc((size_t)n * sizeof(idx_t));

    if (!adj_bits || !eliminated || !degree) {
        free(adj_bits); free(eliminated); free(degree);
        free(adj_ptr); free(adj_list);
        return SPARSE_ERR_ALLOC;
    }

    /* Initialize bitset adjacency from CSR graph */
    for (idx_t i = 0; i < n; i++) {
        bword_t *row = &adj_bits[(size_t)i * (size_t)nwords];
        for (idx_t k = adj_ptr[i]; k < adj_ptr[i + 1]; k++)
            bset(row, adj_list[k]);
        degree[i] = adj_ptr[i + 1] - adj_ptr[i];
    }

    free(adj_ptr);
    free(adj_list);

    /* Main elimination loop */
    for (idx_t step = 0; step < n; step++) {
        /* Find uneliminated node with minimum degree */
        idx_t best = -1;
        idx_t best_deg = n + 1;
        for (idx_t i = 0; i < n; i++) {
            if (!eliminated[i] && degree[i] < best_deg) {
                best_deg = degree[i];
                best = i;
            }
        }

        perm[step] = best;
        eliminated[best] = 1;

        bword_t *best_row = &adj_bits[(size_t)best * (size_t)nwords];

        /* For each uneliminated neighbor u of best:
         *   adj[u] = (adj[u] | adj[best]) \ {best, u}
         * This merges the eliminated node's connections into u's,
         * modeling the fill-in that LU elimination would create. */
        for (idx_t u = 0; u < n; u++) {
            if (eliminated[u] || !btest(best_row, u)) continue;

            bword_t *u_row = &adj_bits[(size_t)u * (size_t)nwords];
            bunion(u_row, best_row, nwords);
            bclr(u_row, best);  /* remove eliminated node */
            bclr(u_row, u);     /* no self-loop */

            /* Recount degree (number of uneliminated neighbors) */
            idx_t deg = 0;
            for (idx_t w = 0; w < nwords; w++) {
                bword_t v = u_row[w];
                while (v) { v &= v - 1; deg++; }
            }
            /* Subtract eliminated nodes from degree */
            idx_t elim_in_adj = 0;
            for (idx_t j = 0; j < n; j++) {
                if (eliminated[j] && btest(u_row, j))
                    elim_in_adj++;
            }
            degree[u] = deg - elim_in_adj;
        }

        /* Clear best from all adjacency rows */
        for (idx_t i = 0; i < n; i++) {
            bclr(&adj_bits[(size_t)i * (size_t)nwords], best);
        }
    }

    free(adj_bits);
    free(eliminated);
    free(degree);
    return SPARSE_OK;
}
