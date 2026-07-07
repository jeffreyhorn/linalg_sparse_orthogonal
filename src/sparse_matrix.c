#include "sparse_matrix.h"
#include "sparse_alloc_internal.h"
#include "sparse_matrix_internal.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifdef SPARSE_OPENMP
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <omp.h>
#pragma GCC diagnostic pop
#endif

/* ─── Pool allocator ─────────────────────────────────────────────────── */

Node *pool_alloc(NodePool *pool) {
    /* Try the free list first */
    if (pool->free_list) {
        Node *node = pool->free_list;
        pool->free_list = node->right;
        return node;
    }

    /* Allocate from current slab or create a new one */
    if (!pool->current || pool->current->used >= NODES_PER_SLAB) {
        NodeSlab *slab = malloc(sizeof(NodeSlab));
        if (!slab)
            return NULL;
        slab->used = 0;
        slab->next = NULL;
        if (pool->current)
            pool->current->next = slab;
        else
            pool->head = slab;
        pool->current = slab;
        pool->num_slabs++;
    }

    return &pool->current->nodes[pool->current->used++];
}

void pool_release(NodePool *pool, Node *node) {
    /* Push onto the free list (reuse ->right as the next pointer) */
    node->right = pool->free_list;
    pool->free_list = node;
}

void pool_free_all(NodePool *pool) {
    NodeSlab *slab = pool->head;
    while (slab) {
        NodeSlab *next = slab->next;
        free(slab);
        slab = next;
    }
    pool->head = NULL;
    pool->current = NULL;
    pool->free_list = NULL;
    pool->num_slabs = 0;
}

/* ─── Helpers ────────────────────────────────────────────────────────── */

sparse_err_t sparse_stream_vprintf_checked(FILE *stream, const char *fmt, va_list ap) {
    int rc = 0;
    if (!stream || !fmt)
        return SPARSE_ERR_NULL;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
    errno = 0;
    rc = vfprintf(stream, fmt, ap);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    if (rc < 0) {
        sparse_set_errno_(errno != 0 ? errno : EIO);
        return SPARSE_ERR_IO;
    }
    return SPARSE_OK;
}

sparse_err_t sparse_stream_printf_checked(FILE *stream, const char *fmt, ...) {
    sparse_err_t err;
    va_list ap;
    va_start(ap, fmt);
    err = sparse_stream_vprintf_checked(stream, fmt, ap);
    va_end(ap);
    return err;
}

Node *sparse_matrix_make_node(SparseMatrix *mat, idx_t r, idx_t c, sparse_scalar_t v) {
    Node *n = pool_alloc(&mat->pool);
    if (!n)
        return NULL;
    n->row = r;
    n->col = c;
    n->value = v;
    n->right = NULL;
    n->down = NULL;
    return n;
}

static int sparse_matrix_has_non_identity_row_col_perms(const SparseMatrix *mat) {
    if (!mat)
        return 0;
    for (idx_t i = 0; i < mat->rows; i++) {
        if ((mat->row_perm && mat->row_perm[i] != i) ||
            (mat->inv_row_perm && mat->inv_row_perm[i] != i))
            return 1;
    }
    for (idx_t i = 0; i < mat->cols; i++) {
        if ((mat->col_perm && mat->col_perm[i] != i) ||
            (mat->inv_col_perm && mat->inv_col_perm[i] != i))
            return 1;
    }
    return 0;
}

static void sparse_matrix_free_shell_buffers(SparseMatrix *mat) {
    if (!mat)
        return;
    free(mat->row_headers);
    free(mat->col_headers);
    free(mat->row_perm);
    free(mat->inv_row_perm);
    free(mat->col_perm);
    free(mat->inv_col_perm);
    mat->row_headers = NULL;
    mat->col_headers = NULL;
    mat->row_perm = NULL;
    mat->inv_row_perm = NULL;
    mat->col_perm = NULL;
    mat->inv_col_perm = NULL;
}

static sparse_err_t sparse_matrix_alloc_shell_buffers(SparseMatrix *mat, idx_t rows, idx_t cols) {
    if (!mat)
        return SPARSE_ERR_NULL;
    if (sparse_calloc_idx_array(rows, sizeof(Node *), (void **)&mat->row_headers) != SPARSE_OK ||
        sparse_calloc_idx_array(cols, sizeof(Node *), (void **)&mat->col_headers) != SPARSE_OK ||
        sparse_malloc_idx_array(rows, sizeof(idx_t), (void **)&mat->row_perm) != SPARSE_OK ||
        sparse_malloc_idx_array(rows, sizeof(idx_t), (void **)&mat->inv_row_perm) != SPARSE_OK ||
        sparse_malloc_idx_array(cols, sizeof(idx_t), (void **)&mat->col_perm) != SPARSE_OK ||
        sparse_malloc_idx_array(cols, sizeof(idx_t), (void **)&mat->inv_col_perm) != SPARSE_OK) {
        sparse_matrix_free_shell_buffers(mat);
        return SPARSE_ERR_ALLOC;
    }
    return SPARSE_OK;
}

static int sparse_memory_usage_add(size_t *total, size_t addend) {
    size_t next = 0;
    if (sparse_size_add_overflow(*total, addend, &next)) {
        *total = SIZE_MAX;
        return 1;
    }
    *total = next;
    return 0;
}

static int sparse_memory_usage_add_idx_bytes(size_t *total, idx_t count, size_t elem_size) {
    size_t bytes = 0;
    if (sparse_idx_count_bytes_overflow(count, elem_size, &bytes)) {
        *total = SIZE_MAX;
        return 1;
    }
    return sparse_memory_usage_add(total, bytes);
}

/* ─── Lifecycle ──────────────────────────────────────────────────────── */

SparseMatrix *sparse_create(idx_t rows, idx_t cols) {
    if (rows <= 0 || cols <= 0)
        return NULL;

    SparseMatrix *mat = malloc(sizeof(SparseMatrix));
    if (!mat)
        return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->nnz = 0;
    mat->cached_norm = -1.0;
    mat->factor_norm = -1.0;
    mat->factored = 0;
    mat->factor_state = NULL;
    mat->reorder_perm = NULL;
    mat->row_headers = NULL;
    mat->col_headers = NULL;
    mat->row_perm = NULL;
    mat->inv_row_perm = NULL;
    mat->col_perm = NULL;
    mat->inv_col_perm = NULL;

    if (sparse_matrix_alloc_shell_buffers(mat, rows, cols) != SPARSE_OK) {
        free(mat);
        return NULL;
    }

#ifdef SPARSE_MUTEX
    if (pthread_mutex_init(&mat->mtx, NULL) != 0) {
        sparse_matrix_free_shell_buffers(mat);
        free(mat);
        return NULL;
    }
#endif

    for (idx_t i = 0; i < rows; i++) {
        mat->row_perm[i] = i;
        mat->inv_row_perm[i] = i;
    }
    for (idx_t j = 0; j < cols; j++) {
        mat->col_perm[j] = j;
        mat->inv_col_perm[j] = j;
    }

    mat->pool.head = NULL;
    mat->pool.current = NULL;
    mat->pool.free_list = NULL;
    mat->pool.num_slabs = 0;

    return mat;
}

void sparse_free(SparseMatrix *mat) {
    if (!mat)
        return;
    pool_free_all(&mat->pool);
    sparse_matrix_free_shell_buffers(mat);
    free(mat->factor_state);
    free(mat->reorder_perm);
#ifdef SPARSE_MUTEX
    pthread_mutex_destroy(&mat->mtx);
#endif
    free(mat);
}

SparseMatrix *sparse_copy(const SparseMatrix *mat) {
    SparseBuildEntry *entries = NULL;
    size_t row_perm_bytes = 0;
    size_t col_perm_bytes = 0;
    if (!mat)
        return NULL;

    SparseMatrix *copy = NULL;
    if (sparse_malloc_idx_array(mat->nnz, sizeof(*entries), (void **)&entries) != SPARSE_OK)
        return NULL;

    idx_t entry_idx = 0;
    for (idx_t i = 0; i < mat->rows; i++) {
        Node *src = mat->row_headers[i];
        while (src) {
            entries[entry_idx] = (SparseBuildEntry){
                .row = src->row,
                .col = src->col,
                .value = src->value,
                .order = entry_idx,
            };
            entry_idx++;
            src = src->right;
        }
    }

    if (sparse_matrix_build_from_entries(mat->rows, mat->cols, entries, entry_idx,
                                         /*entries_sorted=*/1, &copy) != SPARSE_OK) {
        free(entries);
        return NULL;
    }
    free(entries);

    /* Copy permutation arrays */
    if (sparse_idx_count_bytes_overflow(mat->rows, sizeof(idx_t), &row_perm_bytes) ||
        sparse_idx_count_bytes_overflow(mat->cols, sizeof(idx_t), &col_perm_bytes)) {
        sparse_free(copy);
        return NULL;
    }
    memcpy(copy->row_perm, mat->row_perm, row_perm_bytes);
    memcpy(copy->inv_row_perm, mat->inv_row_perm, row_perm_bytes);
    memcpy(copy->col_perm, mat->col_perm, col_perm_bytes);
    memcpy(copy->inv_col_perm, mat->inv_col_perm, col_perm_bytes);

    /* Preserve cached norm, factor norm, and factored flag from source */
    copy->cached_norm = mat->cached_norm;
    copy->factor_norm = mat->factor_norm;
    copy->factored = mat->factored;
    if (sparse_factor_state_clone(copy, mat) != SPARSE_OK) {
        sparse_free(copy);
        return NULL;
    }

    /* Copy reorder permutation if present */
    if (mat->reorder_perm) {
        if (sparse_malloc_idx_array(mat->rows, sizeof(idx_t), (void **)&copy->reorder_perm) !=
            SPARSE_OK) {
            sparse_free(copy);
            return NULL;
        }
        memcpy(copy->reorder_perm, mat->reorder_perm, row_perm_bytes);
    }

    return copy;
}

SparseMatrix *sparse_transpose(const SparseMatrix *A) {
    SparseBuildEntry *entries = NULL;
    if (!A)
        return NULL;

    if (sparse_malloc_idx_array(A->nnz, sizeof(*entries), (void **)&entries) != SPARSE_OK)
        return NULL;

    idx_t entry_idx = 0;
    for (idx_t i = 0; i < A->rows; i++) {
        Node *nd = A->row_headers[i];
        while (nd) {
            entries[entry_idx] = (SparseBuildEntry){
                .row = nd->col,
                .col = nd->row,
                .value = nd->value,
                .order = entry_idx,
            };
            entry_idx++;
            nd = nd->right;
        }
    }

    SparseMatrix *T = NULL;
    if (sparse_matrix_build_from_entries(A->cols, A->rows, entries, entry_idx,
                                         /*entries_sorted=*/0, &T) != SPARSE_OK) {
        free(entries);
        return NULL;
    }
    free(entries);
    return T;
}

/* ─── Element access (physical) ──────────────────────────────────────── */

static sparse_err_t sparse_remove_internal(SparseMatrix *mat, idx_t row, idx_t col);

sparse_err_t sparse_insert(SparseMatrix *mat, idx_t row, idx_t col, sparse_scalar_t val) {
    if (!mat)
        return SPARSE_ERR_NULL;
    if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return SPARSE_ERR_BOUNDS;

    SPARSE_LOCK(mat);
    sparse_factor_state_set_factored(mat, 0);

    if (val == 0.0) {
        sparse_err_t err = sparse_remove_internal(mat, row, col);
        SPARSE_UNLOCK(mat);
        return err;
    }

    /* Walk the row list to find insertion point (sorted by col) */
    Node *prev_r = NULL;
    Node *curr_r = mat->row_headers[row];
    while (curr_r && curr_r->col < col) {
        prev_r = curr_r;
        curr_r = curr_r->right;
    }

    /* If node already exists, update its value */
    if (curr_r && curr_r->col == col) {
        curr_r->value = val;
        mat->cached_norm = -1.0;
        SPARSE_UNLOCK(mat);
        return SPARSE_OK;
    }

    /* Create a new node */
    Node *node = sparse_matrix_make_node(mat, row, col, val);
    if (!node) {
        SPARSE_UNLOCK(mat);
        return SPARSE_ERR_ALLOC;
    }
    mat->nnz++;
    mat->cached_norm = -1.0;

    /* Link into row list */
    node->right = curr_r;
    if (prev_r)
        prev_r->right = node;
    else
        mat->row_headers[row] = node;

    /* Link into column list (sorted by row) */
    Node *prev_c = NULL;
    Node *curr_c = mat->col_headers[col];
    while (curr_c && curr_c->row < row) {
        prev_c = curr_c;
        curr_c = curr_c->down;
    }
    node->down = curr_c;
    if (prev_c)
        prev_c->down = node;
    else
        mat->col_headers[col] = node;

    SPARSE_UNLOCK(mat);
    return SPARSE_OK;
}

/* Internal remove (no locking — called from within locked sparse_insert) */
static sparse_err_t sparse_remove_internal(SparseMatrix *mat, idx_t row, idx_t col) {
    /* Find and unlink from row list */
    Node *prev = NULL;
    Node *curr = mat->row_headers[row];
    while (curr && curr->col != col) {
        prev = curr;
        curr = curr->right;
    }
    if (!curr)
        return SPARSE_OK; /* Not present — not an error */

    if (prev)
        prev->right = curr->right;
    else
        mat->row_headers[row] = curr->right;

    /* Find and unlink from column list */
    prev = NULL;
    Node *ccol = mat->col_headers[col];
    while (ccol && ccol->row != row) {
        prev = ccol;
        ccol = ccol->down;
    }
    if (prev)
        prev->down = ccol->down; // NOLINT(clang-analyzer-core.NullDereference)
    else
        mat->col_headers[col] = ccol->down; // NOLINT(clang-analyzer-core.NullDereference)

    pool_release(&mat->pool, curr);
    mat->nnz--;
    mat->cached_norm = -1.0;

    return SPARSE_OK;
}

sparse_err_t sparse_remove(SparseMatrix *mat, idx_t row, idx_t col) {
    if (!mat)
        return SPARSE_ERR_NULL;
    if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return SPARSE_ERR_BOUNDS;
    SPARSE_LOCK(mat);
    sparse_factor_state_clear(mat);
    sparse_err_t err = sparse_remove_internal(mat, row, col);
    SPARSE_UNLOCK(mat);
    return err;
}

sparse_scalar_t sparse_get_phys(const SparseMatrix *mat, idx_t row, idx_t col) {
    if (!mat || row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return 0.0;

    Node *curr = mat->row_headers[row];
    while (curr && curr->col < col)
        curr = curr->right;

    return (curr && curr->col == col) ? curr->value : 0.0;
}

/* ─── Element access (logical — through permutations) ────────────────── */

sparse_scalar_t sparse_get(const SparseMatrix *mat, idx_t row, idx_t col) {
    if (!mat || row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return 0.0;
    return sparse_get_phys(mat, mat->row_perm[row], mat->col_perm[col]);
}

sparse_err_t sparse_set(SparseMatrix *mat, idx_t row, idx_t col, sparse_scalar_t val) {
    if (!mat)
        return SPARSE_ERR_NULL;
    if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return SPARSE_ERR_BOUNDS;
    return sparse_insert(mat, mat->row_perm[row], mat->col_perm[col], val);
}

/* ─── Matrix information ─────────────────────────────────────────────── */

idx_t sparse_rows(const SparseMatrix *mat) { return mat ? mat->rows : 0; }

idx_t sparse_cols(const SparseMatrix *mat) { return mat ? mat->cols : 0; }

idx_t sparse_nnz(const SparseMatrix *mat) { return mat ? mat->nnz : 0; }

size_t sparse_memory_usage(const SparseMatrix *mat) {
    if (!mat)
        return 0;
    size_t total = sizeof(SparseMatrix);
    if (sparse_memory_usage_add_idx_bytes(&total, mat->rows, sizeof(Node *)) ||
        sparse_memory_usage_add_idx_bytes(&total, mat->cols, sizeof(Node *)) ||
        sparse_memory_usage_add_idx_bytes(&total, mat->rows, 2 * sizeof(idx_t)) ||
        sparse_memory_usage_add_idx_bytes(&total, mat->cols, 2 * sizeof(idx_t)) ||
        (mat->reorder_perm &&
         sparse_memory_usage_add_idx_bytes(&total, mat->rows, sizeof(idx_t))) ||
        sparse_memory_usage_add_idx_bytes(&total, mat->pool.num_slabs, sizeof(NodeSlab)))
        return SIZE_MAX;
    return total;
}

/* ─── Symmetry check ─────────────────────────────────────────────────── */

int sparse_is_symmetric(const SparseMatrix *mat, sparse_scalar_t tol) {
    if (!mat)
        return 0;
    if (mat->rows != mat->cols)
        return 0;
    if (!(tol >= 0.0))
        return 0; /* rejects negative and NaN */

    /* O(nnz) check: for each entry A(i,j), walk column j's list to find
     * the matching A(j,i) entry. Since column lists are sorted by row,
     * we can do a two-pointer walk per row/column pair. But the simplest
     * O(nnz) approach: for each row i, compare the row list against
     * column i's list — both are sorted, so a single parallel scan suffices. */
    for (idx_t i = 0; i < mat->rows; i++) {
        Node *row_node = mat->row_headers[i]; /* entries in row i, sorted by col */
        Node *col_node = mat->col_headers[i]; /* entries in col i, sorted by row */
        while (row_node && col_node) {
            if (row_node->col < col_node->row) {
                /* Entry in row i, col j with no matching entry in col i, row j */
                if (fabs(row_node->value) > tol)
                    return 0;
                row_node = row_node->right;
            } else if (row_node->col > col_node->row) {
                if (fabs(col_node->value) > tol)
                    return 0;
                col_node = col_node->down;
            } else {
                /* Same position: check A(i,j) == A(j,i) */
                if (fabs(row_node->value - col_node->value) > tol)
                    return 0;
                row_node = row_node->right;
                col_node = col_node->down;
            }
        }
        /* Any remaining entries must be within tolerance of zero */
        while (row_node) {
            if (fabs(row_node->value) > tol)
                return 0;
            row_node = row_node->right;
        }
        while (col_node) {
            if (fabs(col_node->value) > tol)
                return 0;
            col_node = col_node->down;
        }
    }
    return 1;
}

/* ─── Infinity norm ──────────────────────────────────────────────────── */

sparse_err_t sparse_norminf(SparseMatrix *mat, sparse_scalar_t *norm) {
    if (!mat || !norm)
        return SPARSE_ERR_NULL;

    /* Return cached value if valid.  Relaxed ordering suffices: the cached
     * value is idempotent (all threads compute the same result from the
     * same immutable linked-list structure). */
    sparse_scalar_t cached = atomic_load_explicit(&mat->cached_norm, memory_order_relaxed);
    if (cached >= 0.0) {
        *norm = cached;
        return SPARSE_OK;
    }

    sparse_scalar_t max_row_sum = 0.0;
    for (idx_t i = 0; i < mat->rows; i++) {
        sparse_scalar_t row_sum = 0.0;
        Node *node = mat->row_headers[i];
        while (node) {
            row_sum += fabs(node->value);
            node = node->right;
        }
        if (row_sum > max_row_sum)
            max_row_sum = row_sum;
    }

    atomic_store_explicit(&mat->cached_norm, max_row_sum, memory_order_relaxed);
    *norm = max_row_sum;
    return SPARSE_OK;
}

/* ─── Mark as factored ──────────────────────────────────────────────── */

sparse_err_t sparse_mark_factored(SparseMatrix *mat) {
    if (!mat)
        return SPARSE_ERR_NULL;
    if (mat->rows != mat->cols)
        return SPARSE_ERR_SHAPE;
    /* Compute factor_norm if not already set */
    if (sparse_factor_state_factor_norm(mat) < 0.0) {
        sparse_scalar_t norm;
        sparse_err_t err = sparse_norminf(mat, &norm);
        if (err != SPARSE_OK)
            return err;
        sparse_factor_state_set_factor_norm(mat, norm);
    }
    sparse_factor_state_set_factored(mat, 1);
    return SPARSE_OK;
}

/* ─── Matrix arithmetic ──────────────────────────────────────────────── */

sparse_err_t sparse_scale(SparseMatrix *mat, sparse_scalar_t alpha) {
    if (!mat)
        return SPARSE_ERR_NULL;

    if (alpha == 0.0) {
        /* Remove all entries */
        for (idx_t i = 0; i < mat->rows; i++) {
            Node *node = mat->row_headers[i];
            while (node) {
                Node *next = node->right;
                pool_release(&mat->pool, node);
                node = next;
            }
            mat->row_headers[i] = NULL;
        }
        for (idx_t j = 0; j < mat->cols; j++)
            mat->col_headers[j] = NULL;
        mat->nnz = 0;
    } else {
        for (idx_t i = 0; i < mat->rows; i++) {
            Node *node = mat->row_headers[i];
            while (node) {
                node->value *= alpha;
                node = node->right;
            }
        }
    }

    mat->cached_norm = -1.0;
    sparse_factor_state_clear(mat);
    return SPARSE_OK;
}

/* NOTE: sparse_add() and sparse_add_inplace() operate in physical index space.
 * Do not use on matrices with non-identity permutations (e.g., after LU
 * factorization) — results would not correspond to logical matrix entries. */
sparse_err_t sparse_add(const SparseMatrix *A, const SparseMatrix *B, sparse_scalar_t alpha,
                        sparse_scalar_t beta, SparseMatrix **C_out) {
    if (!A || !B || !C_out)
        return SPARSE_ERR_NULL;
    *C_out = NULL;
    if (A->rows != B->rows || A->cols != B->cols)
        return SPARSE_ERR_SHAPE;

    SparseMatrix *C = sparse_create(A->rows, A->cols);
    if (!C)
        return SPARSE_ERR_ALLOC;

    /* Row-wise merge of A and B using sorted row lists (two-pointer walk) */
    for (idx_t i = 0; i < A->rows; i++) {
        Node *nA = A->row_headers[i];
        Node *nB = B->row_headers[i];

        while (nA && nB) {
            sparse_scalar_t val;
            idx_t col;
            if (nA->col < nB->col) {
                val = alpha * nA->value;
                col = nA->col;
                nA = nA->right;
            } else if (nB->col < nA->col) {
                val = beta * nB->value;
                col = nB->col;
                nB = nB->right;
            } else {
                val = alpha * nA->value + beta * nB->value;
                col = nA->col;
                nA = nA->right;
                nB = nB->right;
            }
            if (fabs(val) >= 1e-15) {
                sparse_err_t err = sparse_insert(C, i, col, val);
                if (err != SPARSE_OK) {
                    sparse_free(C);
                    return err;
                }
            }
        }
        while (nA) {
            sparse_scalar_t val = alpha * nA->value;
            if (fabs(val) >= 1e-15) {
                sparse_err_t err = sparse_insert(C, i, nA->col, val);
                if (err != SPARSE_OK) {
                    sparse_free(C);
                    return err;
                }
            }
            nA = nA->right;
        }
        while (nB) {
            sparse_scalar_t val = beta * nB->value;
            if (fabs(val) >= 1e-15) {
                sparse_err_t err = sparse_insert(C, i, nB->col, val);
                if (err != SPARSE_OK) {
                    sparse_free(C);
                    return err;
                }
            }
            nB = nB->right;
        }
    }

    *C_out = C;
    return SPARSE_OK;
}

sparse_err_t sparse_add_inplace(SparseMatrix *A, const SparseMatrix *B, sparse_scalar_t alpha,
                                sparse_scalar_t beta) {
    if (!A || !B)
        return SPARSE_ERR_NULL;
    if (A->rows != B->rows || A->cols != B->cols)
        return SPARSE_ERR_SHAPE;

    /* Invalidate cache early: A will be mutated even on partial failure */
    A->cached_norm = -1.0;
    sparse_factor_state_clear(A);

    /* Scale A by alpha */
    if (alpha != 1.0) {
        sparse_err_t err = sparse_scale(A, alpha);
        if (err != SPARSE_OK)
            return err;
    }

    /* Add beta * B using per-row cursor walk.
     * After insert/remove we must rescan from row head because the linked list
     * structure has changed, but we fast-forward past columns we've already
     * processed (nb->col) to avoid quadratic rescans. */
    for (idx_t i = 0; i < B->rows; i++) {
        Node *nb = B->row_headers[i];
        Node *na = A->row_headers[i];
        while (nb) {
            idx_t target_col = nb->col;
            /* Advance A's cursor to find or pass nb->col */
            while (na && na->col < target_col)
                na = na->right;
            if (na && na->col == target_col) {
                /* Entry exists in A — update in place */
                sparse_scalar_t val = na->value + beta * nb->value;
                if (fabs(val) < 1e-15) {
                    /* Cancellation — remove via insert(0.0) */
                    sparse_err_t ierr = sparse_insert(A, i, target_col, 0.0);
                    if (ierr != SPARSE_OK)
                        return ierr;
                    /* Row structure changed; rescan but skip past target_col */
                    na = A->row_headers[i];
                    while (na && na->col <= target_col)
                        na = na->right;
                } else {
                    na->value = val;
                }
            } else {
                /* No entry in A — insert only if non-negligible */
                sparse_scalar_t val = beta * nb->value;
                if (fabs(val) < 1e-15) {
                    nb = nb->right;
                    continue;
                }
                sparse_err_t err = sparse_insert(A, i, target_col, val);
                if (err != SPARSE_OK)
                    return err;
                /* Row structure changed; rescan but skip past target_col */
                na = A->row_headers[i];
                while (na && na->col <= target_col)
                    na = na->right;
            }
            nb = nb->right;
        }
    }

    return SPARSE_OK;
}

/* ─── Sparse matrix-matrix multiply (Gustavson's algorithm) ──────────── */

static int cmp_idx(const void *a, const void *b) {
    idx_t va = *(const idx_t *)a;
    idx_t vb = *(const idx_t *)b;
    return (va > vb) - (va < vb);
}

sparse_err_t sparse_matmul(const SparseMatrix *A, const SparseMatrix *B, SparseMatrix **C) {
    if (!C)
        return SPARSE_ERR_NULL;
    *C = NULL;
    if (!A || !B)
        return SPARSE_ERR_NULL;
    if (A->cols != B->rows)
        return SPARSE_ERR_SHAPE;

    idx_t m = A->rows;
    idx_t k = A->cols;
    idx_t nc = B->cols;
    (void)k;

    SparseMatrix *out = sparse_create(m, nc);
    if (!out)
        return SPARSE_ERR_ALLOC;

    /* Dense accumulator for one row of C, with compact touched-index list */
    sparse_scalar_t *acc = NULL;
    int *nz_flag = NULL;
    idx_t *touched = NULL;
    if (sparse_calloc_idx_array(nc, sizeof(sparse_scalar_t), (void **)&acc) != SPARSE_OK ||
        sparse_calloc_idx_array(nc, sizeof(int), (void **)&nz_flag) != SPARSE_OK ||
        sparse_malloc_idx_array(nc, sizeof(idx_t), (void **)&touched) != SPARSE_OK) {
        free(acc);
        free(nz_flag);
        free(touched);
        sparse_free(out);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < m; i++) {
        /* Accumulate row i of C: sum over j of A(i,j) * row_j(B) */
        idx_t ntouched = 0;
        Node *a_node = A->row_headers[i];
        while (a_node) {
            idx_t j = a_node->col;
            sparse_scalar_t a_ij = a_node->value;

            /* Add a_ij * row_j(B) to accumulator */
            Node *b_node = B->row_headers[j];
            while (b_node) {
                acc[b_node->col] += a_ij * b_node->value;
                if (!nz_flag[b_node->col]) {
                    nz_flag[b_node->col] = 1;
                    touched[ntouched++] = b_node->col;
                }
                b_node = b_node->right;
            }
            a_node = a_node->right;
        }

        /* Sort touched columns so inserts are in ascending order. sparse_insert
         * scans from the row head each time, so total flush cost per row is
         * O(nnz_row^2) in the worst case. Sorting avoids the pathological
         * reverse-order case and gives good practical performance. */
        if (ntouched > 1)
            qsort(touched, (size_t)ntouched, sizeof(idx_t), cmp_idx);

        /* Flush accumulator to sparse output (only touched columns) */
        for (idx_t t = 0; t < ntouched; t++) {
            idx_t col = touched[t];
            if (fabs(acc[col]) >= 1e-15) {
                sparse_err_t err = sparse_insert(out, i, col, acc[col]);
                if (err != SPARSE_OK) {
                    free(acc);
                    free(nz_flag);
                    free(touched);
                    sparse_free(out);
                    return err;
                }
            }
            acc[col] = 0.0;
            nz_flag[col] = 0;
        }
    }

    free(acc);
    free(nz_flag);
    free(touched);
    *C = out;
    return SPARSE_OK;
}

/* ─── Sparse matrix-vector product ───────────────────────────────────── */

sparse_err_t sparse_matvec(const SparseMatrix *mat, const sparse_scalar_t *x, sparse_scalar_t *y) {
    if (!mat || !x || !y)
        return SPARSE_ERR_NULL;

    idx_t nrows = mat->rows;

    /* Walk each physical row, accumulate y[logical_row].
     * Each row writes to a distinct y[log_i], so rows are independent
     * and safe to parallelize without synchronization. The library does
     * not set an OpenMP thread count here; callers should use the OpenMP
     * runtime (for example OMP_NUM_THREADS) when SPARSE_OPENMP is enabled. */
#ifdef SPARSE_OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (idx_t log_i = 0; log_i < nrows; log_i++) {
        idx_t phys_i = mat->row_perm[log_i];
        Node *node = mat->row_headers[phys_i];
        sparse_scalar_t sum = 0.0;
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            sum += node->value * x[log_j];
            node = node->right;
        }
        y[log_i] = sum;
    }

    return SPARSE_OK;
}

/* ─── Block SpMV: Y = A * X (multiple RHS) ───────────────────────────── */

sparse_err_t sparse_matvec_block(const SparseMatrix *mat, const sparse_scalar_t *X, idx_t nrhs,
                                 sparse_scalar_t *Y) {
    if (!mat || !X || !Y)
        return SPARSE_ERR_NULL;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;
    if (nrhs == 0)
        return SPARSE_OK;

    idx_t m = mat->rows;

    /* Overflow guard: ensure m*nrhs and cols*nrhs fit in size_t */
    if (m > 0 && (size_t)nrhs > SIZE_MAX / (size_t)m)
        return SPARSE_ERR_ALLOC;
    if (mat->cols > 0 && (size_t)nrhs > SIZE_MAX / (size_t)mat->cols)
        return SPARSE_ERR_ALLOC;

    /* Zero output */
    for (idx_t k = 0; k < nrhs; k++) {
        size_t ok = (size_t)m * (size_t)k;
        for (idx_t i = 0; i < m; i++)
            Y[(size_t)i + ok] = 0.0;
    }

    size_t sm = (size_t)m;
    size_t sc = (size_t)mat->cols;

    /* Walk each row once, update all nrhs columns.
     * Each row writes to distinct Y positions, so parallelization is safe.
     * Thread count remains owned by the OpenMP runtime, matching
     * sparse_matvec() above. */
#ifdef SPARSE_OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (idx_t log_i = 0; log_i < m; log_i++) {
        idx_t phys_i = mat->row_perm[log_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            sparse_scalar_t a_ij = node->value;
            sparse_scalar_t *y_ptr = Y + (size_t)log_i;
            const sparse_scalar_t *x_ptr = X + (size_t)log_j;
            for (idx_t k = 0; k < nrhs; k++) {
                *y_ptr += a_ij * (*x_ptr);
                y_ptr += sm;
                x_ptr += sc;
            }
            node = node->right;
        }
    }

    return SPARSE_OK;
}

/* ─── Display / debug ────────────────────────────────────────────────── */

sparse_err_t sparse_print_dense(const SparseMatrix *mat, FILE *stream) {
    if (!mat || !stream)
        return SPARSE_ERR_NULL;

    if (mat->rows > 50 || mat->cols > 50) {
        if (sparse_stream_printf_checked(stream,
                                         "[WARNING: matrix is %" SPARSE_PRIDX "x%" SPARSE_PRIDX
                                         ", dense print may be very large]\n",
                                         mat->rows, mat->cols) != SPARSE_OK)
            return SPARSE_ERR_IO;
    }

    for (idx_t i = 0; i < mat->rows; i++) {
        for (idx_t j = 0; j < mat->cols; j++) {
            if (sparse_stream_printf_checked(stream, "%10.4f ", sparse_get(mat, i, j)) != SPARSE_OK)
                return SPARSE_ERR_IO;
        }
        if (sparse_stream_printf_checked(stream, "\n") != SPARSE_OK)
            return SPARSE_ERR_IO;
    }

    return SPARSE_OK;
}

sparse_err_t sparse_print_entries(const SparseMatrix *mat, FILE *stream) {
    if (!mat || !stream)
        return SPARSE_ERR_NULL;

    for (idx_t log_i = 0; log_i < mat->rows; log_i++) {
        idx_t phys_i = mat->row_perm[log_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            if (sparse_stream_printf_checked(stream,
                                             "  (%" SPARSE_PRIDX ", %" SPARSE_PRIDX ") = %.15g\n",
                                             log_i, log_j, node->value) != SPARSE_OK)
                return SPARSE_ERR_IO;
            node = node->right;
        }
    }

    return SPARSE_OK;
}

sparse_err_t sparse_print_info(const SparseMatrix *mat, FILE *stream) {
    if (!mat || !stream)
        return SPARSE_ERR_NULL;

    if (sparse_stream_printf_checked(stream,
                                     "SparseMatrix: %" SPARSE_PRIDX " x %" SPARSE_PRIDX
                                     ", nnz = %" SPARSE_PRIDX ", memory ~ %zu bytes\n",
                                     mat->rows, mat->cols, mat->nnz,
                                     sparse_memory_usage(mat)) != SPARSE_OK)
        return SPARSE_ERR_IO;

    return SPARSE_OK;
}

/* ─── Permutation access ─────────────────────────────────────────────── */

const idx_t *sparse_row_perm(const SparseMatrix *mat) { return mat ? mat->row_perm : NULL; }

const idx_t *sparse_col_perm(const SparseMatrix *mat) { return mat ? mat->col_perm : NULL; }

const idx_t *sparse_inv_row_perm(const SparseMatrix *mat) { return mat ? mat->inv_row_perm : NULL; }

const idx_t *sparse_inv_col_perm(const SparseMatrix *mat) { return mat ? mat->inv_col_perm : NULL; }

sparse_err_t sparse_reset_perms(SparseMatrix *mat) {
    if (!mat)
        return SPARSE_ERR_NULL;
    int drop_factor_compat =
        (mat->reorder_perm != NULL) || sparse_matrix_has_non_identity_row_col_perms(mat);
    for (idx_t i = 0; i < mat->rows; i++) {
        mat->row_perm[i] = i;
        mat->inv_row_perm[i] = i;
    }
    for (idx_t j = 0; j < mat->cols; j++) {
        mat->col_perm[j] = j;
        mat->inv_col_perm[j] = j;
    }
    sparse_factor_state_replace_reorder_perm(mat, NULL);
    if (drop_factor_compat)
        sparse_factor_state_clear(mat);
    return SPARSE_OK;
}
