#include "sparse_matrix.h"
#include "sparse_matrix_internal.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
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

static Node *make_node(SparseMatrix *mat, idx_t r, idx_t c, double v) {
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
    mat->reorder_perm = NULL;

    mat->row_headers = calloc((size_t)rows, sizeof(Node *));
    mat->col_headers = calloc((size_t)cols, sizeof(Node *));
    mat->row_perm = malloc((size_t)rows * sizeof(idx_t));
    mat->inv_row_perm = malloc((size_t)rows * sizeof(idx_t));
    mat->col_perm = malloc((size_t)cols * sizeof(idx_t));
    mat->inv_col_perm = malloc((size_t)cols * sizeof(idx_t));

    if (!mat->row_headers || !mat->col_headers || !mat->row_perm || !mat->inv_row_perm ||
        !mat->col_perm || !mat->inv_col_perm) {
        free(mat->row_headers);
        free(mat->col_headers);
        free(mat->row_perm);
        free(mat->inv_row_perm);
        free(mat->col_perm);
        free(mat->inv_col_perm);
        free(mat);
        return NULL;
    }

#ifdef SPARSE_MUTEX
    if (pthread_mutex_init(&mat->mtx, NULL) != 0) {
        free(mat->row_headers);
        free(mat->col_headers);
        free(mat->row_perm);
        free(mat->inv_row_perm);
        free(mat->col_perm);
        free(mat->inv_col_perm);
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
    free(mat->row_headers);
    free(mat->col_headers);
    free(mat->row_perm);
    free(mat->inv_row_perm);
    free(mat->col_perm);
    free(mat->inv_col_perm);
    free(mat->reorder_perm);
#ifdef SPARSE_MUTEX
    pthread_mutex_destroy(&mat->mtx);
#endif
    free(mat);
}

SparseMatrix *sparse_copy(const SparseMatrix *mat) {
    if (!mat)
        return NULL;

    SparseMatrix *copy = sparse_create(mat->rows, mat->cols);
    if (!copy)
        return NULL;

    /* Copy permutation arrays */
    memcpy(copy->row_perm, mat->row_perm, (size_t)mat->rows * sizeof(idx_t));
    memcpy(copy->inv_row_perm, mat->inv_row_perm, (size_t)mat->rows * sizeof(idx_t));
    memcpy(copy->col_perm, mat->col_perm, (size_t)mat->cols * sizeof(idx_t));
    memcpy(copy->inv_col_perm, mat->inv_col_perm, (size_t)mat->cols * sizeof(idx_t));

    /* Copy all nodes by walking each row */
    for (idx_t i = 0; i < mat->rows; i++) {
        Node *src = mat->row_headers[i];
        while (src) {
            if (sparse_insert(copy, src->row, src->col, src->value) != SPARSE_OK) {
                sparse_free(copy);
                return NULL;
            }
            src = src->right;
        }
    }

    /* Preserve cached norm and factor norm from source */
    copy->cached_norm = mat->cached_norm;
    copy->factor_norm = mat->factor_norm;

    /* Copy reorder permutation if present */
    if (mat->reorder_perm) {
        copy->reorder_perm = malloc((size_t)mat->rows * sizeof(idx_t));
        if (!copy->reorder_perm) {
            sparse_free(copy);
            return NULL;
        }
        memcpy(copy->reorder_perm, mat->reorder_perm, (size_t)mat->rows * sizeof(idx_t));
    }

    return copy;
}

SparseMatrix *sparse_transpose(const SparseMatrix *A) {
    if (!A)
        return NULL;

    SparseMatrix *T = sparse_create(A->cols, A->rows);
    if (!T)
        return NULL;

    /* For each nonzero A(i,j) = v, insert T(j,i) = v */
    for (idx_t i = 0; i < A->rows; i++) {
        Node *nd = A->row_headers[i];
        while (nd) {
            if (sparse_insert(T, nd->col, nd->row, nd->value) != SPARSE_OK) {
                sparse_free(T);
                return NULL;
            }
            nd = nd->right;
        }
    }

    return T;
}

/* ─── Element access (physical) ──────────────────────────────────────── */

static sparse_err_t sparse_remove_internal(SparseMatrix *mat, idx_t row, idx_t col);

sparse_err_t sparse_insert(SparseMatrix *mat, idx_t row, idx_t col, double val) {
    if (!mat)
        return SPARSE_ERR_NULL;
    if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return SPARSE_ERR_BOUNDS;

    SPARSE_LOCK(mat);

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
    Node *node = make_node(mat, row, col, val);
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
    sparse_err_t err = sparse_remove_internal(mat, row, col);
    SPARSE_UNLOCK(mat);
    return err;
}

double sparse_get_phys(const SparseMatrix *mat, idx_t row, idx_t col) {
    if (!mat || row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return 0.0;

    Node *curr = mat->row_headers[row];
    while (curr && curr->col < col)
        curr = curr->right;

    return (curr && curr->col == col) ? curr->value : 0.0;
}

/* ─── Element access (logical — through permutations) ────────────────── */

double sparse_get(const SparseMatrix *mat, idx_t row, idx_t col) {
    if (!mat || row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return 0.0;
    return sparse_get_phys(mat, mat->row_perm[row], mat->col_perm[col]);
}

sparse_err_t sparse_set(SparseMatrix *mat, idx_t row, idx_t col, double val) {
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
    size_t reorder_size = mat->reorder_perm ? (size_t)mat->rows * sizeof(idx_t) : 0;
    return sizeof(SparseMatrix) + (size_t)mat->rows * sizeof(Node *) /* row_headers */
           + (size_t)mat->cols * sizeof(Node *)                      /* col_headers */
           + (size_t)mat->rows * 2 * sizeof(idx_t)                   /* row perms */
           + (size_t)mat->cols * 2 * sizeof(idx_t)                   /* col perms */
           + reorder_size                                            /* reorder_perm */
           + (size_t)mat->pool.num_slabs * sizeof(NodeSlab);
}

/* ─── Symmetry check ─────────────────────────────────────────────────── */

int sparse_is_symmetric(const SparseMatrix *mat, double tol) {
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

sparse_err_t sparse_norminf(SparseMatrix *mat, double *norm) {
    if (!mat || !norm)
        return SPARSE_ERR_NULL;

    /* Return cached value if valid */
    if (mat->cached_norm >= 0.0) {
        *norm = mat->cached_norm;
        return SPARSE_OK;
    }

    double max_row_sum = 0.0;
    for (idx_t i = 0; i < mat->rows; i++) {
        double row_sum = 0.0;
        Node *node = mat->row_headers[i];
        while (node) {
            row_sum += fabs(node->value);
            node = node->right;
        }
        if (row_sum > max_row_sum)
            max_row_sum = row_sum;
    }

    mat->cached_norm = max_row_sum;
    *norm = max_row_sum;
    return SPARSE_OK;
}

/* ─── Matrix arithmetic ──────────────────────────────────────────────── */

sparse_err_t sparse_scale(SparseMatrix *mat, double alpha) {
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
    return SPARSE_OK;
}

/* NOTE: sparse_add() and sparse_add_inplace() operate in physical index space.
 * Do not use on matrices with non-identity permutations (e.g., after LU
 * factorization) — results would not correspond to logical matrix entries. */
sparse_err_t sparse_add(const SparseMatrix *A, const SparseMatrix *B, double alpha, double beta,
                        SparseMatrix **C_out) {
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
            double val;
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
            double val = alpha * nA->value;
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
            double val = beta * nB->value;
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

sparse_err_t sparse_add_inplace(SparseMatrix *A, const SparseMatrix *B, double alpha, double beta) {
    if (!A || !B)
        return SPARSE_ERR_NULL;
    if (A->rows != B->rows || A->cols != B->cols)
        return SPARSE_ERR_SHAPE;

    /* Invalidate cache early: A will be mutated even on partial failure */
    A->cached_norm = -1.0;

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
                double val = na->value + beta * nb->value;
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
                double val = beta * nb->value;
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
    double *acc = calloc((size_t)nc, sizeof(double));
    int *nz_flag = calloc((size_t)nc, sizeof(int));
    idx_t *touched = malloc((size_t)nc * sizeof(idx_t));
    if (!acc || !nz_flag || !touched) {
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
            double a_ij = a_node->value;

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

sparse_err_t sparse_matvec(const SparseMatrix *mat, const double *x, double *y) {
    if (!mat || !x || !y)
        return SPARSE_ERR_NULL;

    idx_t nrows = mat->rows;

    /* Walk each physical row, accumulate y[logical_row].
     * Each row writes to a distinct y[log_i], so rows are independent
     * and safe to parallelize without synchronization. */
#ifdef SPARSE_OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (idx_t log_i = 0; log_i < nrows; log_i++) {
        idx_t phys_i = mat->row_perm[log_i];
        Node *node = mat->row_headers[phys_i];
        double sum = 0.0;
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

sparse_err_t sparse_matvec_block(const SparseMatrix *mat, const double *X, idx_t nrhs, double *Y) {
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

    /* Precompute per-column base offsets to avoid redundant multiplies */
    if ((size_t)nrhs > SIZE_MAX / sizeof(size_t))
        return SPARSE_ERR_ALLOC;
    size_t *y_off = malloc((size_t)nrhs * sizeof(size_t));
    size_t *x_off = malloc((size_t)nrhs * sizeof(size_t));
    if (!y_off || !x_off) {
        free(y_off);
        free(x_off);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t k = 0; k < nrhs; k++) {
        y_off[k] = (size_t)m * (size_t)k;
        x_off[k] = (size_t)mat->cols * (size_t)k;
    }

    /* Walk each row once, update all nrhs columns */
    for (idx_t log_i = 0; log_i < m; log_i++) {
        idx_t phys_i = mat->row_perm[log_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            double a_ij = node->value;
            for (idx_t k = 0; k < nrhs; k++)
                Y[(size_t)log_i + y_off[k]] += a_ij * X[(size_t)log_j + x_off[k]];
            node = node->right;
        }
    }
    free(y_off);
    free(x_off);

    return SPARSE_OK;
}

/* ─── Matrix Market I/O ──────────────────────────────────────────────── */

sparse_err_t sparse_save_mm(const SparseMatrix *mat, const char *filename) {
    if (!mat || !filename)
        return SPARSE_ERR_NULL;

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        sparse_set_errno_(errno);
        return SPARSE_ERR_IO;
    }

    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%" PRId32 " %" PRId32 " %" PRId32 "\n", mat->rows, mat->cols, mat->nnz);

    for (idx_t log_i = 0; log_i < mat->rows; log_i++) {
        idx_t phys_i = mat->row_perm[log_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            fprintf(fp, "%" PRId32 " %" PRId32 " %.15g\n", log_i + 1, log_j + 1, node->value);
            node = node->right;
        }
    }

    if (fclose(fp) != 0) {
        sparse_set_errno_(errno);
        return SPARSE_ERR_IO;
    }
    sparse_set_errno_(0);
    return SPARSE_OK;
}

sparse_err_t sparse_load_mm(SparseMatrix **mat_out, const char *filename) {
    if (!mat_out || !filename)
        return SPARSE_ERR_NULL;
    *mat_out = NULL;

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        sparse_set_errno_(errno);
        return SPARSE_ERR_IO;
    }

    char line[1024];
    if (!fgets(line, (int)sizeof(line), fp)) {
        if (ferror(fp)) {
            sparse_set_errno_(errno);
            fclose(fp);
            return SPARSE_ERR_IO;
        }
        fclose(fp);
        return SPARSE_ERR_PARSE; /* empty file */
    }

    if (strstr(line, "MatrixMarket") == NULL || strstr(line, "coordinate") == NULL) {
        fclose(fp);
        return SPARSE_ERR_PARSE;
    }

    /* Detect symmetric and pattern-only formats from the header */
    int is_symmetric = (strstr(line, "symmetric") != NULL);
    int is_pattern = (strstr(line, "pattern") != NULL);

    /* Skip comment lines */
    while (fgets(line, (int)sizeof(line), fp)) {
        if (line[0] != '%')
            break;
    }

    idx_t m, n, nnz_file;
    if (sscanf(line, "%" PRId32 " %" PRId32 " %" PRId32, &m, &n, &nnz_file) != 3) {
        fclose(fp);
        return SPARSE_ERR_PARSE;
    }

    SparseMatrix *mat = sparse_create(m, n);
    if (!mat) {
        fclose(fp);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t k = 0; k < nnz_file; k++) {
        idx_t i, j;
        double v = 1.0; /* default for pattern matrices */
        if (is_pattern) {
            if (fscanf(fp, "%" PRId32 " %" PRId32, &i, &j) != 2) {
                sparse_err_t ioerr =
                    ferror(fp) ? (sparse_set_errno_(errno), SPARSE_ERR_IO) : SPARSE_ERR_PARSE;
                sparse_free(mat);
                fclose(fp);
                return ioerr;
            }
        } else {
            if (fscanf(fp, "%" PRId32 " %" PRId32 " %lf", &i, &j, &v) != 3) {
                sparse_err_t ioerr =
                    ferror(fp) ? (sparse_set_errno_(errno), SPARSE_ERR_IO) : SPARSE_ERR_PARSE;
                sparse_free(mat);
                fclose(fp);
                return ioerr;
            }
        }
        i--; /* 1-based -> 0-based */
        j--;
        if (i >= 0 && i < m && j >= 0 && j < n) {
            sparse_err_t err = sparse_insert(mat, i, j, v);
            if (err != SPARSE_OK) {
                sparse_free(mat);
                fclose(fp);
                return err;
            }
            /* For symmetric matrices, also insert the mirror entry */
            if (is_symmetric && i != j && j < m && i < n) {
                err = sparse_insert(mat, j, i, v);
                if (err != SPARSE_OK) {
                    sparse_free(mat);
                    fclose(fp);
                    return err;
                }
            }
        }
    }

    if (fclose(fp) != 0) {
        sparse_set_errno_(errno);
        sparse_free(mat);
        return SPARSE_ERR_IO;
    }
    sparse_set_errno_(0);
    *mat_out = mat;
    return SPARSE_OK;
}

/* ─── Display / debug ────────────────────────────────────────────────── */

sparse_err_t sparse_print_dense(const SparseMatrix *mat, FILE *stream) {
    if (!mat || !stream)
        return SPARSE_ERR_NULL;

    if (mat->rows > 50 || mat->cols > 50) {
        fprintf(stream,
                "[WARNING: matrix is %" PRId32 "x%" PRId32 ", dense print may be very large]\n",
                mat->rows, mat->cols);
    }

    for (idx_t i = 0; i < mat->rows; i++) {
        for (idx_t j = 0; j < mat->cols; j++) {
            fprintf(stream, "%10.4f ", sparse_get(mat, i, j));
        }
        fprintf(stream, "\n");
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
            fprintf(stream, "  (%" PRId32 ", %" PRId32 ") = %.15g\n", log_i, log_j, node->value);
            node = node->right;
        }
    }

    return SPARSE_OK;
}

sparse_err_t sparse_print_info(const SparseMatrix *mat, FILE *stream) {
    if (!mat || !stream)
        return SPARSE_ERR_NULL;

    fprintf(stream,
            "SparseMatrix: %" PRId32 " x %" PRId32 ", nnz = %" PRId32 ", memory ~ %zu bytes\n",
            mat->rows, mat->cols, mat->nnz, sparse_memory_usage(mat));

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
    for (idx_t i = 0; i < mat->rows; i++) {
        mat->row_perm[i] = i;
        mat->inv_row_perm[i] = i;
    }
    for (idx_t j = 0; j < mat->cols; j++) {
        mat->col_perm[j] = j;
        mat->inv_col_perm[j] = j;
    }
    return SPARSE_OK;
}
