#include "sparse_matrix.h"
#include "sparse_matrix_internal.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

/* ─── Pool allocator ─────────────────────────────────────────────────── */

Node *pool_alloc(NodePool *pool)
{
    /* Try the free list first */
    if (pool->free_list) {
        Node *node = pool->free_list;
        pool->free_list = node->right;
        return node;
    }

    /* Allocate from current slab or create a new one */
    if (!pool->current || pool->current->used >= NODES_PER_SLAB) {
        NodeSlab *slab = malloc(sizeof(NodeSlab));
        if (!slab) return NULL;
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

void pool_release(NodePool *pool, Node *node)
{
    /* Push onto the free list (reuse ->right as the next pointer) */
    node->right = pool->free_list;
    pool->free_list = node;
}

void pool_free_all(NodePool *pool)
{
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

static Node *make_node(SparseMatrix *mat, idx_t r, idx_t c, double v)
{
    Node *n = pool_alloc(&mat->pool);
    if (!n) return NULL;
    n->row   = r;
    n->col   = c;
    n->value = v;
    n->right = NULL;
    n->down  = NULL;
    return n;
}

/* ─── Lifecycle ──────────────────────────────────────────────────────── */

SparseMatrix *sparse_create(idx_t rows, idx_t cols)
{
    if (rows <= 0 || cols <= 0) return NULL;

    SparseMatrix *mat = malloc(sizeof(SparseMatrix));
    if (!mat) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->nnz  = 0;

    mat->row_headers  = calloc((size_t)rows, sizeof(Node *));
    mat->col_headers  = calloc((size_t)cols, sizeof(Node *));
    mat->row_perm     = malloc((size_t)rows * sizeof(idx_t));
    mat->inv_row_perm = malloc((size_t)rows * sizeof(idx_t));
    mat->col_perm     = malloc((size_t)cols * sizeof(idx_t));
    mat->inv_col_perm = malloc((size_t)cols * sizeof(idx_t));

    if (!mat->row_headers  || !mat->col_headers ||
        !mat->row_perm     || !mat->inv_row_perm ||
        !mat->col_perm     || !mat->inv_col_perm) {
        free(mat->row_headers);
        free(mat->col_headers);
        free(mat->row_perm);
        free(mat->inv_row_perm);
        free(mat->col_perm);
        free(mat->inv_col_perm);
        free(mat);
        return NULL;
    }

    for (idx_t i = 0; i < rows; i++) {
        mat->row_perm[i]     = i;
        mat->inv_row_perm[i] = i;
    }
    for (idx_t j = 0; j < cols; j++) {
        mat->col_perm[j]     = j;
        mat->inv_col_perm[j] = j;
    }

    mat->pool.head      = NULL;
    mat->pool.current   = NULL;
    mat->pool.free_list = NULL;
    mat->pool.num_slabs = 0;

    return mat;
}

void sparse_free(SparseMatrix *mat)
{
    if (!mat) return;
    pool_free_all(&mat->pool);
    free(mat->row_headers);
    free(mat->col_headers);
    free(mat->row_perm);
    free(mat->inv_row_perm);
    free(mat->col_perm);
    free(mat->inv_col_perm);
    free(mat);
}

SparseMatrix *sparse_copy(const SparseMatrix *mat)
{
    if (!mat) return NULL;

    SparseMatrix *copy = sparse_create(mat->rows, mat->cols);
    if (!copy) return NULL;

    /* Copy permutation arrays */
    memcpy(copy->row_perm,     mat->row_perm,     (size_t)mat->rows * sizeof(idx_t));
    memcpy(copy->inv_row_perm, mat->inv_row_perm, (size_t)mat->rows * sizeof(idx_t));
    memcpy(copy->col_perm,     mat->col_perm,     (size_t)mat->cols * sizeof(idx_t));
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

    return copy;
}

/* ─── Element access (physical) ──────────────────────────────────────── */

sparse_err_t sparse_insert(SparseMatrix *mat, idx_t row, idx_t col, double val)
{
    if (!mat) return SPARSE_ERR_NULL;
    if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return SPARSE_ERR_BOUNDS;

    if (val == 0.0)
        return sparse_remove(mat, row, col);

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
        return SPARSE_OK;
    }

    /* Create a new node */
    Node *node = make_node(mat, row, col, val);
    if (!node) return SPARSE_ERR_ALLOC;
    mat->nnz++;

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

    return SPARSE_OK;
}

sparse_err_t sparse_remove(SparseMatrix *mat, idx_t row, idx_t col)
{
    if (!mat) return SPARSE_ERR_NULL;
    if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return SPARSE_ERR_BOUNDS;

    /* Find and unlink from row list */
    Node *prev = NULL;
    Node *curr = mat->row_headers[row];
    while (curr && curr->col != col) {
        prev = curr;
        curr = curr->right;
    }
    if (!curr) return SPARSE_OK;  /* Not present — not an error */

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
        prev->down = ccol->down;
    else
        mat->col_headers[col] = ccol->down;

    pool_release(&mat->pool, curr);
    mat->nnz--;

    return SPARSE_OK;
}

double sparse_get_phys(const SparseMatrix *mat, idx_t row, idx_t col)
{
    if (!mat || row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return 0.0;

    Node *curr = mat->row_headers[row];
    while (curr && curr->col < col)
        curr = curr->right;

    return (curr && curr->col == col) ? curr->value : 0.0;
}

/* ─── Element access (logical — through permutations) ────────────────── */

double sparse_get(const SparseMatrix *mat, idx_t row, idx_t col)
{
    if (!mat || row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return 0.0;
    return sparse_get_phys(mat, mat->row_perm[row], mat->col_perm[col]);
}

sparse_err_t sparse_set(SparseMatrix *mat, idx_t row, idx_t col, double val)
{
    if (!mat) return SPARSE_ERR_NULL;
    if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols)
        return SPARSE_ERR_BOUNDS;
    return sparse_insert(mat, mat->row_perm[row], mat->col_perm[col], val);
}

/* ─── Matrix information ─────────────────────────────────────────────── */

idx_t sparse_rows(const SparseMatrix *mat)
{
    return mat ? mat->rows : 0;
}

idx_t sparse_cols(const SparseMatrix *mat)
{
    return mat ? mat->cols : 0;
}

idx_t sparse_nnz(const SparseMatrix *mat)
{
    return mat ? mat->nnz : 0;
}

size_t sparse_memory_usage(const SparseMatrix *mat)
{
    if (!mat) return 0;
    return sizeof(SparseMatrix)
         + (size_t)mat->rows * sizeof(Node *)       /* row_headers */
         + (size_t)mat->cols * sizeof(Node *)        /* col_headers */
         + (size_t)mat->rows * 2 * sizeof(idx_t)     /* row perms */
         + (size_t)mat->cols * 2 * sizeof(idx_t)     /* col perms */
         + (size_t)mat->pool.num_slabs * sizeof(NodeSlab);
}

/* ─── Sparse matrix-vector product ───────────────────────────────────── */

sparse_err_t sparse_matvec(const SparseMatrix *mat,
                           const double *x, double *y)
{
    if (!mat || !x || !y) return SPARSE_ERR_NULL;

    /* Zero the output */
    for (idx_t i = 0; i < mat->rows; i++)
        y[i] = 0.0;

    /* Walk each physical row, accumulate y[logical_row] */
    for (idx_t log_i = 0; log_i < mat->rows; log_i++) {
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

/* ─── Matrix Market I/O ──────────────────────────────────────────────── */

sparse_err_t sparse_save_mm(const SparseMatrix *mat, const char *filename)
{
    if (!mat || !filename) return SPARSE_ERR_NULL;

    FILE *fp = fopen(filename, "w");
    if (!fp) return SPARSE_ERR_FOPEN;

    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%" PRId32 " %" PRId32 " %" PRId32 "\n",
            mat->rows, mat->cols, mat->nnz);

    for (idx_t log_i = 0; log_i < mat->rows; log_i++) {
        idx_t phys_i = mat->row_perm[log_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            fprintf(fp, "%" PRId32 " %" PRId32 " %.15g\n",
                    log_i + 1, log_j + 1, node->value);
            node = node->right;
        }
    }

    if (fclose(fp) != 0) return SPARSE_ERR_FWRITE;
    return SPARSE_OK;
}

sparse_err_t sparse_load_mm(SparseMatrix **mat_out, const char *filename)
{
    if (!mat_out || !filename) return SPARSE_ERR_NULL;

    FILE *fp = fopen(filename, "r");
    if (!fp) return SPARSE_ERR_FOPEN;

    char line[1024];
    if (!fgets(line, (int)sizeof(line), fp)) {
        fclose(fp);
        return SPARSE_ERR_FREAD;
    }

    if (strstr(line, "MatrixMarket") == NULL ||
        strstr(line, "coordinate")  == NULL) {
        fclose(fp);
        return SPARSE_ERR_PARSE;
    }

    /* Skip comment lines */
    while (fgets(line, (int)sizeof(line), fp)) {
        if (line[0] != '%') break;
    }

    idx_t m, n, nnz_file;
    if (sscanf(line, "%" PRId32 " %" PRId32 " %" PRId32,
               &m, &n, &nnz_file) != 3) {
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
        double v;
        if (fscanf(fp, "%" PRId32 " %" PRId32 " %lf", &i, &j, &v) != 3) {
            sparse_free(mat);
            fclose(fp);
            return SPARSE_ERR_FREAD;
        }
        i--;  /* 1-based -> 0-based */
        j--;
        if (i >= 0 && i < m && j >= 0 && j < n) {
            sparse_err_t err = sparse_insert(mat, i, j, v);
            if (err != SPARSE_OK) {
                sparse_free(mat);
                fclose(fp);
                return err;
            }
        }
    }

    fclose(fp);
    *mat_out = mat;
    return SPARSE_OK;
}

/* ─── Display / debug ────────────────────────────────────────────────── */

sparse_err_t sparse_print_dense(const SparseMatrix *mat, FILE *stream)
{
    if (!mat || !stream) return SPARSE_ERR_NULL;

    if (mat->rows > 50 || mat->cols > 50) {
        fprintf(stream, "[WARNING: matrix is %" PRId32 "x%" PRId32
                ", dense print may be very large]\n", mat->rows, mat->cols);
    }

    for (idx_t i = 0; i < mat->rows; i++) {
        for (idx_t j = 0; j < mat->cols; j++) {
            fprintf(stream, "%10.4f ", sparse_get(mat, i, j));
        }
        fprintf(stream, "\n");
    }

    return SPARSE_OK;
}

sparse_err_t sparse_print_entries(const SparseMatrix *mat, FILE *stream)
{
    if (!mat || !stream) return SPARSE_ERR_NULL;

    for (idx_t log_i = 0; log_i < mat->rows; log_i++) {
        idx_t phys_i = mat->row_perm[log_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            fprintf(stream, "  (%" PRId32 ", %" PRId32 ") = %.15g\n",
                    log_i, log_j, node->value);
            node = node->right;
        }
    }

    return SPARSE_OK;
}

sparse_err_t sparse_print_info(const SparseMatrix *mat, FILE *stream)
{
    if (!mat || !stream) return SPARSE_ERR_NULL;

    fprintf(stream, "SparseMatrix: %" PRId32 " x %" PRId32
            ", nnz = %" PRId32 ", memory ~ %zu bytes\n",
            mat->rows, mat->cols, mat->nnz, sparse_memory_usage(mat));

    return SPARSE_OK;
}

/* ─── Permutation access ─────────────────────────────────────────────── */

const idx_t *sparse_row_perm(const SparseMatrix *mat)
{
    return mat ? mat->row_perm : NULL;
}

const idx_t *sparse_col_perm(const SparseMatrix *mat)
{
    return mat ? mat->col_perm : NULL;
}

const idx_t *sparse_inv_row_perm(const SparseMatrix *mat)
{
    return mat ? mat->inv_row_perm : NULL;
}

const idx_t *sparse_inv_col_perm(const SparseMatrix *mat)
{
    return mat ? mat->inv_col_perm : NULL;
}

sparse_err_t sparse_reset_perms(SparseMatrix *mat)
{
    if (!mat) return SPARSE_ERR_NULL;
    for (idx_t i = 0; i < mat->rows; i++) {
        mat->row_perm[i]     = i;
        mat->inv_row_perm[i] = i;
    }
    for (idx_t j = 0; j < mat->cols; j++) {
        mat->col_perm[j]     = j;
        mat->inv_col_perm[j] = j;
    }
    return SPARSE_OK;
}
