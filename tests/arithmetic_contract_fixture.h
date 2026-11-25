#ifndef ARITHMETIC_CONTRACT_FIXTURE_H
#define ARITHMETIC_CONTRACT_FIXTURE_H

/*
 * Focused Sprint 227 fixtures for the matrix-arithmetic state contract.
 *
 * These helpers deliberately construct permutation state by updating both
 * directions of each map.  They do not call the arithmetic implementation
 * under test when producing dense expected values.
 */

#include "sparse_matrix.h"
#include "sparse_matrix_internal.h"
#include <math.h>
#include <stdatomic.h>

#define S227_FIXTURE_MAX_DIM 6
#define S227_FIXTURE_MAX_ENTRIES (S227_FIXTURE_MAX_DIM * S227_FIXTURE_MAX_DIM)

typedef struct {
    idx_t rows;
    idx_t cols;
    idx_t nnz;
    idx_t row_perm[S227_FIXTURE_MAX_DIM];
    idx_t inv_row_perm[S227_FIXTURE_MAX_DIM];
    idx_t col_perm[S227_FIXTURE_MAX_DIM];
    idx_t inv_col_perm[S227_FIXTURE_MAX_DIM];
    double logical[S227_FIXTURE_MAX_ENTRIES];
    double cached_norm;
    double factor_norm;
    int factored;
    const void *row_headers_storage;
    const void *col_headers_storage;
    const void *row_perm_storage;
    const void *inv_row_perm_storage;
    const void *col_perm_storage;
    const void *inv_col_perm_storage;
    const void *factor_state_storage;
    const void *reorder_perm_storage;
    const void *pool_head;
    const void *pool_current;
    const void *pool_free_list;
    idx_t pool_num_slabs;
    idx_t pool_current_used;
    idx_t node_count;
    const Node *row_headers[S227_FIXTURE_MAX_DIM];
    const Node *col_headers[S227_FIXTURE_MAX_DIM];
    const Node *nodes[S227_FIXTURE_MAX_ENTRIES];
    const Node *node_right[S227_FIXTURE_MAX_ENTRIES];
    const Node *node_down[S227_FIXTURE_MAX_ENTRIES];
    idx_t node_row[S227_FIXTURE_MAX_ENTRIES];
    idx_t node_col[S227_FIXTURE_MAX_ENTRIES];
    sparse_scalar_t node_value[S227_FIXTURE_MAX_ENTRIES];
} s227_matrix_snapshot_t;

static inline SparseMatrix *s227_matrix_from_dense(idx_t rows, idx_t cols, const double *dense) {
    SparseMatrix *matrix = sparse_create(rows, cols);
    if (!matrix)
        return NULL;

    for (idx_t row = 0; row < rows; row++) {
        for (idx_t col = 0; col < cols; col++) {
            double value = dense[row * cols + col];
            if (value != 0.0 && sparse_insert(matrix, row, col, value) != SPARSE_OK) {
                sparse_free(matrix);
                return NULL;
            }
        }
    }
    return matrix;
}

static inline int s227_set_permutation(idx_t size, idx_t *forward, idx_t *inverse,
                                       const idx_t *logical_to_physical) {
    unsigned char seen[S227_FIXTURE_MAX_DIM] = {0};
    if (size < 0 || size > S227_FIXTURE_MAX_DIM || !forward || !inverse || !logical_to_physical)
        return 0;

    for (idx_t logical = 0; logical < size; logical++) {
        idx_t physical = logical_to_physical[logical];
        if (physical < 0 || physical >= size || seen[physical])
            return 0;
        seen[physical] = 1;
        forward[logical] = physical;
        inverse[physical] = logical;
    }
    return 1;
}

static inline int s227_set_row_permutation(SparseMatrix *matrix, const idx_t *logical_to_physical) {
    return matrix && s227_set_permutation(matrix->rows, matrix->row_perm, matrix->inv_row_perm,
                                          logical_to_physical);
}

static inline int s227_set_col_permutation(SparseMatrix *matrix, const idx_t *logical_to_physical) {
    return matrix && s227_set_permutation(matrix->cols, matrix->col_perm, matrix->inv_col_perm,
                                          logical_to_physical);
}

static inline void s227_copy_logical_dense(const SparseMatrix *matrix, double *dense) {
    for (idx_t row = 0; row < matrix->rows; row++)
        for (idx_t col = 0; col < matrix->cols; col++)
            dense[row * matrix->cols + col] = sparse_get(matrix, row, col);
}

static inline void s227_dense_add(idx_t rows, idx_t cols, const double *a, const double *b,
                                  double alpha, double beta, double *expected) {
    for (idx_t i = 0; i < rows * cols; i++)
        expected[i] = alpha * a[i] + beta * b[i];
}

static inline void s227_dense_matmul(idx_t rows, idx_t inner, idx_t cols, const double *a,
                                     const double *b, double *expected) {
    for (idx_t row = 0; row < rows; row++) {
        for (idx_t col = 0; col < cols; col++) {
            double sum = 0.0;
            for (idx_t k = 0; k < inner; k++)
                sum += a[row * inner + k] * b[k * cols + col];
            expected[row * cols + col] = sum;
        }
    }
}

static inline int s227_maps_are_valid(const SparseMatrix *matrix) {
    if (!matrix)
        return 0;
    for (idx_t logical = 0; logical < matrix->rows; logical++) {
        idx_t physical = matrix->row_perm[logical];
        if (physical < 0 || physical >= matrix->rows || matrix->inv_row_perm[physical] != logical)
            return 0;
    }
    for (idx_t logical = 0; logical < matrix->cols; logical++) {
        idx_t physical = matrix->col_perm[logical];
        if (physical < 0 || physical >= matrix->cols || matrix->inv_col_perm[physical] != logical)
            return 0;
    }
    return 1;
}

static inline int s227_links_are_valid(const SparseMatrix *matrix) {
    idx_t row_count = 0;
    idx_t col_count = 0;
    if (!matrix)
        return 0;

    for (idx_t row = 0; row < matrix->rows; row++) {
        idx_t previous_col = -1;
        for (const Node *node = matrix->row_headers[row]; node; node = node->right) {
            if (node->row != row || node->col <= previous_col || node->col >= matrix->cols)
                return 0;
            idx_t column_occurrences = 0;
            for (const Node *column_node = matrix->col_headers[node->col]; column_node;
                 column_node = column_node->down) {
                if (column_node == node)
                    column_occurrences++;
            }
            if (column_occurrences != 1)
                return 0;
            previous_col = node->col;
            row_count++;
        }
    }
    for (idx_t col = 0; col < matrix->cols; col++) {
        idx_t previous_row = -1;
        for (const Node *node = matrix->col_headers[col]; node; node = node->down) {
            if (node->col != col || node->row <= previous_row || node->row >= matrix->rows)
                return 0;
            idx_t row_occurrences = 0;
            for (const Node *row_node = matrix->row_headers[node->row]; row_node;
                 row_node = row_node->right) {
                if (row_node == node)
                    row_occurrences++;
            }
            if (row_occurrences != 1)
                return 0;
            previous_row = node->row;
            col_count++;
        }
    }
    return row_count == matrix->nnz && col_count == matrix->nnz;
}

static inline int s227_structure_is_valid(const SparseMatrix *matrix) {
    return s227_maps_are_valid(matrix) && s227_links_are_valid(matrix);
}

static inline int s227_take_snapshot(const SparseMatrix *matrix, s227_matrix_snapshot_t *snapshot) {
    if (!matrix || !snapshot || matrix->rows > S227_FIXTURE_MAX_DIM ||
        matrix->cols > S227_FIXTURE_MAX_DIM)
        return 0;

    snapshot->rows = matrix->rows;
    snapshot->cols = matrix->cols;
    snapshot->nnz = matrix->nnz;
    snapshot->cached_norm = atomic_load_explicit(&matrix->cached_norm, memory_order_relaxed);
    snapshot->factor_norm = matrix->factor_norm;
    snapshot->factored = matrix->factored;
    snapshot->row_headers_storage = matrix->row_headers;
    snapshot->col_headers_storage = matrix->col_headers;
    snapshot->row_perm_storage = matrix->row_perm;
    snapshot->inv_row_perm_storage = matrix->inv_row_perm;
    snapshot->col_perm_storage = matrix->col_perm;
    snapshot->inv_col_perm_storage = matrix->inv_col_perm;
    snapshot->factor_state_storage = matrix->factor_state;
    snapshot->reorder_perm_storage = matrix->reorder_perm;
    snapshot->pool_head = matrix->pool.head;
    snapshot->pool_current = matrix->pool.current;
    snapshot->pool_free_list = matrix->pool.free_list;
    snapshot->pool_num_slabs = matrix->pool.num_slabs;
    snapshot->pool_current_used = matrix->pool.current ? matrix->pool.current->used : 0;
    snapshot->node_count = 0;
    for (idx_t i = 0; i < matrix->rows; i++) {
        snapshot->row_perm[i] = matrix->row_perm[i];
        snapshot->inv_row_perm[i] = matrix->inv_row_perm[i];
        snapshot->row_headers[i] = matrix->row_headers[i];
        for (const Node *node = matrix->row_headers[i]; node; node = node->right) {
            if (snapshot->node_count >= S227_FIXTURE_MAX_ENTRIES)
                return 0;
            idx_t index = snapshot->node_count++;
            snapshot->nodes[index] = node;
            snapshot->node_right[index] = node->right;
            snapshot->node_down[index] = node->down;
            snapshot->node_row[index] = node->row;
            snapshot->node_col[index] = node->col;
            snapshot->node_value[index] = node->value;
        }
    }
    for (idx_t i = 0; i < matrix->cols; i++) {
        snapshot->col_perm[i] = matrix->col_perm[i];
        snapshot->inv_col_perm[i] = matrix->inv_col_perm[i];
        snapshot->col_headers[i] = matrix->col_headers[i];
    }
    s227_copy_logical_dense(matrix, snapshot->logical);
    return 1;
}

static inline int s227_matches_snapshot(const SparseMatrix *matrix,
                                        const s227_matrix_snapshot_t *snapshot) {
    if (!matrix || !snapshot || matrix->rows != snapshot->rows || matrix->cols != snapshot->cols ||
        matrix->nnz != snapshot->nnz ||
        atomic_load_explicit(&matrix->cached_norm, memory_order_relaxed) != snapshot->cached_norm ||
        matrix->factor_norm != snapshot->factor_norm || matrix->factored != snapshot->factored ||
        matrix->row_headers != snapshot->row_headers_storage ||
        matrix->col_headers != snapshot->col_headers_storage ||
        matrix->row_perm != snapshot->row_perm_storage ||
        matrix->inv_row_perm != snapshot->inv_row_perm_storage ||
        matrix->col_perm != snapshot->col_perm_storage ||
        matrix->inv_col_perm != snapshot->inv_col_perm_storage ||
        matrix->factor_state != snapshot->factor_state_storage ||
        matrix->reorder_perm != snapshot->reorder_perm_storage ||
        matrix->pool.head != snapshot->pool_head ||
        matrix->pool.current != snapshot->pool_current ||
        matrix->pool.free_list != snapshot->pool_free_list ||
        matrix->pool.num_slabs != snapshot->pool_num_slabs ||
        (matrix->pool.current ? matrix->pool.current->used : 0) != snapshot->pool_current_used)
        return 0;

    for (idx_t i = 0; i < matrix->rows; i++) {
        if (matrix->row_perm[i] != snapshot->row_perm[i] ||
            matrix->inv_row_perm[i] != snapshot->inv_row_perm[i] ||
            matrix->row_headers[i] != snapshot->row_headers[i])
            return 0;
    }
    for (idx_t i = 0; i < matrix->cols; i++) {
        if (matrix->col_perm[i] != snapshot->col_perm[i] ||
            matrix->inv_col_perm[i] != snapshot->inv_col_perm[i] ||
            matrix->col_headers[i] != snapshot->col_headers[i])
            return 0;
    }
    idx_t node_count = 0;
    for (idx_t row = 0; row < matrix->rows; row++) {
        for (const Node *node = matrix->row_headers[row]; node; node = node->right) {
            if (node_count >= snapshot->node_count || node != snapshot->nodes[node_count] ||
                node->right != snapshot->node_right[node_count] ||
                node->down != snapshot->node_down[node_count] ||
                node->row != snapshot->node_row[node_count] ||
                node->col != snapshot->node_col[node_count] ||
                node->value != snapshot->node_value[node_count])
                return 0;
            node_count++;
        }
    }
    if (node_count != snapshot->node_count)
        return 0;
    for (idx_t row = 0; row < matrix->rows; row++)
        for (idx_t col = 0; col < matrix->cols; col++)
            if (sparse_get(matrix, row, col) != snapshot->logical[row * matrix->cols + col])
                return 0;
    return s227_links_are_valid(matrix);
}

#endif
