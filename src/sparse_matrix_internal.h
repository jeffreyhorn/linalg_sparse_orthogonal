#ifndef SPARSE_MATRIX_INTERNAL_H
#define SPARSE_MATRIX_INTERNAL_H

/*
 * Private header: internal struct definitions for sparse_matrix.c and
 * sparse_lu.c. NOT part of the public API.
 */

#include "sparse_matrix.h"  /* picks up SPARSE_NODES_PER_SLAB, SPARSE_DROP_TOL */

/*
 * Optional mutex support: compile with -DSPARSE_MUTEX to enable.
 * When enabled, sparse_insert/sparse_remove/sparse_lu_factor lock
 * the matrix mutex to allow safe concurrent mutation of the same matrix.
 * Default: disabled (zero overhead).
 */
#ifdef SPARSE_MUTEX
#include <pthread.h>
#define SPARSE_LOCK(mat)   pthread_mutex_lock(&(mat)->mtx)
#define SPARSE_UNLOCK(mat) pthread_mutex_unlock(&(mat)->mtx)
#else
#define SPARSE_LOCK(mat)   ((void)0)
#define SPARSE_UNLOCK(mat) ((void)0)
#endif

#define NODES_PER_SLAB SPARSE_NODES_PER_SLAB
#define DROP_TOL       SPARSE_DROP_TOL

typedef struct Node {
    idx_t row;
    idx_t col;
    double value;
    struct Node *right;  /* next in same row (sorted by col)  */
    struct Node *down;   /* next in same col (sorted by row)  */
} Node;

typedef struct NodeSlab {
    Node nodes[NODES_PER_SLAB];
    struct NodeSlab *next;
    idx_t used;
} NodeSlab;

typedef struct NodePool {
    NodeSlab *head;
    NodeSlab *current;
    Node     *free_list;   /* singly-linked free list via ->right */
    idx_t     num_slabs;
} NodePool;

typedef struct SparseMatrix {
    idx_t rows;
    idx_t cols;
    Node **row_headers;
    Node **col_headers;
    idx_t *row_perm;       /* logical -> physical row */
    idx_t *inv_row_perm;   /* physical -> logical row */
    idx_t *col_perm;       /* logical -> physical col */
    idx_t *inv_col_perm;   /* physical -> logical col */
    NodePool pool;
    idx_t nnz;
    double cached_norm;    /* cached ||A||_inf, -1.0 = invalid */
    double factor_norm;    /* ||A||_inf at factorization time, for relative tol */
    idx_t *reorder_perm;   /* fill-reducing reorder: perm[new] = old, or NULL */
#ifdef SPARSE_MUTEX
    pthread_mutex_t mtx;   /* optional mutex for concurrent mutation */
#endif
} SparseMatrix;

/*
 * Internal errno capture (defined in sparse_types.c)
 */
void sparse_set_errno_(int errnum);

/*
 * Internal pool operations (used by sparse_matrix.c and sparse_lu.c)
 */
Node *pool_alloc(NodePool *pool);
void  pool_release(NodePool *pool, Node *node);
void  pool_free_all(NodePool *pool);

/*
 * Build CSR adjacency graph of A+A^T (symmetrized, no self-loops).
 * adj_ptr[i]..adj_ptr[i+1]-1 index into adj_list for neighbors of i.
 * Caller must free both adj_ptr and adj_list.
 */
sparse_err_t sparse_build_adj(const SparseMatrix *A,
                              idx_t **adj_ptr, idx_t **adj_list);

#endif /* SPARSE_MATRIX_INTERNAL_H */
