#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "sparse_types.h"
#include <stdio.h>

/*
 * Tuning constants (override at compile time with -D if needed):
 *
 * SPARSE_NODES_PER_SLAB — number of Node entries per slab in the pool
 *                         allocator. Larger values reduce malloc overhead
 *                         but may waste memory for small matrices.
 *
 * SPARSE_DROP_TOL       — entries with |value| < DROP_TOL * pivot are
 *                         dropped to zero during LU factorization to
 *                         reduce fill-in.
 */
#ifndef SPARSE_NODES_PER_SLAB
#define SPARSE_NODES_PER_SLAB 4096
#endif

#ifndef SPARSE_DROP_TOL
#define SPARSE_DROP_TOL 1e-14
#endif

/* Opaque sparse matrix type */
typedef struct SparseMatrix SparseMatrix;

/*
 * Lifecycle
 */

/* Create a sparse matrix with given dimensions. Returns NULL on failure. */
SparseMatrix *sparse_create(idx_t rows, idx_t cols);

/* Free a sparse matrix and all associated memory. Safe to call with NULL. */
void sparse_free(SparseMatrix *mat);

/* Deep-copy a sparse matrix. Returns NULL on failure. */
SparseMatrix *sparse_copy(const SparseMatrix *mat);

/*
 * Element access (physical indices — direct row/col without permutation)
 */

/* Insert or update a value at physical (row, col). Inserting 0.0 removes. */
sparse_err_t sparse_insert(SparseMatrix *mat, idx_t row, idx_t col, double val);

/* Remove the element at physical (row, col). No-op if not present. */
sparse_err_t sparse_remove(SparseMatrix *mat, idx_t row, idx_t col);

/* Get the value at physical (row, col). Returns 0.0 for missing entries. */
double sparse_get_phys(const SparseMatrix *mat, idx_t row, idx_t col);

/*
 * Element access (logical indices — through permutation arrays)
 */

/* Get value at logical (row, col), applying current permutations. */
double sparse_get(const SparseMatrix *mat, idx_t row, idx_t col);

/* Set value at logical (row, col), applying current permutations. */
sparse_err_t sparse_set(SparseMatrix *mat, idx_t row, idx_t col, double val);

/*
 * Matrix information
 */

idx_t  sparse_rows(const SparseMatrix *mat);
idx_t  sparse_cols(const SparseMatrix *mat);
idx_t  sparse_nnz(const SparseMatrix *mat);
size_t sparse_memory_usage(const SparseMatrix *mat);

/*
 * Matrix-vector product: y = A * x (uses logical ordering)
 * Caller must allocate y (length rows) and x (length cols).
 */
sparse_err_t sparse_matvec(const SparseMatrix *mat,
                           const double *x, double *y);

/*
 * Matrix Market I/O (coordinate, real, general)
 */

/* Save matrix to a Matrix Market file. */
sparse_err_t sparse_save_mm(const SparseMatrix *mat, const char *filename);

/* Load matrix from a Matrix Market file. Caller frees *mat_out. */
sparse_err_t sparse_load_mm(SparseMatrix **mat_out, const char *filename);

/*
 * Display / debug
 */

/* Print the matrix in dense format to stream. Warns if n > 50. */
sparse_err_t sparse_print_dense(const SparseMatrix *mat, FILE *stream);

/* Print only non-zero entries as (row, col, val) triples. */
sparse_err_t sparse_print_entries(const SparseMatrix *mat, FILE *stream);

/* Print summary info (dimensions, nnz, memory). */
sparse_err_t sparse_print_info(const SparseMatrix *mat, FILE *stream);

/*
 * Permutation access (used by LU module; also available to callers)
 */

const idx_t *sparse_row_perm(const SparseMatrix *mat);
const idx_t *sparse_col_perm(const SparseMatrix *mat);
const idx_t *sparse_inv_row_perm(const SparseMatrix *mat);
const idx_t *sparse_inv_col_perm(const SparseMatrix *mat);

/* Reset permutations to identity */
sparse_err_t sparse_reset_perms(SparseMatrix *mat);

#endif /* SPARSE_MATRIX_H */
