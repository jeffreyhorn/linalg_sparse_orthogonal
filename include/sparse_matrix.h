#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

/**
 * @file sparse_matrix.h
 * @brief Public API for the orthogonal linked-list sparse matrix.
 *
 * The SparseMatrix type stores non-zero elements in a cross-linked structure:
 * each non-zero node is linked into both a row list (sorted by column) and a
 * column list (sorted by row). This enables efficient traversal in both
 * directions, which is essential for the LU factorization algorithm.
 *
 * Memory is managed by a slab pool allocator with a free-list for node reuse.
 * Tuning constants (slab size, drop tolerance) can be overridden at compile
 * time with @c -DSPARSE_NODES_PER_SLAB=N and @c -DSPARSE_DROP_TOL=val.
 */

#include "sparse_types.h"
#include <stdio.h>

/**
 * @brief Number of Node entries per slab in the pool allocator.
 *
 * Larger values reduce malloc overhead but may waste memory for small matrices.
 * Override at compile time with @c -DSPARSE_NODES_PER_SLAB=N.
 */
#ifndef SPARSE_NODES_PER_SLAB
#define SPARSE_NODES_PER_SLAB 4096
#endif

/**
 * @brief Drop tolerance for LU factorization fill-in control.
 *
 * During factorization, entries with |value| < DROP_TOL * |pivot| are dropped
 * to zero. Override at compile time with @c -DSPARSE_DROP_TOL=val.
 */
#ifndef SPARSE_DROP_TOL
#define SPARSE_DROP_TOL 1e-14
#endif

/** @brief Opaque sparse matrix type. */
typedef struct SparseMatrix SparseMatrix;

/* ═══════════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Create an empty sparse matrix with the given dimensions.
 *
 * Allocates row/column header arrays, permutation arrays (initialized to
 * identity), and the initial pool slab. Supports rectangular matrices.
 *
 * @param rows  Number of rows (must be > 0).
 * @param cols  Number of columns (must be > 0).
 * @return A new SparseMatrix, or NULL on allocation failure or invalid dimensions.
 *
 * @note The caller owns the returned matrix and must free it with sparse_free().
 */
SparseMatrix *sparse_create(idx_t rows, idx_t cols);

/**
 * @brief Free a sparse matrix and all associated memory.
 *
 * Releases the pool allocator (all slabs), header arrays, permutation arrays,
 * and the matrix struct itself. Safe to call with NULL.
 *
 * @param mat  The matrix to free, or NULL (no-op).
 */
void sparse_free(SparseMatrix *mat);

/**
 * @brief Create a deep copy of a sparse matrix.
 *
 * Copies all non-zero elements, permutation arrays, and allocates a fresh
 * pool. The copy is independent — modifying one does not affect the other.
 *
 * @param mat  The matrix to copy (must not be NULL).
 * @return A new SparseMatrix with identical contents, or NULL on failure.
 */
SparseMatrix *sparse_copy(const SparseMatrix *mat);

/* ═══════════════════════════════════════════════════════════════════════════
 * Element access (physical indices)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Insert or update a value at a physical (row, col) position.
 *
 * If an entry already exists at (row, col), its value is overwritten.
 * Inserting 0.0 removes the entry (equivalent to sparse_remove).
 *
 * @param mat  The matrix.
 * @param row  Physical row index (0-based).
 * @param col  Physical column index (0-based).
 * @param val  The value to insert. If 0.0, the entry is removed instead.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat is NULL,
 *         SPARSE_ERR_BOUNDS if indices are out of range.
 */
sparse_err_t sparse_insert(SparseMatrix *mat, idx_t row, idx_t col, double val);

/**
 * @brief Remove the element at a physical (row, col) position.
 *
 * The removed node is returned to the pool's free-list for reuse.
 * No-op if the entry does not exist.
 *
 * @param mat  The matrix.
 * @param row  Physical row index (0-based).
 * @param col  Physical column index (0-based).
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat is NULL,
 *         SPARSE_ERR_BOUNDS if indices are out of range.
 */
sparse_err_t sparse_remove(SparseMatrix *mat, idx_t row, idx_t col);

/**
 * @brief Get the value at a physical (row, col) position.
 *
 * @param mat  The matrix (may be NULL, returns 0.0).
 * @param row  Physical row index (0-based).
 * @param col  Physical column index (0-based).
 * @return The stored value, or 0.0 if the entry is absent or indices are invalid.
 */
double sparse_get_phys(const SparseMatrix *mat, idx_t row, idx_t col);

/* ═══════════════════════════════════════════════════════════════════════════
 * Element access (logical indices — through permutation arrays)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Get the value at a logical (row, col) position.
 *
 * Translates logical indices to physical using the current row and column
 * permutation arrays, then reads the value.
 *
 * @param mat  The matrix (may be NULL, returns 0.0).
 * @param row  Logical row index (0-based).
 * @param col  Logical column index (0-based).
 * @return The stored value, or 0.0 if absent or invalid.
 */
double sparse_get(const SparseMatrix *mat, idx_t row, idx_t col);

/**
 * @brief Set a value at a logical (row, col) position.
 *
 * Translates logical indices to physical using the current permutation
 * arrays, then inserts (or removes if val is 0.0).
 *
 * @param mat  The matrix.
 * @param row  Logical row index (0-based).
 * @param col  Logical column index (0-based).
 * @param val  The value to set. If 0.0, the entry is removed.
 * @return SPARSE_OK on success, or an error code.
 */
sparse_err_t sparse_set(SparseMatrix *mat, idx_t row, idx_t col, double val);

/* ═══════════════════════════════════════════════════════════════════════════
 * Matrix information
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Return the number of rows. Returns 0 if mat is NULL. */
idx_t  sparse_rows(const SparseMatrix *mat);

/** @brief Return the number of columns. Returns 0 if mat is NULL. */
idx_t  sparse_cols(const SparseMatrix *mat);

/** @brief Return the number of stored non-zero entries. Returns 0 if mat is NULL. */
idx_t  sparse_nnz(const SparseMatrix *mat);

/**
 * @brief Estimate the memory used by the matrix (bytes).
 *
 * Includes the struct, header arrays, permutation arrays, and all pool slabs.
 * This is a lower bound — actual usage may be slightly higher due to malloc
 * overhead and alignment.
 *
 * @param mat  The matrix (returns 0 if NULL).
 * @return Estimated memory usage in bytes.
 */
size_t sparse_memory_usage(const SparseMatrix *mat);

/**
 * @brief Compute the infinity norm of the matrix: ||A||_inf = max_i sum_j |a_ij|.
 *
 * The result is cached internally and invalidated when the matrix is modified
 * (via sparse_insert, sparse_remove, or sparse_set). Repeated calls without
 * modification return the cached value in O(1).
 *
 * @param mat       The matrix (must not be NULL).
 * @param[out] norm Pointer to receive the computed norm.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat or norm is NULL.
 */
sparse_err_t sparse_norminf(const SparseMatrix *mat, double *norm);

/* ═══════════════════════════════════════════════════════════════════════════
 * Sparse matrix-vector product
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Compute y = A * x (sparse matrix-vector product).
 *
 * Traverses each physical row, accumulating into y using physical ordering.
 * The caller must allocate y (length = rows) and x (length = cols).
 * y is zeroed before accumulation.
 *
 * @param mat  The matrix.
 * @param x    Input vector of length cols.
 * @param y    Output vector of length rows (overwritten).
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_matvec(const SparseMatrix *mat,
                           const double *x, double *y);

/* ═══════════════════════════════════════════════════════════════════════════
 * Matrix arithmetic
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Scale all entries of a matrix in-place: A = alpha * A.
 *
 * Multiplies every stored non-zero by alpha. If alpha is 0.0, all entries
 * are removed and nnz becomes 0. Invalidates the cached infinity norm.
 *
 * @param mat    The matrix to scale.
 * @param alpha  The scalar multiplier.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat is NULL.
 */
sparse_err_t sparse_scale(SparseMatrix *mat, double alpha);

/**
 * @brief Compute C = alpha*A + beta*B (sparse matrix addition with scaling).
 *
 * A and B must have the same dimensions. C is a newly allocated matrix.
 * Entries that cancel to zero (|value| < 1e-15) are not stored.
 *
 * @param A       First input matrix.
 * @param B       Second input matrix.
 * @param alpha   Scalar for A.
 * @param beta    Scalar for B.
 * @param[out] C_out  Pointer to receive the result matrix. The caller must
 *                    free with sparse_free().
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if any pointer is NULL,
 *         SPARSE_ERR_SHAPE if dimensions mismatch, SPARSE_ERR_ALLOC on
 *         memory failure.
 */
sparse_err_t sparse_add(const SparseMatrix *A, const SparseMatrix *B,
                         double alpha, double beta, SparseMatrix **C_out);

/**
 * @brief Compute A = alpha*A + beta*B in-place.
 *
 * A and B must have the same dimensions. A is modified in-place.
 * Entries that cancel to zero are removed.
 *
 * @param A       Matrix to modify in-place (receives the result).
 * @param B       Second input matrix (read-only).
 * @param alpha   Scalar for A.
 * @param beta    Scalar for B.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if any pointer is NULL,
 *         SPARSE_ERR_SHAPE if dimensions mismatch.
 */
sparse_err_t sparse_add_inplace(SparseMatrix *A, const SparseMatrix *B,
                                 double alpha, double beta);

/* ═══════════════════════════════════════════════════════════════════════════
 * Matrix Market I/O
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Save a matrix to a Matrix Market file (coordinate real general).
 *
 * Writes the matrix in "%%MatrixMarket matrix coordinate real general" format
 * with full double-precision values (%.15g). Only stored non-zeros are written.
 *
 * On I/O failure, returns SPARSE_ERR_IO and captures the system errno,
 * retrievable via sparse_errno(). On success, sparse_errno() is reset to 0.
 *
 * @param mat       The matrix to save.
 * @param filename  Path to the output file.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if arguments are NULL,
 *         SPARSE_ERR_IO on file open/write failure.
 */
sparse_err_t sparse_save_mm(const SparseMatrix *mat, const char *filename);

/**
 * @brief Load a matrix from a Matrix Market file.
 *
 * Supports coordinate format with real, pattern, or integer value types,
 * and general or symmetric symmetry. Symmetric matrices have their lower
 * triangle mirrored to the upper triangle. Pattern matrices use value 1.0.
 *
 * On I/O failure, returns SPARSE_ERR_IO and captures the system errno,
 * retrievable via sparse_errno(). On success, sparse_errno() is reset to 0.
 * Format errors (bad header, dimension mismatch) return SPARSE_ERR_PARSE.
 *
 * @param[out] mat_out  Pointer to receive the loaded matrix. Set to NULL on error.
 *                      The caller must free the matrix with sparse_free().
 * @param      filename Path to the input .mtx file.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if arguments are NULL,
 *         SPARSE_ERR_IO on file open/read failure, SPARSE_ERR_PARSE on
 *         format error, SPARSE_ERR_ALLOC on memory failure.
 */
sparse_err_t sparse_load_mm(SparseMatrix **mat_out, const char *filename);

/* ═══════════════════════════════════════════════════════════════════════════
 * Display / debug
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Print the matrix in dense format to a stream.
 *
 * Prints an n-by-m grid of values (including zeros). Emits a warning if
 * either dimension exceeds 50.
 *
 * @param mat     The matrix.
 * @param stream  Output stream (e.g., stdout).
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat or stream is NULL.
 */
sparse_err_t sparse_print_dense(const SparseMatrix *mat, FILE *stream);

/**
 * @brief Print only non-zero entries as (row, col, value) triples.
 *
 * @param mat     The matrix.
 * @param stream  Output stream.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat or stream is NULL.
 */
sparse_err_t sparse_print_entries(const SparseMatrix *mat, FILE *stream);

/**
 * @brief Print summary information (dimensions, nnz, memory usage).
 *
 * @param mat     The matrix.
 * @param stream  Output stream.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat or stream is NULL.
 */
sparse_err_t sparse_print_info(const SparseMatrix *mat, FILE *stream);

/* ═══════════════════════════════════════════════════════════════════════════
 * Permutation access
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Access the row permutation array (logical → physical).
 *
 * After LU factorization, row_perm encodes the row permutation P such that
 * P[i] = row_perm[i] maps logical row i to physical row row_perm[i].
 *
 * @param mat  The matrix (returns NULL if mat is NULL).
 * @return Pointer to the internal row_perm array (length = rows). Do not free.
 */
const idx_t *sparse_row_perm(const SparseMatrix *mat);

/** @brief Access the column permutation array (logical → physical). */
const idx_t *sparse_col_perm(const SparseMatrix *mat);

/** @brief Access the inverse row permutation array (physical → logical). */
const idx_t *sparse_inv_row_perm(const SparseMatrix *mat);

/** @brief Access the inverse column permutation array (physical → logical). */
const idx_t *sparse_inv_col_perm(const SparseMatrix *mat);

/**
 * @brief Reset all permutation arrays to identity.
 *
 * Useful for reusing a matrix after factorization has permuted it.
 *
 * @param mat  The matrix.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat is NULL.
 */
sparse_err_t sparse_reset_perms(SparseMatrix *mat);

#endif /* SPARSE_MATRIX_H */
