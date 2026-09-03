#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

/**
 * @file sparse_matrix.h
 * @brief Public API for the orthogonal linked-list sparse matrix shell.
 *
 * SparseMatrix is the public mutable matrix shell. Each non-zero is linked in
 * both row and column order, so public helpers can traverse either direction.
 * Objects returned as `SparseMatrix *` are caller-owned unless a function
 * explicitly documents otherwise; release them with `sparse_free()`.
 *
 * Memory is managed by a slab pool allocator with a free-list for node reuse.
 * Tuning constants (slab size, drop tolerance) can be overridden at compile
 * time with @c -DSPARSE_NODES_PER_SLAB=N and @c -DSPARSE_DROP_TOL=val.
 *
 * The matrix shell follows the compile-time width contract in
 * `sparse_types.h`: build the library and downstream callers with the same
 * `SPARSE_IDX_BITS` setting and use @c idx_t for persisted or exchanged
 * dimensions, indices, and nnz counts.
 *
 * This type owns public sparse construction, compressed import/export, and
 * one-shot direct-workflow compatibility. Reusable symbolic/factor workspace
 * for stable sparsity patterns lives in `sparse_analysis.h`.
 *
 * Dense scalar buffers on helper paths use `sparse_scalar_t`. The current
 * scalar contract is real-only `double`; this does not imply broad numeric
 * genericity or complex support.
 *
 * Use this header for exact matrix-shell declarations, ownership contracts,
 * and matrix helper status semantics.
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

/**
 * @brief Dimension crossover for the CSC Cholesky backend.
 *
 * `sparse_cholesky_factor_opts` with `backend == SPARSE_CHOL_BACKEND_AUTO`
 * dispatches matrices with `rows >= SPARSE_CSC_THRESHOLD` to the CSC
 * working-format kernel and routes smaller matrices through the linked-list
 * scalar kernel. The default is local dispatch policy, not a portable
 * performance threshold.
 *
 * Callers with a known structure can override with
 * `-DSPARSE_CSC_THRESHOLD=N` at compile time, or set
 * `sparse_cholesky_opts_t::backend` explicitly to force one branch per call.
 * Forcing a branch requests that implementation path only and does not change
 * package, ABI, platform, or solver correctness support.
 */
#ifndef SPARSE_CSC_THRESHOLD
#define SPARSE_CSC_THRESHOLD 100
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
 * pool. The returned matrix is caller-owned and independent — modifying one
 * matrix does not affect the other.
 * Any current one-shot factor/permutation compatibility state on the source
 * matrix is copied too, so copying a factored matrix preserves its matrix-shell
 * solve contract until later matrix-shell mutation or `sparse_reset_perms()`
 * drops that compatibility. To preserve only the original coefficients, copy
 * the unfactored source matrix before entering a one-shot factorization lane.
 *
 * @param mat  The matrix to copy (must not be NULL).
 * @return A new SparseMatrix with identical contents, or NULL on failure.
 */
SparseMatrix *sparse_copy(const SparseMatrix *mat);

/**
 * @brief Compute the transpose of a sparse matrix.
 *
 * Returns a new caller-owned matrix B = A^T where B(j,i) = A(i,j) for every
 * nonzero entry in physical storage. The result has dimensions
 * (cols_A × rows_A). Works on rectangular matrices.
 *
 * @note Operates on physical storage indices. If A has non-identity
 *       row/col permutations, the transpose reflects the physical layout,
 *       not the logical view.
 *
 * @param A  The matrix to transpose (not modified and not retained). May be
 *           NULL, in which case NULL is returned.
 * @return A new SparseMatrix containing A^T, or NULL on failure or if A is NULL.
 */
SparseMatrix *sparse_transpose(const SparseMatrix *A);

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
sparse_err_t sparse_insert(SparseMatrix *mat, idx_t row, idx_t col, sparse_scalar_t val);

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
 *
 * @note **Silent-zero contract:** this function returns `0.0` in three
 *       distinct cases, which are NOT
 *       distinguished by the return value:
 *         1. `mat == NULL`.
 *         2. `row` or `col` is out of `[0, mat->rows)` / `[0,
 *            mat->cols)`.
 *         3. The (row, col) entry is absent (sparse — never
 *            inserted, or removed via `sparse_remove`).
 *
 *       This is intentional API design: the dominant caller pattern
 *       is in-bounds reads against a populated matrix, where "entry
 *       not stored" naturally means zero (the sparse-matrix
 *       convention).  Callers needing to distinguish absent-vs-OOB
 *       should pre-validate indices via `sparse_rows` / `sparse_cols`.
 */
sparse_scalar_t sparse_get_phys(const SparseMatrix *mat, idx_t row, idx_t col);

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
 *
 * @note **Silent-zero contract:** mirrors
 *       `sparse_get_phys` — returns `0.0` for NULL `mat`, out-of-
 *       range indices, and absent entries.  These three cases are
 *       NOT distinguishable from the return value.
 */
sparse_scalar_t sparse_get(const SparseMatrix *mat, idx_t row, idx_t col);

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
sparse_err_t sparse_set(SparseMatrix *mat, idx_t row, idx_t col, sparse_scalar_t val);

/* ═══════════════════════════════════════════════════════════════════════════
 * Matrix information
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Return the number of rows.
 * @param mat  The matrix (may be NULL).
 * @return Number of rows, or 0 if mat is NULL.
 *
 * @note **Silent-zero contract:** returns 0 on
 *       NULL.  This is indistinguishable from a 0-row matrix
 *       (legitimate corner case).
 */
idx_t sparse_rows(const SparseMatrix *mat);

/**
 * @brief Return the number of columns.
 * @param mat  The matrix (may be NULL).
 * @return Number of columns, or 0 if mat is NULL.
 *
 * @note **Silent-zero contract:** see `sparse_rows()` above.
 */
idx_t sparse_cols(const SparseMatrix *mat);

/**
 * @brief Return the number of stored non-zero entries.
 * @param mat  The matrix (may be NULL).
 * @return Number of stored non-zeros, or 0 if mat is NULL.
 *
 * @note **Silent-zero contract:** see `sparse_rows()` above.
 */
idx_t sparse_nnz(const SparseMatrix *mat);

/**
 * @brief Estimate the memory used by the matrix (bytes).
 *
 * Includes the struct, header arrays, permutation arrays, and all pool slabs.
 * This is a lower bound — actual usage may be slightly higher due to malloc
 * overhead and alignment. If the estimate would overflow @c size_t, the
 * function returns @c SIZE_MAX.
 *
 * @param mat  The matrix (returns 0 if NULL).
 * @return Estimated memory usage in bytes.
 */
size_t sparse_memory_usage(const SparseMatrix *mat);

/**
 * @brief Check whether a matrix is symmetric within a tolerance.
 *
 * Returns 1 if for all nonzero entries A(i,j), |A(i,j) - A(j,i)| <= tol.
 * Also checks that A is square.
 *
 * @note Operates in physical index space. Do not use on matrices with
 *       non-identity permutations (e.g., after LU factorization).
 *
 * @param mat  Input matrix (not modified).
 * @param tol  Absolute tolerance for symmetry check.
 * @return 1 if symmetric, 0 if not symmetric or mat is NULL/non-square.
 */
int sparse_is_symmetric(const SparseMatrix *mat, sparse_scalar_t tol);

/**
 * @brief Compute the infinity norm of the matrix: ||A||_inf = max_i sum_j |a_ij|.
 *
 * The result is cached internally and invalidated when the matrix is modified
 * (via sparse_insert, sparse_remove, sparse_set, sparse_scale, or
 * sparse_add_inplace). Repeated calls without modification return the cached
 * value in O(1).
 *
 * @param mat       The matrix (must not be NULL). May be mutated internally
 *                  to update the cached norm value.
 * @param[out] norm Pointer to receive the computed norm.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat or norm is NULL.
 */
sparse_err_t sparse_norminf(SparseMatrix *mat, sparse_scalar_t *norm);

/**
 * @brief Mark a matrix as factored so that solve functions accept it.
 *
 * Solve functions (sparse_lu_solve, sparse_cholesky_solve, etc.) check an
 * internal flag and return SPARSE_ERR_BADARG on unfactored matrices.  This
 * function sets that flag for matrices whose one-shot matrix-shell factors
 * were constructed externally (for example, imported from CSR) rather than
 * via the library's own factorization routines.
 *
 * This is a compatibility hook for matrix-shell solve entry points. It is not
 * the long-lived repeated-run ownership path; reusable direct factors and
 * symbolic analysis belong in `sparse_analysis.h`.
 *
 * Also computes and caches ||A||_inf so that solve-path singularity
 * detection works correctly.
 *
 * @param mat  The matrix to mark as factored. Must be square.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if mat is NULL.
 * @return SPARSE_ERR_SHAPE if mat is not square.
 */
sparse_err_t sparse_mark_factored(SparseMatrix *mat);

/* ═══════════════════════════════════════════════════════════════════════════
 * Sparse matrix-vector product
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Compute y = A * x (sparse matrix-vector product).
 *
 * Computes the product by traversing each row's entries in physical ordering.
 * The caller owns and must allocate x (length = cols) and y (length = rows).
 * The library borrows both buffers only for the duration of the call. Each
 * y[i] is fully overwritten (not accumulated into). If argument validation
 * fails, y is not a completed output and callers should treat its contents as
 * unchanged/unspecified by this call.
 *
 * @param mat  The matrix (borrowed, not modified).
 * @param x    Caller-owned input vector of length cols.
 * @param y    Caller-owned output vector of length rows (overwritten).
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_matvec(const SparseMatrix *mat, const sparse_scalar_t *x, sparse_scalar_t *y);

/**
 * @brief Sparse matrix × dense block multiply: Y = A * X.
 *
 * Computes Y(:,k) = A * X(:,k) for k = 0..nrhs-1 in a single pass
 * over the sparse structure, amortizing row traversal across all RHS.
 *
 * If @p nrhs is 0, this function is a supported no-op and returns
 * SPARSE_OK without reading from @p X or writing to @p Y.
 *
 * @param mat   Sparse matrix (m × n, not modified).
 * @param X     Caller-owned dense input matrix, n × nrhs column-major. Must be
 *              non-NULL even when @p nrhs is 0. Borrowed for the call only.
 * @param nrhs  Number of columns in X and Y. If 0, the call is a no-op.
 * @param Y     Caller-owned dense output matrix, m × nrhs column-major
 *              (overwritten on success). Must be non-NULL even when @p nrhs
 *              is 0. On error, callers should not consume @p Y as a
 *              completed block product.
 * @return SPARSE_OK on success (including the no-op case when @p nrhs is 0).
 * @return SPARSE_ERR_NULL if @p mat, @p X, or @p Y is NULL.
 * @return SPARSE_ERR_BADARG if @p nrhs is negative.
 * @return SPARSE_ERR_ALLOC if any internal size calculation overflows @c size_t
 *         (including output or input stride calculations).
 */
sparse_err_t sparse_matvec_block(const SparseMatrix *mat, const sparse_scalar_t *X, idx_t nrhs,
                                 sparse_scalar_t *Y);

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
sparse_err_t sparse_scale(SparseMatrix *mat, sparse_scalar_t alpha);

/**
 * @brief Compute C = alpha*A + beta*B (sparse matrix addition with scaling).
 *
 * A and B must have the same dimensions. After A, B, and C_out are validated,
 * *C_out is set to NULL before shape checks/allocation; on success it receives
 * a newly allocated caller-owned matrix.
 * Entries that cancel to zero (|value| < 1e-15) are not stored.
 *
 * @note Operates in physical index space. Do not use on matrices with
 *       non-identity permutations (e.g., after LU factorization).
 *
 * @param A       First input matrix (borrowed, not modified).
 * @param B       Second input matrix (borrowed, not modified).
 * @param alpha   Scalar for A.
 * @param beta    Scalar for B.
 * @param[out] C_out  Pointer to receive the result matrix. Set to NULL after
 *                    pointer validation and left NULL on later errors. The
 *                    caller must free a successful result with sparse_free().
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if any pointer is NULL,
 *         SPARSE_ERR_SHAPE if dimensions mismatch, SPARSE_ERR_ALLOC on
 *         memory failure.
 */
sparse_err_t sparse_add(const SparseMatrix *A, const SparseMatrix *B, sparse_scalar_t alpha,
                        sparse_scalar_t beta, SparseMatrix **C_out);

/**
 * @brief Compute A = alpha*A + beta*B in-place.
 *
 * A and B must have the same dimensions. A is modified in-place.
 * Entries that cancel to zero are removed.
 *
 * @note Operates in physical index space. Do not use on matrices with
 *       non-identity permutations (e.g., after LU factorization).
 *
 * @param A       Matrix to modify in-place (receives the result). May already
 *                be partially updated if an allocation or insertion error is
 *                reported after validation succeeds.
 * @param B       Second input matrix (read-only).
 * @param alpha   Scalar for A.
 * @param beta    Scalar for B.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if any pointer is NULL,
 *         SPARSE_ERR_SHAPE if dimensions mismatch.
 */
sparse_err_t sparse_add_inplace(SparseMatrix *A, const SparseMatrix *B, sparse_scalar_t alpha,
                                sparse_scalar_t beta);

/**
 * @brief Compute C = A * B (sparse matrix-matrix multiply).
 *
 * Uses Gustavson's row-wise algorithm: for each row i of A, row i of C
 * is a linear combination of rows of B, weighted by A's entries. A dense
 * accumulator is used per row and flushed to sparse output.
 *
 * A must be m×k and B must be k×n; C will be m×n.
 * Entries with |value| < 1e-15 are dropped.
 *
 * @note Operates in physical index space. Do not use on matrices with
 *       non-identity permutations (e.g., after LU factorization).
 *
 * @param A       Left input matrix (m×k).
 * @param B       Right input matrix (k×n).
 * @param[out] C  Pointer to receive the caller-owned product matrix. Set to
 *                NULL on entry and left NULL on error. Caller must free a
 *                successful result with sparse_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_SHAPE if inner dimensions mismatch (A->cols != B->rows).
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_matmul(const SparseMatrix *A, const SparseMatrix *B, SparseMatrix **C);

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
 * and general or symmetric symmetry. Symmetric off-diagonal entries are
 * mirrored in the returned matrix, and symmetric inputs must be square.
 * Pattern matrices use value 1.0. Duplicate coordinates are resolved by
 * last entry in file order, and a final value of 0.0 is omitted from stored
 * sparse entries.
 *
 * On I/O failure, returns SPARSE_ERR_IO and captures the system errno,
 * retrievable via sparse_errno(). On success, sparse_errno() is reset to 0.
 * Format errors (bad header, malformed dimensions/data, unsupported format,
 * out-of-range coordinates, zero coordinates, or rectangular symmetric
 * input) return SPARSE_ERR_PARSE.
 *
 * @param[out] mat_out  Pointer to receive the loaded caller-owned matrix. Set
 *                      to NULL after argument validation and left NULL on later
 *                      errors. The caller must free a successful matrix with
 *                      sparse_free().
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
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat or stream is NULL,
 *         SPARSE_ERR_IO on stream write failure.
 */
sparse_err_t sparse_print_dense(const SparseMatrix *mat, FILE *stream);

/**
 * @brief Print only non-zero entries as (row, col, value) triples.
 *
 * @param mat     The matrix.
 * @param stream  Output stream.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat or stream is NULL,
 *         SPARSE_ERR_IO on stream write failure.
 */
sparse_err_t sparse_print_entries(const SparseMatrix *mat, FILE *stream);

/**
 * @brief Print summary information (dimensions, nnz, memory usage).
 *
 * @param mat     The matrix.
 * @param stream  Output stream.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat or stream is NULL,
 *         SPARSE_ERR_IO on stream write failure.
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

/**
 * @brief Access the column permutation array (logical -> physical).
 * @param mat  The matrix (returns NULL if mat is NULL).
 * @return Pointer to the internal col_perm array (length = cols). Do not free.
 */
const idx_t *sparse_col_perm(const SparseMatrix *mat);

/**
 * @brief Access the inverse row permutation array (physical -> logical).
 * @param mat  The matrix (returns NULL if mat is NULL).
 * @return Pointer to the internal inv_row_perm array (length = rows). Do not free.
 */
const idx_t *sparse_inv_row_perm(const SparseMatrix *mat);

/**
 * @brief Access the inverse column permutation array (physical -> logical).
 * @param mat  The matrix (returns NULL if mat is NULL).
 * @return Pointer to the internal inv_col_perm array (length = cols). Do not free.
 */
const idx_t *sparse_inv_col_perm(const SparseMatrix *mat);

/**
 * @brief Reset all permutation arrays to identity.
 *
 * Useful for recovering a plain matrix shell after a one-shot factorization or
 * reorder has permuted it.
 *
 * If the matrix currently carries one-shot reordered/factored compatibility
 * state, resetting the permutation shell drops that compatibility so later
 * solve calls reject the matrix until it is factorized again. This keeps
 * `SparseMatrix` focused on matrix-shell ownership instead of preserving a
 * stale solve handle after the caller has rewritten its coordinate mapping.
 *
 * @param mat  The matrix.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if mat is NULL.
 */
sparse_err_t sparse_reset_perms(SparseMatrix *mat);

#endif /* SPARSE_MATRIX_H */
