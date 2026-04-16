#ifndef SPARSE_TYPES_H
#define SPARSE_TYPES_H

/**
 * @file sparse_types.h
 * @brief Core type definitions and error codes for the sparse matrix library.
 *
 * This header defines the index type, error code enumeration, and pivoting
 * strategy enumeration used throughout the library.
 */

#include <stddef.h>
#include <stdint.h>

/* ── Library version (generated from VERSION file) ───────────────────── */
#include "sparse_version.h"

/**
 * @brief Signed 32-bit index type for matrix dimensions and indices.
 *
 * All matrix dimensions, row/column indices, and nonzero counts use this
 * type.  The 32-bit limit caps matrix dimensions at ~2.1 billion and
 * nonzero counts at INT32_MAX (~2.1 billion entries).
 *
 * **Rationale:** 32-bit indices halve memory for index arrays compared to
 * 64-bit, improving cache efficiency for typical sparse matrix sizes.
 * Most sparse matrices in practice have well under 2 billion nonzeros.
 *
 * **Migration path:** To support larger matrices, change this typedef to
 * @c int64_t and recompile.  The library uses @c idx_t consistently, so
 * no other source changes are needed — but all downstream code that
 * stores or passes index values must also use @c idx_t (not bare @c int).
 */
typedef int32_t idx_t;

/**
 * @brief Error codes returned by library functions.
 *
 * All functions that can fail return a @c sparse_err_t value.
 * SPARSE_OK (0) indicates success; all other values indicate an error.
 * Use sparse_strerror() to obtain a human-readable description.
 */
typedef enum {
    SPARSE_OK = 0,           /**< Success */
    SPARSE_ERR_NULL = 1,     /**< NULL pointer argument */
    SPARSE_ERR_ALLOC = 2,    /**< Memory allocation failure */
    SPARSE_ERR_BOUNDS = 3,   /**< Index out of bounds */
    SPARSE_ERR_SINGULAR = 4, /**< Matrix is singular or nearly singular */
    SPARSE_ERR_FOPEN = 5,    /**< File open failure */
    SPARSE_ERR_FREAD = 6,    /**< File read/parse failure */
    SPARSE_ERR_FWRITE = 7,   /**< File write failure */
    SPARSE_ERR_PARSE = 8,    /**< File format parse error */
    SPARSE_ERR_SHAPE = 9,    /**< Matrix shape mismatch (e.g., non-square for LU) */
    SPARSE_ERR_IO = 10,      /**< I/O error with errno context (use sparse_errno()) */
    SPARSE_ERR_BADARG = 11,  /**< Invalid argument (e.g., unfactored matrix passed to condest) */
    SPARSE_ERR_NOT_SPD = 12, /**< Matrix is not symmetric positive-definite */
    SPARSE_ERR_NOT_CONVERGED = 13, /**< Iterative solver did not converge within max iterations */
    SPARSE_ERR_NUMERIC = 14, /**< Numerical failure (NaN or Inf produced during computation) */
} sparse_err_t;

/**
 * @brief Pivoting strategy for LU factorization.
 *
 * Complete pivoting searches the entire remaining submatrix for the largest
 * element, providing better numerical stability at the cost of O(n^2) search
 * per step. Partial pivoting searches only the pivot column (O(n) per step)
 * and is strongly preferred for banded or structured matrices.
 */
typedef enum {
    SPARSE_PIVOT_COMPLETE = 0, /**< Complete pivoting (max over submatrix) */
    SPARSE_PIVOT_PARTIAL = 1,  /**< Partial pivoting (max in pivot column) */
} sparse_pivot_t;

/**
 * @brief Reordering strategy for fill-reducing permutation.
 *
 * RCM and AMD compute symmetric permutations (P*A*P^T) for square matrices,
 * reducing fill-in during LU/Cholesky/LDL^T factorization. COLAMD computes
 * a column-only permutation for unsymmetric/rectangular matrices, reducing
 * fill in QR or column-pivoted LU.
 *
 * - NONE: natural ordering (no reordering)
 * - RCM: Reverse Cuthill-McKee — BFS-based bandwidth reduction. Square only.
 * - AMD: Approximate Minimum Degree — symmetric fill reduction. Square only.
 * - COLAMD: Column Approximate Minimum Degree — column fill reduction for
 *   unsymmetric/QR problems. Handles rectangular matrices.
 */
typedef enum {
    SPARSE_REORDER_NONE = 0,   /**< No reordering (natural order) */
    SPARSE_REORDER_RCM = 1,    /**< Reverse Cuthill-McKee ordering */
    SPARSE_REORDER_AMD = 2,    /**< Approximate Minimum Degree ordering */
    SPARSE_REORDER_COLAMD = 3, /**< Column Approximate Minimum Degree ordering */
} sparse_reorder_t;

/**
 * @brief Return a human-readable string for an error code.
 *
 * @param err  The error code to describe.
 * @return A static string such as "OK", "NULL pointer", etc.
 *         Returns "Unknown error" for unrecognized codes.
 */
const char *sparse_strerror(sparse_err_t err);

/**
 * @brief Return the errno captured by the last I/O operation that failed.
 *
 * When a library function returns SPARSE_ERR_IO, this function returns the
 * system errno that was active at the point of failure. Returns 0 if no
 * errno has been captured or after a successful I/O operation.
 *
 * @return The captured errno value.
 */
int sparse_errno(void);

#endif /* SPARSE_TYPES_H */
