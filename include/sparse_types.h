#ifndef SPARSE_TYPES_H
#define SPARSE_TYPES_H

/**
 * @file sparse_types.h
 * @brief Core type definitions and error codes for the sparse matrix library.
 *
 * This header defines the index type, error code enumeration, and pivoting
 * strategy enumeration used throughout the library.
 */

#include <stdint.h>
#include <stddef.h>

/** @brief Signed 32-bit index type for matrix dimensions and indices. */
typedef int32_t idx_t;

/**
 * @brief Error codes returned by library functions.
 *
 * All functions that can fail return a @c sparse_err_t value.
 * SPARSE_OK (0) indicates success; all other values indicate an error.
 * Use sparse_strerror() to obtain a human-readable description.
 */
typedef enum {
    SPARSE_OK           = 0,   /**< Success */
    SPARSE_ERR_NULL     = 1,   /**< NULL pointer argument */
    SPARSE_ERR_ALLOC    = 2,   /**< Memory allocation failure */
    SPARSE_ERR_BOUNDS   = 3,   /**< Index out of bounds */
    SPARSE_ERR_SINGULAR = 4,   /**< Matrix is singular or nearly singular */
    SPARSE_ERR_FOPEN    = 5,   /**< File open failure */
    SPARSE_ERR_FREAD    = 6,   /**< File read/parse failure */
    SPARSE_ERR_FWRITE   = 7,   /**< File write failure */
    SPARSE_ERR_PARSE    = 8,   /**< File format parse error */
    SPARSE_ERR_SHAPE    = 9,   /**< Matrix shape mismatch (e.g., non-square for LU) */
    SPARSE_ERR_IO       = 10,  /**< I/O error with errno context (use sparse_errno()) */
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
    SPARSE_PIVOT_COMPLETE = 0,   /**< Complete pivoting (max over submatrix) */
    SPARSE_PIVOT_PARTIAL  = 1,   /**< Partial pivoting (max in pivot column) */
} sparse_pivot_t;

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
