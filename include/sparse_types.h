#ifndef SPARSE_TYPES_H
#define SPARSE_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* Index type for matrix dimensions and indices */
typedef int32_t idx_t;

/* Error codes returned by library functions */
typedef enum {
    SPARSE_OK           = 0,
    SPARSE_ERR_NULL     = 1,   /* NULL pointer argument */
    SPARSE_ERR_ALLOC    = 2,   /* Memory allocation failure */
    SPARSE_ERR_BOUNDS   = 3,   /* Index out of bounds */
    SPARSE_ERR_SINGULAR = 4,   /* Matrix is singular or nearly singular */
    SPARSE_ERR_FOPEN    = 5,   /* File open failure */
    SPARSE_ERR_FREAD    = 6,   /* File read failure */
    SPARSE_ERR_FWRITE   = 7,   /* File write failure */
    SPARSE_ERR_PARSE    = 8,   /* File format parse error */
    SPARSE_ERR_SHAPE    = 9,   /* Matrix shape mismatch (e.g., non-square for LU) */
} sparse_err_t;

/* Pivoting strategy for LU factorization */
typedef enum {
    SPARSE_PIVOT_COMPLETE = 0,   /* Complete pivoting (max over entire submatrix) */
    SPARSE_PIVOT_PARTIAL  = 1,   /* Partial pivoting (max in pivot column only) */
} sparse_pivot_t;

/* Return a human-readable string for an error code */
const char *sparse_strerror(sparse_err_t err);

#endif /* SPARSE_TYPES_H */
