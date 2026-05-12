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
 * @brief Maximum representable value of @c idx_t.
 *
 * Tracks the @c idx_t typedef above — pinned at @c INT32_MAX while
 * @c idx_t is @c int32_t.  The migration-to-int64_t path documented
 * above is "change the typedef and recompile"; this macro must be
 * updated alongside that change so callers that need a width-aware
 * upper bound (e.g. integer-overflow guards in size-computing
 * allocators) keep tracking the real @c idx_t range.
 */
#define IDX_MAX INT32_MAX

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
    SPARSE_ERR_NUMERIC = 14,   /**< Numerical failure (NaN or Inf produced during computation) */
    SPARSE_ERR_CANCELLED = 15, /**< Operation cancelled via opts.progress_cb (Sprint 29 Day 6).
                                    Callback returned non-zero; library freed intermediate state
                                    and aborted.  For in-place factorisations, the caller-visible
                                    matrix may be in a partially-eliminated state if cancellation
                                    happened mid-iteration; on cancellation at step=0 (before any
                                    mutation) the input matrix is bit-identical to entry.  See
                                    `sparse_progress_cb_t` below. */
} sparse_err_t;

/* ─── Progress / cancel callback (Sprint 29 Day 6, Item 4) ──────────── */

/**
 * @brief Progress event payload passed to `sparse_progress_cb_t`.
 *
 * Emitted at meaningful iteration boundaries inside long-running
 * routines (per column for scalar elimination, per supernode for
 * supernodal paths).  Fields are library-owned; the callback must
 * treat the struct as read-only and must not retain pointers past
 * the callback return.
 */
typedef struct {
    const char *phase; /**< Short phase identifier, e.g. "lu_factor",
                        *  "cholesky_factor", "ldlt_factor".  Static
                        *  string with the library's lifetime. */
    idx_t step;        /**< Monotonic 0-indexed counter within `phase`.
                        *  Increments by 1 per emission. */
    idx_t total;       /**< Expected total steps if known (e.g. matrix
                        *  dimension `n` for column-major elimination);
                        *  0 if the total isn't predictable. */
    double elapsed_s;  /**< Wall time in seconds since the routine
                        *  entered `phase` (CLOCK_MONOTONIC). */
} sparse_progress_t;

/**
 * @brief Progress / cancellation callback type.
 *
 * Long-running routines call this at iteration boundaries (per
 * column for scalar elimination, per supernode for supernodal
 * paths).  Return 0 to continue; return any non-zero value to
 * cancel the operation — the library will free any intermediate
 * state and return `SPARSE_ERR_CANCELLED` from the outer call.
 *
 * **In-place factorisations:** when the routine writes results
 * back to the input matrix (LU, Cholesky), cancellation mid-
 * iteration leaves the matrix in a partially-eliminated state.
 * Callers that need bit-identical input on cancellation should
 * return non-zero on the first callback invocation (`step == 0`),
 * which fires before any mutation has happened.
 *
 * The callback runs synchronously inside the call thread; it
 * should be fast (no I/O, no long computation).  Library guarantees
 * single-threaded invocation per outer routine call — the callback
 * does not need to be reentrant.
 *
 * @param p     Progress event payload (read-only, library-owned).
 * @param user  Opaque context pointer from `opts->progress_user`
 *              (or whichever options field carries it).
 * @return 0 to continue; non-zero to cancel.
 */
typedef int (*sparse_progress_cb_t)(const sparse_progress_t *p, void *user);

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
 * RCM, AMD, and ND compute symmetric permutations (P*A*P^T) for
 * square matrices, reducing fill-in during LU/Cholesky/LDL^T
 * factorization. COLAMD computes a column-only permutation for
 * unsymmetric/rectangular matrices, reducing fill in QR or
 * column-pivoted LU.
 *
 * - NONE: natural ordering (no reordering)
 * - RCM: Reverse Cuthill-McKee — BFS-based bandwidth reduction. Square only.
 * - AMD: Approximate Minimum Degree — symmetric fill reduction. Square only.
 * - COLAMD: Column Approximate Minimum Degree — column fill reduction for
 *   unsymmetric/QR problems. Handles rectangular matrices.
 * - ND: Nested Dissection — multilevel vertex-separator ordering.
 *   Best on 2D / 3D PDE meshes; on irregular sparsity AMD is often
 *   comparable.  Square only.
 */
typedef enum {
    SPARSE_REORDER_NONE = 0,   /**< No reordering (natural order) */
    SPARSE_REORDER_RCM = 1,    /**< Reverse Cuthill-McKee ordering */
    SPARSE_REORDER_AMD = 2,    /**< Approximate Minimum Degree ordering */
    SPARSE_REORDER_COLAMD = 3, /**< Column Approximate Minimum Degree ordering */
    SPARSE_REORDER_ND = 4,     /**< Nested Dissection (multilevel vertex separator) ordering */
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
