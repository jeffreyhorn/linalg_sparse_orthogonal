#ifndef SPARSE_ANALYSIS_H
#define SPARSE_ANALYSIS_H

/**
 * @file sparse_analysis.h
 * @brief Symbolic analysis and numeric factorization API.
 *
 * Separates symbolic analysis from numeric factorization for LU, Cholesky,
 * and LDL^T. This enables repeated numeric refactorization on the same
 * sparsity pattern without redoing ordering and symbolic work — critical
 * for nonlinear solvers, time-stepping codes, and optimization loops.
 *
 * @note For LU, the one-time symbolic analysis can be substantially more
 * expensive than the subsequent numeric refactorizations may suggest. In
 * particular, LU symbolic analysis builds the structural pattern of A^T*A,
 * which can require O(sum_i row_nnz(i)^2) work and comparable temporary
 * memory growth in the worst case, especially when A has dense rows.
 *
 * **Workflow:**
 * @code
 *   // 1. Analyze once
 *   sparse_analysis_opts_t opts = { SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_AMD };
 *   sparse_analysis_t analysis = {0};
 *   sparse_factors_t factors = {0};
 *   sparse_err_t err = sparse_analyze(A, &opts, &analysis);
 *
 *   // 2. Factor with precomputed analysis
 *   err = sparse_factor_numeric(A, &analysis, &factors);
 *   sparse_factor_solve(&factors, &analysis, b, x);
 *
 *   // 3. Change values, refactor (no re-analysis)
 *   // ... modify A values (same sparsity pattern) ...
 *   sparse_refactor_numeric(A_new, &analysis, &factors);
 *   sparse_factor_solve(&factors, &analysis, b2, x2);
 *
 *   // 4. Clean up
 *   sparse_factor_free(&factors);
 *   sparse_analysis_free(&analysis);
 * @endcode
 */

#include "sparse_matrix.h"
#include "sparse_reorder.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Factorization type
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Factorization type for symbolic analysis.
 *
 * Determines which factorization the symbolic analysis prepares for.
 * The analysis must match the subsequent numeric factorization type.
 */
typedef enum {
    SPARSE_FACTOR_CHOLESKY = 0, /**< Cholesky (A = L*L^T, SPD matrices) */
    SPARSE_FACTOR_LU = 1,       /**< LU (P*A*Q = L*U, general matrices) */
    SPARSE_FACTOR_LDLT = 2,     /**< LDL^T (P*A*P^T = L*D*L^T, symmetric indefinite) */
} sparse_factor_type_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * Analysis options
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Options controlling symbolic analysis.
 *
 * @code
 *   sparse_analysis_opts_t opts = { SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_AMD };
 * @endcode
 */
typedef struct {
    sparse_factor_type_t factor_type; /**< Target factorization type */
    sparse_reorder_t reorder;         /**< Fill-reducing reordering (NONE/RCM/AMD) */
} sparse_analysis_opts_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * Symbolic structure (compressed-column format)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Compressed-column symbolic structure of a triangular factor.
 *
 * col_ptr[j]..col_ptr[j+1]-1 index into row_idx for the row indices
 * of nonzeros in column j. Row indices within each column are sorted
 * in ascending order.
 */
typedef struct {
    idx_t *col_ptr; /**< Column pointers (length n+1). */
    idx_t *row_idx; /**< Row indices (length nnz). */
    idx_t n;        /**< Matrix dimension. */
    idx_t nnz;      /**< Total nonzeros. */
} sparse_symbolic_pub_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * Analysis result
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Precomputed symbolic analysis for repeated numeric factorization.
 *
 * Contains the elimination tree, fill-reducing permutation, symbolic
 * structure of the factor(s), and cached matrix norm. Once computed by
 * sparse_analyze(), this object can be reused for multiple numeric
 * factorizations on matrices with the same sparsity pattern.
 *
 * Callers must call sparse_analysis_free() when done. The free function
 * is safe on a zeroed struct.
 */
typedef struct {
    idx_t n;                     /**< Matrix dimension */
    idx_t *perm;                 /**< Fill-reducing permutation (length n), or NULL */
    idx_t *etree;                /**< Elimination tree parent pointers (length n);
                                     NULL when type == SPARSE_FACTOR_LU. */
    idx_t *postorder;            /**< Etree postorder traversal (length n);
                                     NULL when type == SPARSE_FACTOR_LU. */
    sparse_symbolic_pub_t sym_L; /**< Symbolic structure of L.
                                     Cholesky: exact symbolic structure
                                     (upper bound on stored numeric factor
                                     when drop-tolerance pruning is enabled).
                                     LU: upper bound on L columns.
                                     LDL^T: upper bound (Bunch-Kaufman
                                     pivoting may produce less fill). */
    sparse_symbolic_pub_t sym_U; /**< Symbolic structure of U (zeroed for Cholesky/LDL^T).
                                     LU: upper bound on U columns. */
    sparse_factor_type_t type;   /**< Factorization type */
    double analysis_norm;        /**< Cached ||A||_inf at analysis time */
} sparse_analysis_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Compute symbolic analysis for a sparse matrix.
 *
 * Performs fill-reducing reordering (if requested), computes the
 * elimination tree, column counts, and symbolic factorization structure.
 * The result can be reused for multiple numeric factorizations on
 * matrices with the same sparsity pattern.
 *
 * @pre A must be square. For CHOLESKY and LDLT, A should be symmetric
 *      (only the lower triangle is used). For LU, A may be unsymmetric.
 *
 * @param A         The sparse matrix to analyze.
 * @param opts      Analysis options (factorization type and reordering).
 *                  If NULL, defaults to Cholesky with no reordering.
 * @param analysis  Output analysis object (caller frees with
 *                  sparse_analysis_free). Must be zeroed or freshly
 *                  initialized before first call.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or analysis is NULL.
 * @return SPARSE_ERR_BADARG if A is already factored, has non-identity
 *         row/col permutations, or an invalid reorder enum is specified.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_NOT_SPD if A is not symmetric (Cholesky/LDL^T only).
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return Error codes from the reorder routines may also be propagated.
 *
 * @note The analysis object does not hold a reference to A. The caller
 *       may modify or free A after analysis, as long as the sparsity
 *       pattern is preserved for subsequent numeric factorization.
 *
 * @note COLAMD is a column ordering, but sparse_analyze() applies it as
 *       a symmetric permutation (rows and columns) since the analysis API
 *       is built around symmetric factorizations. For column-only COLAMD
 *       application, use sparse_qr_factor_opts() with SPARSE_REORDER_COLAMD.
 *
 * @par Thread safety: Read-only on A. Safe to call concurrently on the
 *      same matrix with different output analysis objects.
 */
sparse_err_t sparse_analyze(const SparseMatrix *A, const sparse_analysis_opts_t *opts,
                            sparse_analysis_t *analysis);

/**
 * @brief Free all memory associated with a symbolic analysis.
 *
 * Releases the permutation, etree, postorder, and symbolic structure
 * arrays. Safe to call on a zeroed struct (no-op). After freeing,
 * the struct is zeroed.
 *
 * @param analysis  The analysis object to free, or NULL (no-op).
 */
void sparse_analysis_free(sparse_analysis_t *analysis);

/* ═══════════════════════════════════════════════════════════════════════════
 * Numeric factorization using precomputed analysis
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Factorization result from sparse_factor_numeric().
 *
 * Wraps the factored matrix (L for Cholesky, L+U for LU, L+D for LDL^T)
 * along with the factorization type and permutation needed for solve.
 *
 * Callers must call sparse_factor_free() when done.
 */
typedef struct {
    SparseMatrix *F;           /**< Factored matrix (L for Cholesky, L+U for LU, L for LDL^T) */
    sparse_factor_type_t type; /**< Factorization type that produced this */
    idx_t n;                   /**< Matrix dimension */
    double factor_norm;        /**< ||A||_inf at factorization time */
    /* LDL^T-specific fields (NULL/zero for Cholesky and LU) */
    double *D;         /**< D diagonal (length n), or NULL */
    double *D_offdiag; /**< D off-diagonal for 2x2 pivots (length n), or NULL */
    int *pivot_size;   /**< Pivot block sizes (length n), or NULL */
    idx_t *ldlt_perm;  /**< LDL^T pivot permutation (length n), or NULL */
} sparse_factors_t;

/**
 * @brief Perform numeric factorization using a precomputed analysis.
 *
 * Applies the fill-reducing permutation from the analysis (if any),
 * builds a permuted copy, and delegates to the appropriate one-shot
 * factorization routine (sparse_cholesky_factor, sparse_lu_factor, or
 * sparse_ldlt_factor). The symbolic structure (etree, column counts,
 * sym_L/sym_U) computed by sparse_analyze() is available for future
 * optimizations but is not currently used to bypass internal symbolic
 * work in the underlying factorization routines.
 *
 * For Cholesky: computes L such that P*A*P^T = L*L^T.
 * For LU: computes L and U such that P*A*Q = L*U (with pivoting).
 *   Uses partial pivoting with tolerance 1e-12.
 * For LDL^T: computes L and D such that P*A*P^T = L*D*L^T.
 *
 * @note The LU path currently uses fixed parameters (partial pivoting,
 *       tol=1e-12). Future versions may expose these via the analysis
 *       options.
 *
 * @pre analysis must have been computed by sparse_analyze().
 * @pre A must have identity row/col permutations and not be factored.
 *
 * @param A         The matrix to factor (not modified).
 * @param analysis  Precomputed symbolic analysis (provides permutation
 *                  and factorization type).
 * @param factors   Output factorization result. Must be zeroed or
 *                  freshly initialized. Caller frees with sparse_factor_free().
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_BADARG if A is already factored or has non-identity
 *         row/col permutations.
 * @return SPARSE_ERR_SHAPE if dimensions don't match the analysis.
 * @return SPARSE_ERR_NOT_SPD if the matrix is not symmetric
 *         (Cholesky/LDL^T) or not positive-definite (Cholesky).
 * @return SPARSE_ERR_SINGULAR if a zero pivot is encountered.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return Error codes from the delegated factorization routine may also
 *         be propagated.
 */
sparse_err_t sparse_factor_numeric(const SparseMatrix *A, const sparse_analysis_t *analysis,
                                   sparse_factors_t *factors);

/**
 * @brief Solve A*x = b using precomputed factors.
 *
 * Uses the factorization from sparse_factor_numeric() to solve a linear
 * system. The permutation from the analysis is applied automatically.
 *
 * @param factors   Factorization from sparse_factor_numeric().
 * @param analysis  The analysis used to produce the factors (provides perm).
 * @param b         Right-hand side vector of length n.
 * @param x         Solution vector of length n (overwritten). May alias b.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_BADARG if factors->F is NULL or analysis->type
 *         doesn't match factors->type.
 * @return SPARSE_ERR_SHAPE if analysis->n doesn't match factors->n.
 * @return SPARSE_ERR_SINGULAR if a near-zero diagonal is encountered.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 */
sparse_err_t sparse_factor_solve(const sparse_factors_t *factors, const sparse_analysis_t *analysis,
                                 const double *b, double *x);

/**
 * @brief Free all memory associated with a factorization result.
 *
 * Safe to call on a zeroed struct (no-op). After freeing, the struct
 * is zeroed.
 *
 * @param factors  The factors to free, or NULL (no-op).
 */
void sparse_factor_free(sparse_factors_t *factors);

/**
 * @brief Refactor using new numeric values with the analyzed structure.
 *
 * Attempts a new numeric factorization using the new matrix and the same
 * symbolic analysis. The matrix A_new must have dimensions compatible with
 * the analysis. Structural compatibility (same sparsity pattern) is a
 * caller precondition but is not validated.
 *
 * On success, the existing factors are replaced with the newly computed
 * factorization. On failure, the previous factors are left unchanged.
 *
 * This avoids the cost of repeated symbolic analysis when solving
 * multiple systems with the same structure but different values
 * (e.g., nonlinear solvers, time-stepping).
 *
 * @pre A_new must be structurally compatible with the analyzed matrix.
 * @pre factors must be zeroed or contain a valid existing factorization.
 *
 * @param A_new     The new matrix to factor (not modified). Must have
 *                  dimensions compatible with the original analysis.
 * @param analysis  Precomputed symbolic analysis (from sparse_analyze).
 * @param factors   Existing factors to overwrite on success. If
 *                  refactorization fails, the previous factorization is
 *                  preserved.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_SHAPE if matrix dimensions don't match.
 * @return SPARSE_ERR_NOT_SPD if A_new is not symmetric (Cholesky/LDL^T)
 *         or not positive-definite (Cholesky).
 * @return SPARSE_ERR_SINGULAR if a zero pivot is encountered.
 */
sparse_err_t sparse_refactor_numeric(const SparseMatrix *A_new, const sparse_analysis_t *analysis,
                                     sparse_factors_t *factors);

#endif /* SPARSE_ANALYSIS_H */
