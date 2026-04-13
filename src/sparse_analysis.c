#include "sparse_analysis.h"
#include "sparse_analysis_internal.h"
#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_lu.h"
#include "sparse_reorder.h"

#include <math.h>
#include <string.h>

/* Check that the matrix has identity row/col permutations and is not factored.
 * Analysis and factorization operate on physical index space, so non-identity
 * perms or factored state would produce incorrect results. */
static int has_identity_perms(const SparseMatrix *A) {
    idx_t n = A->rows;
    for (idx_t i = 0; i < n; i++) {
        if (A->row_perm[i] != i || A->col_perm[i] != i)
            return 0;
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_analyze — compute symbolic analysis
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_analyze(const SparseMatrix *A, const sparse_analysis_opts_t *opts,
                            sparse_analysis_t *analysis) {
    if (!A || !analysis)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;
    if (A->factored || !has_identity_perms(A))
        return SPARSE_ERR_BADARG;

    /* Default options: Cholesky, no reordering */
    sparse_factor_type_t ftype = SPARSE_FACTOR_CHOLESKY;
    sparse_reorder_t reorder = SPARSE_REORDER_NONE;
    if (opts) {
        ftype = opts->factor_type;
        reorder = opts->reorder;
    }

    idx_t n = A->rows;
    sparse_analysis_free(analysis); /* free any prior contents */
    analysis->n = n;
    analysis->type = ftype;

    /* Cache ||A||_inf */
    analysis->analysis_norm = sparse_norminf_const(A);

    /* Compute fill-reducing permutation if requested */
    sparse_err_t err = SPARSE_OK;
    if (reorder != SPARSE_REORDER_NONE && n > 0) {
        if ((size_t)n > SIZE_MAX / sizeof(idx_t)) {
            sparse_analysis_free(analysis);
            return SPARSE_ERR_ALLOC;
        }
        analysis->perm = malloc((size_t)n * sizeof(idx_t));
        if (!analysis->perm) {
            sparse_analysis_free(analysis);
            return SPARSE_ERR_ALLOC;
        }
        if (reorder == SPARSE_REORDER_RCM) {
            err = sparse_reorder_rcm(A, analysis->perm);
        } else if (reorder == SPARSE_REORDER_AMD) {
            err = sparse_reorder_amd(A, analysis->perm);
        } else {
            sparse_analysis_free(analysis);
            return SPARSE_ERR_BADARG;
        }
        if (err != SPARSE_OK) {
            sparse_analysis_free(analysis);
            return err;
        }
    }

    /* Dispatch by factorization type */
    switch (ftype) {
    case SPARSE_FACTOR_CHOLESKY:
    case SPARSE_FACTOR_LDLT: {
        /* Validate symmetry early to avoid producing a meaningless etree */
        if (!sparse_is_symmetric(A, 1e-12)) {
            sparse_analysis_free(analysis);
            return SPARSE_ERR_NOT_SPD;
        }

        /* Symmetric path: etree + postorder + colcount + symbolic Cholesky.
         * If a permutation was computed, build a symmetrically permuted
         * copy using sparse_permute (perm[new] = old convention). */
        const SparseMatrix *B = A;
        SparseMatrix *B_perm = NULL;

        if (analysis->perm) {
            err = sparse_permute(A, analysis->perm, analysis->perm, &B_perm);
            if (err) {
                sparse_analysis_free(analysis);
                return err;
            }
            B = B_perm;
        }

        /* Allocate etree and postorder (overflow already checked above for perm) */
        if ((size_t)n > SIZE_MAX / sizeof(idx_t)) {
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return SPARSE_ERR_ALLOC;
        }
        analysis->etree = malloc((size_t)n * sizeof(idx_t));
        analysis->postorder = malloc((size_t)n * sizeof(idx_t));
        if (!analysis->etree || !analysis->postorder) {
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return SPARSE_ERR_ALLOC;
        }

        err = sparse_etree_compute(B, analysis->etree);
        if (err) {
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return err;
        }

        err = sparse_etree_postorder(analysis->etree, n, analysis->postorder);
        if (err) {
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return err;
        }

        idx_t *cc = malloc((size_t)n * sizeof(idx_t));
        if (!cc) {
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return SPARSE_ERR_ALLOC;
        }

        err = sparse_colcount(B, analysis->etree, analysis->postorder, cc);
        if (err) {
            free(cc);
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return err;
        }

        /* Compute symbolic structure */
        sparse_symbolic_t sym_internal;
        err = sparse_symbolic_cholesky(B, analysis->etree, analysis->postorder, cc, &sym_internal);
        free(cc);
        sparse_free(B_perm);

        if (err) {
            sparse_analysis_free(analysis);
            return err;
        }

        /* Copy internal symbolic to public struct (same layout) */
        analysis->sym_L.col_ptr = sym_internal.col_ptr;
        analysis->sym_L.row_idx = sym_internal.row_idx;
        analysis->sym_L.n = sym_internal.n;
        analysis->sym_L.nnz = sym_internal.nnz;
        break;
    }

    case SPARSE_FACTOR_LU: {
        /* Unsymmetric path: use sparse_symbolic_lu which computes
         * column etree of A^T*A and produces L and U bounds. */
        sparse_symbolic_t sym_L_int, sym_U_int;
        err = sparse_symbolic_lu(A, analysis->perm, &sym_L_int, &sym_U_int);
        if (err) {
            sparse_analysis_free(analysis);
            return err;
        }

        analysis->sym_L.col_ptr = sym_L_int.col_ptr;
        analysis->sym_L.row_idx = sym_L_int.row_idx;
        analysis->sym_L.n = sym_L_int.n;
        analysis->sym_L.nnz = sym_L_int.nnz;

        analysis->sym_U.col_ptr = sym_U_int.col_ptr;
        analysis->sym_U.row_idx = sym_U_int.row_idx;
        analysis->sym_U.n = sym_U_int.n;
        analysis->sym_U.nnz = sym_U_int.nnz;

        /* The etree/postorder are computed internally by sparse_symbolic_lu.
         * We don't expose them for the LU path since they're of the
         * symmetrized pattern, not A itself. */
        break;
    }

    default:
        sparse_analysis_free(analysis);
        return SPARSE_ERR_BADARG;
    }

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_analysis_free
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_analysis_free(sparse_analysis_t *analysis) {
    if (!analysis)
        return;
    free(analysis->perm);
    free(analysis->etree);
    free(analysis->postorder);
    free(analysis->sym_L.col_ptr);
    free(analysis->sym_L.row_idx);
    free(analysis->sym_U.col_ptr);
    free(analysis->sym_U.row_idx);
    memset(analysis, 0, sizeof(*analysis));
}

/* ═══════════════════════════════════════════════════════════════════════
 * Helper: build symmetrically permuted copy of A
 * ═══════════════════════════════════════════════════════════════════════ */

/* Reset permutation state on a working copy so the underlying solvers
 * don't apply stale reorder/pivot permutations from the original matrix. */
static SparseMatrix *sanitize_working_copy(SparseMatrix *B) {
    if (!B)
        return NULL;
    sparse_reset_perms(B);
    free(B->reorder_perm);
    B->reorder_perm = NULL;
    return B;
}

static sparse_err_t build_permuted_copy(const SparseMatrix *A, const idx_t *perm,
                                        SparseMatrix **out) {
    SparseMatrix *B = NULL;
    sparse_err_t err;
    if (perm) {
        err = sparse_permute(A, perm, perm, &B);
        if (err != SPARSE_OK)
            return err;
    } else {
        B = sparse_copy(A);
        if (!B)
            return SPARSE_ERR_ALLOC;
    }
    *out = sanitize_working_copy(B);
    return *out ? SPARSE_OK : SPARSE_ERR_ALLOC;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_factor_numeric
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_factor_numeric(const SparseMatrix *A, const sparse_analysis_t *analysis,
                                   sparse_factors_t *factors) {
    if (!A || !analysis || !factors)
        return SPARSE_ERR_NULL;
    if (A->rows != analysis->n || A->cols != analysis->n)
        return SPARSE_ERR_SHAPE;
    if (A->factored || !has_identity_perms(A))
        return SPARSE_ERR_BADARG;

    idx_t n = analysis->n;
    sparse_factor_free(factors); /* free any prior contents */
    factors->type = analysis->type;
    factors->n = n;

    switch (analysis->type) {
    case SPARSE_FACTOR_CHOLESKY: {
        /* Build (optionally permuted) copy and factor with existing Cholesky */
        SparseMatrix *L = NULL;
        sparse_err_t err = build_permuted_copy(A, analysis->perm, &L);
        if (err != SPARSE_OK)
            return err;

        err = sparse_cholesky_factor(L);
        if (err != SPARSE_OK) {
            sparse_free(L);
            return err;
        }

        factors->F = L;
        factors->factor_norm = L->factor_norm;
        break;
    }

    case SPARSE_FACTOR_LU: {
        /* Build (optionally permuted) copy and factor with existing LU */
        SparseMatrix *LU = NULL;
        sparse_err_t err = build_permuted_copy(A, analysis->perm, &LU);
        if (err != SPARSE_OK)
            return err;

        err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
        if (err != SPARSE_OK) {
            sparse_free(LU);
            return err;
        }

        factors->F = LU;
        factors->factor_norm = LU->factor_norm;
        break;
    }

    case SPARSE_FACTOR_LDLT: {
        /* Build (optionally permuted) copy and factor with existing LDL^T */
        SparseMatrix *B = NULL;
        sparse_err_t err = build_permuted_copy(A, analysis->perm, &B);
        if (err != SPARSE_OK)
            return err;

        sparse_ldlt_t ldlt;
        err = sparse_ldlt_factor(B, &ldlt);
        sparse_free(B);
        if (err != SPARSE_OK)
            return err;

        /* Transfer ownership from ldlt to factors */
        factors->F = ldlt.L;
        factors->factor_norm = ldlt.factor_norm;
        factors->D = ldlt.D;
        factors->D_offdiag = ldlt.D_offdiag;
        factors->pivot_size = ldlt.pivot_size;

        /* Store the LDL^T pivot permutation separately; solve applies the
         * analysis permutation first and then this LDL^T permutation. */
        factors->ldlt_perm = ldlt.perm;

        /* Null out ldlt pointers so sparse_ldlt_free doesn't double-free */
        ldlt.L = NULL;
        ldlt.D = NULL;
        ldlt.D_offdiag = NULL;
        ldlt.pivot_size = NULL;
        ldlt.perm = NULL;
        sparse_ldlt_free(&ldlt);
        break;
    }

    default:
        return SPARSE_ERR_BADARG;
    }

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_factor_solve
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_factor_solve(const sparse_factors_t *factors, const sparse_analysis_t *analysis,
                                 const double *b, double *x) {
    if (!factors || !analysis || !b || !x)
        return SPARSE_ERR_NULL;
    if (!factors->F)
        return SPARSE_ERR_BADARG;
    if (analysis->n != factors->n)
        return SPARSE_ERR_SHAPE;
    if (analysis->type != factors->type)
        return SPARSE_ERR_BADARG;

    idx_t n = factors->n;
    const idx_t *perm = analysis->perm;

    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

    /* Permute b if a fill-reducing permutation was used.
     * perm[new] = old convention: b_perm[new_i] = b[perm[new_i]] */
    double *b_perm = NULL;
    const double *b_eff = b;
    if (perm) {
        b_perm = malloc((size_t)n * sizeof(double));
        if (!b_perm)
            return SPARSE_ERR_ALLOC;
        for (idx_t i = 0; i < n; i++)
            b_perm[i] = b[perm[i]];
        b_eff = b_perm;
    }

    sparse_err_t err;
    double *x_tmp = malloc((size_t)n * sizeof(double));
    if (!x_tmp) {
        free(b_perm);
        return SPARSE_ERR_ALLOC;
    }

    switch (factors->type) {
    case SPARSE_FACTOR_CHOLESKY:
        err = sparse_cholesky_solve(factors->F, b_eff, x_tmp);
        break;
    case SPARSE_FACTOR_LU:
        err = sparse_lu_solve(factors->F, b_eff, x_tmp);
        break;
    case SPARSE_FACTOR_LDLT: {
        /* Reconstruct a temporary sparse_ldlt_t for the solve call */
        sparse_ldlt_t ldlt_tmp;
        ldlt_tmp.L = factors->F;
        ldlt_tmp.D = factors->D;
        ldlt_tmp.D_offdiag = factors->D_offdiag;
        ldlt_tmp.pivot_size = factors->pivot_size;
        ldlt_tmp.perm = factors->ldlt_perm;
        ldlt_tmp.n = factors->n;
        ldlt_tmp.factor_norm = factors->factor_norm;
        ldlt_tmp.tol = SPARSE_DROP_TOL;
        err = sparse_ldlt_solve(&ldlt_tmp, b_eff, x_tmp);
        break;
    }
    default:
        err = SPARSE_ERR_BADARG;
        break;
    }

    if (err != SPARSE_OK) {
        free(b_perm);
        free(x_tmp);
        return err;
    }

    /* Unpermute the solution: perm[new] = old, so x[old] = x_tmp[new] */
    if (perm) {
        for (idx_t i = 0; i < n; i++)
            x[perm[i]] = x_tmp[i];
    } else {
        memcpy(x, x_tmp, (size_t)n * sizeof(double));
    }

    free(b_perm);
    free(x_tmp);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_factor_free
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_factor_free(sparse_factors_t *factors) {
    if (!factors)
        return;
    sparse_free(factors->F);
    free(factors->D);
    free(factors->D_offdiag);
    free(factors->pivot_size);
    free(factors->ldlt_perm);
    memset(factors, 0, sizeof(*factors));
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_refactor_numeric
 *
 * Convenience wrapper around sparse_factor_numeric() using an existing
 * symbolic analysis. Performs a full numeric refactorization and does
 * not attempt to validate or reuse the previous numeric structure.
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_refactor_numeric(const SparseMatrix *A_new, const sparse_analysis_t *analysis,
                                     sparse_factors_t *factors) {
    if (!A_new || !analysis || !factors)
        return SPARSE_ERR_NULL;

    if (A_new->rows != analysis->n || A_new->cols != analysis->n)
        return SPARSE_ERR_SHAPE;

    /* Factor into a temporary first so old factors survive on error */
    sparse_factors_t new_factors;
    memset(&new_factors, 0, sizeof(new_factors));
    sparse_err_t err = sparse_factor_numeric(A_new, analysis, &new_factors);
    if (err != SPARSE_OK)
        return err;

    sparse_factor_free(factors);
    *factors = new_factors;
    return SPARSE_OK;
}
