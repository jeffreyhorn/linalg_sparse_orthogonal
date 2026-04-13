#include "sparse_analysis.h"
#include "sparse_analysis_internal.h"
#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_lu.h"
#include "sparse_reorder.h"

#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_analyze — compute symbolic analysis
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_analyze(const SparseMatrix *A, const sparse_analysis_opts_t *opts,
                            sparse_analysis_t *analysis) {
    if (!A || !analysis)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    /* Default options: Cholesky, no reordering */
    sparse_factor_type_t ftype = SPARSE_FACTOR_CHOLESKY;
    sparse_reorder_t reorder = SPARSE_REORDER_NONE;
    if (opts) {
        ftype = opts->factor_type;
        reorder = opts->reorder;
    }

    idx_t n = A->rows;
    memset(analysis, 0, sizeof(*analysis));
    analysis->n = n;
    analysis->type = ftype;

    /* Cache ||A||_inf */
    analysis->analysis_norm = sparse_norminf_const(A);

    /* Compute fill-reducing permutation if requested */
    sparse_err_t err = SPARSE_OK;
    if (reorder != SPARSE_REORDER_NONE && n > 0) {
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

        /* Allocate etree and postorder */
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

static SparseMatrix *build_permuted_copy(const SparseMatrix *A, const idx_t *perm) {
    if (perm) {
        SparseMatrix *B = NULL;
        if (sparse_permute(A, perm, perm, &B) != SPARSE_OK)
            return NULL;
        return B;
    }
    return sparse_copy(A);
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

    idx_t n = analysis->n;
    memset(factors, 0, sizeof(*factors));
    factors->type = analysis->type;
    factors->n = n;

    switch (analysis->type) {
    case SPARSE_FACTOR_CHOLESKY: {
        /* Build (optionally permuted) copy and factor with existing Cholesky */
        SparseMatrix *L = build_permuted_copy(A, analysis->perm);
        if (!L)
            return SPARSE_ERR_ALLOC;

        sparse_err_t err = sparse_cholesky_factor(L);
        if (err != SPARSE_OK) {
            sparse_free(L);
            return err;
        }

        factors->LU = L;
        factors->factor_norm = L->factor_norm;
        break;
    }

    case SPARSE_FACTOR_LU: {
        /* Build (optionally permuted) copy and factor with existing LU */
        SparseMatrix *LU = build_permuted_copy(A, analysis->perm);
        if (!LU)
            return SPARSE_ERR_ALLOC;

        sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
        if (err != SPARSE_OK) {
            sparse_free(LU);
            return err;
        }

        factors->LU = LU;
        factors->factor_norm = LU->factor_norm;
        break;
    }

    case SPARSE_FACTOR_LDLT: {
        /* Build (optionally permuted) copy and factor with existing LDL^T */
        SparseMatrix *B = build_permuted_copy(A, analysis->perm);
        if (!B)
            return SPARSE_ERR_ALLOC;

        sparse_ldlt_t ldlt;
        sparse_err_t err = sparse_ldlt_factor(B, &ldlt);
        sparse_free(B);
        if (err != SPARSE_OK)
            return err;

        /* Transfer ownership from ldlt to factors */
        factors->LU = ldlt.L;
        factors->factor_norm = ldlt.factor_norm;
        factors->D = ldlt.D;
        factors->D_offdiag = ldlt.D_offdiag;
        factors->pivot_size = ldlt.pivot_size;

        /* Compose LDL^T pivot perm with analysis perm if needed */
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
    if (!factors->LU)
        return SPARSE_ERR_BADARG;

    idx_t n = factors->n;
    const idx_t *perm = analysis->perm;

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
        err = sparse_cholesky_solve(factors->LU, b_eff, x_tmp);
        break;
    case SPARSE_FACTOR_LU:
        err = sparse_lu_solve(factors->LU, b_eff, x_tmp);
        break;
    case SPARSE_FACTOR_LDLT: {
        /* Reconstruct a temporary sparse_ldlt_t for the solve call */
        sparse_ldlt_t ldlt_tmp;
        ldlt_tmp.L = factors->LU;
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
    sparse_free(factors->LU);
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
