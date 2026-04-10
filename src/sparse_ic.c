#include "sparse_ic.h"
#include "sparse_matrix_internal.h"
#include "sparse_vector.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * IC(0) factorization
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ic_factor(const SparseMatrix *A, sparse_ilu_t *ic) {
    if (!ic)
        return SPARSE_ERR_NULL;
    /* Zero-initialize output so sparse_ic_free() is safe on any error path */
    ic->L = NULL;
    ic->U = NULL;
    ic->n = 0;
    ic->perm = NULL;
    ic->factor_norm = 0.0;
    if (!A)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;
    ic->n = n;

    /* Empty matrix: valid no-op */
    if (n == 0)
        return SPARSE_OK;

    /* Symmetry check — IC(0) requires a symmetric matrix */
    if (!sparse_is_symmetric(A, 1e-12))
        return SPARSE_ERR_NOT_SPD;

    /* Reject factored matrices — IC(0) needs the original entries */
    if (A->factored)
        return SPARSE_ERR_BADARG;

    /* Reject matrices with non-identity permutations */
    {
        const idx_t *rp = sparse_row_perm(A);
        const idx_t *cp = sparse_col_perm(A);
        if (rp && cp) {
            for (idx_t i = 0; i < n; i++) {
                if (rp[i] != i || cp[i] != i)
                    return SPARSE_ERR_BADARG;
            }
        }
    }

    /* Compute ||A||_inf for relative tolerance */
    ic->factor_norm = sparse_norminf_const(A);

    /* ── Symbolic phase: extract lower triangular pattern ──────────── */

    /* Allocate a dense workspace for one column of L at a time.
     * val[i] holds the accumulated value for row i in the current column.
     * pattern[0..pat_len-1] holds the row indices with nonzero entries. */
    if ((size_t)n > SIZE_MAX / sizeof(double) || (size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;
    double *val = calloc((size_t)n, sizeof(double));
    idx_t *pattern = malloc((size_t)n * sizeof(idx_t));
    if (!val || !pattern) {
        free(val);
        free(pattern);
        return SPARSE_ERR_ALLOC;
    }

    /* Build L column-by-column using the left-looking IC(0) algorithm.
     * IC(0) preserves the sparsity pattern of the lower triangle of A:
     * L(i,k) is nonzero only if A(i,k) != 0 and i >= k. */

    SparseMatrix *L = sparse_create(n, n);
    if (!L) {
        free(val);
        free(pattern);
        return SPARSE_ERR_ALLOC;
    }

    /* Boolean marker: in_pat[i] is nonzero when row i is in the current
     * column's sparsity pattern.  This is used instead of testing val[i]!=0
     * because val[i] can become exactly zero after subtractions. */
    char *in_pat = calloc((size_t)n, sizeof(char));
    if (!in_pat) {
        free(val);
        free(pattern);
        sparse_free(L);
        return SPARSE_ERR_ALLOC;
    }

    double tol = sparse_rel_tol(ic->factor_norm, DROP_TOL);
    sparse_err_t err = SPARSE_OK;

    for (idx_t k = 0; k < n; k++) {
        /* Gather column k of the lower triangle of A into the dense workspace */
        idx_t pat_len = 0;

        for (Node *nd = A->col_headers[k]; nd; nd = nd->down) {
            idx_t i = nd->row;
            if (i >= k && pat_len < n) {
                val[i] = nd->value;
                in_pat[i] = 1;
                pattern[pat_len++] = i;
            }
        }

        /* Subtract contributions from previously computed columns of L:
         * For each j < k where L(k,j) != 0:
         *   For each i >= k where L(i,j) != 0 AND A(i,k) != 0:
         *     val[i] -= L(k,j) * L(i,j) */
        for (Node *lkj = L->row_headers[k]; lkj; lkj = lkj->right) {
            idx_t j = lkj->col;
            if (j >= k)
                break;
            double lkj_val = lkj->value;

            /* Walk column j of L for rows >= k that are in our pattern */
            /* Skip column entries with row < k (column list is sorted by row) */
            Node *lij = L->col_headers[j];
            while (lij && lij->row < k)
                lij = lij->down;
            for (; lij; lij = lij->down) {
                /* Only update if (i,k) is in the sparsity pattern of A */
                if (in_pat[lij->row])
                    val[lij->row] -= lkj_val * lij->value;
            }
        }

        /* Compute L(k,k) = sqrt(val[k]) — the diagonal entry */
        if (val[k] <= tol) {
            err = SPARSE_ERR_NOT_SPD;
            /* Clean up the dense workspace before breaking */
            for (idx_t p = 0; p < pat_len; p++) {
                val[pattern[p]] = 0.0;
                in_pat[pattern[p]] = 0;
            }
            break;
        }
        double lkk = sqrt(val[k]);
        err = sparse_insert(L, k, k, lkk);
        if (err != SPARSE_OK) {
            for (idx_t p = 0; p < pat_len; p++) {
                val[pattern[p]] = 0.0;
                in_pat[pattern[p]] = 0;
            }
            break;
        }

        /* Compute off-diagonal entries: L(i,k) = val[i] / L(k,k) for i > k */
        for (idx_t p = 0; p < pat_len; p++) {
            idx_t i = pattern[p];
            if (i == k)
                continue;
            double lik = val[i] / lkk;
            if (fabs(lik) > 0.0) {
                err = sparse_insert(L, i, k, lik);
                if (err != SPARSE_OK) {
                    for (idx_t q = 0; q < pat_len; q++) {
                        val[pattern[q]] = 0.0;
                        in_pat[pattern[q]] = 0;
                    }
                    goto cleanup;
                }
            }
        }

        /* Clear the dense workspace */
        for (idx_t p = 0; p < pat_len; p++) {
            val[pattern[p]] = 0.0;
            in_pat[pattern[p]] = 0;
        }
    }

cleanup:
    free(val);
    free(pattern);
    free(in_pat);

    if (err != SPARSE_OK) {
        sparse_free(L);
        return err;
    }

    /* Build U = L^T */
    SparseMatrix *U = sparse_transpose(L);
    if (!U) {
        sparse_free(L);
        return SPARSE_ERR_ALLOC;
    }

    ic->L = L;
    ic->U = U;

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * IC(0) solve: forward/backward substitution with L and L^T
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ic_solve(const sparse_ilu_t *ic, const double *r, double *z) {
    if (!ic || !r || !z)
        return SPARSE_ERR_NULL;
    if (ic->n == 0)
        return SPARSE_OK;
    if (!ic->L || !ic->U)
        return SPARSE_ERR_NULL;

    idx_t n = ic->n;
    /* L diagonals scale like sqrt(||A||), so use sqrt(factor_norm) as
     * the reference for singularity detection (not factor_norm itself). */
    double ref = ic->factor_norm > 0.0 ? sqrt(ic->factor_norm) : ic->factor_norm;
    double tol = sparse_rel_tol(ref, DROP_TOL);

    /* Forward substitution: L*y = r  (L has non-unit diagonal)
     * y[i] = (r[i] - sum_{j<i} L(i,j)*y[j]) / L(i,i)
     * Store y in z to avoid extra allocation. */
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        double diag = 0.0;
        for (Node *nd = ic->L->row_headers[i]; nd; nd = nd->right) {
            if (nd->col < i)
                sum += nd->value * z[nd->col];
            else if (nd->col == i)
                diag = nd->value;
        }
        if (fabs(diag) < tol)
            return SPARSE_ERR_SINGULAR;
        z[i] = (r[i] - sum) / diag;
    }

    /* Backward substitution: L^T*x = y  (L^T = U, upper triangular)
     * x[i] = (y[i] - sum_{j>i} U(i,j)*x[j]) / U(i,i)
     * Overwrite z in place. */
    for (idx_t i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        double diag = 0.0;
        for (Node *nd = ic->U->row_headers[i]; nd; nd = nd->right) {
            if (nd->col == i)
                diag = nd->value;
            else if (nd->col > i)
                sum += nd->value * z[nd->col];
        }
        if (fabs(diag) < tol)
            return SPARSE_ERR_SINGULAR;
        z[i] = (z[i] - sum) / diag;
    }

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * IC(0) preconditioner callback
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ic_precond(const void *ctx, idx_t n, const double *r, double *z) {
    if (!ctx || !r || !z)
        return SPARSE_ERR_NULL;
    const sparse_ilu_t *ic = (const sparse_ilu_t *)ctx;
    if (n != ic->n)
        return SPARSE_ERR_SHAPE;
    return sparse_ic_solve(ic, r, z);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Free
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_ic_free(sparse_ilu_t *ic) { sparse_ilu_free(ic); }
