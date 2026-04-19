#include "sparse_cholesky.h"
#include "sparse_analysis.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_matrix_internal.h"
#include "sparse_reorder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── Cholesky factorization ─────────────────────────────────────────── */

/*
 * Sparse Cholesky: A = L * L^T  (in-place, lower triangle only)
 *
 * Algorithm (left-looking / column-by-column):
 *   for k = 0 to n-1:
 *     L(k,k) = sqrt( A(k,k) - sum_{j<k} L(k,j)^2 )
 *     for each i > k where A(i,k) != 0 or fill-in occurs:
 *       L(i,k) = ( A(i,k) - sum_{j<k} L(i,j)*L(k,j) ) / L(k,k)
 *
 * We use a dense column accumulator for each column k to handle fill-in
 * efficiently, then write back nonzeros into the sparse structure.
 */
sparse_err_t sparse_cholesky_factor(SparseMatrix *mat) {
    if (!mat)
        return SPARSE_ERR_NULL;
    idx_t n = mat->rows;
    if (n != mat->cols)
        return SPARSE_ERR_SHAPE;
    if (n == 0)
        return SPARSE_OK;

    /* Clear factored flag immediately so an aborted factorization
     * cannot leave a stale 'factored' state. */
    mat->factored = 0;

    /* Validate symmetry before allocating or modifying anything */
    if (!sparse_is_symmetric(mat, 1e-12))
        return SPARSE_ERR_NOT_SPD;

    /* Compute and cache ||A||_inf before factorization for relative tolerance */
    double anorm;
    sparse_err_t nerr = sparse_norminf(mat, &anorm);
    if (nerr != SPARSE_OK)
        return nerr;
    mat->factor_norm = anorm;

    /* Dense accumulator for column k (indices k..n-1) */
    double *col_acc = calloc((size_t)n, sizeof(double));
    int *nz_row = calloc((size_t)n, sizeof(int));
    idx_t *nz_rows = malloc((size_t)n * sizeof(idx_t));
    if (!col_acc || !nz_row || !nz_rows) {
        free(col_acc);
        free(nz_row);
        free(nz_rows);
        return SPARSE_ERR_ALLOC;
    }

    /* Remove upper triangle entries (we only store L) */
    for (idx_t i = 0; i < n; i++) {
        Node *node = mat->row_headers[i];
        while (node) {
            Node *next = node->right;
            if (node->col > i) {
                sparse_insert(mat, i, node->col, 0.0); /* removes entry */
            }
            node = next;
        }
    }

    for (idx_t k = 0; k < n; k++) {
        /* Initialize accumulator with column k of current matrix (rows >= k).
         * Track nonzero rows in nz_rows[] / nz_row[] for O(nnz) flush. */
        idx_t nnz_rows = 0;
        Node *node = mat->col_headers[k];
        while (node) {
            if (node->row >= k) {
                col_acc[node->row] = node->value;
                if (!nz_row[node->row]) {
                    nz_row[node->row] = 1;
                    nz_rows[nnz_rows++] = node->row; // NOLINT(clang-analyzer-security.ArrayBound)
                }
            }
            node = node->down;
        }

        /* Subtract contributions from previously computed columns of L:
         * For each j < k where L(k,j) != 0:
         *   col_acc[i] -= L(i,j) * L(k,j) for all i >= k */
        Node *row_k = mat->row_headers[k];
        while (row_k) {
            idx_t j = row_k->col;
            if (j >= k) {
                row_k = row_k->right;
                continue;
            }
            double l_kj = row_k->value;

            /* Walk column j for entries L(i,j) with i >= k */
            Node *col_j = mat->col_headers[j];
            while (col_j) {
                if (col_j->row >= k) {
                    col_acc[col_j->row] -= col_j->value * l_kj;
                    if (!nz_row[col_j->row]) {
                        nz_row[col_j->row] = 1;
                        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
                        nz_rows[nnz_rows++] = col_j->row;
                    }
                }
                col_j = col_j->down;
            }
            row_k = row_k->right;
        }

        /* Diagonal: L(k,k) = sqrt(col_acc[k]) */
        if (col_acc[k] <= 0.0) {
            free(col_acc);
            free(nz_row);
            free(nz_rows);
            return SPARSE_ERR_NOT_SPD;
        }
        double l_kk = sqrt(col_acc[k]);

        /* Write L(k,k) */
        sparse_err_t err = sparse_insert(mat, k, k, l_kk);
        if (err != SPARSE_OK) {
            free(col_acc);
            free(nz_row);
            free(nz_rows);
            return err;
        }

        /* Off-diagonal: walk only rows with nonzero accumulator values */
        for (idx_t t = 0; t < nnz_rows; t++) {
            idx_t i = nz_rows[t];
            if (i <= k) {
                /* Reset tracking for row k (diagonal already handled) */
                col_acc[i] = 0.0;
                nz_row[i] = 0;
                continue;
            }
            double l_ik = col_acc[i] / l_kk;
            if (fabs(l_ik) < DROP_TOL * l_kk) {
                /* Drop small fill-in; remove if entry exists */
                if (sparse_get_phys(mat, i, k) != 0.0)
                    sparse_insert(mat, i, k, 0.0);
            } else {
                err = sparse_insert(mat, i, k, l_ik);
                if (err != SPARSE_OK) {
                    free(col_acc);
                    free(nz_row);
                    free(nz_rows);
                    return err;
                }
            }
            col_acc[i] = 0.0;
            nz_row[i] = 0;
        }
    }

    free(col_acc);
    free(nz_row);
    free(nz_rows);
    mat->factored = 1;
    return SPARSE_OK;
}

/* ─── Cholesky factorization with options ────────────────────────────── */

sparse_err_t sparse_cholesky_factor_opts(SparseMatrix *mat, const sparse_cholesky_opts_t *opts) {
    if (!mat || !opts)
        return SPARSE_ERR_NULL;
    idx_t n = mat->rows;
    if (n != mat->cols)
        return SPARSE_ERR_SHAPE;

    /* Clear any previous reorder permutation */
    free(mat->reorder_perm);
    mat->reorder_perm = NULL;

    /* Apply fill-reducing reordering if requested.  This permutes the
     * SparseMatrix into its final coordinate space and resets the
     * internal row/col perms to identity — the same end-state both
     * the linked-list factor and the CSC factor expect to start from. */
    if (opts->reorder != SPARSE_REORDER_NONE && n > 1) {
        idx_t *perm = malloc((size_t)n * sizeof(idx_t));
        if (!perm)
            return SPARSE_ERR_ALLOC;

        sparse_err_t err;
        switch (opts->reorder) {
        case SPARSE_REORDER_RCM:
            err = sparse_reorder_rcm(mat, perm);
            break;
        case SPARSE_REORDER_AMD:
            err = sparse_reorder_amd(mat, perm);
            break;
        default:
            free(perm);
            return SPARSE_ERR_BADARG;
        }

        if (err != SPARSE_OK) {
            free(perm);
            return err;
        }

        /* Apply symmetric permutation in-place */
        SparseMatrix *PA = NULL;
        err = sparse_permute(mat, perm, perm, &PA);
        if (err != SPARSE_OK) {
            free(perm);
            return err;
        }

        /* Swap internal data from PA into mat */
        pool_free_all(&mat->pool);
        free(mat->row_headers);
        free(mat->col_headers);

        mat->row_headers = PA->row_headers;
        mat->col_headers = PA->col_headers;
        mat->pool = PA->pool;
        mat->nnz = PA->nnz;
        mat->cached_norm = PA->cached_norm;
        mat->factor_norm = -1.0; /* reset: not Cholesky-factored */

        PA->row_headers = NULL;
        PA->col_headers = NULL;
        PA->pool.head = NULL;
        PA->pool.current = NULL;
        PA->pool.free_list = NULL;
        sparse_free(PA);

        /* Reset permutations to identity */
        for (idx_t i = 0; i < n; i++) {
            mat->row_perm[i] = i;
            mat->inv_row_perm[i] = i;
            mat->col_perm[i] = i;
            mat->inv_col_perm[i] = i;
        }

        /* Store reorder permutation for solve to unpermute */
        mat->reorder_perm = perm;
    }

    /* Backend dispatch: AUTO compares against SPARSE_CSC_THRESHOLD;
     * LINKED_LIST / CSC force one branch.  Records the chosen path
     * back through the caller's opts->used_csc_path if provided. */
    int use_csc;
    switch (opts->backend) {
    case SPARSE_CHOL_BACKEND_LINKED_LIST:
        use_csc = 0;
        break;
    case SPARSE_CHOL_BACKEND_CSC:
        use_csc = 1;
        break;
    case SPARSE_CHOL_BACKEND_AUTO:
    default:
        use_csc = (n >= SPARSE_CSC_THRESHOLD);
        break;
    }
    if (opts->used_csc_path)
        *opts->used_csc_path = use_csc ? 1 : 0;

    if (!use_csc)
        return sparse_cholesky_factor(mat);

    /* CSC path: analyse, convert with the full symbolic L pattern,
     * factor via the batched supernodal kernel, transplant the result
     * back into `mat`.  After the reorder + transplant above mat lives
     * in the final coordinate space, so we run the symbolic analysis
     * with REORDER_NONE — the ordering has already been folded in.
     * `chol_csc_from_sparse_with_analysis` pre-populates the full
     * sym_L pattern (Sprint 18 Day 12), which the supernodal extract
     * requires to see every fill row of each supernode's panel. */
    sparse_analysis_opts_t an_opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};
    sparse_analysis_t an = {0};
    sparse_err_t err = sparse_analyze(mat, &an_opts, &an);
    if (err != SPARSE_OK)
        return err;

    CholCsc *L_csc = NULL;
    err = chol_csc_from_sparse_with_analysis(mat, &an, &L_csc);
    if (err != SPARSE_OK) {
        sparse_analysis_free(&an);
        return err;
    }

    err = chol_csc_eliminate_supernodal(L_csc, 4);
    if (err != SPARSE_OK) {
        chol_csc_free(L_csc);
        sparse_analysis_free(&an);
        return err;
    }

    /* Writeback overwrites mat's storage with L and re-publishes the
     * reorder perm; we pass mat->reorder_perm so writeback's internal
     * memcpy + free dance produces the same pointer identity the
     * linked-list path would. */
    err = chol_csc_writeback_to_sparse(L_csc, mat, mat->reorder_perm);
    chol_csc_free(L_csc);
    sparse_analysis_free(&an);
    return err;
}

/* ─── Cholesky solve ─────────────────────────────────────────────────── */

sparse_err_t sparse_cholesky_solve(const SparseMatrix *mat, const double *b, double *x) {
    if (!mat || !b || !x)
        return SPARSE_ERR_NULL;
    if (!mat->factored)
        return SPARSE_ERR_BADARG;
    idx_t n = mat->rows;
    const idx_t *rperm = mat->reorder_perm;

    double *y = malloc((size_t)n * sizeof(double));
    double *b_perm = NULL;
    if (rperm) {
        b_perm = malloc((size_t)n * sizeof(double));
        if (!b_perm) {
            free(y);
            return SPARSE_ERR_ALLOC;
        }
    }
    if (!y) {
        free(b_perm);
        return SPARSE_ERR_ALLOC;
    }

    /* If reorder permutation exists, permute b */
    const double *b_eff = b;
    if (rperm) {
        for (idx_t i = 0; i < n; i++)
            b_perm[i] = b[rperm[i]];
        b_eff = b_perm;
    }

    /* Singularity threshold: relative to ||L||_inf ≈ sqrt(||A||_inf).
     * L entries scale as sqrt of A entries in Cholesky, so use the
     * square root of the original matrix norm as reference. */
    double l_norm = (mat->factor_norm > 0.0) ? sqrt(mat->factor_norm) : 0.0;
    double sing_tol = sparse_rel_tol(l_norm, DROP_TOL);

    /* Forward substitution: solve L*y = b_eff
     * L is lower triangular stored in lower triangle (physical indices).
     * y[i] = (b[i] - sum_{j<i} L(i,j)*y[j]) / L(i,i) */
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        double l_ii = 0.0;
        Node *node = mat->row_headers[i];
        while (node) {
            if (node->col < i)
                sum += node->value * y[node->col]; // NOLINT(clang-analyzer-security.ArrayBound)
            else if (node->col == i)
                l_ii = node->value;
            node = node->right;
        }
        if (fabs(l_ii) < sing_tol) {
            free(y);
            free(b_perm);
            return SPARSE_ERR_SINGULAR;
        }
        y[i] = (b_eff[i] - sum) / l_ii;
    }

    /* Backward substitution: solve L^T*x = y
     * L^T is upper triangular. L^T(i,j) = L(j,i).
     * x[i] = (y[i] - sum_{j>i} L(j,i)*x[j]) / L(i,i)
     * Walk column i to find L entries with row > i (these are L^T entries). */
    for (idx_t i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        double l_ii = 0.0;
        Node *node = mat->col_headers[i];
        while (node) {
            if (node->row > i)
                sum += node->value * x[node->row];
            else if (node->row == i)
                l_ii = node->value;
            node = node->down;
        }
        if (fabs(l_ii) < sing_tol) {
            free(y);
            free(b_perm);
            return SPARSE_ERR_SINGULAR;
        }
        x[i] = (y[i] - sum) / l_ii; // NOLINT(clang-analyzer-core.UndefinedBinaryOperatorResult)
    }

    /* If reorder permutation exists, unpermute x */
    if (rperm) {
        memcpy(y, x, (size_t)n * sizeof(double)); /* reuse y as temp */
        for (idx_t i = 0; i < n; i++)
            x[rperm[i]] = y[i];
    }

    free(y);
    free(b_perm);
    return SPARSE_OK;
}
