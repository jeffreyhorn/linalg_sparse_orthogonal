#include "sparse_cholesky.h"
#include "sparse_reorder.h"
#include "sparse_matrix_internal.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

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
sparse_err_t sparse_cholesky_factor(SparseMatrix *mat)
{
    if (!mat) return SPARSE_ERR_NULL;
    idx_t n = mat->rows;
    if (n != mat->cols) return SPARSE_ERR_SHAPE;
    if (n == 0) return SPARSE_OK;

    /* Dense accumulator for column k (indices k..n-1) */
    double *col_acc = calloc((size_t)n, sizeof(double));
    if (!col_acc) return SPARSE_ERR_ALLOC;

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
        /* Initialize accumulator with column k of current matrix (rows >= k) */
        memset(col_acc, 0, (size_t)n * sizeof(double));
        Node *node = mat->col_headers[k];
        while (node) {
            if (node->row >= k)
                col_acc[node->row] = node->value;
            node = node->down;
        }

        /* Subtract contributions from previously computed columns of L:
         * For each j < k where L(k,j) != 0:
         *   col_acc[i] -= L(i,j) * L(k,j) for all i >= k */
        Node *row_k = mat->row_headers[k];
        while (row_k) {
            idx_t j = row_k->col;
            if (j >= k) { row_k = row_k->right; continue; }
            double l_kj = row_k->value;

            /* Walk column j for entries L(i,j) with i >= k */
            Node *col_j = mat->col_headers[j];
            while (col_j) {
                if (col_j->row >= k)
                    col_acc[col_j->row] -= col_j->value * l_kj;
                col_j = col_j->down;
            }
            row_k = row_k->right;
        }

        /* Diagonal: L(k,k) = sqrt(col_acc[k]) */
        if (col_acc[k] <= 0.0) {
            free(col_acc);
            return SPARSE_ERR_NOT_SPD;
        }
        double l_kk = sqrt(col_acc[k]);

        /* Write L(k,k) */
        sparse_err_t err = sparse_insert(mat, k, k, l_kk);
        if (err != SPARSE_OK) { free(col_acc); return err; }

        /* Off-diagonal: L(i,k) = col_acc[i] / L(k,k) for i > k */
        for (idx_t i = k + 1; i < n; i++) {
            double l_ik = col_acc[i] / l_kk;
            if (fabs(l_ik) < DROP_TOL * l_kk) {
                /* Drop small fill-in; remove if entry exists */
                if (sparse_get_phys(mat, i, k) != 0.0)
                    sparse_insert(mat, i, k, 0.0);
            } else {
                err = sparse_insert(mat, i, k, l_ik);
                if (err != SPARSE_OK) { free(col_acc); return err; }
            }
        }
    }

    free(col_acc);
    return SPARSE_OK;
}

/* ─── Cholesky factorization with options ────────────────────────────── */

sparse_err_t sparse_cholesky_factor_opts(SparseMatrix *mat,
                                         const sparse_cholesky_opts_t *opts)
{
    if (!mat || !opts) return SPARSE_ERR_NULL;
    if (mat->rows != mat->cols) return SPARSE_ERR_SHAPE;
    /* TODO: implement reordering in Day 3 */
    return sparse_cholesky_factor(mat);
}

/* ─── Cholesky solve ─────────────────────────────────────────────────── */

sparse_err_t sparse_cholesky_solve(const SparseMatrix *mat,
                                   const double *b, double *x)
{
    if (!mat || !b || !x) return SPARSE_ERR_NULL;
    /* TODO: implement in Day 3 */
    (void)mat; (void)b; (void)x;
    return SPARSE_ERR_BADARG;
}
