#include "sparse_ilu.h"
#include "sparse_matrix_internal.h"
#include "sparse_vector.h"
#include <stdlib.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * ILU(0) factorization
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ilu_factor(const SparseMatrix *A, sparse_ilu_t *ilu)
{
    if (!A || !ilu) return SPARSE_ERR_NULL;
    if (A->rows != A->cols) return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;
    ilu->L = NULL;
    ilu->U = NULL;
    ilu->n = n;

    /* Empty matrix: treat as a valid no-op factorization.
     * Leave L and U as NULL; sparse_ilu_solve handles n==0 early. */
    if (n == 0) return SPARSE_OK;

    /* Work on a copy of A (using physical indices, no permutations) */
    SparseMatrix *W = sparse_copy(A);
    if (!W) return SPARSE_ERR_ALLOC;

    /* Reset permutations on the working copy so we operate in natural order */
    sparse_reset_perms(W);

    /* IKJ variant of ILU(0) Gaussian elimination:
     * For each row i = 1..n-1:
     *   For each k < i where W(i,k) != 0:
     *     W(i,k) = W(i,k) / W(k,k)       (store multiplier in L position)
     *     For each j > k where W(k,j) != 0:
     *       If W(i,j) exists in sparsity pattern of A:
     *         W(i,j) -= W(i,k) * W(k,j)
     *       Else: drop (ILU(0) rule)
     */
    for (idx_t i = 1; i < n; i++) {
        /* Collect column indices k < i where W(i,k) != 0.
         * We need to process them in ascending order of k. */
        Node *node_ik = W->row_headers[i];
        while (node_ik) {
            idx_t k = node_ik->col;
            if (k >= i) break;  /* only process k < i (lower triangle) */

            double wkk = sparse_get_phys(W, k, k);
            if (fabs(wkk) < 1e-30) {
                sparse_free(W);
                return SPARSE_ERR_SINGULAR;
            }

            /* W(i,k) /= W(k,k) — store the multiplier */
            double mult = node_ik->value / wkk;
            node_ik->value = mult;

            /* For each j > k where W(k,j) != 0: update W(i,j) if it exists */
            Node *node_kj = W->row_headers[k];
            Node *scan = W->row_headers[i];
            while (node_kj) {
                idx_t j = node_kj->col;
                if (j > k) {
                    /* Advance scan pointer to column >= j.
                     * Row i entries are sorted by column, so we only
                     * move forward (never restart from head). */
                    while (scan && scan->col < j)
                        scan = scan->right;
                    if (scan && scan->col == j) {
                        scan->value -= mult * node_kj->value;
                    }
                    /* else: drop (ILU(0) no-fill rule) */
                }
                node_kj = node_kj->right;
            }

            node_ik = node_ik->right;
        }
    }

    /* Verify all diagonal entries are nonzero (needed for backward sub) */
    for (idx_t i = 0; i < n; i++) {
        double wii = sparse_get_phys(W, i, i);
        if (fabs(wii) < 1e-30) {
            sparse_free(W);
            return SPARSE_ERR_SINGULAR;
        }
    }

    /* Extract L (unit lower triangular) and U (upper triangular with diagonal) */
    SparseMatrix *L = sparse_create(n, n);
    SparseMatrix *U = sparse_create(n, n);
    if (!L || !U) {
        sparse_free(L);
        sparse_free(U);
        sparse_free(W);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++) {
        /* Unit diagonal for L */
        sparse_insert(L, i, i, 1.0);

        Node *node = W->row_headers[i];
        while (node) {
            idx_t j = node->col;
            double val = node->value;
            if (j < i) {
                /* Lower triangle → L */
                sparse_insert(L, i, j, val);
            } else {
                /* Diagonal and upper triangle → U */
                sparse_insert(U, i, j, val);
            }
            node = node->right;
        }
    }

    ilu->L = L;
    ilu->U = U;
    ilu->n = n;

    sparse_free(W);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * ILU(0) solve: L*U*z = r  →  L*y = r (forward), U*z = y (backward)
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ilu_solve(const sparse_ilu_t *ilu,
                               const double *r, double *z)
{
    if (!ilu || !r || !z) return SPARSE_ERR_NULL;

    idx_t n = ilu->n;
    if (n == 0) return SPARSE_OK;

    if (!ilu->L || !ilu->U) return SPARSE_ERR_NULL;

    /* Allocate workspace for intermediate vector y */
    double *y = malloc((size_t)n * sizeof(double));
    if (!y) return SPARSE_ERR_ALLOC;

    /* Forward substitution: L*y = r
     * L is unit lower triangular, so y[i] = r[i] - sum_{j<i} L(i,j)*y[j] */
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        Node *node = ilu->L->row_headers[i];
        while (node) {
            if (node->col < i) {
                sum += node->value * y[node->col];
            }
            node = node->right;
        }
        y[i] = r[i] - sum;
    }

    /* Backward substitution: U*z = y
     * U is upper triangular with diagonal */
    for (idx_t i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        double diag = 0.0;
        Node *node = ilu->U->row_headers[i];
        while (node) {
            if (node->col == i) {
                diag = node->value;
            } else if (node->col > i) {
                sum += node->value * z[node->col];
            }
            node = node->right;
        }
        if (fabs(diag) < 1e-30) {
            free(y);
            return SPARSE_ERR_SINGULAR;
        }
        z[i] = (y[i] - sum) / diag;
    }

    free(y);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Free & preconditioner callback
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_ilu_free(sparse_ilu_t *ilu)
{
    if (!ilu) return;
    sparse_free(ilu->L);
    sparse_free(ilu->U);
    ilu->L = NULL;
    ilu->U = NULL;
    ilu->n = 0;
}

sparse_err_t sparse_ilu_precond(const void *ctx, idx_t n,
                                 const double *r, double *z)
{
    if (!ctx) return SPARSE_ERR_NULL;
    const sparse_ilu_t *ilu = (const sparse_ilu_t *)ctx;
    if (n != ilu->n) return SPARSE_ERR_SHAPE;
    return sparse_ilu_solve(ilu, r, z);
}
