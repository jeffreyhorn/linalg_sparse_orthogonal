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

    /* Reject matrices with non-identity permutations (e.g., after LU pivoting).
     * ILU(0) operates on physical storage and assumes identity perms. */
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

    /* Empty matrix: treat as a valid no-op factorization.
     * Leave L and U as NULL; sparse_ilu_solve handles n==0 early. */
    if (n == 0) return SPARSE_OK;

    /* Work on a copy of A (using physical indices, no permutations) */
    SparseMatrix *W = sparse_copy(A);
    if (!W) return SPARSE_ERR_ALLOC;

    /* Reset permutations on the working copy so we operate in natural order.
     * Note: this only resets the perm arrays to identity; it does not reorder
     * physical storage.  The caller must pass an unfactored matrix (one whose
     * permutations are still identity) for correct results. */
    sparse_reset_perms(W);

    /* Cache diagonal node pointers for O(1) pivot access during elimination.
     * This avoids repeated O(nnz_row) scans via sparse_get_phys(). */
    Node **diag_nodes = malloc((size_t)n * sizeof(Node *));
    if (!diag_nodes) { sparse_free(W); return SPARSE_ERR_ALLOC; }
    for (idx_t i = 0; i < n; i++) {
        diag_nodes[i] = NULL;
        Node *nd = W->row_headers[i];
        while (nd) {
            if (nd->col == i) { diag_nodes[i] = nd; break; }
            if (nd->col > i) break;
            nd = nd->right;
        }
    }

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
        Node *node_ik = W->row_headers[i];
        while (node_ik) {
            idx_t k = node_ik->col;
            if (k >= i) break;

            if (!diag_nodes[k] || fabs(diag_nodes[k]->value) < 1e-30) {
                free(diag_nodes);
                sparse_free(W);
                return SPARSE_ERR_SINGULAR;
            }

            double mult = node_ik->value / diag_nodes[k]->value;
            node_ik->value = mult;

            Node *node_kj = W->row_headers[k];
            Node *scan = W->row_headers[i];
            while (node_kj) {
                idx_t j = node_kj->col;
                if (j > k) {
                    while (scan && scan->col < j)
                        scan = scan->right;
                    if (scan && scan->col == j) {
                        scan->value -= mult * node_kj->value;
                    }
                }
                node_kj = node_kj->right;
            }

            node_ik = node_ik->right;
        }
    }

    /* Verify all diagonal entries are nonzero (needed for backward sub) */
    for (idx_t i = 0; i < n; i++) {
        double wii = diag_nodes[i] ? diag_nodes[i]->value : 0.0;
        if (fabs(wii) < 1e-30) {
            free(diag_nodes);
            sparse_free(W);
            return SPARSE_ERR_SINGULAR;
        }
    }
    free(diag_nodes);

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
        if (sparse_insert(L, i, i, 1.0) != SPARSE_OK) {
            sparse_free(L); sparse_free(U); sparse_free(W);
            return SPARSE_ERR_ALLOC;
        }

        Node *node = W->row_headers[i];
        while (node) {
            idx_t j = node->col;
            double val = node->value;
            sparse_err_t ins_err;
            if (j < i) {
                /* Lower triangle → L */
                ins_err = sparse_insert(L, i, j, val);
            } else {
                /* Diagonal and upper triangle → U */
                ins_err = sparse_insert(U, i, j, val);
            }
            if (ins_err != SPARSE_OK) {
                sparse_free(L); sparse_free(U); sparse_free(W);
                return SPARSE_ERR_ALLOC;
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

    /* Forward substitution: L*y = r  (store y in z to avoid allocation)
     * L is unit lower triangular, so z[i] = r[i] - sum_{j<i} L(i,j)*z[j] */
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        Node *node = ilu->L->row_headers[i];
        while (node) {
            if (node->col < i) {
                sum += node->value * z[node->col];
            }
            node = node->right;
        }
        z[i] = r[i] - sum;
    }

    /* Backward substitution: U*z = y  (y is already in z)
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
            return SPARSE_ERR_SINGULAR;
        }
        z[i] = (z[i] - sum) / diag;
    }

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
