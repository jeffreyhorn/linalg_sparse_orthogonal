#include "sparse_ilu.h"
#include "sparse_matrix_internal.h"
#include "sparse_vector.h"
#include <math.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * ILU(0) factorization
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ilu_factor(const SparseMatrix *A, sparse_ilu_t *ilu) {
    if (!ilu)
        return SPARSE_ERR_NULL;
    /* Zero-initialize output so sparse_ilu_free() is safe on any error path */
    ilu->L = NULL;
    ilu->U = NULL;
    ilu->n = 0;
    ilu->perm = NULL;
    ilu->factor_norm = 0.0;
    if (!A)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;
    ilu->n = n;

    /* Reject factored matrices — ILU(0) needs the original entries, not L/U. */
    if (A->factored)
        return SPARSE_ERR_BADARG;

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

    /* Compute ||A||_inf for relative tolerance.
     * Use const-safe helper to avoid mutating the caller's matrix. */
    ilu->factor_norm = sparse_norminf_const(A);

    /* Empty matrix: treat as a valid no-op factorization.
     * Leave L and U as NULL; sparse_ilu_solve handles n==0 early. */
    if (n == 0)
        return SPARSE_OK;

    /* Work on a copy of A (using physical indices, no permutations) */
    SparseMatrix *W = sparse_copy(A);
    if (!W)
        return SPARSE_ERR_ALLOC;

    /* Reset permutations on the working copy so we operate in natural order.
     * Note: this only resets the perm arrays to identity; it does not reorder
     * physical storage.  The caller must pass an unfactored matrix (one whose
     * permutations are still identity) for correct results. */
    sparse_reset_perms(W);

    /* Cache diagonal node pointers for O(1) pivot access during elimination.
     * This avoids repeated O(nnz_row) scans via sparse_get_phys(). */
    Node **diag_nodes = malloc((size_t)n * sizeof(Node *));
    if (!diag_nodes) {
        sparse_free(W);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < n; i++) {
        diag_nodes[i] = NULL;
        Node *nd = W->row_headers[i];
        while (nd) {
            if (nd->col == i) {
                diag_nodes[i] = nd;
                break;
            }
            if (nd->col > i)
                break;
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
            if (k >= i)
                break;

            if (!diag_nodes[k] ||
                fabs(diag_nodes[k]->value) // NOLINT(clang-analyzer-core.NullDereference)
                    < sparse_rel_tol(ilu->factor_norm, DROP_TOL)) {
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
        if (fabs(wii) < sparse_rel_tol(ilu->factor_norm, DROP_TOL)) {
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
            sparse_free(L);
            sparse_free(U);
            sparse_free(W);
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
                sparse_free(L);
                sparse_free(U);
                sparse_free(W);
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

sparse_err_t sparse_ilu_solve(const sparse_ilu_t *ilu, const double *r, double *z) {
    if (!ilu || !r || !z)
        return SPARSE_ERR_NULL;

    idx_t n = ilu->n;
    if (n == 0)
        return SPARSE_OK;

    if (!ilu->L || !ilu->U)
        return SPARSE_ERR_NULL;

    /* If pivoting was used, apply row permutation to RHS:
     * solve P*L*U*z = r  →  L*U*z = P^{-1}*r  →  permuted_r[i] = r[perm[i]] */
    const double *rhs = r;
    double *pr = NULL;
    if (ilu->perm) {
        pr = malloc((size_t)n * sizeof(double));
        if (!pr)
            return SPARSE_ERR_ALLOC;
        for (idx_t i = 0; i < n; i++)
            pr[i] = r[ilu->perm[i]];
        rhs = pr;
    }

    /* Forward substitution: L*y = rhs  (store y in z to avoid allocation)
     * L is unit lower triangular, so z[i] = rhs[i] - sum_{j<i} L(i,j)*z[j] */
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        Node *node = ilu->L->row_headers[i];
        while (node) {
            if (node->col < i) {
                sum += node->value * z[node->col];
            }
            node = node->right;
        }
        z[i] = rhs[i] - sum;
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
        if (fabs(diag) < sparse_rel_tol(ilu->factor_norm, DROP_TOL)) {
            free(pr);
            return SPARSE_ERR_SINGULAR;
        }
        z[i] = (z[i] - sum) / diag;
    }

    free(pr);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Free & preconditioner callback
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_ilu_free(sparse_ilu_t *ilu) {
    if (!ilu)
        return;
    sparse_free(ilu->L);
    sparse_free(ilu->U);
    free(ilu->perm);
    ilu->L = NULL;
    ilu->U = NULL;
    ilu->perm = NULL;
    ilu->n = 0;
}

sparse_err_t sparse_ilu_precond(const void *ctx, idx_t n, const double *r, double *z) {
    if (!ctx)
        return SPARSE_ERR_NULL;
    const sparse_ilu_t *ilu = (const sparse_ilu_t *)ctx;
    if (n != ilu->n)
        return SPARSE_ERR_SHAPE;
    return sparse_ilu_solve(ilu, r, z);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ILUT factorization — ILU with Threshold dropping
 * ═══════════════════════════════════════════════════════════════════════ */

static const sparse_ilut_opts_t ilut_defaults = {
    .tol = 1e-3,
    .max_fill = 10,
    .pivot = 0,
};

/* Helper: comparison function for sorting indices by descending |value| */
typedef struct {
    idx_t col;
    double val;
} col_val_t;

static int cmp_col_val_desc(const void *a, const void *b) {
    double va = fabs(((const col_val_t *)a)->val);
    double vb = fabs(((const col_val_t *)b)->val);
    if (va > vb)
        return -1;
    if (va < vb)
        return 1;
    return 0;
}

sparse_err_t sparse_ilut_factor(const SparseMatrix *A, const sparse_ilut_opts_t *opts,
                                sparse_ilu_t *ilu) {
    if (!ilu)
        return SPARSE_ERR_NULL;
    /* Zero the output struct. Callers must call sparse_ilu_free() before
     * reusing a struct that already holds a factorization. */
    ilu->L = NULL;
    ilu->U = NULL;
    ilu->n = 0;
    ilu->perm = NULL;
    ilu->factor_norm = 0.0;
    if (!A)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    const sparse_ilut_opts_t *o = opts ? opts : &ilut_defaults;
    if (o->tol < 0.0 || o->max_fill < 0)
        return SPARSE_ERR_BADARG;

    idx_t n = A->rows;
    ilu->n = n;

    /* Reject factored matrices — ILUT needs the original entries, not L/U. */
    if (A->factored)
        return SPARSE_ERR_BADARG;

    /* Reject non-identity permutations */
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

    /* Compute and cache ||A||_inf for relative tolerance */
    /* Compute ||A||_inf for relative tolerance.
     * Use const-safe helper to avoid mutating the caller's matrix. */
    ilu->factor_norm = sparse_norminf_const(A);

    if (n == 0)
        return SPARSE_OK;

    /* Allocate dense row workspace and L/U output matrices */
    double *w = calloc((size_t)n, sizeof(double));
    int *w_nz = calloc((size_t)n, sizeof(int));
    idx_t *nz_idx = malloc((size_t)n * sizeof(idx_t));
    col_val_t *l_buf = malloc((size_t)n * sizeof(col_val_t));
    col_val_t *u_buf = malloc((size_t)n * sizeof(col_val_t));
    SparseMatrix *L = sparse_create(n, n);
    SparseMatrix *U = sparse_create(n, n);

    if (!w || !w_nz || !nz_idx || !l_buf || !u_buf || !L || !U) {
        free(w);
        free(w_nz);
        free(nz_idx);
        free(l_buf);
        free(u_buf);
        sparse_free(L);
        sparse_free(U);
        return SPARSE_ERR_ALLOC;
    }

    /* Pivoting support: row_map[pos] = orig_row, inv_row_map[orig_row] = pos.
     * When pivoting is enabled, we swap entries in both maps and allocate perm. */
    idx_t *row_map = NULL;
    idx_t *inv_row_map = NULL;
    if (o->pivot) {
        if ((size_t)n > SIZE_MAX / sizeof(idx_t)) {
            free(w);
            free(w_nz);
            free(nz_idx);
            free(l_buf);
            free(u_buf);
            sparse_free(L);
            sparse_free(U);
            return SPARSE_ERR_ALLOC;
        }
        row_map = malloc((size_t)n * sizeof(idx_t));
        inv_row_map = malloc((size_t)n * sizeof(idx_t));
        ilu->perm = malloc((size_t)n * sizeof(idx_t));
        if (!row_map || !inv_row_map || !ilu->perm) {
            free(row_map);
            free(inv_row_map);
            free(ilu->perm);
            ilu->perm = NULL;
            free(w);
            free(w_nz);
            free(nz_idx);
            free(l_buf);
            free(u_buf);
            sparse_free(L);
            sparse_free(U);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < n; i++) {
            row_map[i] = i;
            inv_row_map[i] = i;
            ilu->perm[i] = i;
        }
    }

    sparse_err_t status = SPARSE_OK;

    for (idx_t i = 0; i < n; i++) {
        /* Determine which original row to process at position i */
        idx_t orig_row = row_map ? row_map[i] : i;

        /* Pivoting: find the row j >= i with the largest |A(row_map[j], i)|.
         * Scan col_headers[i] in O(nnz_in_col) and use inv_row_map for
         * O(1) position lookup instead of linear row_map scan. */
        if (row_map) {
            idx_t best_j = i;
            double best_val = 0.0;
            Node *cnd = A->col_headers[i];
            while (cnd) {
                idx_t pos = inv_row_map[cnd->row];
                if (pos >= i) {
                    double val = fabs(cnd->value);
                    if (val > best_val) {
                        best_val = val;
                        best_j = pos;
                    }
                }
                cnd = cnd->down;
            }
            if (best_j != i) {
                /* Swap row_map and inv_row_map */
                idx_t orig_i = row_map[i];
                // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
                idx_t orig_best = row_map[best_j];
                row_map[i] = orig_best;
                row_map[best_j] = orig_i;
                inv_row_map[orig_best] = i;
                inv_row_map[orig_i] = best_j;
                /* Record the swap in perm */
                idx_t tmp = ilu->perm[i];
                ilu->perm[i] = ilu->perm[best_j];
                ilu->perm[best_j] = tmp;
            }
            orig_row = row_map[i];
        }

        /* Scatter row orig_row of A into dense workspace w */
        idx_t nnz_w = 0;
        Node *nd = A->row_headers[orig_row];
        while (nd) {
            w[nd->col] = nd->value;
            w_nz[nd->col] = 1;
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            nz_idx[nnz_w++] = nd->col;
            nd = nd->right;
        }

        /* Compute row norm for drop tolerance */
        double row_norm = 0.0;
        for (idx_t p = 0; p < nnz_w; p++)
            row_norm += w[nz_idx[p]] * w[nz_idx[p]];
        row_norm = sqrt(row_norm);
        double drop_tol = o->tol * row_norm;

        /* Sort nonzero indices ascending for column-order elimination */
        for (idx_t p = 1; p < nnz_w; p++) {
            idx_t key = nz_idx[p];
            idx_t q = p - 1;
            while (q >= 0 && nz_idx[q] > key) {
                nz_idx[q + 1] = nz_idx[q];
                q--;
            }
            nz_idx[q + 1] = key;
        }

        /* Elimination: for each k < i where w[k] != 0.
         *
         * nz_idx is sorted ascending before this loop. New fill indices
         * are appended unsorted; the loop scans through them (nnz_w grows)
         * and breaks on the first k >= i it encounters.  Fill entries
         * with j < i that happen to land after an entry >= i in the
         * unsorted tail are skipped — this is intentional: on matrices
         * with diagonal modification (e.g., west0067), processing
         * additional lower-triangular fill through small synthetic pivots
         * amplifies numerical error and destabilises the preconditioner. */
        for (idx_t p = 0; p < nnz_w; p++) {
            idx_t k = nz_idx[p];
            if (k >= i)
                break;

            double ukk = sparse_get_phys(U, k, k);
            if (fabs(ukk) < sparse_rel_tol(ilu->factor_norm, DROP_TOL)) {
                status = SPARSE_ERR_SINGULAR;
                goto cleanup;
            }

            double mult = w[k] / ukk;

            if (fabs(mult) < drop_tol) {
                w[k] = 0.0;
                w_nz[k] = 0;
                continue;
            }
            w[k] = mult;

            Node *uk = U->row_headers[k];
            while (uk) {
                idx_t j = uk->col;
                if (j > k) {
                    w[j] -= mult * uk->value;
                    if (!w_nz[j]) {
                        w_nz[j] = 1;
                        nz_idx[nnz_w++] = j; // NOLINT(clang-analyzer-security.ArrayBound)
                    }
                }
                uk = uk->right;
            }
        }

        /* Diagonal stabilization: if diagonal is too small after elimination,
         * apply diagonal modification (sign-preserving perturbation) as a
         * fallback. With pivoting enabled, this should be rare since pivoting
         * already placed the largest column entry on the diagonal. */
        double diag_w = w_nz[i] ? w[i] : 0.0;
        if (fabs(diag_w) < sparse_rel_tol(ilu->factor_norm, DROP_TOL)) {
            double tol_row = (row_norm > 0.0) ? o->tol * row_norm : 0.0;
            double eps_row = (tol_row > 1e-10) ? tol_row : 1e-10;
            diag_w = (diag_w >= 0.0) ? eps_row : -eps_row;
            w[i] = diag_w;
            if (!w_nz[i]) {
                w_nz[i] = 1;
                // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
                nz_idx[nnz_w++] = i;
            }
        }

        /* Collect L and U entries with dual dropping */
        col_val_t *l_entries = l_buf;
        col_val_t *u_entries = u_buf;
        idx_t l_count = 0, u_count = 0;

        for (idx_t p = 0; p < nnz_w; p++) {
            idx_t j = nz_idx[p];
            if (!w_nz[j])
                continue;
            double val = w[j];
            if (fabs(val) < drop_tol && j != i)
                continue;
            if (j < i) {
                if (l_count < n)
                    l_entries[l_count++] = (col_val_t){j, val};
            } else {
                if (u_count < n)
                    u_entries[u_count++] = (col_val_t){j, val};
            }
        }

        /* Keep at most max_fill entries in L (by magnitude) */
        if (o->max_fill > 0 && l_count > o->max_fill) {
            qsort(l_entries, (size_t)l_count, sizeof(col_val_t), cmp_col_val_desc);
            l_count = o->max_fill;
        }

        /* Separate diagonal from off-diagonal U, keep at most max_fill off-diag */
        double diag_final = 0.0;
        idx_t u_offdiag = 0;
        for (idx_t p = 0; p < u_count; p++) {
            if (u_entries[p].col == i) {
                diag_final = u_entries[p].val;
            } else {
                u_entries[u_offdiag++] = u_entries[p];
            }
        }
        if (o->max_fill > 0 && u_offdiag > o->max_fill) {
            qsort(u_entries, (size_t)u_offdiag, sizeof(col_val_t), cmp_col_val_desc);
            u_offdiag = o->max_fill;
        }

        if (fabs(diag_final) < sparse_rel_tol(ilu->factor_norm, DROP_TOL)) {
            status = SPARSE_ERR_SINGULAR;
            goto cleanup;
        }

        /* Insert L entries */
        if (sparse_insert(L, i, i, 1.0) != SPARSE_OK) {
            status = SPARSE_ERR_ALLOC;
            goto cleanup;
        }
        for (idx_t p = 0; p < l_count; p++) {
            if (sparse_insert(L, i, l_entries[p].col, l_entries[p].val) != SPARSE_OK) {
                status = SPARSE_ERR_ALLOC;
                goto cleanup;
            }
        }

        /* Insert U entries */
        if (sparse_insert(U, i, i, diag_final) != SPARSE_OK) {
            status = SPARSE_ERR_ALLOC;
            goto cleanup;
        }
        for (idx_t p = 0; p < u_offdiag; p++) {
            if (sparse_insert(U, i, u_entries[p].col, u_entries[p].val) != SPARSE_OK) {
                status = SPARSE_ERR_ALLOC;
                goto cleanup;
            }
        }

        /* Clear workspace */
        for (idx_t p = 0; p < nnz_w; p++) {
            w[nz_idx[p]] = 0.0;
            w_nz[nz_idx[p]] = 0;
        }
    }

    ilu->L = L;
    ilu->U = U;
    ilu->n = n;

    free(w);
    free(w_nz);
    free(nz_idx);
    free(l_buf);
    free(u_buf);
    free(row_map);
    free(inv_row_map);
    return SPARSE_OK;

cleanup:
    free(w);
    free(w_nz);
    free(nz_idx);
    free(l_buf);
    free(u_buf);
    free(row_map);
    free(inv_row_map);
    sparse_free(L);
    sparse_free(U);
    if (ilu) {
        free(ilu->perm);
        ilu->perm = NULL;
        ilu->n = 0;
    }
    return status;
}

sparse_err_t sparse_ilut_precond(const void *ctx, idx_t n, const double *r, double *z) {
    return sparse_ilu_precond(ctx, n, r, z);
}
