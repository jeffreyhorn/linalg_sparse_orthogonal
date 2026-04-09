#include "sparse_ldlt.h"
#include "sparse_matrix_internal.h"
#include "sparse_reorder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Free
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_ldlt_free(sparse_ldlt_t *ldlt) {
    if (!ldlt)
        return;
    sparse_free(ldlt->L);
    free(ldlt->D);
    free(ldlt->D_offdiag);
    free(ldlt->pivot_size);
    free(ldlt->perm);
    ldlt->L = NULL;
    ldlt->D = NULL;
    ldlt->D_offdiag = NULL;
    ldlt->pivot_size = NULL;
    ldlt->perm = NULL;
    ldlt->n = 0;
    ldlt->factor_norm = 0.0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers for Bunch-Kaufman pivoting
 * ═══════════════════════════════════════════════════════════════════════ */

/* Symmetric row/column swap: P*M*P^T where P transposes indices p and q.
 * M must be symmetric.  Uses dense row collection + remove/reinsert. */
static sparse_err_t swap_sym_rc(SparseMatrix *M, idx_t p, idx_t q) {
    if (p == q)
        return SPARSE_OK;
    idx_t n = M->rows;

    double *rp = calloc((size_t)n, sizeof(double));
    double *rq = calloc((size_t)n, sizeof(double));
    if (!rp || !rq) {
        free(rp);
        free(rq);
        return SPARSE_ERR_ALLOC;
    }

    /* Collect rows p and q */
    for (Node *nd = M->row_headers[p]; nd; nd = nd->right)
        rp[nd->col] = nd->value;
    for (Node *nd = M->row_headers[q]; nd; nd = nd->right)
        rq[nd->col] = nd->value;

    /* Remove rows p and q */
    for (idx_t j = 0; j < n; j++) {
        if (rp[j] != 0.0)
            sparse_remove(M, p, j);
        if (rq[j] != 0.0)
            sparse_remove(M, q, j);
    }

    /* Remove remaining entries in columns p and q (rows != p,q) */
    while (M->col_headers[p])
        sparse_remove(M, M->col_headers[p]->row, p);
    while (M->col_headers[q])
        sparse_remove(M, M->col_headers[q]->row, q);

    /* Re-insert with swapped indices */
    for (idx_t j = 0; j < n; j++) {
        idx_t nj = (j == p) ? q : (j == q) ? p : j;
        if (rp[j] != 0.0)
            sparse_insert(M, q, nj, rp[j]);
        if (rq[j] != 0.0)
            sparse_insert(M, p, nj, rq[j]);
    }

    /* Symmetric counterparts for rows != p,q */
    for (idx_t i = 0; i < n; i++) {
        if (i == p || i == q)
            continue;
        if (rp[i] != 0.0)
            sparse_insert(M, i, q, rp[i]);
        if (rq[i] != 0.0)
            sparse_insert(M, i, p, rq[i]);
    }

    free(rp);
    free(rq);
    return SPARSE_OK;
}

/* Swap rows p and q of L for all columns j < max_col.
 * Diagonal entries (L(i,i) = 1) are not touched since p,q >= max_col. */
static sparse_err_t swap_L_rows(SparseMatrix *L, idx_t p, idx_t q, idx_t max_col) {
    if (p == q || max_col == 0)
        return SPARSE_OK;

    idx_t np = 0, nq = 0;
    for (Node *nd = L->row_headers[p]; nd; nd = nd->right)
        if (nd->col < max_col)
            np++;
    for (Node *nd = L->row_headers[q]; nd; nd = nd->right)
        if (nd->col < max_col)
            nq++;

    idx_t *cols_p = NULL, *cols_q = NULL;
    double *vals_p = NULL, *vals_q = NULL;

    if (np > 0) {
        cols_p = malloc((size_t)np * sizeof(idx_t));
        vals_p = malloc((size_t)np * sizeof(double));
        if (!cols_p || !vals_p) {
            free(cols_p);
            free(vals_p);
            return SPARSE_ERR_ALLOC;
        }
        idx_t ix = 0;
        for (Node *nd = L->row_headers[p]; nd; nd = nd->right)
            if (nd->col < max_col) {
                cols_p[ix] = nd->col; // NOLINT
                vals_p[ix] = nd->value;
                ix++;
            }
    }

    if (nq > 0) {
        cols_q = malloc((size_t)nq * sizeof(idx_t));
        vals_q = malloc((size_t)nq * sizeof(double));
        if (!cols_q || !vals_q) {
            free(cols_p);
            free(vals_p);
            free(cols_q);
            free(vals_q);
            return SPARSE_ERR_ALLOC;
        }
        idx_t ix = 0;
        for (Node *nd = L->row_headers[q]; nd; nd = nd->right)
            if (nd->col < max_col) {
                cols_q[ix] = nd->col; // NOLINT
                vals_q[ix] = nd->value;
                ix++;
            }
    }

    for (idx_t i = 0; i < np; i++)
        sparse_remove(L, p, cols_p[i]); // NOLINT
    for (idx_t i = 0; i < nq; i++)
        sparse_remove(L, q, cols_q[i]); // NOLINT

    for (idx_t i = 0; i < np; i++)
        sparse_insert(L, q, cols_p[i], vals_p[i]);
    for (idx_t i = 0; i < nq; i++)
        sparse_insert(L, p, cols_q[i], vals_q[i]);

    free(cols_p);
    free(vals_p);
    free(cols_q);
    free(vals_q);
    return SPARSE_OK;
}

/* Accumulate column `col` of the Schur complement at elimination step `step_k`.
 *
 *   S(i, col) = W(i, col) − Σ_{j<step_k} L(i,j)·D[j]·L(col,j)
 *             − cross-terms for 2×2 pivot blocks
 *
 * Fills acc[i] for rows i >= step_k.  Returns the number of nonzeros. */
static idx_t acc_schur_col(const SparseMatrix *W, const SparseMatrix *L, const double *D,
                           const double *D_offdiag, const int *pivot_size, idx_t col, idx_t step_k,
                           double *acc, int *flag, idx_t *list) {
    idx_t nnz = 0;

    /* Load column col of W (rows >= step_k) */
    Node *nd = W->col_headers[col];
    while (nd) {
        if (nd->row >= step_k) {
            acc[nd->row] = nd->value;
            if (!flag[nd->row]) {
                flag[nd->row] = 1;
                list[nnz++] = nd->row;
            }
        }
        nd = nd->down;
    }

    /* Subtract diagonal-D contributions: for j < step_k where L(col,j) ≠ 0 */
    Node *lk = L->row_headers[col];
    while (lk) {
        idx_t j = lk->col;
        if (j >= step_k) {
            lk = lk->right;
            continue;
        }
        double l_cj_dj = lk->value * D[j]; // NOLINT
        Node *lij = L->col_headers[j];
        while (lij) {
            if (lij->row >= step_k) {
                acc[lij->row] -= lij->value * l_cj_dj;
                if (!flag[lij->row]) {
                    flag[lij->row] = 1;
                    list[nnz++] = lij->row;
                }
            }
            lij = lij->down;
        }
        lk = lk->right;
    }

    /* Cross-term corrections for 2×2 pivot blocks.
     * For each 2×2 block at (j, j+1):
     *   acc[i] -= L(i,j)·D_off·L(col,j+1) + L(i,j+1)·D_off·L(col,j) */
    for (idx_t j = 0; j < step_k;) {
        if (pivot_size[j] == 2) {
            double d_off = D_offdiag[j];
            if (d_off != 0.0) {
                double l_cj = sparse_get_phys(L, col, j);
                double l_cj1 = sparse_get_phys(L, col, j + 1);
                double ct1 = d_off * l_cj1;
                double ct2 = d_off * l_cj;
                if (ct1 != 0.0) {
                    Node *lij = L->col_headers[j];
                    while (lij) {
                        if (lij->row >= step_k) {
                            acc[lij->row] -= lij->value * ct1;
                            if (!flag[lij->row]) {
                                flag[lij->row] = 1;
                                list[nnz++] = lij->row;
                            }
                        }
                        lij = lij->down;
                    }
                }
                if (ct2 != 0.0) {
                    Node *lij = L->col_headers[j + 1];
                    while (lij) {
                        if (lij->row >= step_k) {
                            acc[lij->row] -= lij->value * ct2;
                            if (!flag[lij->row]) {
                                flag[lij->row] = 1;
                                list[nnz++] = lij->row;
                            }
                        }
                        lij = lij->down;
                    }
                }
            }
            j += 2;
        } else {
            j++;
        }
    }
    return nnz;
}

/* Clear dense accumulator entries tracked by nz list */
static void clear_acc(double *acc, int *flag, const idx_t *list, idx_t nnz) {
    for (idx_t t = 0; t < nnz; t++) {
        acc[list[t]] = 0.0; // NOLINT
        flag[list[t]] = 0;  // NOLINT
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * LDL^T factorization
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ldlt_factor(const SparseMatrix *A, sparse_ldlt_t *ldlt) {
    if (!ldlt)
        return SPARSE_ERR_NULL;
    /* Zero-initialize output so sparse_ldlt_free() is safe on any error path */
    ldlt->L = NULL;
    ldlt->D = NULL;
    ldlt->D_offdiag = NULL;
    ldlt->pivot_size = NULL;
    ldlt->perm = NULL;
    ldlt->n = 0;
    ldlt->factor_norm = 0.0;
    if (!A)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;
    ldlt->n = n;

    /* Empty matrix: trivial factorization */
    if (n == 0)
        return SPARSE_OK;

    /* Reject matrices with non-identity permutations */
    if (A->factored)
        return SPARSE_ERR_BADARG;
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

    /* Validate symmetry */
    if (!sparse_is_symmetric(A, 1e-12))
        return SPARSE_ERR_NOT_SPD;

    /* Compute ||A||_inf for relative tolerance */
    ldlt->factor_norm = sparse_norminf_const(A);

    /* Allocate D, D_offdiag, pivot_size arrays */
    ldlt->D = calloc((size_t)n, sizeof(double));
    ldlt->D_offdiag = calloc((size_t)n, sizeof(double));
    ldlt->pivot_size = calloc((size_t)n, sizeof(int));
    if (!ldlt->D || !ldlt->D_offdiag || !ldlt->pivot_size) {
        sparse_ldlt_free(ldlt);
        return SPARSE_ERR_ALLOC;
    }

    /* Create working copy of A (lower triangle only) */
    SparseMatrix *W = sparse_copy(A);
    if (!W) {
        sparse_ldlt_free(ldlt);
        return SPARSE_ERR_ALLOC;
    }

    /* Create L as unit lower triangular (start with identity diagonal) */
    ldlt->L = sparse_create(n, n);
    if (!ldlt->L) {
        sparse_free(W);
        sparse_ldlt_free(ldlt);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < n; i++) {
        sparse_err_t ierr = sparse_insert(ldlt->L, i, i, 1.0);
        if (ierr != SPARSE_OK) {
            sparse_free(W);
            sparse_ldlt_free(ldlt);
            return ierr;
        }
    }

    /* Allocate pivot permutation (identity initially) */
    ldlt->perm = malloc((size_t)n * sizeof(idx_t));
    if (!ldlt->perm) {
        sparse_free(W);
        sparse_ldlt_free(ldlt);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < n; i++)
        ldlt->perm[i] = i;

    /* ── Bunch-Kaufman LDL^T elimination ─────────────────────────────────
     *
     * Left-looking symmetric elimination with Bunch-Kaufman pivoting.
     * Supports both 1×1 and 2×2 pivot blocks for symmetric indefinite
     * matrices.  The parameter α = (1+√17)/8 ≈ 0.6404 controls when a
     * 2×2 pivot is preferred over a 1×1 pivot.
     *
     * Element growth control: L entries exceeding GROWTH_BOUND indicate
     * numerical breakdown (near-singular pivot).  Bunch-Kaufman bounds
     * growth in exact arithmetic, but finite precision + drop tolerance
     * can still produce large L entries on ill-conditioned problems.
     */
    double sing_tol = sparse_rel_tol(ldlt->factor_norm, DROP_TOL);
    double alpha_bk = (1.0 + sqrt(17.0)) / 8.0;
    /* L entry magnitude bound: 1/(100·DROP_TOL) ≈ 1e12 with default tol */
    double growth_bound = 1.0 / (100.0 * DROP_TOL);

    /* Dense column accumulators (two sets: column k and column r) */
    double *col_acc = calloc((size_t)n, sizeof(double));
    int *nz_flag = calloc((size_t)n, sizeof(int));
    idx_t *nz_list = malloc((size_t)n * sizeof(idx_t));
    double *col_acc_r = calloc((size_t)n, sizeof(double));
    int *nz_flag_r = calloc((size_t)n, sizeof(int));
    idx_t *nz_list_r = malloc((size_t)n * sizeof(idx_t));
    if (!col_acc || !nz_flag || !nz_list || !col_acc_r || !nz_flag_r || !nz_list_r) {
        free(col_acc);
        free(nz_flag);
        free(nz_list);
        free(col_acc_r);
        free(nz_flag_r);
        free(nz_list_r);
        sparse_free(W);
        sparse_ldlt_free(ldlt);
        return SPARSE_ERR_ALLOC;
    }

    sparse_err_t rc = SPARSE_OK;
    idx_t k = 0;
    while (k < n) {
        /* Step 1: Accumulate column k of the Schur complement */
        idx_t nnz_acc = acc_schur_col(W, ldlt->L, ldlt->D, ldlt->D_offdiag, ldlt->pivot_size, k, k,
                                      col_acc, nz_flag, nz_list);

        /* Step 2: Bunch-Kaufman pivot decision */
        double max_offdiag = 0.0;
        idx_t r = k; /* sentinel: no off-diagonal found */
        for (idx_t t = 0; t < nnz_acc; t++) {
            idx_t i = nz_list[t];
            if (i > k && fabs(col_acc[i]) > max_offdiag) { // NOLINT
                max_offdiag = fabs(col_acc[i]);
                r = i;
            }
        }

        int use_2x2 = 0;
        idx_t nnz_r = 0;

        /* Try to avoid 1×1 pivot only when the diagonal is small relative
         * to the off-diagonal and a 2×2 pivot is geometrically possible. */
        if (max_offdiag > 0.0 && k + 1 < n && fabs(col_acc[k]) < alpha_bk * max_offdiag) {
            /* Need column r for Bunch-Kaufman criteria 2–4 */
            nnz_r = acc_schur_col(W, ldlt->L, ldlt->D, ldlt->D_offdiag, ldlt->pivot_size, r, k,
                                  col_acc_r, nz_flag_r, nz_list_r);

            /* σ_r = max |S(i,r)| for i ≥ k, i ≠ r */
            double sigma_r = 0.0;
            for (idx_t t = 0; t < nnz_r; t++) {
                idx_t i = nz_list_r[t];
                if (i >= k && i != r && fabs(col_acc_r[i]) > sigma_r)
                    sigma_r = fabs(col_acc_r[i]);
            }

            if (fabs(col_acc[k]) * sigma_r >= alpha_bk * max_offdiag * max_offdiag) {
                /* Criterion 2: 1×1 at (k,k) */
                clear_acc(col_acc_r, nz_flag_r, nz_list_r, nnz_r);
                nnz_r = 0;
            } else if (fabs(col_acc_r[r]) >= alpha_bk * sigma_r) {
                /* Criterion 3: 1×1 at (r,r) — swap k ↔ r */
                rc = swap_sym_rc(W, k, r);
                if (rc != SPARSE_OK)
                    goto err_cleanup;
                if (k > 0) {
                    rc = swap_L_rows(ldlt->L, k, r, k);
                    if (rc != SPARSE_OK)
                        goto err_cleanup;
                }
                { /* swap perm */
                    idx_t tmp = ldlt->perm[k];
                    ldlt->perm[k] = ldlt->perm[r];
                    ldlt->perm[r] = tmp;
                }
                /* Replace col_acc with col_acc_r remapped through k↔r */
                clear_acc(col_acc, nz_flag, nz_list, nnz_acc);
                nnz_acc = 0;
                for (idx_t t = 0; t < nnz_r; t++) {
                    idx_t i = nz_list_r[t];
                    idx_t m = (i == k) ? r : (i == r) ? k : i;
                    col_acc[m] = col_acc_r[i]; // NOLINT
                    if (!nz_flag[m]) {
                        nz_flag[m] = 1;
                        nz_list[nnz_acc++] = m; // NOLINT
                    }
                    col_acc_r[i] = 0.0;
                    nz_flag_r[i] = 0;
                }
                nnz_r = 0;
            } else {
                /* Criterion 4: 2×2 pivot at (k, r) */
                use_2x2 = 1;
                if (r != k + 1) {
                    rc = swap_sym_rc(W, k + 1, r);
                    if (rc != SPARSE_OK)
                        goto err_cleanup;
                    if (k > 0) {
                        rc = swap_L_rows(ldlt->L, k + 1, r, k);
                        if (rc != SPARSE_OK)
                            goto err_cleanup;
                    }
                    { /* swap perm */
                        idx_t tmp = ldlt->perm[k + 1];
                        ldlt->perm[k + 1] = ldlt->perm[r];
                        ldlt->perm[r] = tmp;
                    }
                    /* Swap entries k+1 ↔ r in both accumulators */
                    {
                        double tv;
                        tv = col_acc[k + 1];
                        col_acc[k + 1] = col_acc[r];
                        col_acc[r] = tv;
                        tv = col_acc_r[k + 1];
                        col_acc_r[k + 1] = col_acc_r[r];
                        col_acc_r[r] = tv;
                    }
                    /* Ensure nz tracking covers swapped positions */
                    if (col_acc[k + 1] != 0.0 && !nz_flag[k + 1]) {
                        nz_flag[k + 1] = 1;
                        nz_list[nnz_acc++] = k + 1; // NOLINT
                    }
                    if (col_acc[r] != 0.0 && !nz_flag[r]) {
                        nz_flag[r] = 1;
                        nz_list[nnz_acc++] = r; // NOLINT
                    }
                    if (col_acc_r[k + 1] != 0.0 && !nz_flag_r[k + 1]) {
                        nz_flag_r[k + 1] = 1;
                        nz_list_r[nnz_r++] = k + 1; // NOLINT
                    }
                    if (col_acc_r[r] != 0.0 && !nz_flag_r[r]) {
                        nz_flag_r[r] = 1;
                        nz_list_r[nnz_r++] = r; // NOLINT
                    }
                }
            }
        }

        if (!use_2x2) {
            /* ── 1×1 pivot ───────────────────────────────────────── */
            double dk = col_acc[k];

            if (fabs(dk) < sing_tol) {
                clear_acc(col_acc, nz_flag, nz_list, nnz_acc);
                rc = SPARSE_ERR_SINGULAR;
                goto err_cleanup;
            }

            ldlt->D[k] = dk;
            ldlt->D_offdiag[k] = 0.0;
            ldlt->pivot_size[k] = 1;

            for (idx_t t = 0; t < nnz_acc; t++) {
                idx_t i = nz_list[t];
                if (i <= k) {
                    col_acc[i] = 0.0;
                    nz_flag[i] = 0;
                    continue;
                }
                double l_ik = col_acc[i] / dk;
                /* Element growth check: very large L entries indicate
                 * near-singular pivot — numerical breakdown. */
                if (fabs(l_ik) > growth_bound) {
                    clear_acc(col_acc, nz_flag, nz_list, nnz_acc);
                    rc = SPARSE_ERR_SINGULAR;
                    goto err_cleanup;
                }
                if (fabs(l_ik) >= DROP_TOL) {
                    sparse_err_t ierr = sparse_insert(ldlt->L, i, k, l_ik);
                    if (ierr != SPARSE_OK) {
                        clear_acc(col_acc, nz_flag, nz_list, nnz_acc);
                        rc = ierr;
                        goto err_cleanup;
                    }
                }
                col_acc[i] = 0.0;
                nz_flag[i] = 0;
            }
            k++;
        } else {
            /* ── 2×2 pivot at (k, k+1) ──────────────────────────── */
            double d11 = col_acc[k];
            double d21 = col_acc[k + 1];
            double d22 = col_acc_r[k + 1];
            double det = d11 * d22 - d21 * d21;

            /* Block-relative singularity check: compare |det| against
             * the squared block scale rather than the global sing_tol².
             * This correctly handles Schur complement entries that have
             * grown or shrunk relative to ||A||_inf. */
            {
                double bscale = fabs(d11) + fabs(d22) + fabs(d21);
                double det_tol = (bscale > 0.0) ? DROP_TOL * bscale * bscale : sing_tol * sing_tol;
                if (fabs(det) < det_tol) {
                    clear_acc(col_acc, nz_flag, nz_list, nnz_acc);
                    clear_acc(col_acc_r, nz_flag_r, nz_list_r, nnz_r);
                    rc = SPARSE_ERR_SINGULAR;
                    goto err_cleanup;
                }
            }

            ldlt->D[k] = d11;
            ldlt->D[k + 1] = d22;
            ldlt->D_offdiag[k] = d21;
            ldlt->D_offdiag[k + 1] = 0.0;
            ldlt->pivot_size[k] = 2;
            ldlt->pivot_size[k + 1] = 2;

            double inv_det = 1.0 / det;
            double drop_2x2 = DROP_TOL * (fabs(d11) + fabs(d22) + fabs(d21));

            /* Merge nonzero sets so we visit every row that appears
             * in either accumulated column */
            for (idx_t t = 0; t < nnz_r; t++) {
                idx_t i = nz_list_r[t];
                if (!nz_flag[i]) { // NOLINT
                    nz_flag[i] = 1;
                    nz_list[nnz_acc++] = i; // NOLINT
                }
            }

            /* L(i,k) and L(i,k+1) for i > k+1 */
            for (idx_t t = 0; t < nnz_acc; t++) {
                idx_t i = nz_list[t];
                if (i <= k + 1) {
                    col_acc[i] = 0.0;
                    nz_flag[i] = 0;
                    continue;
                }
                double s_ik = col_acc[i];    // NOLINT
                double s_ik1 = col_acc_r[i]; // NOLINT
                double l_ik = (s_ik * d22 - s_ik1 * d21) * inv_det;
                double l_ik1 = (-s_ik * d21 + s_ik1 * d11) * inv_det;

                /* Element growth check for 2×2 pivot L entries */
                if (fabs(l_ik) > growth_bound || fabs(l_ik1) > growth_bound) {
                    rc = SPARSE_ERR_SINGULAR;
                    goto err_cleanup;
                }

                if (fabs(l_ik) >= drop_2x2) {
                    sparse_err_t ierr = sparse_insert(ldlt->L, i, k, l_ik);
                    if (ierr != SPARSE_OK) {
                        rc = ierr;
                        goto err_cleanup;
                    }
                }
                if (fabs(l_ik1) >= drop_2x2) {
                    sparse_err_t ierr = sparse_insert(ldlt->L, i, k + 1, l_ik1);
                    if (ierr != SPARSE_OK) {
                        rc = ierr;
                        goto err_cleanup;
                    }
                }
                col_acc[i] = 0.0;
                nz_flag[i] = 0;
            }

            /* Clean up col_acc_r */
            for (idx_t t = 0; t < nnz_r; t++) {
                col_acc_r[nz_list_r[t]] = 0.0;
                nz_flag_r[nz_list_r[t]] = 0;
            }

            k += 2;
        }
    }

    free(col_acc);
    free(nz_flag);
    free(nz_list);
    free(col_acc_r);
    free(nz_flag_r);
    free(nz_list_r);
    sparse_free(W);
    return SPARSE_OK;

err_cleanup:
    free(col_acc);
    free(nz_flag);
    free(nz_list);
    free(col_acc_r);
    free(nz_flag_r);
    free(nz_list_r);
    sparse_free(W);
    sparse_ldlt_free(ldlt);
    return rc;
}

/* ─── Factor with options (reordering) ─────────────────────────────── */

sparse_err_t sparse_ldlt_factor_opts(const SparseMatrix *A, const sparse_ldlt_opts_t *opts,
                                     sparse_ldlt_t *ldlt) {
    if (!ldlt)
        return SPARSE_ERR_NULL;
    ldlt->L = NULL;
    ldlt->D = NULL;
    ldlt->D_offdiag = NULL;
    ldlt->pivot_size = NULL;
    ldlt->perm = NULL;
    ldlt->n = 0;
    ldlt->factor_norm = 0.0;
    if (!A)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    const sparse_ldlt_opts_t defaults = {SPARSE_REORDER_NONE, 0.0};
    const sparse_ldlt_opts_t *o = opts ? opts : &defaults;

    idx_t n = A->rows;

    /* Apply fill-reducing reordering if requested */
    if (o->reorder != SPARSE_REORDER_NONE && n > 1) {
        idx_t *perm = malloc((size_t)n * sizeof(idx_t));
        if (!perm)
            return SPARSE_ERR_ALLOC;

        sparse_err_t err;
        switch (o->reorder) {
        case SPARSE_REORDER_RCM:
            err = sparse_reorder_rcm(A, perm);
            break;
        case SPARSE_REORDER_AMD:
            err = sparse_reorder_amd(A, perm);
            break;
        default:
            free(perm);
            return SPARSE_ERR_BADARG;
        }

        if (err != SPARSE_OK) {
            free(perm);
            return err;
        }

        /* Apply symmetric permutation P*A*P^T */
        SparseMatrix *PA = NULL;
        err = sparse_permute(A, perm, perm, &PA);
        if (err != SPARSE_OK) {
            free(perm);
            return err;
        }

        /* Reset permutations on PA to identity so factor accepts it */
        sparse_reset_perms(PA);

        /* Factor the permuted matrix */
        err = sparse_ldlt_factor(PA, ldlt);
        sparse_free(PA);

        if (err != SPARSE_OK) {
            free(perm);
            return err;
        }

        /* Compose reorder permutation with pivot permutation.
         * ldlt->perm currently holds the Bunch-Kaufman pivot perm (identity
         * for now).  Compose: final_perm[i] = reorder_perm[bk_perm[i]]. */
        idx_t *composed = malloc((size_t)n * sizeof(idx_t));
        if (!composed) {
            free(perm);
            sparse_ldlt_free(ldlt);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < n; i++)
            composed[i] = perm[ldlt->perm[i]]; // NOLINT
        free(ldlt->perm);
        ldlt->perm = composed;
        free(perm);

        return SPARSE_OK;
    }

    /* No reordering — delegate directly */
    return sparse_ldlt_factor(A, ldlt);
}

/* ═══════════════════════════════════════════════════════════════════════
 * LDL^T solve
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ldlt_solve(const sparse_ldlt_t *ldlt, const double *b, double *x) {
    if (!ldlt || !b || !x)
        return SPARSE_ERR_NULL;
    if (!ldlt->L || !ldlt->D || !ldlt->pivot_size)
        return SPARSE_ERR_BADARG;

    idx_t n = ldlt->n;
    if (n == 0)
        return SPARSE_OK;

    double *y = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double));
    if (!y || !z) {
        free(y);
        free(z);
        return SPARSE_ERR_ALLOC;
    }

    /* Phase 0: Apply permutation — y[i] = b[perm[i]] */
    if (ldlt->perm) {
        for (idx_t i = 0; i < n; i++)
            y[i] = b[ldlt->perm[i]];
    } else {
        memcpy(y, b, (size_t)n * sizeof(double));
    }

    /* Phase 1: Forward substitution — L * w = y (L is unit lower triangular)
     * w[i] = y[i] - sum_{j<i} L(i,j) * w[j] */
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        Node *node = ldlt->L->row_headers[i];
        while (node) {
            if (node->col < i)
                sum += node->value * y[node->col]; // NOLINT
            node = node->right;
        }
        y[i] -= sum; /* y is now w */
    }

    /* Phase 2: Diagonal solve — D * z = w
     * Handle 1x1 and 2x2 pivot blocks */
    double sing_tol = sparse_rel_tol(ldlt->factor_norm, DROP_TOL);
    for (idx_t k = 0; k < n;) {
        if (ldlt->pivot_size[k] == 1) {
            if (fabs(ldlt->D[k]) < sing_tol) {
                free(y);
                free(z);
                return SPARSE_ERR_SINGULAR;
            }
            z[k] = y[k] / ldlt->D[k];
            k++;
        } else {
            /* 2x2 block: [D[k] D_off; D_off D[k+1]] */
            double d11 = ldlt->D[k];
            double d22 = ldlt->D[k + 1];
            double d21 = ldlt->D_offdiag[k];
            double det = d11 * d22 - d21 * d21;
            if (fabs(det) < sing_tol * sing_tol) {
                free(y);
                free(z);
                return SPARSE_ERR_SINGULAR;
            }
            z[k] = (d22 * y[k] - d21 * y[k + 1]) / det;     // NOLINT
            z[k + 1] = (d11 * y[k + 1] - d21 * y[k]) / det; // NOLINT
            k += 2;
        }
    }

    /* Phase 3: Backward substitution — L^T * w = z
     * w[i] = z[i] - sum_{j>i} L(j,i) * w[j]
     * Walk column i of L for entries with row > i */
    for (idx_t i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        Node *node = ldlt->L->col_headers[i];
        while (node) {
            if (node->row > i)
                sum += node->value * z[node->row];
            node = node->down;
        }
        z[i] -= sum;
    }

    /* Phase 4: Apply inverse permutation — x[perm[i]] = z[i] */
    if (ldlt->perm) {
        for (idx_t i = 0; i < n; i++)
            x[ldlt->perm[i]] = z[i];
    } else {
        memcpy(x, z, (size_t)n * sizeof(double));
    }

    free(y);
    free(z);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Inertia
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ldlt_inertia(const sparse_ldlt_t *ldlt, idx_t *n_pos, idx_t *n_neg,
                                 idx_t *n_zero) {
    if (!ldlt)
        return SPARSE_ERR_NULL;
    if (!ldlt->D || !ldlt->pivot_size)
        return SPARSE_ERR_BADARG;

    idx_t pos = 0, neg = 0, zero = 0;
    idx_t n = ldlt->n;

    for (idx_t k = 0; k < n;) {
        if (ldlt->pivot_size[k] == 1) {
            if (ldlt->D[k] > 0.0)
                pos++;
            else if (ldlt->D[k] < 0.0)
                neg++;
            else
                zero++;
            k++;
        } else {
            /* 2x2 block: eigenvalues from det and trace */
            double d11 = ldlt->D[k];
            double d22 = ldlt->D[k + 1];
            double d21 = ldlt->D_offdiag[k];
            double det = d11 * d22 - d21 * d21;
            if (det < 0.0) {
                /* One positive, one negative eigenvalue */
                pos++;
                neg++;
            } else {
                double tr = d11 + d22;
                if (tr > 0.0)
                    pos += 2;
                else if (tr < 0.0)
                    neg += 2;
                else
                    zero += 2;
            }
            k += 2;
        }
    }

    if (n_pos)
        *n_pos = pos;
    if (n_neg)
        *n_neg = neg;
    if (n_zero)
        *n_zero = zero;
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Iterative refinement
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ldlt_refine(const SparseMatrix *A, const sparse_ldlt_t *ldlt, const double *b,
                                double *x, int max_iters, double tol) {
    if (!A || !ldlt || !b || !x)
        return SPARSE_ERR_NULL;
    if (!ldlt->L || !ldlt->D || !ldlt->pivot_size)
        return SPARSE_ERR_BADARG;

    idx_t n = ldlt->n;
    if (n == 0)
        return SPARSE_OK;

    double *r = malloc((size_t)n * sizeof(double));
    double *d = malloc((size_t)n * sizeof(double));
    if (!r || !d) {
        free(r);
        free(d);
        return SPARSE_ERR_ALLOC;
    }

    /* Compute ||b||_inf for relative residual */
    double norm_b = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ab = fabs(b[i]);
        if (ab > norm_b)
            norm_b = ab;
    }
    if (norm_b == 0.0)
        norm_b = 1.0;

    for (int iter = 0; iter < max_iters; iter++) {
        /* r = b - A*x */
        sparse_err_t err = sparse_matvec(A, x, r);
        if (err != SPARSE_OK) {
            free(r);
            free(d);
            return err;
        }

        double norm_r = 0.0;
        for (idx_t i = 0; i < n; i++) {
            r[i] = b[i] - r[i];
            double ar = fabs(r[i]);
            if (ar > norm_r)
                norm_r = ar;
        }

        if (norm_r / norm_b < tol)
            break;

        /* Solve A*d = r using existing LDL^T */
        err = sparse_ldlt_solve(ldlt, r, d);
        if (err != SPARSE_OK) {
            free(r);
            free(d);
            return err;
        }

        /* x += d */
        for (idx_t i = 0; i < n; i++)
            x[i] += d[i];
    }

    free(r);
    free(d);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Condition estimation
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ldlt_condest(const SparseMatrix *A, const sparse_ldlt_t *ldlt,
                                 double *condest) {
    if (!A || !ldlt || !condest)
        return SPARSE_ERR_NULL;
    if (!ldlt->L || !ldlt->D || !ldlt->pivot_size)
        return SPARSE_ERR_BADARG;

    idx_t n = ldlt->n;
    if (n == 0) {
        *condest = 0.0;
        return SPARSE_OK;
    }

    /* Compute ||A||_1 (max column sum of absolute values) */
    double norm_A = 0.0;
    for (idx_t j = 0; j < n; j++) {
        double col_sum = 0.0;
        Node *node = A->col_headers[j];
        while (node) {
            col_sum += fabs(node->value);
            node = node->down;
        }
        if (col_sum > norm_A)
            norm_A = col_sum;
    }
    if (norm_A == 0.0) {
        *condest = INFINITY;
        return SPARSE_OK;
    }

    /* Hager/Higham algorithm to estimate ||A^{-1}||_1.
     * Since A is symmetric, A^T = A and sparse_ldlt_solve works for both
     * forward and transpose solves. */
    double *x = malloc((size_t)n * sizeof(double));
    double *w = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double));
    if (!x || !w || !z) {
        free(x);
        free(w);
        free(z);
        return SPARSE_ERR_ALLOC;
    }

    double inv_n = 1.0 / (double)n;
    for (idx_t i = 0; i < n; i++)
        x[i] = inv_n;

    double est = 0.0;
    sparse_err_t err;

    for (int iter = 0; iter < 5; iter++) {
        /* Solve A*w = x */
        err = sparse_ldlt_solve(ldlt, x, w);
        if (err != SPARSE_OK) {
            free(x);
            free(w);
            free(z);
            return err;
        }

        /* ||w||_1 */
        double w_norm1 = 0.0;
        for (idx_t i = 0; i < n; i++)
            w_norm1 += fabs(w[i]);
        est = w_norm1;

        /* xi = sign(w) */
        for (idx_t i = 0; i < n; i++)
            x[i] = (w[i] >= 0.0) ? 1.0 : -1.0;

        /* Solve A^T*z = xi — for symmetric A, same as A*z = xi */
        err = sparse_ldlt_solve(ldlt, x, z);
        if (err != SPARSE_OK) {
            free(x);
            free(w);
            free(z);
            return err;
        }

        /* Check convergence: ||z||_inf <= z^T * w */
        double z_inf = 0.0;
        double zt_w = 0.0;
        idx_t j_max = 0;
        for (idx_t i = 0; i < n; i++) {
            double az = fabs(z[i]);
            if (az > z_inf) {
                z_inf = az;
                j_max = i;
            }
            zt_w += z[i] * w[i];
        }

        if (z_inf <= zt_w || iter == 4)
            break;

        /* x = e_{j_max} */
        for (idx_t i = 0; i < n; i++)
            x[i] = 0.0;
        x[j_max] = 1.0; // NOLINT
    }

    *condest = norm_A * est;

    free(x);
    free(w);
    free(z);
    return SPARSE_OK;
}
