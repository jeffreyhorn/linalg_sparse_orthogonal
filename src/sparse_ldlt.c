#include "sparse_ldlt.h"
#include "sparse_analysis.h"
#include "sparse_ldlt_csc_internal.h"
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
    ldlt->tol = 0.0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers for Bunch-Kaufman pivoting
 * ═══════════════════════════════════════════════════════════════════════ */

static idx_t count_row_nnz(const SparseMatrix *M, idx_t row) {
    idx_t count = 0;
    for (Node *nd = M->row_headers[row]; nd; nd = nd->right)
        count++;
    return count;
}

/* Symmetric row/column swap: P*M*P^T where P transposes indices p and q.
 * M must be symmetric.  Tracks row nonzeros explicitly so work is
 * proportional to the swapped rows/columns rather than all n columns. */
static sparse_err_t swap_sym_rc(SparseMatrix *M, idx_t p, idx_t q) {
    if (p == q)
        return SPARSE_OK;

    idx_t p_nnz = count_row_nnz(M, p);
    idx_t q_nnz = count_row_nnz(M, q);
    idx_t *p_cols = p_nnz ? malloc((size_t)p_nnz * sizeof(idx_t)) : NULL;
    idx_t *q_cols = q_nnz ? malloc((size_t)q_nnz * sizeof(idx_t)) : NULL;
    double *p_vals = p_nnz ? malloc((size_t)p_nnz * sizeof(double)) : NULL;
    double *q_vals = q_nnz ? malloc((size_t)q_nnz * sizeof(double)) : NULL;

    if ((p_nnz && (!p_cols || !p_vals)) || (q_nnz && (!q_cols || !q_vals))) {
        free(p_cols);
        free(q_cols);
        free(p_vals);
        free(q_vals);
        return SPARSE_ERR_ALLOC;
    }

    idx_t k = 0;
    for (Node *nd = M->row_headers[p]; nd; nd = nd->right) {
        p_cols[k] = nd->col; // NOLINT
        p_vals[k] = nd->value;
        k++;
    }
    k = 0;
    for (Node *nd = M->row_headers[q]; nd; nd = nd->right) {
        q_cols[k] = nd->col; // NOLINT
        q_vals[k] = nd->value;
        k++;
    }

    /* Remove rows p and q using only their stored nonzeros */
    for (idx_t i = 0; i < p_nnz; i++)
        sparse_remove(M, p, p_cols[i]); // NOLINT
    for (idx_t i = 0; i < q_nnz; i++)
        sparse_remove(M, q, q_cols[i]); // NOLINT

    /* Remove remaining entries in columns p and q (rows != p,q) */
    while (M->col_headers[p])
        sparse_remove(M, M->col_headers[p]->row, p);
    while (M->col_headers[q])
        sparse_remove(M, M->col_headers[q]->row, q);

    /* Re-insert swapped rows */
    sparse_err_t rc = SPARSE_OK;
    for (idx_t i = 0; i < p_nnz && rc == SPARSE_OK; i++) {
        idx_t j = p_cols[i]; // NOLINT
        idx_t nj = (j == p) ? q : (j == q) ? p : j;
        rc = sparse_insert(M, q, nj, p_vals[i]);
    }
    for (idx_t i = 0; i < q_nnz && rc == SPARSE_OK; i++) {
        idx_t j = q_cols[i]; // NOLINT
        idx_t nj = (j == p) ? q : (j == q) ? p : j;
        rc = sparse_insert(M, p, nj, q_vals[i]);
    }

    /* Symmetric counterparts for rows != p,q */
    for (idx_t i = 0; i < p_nnz && rc == SPARSE_OK; i++) {
        idx_t row = p_cols[i]; // NOLINT
        if (row == p || row == q)
            continue;
        rc = sparse_insert(M, row, q, p_vals[i]);
    }
    for (idx_t i = 0; i < q_nnz && rc == SPARSE_OK; i++) {
        idx_t row = q_cols[i]; // NOLINT
        if (row == p || row == q)
            continue;
        rc = sparse_insert(M, row, p, q_vals[i]);
    }

    free(p_cols);
    free(q_cols);
    free(p_vals);
    free(q_vals);
    return rc;
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

    sparse_err_t rc = SPARSE_OK;
    for (idx_t i = 0; i < np && rc == SPARSE_OK; i++)
        rc = sparse_remove(L, p, cols_p[i]); // NOLINT
    for (idx_t i = 0; i < nq && rc == SPARSE_OK; i++)
        rc = sparse_remove(L, q, cols_q[i]); // NOLINT

    for (idx_t i = 0; i < np && rc == SPARSE_OK; i++)
        rc = sparse_insert(L, q, cols_p[i], vals_p[i]);
    for (idx_t i = 0; i < nq && rc == SPARSE_OK; i++)
        rc = sparse_insert(L, p, cols_q[i], vals_q[i]);

    free(cols_p);
    free(vals_p);
    free(cols_q);
    free(vals_q);
    return rc;
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
     *   acc[i] -= L(i,j)·D_off·L(col,j+1) + L(i,j+1)·D_off·L(col,j)
     *
     * Collect L(col, *) entries once via a single row scan to avoid
     * repeated O(nnz_in_row) sparse_get_phys() probes. */
    {
        /* Build a sparse map of L(col, j) for j < step_k by scanning row col */
        idx_t n_lc = 0;
        Node *lc = L->row_headers[col];
        while (lc) {
            if (lc->col < step_k)
                n_lc++;
            lc = lc->right;
        }
        idx_t *lc_cols = NULL;
        double *lc_vals = NULL;
        if (n_lc > 0) {
            lc_cols = malloc((size_t)n_lc * sizeof(idx_t));
            lc_vals = malloc((size_t)n_lc * sizeof(double));
            if (lc_cols && lc_vals) {
                idx_t ix = 0;
                lc = L->row_headers[col];
                while (lc) {
                    if (lc->col < step_k) {
                        lc_cols[ix] = lc->col;
                        lc_vals[ix] = lc->value;
                        ix++;
                    }
                    lc = lc->right;
                }
            } else {
                /* Alloc failed — fall back to sparse_get_phys() probes
                 * so cross-term corrections remain correct under OOM. */
                free(lc_cols);
                free(lc_vals);
                n_lc = -1; /* sentinel: use fallback path */
                lc_cols = NULL;
                lc_vals = NULL;
            }
        }
        /* Look up L(col, j) from cache or fallback to sparse_get_phys.
         * lc_cols is sorted, so we advance a cursor through the cache
         * instead of rescanning from the start for each 2x2 block. */
        idx_t lc_cursor = 0;
        for (idx_t j = 0; j < step_k;) {
            if (pivot_size[j] == 2) {
                double d_off = D_offdiag[j];
                if (d_off != 0.0) {
                    double l_cj = 0.0, l_cj1 = 0.0;
                    if (n_lc >= 0) {
                        /* Advance cursor past entries before j */
                        while (lc_cursor < n_lc && lc_cols[lc_cursor] < j)
                            lc_cursor++;
                        if (lc_cursor < n_lc && lc_cols[lc_cursor] == j)
                            l_cj = lc_vals[lc_cursor];
                        idx_t t2 = lc_cursor;
                        while (t2 < n_lc && lc_cols[t2] <= j)
                            t2++;
                        if (t2 < n_lc && lc_cols[t2] == j + 1)
                            l_cj1 = lc_vals[t2];
                    } else {
                        /* OOM fallback: probe directly */
                        l_cj = sparse_get_phys(L, col, j);
                        l_cj1 = sparse_get_phys(L, col, j + 1);
                    }
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
        free(lc_cols);
        free(lc_vals);
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

/* Internal factorization with caller-specified pivot tolerance.
 * user_tol <= 0 means use DROP_TOL (the compile-time default). */
static sparse_err_t ldlt_factor_internal(const SparseMatrix *A, sparse_ldlt_t *ldlt,
                                         double user_tol) {
    double tol = (user_tol > 0.0) ? user_tol : DROP_TOL;
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
    ldlt->tol = 0.0;
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
        for (idx_t i = 0; i < n; i++) {
            if ((rp && rp[i] != i) || (cp && cp[i] != i))
                return SPARSE_ERR_BADARG;
        }
    }

    /* Validate symmetry */
    if (!sparse_is_symmetric(A, 1e-12))
        return SPARSE_ERR_NOT_SPD;

    /* Compute ||A||_inf for relative tolerance */
    ldlt->factor_norm = sparse_norminf_const(A);
    ldlt->tol = tol;

    /* Overflow guard for n-sized allocations (double is the largest element) */
    if ((size_t)n > SIZE_MAX / sizeof(double)) {
        sparse_ldlt_free(ldlt);
        return SPARSE_ERR_ALLOC;
    }

    /* Allocate D, D_offdiag, pivot_size arrays */
    ldlt->D = calloc((size_t)n, sizeof(double));
    ldlt->D_offdiag = calloc((size_t)n, sizeof(double));
    ldlt->pivot_size = calloc((size_t)n, sizeof(int));
    if (!ldlt->D || !ldlt->D_offdiag || !ldlt->pivot_size) {
        sparse_ldlt_free(ldlt);
        return SPARSE_ERR_ALLOC;
    }

    /* Create working copy of A (full symmetric matrix) */
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
    double sing_tol = sparse_rel_tol(ldlt->factor_norm, tol);
    double alpha_bk = (1.0 + sqrt(17.0)) / 8.0;
    /* L entry magnitude bound: 1/(100·tol) ≈ 1e12 with default tol */
    double growth_bound = 1.0 / (100.0 * tol);

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
                if (fabs(l_ik) >= tol) {
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
                double det_tol = (bscale > 0.0) ? tol * bscale * bscale : sing_tol * sing_tol;
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
            double drop_2x2 = tol * (fabs(d11) + fabs(d22) + fabs(d21));

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

/* ─── Sprint 20 Day 5: CSC-path factor helper ────────────────────────── */

/* Factor `A_work` (already in its natural coordinate space — the
 * caller has applied any user-requested fill-reducing reorder
 * before passing it in) via the CSC supernodal pipeline.  Populates
 * `ldlt_out` with the factored state; the caller is responsible for
 * composing any outer reorder permutation onto `ldlt_out->perm` and
 * reporting `used_csc_path` telemetry.
 *
 * Workflow (the "Option D" two-pass pattern validated in Day 3):
 *
 *   1. Scalar pre-pass on heuristic CSC → F_pre with the Bunch-
 *      Kaufman permutation and pivot_size pattern.
 *   2. Symmetrically permute A_work by F_pre->perm → A_perm.  On
 *      the pre-permuted input BK will not swap again during the
 *      batched factor, so sym_L is complete.
 *   3. sparse_analyze(A_perm, LDLT, REORDER_NONE) → symbolic sym_L.
 *   4. ldlt_csc_from_sparse_with_analysis(A_perm, &an, &F_batched).
 *   5. Seed F_batched->pivot_size from F_pre.
 *   6. ldlt_csc_eliminate_supernodal(F_batched, min_size=2).  On
 *      SPARSE_ERR_BADARG (pivot-stability check tripped — can
 *      happen when numerical drift shifts a BK decision between
 *      the two passes) fall back to F_pre's scalar factor, which
 *      is already valid.  Either way we emit ldlt->perm =
 *      F_pre->perm below (the effective BK permutation).
 *   7. Writeback via ldlt_csc_writeback_to_ldlt.  Before writeback
 *      we overwrite the source factor's perm field with
 *      F_pre->perm so the public `sparse_ldlt_t.perm` carries the
 *      correct BK permutation regardless of which path succeeded. */
static sparse_err_t ldlt_factor_csc_path(const SparseMatrix *A_work, double tol,
                                         sparse_ldlt_t *ldlt_out) {
    if (A_work->rows != A_work->cols)
        return SPARSE_ERR_SHAPE;
    idx_t n = A_work->rows;

    /* Day 5 CSC path supports n >= 1 only.  The n == 0 edge case is
     * handled by the caller (falls through to the linked-list
     * path). */
    if (n == 0)
        return SPARSE_ERR_BADARG;

    /* Step 1: scalar pre-pass on heuristic CSC. */
    LdltCsc *F_pre = NULL;
    sparse_err_t err = ldlt_csc_from_sparse(A_work, NULL, 2.0, &F_pre);
    if (err != SPARSE_OK)
        return err;
    err = ldlt_csc_eliminate_native(F_pre);
    if (err != SPARSE_OK) {
        ldlt_csc_free(F_pre);
        return err;
    }

    /* Step 2: symmetric permutation A_perm = P · A_work · P^T. */
    SparseMatrix *A_perm = NULL;
    err = sparse_permute(A_work, F_pre->perm, F_pre->perm, &A_perm);
    if (err != SPARSE_OK) {
        ldlt_csc_free(F_pre);
        return err;
    }
    sparse_reset_perms(A_perm);

    /* Step 3: analyze the pre-permuted matrix. */
    sparse_analysis_opts_t an_opts = {SPARSE_FACTOR_LDLT, SPARSE_REORDER_NONE};
    sparse_analysis_t an = {0};
    err = sparse_analyze(A_perm, &an_opts, &an);
    if (err != SPARSE_OK) {
        sparse_analysis_free(&an);
        sparse_free(A_perm);
        ldlt_csc_free(F_pre);
        return err;
    }

    /* Step 4: build F_batched with full sym_L pre-allocation. */
    LdltCsc *F_batched = NULL;
    err = ldlt_csc_from_sparse_with_analysis(A_perm, &an, &F_batched);
    if (err != SPARSE_OK) {
        sparse_analysis_free(&an);
        sparse_free(A_perm);
        ldlt_csc_free(F_pre);
        return err;
    }

    /* Step 5: seed pivot_size from scalar pass so supernode
     * detection respects the 2x2-aware boundaries BK produced. */
    for (idx_t k = 0; k < n; k++)
        F_batched->pivot_size[k] = F_pre->pivot_size[k];

    /* Step 6: batched supernodal factor with structural fallback.
     * F_pre already holds a valid scalar factor by this point, so
     * ANY failure on the batched side (pivot-stability BADARG,
     * numerical tolerance SINGULAR tripped by drift between the
     * two passes, or other) falls back to F_pre.  Sprint 19's
     * --supernodal bench on bcsstk14 / s3rmt3m3 shows the batched
     * path can produce factors that either trip the singularity
     * threshold or produce garbage residuals on those matrices —
     * see the post-Sprint-19 NOTE in `bench_ldlt_csc.c`.  The
     * fallback always gives a correct factor; the telemetry flag
     * continues to report `used_csc_path = 1` because the CSC
     * kernel chain handled the factor end-to-end. */
    LdltCsc *source = NULL;
    err = ldlt_csc_eliminate_supernodal(F_batched, /*min_size=*/2);
    if (err == SPARSE_OK) {
        /* Batched numeric factor succeeded — F_batched->perm is
         * identity in A_perm's coordinate space.  Overwrite with
         * F_pre->perm so the public ldlt->perm carries the
         * effective BK permutation, matching the behaviour of
         * the linked-list path. */
        for (idx_t i = 0; i < n; i++)
            F_batched->perm[i] = F_pre->perm[i];
        source = F_batched;
    } else {
        /* Any batched failure → fall back to F_pre's scalar
         * factor.  The pivot-stability BADARG case is the
         * intended fast-path trip; SINGULAR and other codes are
         * numerical drift between the two passes, where F_pre's
         * factor remains valid. */
        source = F_pre;
    }

    /* Step 7: writeback.
     *
     * The CSC elimination pipeline (ldlt_csc_eliminate_scalar /
     * ldlt_csc_eliminate_supernodal) currently enforces an internal
     * tolerance floor of `SPARSE_DROP_TOL`.  Thread the caller-
     * provided `tol` into the public LDLT object's recorded
     * tolerance, but never below that CSC floor, so
     * `sparse_ldlt_solve` sees a tolerance that both reflects an
     * explicitly stricter caller request and remains compatible
     * with the factorization that actually ran.  This avoids
     * silently discarding `tol` when the CSC path is selected while
     * also preventing a user-supplied tolerance smaller than
     * `SPARSE_DROP_TOL` from recording a looser check than the CSC
     * kernels applied.  See the backend caveat documented on
     * `sparse_ldlt_opts_t::tol` in include/sparse_ldlt.h. */
    const double effective_tol = (tol > SPARSE_DROP_TOL) ? tol : SPARSE_DROP_TOL;
    err = ldlt_csc_writeback_to_ldlt(source, effective_tol, ldlt_out);

    ldlt_csc_free(F_batched);
    ldlt_csc_free(F_pre);
    sparse_analysis_free(&an);
    sparse_free(A_perm);
    return err;
}

/* ─── Public factor API (delegates to internal with default tol) ────── */

sparse_err_t sparse_ldlt_factor(const SparseMatrix *A, sparse_ldlt_t *ldlt) {
    return ldlt_factor_internal(A, ldlt, 0.0);
}

/* ─── Factor with options (reordering + tolerance) ────────────────── */

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
    ldlt->tol = 0.0;
    if (!A)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    const sparse_ldlt_opts_t defaults = {SPARSE_REORDER_NONE, 0.0, SPARSE_LDLT_BACKEND_AUTO, NULL};
    const sparse_ldlt_opts_t *o = opts ? opts : &defaults;

    /* Sprint 20 Days 4-5 dispatch: validate the backend selector and
     * decide whether to take the CSC supernodal path.
     *
     *   LINKED_LIST → existing linked-list kernel (pre-Sprint-20).
     *   CSC         → Day 5 CSC path unconditionally.
     *   AUTO        → CSC when n >= SPARSE_CSC_THRESHOLD, else
     *                 linked-list (matches the Sprint 18 Cholesky
     *                 dispatch heuristic).
     *
     * For n == 0 the CSC pipeline is undefined (scalar pre-pass
     * can't factor an empty matrix); fall through to linked-list
     * regardless of the selector. */
    if (o->backend != SPARSE_LDLT_BACKEND_AUTO && o->backend != SPARSE_LDLT_BACKEND_LINKED_LIST &&
        o->backend != SPARSE_LDLT_BACKEND_CSC)
        return SPARSE_ERR_BADARG;

    idx_t n = A->rows;

    int use_csc;
    switch (o->backend) {
    case SPARSE_LDLT_BACKEND_LINKED_LIST:
        use_csc = 0;
        break;
    case SPARSE_LDLT_BACKEND_CSC:
        use_csc = 1;
        break;
    case SPARSE_LDLT_BACKEND_AUTO:
    default:
        use_csc = (n >= SPARSE_CSC_THRESHOLD);
        break;
    }
    if (n == 0)
        use_csc = 0;

    /* Apply fill-reducing reordering if requested */
    if (o->reorder != SPARSE_REORDER_NONE && n > 1) {
        if ((size_t)n > SIZE_MAX / sizeof(idx_t))
            return SPARSE_ERR_ALLOC;
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

        /* Factor the permuted matrix — CSC supernodal pipeline or
         * linked-list kernel depending on the Day 5 dispatch. */
        if (use_csc)
            err = ldlt_factor_csc_path(PA, o->tol, ldlt);
        else
            err = ldlt_factor_internal(PA, ldlt, o->tol);
        sparse_free(PA);

        if (err != SPARSE_OK) {
            free(perm);
            return err;
        }

        /* Compose reorder permutation with pivot permutation.
         * ldlt->perm currently holds the Bunch-Kaufman pivot permutation
         * produced during factorization and may be non-identity because of
         * symmetric swaps.  Compose: final_perm[i] = reorder_perm[bk_perm[i]]. */
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

        if (o->used_csc_path)
            *o->used_csc_path = use_csc ? 1 : 0;
        return SPARSE_OK;
    }

    /* No reordering — delegate directly to the chosen kernel. */
    sparse_err_t err_factor =
        use_csc ? ldlt_factor_csc_path(A, o->tol, ldlt) : ldlt_factor_internal(A, ldlt, o->tol);
    if (err_factor == SPARSE_OK && o->used_csc_path)
        *o->used_csc_path = use_csc ? 1 : 0;
    return err_factor;
}

/* ═══════════════════════════════════════════════════════════════════════
 * LDL^T solve
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_ldlt_solve(const sparse_ldlt_t *ldlt, const double *b, double *x) {
    if (!ldlt || !b || !x)
        return SPARSE_ERR_NULL;

    idx_t n = ldlt->n;
    if (n == 0)
        return SPARSE_OK;

    if (!ldlt->L || !ldlt->D || !ldlt->pivot_size)
        return SPARSE_ERR_BADARG;

    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

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
     * Handle 1x1 and 2x2 pivot blocks.
     * Use ldlt->tol (effective tolerance from factorization) for consistency. */
    double solve_tol = (ldlt->tol > 0.0) ? ldlt->tol : DROP_TOL;
    double sing_tol = sparse_rel_tol(ldlt->factor_norm, solve_tol);
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
            /* Block-relative singularity check (matches factorization) */
            double bscale = fabs(d11) + fabs(d22) + fabs(d21);
            double det_tol = (bscale > 0.0) ? solve_tol * bscale * bscale : sing_tol * sing_tol;
            if (fabs(det) < det_tol) {
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

    idx_t n = ldlt->n;
    if (n == 0) {
        if (n_pos)
            *n_pos = 0;
        if (n_neg)
            *n_neg = 0;
        if (n_zero)
            *n_zero = 0;
        return SPARSE_OK;
    }

    if (!ldlt->D || !ldlt->pivot_size)
        return SPARSE_ERR_BADARG;

    idx_t pos = 0, neg = 0, zero = 0;

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

    idx_t n = ldlt->n;
    if (n == 0)
        return SPARSE_OK;

    if (A->rows != n || A->cols != n)
        return SPARSE_ERR_SHAPE;

    if (!ldlt->L || !ldlt->D || !ldlt->pivot_size)
        return SPARSE_ERR_BADARG;

    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

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

    idx_t n = ldlt->n;
    if (n == 0) {
        *condest = 0.0;
        return SPARSE_OK;
    }

    if (A->rows != n || A->cols != n)
        return SPARSE_ERR_SHAPE;

    if (!ldlt->L || !ldlt->D || !ldlt->pivot_size)
        return SPARSE_ERR_BADARG;

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
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

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
