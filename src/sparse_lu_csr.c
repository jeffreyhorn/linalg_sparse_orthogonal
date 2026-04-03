#include "sparse_lu_csr.h"
#include "sparse_matrix_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── Free ───────────────────────────────────────────────────────────── */

void lu_csr_free(LuCsr *csr) {
    if (!csr)
        return;
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->values);
    free(csr);
}

/* ─── Convert SparseMatrix → LuCsr (logical index order) ────────────── */

sparse_err_t lu_csr_from_sparse(const SparseMatrix *mat, double fill_factor, LuCsr **csr_out) {
    if (!csr_out)
        return SPARSE_ERR_NULL;
    *csr_out = NULL;
    if (!mat)
        return SPARSE_ERR_NULL;

    idx_t n = mat->rows;
    if (n != mat->cols)
        return SPARSE_ERR_SHAPE;

    /* Clamp fill factor */
    if (fill_factor < 1.0)
        fill_factor = 1.0;
    if (fill_factor > 20.0)
        fill_factor = 20.0;

    idx_t nnz = mat->nnz;
    idx_t cap = (idx_t)(nnz * fill_factor);
    if (cap < nnz)
        cap = nnz; /* overflow guard */
    if (cap < 1)
        cap = 1;

    LuCsr *csr = malloc(sizeof(LuCsr));
    if (!csr)
        return SPARSE_ERR_ALLOC;

    csr->n = n;
    csr->nnz = nnz;
    csr->capacity = cap;
    csr->row_ptr = malloc((size_t)(n + 1) * sizeof(idx_t));
    csr->col_idx = malloc((size_t)cap * sizeof(idx_t));
    csr->values = malloc((size_t)cap * sizeof(double));

    if (!csr->row_ptr || !csr->col_idx || !csr->values) {
        lu_csr_free(csr);
        return SPARSE_ERR_ALLOC;
    }

    /*
     * First pass: count nonzeros per logical row.
     * We traverse each physical row and map to logical row via inv_row_perm.
     */
    memset(csr->row_ptr, 0, (size_t)(n + 1) * sizeof(idx_t));

    for (idx_t phys_i = 0; phys_i < n; phys_i++) {
        idx_t log_i = mat->inv_row_perm[phys_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            (void)mat->inv_col_perm[node->col]; /* col mapping used in 2nd pass */
            csr->row_ptr[log_i + 1]++;
            node = node->right;
        }
    }

    /* Prefix sum to get row pointers */
    for (idx_t i = 0; i < n; i++)
        csr->row_ptr[i + 1] += csr->row_ptr[i];

    /*
     * Second pass: fill col_idx and values in logical order.
     * Use a temporary write-position array (reuse beginning of col_idx
     * would be tricky, so allocate a small temp array).
     */
    idx_t *write_pos = malloc((size_t)n * sizeof(idx_t));
    if (!write_pos) {
        lu_csr_free(csr);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < n; i++)
        write_pos[i] = csr->row_ptr[i];

    for (idx_t phys_i = 0; phys_i < n; phys_i++) {
        idx_t log_i = mat->inv_row_perm[phys_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            idx_t pos = write_pos[log_i]++;
            csr->col_idx[pos] = log_j;
            csr->values[pos] = node->value;
            node = node->right;
        }
    }
    free(write_pos);

    /*
     * Sort each row by logical column index. The linked-list rows are sorted
     * by physical column, but after applying inv_col_perm the logical columns
     * may be out of order (e.g., after complete pivoting).
     * Use insertion sort since rows are typically short and nearly sorted.
     */
    for (idx_t i = 0; i < n; i++) {
        idx_t start = csr->row_ptr[i];
        idx_t end = csr->row_ptr[i + 1];
        for (idx_t j = start + 1; j < end; j++) {
            idx_t key_col = csr->col_idx[j];
            double key_val = csr->values[j];
            idx_t k = j - 1;
            while (k >= start &&
                   csr->col_idx[k] > key_col) { // NOLINT(clang-analyzer-security.ArrayBound)
                csr->col_idx[k + 1] = csr->col_idx[k];
                csr->values[k + 1] = csr->values[k];
                k--;
            }
            csr->col_idx[k + 1] = key_col;
            csr->values[k + 1] = key_val;
        }
    }

    *csr_out = csr;
    return SPARSE_OK;
}

/* ─── Grow CSR arrays ────────────────────────────────────────────────── */

static sparse_err_t lu_csr_grow(LuCsr *csr, idx_t needed) {
    if (needed <= csr->capacity)
        return SPARSE_OK;
    /* Grow by at least 50% or to needed, whichever is larger */
    idx_t new_cap = csr->capacity + csr->capacity / 2;
    if (new_cap < needed)
        new_cap = needed;

    idx_t *new_col = realloc(csr->col_idx, (size_t)new_cap * sizeof(idx_t));
    if (!new_col)
        return SPARSE_ERR_ALLOC;
    csr->col_idx = new_col;

    double *new_val = realloc(csr->values, (size_t)new_cap * sizeof(double));
    if (!new_val)
        return SPARSE_ERR_ALLOC;
    csr->values = new_val;

    csr->capacity = new_cap;
    return SPARSE_OK;
}

/* ─── CSR LU elimination with scatter-gather ─────────────────────────── */

sparse_err_t lu_csr_eliminate(LuCsr *csr, double tol, double drop_tol, idx_t *piv_perm) {
    if (!csr)
        return SPARSE_ERR_NULL;

    idx_t n = csr->n;
    if (n == 0)
        return SPARSE_OK;

    /* Per-row start/end arrays — decoupled from row_ptr so that rewriting
     * one row doesn't corrupt its neighbors. */
    idx_t *rstart = malloc((size_t)n * sizeof(idx_t));
    idx_t *rend = malloc((size_t)n * sizeof(idx_t));
    /* Dense workspace for pivot row (scatter) */
    double *work = calloc((size_t)n, sizeof(double));
    /* Columns present in workspace (for efficient clear) */
    idx_t *work_cols = malloc((size_t)n * sizeof(idx_t));
    /* Row indirection: logical row i → storage slot row_map[i].
     * Swapping rows is O(1) via this indirection. */
    idx_t *row_map = malloc((size_t)n * sizeof(idx_t));

    if (!rstart || !rend || !work || !work_cols || !row_map) {
        free(rstart);
        free(rend);
        free(work);
        free(work_cols);
        free(row_map);
        return SPARSE_ERR_ALLOC;
    }

    /* Initialize per-row bounds from row_ptr */
    for (idx_t i = 0; i < n; i++) {
        rstart[i] = csr->row_ptr[i];
        rend[i] = csr->row_ptr[i + 1];
        row_map[i] = i;
    }

    if (piv_perm) {
        for (idx_t i = 0; i < n; i++)
            piv_perm[i] = i;
    }

    /* write_pos: append position for newly written rows */
    idx_t write_pos = csr->row_ptr[n];

    sparse_err_t err = SPARSE_OK;

    for (idx_t k = 0; k < n; k++) {
        /* ── Partial pivot: find largest |A[i,k]| for i in [k..n-1] ── */
        double max_val = 0.0;
        idx_t pivot_row = k;

        for (idx_t i = k; i < n; i++) {
            idx_t ri = row_map[i];
            idx_t s = rstart[ri], e = rend[ri];
            for (idx_t p = s; p < e; p++) {
                if (csr->col_idx[p] == k) {
                    double av = fabs(csr->values[p]);
                    if (av > max_val) {
                        max_val = av;
                        pivot_row = i;
                    }
                    break;
                }
                if (csr->col_idx[p] > k)
                    break;
            }
        }

        if (max_val < tol) {
            err = SPARSE_ERR_SINGULAR;
            goto cleanup;
        }

        /* ── Swap logical rows k ↔ pivot_row ── */
        if (pivot_row != k) {
            idx_t tmp = row_map[k];
            row_map[k] = row_map[pivot_row];
            row_map[pivot_row] = tmp;
            if (piv_perm) {
                idx_t ptmp = piv_perm[k];
                piv_perm[k] = piv_perm[pivot_row];
                piv_perm[pivot_row] = ptmp;
            }
        }

        /* ── Read pivot value ── */
        idx_t pk = row_map[k];
        double pivot_val = 0.0;
        {
            idx_t s = rstart[pk], e = rend[pk];
            for (idx_t p = s; p < e; p++) {
                if (csr->col_idx[p] == k) {
                    pivot_val = csr->values[p];
                    break;
                }
                if (csr->col_idx[p] > k)
                    break;
            }
        }

        /* ── Scatter pivot row into dense workspace ── */
        idx_t pk_s = rstart[pk], pk_e = rend[pk];
        idx_t wcount = 0;
        for (idx_t p = pk_s; p < pk_e; p++) {
            idx_t j = csr->col_idx[p];
            work[j] = csr->values[p];
            work_cols[wcount++] = j; // NOLINT(clang-analyzer-security.ArrayBound)
        }

        /* ── Elimination loop ── */
        for (idx_t i = k + 1; i < n; i++) {
            idx_t ri = row_map[i];
            idx_t ri_s = rstart[ri], ri_e = rend[ri];

            /* Find A[i,k] in row ri */
            double a_ik = 0.0;
            int found = 0;
            for (idx_t p = ri_s; p < ri_e; p++) {
                if (csr->col_idx[p] == k) {
                    a_ik = csr->values[p];
                    found = 1;
                    break;
                }
                if (csr->col_idx[p] > k)
                    break;
            }
            if (!found)
                continue;

            double mult = a_ik / pivot_val;

            /* Ensure capacity for worst-case fill (row_i_nnz + pivot_nnz) */
            idx_t ri_nnz = ri_e - ri_s;
            idx_t piv_nnz = pk_e - pk_s;
            err = lu_csr_grow(csr, write_pos + ri_nnz + piv_nnz);
            if (err != SPARSE_OK)
                goto cleanup;

            /*
             * Build the new row at write_pos by merging row_i and pivot_row.
             *
             * Columns < k  : copy from row_i (L entries from earlier steps)
             * Column  == k : store multiplier
             * Columns > k  : merge row_i and pivot_row with subtraction
             */
            idx_t new_start = write_pos;
            idx_t nc = 0; /* new entry count */
            idx_t pi = ri_s;

            /* Columns < k: copy from row_i as-is */
            while (pi < ri_e && csr->col_idx[pi] < k) {
                csr->col_idx[new_start + nc] = csr->col_idx[pi];
                csr->values[new_start + nc] = csr->values[pi];
                nc++;
                pi++;
            }

            /* Column k: store multiplier */
            csr->col_idx[new_start + nc] = k;
            csr->values[new_start + nc] = mult;
            nc++;

            /* Skip old column-k entry in row_i */
            if (pi < ri_e && csr->col_idx[pi] == k)
                pi++;

            /* Columns > k: merge row_i and pivot_row.
             * row_i entries are in col_idx[pi..ri_e), sorted.
             * pivot entries with col > k are in work[] (lookup by col). */
            idx_t pp = 0;
            while (pp < wcount && work_cols[pp] <= k)
                pp++;

            while (pi < ri_e || pp < wcount) {
                idx_t col_i = (pi < ri_e) ? csr->col_idx[pi] : n;
                idx_t col_p = (pp < wcount) ? work_cols[pp] : n;

                double new_val;
                idx_t col;

                if (col_i < col_p) {
                    col = col_i;
                    new_val = csr->values[pi];
                    pi++;
                } else if (col_p < col_i) {
                    col = col_p;
                    new_val = -mult * work[col_p]; // NOLINT(clang-analyzer-security.ArrayBound)
                    pp++;
                } else {
                    col = col_i;
                    new_val = csr->values[pi] -
                              mult * work[col_i]; // NOLINT(clang-analyzer-security.ArrayBound)
                    pi++;
                    pp++;
                }

                /* Drop small entries */
                if (fabs(new_val) < drop_tol * max_val)
                    continue;

                csr->col_idx[new_start + nc] = col;
                csr->values[new_start + nc] = new_val;
                nc++;
            }

            /* Update row bounds to point to newly written data */
            rstart[ri] = new_start;
            rend[ri] = new_start + nc;
            write_pos = new_start + nc;
        }

        /* ── Clear pivot row from workspace ── */
        for (idx_t c = 0; c < wcount; c++)
            work[work_cols[c]] = 0.0;
    }

    /* ── Compact: rebuild CSR arrays in logical row order ── */
    {
        idx_t total = 0;
        for (idx_t i = 0; i < n; i++) {
            idx_t ri = row_map[i];
            total += rend[ri] - rstart[ri];
        }

        idx_t *new_rp = malloc((size_t)(n + 1) * sizeof(idx_t));
        idx_t *new_ci = malloc((size_t)(total > 0 ? total : 1) * sizeof(idx_t));
        double *new_v = malloc((size_t)(total > 0 ? total : 1) * sizeof(double));
        if (!new_rp || !new_ci || !new_v) {
            free(new_rp);
            free(new_ci);
            free(new_v);
            err = SPARSE_ERR_ALLOC;
            goto cleanup;
        }

        idx_t pos = 0;
        for (idx_t i = 0; i < n; i++) {
            new_rp[i] = pos;
            idx_t ri = row_map[i];
            idx_t s = rstart[ri], e = rend[ri];
            for (idx_t p = s; p < e; p++) {
                new_ci[pos] = csr->col_idx[p];
                new_v[pos] = csr->values[p];
                pos++;
            }
        }
        new_rp[n] = pos; // NOLINT(clang-analyzer-security.ArrayBound)

        free(csr->row_ptr);
        free(csr->col_idx);
        free(csr->values);
        csr->row_ptr = new_rp;
        csr->col_idx = new_ci;
        csr->values = new_v;
        csr->nnz = total;
        csr->capacity = total > 0 ? total : 1;
    }

cleanup:
    free(rstart);
    free(rend);
    free(work);
    free(work_cols);
    free(row_map);
    return err;
}

/* ─── Convert LuCsr → SparseMatrix ──────────────────────────────────── */

sparse_err_t lu_csr_to_sparse(const LuCsr *csr, SparseMatrix **mat_out) {
    if (!mat_out)
        return SPARSE_ERR_NULL;
    *mat_out = NULL;
    if (!csr)
        return SPARSE_ERR_NULL;

    idx_t n = csr->n;
    SparseMatrix *mat = sparse_create(n, n);
    if (!mat)
        return SPARSE_ERR_ALLOC;

    for (idx_t i = 0; i < n; i++) {
        for (idx_t k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; k++) {
            /* Skip exact zeros (dropped entries) */
            if (csr->values[k] == 0.0)
                continue;
            sparse_err_t err = sparse_insert(mat, i, csr->col_idx[k], csr->values[k]);
            if (err != SPARSE_OK) {
                sparse_free(mat);
                return err;
            }
        }
    }

    *mat_out = mat;
    return SPARSE_OK;
}

/* ─── CSR forward/backward substitution ──────────────────────────────── */

sparse_err_t lu_csr_solve(const LuCsr *csr, const idx_t *piv_perm, const double *b, double *x) {
    if (!csr || !piv_perm || !b || !x)
        return SPARSE_ERR_NULL;

    idx_t n = csr->n;

    /* Step 1: Apply pivot permutation — pb[i] = b[piv_perm[i]] */
    double *y = malloc((size_t)n * sizeof(double));
    if (!y)
        return SPARSE_ERR_ALLOC;

    double *pb = malloc((size_t)n * sizeof(double));
    if (!pb) {
        free(y);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++)
        pb[i] = b[piv_perm[i]];

    /* Step 2: Forward substitution — L*y = pb (L has unit diagonal) */
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (idx_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; p++) {
            idx_t j = csr->col_idx[p];
            if (j < i)
                sum += csr->values[p] * y[j]; // NOLINT(clang-analyzer-security.ArrayBound)
        }
        y[i] = pb[i] - sum;
    }

    /* Step 3: Backward substitution — U*x = y */
    for (idx_t i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        double u_ii = 0.0;
        for (idx_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; p++) {
            idx_t j = csr->col_idx[p];
            if (j == i)
                u_ii = csr->values[p];
            else if (j > i)
                sum += csr->values[p] * x[j];
        }
        if (fabs(u_ii) < 1e-300) {
            free(y);
            free(pb);
            return SPARSE_ERR_SINGULAR;
        }
        x[i] = (y[i] - sum) / u_ii; // NOLINT(clang-analyzer-core.UndefinedBinaryOperatorResult)
    }

    free(y);
    free(pb);
    return SPARSE_OK;
}

/* ─── One-shot CSR factor + solve ────────────────────────────────────── */

sparse_err_t lu_csr_factor_solve(const SparseMatrix *mat, const double *b, double *x, double tol) {
    if (!mat || !b || !x)
        return SPARSE_ERR_NULL;

    idx_t n = sparse_rows(mat);

    /* Convert to CSR */
    LuCsr *csr = NULL;
    sparse_err_t err = lu_csr_from_sparse(mat, 3.0, &csr);
    if (err != SPARSE_OK)
        return err;

    /* Factor */
    idx_t *piv = malloc((size_t)n * sizeof(idx_t));
    if (!piv) {
        lu_csr_free(csr);
        return SPARSE_ERR_ALLOC;
    }

    err = lu_csr_eliminate(csr, tol, 1e-14, piv);
    if (err != SPARSE_OK) {
        free(piv);
        lu_csr_free(csr);
        return err;
    }

    /* Solve */
    err = lu_csr_solve(csr, piv, b, x);

    free(piv);
    lu_csr_free(csr);
    return err;
}
