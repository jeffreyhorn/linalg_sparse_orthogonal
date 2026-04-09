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
    /* Compute capacity in double, clamp to INT32_MAX to avoid UB on cast */
    double cap_d = (double)nnz * fill_factor;
    if (cap_d > (double)INT32_MAX)
        cap_d = (double)INT32_MAX;
    idx_t cap = (idx_t)cap_d;
    if (cap < nnz)
        cap = nnz;
    if (cap < 1)
        cap = 1;

    LuCsr *csr = malloc(sizeof(LuCsr));
    if (!csr)
        return SPARSE_ERR_ALLOC;

    csr->n = n;
    csr->nnz = nnz;
    csr->capacity = cap;
    /* Compute ||A||_inf for relative tolerance in solve paths.
     * Use const-safe helper to avoid mutating the caller's matrix. */
    csr->factor_norm = sparse_norminf_const(mat);
    /* Overflow guard for allocation byte counts */
    if ((size_t)cap > SIZE_MAX / sizeof(double) || (size_t)(n + 1) > SIZE_MAX / sizeof(idx_t)) {
        free(csr);
        return SPARSE_ERR_ALLOC;
    }
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
    /* Reject if needed exceeds idx_t range */
    if (needed > INT32_MAX)
        return SPARSE_ERR_ALLOC;
    /* Grow by at least 50% or to needed, whichever is larger.
     * Guard against idx_t overflow in the addition. */
    idx_t new_cap;
    if (csr->capacity > INT32_MAX - csr->capacity / 2)
        new_cap = INT32_MAX;
    else
        new_cap = csr->capacity + csr->capacity / 2;
    if (new_cap < needed)
        new_cap = needed;
    /* Guard size_t overflow for realloc byte count */
    if ((size_t)new_cap > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

    idx_t *new_col = realloc(csr->col_idx, (size_t)new_cap * sizeof(idx_t));
    if (!new_col)
        return SPARSE_ERR_ALLOC;
    /* Don't update csr->col_idx yet — if values realloc fails we need to
     * keep the struct consistent.  new_col is valid either way (realloc
     * frees the old block on success), so stash it and commit both below. */

    double *new_val = realloc(csr->values, (size_t)new_cap * sizeof(double));
    if (!new_val) {
        /* col_idx was already reallocated (old pointer freed by realloc),
         * so we must still record new_col to avoid a dangling pointer. */
        csr->col_idx = new_col;
        return SPARSE_ERR_ALLOC;
    }

    csr->col_idx = new_col;
    csr->values = new_val;
    csr->capacity = new_cap;
    return SPARSE_OK;
}

/* ─── Dense LU factorization (dgetrf-style) ──────────────────────────── */

sparse_err_t lu_dense_factor(idx_t m, idx_t n, double *A, idx_t lda, idx_t *ipiv, double tol) {
    if (!A || !ipiv)
        return SPARSE_ERR_NULL;
    if (m < 0 || n < 0 || lda < m)
        return SPARSE_ERR_BADARG;

    idx_t mn = (m < n) ? m : n;

    for (idx_t k = 0; k < mn; k++) {
        /* Find pivot: largest |A[i,k]| for i in [k..m-1] */
        double max_val = 0.0;
        idx_t pivot_row = k;
        for (idx_t i = k; i < m; i++) {
            double av = fabs(A[i + lda * k]); /* column-major: A[i,k] = A[i + lda*k] */
            if (av > max_val) {
                max_val = av;
                pivot_row = i;
            }
        }

        if (max_val < tol)
            return SPARSE_ERR_SINGULAR;

        ipiv[k] = pivot_row;

        /* Swap rows k and pivot_row across all columns */
        if (pivot_row != k) {
            for (idx_t j = 0; j < n; j++) {
                double tmp = A[k + lda * j];
                A[k + lda * j] = A[pivot_row + lda * j];
                A[pivot_row + lda * j] = tmp;
            }
        }

        /* Eliminate: for each row i > k */
        double pivot_val = A[k + lda * k];
        for (idx_t i = k + 1; i < m; i++) {
            double mult = A[i + lda * k] / pivot_val;
            A[i + lda * k] = mult; /* Store L entry */
            for (idx_t j = k + 1; j < n; j++) {
                A[i + lda * j] -= mult * A[k + lda * j];
            }
        }
    }

    return SPARSE_OK;
}

/* ─── Dense triangular solve ─────────────────────────────────────────── */

sparse_err_t lu_dense_solve(idx_t n, const double *LU, idx_t lda, const idx_t *ipiv, double *b) {
    if (!LU || !ipiv || !b)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n)
        return SPARSE_ERR_BADARG;

    /* Apply row permutation */
    for (idx_t k = 0; k < n; k++) {
        if (ipiv[k] != k) {
            double tmp = b[k];
            b[k] = b[ipiv[k]];
            b[ipiv[k]] = tmp;
        }
    }

    /* Forward substitution: L*y = Pb (unit diagonal) */
    for (idx_t i = 1; i < n; i++) {
        double sum = 0.0;
        for (idx_t j = 0; j < i; j++)
            sum += LU[i + lda * j] * b[j];
        b[i] -= sum;
    }

    /* Compute infinity norm of dense LU for relative tolerance */
    double lu_norm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (idx_t j = 0; j < n; j++)
            row_sum += fabs(LU[i + lda * j]);
        if (row_sum > lu_norm)
            lu_norm = row_sum;
    }
    double sing_tol = sparse_rel_tol(lu_norm, DROP_TOL);

    /* Backward substitution: U*x = y */
    for (idx_t i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (idx_t j = i + 1; j < n; j++)
            sum += LU[i + lda * j] * b[j];
        double u_ii = LU[i + lda * i];
        if (fabs(u_ii) < sing_tol)
            return SPARSE_ERR_SINGULAR;
        b[i] = (b[i] - sum) / u_ii;
    }

    return SPARSE_OK;
}

/* ─── CSR structural validation ──────────────────────────────────────── */

/**
 * Validate LuCsr structural invariants:
 * - row_ptr[0] == 0
 * - row_ptr is monotone non-decreasing
 * - row_ptr[n] == csr->nnz
 * - csr->nnz <= csr->capacity
 * - all col_idx values in [0, n)
 */
static sparse_err_t lu_csr_validate(const LuCsr *csr) {
    idx_t n = csr->n;
    if (csr->nnz < 0 || csr->capacity < 0 || csr->nnz > csr->capacity)
        return SPARSE_ERR_BADARG;
    if (!csr->row_ptr || !csr->col_idx || !csr->values)
        return SPARSE_ERR_NULL;
    if (csr->row_ptr[0] != 0)
        return SPARSE_ERR_BADARG;
    for (idx_t i = 0; i < n; i++) {
        if (csr->row_ptr[i] < 0 || csr->row_ptr[i] > csr->row_ptr[i + 1])
            return SPARSE_ERR_BADARG;
    }
    if (csr->row_ptr[n] != csr->nnz)
        return SPARSE_ERR_BADARG;
    for (idx_t p = 0; p < csr->nnz; p++) {
        if (csr->col_idx[p] < 0 || csr->col_idx[p] >= n)
            return SPARSE_ERR_BADARG;
    }
    return SPARSE_OK;
}

/* ─── CSR LU elimination with scatter-gather ─────────────────────────── */

sparse_err_t lu_csr_eliminate(LuCsr *csr, double tol, double drop_tol, idx_t *piv_perm) {
    if (!csr)
        return SPARSE_ERR_NULL;

    idx_t n = csr->n;
    if (n == 0)
        return SPARSE_OK;

    /* Overflow guard: ensure n * sizeof(largest_type) fits in size_t */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

    /* Validate CSR structure */
    sparse_err_t verr = lu_csr_validate(csr);
    if (verr != SPARSE_OK)
        return verr;

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
            size_t needed = (size_t)write_pos + (size_t)ri_nnz + (size_t)piv_nnz;
            if (needed > (size_t)INT32_MAX) {
                err = SPARSE_ERR_ALLOC;
                goto cleanup;
            }
            err = lu_csr_grow(csr, (idx_t)needed);
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

/* ─── Block-aware CSR LU elimination ──────────────────────────────────── */

/*
 * Single-pass block-aware elimination. Detects dense diagonal blocks
 * upfront. During elimination, when step k enters a diagonal block, the
 * block is extracted, factored densely, and the L\U result is written
 * back. Steps inside the block are then skipped. Steps outside blocks
 * use the standard sparse scatter-gather path.
 *
 * The Schur complement update for rows outside the block is handled by
 * the sparse elimination when it reaches those rows.
 */
sparse_err_t lu_csr_eliminate_block(LuCsr *csr, double tol, double drop_tol, idx_t min_block,
                                    idx_t *piv_perm) {
    if (!csr)
        return SPARSE_ERR_NULL;

    idx_t n = csr->n;
    if (n == 0)
        return SPARSE_OK;

    /* Validate CSR structure */
    sparse_err_t verr = lu_csr_validate(csr);
    if (verr != SPARSE_OK)
        return verr;

    /* Detect dense diagonal blocks */
    DenseBlock *blks = NULL;
    idx_t nblks = 0;
    sparse_err_t err = lu_detect_dense_blocks(csr, min_block, 0.8, &blks, &nblks);
    if (err != SPARSE_OK)
        return err;

    /* Overflow guard for n-sized allocations (shared by block_at, rstart,
     * rend, work, work_cols, row_map — double is the largest element type) */
    if ((size_t)n > SIZE_MAX / sizeof(double)) {
        free(blks);
        return SPARSE_ERR_ALLOC;
    }

    /* Build a lookup: for each step k, which block (if any) starts there?
     * block_at[k] = block index if a diagonal block starts at step k, else -1. */
    idx_t *block_at = calloc((size_t)n, sizeof(idx_t));
    if (!block_at) {
        free(blks);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < n; i++)
        block_at[i] = -1;

    for (idx_t b = 0; b < nblks; b++) {
        if (blks[b].row_start == blks[b].col_start &&
            blks[b].row_end - blks[b].row_start == blks[b].col_end - blks[b].col_start &&
            blks[b].row_end <= n) {
            idx_t start = blks[b].row_start;
            /* Resolve overlaps: keep the larger block */
            if (block_at[start] >= 0) {
                idx_t old_b = block_at[start];
                idx_t old_size = blks[old_b].row_end - blks[old_b].row_start;
                idx_t new_size = blks[b].row_end - blks[b].row_start;
                if (new_size <= old_size)
                    continue; /* keep existing larger block */
            }
            block_at[start] = b;
            /* Mark interior steps as covered to prevent overlapping blocks */
            idx_t bsz = blks[b].row_end - blks[b].row_start;
            for (idx_t s = start + 1; s < start + bsz && s < n; s++)
                block_at[s] = -2; /* -2 = interior of another block, not a start */
        }
    }

    /* Workspace for sparse elimination (same as lu_csr_eliminate) */
    idx_t *rstart = malloc((size_t)n * sizeof(idx_t));
    idx_t *rend = malloc((size_t)n * sizeof(idx_t));
    double *work = calloc((size_t)n, sizeof(double));
    idx_t *work_cols = malloc((size_t)n * sizeof(idx_t));
    idx_t *row_map = malloc((size_t)n * sizeof(idx_t));

    if (!rstart || !rend || !work || !work_cols || !row_map) {
        free(rstart);
        free(rend);
        free(work);
        free(work_cols);
        free(row_map);
        free(block_at);
        free(blks);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++) {
        rstart[i] = csr->row_ptr[i];
        rend[i] = csr->row_ptr[i + 1];
        row_map[i] = i;
    }
    if (piv_perm) {
        for (idx_t i = 0; i < n; i++)
            piv_perm[i] = i;
    }

    idx_t write_pos = csr->row_ptr[n];

    idx_t k = 0;
    while (k < n) {
        if (block_at[k] >= 0) {
            /* ── Dense block path ── */
            idx_t b = block_at[k];
            idx_t bsize = blks[b].row_end - blks[b].row_start;

            /* Degenerate block — fall back to sparse elimination for this k */
            if (bsize <= 0 || k + bsize > n) {
                block_at[k] = -1;
                goto sparse_fallback_no_alloc;
            }

            /* Overflow guard for dense block allocations */
            if ((size_t)bsize > SIZE_MAX / (size_t)bsize ||
                (size_t)bsize * (size_t)bsize > SIZE_MAX / sizeof(double) ||
                (size_t)bsize > SIZE_MAX / sizeof(idx_t)) {
                err = SPARSE_ERR_ALLOC;
                goto block_cleanup;
            }

            /* Extract the block from current CSR state (using row_map) */
            double *dense = calloc((size_t)bsize * (size_t)bsize, sizeof(double));
            idx_t *ipiv = malloc((size_t)bsize * sizeof(idx_t));
            if (!dense || !ipiv) {
                free(dense);
                free(ipiv);
                err = SPARSE_ERR_ALLOC;
                goto block_cleanup;
            }

            /* Validate block isolation: if any block row has entries in
             * columns >= k+bsize, we cannot skip the Schur complement update.
             * Fall back to sparse path for correctness. */
            int isolated = 1;
            for (idx_t bi = 0; bi < bsize && isolated; bi++) {
                idx_t ri = row_map[k + bi];
                idx_t s = rstart[ri], e = rend[ri];
                for (idx_t p = s; p < e; p++) {
                    if (csr->col_idx[p] >= k + bsize) {
                        isolated = 0;
                        break;
                    }
                }
            }
            if (!isolated) {
                goto sparse_fallback;
            }

            /* Extract: for each logical row in [k, k+bsize), read entries
             * in columns [k, k+bsize) from the CSR using row_map */
            for (idx_t bi = 0; bi < bsize; bi++) {
                idx_t ri = row_map[k + bi];
                idx_t s = rstart[ri], e = rend[ri];
                for (idx_t p = s; p < e; p++) {
                    idx_t c = csr->col_idx[p];
                    if (c >= k && c < k + bsize) {
                        idx_t bj = c - k;
                        dense[bi + bsize * bj] = csr->values[p]; /* col-major */
                    }
                }
            }

            /* Factor densely — on failure, fall back to sparse path */
            err = lu_dense_factor(bsize, bsize, dense, bsize, ipiv, tol);
            if (err != SPARSE_OK) {
                err = SPARSE_OK; /* Reset error — sparse path will handle it */
                goto sparse_fallback;
            }

            /* Apply pivot permutation to row_map and piv_perm */
            for (idx_t bi = 0; bi < bsize; bi++) {
                if (ipiv[bi] != bi && ipiv[bi] >= 0 && ipiv[bi] < bsize) {
                    idx_t r1 = k + bi;
                    idx_t r2 = k + ipiv[bi];
                    idx_t tmp = row_map[r1];
                    row_map[r1] = row_map[r2];
                    row_map[r2] = tmp;
                    if (piv_perm) {
                        idx_t ptmp = piv_perm[r1];
                        piv_perm[r1] = piv_perm[r2];
                        piv_perm[r2] = ptmp;
                    }
                }
            }

            /* Write L\U block entries back into CSR for each block row.
             * For each logical row k+bi, rebuild it: keep entries outside
             * the block columns, replace block columns with dense L\U. */
            for (idx_t bi = 0; bi < bsize; bi++) {
                idx_t ri = row_map[k + bi];
                idx_t s = rstart[ri], e = rend[ri];
                idx_t row_nnz_outside = 0;
                for (idx_t p = s; p < e; p++) {
                    idx_t c = csr->col_idx[p];
                    if (c < k || c >= k + bsize)
                        row_nnz_outside++;
                }

                idx_t new_nnz = row_nnz_outside + bsize; /* worst case: all block entries */
                size_t needed_b = (size_t)write_pos + (size_t)new_nnz;
                if (needed_b > (size_t)INT32_MAX) {
                    err = SPARSE_ERR_ALLOC;
                    free(dense);
                    free(ipiv);
                    goto block_cleanup;
                }
                err = lu_csr_grow(csr, (idx_t)needed_b);
                if (err != SPARSE_OK) {
                    free(dense);
                    free(ipiv);
                    goto block_cleanup;
                }

                idx_t cap = csr->capacity;
                idx_t nc = 0;
                idx_t new_start = write_pos;
                idx_t old_p = s;

                /* Entries before block columns.
                 * Capacity overflow is unreachable (lu_csr_grow guarantees
                 * cap >= needed_b), but guard defensively. */
                while (old_p < e && csr->col_idx[old_p] < k) {
                    if (new_start + nc >= cap) { // NOLINT
                        err = SPARSE_ERR_ALLOC;
                        free(dense);
                        free(ipiv);
                        goto block_cleanup;
                    }
                    csr->col_idx[new_start + nc] = csr->col_idx[old_p];
                    csr->values[new_start + nc] = csr->values[old_p];
                    nc++;
                    old_p++;
                }

                /* Block column entries from dense L\U */
                for (idx_t bj = 0; bj < bsize; bj++) {
                    double v = dense[bi + bsize * bj];
                    if (fabs(v) >= drop_tol || bj == bi) {
                        if (new_start + nc >= cap) { // NOLINT
                            err = SPARSE_ERR_ALLOC;
                            free(dense);
                            free(ipiv);
                            goto block_cleanup;
                        }
                        /* Always keep diagonal (even if small) */
                        csr->col_idx[new_start + nc] = k + bj;
                        csr->values[new_start + nc] = v;
                        nc++;
                    }
                }

                /* Skip old block column entries */
                while (old_p < e && csr->col_idx[old_p] < k + bsize)
                    old_p++;

                /* Entries after block columns */
                while (old_p < e) {
                    if (new_start + nc >= cap) { // NOLINT
                        err = SPARSE_ERR_ALLOC;
                        free(dense);
                        free(ipiv);
                        goto block_cleanup;
                    }
                    csr->col_idx[new_start + nc] = csr->col_idx[old_p];
                    csr->values[new_start + nc] = csr->values[old_p];
                    nc++;
                    old_p++;
                }

                rstart[ri] = new_start;
                rend[ri] = new_start + nc;
                write_pos = new_start + nc;
            }

            free(dense);
            free(ipiv);

            k += bsize; /* Skip block steps */
            continue;

        sparse_fallback:
            free(dense);
            free(ipiv);
            /* Fall through to sparse path for this step */

        sparse_fallback_no_alloc:;
        }

        /* ── Sparse scatter-gather path (same as lu_csr_eliminate) ── */
        {
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
                goto block_cleanup;
            }

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

            idx_t pk_s = rstart[pk], pk_e = rend[pk];
            idx_t wcount = 0;
            for (idx_t p = pk_s; p < pk_e; p++) {
                idx_t j = csr->col_idx[p];
                work[j] = csr->values[p];
                work_cols[wcount++] = j; // NOLINT(clang-analyzer-security.ArrayBound)
            }

            for (idx_t i = k + 1; i < n; i++) {
                idx_t ri = row_map[i];
                idx_t ri_s = rstart[ri], ri_e = rend[ri];

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

                idx_t ri_nnz = ri_e - ri_s;
                idx_t piv_nnz = pk_e - pk_s;
                size_t needed_s = (size_t)write_pos + (size_t)ri_nnz + (size_t)piv_nnz;
                if (needed_s > (size_t)INT32_MAX) {
                    err = SPARSE_ERR_ALLOC;
                    goto block_cleanup;
                }
                err = lu_csr_grow(csr, (idx_t)needed_s);
                if (err != SPARSE_OK)
                    goto block_cleanup;

                idx_t new_start = write_pos;
                idx_t nc = 0;
                idx_t pi = ri_s;

                while (pi < ri_e && csr->col_idx[pi] < k) {
                    csr->col_idx[new_start + nc] = csr->col_idx[pi];
                    csr->values[new_start + nc] = csr->values[pi];
                    nc++;
                    pi++;
                }

                csr->col_idx[new_start + nc] = k;
                csr->values[new_start + nc] = mult;
                nc++;

                if (pi < ri_e && csr->col_idx[pi] == k)
                    pi++;

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

                    if (fabs(new_val) < drop_tol * max_val)
                        continue;

                    csr->col_idx[new_start + nc] = col;
                    csr->values[new_start + nc] = new_val;
                    nc++;
                }

                rstart[ri] = new_start;
                rend[ri] = new_start + nc;
                write_pos = new_start + nc;
            }

            for (idx_t c = 0; c < wcount; c++)
                work[work_cols[c]] = 0.0;
        }

        k++;
    }

    /* ── Compact into clean CSR ── */
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
            goto block_cleanup;
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

block_cleanup:
    free(rstart);
    free(rend);
    free(work);
    free(work_cols);
    free(row_map);
    free(block_at);
    free(blks);
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

    /* Overflow guard for n-length allocations */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

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
        if (fabs(u_ii) < sparse_rel_tol(csr->factor_norm, DROP_TOL)) {
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

/* ─── CSR block solve (multiple RHS) ─────────────────────────────────── */

sparse_err_t lu_csr_solve_block(const LuCsr *csr, const idx_t *piv_perm, const double *B,
                                idx_t nrhs, double *X) {
    if (!csr || !piv_perm || !B || !X)
        return SPARSE_ERR_NULL;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;
    if (nrhs == 0)
        return SPARSE_OK;

    idx_t n = csr->n;

    /* Overflow guard: ensure n*nrhs fits in both size_t and idx_t so that
     * idx_t-based offset arithmetic (i + n*k) cannot overflow. */
    if (n > 0 && (size_t)nrhs > SIZE_MAX / (size_t)n)
        return SPARSE_ERR_ALLOC;
    size_t block_sz = (size_t)n * (size_t)nrhs;
    if (block_sz > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if (block_sz > (size_t)INT32_MAX)
        return SPARSE_ERR_ALLOC;

    double *Y = malloc(block_sz * sizeof(double));
    if (!Y)
        return SPARSE_ERR_ALLOC;

    /* Step 1: Apply pivot permutation — PB[i,k] = B[piv_perm[i], k] */
    double *PB = malloc(block_sz * sizeof(double));
    if (!PB) {
        free(Y);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            PB[i + n * k] = B[piv_perm[i] + n * k];

    /* Step 2: Forward substitution — L*Y = PB (unit diagonal).
     * Traverse each row once, update all nrhs vectors. */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t k = 0; k < nrhs; k++)
            Y[i + n * k] = PB[i + n * k];

        for (idx_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; p++) {
            idx_t j = csr->col_idx[p];
            if (j < i) {
                double l_ij = csr->values[p];
                for (idx_t k = 0; k < nrhs; k++)
                    Y[i + n * k] -= l_ij * Y[j + n * k]; // NOLINT
            }
        }
    }

    /* Step 3: Backward substitution — U*X = Y */
    for (idx_t i = n - 1; i >= 0; i--) {
        for (idx_t k = 0; k < nrhs; k++)
            X[i + n * k] = Y[i + n * k]; // NOLINT(clang-analyzer-core.uninitialized.Assign)

        double u_ii = 0.0;
        for (idx_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; p++) {
            idx_t j = csr->col_idx[p];
            if (j == i) {
                u_ii = csr->values[p];
            } else if (j > i) {
                double u_ij = csr->values[p];
                for (idx_t k = 0; k < nrhs; k++)
                    X[i + n * k] -= u_ij * X[j + n * k];
            }
        }

        if (fabs(u_ii) < sparse_rel_tol(csr->factor_norm, DROP_TOL)) {
            free(Y);
            free(PB);
            return SPARSE_ERR_SINGULAR;
        }
        for (idx_t k = 0; k < nrhs; k++)
            X[i + n * k] /= u_ii;
    }

    free(Y);
    free(PB);
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
    if ((size_t)n > SIZE_MAX / sizeof(idx_t)) {
        lu_csr_free(csr);
        return SPARSE_ERR_ALLOC;
    }
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

/* ─── Dense subblock detection ───────────────────────────────────────── */

/*
 * Build a column-to-row-set signature for supernodal detection.
 *
 * For each column j, we compute which rows in the CSR have an entry in
 * column j. We transpose the CSR: build col_rows[j] = sorted list of
 * rows that contain column j.
 *
 * Then, consecutive columns with identical row sets form a supernode.
 * If the supernode width × height ≥ min_size² and the fill ratio
 * meets the threshold, we report it as a dense block.
 */

sparse_err_t lu_detect_dense_blocks(const LuCsr *csr, idx_t min_size, double threshold,
                                    DenseBlock **blocks_out, idx_t *nblocks_out) {
    if (!csr || !blocks_out || !nblocks_out)
        return SPARSE_ERR_NULL;

    *blocks_out = NULL;
    *nblocks_out = 0;

    idx_t n = csr->n;
    if (n < min_size)
        return SPARSE_OK;

    /* Validate CSR structure */
    sparse_err_t verr = lu_csr_validate(csr);
    if (verr != SPARSE_OK)
        return verr;

    /* Overflow guard for n-sized allocations */
    if ((size_t)n > SIZE_MAX / sizeof(idx_t) || (size_t)(n + 1) > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    /* Build transpose: for each column j, collect the sorted row indices.
     * col_ptr[j]..col_ptr[j+1]-1 index into col_rows[]. */
    idx_t *col_count = calloc((size_t)n, sizeof(idx_t));
    if (!col_count)
        return SPARSE_ERR_ALLOC;

    /* Count entries per column (validate col_idx in [0, n)) */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; p++) {
            idx_t j = csr->col_idx[p];
            if (j < 0 || j >= n) {
                free(col_count);
                return SPARSE_ERR_BADARG;
            }
            col_count[j]++;
        }
    }

    /* Prefix sum → col_ptr */
    idx_t *col_ptr = malloc((size_t)(n + 1) * sizeof(idx_t));
    if (!col_ptr) {
        free(col_count);
        return SPARSE_ERR_ALLOC;
    }
    col_ptr[0] = 0;
    for (idx_t j = 0; j < n; j++)
        col_ptr[j + 1] = col_ptr[j] + col_count[j];

    idx_t total_entries = col_ptr[n]; // NOLINT(clang-analyzer-security.ArrayBound)
    idx_t *col_rows = malloc((size_t)(total_entries > 0 ? total_entries : 1) * sizeof(idx_t));
    if (!col_rows) {
        free(col_count);
        free(col_ptr);
        return SPARSE_ERR_ALLOC;
    }

    /* Reset col_count as write cursors */
    memset(col_count, 0, (size_t)n * sizeof(idx_t));

    /* Fill col_rows (rows appear in order since we iterate i=0..n-1) */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; p++) {
            idx_t j = csr->col_idx[p];
            col_rows[col_ptr[j] + col_count[j]] = i;
            col_count[j]++;
        }
    }
    free(col_count);

    /* Scan for supernodes: groups of consecutive columns with identical row sets.
     * Two columns j and j+1 have identical row sets if:
     * - Same number of rows
     * - Same row indices */

    /* Allocate result buffer (grow as needed) */
    idx_t blk_cap = 16;
    idx_t blk_count = 0;
    DenseBlock *blks = malloc((size_t)blk_cap * sizeof(DenseBlock));
    if (!blks) {
        free(col_ptr);
        free(col_rows);
        return SPARSE_ERR_ALLOC;
    }

    idx_t col_start = 0;
    while (col_start < n) {
        /* Find the end of this supernode: consecutive cols with identical row sets */
        idx_t col_end = col_start + 1;

        idx_t nrows_start = col_ptr[col_start + 1] - col_ptr[col_start];
        while (col_end < n) {
            idx_t nrows_end = col_ptr[col_end + 1] - col_ptr[col_end];
            if (nrows_end != nrows_start)
                break;

            /* Compare row indices */
            int same = 1;
            idx_t base_s = col_ptr[col_start];
            idx_t base_e = col_ptr[col_end];
            for (idx_t r = 0; r < nrows_start; r++) {
                if (col_rows[base_s + r] != col_rows[base_e + r]) { // NOLINT
                    same = 0;
                    break;
                }
            }
            if (!same)
                break;
            col_end++;
        }

        /* This supernode spans columns [col_start, col_end).
         * The row range is the min/max of the row set. */
        idx_t width = col_end - col_start;
        idx_t height = nrows_start;

        if (width >= min_size && height >= min_size) {
            /* Compute actual fill ratio within this block region.
             * The row range for these columns is in col_rows[col_ptr[col_start]..].
             * The block covers rows from min_row to max_row. But for supernodal,
             * we know exactly which rows are present — it's the row set itself.
             * The block region is row_min..row_max × col_start..col_end.
             * Fill ratio = (height * width) / ((row_max - row_min + 1) * width). */
            idx_t row_min = col_rows[col_ptr[col_start]];                   // NOLINT
            idx_t row_max = col_rows[col_ptr[col_start] + nrows_start - 1]; // NOLINT
            idx_t row_span = row_max - row_min + 1;

            /* Count actual entries in the block region.
             * For each row in the row set, count how many of the columns
             * [col_start..col_end) are present. */
            idx_t block_nnz = 0;
            for (idx_t r = 0; r < nrows_start; r++) {
                idx_t row = col_rows[col_ptr[col_start] + r];
                for (idx_t p = csr->row_ptr[row]; p < csr->row_ptr[row + 1]; p++) {
                    idx_t c = csr->col_idx[p];
                    if (c >= col_start && c < col_end)
                        block_nnz++;
                    if (c >= col_end)
                        break;
                }
            }

            double area = (double)row_span * (double)width;
            double fill_ratio = (area > 0.0) ? (double)block_nnz / area : 0.0;

            if (fill_ratio >= threshold) {
                /* Grow result array if needed */
                if (blk_count >= blk_cap) {
                    blk_cap *= 2;
                    DenseBlock *tmp = realloc(blks, (size_t)blk_cap * sizeof(DenseBlock));
                    if (!tmp) {
                        free(blks);
                        free(col_ptr);
                        free(col_rows);
                        return SPARSE_ERR_ALLOC;
                    }
                    blks = tmp;
                }

                blks[blk_count].row_start = row_min;
                blks[blk_count].row_end = row_max + 1;
                blks[blk_count].col_start = col_start;
                blks[blk_count].col_end = col_end;
                blk_count++;
            }
        }

        col_start = col_end;
    }

    free(col_ptr);
    free(col_rows);

    if (blk_count == 0) {
        free(blks);
        *blocks_out = NULL;
    } else {
        *blocks_out = blks;
    }
    *nblocks_out = blk_count;
    return SPARSE_OK;
}

/* ─── Dense block extraction ─────────────────────────────────────────── */

sparse_err_t lu_extract_dense_block(const LuCsr *csr, const DenseBlock *blk, double *dense) {
    if (!csr || !blk || !dense)
        return SPARSE_ERR_NULL;

    idx_t rows = blk->row_end - blk->row_start;
    idx_t cols = blk->col_end - blk->col_start;
    if (rows <= 0 || cols <= 0)
        return SPARSE_OK;

    /* Overflow guard for memset size */
    if ((size_t)rows > SIZE_MAX / (size_t)cols ||
        (size_t)rows * (size_t)cols > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

    /* Zero-initialize (column-major: dense[i + rows*j]) */
    memset(dense, 0, (size_t)rows * (size_t)cols * sizeof(double));

    /* Fill from CSR */
    for (idx_t i = 0; i < rows; i++) {
        idx_t row = blk->row_start + i;
        if (row >= csr->n)
            continue;
        for (idx_t p = csr->row_ptr[row]; p < csr->row_ptr[row + 1]; p++) {
            idx_t c = csr->col_idx[p];
            if (c < blk->col_start)
                continue;
            if (c >= blk->col_end)
                break;
            idx_t j = c - blk->col_start;
            dense[i + rows * j] = csr->values[p]; /* column-major */
        }
    }

    return SPARSE_OK;
}

/* ─── Dense block insertion ──────────────────────────────────────────── */

sparse_err_t lu_insert_dense_block(LuCsr *csr, const DenseBlock *blk, const double *dense,
                                   double drop_tol) {
    if (!csr || !blk || !dense)
        return SPARSE_ERR_NULL;

    idx_t n = csr->n;
    idx_t brows = blk->row_end - blk->row_start;
    idx_t bcols = blk->col_end - blk->col_start;

    /* Rebuild the entire CSR: for rows outside the block, copy as-is.
     * For rows inside the block, merge outside-block entries with dense values. */

    /* First pass: compute total new nnz (use size_t to avoid idx_t overflow) */
    size_t total = 0;
    for (idx_t row = 0; row < n; row++) {
        if (row < blk->row_start || row >= blk->row_end) {
            /* Outside block — copy all entries */
            total += (size_t)(csr->row_ptr[row + 1] - csr->row_ptr[row]);
        } else {
            /* Inside block: entries outside column range + dense entries */
            for (idx_t p = csr->row_ptr[row]; p < csr->row_ptr[row + 1]; p++) {
                idx_t c = csr->col_idx[p];
                if (c < blk->col_start || c >= blk->col_end)
                    total++;
            }
            idx_t bi = row - blk->row_start;
            for (idx_t j = 0; j < bcols; j++) {
                if (fabs(dense[bi + brows * j]) >= drop_tol)
                    total++;
            }
        }
    }

    /* Validate total fits in idx_t and allocation won't overflow */
    if (total > (size_t)INT32_MAX)
        return SPARSE_ERR_ALLOC;
    size_t alloc_n = total > 0 ? total : 1;
    if (alloc_n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

    /* Allocate new arrays */
    idx_t *new_rp = malloc((size_t)(n + 1) * sizeof(idx_t));
    idx_t *new_ci = malloc(alloc_n * sizeof(idx_t));
    double *new_v = malloc(alloc_n * sizeof(double));
    if (!new_rp || !new_ci || !new_v) {
        free(new_rp);
        free(new_ci);
        free(new_v);
        return SPARSE_ERR_ALLOC;
    }

    /* Second pass: build new CSR */
    idx_t pos = 0;
    for (idx_t row = 0; row < n; row++) {
        new_rp[row] = pos;

        if (row < blk->row_start || row >= blk->row_end) {
            /* Copy row as-is */
            for (idx_t p = csr->row_ptr[row]; p < csr->row_ptr[row + 1]; p++) {
                new_ci[pos] = csr->col_idx[p]; // NOLINT(clang-analyzer-security.ArrayBound)
                new_v[pos] = csr->values[p];   // NOLINT(clang-analyzer-security.ArrayBound)
                pos++;
            }
        } else {
            /* Merge: entries before block + dense block + entries after block */
            idx_t old_p = csr->row_ptr[row];
            idx_t old_end = csr->row_ptr[row + 1];
            idx_t bi = row - blk->row_start;

            /* Entries before block column range */
            while (old_p < old_end && csr->col_idx[old_p] < blk->col_start) {
                new_ci[pos] = csr->col_idx[old_p]; // NOLINT(clang-analyzer-security.ArrayBound)
                new_v[pos] = csr->values[old_p];
                pos++;
                old_p++;
            }

            /* Dense block entries */
            for (idx_t j = 0; j < bcols; j++) {
                double v = dense[bi + brows * j];
                if (fabs(v) >= drop_tol) {
                    new_ci[pos] = blk->col_start + j; // NOLINT(clang-analyzer-security.ArrayBound)
                    new_v[pos] = v;
                    pos++;
                }
            }

            /* Skip old entries in block column range */
            while (old_p < old_end && csr->col_idx[old_p] < blk->col_end)
                old_p++;

            /* Entries after block column range */
            while (old_p < old_end) {
                new_ci[pos] = csr->col_idx[old_p]; // NOLINT(clang-analyzer-security.ArrayBound)
                new_v[pos] = csr->values[old_p];
                pos++;
                old_p++;
            }
        }
    }
    new_rp[n] = pos;

    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->values);
    csr->row_ptr = new_rp;
    csr->col_idx = new_ci;
    csr->values = new_v;
    csr->nnz = pos;
    csr->capacity = (idx_t)(total > 0 ? total : 1);

    return SPARSE_OK;
}
