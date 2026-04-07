#include "sparse_lu.h"
#include "sparse_matrix_internal.h"
#include "sparse_reorder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── Permutation swap helpers ───────────────────────────────────────── */

static void swap_row_perm(SparseMatrix *mat, idx_t log_a, idx_t log_b) {
    idx_t tmp = mat->row_perm[log_a];
    mat->row_perm[log_a] = mat->row_perm[log_b];
    mat->row_perm[log_b] = tmp;
    mat->inv_row_perm[mat->row_perm[log_a]] = log_a;
    mat->inv_row_perm[mat->row_perm[log_b]] = log_b;
}

static void swap_col_perm(SparseMatrix *mat, idx_t log_a, idx_t log_b) {
    idx_t tmp = mat->col_perm[log_a];
    mat->col_perm[log_a] = mat->col_perm[log_b];
    mat->col_perm[log_b] = tmp;
    mat->inv_col_perm[mat->col_perm[log_a]] = log_a;
    mat->inv_col_perm[mat->col_perm[log_b]] = log_b;
}

/* ─── LU factorization ───────────────────────────────────────────────── */

sparse_err_t sparse_lu_factor(SparseMatrix *mat, sparse_pivot_t pivot, double tol) {
    if (!mat)
        return SPARSE_ERR_NULL;
    idx_t n = mat->rows;
    if (n != mat->cols)
        return SPARSE_ERR_SHAPE;

    /*
     * Temporary buffer for collecting rows to eliminate.
     * This fixes the bug where the column list is walked while being
     * modified: we snapshot the row indices first, then process them.
     */
    idx_t *elim_rows = malloc((size_t)n * sizeof(idx_t));
    if (!elim_rows)
        return SPARSE_ERR_ALLOC;

    /* Compute and cache ||A||_inf before factorization for relative tolerance */
    double anorm;
    sparse_err_t nerr = sparse_norminf(mat, &anorm);
    if (nerr != SPARSE_OK) {
        free(elim_rows);
        return nerr;
    }
    mat->factor_norm = anorm;

    for (idx_t k = 0; k < n; k++) {
        /* ── Pivot search ── */
        double max_val = 0.0;
        idx_t pivot_log_row = k;
        idx_t pivot_log_col = k;

        if (pivot == SPARSE_PIVOT_PARTIAL) {
            /* Partial pivoting: search only the pivot column (log col k) */
            idx_t phys_k = mat->col_perm[k];
            Node *curr = mat->col_headers[phys_k];
            while (curr) {
                idx_t log_i = mat->inv_row_perm[curr->row];
                if (log_i >= k) {
                    double av = fabs(curr->value);
                    if (av > max_val) {
                        max_val = av;
                        pivot_log_row = log_i;
                    }
                }
                curr = curr->down;
            }
        } else {
            /* Complete pivoting: search entire remaining submatrix */
            for (idx_t log_j = k; log_j < n; log_j++) {
                idx_t phys_j = mat->col_perm[log_j];
                Node *curr = mat->col_headers[phys_j];
                while (curr) {
                    idx_t log_i = mat->inv_row_perm[curr->row];
                    if (log_i >= k) {
                        double av = fabs(curr->value);
                        if (av > max_val) {
                            max_val = av;
                            pivot_log_row = log_i;
                            pivot_log_col = log_j;
                        }
                    }
                    curr = curr->down;
                }
            }
        }

        if (max_val < tol) {
            free(elim_rows);
            return SPARSE_ERR_SINGULAR;
        }

        /* ── Swap rows (always) and columns (complete pivoting only) ── */
        if (pivot_log_row != k)
            swap_row_perm(mat, k, pivot_log_row);
        if (pivot == SPARSE_PIVOT_COMPLETE && pivot_log_col != k)
            swap_col_perm(mat, k, pivot_log_col);

        /* ── Snapshot: collect logical row indices for elimination ── */
        idx_t elim_count = 0;
        {
            idx_t phys_k_col = mat->col_perm[k];
            Node *curr = mat->col_headers[phys_k_col];
            while (curr) {
                idx_t log_i = mat->inv_row_perm[curr->row];
                if (log_i > k) {
                    elim_rows[elim_count++] = log_i; // NOLINT(clang-analyzer-security.ArrayBound)
                }
                curr = curr->down;
            }
        }

        /* ── Elimination ── */
        /* Cache physical indices to avoid repeated permutation lookups */
        idx_t phys_k_col_val = mat->col_perm[k];
        idx_t phys_row_k = mat->row_perm[k];
        double pivot_val = sparse_get_phys(mat, phys_row_k, phys_k_col_val);
        if (fabs(pivot_val) < tol) {
            free(elim_rows);
            return SPARSE_ERR_SINGULAR;
        }

        for (idx_t e = 0; e < elim_count; e++) {
            idx_t log_i = elim_rows[e];
            idx_t phys_i = mat->row_perm[log_i];

            /* Find a_ik using physical coordinates directly */
            double a_ik = sparse_get_phys(mat, phys_i, phys_k_col_val);
            double mult = a_ik / pivot_val;

            /* Store multiplier in L position */
            sparse_err_t err = sparse_set(mat, log_i, k, mult);
            if (err != SPARSE_OK) {
                free(elim_rows);
                return err;
            }

            /* Subtract mult * row_k from row_i for columns j > k.
             * Use sparse_get_phys for reads to skip permutation lookups,
             * but keep sparse_set for writes (maintains correct structure). */
            Node *uj = mat->row_headers[phys_row_k];
            while (uj) {
                idx_t log_j = mat->inv_col_perm[uj->col];
                if (log_j > k) {
                    double u_kj = uj->value;
                    double a_ij = sparse_get_phys(mat, phys_i, uj->col);
                    double new_val = a_ij - mult * u_kj;
                    if (fabs(new_val) < DROP_TOL * max_val) {
                        err = sparse_set(mat, log_i, log_j, 0.0);
                    } else {
                        err = sparse_set(mat, log_i, log_j, new_val);
                    }
                    if (err != SPARSE_OK) {
                        free(elim_rows);
                        return err;
                    }
                }
                uj = uj->right;
            }
        }
    }

    free(elim_rows);
    mat->factored = 1;
    return SPARSE_OK;
}

/* ─── Factor with options (reordering + pivoting) ────────────────────── */

sparse_err_t sparse_lu_factor_opts(SparseMatrix *mat, const sparse_lu_opts_t *opts) {
    if (!mat || !opts)
        return SPARSE_ERR_NULL;
    idx_t n = mat->rows;
    if (n != mat->cols)
        return SPARSE_ERR_SHAPE;

    /* Clear any previous reorder permutation */
    free(mat->reorder_perm);
    mat->reorder_perm = NULL;

    /* Apply fill-reducing reordering if requested */
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

        /* Apply symmetric permutation in-place:
         * Create permuted copy, then swap contents into mat */
        SparseMatrix *PA = NULL;
        err = sparse_permute(mat, perm, perm, &PA);
        if (err != SPARSE_OK) {
            free(perm);
            return err;
        }

        /* Swap internal data from PA into mat:
         * Free mat's old data, steal PA's data */
        pool_free_all(&mat->pool);
        free(mat->row_headers);
        free(mat->col_headers);

        mat->row_headers = PA->row_headers;
        mat->col_headers = PA->col_headers;
        mat->pool = PA->pool;
        mat->nnz = PA->nnz;
        mat->cached_norm = PA->cached_norm;

        /* Prevent PA from freeing the data we stole */
        PA->row_headers = NULL;
        PA->col_headers = NULL;
        PA->pool.head = NULL;
        PA->pool.current = NULL;
        PA->pool.free_list = NULL;
        sparse_free(PA);

        /* Reset row/col permutations to identity — the reordered matrix
         * has fresh physical layout, so LU factorization must start from
         * identity permutations regardless of any prior factorization. */
        for (idx_t i = 0; i < n; i++) {
            mat->row_perm[i] = i;
            mat->inv_row_perm[i] = i;
            mat->col_perm[i] = i;
            mat->inv_col_perm[i] = i;
        }

        /* Store reorder permutation for solve to unpermute */
        free(mat->reorder_perm);
        mat->reorder_perm = perm;
    }

    /* Factor with given pivoting and tolerance */
    return sparse_lu_factor(mat, opts->pivot, opts->tol);
}

/* ─── Transpose solve ────────────────────────────────────────────────── */

sparse_err_t sparse_lu_solve_transpose(const SparseMatrix *mat, const double *b, double *x) {
    if (!mat || !b || !x)
        return SPARSE_ERR_NULL;
    if (!mat->factored)
        return SPARSE_ERR_BADARG;
    idx_t n = mat->rows;
    const idx_t *rperm = mat->reorder_perm;

    double *c = malloc((size_t)n * sizeof(double));
    double *d = malloc((size_t)n * sizeof(double));
    double *w = malloc((size_t)n * sizeof(double));
    double *b_perm = NULL;
    if (rperm) {
        b_perm = malloc((size_t)n * sizeof(double));
        if (!b_perm) {
            free(c);
            free(d);
            free(w);
            return SPARSE_ERR_ALLOC;
        }
    }
    if (!c || !d || !w) {
        free(c);
        free(d);
        free(w);
        free(b_perm);
        return SPARSE_ERR_ALLOC;
    }

    /* If reorder permutation exists, permute b first */
    const double *b_eff = b;
    if (rperm) {
        for (idx_t i = 0; i < n; i++)
            b_perm[i] = b[rperm[i]];
        b_eff = b_perm;
    }

    /* Step 1: c = Q^T * b  →  c[i] = b_eff[col_perm[i]] */
    for (idx_t i = 0; i < n; i++)
        c[i] = b_eff[mat->col_perm[i]];

    /* Step 2: Forward-substitute with U^T (lower triangular).
     * U^T[i][j] = U[j][i], so walk column i to find U entries with log_row <= i.
     * Solve: d[i] = (c[i] - sum_{j<i} U[j][i]*d[j]) / U[i][i] */
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        double u_ii = 0.0;
        idx_t phys_col = mat->col_perm[i];
        Node *node = mat->col_headers[phys_col];
        while (node) {
            idx_t log_j = mat->inv_row_perm[node->row];
            if (log_j == i)
                u_ii = node->value;
            else if (log_j < i)
                sum += node->value * d[log_j];
            node = node->down;
        }
        double sing_tol = (mat->factor_norm > 0.0) ? DROP_TOL * mat->factor_norm : DROP_TOL;
        if (fabs(u_ii) < sing_tol) {
            free(c);
            free(d);
            free(w);
            free(b_perm);
            return SPARSE_ERR_SINGULAR;
        }
        d[i] = (c[i] - sum) / u_ii;
    }

    /* Step 3: Backward-substitute with L^T (upper triangular, unit diagonal).
     * L^T[i][j] = L[j][i], so walk column i to find L entries with log_row > i.
     * Solve: w[i] = d[i] - sum_{j>i} L[j][i]*w[j] */
    for (idx_t i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        idx_t phys_col = mat->col_perm[i];
        Node *node = mat->col_headers[phys_col];
        while (node) {
            idx_t log_j = mat->inv_row_perm[node->row];
            if (log_j > i)
                sum += node->value * w[log_j];
            node = node->down;
        }
        // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
        w[i] = d[i] - sum; /* L^T has unit diagonal */
    }

    /* Step 4: x = P^T * w = P^{-1} * w  →  x[i] = w[inv_row_perm[i]] */
    for (idx_t i = 0; i < n; i++)
        x[i] = w[mat->inv_row_perm[i]];

    /* If reorder permutation exists, unpermute x */
    if (rperm) {
        memcpy(w, x, (size_t)n * sizeof(double)); /* reuse w as temp */
        for (idx_t i = 0; i < n; i++)
            x[rperm[i]] = w[i];
    }

    free(c);
    free(d);
    free(w);
    free(b_perm);
    return SPARSE_OK;
}

/* ─── Condition number estimation ────────────────────────────────────── */

/* Compute ||A||_1 = max column sum of |a_ij| */
static double sparse_norm1(const SparseMatrix *mat) {
    double max_col_sum = 0.0;
    for (idx_t j = 0; j < mat->cols; j++) {
        double col_sum = 0.0;
        Node *node = mat->col_headers[j];
        while (node) {
            col_sum += fabs(node->value);
            node = node->down;
        }
        if (col_sum > max_col_sum)
            max_col_sum = col_sum;
    }
    return max_col_sum;
}

sparse_err_t sparse_lu_condest(const SparseMatrix *mat_orig, const SparseMatrix *mat_lu,
                               double *condest) {
    if (!mat_orig || !mat_lu || !condest)
        return SPARSE_ERR_NULL;
    /* Check that mat_lu has been factored */
    if (!mat_lu->factored)
        return SPARSE_ERR_BADARG;
    /* Check dimensions match and are square */
    if (mat_orig->rows != mat_orig->cols)
        return SPARSE_ERR_SHAPE;
    if (mat_lu->rows != mat_lu->cols)
        return SPARSE_ERR_SHAPE;
    if (mat_orig->rows != mat_lu->rows)
        return SPARSE_ERR_SHAPE;

    idx_t n = mat_orig->rows;
    if (n == 0) {
        *condest = 0.0;
        return SPARSE_OK;
    }

    /* Compute ||A||_1 from the original matrix */
    double norm_A = sparse_norm1(mat_orig);
    if (norm_A == 0.0) {
        *condest = INFINITY;
        return SPARSE_OK;
    }

    /* Allocate workspace for Hager/Higham 1-norm estimator */
    double *x = malloc((size_t)n * sizeof(double));
    double *w = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double));
    if (!x || !w || !z) {
        free(x);
        free(w);
        free(z);
        return SPARSE_ERR_ALLOC;
    }

    /* Hager/Higham algorithm to estimate ||A^{-1}||_1:
     * 1. x = [1/n, ..., 1/n]
     * 2. Solve A*w = x
     * 3. xi = sign(w)
     * 4. Solve A^T*z = xi
     * 5. If ||z||_inf <= z^T*w, converged: ||A^{-1}||_1 ≈ ||w||_1
     * 6. Otherwise x = e_j where j = argmax|z_j|, goto 2
     * Max 5 iterations. */

    double inv_n = 1.0 / (double)n;
    for (idx_t i = 0; i < n; i++)
        x[i] = inv_n;

    double est = 0.0;
    int max_iter = 5;
    sparse_err_t err;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Solve A*w = x */
        err = sparse_lu_solve(mat_lu, x, w);
        if (err != SPARSE_OK) {
            free(x);
            free(w);
            free(z);
            return err;
        }

        /* Compute ||w||_1 */
        double w_norm1 = 0.0;
        for (idx_t i = 0; i < n; i++)
            w_norm1 += fabs(w[i]);
        est = w_norm1;

        /* xi = sign(w) */
        for (idx_t i = 0; i < n; i++)
            x[i] = (w[i] >= 0.0) ? 1.0 : -1.0;

        /* Solve A^T*z = xi */
        err = sparse_lu_solve_transpose(mat_lu, x, z);
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

        if (z_inf <= zt_w || iter == max_iter - 1)
            break;

        /* Set x = e_{j_max} */
        for (idx_t i = 0; i < n; i++)
            x[i] = 0.0;
        x[j_max] = 1.0;
    }

    *condest = norm_A * est;

    free(x);
    free(w);
    free(z);
    return SPARSE_OK;
}

/* ─── Solver phases ──────────────────────────────────────────────────── */

sparse_err_t sparse_apply_row_perm(const SparseMatrix *mat, const double *b, double *pb) {
    if (!mat || !b || !pb)
        return SPARSE_ERR_NULL;
    for (idx_t i = 0; i < mat->rows; i++)
        pb[i] = b[mat->row_perm[i]];
    return SPARSE_OK;
}

sparse_err_t sparse_apply_inv_col_perm(const SparseMatrix *mat, const double *z, double *x) {
    if (!mat || !z || !x)
        return SPARSE_ERR_NULL;
    for (idx_t i = 0; i < mat->cols; i++)
        x[i] = z[mat->inv_col_perm[i]];
    return SPARSE_OK;
}

sparse_err_t sparse_forward_sub(const SparseMatrix *mat, const double *pb, double *y) {
    if (!mat || !pb || !y)
        return SPARSE_ERR_NULL;
    idx_t n = mat->rows;

    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        idx_t phys_i = mat->row_perm[i];
        Node *node = mat->row_headers[phys_i];

        /*
         * Walk the ENTIRE row and accumulate only L entries (log_col < i).
         * We do NOT break early because after pivoting, physical column
         * order does not correspond to logical column order.
         */
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            if (log_j < i) {
                sum += node->value * y[log_j]; // NOLINT(clang-analyzer-security.ArrayBound)
            }
            node = node->right;
        }
        y[i] = pb[i] - sum; /* L has unit diagonal */
    }

    return SPARSE_OK;
}

sparse_err_t sparse_backward_sub(const SparseMatrix *mat, const double *y, double *z) {
    if (!mat || !y || !z)
        return SPARSE_ERR_NULL;
    idx_t n = mat->rows;

    for (idx_t i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        double u_ii = 0.0;
        idx_t phys_i = mat->row_perm[i];
        Node *node = mat->row_headers[phys_i];

        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            if (log_j == i)
                u_ii = node->value;
            else if (log_j > i)
                sum += node->value * z[log_j];
            node = node->right;
        }

        /* Use relative tolerance if factor_norm is available, else absolute */
        double sing_tol = (mat->factor_norm > 0.0) ? DROP_TOL * mat->factor_norm : DROP_TOL;
        if (fabs(u_ii) < sing_tol)
            return SPARSE_ERR_SINGULAR;

        z[i] = (y[i] - sum) / u_ii; // NOLINT(clang-analyzer-core.UndefinedBinaryOperatorResult)
    }

    return SPARSE_OK;
}

/* ─── Full LU solve ──────────────────────────────────────────────────── */

sparse_err_t sparse_lu_solve(const SparseMatrix *mat, const double *b, double *x) {
    if (!mat || !b || !x)
        return SPARSE_ERR_NULL;
    if (!mat->factored)
        return SPARSE_ERR_BADARG;
    idx_t n = mat->rows;
    const idx_t *rperm = mat->reorder_perm;

    double *pb = malloc((size_t)n * sizeof(double));
    double *y = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double));
    double *b_perm = NULL;
    if (rperm) {
        b_perm = malloc((size_t)n * sizeof(double));
        if (!b_perm) {
            free(pb);
            free(y);
            free(z);
            return SPARSE_ERR_ALLOC;
        }
    }
    if (!pb || !y || !z) {
        free(pb);
        free(y);
        free(z);
        free(b_perm);
        return SPARSE_ERR_ALLOC;
    }

    /* If reorder permutation exists, permute b first */
    const double *b_eff = b;
    if (rperm) {
        for (idx_t i = 0; i < n; i++)
            b_perm[i] = b[rperm[i]];
        b_eff = b_perm;
    }

    sparse_err_t err;

    err = sparse_apply_row_perm(mat, b_eff, pb);
    if (err != SPARSE_OK)
        goto cleanup;

    err = sparse_forward_sub(mat, pb, y);
    if (err != SPARSE_OK)
        goto cleanup;

    err = sparse_backward_sub(mat, y, z);
    if (err != SPARSE_OK)
        goto cleanup;

    err = sparse_apply_inv_col_perm(mat, z, x);
    if (err != SPARSE_OK)
        goto cleanup;

    /* If reorder permutation exists, unpermute x */
    if (rperm) {
        /* x currently holds solution in permuted space.
         * We need x_orig[rperm[i]] = x_perm[i] */
        memcpy(z, x, (size_t)n * sizeof(double)); /* reuse z as temp */
        for (idx_t i = 0; i < n; i++)
            x[rperm[i]] = z[i];
    }

cleanup:
    free(pb);
    free(y);
    free(z);
    free(b_perm);
    return err;
}

/* ─── Block LU solve (multiple RHS) ──────────────────────────────────── */

sparse_err_t sparse_lu_solve_block(const SparseMatrix *mat, const double *B, idx_t nrhs,
                                   double *X) {
    if (!mat || !B || !X)
        return SPARSE_ERR_NULL;
    if (!mat->factored)
        return SPARSE_ERR_BADARG;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;
    if (nrhs == 0)
        return SPARSE_OK;

    idx_t n = mat->rows;
    const idx_t *rperm = mat->reorder_perm;

    /* Overflow guard: ensure n*nrhs fits in both size_t and idx_t so that
     * idx_t-based offset arithmetic (i + n*k) cannot overflow. */
    if (n > 0 && (size_t)nrhs > SIZE_MAX / (size_t)n)
        return SPARSE_ERR_ALLOC;
    size_t block_sz = (size_t)n * (size_t)nrhs;
    if (block_sz > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if (block_sz > (size_t)INT32_MAX)
        return SPARSE_ERR_ALLOC;

    /* Allocate workspace: PB (permuted B), Y (forward sub result) for all RHS */
    double *PB = malloc(block_sz * sizeof(double));
    double *Y = malloc(block_sz * sizeof(double));
    double *Z = malloc(block_sz * sizeof(double));
    if (!PB || !Y || !Z) {
        free(PB);
        free(Y);
        free(Z);
        return SPARSE_ERR_ALLOC;
    }

    /* Step 1: Apply reorder permutation (if any) and row permutation.
     * PB[i + n*k] = B[row_perm[i] ... ] with reorder applied. */
    for (idx_t k = 0; k < nrhs; k++) {
        for (idx_t i = 0; i < n; i++) {
            idx_t src = mat->row_perm[i];
            if (rperm)
                PB[i + n * k] = B[rperm[src] + n * k];
            else
                PB[i + n * k] = B[src + n * k];
        }
    }

    /* Step 2: Forward substitution — L*Y = PB (L has unit diagonal).
     * For each row i, traverse the row once and update all nrhs vectors. */
    for (idx_t i = 0; i < n; i++) {
        /* Initialize Y[i,k] = PB[i,k] for all k */
        for (idx_t k = 0; k < nrhs; k++)
            Y[i + n * k] = PB[i + n * k];

        /* Subtract L[i,j] * Y[j,k] for j < i, all k */
        idx_t phys_i = mat->row_perm[i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            if (log_j < i) {
                double l_ij = node->value;
                for (idx_t k = 0; k < nrhs; k++)
                    Y[i + n * k] -=
                        l_ij * Y[log_j + n * k]; // NOLINT(clang-analyzer-security.ArrayBound)
            }
            node = node->right;
        }
    }

    /* Step 3: Backward substitution — U*Z = Y.
     * For each row i (reverse), traverse once and update all nrhs vectors. */
    double sing_tol = (mat->factor_norm > 0.0) ? DROP_TOL * mat->factor_norm : DROP_TOL;

    for (idx_t i = n - 1; i >= 0; i--) {
        /* Initialize Z[i,k] = Y[i,k] */
        for (idx_t k = 0; k < nrhs; k++)
            Z[i + n * k] = Y[i + n * k]; // NOLINT(clang-analyzer-core.uninitialized.Assign)

        double u_ii = 0.0;
        idx_t phys_i = mat->row_perm[i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            if (log_j == i) {
                u_ii = node->value;
            } else if (log_j > i) {
                double u_ij = node->value;
                for (idx_t k = 0; k < nrhs; k++)
                    Z[i + n * k] -= u_ij * Z[log_j + n * k];
            }
            node = node->right;
        }

        if (fabs(u_ii) < sing_tol) {
            free(PB);
            free(Y);
            free(Z);
            return SPARSE_ERR_SINGULAR;
        }

        for (idx_t k = 0; k < nrhs; k++)
            Z[i + n * k] /= u_ii;
    }

    /* Step 4: Apply inverse column permutation and reorder unpermutation.
     * X[col, k] = Z[inv_col_perm[col], k], with reorder applied. */
    for (idx_t k = 0; k < nrhs; k++) {
        for (idx_t i = 0; i < n; i++) {
            double val = Z[mat->inv_col_perm[i] + n * k];
            if (rperm) {
                /* Store into reorder-unpermuted position */
                X[rperm[i] + n * k] = val; // NOLINT(clang-analyzer-security.ArrayBound)
            } else {
                X[i + n * k] = val;
            }
        }
    }

    free(PB);
    free(Y);
    free(Z);
    return SPARSE_OK;
}

/* ─── Iterative refinement ───────────────────────────────────────────── */

sparse_err_t sparse_lu_refine(const SparseMatrix *mat_orig, const SparseMatrix *mat_lu,
                              const double *b, double *x, int max_iters, double tol) {
    if (!mat_orig || !mat_lu || !b || !x)
        return SPARSE_ERR_NULL;
    if (!mat_lu->factored)
        return SPARSE_ERR_BADARG;
    idx_t n = mat_orig->rows;

    double *r = malloc((size_t)n * sizeof(double));
    double *d = malloc((size_t)n * sizeof(double));
    if (!r || !d) {
        free(r);
        free(d);
        return SPARSE_ERR_ALLOC;
    }

    /* Compute ||b|| for relative residual */
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
        sparse_err_t err = sparse_matvec(mat_orig, x, r);
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

        /* Solve A*d = r using existing LU */
        err = sparse_lu_solve(mat_lu, r, d);
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
