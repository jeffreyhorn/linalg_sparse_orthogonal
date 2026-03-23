#include "sparse_lu.h"
#include "sparse_matrix_internal.h"

#include <stdlib.h>
#include <math.h>

/* ─── Permutation swap helpers ───────────────────────────────────────── */

static void swap_row_perm(SparseMatrix *mat, idx_t log_a, idx_t log_b)
{
    idx_t tmp = mat->row_perm[log_a];
    mat->row_perm[log_a] = mat->row_perm[log_b];
    mat->row_perm[log_b] = tmp;
    mat->inv_row_perm[mat->row_perm[log_a]] = log_a;
    mat->inv_row_perm[mat->row_perm[log_b]] = log_b;
}

static void swap_col_perm(SparseMatrix *mat, idx_t log_a, idx_t log_b)
{
    idx_t tmp = mat->col_perm[log_a];
    mat->col_perm[log_a] = mat->col_perm[log_b];
    mat->col_perm[log_b] = tmp;
    mat->inv_col_perm[mat->col_perm[log_a]] = log_a;
    mat->inv_col_perm[mat->col_perm[log_b]] = log_b;
}

/* ─── LU factorization ───────────────────────────────────────────────── */

sparse_err_t sparse_lu_factor(SparseMatrix *mat, sparse_pivot_t pivot,
                              double tol)
{
    if (!mat) return SPARSE_ERR_NULL;
    idx_t n = mat->rows;
    if (n != mat->cols) return SPARSE_ERR_SHAPE;

    /*
     * Temporary buffer for collecting rows to eliminate.
     * This fixes the bug where the column list is walked while being
     * modified: we snapshot the row indices first, then process them.
     */
    idx_t *elim_rows = malloc((size_t)n * sizeof(idx_t));
    if (!elim_rows) return SPARSE_ERR_ALLOC;

    /* Compute and cache ||A||_inf before factorization for relative tolerance */
    double anorm;
    sparse_err_t nerr = sparse_norminf(mat, &anorm);
    if (nerr != SPARSE_OK) { free(elim_rows); return nerr; }
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
                    elim_rows[elim_count++] = log_i;
                }
                curr = curr->down;
            }
        }

        /* ── Elimination ── */
        double pivot_val = sparse_get(mat, k, k);
        if (fabs(pivot_val) < tol) {
            free(elim_rows);
            return SPARSE_ERR_SINGULAR;
        }

        for (idx_t e = 0; e < elim_count; e++) {
            idx_t log_i = elim_rows[e];
            double a_ik = sparse_get(mat, log_i, k);
            double mult = a_ik / pivot_val;

            /* Store multiplier in L position */
            sparse_err_t err = sparse_set(mat, log_i, k, mult);
            if (err != SPARSE_OK) { free(elim_rows); return err; }

            /* Subtract mult * row_k from row_i for columns j > k */
            idx_t phys_row_k = mat->row_perm[k];
            Node *uj = mat->row_headers[phys_row_k];
            while (uj) {
                idx_t log_j = mat->inv_col_perm[uj->col];
                if (log_j > k) {
                    double u_kj = uj->value;
                    double a_ij = sparse_get(mat, log_i, log_j);
                    double new_val = a_ij - mult * u_kj;
                    if (fabs(new_val) < DROP_TOL * max_val) {
                        err = sparse_set(mat, log_i, log_j, 0.0);
                    } else {
                        err = sparse_set(mat, log_i, log_j, new_val);
                    }
                    if (err != SPARSE_OK) { free(elim_rows); return err; }
                }
                uj = uj->right;
            }
        }
    }

    free(elim_rows);
    return SPARSE_OK;
}

/* ─── Transpose solve ────────────────────────────────────────────────── */

sparse_err_t sparse_lu_solve_transpose(const SparseMatrix *mat,
                                       const double *b, double *x)
{
    if (!mat || !b || !x) return SPARSE_ERR_NULL;
    idx_t n = mat->rows;

    double *c = malloc((size_t)n * sizeof(double));
    double *d = malloc((size_t)n * sizeof(double));
    double *w = malloc((size_t)n * sizeof(double));
    if (!c || !d || !w) {
        free(c); free(d); free(w);
        return SPARSE_ERR_ALLOC;
    }

    /* Step 1: c = Q^T * b  →  c[i] = b[col_perm[i]] */
    for (idx_t i = 0; i < n; i++)
        c[i] = b[mat->col_perm[i]];

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
        double sing_tol = (mat->factor_norm > 0.0)
                        ? DROP_TOL * mat->factor_norm
                        : DROP_TOL;
        if (fabs(u_ii) < sing_tol) {
            free(c); free(d); free(w);
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
        w[i] = d[i] - sum;  /* L^T has unit diagonal */
    }

    /* Step 4: x = P^T * w = P^{-1} * w  →  x[i] = w[inv_row_perm[i]] */
    for (idx_t i = 0; i < n; i++)
        x[i] = w[mat->inv_row_perm[i]];

    free(c); free(d); free(w);
    return SPARSE_OK;
}

/* ─── Condition number estimation ────────────────────────────────────── */

sparse_err_t sparse_lu_condest(const SparseMatrix *mat_orig,
                               const SparseMatrix *mat_lu,
                               double *condest)
{
    if (!mat_orig || !mat_lu || !condest) return SPARSE_ERR_NULL;
    /* Check that mat_lu has been factored (factor_norm is set during factorization) */
    if (mat_lu->factor_norm < 0.0) return SPARSE_ERR_BADARG;
    /* TODO: implement Hager's algorithm in Day 3 */
    (void)mat_orig;
    *condest = -1.0;
    return SPARSE_ERR_BADARG;
}

/* ─── Solver phases ──────────────────────────────────────────────────── */

sparse_err_t sparse_apply_row_perm(const SparseMatrix *mat,
                                   const double *b, double *pb)
{
    if (!mat || !b || !pb) return SPARSE_ERR_NULL;
    for (idx_t i = 0; i < mat->rows; i++)
        pb[i] = b[mat->row_perm[i]];
    return SPARSE_OK;
}

sparse_err_t sparse_apply_inv_col_perm(const SparseMatrix *mat,
                                       const double *z, double *x)
{
    if (!mat || !z || !x) return SPARSE_ERR_NULL;
    for (idx_t i = 0; i < mat->cols; i++)
        x[i] = z[mat->inv_col_perm[i]];
    return SPARSE_OK;
}

sparse_err_t sparse_forward_sub(const SparseMatrix *mat,
                                const double *pb, double *y)
{
    if (!mat || !pb || !y) return SPARSE_ERR_NULL;
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
                sum += node->value * y[log_j];
            }
            node = node->right;
        }
        y[i] = pb[i] - sum;  /* L has unit diagonal */
    }

    return SPARSE_OK;
}

sparse_err_t sparse_backward_sub(const SparseMatrix *mat,
                                 const double *y, double *z)
{
    if (!mat || !y || !z) return SPARSE_ERR_NULL;
    idx_t n = mat->rows;

    for (idx_t i = n - 1; i >= 0; i--) {
        double sum  = 0.0;
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
        double sing_tol = (mat->factor_norm > 0.0)
                        ? DROP_TOL * mat->factor_norm
                        : DROP_TOL;
        if (fabs(u_ii) < sing_tol)
            return SPARSE_ERR_SINGULAR;

        z[i] = (y[i] - sum) / u_ii;
    }

    return SPARSE_OK;
}

/* ─── Full LU solve ──────────────────────────────────────────────────── */

sparse_err_t sparse_lu_solve(const SparseMatrix *mat,
                             const double *b, double *x)
{
    if (!mat || !b || !x) return SPARSE_ERR_NULL;
    idx_t n = mat->rows;

    double *pb = malloc((size_t)n * sizeof(double));
    double *y  = malloc((size_t)n * sizeof(double));
    double *z  = malloc((size_t)n * sizeof(double));
    if (!pb || !y || !z) {
        free(pb); free(y); free(z);
        return SPARSE_ERR_ALLOC;
    }

    sparse_err_t err;

    err = sparse_apply_row_perm(mat, b, pb);
    if (err != SPARSE_OK) goto cleanup;

    err = sparse_forward_sub(mat, pb, y);
    if (err != SPARSE_OK) goto cleanup;

    err = sparse_backward_sub(mat, y, z);
    if (err != SPARSE_OK) goto cleanup;

    err = sparse_apply_inv_col_perm(mat, z, x);

cleanup:
    free(pb);
    free(y);
    free(z);
    return err;
}

/* ─── Iterative refinement ───────────────────────────────────────────── */

sparse_err_t sparse_lu_refine(const SparseMatrix *mat_orig,
                              const SparseMatrix *mat_lu,
                              const double *b, double *x,
                              int max_iters, double tol)
{
    if (!mat_orig || !mat_lu || !b || !x) return SPARSE_ERR_NULL;
    idx_t n = mat_orig->rows;

    double *r = malloc((size_t)n * sizeof(double));
    double *d = malloc((size_t)n * sizeof(double));
    if (!r || !d) {
        free(r); free(d);
        return SPARSE_ERR_ALLOC;
    }

    /* Compute ||b|| for relative residual */
    double norm_b = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ab = fabs(b[i]);
        if (ab > norm_b) norm_b = ab;
    }
    if (norm_b == 0.0) norm_b = 1.0;

    for (int iter = 0; iter < max_iters; iter++) {
        /* r = b - A*x */
        sparse_err_t err = sparse_matvec(mat_orig, x, r);
        if (err != SPARSE_OK) { free(r); free(d); return err; }

        double norm_r = 0.0;
        for (idx_t i = 0; i < n; i++) {
            r[i] = b[i] - r[i];
            double ar = fabs(r[i]);
            if (ar > norm_r) norm_r = ar;
        }

        if (norm_r / norm_b < tol) break;

        /* Solve A*d = r using existing LU */
        err = sparse_lu_solve(mat_lu, r, d);
        if (err != SPARSE_OK) { free(r); free(d); return err; }

        /* x += d */
        for (idx_t i = 0; i < n; i++)
            x[i] += d[i];
    }

    free(r);
    free(d);
    return SPARSE_OK;
}
