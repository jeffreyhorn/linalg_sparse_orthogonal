/**
 * @file sparse_ldlt_csc_supernodal.c
 * @brief Supernodal helper cluster for the LDL^T CSC numeric backend.
 *
 * Owns the extracted dense-panel path used by
 * `ldlt_csc_eliminate_supernodal`: supernode extract/writeback,
 * diagonal-block factor/update, panel solve, and the supernodal
 * driver-local row-map lookup helper.  Public orchestration,
 * lifecycle/conversion, and the scalar/native kernel remain in
 * `sparse_ldlt_csc.c`.
 */

#include "sparse_ldlt_csc_internal.h"

#include <math.h>
#include <stdlib.h>

/* Local copy of `chol_csc_bsearch_row_map` (static in
 * sparse_chol_csc_supernodal.c).  Could be shared via the chol internal header,
 * but it's a five-line function and duplicating keeps the LDL^T side
 * loosely coupled. */
static idx_t ldlt_csc_bsearch_row_map(const idx_t *row_map, idx_t panel_height, idx_t target) {
    idx_t lo = 0;
    idx_t hi = panel_height;
    while (lo < hi) {
        idx_t mid = lo + (hi - lo) / 2;
        if (row_map[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < panel_height && row_map[lo] == target) {
        return lo;
    }
    return panel_height;
}

sparse_err_t ldlt_csc_supernode_extract(const LdltCsc *F, idx_t s_start, idx_t s_size,
                                        double *dense, idx_t lda, idx_t *row_map,
                                        idx_t *panel_height_out) {
    if (!F || !dense || !row_map || !panel_height_out)
        return SPARSE_ERR_NULL;
    if (!F->L)
        return SPARSE_ERR_NULL;
    if (s_start < 0 || s_size < 1 || s_start > F->n - s_size)
        return SPARSE_ERR_BADARG;

    const CholCsc *L = F->L;
    idx_t first_start = L->col_ptr[s_start];
    idx_t panel_height = L->col_ptr[s_start + 1] - first_start;
    if (lda < panel_height)
        return SPARSE_ERR_BADARG;
    if (panel_height < s_size)
        return SPARSE_ERR_BADARG;

    /* Seed row_map from the first column.  Supernodal invariant:
     * the first s_size entries are [s_start, ..., s_start + s_size - 1]
     * in order (the diagonal block precedes the shared panel rows). */
    for (idx_t i = 0; i < panel_height; i++)
        row_map[i] = L->row_idx[first_start + i];
    for (idx_t i = 0; i < s_size; i++) {
        if (row_map[i] != s_start + i)
            return SPARSE_ERR_BADARG;
    }
    *panel_height_out = panel_height;

    for (idx_t j = 0; j < s_size; j++) {
        idx_t c = s_start + j;
        idx_t cstart = L->col_ptr[c];
        idx_t cend = L->col_ptr[c + 1];
        for (idx_t p = cstart; p < cend; p++) {
            idx_t row = L->row_idx[p];
            idx_t local = ldlt_csc_bsearch_row_map(row_map, panel_height, row);
            if (local >= panel_height)
                return SPARSE_ERR_BADARG;
            dense[local + j * lda] = L->values[p];
        }
    }

    return SPARSE_OK;
}

sparse_err_t ldlt_csc_supernode_writeback(LdltCsc *F, idx_t s_start, idx_t s_size,
                                          const double *dense, idx_t lda, const idx_t *row_map,
                                          idx_t panel_height, const double *D_block,
                                          const double *D_offdiag_block,
                                          const idx_t *pivot_size_block, double drop_tol) {
    if (!F || !dense || !row_map || !D_block || !D_offdiag_block || !pivot_size_block)
        return SPARSE_ERR_NULL;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size)
        return SPARSE_ERR_NULL;
    if (s_start < 0 || s_size < 1 || s_start > F->n - s_size)
        return SPARSE_ERR_BADARG;
    if (panel_height < s_size || lda < panel_height)
        return SPARSE_ERR_BADARG;

    CholCsc *L = F->L;

    /* Precompute per-column drop thresholds matching the scalar
     * `chol_csc_gather` invocations in `ldlt_csc_eliminate_native`:
     *   - 1x1 pivot at column j: threshold = drop_tol  (the scalar
     *     path passes raw drop_tol because dense_col[j] == 1.0).
     *   - 2x2 pair (j_first, j_first+1): threshold for both columns =
     *     drop_tol * (|d11| + |d22| + |d21|).  We disambiguate first
     *     vs second via D_offdiag_block (non-zero on first, zero on
     *     second) — robust against adjacent 2x2 pairs where
     *     pivot_size_block alone is ambiguous.
     *
     * Thresholds are allocated on the heap with an explicit overflow
     * guard. Supernodes are still bounded for cache efficiency, but the
     * heap allocation keeps this helper from depending on large
     * stack-resident scratch storage. */
    if ((size_t)s_size > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *thresholds = malloc((size_t)s_size * sizeof(double));
    if (!thresholds)
        return SPARSE_ERR_ALLOC;

    for (idx_t j = 0; j < s_size; j++) {
        if (pivot_size_block[j] == 1) {
            thresholds[j] = drop_tol;
        } else if (pivot_size_block[j] == 2) {
            /* Identify the first of this pair to source d11/d22/d21. */
            idx_t j_first = (D_offdiag_block[j] != 0.0) ? j : j - 1;
            if (j_first < 0 || j_first + 1 >= s_size) {
                free(thresholds);
                return SPARSE_ERR_BADARG;
            }
            double d11 = D_block[j_first];
            double d22 = D_block[j_first + 1];
            double d21 = D_offdiag_block[j_first];
            double bscale = fabs(d11) + fabs(d22) + fabs(d21);
            thresholds[j] = drop_tol * bscale;
        } else {
            free(thresholds);
            return SPARSE_ERR_BADARG;
        }
    }

    /* Per-column gather: walk the existing CSC slot, translate row ->
     * local via row_map, write dense[local + j*lda] back into
     * values[p].  Drop below-diagonal entries below threshold; never
     * drop the diagonal (local == j by the supernodal invariant on
     * row_map's first s_size slots). */
    for (idx_t j = 0; j < s_size; j++) {
        idx_t c = s_start + j;
        idx_t cstart = L->col_ptr[c];
        idx_t cend = L->col_ptr[c + 1];
        double threshold = thresholds[j];
        for (idx_t p = cstart; p < cend; p++) {
            idx_t row = L->row_idx[p];
            idx_t local = ldlt_csc_bsearch_row_map(row_map, panel_height, row);
            if (local >= panel_height) {
                free(thresholds);
                return SPARSE_ERR_BADARG;
            }
            double v = dense[local + j * lda];
            if (local != j && fabs(v) < threshold)
                v = 0.0;
            L->values[p] = v;
        }

        /* Distribute the dense-block-factor's auxiliary outputs into
         * the LdltCsc.  These are owned by the caller's local scratch
         * and copied verbatim here. */
        F->D[s_start + j] = D_block[j];
        F->D_offdiag[s_start + j] = D_offdiag_block[j];
        F->pivot_size[s_start + j] = pivot_size_block[j];
    }

    free(thresholds);
    return SPARSE_OK;
}

sparse_err_t ldlt_csc_supernode_eliminate_diag(const LdltCsc *F, idx_t s_start, idx_t s_size,
                                               double *dense, idx_t lda, const idx_t *row_map,
                                               idx_t panel_height, double *D_block,
                                               double *D_offdiag_block, idx_t *pivot_size_block,
                                               double tol) {
    if (!F || !F->L || !F->D || !F->D_offdiag || !F->pivot_size || !dense || !row_map || !D_block ||
        !D_offdiag_block || !pivot_size_block)
        return SPARSE_ERR_NULL;
    if (s_start < 0 || s_size < 1 || s_start > F->n - s_size)
        return SPARSE_ERR_BADARG;
    if (panel_height < s_size || lda < panel_height)
        return SPARSE_ERR_BADARG;

    const CholCsc *L = F->L;

    if ((size_t)s_size > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *L_col_k = malloc((size_t)s_size * sizeof(double));
    double *L_col_k1 = malloc((size_t)s_size * sizeof(double));
    if (!L_col_k || !L_col_k1) {
        free(L_col_k);
        free(L_col_k1);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t k = 0; k < s_start;) {
        idx_t pk = F->pivot_size[k];
        int is_2x2 = (pk == 2 && F->D_offdiag[k] != 0.0);
        if (is_2x2 && k + 1 >= s_start) {
            free(L_col_k);
            free(L_col_k1);
            return SPARSE_ERR_BADARG;
        }
        idx_t step = is_2x2 ? 2 : 1;

        for (idx_t j = 0; j < s_size; j++)
            L_col_k[j] = 0.0;
        if (is_2x2)
            for (idx_t j = 0; j < s_size; j++)
                L_col_k1[j] = 0.0;

        int saw_super_row = 0;
        idx_t cstart = L->col_ptr[k];
        idx_t cend = L->col_ptr[k + 1];
        for (idx_t p = cstart; p < cend; p++) {
            idx_t row = L->row_idx[p];
            if (row < s_start)
                continue;
            if (row >= s_start + s_size)
                break;
            L_col_k[row - s_start] = L->values[p];
            saw_super_row = 1;
        }
        idx_t cstart1 = 0;
        idx_t cend1 = 0;
        if (is_2x2) {
            cstart1 = L->col_ptr[k + 1];
            cend1 = L->col_ptr[k + 2];
            for (idx_t p = cstart1; p < cend1; p++) {
                idx_t row = L->row_idx[p];
                if (row < s_start)
                    continue;
                if (row >= s_start + s_size)
                    break;
                L_col_k1[row - s_start] = L->values[p];
                saw_super_row = 1;
            }
        }

        if (!saw_super_row) {
            k += step;
            continue;
        }

        if (!is_2x2) {
            double dk = F->D[k];
            for (idx_t p = cstart; p < cend; p++) {
                idx_t row = L->row_idx[p];
                idx_t local = ldlt_csc_bsearch_row_map(row_map, panel_height, row);
                if (local >= panel_height)
                    continue;
                double v_r_k = L->values[p];
                double factor = v_r_k * dk;
                for (idx_t j = 0; j < s_size; j++) {
                    double ljk = L_col_k[j];
                    if (ljk != 0.0)
                        dense[local + j * lda] -= factor * ljk;
                }
            }
        } else {
            double dk = F->D[k];
            double dk1 = F->D[k + 1];
            double doff = F->D_offdiag[k];

            for (idx_t p = cstart; p < cend; p++) {
                idx_t row = L->row_idx[p];
                idx_t local = ldlt_csc_bsearch_row_map(row_map, panel_height, row);
                if (local >= panel_height)
                    continue;
                double v_r_k = L->values[p];
                for (idx_t j = 0; j < s_size; j++) {
                    double term = dk * L_col_k[j] + doff * L_col_k1[j];
                    if (term != 0.0)
                        dense[local + j * lda] -= v_r_k * term;
                }
            }

            for (idx_t p = cstart1; p < cend1; p++) {
                idx_t row = L->row_idx[p];
                idx_t local = ldlt_csc_bsearch_row_map(row_map, panel_height, row);
                if (local >= panel_height)
                    continue;
                double v_r_k1 = L->values[p];
                for (idx_t j = 0; j < s_size; j++) {
                    double term = dk1 * L_col_k1[j] + doff * L_col_k[j];
                    if (term != 0.0)
                        dense[local + j * lda] -= v_r_k1 * term;
                }
            }
        }

        k += step;
    }

    free(L_col_k);
    free(L_col_k1);

    for (idx_t j = 0; j < s_size; j++) {
        for (idx_t i = j + 1; i < s_size; i++)
            dense[j + i * lda] = dense[i + j * lda];
    }

    sparse_err_t err = ldlt_dense_factor(dense, D_block, D_offdiag_block, pivot_size_block, s_size,
                                         lda, tol, NULL);
    if (err != SPARSE_OK)
        return err;

    for (idx_t j = 0; j < s_size; j++) {
        if (pivot_size_block[j] != F->pivot_size[s_start + j])
            return SPARSE_ERR_PIVOT_REJECTED;
    }

    return SPARSE_OK;
}

sparse_err_t ldlt_csc_supernode_eliminate_panel(const double *L_diag, const double *D_block,
                                                const double *D_offdiag_block,
                                                const idx_t *pivot_size_block, idx_t s_size,
                                                idx_t lda_diag, double *panel, idx_t lda_panel,
                                                idx_t panel_rows) {
    if (!L_diag || !D_block || !D_offdiag_block || !pivot_size_block)
        return SPARSE_ERR_NULL;
    if (s_size < 1 || lda_diag < s_size || panel_rows < 0)
        return SPARSE_ERR_BADARG;
    if (panel_rows == 0)
        return SPARSE_OK;
    if (!panel)
        return SPARSE_ERR_NULL;
    if (lda_panel < panel_rows)
        return SPARSE_ERR_BADARG;

    if ((size_t)s_size > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *row_buf = malloc((size_t)s_size * sizeof(double));
    if (!row_buf)
        return SPARSE_ERR_ALLOC;

    for (idx_t i = 0; i < panel_rows; i++) {
        for (idx_t j = 0; j < s_size; j++)
            row_buf[j] = panel[i + j * lda_panel];

        for (idx_t j = 0; j < s_size; j++) {
            double sum = row_buf[j];
            for (idx_t q = 0; q < j; q++)
                sum -= L_diag[j + q * lda_diag] * row_buf[q];
            row_buf[j] = sum;
        }

        idx_t j = 0;
        while (j < s_size) {
            if (pivot_size_block[j] == 1) {
                double dk = D_block[j];
                if (dk == 0.0) {
                    free(row_buf);
                    return SPARSE_ERR_SINGULAR;
                }
                row_buf[j] = row_buf[j] / dk;
                j += 1;
            } else if (pivot_size_block[j] == 2 && j + 1 < s_size && pivot_size_block[j + 1] == 2 &&
                       D_offdiag_block[j] != 0.0) {
                double d11 = D_block[j];
                double d22 = D_block[j + 1];
                double d21 = D_offdiag_block[j];
                double det = d11 * d22 - d21 * d21;
                if (det == 0.0) {
                    free(row_buf);
                    return SPARSE_ERR_SINGULAR;
                }
                double inv_det = 1.0 / det;
                double y0 = row_buf[j];
                double y1 = row_buf[j + 1];
                row_buf[j] = (d22 * y0 - d21 * y1) * inv_det;
                row_buf[j + 1] = (-d21 * y0 + d11 * y1) * inv_det;
                j += 2;
            } else {
                free(row_buf);
                return SPARSE_ERR_BADARG;
            }
        }

        for (idx_t q = 0; q < s_size; q++)
            panel[i + q * lda_panel] = row_buf[q];
    }

    free(row_buf);
    return SPARSE_OK;
}
