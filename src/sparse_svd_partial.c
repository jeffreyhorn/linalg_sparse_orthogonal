#include "sparse_alloc_internal.h"
#include "sparse_matrix_internal.h"
#include "sparse_matrix_state_internal.h"
#include "sparse_svd.h"
#include "sparse_svd_internal.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static sparse_err_t partial_svd_full_truncate(const SparseMatrix *A, idx_t kk,
                                              const sparse_svd_opts_t *opts, sparse_svd_t *svd) {
    sparse_svd_opts_t full_opts = {0};
    if (opts)
        full_opts = *opts;
    full_opts.max_iter = 0;

    sparse_svd_t full = {0};
    sparse_err_t err = sparse_svd_compute(A, opts ? &full_opts : NULL, &full);
    if (err != SPARSE_OK)
        return err;

    size_t kk_size = 0;
    size_t m_size = 0;
    size_t n_size = 0;
    if (sparse_idx_to_size_checked(kk, &kk_size) || sparse_idx_to_size_checked(full.m, &m_size) ||
        sparse_idx_to_size_checked(full.n, &n_size)) {
        sparse_svd_free(&full);
        return SPARSE_ERR_ALLOC;
    }

    double *sigma = NULL;
    if (sparse_malloc_array(kk_size, sizeof(double), (void **)&sigma) != SPARSE_OK) {
        sparse_svd_free(&full);
        return SPARSE_ERR_ALLOC;
    }
    memcpy(sigma, full.sigma, kk_size * sizeof(double));

    svd->sigma = sigma;
    svd->m = full.m;
    svd->n = full.n;
    svd->k = kk;
    svd->economy = opts ? opts->economy : 0;

    if (opts && opts->compute_uv) {
        size_t u_count = 0;
        size_t vt_count = 0;
        if (sparse_size_mul_overflow(m_size, kk_size, &u_count) ||
            sparse_size_mul_overflow(n_size, kk_size, &vt_count)) {
            sparse_svd_free(&full);
            sparse_svd_free(svd);
            return SPARSE_ERR_ALLOC;
        }

        double *U = NULL;
        double *Vt = NULL;
        if (sparse_malloc_array(u_count, sizeof(double), (void **)&U) != SPARSE_OK ||
            sparse_malloc_array(vt_count, sizeof(double), (void **)&Vt) != SPARSE_OK) {
            free(U);
            free(Vt);
            sparse_svd_free(&full);
            sparse_svd_free(svd);
            return SPARSE_ERR_ALLOC;
        }

        for (idx_t s = 0; s < kk; s++) {
            memcpy(&U[(size_t)s * m_size], &full.U[(size_t)s * m_size], m_size * sizeof(double));
        }
        for (idx_t j = 0; j < full.n; j++) {
            for (idx_t s = 0; s < kk; s++) {
                Vt[(size_t)j * kk_size + (size_t)s] =
                    full.Vt[(size_t)j * (size_t)full.k + (size_t)s];
            }
        }

        svd->U = U;
        svd->Vt = Vt;
        svd->economy = 1;
    }

    sparse_svd_free(&full);
    return SPARSE_OK;
}

sparse_err_t sparse_svd_partial(const SparseMatrix *A, idx_t kk, const sparse_svd_opts_t *opts,
                                sparse_svd_t *svd) {
    if (!svd)
        return SPARSE_ERR_NULL;
    memset(svd, 0, sizeof(*svd));
    if (!A)
        return SPARSE_ERR_NULL;

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);
    idx_t kmax = (m < n) ? m : n;

    if (kk <= 0 || kk > kmax)
        return SPARSE_ERR_BADARG;

    /* Reject negative max_iter/tol for consistent API semantics */
    if (opts && (opts->max_iter < 0 || opts->tol < 0.0))
        return SPARSE_ERR_BADARG;

    /* Enforce compute_uv requires economy=1 (same as sparse_svd_compute) */
    if (opts && opts->compute_uv && !opts->economy)
        return SPARSE_ERR_BADARG;

    /* Partial SVD consumes the original physical row/col view. */
    if (sparse_matrix_require_original_row_col_state(A) != SPARSE_OK)
        return SPARSE_ERR_BADARG;

    svd->m = m;
    svd->n = n;
    svd->k = kk;

    /* Use more Lanczos steps than k for better convergence.
     * Clustered spectra (e.g. stiffness matrices) need a larger subspace.
     * Compute in int64_t to avoid overflow when kk is large. */
    int64_t lanczos_k64 = 2LL * (int64_t)kk + 20;
    if (lanczos_k64 < (int64_t)kk + 30)
        lanczos_k64 = (int64_t)kk + 30;
    if (lanczos_k64 > (int64_t)kmax)
        lanczos_k64 = (int64_t)kmax;
    idx_t lanczos_k = (idx_t)lanczos_k64;
    if ((!opts || opts->max_iter == 0) && lanczos_k == kmax)
        return partial_svd_full_truncate(A, kk, opts, svd);

    size_t m_size = 0;
    size_t n_size = 0;
    size_t lanczos_k_size = 0;
    size_t lanczos_k1_size = 0;
    size_t kk_size = 0;
    if (sparse_idx_to_size_checked(lanczos_k, &lanczos_k_size) ||
        sparse_size_add_overflow(lanczos_k_size, 1, &lanczos_k1_size))
        return SPARSE_ERR_ALLOC;
    if (sparse_idx_to_size_checked(m, &m_size) || sparse_idx_to_size_checked(n, &n_size) ||
        sparse_idx_to_size_checked(kk, &kk_size)) {
        return SPARSE_ERR_ALLOC;
    }

    /* Transpose for A^T * x operations */
    SparseMatrix *At = sparse_transpose(A);
    if (!At)
        return SPARSE_ERR_ALLOC;

    /* Allocate Lanczos vectors: P (m x lanczos_k) and Q (n x (lanczos_k+1)) */
    size_t sz_p, sz_q;
    size_t sz_alpha = lanczos_k_size;
    size_t sz_beta = lanczos_k1_size;
    if (sparse_size_mul_overflow(m_size, lanczos_k_size, &sz_p) ||
        sparse_size_mul_overflow(n_size, lanczos_k1_size, &sz_q)) {
        sparse_free(At);
        return SPARSE_ERR_ALLOC;
    }
    double *P = NULL;
    double *Q = NULL;
    double *alpha = NULL;
    double *beta = NULL;
    if (sparse_calloc_array(sz_p, sizeof(double), (void **)&P) != SPARSE_OK ||
        sparse_calloc_array(sz_q, sizeof(double), (void **)&Q) != SPARSE_OK ||
        sparse_calloc_array(sz_alpha, sizeof(double), (void **)&alpha) != SPARSE_OK ||
        sparse_calloc_array(sz_beta, sizeof(double), (void **)&beta) != SPARSE_OK) {
        free(P);
        free(Q);
        free(alpha);
        free(beta);
        sparse_free(At);
        return SPARSE_ERR_ALLOC;
    }

    /* Initialize q_0 = [1/sqrt(n), ...] (unit vector) */
    {
        double inv_sqrt_n = 1.0 / sqrt((double)n);
        for (idx_t i = 0; i < n; i++)
            Q[i] = inv_sqrt_n;
    }

    beta[0] = 0.0;

    for (idx_t j = 0; j < lanczos_k; j++) {
        double *qj = &Q[(size_t)j * (size_t)n];
        double *pj = &P[(size_t)j * (size_t)m];

        /* p_j = A * q_j */
        {
            sparse_err_t mv_err = sparse_matvec(A, qj, pj);
            if (mv_err != SPARSE_OK) {
                free(P);
                free(Q);
                free(alpha);
                free(beta);
                sparse_free(At);
                return mv_err;
            }
        }

        /* p_j = p_j - beta_j * p_{j-1} */
        if (j > 0) {
            double *pjm1 = &P[(size_t)(j - 1) * (size_t)m];
            for (idx_t i = 0; i < m; i++)
                pj[i] -= beta[j] * pjm1[i];
        }

        /* Reorthogonalize p_j against p_0..p_{j-1} */
        for (idx_t r = 0; r < j; r++) {
            double *pr = &P[(size_t)r * (size_t)m];
            double dot = 0.0;
            for (idx_t i = 0; i < m; i++)
                dot += pr[i] * pj[i];
            for (idx_t i = 0; i < m; i++)
                pj[i] -= dot * pr[i];
        }

        /* alpha_j = ||p_j|| */
        double anorm = 0.0;
        for (idx_t i = 0; i < m; i++)
            anorm += pj[i] * pj[i];
        anorm = sqrt(anorm);
        alpha[j] = anorm;

        if (anorm > sparse_rel_tol(0, DROP_TOL)) {
            double inv = 1.0 / anorm;
            for (idx_t i = 0; i < m; i++)
                pj[i] *= inv;
        }

        /* r = A^T * p_j - alpha_j * q_j */
        double *qj1 = &Q[(size_t)(j + 1) * (size_t)n];
        {
            sparse_err_t mv_err = sparse_matvec(At, pj, qj1);
            if (mv_err != SPARSE_OK) {
                free(P);
                free(Q);
                free(alpha);
                free(beta);
                sparse_free(At);
                return mv_err;
            }
        }
        for (idx_t i = 0; i < n; i++)
            qj1[i] -= alpha[j] * qj[i];

        /* Reorthogonalize q_{j+1} against q_0..q_j */
        for (idx_t r = 0; r <= j; r++) {
            double *qr = &Q[(size_t)r * (size_t)n];
            double dot = 0.0;
            for (idx_t i = 0; i < n; i++)
                dot += qr[i] * qj1[i];
            for (idx_t i = 0; i < n; i++)
                qj1[i] -= dot * qr[i];
        }

        /* beta_{j+1} = ||q_{j+1}|| */
        double bnorm = 0.0;
        for (idx_t i = 0; i < n; i++)
            bnorm += qj1[i] * qj1[i];
        bnorm = sqrt(bnorm);
        beta[j + 1] = bnorm;

        if (j + 1 < lanczos_k && bnorm > sparse_rel_tol(0, DROP_TOL)) {
            double inv = 1.0 / bnorm;
            for (idx_t i = 0; i < n; i++)
                qj1[i] *= inv;
        }
    }

    sparse_free(At);

    /* Check if singular vectors are requested.
     * Free P/Q early when not needed to reduce peak memory. */
    int compute_uv = opts ? opts->compute_uv : 0;
    if (!compute_uv) {
        free(P);
        free(Q);
        P = NULL;
        Q = NULL;
    }

    /* Now we have a lanczos_k x lanczos_k bidiagonal with
     * diag=alpha, superdiag=beta[1..lanczos_k-1] */
    double *bd_super = NULL;
    if (lanczos_k > 1) {
        if (sparse_malloc_idx_array(lanczos_k - 1, sizeof(double), (void **)&bd_super) !=
            SPARSE_OK) {
            free(alpha);
            free(beta);
            free(P);
            free(Q);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < lanczos_k - 1; i++)
            bd_super[i] = beta[i + 1];
    }
    free(beta);

    /* Allocate small U and V matrices for bidiag SVD if computing vectors */
    double *U_small = NULL;
    double *V_small = NULL;
    if (compute_uv) {
        size_t lk2;
        if (sparse_size_mul_overflow(lanczos_k_size, lanczos_k_size, &lk2)) {
            free(alpha);
            free(bd_super);
            free(P);
            free(Q);
            return SPARSE_ERR_ALLOC;
        }
        if (sparse_calloc_array(lk2, sizeof(double), (void **)&U_small) != SPARSE_OK ||
            sparse_calloc_array(lk2, sizeof(double), (void **)&V_small) != SPARSE_OK) {
            free(alpha);
            free(bd_super);
            free(P);
            free(Q);
            free(U_small);
            free(V_small);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < lanczos_k; i++) {
            U_small[(size_t)i * (size_t)lanczos_k + (size_t)i] = 1.0;
            V_small[(size_t)i * (size_t)lanczos_k + (size_t)i] = 1.0;
        }
    }

    /* Run bidiagonal SVD iteration on the small lanczos_k x lanczos_k bidiag */
    idx_t max_iter_val = opts ? opts->max_iter : 0;
    double tol_val = opts ? opts->tol : 0.0;
    sparse_err_t err = bidiag_svd_iterate(alpha, bd_super, lanczos_k, U_small, lanczos_k, V_small,
                                          lanczos_k, max_iter_val, tol_val);
    free(bd_super);

    if (err != SPARSE_OK) {
        free(alpha);
        free(P);
        free(Q);
        free(U_small);
        free(V_small);
        return err;
    }

    /* Sort singular values descending, tracking permutation only when
     * compute_uv is set (perm is only needed for vector recovery). */
    idx_t *perm = NULL;
    if (compute_uv) {
        if (sparse_malloc_idx_array(lanczos_k, sizeof(idx_t), (void **)&perm) != SPARSE_OK) {
            free(alpha);
            free(P);
            free(Q);
            free(U_small);
            free(V_small);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < lanczos_k; i++)
            perm[i] = i;
    }

    for (idx_t i = 0; i < lanczos_k; i++)
        if (alpha[i] < 0.0)
            alpha[i] = -alpha[i];

    for (idx_t i = 0; i < lanczos_k - 1; i++) {
        idx_t best = i;
        for (idx_t j = i + 1; j < lanczos_k; j++)
            if (alpha[j] > alpha[best])
                best = j;
        if (best != i) {
            double tmp = alpha[i];
            alpha[i] = alpha[best];
            alpha[best] = tmp;
            if (perm) {
                idx_t ptmp = perm[i];
                perm[i] = perm[best];
                perm[best] = ptmp;
            }
        }
    }

    double *sigma = NULL;
    if (sparse_malloc_array(kk_size, sizeof(double), (void **)&sigma) != SPARSE_OK) {
        free(alpha);
        free(perm);
        free(P);
        free(Q);
        free(U_small);
        free(V_small);
        return SPARSE_ERR_ALLOC;
    }
    memcpy(sigma, alpha, kk_size * sizeof(double));
    free(alpha);

    svd->sigma = sigma;

    if (compute_uv) {
        size_t sz_u_out, sz_vt_out;
        if (sparse_size_mul_overflow(m_size, kk_size, &sz_u_out) ||
            sparse_size_mul_overflow(kk_size, n_size, &sz_vt_out)) {
            free(perm);
            free(P);
            free(Q);
            free(U_small);
            free(V_small);
            sparse_svd_free(svd);
            return SPARSE_ERR_ALLOC;
        }
        double *U_out = NULL;
        double *Vt_out = NULL;
        if (sparse_calloc_array(sz_u_out, sizeof(double), (void **)&U_out) != SPARSE_OK ||
            sparse_calloc_array(sz_vt_out, sizeof(double), (void **)&Vt_out) != SPARSE_OK) {
            free(U_out);
            free(Vt_out);
            free(perm);
            free(P);
            free(Q);
            free(U_small);
            free(V_small);
            sparse_svd_free(svd);
            return SPARSE_ERR_ALLOC;
        }

        /* U_out[s*m + i] = sum_t P[t*m + i] * U_small[perm[s]*lanczos_k + t] */
        for (idx_t s = 0; s < kk; s++) {
            idx_t ps = perm[s];
            for (idx_t t = 0; t < lanczos_k; t++) {
                double coeff = U_small[(size_t)ps * (size_t)lanczos_k + (size_t)t];
                if (coeff == 0.0)
                    continue;
                double *p_col = &P[(size_t)t * (size_t)m];
                double *u_col = &U_out[(size_t)s * (size_t)m];
                for (idx_t i = 0; i < m; i++)
                    u_col[i] += coeff * p_col[i];
            }
        }

        /* Vt_out[j*kk + s] = sum_t Q[t*n + j] * V_small[perm[s]*lanczos_k + t] */
        for (idx_t s = 0; s < kk; s++) {
            idx_t ps = perm[s];
            for (idx_t t = 0; t < lanczos_k; t++) {
                double coeff = V_small[(size_t)ps * (size_t)lanczos_k + (size_t)t];
                if (coeff == 0.0)
                    continue;
                double *q_col = &Q[(size_t)t * (size_t)n];
                for (idx_t j = 0; j < n; j++)
                    Vt_out[(size_t)j * (size_t)kk + (size_t)s] += coeff * q_col[j];
            }
        }

        svd->U = U_out;
        svd->Vt = Vt_out;
        svd->economy = 1;
    }

    free(perm);
    free(P);
    free(Q);
    free(U_small);
    free(V_small);
    return SPARSE_OK;
}
