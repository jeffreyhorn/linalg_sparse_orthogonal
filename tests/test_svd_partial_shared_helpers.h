#ifndef TEST_SVD_PARTIAL_SHARED_HELPERS_H
#define TEST_SVD_PARTIAL_SHARED_HELPERS_H

#include "sparse_matrix.h"
#include "sparse_svd.h"
#include "test_framework.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int partial_svd_max_residuals(const SparseMatrix *A, const sparse_svd_t *svd, idx_t k,
                                     double *max_av_resid, double *max_atu_resid) {
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(svd);
    ASSERT_NOT_NULL(max_av_resid);
    if (!A || !svd || !max_av_resid)
        return 0;
    ASSERT_NOT_NULL(svd->U);
    ASSERT_NOT_NULL(svd->Vt);
    ASSERT_NOT_NULL(svd->sigma);
    ASSERT_TRUE(svd->m > 0);
    ASSERT_TRUE(svd->n > 0);
    ASSERT_TRUE(svd->k > 0);
    if (!svd->U || !svd->Vt || !svd->sigma || svd->m <= 0 || svd->n <= 0 || svd->k <= 0)
        return 0;

    idx_t n_vecs = k;
    if (n_vecs > svd->k)
        n_vecs = svd->k;
    if (n_vecs <= 0)
        return 0;

    SparseMatrix *At = NULL;
    if (max_atu_resid) {
        At = sparse_transpose(A);
        ASSERT_NOT_NULL(At);
        if (!At)
            return 0;
    }

    double *Av = calloc((size_t)svd->m, sizeof(double));
    double *Atu = max_atu_resid ? calloc((size_t)svd->n, sizeof(double)) : NULL;
    double *v = calloc((size_t)svd->n, sizeof(double));
    ASSERT_NOT_NULL(Av);
    if (max_atu_resid)
        ASSERT_NOT_NULL(Atu);
    ASSERT_NOT_NULL(v);
    if (!Av || (max_atu_resid && !Atu) || !v) {
        free(Av);
        free(Atu);
        free(v);
        sparse_free(At);
        return 0;
    }

    *max_av_resid = 0.0;
    if (max_atu_resid)
        *max_atu_resid = 0.0;
    for (idx_t s = 0; s < n_vecs; s++) {
        for (idx_t j = 0; j < svd->n; j++)
            v[j] = svd->Vt[(size_t)j * (size_t)svd->k + (size_t)s];

        memset(Av, 0, (size_t)svd->m * sizeof(double));
        sparse_matvec(A, v, Av);
        double av_resid = 0.0;
        for (idx_t i = 0; i < svd->m; i++) {
            double diff = Av[i] - svd->sigma[s] * svd->U[(size_t)s * (size_t)svd->m + (size_t)i];
            av_resid += diff * diff;
        }
        av_resid = sqrt(av_resid);
        if (av_resid > *max_av_resid)
            *max_av_resid = av_resid;

        if (!max_atu_resid)
            continue;

        memset(Atu, 0, (size_t)svd->n * sizeof(double));
        sparse_matvec(At, &svd->U[(size_t)s * (size_t)svd->m], Atu);
        double atu_resid = 0.0;
        for (idx_t j = 0; j < svd->n; j++) {
            double diff = Atu[j] - svd->sigma[s] * v[j];
            atu_resid += diff * diff;
        }
        atu_resid = sqrt(atu_resid);
        if (atu_resid > *max_atu_resid)
            *max_atu_resid = atu_resid;
    }

    free(Av);
    free(Atu);
    free(v);
    sparse_free(At);
    return 1;
}

static int partial_svd_max_triplet_residuals(const SparseMatrix *A, const sparse_svd_t *svd,
                                             idx_t k, double *max_av_resid, double *max_atu_resid) {
    ASSERT_NOT_NULL(max_atu_resid);
    if (!max_atu_resid)
        return 0;
    return partial_svd_max_residuals(A, svd, k, max_av_resid, max_atu_resid);
}

static double partial_svd_u_coordinate_range_projector_error(const sparse_svd_t *svd,
                                                             idx_t range_rank) {
    double frob_sq = 0.0;
    for (idx_t row = 0; row < svd->m; row++) {
        for (idx_t col = 0; col < svd->m; col++) {
            double actual = 0.0;
            for (idx_t s = 0; s < range_rank; s++)
                actual += svd->U[(size_t)s * (size_t)svd->m + (size_t)row] *
                          svd->U[(size_t)s * (size_t)svd->m + (size_t)col];
            double expected = (row == col && row < range_rank) ? 1.0 : 0.0;
            double diff = actual - expected;
            frob_sq += diff * diff;
        }
    }
    return sqrt(frob_sq);
}

static double partial_svd_v_coordinate_range_projector_error(const sparse_svd_t *svd,
                                                             idx_t range_rank) {
    double frob_sq = 0.0;
    idx_t vt_ld = svd->economy ? svd->k : svd->n;
    for (idx_t row = 0; row < svd->n; row++) {
        for (idx_t col = 0; col < svd->n; col++) {
            double actual = 0.0;
            for (idx_t s = 0; s < range_rank; s++)
                actual += svd->Vt[(size_t)row * (size_t)vt_ld + (size_t)s] *
                          svd->Vt[(size_t)col * (size_t)vt_ld + (size_t)s];
            double expected = (row == col && row < range_rank) ? 1.0 : 0.0;
            double diff = actual - expected;
            frob_sq += diff * diff;
        }
    }
    return sqrt(frob_sq);
}

#endif
