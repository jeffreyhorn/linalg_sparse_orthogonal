#ifndef TEST_SVD_PARTIAL_HELPERS_H
#define TEST_SVD_PARTIAL_HELPERS_H

/* ═══════════════════════════════════════════════════════════════════════
 * Partial SVD via Lanczos bidiagonalization (Sprint 8 Day 11)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_partial_svd_null(void) {
    sparse_svd_t svd;
    ASSERT_ERR(sparse_svd_partial(NULL, 3, NULL, &svd), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_svd_partial(NULL, 3, NULL, NULL), SPARSE_ERR_NULL);
}

static void test_partial_svd_bad_k(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    sparse_svd_t svd;
    ASSERT_ERR(sparse_svd_partial(A, 0, NULL, &svd), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_svd_partial(A, -1, NULL, &svd), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_svd_partial(A, 6, NULL, &svd), SPARSE_ERR_BADARG);

    sparse_free(A);
}

static void test_partial_svd_diag_10x10(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(n - i));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 3, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 3);
    ASSERT_EQ(partial.m, n);
    ASSERT_EQ(partial.n, n);
    for (idx_t i = 0; i < 3; i++) {
        printf("    partial sigma[%d]=%.6f, full sigma[%d]=%.6f\n", (int)i, partial.sigma[i],
               (int)i, full.sigma[i]);
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 1e-10);
    }
    ASSERT_TRUE(partial.U == NULL);
    ASSERT_TRUE(partial.Vt == NULL);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

static void test_partial_svd_full_k(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(n - i));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, n, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, n);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 1e-10);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

static void test_partial_svd_dense_8x8(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, 1.0 / (double)(i + j + 1));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t kk = 4;
    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, kk, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, kk);
    printf("    Hilbert 8x8 partial SVD (k=%d):\n", (int)kk);
    for (idx_t i = 0; i < kk; i++) {
        printf("      sigma[%d]: partial=%.8f, full=%.8f\n", (int)i, partial.sigma[i],
               full.sigma[i]);
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 1e-8);
    }

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

static void test_partial_svd_tall(void) {
    idx_t m = 10, nc = 5;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < nc; i++)
        sparse_insert(A, i, i, (double)(nc - i));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 3, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 3);
    for (idx_t i = 0; i < 3; i++)
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 1e-10);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

static void test_partial_svd_wide(void) {
    idx_t m = 5, nc = 10;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        sparse_insert(A, i, i, (double)(m - i));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 3, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 3);
    for (idx_t i = 0; i < 3; i++)
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 0.05 * full.sigma[i]);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

static void test_partial_svd_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t kk = 5;
    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, kk, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, kk);
    printf("    nos4 partial SVD (k=%d):\n", (int)kk);
    for (idx_t i = 0; i < kk; i++) {
        printf("      sigma[%d]: partial=%.6f, full=%.6f\n", (int)i, partial.sigma[i],
               full.sigma[i]);
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 0.1 * full.sigma[i]);
    }

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Partial SVD validation (Sprint 8 Day 12)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_partial_svd_k1(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(n - i));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 1, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 1);
    printf("    k=1: partial=%.6f, full=%.6f\n", partial.sigma[0], full.sigma[0]);
    ASSERT_NEAR(partial.sigma[0], full.sigma[0], 1e-10);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

static void test_partial_svd_rank_deficient(void) {
    SparseMatrix *A = sparse_create(6, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 6; i++) {
        sparse_insert(A, i, 0, (double)(i + 1));
        sparse_insert(A, i, 1, (double)(i + 1));
        sparse_insert(A, i, 2, (double)(i * 2 + 1));
        sparse_insert(A, i, 3, (double)(i * 2 + 1));
    }

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 4, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 4);
    printf("    rank-def: sigma=[%.4f, %.4f, %.4f, %.4f]\n", partial.sigma[0], partial.sigma[1],
           partial.sigma[2], partial.sigma[3]);
    ASSERT_TRUE(partial.sigma[0] > 0.1);
    ASSERT_TRUE(partial.sigma[1] > 0.1);
    ASSERT_NEAR(partial.sigma[2], 0.0, 1e-8);
    ASSERT_NEAR(partial.sigma[3], 0.0, 1e-8);
    ASSERT_NEAR(partial.sigma[0], full.sigma[0], 1e-8);
    ASSERT_NEAR(partial.sigma[1], full.sigma[1], 1e-8);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

static void test_partial_svd_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t kk = 5;
    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, kk, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, kk);
    printf("    west0067 partial SVD (k=%d):\n", (int)kk);
    for (idx_t i = 0; i < kk; i++) {
        printf("      sigma[%d]: partial=%.6f, full=%.6f\n", (int)i, partial.sigma[i],
               full.sigma[i]);
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 0.1 * full.sigma[i]);
    }

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

static void test_partial_svd_descending(void) {
    idx_t n = 15;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0 * (double)(i + 1));
        if (i + 1 < n)
            sparse_insert(A, i, i + 1, 1.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, 1.0);
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 7, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 7);
    for (idx_t i = 0; i < 7; i++)
        ASSERT_TRUE(partial.sigma[i] >= 0.0);
    for (idx_t i = 1; i < 7; i++)
        ASSERT_TRUE(partial.sigma[i] <= partial.sigma[i - 1] + 1e-10);

    printf("    descending: sigma[0]=%.4f, sigma[6]=%.4f\n", partial.sigma[0], partial.sigma[6]);

    sparse_svd_free(&partial);
    sparse_free(A);
}

static void test_partial_svd_timing(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    clock_t t0 = clock();
    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    double full_time = (double)(clock() - t0) / CLOCKS_PER_SEC;
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    t0 = clock();
    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 5, NULL, &partial);
    double partial_time = (double)(clock() - t0) / CLOCKS_PER_SEC;
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    printf("    nos4 timing: full=%.4f s, partial(k=5)=%.4f s\n", full_time, partial_time);
    ASSERT_EQ(partial.k, 5);
    ASSERT_TRUE(partial.sigma[0] > 0.0);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

static void test_partial_svd_nonsymmetric(void) {
    idx_t m = 10, nc = 8;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            if ((i + j) % 3 != 0)
                sparse_insert(A, i, j, (double)(i + 1) / (double)(j + 1));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 4, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 4);
    printf("    non-symmetric 10x8 partial (k=4):\n");
    for (idx_t i = 0; i < 4; i++) {
        printf("      sigma[%d]: partial=%.6f, full=%.6f\n", (int)i, partial.sigma[i],
               full.sigma[i]);
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 0.05 * full.sigma[i] + 1e-10);
    }

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Partial SVD with singular vectors (Sprint 9 Day 4)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_partial_svd_vectors_ortho(void) {
    SparseMatrix *A = sparse_create(10, 10);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 10; i++) {
        sparse_insert(A, i, i, 2.0 * (double)(i + 1));
        if (i + 1 < 10)
            sparse_insert(A, i, i + 1, 1.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, 1.0);
    }

    idx_t k = 3;
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_partial(A, k, &opts, &svd);
    ASSERT_EQ(err, SPARSE_OK);
    ASSERT_NOT_NULL(svd.U);
    ASSERT_NOT_NULL(svd.Vt);

    idx_t m_val = svd.m;
    for (idx_t p = 0; p < k; p++) {
        for (idx_t q = p; q < k; q++) {
            double dot = 0.0;
            for (idx_t i = 0; i < m_val; i++)
                dot += svd.U[(size_t)p * (size_t)m_val + (size_t)i] *
                       svd.U[(size_t)q * (size_t)m_val + (size_t)i];
            double expected = (p == q) ? 1.0 : 0.0;
            if (fabs(dot - expected) > 1e-6)
                printf("    U ortho: (%d,%d) = %.6f (expected %.1f)\n", (int)p, (int)q, dot,
                       expected);
            ASSERT_TRUE(fabs(dot - expected) < 1e-6);
        }
    }

    idx_t n_val = svd.n;
    for (idx_t p = 0; p < k; p++) {
        for (idx_t q = p; q < k; q++) {
            double dot = 0.0;
            for (idx_t j = 0; j < n_val; j++)
                dot += svd.Vt[(size_t)j * (size_t)k + (size_t)p] *
                       svd.Vt[(size_t)j * (size_t)k + (size_t)q];
            double expected = (p == q) ? 1.0 : 0.0;
            if (fabs(dot - expected) > 1e-6)
                printf("    Vt ortho: (%d,%d) = %.6f (expected %.1f)\n", (int)p, (int)q, dot,
                       expected);
            ASSERT_TRUE(fabs(dot - expected) < 1e-6);
        }
    }
    printf("    partial SVD vectors ortho: PASS (k=%d, m=%d, n=%d)\n", (int)k, (int)m_val,
           (int)n_val);

    sparse_svd_free(&svd);
    sparse_free(A);
}

static int partial_svd_max_av_residual(const SparseMatrix *A, const sparse_svd_t *svd, idx_t k,
                                       double *max_resid) {
    if (!A || !svd || !max_resid || !svd->U || !svd->Vt || !svd->sigma)
        return 0;
    if (svd->m <= 0 || svd->n <= 0 || svd->k <= 0)
        return 0;

    idx_t n_vecs = k;
    if (n_vecs > svd->k)
        n_vecs = svd->k;
    if (n_vecs <= 0)
        return 0;

    double *Av = calloc((size_t)svd->m, sizeof(double));
    double *v = calloc((size_t)svd->n, sizeof(double));
    ASSERT_NOT_NULL(Av);
    ASSERT_NOT_NULL(v);
    if (!Av || !v) {
        free(Av);
        free(v);
        return 0;
    }

    *max_resid = 0.0;
    for (idx_t s = 0; s < n_vecs; s++) {
        for (idx_t j = 0; j < svd->n; j++)
            v[j] = svd->Vt[(size_t)j * (size_t)svd->k + (size_t)s];

        memset(Av, 0, (size_t)svd->m * sizeof(double));
        sparse_matvec(A, v, Av);

        double resid = 0.0;
        for (idx_t i = 0; i < svd->m; i++) {
            double diff = Av[i] - svd->sigma[s] * svd->U[(size_t)s * (size_t)svd->m + (size_t)i];
            resid += diff * diff;
        }
        resid = sqrt(resid);
        if (resid > *max_resid)
            *max_resid = resid;
    }

    free(Av);
    free(v);
    return 1;
}

static void test_partial_svd_vectors_Av(void) {
    SparseMatrix *A = sparse_create(8, 8);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 8; i++) {
        sparse_insert(A, i, i, 3.0 * (double)(i + 1));
        if (i + 1 < 8)
            sparse_insert(A, i, i + 1, 1.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, 1.0);
    }

    idx_t k = 3;
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_partial(A, k, &opts, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    double max_resid = 0.0;
    if (!partial_svd_max_av_residual(A, &svd, k, &max_resid)) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }
    printf("    partial SVD A*v ≈ sigma*u: max_resid=%.2e\n", max_resid);
    ASSERT_TRUE(max_resid < 1e-6);

    sparse_svd_free(&svd);
    sparse_free(A);
}

static void test_partial_svd_vectors_vs_full(void) {
    SparseMatrix *A = sparse_create(6, 6);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    double diag_vals[] = {10.0, 7.0, 5.0, 3.0, 2.0, 1.0};
    for (idx_t i = 0; i < 6; i++)
        sparse_insert(A, i, i, diag_vals[i]);

    sparse_svd_opts_t full_opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t full_svd;
    sparse_err_t err = sparse_svd_compute(A, &full_opts, &full_svd);
    ASSERT_EQ(err, SPARSE_OK);

    idx_t k = 3;
    sparse_svd_opts_t part_opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t part_svd;
    err = sparse_svd_partial(A, k, &part_opts, &part_svd);
    ASSERT_EQ(err, SPARSE_OK);

    for (idx_t s = 0; s < k; s++) {
        printf("    sigma[%d]: partial=%.4f, full=%.4f\n", (int)s, part_svd.sigma[s],
               full_svd.sigma[s]);
        ASSERT_TRUE(fabs(part_svd.sigma[s] - full_svd.sigma[s]) < 0.1);
    }

    for (idx_t s = 0; s < k; s++) {
        double dot_u = 0.0;
        for (idx_t i = 0; i < 6; i++)
            dot_u += part_svd.U[(size_t)s * 6 + (size_t)i] * full_svd.U[(size_t)s * 6 + (size_t)i];
        printf("    |u_part . u_full|[%d] = %.4f\n", (int)s, fabs(dot_u));
        ASSERT_TRUE(fabs(fabs(dot_u) - 1.0) < 1e-4);
    }

    sparse_svd_free(&full_svd);
    sparse_svd_free(&part_svd);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Partial SVD vector validation (Sprint 9 Day 5)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_partial_svd_vectors_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    if (lerr != SPARSE_OK || !A) {
        printf("    SKIP: nos4.mtx not found\n");
        return;
    }

    idx_t k = 5;
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t part;
    sparse_err_t err = sparse_svd_partial(A, k, &opts, &part);
    ASSERT_EQ(err, SPARSE_OK);

    sparse_svd_t full;
    err = sparse_svd_compute(A, NULL, &full);
    ASSERT_EQ(err, SPARSE_OK);

    double max_sigma_err = 0.0;
    for (idx_t s = 0; s < k; s++) {
        double e = fabs(part.sigma[s] - full.sigma[s]);
        if (e > max_sigma_err)
            max_sigma_err = e;
    }
    printf("    nos4 partial vectors: max sigma err=%.2e\n", max_sigma_err);

    double *Av = calloc((size_t)part.m, sizeof(double));
    double *v = calloc((size_t)part.n, sizeof(double));
    if (!Av || !v) {
        free(Av);
        free(v);
        sparse_svd_free(&part);
        sparse_svd_free(&full);
        sparse_free(A);
        ASSERT_NOT_NULL(Av);
        return;
    }
    double max_resid = 0.0;
    for (idx_t s = 0; s < k; s++) {
        for (idx_t j = 0; j < part.n; j++)
            v[j] = part.Vt[(size_t)j * (size_t)k + (size_t)s];
        memset(Av, 0, (size_t)part.m * sizeof(double));
        sparse_matvec(A, v, Av);
        double resid = 0.0;
        for (idx_t i = 0; i < part.m; i++) {
            double diff = Av[i] - part.sigma[s] * part.U[(size_t)s * (size_t)part.m + (size_t)i];
            resid += diff * diff;
        }
        resid = sqrt(resid);
        if (resid > max_resid)
            max_resid = resid;
    }
    printf("    nos4 A*v ≈ sigma*u max_resid=%.2e\n", max_resid);
    ASSERT_TRUE(max_resid < 1e-4);

    free(Av);
    free(v);
    sparse_svd_free(&part);
    sparse_svd_free(&full);
    sparse_free(A);
}

static void test_partial_svd_vectors_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    if (lerr != SPARSE_OK || !A) {
        printf("    SKIP: west0067.mtx not found\n");
        return;
    }

    idx_t k = 3;
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t part;
    sparse_err_t err = sparse_svd_partial(A, k, &opts, &part);
    ASSERT_EQ(err, SPARSE_OK);

    double *Av = calloc((size_t)part.m, sizeof(double));
    double *v = calloc((size_t)part.n, sizeof(double));
    if (!Av || !v) {
        free(Av);
        free(v);
        sparse_svd_free(&part);
        sparse_free(A);
        ASSERT_NOT_NULL(Av);
        return;
    }
    double max_resid = 0.0;
    for (idx_t s = 0; s < k; s++) {
        for (idx_t j = 0; j < part.n; j++)
            v[j] = part.Vt[(size_t)j * (size_t)k + (size_t)s];
        memset(Av, 0, (size_t)part.m * sizeof(double));
        sparse_matvec(A, v, Av);
        double resid = 0.0;
        for (idx_t i = 0; i < part.m; i++) {
            double diff = Av[i] - part.sigma[s] * part.U[(size_t)s * (size_t)part.m + (size_t)i];
            resid += diff * diff;
        }
        resid = sqrt(resid);
        if (resid > max_resid)
            max_resid = resid;
    }
    printf("    west0067 partial vectors: max_resid=%.2e\n", max_resid);
    ASSERT_TRUE(max_resid < 1e-4);

    free(Av);
    free(v);
    sparse_svd_free(&part);
    sparse_free(A);
}

static void test_partial_svd_vectors_recon(void) {
    SparseMatrix *A = sparse_create(10, 10);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 10; i++) {
        sparse_insert(A, i, i, 2.0 * (double)(i + 1));
        if (i + 1 < 10)
            sparse_insert(A, i, i + 1, 1.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, 1.0);
    }

    sparse_svd_t full;
    sparse_err_t err = sparse_svd_compute(A, NULL, &full);
    ASSERT_EQ(err, SPARSE_OK);

    idx_t k = 3;
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t part;
    err = sparse_svd_partial(A, k, &opts, &part);
    ASSERT_EQ(err, SPARSE_OK);

    double frob_sq = 0.0;
    for (idx_t i = 0; i < 10; i++) {
        for (idx_t j = 0; j < 10; j++) {
            double approx = 0.0;
            for (idx_t s = 0; s < k; s++)
                approx += part.U[(size_t)s * 10 + (size_t)i] * part.sigma[s] *
                          part.Vt[(size_t)j * (size_t)k + (size_t)s];
            double diff = sparse_get(A, i, j) - approx;
            frob_sq += diff * diff;
        }
    }
    double frob = sqrt(frob_sq);

    double expected_sq = 0.0;
    for (idx_t i = k; i < full.k; i++)
        expected_sq += full.sigma[i] * full.sigma[i];
    double expected = sqrt(expected_sq);

    printf("    reconstruction error: %.4f, expected: %.4f\n", frob, expected);
    ASSERT_TRUE(frob < expected * 1.2 + 0.1);

    sparse_svd_free(&full);
    sparse_svd_free(&part);
    sparse_free(A);
}

static void test_partial_svd_vectors_k1(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    double d[] = {10.0, 7.0, 5.0, 3.0, 1.0};
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, i, d[i]);

    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_partial(A, 1, &opts, &svd);
    ASSERT_EQ(err, SPARSE_OK);
    ASSERT_NOT_NULL(svd.U);
    ASSERT_NOT_NULL(svd.Vt);

    ASSERT_TRUE(fabs(svd.sigma[0] - 10.0) < 0.1);

    double max_u = 0.0;
    idx_t max_u_idx = 0;
    for (idx_t i = 0; i < 5; i++) {
        if (fabs(svd.U[i]) > max_u) {
            max_u = fabs(svd.U[i]);
            max_u_idx = i;
        }
    }
    ASSERT_EQ(max_u_idx, 0);
    ASSERT_TRUE(max_u > 0.9);

    sparse_svd_free(&svd);
    sparse_free(A);
}

static void test_partial_svd_vectors_wide(void) {
    SparseMatrix *A = sparse_create(4, 8);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 4; i++)
        sparse_insert(A, i, i, (double)(5 - i));

    idx_t k = 2;
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_partial(A, k, &opts, &svd);
    ASSERT_EQ(err, SPARSE_OK);
    ASSERT_NOT_NULL(svd.U);
    ASSERT_NOT_NULL(svd.Vt);

    ASSERT_TRUE(fabs(svd.sigma[0] - 5.0) < 0.1);
    ASSERT_TRUE(fabs(svd.sigma[1] - 4.0) < 0.1);

    double max_resid = 0.0;
    if (!partial_svd_max_av_residual(A, &svd, k, &max_resid)) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }
    printf("    wide 4x8 partial vectors: max_resid=%.2e\n", max_resid);
    ASSERT_TRUE(max_resid < 1e-6);

    sparse_svd_free(&svd);
    sparse_free(A);
}

static void test_partial_svd_no_vectors(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_partial(A, 3, NULL, &svd);
    ASSERT_EQ(err, SPARSE_OK);
    ASSERT_TRUE(svd.U == NULL);
    ASSERT_TRUE(svd.Vt == NULL);
    ASSERT_NOT_NULL(svd.sigma);
    ASSERT_TRUE(fabs(svd.sigma[0] - 5.0) < 0.1);

    sparse_svd_free(&svd);
    sparse_free(A);
}

#endif
