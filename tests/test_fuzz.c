/*
 * test_fuzz.c — Fuzz tests for Matrix Market parser and property-based tests.
 *
 * Exercises sparse_load_mm() with malformed inputs and verifies
 * factorization properties on random matrices.
 */
#include "test_framework.h"

#include "sparse_analysis.h"
#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_svd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Helper: unique temp file for concurrent-safe fuzz testing
 * ═══════════════════════════════════════════════════════════════════════ */

static char fuzz_tmp_path[256];

static void fuzz_init_tmp(void) {
#ifdef _WIN32
    char tmpdir[MAX_PATH + 1];
    char tmppath[MAX_PATH + 1];
    char finalpath[sizeof(fuzz_tmp_path)];
    DWORD len;

    fuzz_tmp_path[0] = '\0';
    len = GetTempPathA((DWORD)sizeof(tmpdir), tmpdir);
    if (len == 0 || len >= (DWORD)sizeof(tmpdir)) {
        fprintf(stderr, "fuzz_init_tmp: GetTempPathA failed\n");
        return;
    }
    if (GetTempFileNameA(tmpdir, "fuz", 0, tmppath) == 0) {
        fprintf(stderr, "fuzz_init_tmp: GetTempFileNameA failed\n");
        return;
    }
    if (snprintf(finalpath, sizeof(finalpath), "%s.mtx", tmppath) >= (int)sizeof(finalpath)) {
        fprintf(stderr, "fuzz_init_tmp: temp path too long\n");
        remove(tmppath);
        return;
    }
    if (!MoveFileA(tmppath, finalpath)) {
        fprintf(stderr, "fuzz_init_tmp: MoveFileA failed\n");
        remove(tmppath);
        fuzz_tmp_path[0] = '\0';
        return;
    }
    snprintf(fuzz_tmp_path, sizeof(fuzz_tmp_path), "%s", finalpath);
#else
    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || !tmpdir[0])
        tmpdir = "/tmp";
    fuzz_tmp_path[0] = '\0';
    snprintf(fuzz_tmp_path, sizeof(fuzz_tmp_path), "%s/fuzz_test_XXXXXX.mtx", tmpdir);
    int fd = mkstemps(fuzz_tmp_path, 4); /* 4 = strlen(".mtx") */
    if (fd < 0) {
        fprintf(stderr, "fuzz_init_tmp: mkstemps failed\n");
        fuzz_tmp_path[0] = '\0';
    } else {
        close(fd);
    }
#endif
}

static void fuzz_cleanup_tmp(void) {
    if (fuzz_tmp_path[0])
        remove(fuzz_tmp_path);
}

static sparse_err_t try_load_mm(const char *content) {
    if (!fuzz_tmp_path[0])
        return SPARSE_ERR_FOPEN; /* temp-file creation failed in init */
    FILE *f = fopen(fuzz_tmp_path, "w");
    if (!f)
        return SPARSE_ERR_FOPEN;
    if (content)
        fputs(content, f);
    fclose(f);

    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, fuzz_tmp_path);
    if (A)
        sparse_free(A);
    return err;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Fuzz tests for Matrix Market parser
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_fuzz_empty_file(void) {
    sparse_err_t err = try_load_mm("");
    ASSERT_TRUE(err != SPARSE_OK);
}

static void test_fuzz_header_only(void) {
    sparse_err_t err = try_load_mm("%%MatrixMarket matrix coordinate real general\n");
    ASSERT_TRUE(err != SPARSE_OK);
}

static void test_fuzz_bad_header(void) {
    sparse_err_t err = try_load_mm("%%MatrixMarket GARBAGE\n5 5 3\n1 1 1.0\n2 2 2.0\n3 3 3.0\n");
    ASSERT_TRUE(err != SPARSE_OK);
}

static void test_fuzz_missing_dimensions(void) {
    sparse_err_t err = try_load_mm("%%MatrixMarket matrix coordinate real general\n");
    ASSERT_TRUE(err != SPARSE_OK);
}

static void test_fuzz_zero_dimensions(void) {
    sparse_err_t err = try_load_mm("%%MatrixMarket matrix coordinate real general\n0 0 0\n");
    /* 0x0 matrix — either error or empty matrix is OK */
    (void)err;
}

static void test_fuzz_negative_dimensions(void) {
    sparse_err_t err =
        try_load_mm("%%MatrixMarket matrix coordinate real general\n-5 5 3\n1 1 1.0\n");
    ASSERT_TRUE(err != SPARSE_OK);
}

static void test_fuzz_truncated_entries(void) {
    /* Claims 3 entries but only provides 1 */
    sparse_err_t err =
        try_load_mm("%%MatrixMarket matrix coordinate real general\n3 3 3\n1 1 1.0\n");
    /* Should fail or produce partial result */
    (void)err; /* either error or partial load is acceptable */
}

static void test_fuzz_out_of_range_indices(void) {
    sparse_err_t err =
        try_load_mm("%%MatrixMarket matrix coordinate real general\n3 3 1\n10 10 1.0\n");
    /* Parser may silently skip out-of-range entries — verify no crash */
    (void)err;
}

static void test_fuzz_zero_index(void) {
    /* MM format is 1-based; 0 maps to -1 after adjustment — verify no crash */
    sparse_err_t err =
        try_load_mm("%%MatrixMarket matrix coordinate real general\n3 3 1\n0 1 1.0\n");
    (void)err;
}

static void test_fuzz_nan_value(void) {
    sparse_err_t err =
        try_load_mm("%%MatrixMarket matrix coordinate real general\n3 3 1\n1 1 NaN\n");
    /* Parser may accept or reject NaN — just verify no crash */
    (void)err;
}

static void test_fuzz_inf_value(void) {
    sparse_err_t err =
        try_load_mm("%%MatrixMarket matrix coordinate real general\n3 3 1\n1 1 Inf\n");
    (void)err;
}

static void test_fuzz_very_large_dimensions(void) {
    /* Dimensions that would overflow memory allocation */
    sparse_err_t err = try_load_mm(
        "%%MatrixMarket matrix coordinate real general\n999999999 999999999 1\n1 1 1.0\n");
    /* Should fail with ALLOC or BADARG, not crash */
    (void)err;
}

static void test_fuzz_binary_garbage(void) {
    FILE *f = fopen(fuzz_tmp_path, "wb");
    if (!f)
        return;
    unsigned char garbage[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
                               0x00, 0x00, 0x00, 0x0D, 0xFF, 0xFE, 0xFD, 0xFC};
    fwrite(garbage, 1, sizeof(garbage), f);
    fclose(f);

    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, fuzz_tmp_path);
    ASSERT_TRUE(err != SPARSE_OK);
    if (A)
        sparse_free(A);
}

static void test_fuzz_utf8_content(void) {
    sparse_err_t err = try_load_mm("%%MatrixMarket matrix coordinate real general\n"
                                   "% Comment with UTF-8: \xc3\xa9\xc3\xa0\xc3\xbc\n"
                                   "2 2 1\n1 1 1.0\n");
    /* UTF-8 in comments should be tolerated */
    (void)err;
}

static void test_fuzz_extra_whitespace(void) {
    sparse_err_t err = try_load_mm("%%MatrixMarket matrix coordinate real general\n"
                                   "  3   3   2  \n"
                                   "  1   1   1.0  \n"
                                   "  2   2   2.0  \n");
    /* Extra whitespace should be handled */
    (void)err;
}

static void test_fuzz_many_comments(void) {
    char buf[4096];
    int pos = 0;
    int written;

    written = snprintf(buf + pos, sizeof(buf) - (size_t)pos,
                       "%%%%MatrixMarket matrix coordinate real general\n");
    ASSERT_TRUE(written > 0 && (size_t)written < sizeof(buf) - (size_t)pos);
    pos += written;

    for (int i = 0; i < 100; i++) {
        written = snprintf(buf + pos, sizeof(buf) - (size_t)pos, "%% comment line %d\n", i);
        ASSERT_TRUE(written > 0 && (size_t)written < sizeof(buf) - (size_t)pos);
        pos += written;
    }

    written = snprintf(buf + pos, sizeof(buf) - (size_t)pos, "2 2 1\n1 1 1.0\n");
    ASSERT_TRUE(written > 0 && (size_t)written < sizeof(buf) - (size_t)pos);

    sparse_err_t err = try_load_mm(buf);
    ASSERT_EQ(err, SPARSE_OK);
}

static void test_fuzz_duplicate_entries(void) {
    sparse_err_t err = try_load_mm("%%MatrixMarket matrix coordinate real general\n"
                                   "2 2 3\n1 1 1.0\n1 1 2.0\n2 2 3.0\n");
    /* Duplicate entries: should overwrite or sum, not crash */
    (void)err;
}

static void test_fuzz_symmetric_flag(void) {
    SparseMatrix *A = NULL;
    FILE *f = fopen(fuzz_tmp_path, "w");
    if (!f)
        return;
    fputs("%%MatrixMarket matrix coordinate real symmetric\n", f);
    fputs("3 3 2\n", f);
    fputs("1 1 4.0\n", f);
    fputs("2 1 1.0\n", f);
    fclose(f);

    sparse_err_t err = sparse_load_mm(&A, fuzz_tmp_path);
    if (err == SPARSE_OK && A) {
        /* Symmetric: (2,1) should be mirrored to (1,2) */
        double v12 = sparse_get(A, 0, 1);
        double v21 = sparse_get(A, 1, 0);
        ASSERT_NEAR(v12, v21, 1e-15);
        sparse_free(A);
    }
}

static void test_fuzz_null_args(void) {
    ASSERT_TRUE(sparse_load_mm(NULL, "foo.mtx") != SPARSE_OK);
    SparseMatrix *A = NULL;
    ASSERT_TRUE(sparse_load_mm(&A, NULL) != SPARSE_OK);
}

static void test_fuzz_nonexistent_file(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, "/nonexistent/path/no_such_file.mtx");
    ASSERT_TRUE(err != SPARSE_OK);
    ASSERT_TRUE(A == NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Property-based tests: random matrices
 * ═══════════════════════════════════════════════════════════════════════ */

/* Generate a random diagonally-dominant matrix (non-singular) */
static SparseMatrix *random_diag_dominant(idx_t n, unsigned seed) {
    srand(seed);
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        double offdiag_sum = 0.0;
        /* ~3 off-diagonal entries per row */
        for (int e = 0; e < 3; e++) {
            idx_t j = (idx_t)(rand() % (int)n);
            if (j != i) {
                double val = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                sparse_insert(A, i, j, val);
                offdiag_sum += fabs(val);
            }
        }
        sparse_insert(A, i, i, offdiag_sum + 1.0); /* ensure diagonal dominance */
    }
    return A;
}

/* Generate a random SPD matrix: A = B^T*B + n*I */
static SparseMatrix *random_spd(idx_t n, unsigned seed) {
    srand(seed);
    SparseMatrix *B = sparse_create(n, n);
    if (!B)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(B, i, i, (double)(rand() % 10 + 1));
        if (i + 1 < n)
            sparse_insert(B, i, i + 1, ((double)rand() / RAND_MAX) * 0.5);
    }
    /* A = B^T*B via matmul + add n*I */
    SparseMatrix *Bt = sparse_transpose(B);
    SparseMatrix *A = NULL;
    sparse_matmul(Bt, B, &A);
    sparse_free(B);
    sparse_free(Bt);
    if (!A)
        return NULL;
    /* Add n*I for strong positive definiteness */
    for (idx_t i = 0; i < n; i++) {
        double cur = sparse_get(A, i, i);
        sparse_insert(A, i, i, cur + (double)n);
    }
    return A;
}

static SparseMatrix *build_large_kkt(idx_t n_top, idx_t n_bot) {
    idx_t n = n_top + n_bot;
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;

    for (idx_t i = 0; i < n_top; i++) {
        sparse_insert(A, i, i, 6.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    for (idx_t j = 0; j < n_bot; j++) {
        sparse_insert(A, n_top + j, j, 1.0);
        sparse_insert(A, j, n_top + j, 1.0);
    }
    return A;
}

static void perturb_large_kkt_values_in_place(SparseMatrix *A, idx_t n_top, idx_t n_bot,
                                              unsigned seed) {
    for (idx_t i = 0; i < n_top; i++) {
        double diag = 6.0 + 0.05 * (double)((int)((seed + 7u * (unsigned)i) % 11u) - 5);
        ASSERT_EQ(sparse_set(A, i, i, diag), SPARSE_OK);
        if (i > 0) {
            double offdiag = -1.0 - 0.02 * (double)((int)((seed + 5u * (unsigned)i) % 7u) - 3);
            ASSERT_EQ(sparse_set(A, i, i - 1, offdiag), SPARSE_OK);
            ASSERT_EQ(sparse_set(A, i - 1, i, offdiag), SPARSE_OK);
        }
    }

    for (idx_t j = 0; j < n_bot; j++) {
        double coupling = 1.0 + 0.03 * (double)((int)((seed + 3u * (unsigned)j) % 9u) - 4);
        ASSERT_EQ(sparse_set(A, n_top + j, j, coupling), SPARSE_OK);
        ASSERT_EQ(sparse_set(A, j, n_top + j, coupling), SPARSE_OK);
    }
}

static void property_assert_vec_near(const double *a, const double *b, idx_t n, double tol) {
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(a[i], b[i], tol);
}

static void property_assert_rel_residual_small(const SparseMatrix *A, const double *x,
                                               const double *b, idx_t n, double tol) {
    double *Ax = malloc((size_t)n * sizeof(double));
    double numer = 0.0;
    double denom = 0.0;
    double resid = 0.0;
    double rhs_norm = 0.0;

    REQUIRE_OK(Ax ? SPARSE_OK : SPARSE_ERR_ALLOC);

    memset(Ax, 0, (size_t)n * sizeof(double));
    sparse_matvec(A, x, Ax);

    for (idx_t i = 0; i < n; i++) {
        const double diff = b[i] - Ax[i];
        numer += diff * diff;
        denom += b[i] * b[i];
    }

    rhs_norm = sqrt(denom);
    resid = sqrt(numer) / (rhs_norm > 0.0 ? rhs_norm : 1.0);
    ASSERT_TRUE(resid <= tol);

    free(Ax);
}

/* Property: LU factor -> solve -> residual small */
static void test_property_lu(void) {
    int pass_count = 0;
    for (unsigned seed = 1; seed <= 10; seed++) {
        idx_t n = 20;
        SparseMatrix *A = random_diag_dominant(n, seed * 137u);
        if (!A)
            continue;

        /* b = A * ones */
        double ones[20], b[20], x[20];
        for (idx_t i = 0; i < n; i++)
            ones[i] = 1.0;
        memset(b, 0, sizeof(b));
        sparse_matvec(A, ones, b);

        SparseMatrix *LU = sparse_copy(A);
        sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-14);
        if (err == SPARSE_OK) {
            sparse_err_t serr = sparse_lu_solve(LU, b, x);
            if (serr == SPARSE_OK) {
                /* Check residual */
                double Ax[20] = {0};
                sparse_matvec(A, x, Ax);
                double resid = 0.0;
                for (idx_t i = 0; i < n; i++) {
                    double d = b[i] - Ax[i];
                    resid += d * d;
                }
                resid = sqrt(resid);
                if (resid < 1e-6)
                    pass_count++;
            }
        }
        sparse_free(LU);
        sparse_free(A);
    }
    printf("    LU property: %d/10 passed\n", pass_count);
    ASSERT_TRUE(pass_count >= 9);
}

/* Property: Cholesky factor -> solve -> residual small */
static void test_property_cholesky(void) {
    int pass_count = 0;
    for (unsigned seed = 1; seed <= 10; seed++) {
        idx_t n = 15;
        SparseMatrix *A = random_spd(n, seed * 251u);
        if (!A)
            continue;

        double ones[15], b[15], x[15];
        for (idx_t i = 0; i < n; i++)
            ones[i] = 1.0;
        memset(b, 0, sizeof(b));
        sparse_matvec(A, ones, b);

        SparseMatrix *L = sparse_copy(A);
        sparse_err_t err = sparse_cholesky_factor(L);
        if (err == SPARSE_OK) {
            sparse_err_t serr = sparse_cholesky_solve(L, b, x);
            if (serr == SPARSE_OK) {
                double Ax[15] = {0};
                sparse_matvec(A, x, Ax);
                double resid = 0.0;
                for (idx_t i = 0; i < n; i++) {
                    double d = b[i] - Ax[i];
                    resid += d * d;
                }
                resid = sqrt(resid);
                if (resid < 1e-6)
                    pass_count++;
            }
        }
        sparse_free(L);
        sparse_free(A);
    }
    printf("    Cholesky property: %d/10 passed\n", pass_count);
    ASSERT_TRUE(pass_count >= 9);
}

/* Property: QR solve -> residual minimality */
static void test_property_qr(void) {
    int pass_count = 0;
    for (unsigned seed = 1; seed <= 10; seed++) {
        srand(seed * 373u);
        idx_t m = 20, n = 10;
        SparseMatrix *A = sparse_create(m, n);
        if (!A)
            continue;
        /* Random tall matrix */
        for (idx_t i = 0; i < m; i++)
            for (idx_t j = 0; j < n; j++) {
                double val = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                if (fabs(val) > 0.3)
                    sparse_insert(A, i, j, val);
            }
        /* Ensure non-degenerate */
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, j, j, sparse_get(A, j, j) + 3.0);

        double b[20];
        for (idx_t i = 0; i < m; i++)
            b[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

        sparse_qr_t qr;
        sparse_err_t err = sparse_qr_factor(A, &qr);
        if (err == SPARSE_OK) {
            double x[10];
            double resid_norm = 0.0;
            err = sparse_qr_solve(&qr, b, x, &resid_norm);
            if (err == SPARSE_OK && resid_norm < 10.0)
                pass_count++;
            sparse_qr_free(&qr);
        }
        sparse_free(A);
    }
    printf("    QR property: %d/10 passed\n", pass_count);
    ASSERT_TRUE(pass_count >= 9);
}

/* Property: SVD -> A ≈ U*Sigma*V^T */
static void test_property_svd(void) {
    int pass_count = 0;
    for (unsigned seed = 1; seed <= 10; seed++) {
        srand(seed * 499u);
        idx_t m = 8, n = 6;
        SparseMatrix *A = sparse_create(m, n);
        if (!A)
            continue;
        for (idx_t i = 0; i < m; i++)
            for (idx_t j = 0; j < n; j++) {
                double val = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                if (fabs(val) > 0.2)
                    sparse_insert(A, i, j, val);
            }

        sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1};
        sparse_svd_t svd;
        sparse_err_t err = sparse_svd_compute(A, &opts, &svd);
        if (err == SPARSE_OK) {
            /* Verify reconstruction */
            double max_err = 0.0;
            idx_t k = svd.k;
            for (idx_t i = 0; i < m; i++) {
                for (idx_t j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (idx_t s = 0; s < k; s++)
                        sum += svd.U[(size_t)s * (size_t)m + (size_t)i] * svd.sigma[s] *
                               svd.Vt[(size_t)j * (size_t)k + (size_t)s];
                    double e = fabs(sum - sparse_get(A, i, j));
                    if (e > max_err)
                        max_err = e;
                }
            }
            if (max_err < 1e-8)
                pass_count++;
            sparse_svd_free(&svd);
        }
        sparse_free(A);
    }
    printf("    SVD property: %d/10 passed\n", pass_count);
    ASSERT_TRUE(pass_count >= 9);
}

/* Property: large-n CSC-backed public one-shot Cholesky and explicit repeated
 * lifecycle stay numerically aligned across same-pattern SPD stages. */
static void test_property_large_n_cholesky_public_lifecycle_same_pattern_csc(void) {
    static const unsigned seeds[] = {701u, 1103u, 1729u};
    const idx_t n = (idx_t)(SPARSE_CSC_THRESHOLD + 12);
    const double tol = 1e-10;
    int pass_count = 0;

    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    for (size_t case_idx = 0; case_idx < sizeof(seeds) / sizeof(seeds[0]); case_idx++) {
        SparseMatrix *A_base = random_spd(n, seeds[case_idx]);
        SparseMatrix *A_ref1 = NULL;
        SparseMatrix *A_ref2 = NULL;
        SparseMatrix *A_one0 = NULL;
        SparseMatrix *A_one1 = NULL;
        SparseMatrix *A_one2 = NULL;
        sparse_analysis_t analysis = {0};
        sparse_factors_t factors = {0};
        double *x_exact = NULL;
        double *b0 = NULL;
        double *b1 = NULL;
        double *b2 = NULL;
        double *x_public0 = NULL;
        double *x_public1 = NULL;
        double *x_public2 = NULL;
        double *x_one0 = NULL;
        double *x_one1 = NULL;
        double *x_one2 = NULL;
        int used_csc_path0 = 0;
        int used_csc_path1 = 0;
        int used_csc_path2 = 0;

        REQUIRE_OK(A_base ? SPARSE_OK : SPARSE_ERR_ALLOC);

        A_ref1 = sparse_copy(A_base);
        A_ref2 = sparse_copy(A_base);
        A_one0 = sparse_copy(A_base);
        A_one1 = sparse_copy(A_base);
        A_one2 = sparse_copy(A_base);
        REQUIRE_OK(A_ref1 && A_ref2 && A_one0 && A_one1 && A_one2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

        sparse_analysis_opts_t analysis_opts = {
            .factor_type = SPARSE_FACTOR_CHOLESKY,
            .reorder = SPARSE_REORDER_AMD,
        };
        sparse_cholesky_opts_t chol_opts0 = {
            .reorder = SPARSE_REORDER_AMD,
            .used_csc_path = &used_csc_path0,
        };
        sparse_cholesky_opts_t chol_opts1 = {
            .reorder = SPARSE_REORDER_AMD,
            .used_csc_path = &used_csc_path1,
        };
        sparse_cholesky_opts_t chol_opts2 = {
            .reorder = SPARSE_REORDER_AMD,
            .used_csc_path = &used_csc_path2,
        };

        REQUIRE_OK(sparse_analyze(A_base, &analysis_opts, &analysis));
        REQUIRE_OK(sparse_factor_numeric(A_base, &analysis, &factors));

        x_exact = malloc((size_t)n * sizeof(double));
        b0 = malloc((size_t)n * sizeof(double));
        b1 = malloc((size_t)n * sizeof(double));
        b2 = malloc((size_t)n * sizeof(double));
        x_public0 = malloc((size_t)n * sizeof(double));
        x_public1 = malloc((size_t)n * sizeof(double));
        x_public2 = malloc((size_t)n * sizeof(double));
        x_one0 = malloc((size_t)n * sizeof(double));
        x_one1 = malloc((size_t)n * sizeof(double));
        x_one2 = malloc((size_t)n * sizeof(double));
        REQUIRE_OK(x_exact && b0 && b1 && b2 && x_public0 && x_public1 && x_public2 && x_one0 &&
                           x_one1 && x_one2
                       ? SPARSE_OK
                       : SPARSE_ERR_ALLOC);

        for (idx_t i = 0; i < n; i++)
            x_exact[i] = 1.0 + 0.01 * (double)i;

        for (idx_t i = 0; i < n; i++) {
            const double base_diag = sparse_get(A_base, i, i);
            const double ref1_diag = base_diag + 0.5;
            const double ref2_diag = base_diag + 1.0 + 0.01 * (double)i;
            ASSERT_EQ(sparse_set(A_ref1, i, i, ref1_diag), SPARSE_OK);
            ASSERT_EQ(sparse_set(A_ref2, i, i, ref2_diag), SPARSE_OK);
            ASSERT_EQ(sparse_set(A_one1, i, i, ref1_diag), SPARSE_OK);
            ASSERT_EQ(sparse_set(A_one2, i, i, ref2_diag), SPARSE_OK);
        }

        sparse_matvec(A_base, x_exact, b0);
        sparse_matvec(A_ref1, x_exact, b1);
        sparse_matvec(A_ref2, x_exact, b2);

        REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b0, x_public0));
        REQUIRE_OK(sparse_cholesky_factor_opts(A_one0, &chol_opts0));
        ASSERT_EQ(used_csc_path0, 1);
        REQUIRE_OK(sparse_cholesky_solve(A_one0, b0, x_one0));
        property_assert_vec_near(x_public0, x_exact, n, tol);
        property_assert_vec_near(x_one0, x_exact, n, tol);
        property_assert_vec_near(x_public0, x_one0, n, tol);

        REQUIRE_OK(sparse_refactor_numeric(A_ref1, &analysis, &factors));
        REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b1, x_public1));
        REQUIRE_OK(sparse_cholesky_factor_opts(A_one1, &chol_opts1));
        ASSERT_EQ(used_csc_path1, 1);
        REQUIRE_OK(sparse_cholesky_solve(A_one1, b1, x_one1));
        property_assert_vec_near(x_public1, x_exact, n, tol);
        property_assert_vec_near(x_one1, x_exact, n, tol);
        property_assert_vec_near(x_public1, x_one1, n, tol);

        REQUIRE_OK(sparse_refactor_numeric(A_ref2, &analysis, &factors));
        REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b2, x_public2));
        REQUIRE_OK(sparse_cholesky_factor_opts(A_one2, &chol_opts2));
        ASSERT_EQ(used_csc_path2, 1);
        REQUIRE_OK(sparse_cholesky_solve(A_one2, b2, x_one2));
        property_assert_vec_near(x_public2, x_exact, n, tol);
        property_assert_vec_near(x_one2, x_exact, n, tol);
        property_assert_vec_near(x_public2, x_one2, n, tol);

        pass_count++;

        free(x_exact);
        free(b0);
        free(b1);
        free(b2);
        free(x_public0);
        free(x_public1);
        free(x_public2);
        free(x_one0);
        free(x_one1);
        free(x_one2);
        sparse_factor_free(&factors);
        sparse_analysis_free(&analysis);
        sparse_free(A_base);
        sparse_free(A_ref1);
        sparse_free(A_ref2);
        sparse_free(A_one0);
        sparse_free(A_one1);
        sparse_free(A_one2);
    }

    printf("    large-n CSC lifecycle property: %d/%zu passed\n", pass_count,
           sizeof(seeds) / sizeof(seeds[0]));
    ASSERT_EQ(pass_count, (int)(sizeof(seeds) / sizeof(seeds[0])));
}

static void test_property_large_n_ldlt_public_lifecycle_same_pattern_csc(void) {
    static const unsigned seeds[] = {809u, 1451u, 2029u};
    const idx_t n_top = (idx_t)(SPARSE_CSC_THRESHOLD + 12);
    const idx_t n_bot = 8;
    const idx_t n = n_top + n_bot;
    const double tol = 1e-9;
    int pass_count = 0;

    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    for (size_t case_idx = 0; case_idx < sizeof(seeds) / sizeof(seeds[0]); case_idx++) {
        SparseMatrix *A_base = build_large_kkt(n_top, n_bot);
        SparseMatrix *A_ref1 = NULL;
        SparseMatrix *A_ref2 = NULL;
        SparseMatrix *A_one0 = NULL;
        SparseMatrix *A_one1 = NULL;
        SparseMatrix *A_one2 = NULL;
        sparse_analysis_t analysis = {0};
        sparse_factors_t factors = {0};
        sparse_ldlt_t ldlt0 = {0};
        sparse_ldlt_t ldlt1 = {0};
        sparse_ldlt_t ldlt2 = {0};
        double *x_exact = NULL;
        double *b0 = NULL;
        double *b1 = NULL;
        double *b2 = NULL;
        double *x_public0 = NULL;
        double *x_public1 = NULL;
        double *x_public2 = NULL;
        double *x_one0 = NULL;
        double *x_one1 = NULL;
        double *x_one2 = NULL;
        int used_csc_path0 = 0;
        int used_csc_path1 = 0;
        int used_csc_path2 = 0;

        REQUIRE_OK(A_base ? SPARSE_OK : SPARSE_ERR_ALLOC);

        A_ref1 = sparse_copy(A_base);
        A_ref2 = sparse_copy(A_base);
        A_one0 = sparse_copy(A_base);
        A_one1 = sparse_copy(A_base);
        A_one2 = sparse_copy(A_base);
        REQUIRE_OK(A_ref1 && A_ref2 && A_one0 && A_one1 && A_one2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

        perturb_large_kkt_values_in_place(A_ref1, n_top, n_bot, seeds[case_idx]);
        perturb_large_kkt_values_in_place(A_ref2, n_top, n_bot, seeds[case_idx] + 97u);
        perturb_large_kkt_values_in_place(A_one1, n_top, n_bot, seeds[case_idx]);
        perturb_large_kkt_values_in_place(A_one2, n_top, n_bot, seeds[case_idx] + 97u);

        sparse_analysis_opts_t analysis_opts = {
            .factor_type = SPARSE_FACTOR_LDLT,
            .reorder = SPARSE_REORDER_AMD,
        };
        sparse_ldlt_opts_t ldlt_opts0 = {
            .reorder = SPARSE_REORDER_AMD,
            .backend = SPARSE_LDLT_BACKEND_AUTO,
            .used_csc_path = &used_csc_path0,
        };
        sparse_ldlt_opts_t ldlt_opts1 = {
            .reorder = SPARSE_REORDER_AMD,
            .backend = SPARSE_LDLT_BACKEND_AUTO,
            .used_csc_path = &used_csc_path1,
        };
        sparse_ldlt_opts_t ldlt_opts2 = {
            .reorder = SPARSE_REORDER_AMD,
            .backend = SPARSE_LDLT_BACKEND_AUTO,
            .used_csc_path = &used_csc_path2,
        };

        REQUIRE_OK(sparse_analyze(A_base, &analysis_opts, &analysis));
        REQUIRE_OK(sparse_factor_numeric(A_base, &analysis, &factors));

        x_exact = malloc((size_t)n * sizeof(double));
        b0 = malloc((size_t)n * sizeof(double));
        b1 = malloc((size_t)n * sizeof(double));
        b2 = malloc((size_t)n * sizeof(double));
        x_public0 = malloc((size_t)n * sizeof(double));
        x_public1 = malloc((size_t)n * sizeof(double));
        x_public2 = malloc((size_t)n * sizeof(double));
        x_one0 = malloc((size_t)n * sizeof(double));
        x_one1 = malloc((size_t)n * sizeof(double));
        x_one2 = malloc((size_t)n * sizeof(double));
        REQUIRE_OK(x_exact && b0 && b1 && b2 && x_public0 && x_public1 && x_public2 && x_one0 &&
                           x_one1 && x_one2
                       ? SPARSE_OK
                       : SPARSE_ERR_ALLOC);

        for (idx_t i = 0; i < n; i++)
            x_exact[i] = 1.0 + 0.005 * (double)i;

        sparse_matvec(A_base, x_exact, b0);
        sparse_matvec(A_ref1, x_exact, b1);
        sparse_matvec(A_ref2, x_exact, b2);

        REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b0, x_public0));
        REQUIRE_OK(sparse_ldlt_factor_opts(A_one0, &ldlt_opts0, &ldlt0));
        ASSERT_EQ(used_csc_path0, 1);
        REQUIRE_OK(sparse_ldlt_solve(&ldlt0, b0, x_one0));
        property_assert_vec_near(x_public0, x_exact, n, tol);
        property_assert_vec_near(x_one0, x_exact, n, tol);
        property_assert_vec_near(x_public0, x_one0, n, tol);

        REQUIRE_OK(sparse_refactor_numeric(A_ref1, &analysis, &factors));
        REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b1, x_public1));
        REQUIRE_OK(sparse_ldlt_factor_opts(A_one1, &ldlt_opts1, &ldlt1));
        ASSERT_EQ(used_csc_path1, 1);
        REQUIRE_OK(sparse_ldlt_solve(&ldlt1, b1, x_one1));
        property_assert_vec_near(x_public1, x_exact, n, tol);
        property_assert_vec_near(x_one1, x_exact, n, tol);
        property_assert_vec_near(x_public1, x_one1, n, tol);

        REQUIRE_OK(sparse_refactor_numeric(A_ref2, &analysis, &factors));
        REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b2, x_public2));
        REQUIRE_OK(sparse_ldlt_factor_opts(A_one2, &ldlt_opts2, &ldlt2));
        ASSERT_EQ(used_csc_path2, 1);
        REQUIRE_OK(sparse_ldlt_solve(&ldlt2, b2, x_one2));
        property_assert_vec_near(x_public2, x_exact, n, tol);
        property_assert_vec_near(x_one2, x_exact, n, tol);
        property_assert_vec_near(x_public2, x_one2, n, tol);

        pass_count++;

        free(x_exact);
        free(b0);
        free(b1);
        free(b2);
        free(x_public0);
        free(x_public1);
        free(x_public2);
        free(x_one0);
        free(x_one1);
        free(x_one2);
        sparse_ldlt_free(&ldlt0);
        sparse_ldlt_free(&ldlt1);
        sparse_ldlt_free(&ldlt2);
        sparse_factor_free(&factors);
        sparse_analysis_free(&analysis);
        sparse_free(A_base);
        sparse_free(A_ref1);
        sparse_free(A_ref2);
        sparse_free(A_one0);
        sparse_free(A_one1);
        sparse_free(A_one2);
    }

    printf("    large-n LDLT CSC lifecycle property: %d/%zu passed\n", pass_count,
           sizeof(seeds) / sizeof(seeds[0]));
    ASSERT_EQ(pass_count, (int)(sizeof(seeds) / sizeof(seeds[0])));
}

static void test_property_large_n_cholesky_csc_reorder_repeat_solve_agreement(void) {
    static const unsigned seeds[] = {911u, 1289u, 1877u};
    const idx_t n = (idx_t)(SPARSE_CSC_THRESHOLD + 12);
    const double tol = 1e-10;
    const double residual_tol = 1e-12;
    int pass_count = 0;

    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    for (size_t case_idx = 0; case_idx < sizeof(seeds) / sizeof(seeds[0]); case_idx++) {
        SparseMatrix *A_base = random_spd(n, seeds[case_idx]);
        SparseMatrix *A_ref = NULL;
        sparse_analysis_t analysis_none = {0};
        sparse_analysis_t analysis_amd = {0};
        sparse_factors_t factors_none = {0};
        sparse_factors_t factors_amd = {0};
        double *x_exact = NULL;
        double *b0 = NULL;
        double *b1 = NULL;
        double *x_none0 = NULL;
        double *x_none0_repeat = NULL;
        double *x_amd0 = NULL;
        double *x_amd0_repeat = NULL;
        double *x_none1 = NULL;
        double *x_none1_repeat = NULL;
        double *x_amd1 = NULL;
        double *x_amd1_repeat = NULL;

        REQUIRE_OK(A_base ? SPARSE_OK : SPARSE_ERR_ALLOC);

        A_ref = sparse_copy(A_base);
        REQUIRE_OK(A_ref ? SPARSE_OK : SPARSE_ERR_ALLOC);

        sparse_analysis_opts_t analysis_opts_none = {
            .factor_type = SPARSE_FACTOR_CHOLESKY,
            .reorder = SPARSE_REORDER_NONE,
        };
        sparse_analysis_opts_t analysis_opts_amd = {
            .factor_type = SPARSE_FACTOR_CHOLESKY,
            .reorder = SPARSE_REORDER_AMD,
        };

        REQUIRE_OK(sparse_analyze(A_base, &analysis_opts_none, &analysis_none));
        REQUIRE_OK(sparse_factor_numeric(A_base, &analysis_none, &factors_none));
        REQUIRE_OK(sparse_analyze(A_base, &analysis_opts_amd, &analysis_amd));
        REQUIRE_OK(sparse_factor_numeric(A_base, &analysis_amd, &factors_amd));

        x_exact = malloc((size_t)n * sizeof(double));
        b0 = malloc((size_t)n * sizeof(double));
        b1 = malloc((size_t)n * sizeof(double));
        x_none0 = malloc((size_t)n * sizeof(double));
        x_none0_repeat = malloc((size_t)n * sizeof(double));
        x_amd0 = malloc((size_t)n * sizeof(double));
        x_amd0_repeat = malloc((size_t)n * sizeof(double));
        x_none1 = malloc((size_t)n * sizeof(double));
        x_none1_repeat = malloc((size_t)n * sizeof(double));
        x_amd1 = malloc((size_t)n * sizeof(double));
        x_amd1_repeat = malloc((size_t)n * sizeof(double));
        REQUIRE_OK(x_exact && b0 && b1 && x_none0 && x_none0_repeat && x_amd0 && x_amd0_repeat &&
                           x_none1 && x_none1_repeat && x_amd1 && x_amd1_repeat
                       ? SPARSE_OK
                       : SPARSE_ERR_ALLOC);

        for (idx_t i = 0; i < n; i++) {
            const double base_diag = sparse_get(A_base, i, i);
            x_exact[i] = 1.0 + 0.01 * (double)i;
            ASSERT_EQ(sparse_set(A_ref, i, i, base_diag + 0.25 + 0.005 * (double)i), SPARSE_OK);
        }

        sparse_matvec(A_base, x_exact, b0);
        sparse_matvec(A_ref, x_exact, b1);

        REQUIRE_OK(sparse_factor_solve(&factors_none, &analysis_none, b0, x_none0));
        REQUIRE_OK(sparse_factor_solve(&factors_none, &analysis_none, b0, x_none0_repeat));
        REQUIRE_OK(sparse_factor_solve(&factors_amd, &analysis_amd, b0, x_amd0));
        REQUIRE_OK(sparse_factor_solve(&factors_amd, &analysis_amd, b0, x_amd0_repeat));

        property_assert_vec_near(x_none0, x_exact, n, tol);
        property_assert_vec_near(x_amd0, x_exact, n, tol);
        property_assert_vec_near(x_none0, x_none0_repeat, n, tol);
        property_assert_vec_near(x_amd0, x_amd0_repeat, n, tol);
        property_assert_vec_near(x_none0, x_amd0, n, tol);
        property_assert_rel_residual_small(A_base, x_none0, b0, n, residual_tol);
        property_assert_rel_residual_small(A_base, x_amd0, b0, n, residual_tol);

        REQUIRE_OK(sparse_refactor_numeric(A_ref, &analysis_none, &factors_none));
        REQUIRE_OK(sparse_refactor_numeric(A_ref, &analysis_amd, &factors_amd));
        REQUIRE_OK(sparse_factor_solve(&factors_none, &analysis_none, b1, x_none1));
        REQUIRE_OK(sparse_factor_solve(&factors_none, &analysis_none, b1, x_none1_repeat));
        REQUIRE_OK(sparse_factor_solve(&factors_amd, &analysis_amd, b1, x_amd1));
        REQUIRE_OK(sparse_factor_solve(&factors_amd, &analysis_amd, b1, x_amd1_repeat));

        property_assert_vec_near(x_none1, x_exact, n, tol);
        property_assert_vec_near(x_amd1, x_exact, n, tol);
        property_assert_vec_near(x_none1, x_none1_repeat, n, tol);
        property_assert_vec_near(x_amd1, x_amd1_repeat, n, tol);
        property_assert_vec_near(x_none1, x_amd1, n, tol);
        property_assert_rel_residual_small(A_ref, x_none1, b1, n, residual_tol);
        property_assert_rel_residual_small(A_ref, x_amd1, b1, n, residual_tol);

        pass_count++;

        free(x_exact);
        free(b0);
        free(b1);
        free(x_none0);
        free(x_none0_repeat);
        free(x_amd0);
        free(x_amd0_repeat);
        free(x_none1);
        free(x_none1_repeat);
        free(x_amd1);
        free(x_amd1_repeat);
        sparse_factor_free(&factors_none);
        sparse_factor_free(&factors_amd);
        sparse_analysis_free(&analysis_none);
        sparse_analysis_free(&analysis_amd);
        sparse_free(A_base);
        sparse_free(A_ref);
    }

    printf("    large-n CSC reorder/repeat property: %d/%zu passed\n", pass_count,
           sizeof(seeds) / sizeof(seeds[0]));
    ASSERT_EQ(pass_count, (int)(sizeof(seeds) / sizeof(seeds[0])));
}

static void test_property_large_n_ldlt_csc_reorder_repeat_solve_agreement(void) {
    static const unsigned seeds[] = {977u, 1543u, 2111u};
    const idx_t n_top = (idx_t)(SPARSE_CSC_THRESHOLD + 12);
    const idx_t n_bot = 8;
    const idx_t n = n_top + n_bot;
    const double tol = 1e-9;
    const double residual_tol = 1e-11;
    int pass_count = 0;

    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    for (size_t case_idx = 0; case_idx < sizeof(seeds) / sizeof(seeds[0]); case_idx++) {
        SparseMatrix *A_base = build_large_kkt(n_top, n_bot);
        SparseMatrix *A_ref = NULL;
        sparse_analysis_t analysis_none = {0};
        sparse_analysis_t analysis_amd = {0};
        sparse_factors_t factors_none = {0};
        sparse_factors_t factors_amd = {0};
        double *x_exact = NULL;
        double *b0 = NULL;
        double *b1 = NULL;
        double *x_none0 = NULL;
        double *x_none0_repeat = NULL;
        double *x_amd0 = NULL;
        double *x_amd0_repeat = NULL;
        double *x_none1 = NULL;
        double *x_none1_repeat = NULL;
        double *x_amd1 = NULL;
        double *x_amd1_repeat = NULL;

        REQUIRE_OK(A_base ? SPARSE_OK : SPARSE_ERR_ALLOC);

        A_ref = sparse_copy(A_base);
        REQUIRE_OK(A_ref ? SPARSE_OK : SPARSE_ERR_ALLOC);
        perturb_large_kkt_values_in_place(A_ref, n_top, n_bot, seeds[case_idx]);

        sparse_analysis_opts_t analysis_opts_none = {
            .factor_type = SPARSE_FACTOR_LDLT,
            .reorder = SPARSE_REORDER_NONE,
        };
        sparse_analysis_opts_t analysis_opts_amd = {
            .factor_type = SPARSE_FACTOR_LDLT,
            .reorder = SPARSE_REORDER_AMD,
        };

        REQUIRE_OK(sparse_analyze(A_base, &analysis_opts_none, &analysis_none));
        REQUIRE_OK(sparse_factor_numeric(A_base, &analysis_none, &factors_none));
        REQUIRE_OK(sparse_analyze(A_base, &analysis_opts_amd, &analysis_amd));
        REQUIRE_OK(sparse_factor_numeric(A_base, &analysis_amd, &factors_amd));

        x_exact = malloc((size_t)n * sizeof(double));
        b0 = malloc((size_t)n * sizeof(double));
        b1 = malloc((size_t)n * sizeof(double));
        x_none0 = malloc((size_t)n * sizeof(double));
        x_none0_repeat = malloc((size_t)n * sizeof(double));
        x_amd0 = malloc((size_t)n * sizeof(double));
        x_amd0_repeat = malloc((size_t)n * sizeof(double));
        x_none1 = malloc((size_t)n * sizeof(double));
        x_none1_repeat = malloc((size_t)n * sizeof(double));
        x_amd1 = malloc((size_t)n * sizeof(double));
        x_amd1_repeat = malloc((size_t)n * sizeof(double));
        REQUIRE_OK(x_exact && b0 && b1 && x_none0 && x_none0_repeat && x_amd0 && x_amd0_repeat &&
                           x_none1 && x_none1_repeat && x_amd1 && x_amd1_repeat
                       ? SPARSE_OK
                       : SPARSE_ERR_ALLOC);

        for (idx_t i = 0; i < n; i++)
            x_exact[i] = 1.0 + 0.005 * (double)i;

        sparse_matvec(A_base, x_exact, b0);
        sparse_matvec(A_ref, x_exact, b1);

        REQUIRE_OK(sparse_factor_solve(&factors_none, &analysis_none, b0, x_none0));
        REQUIRE_OK(sparse_factor_solve(&factors_none, &analysis_none, b0, x_none0_repeat));
        REQUIRE_OK(sparse_factor_solve(&factors_amd, &analysis_amd, b0, x_amd0));
        REQUIRE_OK(sparse_factor_solve(&factors_amd, &analysis_amd, b0, x_amd0_repeat));

        property_assert_vec_near(x_none0, x_exact, n, tol);
        property_assert_vec_near(x_amd0, x_exact, n, tol);
        property_assert_vec_near(x_none0, x_none0_repeat, n, tol);
        property_assert_vec_near(x_amd0, x_amd0_repeat, n, tol);
        property_assert_vec_near(x_none0, x_amd0, n, tol);
        property_assert_rel_residual_small(A_base, x_none0, b0, n, residual_tol);
        property_assert_rel_residual_small(A_base, x_amd0, b0, n, residual_tol);

        REQUIRE_OK(sparse_refactor_numeric(A_ref, &analysis_none, &factors_none));
        REQUIRE_OK(sparse_refactor_numeric(A_ref, &analysis_amd, &factors_amd));
        REQUIRE_OK(sparse_factor_solve(&factors_none, &analysis_none, b1, x_none1));
        REQUIRE_OK(sparse_factor_solve(&factors_none, &analysis_none, b1, x_none1_repeat));
        REQUIRE_OK(sparse_factor_solve(&factors_amd, &analysis_amd, b1, x_amd1));
        REQUIRE_OK(sparse_factor_solve(&factors_amd, &analysis_amd, b1, x_amd1_repeat));

        property_assert_vec_near(x_none1, x_exact, n, tol);
        property_assert_vec_near(x_amd1, x_exact, n, tol);
        property_assert_vec_near(x_none1, x_none1_repeat, n, tol);
        property_assert_vec_near(x_amd1, x_amd1_repeat, n, tol);
        property_assert_vec_near(x_none1, x_amd1, n, tol);
        property_assert_rel_residual_small(A_ref, x_none1, b1, n, residual_tol);
        property_assert_rel_residual_small(A_ref, x_amd1, b1, n, residual_tol);

        pass_count++;

        free(x_exact);
        free(b0);
        free(b1);
        free(x_none0);
        free(x_none0_repeat);
        free(x_amd0);
        free(x_amd0_repeat);
        free(x_none1);
        free(x_none1_repeat);
        free(x_amd1);
        free(x_amd1_repeat);
        sparse_factor_free(&factors_none);
        sparse_factor_free(&factors_amd);
        sparse_analysis_free(&analysis_none);
        sparse_analysis_free(&analysis_amd);
        sparse_free(A_base);
        sparse_free(A_ref);
    }

    printf("    large-n LDLT reorder/repeat property: %d/%zu passed\n", pass_count,
           sizeof(seeds) / sizeof(seeds[0]));
    ASSERT_EQ(pass_count, (int)(sizeof(seeds) / sizeof(seeds[0])));
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test suite
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Fuzz & Property Tests");
    fuzz_init_tmp();

    /* Fuzz tests for MM parser (require writable temp dir) */
    if (fuzz_tmp_path[0]) {
        RUN_TEST(test_fuzz_empty_file);
        RUN_TEST(test_fuzz_header_only);
        RUN_TEST(test_fuzz_bad_header);
        RUN_TEST(test_fuzz_missing_dimensions);
        RUN_TEST(test_fuzz_zero_dimensions);
        RUN_TEST(test_fuzz_negative_dimensions);
        RUN_TEST(test_fuzz_truncated_entries);
        RUN_TEST(test_fuzz_out_of_range_indices);
        RUN_TEST(test_fuzz_zero_index);
        RUN_TEST(test_fuzz_nan_value);
        RUN_TEST(test_fuzz_inf_value);
        RUN_TEST(test_fuzz_very_large_dimensions);
        RUN_TEST(test_fuzz_binary_garbage);
        RUN_TEST(test_fuzz_utf8_content);
        RUN_TEST(test_fuzz_extra_whitespace);
        RUN_TEST(test_fuzz_many_comments);
        RUN_TEST(test_fuzz_duplicate_entries);
        RUN_TEST(test_fuzz_symmetric_flag);
    } else {
        printf("  SKIP: fuzz tests (temp file creation failed)\n");
    }
    /* These don't need a temp file */
    RUN_TEST(test_fuzz_null_args);
    RUN_TEST(test_fuzz_nonexistent_file);

    /* Property-based tests */
    RUN_TEST(test_property_lu);
    RUN_TEST(test_property_cholesky);
    RUN_TEST(test_property_qr);
    RUN_TEST(test_property_svd);
    RUN_TEST(test_property_large_n_cholesky_public_lifecycle_same_pattern_csc);
    RUN_TEST(test_property_large_n_ldlt_public_lifecycle_same_pattern_csc);
    RUN_TEST(test_property_large_n_cholesky_csc_reorder_repeat_solve_agreement);
    RUN_TEST(test_property_large_n_ldlt_csc_reorder_repeat_solve_agreement);

    fuzz_cleanup_tmp();
    TEST_SUITE_END();
}
