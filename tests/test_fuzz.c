/*
 * test_fuzz.c — Fuzz tests for Matrix Market parser and property-based tests.
 *
 * Exercises sparse_load_mm() with malformed inputs and verifies
 * factorization properties on random matrices.
 */
#include "test_framework.h"

#include "sparse_cholesky.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_svd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Helper: unique temp file for concurrent-safe fuzz testing
 * ═══════════════════════════════════════════════════════════════════════ */

static char fuzz_tmp_path[256];

static void fuzz_init_tmp(void) {
    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || !tmpdir[0])
        tmpdir = "/tmp";
    snprintf(fuzz_tmp_path, sizeof(fuzz_tmp_path), "%s/fuzz_test_XXXXXX.mtx", tmpdir);
    int fd = mkstemps(fuzz_tmp_path, 4); /* 4 = strlen(".mtx") */
    if (fd < 0) {
        fprintf(stderr, "fuzz_init_tmp: mkstemps failed\n");
        fuzz_tmp_path[0] = '\0';
    } else {
        close(fd);
    }
}

static void fuzz_cleanup_tmp(void) {
    if (fuzz_tmp_path[0])
        unlink(fuzz_tmp_path);
}

static sparse_err_t try_load_mm(const char *content) {
    if (!fuzz_tmp_path[0])
        return SPARSE_ERR_FOPEN; /* mkstemps failed in init */
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

    fuzz_cleanup_tmp();
    TEST_SUITE_END();
}
