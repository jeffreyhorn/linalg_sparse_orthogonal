#include "sparse_matrix.h"
#include "sparse_svd.h"
#include "test_framework.h"
#include "test_svd_helpers.h"
#include "test_svd_partial_shared_helpers.h"

#include <math.h>

#define PARTIAL_SVD_CORPUS_ROWS 8
#define PARTIAL_SVD_CORPUS_COLS 6
#define PARTIAL_SVD_CORPUS_K 3
#define PARTIAL_SVD_CORPUS_TOL 1e-8

static void sort_descending(double *values, idx_t n) {
    for (idx_t i = 1; i < n; i++) {
        double value = values[i];
        idx_t j = i;
        while (j > 0 && values[j - 1] < value) {
            values[j] = values[j - 1];
            j--;
        }
        values[j] = value;
    }
}

static SparseMatrix *make_partial_svd_clustered_repeated_fixture(void) {
    const double diag[PARTIAL_SVD_CORPUS_COLS] = {10.0, 10.0, 9.999999, 4.0, 1.0, 0.0};
    return tf_svd_make_diag_matrix(PARTIAL_SVD_CORPUS_ROWS, PARTIAL_SVD_CORPUS_COLS, diag,
                                   PARTIAL_SVD_CORPUS_COLS);
}

static int run_partial_svd_default(const SparseMatrix *A, sparse_svd_t *svd) {
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_err_t err = sparse_svd_partial(A, PARTIAL_SVD_CORPUS_K, &opts, svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK)
        return 0;

    ASSERT_EQ(svd->m, PARTIAL_SVD_CORPUS_ROWS);
    ASSERT_EQ(svd->n, PARTIAL_SVD_CORPUS_COLS);
    ASSERT_EQ(svd->k, PARTIAL_SVD_CORPUS_K);
    ASSERT_NOT_NULL(svd->sigma);
    ASSERT_NOT_NULL(svd->U);
    ASSERT_NOT_NULL(svd->Vt);
    return svd->sigma && svd->U && svd->Vt;
}

static double partial_svd_corpus_sigma_error(const sparse_svd_t *svd) {
    const double expected[PARTIAL_SVD_CORPUS_K] = {10.0, 10.0, 9.999999};
    double actual_sorted[PARTIAL_SVD_CORPUS_K] = {0.0, 0.0, 0.0};
    double expected_sorted[PARTIAL_SVD_CORPUS_K] = {expected[0], expected[1], expected[2]};
    double max_error = 0.0;

    for (idx_t i = 0; i < PARTIAL_SVD_CORPUS_K; i++)
        actual_sorted[i] = svd->sigma[i];
    sort_descending(actual_sorted, PARTIAL_SVD_CORPUS_K);
    sort_descending(expected_sorted, PARTIAL_SVD_CORPUS_K);

    for (idx_t i = 0; i < PARTIAL_SVD_CORPUS_K; i++) {
        double error = fabs(actual_sorted[i] - expected_sorted[i]);
        if (error > max_error)
            max_error = error;
    }
    return max_error;
}

static void test_partial_svd_corpus_clustered_repeated_default_success(void) {
    SparseMatrix *A = make_partial_svd_clustered_repeated_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_svd_t svd = {0};
    if (!run_partial_svd_default(A, &svd)) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }

    double sigma_error = partial_svd_corpus_sigma_error(&svd);
    printf("    partial-SVD corpus default success: max sigma error=%.3e\n", sigma_error);
    ASSERT_TRUE(sigma_error <= PARTIAL_SVD_CORPUS_TOL);

    sparse_svd_free(&svd);
    sparse_free(A);
}

static void test_partial_svd_corpus_clustered_repeated_projectors(void) {
    SparseMatrix *A = make_partial_svd_clustered_repeated_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_svd_t svd = {0};
    if (!run_partial_svd_default(A, &svd)) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }

    double left_distance =
        partial_svd_u_coordinate_range_projector_error(&svd, PARTIAL_SVD_CORPUS_K);
    double right_distance =
        partial_svd_v_coordinate_range_projector_error(&svd, PARTIAL_SVD_CORPUS_K);
    printf("    partial-SVD corpus projectors: left=%.3e, right=%.3e\n", left_distance,
           right_distance);
    ASSERT_TRUE(left_distance <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(right_distance <= PARTIAL_SVD_CORPUS_TOL);

    sparse_svd_free(&svd);
    sparse_free(A);
}

static void test_partial_svd_corpus_clustered_repeated_residuals(void) {
    SparseMatrix *A = make_partial_svd_clustered_repeated_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_svd_t svd = {0};
    if (!run_partial_svd_default(A, &svd)) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }

    double max_av_resid = 0.0;
    double max_atu_resid = 0.0;
    if (!partial_svd_max_triplet_residuals(A, &svd, PARTIAL_SVD_CORPUS_K, &max_av_resid,
                                           &max_atu_resid)) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }
    double u_ortho = tf_dense_column_orthogonality_error(svd.U, svd.m, svd.k);
    double v_ortho = tf_svd_vt_row_orthogonality_error(svd.Vt, svd.k, svd.n, svd.k);
    printf("    partial-SVD corpus residuals: Av=%.3e, Atu=%.3e, U_ortho=%.3e, "
           "V_ortho=%.3e\n",
           max_av_resid, max_atu_resid, u_ortho, v_ortho);
    ASSERT_TRUE(max_av_resid <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(max_atu_resid <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(u_ortho <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(v_ortho <= PARTIAL_SVD_CORPUS_TOL);

    sparse_svd_free(&svd);
    sparse_free(A);
}

static void test_partial_svd_corpus_clustered_repeated_tight_budget_fail_closed(void) {
    SparseMatrix *A = make_partial_svd_clustered_repeated_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 1, .tol = 0.0};
    sparse_svd_t svd = {0};
    sparse_err_t err = sparse_svd_partial(A, PARTIAL_SVD_CORPUS_K, &opts, &svd);
    printf("    partial-SVD corpus tight budget: err=%d, sigma=%p, U=%p, Vt=%p\n", (int)err,
           (void *)svd.sigma, (void *)svd.U, (void *)svd.Vt);
    ASSERT_ERR(err, SPARSE_ERR_NOT_CONVERGED);
    ASSERT_EQ(svd.m, PARTIAL_SVD_CORPUS_ROWS);
    ASSERT_EQ(svd.n, PARTIAL_SVD_CORPUS_COLS);
    ASSERT_EQ(svd.k, PARTIAL_SVD_CORPUS_K);
    ASSERT_TRUE(svd.sigma == NULL);
    ASSERT_TRUE(svd.U == NULL);
    ASSERT_TRUE(svd.Vt == NULL);

    sparse_svd_free(&svd);
    sparse_free(A);
}

static void test_partial_svd_corpus_clustered_repeated_recovery_after_failure(void) {
    SparseMatrix *A = make_partial_svd_clustered_repeated_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_svd_opts_t tight = {.compute_uv = 1, .economy = 1, .max_iter = 1, .tol = 0.0};
    sparse_svd_t failed = {0};
    sparse_err_t err = sparse_svd_partial(A, PARTIAL_SVD_CORPUS_K, &tight, &failed);
    ASSERT_ERR(err, SPARSE_ERR_NOT_CONVERGED);
    ASSERT_TRUE(failed.sigma == NULL);
    ASSERT_TRUE(failed.U == NULL);
    ASSERT_TRUE(failed.Vt == NULL);
    sparse_svd_free(&failed);

    sparse_svd_t recovered = {0};
    if (!run_partial_svd_default(A, &recovered)) {
        sparse_svd_free(&recovered);
        sparse_free(A);
        return;
    }

    double sigma_error = partial_svd_corpus_sigma_error(&recovered);
    double max_av_resid = 0.0;
    double max_atu_resid = 0.0;
    if (!partial_svd_max_triplet_residuals(A, &recovered, PARTIAL_SVD_CORPUS_K, &max_av_resid,
                                           &max_atu_resid)) {
        sparse_svd_free(&recovered);
        sparse_free(A);
        return;
    }
    printf("    partial-SVD corpus recovery: sigma=%.3e, Av=%.3e, Atu=%.3e\n", sigma_error,
           max_av_resid, max_atu_resid);
    ASSERT_TRUE(sigma_error <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(max_av_resid <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(max_atu_resid <= PARTIAL_SVD_CORPUS_TOL);

    sparse_svd_free(&recovered);
    sparse_free(A);
}

static void test_partial_svd_corpus_full_rank_truncate_path(void) {
    SparseMatrix *A = make_partial_svd_clustered_repeated_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    const idx_t full_rank_k = PARTIAL_SVD_CORPUS_COLS;
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t svd = {0};
    sparse_err_t err = sparse_svd_partial(A, full_rank_k, &opts, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(svd.m, PARTIAL_SVD_CORPUS_ROWS);
    ASSERT_EQ(svd.n, PARTIAL_SVD_CORPUS_COLS);
    ASSERT_EQ(svd.k, full_rank_k);
    ASSERT_TRUE(svd.economy);
    ASSERT_NOT_NULL(svd.sigma);
    ASSERT_NOT_NULL(svd.U);
    ASSERT_NOT_NULL(svd.Vt);

    double expected[PARTIAL_SVD_CORPUS_COLS] = {10.0, 10.0, 9.999999, 4.0, 1.0, 0.0};
    double actual[PARTIAL_SVD_CORPUS_COLS] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (idx_t i = 0; i < full_rank_k; i++)
        actual[i] = svd.sigma[i];
    sort_descending(expected, PARTIAL_SVD_CORPUS_COLS);
    sort_descending(actual, full_rank_k);

    double max_sigma_error = 0.0;
    for (idx_t i = 0; i < full_rank_k; i++) {
        double error = fabs(actual[i] - expected[i]);
        if (error > max_sigma_error)
            max_sigma_error = error;
    }

    double max_av_resid = 0.0;
    double max_atu_resid = 0.0;
    if (!partial_svd_max_triplet_residuals(A, &svd, full_rank_k, &max_av_resid, &max_atu_resid)) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }
    double u_ortho = tf_dense_column_orthogonality_error(svd.U, svd.m, svd.k);
    double v_ortho = tf_svd_vt_row_orthogonality_error(svd.Vt, svd.k, svd.n, svd.k);
    printf("    partial-SVD corpus full-rank truncate: sigma=%.3e, Av=%.3e, Atu=%.3e, "
           "U_ortho=%.3e, V_ortho=%.3e\n",
           max_sigma_error, max_av_resid, max_atu_resid, u_ortho, v_ortho);
    ASSERT_TRUE(max_sigma_error <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(max_av_resid <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(max_atu_resid <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(u_ortho <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(v_ortho <= PARTIAL_SVD_CORPUS_TOL);

    sparse_svd_free(&svd);
    sparse_free(A);
}

int main(void) {
    TEST_SUITE_BEGIN("Partial SVD Corpus Tests");

    RUN_TEST(test_partial_svd_corpus_clustered_repeated_default_success);
    RUN_TEST(test_partial_svd_corpus_clustered_repeated_projectors);
    RUN_TEST(test_partial_svd_corpus_clustered_repeated_residuals);
    RUN_TEST(test_partial_svd_corpus_clustered_repeated_tight_budget_fail_closed);
    RUN_TEST(test_partial_svd_corpus_clustered_repeated_recovery_after_failure);
    RUN_TEST(test_partial_svd_corpus_full_rank_truncate_path);

    TEST_SUITE_END();
}
