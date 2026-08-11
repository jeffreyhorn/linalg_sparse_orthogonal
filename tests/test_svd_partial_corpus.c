#include "sparse_matrix.h"
#include "sparse_svd.h"
#include "test_framework.h"
#include "test_svd_helpers.h"
#include "test_svd_partial_shared_helpers.h"

#include <math.h>
#include <stdlib.h>

#define PARTIAL_SVD_CORPUS_ROWS 8
#define PARTIAL_SVD_CORPUS_COLS 6
#define PARTIAL_SVD_CORPUS_K 3
#define PARTIAL_SVD_CORPUS_TOL 1e-8

#define PARTIAL_SVD_RANKDEF_FIXTURE_KEY "partial_svd_rankdef_diag6x4_k2_range_projector_v1"
#define PARTIAL_SVD_LOWRANK_FIXTURE_KEY "partial_svd_lowrank_rect5x7_k3_sparse_output_v1"
#define PARTIAL_SVD_FAIL_CLOSED_FIXTURE_KEY "partial_svd_fail_closed_diag6_k2_v1"
#define PARTIAL_SVD_CORPUS_TIGHT_TOL 1e-10

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

static SparseMatrix *make_partial_svd_rankdef_diag6x4_fixture(void) {
    const double diag[4] = {9.0, 6.0, 0.0, 0.0};
    return tf_svd_make_diag_matrix(6, 4, diag, 4);
}

static SparseMatrix *make_partial_svd_lowrank_rect5x7_fixture(void) {
    const double diag[5] = {8.0, 4.0, 2.0, 1.0, 0.0};
    return tf_svd_make_diag_matrix(5, 7, diag, 5);
}

static SparseMatrix *make_partial_svd_fail_closed_diag6_fixture(void) {
    const double diag[6] = {9.0, 6.0, 3.0, 1.0, 0.5, 0.25};
    return tf_svd_make_diag_matrix(6, 6, diag, 6);
}

static int run_partial_svd_fixture_default(const SparseMatrix *A, idx_t rows, idx_t cols, idx_t k,
                                           sparse_svd_t *svd) {
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_err_t err = sparse_svd_partial(A, k, &opts, svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK)
        return 0;

    ASSERT_EQ(svd->m, rows);
    ASSERT_EQ(svd->n, cols);
    ASSERT_EQ(svd->k, k);
    ASSERT_NOT_NULL(svd->sigma);
    ASSERT_NOT_NULL(svd->U);
    ASSERT_NOT_NULL(svd->Vt);
    return svd->sigma && svd->U && svd->Vt;
}

static int run_partial_svd_default(const SparseMatrix *A, sparse_svd_t *svd) {
    return run_partial_svd_fixture_default(A, PARTIAL_SVD_CORPUS_ROWS, PARTIAL_SVD_CORPUS_COLS,
                                           PARTIAL_SVD_CORPUS_K, svd);
}

static double partial_svd_topk_sigma_error(const sparse_svd_t *svd, const double *expected,
                                           idx_t k) {
    double actual_sorted[8] = {0.0};
    double expected_sorted[8] = {0.0};
    double max_error = 0.0;
    ASSERT_TRUE(k > 0);
    ASSERT_TRUE(k <= 8);
    if (k <= 0 || k > 8)
        return INFINITY;

    for (idx_t i = 0; i < k; i++) {
        actual_sorted[i] = svd->sigma[i];
        expected_sorted[i] = expected[i];
    }
    sort_descending(actual_sorted, k);
    sort_descending(expected_sorted, k);

    for (idx_t i = 0; i < k; i++) {
        double error = fabs(actual_sorted[i] - expected_sorted[i]);
        if (error > max_error)
            max_error = error;
    }
    return max_error;
}

static double partial_svd_corpus_sigma_error(const sparse_svd_t *svd) {
    const double expected[PARTIAL_SVD_CORPUS_K] = {10.0, 10.0, 9.999999};
    return partial_svd_topk_sigma_error(svd, expected, PARTIAL_SVD_CORPUS_K);
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

static void test_partial_svd_corpus_rankdef_diag6x4_k2_metadata_and_values(void) {
    SparseMatrix *A = make_partial_svd_rankdef_diag6x4_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    idx_t reported_rank = -1;
    sparse_err_t rank_err = sparse_svd_rank(A, PARTIAL_SVD_CORPUS_TOL, &reported_rank);
    ASSERT_ERR(rank_err, SPARSE_OK);
    if (rank_err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t svd = {0};
    if (!run_partial_svd_fixture_default(A, 6, 4, 2, &svd)) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }

    const double expected_sigma[2] = {9.0, 6.0};
    double sigma_error = partial_svd_topk_sigma_error(&svd, expected_sigma, 2);
    printf("    %s: rank=%d, sigma=%.3e\n", PARTIAL_SVD_RANKDEF_FIXTURE_KEY, (int)reported_rank,
           sigma_error);
    ASSERT_EQ(reported_rank, 2);
    ASSERT_TRUE(sigma_error <= PARTIAL_SVD_CORPUS_TOL);

    sparse_svd_free(&svd);
    sparse_free(A);
}

static void test_partial_svd_corpus_rankdef_diag6x4_k2_projectors_and_residuals(void) {
    SparseMatrix *A = make_partial_svd_rankdef_diag6x4_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_svd_t svd = {0};
    if (!run_partial_svd_fixture_default(A, 6, 4, 2, &svd)) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }

    double left_projector = partial_svd_u_coordinate_range_projector_error(&svd, 2);
    double right_projector = partial_svd_v_coordinate_range_projector_error(&svd, 2);
    double max_av_resid = 0.0;
    double max_atu_resid = 0.0;
    if (!partial_svd_max_triplet_residuals(A, &svd, 2, &max_av_resid, &max_atu_resid)) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }
    double u_ortho = tf_dense_column_orthogonality_error(svd.U, svd.m, svd.k);
    double v_ortho = tf_svd_vt_row_orthogonality_error(svd.Vt, svd.k, svd.n, svd.k);
    printf("    %s: PU=%.3e, PV=%.3e, Av=%.3e, Atu=%.3e, U_ortho=%.3e, "
           "V_ortho=%.3e\n",
           PARTIAL_SVD_RANKDEF_FIXTURE_KEY, left_projector, right_projector, max_av_resid,
           max_atu_resid, u_ortho, v_ortho);
    ASSERT_TRUE(left_projector <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(right_projector <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(max_av_resid <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(max_atu_resid <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(u_ortho <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(v_ortho <= PARTIAL_SVD_CORPUS_TOL);

    sparse_svd_free(&svd);
    sparse_free(A);
}

static void test_partial_svd_corpus_lowrank_rect5x7_k3_sparse_output(void) {
    SparseMatrix *A = make_partial_svd_lowrank_rect5x7_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    double *dense = NULL;
    sparse_err_t err = sparse_svd_lowrank(A, 3, &dense);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(dense);
    if (!dense) {
        sparse_free(A);
        return;
    }

    SparseMatrix *sp = NULL;
    err = sparse_svd_lowrank_sparse(A, 3, 0.0, &sp);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(dense);
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(sp);
    if (!sp) {
        free(dense);
        sparse_free(A);
        return;
    }

    double dense_error = tf_svd_dense_lowrank_frobenius_error(A, dense, 5, 7, 5);
    double sparse_dense_diff = tf_svd_sparse_dense_frobenius_diff(sp, dense, 5, 7, 5);
    idx_t retained_nnz = sparse_nnz(sp);
    printf("    %s: shape=%dx%d, nnz=%d, dense_err=%.12f, sparse_dense=%.3e\n",
           PARTIAL_SVD_LOWRANK_FIXTURE_KEY, (int)sparse_rows(sp), (int)sparse_cols(sp),
           (int)retained_nnz, dense_error, sparse_dense_diff);
    ASSERT_EQ(sparse_rows(sp), 5);
    ASSERT_EQ(sparse_cols(sp), 7);
    ASSERT_EQ(retained_nnz, 3);
    ASSERT_NEAR(sparse_get(sp, 0, 0), 8.0, PARTIAL_SVD_CORPUS_TIGHT_TOL);
    ASSERT_NEAR(sparse_get(sp, 1, 1), 4.0, PARTIAL_SVD_CORPUS_TIGHT_TOL);
    ASSERT_NEAR(sparse_get(sp, 2, 2), 2.0, PARTIAL_SVD_CORPUS_TIGHT_TOL);
    ASSERT_NEAR(sparse_get(sp, 3, 3), 0.0, PARTIAL_SVD_CORPUS_TIGHT_TOL);
    ASSERT_NEAR(dense_error, 1.0, PARTIAL_SVD_CORPUS_TIGHT_TOL);
    ASSERT_NEAR(sparse_dense_diff, 0.0, PARTIAL_SVD_CORPUS_TIGHT_TOL);

    free(dense);
    sparse_free(sp);
    sparse_free(A);
}

static void test_partial_svd_corpus_fail_closed_diag6_k2_recovery(void) {
    SparseMatrix *A = make_partial_svd_fail_closed_diag6_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_svd_opts_t tight_budget = {.compute_uv = 1, .economy = 1, .max_iter = 1, .tol = 0.0};
    sparse_svd_t failed = {0};
    sparse_err_t err = sparse_svd_partial(A, 2, &tight_budget, &failed);
    printf("    %s: tight_err=%d, failed_sigma=%p, failed_U=%p, failed_Vt=%p\n",
           PARTIAL_SVD_FAIL_CLOSED_FIXTURE_KEY, (int)err, (void *)failed.sigma, (void *)failed.U,
           (void *)failed.Vt);
    ASSERT_ERR(err, SPARSE_ERR_NOT_CONVERGED);
    ASSERT_EQ(failed.m, 6);
    ASSERT_EQ(failed.n, 6);
    ASSERT_EQ(failed.k, 2);
    ASSERT_TRUE(failed.sigma == NULL);
    ASSERT_TRUE(failed.U == NULL);
    ASSERT_TRUE(failed.Vt == NULL);
    sparse_svd_free(&failed);

    sparse_svd_t recovered = {0};
    if (!run_partial_svd_fixture_default(A, 6, 6, 2, &recovered)) {
        sparse_svd_free(&recovered);
        sparse_free(A);
        return;
    }

    const double expected_sigma[2] = {9.0, 6.0};
    double sigma_error = partial_svd_topk_sigma_error(&recovered, expected_sigma, 2);
    double max_av_resid = 0.0;
    double max_atu_resid = 0.0;
    if (!partial_svd_max_triplet_residuals(A, &recovered, 2, &max_av_resid, &max_atu_resid)) {
        sparse_svd_free(&recovered);
        sparse_free(A);
        return;
    }
    printf("    %s: recovery_sigma=%.3e, Av=%.3e, Atu=%.3e\n", PARTIAL_SVD_FAIL_CLOSED_FIXTURE_KEY,
           sigma_error, max_av_resid, max_atu_resid);
    ASSERT_TRUE(sigma_error <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(max_av_resid <= PARTIAL_SVD_CORPUS_TOL);
    ASSERT_TRUE(max_atu_resid <= PARTIAL_SVD_CORPUS_TOL);

    sparse_svd_free(&recovered);
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
    RUN_TEST(test_partial_svd_corpus_rankdef_diag6x4_k2_metadata_and_values);
    RUN_TEST(test_partial_svd_corpus_rankdef_diag6x4_k2_projectors_and_residuals);
    RUN_TEST(test_partial_svd_corpus_lowrank_rect5x7_k3_sparse_output);
    RUN_TEST(test_partial_svd_corpus_fail_closed_diag6_k2_recovery);

    TEST_SUITE_END();
}
