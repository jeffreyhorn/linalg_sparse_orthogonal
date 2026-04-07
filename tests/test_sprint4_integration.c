#include "sparse_cholesky.h"
#include "sparse_csr.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Cholesky → CSR → solve integration
 * ═══════════════════════════════════════════════════════════════════════ */

/* Factor → export L to CSR → import back → solve → correct result */
static void test_cholesky_csr_roundtrip_solve(void) {
    /* A = [[4,2,0],[2,5,1],[0,1,3]] — SPD */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 3.0);

    double b[] = {8.0, 15.0, 11.0};

    /* Factor */
    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    /* Export L to CSR */
    SparseCsr *csr = NULL;
    ASSERT_ERR(sparse_to_csr(L, &csr), SPARSE_OK);

    /* Verify CSR is lower triangular: all col_idx[k] <= row for that row */
    for (idx_t i = 0; i < 3; i++) {
        for (idx_t k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; k++) {
            ASSERT_TRUE(csr->col_idx[k] <= i);
        }
    }

    /* Import back from CSR */
    SparseMatrix *L2 = NULL;
    ASSERT_ERR(sparse_from_csr(csr, &L2), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(L2), sparse_nnz(L));

    /* Mark L2 as factored — the CSR roundtrip doesn't preserve factored
     * state.  This also computes factor_norm for the solve path. */
    ASSERT_ERR(sparse_mark_factored(L2), SPARSE_OK);
    double x[3];
    ASSERT_ERR(sparse_cholesky_solve(L2, b, x), SPARSE_OK);

    /* Verify via residual */
    double r[3];
    sparse_matvec(A, x, r);
    double rnorm = 0.0;
    for (int i = 0; i < 3; i++) {
        r[i] -= b[i];
        double a = fabs(r[i]);
        if (a > rnorm)
            rnorm = a;
    }
    ASSERT_TRUE(rnorm < 1e-12);

    sparse_csr_free(csr);
    sparse_free(A);
    sparse_free(L);
    sparse_free(L2);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SpMM: L * L^T reconstruction
 * ═══════════════════════════════════════════════════════════════════════ */

static SparseMatrix *build_lower_transpose(const SparseMatrix *L, idx_t n) {
    SparseMatrix *LT = sparse_create(n, n);
    if (!LT)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double val = sparse_get_phys(L, i, j);
            if (val != 0.0)
                sparse_insert(LT, j, i, val);
        }
    }
    return LT;
}

/* L * L^T should reconstruct A on a SuiteSparse matrix */
static void test_spmm_cholesky_reconstruct_nos4(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/nos4.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    SparseMatrix *LT = build_lower_transpose(L, n);
    ASSERT_NOT_NULL(LT);

    SparseMatrix *LLT = NULL;
    ASSERT_ERR(sparse_matmul(L, LT, &LLT), SPARSE_OK);

    /* Verify (L*L^T)*x ≈ A*x for x = ones */
    double *ones = malloc((size_t)n * sizeof(double));
    double *Ax = malloc((size_t)n * sizeof(double));
    double *LLTx = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(Ax);
    ASSERT_NOT_NULL(LLTx);
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;

    sparse_matvec(A, ones, Ax);
    sparse_matvec(LLT, ones, LLTx);

    double maxdiff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(Ax[i] - LLTx[i]);
        if (d > maxdiff)
            maxdiff = d;
    }
    printf("    nos4: L*L^T reconstruction maxdiff = %.2e\n", maxdiff);
    ASSERT_TRUE(maxdiff < 1e-8);

    free(ones);
    free(Ax);
    free(LLTx);
    sparse_free(A);
    sparse_free(L);
    sparse_free(LT);
    sparse_free(LLT);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Condest with Cholesky
 * ═══════════════════════════════════════════════════════════════════════ */

/* Factor SPD with Cholesky, use separate LU for condest, verify reasonable */
static void test_condest_via_lu_on_spd(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/nos4.mtx"), SPARSE_OK);

    /* Cholesky factor */
    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    /* LU factor for condest */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    double cond;
    ASSERT_ERR(sparse_lu_condest(A, LU, &cond), SPARSE_OK);
    ASSERT_TRUE(cond > 0.0);
    printf("    nos4: condest = %.3e (via LU on SPD matrix)\n", cond);

    /* Cholesky solve should produce residual consistent with condest */
    idx_t n = sparse_rows(A);
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x);
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);

    double maxerr = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(x[i] - 1.0);
        if (d > maxerr)
            maxerr = d;
    }
    ASSERT_TRUE(maxerr < 1e-10);

    free(ones);
    free(b);
    free(x);
    sparse_free(A);
    sparse_free(L);
    sparse_free(LU);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Concurrent Cholesky on SuiteSparse
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int thread_id;
    int success;
} chol_ss_arg_t;

static void *thread_cholesky_suitesparse(void *arg) {
    chol_ss_arg_t *ca = (chol_ss_arg_t *)arg;
    ca->success = 0;

    /* Each thread loads, factors, and solves independently */
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, SS_DIR "/nos4.mtx") != SPARSE_OK)
        return NULL;
    idx_t n = sparse_rows(A);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    if (!ones || !b || !x) {
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return NULL;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* Scale b by thread_id for variety */
    for (idx_t i = 0; i < n; i++)
        b[i] *= (double)(ca->thread_id + 1);

    SparseMatrix *L = sparse_copy(A);
    if (!L) {
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return NULL;
    }
    if (sparse_cholesky_factor(L) != SPARSE_OK) {
        sparse_free(L);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return NULL;
    }

    if (sparse_cholesky_solve(L, b, x) != SPARSE_OK) {
        sparse_free(L);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return NULL;
    }

    /* Verify residual */
    double *r = malloc((size_t)n * sizeof(double));
    if (!r) {
        sparse_free(L);
        free(ones);
        free(b);
        free(x);
        sparse_free(A);
        return NULL;
    }
    sparse_matvec(A, x, r);
    double maxerr = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(r[i] - b[i]);
        if (d > maxerr)
            maxerr = d;
    }

    ca->success = (maxerr < 1e-8) ? 1 : 0;

    free(ones);
    free(b);
    free(x);
    free(r);
    sparse_free(A);
    sparse_free(L);
    return NULL;
}

static void test_concurrent_cholesky_suitesparse(void) {
    int nthreads = 4;
    pthread_t threads[4];
    chol_ss_arg_t args[4];

    for (int t = 0; t < nthreads; t++) {
        args[t].thread_id = t;
        int rc = pthread_create(&threads[t], NULL, thread_cholesky_suitesparse, &args[t]);
        ASSERT_EQ(rc, 0);
    }

    int all_pass = 1;
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        if (!args[t].success)
            all_pass = 0;
    }
    printf("    concurrent Cholesky on nos4: %d threads, all_pass=%d\n", nthreads, all_pass);
    ASSERT_TRUE(all_pass);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CSR export of Cholesky factor → verify triangular structure
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_csr_cholesky_triangular(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/nos4.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    SparseCsr *csr = NULL;
    ASSERT_ERR(sparse_to_csr(L, &csr), SPARSE_OK);

    /* Every entry must be in the lower triangle: col_idx[k] <= i */
    int all_lower = 1;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; k++) {
            if (csr->col_idx[k] > i) {
                all_lower = 0;
                break;
            }
        }
    }
    ASSERT_TRUE(all_lower);
    printf("    nos4 Cholesky CSR: all %d entries in lower triangle\n", (int)csr->nnz);

    sparse_csr_free(csr);
    sparse_free(A);
    sparse_free(L);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 4 Integration Tests");

    RUN_TEST(test_cholesky_csr_roundtrip_solve);
    RUN_TEST(test_spmm_cholesky_reconstruct_nos4);
    RUN_TEST(test_condest_via_lu_on_spd);
    RUN_TEST(test_concurrent_cholesky_suitesparse);
    RUN_TEST(test_csr_cholesky_triangular);

    TEST_SUITE_END();
}
