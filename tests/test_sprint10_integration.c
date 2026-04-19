#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 199309L)
#define _POSIX_C_SOURCE 199309L
#endif
/**
 * Sprint 10 cross-feature integration tests.
 *
 * Validates that CSR LU, block solvers, and packaging features work
 * together correctly on real SuiteSparse matrices.
 */
#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_lu.h"
#include "sparse_lu_csr.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ── helpers ─────────────────────────────────────────────────────────── */

static double wall_time(void) {
#ifdef _WIN32
    /* Windows: use timespec_get (C11) as fallback */
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static double local_norm2(const double *v, idx_t n) {
    double s = 0.0;
    for (idx_t i = 0; i < n; i++)
        s += v[i] * v[i];
    return sqrt(s);
}

/* Compute ||Ax - b||/||b|| for a single RHS */
static double relative_residual(const SparseMatrix *A, const double *b, const double *x, idx_t n) {
    double *r = calloc((size_t)n, sizeof(double));
    if (!r)
        return INFINITY;
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];
    double rnorm = local_norm2(r, n);
    double bnorm = local_norm2(b, n);
    free(r);
    return (bnorm > 0.0) ? rnorm / bnorm : rnorm;
}

/* Compute max-column ||A*X(:,k) - B(:,k)||/||B(:,k)|| for block RHS */
static double block_relative_residual(const SparseMatrix *A, const double *B, const double *X,
                                      idx_t n, idx_t nrhs) {
    double *Y = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    if (!Y)
        return INFINITY;
    sparse_matvec_block(A, X, nrhs, Y);
    double worst = 0.0;
    for (idx_t k = 0; k < nrhs; k++) {
        double rnorm = 0.0, bnorm = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double ri = B[i + k * n] - Y[i + k * n];
            rnorm += ri * ri;
            bnorm += B[i + k * n] * B[i + k * n];
        }
        rnorm = sqrt(rnorm);
        bnorm = sqrt(bnorm);
        double rel = (bnorm > 0.0) ? rnorm / bnorm : rnorm;
        if (rel > worst)
            worst = rel;
    }
    free(Y);
    return worst;
}

/* Generate b = A * x_exact where x_exact = [1, 2, ..., n] */
static void make_rhs(const SparseMatrix *A, double *b, double *x_exact, idx_t n) {
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);
}

/* Generate B = A * X_exact for nrhs columns */
static void make_block_rhs(const SparseMatrix *A, double *B, double *X_exact, idx_t n, idx_t nrhs) {
    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            X_exact[i + k * n] = (double)(i + 1) * (double)(k + 1);
    sparse_matvec_block(A, X_exact, nrhs, B);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 1. CSR LU on all SuiteSparse matrices
 * ═══════════════════════════════════════════════════════════════════════ */

static void csr_solve_matrix(const char *path, const char *name) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, path);
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t n = sparse_rows(A);
    double *b = calloc((size_t)n, sizeof(double));
    double *x_exact = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    if (!b || !x_exact || !x) {
        free(x);
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }

    make_rhs(A, b, x_exact, n);
    sparse_err_t err = lu_csr_factor_solve(A, b, x, 1e-12);
    ASSERT_ERR(err, SPARSE_OK);

    double rr = relative_residual(A, b, x, n);
    printf("    %s (%d×%d): relres = %.2e\n", name, (int)n, (int)n, rr);
    ASSERT_TRUE(rr < 1e-10);

    free(x);
    free(x_exact);
    free(b);
    sparse_free(A);
}

static void test_csr_solve_nos4(void) { csr_solve_matrix(SS_DIR "/nos4.mtx", "nos4"); }

static void test_csr_solve_bcsstk04(void) { csr_solve_matrix(SS_DIR "/bcsstk04.mtx", "bcsstk04"); }

static void test_csr_solve_west0067(void) { csr_solve_matrix(SS_DIR "/west0067.mtx", "west0067"); }

static void test_csr_solve_steam1(void) { csr_solve_matrix(SS_DIR "/steam1.mtx", "steam1"); }

static void test_csr_solve_fs_541_1(void) { csr_solve_matrix(SS_DIR "/fs_541_1.mtx", "fs_541_1"); }

static void test_csr_solve_orsirr_1(void) { csr_solve_matrix(SS_DIR "/orsirr_1.mtx", "orsirr_1"); }

/* ═══════════════════════════════════════════════════════════════════════
 * 2. CSR vs linked-list residual agreement
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_csr_vs_linkedlist_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t n = sparse_rows(A);
    double *b = calloc((size_t)n, sizeof(double));
    double *x_exact = calloc((size_t)n, sizeof(double));
    double *x_csr = calloc((size_t)n, sizeof(double));
    double *x_ll = calloc((size_t)n, sizeof(double));
    if (!b || !x_exact || !x_csr || !x_ll) {
        free(x_ll);
        free(x_csr);
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }

    make_rhs(A, b, x_exact, n);

    /* CSR path */
    ASSERT_ERR(lu_csr_factor_solve(A, b, x_csr, 1e-12), SPARSE_OK);

    /* Linked-list path */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    if (!LU) {
        free(x_ll);
        free(x_csr);
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU, b, x_ll), SPARSE_OK);

    double rr_csr = relative_residual(A, b, x_csr, n);
    double rr_ll = relative_residual(A, b, x_ll, n);
    printf("    nos4: CSR relres=%.2e  LL relres=%.2e\n", rr_csr, rr_ll);
    ASSERT_TRUE(rr_csr < 1e-10);
    ASSERT_TRUE(rr_ll < 1e-10);

    /* Solutions should agree to machine precision (same problem) */
    double diff = 0.0;
    for (idx_t i = 0; i < n; i++)
        diff += (x_csr[i] - x_ll[i]) * (x_csr[i] - x_ll[i]);
    diff = sqrt(diff) / local_norm2(x_csr, n);
    printf("    solution diff: %.2e\n", diff);
    ASSERT_TRUE(diff < 1e-8);

    sparse_free(LU);
    free(x_ll);
    free(x_csr);
    free(x_exact);
    free(b);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 3. CSR LU speedup benchmark on orsirr_1
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_csr_speedup_orsirr_1(void) {
    /* Skip by default to keep unit tests fast; set RUN_BENCH=1 to enable */
    if (!getenv("RUN_BENCH")) {
        printf("    (skipped — set RUN_BENCH=1 to run timing benchmark)\n");
        return;
    }

    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/orsirr_1.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t n = sparse_rows(A);
    double *b = calloc((size_t)n, sizeof(double));
    double *x_exact = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    if (!b || !x_exact || !x) {
        free(x);
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    make_rhs(A, b, x_exact, n);

    int reps = 2;

    /* Linked-list path */
    double t0 = wall_time();
    for (int r = 0; r < reps; r++) {
        SparseMatrix *LU = sparse_copy(A);
        if (!LU) {
            free(x);
            free(x_exact);
            free(b);
            sparse_free(A);
            return;
        }
        ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
        ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);
        sparse_free(LU);
    }
    double t_ll = (wall_time() - t0) / reps;

    /* CSR path */
    t0 = wall_time();
    for (int r = 0; r < reps; r++) {
        sparse_err_t csr_rc = lu_csr_factor_solve(A, b, x, 1e-12);
        ASSERT_ERR(csr_rc, SPARSE_OK);
        if (csr_rc != SPARSE_OK) {
            free(x);
            free(x_exact);
            free(b);
            sparse_free(A);
            return;
        }
    }
    double t_csr = (wall_time() - t0) / reps;

    double speedup = t_ll / t_csr;
    printf("    orsirr_1 (%d×%d): LL=%.4fs  CSR=%.4fs  speedup=%.1fx\n", (int)n, (int)n, t_ll,
           t_csr, speedup);

    /* Log speedup but don't assert — wall-clock timing is flaky across CI */
    if (speedup < 2.0)
        printf("    NOTE: speedup %.1fx below 2x target (may vary by platform)\n", speedup);
    if (speedup <= 1.0)
        printf("    NOTE: CSR was not faster in this run (timing may vary by platform)\n");

    free(x);
    free(x_exact);
    free(b);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 4. Block solver cross-validation: LU, CG, GMRES on same SPD system
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_block_solvers_cross_validate_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t n = sparse_rows(A);
    idx_t nrhs = 5;
    size_t block_sz = (size_t)n * (size_t)nrhs;
    double *B = calloc(block_sz, sizeof(double));
    double *X_exact = calloc(block_sz, sizeof(double));
    double *X_lu = calloc(block_sz, sizeof(double));
    double *X_cg = calloc(block_sz, sizeof(double));
    double *X_gmres = calloc(block_sz, sizeof(double));
    if (!B || !X_exact || !X_lu || !X_cg || !X_gmres) {
        free(X_gmres);
        free(X_cg);
        free(X_lu);
        free(X_exact);
        free(B);
        sparse_free(A);
        return;
    }

    make_block_rhs(A, B, X_exact, n, nrhs);

    /* Block LU solve (must factor first) */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    if (!LU) {
        free(X_gmres);
        free(X_cg);
        free(X_lu);
        free(X_exact);
        free(B);
        sparse_free(A);
        return;
    }
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve_block(LU, B, nrhs, X_lu), SPARSE_OK);
    sparse_free(LU);
    double rr_lu = block_relative_residual(A, B, X_lu, n, nrhs);
    printf("    block LU:    max relres = %.2e\n", rr_lu);
    ASSERT_TRUE(rr_lu < 1e-10);

    /* Block CG solve (nos4 is SPD) */
    sparse_iter_opts_t cg_opts = {.max_iter = 2000, .tol = 1e-12};
    sparse_iter_result_t cg_res;
    ASSERT_ERR(sparse_cg_solve_block(A, B, nrhs, X_cg, &cg_opts, NULL, NULL, &cg_res), SPARSE_OK);
    double rr_cg = block_relative_residual(A, B, X_cg, n, nrhs);
    printf("    block CG:    max relres = %.2e  iters = %d\n", rr_cg, (int)cg_res.iterations);
    ASSERT_TRUE(rr_cg < 1e-8);

    /* Block GMRES solve */
    sparse_gmres_opts_t gm_opts = {.max_iter = 2000, .restart = 50, .tol = 1e-12};
    sparse_iter_result_t gm_res;
    ASSERT_ERR(sparse_gmres_solve_block(A, B, nrhs, X_gmres, &gm_opts, NULL, NULL, &gm_res),
               SPARSE_OK);
    double rr_gm = block_relative_residual(A, B, X_gmres, n, nrhs);
    printf("    block GMRES: max relres = %.2e  iters = %d\n", rr_gm, (int)gm_res.iterations);
    ASSERT_TRUE(rr_gm < 1e-8);

    /* All three solutions should agree */
    double max_diff = 0.0;
    for (idx_t k = 0; k < nrhs; k++) {
        for (idx_t i = 0; i < n; i++) {
            double d1 = fabs(X_lu[i + k * n] - X_cg[i + k * n]);
            double d2 = fabs(X_lu[i + k * n] - X_gmres[i + k * n]);
            if (d1 > max_diff)
                max_diff = d1;
            if (d2 > max_diff)
                max_diff = d2;
        }
    }
    printf("    max |x_lu - x_cg/gmres| = %.2e\n", max_diff);
    ASSERT_TRUE(max_diff < 1e-6);

    free(X_gmres);
    free(X_cg);
    free(X_lu);
    free(X_exact);
    free(B);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 5. Preconditioned block solvers on steam1 (general, non-SPD)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_preconditioned_block_gmres_steam1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/steam1.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t n = sparse_rows(A);
    idx_t nrhs = 3;
    size_t block_sz = (size_t)n * (size_t)nrhs;
    double *B = calloc(block_sz, sizeof(double));
    double *X_exact = calloc(block_sz, sizeof(double));
    double *X = calloc(block_sz, sizeof(double));
    if (!B || !X_exact || !X) {
        free(X);
        free(X_exact);
        free(B);
        sparse_free(A);
        return;
    }

    make_block_rhs(A, B, X_exact, n, nrhs);

    /* ILU(0) preconditioner */
    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    sparse_err_t ilu_err = sparse_ilu_factor(A, &ilu);
    ASSERT_ERR(ilu_err, SPARSE_OK);
    if (ilu_err != SPARSE_OK) {
        sparse_ilu_free(&ilu);
        free(X);
        free(X_exact);
        free(B);
        sparse_free(A);
        return;
    }

    sparse_gmres_opts_t opts = {.max_iter = 1000, .restart = 50, .tol = 1e-10};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_gmres_solve_block(A, B, nrhs, X, &opts, sparse_ilu_precond, &ilu, &res),
               SPARSE_OK);

    double rr = block_relative_residual(A, B, X, n, nrhs);
    printf("    steam1 (%d×%d, %d RHS): ILU+GMRES relres=%.2e iters=%d\n", (int)n, (int)n,
           (int)nrhs, rr, (int)res.iterations);
    ASSERT_TRUE(rr < 1e-6);

    sparse_ilu_free(&ilu);
    free(X);
    free(X_exact);
    free(B);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 6. CSR block solve matches single-column solve
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_csr_block_vs_single_solve(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t n = sparse_rows(A);
    idx_t nrhs = 4;

    double *B = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    double *X_exact = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    double *X_block = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    double *x_single = calloc((size_t)n, sizeof(double));
    if (!B || !X_exact || !X_block || !x_single) {
        free(x_single);
        free(X_block);
        free(X_exact);
        free(B);
        sparse_free(A);
        return;
    }

    make_block_rhs(A, B, X_exact, n, nrhs);

    /* Factor once */
    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);
    if (!csr) {
        free(x_single);
        free(X_block);
        free(X_exact);
        free(B);
        sparse_free(A);
        return;
    }
    idx_t *piv = calloc((size_t)n, sizeof(idx_t));
    if (!piv) {
        lu_csr_free(csr);
        free(x_single);
        free(X_block);
        free(X_exact);
        free(B);
        sparse_free(A);
        return;
    }
    sparse_err_t elim_err = lu_csr_eliminate(csr, 1e-12, 1e-14, piv);
    ASSERT_ERR(elim_err, SPARSE_OK);
    if (elim_err != SPARSE_OK) {
        free(piv);
        lu_csr_free(csr);
        free(x_single);
        free(X_block);
        free(X_exact);
        free(B);
        sparse_free(A);
        return;
    }

    /* Block solve */
    ASSERT_ERR(lu_csr_solve_block(csr, piv, B, nrhs, X_block), SPARSE_OK);

    /* Single-column solves must match */
    double max_diff = 0.0;
    for (idx_t k = 0; k < nrhs; k++) {
        ASSERT_ERR(lu_csr_solve(csr, piv, B + k * n, x_single), SPARSE_OK);
        for (idx_t i = 0; i < n; i++) {
            double d = fabs(X_block[i + k * n] - x_single[i]);
            if (d > max_diff)
                max_diff = d;
        }
    }
    printf("    west0067: block vs single max diff = %.2e\n", max_diff);
    ASSERT_TRUE(max_diff < 1e-11);

    free(piv);
    lu_csr_free(csr);
    free(x_single);
    free(X_block);
    free(X_exact);
    free(B);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 7. Version macros sanity check
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_version_macros(void) {
    /* Sprint 18 bumped the library to 2.0.0 because the
     * `sparse_cholesky_opts_t` ABI changed (new `backend` /
     * `used_csc_path` fields).  If you're touching this test and the
     * VERSION file says something different, update both. */
    ASSERT_EQ(SPARSE_VERSION_MAJOR, 2);
    ASSERT_EQ(SPARSE_VERSION_MINOR, 0);
    ASSERT_EQ(SPARSE_VERSION_PATCH, 0);
    ASSERT_EQ(SPARSE_VERSION, 20000);
    ASSERT_EQ(SPARSE_VERSION_ENCODE(2, 0, 0), SPARSE_VERSION);
    ASSERT_TRUE(SPARSE_VERSION_ENCODE(3, 0, 0) > SPARSE_VERSION);
    ASSERT_TRUE(SPARSE_VERSION_ENCODE(1, 0, 0) < SPARSE_VERSION);
    ASSERT_TRUE(strcmp(SPARSE_VERSION_STRING, "2.0.0") == 0);
    printf("    version: %s (int %d)\n", SPARSE_VERSION_STRING, SPARSE_VERSION);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 8. Backward compatibility: Sprint 9 API still works
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_backward_compat_lu_solve(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/nos4.mtx"), SPARSE_OK);
    if (!A)
        return;

    idx_t n = sparse_rows(A);
    double *b = calloc((size_t)n, sizeof(double));
    double *x_exact = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    if (!b || !x_exact || !x) {
        free(x);
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    make_rhs(A, b, x_exact, n);

    /* Old single-RHS API */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    if (!LU) {
        free(x);
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    double rr = relative_residual(A, b, x, n);
    printf("    backward compat LU: relres = %.2e\n", rr);
    ASSERT_TRUE(rr < 1e-10);

    sparse_free(LU);
    free(x);
    free(x_exact);
    free(b);
    sparse_free(A);
}

static void test_backward_compat_cg_single(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/nos4.mtx"), SPARSE_OK);
    if (!A)
        return;

    idx_t n = sparse_rows(A);
    double *b = calloc((size_t)n, sizeof(double));
    double *x_exact = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    if (!b || !x_exact || !x) {
        free(x);
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    make_rhs(A, b, x_exact, n);

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-12};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &res), SPARSE_OK);

    double rr = relative_residual(A, b, x, n);
    printf("    backward compat CG: relres = %.2e  iters = %d\n", rr, (int)res.iterations);
    ASSERT_TRUE(rr < 1e-8);

    free(x);
    free(x_exact);
    free(b);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 10 Integration");

    /* CSR LU on all SuiteSparse matrices */
    RUN_TEST(test_csr_solve_nos4);
    RUN_TEST(test_csr_solve_bcsstk04);
    RUN_TEST(test_csr_solve_west0067);
    RUN_TEST(test_csr_solve_steam1);
    RUN_TEST(test_csr_solve_fs_541_1);
    RUN_TEST(test_csr_solve_orsirr_1);

    /* CSR vs linked-list agreement */
    RUN_TEST(test_csr_vs_linkedlist_nos4);

    /* CSR speedup benchmark */
    RUN_TEST(test_csr_speedup_orsirr_1);

    /* Block solver cross-validation */
    RUN_TEST(test_block_solvers_cross_validate_nos4);

    /* Preconditioned block GMRES */
    RUN_TEST(test_preconditioned_block_gmres_steam1);

    /* CSR block vs single solve */
    RUN_TEST(test_csr_block_vs_single_solve);

    /* Version macros */
    RUN_TEST(test_version_macros);

    /* Backward compatibility */
    RUN_TEST(test_backward_compat_lu_solve);
    RUN_TEST(test_backward_compat_cg_single);

    TEST_SUITE_END();
}
