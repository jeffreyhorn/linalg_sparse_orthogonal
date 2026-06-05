/*
 * bench_iterative_reuse.c — repeated one-shot vs public repeated-run-handle
 * iterative solve comparison for Sprint 45 / Sprint 49.
 *
 * Keeps the scope intentionally narrow:
 *   - scalar CG on a generated SPD tridiagonal
 *   - scalar GMRES on a generated nonsymmetric tridiagonal
 *   - scalar MINRES on a generated symmetric-indefinite KKT system
 *
 * Reports repeated-call wall time, last-iteration summary, and a simple
 * speedup ratio. This is evidence for allocator-churn reduction through the
 * final public repeated-run contract, not a machine-independent performance
 * claim.
 */
#define _POSIX_C_SOURCE 200809L
#include "sparse_iterative.h"
#include "sparse_matrix.h"
#include "sparse_vector.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static SparseMatrix *make_spd_tridiag(idx_t n, double diag, double offdiag) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        if (sparse_insert(A, i, i, diag) != SPARSE_OK)
            goto fail;
        if (i > 0) {
            if (sparse_insert(A, i, i - 1, offdiag) != SPARSE_OK)
                goto fail;
            if (sparse_insert(A, i - 1, i, offdiag) != SPARSE_OK)
                goto fail;
        }
    }
    return A;
fail:
    sparse_free(A);
    return NULL;
}

static SparseMatrix *make_unsym_tridiag(idx_t n, double diag, double lower, double upper) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        if (sparse_insert(A, i, i, diag) != SPARSE_OK)
            goto fail;
        if (i > 0 && sparse_insert(A, i, i - 1, lower) != SPARSE_OK)
            goto fail;
        if (i + 1 < n && sparse_insert(A, i, i + 1, upper) != SPARSE_OK)
            goto fail;
    }
    return A;
fail:
    sparse_free(A);
    return NULL;
}

static SparseMatrix *make_kkt(idx_t nh, idx_t nc) {
    idx_t n = nh + nc;
    SparseMatrix *K = sparse_create(n, n);
    if (!K)
        return NULL;
    for (idx_t i = 0; i < nh; i++) {
        if (sparse_insert(K, i, i, 4.0) != SPARSE_OK)
            goto fail;
        if (i > 0) {
            if (sparse_insert(K, i, i - 1, -1.0) != SPARSE_OK)
                goto fail;
            if (sparse_insert(K, i - 1, i, -1.0) != SPARSE_OK)
                goto fail;
        }
    }
    for (idx_t c = 0; c < nc; c++) {
        idx_t j0 = (c * 2) % nh;
        idx_t j1 = (j0 + 1) % nh;
        if (sparse_insert(K, nh + c, j0, 1.0) != SPARSE_OK)
            goto fail;
        if (sparse_insert(K, j0, nh + c, 1.0) != SPARSE_OK)
            goto fail;
        if (sparse_insert(K, nh + c, j1, 1.0) != SPARSE_OK)
            goto fail;
        if (sparse_insert(K, j1, nh + c, 1.0) != SPARSE_OK)
            goto fail;
    }
    return K;
fail:
    sparse_free(K);
    return NULL;
}

static double compute_rel_residual(const SparseMatrix *A, const double *b, const double *x,
                                   idx_t n) {
    double *r = malloc((size_t)n * sizeof(double));
    if (!r)
        return nan("");
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];
    double rn = vec_norm2(r, n);
    double bn = vec_norm2(b, n);
    free(r);
    return (bn > 0.0) ? rn / bn : 0.0;
}

static sparse_err_t run_cg_repeated_case(const char *name, SparseMatrix *A, idx_t repeats) {
    idx_t n = sparse_rows(A);
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    if (!x_exact || !b || !x) {
        free(x_exact);
        free(b);
        free(x);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t one_shot = {0};
    sparse_iter_result_t reuse = {0};
    sparse_err_t err = SPARSE_OK;

    double t0 = wall_time();
    for (idx_t rep = 0; rep < repeats; rep++) {
        vec_zero(x, n);
        err = sparse_solve_cg(A, b, x, &opts, NULL, NULL, &one_shot);
        if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED)
            break;
    }
    double t_one_shot = wall_time() - t0;

    double one_shot_rr = compute_rel_residual(A, b, x, n);

    sparse_iter_handle_t handle = {0};
    err = sparse_iter_handle_prepare_cg(&handle, n);
    if (err != SPARSE_OK)
        goto cleanup;

    t0 = wall_time();
    for (idx_t rep = 0; rep < repeats; rep++) {
        vec_zero(x, n);
        err = sparse_solve_cg_with_handle(A, b, x, &opts, NULL, NULL, &reuse, &handle);
        if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED)
            break;
    }
    double t_reuse = wall_time() - t0;
    double reuse_rr = compute_rel_residual(A, b, x, n);
    sparse_iter_handle_free(&handle);

    if (err == SPARSE_OK || err == SPARSE_ERR_NOT_CONVERGED) {
        printf("  %-18s one-shot=%8.4f ms  reuse=%8.4f ms  speedup=%5.2fx\n", name,
               t_one_shot * 1000.0, t_reuse * 1000.0, t_reuse > 0.0 ? t_one_shot / t_reuse : 0.0);
        printf("    last one-shot: iters=%4d relres=%.3e conv=%d\n", (int)one_shot.iterations,
               one_shot_rr, one_shot.converged);
        printf("    last reuse:    iters=%4d relres=%.3e conv=%d\n", (int)reuse.iterations,
               reuse_rr, reuse.converged);
    }

cleanup:
    free(x_exact);
    free(b);
    free(x);
    return err;
}

static sparse_err_t run_gmres_repeated_case(const char *name, SparseMatrix *A, idx_t repeats) {
    idx_t n = sparse_rows(A);
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    if (!x_exact || !b || !x) {
        free(x_exact);
        free(b);
        free(x);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_gmres_opts_t opts = {.max_iter = 1000, .restart = 30, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t one_shot = {0};
    sparse_iter_result_t reuse = {0};
    sparse_err_t err = SPARSE_OK;

    double t0 = wall_time();
    for (idx_t rep = 0; rep < repeats; rep++) {
        vec_zero(x, n);
        err = sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &one_shot);
        if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED)
            break;
    }
    double t_one_shot = wall_time() - t0;

    double one_shot_rr = compute_rel_residual(A, b, x, n);

    sparse_iter_handle_t handle = {0};
    err = sparse_iter_handle_prepare_gmres(&handle, n, opts.restart);
    if (err != SPARSE_OK)
        goto cleanup;

    t0 = wall_time();
    for (idx_t rep = 0; rep < repeats; rep++) {
        vec_zero(x, n);
        err = sparse_solve_gmres_with_handle(A, b, x, &opts, NULL, NULL, &reuse, &handle);
        if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED)
            break;
    }
    double t_reuse = wall_time() - t0;
    double reuse_rr = compute_rel_residual(A, b, x, n);
    sparse_iter_handle_free(&handle);

    if (err == SPARSE_OK || err == SPARSE_ERR_NOT_CONVERGED) {
        printf("  %-18s one-shot=%8.4f ms  reuse=%8.4f ms  speedup=%5.2fx\n", name,
               t_one_shot * 1000.0, t_reuse * 1000.0, t_reuse > 0.0 ? t_one_shot / t_reuse : 0.0);
        printf("    last one-shot: iters=%4d relres=%.3e conv=%d\n", (int)one_shot.iterations,
               one_shot_rr, one_shot.converged);
        printf("    last reuse:    iters=%4d relres=%.3e conv=%d\n", (int)reuse.iterations,
               reuse_rr, reuse.converged);
    }

cleanup:
    free(x_exact);
    free(b);
    free(x);
    return err;
}

static sparse_err_t run_minres_repeated_case(const char *name, SparseMatrix *A, idx_t repeats) {
    idx_t n = sparse_rows(A);
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    if (!x_exact || !b || !x) {
        free(x_exact);
        free(b);
        free(x);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 1000, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t one_shot = {0};
    sparse_iter_result_t reuse = {0};
    sparse_err_t err = SPARSE_OK;

    double t0 = wall_time();
    for (idx_t rep = 0; rep < repeats; rep++) {
        vec_zero(x, n);
        err = sparse_solve_minres(A, b, x, &opts, NULL, NULL, &one_shot);
        if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED)
            break;
    }
    double t_one_shot = wall_time() - t0;
    if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED)
        goto cleanup;

    double one_shot_rr = compute_rel_residual(A, b, x, n);

    sparse_iter_handle_t handle = {0};
    err = sparse_iter_handle_prepare_minres(&handle, n);
    if (err != SPARSE_OK)
        goto cleanup;

    t0 = wall_time();
    for (idx_t rep = 0; rep < repeats; rep++) {
        vec_zero(x, n);
        err = sparse_solve_minres_with_handle(A, b, x, &opts, NULL, NULL, &reuse, &handle);
        if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED)
            break;
    }
    double t_reuse = wall_time() - t0;
    double reuse_rr = compute_rel_residual(A, b, x, n);
    sparse_iter_handle_free(&handle);

    if (err == SPARSE_OK || err == SPARSE_ERR_NOT_CONVERGED) {
        printf("  %-18s one-shot=%8.4f ms  reuse=%8.4f ms  speedup=%5.2fx\n", name,
               t_one_shot * 1000.0, t_reuse * 1000.0, t_reuse > 0.0 ? t_one_shot / t_reuse : 0.0);
        printf("    last one-shot: iters=%4d relres=%.3e conv=%d\n", (int)one_shot.iterations,
               one_shot_rr, one_shot.converged);
        printf("    last reuse:    iters=%4d relres=%.3e conv=%d\n", (int)reuse.iterations,
               reuse_rr, reuse.converged);
    }

cleanup:
    free(x_exact);
    free(b);
    free(x);
    return err;
}

int main(void) {
    const idx_t cg_repeats = 400;
    const idx_t gmres_repeats = 300;
    const idx_t minres_repeats = 250;

    printf("=== Sprint 45/49/54 Iterative Repeated-Run Benchmark ===\n\n");
    printf("Repeated-call comparison only; results are local evidence, not universal performance "
           "claims. The reuse path now exercises the public handle API.\n\n");

    SparseMatrix *cg_A = make_spd_tridiag(300, 4.0, -1.0);
    if (!cg_A) {
        fprintf(stderr, "Failed to build CG benchmark matrix\n");
        return 1;
    }
    SparseMatrix *gmres_A = make_unsym_tridiag(220, 4.0, -1.0, -0.5);
    if (!gmres_A) {
        fprintf(stderr, "Failed to build GMRES benchmark matrix\n");
        sparse_free(cg_A);
        return 1;
    }
    SparseMatrix *minres_A = make_kkt(30, 12);
    if (!minres_A) {
        fprintf(stderr, "Failed to build MINRES benchmark matrix\n");
        sparse_free(cg_A);
        sparse_free(gmres_A);
        return 1;
    }

    printf("CG repeated-solve case (SPD tridiag, repeats=%d)\n", (int)cg_repeats);
    sparse_err_t err = run_cg_repeated_case("cg-tridiag-300", cg_A, cg_repeats);
    if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED) {
        fprintf(stderr, "CG repeated benchmark failed: %s\n", sparse_strerror(err));
        sparse_free(cg_A);
        sparse_free(gmres_A);
        return 1;
    }

    printf("\nGMRES repeated-solve case (nonsymmetric tridiag, repeats=%d)\n", (int)gmres_repeats);
    err = run_gmres_repeated_case("gmres-unsym-220", gmres_A, gmres_repeats);
    if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED) {
        fprintf(stderr, "GMRES repeated benchmark failed: %s\n", sparse_strerror(err));
        sparse_free(cg_A);
        sparse_free(gmres_A);
        sparse_free(minres_A);
        return 1;
    }

    printf("\nMINRES repeated-solve case (KKT 42x42, repeats=%d)\n", (int)minres_repeats);
    err = run_minres_repeated_case("minres-kkt-42", minres_A, minres_repeats);
    if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED) {
        fprintf(stderr, "MINRES repeated benchmark failed: %s\n", sparse_strerror(err));
        sparse_free(cg_A);
        sparse_free(gmres_A);
        sparse_free(minres_A);
        return 1;
    }

    sparse_free(cg_A);
    sparse_free(gmres_A);
    sparse_free(minres_A);
    return 0;
}
