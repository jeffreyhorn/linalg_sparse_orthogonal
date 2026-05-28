/*
 * bench_eigs_reuse.c — repeated one-shot vs reusable-workspace eigensolver
 * comparison for Sprint 46.
 *
 * Scope intentionally stays narrow:
 *   - grow-m Lanczos on nos4
 *   - thick-restart Lanczos on bcsstk14
 *
 * Reports median repeated-call wall time, last-run solver summaries, and a
 * simple speedup ratio. This is local evidence for reduced repeated allocation
 * churn, not a machine-independent performance claim.
 */
#define _POSIX_C_SOURCE 200809L

#include "sparse_eigs.h"
#include "sparse_eigs_internal.h"
#include "sparse_eigs_workspace_internal.h"
#include "sparse_matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y)
        return -1;
    if (x > y)
        return 1;
    return 0;
}

static double median_double(double *arr, idx_t n) {
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    return arr[n / 2];
}

static double max_abs_eig_diff(const sparse_eigs_t *a, const sparse_eigs_t *b) {
    idx_t common = (a->n_converged < b->n_converged) ? a->n_converged : b->n_converged;
    double max_diff = 0.0;
    for (idx_t i = 0; i < common; i++) {
        double diff = fabs(a->eigenvalues[i] - b->eigenvalues[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

static sparse_err_t run_repeated_case(const char *name, const char *path, idx_t k,
                                      sparse_eigs_backend_t backend, idx_t repeats) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, path);
    if (err != SPARSE_OK || !A)
        return (err == SPARSE_OK) ? SPARSE_ERR_ALLOC : err;

    double *vals_one = calloc((size_t)k, sizeof(double));
    double *vals_reuse = calloc((size_t)k, sizeof(double));
    double *times_one = calloc((size_t)repeats, sizeof(double));
    double *times_reuse = calloc((size_t)repeats, sizeof(double));
    if (!vals_one || !vals_reuse || !times_one || !times_reuse) {
        free(vals_one);
        free(vals_reuse);
        free(times_one);
        free(times_reuse);
        sparse_free(A);
        return SPARSE_ERR_ALLOC;
    }

    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .sigma = 0.0,
        .max_iterations = 0,
        .tol = 1e-10,
        .reorthogonalize = 1,
        .compute_vectors = 0,
        .backend = backend,
        .lobpcg_soft_lock = 1,
    };
    sparse_eigs_t one_shot = {.eigenvalues = vals_one};
    sparse_eigs_t reuse = {.eigenvalues = vals_reuse};
    sparse_err_t one_err = SPARSE_OK;
    sparse_err_t reuse_err = SPARSE_OK;

    for (idx_t rep = 0; rep < repeats; rep++) {
        memset(vals_one, 0, (size_t)k * sizeof(double));
        one_shot = (sparse_eigs_t){.eigenvalues = vals_one};
        double t0 = wall_time();
        err = sparse_eigs_sym(A, k, &opts, &one_shot);
        one_err = err;
        times_one[rep] = wall_time() - t0;
        if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED)
            goto cleanup;
    }

    sparse_eigs_workspace_t workspace;
    sparse_eigs_workspace_init(&workspace);
    for (idx_t rep = 0; rep < repeats; rep++) {
        memset(vals_reuse, 0, (size_t)k * sizeof(double));
        reuse = (sparse_eigs_t){.eigenvalues = vals_reuse};
        double t0 = wall_time();
        err = sparse_eigs_sym_with_workspace_internal(A, k, &opts, &reuse, &workspace);
        reuse_err = err;
        times_reuse[rep] = wall_time() - t0;
        if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED) {
            sparse_eigs_workspace_free(&workspace);
            goto cleanup;
        }
    }
    sparse_eigs_workspace_free(&workspace);

    double med_one = median_double(times_one, repeats);
    double med_reuse = median_double(times_reuse, repeats);
    int parity_match = (one_err == reuse_err) && (one_shot.n_converged == reuse.n_converged);
    double max_eig_diff = parity_match ? max_abs_eig_diff(&one_shot, &reuse) : NAN;

    printf("  %-24s one-shot=%8.4f ms  reuse=%8.4f ms  speedup=%5.2fx\n", name, med_one * 1000.0,
           med_reuse * 1000.0, med_reuse > 0.0 ? med_one / med_reuse : 0.0);
    printf("    last one-shot: iters=%4d conv=%d nconv=%d relres=%.3e peak=%d\n",
           (int)one_shot.iterations, (one_err == SPARSE_OK), (int)one_shot.n_converged,
           one_shot.residual_norm, (int)one_shot.peak_basis_size);
    printf("    last reuse:    iters=%4d conv=%d nconv=%d relres=%.3e peak=%d\n",
           (int)reuse.iterations, (reuse_err == SPARSE_OK), (int)reuse.n_converged,
           reuse.residual_norm, (int)reuse.peak_basis_size);
    if (!parity_match) {
        printf("    parity:        FAILED status_one=%d status_reuse=%d nconv_one=%d "
               "nconv_reuse=%d\n",
               (int)one_err, (int)reuse_err, (int)one_shot.n_converged, (int)reuse.n_converged);
        err = SPARSE_ERR_BADARG;
        goto cleanup;
    }
    if (one_err != SPARSE_OK) {
        printf("    parity:        FAILED matched non-success status=%d\n", (int)one_err);
        err = SPARSE_ERR_NOT_CONVERGED;
        goto cleanup;
    }
    printf("    parity:        |lambda|max diff=%.3e backend=%d\n", max_eig_diff,
           (int)reuse.backend_used);

cleanup:
    free(vals_one);
    free(vals_reuse);
    free(times_one);
    free(times_reuse);
    sparse_free(A);
    return err;
}

int main(void) {
    const idx_t growm_repeats = 40;
    const idx_t thick_repeats = 8;

    printf("=== Sprint 46 Eigensolver Workspace Reuse Benchmark ===\n\n");
    printf("Repeated-call comparison only; results are local evidence, not universal performance "
           "claims.\n\n");

    printf("Grow-m Lanczos repeated-run case (nos4, k=5, repeats=%d)\n", (int)growm_repeats);
    sparse_err_t err = run_repeated_case("growm-nos4-k5", "tests/data/suitesparse/nos4.mtx", 5,
                                         SPARSE_EIGS_BACKEND_LANCZOS, growm_repeats);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Grow-m repeated benchmark failed: %s\n", sparse_strerror(err));
        return 1;
    }

    printf("\nThick-restart repeated-run case (bcsstk14, k=5, repeats=%d)\n", (int)thick_repeats);
    err = run_repeated_case("thick-bcsstk14-k5", "tests/data/suitesparse/bcsstk14.mtx", 5,
                            SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART, thick_repeats);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Thick-restart repeated benchmark failed: %s\n", sparse_strerror(err));
        return 1;
    }

    return 0;
}
