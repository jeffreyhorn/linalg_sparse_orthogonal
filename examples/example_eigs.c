/*
 * example_eigs.c — Compute symmetric eigenpairs with sparse_eigs_sym.
 *
 * Demonstrates the full Sprint 20-21 eigensolver surface across the
 * three concrete backends and the preconditioning hook:
 *
 *   (a) Five largest eigenvalues of a small SPD SuiteSparse matrix
 *       (nos4.mtx, n = 100) — the typical "modal analysis" query.
 *       Sprint 20 grow-m Lanczos (the AUTO default for n < 500).
 *   (b) Three eigenvalues nearest a shift point sigma on a KKT-style
 *       indefinite saddle-point matrix — exercises shift-invert mode
 *       and the composition with the Sprint 20 Days 4-6 LDL^T
 *       dispatch.
 *   (c) Sprint 21 LOBPCG with IC(0) preconditioning on bcsstk04
 *       (n = 132, cond ≈ 5e6) k = 3 SMALLEST — vanilla LOBPCG
 *       saturates the iteration cap on this fixture; with the
 *       Sprint 13 IC(0) factor plugged in via opts->precond, the
 *       same problem converges in dramatically fewer iterations.
 *
 * Each demo prints a per-pair residual check
 * (||A*v - lambda*v|| / (|lambda| * ||v||)) confirming each returned
 * Ritz pair satisfies the eigen-equation independently of the
 * solver's internal Wu/Simon bound.
 *
 * Build:
 *   cc -O2 -Iinclude -o example_eigs examples/example_eigs.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 *
 * Run (from the project root so the SuiteSparse fixtures resolve):
 *   ./build/example_eigs
 */
#include "sparse_eigs.h"
#include "sparse_ic.h"
#include "sparse_ilu.h"
#include "sparse_matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Build a small KKT-style indefinite saddle-point matrix:
 *   [ H  A^T ]       H = 4x4 SPD tridiag (diag 4, off-diag -1)
 *   [ A   0  ]       A = [[1,0,1,0],[0,1,0,1]]
 * Dimension 6, symmetric indefinite.  The eigenvalues are a mix of
 * positive and negative values — a natural shift-invert target. */
static SparseMatrix *build_kkt(void) {
    idx_t nh = 4;
    idx_t nc = 2;
    idx_t n = nh + nc;
    SparseMatrix *K = sparse_create(n, n);
    if (!K)
        return NULL;
    for (idx_t i = 0; i < nh; i++) {
        sparse_insert(K, i, i, 4.0);
        if (i > 0) {
            sparse_insert(K, i, i - 1, -1.0);
            sparse_insert(K, i - 1, i, -1.0);
        }
    }
    sparse_insert(K, nh, 0, 1.0);
    sparse_insert(K, 0, nh, 1.0);
    sparse_insert(K, nh, 2, 1.0);
    sparse_insert(K, 2, nh, 1.0);
    sparse_insert(K, nh + 1, 1, 1.0);
    sparse_insert(K, 1, nh + 1, 1.0);
    sparse_insert(K, nh + 1, 3, 1.0);
    sparse_insert(K, 3, nh + 1, 1.0);
    return K;
}

/* Compute ||A*v - lambda*v|| / (|lambda| * ||v||) for a Ritz pair. */
static double ritz_residual(const SparseMatrix *A, double lambda, const double *v, idx_t n,
                            double *Av_scratch) {
    sparse_matvec(A, v, Av_scratch);
    double num = 0.0;
    double v_sq = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double r = Av_scratch[i] - lambda * v[i];
        num += r * r;
        v_sq += v[i] * v[i];
    }
    double lambda_abs = fabs(lambda);
    double anchor = (lambda_abs > 0.0 ? lambda_abs : 1.0) * (v_sq > 0.0 ? sqrt(v_sq) : 1.0);
    return sqrt(num) / anchor;
}

int main(void) {
    printf("=== Sparse symmetric eigensolver (Sprints 20-21) ===\n\n");

    /* ── (a) Five largest eigenvalues of nos4 (SuiteSparse SPD) ───── */
    printf("(a) Five largest eigenvalues of nos4.mtx (n = 100 SPD)\n");
    printf("---------------------------------------------------------\n");
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, "tests/data/suitesparse/nos4.mtx");
    if (err != SPARSE_OK) {
        fprintf(stderr, "Could not load nos4.mtx (run from project root): %s\n",
                sparse_strerror(err));
        return 1;
    }
    idx_t n = sparse_rows(A);
    printf("  Loaded A: n = %d, nnz = %d\n", (int)n, (int)sparse_nnz(A));

    idx_t k = 5;
    double vals[5] = {0};
    double *vecs = calloc((size_t)n * (size_t)k, sizeof(double));
    if (!vecs) {
        fprintf(stderr, "Allocation failed\n");
        sparse_free(A);
        return 1;
    }
    sparse_eigs_t res = {.eigenvalues = vals, .eigenvectors = vecs};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-10,
        .compute_vectors = 1,
        .reorthogonalize = 1,
    };
    err = sparse_eigs_sym(A, k, &opts, &res);
    if (err != SPARSE_OK) {
        fprintf(stderr, "sparse_eigs_sym failed: %s\n", sparse_strerror(err));
        free(vecs);
        sparse_free(A);
        return 1;
    }
    printf("  Converged: %d / %d pairs in %d Lanczos iterations\n", (int)res.n_converged, (int)k,
           (int)res.iterations);
    printf("  Reported residual_norm (Wu/Simon): %.3e\n", res.residual_norm);

    /* Check each pair against the eigen-equation (self-validating —
     * the solver's Wu/Simon bound and this direct check should agree
     * to within round-off for a converged run). */
    double *Av = malloc((size_t)n * sizeof(double));
    if (!Av) {
        fprintf(stderr, "Allocation failed\n");
        free(vecs);
        sparse_free(A);
        return 1;
    }
    printf("  %-4s %-22s %-12s\n", "j", "lambda_j", "||Av-lv||/(|l|.|v|)");
    for (idx_t j = 0; j < res.n_converged; j++) {
        double rel = ritz_residual(A, vals[j], vecs + (size_t)j * (size_t)n, n, Av);
        printf("  %-4d %-22.15g %-12.3e\n", (int)j, vals[j], rel);
    }
    free(Av);
    free(vecs);
    sparse_free(A);

    /* ── (b) Three eigenvalues nearest sigma on a KKT fixture ─────── */
    printf("\n(b) Three eigenvalues nearest sigma = 0 on a KKT indefinite matrix\n");
    printf("-------------------------------------------------------------------\n");
    SparseMatrix *K = build_kkt();
    if (!K) {
        fprintf(stderr, "KKT allocation failed\n");
        return 1;
    }
    idx_t nk = sparse_rows(K);
    printf("  KKT matrix K: n = %d, nnz = %d (symmetric indefinite)\n", (int)nk,
           (int)sparse_nnz(K));

    double kvals[3] = {0};
    double *kvecs = calloc((size_t)nk * 3, sizeof(double));
    if (!kvecs) {
        fprintf(stderr, "Allocation failed\n");
        sparse_free(K);
        return 1;
    }
    sparse_eigs_t kres = {.eigenvalues = kvals, .eigenvectors = kvecs};
    sparse_eigs_opts_t kopts = {
        .which = SPARSE_EIGS_NEAREST_SIGMA,
        .sigma = 0.0,
        .tol = 1e-10,
        .compute_vectors = 1,
        .reorthogonalize = 1,
    };
    err = sparse_eigs_sym(K, 3, &kopts, &kres);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Shift-invert sparse_eigs_sym failed: %s\n", sparse_strerror(err));
        free(kvecs);
        sparse_free(K);
        return 1;
    }
    printf("  Converged: %d / 3 pairs in %d Lanczos iterations\n", (int)kres.n_converged,
           (int)kres.iterations);
    printf("  Inner LDL^T factor routed through CSC supernodal: %s\n",
           kres.used_csc_path_ldlt ? "yes" : "no (below threshold — linked-list path)");

    double *KAv = malloc((size_t)nk * sizeof(double));
    if (!KAv) {
        fprintf(stderr, "Allocation failed\n");
        free(kvecs);
        sparse_free(K);
        return 1;
    }
    printf("  %-4s %-22s %-12s\n", "j", "lambda_j", "|lambda - sigma|");
    for (idx_t j = 0; j < kres.n_converged; j++) {
        double rel = ritz_residual(K, kvals[j], kvecs + (size_t)j * (size_t)nk, nk, KAv);
        printf("  %-4d %-22.15g dist = %-8.4g residual = %.3e\n", (int)j, kvals[j],
               fabs(kvals[j] - kopts.sigma), rel);
    }
    free(KAv);
    free(kvecs);
    sparse_free(K);

    /* ── (c) LOBPCG with IC(0) preconditioning on bcsstk04 ─────────── */
    printf("\n(c) Three smallest eigenvalues of bcsstk04 (n = 132 SPD, cond ~ 5e6)\n");
    printf("    LOBPCG with Sprint 13 IC(0) preconditioning.\n");
    printf("---------------------------------------------------------------------\n");
    SparseMatrix *B = NULL;
    err = sparse_load_mm(&B, "tests/data/suitesparse/bcsstk04.mtx");
    if (err != SPARSE_OK) {
        fprintf(stderr, "Could not load bcsstk04.mtx (run from project root): %s\n",
                sparse_strerror(err));
        return 1;
    }
    idx_t nb = sparse_rows(B);
    printf("  Loaded B: n = %d, nnz = %d\n", (int)nb, (int)sparse_nnz(B));

    /* Build the IC(0) factor once.  sparse_ic_precond + the factor
     * struct plug straight into the Sprint 21 sparse_eigs_opts_t
     * preconditioner hook — no adapter glue required. */
    sparse_ilu_t ic = {0};
    err = sparse_ic_factor(B, &ic);
    if (err != SPARSE_OK) {
        fprintf(stderr, "sparse_ic_factor failed: %s\n", sparse_strerror(err));
        sparse_free(B);
        return 1;
    }

    idx_t kb = 3;
    double bvals[3] = {0};
    double *bvecs = calloc((size_t)nb * (size_t)kb, sizeof(double));
    if (!bvecs) {
        fprintf(stderr, "Allocation failed\n");
        sparse_ic_free(&ic);
        sparse_free(B);
        return 1;
    }
    sparse_eigs_t bres = {.eigenvalues = bvals, .eigenvectors = bvecs};
    sparse_eigs_opts_t bopts = {
        .which = SPARSE_EIGS_SMALLEST,
        .tol = 1e-8,
        .compute_vectors = 1,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_LOBPCG,
        .max_iterations = 200,
        .precond = sparse_ic_precond,
        .precond_ctx = &ic,
        .lobpcg_soft_lock = 1,
    };
    err = sparse_eigs_sym(B, kb, &bopts, &bres);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Preconditioned LOBPCG failed: %s\n", sparse_strerror(err));
        free(bvecs);
        sparse_ic_free(&ic);
        sparse_free(B);
        return 1;
    }
    printf("  Converged: %d / %d pairs in %d outer iterations\n", (int)bres.n_converged, (int)kb,
           (int)bres.iterations);
    printf("  Backend used (per AUTO/explicit dispatch): %s\n",
           bres.backend_used == SPARSE_EIGS_BACKEND_LOBPCG ? "LOBPCG" : "(other)");
    printf("  Reported residual_norm: %.3e\n", bres.residual_norm);

    double *BAv = malloc((size_t)nb * sizeof(double));
    if (!BAv) {
        fprintf(stderr, "Allocation failed\n");
        free(bvecs);
        sparse_ic_free(&ic);
        sparse_free(B);
        return 1;
    }
    printf("  %-4s %-22s %-12s\n", "j", "lambda_j", "||Av-lv||/(|l|.|v|)");
    for (idx_t j = 0; j < bres.n_converged; j++) {
        double rel = ritz_residual(B, bvals[j], bvecs + (size_t)j * (size_t)nb, nb, BAv);
        printf("  %-4d %-22.15g %-12.3e\n", (int)j, bvals[j], rel);
    }
    free(BAv);
    free(bvecs);
    sparse_ic_free(&ic);
    sparse_free(B);

    printf("\nAll done.\n");
    return 0;
}
