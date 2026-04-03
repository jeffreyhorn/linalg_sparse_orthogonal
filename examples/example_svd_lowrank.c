/*
 * example_svd_lowrank.c — Low-rank matrix approximation via SVD.
 *
 * Demonstrates:
 *   - Full SVD computation
 *   - Singular value spectrum
 *   - Best rank-k approximation
 *   - Compression ratio and approximation error
 *   - Condition number estimation
 *
 * Build:
 *   cc -O2 -Iinclude -o example_svd_lowrank examples/example_svd_lowrank.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_matrix.h"
#include "sparse_svd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== SVD Low-Rank Approximation Example ===\n\n");

    /* Create an 8x8 matrix with a rapidly decaying spectrum.
     * A = sum_{i=0}^{7} sigma_i * e_i * e_i^T  with sigma = [100, 50, 10, 5, 1, 0.1, 0.01, 0.001]
     * Plus some off-diagonal structure. */
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    if (!A) {
        fprintf(stderr, "Failed to create matrix\n");
        return 1;
    }

    double diag_vals[] = {100.0, 50.0, 10.0, 5.0, 1.0, 0.1, 0.01, 0.001};
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag_vals[i]);
        /* Add small off-diagonal coupling */
        if (i + 1 < n)
            sparse_insert(A, i, i + 1, 0.5);
        if (i > 0)
            sparse_insert(A, i, i - 1, 0.5);
    }

    printf("Matrix A (%d x %d, %d nonzeros)\n", (int)n, (int)n, (int)sparse_nnz(A));

    /* Full SVD (singular values only) */
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    if (err != SPARSE_OK) {
        fprintf(stderr, "SVD failed (err=%d)\n", (int)err);
        sparse_free(A);
        return 1;
    }

    printf("\nSingular value spectrum:\n");
    for (idx_t i = 0; i < svd.k; i++)
        printf("  sigma[%d] = %12.6f\n", (int)i, svd.sigma[i]);

    /* Condition number */
    sparse_err_t cond_err;
    double cond = sparse_cond(A, &cond_err);
    printf("\nCondition number: %.2e\n", cond);

    /* Numerical rank at different tolerances */
    idx_t rank_default = 0;
    sparse_svd_rank(A, 0.0, &rank_default);
    idx_t rank_strict = 0;
    sparse_svd_rank(A, 1e-6, &rank_strict);
    printf("Numerical rank (default tol): %d\n", (int)rank_default);
    printf("Numerical rank (tol=1e-6):    %d\n", (int)rank_strict);

    /* Low-rank approximation for several values of k */
    printf("\nLow-rank approximation quality:\n");
    printf("  %-4s  %-15s  %-15s  %-10s\n", "k", "||A - A_k||_F", "sigma_{k+1}", "ratio");
    printf("  %-4s  %-15s  %-15s  %-10s\n", "----", "---------------", "---------------",
           "----------");

    for (idx_t rank_k = 1; rank_k <= 5; rank_k++) {
        double *lowrank = NULL;
        err = sparse_svd_lowrank(A, rank_k, &lowrank);
        if (err != SPARSE_OK || !lowrank)
            continue;

        /* ||A - A_k||_F */
        double frob_sq = 0.0;
        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < n; j++) {
                double a_ij = sparse_get(A, i, j);
                double lr_ij = lowrank[(size_t)j * (size_t)n + (size_t)i]; /* col-major */
                double diff = a_ij - lr_ij;
                frob_sq += diff * diff;
            }
        }
        double frob = sqrt(frob_sq);

        /* Theoretical error = sqrt(sum sigma_{k+1}^2 + ...) */
        double sigma_next = (rank_k < svd.k) ? svd.sigma[rank_k] : 0.0;

        printf("  %-4d  %-15.6f  %-15.6f  %.2f%%\n", (int)rank_k, frob, sigma_next,
               (svd.sigma[0] > 0) ? frob / svd.sigma[0] * 100.0 : 0.0);

        free(lowrank);
    }

    /* Sparse low-rank */
    printf("\nSparse low-rank (k=2, drop_tol=0.1):\n");
    SparseMatrix *sp_lr = NULL;
    err = sparse_svd_lowrank_sparse(A, 2, 0.1, &sp_lr);
    if (err == SPARSE_OK && sp_lr) {
        printf("  Original nnz: %d\n", (int)sparse_nnz(A));
        printf("  Low-rank nnz: %d\n", (int)sparse_nnz(sp_lr));
        printf("  Compression:  %.1fx\n",
               sparse_nnz(sp_lr) > 0 ? (double)sparse_nnz(A) / (double)sparse_nnz(sp_lr) : 0.0);
        sparse_free(sp_lr);
    }

    sparse_svd_free(&svd);
    sparse_free(A);
    return 0;
}
