/*
 * example_colamd.c — COLAMD column ordering for fill reduction.
 *
 * Demonstrates:
 *   - Computing COLAMD ordering on an unsymmetric matrix
 *   - Comparing LU fill-in: natural vs COLAMD ordering
 *   - Using COLAMD with QR factorization
 *
 * Build:
 *   cc -O2 -Iinclude -o example_colamd examples/example_colamd.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_reorder.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== COLAMD Column Ordering Example ===\n\n");

    /* Build a 10x10 unsymmetric "arrow + band" matrix */
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    if (!A) {
        fprintf(stderr, "Failed to create matrix\n");
        return 1;
    }

    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -0.5);
        /* Arrow pattern: dense last column */
        sparse_insert(A, i, n - 1, 0.1);
        if (i < n - 1)
            sparse_insert(A, n - 1, i, 0.1);
    }

    printf("Matrix: %dx%d unsymmetric (arrow + band), nnz = %d\n\n", (int)n, (int)n,
           (int)sparse_nnz(A));

    /* Compute COLAMD ordering */
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    if (!perm) {
        sparse_free(A);
        return 1;
    }

    sparse_err_t err = sparse_reorder_colamd(A, perm);
    if (err != SPARSE_OK) {
        fprintf(stderr, "COLAMD failed: %s\n", sparse_strerror(err));
        free(perm);
        sparse_free(A);
        return 1;
    }

    printf("COLAMD permutation: [");
    for (idx_t i = 0; i < n; i++)
        printf("%s%d", i > 0 ? ", " : "", (int)perm[i]);
    printf("]\n\n");

    /* Compare LU fill-in (COLAMD = column-only permutation) */
    SparseMatrix *LU_nat = sparse_copy(A);
    err = sparse_lu_factor(LU_nat, SPARSE_PIVOT_PARTIAL, 1e-12);
    idx_t fill_nat = (err == SPARSE_OK) ? sparse_nnz(LU_nat) : -1;

    /* Apply COLAMD as column-only permutation (identity row perm) */
    idx_t *id_perm = malloc((size_t)n * sizeof(idx_t));
    if (id_perm) {
        for (idx_t i = 0; i < n; i++)
            id_perm[i] = i;
    }
    SparseMatrix *PA = NULL;
    if (id_perm)
        sparse_permute(A, id_perm, perm, &PA);
    free(id_perm);
    SparseMatrix *LU_col = PA ? sparse_copy(PA) : NULL;
    sparse_err_t err2 =
        LU_col ? sparse_lu_factor(LU_col, SPARSE_PIVOT_PARTIAL, 1e-12) : SPARSE_ERR_ALLOC;
    idx_t fill_col = (err2 == SPARSE_OK) ? sparse_nnz(LU_col) : -1;

    printf("LU fill-in comparison:\n");
    printf("  Natural ordering: nnz(LU) = %d\n", (int)fill_nat);
    printf("  COLAMD ordering:  nnz(LU) = %d", (int)fill_col);
    if (fill_nat > 0 && fill_col > 0) {
        double reduction = 100.0 * (1.0 - (double)fill_col / (double)fill_nat);
        printf(" (%.0f%% %s)", reduction > 0 ? reduction : -reduction,
               reduction > 0 ? "reduction" : "increase");
    }
    printf("\n\n");

    /* QR with COLAMD */
    sparse_qr_t qr;
    sparse_qr_opts_t qr_opts = {SPARSE_REORDER_COLAMD, 0, 0};
    err = sparse_qr_factor_opts(A, &qr_opts, &qr);
    if (err == SPARSE_OK) {
        double b[10], x[10];
        for (idx_t i = 0; i < n; i++)
            b[i] = 1.0;
        double resid;
        sparse_qr_solve(&qr, b, x, &resid);
        printf("QR+COLAMD solve: residual = %.2e\n", resid);

        /* Rank diagnostics */
        sparse_qr_rank_info_t info;
        sparse_qr_rank_info(&qr, 0, &info);
        printf("Rank info: rank=%d/%d, condest=%.2f\n", (int)info.rank, (int)info.k, info.condest);

        sparse_qr_free(&qr);
    }

    free(perm);
    sparse_free(LU_col);
    sparse_free(PA);
    sparse_free(LU_nat);
    sparse_free(A);

    printf("\nDone.\n");
    return 0;
}
