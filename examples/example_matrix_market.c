/*
 * example_matrix_market.c - Load a Matrix Market file and use it in a solve.
 *
 * Demonstrates:
 *   - Loading a public SparseMatrix shell from a Matrix Market file
 *   - Handling parse/I/O errors through sparse_err_t and sparse_errno()
 *   - Using the loaded matrix with the normal one-shot LU workflow
 *   - Freeing the loaded matrix with sparse_free()
 *
 * Run from the project root so tests/data/tridiagonal_20.mtx resolves:
 *   ./build/example_matrix_market
 */
#include "example_alloc_helpers.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const char *path = "tests/data/tridiagonal_20.mtx";
    printf("=== Matrix Market Load/Use Example ===\n\n");

    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, path);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Could not load %s: %s", path, sparse_strerror(err));
        if (err == SPARSE_ERR_IO)
            fprintf(stderr, " (errno=%d)", sparse_errno());
        fputc('\n', stderr);
        return 1;
    }

    idx_t n = sparse_rows(A);
    if (n != sparse_cols(A)) {
        fprintf(stderr, "Loaded matrix is not square: %" SPARSE_PRIDX " x %" SPARSE_PRIDX "\n", n,
                sparse_cols(A));
        sparse_free(A);
        return 1;
    }

    printf("Loaded %s: %" SPARSE_PRIDX " x %" SPARSE_PRIDX ", nnz = %" SPARSE_PRIDX "\n", path, n,
           sparse_cols(A), sparse_nnz(A));

    double *x_exact = NULL;
    double *b = NULL;
    double *x = NULL;
    double *Ax = NULL;
    if (example_malloc_array(n, sizeof(double), (void **)&x_exact) != SPARSE_OK ||
        example_calloc_array(n, sizeof(double), (void **)&b) != SPARSE_OK ||
        example_calloc_array(n, sizeof(double), (void **)&x) != SPARSE_OK ||
        example_calloc_array(n, sizeof(double), (void **)&Ax) != SPARSE_OK) {
        fprintf(stderr, "Allocation failed\n");
        free(Ax);
        free(x);
        free(b);
        free(x_exact);
        sparse_free(A);
        return 1;
    }

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A, x_exact, b);

    SparseMatrix *LU = sparse_copy(A);
    if (!LU) {
        fprintf(stderr, "Failed to copy loaded matrix\n");
        free(Ax);
        free(x);
        free(b);
        free(x_exact);
        sparse_free(A);
        return 1;
    }

    err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-14);
    if (err != SPARSE_OK) {
        fprintf(stderr, "LU factorization failed: %s\n", sparse_strerror(err));
        sparse_free(LU);
        free(Ax);
        free(x);
        free(b);
        free(x_exact);
        sparse_free(A);
        return 1;
    }

    err = sparse_lu_solve(LU, b, x);
    sparse_free(LU);
    if (err != SPARSE_OK) {
        fprintf(stderr, "LU solve failed: %s\n", sparse_strerror(err));
        free(Ax);
        free(x);
        free(b);
        free(x_exact);
        sparse_free(A);
        return 1;
    }

    sparse_matvec(A, x, Ax);
    double residual_sq = 0.0;
    double error_sq = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ri = b[i] - Ax[i];
        double ei = x[i] - x_exact[i];
        residual_sq += ri * ri;
        error_sq += ei * ei;
    }

    printf("Solved A*x = b where b = A*1\n");
    printf("Residual ||b - A*x||_2 = %.2e\n", sqrt(residual_sq));
    printf("Error    ||x - 1||_2   = %.2e\n", sqrt(error_sq));

    free(Ax);
    free(x);
    free(b);
    free(x_exact);
    sparse_free(A);
    return 0;
}
