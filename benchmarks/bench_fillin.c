/*
 * bench_fillin.c — Measure fill-in during LU factorization
 *
 * Generates matrices with known sparsity patterns and reports
 * nnz before/after factorization to characterize fill-in behavior.
 */
#include "sparse_matrix.h"
#include "sparse_lu.h"
#include <stdio.h>
#include <stdlib.h>

/* ─── Matrix generators ──────────────────────────────────────────────── */

/* Tridiagonal: minimal fill-in expected */
static SparseMatrix *make_tridiag(idx_t n)
{
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0);
        if (i > 0)     sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1) sparse_insert(A, i, i + 1, -1.0);
    }
    return A;
}

/* Pentadiagonal (bandwidth 2): moderate fill-in */
static SparseMatrix *make_pentadiag(idx_t n)
{
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 6.0);
        if (i > 0)     sparse_insert(A, i, i - 1, -1.0);
        if (i > 1)     sparse_insert(A, i, i - 2, -0.5);
        if (i < n - 1) sparse_insert(A, i, i + 1, -1.0);
        if (i < n - 2) sparse_insert(A, i, i + 2, -0.5);
    }
    return A;
}

/* Arrow matrix: dense first row/col, diagonal elsewhere.
 * Known to cause significant fill-in. */
static SparseMatrix *make_arrow(idx_t n)
{
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));  /* diagonal */
        if (i > 0) {
            sparse_insert(A, 0, i, 1.0);  /* dense first row */
            sparse_insert(A, i, 0, 1.0);  /* dense first col */
        }
    }
    return A;
}

/* Random sparse with ~k entries per row */
static SparseMatrix *make_random_sparse(idx_t n, int k, unsigned seed)
{
    srand(seed);
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    for (idx_t i = 0; i < n; i++) {
        double diag_sum = 0.0;
        for (int e = 0; e < k; e++) {
            idx_t j = (idx_t)(rand() % n);
            if (j == i) continue;
            double v = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
            sparse_insert(A, i, j, v);
            diag_sum += (v > 0 ? v : -v);
        }
        sparse_insert(A, i, i, diag_sum + 1.0);
    }
    return A;
}

/* Dense: maximum fill-in (already full) */
static SparseMatrix *make_dense(idx_t n)
{
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    srand(123);
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            double v = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
            if (i == j) v += (double)n;
            sparse_insert(A, i, j, v);
        }
    }
    return A;
}

/* ─── Run fill-in analysis ──────────────────────────────────────────── */

static void analyze(const char *label, SparseMatrix *A, sparse_pivot_t pivot)
{
    if (!A) { printf("%-25s  [allocation failed]\n", label); return; }

    idx_t n = sparse_rows(A);
    idx_t nnz_before = sparse_nnz(A);
    idx_t max_nnz = n * n;
    double density_before = (double)nnz_before / (double)max_nnz * 100.0;

    SparseMatrix *LU = sparse_copy(A);
    sparse_err_t err = sparse_lu_factor(LU, pivot, 1e-12);

    if (err != SPARSE_OK) {
        printf("%-25s  n=%-5d  nnz=%-8d  FACTOR FAILED: %s\n",
               label, (int)n, (int)nnz_before, sparse_strerror(err));
        sparse_free(LU);
        return;
    }

    idx_t nnz_after = sparse_nnz(LU);
    double density_after = (double)nnz_after / (double)max_nnz * 100.0;
    double ratio = (double)nnz_after / (double)nnz_before;

    printf("%-25s  n=%-5d  nnz_before=%-8d (%.1f%%)  "
           "nnz_after=%-8d (%.1f%%)  ratio=%.2f\n",
           label, (int)n, (int)nnz_before, density_before,
           (int)nnz_after, density_after, ratio);

    sparse_free(LU);
}

int main(void)
{
    printf("Fill-in Analysis\n");
    printf("================\n\n");

    printf("--- Complete Pivoting ---\n");
    for (idx_t n = 50; n <= 500; n *= 2) {
        analyze("Tridiagonal", make_tridiag(n), SPARSE_PIVOT_COMPLETE);
    }
    printf("\n");

    for (idx_t n = 50; n <= 500; n *= 2) {
        analyze("Pentadiagonal", make_pentadiag(n), SPARSE_PIVOT_COMPLETE);
    }
    printf("\n");

    for (idx_t n = 50; n <= 500; n *= 2) {
        analyze("Arrow", make_arrow(n), SPARSE_PIVOT_COMPLETE);
    }
    printf("\n");

    for (idx_t n = 50; n <= 200; n *= 2) {
        analyze("Random sparse (k=5)", make_random_sparse(n, 5, 42),
                SPARSE_PIVOT_COMPLETE);
    }
    printf("\n");

    for (idx_t n = 20; n <= 80; n *= 2) {
        analyze("Dense", make_dense(n), SPARSE_PIVOT_COMPLETE);
    }
    printf("\n");

    printf("--- Partial Pivoting ---\n");
    for (idx_t n = 50; n <= 500; n *= 2) {
        analyze("Tridiagonal", make_tridiag(n), SPARSE_PIVOT_PARTIAL);
    }
    printf("\n");

    for (idx_t n = 50; n <= 500; n *= 2) {
        analyze("Arrow", make_arrow(n), SPARSE_PIVOT_PARTIAL);
    }
    printf("\n");

    return 0;
}
