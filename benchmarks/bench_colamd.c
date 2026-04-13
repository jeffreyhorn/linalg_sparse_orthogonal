/*
 * bench_colamd.c — Compare fill-in: COLAMD vs AMD vs natural ordering.
 *
 * Reports nnz(R) for QR factorization with each ordering strategy
 * on built-in test matrices and SuiteSparse matrices.
 *
 * Build:
 *   cc -O2 -Iinclude benchmarks/bench_colamd.c -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_reorder.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

static void bench_qr_fill(const char *name, const SparseMatrix *A) {
    if (!A) {
        printf("  %-20s  [SKIP] matrix not available\n", name);
        return;
    }

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);

    sparse_qr_t qr_none, qr_amd, qr_colamd;
    sparse_qr_opts_t opts_none = {SPARSE_REORDER_NONE, 0, 0};
    sparse_qr_opts_t opts_amd = {SPARSE_REORDER_AMD, 0, 0};
    sparse_qr_opts_t opts_colamd = {SPARSE_REORDER_COLAMD, 0, 0};

    idx_t nnz_none = -1, nnz_amd = -1, nnz_colamd = -1;

    if (sparse_qr_factor_opts(A, &opts_none, &qr_none) == SPARSE_OK) {
        nnz_none = sparse_nnz(qr_none.R);
        sparse_qr_free(&qr_none);
    }
    if (sparse_qr_factor_opts(A, &opts_amd, &qr_amd) == SPARSE_OK) {
        nnz_amd = sparse_nnz(qr_amd.R);
        sparse_qr_free(&qr_amd);
    }
    if (sparse_qr_factor_opts(A, &opts_colamd, &qr_colamd) == SPARSE_OK) {
        nnz_colamd = sparse_nnz(qr_colamd.R);
        sparse_qr_free(&qr_colamd);
    }

    printf("  %-20s  %dx%d  none=%-6d amd=%-6d colamd=%-6d", name, (int)m, (int)n, (int)nnz_none,
           (int)nnz_amd, (int)nnz_colamd);
    if (nnz_colamd >= 0 && nnz_none > 0) {
        printf("  (%.0f%% vs natural)", 100.0 * (1.0 - (double)nnz_colamd / (double)nnz_none));
    }
    printf("\n");
}

int main(void) {
    printf("=== QR Fill-In Comparison: COLAMD vs AMD vs Natural ===\n\n");
    printf("  %-20s  %-8s  %-10s %-10s %-10s\n", "Matrix", "Size", "nnz(R)nat", "nnz(R)amd",
           "nnz(R)col");
    printf("  %-20s  %-8s  %-10s %-10s %-10s\n", "------", "----", "---------", "---------",
           "---------");

    /* Built-in test matrices */
    {
        idx_t n = 20;
        SparseMatrix *A = sparse_create(n, n);
        if (A) {
            for (idx_t i = 0; i < n; i++) {
                sparse_insert(A, i, i, 4.0);
                if (i > 0) {
                    sparse_insert(A, i, i - 1, -1.0);
                    sparse_insert(A, i - 1, i, -1.0);
                }
            }
            bench_qr_fill("tridiag-20", A);
            sparse_free(A);
        }
    }

    {
        idx_t n = 15;
        SparseMatrix *A = sparse_create(n, n);
        if (A) {
            for (idx_t i = 0; i < n; i++) {
                sparse_insert(A, i, i, 4.0);
                if (i < n - 1) {
                    sparse_insert(A, i, n - 1, -1.0);
                    sparse_insert(A, n - 1, i, -1.0);
                }
            }
            bench_qr_fill("arrow-15", A);
            sparse_free(A);
        }
    }

    /* SuiteSparse matrices */
    {
        SparseMatrix *A = NULL;
        if (sparse_load_mm(&A, SS_DIR "/west0067.mtx") == SPARSE_OK) {
            bench_qr_fill("west0067", A);
            sparse_free(A);
        } else {
            printf("  %-20s  [SKIP] not found\n", "west0067");
        }
    }

    printf("\nDone.\n");
    return 0;
}
