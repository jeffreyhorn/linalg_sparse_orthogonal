#ifndef TEST_INTEGRATION_FIXTURES_H
#define TEST_INTEGRATION_FIXTURES_H

#include "sparse_csr.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"

typedef struct {
    idx_t n_calls;
    idx_t cancel_after_step; /* return non-zero when step == cancel_after_step; -1 = never */
    idx_t last_step;
    idx_t last_total;
    const char *last_phase;
    double last_elapsed_s;
} integration_progress_counter_t;

static int integration_progress_count_cb(const sparse_progress_t *p, void *user) {
    integration_progress_counter_t *ctx = (integration_progress_counter_t *)user;
    ctx->n_calls++;
    ctx->last_step = p->step;
    ctx->last_total = p->total;
    ctx->last_phase = p->phase;
    ctx->last_elapsed_s = p->elapsed_s;
    if (ctx->cancel_after_step >= 0 && p->step >= ctx->cancel_after_step)
        return 1;
    return 0;
}

static int integration_insert_or_free(SparseMatrix **A, idx_t row, idx_t col, double value) {
    if (sparse_insert(*A, row, col, value) != SPARSE_OK) {
        sparse_free(*A);
        *A = NULL;
        return 0;
    }
    return 1;
}

/* Shared SPD fixture for integration paths that need LU, Cholesky, LDLT,
 * direct lifecycle, QR, eigensolver, or iterative solver progress proof.
 */
static SparseMatrix *integration_build_tridiag_spd(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        if (!integration_insert_or_free(&A, i, i, 4.0))
            return NULL;
        if (i > 0) {
            if (!integration_insert_or_free(&A, i, i - 1, -1.0))
                return NULL;
            if (!integration_insert_or_free(&A, i - 1, i, -1.0))
                return NULL;
        }
    }
    return A;
}

static SparseMatrix *integration_build_unsym_4x4(void) {
    SparseMatrix *A = sparse_create(4, 4);
    if (!A)
        return NULL;

    if (!integration_insert_or_free(&A, 0, 0, 6.0))
        return NULL;
    if (!integration_insert_or_free(&A, 0, 1, -1.0))
        return NULL;
    if (!integration_insert_or_free(&A, 0, 3, 0.5))
        return NULL;
    if (!integration_insert_or_free(&A, 1, 0, 2.0))
        return NULL;
    if (!integration_insert_or_free(&A, 1, 1, 7.0))
        return NULL;
    if (!integration_insert_or_free(&A, 1, 2, -1.0))
        return NULL;
    if (!integration_insert_or_free(&A, 2, 1, 1.5))
        return NULL;
    if (!integration_insert_or_free(&A, 2, 2, 8.0))
        return NULL;
    if (!integration_insert_or_free(&A, 2, 3, -2.0))
        return NULL;
    if (!integration_insert_or_free(&A, 3, 0, -0.5))
        return NULL;
    if (!integration_insert_or_free(&A, 3, 2, 1.0))
        return NULL;
    if (!integration_insert_or_free(&A, 3, 3, 5.0))
        return NULL;
    return A;
}

/* KKT-style saddle-point indefinite matrix:
 *   [ H    B^T ]
 *   [ B    0   ]
 * H is `n_top` x `n_top` tridiagonal SPD (diag 6, off-diag -1).
 * B = [I_k | 0] where k = n_bot. This keeps the system symmetric,
 * indefinite, and nonsingular for n_top >= n_bot.
 */
static SparseMatrix *integration_build_kkt(idx_t n_top, idx_t n_bot) {
    idx_t n = n_top + n_bot;
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n_top; i++) {
        if (!integration_insert_or_free(&A, i, i, 6.0))
            return NULL;
        if (i > 0) {
            if (!integration_insert_or_free(&A, i, i - 1, -1.0))
                return NULL;
            if (!integration_insert_or_free(&A, i - 1, i, -1.0))
                return NULL;
        }
    }
    for (idx_t j = 0; j < n_bot; j++) {
        if (!integration_insert_or_free(&A, n_top + j, j, 1.0))
            return NULL;
        if (!integration_insert_or_free(&A, j, n_top + j, 1.0))
            return NULL;
    }
    return A;
}

static void integration_perturb_kkt_values_in_place(SparseMatrix *A, idx_t n_top, idx_t n_bot,
                                                    double scale) {
    for (idx_t i = 0; i < n_top; i++) {
        double diag = 6.0 + scale * (double)((i % 7) - 3);
        ASSERT_EQ(sparse_set(A, i, i, diag), SPARSE_OK);
        if (i > 0) {
            double offdiag = -1.0 - 0.1 * scale * (double)(i % 3);
            ASSERT_EQ(sparse_set(A, i, i - 1, offdiag), SPARSE_OK);
            ASSERT_EQ(sparse_set(A, i - 1, i, offdiag), SPARSE_OK);
        }
    }

    for (idx_t j = 0; j < n_bot; j++) {
        double coupling = 1.0 + 0.05 * scale * (double)((j % 5) - 2);
        ASSERT_EQ(sparse_set(A, n_top + j, j, coupling), SPARSE_OK);
        ASSERT_EQ(sparse_set(A, j, n_top + j, coupling), SPARSE_OK);
    }
}

static SparseMatrix *integration_build_from_csr_constructor(const SparseMatrix *src) {
    SparseCsr *csr = NULL;
    SparseMatrix *out = NULL;

    if (!src)
        return NULL;
    if (sparse_to_csr(src, &csr) != SPARSE_OK)
        return NULL;

    out = sparse_create_from_csr(csr);
    sparse_csr_free(csr);
    return out;
}

static SparseMatrix *integration_build_from_csc_constructor(const SparseMatrix *src) {
    SparseCsc *csc = NULL;
    SparseMatrix *out = NULL;

    if (!src)
        return NULL;
    if (sparse_to_csc(src, &csc) != SPARSE_OK)
        return NULL;

    out = sparse_create_from_csc(csc);
    sparse_csc_free(csc);
    return out;
}

#endif
