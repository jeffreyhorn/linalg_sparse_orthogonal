/* Sprint 29 Day 8 (Item 5): feature-test macro for clock_gettime. */
#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 199309L)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _POSIX_C_SOURCE 199309L
#endif

#include "sparse_qr_internal.h"

#include "sparse_matrix_internal.h"

#include <math.h>
#include <string.h>
#include <time.h>

/* Sprint 29 Day 7 (Item 4): progress / cancel wiring. */
double sparse_qr_now_s(void) {
    struct timespec ts;
#ifdef _WIN32
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

double sparse_qr_householder_compute(const double *x, double *v, idx_t len) {
    if (len <= 0)
        return 0.0;

    memcpy(v, x, (size_t)len * sizeof(double));

    double sigma = 0.0;
    for (idx_t i = 1; i < len; i++)
        sigma += v[i] * v[i];

    if (sigma == 0.0 && v[0] >= 0.0)
        return 0.0;

    double xnorm = sqrt(v[0] * v[0] + sigma);
    if (v[0] >= 0.0)
        v[0] += xnorm;
    else
        v[0] -= xnorm;

    double vtv = v[0] * v[0] + sigma;
    if (vtv == 0.0)
        return 0.0;

    return 2.0 / vtv;
}

void sparse_qr_householder_apply(const double *v, double beta, double *y, idx_t len) {
    if (beta == 0.0)
        return;

    double vty = 0.0;
    for (idx_t i = 0; i < len; i++)
        vty += v[i] * y[i];

    double scale = beta * vty;
    for (idx_t i = 0; i < len; i++)
        y[i] -= scale * v[i];
}

void sparse_qr_extract_column(const SparseMatrix *A, idx_t col, double *dense) {
    Node *nd = A->col_headers[col];
    while (nd) {
        dense[nd->row] = nd->value;
        nd = nd->down;
    }
}

void sparse_qr_householder_apply_to_column(const double *v, double beta, double *dense, idx_t start,
                                           idx_t m) {
    if (beta == 0.0)
        return;
    idx_t len = m - start;
    sparse_qr_householder_apply(v, beta, &dense[start], len);
}
