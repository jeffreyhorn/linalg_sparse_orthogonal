#ifndef SPARSE_BICGSTAB_INTERNAL_H
#define SPARSE_BICGSTAB_INTERNAL_H

/**
 * @file sparse_bicgstab_internal.h
 * @brief Internal data structures for BiCGSTAB solver.
 *
 * BiCGSTAB (Bi-Conjugate Gradient Stabilized, Van der Vorst 1992) solves
 * general nonsymmetric linear systems A*x = b. It combines the BiCG
 * two-sided Lanczos approach with a polynomial stabilization step,
 * yielding smoother convergence than CGS without requiring A^T.
 *
 * Each iteration performs two SpMVs and (optionally) two preconditioner
 * applications. The recurrence maintains six work vectors of length n.
 *
 * Not part of the public API.
 */

#include "sparse_matrix_internal.h"
#include <stdint.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * BiCGSTAB workspace
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Number of work vectors for unpreconditioned BiCGSTAB: r, r_hat, p, v, s, t.
 * Preconditioned BiCGSTAB adds two more: p_hat, s_hat.
 */
#define BICGSTAB_NVEC_NOPRECOND 6
#define BICGSTAB_NVEC_PRECOND 8

/**
 * BiCGSTAB workspace — all vectors stored in a single contiguous allocation.
 *
 * For unpreconditioned BiCGSTAB (6 vectors):
 *   r      — current residual
 *   r_hat  — shadow residual (fixed: r_hat = r_0)
 *   p      — search direction
 *   v      — A * p_hat (or A * p when unpreconditioned)
 *   s      — intermediate residual: s = r - alpha * v
 *   t      — A * s_hat (or A * s when unpreconditioned)
 *
 * For preconditioned BiCGSTAB (8 vectors), add:
 *   p_hat  — preconditioned p: M^{-1} * p
 *   s_hat  — preconditioned s: M^{-1} * s
 */
typedef struct {
    double *mem;   /**< Single allocation for all work vectors */
    double *r;     /**< Current residual (length n) */
    double *r_hat; /**< Shadow residual r_hat_0 (length n) */
    double *p;     /**< Search direction (length n) */
    double *v;     /**< v = A * p_hat (length n) */
    double *s;     /**< s = r - alpha * v (length n) */
    double *t;     /**< t = A * s_hat (length n) */
    double *p_hat; /**< Preconditioned p (length n, NULL if no precond) */
    double *s_hat; /**< Preconditioned s (length n, NULL if no precond) */
    idx_t n;       /**< System dimension */
} bicgstab_workspace_t;

/**
 * Allocate BiCGSTAB workspace for a system of dimension n.
 *
 * @param n         System dimension.
 * @param precond   Non-zero if a preconditioner will be used (allocates 8 vectors
 *                  instead of 6).
 * @param ws        Output workspace struct (zeroed on failure).
 * @return SPARSE_OK on success, SPARSE_ERR_ALLOC on failure.
 */
static inline sparse_err_t bicgstab_workspace_alloc(idx_t n, int precond,
                                                    bicgstab_workspace_t *ws) {
    idx_t nvec = precond ? BICGSTAB_NVEC_PRECOND : BICGSTAB_NVEC_NOPRECOND;

    if ((size_t)n > SIZE_MAX / ((size_t)nvec * sizeof(double))) {
        *ws = (bicgstab_workspace_t){0};
        return SPARSE_ERR_ALLOC;
    }

    double *mem = calloc((size_t)nvec * (size_t)n, sizeof(double));
    if (!mem) {
        *ws = (bicgstab_workspace_t){0};
        return SPARSE_ERR_ALLOC;
    }

    ws->mem = mem;
    ws->n = n;
    ws->r = mem;
    ws->r_hat = mem + (size_t)n;
    ws->p = mem + 2 * (size_t)n;
    ws->v = mem + 3 * (size_t)n;
    ws->s = mem + 4 * (size_t)n;
    ws->t = mem + 5 * (size_t)n;

    if (precond) {
        ws->p_hat = mem + 6 * (size_t)n;
        ws->s_hat = mem + 7 * (size_t)n;
    } else {
        ws->p_hat = NULL;
        ws->s_hat = NULL;
    }

    return SPARSE_OK;
}

/**
 * Free BiCGSTAB workspace. Safe to call on a zeroed struct.
 */
static inline void bicgstab_workspace_free(bicgstab_workspace_t *ws) {
    if (ws) {
        free(ws->mem);
        *ws = (bicgstab_workspace_t){0};
    }
}

#endif /* SPARSE_BICGSTAB_INTERNAL_H */
