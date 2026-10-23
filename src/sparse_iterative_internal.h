#ifndef SPARSE_ITERATIVE_INTERNAL_H
#define SPARSE_ITERATIVE_INTERNAL_H

/*
 * Private header: iterative-solver shared internal entry points and helper
 * declarations for reusable workspace benchmarking, internal composition, and
 * split implementation ownership. Not part of the public API.
 */

#include "sparse_iterative.h"
#include "sparse_iterative_workspace_internal.h"
#include <math.h>
#include <stdio.h>

typedef struct {
    double *buf;
    idx_t capacity;
    idx_t count;
    idx_t head;
    double tol;
} stag_tracker_t;

typedef struct {
    double *buf;
    idx_t len;
    idx_t count;
} reshist_t;

double s29_iter_now_s(void);
sparse_err_t s49_iter_handle_ensure(sparse_iter_handle_t *handle,
                                    sparse_iter_workspace_t **workspace_out);

/*
 * Shared iterative result/default helpers used by split solver owners.
 * Keep these private to source files under src/.
 */
const sparse_iter_opts_t *s85_iter_cg_defaults(void);
void s85_iter_result_reset(sparse_iter_result_t *result);
void s85_iter_result_mark_converged(sparse_iter_result_t *result);

static inline int s217_iter_vector_finite(const double *values, idx_t n) {
    for (idx_t i = 0; i < n; i++) {
        if (!isfinite(values[i]))
            return 0;
    }
    return 1;
}

static inline double s217_iter_relative_residual(double rnorm, double bnorm) {
    if (!isfinite(rnorm) || !isfinite(bnorm) || rnorm < 0.0 || bnorm <= 0.0)
        return NAN;
    double rel = rnorm / bnorm;
    if (!isfinite(rel) || (rel == 0.0 && rnorm != 0.0))
        return NAN;
    return rel;
}

static inline void s217_iter_numeric_result(sparse_iter_result_t *result, idx_t iterations,
                                            idx_t history_count) {
    if (result) {
        result->iterations = iterations;
        result->residual_norm = NAN;
        result->converged = 0;
        result->residual_history_count = history_count;
    }
}
sparse_err_t sparse_iter_stag_init(stag_tracker_t *st, idx_t window);
void sparse_iter_stag_free(stag_tracker_t *st);
void sparse_iter_stag_record(stag_tracker_t *st, double residual);
int sparse_iter_stag_check(const stag_tracker_t *st);

static inline reshist_t reshist_make(double *buf, idx_t len) {
    return (reshist_t){.buf = buf, .len = (buf && len > 0) ? len : 0, .count = 0};
}

static inline void reshist_record(reshist_t *rh, double relres) {
    if (rh->count < rh->len)
        rh->buf[rh->count] = relres;
    rh->count++;
}

static inline void iter_report(sparse_iter_callback_fn cb, void *cb_ctx, int verbose,
                               const char *solver, idx_t iteration, double residual_norm) {
    if (cb) {
        sparse_iter_progress_t p = {
            .iteration = iteration, .residual_norm = residual_norm, .solver = solver};
        cb(&p, cb_ctx);
    } else if (verbose) {
        fprintf(stderr, "  %s iter %4d: ||r||/||b|| = %.6e\n", solver, (int)iteration,
                residual_norm);
    }
}

sparse_err_t sparse_solve_cg_with_workspace_internal(const SparseMatrix *A, const double *b,
                                                     double *x, const sparse_iter_opts_t *opts,
                                                     sparse_precond_fn precond,
                                                     const void *precond_ctx,
                                                     sparse_iter_result_t *result,
                                                     sparse_iter_workspace_t *workspace);

sparse_err_t sparse_solve_gmres_with_workspace_internal(const SparseMatrix *A, const double *b,
                                                        double *x, const sparse_gmres_opts_t *opts,
                                                        sparse_precond_fn precond,
                                                        const void *precond_ctx,
                                                        sparse_iter_result_t *result,
                                                        sparse_iter_workspace_t *workspace);

sparse_err_t sparse_solve_minres_with_workspace_internal(const SparseMatrix *A, const double *b,
                                                         double *x, const sparse_iter_opts_t *opts,
                                                         sparse_precond_fn precond,
                                                         const void *precond_ctx,
                                                         sparse_iter_result_t *result,
                                                         sparse_iter_workspace_t *workspace);

#endif /* SPARSE_ITERATIVE_INTERNAL_H */
