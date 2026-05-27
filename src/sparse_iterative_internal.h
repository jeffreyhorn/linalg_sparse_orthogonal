#ifndef SPARSE_ITERATIVE_INTERNAL_H
#define SPARSE_ITERATIVE_INTERNAL_H

/*
 * Private header: iterative-solver internal entry points for reusable
 * workspace benchmarking and internal composition. Not part of the public API.
 */

#include "sparse_iterative.h"
#include "sparse_iterative_workspace_internal.h"

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

#endif /* SPARSE_ITERATIVE_INTERNAL_H */
