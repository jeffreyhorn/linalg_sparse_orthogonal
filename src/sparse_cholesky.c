#include "sparse_cholesky.h"
#include "sparse_reorder.h"
#include "sparse_matrix_internal.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── Cholesky factorization ─────────────────────────────────────────── */

sparse_err_t sparse_cholesky_factor(SparseMatrix *mat)
{
    if (!mat) return SPARSE_ERR_NULL;
    if (mat->rows != mat->cols) return SPARSE_ERR_SHAPE;
    /* TODO: implement in Day 2 */
    (void)mat;
    return SPARSE_ERR_BADARG;
}

/* ─── Cholesky factorization with options ────────────────────────────── */

sparse_err_t sparse_cholesky_factor_opts(SparseMatrix *mat,
                                         const sparse_cholesky_opts_t *opts)
{
    if (!mat || !opts) return SPARSE_ERR_NULL;
    if (mat->rows != mat->cols) return SPARSE_ERR_SHAPE;
    /* TODO: implement reordering in Day 3 */
    return sparse_cholesky_factor(mat);
}

/* ─── Cholesky solve ─────────────────────────────────────────────────── */

sparse_err_t sparse_cholesky_solve(const SparseMatrix *mat,
                                   const double *b, double *x)
{
    if (!mat || !b || !x) return SPARSE_ERR_NULL;
    /* TODO: implement in Day 3 */
    (void)mat; (void)b; (void)x;
    return SPARSE_ERR_BADARG;
}
