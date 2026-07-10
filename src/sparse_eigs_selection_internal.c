#include "sparse_eigs_internal.h"

#include <math.h>
#include <stddef.h>

/* Select `min(k_want, m)` indices into `theta[0..m)` by `which`:
 *
 *   LARGEST       - descending theta (sel_idx[0] = m - 1 etc.)
 *   SMALLEST      - ascending theta (sel_idx[0] = 0 etc.)
 *   NEAREST_SIGMA - descending |theta| via a two-pointer sweep over
 *                   the ascending list (largest-|theta| lives at one
 *                   of the two ends; under shift-invert this means
 *                   the Ritz value closest to sigma in the original
 *                   lambda-space).
 *
 * Assumes theta is sorted ascending (as `tridiag_qr_eigenpairs`
 * returns it).  Returns the number of indices written. */
idx_t s20_select_indices(const double *theta, idx_t m, sparse_eigs_which_t which, idx_t k_want,
                         idx_t *sel_idx) {
    idx_t take = k_want < m ? k_want : m;
    if (take < 1)
        return 0;
    if (which == SPARSE_EIGS_LARGEST) {
        for (idx_t j = 0; j < take; j++)
            sel_idx[j] = m - 1 - j;
    } else if (which == SPARSE_EIGS_SMALLEST) {
        for (idx_t j = 0; j < take; j++)
            sel_idx[j] = j;
    } else {
        /* NEAREST_SIGMA: largest-|theta| first.  Two-pointer scan;
         * left runs up from 0, right runs down from m-1.  The loop
         * body bounds-checks both pointers so a partial overlap at
         * the centre of the array can't under/overflow. */
        idx_t left = 0;
        idx_t right = m - 1;
        for (idx_t j = 0; j < take; j++) {
            if (left > right)
                break;
            if (fabs(theta[left]) > fabs(theta[right])) {
                sel_idx[j] = left;
                left++;
            } else {
                sel_idx[j] = right;
                if (right == 0)
                    break;
                right--;
            }
        }
    }
    return take;
}

/* Ritz vector lift: for each j in [0, take), write column j of
 * `eigenvectors_out` (n x take, column-major) with
 *   eigenvector_j = V * Y[:, idx_j]
 * where V is the Lanczos basis (n x m, column-major) and idx_j is
 * the m-space column index of the j-th selected Ritz pair.  Assumes
 * V's columns are already orthonormal (assuming full
 * reorthogonalization) so the lifted vectors inherit unit norm up
 * to the MGS drift bound.  Ritz vectors of (A - sigma I)^-1 are also
 * eigenvectors of A (same eigenspaces), so the same lift works for
 * shift-invert mode. */
void s20_lift_ritz_vectors(const double *V, const double *Y, idx_t n, idx_t m, idx_t take,
                           const idx_t *idx, double *eigenvectors_out) {
    for (idx_t j = 0; j < take; j++) {
        const double *y = Y + (size_t)idx[j] * (size_t)m;
        double *out = eigenvectors_out + (size_t)j * (size_t)n;
        for (idx_t i = 0; i < n; i++)
            out[i] = 0.0;
        for (idx_t c = 0; c < m; c++) {
            double yc = y[c];
            if (yc == 0.0)
                continue;
            const double *v_c = V + (size_t)c * (size_t)n;
            for (idx_t i = 0; i < n; i++)
                out[i] += yc * v_c[i];
        }
    }
}
