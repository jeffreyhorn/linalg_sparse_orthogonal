#include "sparse_eigs_internal.h"

#include <math.h>

/* ─── Dense Symmetric Eigensolver (Jacobi) ───────────────────────── */

/* Classical Jacobi sweeps on a dense symmetric K × K matrix.
 * Returns ascending eigenvalues in `theta_out[0..K-1]` and the
 * corresponding orthonormal eigenvectors as columns of `Q_out`
 * (K × K, column-major).  Used for arrowhead Ritz extraction
 * because the arrowhead does not have the tridiagonal shape
 * `tridiag_qr_eigenpairs` expects.
 *
 * Cost: O(K^3) per sweep × O(log K) sweeps.  For K ≤ 100 this is
 * microsecond-scale; acceptable as long as m_restart stays bounded.
 *
 * Input `A_scratch` is destroyed (overwritten with the diagonalised
 * form as a side effect). */
sparse_err_t s21_dense_sym_jacobi(double *A_scratch, idx_t K, double *theta_out, double *Q_out) {
    if (!A_scratch || !theta_out || !Q_out)
        return SPARSE_ERR_NULL;
    if (K < 1)
        return SPARSE_ERR_BADARG;

    /* Q := I. */
    for (idx_t j = 0; j < K; j++) {
        for (idx_t i = 0; i < K; i++)
            Q_out[(size_t)i + (size_t)j * (size_t)K] = (i == j) ? 1.0 : 0.0;
    }

    if (K == 1) {
        theta_out[0] = A_scratch[0];
        return SPARSE_OK;
    }

    const idx_t max_sweeps = 100;
    const double tol = 1e-14;

    for (idx_t sweep = 0; sweep < max_sweeps; sweep++) {
        /* off-diagonal Frobenius norm */
        double off = 0.0;
        for (idx_t i = 0; i < K; i++) {
            for (idx_t j = i + 1; j < K; j++) {
                double aij = A_scratch[(size_t)i + (size_t)j * (size_t)K];
                off += aij * aij;
            }
        }
        if (sqrt(off) < tol)
            break;

        for (idx_t p = 0; p < K; p++) {
            for (idx_t q = p + 1; q < K; q++) {
                size_t pq = (size_t)p + (size_t)q * (size_t)K;
                double apq = A_scratch[pq];
                if (fabs(apq) < tol)
                    continue;
                double app = A_scratch[(size_t)p + (size_t)p * (size_t)K];
                double aqq = A_scratch[(size_t)q + (size_t)q * (size_t)K];
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (fabs(theta) > 1e15) {
                    t = 1.0 / (2.0 * theta);
                } else {
                    double sign_t = theta >= 0.0 ? 1.0 : -1.0;
                    t = sign_t / (fabs(theta) + sqrt(theta * theta + 1.0));
                }
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                /* Update rows/cols p, q of A (symmetric). */
                for (idx_t i = 0; i < K; i++) {
                    if (i == p || i == q)
                        continue;
                    double aip = A_scratch[(size_t)i + (size_t)p * (size_t)K];
                    double aiq = A_scratch[(size_t)i + (size_t)q * (size_t)K];
                    double new_ip = c * aip - s * aiq;
                    double new_iq = s * aip + c * aiq;
                    A_scratch[(size_t)i + (size_t)p * (size_t)K] = new_ip;
                    A_scratch[(size_t)p + (size_t)i * (size_t)K] = new_ip;
                    A_scratch[(size_t)i + (size_t)q * (size_t)K] = new_iq;
                    A_scratch[(size_t)q + (size_t)i * (size_t)K] = new_iq;
                }
                A_scratch[(size_t)p + (size_t)p * (size_t)K] =
                    c * c * app - 2.0 * s * c * apq + s * s * aqq;
                A_scratch[(size_t)q + (size_t)q * (size_t)K] =
                    s * s * app + 2.0 * s * c * apq + c * c * aqq;
                A_scratch[(size_t)p + (size_t)q * (size_t)K] = 0.0;
                A_scratch[(size_t)q + (size_t)p * (size_t)K] = 0.0;

                /* Update Q's rows p, q (equivalently cols p, q
                 * since we're building Q s.t. A = Q * diag * Q^T;
                 * each rotation is applied from the right to Q). */
                for (idx_t i = 0; i < K; i++) {
                    size_t ip = (size_t)i + (size_t)p * (size_t)K;
                    size_t iq = (size_t)i + (size_t)q * (size_t)K;
                    double qip = Q_out[ip];
                    double qiq = Q_out[iq];
                    Q_out[ip] = c * qip - s * qiq;
                    Q_out[iq] = s * qip + c * qiq;
                }
            }
        }
    }

    /* Sort eigenvalues ascending; permute Q columns to match. */
    for (idx_t i = 0; i < K; i++)
        theta_out[i] = A_scratch[(size_t)i + (size_t)i * (size_t)K];
    /* Simple selection sort — K is small. */
    for (idx_t i = 0; i < K; i++) {
        idx_t min_idx = i;
        for (idx_t j = i + 1; j < K; j++) {
            if (theta_out[j] < theta_out[min_idx])
                min_idx = j;
        }
        if (min_idx != i) {
            double tmp = theta_out[i];
            theta_out[i] = theta_out[min_idx];
            theta_out[min_idx] = tmp;
            for (idx_t r = 0; r < K; r++) {
                double q_tmp = Q_out[(size_t)r + (size_t)i * (size_t)K];
                Q_out[(size_t)r + (size_t)i * (size_t)K] =
                    Q_out[(size_t)r + (size_t)min_idx * (size_t)K];
                Q_out[(size_t)r + (size_t)min_idx * (size_t)K] = q_tmp;
            }
        }
    }

    return SPARSE_OK;
}
