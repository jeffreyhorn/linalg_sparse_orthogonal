#ifndef SPARSE_VECTOR_H
#define SPARSE_VECTOR_H

/**
 * @file sparse_vector.h
 * @brief Dense vector utility functions.
 *
 * Small helpers for BLAS-like operations on caller-allocated arrays.
 * Used by the LU solver, iterative refinement, and benchmarks.
 * All functions accept NULL pointers: mutating helpers do nothing, and
 * scalar-returning helpers return 0.0. This is a compatibility convention,
 * not a mathematical norm or dot product for a missing vector.
 */

#include "sparse_types.h"

/**
 * @brief Compute the Euclidean (2-) norm using scaled sum of squares.
 *
 * Scaling avoids a false infinity or zero from directly squaring finite
 * components when the final nonzero norm is representable. Tiny individual
 * contributions can still round away. It does not extend the range of the
 * double return type: a final norm too large for double is +infinity.
 * NaN in any element takes precedence over infinity, regardless of order.
 * An infinite element without NaN yields +infinity; a finite zero vector
 * yields +0.0. No error code or allocation is involved.
 *
 * @param v  Input vector (NULL returns +0.0 without dereferencing).
 * @param n  Vector length (n <= 0 returns +0.0 without dereferencing).
 * @return The nonnegative 2-norm if representable, +infinity on final-result
 *         overflow or an infinite element, or NaN if any element is NaN.
 */
double vec_norm2(const double *v, idx_t n);

/**
 * @brief Compute the infinity norm: max(|v[i]|).
 *
 * NaN in any element returns NaN even if another element is infinite.
 * Otherwise an infinite element returns +infinity, and an all-zero vector
 * returns +0.0. Finite inputs cannot overflow this maximum.
 *
 * @param v  Input vector (NULL returns +0.0 without dereferencing).
 * @param n  Vector length (n <= 0 returns +0.0 without dereferencing).
 * @return The nonnegative infinity norm, or NaN if any element is NaN.
 */
double vec_norminf(const double *v, idx_t n);

/**
 * @brief AXPY operation: y[i] += a * x[i] for i = 0..n-1.
 *
 * @param a  Scalar multiplier.
 * @param x  Input vector (NULL is a no-op).
 * @param y  Input/output vector (NULL is a no-op).
 * @param n  Vector length.
 */
void vec_axpy(double a, const double *x, double *y, idx_t n);

/**
 * @brief Copy: dst[i] = src[i] for i = 0..n-1.
 *
 * @param src  Source vector (NULL is a no-op).
 * @param dst  Destination vector (NULL is a no-op).
 * @param n    Vector length.
 */
void vec_copy(const double *src, double *dst, idx_t n);

/**
 * @brief Zero fill: v[i] = 0 for i = 0..n-1.
 *
 * @param v  Vector to zero (NULL is a no-op).
 * @param n  Vector length.
 */
void vec_zero(double *v, idx_t n);

/**
 * @brief Dot product: sum(x[i] * y[i]) for i = 0..n-1.
 *
 * @param x  First input vector (NULL returns 0.0).
 * @param y  Second input vector (NULL returns 0.0).
 * @param n  Vector length (n <= 0 returns 0.0).
 * @return The dot product of x and y.
 */
double vec_dot(const double *x, const double *y, idx_t n);

#endif /* SPARSE_VECTOR_H */
