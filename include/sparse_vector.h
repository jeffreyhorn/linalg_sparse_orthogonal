#ifndef SPARSE_VECTOR_H
#define SPARSE_VECTOR_H

#include "sparse_types.h"

/*
 * Dense vector utility functions.
 *
 * These are small helpers used by the LU solver, iterative refinement,
 * and benchmarks. All operate on caller-allocated arrays of length n.
 */

/* 2-norm: sqrt(sum(v[i]^2)) */
double vec_norm2(const double *v, idx_t n);

/* Infinity-norm: max(|v[i]|) */
double vec_norminf(const double *v, idx_t n);

/* AXPY: y[i] += a * x[i] for i = 0..n-1 */
void vec_axpy(double a, const double *x, double *y, idx_t n);

/* Copy: dst[i] = src[i] for i = 0..n-1 */
void vec_copy(const double *src, double *dst, idx_t n);

/* Zero: v[i] = 0 for i = 0..n-1 */
void vec_zero(double *v, idx_t n);

/* Dot product: sum(x[i] * y[i]) */
double vec_dot(const double *x, const double *y, idx_t n);

#endif /* SPARSE_VECTOR_H */
