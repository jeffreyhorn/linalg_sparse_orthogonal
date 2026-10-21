#include "sparse_vector.h"
#include <float.h>
#include <math.h>

static int vec_norm2_overflows(const double *v, idx_t n) {
    double hi = 0.0;
    double lo = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double a = scalbn(fabs(v[i]), 1 - DBL_MAX_EXP);
        double square = a * a;
        double square_error = fma(a, a, -square);
        double sum = hi + square;
        double part = sum - hi;
        double add_error = (hi - (sum - part)) + (square - part);
        double correction = lo + (square_error + add_error);
        hi = sum + correction;
        lo = (sum - hi) + correction;
        if (hi >= 4.0)
            return 1;
    }
    /* The squared round-to-nearest overflow midpoint after exponent scaling. */
    double threshold_hi = nextafter(4.0, 0.0);
    return hi > threshold_hi || (hi == threshold_hi && lo >= scalbn(1.0, -2 * DBL_MANT_DIG));
}

double vec_norm2(const double *v, idx_t n) {
    if (!v || n <= 0)
        return 0.0;
    double scale = 0.0;
    double ssq = 1.0;
    int has_infinity = 0;
    for (idx_t i = 0; i < n; i++) {
        double a = fabs(v[i]);
        if (isnan(a))
            return NAN;
        if (isinf(a)) {
            has_infinity = 1;
            continue;
        }
        if (a == 0.0)
            continue;
        if (a > scale) {
            double ratio = scale / a;
            ssq = 1.0 + ssq * ratio * ratio;
            scale = a;
        } else {
            double ratio = a / scale;
            ssq += ratio * ratio;
        }
    }
    if (has_infinity)
        return INFINITY;
    if (scale == 0.0)
        return 0.0;
    double norm = scale * sqrt(ssq);
    if (norm >= DBL_MAX / 2.0 && vec_norm2_overflows(v, n))
        return INFINITY;
    if (isinf(norm))
        return DBL_MAX;
    return norm;
}

double vec_norminf(const double *v, idx_t n) {
    if (!v || n <= 0)
        return 0.0;
    double mx = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double a = fabs(v[i]);
        if (isnan(a))
            return NAN;
        if (a > mx)
            mx = a;
    }
    return mx;
}

void vec_axpy(double a, const double *x, double *y, idx_t n) {
    if (!x || !y)
        return;
    for (idx_t i = 0; i < n; i++)
        y[i] += a * x[i];
}

void vec_copy(const double *src, double *dst, idx_t n) {
    if (!src || !dst)
        return;
    for (idx_t i = 0; i < n; i++)
        dst[i] = src[i];
}

void vec_zero(double *v, idx_t n) {
    if (!v)
        return;
    for (idx_t i = 0; i < n; i++)
        v[i] = 0.0;
}

double vec_dot(const double *x, const double *y, idx_t n) {
    if (!x || !y || n <= 0)
        return 0.0;
    double sum = 0.0;
    for (idx_t i = 0; i < n; i++)
        sum += x[i] * y[i];
    return sum;
}
