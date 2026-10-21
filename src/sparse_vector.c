#include "sparse_vector.h"
#include <float.h>
#include <math.h>
#include <stdint.h>

enum {
    VEC_EXACT_BASE_EXP = 2 * (DBL_MIN_EXP - 2 * DBL_MANT_DIG + 1),
    VEC_EXACT_WORDS = (2 * DBL_MAX_EXP - VEC_EXACT_BASE_EXP + SPARSE_IDX_BITS + 31) / 32
};

_Static_assert(FLT_RADIX == 2 && DBL_MANT_DIG == 53, "binary64 required for exact norm comparison");

static void vec_exact_add_word(uint32_t *words, uint32_t value, unsigned shift) {
    size_t index = shift / 32;
    unsigned offset = shift % 32;
    uint64_t chunk = (uint64_t)value << offset;
    while (chunk != 0 && index < VEC_EXACT_WORDS) {
        uint64_t total = (uint64_t)words[index] + (uint32_t)chunk;
        words[index] = (uint32_t)total;
        chunk = (chunk >> 32) + (total >> 32);
        index++;
    }
}

static int vec_norm2_exact_overflows(const double *v, idx_t n) {
    uint32_t sum[VEC_EXACT_WORDS] = {0};
    uint32_t threshold[VEC_EXACT_WORDS] = {0};
    for (idx_t i = 0; i < n; i++) {
        if (v[i] == 0.0)
            continue;
        int exponent;
        uint64_t mantissa = (uint64_t)scalbn(frexp(fabs(v[i]), &exponent), DBL_MANT_DIG);
        uint32_t parts[2] = {(uint32_t)mantissa, (uint32_t)(mantissa >> 32)};
        uint32_t square[4] = {0};
        for (size_t j = 0; j < 2; j++) {
            uint64_t carry = 0;
            for (size_t k = 0; k < 2; k++) {
                uint64_t product = (uint64_t)parts[j] * parts[k] + square[j + k] + carry;
                square[j + k] = (uint32_t)product;
                carry = product >> 32;
            }
            square[j + 2] = (uint32_t)carry;
        }
        unsigned shift = (unsigned)(2 * (exponent - DBL_MANT_DIG) - VEC_EXACT_BASE_EXP);
        for (unsigned j = 0; j < 4; j++)
            vec_exact_add_word(sum, square[j], shift + 32 * j);
    }

    /* (2^E - 2^(E-p-1))^2 = 2^(2E) - 2^(2E-p) + 2^(2E-2p-2). */
    for (int bit = 2 * DBL_MAX_EXP - DBL_MANT_DIG; bit < 2 * DBL_MAX_EXP; bit++) {
        unsigned shift = (unsigned)(bit - VEC_EXACT_BASE_EXP);
        threshold[shift / 32] |= UINT32_C(1) << (shift % 32);
    }
    unsigned low_bit = (unsigned)(2 * DBL_MAX_EXP - 2 * DBL_MANT_DIG - 2 - VEC_EXACT_BASE_EXP);
    threshold[low_bit / 32] |= UINT32_C(1) << (low_bit % 32);
    for (size_t i = VEC_EXACT_WORDS; i-- > 0;) {
        if (sum[i] != threshold[i])
            return sum[i] > threshold[i];
    }
    return 1;
}

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
    }
    /* The squared round-to-nearest overflow midpoint after exponent scaling. */
    double threshold_hi = nextafter(4.0, 0.0);
    double threshold_lo = scalbn(1.0, -2 * DBL_MANT_DIG);
    double difference = (hi - threshold_hi) + (lo - threshold_lo);
    double count = (double)n;
    double error_bound = 64.0 * count * count * DBL_EPSILON;
    if (difference > error_bound)
        return 1;
    if (difference < -error_bound)
        return 0;
    return vec_norm2_exact_overflows(v, n);
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
