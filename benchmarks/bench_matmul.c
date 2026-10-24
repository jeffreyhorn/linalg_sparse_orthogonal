/* Pinned sparse_matmul fixtures. One invocation measures one multiply. */

/* Expose clock_gettime and CLOCK_MONOTONIC under strict C11 on POSIX. */
#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _POSIX_C_SOURCE 200809L
#endif

#include "sparse_matrix.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *name;
    idx_t a_rows;
    idx_t a_cols;
    idx_t b_cols;
    idx_t a_nnz;
    idx_t b_nnz;
    idx_t c_nnz;
} matmul_fixture_t;

static const matmul_fixture_t fixtures[] = {
    {"banded", 256, 256, 256, 256, 766, 766},
    {"wide-row", 1, 1024, 1024, 1024, 1024, 1024},
    {"rectangular", 64, 32, 96, 2048, 3072, 6144},
};

static double monotonic_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return -1.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int insert_checked(SparseMatrix *matrix, idx_t row, idx_t col, double value) {
    return sparse_insert(matrix, row, col, value) == SPARSE_OK;
}

static int populate(const matmul_fixture_t *fixture, SparseMatrix *a, SparseMatrix *b) {
    if (strcmp(fixture->name, "banded") == 0) {
        for (idx_t i = 0; i < fixture->a_rows; i++) {
            if (!insert_checked(a, i, i, 2.0))
                return 0;
            for (idx_t j = i > 0 ? i - 1 : i; j <= i + 1 && j < fixture->b_cols; j++) {
                if (!insert_checked(b, i, j, 1.0))
                    return 0;
            }
        }
    } else if (strcmp(fixture->name, "wide-row") == 0) {
        for (idx_t i = 0; i < fixture->a_cols; i++) {
            if (!insert_checked(a, 0, i, 1.0) || !insert_checked(b, i, i, 1.0))
                return 0;
        }
    } else {
        for (idx_t i = 0; i < fixture->a_rows; i++) {
            for (idx_t j = 0; j < fixture->a_cols; j++) {
                if (!insert_checked(a, i, j, 1.0 + (double)((i + j) % 7) / 16.0))
                    return 0;
            }
        }
        for (idx_t i = 0; i < fixture->a_cols; i++) {
            for (idx_t j = 0; j < fixture->b_cols; j++) {
                if (!insert_checked(b, i, j, 1.0 + (double)((2 * i + j) % 11) / 16.0))
                    return 0;
            }
        }
    }
    return 1;
}

static double expected_value(const matmul_fixture_t *fixture, idx_t row, idx_t col) {
    if (strcmp(fixture->name, "banded") == 0)
        return col == row || (row > 0 && col == row - 1) || col == row + 1 ? 2.0 : 0.0;
    if (strcmp(fixture->name, "wide-row") == 0)
        return 1.0;

    double value = 0.0;
    for (idx_t k = 0; k < fixture->a_cols; k++) {
        double a = 1.0 + (double)((row + k) % 7) / 16.0;
        double b = 1.0 + (double)((2 * k + col) % 11) / 16.0;
        value += a * b;
    }
    return value;
}

static int validate_output(const matmul_fixture_t *fixture, const SparseMatrix *c) {
    if (!c || sparse_rows(c) != fixture->a_rows || sparse_cols(c) != fixture->b_cols ||
        sparse_nnz(c) != fixture->c_nnz)
        return 0;
    for (idx_t row = 0; row < fixture->a_rows; row++) {
        for (idx_t col = 0; col < fixture->b_cols; col++) {
            double expected = expected_value(fixture, row, col);
            double actual = sparse_get_phys(c, row, col);
            if (!isfinite(actual) || fabs(actual - expected) > 1e-11 * (1.0 + fabs(expected)))
                return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s {banded|wide-row|rectangular}\n", argv[0]);
        return 2;
    }

    const matmul_fixture_t *fixture = NULL;
    for (size_t i = 0; i < sizeof(fixtures) / sizeof(fixtures[0]); i++) {
        if (strcmp(argv[1], fixtures[i].name) == 0) {
            fixture = &fixtures[i];
            break;
        }
    }
    if (!fixture) {
        fprintf(stderr, "unknown fixture: %s\n", argv[1]);
        return 2;
    }

    SparseMatrix *a = sparse_create(fixture->a_rows, fixture->a_cols);
    SparseMatrix *b = sparse_create(fixture->a_cols, fixture->b_cols);
    SparseMatrix *c = NULL;
    int status = 1;
    if (!a || !b || !populate(fixture, a, b) || sparse_nnz(a) != fixture->a_nnz ||
        sparse_nnz(b) != fixture->b_nnz) {
        fprintf(stderr, "fixture setup failed: %s\n", fixture->name);
        goto done;
    }

    double start = monotonic_seconds();
    if (start < 0.0) {
        fprintf(stderr, "clock failed\n");
        goto done;
    }
    sparse_err_t rc = sparse_matmul(a, b, &c);
    double end = monotonic_seconds();
    if (rc != SPARSE_OK || end < start || !validate_output(fixture, c)) {
        fprintf(stderr, "matmul failed: %s, status=%d, nnz=%lld\n", fixture->name, (int)rc,
                (long long)(c ? sparse_nnz(c) : -1));
        goto done;
    }

    printf("fixture,a_rows,a_cols,b_cols,a_nnz,b_nnz,c_nnz,zero_drops,elapsed_ms\n");
    printf("%s,%lld,%lld,%lld,%lld,%lld,%lld,0,%.6f\n", fixture->name, (long long)fixture->a_rows,
           (long long)fixture->a_cols, (long long)fixture->b_cols, (long long)fixture->a_nnz,
           (long long)fixture->b_nnz, (long long)fixture->c_nnz, (end - start) * 1000.0);
    status = 0;

done:
    sparse_free(c);
    sparse_free(b);
    sparse_free(a);
    return status;
}
