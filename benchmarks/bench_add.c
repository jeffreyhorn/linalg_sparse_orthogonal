/* Pinned sparse_add construction profiles. One invocation measures one add. */

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

typedef enum {
    PROFILE_WIDE_ROW,
    PROFILE_VARIED_ROWS,
    PROFILE_NARROW_ROWS,
} add_profile_kind_t;

typedef struct {
    const char *name;
    const char *workload_id;
    add_profile_kind_t kind;
    idx_t rows;
    idx_t cols;
    idx_t expected_a_nnz;
    idx_t expected_b_nnz;
    idx_t expected_c_nnz;
} add_fixture_t;

static const add_fixture_t fixtures[] = {
    {"wide-row", "s227-add-wide-r1-c4096-v1", PROFILE_WIDE_ROW, 1, 4096, 2048, 2048, 4096},
    {"varied-rows", "s227-add-varied-r64-c1024-v1", PROFILE_VARIED_ROWS, 64, 1024, 4608, 4608,
     9216},
    {"narrow-rows", "s227-add-narrow-r1024-c64-v1", PROFILE_NARROW_ROWS, 1024, 64, 4096, 4096,
     8192},
};

static double monotonic_seconds(void) {
    struct timespec ts;
#ifdef _WIN32
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC)
        return -1.0;
#else
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return -1.0;
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static idx_t entries_per_side(const add_fixture_t *fixture, idx_t row) {
    if (fixture->kind == PROFILE_WIDE_ROW)
        return 2048;
    if (fixture->kind == PROFILE_VARIED_ROWS)
        return 16 + (row % 8) * 16;
    return 4;
}

static idx_t fixture_col(const add_fixture_t *fixture, idx_t row, idx_t entry, int side_b) {
    if (fixture->kind == PROFILE_WIDE_ROW)
        return 2 * entry + side_b;
    if (fixture->kind == PROFILE_VARIED_ROWS)
        return 2 * entry + side_b;
    return (8 * row + 2 * entry + side_b) % fixture->cols;
}

static double fixture_value(idx_t row, idx_t entry, int side_b) {
    double magnitude = 1.0 + (double)((3 * row + 5 * entry + 7 * side_b) % 17) / 16.0;
    return side_b ? -magnitude : magnitude;
}

static int populate(const add_fixture_t *fixture, SparseMatrix *a, SparseMatrix *b) {
    for (idx_t row = 0; row < fixture->rows; row++) {
        idx_t width = entries_per_side(fixture, row);
        for (idx_t entry = 0; entry < width; entry++) {
            idx_t a_col = fixture_col(fixture, row, entry, 0);
            idx_t b_col = fixture_col(fixture, row, entry, 1);
            if (sparse_insert(a, row, a_col, fixture_value(row, entry, 0)) != SPARSE_OK ||
                sparse_insert(b, row, b_col, fixture_value(row, entry, 1)) != SPARSE_OK)
                return 0;
        }
    }
    return sparse_nnz(a) == fixture->expected_a_nnz && sparse_nnz(b) == fixture->expected_b_nnz;
}

static double expected_value(const add_fixture_t *fixture, idx_t row, idx_t col, int matrix) {
    idx_t width = entries_per_side(fixture, row);
    for (idx_t entry = 0; entry < width; entry++) {
        if (matrix != 2 && col == fixture_col(fixture, row, entry, 0))
            return fixture_value(row, entry, 0);
        if (matrix != 1 && col == fixture_col(fixture, row, entry, 1))
            return fixture_value(row, entry, 1);
    }
    return 0.0;
}

static int validate_matrix(const add_fixture_t *fixture, const SparseMatrix *matrix, int which) {
    idx_t expected_nnz = fixture->expected_c_nnz;
    if (which == 1)
        expected_nnz = fixture->expected_a_nnz;
    else if (which == 2)
        expected_nnz = fixture->expected_b_nnz;
    if (!matrix || sparse_rows(matrix) != fixture->rows || sparse_cols(matrix) != fixture->cols ||
        sparse_nnz(matrix) != expected_nnz)
        return 0;

    for (idx_t row = 0; row < fixture->rows; row++) {
        for (idx_t col = 0; col < fixture->cols; col++) {
            double expected = expected_value(fixture, row, col, which);
            double actual = sparse_get_phys(matrix, row, col);
            if (!isfinite(actual) || actual != expected)
                return 0;
        }
    }
    return 1;
}

static int run_fixture(const add_fixture_t *fixture, int print_header) {
    SparseMatrix *a = sparse_create(fixture->rows, fixture->cols);
    SparseMatrix *b = sparse_create(fixture->rows, fixture->cols);
    SparseMatrix *c = NULL;
    int status = 1;
    if (!a || !b || !populate(fixture, a, b)) {
        fprintf(stderr, "fixture setup failed: %s\n", fixture->name);
        goto done;
    }

    double start = monotonic_seconds();
    if (start < 0.0) {
        fprintf(stderr, "clock failed\n");
        goto done;
    }
    sparse_err_t rc = sparse_add(a, b, 1.0, 1.0, &c);
    double end = monotonic_seconds();
    if (rc != SPARSE_OK || end < start || !validate_matrix(fixture, c, 0) ||
        !validate_matrix(fixture, a, 1) || !validate_matrix(fixture, b, 2)) {
        fprintf(stderr, "add failed: %s, status=%d, nnz=%lld\n", fixture->name, (int)rc,
                (long long)(c ? sparse_nnz(c) : -1));
        goto done;
    }

    if (print_header)
        printf("fixture,workload_id,rows,cols,a_nnz,b_nnz,c_nnz,max_output_row_width,"
               "elapsed_ms\n");
    idx_t max_width =
        fixture->kind == PROFILE_WIDE_ROW ? 4096 : (fixture->kind == PROFILE_VARIED_ROWS ? 256 : 8);
    printf("%s,%s,%lld,%lld,%lld,%lld,%lld,%lld,%.6f\n", fixture->name, fixture->workload_id,
           (long long)fixture->rows, (long long)fixture->cols, (long long)fixture->expected_a_nnz,
           (long long)fixture->expected_b_nnz, (long long)fixture->expected_c_nnz,
           (long long)max_width, (end - start) * 1000.0);
    status = 0;

done:
    sparse_free(c);
    sparse_free(b);
    sparse_free(a);
    return status;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        for (size_t i = 0; i < sizeof(fixtures) / sizeof(fixtures[0]); i++) {
            if (run_fixture(&fixtures[i], i == 0) != 0)
                return 1;
        }
        return 0;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s [wide-row|varied-rows|narrow-rows]\n", argv[0]);
        return 2;
    }

    for (size_t i = 0; i < sizeof(fixtures) / sizeof(fixtures[0]); i++) {
        if (strcmp(argv[1], fixtures[i].name) == 0)
            return run_fixture(&fixtures[i], 1);
    }
    fprintf(stderr, "unknown fixture: %s\n", argv[1]);
    return 2;
}
