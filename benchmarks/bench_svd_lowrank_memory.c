#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#if !defined(_WIN32) && !defined(__APPLE__) &&                                                     \
    (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L)
#define _POSIX_C_SOURCE 200809L
#endif

#include "sparse_matrix.h"
#include "sparse_svd.h"
#include "sparse_svd_internal.h"
#include "sparse_types.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/resource.h>
#include <sys/utsname.h>
#endif

typedef struct {
    const char *name;
    idx_t rows;
    idx_t cols;
    idx_t rank;
    idx_t effective_rank;
    int first_power;
    double drop_tol;
    int zero;
} workload_t;

static const workload_t workloads[] = {
    {"measure-square-512-r16", 512, 512, 16, 32, 5, 1e-12, 0},
    {"measure-tall-1024x256-r8", 1024, 256, 8, 8, 3, 0.0, 0},
    {"measure-wide-256x1024-r24", 256, 1024, 24, 32, 4, 1e-9, 0},
    {"measure-zero-512-r16", 512, 512, 16, 0, 0, 0.0, 1},
};

static uint64_t monotonic_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&counter))
        return 0;
    return (uint64_t)((counter.QuadPart * INT64_C(1000000000)) / frequency.QuadPart);
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
#endif
}

static uint64_t peak_rss_bytes(void) {
#if defined(_WIN32)
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        return 0;
#if defined(__APPLE__)
    return (uint64_t)usage.ru_maxrss;
#else
    return (uint64_t)usage.ru_maxrss * UINT64_C(1024);
#endif
#endif
}

static const workload_t *find_workload(const char *name) {
    for (size_t i = 0; i < sizeof(workloads) / sizeof(workloads[0]); i++)
        if (strcmp(workloads[i].name, name) == 0)
            return &workloads[i];
    return NULL;
}

static SparseMatrix *build_workload(const workload_t *workload) {
    SparseMatrix *matrix = sparse_create(workload->rows, workload->cols);
    if (!matrix || workload->zero)
        return matrix;
    for (idx_t i = 0; i < workload->effective_rank; i++) {
        double value = ldexp(1.0, workload->first_power - (int)i);
        if (sparse_insert(matrix, i, i, value) != SPARSE_OK) {
            sparse_free(matrix);
            return NULL;
        }
    }
    return matrix;
}

static uint64_t checksum_output(const SparseMatrix *matrix) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (idx_t row = 0; row < sparse_rows(matrix); row++) {
        for (idx_t col = 0; col < sparse_cols(matrix); col++) {
            double value = sparse_get(matrix, row, col);
            if (value == 0.0)
                continue;
            uint64_t bits = 0;
            memcpy(&bits, &value, sizeof(bits));
            hash ^= (uint64_t)row;
            hash *= UINT64_C(1099511628211);
            hash ^= (uint64_t)col;
            hash *= UINT64_C(1099511628211);
            hash ^= bits;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static int output_is_correct(const workload_t *workload, const SparseMatrix *output) {
    idx_t expected_nnz = workload->zero ? 0 : workload->rank;
    if (!output || sparse_rows(output) != workload->rows || sparse_cols(output) != workload->cols ||
        sparse_nnz(output) != expected_nnz)
        return 0;
    for (idx_t i = 0; i < expected_nnz; i++) {
        double expected = ldexp(1.0, workload->first_power - (int)i);
        double actual = sparse_get(output, i, i);
        if (!isfinite(actual) || fabs(actual - expected) > 1e-11 * fabs(expected))
            return 0;
    }
    return 1;
}

static int set_route(const char *route) {
#if defined(_WIN32)
    if (strcmp(route, "default") == 0)
        return _putenv_s("SPARSE_SVD_LOWRANK_OUTER", "");
    return _putenv_s("SPARSE_SVD_LOWRANK_OUTER", "off");
#else
    if (strcmp(route, "default") == 0)
        return unsetenv("SPARSE_SVD_LOWRANK_OUTER");
    return setenv("SPARSE_SVD_LOWRANK_OUTER", "off", 1);
#endif
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[2], "default") != 0 && strcmp(argv[2], "dense") != 0)) {
        fprintf(stderr, "usage: %s WORKLOAD default|dense\n", argv[0]);
        return 2;
    }
    const workload_t *workload = find_workload(argv[1]);
    if (!workload || set_route(argv[2]) != 0) {
        fprintf(stderr, "invalid workload or route setup failed: %s\n", strerror(errno));
        return 2;
    }

    SparseMatrix *input = build_workload(workload);
    if (!input) {
        fprintf(stderr, "input construction failed\n");
        return 2;
    }
    uint64_t rss_before = peak_rss_bytes();
    uint64_t start = monotonic_ns();
    SparseMatrix *output = NULL;
    sparse_err_t status =
        sparse_svd_lowrank_sparse(input, workload->rank, workload->drop_tol, &output);
    uint64_t elapsed = monotonic_ns() - start;
    uint64_t rss_after = peak_rss_bytes();
    sparse_svd_lowrank_metrics_t metrics;
    sparse_svd_lowrank_metrics_get(&metrics);

    size_t expected_scratch = 0;
    if (!workload->zero && strcmp(argv[2], "dense") == 0)
        expected_scratch = (size_t)workload->rows * (size_t)workload->cols * sizeof(double);
    int correct = status == SPARSE_OK && output_is_correct(workload, output) && metrics.valid &&
                  metrics.reconstruction_scratch_peak_bytes == expected_scratch &&
                  metrics.output_nnz == (workload->zero ? 0 : workload->rank);
    uint64_t checksum = output ? checksum_output(output) : 0;

    printf("{\"schema\":\"s228-memory-v1\",\"workload\":\"%s\","
           "\"route\":\"%s\",\"selector\":\"%s\",\"rows\":%" SPARSE_PRIDX ",\"cols\":%" SPARSE_PRIDX
           ",\"rank\":%" SPARSE_PRIDX ",\"effective_rank\":%" SPARSE_PRIDX ",\"drop_tol\":%.17g,"
           "\"status\":%d,\"correct\":%s,\"nnz\":%" SPARSE_PRIDX ",\"checksum\":\"%016" PRIx64
           "\",\"wall_ns\":%" PRIu64 ",\"rss_before_bytes\":%" PRIu64 ",\"peak_rss_bytes\":%" PRIu64
           ",\"bidiag_work_peak_bytes\":%zu,\"bidiag_work_allocations\":%zu,"
           "\"retained_factor_bytes\":%zu,\"retained_factor_allocations\":%zu,"
           "\"reconstruction_scratch_peak_bytes\":%zu,"
           "\"reconstruction_scratch_allocations\":%zu,"
           "\"sparse_shell_map_bytes\":%zu,\"sparse_shell_map_allocations\":%zu,"
           "\"sparse_node_slab_bytes\":%zu,\"sparse_node_slab_allocations\":%zu}\n",
           workload->name, argv[2], strcmp(argv[2], "default") == 0 ? "unset" : "off",
           workload->rows, workload->cols, workload->rank, workload->effective_rank,
           workload->drop_tol, (int)status, correct ? "true" : "false",
           output ? sparse_nnz(output) : 0, checksum, elapsed, rss_before, rss_after,
           metrics.bidiag_work_peak_bytes, metrics.bidiag_work_allocations,
           metrics.retained_factor_bytes, metrics.retained_factor_allocations,
           metrics.reconstruction_scratch_peak_bytes, metrics.reconstruction_scratch_allocations,
           metrics.sparse_shell_map_bytes, metrics.sparse_shell_map_allocations,
           metrics.sparse_node_slab_bytes, metrics.sparse_node_slab_allocations);

    sparse_free(output);
    sparse_free(input);
    return correct ? 0 : 1;
}
