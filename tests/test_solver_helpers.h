#ifndef TEST_SOLVER_HELPERS_H
#define TEST_SOLVER_HELPERS_H

#include "sparse_matrix.h"

#include <math.h>
#include <stdlib.h>

/* Cluster-scoped helper layer for solver/integration tests.
 *
 * Sprint 37 Day 5 starts by consolidating the repeated L2 residual
 * calculations that were drifting across iterative/preconditioner and
 * integration tests.  Keep this header narrow and explicit instead of
 * growing a broad generic test framework.
 */

static inline double tf_vec_norm2(const double *v, idx_t n) {
    double sum = 0.0;
    for (idx_t i = 0; i < n; i++)
        sum += v[i] * v[i];
    return sqrt(sum);
}

static inline double tf_relative_residual_l2(const SparseMatrix *A, const double *b,
                                             const double *x, idx_t n, double alloc_fail_sentinel) {
    if (n == 0)
        return 0.0;

    double *r = calloc((size_t)n, sizeof(double));
    if (!r)
        return alloc_fail_sentinel;

    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];

    double rnorm = tf_vec_norm2(r, n);
    double bnorm = tf_vec_norm2(b, n);
    free(r);
    return (bnorm > 0.0) ? rnorm / bnorm : rnorm;
}

static inline double tf_block_relative_residual_l2(const SparseMatrix *A, const double *B,
                                                   const double *X, idx_t n, idx_t nrhs,
                                                   double alloc_fail_sentinel) {
    if (n == 0 || nrhs == 0)
        return 0.0;

    double *Y = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    if (!Y)
        return alloc_fail_sentinel;

    sparse_matvec_block(A, X, nrhs, Y);
    double worst = 0.0;
    for (idx_t k = 0; k < nrhs; k++) {
        double rnorm_sq = 0.0;
        double bnorm_sq = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double ri = B[i + k * n] - Y[i + k * n];
            rnorm_sq += ri * ri;
            bnorm_sq += B[i + k * n] * B[i + k * n];
        }
        double rnorm = sqrt(rnorm_sq);
        double bnorm = sqrt(bnorm_sq);
        double rel = (bnorm > 0.0) ? rnorm / bnorm : rnorm;
        if (rel > worst)
            worst = rel;
    }
    free(Y);
    return worst;
}

#ifdef TF_ENABLE_EXTERNAL_REFERENCE_HELPER

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#define tf_external_ref_popen _popen
#define tf_external_ref_pclose _pclose
#else
extern FILE *popen(const char *command, const char *mode);
extern int pclose(FILE *stream);
#define tf_external_ref_popen popen
#define tf_external_ref_pclose pclose
#endif

typedef enum {
    TF_EXTERNAL_REFERENCE_ERROR = -1,
    TF_EXTERNAL_REFERENCE_SKIP = 0,
    TF_EXTERNAL_REFERENCE_OK = 1
} tf_external_reference_status_t;

static inline void tf_external_reference_copy_reason(char *reason, size_t reason_cap,
                                                     const char *text) {
    if (!reason || reason_cap == 0)
        return;
    snprintf(reason, reason_cap, "%s", text ? text : "");
    size_t len = strlen(reason);
    while (len > 0 && (reason[len - 1] == '\n' || reason[len - 1] == '\r')) {
        reason[len - 1] = '\0';
        len--;
    }
}

static inline tf_external_reference_status_t
tf_read_external_reference_vector(const char *cmd, const char *label, double *x_out, idx_t n,
                                  char *reason, size_t reason_cap) {
    if (!reason || reason_cap == 0)
        return TF_EXTERNAL_REFERENCE_ERROR;
    if (!cmd || !label || !x_out) {
        snprintf(reason, reason_cap, "external reference invalid arguments");
        return TF_EXTERNAL_REFERENCE_ERROR;
    }

    FILE *pipe = tf_external_ref_popen(cmd, "r");
    if (!pipe) {
        snprintf(reason, reason_cap, "python3 pipe open failed");
        return TF_EXTERNAL_REFERENCE_SKIP;
    }

    char line[256];
    if (!fgets(line, sizeof(line), pipe)) {
        tf_external_ref_pclose(pipe);
        snprintf(reason, reason_cap, "%s produced no output", label);
        return TF_EXTERNAL_REFERENCE_ERROR;
    }

    if (strncmp(line, "SKIP ", 5) == 0) {
        tf_external_ref_pclose(pipe);
        tf_external_reference_copy_reason(reason, reason_cap, line + 5);
        return TF_EXTERNAL_REFERENCE_SKIP;
    }
    if (strncmp(line, "ERROR ", 6) == 0) {
        tf_external_ref_pclose(pipe);
        tf_external_reference_copy_reason(reason, reason_cap, line + 6);
        return TF_EXTERNAL_REFERENCE_ERROR;
    }

    idx_t got_n = -1;
    int consumed = 0;
    if (sscanf(line, "OK %" SPARSE_SCNIDX "%n", &got_n, &consumed) != 1 || got_n != n) {
        tf_external_ref_pclose(pipe);
        snprintf(reason, reason_cap, "%s returned invalid dimension header", label);
        return TF_EXTERNAL_REFERENCE_ERROR;
    }
    const char *header_end = line + consumed;
    while (*header_end != '\0') {
        if (!isspace((unsigned char)*header_end)) {
            tf_external_ref_pclose(pipe);
            snprintf(reason, reason_cap, "%s trailing data in dimension header", label);
            return TF_EXTERNAL_REFERENCE_ERROR;
        }
        header_end++;
    }

    for (idx_t i = 0; i < n; i++) {
        if (!fgets(line, sizeof(line), pipe)) {
            tf_external_ref_pclose(pipe);
            snprintf(reason, reason_cap, "%s truncated at entry %" SPARSE_PRIDX, label, i);
            return TF_EXTERNAL_REFERENCE_ERROR;
        }
        char *end = NULL;
        x_out[i] = strtod(line, &end);
        if (end == line) {
            tf_external_ref_pclose(pipe);
            snprintf(reason, reason_cap, "%s parse failure at entry %" SPARSE_PRIDX, label, i);
            return TF_EXTERNAL_REFERENCE_ERROR;
        }
        while (*end != '\0') {
            if (!isspace((unsigned char)*end)) {
                tf_external_ref_pclose(pipe);
                snprintf(reason, reason_cap, "%s trailing data at entry %" SPARSE_PRIDX, label, i);
                return TF_EXTERNAL_REFERENCE_ERROR;
            }
            end++;
        }
    }

    while (fgets(line, sizeof(line), pipe)) {
        const char *extra = line;
        while (*extra != '\0') {
            if (!isspace((unsigned char)*extra)) {
                tf_external_ref_pclose(pipe);
                snprintf(reason, reason_cap, "%s produced trailing output", label);
                return TF_EXTERNAL_REFERENCE_ERROR;
            }
            extra++;
        }
    }

    if (tf_external_ref_pclose(pipe) != 0) {
        snprintf(reason, reason_cap, "%s exited non-zero", label);
        return TF_EXTERNAL_REFERENCE_ERROR;
    }
    return TF_EXTERNAL_REFERENCE_OK;
}

#endif

#endif
