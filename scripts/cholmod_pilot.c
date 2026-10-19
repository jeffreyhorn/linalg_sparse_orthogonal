#define _GNU_SOURCE

#include "sparse_cholesky.h"
#include <cholmod.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const double expected[4][4] = {
    {10.0, 1.0, 2.0, 0.0},
    {1.0, 12.0, 1.0, 0.0},
    {2.0, 1.0, 14.0, 0.0},
    {0.0, 0.0, 0.0, 16.0},
};
static const double rhs[4] = {13.0, 14.0, 17.0, 16.0};

static int check_input(const SparseMatrix *a) {
    if (sparse_rows(a) != 4 || sparse_cols(a) != 4 || sparse_nnz(a) != 10) {
        return 0;
    }
    for (idx_t row = 0; row < 4; ++row) {
        double sum = 0.0;
        for (idx_t col = 0; col < 4; ++col) {
            double value = sparse_get(a, row, col);
            if (!isfinite(value) || value != expected[row][col]) {
                return 0;
            }
            sum += value;
        }
        if (sum != rhs[row]) {
            return 0;
        }
    }
    return 1;
}

static void print_solution(const double x[4]) {
    printf("x=%.17g,%.17g,%.17g,%.17g\n", x[0], x[1], x[2], x[3]);
}

static int run_project(const SparseMatrix *a) {
    SparseMatrix *factor = sparse_copy(a);
    if (!factor) {
        fprintf(stderr, "project copy failed\n");
        return 2;
    }
    int used_csc_path = -1;
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
        .used_csc_path = &used_csc_path,
    };
    sparse_err_t factor_status = sparse_cholesky_factor_opts(factor, &opts);
    printf("factor_status=%d\nused_csc_path=%d\n", (int)factor_status, used_csc_path);
    if (factor_status != SPARSE_OK || used_csc_path != 0) {
        sparse_free(factor);
        return 2;
    }
    double x[4] = {0.0};
    sparse_err_t solve_status = sparse_cholesky_solve(factor, rhs, x);
    printf("solve_status=%d\n", (int)solve_status);
    if (solve_status == SPARSE_OK) {
        print_solution(x);
    }
    sparse_free(factor);
    return solve_status == SPARSE_OK ? 0 : 2;
}

static int run_baseline(const SparseMatrix *a) {
    cholmod_common common;
    if (!cholmod_start(&common)) {
        fprintf(stderr, "cholmod_start failed\n");
        return 2;
    }
    common.nmethods = 1;
    common.method[0].ordering = CHOLMOD_NATURAL;
    common.postorder = 0;
    common.supernodal = CHOLMOD_SIMPLICIAL;

    cholmod_sparse *matrix =
        cholmod_allocate_sparse(4, 4, 7, 1, 1, -1, CHOLMOD_REAL + CHOLMOD_DOUBLE, &common);
    cholmod_dense *b = cholmod_allocate_dense(4, 1, 4, CHOLMOD_REAL + CHOLMOD_DOUBLE, &common);
    cholmod_factor *factor = NULL;
    cholmod_dense *solution = NULL;
    int result = 2;
    if (!matrix || !b || matrix->itype != CHOLMOD_INT || matrix->dtype != CHOLMOD_DOUBLE) {
        fprintf(stderr, "CHOLMOD allocation or ABI mismatch: status=%d\n", common.status);
        goto done;
    }

    int32_t *p = (int32_t *)matrix->p;
    int32_t *i = (int32_t *)matrix->i;
    double *values = (double *)matrix->x;
    int32_t cursor = 0;
    for (int32_t col = 0; col < 4; ++col) {
        p[col] = cursor;
        for (int32_t row = col; row < 4; ++row) {
            double value = sparse_get(a, row, col);
            if (value != 0.0) {
                i[cursor] = row;
                values[cursor] = value;
                ++cursor;
            }
        }
    }
    p[4] = cursor;
    if (cursor != 7) {
        fprintf(stderr, "CHOLMOD CSC conversion has %d entries, expected 7\n", (int)cursor);
        goto done;
    }
    memcpy(b->x, rhs, sizeof(rhs));
    printf("csc_nnz=%d\n", (int)cursor);
    printf("csc_stype=%d\ncsc_sorted=%d\ncsc_packed=%d\ncsc_itype=%d\n", matrix->stype,
           matrix->sorted, matrix->packed, matrix->itype);
    factor = cholmod_analyze(matrix, &common);
    printf("analyze_status=%d\n", common.status);
    if (!factor || common.status != CHOLMOD_OK) {
        goto done;
    }
    int factor_ok = cholmod_factorize(matrix, factor, &common);
    printf("factor_status=%d\ncholmod_status=%d\nfactor_minor=%zu\n", factor_ok, common.status,
           factor->minor);
    printf("ordering=%d\nis_super=%d\nis_ll=%d\n", factor->ordering, factor->is_super,
           factor->is_ll);
    if (!factor_ok || common.status != CHOLMOD_OK || factor->minor != 4 ||
        factor->ordering != CHOLMOD_NATURAL || factor->is_super != 0) {
        goto done;
    }
    solution = cholmod_solve(CHOLMOD_A, factor, b, &common);
    printf("solve_status=%d\ncholmod_solve_status=%d\n", solution != NULL, common.status);
    if (!solution || common.status != CHOLMOD_OK || solution->xtype != CHOLMOD_REAL ||
        solution->dtype != CHOLMOD_DOUBLE) {
        goto done;
    }
    print_solution((const double *)solution->x);
    result = 0;

done:
    cholmod_free_dense(&solution, &common);
    cholmod_free_factor(&factor, &common);
    cholmod_free_dense(&b, &common);
    cholmod_free_sparse(&matrix, &common);
    cholmod_finish(&common);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[1], "project") != 0 && strcmp(argv[1], "baseline") != 0)) {
        fprintf(stderr, "usage: cholmod_pilot project|baseline <symmetric_4.mtx>\n");
        return 2;
    }
    void *symbol = dlsym(RTLD_DEFAULT, "cholmod_start");
    Dl_info info;
    if (symbol && dladdr(symbol, &info) && info.dli_fname) {
        printf("loaded_cholmod_path=%s\n", info.dli_fname);
    }
    SparseMatrix *a = NULL;
    sparse_err_t load_status = sparse_load_mm(&a, argv[2]);
    printf("side=%s\nload_status=%d\n", argv[1], (int)load_status);
    if (load_status != SPARSE_OK || !a || !check_input(a)) {
        fprintf(stderr, "fixture load or coefficient validation failed\n");
        sparse_free(a);
        return 2;
    }
    printf("matrix_nnz=%lld\n", (long long)sparse_nnz(a));
    int result = strcmp(argv[1], "project") == 0 ? run_project(a) : run_baseline(a);
    sparse_free(a);
    return result;
}
