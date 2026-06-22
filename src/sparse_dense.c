#include "sparse_dense.h"
#include "sparse_alloc_internal.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_matrix_internal.h"
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <dlfcn.h>
#endif

dense_matrix_t *dense_create(idx_t rows, idx_t cols) {
    size_t rows_size = 0;
    size_t cols_size = 0;
    if (rows < 0 || cols < 0)
        return NULL;

    dense_matrix_t *M = malloc(sizeof(dense_matrix_t));
    if (!M)
        return NULL;

    M->rows = rows;
    M->cols = cols;

    if (rows == 0 || cols == 0) {
        M->data = NULL;
        return M;
    }

    /* Shared helper path: validate rows*cols and allocate dense storage. */
    size_t n = 0;
    void *data = NULL;
    if (sparse_idx_to_size_checked(rows, &rows_size) ||
        sparse_idx_to_size_checked(cols, &cols_size) ||
        sparse_size_mul_overflow(rows_size, cols_size, &n) ||
        sparse_calloc_array(n, sizeof(double), &data) != SPARSE_OK) {
        free(M);
        return NULL;
    }
    M->data = data;

    return M;
}

void dense_free(dense_matrix_t *M) {
    if (!M)
        return;
    free(M->data);
    free(M);
}

sparse_err_t dense_gemm(const dense_matrix_t *A, const dense_matrix_t *B, dense_matrix_t *C) {
    if (!A || !B || !C)
        return SPARSE_ERR_NULL;
    if (A->cols != B->rows)
        return SPARSE_ERR_SHAPE;
    if (C->rows != A->rows || C->cols != B->cols)
        return SPARSE_ERR_SHAPE;

    idx_t m = A->rows;
    idx_t k = A->cols;
    idx_t n = B->cols;
    size_t m_size = 0;
    size_t k_size = 0;
    size_t n_size = 0;

    /* Zero-sized matrices: C = 0 (any zero dimension means empty product) */
    if (m == 0 || k == 0 || n == 0) {
        if (m > 0 && n > 0) {
            size_t mn = 0;
            size_t c_bytes = 0;
            if (!C->data)
                return SPARSE_ERR_NULL;
            if (sparse_idx_to_size_checked(m, &m_size) || sparse_idx_to_size_checked(n, &n_size) ||
                sparse_size_mul_overflow(m_size, n_size, &mn) ||
                sparse_count_bytes_overflow(mn, sizeof(double), &c_bytes))
                return SPARSE_ERR_ALLOC;
            memset(C->data, 0, c_bytes);
        }
        return SPARSE_OK;
    }

    if (!A->data || !B->data || !C->data)
        return SPARSE_ERR_NULL;

    if (sparse_idx_to_size_checked(m, &m_size) || sparse_idx_to_size_checked(k, &k_size) ||
        sparse_idx_to_size_checked(n, &n_size))
        return SPARSE_ERR_ALLOC;

    /* Overflow-safe byte count for C */
    size_t mn = 0;
    size_t c_bytes = 0;
    if (sparse_size_mul_overflow(m_size, n_size, &mn) ||
        sparse_count_bytes_overflow(mn, sizeof(double), &c_bytes))
        return SPARSE_ERR_ALLOC;

    /* Zero C */
    memset(C->data, 0, c_bytes);

    /* C(i,j) = sum_p A(i,p) * B(p,j)
     * Column-major: loop over j (output column), then p, then i for cache. */
    for (idx_t j = 0; j < n; j++) {
        for (idx_t p = 0; p < k; p++) {
            double b_pj = DENSE_AT(B, p, j);
            if (b_pj == 0.0)
                continue;
            for (idx_t i = 0; i < m; i++) {
                DENSE_AT(C, i, j) += DENSE_AT(A, i, p) * b_pj;
            }
        }
    }

    return SPARSE_OK;
}

sparse_err_t dense_gemv(const dense_matrix_t *A, const double *x, double *y) {
    if (!A || !x || !y)
        return SPARSE_ERR_NULL;

    idx_t m = A->rows;
    idx_t n = A->cols;
    size_t m_size = 0;

    if (m == 0)
        return SPARSE_OK;
    if (sparse_idx_to_size_checked(m, &m_size))
        return SPARSE_ERR_ALLOC;

    /* Overflow check for m * sizeof(double) */
    size_t y_bytes = 0;
    if (sparse_count_bytes_overflow(m_size, sizeof(double), &y_bytes))
        return SPARSE_ERR_ALLOC;

    if (n == 0) {
        /* A is m×0: y should be the zero vector */
        memset(y, 0, y_bytes);
        return SPARSE_OK;
    }

    if (!A->data)
        return SPARSE_ERR_NULL;

    /* y = 0 */
    memset(y, 0, y_bytes);

    /* y(i) = sum_j A(i,j) * x(j)
     * Column-major: loop over j (column), then i for cache. */
    for (idx_t j = 0; j < n; j++) {
        double xj = x[j];
        if (xj == 0.0)
            continue;
        for (idx_t i = 0; i < m; i++) {
            y[i] += DENSE_AT(A, i, j) * xj;
        }
    }

    return SPARSE_OK;
}

sparse_err_t chol_dense_factor(double *A, idx_t n, idx_t lda, double tol) {
    if (!A)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n)
        return SPARSE_ERR_BADARG;
    if (n == 0)
        return SPARSE_OK;

    /* Approximate reference norm from A's initial diagonal (before any
     * updates) for relative tolerance scaling.  Keeps the kernel
     * self-contained without forcing callers to pass ||A||_inf. */
    double ref_norm = 0.0;
    for (idx_t j = 0; j < n; j++) {
        double d = fabs(A[j + j * lda]);
        if (d > ref_norm)
            ref_norm = d;
    }
    double sing_tol = sparse_rel_tol(ref_norm, tol > 0.0 ? tol : SPARSE_DROP_TOL);

    for (idx_t k = 0; k < n; k++) {
        /* Diagonal accumulator: A[k,k] - sum_{j<k} L[k,j]^2. */
        double s = A[k + k * lda];
        for (idx_t j = 0; j < k; j++) {
            double l_kj = A[k + j * lda];
            s -= l_kj * l_kj;
        }
        if (s < sing_tol)
            return SPARSE_ERR_NOT_SPD;
        double l_kk = sqrt(s);
        A[k + k * lda] = l_kk;
        double inv_l_kk = 1.0 / l_kk;

        /* Below-diagonal column: L[i, k] = (A[i,k] - sum_{j<k} L[i,j]*L[k,j]) / L[k,k]. */
        for (idx_t i = k + 1; i < n; i++) {
            double t = A[i + k * lda];
            for (idx_t j = 0; j < k; j++)
                t -= A[i + j * lda] * A[k + j * lda];
            A[i + k * lda] = t * inv_l_kk;
        }
    }
    return SPARSE_OK;
}

sparse_err_t chol_dense_solve_lower(const double *L, idx_t n, idx_t lda, double *b) {
    if (!L || !b)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n)
        return SPARSE_ERR_BADARG;
    if (n == 0)
        return SPARSE_OK;

    /* Forward substitution: for each row i, b[i] -= L[i, j] * b[j] for
     * j < i, then b[i] /= L[i, i]. */
    for (idx_t i = 0; i < n; i++) {
        double sum = b[i];
        for (idx_t j = 0; j < i; j++)
            sum -= L[i + j * lda] * b[j];
        double l_ii = L[i + i * lda];
        if (l_ii == 0.0)
            return SPARSE_ERR_SINGULAR;
        b[i] = sum / l_ii;
    }
    return SPARSE_OK;
}

sparse_err_t chol_dense_solve_panel(const double *L, idx_t n, idx_t lda, double *panel, idx_t ldb,
                                    idx_t panel_rows) {
    if (!L)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n || panel_rows < 0)
        return SPARSE_ERR_BADARG;
    if (n == 0 || panel_rows == 0)
        return SPARSE_OK;
    if (!panel)
        return SPARSE_ERR_NULL;
    if (ldb < panel_rows)
        return SPARSE_ERR_BADARG;

    /* Solve all panel rows against the same L in place. Each panel column j
     * stores the j-th solve-dimension entry for every panel row, so forward
     * substitution can update the whole column strip before moving on. */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < i; j++) {
            double l_ij = L[i + j * lda];
            if (l_ij == 0.0)
                continue;
            for (idx_t r = 0; r < panel_rows; r++)
                panel[r + i * ldb] -= l_ij * panel[r + j * ldb];
        }
        double l_ii = L[i + i * lda];
        if (l_ii == 0.0)
            return SPARSE_ERR_SINGULAR;
        double inv_l_ii = 1.0 / l_ii;
        for (idx_t r = 0; r < panel_rows; r++)
            panel[r + i * ldb] *= inv_l_ii;
    }

    return SPARSE_OK;
}

static int s64_idx_to_blas_int_checked(idx_t value, int *out) {
    if (!out)
        return 0;
    if (value < 0)
        return 0;
#if SPARSE_IDX_BITS > 32
    if (value > (idx_t)INT_MAX)
        return 0;
#endif
    *out = (int)value;
    return 1;
}

#ifdef __APPLE__
typedef void (*s64_accel_dpotrf_fn)(const char *uplo, const int *n, double *a, const int *lda,
                                    int *info);
typedef void (*s64_accel_dtrsv_fn)(const char *uplo, const char *trans, const char *diag,
                                   const int *n, const double *a, const int *lda, double *x,
                                   const int *incx);
typedef void (*s64_accel_dtrsm_fn)(const char *side, const char *uplo, const char *transa,
                                   const char *diag, const int *m, const int *n,
                                   const double *alpha, const double *a, const int *lda, double *b,
                                   const int *ldb);

static void *s64_accel_handle = NULL;
static s64_accel_dpotrf_fn s64_accel_dpotrf = NULL;
static s64_accel_dtrsv_fn s64_accel_dtrsv = NULL;
static s64_accel_dtrsm_fn s64_accel_dtrsm = NULL;
static atomic_int s64_accel_probe_state = ATOMIC_VAR_INIT(0);

static int s64_accel_probe_dense_kernels(void) {
    enum {
        S64_ACCEL_UNINITIALIZED = 0,
        S64_ACCEL_INITIALIZING = 1,
        S64_ACCEL_READY = 2,
        S64_ACCEL_FAILED = 3,
    };

    int state = atomic_load_explicit(&s64_accel_probe_state, memory_order_acquire);
    if (state == S64_ACCEL_READY)
        return 1;
    if (state == S64_ACCEL_FAILED)
        return 0;

    int expected = S64_ACCEL_UNINITIALIZED;
    if (atomic_compare_exchange_strong_explicit(&s64_accel_probe_state, &expected,
                                                S64_ACCEL_INITIALIZING, memory_order_acq_rel,
                                                memory_order_acquire)) {
        void *handle = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate",
                              RTLD_LAZY | RTLD_LOCAL);
        s64_accel_dpotrf_fn dpotrf = NULL;
        s64_accel_dtrsv_fn dtrsv = NULL;
        s64_accel_dtrsm_fn dtrsm = NULL;

        if (handle) {
            dpotrf = (s64_accel_dpotrf_fn)dlsym(handle, "dpotrf_");
            dtrsv = (s64_accel_dtrsv_fn)dlsym(handle, "dtrsv_");
            dtrsm = (s64_accel_dtrsm_fn)dlsym(handle, "dtrsm_");
        }
        if (!handle || !dpotrf || !dtrsv || !dtrsm) {
            if (handle)
                dlclose(handle);
            s64_accel_handle = NULL;
            s64_accel_dpotrf = NULL;
            s64_accel_dtrsv = NULL;
            s64_accel_dtrsm = NULL;
            atomic_store_explicit(&s64_accel_probe_state, S64_ACCEL_FAILED, memory_order_release);
            return 0;
        }

        s64_accel_handle = handle;
        s64_accel_dpotrf = dpotrf;
        s64_accel_dtrsv = dtrsv;
        s64_accel_dtrsm = dtrsm;
        atomic_store_explicit(&s64_accel_probe_state, S64_ACCEL_READY, memory_order_release);
        return 1;
    }

    do {
        state = atomic_load_explicit(&s64_accel_probe_state, memory_order_acquire);
    } while (state == S64_ACCEL_INITIALIZING);

    return state == S64_ACCEL_READY;
}

static sparse_err_t s64_accelerate_chol_dense_factor(double *A, idx_t n, idx_t lda, double tol) {
    if (!A)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n)
        return SPARSE_ERR_BADARG;
    if (n == 0)
        return SPARSE_OK;
    if (!s64_accel_probe_dense_kernels())
        return SPARSE_ERR_BACKEND_CONTRACT;

    int n_blas = 0;
    int lda_blas = 0;
    if (!s64_idx_to_blas_int_checked(n, &n_blas) || !s64_idx_to_blas_int_checked(lda, &lda_blas))
        /* BLAS-int width overflow is an optional-backend contract limit, not OOM. */
        return SPARSE_ERR_BACKEND_CONTRACT;

    double ref_norm = 0.0;
    for (idx_t j = 0; j < n; j++) {
        double d = fabs(A[j + j * lda]);
        if (d > ref_norm)
            ref_norm = d;
    }
    double sing_tol = sparse_rel_tol(ref_norm, tol > 0.0 ? tol : SPARSE_DROP_TOL);
    double diag_tol = sqrt(sing_tol);

    const char uplo = 'L';
    int info = 0;
    s64_accel_dpotrf(&uplo, &n_blas, A, &lda_blas, &info);
    if (info < 0)
        return SPARSE_ERR_BACKEND_CONTRACT;
    if (info > 0)
        return SPARSE_ERR_NOT_SPD;

    for (idx_t j = 0; j < n; j++) {
        if (A[j + j * lda] < diag_tol)
            return SPARSE_ERR_NOT_SPD;
    }
    return SPARSE_OK;
}

static sparse_err_t s64_accelerate_chol_dense_solve_lower(const double *L, idx_t n, idx_t lda,
                                                          double *b) {
    if (!L || !b)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n)
        return SPARSE_ERR_BADARG;
    if (n == 0)
        return SPARSE_OK;
    if (!s64_accel_probe_dense_kernels())
        return SPARSE_ERR_BACKEND_CONTRACT;

    int n_blas = 0;
    int lda_blas = 0;
    if (!s64_idx_to_blas_int_checked(n, &n_blas) || !s64_idx_to_blas_int_checked(lda, &lda_blas))
        return SPARSE_ERR_BACKEND_CONTRACT;

    for (idx_t i = 0; i < n; i++) {
        if (L[i + i * lda] == 0.0)
            return SPARSE_ERR_SINGULAR;
    }

    const char uplo = 'L';
    const char trans = 'N';
    const char diag = 'N';
    const int incx = 1;
    s64_accel_dtrsv(&uplo, &trans, &diag, &n_blas, L, &lda_blas, b, &incx);
    return SPARSE_OK;
}

static sparse_err_t s64_accelerate_chol_dense_solve_panel(const double *L, idx_t n, idx_t lda,
                                                          double *panel, idx_t ldb,
                                                          idx_t panel_rows) {
    if (!L)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n || panel_rows < 0)
        return SPARSE_ERR_BADARG;
    if (n == 0 || panel_rows == 0)
        return SPARSE_OK;
    if (!panel)
        return SPARSE_ERR_NULL;
    if (ldb < panel_rows)
        return SPARSE_ERR_BADARG;
    if (!s64_accel_probe_dense_kernels())
        return SPARSE_ERR_BACKEND_CONTRACT;

    int n_blas = 0;
    int lda_blas = 0;
    int panel_rows_blas = 0;
    int ldb_blas = 0;
    if (!s64_idx_to_blas_int_checked(n, &n_blas) || !s64_idx_to_blas_int_checked(lda, &lda_blas) ||
        !s64_idx_to_blas_int_checked(panel_rows, &panel_rows_blas) ||
        !s64_idx_to_blas_int_checked(ldb, &ldb_blas))
        return SPARSE_ERR_BACKEND_CONTRACT;

    for (idx_t i = 0; i < n; i++) {
        if (L[i + i * lda] == 0.0)
            return SPARSE_ERR_SINGULAR;
    }

    const char side = 'R';
    const char uplo = 'L';
    const char transa = 'T';
    const char diag = 'N';
    const double alpha = 1.0;
    s64_accel_dtrsm(&side, &uplo, &transa, &diag, &panel_rows_blas, &n_blas, &alpha, L, &lda_blas,
                    panel, &ldb_blas);
    return SPARSE_OK;
}
#endif

typedef enum {
    S64_CHOL_DENSE_BACKEND_BUILTIN = 0,
    S64_CHOL_DENSE_BACKEND_ACCELERATE = 1,
} s64_chol_dense_backend_t;

static s64_chol_dense_backend_t s64_read_chol_dense_backend_env(void) {
    const char *value = getenv("SPARSE_CHOL_DENSE_BACKEND");
    if (!value || value[0] == '\0')
        return S64_CHOL_DENSE_BACKEND_BUILTIN;
    if (strcmp(value, "accelerate") == 0)
        return S64_CHOL_DENSE_BACKEND_ACCELERATE;
    return S64_CHOL_DENSE_BACKEND_BUILTIN;
}

static const chol_dense_kernels_t s64_builtin_chol_dense_kernels = {
    .name = "builtin",
    .factor = chol_dense_factor,
    .solve_lower = chol_dense_solve_lower,
    .solve_panel = chol_dense_solve_panel,
};

#ifdef __APPLE__
static const chol_dense_kernels_t s64_accelerate_chol_dense_kernels = {
    .name = "accelerate",
    .factor = s64_accelerate_chol_dense_factor,
    .solve_lower = s64_accelerate_chol_dense_solve_lower,
    .solve_panel = s64_accelerate_chol_dense_solve_panel,
};
#endif

static const chol_dense_kernels_t *s64_test_override_dense_kernels = NULL;
static int s64_test_override_dense_kernels_enabled = 0;

const chol_dense_kernels_t *chol_csc_supernodal_dense_kernels(void) {
    if (s64_test_override_dense_kernels_enabled)
        return s64_test_override_dense_kernels;
#ifdef __APPLE__
    if (s64_read_chol_dense_backend_env() == S64_CHOL_DENSE_BACKEND_ACCELERATE &&
        s64_accel_probe_dense_kernels())
        return &s64_accelerate_chol_dense_kernels;
#endif
    return &s64_builtin_chol_dense_kernels;
}

void chol_csc_supernodal_set_dense_kernels_override_for_test(const chol_dense_kernels_t *kernels) {
    s64_test_override_dense_kernels = kernels;
    s64_test_override_dense_kernels_enabled = 1;
}

void chol_csc_supernodal_clear_dense_kernels_override_for_test(void) {
    s64_test_override_dense_kernels = NULL;
    s64_test_override_dense_kernels_enabled = 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Givens rotations
 * ═══════════════════════════════════════════════════════════════════════ */

void givens_compute(double a, double b, double *c, double *s) {
    if (b == 0.0) {
        *c = 1.0;
        *s = 0.0;
    } else if (a == 0.0) {
        *c = 0.0;
        *s = (b > 0.0) ? 1.0 : -1.0;
    } else {
        double r = hypot(a, b);
        *c = a / r;
        *s = b / r;
    }
}

void givens_apply_left(double c, double s, double *x, double *y, idx_t n) {
    for (idx_t k = 0; k < n; k++) {
        double xk = x[k];
        double yk = y[k];
        x[k] = c * xk + s * yk;
        y[k] = -s * xk + c * yk;
    }
}

void givens_apply_right(double c, double s, double *x, double *y, idx_t n) {
    for (idx_t k = 0; k < n; k++) {
        double xk = x[k];
        double yk = y[k];
        x[k] = c * xk - s * yk;
        y[k] = s * xk + c * yk;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * 2×2 symmetric eigenvalue solver
 * ═══════════════════════════════════════════════════════════════════════ */

void eigen2x2(double a, double b, double d, double *lambda1, double *lambda2) {
    /* Eigenvalues of [[a, b], [b, d]]:
     * lambda = (a+d)/2 ± sqrt(((a-d)/2)^2 + b^2)
     * Use the numerically stable form to avoid catastrophic cancellation. */
    double trace = a + d;
    double half_diff = (a - d) * 0.5;
    double disc = sqrt(half_diff * half_diff + b * b);

    /* Return in ascending order */
    *lambda1 = trace * 0.5 - disc;
    *lambda2 = trace * 0.5 + disc;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Symmetric tridiagonal eigenvalue solver (implicit QR with Wilkinson shift)
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * One implicit QR step on the unreduced tridiagonal block diag[lo..hi],
 * subdiag[lo..hi-1] using a Wilkinson shift.
 */
static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db)
        return -1;
    if (da > db)
        return 1;
    return 0;
}

static void tridiag_qr_step(double *diag, double *subdiag, idx_t lo, idx_t hi) {
    /* Wilkinson shift: eigenvalue of trailing 2×2 closer to diag[hi] */
    double l1, l2;
    eigen2x2(diag[hi - 1], subdiag[hi - 1], diag[hi], &l1, &l2);
    double shift = (fabs(l1 - diag[hi]) < fabs(l2 - diag[hi])) ? l1 : l2;

    /* Initial bulge: Givens to zero subdiag[lo] in (T - shift*I) */
    double x = diag[lo] - shift;
    double z = subdiag[lo];
    double c, s;

    for (idx_t k = lo; k < hi; k++) {
        givens_compute(x, z, &c, &s);

        /* Apply Givens rotation G(k, k+1) from both sides to T.
         * This is the implicit symmetric tridiagonal QR step.
         * Only affects rows/cols k, k+1 (and neighbours). */

        if (k > lo) {
            subdiag[k - 1] = c * subdiag[k - 1] + s * z;
            /* The bulge at (k-1, k+1) is zeroed by this rotation */
        }

        double dk = diag[k];
        double dk1 = diag[k + 1];
        double ek = subdiag[k];

        /* Update 2×2 block [dk, ek; ek, dk1] under similarity transform */
        diag[k] = c * c * dk + 2.0 * c * s * ek + s * s * dk1;
        diag[k + 1] = s * s * dk - 2.0 * c * s * ek + c * c * dk1;
        subdiag[k] = c * s * (dk1 - dk) + (c * c - s * s) * ek;

        /* Prepare bulge for next iteration */
        if (k + 1 < hi) {
            z = s * subdiag[k + 1];
            subdiag[k + 1] = c * subdiag[k + 1];
            x = subdiag[k];
        }
    }
}

sparse_err_t tridiag_qr_eigenvalues(double *diag, double *subdiag, idx_t n, idx_t max_iter) {
    if (n <= 0)
        return SPARSE_OK;
    if (!diag)
        return SPARSE_ERR_NULL;
    if (n == 1)
        return SPARSE_OK;
    if (!subdiag)
        return SPARSE_ERR_NULL;

    if (max_iter <= 0) {
        int64_t default_iter = (int64_t)30 * (int64_t)n;
        max_iter = (default_iter > INT32_MAX) ? INT32_MAX : (idx_t)default_iter;
    }

    /* Deflation tolerance */
    double tol = 1e-14;

    idx_t total_iter = 0;
    idx_t hi = n - 1; /* top of active block */

    while (hi > 0 && total_iter < max_iter) {
        /* Check for deflation at the bottom */
        double off = fabs(subdiag[hi - 1]);
        double diag_sum = fabs(diag[hi - 1]) + fabs(diag[hi]);
        if (off <= tol * diag_sum || off < sparse_rel_tol(diag_sum, DROP_TOL)) {
            subdiag[hi - 1] = 0.0;
            hi--;
            continue;
        }

        /* Find the start of the unreduced block */
        idx_t lo = hi - 1;
        while (lo > 0) {
            double off_lo = fabs(subdiag[lo - 1]);
            double ds_lo = fabs(diag[lo - 1]) + fabs(diag[lo]);
            if (off_lo <= tol * ds_lo || off_lo < sparse_rel_tol(ds_lo, DROP_TOL)) {
                subdiag[lo - 1] = 0.0;
                break;
            }
            lo--;
        }

        /* One QR step on block [lo..hi] */
        tridiag_qr_step(diag, subdiag, lo, hi);
        total_iter++;
    }

    if (hi > 0)
        return SPARSE_ERR_NOT_CONVERGED;

    /* Sort eigenvalues in ascending order */
    qsort(diag, (size_t)n, sizeof(double), cmp_double_asc);

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Symmetric tridiagonal eigenpair solver
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Mirrors the tridiagonal QR above, additionally accumulating each
 * Givens rotation into an orthogonal matrix Q.  The rotation applied
 * to T is a similarity transform T_new = G^T · T · G where G (in the
 * 2×2 block acting on indices k, k+1) is [[c, -s], [s, c]].  After
 * every QR step converges, the accumulated Q satisfies
 * Q_total = G_1 · G_2 · … · G_last and T = Q_total · diag(lambda) ·
 * Q_total^T — column j of Q_total is the eigenvector for
 * eigenvalue diag[j].
 *
 * Accumulation rule: right-multiplying Q by G in columns k, k+1
 * updates every row (q_i,k, q_i,k+1) to (c·q_i,k + s·q_i,k+1,
 * -s·q_i,k + c·q_i,k+1).
 *
 * Permutation-on-sort: once eigenvalues converge we sort them
 * ascending and apply the same permutation to Q's columns so
 * column j remains the eigenvector for the new diag[j].  A scratch
 * `pair_t` array handles the indirect sort; n is typically the
 * Lanczos basis size (≤ a few hundred in practice), so this is
 * cheap.
 */

static void tridiag_qr_step_with_Q(double *diag, double *subdiag, idx_t lo, idx_t hi, double *Q,
                                   idx_t Q_rows) {
    /* Wilkinson shift: eigenvalue of trailing 2×2 closer to diag[hi] */
    double l1, l2;
    eigen2x2(diag[hi - 1], subdiag[hi - 1], diag[hi], &l1, &l2);
    double shift = (fabs(l1 - diag[hi]) < fabs(l2 - diag[hi])) ? l1 : l2;

    /* Initial bulge: Givens to zero subdiag[lo] in (T - shift*I) */
    double x = diag[lo] - shift;
    double z = subdiag[lo];
    double c, s;

    for (idx_t k = lo; k < hi; k++) {
        givens_compute(x, z, &c, &s);

        if (k > lo) {
            subdiag[k - 1] = c * subdiag[k - 1] + s * z;
        }

        double dk = diag[k];
        double dk1 = diag[k + 1];
        double ek = subdiag[k];

        diag[k] = c * c * dk + 2.0 * c * s * ek + s * s * dk1;
        diag[k + 1] = s * s * dk - 2.0 * c * s * ek + c * c * dk1;
        subdiag[k] = c * s * (dk1 - dk) + (c * c - s * s) * ek;

        /* Q := Q · G in cols (k, k+1) with G = [[c, -s], [s, c]]. */
        double *col_k = Q + (size_t)k * (size_t)Q_rows;
        double *col_k1 = Q + (size_t)(k + 1) * (size_t)Q_rows;
        for (idx_t i = 0; i < Q_rows; i++) {
            double a = col_k[i];
            double b = col_k1[i];
            col_k[i] = c * a + s * b;
            col_k1[i] = -s * a + c * b;
        }

        if (k + 1 < hi) {
            z = s * subdiag[k + 1];
            subdiag[k + 1] = c * subdiag[k + 1];
            x = subdiag[k];
        }
    }
}

typedef struct {
    double eigval;
    idx_t idx;
} tridiag_pair_t;

static int cmp_pair_asc(const void *a, const void *b) {
    double da = ((const tridiag_pair_t *)a)->eigval;
    double db = ((const tridiag_pair_t *)b)->eigval;
    if (da < db)
        return -1;
    if (da > db)
        return 1;
    return 0;
}

sparse_err_t tridiag_qr_eigenpairs(double *diag, double *subdiag, double *Q, idx_t n,
                                   idx_t max_iter) {
    if (n <= 0)
        return SPARSE_OK;
    if (!diag || !Q)
        return SPARSE_ERR_NULL;
    if (n >= 2 && !subdiag)
        return SPARSE_ERR_NULL;

    /* Overflow-check n*n before using it in byte-sized memset /
     * memcpy / malloc.  For very large n on a 32-bit size_t target
     * (or any target where n² overflows size_t) this prevents the
     * silent undersized buffer that would follow. */
    size_t n_size = 0;
    size_t n2 = 0;
    size_t n2_bytes = 0;
    if (sparse_idx_to_size_checked(n, &n_size) || sparse_size_mul_overflow(n_size, n_size, &n2) ||
        sparse_size_mul_overflow(n2, sizeof(double), &n2_bytes))
        return SPARSE_ERR_ALLOC;

    /* Initialise Q = I_n. */
    memset(Q, 0, n2_bytes);
    for (idx_t i = 0; i < n; i++)
        Q[(size_t)i * n_size + (size_t)i] = 1.0;

    if (n == 1)
        return SPARSE_OK;

    if (max_iter <= 0) {
        int64_t default_iter = (int64_t)30 * (int64_t)n;
        max_iter = (default_iter > INT32_MAX) ? INT32_MAX : (idx_t)default_iter;
    }

    double tol = 1e-14;
    idx_t total_iter = 0;
    idx_t hi = n - 1;

    while (hi > 0 && total_iter < max_iter) {
        double off = fabs(subdiag[hi - 1]);
        double diag_sum = fabs(diag[hi - 1]) + fabs(diag[hi]);
        if (off <= tol * diag_sum || off < sparse_rel_tol(diag_sum, DROP_TOL)) {
            subdiag[hi - 1] = 0.0;
            hi--;
            continue;
        }

        idx_t lo = hi - 1;
        while (lo > 0) {
            double off_lo = fabs(subdiag[lo - 1]);
            double ds_lo = fabs(diag[lo - 1]) + fabs(diag[lo]);
            if (off_lo <= tol * ds_lo || off_lo < sparse_rel_tol(ds_lo, DROP_TOL)) {
                subdiag[lo - 1] = 0.0;
                break;
            }
            lo--;
        }

        tridiag_qr_step_with_Q(diag, subdiag, lo, hi, Q, n);
        total_iter++;
    }

    if (hi > 0)
        return SPARSE_ERR_NOT_CONVERGED;

    /* Sort eigenvalues ascending and permute Q's columns to match.
     * Indirect sort through a (eigval, orig-index) pair array.
     * `n2_bytes` was overflow-validated above. */
    tridiag_pair_t *pairs = NULL;
    double *Q_sorted = NULL;
    double *diag_sorted = NULL;
    if (sparse_malloc_idx_array(n, sizeof(tridiag_pair_t), (void **)&pairs) != SPARSE_OK ||
        sparse_malloc_array(n2, sizeof(double), (void **)&Q_sorted) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(double), (void **)&diag_sorted) != SPARSE_OK) {
        free(pairs);
        free(Q_sorted);
        free(diag_sorted);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < n; i++) {
        pairs[i].eigval = diag[i];
        pairs[i].idx = i;
    }
    qsort(pairs, (size_t)n, sizeof(tridiag_pair_t), cmp_pair_asc);
    for (idx_t i = 0; i < n; i++) {
        diag_sorted[i] = pairs[i].eigval;
        memcpy(Q_sorted + (size_t)i * (size_t)n, Q + (size_t)pairs[i].idx * (size_t)n,
               (size_t)n * sizeof(double));
    }
    memcpy(diag, diag_sorted, (size_t)n * sizeof(double));
    memcpy(Q, Q_sorted, n2_bytes);
    free(pairs);
    free(Q_sorted);
    free(diag_sorted);

    return SPARSE_OK;
}
