/**
 * @file sparse_ldlt_dense.c
 * @brief Dense LDL^T block factor and backend selection for CSC LDL^T.
 *
 * This file owns the dense Bunch-Kaufman primitive used by the LDL^T CSC
 * supernodal path, plus the bounded optional runtime-selected external
 * BLAS/LAPACK-class backend.
 */

#include "sparse_alloc_internal.h"
#include "sparse_ldlt_csc_internal.h"

#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <dlfcn.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Dense LDL^T primitive (Bunch-Kaufman)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * `ldlt_dense_factor` runs Bunch-Kaufman LDL^T on a column-major
 * n×n dense buffer.  Mirrors the sparse reference in
 * `src/sparse_ldlt.c` column-by-column but on dense storage, so the
 * batched supernodal path can call it per supernode diagonal block.
 *
 * Input convention: `A` is n×n column-major symmetric with BOTH
 * triangles populated — `A[i + j*lda] == A[j + i*lda]` at call time.
 * Supernodal callers should scatter the diagonal block that way
 * before calling.  The upper triangle gets overwritten as scratch
 * during symmetric swaps and trailing updates; not preserved post-factor.
 */

/* Symmetric row + column swap of a dense column-major n×n matrix.
 * Maintains symmetry by swapping both rows AND columns in sync. */
static void ldlt_dense_sym_swap(double *A, idx_t n, idx_t lda, idx_t a, idx_t b) {
    if (a == b)
        return;
    for (idx_t c = 0; c < n; c++) {
        double tmp = A[a + c * lda];
        A[a + c * lda] = A[b + c * lda];
        A[b + c * lda] = tmp;
    }
    for (idx_t r = 0; r < n; r++) {
        double tmp = A[r + a * lda];
        A[r + a * lda] = A[r + b * lda];
        A[r + b * lda] = tmp;
    }
}

static int s64_ldlt_idx_to_blas_int_checked(idx_t value, int *out) {
    if (!out || value < 0)
        return 0;
#if SPARSE_IDX_BITS > 32
    if (value > (idx_t)INT_MAX)
        return 0;
#endif
    *out = (int)value;
    return 1;
}

static int s64_ldlt_accel_accepts_noperm_2x2_pivot(const int *ipiv, idx_t k, idx_t n) {
    int pivot_tag = 0;

    if (!ipiv || k < 0 || n < 2 || k >= n - 1)
        return 0;
    if (!s64_ldlt_idx_to_blas_int_checked(k + 2, &pivot_tag))
        return 0;
    /* For lower-storage `dsytrf`, a no-interchange 2x2 block at zero-based
     * indices `(k, k+1)` is encoded as `ipiv[k] == ipiv[k+1] == -(k+2)`.
     * Any other negative pattern implies a row/column interchange that this
     * bounded Accelerate path does not currently reconstruct safely. */
    return ipiv[k] < 0 && ipiv[k + 1] == ipiv[k] && ipiv[k] == -pivot_tag;
}

#ifndef _WIN32
typedef void (*s64_accel_dsytrf_fn)(const char *uplo, const int *n, double *a, const int *lda,
                                    int *ipiv, double *work, const int *lwork, int *info);

typedef enum {
    S64_LDLT_EXT_PROVIDER_NONE = 0,
    S64_LDLT_EXT_PROVIDER_ACCELERATE = 1,
    S64_LDLT_EXT_PROVIDER_BLAS_LAPACK = 2,
} s64_ldlt_ext_provider_t;

static void *s64_ldlt_ext_handle = NULL;
static s64_accel_dsytrf_fn s64_ldlt_ext_dsytrf = NULL;
static s64_ldlt_ext_provider_t s64_ldlt_ext_provider = S64_LDLT_EXT_PROVIDER_NONE;
static atomic_int s64_ldlt_ext_probe_state = ATOMIC_VAR_INIT(0);

/* Copy the raw dlsym result into a typed function pointer without a direct
 * object-pointer-to-function-pointer cast, which trips -Wpedantic on Linux. */
static void s64_ldlt_store_symbol(void *symbol, void *fn_out, size_t fn_size) {
    if (!fn_out || fn_size == 0)
        return;
    memset(fn_out, 0, fn_size);
    if (!symbol || fn_size != sizeof(symbol))
        return;
    memcpy(fn_out, &symbol, sizeof(symbol));
}

static int s64_ldlt_ext_probe_dense_factor(void) {
    enum {
        S64_LDLT_ACCEL_UNINITIALIZED = 0,
        S64_LDLT_ACCEL_INITIALIZING = 1,
        S64_LDLT_ACCEL_READY = 2,
        S64_LDLT_ACCEL_FAILED = 3,
    };

    int state = atomic_load_explicit(&s64_ldlt_ext_probe_state, memory_order_acquire);
    if (state == S64_LDLT_ACCEL_READY)
        return 1;
    if (state == S64_LDLT_ACCEL_FAILED)
        return 0;

    int expected = S64_LDLT_ACCEL_UNINITIALIZED;
    if (atomic_compare_exchange_strong_explicit(&s64_ldlt_ext_probe_state, &expected,
                                                S64_LDLT_ACCEL_INITIALIZING, memory_order_acq_rel,
                                                memory_order_acquire)) {
        static const struct {
            const char *path;
            s64_ldlt_ext_provider_t provider;
        } candidates[] = {
#ifdef __APPLE__
            {"/System/Library/Frameworks/Accelerate.framework/Accelerate",
             S64_LDLT_EXT_PROVIDER_ACCELERATE},
            {"/opt/homebrew/opt/openblas/lib/libopenblas.dylib", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
            {"/usr/local/opt/openblas/lib/libopenblas.dylib", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
            {"libopenblas.dylib", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
            {"libblas.dylib", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
            {"liblapack.dylib", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
#else
            {"libopenblas.so", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
            {"libopenblas.so.0", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
            {"libblas.so.3", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
            {"libblas.so", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
            {"liblapack.so.3", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
            {"liblapack.so", S64_LDLT_EXT_PROVIDER_BLAS_LAPACK},
#endif
        };

        for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
            void *handle = dlopen(candidates[i].path, RTLD_LAZY | RTLD_LOCAL);
            s64_accel_dsytrf_fn dsytrf = NULL;

            if (handle)
                s64_ldlt_store_symbol(dlsym(handle, "dsytrf_"), &dsytrf, sizeof(dsytrf));
            if (!handle || !dsytrf) {
                if (handle)
                    dlclose(handle);
                continue;
            }

            s64_ldlt_ext_handle = handle;
            s64_ldlt_ext_dsytrf = dsytrf;
            s64_ldlt_ext_provider = candidates[i].provider;
            atomic_store_explicit(&s64_ldlt_ext_probe_state, S64_LDLT_ACCEL_READY,
                                  memory_order_release);
            return 1;
        }

        s64_ldlt_ext_handle = NULL;
        s64_ldlt_ext_dsytrf = NULL;
        s64_ldlt_ext_provider = S64_LDLT_EXT_PROVIDER_NONE;
        atomic_store_explicit(&s64_ldlt_ext_probe_state, S64_LDLT_ACCEL_FAILED,
                              memory_order_release);
        return 0;
    }

    do {
        state = atomic_load_explicit(&s64_ldlt_ext_probe_state, memory_order_acquire);
    } while (state == S64_LDLT_ACCEL_INITIALIZING);

    return state == S64_LDLT_ACCEL_READY;
}

static sparse_err_t s64_external_ldlt_dense_factor(double *A, double *D, double *D_offdiag,
                                                   idx_t *pivot_size, idx_t n, idx_t lda,
                                                   double tol, double *elem_growth_out) {
    if (!A || !D || !D_offdiag || !pivot_size)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n)
        return SPARSE_ERR_BADARG;
    if (elem_growth_out)
        *elem_growth_out = 0.0;
    if (n == 0)
        return SPARSE_OK;
    if (!s64_ldlt_ext_probe_dense_factor())
        return SPARSE_ERR_PIVOT_REJECTED;

    int n_blas = 0;
    int lda_blas = 0;
    if (!s64_ldlt_idx_to_blas_int_checked(n, &n_blas) ||
        !s64_ldlt_idx_to_blas_int_checked(lda, &lda_blas))
        return SPARSE_ERR_PIVOT_REJECTED;

    double ref_norm = 0.0;
    for (idx_t j = 0; j < n; j++) {
        double d = fabs(A[j + j * lda]);
        if (d > ref_norm)
            ref_norm = d;
    }
    double eff_tol = tol > 0.0 ? tol : SPARSE_DROP_TOL;
    double sing_tol = sparse_rel_tol(ref_norm, eff_tol);
    double growth_bound = 1.0 / (100.0 * eff_tol);
    double max_growth = 0.0;

    int *ipiv = NULL;
    sparse_err_t alloc_err = sparse_malloc_idx_array(n, sizeof(int), (void **)&ipiv);
    if (alloc_err != SPARSE_OK)
        return alloc_err;

    char uplo = 'L';
    int lwork = -1;
    int info = 0;
    double work_query = 0.0;
    s64_ldlt_ext_dsytrf(&uplo, &n_blas, A, &lda_blas, ipiv, &work_query, &lwork, &info);
    if (info != 0) {
        free(ipiv);
        return info < 0 ? SPARSE_ERR_PIVOT_REJECTED : SPARSE_ERR_SINGULAR;
    }

    if (!(work_query >= 1.0) || work_query > (double)INT_MAX) {
        free(ipiv);
        return SPARSE_ERR_PIVOT_REJECTED;
    }
    int lwork_int = (int)work_query;
    double *work = NULL;
    alloc_err = sparse_malloc_array((size_t)lwork_int, sizeof(double), (void **)&work);
    if (alloc_err != SPARSE_OK) {
        free(ipiv);
        return alloc_err;
    }

    s64_ldlt_ext_dsytrf(&uplo, &n_blas, A, &lda_blas, ipiv, work, &lwork_int, &info);
    free(work);
    if (info != 0) {
        free(ipiv);
        return info < 0 ? SPARSE_ERR_PIVOT_REJECTED : SPARSE_ERR_SINGULAR;
    }

    sparse_err_t err = SPARSE_OK;
    idx_t k = 0;
    while (k < n) {
        if (ipiv[k] > 0) {
            if (ipiv[k] != k + 1) {
                err = SPARSE_ERR_PIVOT_REJECTED;
                goto cleanup;
            }

            double dk = A[k + k * lda];
            if (fabs(dk) < sing_tol) {
                err = SPARSE_ERR_SINGULAR;
                goto cleanup;
            }
            D[k] = dk;
            D_offdiag[k] = 0.0;
            pivot_size[k] = 1;

            for (idx_t i = k + 1; i < n; i++) {
                double l_ik = A[i + k * lda];
                if (fabs(l_ik) > growth_bound) {
                    err = SPARSE_ERR_SINGULAR;
                    goto cleanup;
                }
                if (fabs(l_ik) > max_growth)
                    max_growth = fabs(l_ik);
                A[k + i * lda] = l_ik;
            }
            A[k + k * lda] = 1.0;
            k += 1;
            continue;
        }

        if (!s64_ldlt_accel_accepts_noperm_2x2_pivot(ipiv, k, n)) {
            err = SPARSE_ERR_PIVOT_REJECTED;
            goto cleanup;
        }

        double d11 = A[k + k * lda];
        double d21 = A[(k + 1) + k * lda];
        double d22 = A[(k + 1) + (k + 1) * lda];
        double det = d11 * d22 - d21 * d21;
        double bscale = fabs(d11) + fabs(d22) + fabs(d21);
        double det_tol = (bscale > 0.0) ? eff_tol * bscale * bscale : sing_tol * sing_tol;
        if (fabs(det) < det_tol) {
            err = SPARSE_ERR_SINGULAR;
            goto cleanup;
        }

        D[k] = d11;
        D[k + 1] = d22;
        D_offdiag[k] = d21;
        D_offdiag[k + 1] = 0.0;
        pivot_size[k] = 2;
        pivot_size[k + 1] = 2;

        for (idx_t i = k + 2; i < n; i++) {
            double l_ik = A[i + k * lda];
            double l_ik1 = A[i + (k + 1) * lda];
            if (fabs(l_ik) > growth_bound || fabs(l_ik1) > growth_bound) {
                err = SPARSE_ERR_SINGULAR;
                goto cleanup;
            }
            if (fabs(l_ik) > max_growth)
                max_growth = fabs(l_ik);
            if (fabs(l_ik1) > max_growth)
                max_growth = fabs(l_ik1);
            A[k + i * lda] = l_ik;
            A[(k + 1) + i * lda] = l_ik1;
        }

        A[k + k * lda] = 1.0;
        A[(k + 1) + k * lda] = 0.0;
        A[k + (k + 1) * lda] = 0.0;
        A[(k + 1) + (k + 1) * lda] = 1.0;
        k += 2;
    }

    if (elem_growth_out)
        *elem_growth_out = max_growth;

cleanup:
    free(ipiv);
    return err;
}
#endif

typedef enum {
    S64_LDLT_DENSE_BACKEND_BUILTIN = 0,
    S64_LDLT_DENSE_BACKEND_EXTERNAL = 1,
    S64_LDLT_DENSE_BACKEND_ACCELERATE = 2,
} s64_ldlt_dense_backend_t;

static s64_ldlt_dense_backend_t s64_read_ldlt_dense_backend_env(void) {
    const char *value = getenv("SPARSE_LDLT_DENSE_BACKEND");
    if (!value || strcmp(value, "builtin") == 0)
        return S64_LDLT_DENSE_BACKEND_BUILTIN;
    if (strcmp(value, "external") == 0 || strcmp(value, "blas") == 0 ||
        strcmp(value, "lapack") == 0)
        return S64_LDLT_DENSE_BACKEND_EXTERNAL;
#ifdef __APPLE__
    if (strcmp(value, "accelerate") == 0)
        return S64_LDLT_DENSE_BACKEND_ACCELERATE;
#endif
    return S64_LDLT_DENSE_BACKEND_BUILTIN;
}

const char *ldlt_dense_factor_backend_name(void) {
#ifndef _WIN32
    s64_ldlt_dense_backend_t backend = s64_read_ldlt_dense_backend_env();
    if (backend == S64_LDLT_DENSE_BACKEND_EXTERNAL && s64_ldlt_ext_probe_dense_factor()) {
        if (s64_ldlt_ext_provider == S64_LDLT_EXT_PROVIDER_ACCELERATE)
            return "accelerate";
        if (s64_ldlt_ext_provider == S64_LDLT_EXT_PROVIDER_BLAS_LAPACK)
            return "blas-lapack";
    }
#ifdef __APPLE__
    if (backend == S64_LDLT_DENSE_BACKEND_ACCELERATE && s64_ldlt_ext_probe_dense_factor() &&
        s64_ldlt_ext_provider == S64_LDLT_EXT_PROVIDER_ACCELERATE)
        return "accelerate";
#endif
#endif
    return "builtin";
}

sparse_err_t ldlt_dense_factor(double *A, double *D, double *D_offdiag, idx_t *pivot_size, idx_t n,
                               idx_t lda, double tol, double *elem_growth_out) {
    if (!A || !D || !D_offdiag || !pivot_size)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n)
        return SPARSE_ERR_BADARG;
    if (elem_growth_out)
        *elem_growth_out = 0.0;
    if (n == 0)
        return SPARSE_OK;

    /* Relative tolerance tied to the initial diagonal max — matches
     * `chol_dense_factor`'s self-contained convention. */
    double ref_norm = 0.0;
    for (idx_t j = 0; j < n; j++) {
        double d = fabs(A[j + j * lda]);
        if (d > ref_norm)
            ref_norm = d;
    }
    double eff_tol = tol > 0.0 ? tol : SPARSE_DROP_TOL;
    double sing_tol = sparse_rel_tol(ref_norm, eff_tol);
    double growth_bound = 1.0 / (100.0 * eff_tol);
    double alpha_bk = (1.0 + sqrt(17.0)) / 8.0; /* ≈ 0.6404 */
    double max_growth = 0.0;
    sparse_err_t err = SPARSE_OK;

    /* Single per-call scratch buffer for 2×2-pivot L multipliers
     * (`l_col_k` || `l_col_k1`).  Sized for the worst-case tail
     * (k = 0 → tail_len = n - 2, two columns), then reused across
     * every 2×2 pivot in this call to avoid per-pivot malloc/free
     * churn in the hot supernode kernel.  When n <= 2 the 2×2 path's
     * tail_len is always 0, so no scratch is needed. */
    double *pivot_scratch = NULL;
    if (n > 2) {
        size_t tail_len_max = 0;
        size_t pivot_scratch_count = 0;
        if (sparse_idx_to_size_checked(n - 2, &tail_len_max) ||
            sparse_size_mul_overflow(tail_len_max, 2, &pivot_scratch_count)) {
            return SPARSE_ERR_ALLOC;
        }
        err = sparse_malloc_array(pivot_scratch_count, sizeof(double), (void **)&pivot_scratch);
        if (err != SPARSE_OK)
            return err;
    }

    idx_t k = 0;
    while (k < n) {
        /* ── Pivot selection: four-criteria Bunch-Kaufman ──────────────── */
        double lambda = 0.0;
        idx_t r = k;
        for (idx_t i = k + 1; i < n; i++) {
            double v = fabs(A[i + k * lda]);
            if (v > lambda) {
                lambda = v;
                r = i;
            }
        }

        int use_2x2 = 0;
        if (lambda == 0.0 || fabs(A[k + k * lda]) >= alpha_bk * lambda) {
            use_2x2 = 0;
        } else {
            double sigma_r = 0.0;
            for (idx_t i = k; i < n; i++) {
                if (i == r)
                    continue;
                double v = fabs(A[i + r * lda]);
                if (v > sigma_r)
                    sigma_r = v;
            }
            if (fabs(A[k + k * lda]) * sigma_r >= alpha_bk * lambda * lambda) {
                use_2x2 = 0;
            } else if (fabs(A[r + r * lda]) >= alpha_bk * sigma_r) {
                ldlt_dense_sym_swap(A, n, lda, k, r);
                use_2x2 = 0;
            } else {
                if (r != k + 1)
                    ldlt_dense_sym_swap(A, n, lda, k + 1, r);
                use_2x2 = 1;
            }
        }

        if (!use_2x2) {
            double dk = A[k + k * lda];
            if (fabs(dk) < sing_tol) {
                err = SPARSE_ERR_SINGULAR;
                goto cleanup;
            }

            D[k] = dk;
            D_offdiag[k] = 0.0;
            pivot_size[k] = 1;

            double inv_dk = 1.0 / dk;
            for (idx_t j = k + 1; j < n; j++) {
                double ajk = A[j + k * lda];
                double factor = ajk * inv_dk;
                for (idx_t i = j; i < n; i++) {
                    double update = A[i + k * lda] * factor;
                    A[i + j * lda] -= update;
                    if (i != j)
                        A[j + i * lda] = A[i + j * lda];
                }
            }

            for (idx_t i = k + 1; i < n; i++) {
                double l_ik = A[i + k * lda] * inv_dk;
                if (fabs(l_ik) > growth_bound) {
                    err = SPARSE_ERR_SINGULAR;
                    goto cleanup;
                }
                if (fabs(l_ik) > max_growth)
                    max_growth = fabs(l_ik);
                A[i + k * lda] = l_ik;
                A[k + i * lda] = l_ik;
            }
            A[k + k * lda] = 1.0;

            k += 1;
        } else {
            double d11 = A[k + k * lda];
            double d21 = A[(k + 1) + k * lda];
            double d22 = A[(k + 1) + (k + 1) * lda];
            double det = d11 * d22 - d21 * d21;
            double bscale = fabs(d11) + fabs(d22) + fabs(d21);
            double det_tol = (bscale > 0.0) ? eff_tol * bscale * bscale : sing_tol * sing_tol;
            if (fabs(det) < det_tol) {
                err = SPARSE_ERR_SINGULAR;
                goto cleanup;
            }

            D[k] = d11;
            D[k + 1] = d22;
            D_offdiag[k] = d21;
            D_offdiag[k + 1] = 0.0;
            pivot_size[k] = 2;
            pivot_size[k + 1] = 2;

            double inv_det = 1.0 / det;
            idx_t tail_len = n - (k + 2);
            if (tail_len > 0) {
                double *l_col_k = pivot_scratch;
                double *l_col_k1 = pivot_scratch + tail_len;
                for (idx_t i = k + 2; i < n; i++) {
                    double aik = A[i + k * lda];
                    double aik1 = A[i + (k + 1) * lda];
                    double l_ik = (aik * d22 - aik1 * d21) * inv_det;
                    double l_ik1 = (-aik * d21 + aik1 * d11) * inv_det;
                    if (fabs(l_ik) > growth_bound || fabs(l_ik1) > growth_bound) {
                        err = SPARSE_ERR_SINGULAR;
                        goto cleanup;
                    }
                    if (fabs(l_ik) > max_growth)
                        max_growth = fabs(l_ik);
                    if (fabs(l_ik1) > max_growth)
                        max_growth = fabs(l_ik1);
                    l_col_k[i - (k + 2)] = l_ik;
                    l_col_k1[i - (k + 2)] = l_ik1;
                }

                for (idx_t j = k + 2; j < n; j++) {
                    double ajk = A[j + k * lda];
                    double ajk1 = A[j + (k + 1) * lda];
                    for (idx_t i = j; i < n; i++) {
                        idx_t ti = i - (k + 2);
                        double update = l_col_k[ti] * ajk + l_col_k1[ti] * ajk1;
                        A[i + j * lda] -= update;
                        if (i != j)
                            A[j + i * lda] = A[i + j * lda];
                    }
                }

                for (idx_t i = k + 2; i < n; i++) {
                    idx_t ti = i - (k + 2);
                    double l_ik = l_col_k[ti];
                    double l_ik1 = l_col_k1[ti];
                    A[i + k * lda] = l_ik;
                    A[i + (k + 1) * lda] = l_ik1;
                    A[k + i * lda] = l_ik;
                    A[(k + 1) + i * lda] = l_ik1;
                }
            }

            A[k + k * lda] = 1.0;
            A[(k + 1) + k * lda] = 0.0;
            A[k + (k + 1) * lda] = 0.0;
            A[(k + 1) + (k + 1) * lda] = 1.0;

            k += 2;
        }
    }

    if (elem_growth_out)
        *elem_growth_out = max_growth;

cleanup:
    free(pivot_scratch);
    return err;
}

sparse_err_t ldlt_dense_factor_selected(double *A, double *D, double *D_offdiag, idx_t *pivot_size,
                                        idx_t n, idx_t lda, double tol, double *elem_growth_out) {
#ifndef _WIN32
    s64_ldlt_dense_backend_t backend = s64_read_ldlt_dense_backend_env();
    if (backend == S64_LDLT_DENSE_BACKEND_EXTERNAL && s64_ldlt_ext_probe_dense_factor())
        return s64_external_ldlt_dense_factor(A, D, D_offdiag, pivot_size, n, lda, tol,
                                              elem_growth_out);
#ifdef __APPLE__
    if (backend == S64_LDLT_DENSE_BACKEND_ACCELERATE && s64_ldlt_ext_probe_dense_factor() &&
        s64_ldlt_ext_provider == S64_LDLT_EXT_PROVIDER_ACCELERATE)
        return s64_external_ldlt_dense_factor(A, D, D_offdiag, pivot_size, n, lda, tol,
                                              elem_growth_out);
#endif
#endif
    return ldlt_dense_factor(A, D, D_offdiag, pivot_size, n, lda, tol, elem_growth_out);
}
