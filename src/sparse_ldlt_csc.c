/**
 * @file sparse_ldlt_csc.c
 * @brief CSC working-format numeric backend for symmetric indefinite
 *        LDL^T factorization.
 *
 * This backend runs Bunch-Kaufman directly on CSC storage. The factor keeps
 * its unit lower-triangular L in the shared `CholCsc` layout and carries the
 * block-diagonal D state in the companion `D`, `D_offdiag`, `pivot_size`,
 * and `perm` arrays.
 *
 * The file owns:
 *
 * - the native CSC LDL^T elimination kernel
 * - the linked-list compatibility wrapper path used by tests and A/B benches
 * - row-adjacency support for sparse cmod updates
 * - the scalar solve path
 * - the dense LDL^T primitive and bounded backend-selection seam used by the
 *   supernodal path
 * - top-level orchestration for the batched supernodal LDL^T path
 *
 * The batched supernodal path mirrors the Cholesky CSC structure but adds one
 * LDL^T-specific rule: a 2x2 pivot block is atomic and cannot be split across
 * a supernode boundary. The detection and elimination helpers that implement
 * that rule live in the paired internal contract and the extracted
 * `src/sparse_ldlt_csc_supernodal.c` file.
 *
 * For indefinite matrices, the analysis-aware conversion path matters for the
 * same reason it does on the Cholesky side: the batched path needs the full
 * structural pattern available up front so panel writeback does not silently
 * drop valid fill rows. Callers that cannot provide that pattern should stay
 * on the scalar CSC path.
 */

#include "sparse_alloc_internal.h"
#include "sparse_ldlt.h"
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
 * Sprint 19 Day 11: Dense LDL^T primitive (Bunch-Kaufman)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * `ldlt_dense_factor` runs Bunch-Kaufman LDL^T on a column-major
 * n×n dense buffer.  Mirrors the sparse reference in
 * `src/sparse_ldlt.c` column-by-column but on dense storage, so the
 * Sprint 19 Days 12-14 batched supernodal path can call it per
 * supernode's diagonal block.
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

    if ((size_t)n > SIZE_MAX / sizeof(int))
        return SPARSE_ERR_ALLOC;
    int *ipiv = malloc((size_t)n * sizeof(int));
    if (!ipiv)
        return SPARSE_ERR_ALLOC;

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
    if ((size_t)lwork_int > SIZE_MAX / sizeof(double)) {
        free(ipiv);
        return SPARSE_ERR_ALLOC;
    }
    double *work = malloc((size_t)lwork_int * sizeof(double));
    if (!work) {
        free(ipiv);
        return SPARSE_ERR_ALLOC;
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
            if (ipiv[k] != (int)(k + 1)) {
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
        size_t pivot_scratch_count = 0;
        if (sparse_idx_count_bytes_overflow(n - 2, sizeof(double), &pivot_scratch_count) ||
            sparse_size_mul_overflow(pivot_scratch_count, 2, &pivot_scratch_count)) {
            return SPARSE_ERR_ALLOC;
        }
        pivot_scratch = malloc(pivot_scratch_count);
        if (!pivot_scratch)
            return SPARSE_ERR_ALLOC;
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

/* ─── Free ───────────────────────────────────────────────────────────── */

void ldlt_csc_free(LdltCsc *m) {
    if (!m)
        return;
    chol_csc_free(m->L);
    free(m->D);
    free(m->D_offdiag);
    free(m->pivot_size);
    free(m->perm);
    /* Release per-row adjacency lists.  Each entry in `row_adj` is
     * NULL until a column gets appended to that row.
     * `m->row_adj` is alloc'd with length `max(m->n, 1)` by
     * `ldlt_csc_alloc`, so iterating `[0, m->n)` on a non-NULL
     * `m->row_adj` is always in bounds — clang-analyzer can't see
     * that invariant across the alloc/free boundary. */
    if (m->row_adj) {
        for (idx_t r = 0; r < m->n; r++)
            free(m->row_adj[r]); // NOLINT(clang-analyzer-security.ArrayBound)
        free(m->row_adj);
    }
    free(m->row_adj_count);
    free(m->row_adj_cap);
    free(m);
}

/* ─── Allocate ───────────────────────────────────────────────────────── */

sparse_err_t ldlt_csc_alloc(idx_t n, idx_t initial_nnz, LdltCsc **out) {
    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    if (n < 0)
        return SPARSE_ERR_BADARG;

    /* Overflow guards for byte counts (n known non-negative above). */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if ((size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    LdltCsc *m = calloc(1, sizeof(LdltCsc));
    if (!m)
        return SPARSE_ERR_ALLOC;

    m->n = n;
    m->factor_norm = 0.0;

    sparse_err_t err = chol_csc_alloc(n, initial_nnz, &m->L);
    if (err != SPARSE_OK) {
        ldlt_csc_free(m);
        return err;
    }

    /* Always allocate at least one slot so array pointers are non-NULL
     * even for n == 0 — keeps invariant checks simple. */
    size_t alloc_n = n > 0 ? (size_t)n : 1;
    m->D = calloc(alloc_n, sizeof(double));
    m->D_offdiag = calloc(alloc_n, sizeof(double));
    m->pivot_size = calloc(alloc_n, sizeof(idx_t));
    m->perm = calloc(alloc_n, sizeof(idx_t));
    /* Row-adjacency index starts with all rows empty.
     * Per-row arrays allocated lazily by `ldlt_csc_row_adj_append` on
     * first append to each row.  `calloc` zeros all three so every
     * row_adj[r] slot is NULL and every count/cap is 0 until written. */
    m->row_adj = calloc(alloc_n, sizeof(idx_t *));
    m->row_adj_count = calloc(alloc_n, sizeof(idx_t));
    m->row_adj_cap = calloc(alloc_n, sizeof(idx_t));
    if (!m->D || !m->D_offdiag || !m->pivot_size || !m->perm || !m->row_adj || !m->row_adj_count ||
        !m->row_adj_cap) {
        ldlt_csc_free(m);
        return SPARSE_ERR_ALLOC;
    }

    /* Defaults: every step is a 1x1 pivot; perm is the identity.  Both
     * are overwritten during elimination. */
    for (idx_t i = 0; i < n; i++) {
        m->pivot_size[i] = 1;
        m->perm[i] = i;
    }

    *out = m;
    return SPARSE_OK;
}

/* ─── Row-adjacency append ───────────────────────────────────────── */

sparse_err_t ldlt_csc_row_adj_append(LdltCsc *F, idx_t row, idx_t col) {
    if (!F)
        return SPARSE_ERR_NULL;
    if (row < 0 || row >= F->n || col < 0 || col >= F->n)
        return SPARSE_ERR_BADARG;

    idx_t cap = F->row_adj_cap[row];
    idx_t count = F->row_adj_count[row];
    if (count >= cap) {
        /* Geometric growth (2×), starting at 4 for first-touch rows so
         * short row-adjacency lists don't pay a per-append reallocation
         * when the fill pattern is modest. */
        idx_t new_cap = 4;
        if (cap > 0) {
            if (cap > IDX_MAX / 2)
                return SPARSE_ERR_ALLOC;
            new_cap = cap * 2;
        }
        if ((size_t)new_cap > SIZE_MAX / sizeof(idx_t))
            return SPARSE_ERR_ALLOC;
        idx_t *resized = realloc(F->row_adj[row], (size_t)new_cap * sizeof(idx_t));
        if (!resized)
            return SPARSE_ERR_ALLOC;
        F->row_adj[row] = resized;
        F->row_adj_cap[row] = new_cap;
    }
    F->row_adj[row][count] = col;
    F->row_adj_count[row] = count + 1;
    return SPARSE_OK;
}

/* ─── 2×2-aware supernode detection ──────────────────────────────── */

/* Return 1 iff columns `prev` and `prev + 1` belong to the same
 * fundamental supernode of `L` (Liu-Ng-Peyton three-condition check,
 * same test as `chol_csc.c`'s static helper; duplicated here to keep
 * the LDL^T side loosely coupled). */
static int ldlt_csc_same_supernode(const CholCsc *L, idx_t prev) {
    idx_t curr = prev + 1;
    idx_t prev_start = L->col_ptr[prev];
    idx_t prev_end = L->col_ptr[prev + 1];
    idx_t curr_start = L->col_ptr[curr];
    idx_t curr_end = L->col_ptr[curr + 1];

    idx_t prev_size = prev_end - prev_start;
    idx_t curr_size = curr_end - curr_start;

    if (prev_size < 2)
        return 0;
    if (L->row_idx[prev_start + 1] != curr)
        return 0;
    if (curr_size != prev_size - 1)
        return 0;

    idx_t tail_len = curr_size - 1;
    for (idx_t t = 0; t < tail_len; t++) {
        if (L->row_idx[curr_start + 1 + t] != L->row_idx[prev_start + 2 + t])
            return 0;
    }
    return 1;
}

/* A column `k` is "the first of a 2×2 pair" when `pivot_size[k] == 2`
 * and `k == 0 || pivot_size[k-1] == 1`.  The second of the pair
 * always follows immediately at `k + 1`. */
static int ldlt_csc_is_first_of_2x2(const idx_t *pivot_size, idx_t k) {
    if (pivot_size[k] != 2)
        return 0;
    if (k == 0)
        return 1;
    return pivot_size[k - 1] == 1;
}

/* A column `k` is "the second of a 2×2 pair" when both `pivot_size[k]`
 * and `pivot_size[k-1]` are 2 AND `pivot_size[k-1]` itself is the
 * first of a 2×2 (recursively — but the pairing is always adjacent,
 * so this simplifies to checking the window [k-2, k-1, k]).  For our
 * forward scan we call this only when k >= 1. */
static int ldlt_csc_is_second_of_2x2(const idx_t *pivot_size, idx_t k) {
    if (k < 1)
        return 0;
    if (pivot_size[k] != 2 || pivot_size[k - 1] != 2)
        return 0;
    /* pivot_size[k-2] == 1 confirms the 2×2 starts at k-1, not k-2.
     * k == 1 implies k-1 == 0 which is by definition the first of a
     * 2×2 pair, so k is the second. */
    if (k == 1)
        return 1;
    return pivot_size[k - 2] == 1;
}

sparse_err_t ldlt_csc_detect_supernodes(const LdltCsc *F, idx_t min_size, idx_t *super_starts,
                                        idx_t *super_sizes, idx_t *count) {
    if (!F || !F->L || !F->pivot_size || !super_starts || !super_sizes || !count)
        return SPARSE_ERR_NULL;
    if (min_size < 1)
        return SPARSE_ERR_BADARG;

    idx_t n = F->n;
    const CholCsc *L = F->L;
    const idx_t *pivot_size = F->pivot_size;
    idx_t written = 0;
    idx_t j = 0;

    while (j < n) {
        /* Skip a column if it's the second of a 2×2 pair — the prior
         * iteration already decided what to do with the pair, and we
         * must not start a supernode mid-pair (see the Day 10 design
         * block for atomicity). */
        if (ldlt_csc_is_second_of_2x2(pivot_size, j)) {
            j++;
            continue;
        }

        /* Extend the supernode as long as the Liu-Ng-Peyton pattern
         * check allows. */
        idx_t end = j + 1;
        while (end < n && ldlt_csc_same_supernode(L, end - 1))
            end++;

        /* 2×2 atomicity at the upper boundary: `end-1` must not be
         * the first of a 2×2 pair.  If it is, try to extend by 1 to
         * include the pair's second column; if the pattern blocks
         * the extension, retract `end-- ` so the 2×2 stays outside
         * the supernode (scalar handles it). */
        if (end - 1 >= j && ldlt_csc_is_first_of_2x2(pivot_size, end - 1)) {
            if (end < n && ldlt_csc_same_supernode(L, end - 1)) {
                end++;
            } else {
                end--;
            }
        }

        idx_t size = end - j;
        if (size >= min_size) {
            super_starts[written] = j;
            super_sizes[written] = size;
            written++;
        }

        /* Ensure progress even when the atomicity retraction produced
         * an empty supernode (end <= j): advance `j` by 1 so scalar
         * handles this column and we keep scanning. */
        if (end <= j) {
            j = j + 1;
        } else {
            j = end;
        }
    }

    *count = written;
    return SPARSE_OK;
}

/* ─── Convert SparseMatrix → LdltCsc ─────────────────────────────────── */

sparse_err_t ldlt_csc_from_sparse(const SparseMatrix *mat, const idx_t *perm_in, double fill_factor,
                                  LdltCsc **ldlt_out) {
    if (!ldlt_out)
        return SPARSE_ERR_NULL;
    *ldlt_out = NULL;
    if (!mat)
        return SPARSE_ERR_NULL;
    if (mat->rows != mat->cols)
        return SPARSE_ERR_SHAPE;

    /* Reject factored / non-identity-perm matrices up front — the same
     * precondition `sparse_ldlt_factor` enforces.  Without this, the
     * `sparse_is_symmetric` check below walks physical storage while
     * `chol_csc_from_sparse` later walks logical storage via
     * inv_row_perm / inv_col_perm, so the two views could disagree on
     * a matrix with a non-identity internal permutation. */
    idx_t n = mat->rows;
    if (mat->factored)
        return SPARSE_ERR_BADARG;
    {
        const idx_t *rp = sparse_row_perm(mat);
        const idx_t *cp = sparse_col_perm(mat);
        for (idx_t i = 0; i < n; i++) {
            if ((rp && rp[i] != i) || (cp && cp[i] != i))
                return SPARSE_ERR_BADARG;
        }
    }

    /* LDL^T requires a symmetric input; reject non-symmetric A the same
     * way the linked-list `sparse_ldlt_factor` does, with the shared
     * SPARSE_ERR_NOT_SPD code (that enum also covers "not symmetric"
     * per sparse_ldlt.h's documented contract). */
    if (!sparse_is_symmetric(mat, 1e-12))
        return SPARSE_ERR_NOT_SPD;

    /* Build L via the Cholesky CSC converter; this validates perm_in and
     * caches L->factor_norm.  Errors propagate unchanged. */
    CholCsc *L = NULL;
    sparse_err_t err = chol_csc_from_sparse(mat, perm_in, fill_factor, &L);
    if (err != SPARSE_OK)
        return err;

    LdltCsc *m = calloc(1, sizeof(LdltCsc));
    if (!m) {
        chol_csc_free(L);
        return SPARSE_ERR_ALLOC;
    }
    m->n = n;
    m->L = L;
    m->factor_norm = L->factor_norm;

    size_t alloc_n = n > 0 ? (size_t)n : 1;
    m->D = calloc(alloc_n, sizeof(double));
    m->D_offdiag = calloc(alloc_n, sizeof(double));
    m->pivot_size = calloc(alloc_n, sizeof(idx_t));
    m->perm = calloc(alloc_n, sizeof(idx_t));
    /* Sprint 19 Day 8: row-adjacency index — same calloc-zero initial
     * state as `ldlt_csc_alloc` so Day 9's cmod_unified can read
     * row_adj_count[col] == 0 and short-circuit before any column has
     * been factored. */
    m->row_adj = calloc(alloc_n, sizeof(idx_t *));
    m->row_adj_count = calloc(alloc_n, sizeof(idx_t));
    m->row_adj_cap = calloc(alloc_n, sizeof(idx_t));
    if (!m->D || !m->D_offdiag || !m->pivot_size || !m->perm || !m->row_adj || !m->row_adj_count ||
        !m->row_adj_cap) {
        ldlt_csc_free(m);
        return SPARSE_ERR_ALLOC;
    }

    /* Initial pivot_size is 1 everywhere; elimination overwrites. */
    for (idx_t i = 0; i < n; i++)
        m->pivot_size[i] = 1;

    /* Initial perm: the caller-supplied fill-reducing permutation if any,
     * else identity.  Bunch-Kaufman pivoting (Day 8) composes further
     * swaps into this array. */
    if (perm_in) {
        for (idx_t i = 0; i < n; i++)
            m->perm[i] = perm_in[i];
    } else {
        for (idx_t i = 0; i < n; i++)
            m->perm[i] = i;
    }

    *ldlt_out = m;
    return SPARSE_OK;
}

/* ─── Public: symbolic-analysis-aware LDL^T conversion ───────────────── */

/* Sprint 20 Days 1-2: design + implementation.
 *
 * `ldlt_csc_from_sparse_with_analysis` mirrors
 * `chol_csc_from_sparse_with_analysis` (Sprint 18 Day 12 / Sprint 19
 * Day 6) for the LDL^T side.  It pre-allocates the embedded `L` with
 * every column's full sym_L pattern rather than the heuristic
 * `fill_factor × A.nnz` pattern that `ldlt_csc_from_sparse` produces.
 * This closes the indefinite-fill hole documented in the Sprint 19
 * NOTE in `tests/test_direct_csc_regression.c` (search for
 * "NOTE on the indefinite supernodal path's current scope"): the
 * batched `ldlt_csc_eliminate_supernodal` writeback silently dropped
 * cmod fill rows on KKT-style saddle points, producing residuals of
 * 1e-2..1e-6 instead of round-off.
 *
 * Symbolic-pattern reuse:
 *   `sparse_analyze` treats `SPARSE_FACTOR_CHOLESKY` and
 *   `SPARSE_FACTOR_LDLT` identically for the symbolic pipeline (see
 *   the shared `case SPARSE_FACTOR_CHOLESKY: case SPARSE_FACTOR_LDLT:`
 *   dispatch in `src/sparse_analysis.c`): both run
 *   `sparse_etree_compute` → `sparse_colcount` →
 *   `sparse_symbolic_cholesky` on the symmetric input and produce
 *   identical `sym_L` patterns.  This function therefore accepts
 *   either type without extra per-column buffering.
 *
 * 2×2-pivot handling — "Option D" in `SPRINT_20/PLAN.md`:
 *   Bunch-Kaufman symmetric swaps during elimination CAN introduce
 *   rows not present in `sym_L(A)` because sym_L is pattern-dependent
 *   and symmetric swaps permute the pattern.  The alternative
 *   approaches considered during Day 1 were:
 *     Option A: reuse `sym_L(A)` as-is, accept that BK 2×2 fill can
 *               overflow.  Incorrect on indefinite inputs — defeats
 *               the purpose of the shim.
 *     Option B: run a dedicated LDL^T symbolic pass that accounts
 *               for potential 2×2 pivot fill.  Requires bespoke
 *               infrastructure; high cost for a niche correctness
 *               case.
 *     Option C: use sym_L(A) + per-column 2× over-allocation to
 *               cover BK 2×2 fill.  Bounded but wasteful on SPD
 *               inputs where no 2×2 pivots occur.
 *     Option D: handle 2×2 pivot fill at the workflow level, not
 *               per-column.  SPD inputs (all 1×1 pivots, no swaps)
 *               use sym_L(A) directly; indefinite inputs run a
 *               scalar pre-pass to resolve BK swaps, symmetrically
 *               permute A by the resulting perm, and run
 *               `sparse_analyze` on the pre-permuted matrix.  After
 *               pre-permutation BK cannot swap again during the
 *               batched factor, so sym_L on the pre-permuted matrix
 *               is complete without over-allocation.
 *   Option D selected: the transparent dispatch added in Sprint 20
 *   Days 4-6 wraps the pre-pass workflow behind
 *   `sparse_ldlt_factor_opts` so public-API callers never see it,
 *   while batched-test helpers (e.g. `s19_supernodal_matches_scalar`)
 *   already use the same two-pass structure.
 *
 * Implementation (Day 2):
 *   1. Delegate the L layout + A-scatter to
 *      `chol_csc_from_sparse_with_analysis` — this sets
 *      `L->sym_L_preallocated = 1` for free and re-uses the
 *      bsearch-into-row-range scatter loop that the Cholesky path
 *      already validated on the Sprint 18 corpus.
 *   2. Wrap the returned `CholCsc *L` in an `LdltCsc` with D /
 *      D_offdiag / pivot_size / perm / row_adj allocated in the
 *      same zero-initialised shape as `ldlt_csc_from_sparse` (the
 *      calloc + identity-perm initialisation below mirrors lines
 *      484-524 of `ldlt_csc_from_sparse`).
 *   3. `pivot_size[i] = 1` default; `perm` copied from
 *      `analysis->perm` when present, else identity — matches the
 *      `ldlt_csc_from_sparse` convention when the caller supplied
 *      a fill-reducing perm.  Bunch-Kaufman pivoting composes
 *      further swaps into this array during eliminate.
 *
 * Day 3 wires the resulting `L->sym_L_preallocated == 1` state into
 * `ldlt_csc_eliminate_supernodal`'s writeback fast-path so the
 * indefinite KKT fixture drops from 1e-2..1e-6 residual to
 * round-off. */
sparse_err_t ldlt_csc_from_sparse_with_analysis(const SparseMatrix *mat,
                                                const sparse_analysis_t *analysis,
                                                LdltCsc **ldlt_out) {
    if (!ldlt_out)
        return SPARSE_ERR_NULL;
    *ldlt_out = NULL;
    if (!mat || !analysis)
        return SPARSE_ERR_NULL;
    if (analysis->type != SPARSE_FACTOR_CHOLESKY && analysis->type != SPARSE_FACTOR_LDLT)
        return SPARSE_ERR_BADARG;
    if (mat->rows != mat->cols)
        return SPARSE_ERR_SHAPE;
    if (mat->rows != analysis->n)
        return SPARSE_ERR_SHAPE;

    /* LDL^T requires a symmetric input; reject non-symmetric `mat`
     * with the same SPARSE_ERR_NOT_SPD code the scalar
     * `ldlt_csc_from_sparse` entry point uses (mirrors the
     * documented contract in the function docstring above). */
    if (!sparse_is_symmetric(mat, 1e-12))
        return SPARSE_ERR_NOT_SPD;

    /* Delegate L layout + sym_L pre-allocation + A-scatter to the
     * Cholesky converter.  Sets `L->sym_L_preallocated = 1` and
     * caches `L->factor_norm` from `analysis->analysis_norm`. */
    CholCsc *L = NULL;
    sparse_err_t err = chol_csc_from_sparse_with_analysis(mat, analysis, &L);
    if (err != SPARSE_OK)
        return err;

    /* Wrap L in an LdltCsc with D / D_offdiag / pivot_size / perm /
     * row_adj zero-initialised, matching `ldlt_csc_from_sparse`. */
    idx_t n = mat->rows;
    LdltCsc *m = calloc(1, sizeof(LdltCsc));
    if (!m) {
        chol_csc_free(L);
        return SPARSE_ERR_ALLOC;
    }
    m->n = n;
    m->L = L;
    m->factor_norm = L->factor_norm;

    size_t alloc_n = n > 0 ? (size_t)n : 1;
    m->D = calloc(alloc_n, sizeof(double));
    m->D_offdiag = calloc(alloc_n, sizeof(double));
    m->pivot_size = calloc(alloc_n, sizeof(idx_t));
    m->perm = calloc(alloc_n, sizeof(idx_t));
    m->row_adj = calloc(alloc_n, sizeof(idx_t *));
    m->row_adj_count = calloc(alloc_n, sizeof(idx_t));
    m->row_adj_cap = calloc(alloc_n, sizeof(idx_t));
    if (!m->D || !m->D_offdiag || !m->pivot_size || !m->perm || !m->row_adj || !m->row_adj_count ||
        !m->row_adj_cap) {
        ldlt_csc_free(m);
        return SPARSE_ERR_ALLOC;
    }

    /* Initial pivot_size is 1 everywhere; elimination overwrites. */
    for (idx_t i = 0; i < n; i++)
        m->pivot_size[i] = 1;

    /* Initial perm: caller-supplied fill-reducing permutation from
     * the analysis, else identity.  Bunch-Kaufman pivoting composes
     * further swaps into this array during eliminate. */
    if (analysis->perm) {
        for (idx_t i = 0; i < n; i++)
            m->perm[i] = analysis->perm[i];
    } else {
        for (idx_t i = 0; i < n; i++)
            m->perm[i] = i;
    }

    *ldlt_out = m;
    return SPARSE_OK;
}

/* ─── Convert LdltCsc → SparseMatrix (L lower triangle only) ─────────── */

sparse_err_t ldlt_csc_to_sparse(const LdltCsc *ldlt, const idx_t *perm_out,
                                SparseMatrix **mat_out) {
    if (!ldlt)
        return SPARSE_ERR_NULL;
    return chol_csc_to_sparse(ldlt->L, perm_out, mat_out);
}

/* ─── Writeback CSC factor → public sparse_ldlt_t ─────────────────────── */

static sparse_err_t ldlt_csc_writeback_build_public_l(const LdltCsc *F, SparseMatrix **L_out) {
    if (!F || !L_out)
        return SPARSE_ERR_NULL;
    *L_out = NULL;

    if (F->n == 0)
        return SPARSE_OK;

    SparseMatrix *L_sparse = sparse_create(F->n, F->n);
    if (!L_sparse)
        return SPARSE_ERR_ALLOC;

    const CholCsc *L = F->L;
    for (idx_t j = 0; j < F->n; j++) {
        idx_t cstart = L->col_ptr[j];
        idx_t cend = L->col_ptr[j + 1];
        if (cstart == cend)
            continue;
        double abs_l_jj = (L->row_idx[cstart] == j) ? fabs(L->values[cstart]) : 0.0;
        double threshold = SPARSE_DROP_TOL * abs_l_jj;
        for (idx_t p = cstart; p < cend; p++) {
            idx_t i = L->row_idx[p];
            double v = L->values[p];
            if (v == 0.0)
                continue;
            if (i != j && fabs(v) < threshold)
                continue;
            sparse_err_t err = sparse_insert(L_sparse, i, j, v);
            if (err != SPARSE_OK) {
                sparse_free(L_sparse);
                return err;
            }
        }
    }

    *L_out = L_sparse;
    return SPARSE_OK;
}

static sparse_err_t ldlt_csc_writeback_copy_public_aux(const LdltCsc *F, double tol,
                                                       SparseMatrix *L_out,
                                                       sparse_ldlt_t *ldlt_out) {
    if (!F || !ldlt_out)
        return SPARSE_ERR_NULL;

    size_t alloc_n = F->n > 0 ? (size_t)F->n : 1;
    double *D = calloc(alloc_n, sizeof(double));
    double *D_off = calloc(alloc_n, sizeof(double));
    int *ps = calloc(alloc_n, sizeof(int));
    idx_t *perm = calloc(alloc_n, sizeof(idx_t));
    if (!D || !D_off || !ps || !perm) {
        free(D);
        free(D_off);
        free(ps);
        free(perm);
        sparse_free(L_out);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < F->n; i++) {
        D[i] = F->D[i];
        D_off[i] = F->D_offdiag[i];
        ps[i] = (int)F->pivot_size[i];
        perm[i] = F->perm[i];
    }

    ldlt_out->L = L_out;
    ldlt_out->D = D;
    ldlt_out->D_offdiag = D_off;
    ldlt_out->pivot_size = ps;
    ldlt_out->perm = perm;
    ldlt_out->n = F->n;
    ldlt_out->factor_norm = F->factor_norm;
    ldlt_out->tol = tol;
    return SPARSE_OK;
}

/* Transplant a factored LdltCsc into the `sparse_ldlt_t` shape the
 * public API documents.  Mirrors
 * `chol_csc_writeback_to_sparse` on the Cholesky side except that
 * the output is a separately-allocated result struct (not an
 * overwrite of the input matrix) — matching the LDL^T API's
 * separation between input `A` and output `ldlt->L`. */
sparse_err_t ldlt_csc_writeback_to_ldlt(const LdltCsc *F, double tol, sparse_ldlt_t *ldlt_out) {
    if (!F || !ldlt_out)
        return SPARSE_ERR_NULL;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    if (F->n < 0)
        return SPARSE_ERR_BADARG;

    /* Build the L SparseMatrix column-by-column, mirroring
     * `chol_csc_writeback_to_sparse`'s filter: skip exact zeros
     * (common when the CSC was pre-populated via
     * `ldlt_csc_from_sparse_with_analysis` and some sym_L fill
     * positions never received a non-zero value) and drop below-
     * diagonal entries below `SPARSE_DROP_TOL * |L[j, j]|` so the
     * transplanted `SparseMatrix` sparsity matches what the
     * linked-list kernel publishes.  The diagonal (row_idx[col_ptr[j]]
     * == j by CSC invariant) is inserted whenever its stored value is
     * non-zero, which covers every factor produced by this backend
     * because LDL^T stores a unit-diagonal L (`L[j, j] == 1.0`; see
     * the `unit diagonal` references in `sparse_ldlt_csc.c:86` and
     * the elimination kernels).  The `v == 0.0` filter below
     * therefore never drops the diagonal in practice, and the `i != j`
     * guard on the below-diagonal threshold keeps the unit diagonal
     * from being accidentally filtered by the drop_tol test. */
    SparseMatrix *L_out = NULL;
    sparse_err_t err = ldlt_csc_writeback_build_public_l(F, &L_out);
    if (err != SPARSE_OK)
        return err;

    /* Allocate and copy the auxiliary arrays.  Use alloc_n = max(1, n)
     * so n == 0 is still a valid writeback producing non-NULL
     * pointers (matches ldlt_csc_from_sparse's convention). */
    return ldlt_csc_writeback_copy_public_aux(F, tol, L_out, ldlt_out);
}

/* ─── Invariant checker ─────────────────────────────────────────────── */

sparse_err_t ldlt_csc_validate(const LdltCsc *ldlt) {
    if (!ldlt)
        return SPARSE_ERR_NULL;
    sparse_err_t err = chol_csc_validate(ldlt->L);
    if (err != SPARSE_OK)
        return err;
    if (ldlt->n < 0)
        return SPARSE_ERR_BADARG;
    if (ldlt->n > 0) {
        if (!ldlt->D || !ldlt->D_offdiag || !ldlt->pivot_size || !ldlt->perm)
            return SPARSE_ERR_BADARG;
    }

    /* pivot_size must be 1 or 2, and 2x2 pivots must cover consecutive
     * indices (pivot_size[i] = pivot_size[i+1] = 2). */
    for (idx_t i = 0; i < ldlt->n; i++) {
        idx_t s = ldlt->pivot_size[i];
        if (s != 1 && s != 2)
            return SPARSE_ERR_BADARG;
        if (s == 2) {
            if (i + 1 >= ldlt->n)
                return SPARSE_ERR_BADARG;
            if (ldlt->pivot_size[i + 1] != 2)
                return SPARSE_ERR_BADARG;
            /* Skip the second index of the 2x2 pair — it's been checked. */
            i++;
        }
    }

    /* perm must be a permutation of [0, n): every index appears exactly
     * once.  Use a small bit vector to detect duplicates / out-of-range. */
    if (ldlt->n == 0)
        return SPARSE_OK;
    char *seen = calloc((size_t)ldlt->n, sizeof(char));
    if (!seen)
        return SPARSE_ERR_ALLOC;
    for (idx_t i = 0; i < ldlt->n; i++) {
        idx_t p = ldlt->perm[i];
        if (p < 0 || p >= ldlt->n || seen[p]) {
            free(seen);
            return SPARSE_ERR_BADARG;
        }
        seen[p] = 1;
    }
    free(seen);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Linked-list Bunch-Kaufman elimination path
 * ═══════════════════════════════════════════════════════════════════════ */

/* Expand the lower-triangle CSC into a full symmetric SparseMatrix —
 * every off-diagonal entry is mirrored across the diagonal so that
 * sparse_ldlt_factor's symmetry check passes. */
static sparse_err_t csc_to_full_symmetric_matrix(const CholCsc *csc, SparseMatrix **out) {
    if (!csc || !out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    idx_t n = csc->n;
    if (n <= 0)
        return SPARSE_ERR_BADARG;

    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return SPARSE_ERR_ALLOC;

    for (idx_t j = 0; j < n; j++) {
        for (idx_t p = csc->col_ptr[j]; p < csc->col_ptr[j + 1]; p++) {
            idx_t i = csc->row_idx[p];
            double v = csc->values[p];
            sparse_err_t ierr = sparse_insert(A, i, j, v);
            if (ierr != SPARSE_OK) {
                sparse_free(A);
                return ierr;
            }
            if (i != j) {
                ierr = sparse_insert(A, j, i, v);
                if (ierr != SPARSE_OK) {
                    sparse_free(A);
                    return ierr;
                }
            }
        }
    }

    *out = A;
    return SPARSE_OK;
}

static sparse_err_t ldlt_csc_wrapper_validate_input(const LdltCsc *F, idx_t *l_nnz_out) {
    if (!F || !l_nnz_out)
        return SPARSE_ERR_NULL;
    *l_nnz_out = 0;

    idx_t n = F->n;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    if (F->L->n != n || !F->L->col_ptr)
        return SPARSE_ERR_BADARG;
    if (F->L->col_ptr[0] != 0)
        return SPARSE_ERR_BADARG;

    idx_t l_nnz = F->L->col_ptr[n];
    if (l_nnz < 0)
        return SPARSE_ERR_BADARG;
    for (idx_t j = 0; j < n; j++) {
        idx_t col_start = F->L->col_ptr[j];
        idx_t col_end = F->L->col_ptr[j + 1];
        if (col_start < 0 || col_end < 0 || col_start > col_end || col_start > l_nnz ||
            col_end > l_nnz)
            return SPARSE_ERR_BADARG;
    }
    if (l_nnz > 0 && (!F->L->row_idx || !F->L->values))
        return SPARSE_ERR_NULL;
    for (idx_t p = 0; p < l_nnz; p++) {
        if (F->L->row_idx[p] < 0 || F->L->row_idx[p] >= n)
            return SPARSE_ERR_BADARG;
    }

    if ((size_t)n > SIZE_MAX / sizeof(unsigned char))
        return SPARSE_ERR_ALLOC;
    unsigned char *seen = calloc((size_t)n, sizeof(unsigned char));
    if (!seen)
        return SPARSE_ERR_ALLOC;
    for (idx_t j = 0; j < n; j++) {
        idx_t pj = F->perm[j];
        if (pj < 0 || pj >= n || seen[pj]) {
            free(seen);
            return SPARSE_ERR_BADARG;
        }
        seen[pj] = 1;
    }
    free(seen);

    *l_nnz_out = l_nnz;
    return SPARSE_OK;
}

static sparse_err_t ldlt_csc_wrapper_copy_input_perm(const LdltCsc *F, idx_t **perm_in_out) {
    if (!F || !perm_in_out)
        return SPARSE_ERR_NULL;
    *perm_in_out = NULL;

    idx_t n = F->n;
    if ((size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;
    size_t perm_bytes = (size_t)n * sizeof(idx_t);
    idx_t *perm_in = malloc(perm_bytes);
    if (!perm_in)
        return SPARSE_ERR_ALLOC;
    memcpy(perm_in, F->perm, perm_bytes);
    *perm_in_out = perm_in;
    return SPARSE_OK;
}

static void ldlt_csc_wrapper_publish_factor_payload(LdltCsc *F, const sparse_ldlt_t *ll,
                                                    const idx_t *perm_in) {
    for (idx_t k = 0; k < F->n; k++) {
        F->D[k] = ll->D[k];
        F->D_offdiag[k] = ll->D_offdiag[k];
        F->pivot_size[k] = (idx_t)ll->pivot_size[k];
    }

    if (ll->perm) {
        for (idx_t k = 0; k < F->n; k++)
            F->perm[k] = perm_in[ll->perm[k]];
    } else {
        for (idx_t k = 0; k < F->n; k++)
            F->perm[k] = perm_in[k];
    }

    F->factor_norm = ll->factor_norm;
}

static sparse_err_t ldlt_csc_wrapper_rebuild_csc_factor(LdltCsc *F, const sparse_ldlt_t *ll) {
    if (!F || !ll)
        return SPARSE_ERR_NULL;

    CholCsc *new_L = NULL;
    sparse_err_t err = chol_csc_from_sparse(ll->L, NULL, 1.0, &new_L);
    if (err != SPARSE_OK)
        return err;
    chol_csc_free(F->L);
    F->L = new_L;
    return SPARSE_OK;
}

sparse_err_t ldlt_csc_eliminate_wrapper(LdltCsc *F) {
    if (!F)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (n <= 0)
        return SPARSE_OK;

    /* Validate only the invariants elimination actually needs.  The
     * full `ldlt_csc_validate(F)` call is too strict here because it
     * delegates to `chol_csc_validate(F->L)`, which rejects non-empty
     * CSC columns that don't start with an explicit diagonal entry —
     * but the linked-list `sparse_ldlt_factor` legitimately accepts
     * A's with a structurally-missing diagonal (treats them as zero
     * and either forms a 2x2 BK pivot or returns SPARSE_ERR_SINGULAR
     * later).  So we keep the safety checks that prevent crashes on
     * partially-initialised inputs but drop the diagonal-first
     * requirement and the sorted/distinct-row-index requirement that
     * full validate imposes. */
    idx_t l_nnz = 0;
    sparse_err_t err = ldlt_csc_wrapper_validate_input(F, &l_nnz);
    if (err != SPARSE_OK)
        return err;

    /* Save the pre-elimination perm (fill-reducing) so we can compose it
     * with the Bunch-Kaufman perm chosen during factorization.  Guard
     * `n * sizeof(idx_t)` against size_t overflow on 32-bit platforms
     * (or absurdly large n) before computing the byte count. */
    idx_t *perm_in = NULL;
    err = ldlt_csc_wrapper_copy_input_perm(F, &perm_in);
    if (err != SPARSE_OK)
        return err;

    /* Expand F->L's stored lower triangle to a full symmetric matrix so
     * the linked-list factor's symmetry check passes. */
    SparseMatrix *A_work = NULL;
    err = csc_to_full_symmetric_matrix(F->L, &A_work);
    if (err != SPARSE_OK) {
        free(perm_in);
        return err;
    }

    /* Run the linked-list LDL^T factorization on the expanded matrix. */
    sparse_ldlt_t ll = {0};
    err = sparse_ldlt_factor(A_work, &ll);
    sparse_free(A_work);
    if (err != SPARSE_OK) {
        free(perm_in);
        return err;
    }

    ldlt_csc_wrapper_publish_factor_payload(F, &ll, perm_in);

    /* `sparse_ldlt_factor` initialises ll.L with a full identity
     * diagonal (`sparse_insert(L, i, i, 1.0)` for every i) before the
     * Bunch-Kaufman sweep, so the CSC conversion below can rely on
     * every column already containing its unit diagonal — no extra
     * injection loop is needed here. */

    /* Replace F->L with a CSC built from ll.L.  The linked-list factor
     * is already complete — no further fill-in will be introduced — so
     * allocate the CSC at exact capacity (fill_factor = 1.0) to avoid
     * a spurious 2x over-allocation on large factors. */
    err = ldlt_csc_wrapper_rebuild_csc_factor(F, &ll);
    if (err != SPARSE_OK) {
        sparse_ldlt_free(&ll);
        free(perm_in);
        return err;
    }

    sparse_ldlt_free(&ll);
    free(perm_in);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 2: In-place symmetric swap primitive
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Algorithm (see header comment for the why):
 *
 *   Normalise i < j.
 *
 *   Phase A (cols [0, i)):
 *     For each column c, look up rows i and j in c's sorted row_idx
 *     slice.  Four cases:
 *       - Both present: swap values in place.
 *       - Only i: rename row_idx[pos_i] = j and bubble forward.
 *       - Only j: rename row_idx[pos_j] = i and bubble backward.
 *       - Neither: no-op.
 *     Column sizes and col_ptr are unchanged.
 *
 *   Phase B (cols [i, j]):
 *     Gather every entry in the block into (row, col, value) triples,
 *     apply σ to both coordinates, reflect to lower triangle if
 *     needed, bucket by new column and insertion-sort each bucket by
 *     row.  Write the result back into the same CSC slots (total block
 *     nnz is preserved so col_ptr[j+1] stays put — only col_ptr[i+1..j]
 *     shift within the block).
 *
 *   Phase C: swap F->D[i] ↔ F->D[j], F->D_offdiag[i] ↔ F->D_offdiag[j],
 *   F->pivot_size[i] ↔ F->pivot_size[j], F->perm[i] ↔ F->perm[j].
 *
 * Cols (j, n) are untouched: row indices in column c > j are all >= c,
 * so they never equal i or j. */

sparse_err_t ldlt_csc_symmetric_swap(LdltCsc *F, idx_t i, idx_t j) {
    if (!F)
        return SPARSE_ERR_NULL;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (i < 0 || i >= n || j < 0 || j >= n)
        return SPARSE_ERR_BADARG;
    if (i == j)
        return SPARSE_OK;
    if (i > j) {
        idx_t tmp = i;
        i = j;
        j = tmp;
    }

    CholCsc *L = F->L;
    idx_t *col_ptr = L->col_ptr;
    idx_t *row_idx = L->row_idx;
    double *values = L->values;

    /* ── Phase A: cols [0, i) — swap row i ↔ row j per column ──────── */
    for (idx_t c = 0; c < i; c++) {
        idx_t start = col_ptr[c];
        idx_t end = col_ptr[c + 1];
        idx_t pos_i = end;
        idx_t pos_j = end;
        /* Linear scan is fine: columns are typically small (O(nnz/n))
         * and the scan stops as soon as row_idx[p] > j since row_idx
         * is sorted ascending. */
        for (idx_t p = start; p < end; p++) {
            idx_t r = row_idx[p];
            if (r == i) {
                pos_i = p;
            } else if (r == j) {
                pos_j = p;
                break; /* j is the larger target; beyond this, no matches. */
            } else if (r > j) {
                break;
            }
        }
        if (pos_i < end && pos_j < end) {
            /* Both present: swap values; row_idx slots keep their order. */
            double tmp = values[pos_i];
            values[pos_i] = values[pos_j];
            values[pos_j] = tmp;
        } else if (pos_i < end) {
            /* Only i present: rename to j and bubble forward to keep
             * row_idx sorted ascending. */
            double v = values[pos_i];
            idx_t p = pos_i;
            while (p + 1 < end && row_idx[p + 1] < j) {
                row_idx[p] = row_idx[p + 1];
                values[p] = values[p + 1];
                p++;
            }
            row_idx[p] = j;
            values[p] = v;
        } else if (pos_j < end) {
            /* Only j present: rename to i and bubble backward. */
            double v = values[pos_j];
            idx_t p = pos_j;
            while (p > start && row_idx[p - 1] > i) {
                row_idx[p] = row_idx[p - 1];
                values[p] = values[p - 1];
                p--;
            }
            row_idx[p] = i;
            values[p] = v;
        }
    }

    /* ── Phase B: cols [i, j] — gather-permute-scatter ─────────────── */
    idx_t block_start = col_ptr[i];
    idx_t block_end = col_ptr[j + 1];
    idx_t block_nnz = block_end - block_start;

    if (block_nnz > 0) {
        /* Temporary buffers: (row, col, value) triples for the gathered
         * block.  Total nnz is preserved through the permutation, so the
         * same block slot holds the rebuilt content without shifting
         * cols (j, n-1]. */
        if ((size_t)block_nnz > SIZE_MAX / sizeof(idx_t))
            return SPARSE_ERR_ALLOC;
        if ((size_t)block_nnz > SIZE_MAX / sizeof(double))
            return SPARSE_ERR_ALLOC;
        idx_t block_width = j - i + 1;
        /* Guard `block_width * sizeof(idx_t)` before the calloc below
         * so a 32-bit `size_t` platform (or a pathologically large n)
         * can't wrap and under-allocate `new_col_count`. */
        if ((size_t)block_width > SIZE_MAX / sizeof(idx_t))
            return SPARSE_ERR_ALLOC;

        idx_t *tmp_rows = malloc((size_t)block_nnz * sizeof(idx_t));
        idx_t *tmp_cols = malloc((size_t)block_nnz * sizeof(idx_t));
        double *tmp_vals = malloc((size_t)block_nnz * sizeof(double));
        idx_t *new_col_count = calloc((size_t)block_width, sizeof(idx_t));
        if (!tmp_rows || !tmp_cols || !tmp_vals || !new_col_count) {
            free(tmp_rows);
            free(tmp_cols);
            free(tmp_vals);
            free(new_col_count);
            return SPARSE_ERR_ALLOC;
        }

        /* Gather: walk cols [i, j], apply σ to (row, col), reflect to
         * lower triangle. */
        idx_t k = 0;
        for (idx_t c = i; c <= j; c++) {
            idx_t cstart = col_ptr[c];
            idx_t cend = col_ptr[c + 1];
            for (idx_t p = cstart; p < cend; p++) {
                idx_t r = row_idx[p];
                double v = values[p];
                idx_t rn = (r == i) ? j : ((r == j) ? i : r);
                idx_t cn = (c == i) ? j : ((c == j) ? i : c);
                /* Lower-triangle reflection by symmetry of the underlying
                 * matrix: (rn, cn) and (cn, rn) hold the same value, so
                 * pick whichever is lower-triangular. */
                if (rn < cn) {
                    idx_t t = rn;
                    rn = cn;
                    cn = t;
                }
                tmp_rows[k] = rn;
                tmp_cols[k] = cn;
                tmp_vals[k] = v;
                /* Invariant: cn ∈ [i, j] after the σ/reflect above, so
                 * `cn - i` ∈ [0, block_width).  The static analyzer
                 * can't prove this through the ternary + conditional
                 * swap chain, so the bound is asserted here for
                 * documentation and the access is NOLINT-suppressed. */
                new_col_count[cn - i]++; // NOLINT(clang-analyzer-security.ArrayBound)
                k++;
            }
        }

        /* Compute the new per-column write cursors within the existing
         * block slot.  Block boundaries (col_ptr[i], col_ptr[j+1]) do
         * not move; only col_ptr[i+1..j] shift within the block. */
        idx_t cursor = block_start;
        idx_t *col_write = new_col_count; /* reuse: turn into write cursors */
        for (idx_t c = 0; c < block_width; c++) {
            idx_t count = new_col_count[c];
            col_ptr[i + c] = cursor;
            col_write[c] = cursor;
            cursor += count;
        }
        /* col_ptr[j + 1] already equals block_end; leave it. */

        /* Scatter: bucket each triple by new column.  `tmp_cols[t]` is
         * written for every t in [0, block_nnz) during the gather loop
         * above (k ends at exactly block_nnz), and `tmp_cols[t]` is
         * always in [i, j] by the σ/reflect invariant — so `c_local` is
         * always a valid index into col_write[0..block_width).  NOLINT
         * suppresses a false positive where the analyser can't see the
         * 1:1 correspondence between the gather and scatter loops. */
        for (idx_t t = 0; t < block_nnz; t++) {
            idx_t c_local =
                tmp_cols[t] - i; // NOLINT(clang-analyzer-core.UndefinedBinaryOperatorResult)
            idx_t pos = col_write[c_local]++;
            row_idx[pos] = tmp_rows[t];
            values[pos] = tmp_vals[t];
        }

        /* Sort each rebuilt column's slot ascending by row.  Insertion
         * sort matches `sort_column_entries` in sparse_chol_csc.c —
         * columns are typically small and nearly sorted after scatter. */
        for (idx_t c = i; c <= j; c++) {
            idx_t cstart = col_ptr[c];
            idx_t cend = (c + 1 <= j) ? col_ptr[c + 1] : block_end;
            for (idx_t p = cstart + 1; p < cend; p++) {
                idx_t key_row = row_idx[p];
                double key_val = values[p];
                idx_t q = p;
                while (q > cstart && row_idx[q - 1] > key_row) {
                    row_idx[q] = row_idx[q - 1];
                    values[q] = values[q - 1];
                    q--;
                }
                row_idx[q] = key_row;
                values[q] = key_val;
            }
        }

        free(tmp_rows);
        free(tmp_cols);
        free(tmp_vals);
        free(new_col_count); /* was aliased as col_write — same buffer. */
    }

    /* ── Phase C: swap auxiliary arrays at positions i and j ──────── */
    {
        double tmp = F->D[i];
        F->D[i] = F->D[j];
        F->D[j] = tmp;
    }
    {
        double tmp = F->D_offdiag[i];
        F->D_offdiag[i] = F->D_offdiag[j];
        F->D_offdiag[j] = tmp;
    }
    {
        idx_t tmp = F->pivot_size[i];
        F->pivot_size[i] = F->pivot_size[j];
        F->pivot_size[j] = tmp;
    }
    {
        idx_t tmp = F->perm[i];
        F->perm[i] = F->perm[j];
        F->perm[j] = tmp;
    }

    /* ── Phase D: swap row-adjacency slots (Sprint 19 Day 9) ────────
     *
     * The swap σ = (i, j) renames row i ↔ row j in every factored
     * column c ∈ [0, i).  `row_adj[i]` lists priors whose entries
     * landed at row i BEFORE the swap; those entries now live at row
     * j, and vice versa — so swap the two rows' entire adjacency
     * lists (pointer, count, cap) in lockstep.
     *
     * Rows other than i and j are unaffected: column c may have had
     * entries at other rows, but those rows' indices didn't change.
     * Hence we only need to swap the two slots, not rebuild the
     * whole index. */
    if (F->row_adj) {
        idx_t *tmp_ptr = F->row_adj[i];
        F->row_adj[i] = F->row_adj[j];
        F->row_adj[j] = tmp_ptr;
    }
    if (F->row_adj_count) {
        idx_t tmp_cnt = F->row_adj_count[i];
        F->row_adj_count[i] = F->row_adj_count[j];
        F->row_adj_count[j] = tmp_cnt;
    }
    if (F->row_adj_cap) {
        idx_t tmp_cap = F->row_adj_cap[i];
        F->row_adj_cap[i] = F->row_adj_cap[j];
        F->row_adj_cap[j] = tmp_cap;
    }

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18: Native CSC Bunch-Kaufman — scaffolding
 * ═══════════════════════════════════════════════════════════════════════ */

/* ─── Kernel selection & runtime override ──────────────────────────── */

/* Process-scope override.  Default = 0 (LDLT_CSC_KERNEL_DEFAULT) means
 * "use the compile-time default from LDLT_CSC_USE_NATIVE".  Tests and
 * benchmarks set this via `ldlt_csc_set_kernel_override` to exercise a
 * specific path on the current binary. */
static LdltCscKernelOverride g_ldlt_csc_kernel_override = LDLT_CSC_KERNEL_DEFAULT;

void ldlt_csc_set_kernel_override(LdltCscKernelOverride mode) { g_ldlt_csc_kernel_override = mode; }

LdltCscKernelOverride ldlt_csc_get_kernel_override(void) { return g_ldlt_csc_kernel_override; }

sparse_err_t ldlt_csc_eliminate(LdltCsc *F) {
    /* Resolve any DEFAULT override to the compile-time-selected kernel.
     * Writing it as an if + early return (rather than a switch with a
     * separate DEFAULT case) avoids a bugprone-branch-clone lint when
     * LDLT_CSC_USE_NATIVE == 1 makes the DEFAULT and NATIVE bodies
     * identical. */
    LdltCscKernelOverride mode = g_ldlt_csc_kernel_override;
    if (mode == LDLT_CSC_KERNEL_DEFAULT) {
#if LDLT_CSC_USE_NATIVE
        mode = LDLT_CSC_KERNEL_NATIVE;
#else
        mode = LDLT_CSC_KERNEL_WRAPPER;
#endif
    }
    if (mode == LDLT_CSC_KERNEL_NATIVE)
        return ldlt_csc_eliminate_native(F);
    return ldlt_csc_eliminate_wrapper(F);
}

/* ─── Native-kernel workspace lifecycle ────────────────────────────── */

void ldlt_csc_workspace_free(LdltCscWorkspace *ws) {
    if (!ws)
        return;
    free(ws->dense_col);
    free(ws->dense_pattern);
    free(ws->dense_marker);
    free(ws->dense_col_r);
    free(ws->dense_pattern_r);
    free(ws->dense_marker_r);
    free(ws);
}

sparse_err_t ldlt_csc_workspace_alloc(idx_t n, LdltCscWorkspace **out) {
    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    if (n < 0)
        return SPARSE_ERR_BADARG;

    /* Overflow guards: six length-n arrays. */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if ((size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    LdltCscWorkspace *ws = calloc(1, sizeof(LdltCscWorkspace));
    if (!ws)
        return SPARSE_ERR_ALLOC;
    ws->n = n;

    /* Allocate at least 1 slot so later subscripts never trip over a
     * null pointer even for n == 0 — matches `chol_csc_workspace_alloc`. */
    size_t alloc_n = n > 0 ? (size_t)n : 1;
    ws->dense_col = calloc(alloc_n, sizeof(double));
    ws->dense_pattern = calloc(alloc_n, sizeof(idx_t));
    ws->dense_marker = calloc(alloc_n, sizeof(int8_t));
    ws->dense_col_r = calloc(alloc_n, sizeof(double));
    ws->dense_pattern_r = calloc(alloc_n, sizeof(idx_t));
    ws->dense_marker_r = calloc(alloc_n, sizeof(int8_t));
    if (!ws->dense_col || !ws->dense_pattern || !ws->dense_marker || !ws->dense_col_r ||
        !ws->dense_pattern_r || !ws->dense_marker_r) {
        ldlt_csc_workspace_free(ws);
        return SPARSE_ERR_ALLOC;
    }

    *out = ws;
    return SPARSE_OK;
}

/* ─── Ported BK pivot scanner (phase 1) ────────────────────────────── */

/* Bunch-Kaufman alpha = (1 + sqrt(17)) / 8 ≈ 0.6404.  Computed once at
 * first call rather than baking in a double literal — matches
 * `sparse_ldlt.c`'s runtime computation, so any future sqrt-precision
 * tweak applies to both kernels. */
static double ldlt_csc_bk_alpha(void) { return (1.0 + sqrt(17.0)) / 8.0; }

/* Scan the dense column accumulator for the largest off-diagonal
 * magnitude at rows i > k.  Returns the magnitude (0.0 if none) and
 * writes the row index of that off-diagonal into *r_out (k when the
 * column has no below-diagonal fill, matching `sparse_ldlt.c`'s
 * sentinel convention).
 *
 * Ported from the inline block at src/sparse_ldlt.c:498-507.  Day 3
 * wires this into `ldlt_csc_eliminate_native`'s column loop.
 */
static double ldlt_csc_bk_scan_offdiag(const double *dense_col, const idx_t *pattern,
                                       idx_t pattern_count, idx_t k, idx_t *r_out) {
    double max_offdiag = 0.0;
    idx_t r = k;
    for (idx_t t = 0; t < pattern_count; t++) {
        idx_t i = pattern[t];
        if (i > k) {
            double mag = fabs(dense_col[i]);
            if (mag > max_offdiag) {
                max_offdiag = mag;
                r = i;
            }
        }
    }
    *r_out = r;
    return max_offdiag;
}

/* ─── Scatter + cmod helpers (Sprint 18 Day 3) ──────────────────── */

/* Scatter the symmetric column `col` at step k into a dense accumulator.
 *
 * At step `step_k`, F->L stores factored L (with unit diagonal) in
 * columns [0, step_k) and A's lower triangle in columns [step_k, n).
 * Scattering "column col" in the symmetric sense means picking up:
 *   - lower-tri stored entries of column col with row >= step_k
 *     (iterate col_ptr[col]..col_ptr[col+1]); and
 *   - reflected upper-tri entries at rows in [step_k, col) — by
 *     symmetry A[col, c] == A[c, col], so for each column c in
 *     [step_k, col) we binary-search for row `col` in c's slice and
 *     place the hit into dense[c].
 *
 * The upper-tri loop is empty when `col == step_k` (primary pivot
 * column k) and non-empty when `col > step_k` (BK phase-2 partner r).
 */
static void ldlt_csc_scatter_symmetric(const CholCsc *L, idx_t col, idx_t step_k, double *dense,
                                       idx_t *pattern, int8_t *marker, idx_t *pattern_count) {
    idx_t cstart = L->col_ptr[col];
    idx_t cend = L->col_ptr[col + 1];
    for (idx_t p = cstart; p < cend; p++) {
        idx_t i = L->row_idx[p];
        if (i < step_k)
            continue;
        dense[i] = L->values[p];
        if (!marker[i]) {
            marker[i] = 1;
            pattern[(*pattern_count)++] = i;
        }
    }
    for (idx_t c = step_k; c < col; c++) {
        idx_t start = L->col_ptr[c];
        idx_t end = L->col_ptr[c + 1];
        idx_t lo = start;
        idx_t hi = end;
        while (lo < hi) {
            idx_t mid = lo + (hi - lo) / 2;
            if (L->row_idx[mid] < col) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < end && L->row_idx[lo] == col) {
            dense[c] = L->values[lo];
            if (!marker[c]) {
                marker[c] = 1;
                pattern[(*pattern_count)++] = c;
            }
        }
    }
}

/* Binary-search for `target` in the sorted row_idx slice
 * [cstart, cend) of a CSC column.  Returns the L value at that row, or
 * 0.0 if not present. */
static double ldlt_csc_lookup_Lrc(const CholCsc *L, idx_t cstart, idx_t cend, idx_t target) {
    idx_t lo = cstart;
    idx_t hi = cend;
    while (lo < hi) {
        idx_t mid = lo + (hi - lo) / 2;
        if (L->row_idx[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < cend && L->row_idx[lo] == target) {
        return L->values[lo];
    }
    return 0.0;
}

/* Apply cmod contributions from prior factored columns kp in [0, step_k)
 * to the dense accumulator for column `col`.
 *
 * Two passes:
 *
 *   Phase A: for every prior column kp where `L[col, kp] != 0`,
 *   subtract `L[i, kp] * D[kp] * L[col, kp]` from dense[i] for every
 *   stored row i >= step_k in column kp.  D[kp] here is the per-
 *   column diagonal entry (the block-diagonal element for 2×2
 *   priors, NOT the 2×2 inverse).
 *
 *   Phase B: for every 2×2 block pivot pair (kp, kp+1) where
 *   `L[col, kp] != 0` or `L[col, kp+1] != 0`, add the off-diagonal
 *   cross-term `L[i, kp] * D_off[kp] * L[col, kp+1]
 *   + L[i, kp+1] * D_off[kp] * L[col, kp]` to the subtraction.
 *
 * Phase A + Phase B together reproduce the reference acc_schur_col
 * semantics in sparse_ldlt.c exactly.  New rows touched by either pass
 * get appended to the pattern via the marker check.
 *
 * Sprint 19 Day 9 switch: both phases iterate `F->row_adj[col]`
 * instead of `[0, step_k)`.  `row_adj[col]` is populated by
 * `ldlt_csc_eliminate_native` after every column writeback with the
 * list of prior columns that have a stored non-zero at row `col`.
 * Iterating only those columns matches the linked-list reference's
 * `acc_schur_col` sparse-row scaling (the linked-list's cross-linked
 * row iterator visits exactly the columns that contribute); the
 * pre-Sprint-19 kernel iterated every `kp in [0, step_k)` with a
 * binary search per `kp`, giving O(step_k · log nnz) per cmod even
 * when `col`'s row was very sparse. */
static void ldlt_csc_cmod_unified(const LdltCsc *F, idx_t col, idx_t step_k, double *dense,
                                  idx_t *pattern, int8_t *marker, idx_t *pattern_count) {
    const CholCsc *L = F->L;
    idx_t row_adj_count = (col >= 0 && col < F->n) ? F->row_adj_count[col] : 0;
    idx_t *row_adj = (col >= 0 && col < F->n) ? F->row_adj[col] : NULL;

    /* Phase A: per-column diagonal contribution — walk `row_adj[col]`. */
    for (idx_t idx = 0; idx < row_adj_count; idx++) {
        idx_t kp = row_adj[idx];
        /* Defensive: `row_adj` may contain entries ≥ step_k if the
         * caller repopulated across elimination restarts (shouldn't
         * happen today but the guard is free).  Skip them so they
         * don't double-count into a future step. */
        if (kp >= step_k)
            continue;

        idx_t cstart = L->col_ptr[kp];
        idx_t cend = L->col_ptr[kp + 1];
        double L_col_kp = ldlt_csc_lookup_Lrc(L, cstart, cend, col);
        if (L_col_kp == 0.0)
            continue; /* should not happen if row_adj is accurate */
        double factor = F->D[kp] * L_col_kp;
        for (idx_t p = cstart; p < cend; p++) {
            idx_t i = L->row_idx[p];
            if (i < step_k)
                continue;
            if (!marker[i]) {
                marker[i] = 1;
                /* pattern is allocated with n slots and marker-gated
                 * uniqueness bounds *pattern_count by n throughout the
                 * elimination; analyzer can't follow this across the
                 * helper boundary so the access is NOLINT-suppressed. */
                pattern[(*pattern_count)++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
            }
            dense[i] -= L->values[p] * factor;
        }
    }

    /* Phase B: cross-term correction for 2×2 block priors — walk the
     * same `row_adj[col]` list, detecting 2×2 pair membership via
     * `pivot_size`.  Each entry `kp` in the list triggers at most one
     * inner loop:
     *
     *   - If `kp` is the FIRST of a 2×2 pair `(kp, kp+1)`:
     *     `ct2 = D_off[kp] * L[col, kp]` (non-zero since `kp` is in
     *     `row_adj`), and the inner loop runs on column `kp+1`
     *     (updating `dense[i] -= L[i, kp+1] * ct2`).
     *
     *   - If `kp` is the SECOND of a 2×2 pair `(kp-1, kp)`:
     *     `ct1 = D_off[kp-1] * L[col, kp]`, inner loop on column
     *     `kp-1` (updating `dense[i] -= L[i, kp-1] * ct1`).
     *
     * The two inner-loop cases are the row-adj-driven counterparts of
     * the old Phase B's `ct1 != 0` / `ct2 != 0` branches.  If `col`
     * has no stored entry in either member of a 2×2 pair, the pair
     * contributes zero and is correctly skipped (neither member
     * appears in `row_adj[col]`). */
    for (idx_t idx = 0; idx < row_adj_count; idx++) {
        idx_t kp = row_adj[idx];
        if (kp >= step_k)
            continue;
        if (F->pivot_size[kp] != 2)
            continue;

        idx_t pair_other;
        int kp_is_first;
        if (kp + 1 < step_k && F->pivot_size[kp + 1] == 2 &&
            (kp == 0 || F->pivot_size[kp - 1] != 2)) {
            /* kp is the first of a 2×2 at (kp, kp+1). */
            pair_other = kp + 1;
            kp_is_first = 1;
        } else if (kp >= 1 && F->pivot_size[kp - 1] == 2) {
            /* kp is the second of a 2×2 at (kp-1, kp). */
            pair_other = kp - 1;
            kp_is_first = 0;
        } else {
            continue; /* defensive: malformed pivot_size[] */
        }

        idx_t d_off_idx = kp_is_first ? kp : pair_other;
        double d_off = F->D_offdiag[d_off_idx];
        if (d_off == 0.0)
            continue;

        /* L[col, kp] is guaranteed non-zero (kp is in row_adj). */
        idx_t cstart_kp = L->col_ptr[kp];
        idx_t cend_kp = L->col_ptr[kp + 1];
        double L_col_kp = ldlt_csc_lookup_Lrc(L, cstart_kp, cend_kp, col);
        double ct = d_off * L_col_kp; /* ct2 when kp is first, ct1 when kp is second */
        if (ct == 0.0)
            continue;

        /* Inner loop on the OTHER column of the pair. */
        idx_t cstart_o = L->col_ptr[pair_other];
        idx_t cend_o = L->col_ptr[pair_other + 1];
        for (idx_t p = cstart_o; p < cend_o; p++) {
            idx_t i = L->row_idx[p];
            if (i < step_k)
                continue;
            if (!marker[i]) {
                marker[i] = 1;
                pattern[(*pattern_count)++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
            }
            dense[i] -= L->values[p] * ct;
        }
    }
}

/* Sprint 19 Day 9: populate `F->row_adj` for every prior column `col`
 * contributes to, by walking column `col`'s storage in `F->L` after
 * gather.  For each stored row `i > col`, append `col` to
 * `F->row_adj[i]`.  The diagonal entry (row == col) is skipped — a
 * column is not its own prior.
 *
 * Called once per column writeback in `ldlt_csc_eliminate_native`;
 * together with the row-adj-driven cmod this reproduces the linked-
 * list reference's sparse-row iteration without the O(step_k) scan. */
static sparse_err_t ldlt_csc_populate_row_adj(LdltCsc *F, idx_t col) {
    idx_t cstart = F->L->col_ptr[col];
    idx_t cend = F->L->col_ptr[col + 1];
    for (idx_t p = cstart; p < cend; p++) {
        idx_t i = F->L->row_idx[p];
        if (i > col) {
            sparse_err_t err = ldlt_csc_row_adj_append(F, i, col);
            if (err != SPARSE_OK)
                return err;
        }
    }
    return SPARSE_OK;
}

/* Clear the primary accumulator's touched entries.
 *
 * pattern_count is maintained by the column loop to never exceed the
 * allocation size (ws->n); the marker-gated increments in scatter and
 * cmod ensure uniqueness.  Analyzer can't track that through the helper
 * boundary, hence the NOLINT suppression on the access. */
static void ldlt_csc_clear_dense_col(LdltCscWorkspace *ws) {
    for (idx_t t = 0; t < ws->pattern_count; t++) {
        idx_t i = ws->dense_pattern[t]; // NOLINT(clang-analyzer-security.ArrayBound)
        ws->dense_col[i] = 0.0;
        ws->dense_marker[i] = 0;
    }
    ws->pattern_count = 0;
}

/* Clear the partner accumulator's touched entries (same invariants as
 * `ldlt_csc_clear_dense_col`). */
static void ldlt_csc_clear_dense_col_r(LdltCscWorkspace *ws) {
    for (idx_t t = 0; t < ws->pattern_count_r; t++) {
        idx_t i = ws->dense_pattern_r[t]; // NOLINT(clang-analyzer-security.ArrayBound)
        ws->dense_col_r[i] = 0.0;
        ws->dense_marker_r[i] = 0;
    }
    ws->pattern_count_r = 0;
}

/* ─── Native kernel (Sprint 18 Days 3-4: full 1×1 + 2×2) ─────────── */

/* Sprint 19 Day 13 refactor: the body of `ldlt_csc_eliminate_native`'s
 * column loop is extracted here so `ldlt_csc_eliminate_supernodal`
 * (Day 13) can interleave it with the batched supernodal path on
 * non-supernodal columns.
 *
 * One step factors a single 1×1 pivot or a 2×2 pivot pair starting at
 * column `*k_inout`, advancing it by 1 or 2 on success.  All temporary
 * accumulator state lives on `ws`; the caller (eliminate_native /
 * eliminate_supernodal) owns `ws`'s lifecycle.  On error the
 * accumulator may be left dirty — the caller's policy is to free `ws`
 * (and not reuse it) before returning, so clearing is unnecessary. */
static sparse_err_t ldlt_csc_eliminate_one_step(LdltCsc *F, LdltCscWorkspace *ws, idx_t *k_inout,
                                                double drop_tol, double sing_tol, double alpha_bk,
                                                double growth_bound) {
    idx_t n = F->n;
    idx_t k = *k_inout;
    sparse_err_t rc = SPARSE_OK;

    /* ── Scatter + cmod for column k ────────────────────────── */
    ldlt_csc_scatter_symmetric(F->L, k, k, ws->dense_col, ws->dense_pattern, ws->dense_marker,
                               &ws->pattern_count);
    ldlt_csc_cmod_unified(F, k, k, ws->dense_col, ws->dense_pattern, ws->dense_marker,
                          &ws->pattern_count);

    if (!ws->dense_marker[k]) {
        ws->dense_marker[k] = 1;
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        ws->dense_pattern[ws->pattern_count++] = k;
    }

    /* ── Phase 1 BK scan ────────────────────────────────────── */
    idx_t r = k;
    double max_offdiag =
        ldlt_csc_bk_scan_offdiag(ws->dense_col, ws->dense_pattern, ws->pattern_count, k, &r);
    double diag_k = ws->dense_col[k];

    int use_2x2 = 0;

    /* ── Phase 2 BK criteria (only when the diagonal is small) ── */
    if (max_offdiag > 0.0 && k + 1 < n && fabs(diag_k) < alpha_bk * max_offdiag) {
        ldlt_csc_scatter_symmetric(F->L, r, k, ws->dense_col_r, ws->dense_pattern_r,
                                   ws->dense_marker_r, &ws->pattern_count_r);
        ldlt_csc_cmod_unified(F, r, k, ws->dense_col_r, ws->dense_pattern_r, ws->dense_marker_r,
                              &ws->pattern_count_r);

        double sigma_r = 0.0;
        for (idx_t t = 0; t < ws->pattern_count_r; t++) {
            idx_t i = ws->dense_pattern_r[t];
            if (i >= k && i != r) {
                double m = fabs(ws->dense_col_r[i]);
                if (m > sigma_r)
                    sigma_r = m;
            }
        }

        if (fabs(diag_k) * sigma_r >= alpha_bk * max_offdiag * max_offdiag) {
            ldlt_csc_clear_dense_col_r(ws);
        } else if (fabs(ws->dense_col_r[r]) >= alpha_bk * sigma_r) {
            rc = ldlt_csc_symmetric_swap(F, k, r);
            if (rc != SPARSE_OK)
                return rc;

            ldlt_csc_clear_dense_col(ws);
            for (idx_t t = 0; t < ws->pattern_count_r; t++) {
                idx_t i = ws->dense_pattern_r[t];
                idx_t m = (i == k) ? r : (i == r ? k : i);
                ws->dense_col[m] = ws->dense_col_r[i];
                if (!ws->dense_marker[m]) {
                    ws->dense_marker[m] = 1;
                    ws->dense_pattern[ws->pattern_count++] = m;
                }
            }
            ldlt_csc_clear_dense_col_r(ws);
            diag_k = ws->dense_col[k];
            if (!ws->dense_marker[k]) {
                ws->dense_marker[k] = 1;
                ws->dense_pattern[ws->pattern_count++] = k;
            }
        } else {
            use_2x2 = 1;
        }
    }

    if (!use_2x2) {
        /* ── 1×1 apply ────────────────────────────────────── */
        if (fabs(diag_k) < sing_tol)
            return SPARSE_ERR_SINGULAR;
        F->D[k] = diag_k;
        F->D_offdiag[k] = 0.0;
        F->pivot_size[k] = 1;

        for (idx_t t = 0; t < ws->pattern_count; t++) {
            idx_t i = ws->dense_pattern[t];
            if (i > k) {
                double v = ws->dense_col[i] / diag_k;
                if (fabs(v) > growth_bound)
                    return SPARSE_ERR_SINGULAR;
                ws->dense_col[i] = v;
            }
        }

        ws->dense_col[k] = 1.0;

        CholCscWorkspace view;
        view.n = ws->n;
        view.dense_col = ws->dense_col;
        view.dense_pattern = ws->dense_pattern;
        view.dense_marker = ws->dense_marker;
        view.pattern_count = ws->pattern_count;
        rc = chol_csc_gather(F->L, k, &view, drop_tol);
        ws->pattern_count = view.pattern_count;
        if (rc != SPARSE_OK)
            return rc;

        rc = ldlt_csc_populate_row_adj(F, k);
        if (rc != SPARSE_OK)
            return rc;

        ldlt_csc_clear_dense_col(ws);
        *k_inout = k + 1;
        return SPARSE_OK;
    }

    /* ── 2×2 block pivot at (k, r) ─────────────────────────── */
    if (r != k + 1) {
        rc = ldlt_csc_symmetric_swap(F, r, k + 1);
        if (rc != SPARSE_OK)
            return rc;

        double tmp_d = ws->dense_col[r];
        ws->dense_col[r] = ws->dense_col[k + 1];
        ws->dense_col[k + 1] = tmp_d;
        int8_t tmp_m = ws->dense_marker[r];
        ws->dense_marker[r] = ws->dense_marker[k + 1];
        ws->dense_marker[k + 1] = tmp_m;
        for (idx_t t = 0; t < ws->pattern_count; t++) {
            if (ws->dense_pattern[t] == r) {
                ws->dense_pattern[t] = k + 1;
            } else if (ws->dense_pattern[t] == k + 1) {
                ws->dense_pattern[t] = r;
            }
        }

        tmp_d = ws->dense_col_r[r];
        ws->dense_col_r[r] = ws->dense_col_r[k + 1];
        ws->dense_col_r[k + 1] = tmp_d;
        tmp_m = ws->dense_marker_r[r];
        ws->dense_marker_r[r] = ws->dense_marker_r[k + 1];
        ws->dense_marker_r[k + 1] = tmp_m;
        for (idx_t t = 0; t < ws->pattern_count_r; t++) {
            if (ws->dense_pattern_r[t] == r) {
                ws->dense_pattern_r[t] = k + 1;
            } else if (ws->dense_pattern_r[t] == k + 1) {
                ws->dense_pattern_r[t] = r;
            }
        }
    }

    double d11 = ws->dense_col[k];
    double d21 = ws->dense_col[k + 1];
    double d22 = ws->dense_col_r[k + 1];
    double det = d11 * d22 - d21 * d21;
    double bscale = fabs(d11) + fabs(d22) + fabs(d21);
    double det_tol = (bscale > 0.0) ? drop_tol * bscale * bscale : sing_tol * sing_tol;
    if (fabs(det) < det_tol)
        return SPARSE_ERR_SINGULAR;
    double inv_det = 1.0 / det;
    double drop_2x2 = (bscale > 0.0) ? drop_tol * bscale : drop_tol;

    F->D[k] = d11;
    F->D[k + 1] = d22;
    F->D_offdiag[k] = d21;
    F->D_offdiag[k + 1] = 0.0;
    F->pivot_size[k] = 2;
    F->pivot_size[k + 1] = 2;

    for (idx_t t = 0; t < ws->pattern_count_r; t++) {
        idx_t i = ws->dense_pattern_r[t];
        if (!ws->dense_marker[i]) {
            ws->dense_marker[i] = 1;
            ws->dense_pattern[ws->pattern_count++] = i;
        }
    }

    for (idx_t t = 0; t < ws->pattern_count; t++) {
        idx_t i = ws->dense_pattern[t];
        if (i <= k + 1)
            continue;
        double s_ik = ws->dense_col[i];
        double s_ik1 = ws->dense_col_r[i];
        double l_ik = (s_ik * d22 - s_ik1 * d21) * inv_det;
        double l_ik1 = (-s_ik * d21 + s_ik1 * d11) * inv_det;
        if (fabs(l_ik) > growth_bound || fabs(l_ik1) > growth_bound)
            return SPARSE_ERR_SINGULAR;
        ws->dense_col[i] = l_ik;
        ws->dense_col_r[i] = l_ik1;
        if (!ws->dense_marker_r[i]) {
            ws->dense_marker_r[i] = 1;
            ws->dense_pattern_r[ws->pattern_count_r++] = i;
        }
    }

    ws->dense_col[k] = 1.0;
    ws->dense_col[k + 1] = 0.0;

    CholCscWorkspace view_k;
    view_k.n = ws->n;
    view_k.dense_col = ws->dense_col;
    view_k.dense_pattern = ws->dense_pattern;
    view_k.dense_marker = ws->dense_marker;
    view_k.pattern_count = ws->pattern_count;
    rc = chol_csc_gather(F->L, k, &view_k, drop_2x2);
    ws->pattern_count = view_k.pattern_count;
    if (rc != SPARSE_OK)
        return rc;

    ws->dense_col_r[k + 1] = 1.0;
    ws->dense_col_r[k] = 0.0;
    if (!ws->dense_marker[k + 1]) {
        ws->dense_marker[k + 1] = 1;
        ws->dense_pattern[ws->pattern_count++] = k + 1;
    }
    if (!ws->dense_marker_r[k + 1]) {
        ws->dense_marker_r[k + 1] = 1;
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        ws->dense_pattern_r[ws->pattern_count_r++] = k + 1;
    }
    if (!ws->dense_marker_r[k]) {
        ws->dense_marker_r[k] = 1;
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        ws->dense_pattern_r[ws->pattern_count_r++] = k;
    }

    CholCscWorkspace view_k1;
    view_k1.n = ws->n;
    view_k1.dense_col = ws->dense_col_r;
    view_k1.dense_pattern = ws->dense_pattern;
    view_k1.dense_marker = ws->dense_marker;
    view_k1.pattern_count = ws->pattern_count;
    rc = chol_csc_gather(F->L, k + 1, &view_k1, drop_2x2);
    ws->pattern_count = view_k1.pattern_count;
    if (rc != SPARSE_OK)
        return rc;

    rc = ldlt_csc_populate_row_adj(F, k);
    if (rc != SPARSE_OK)
        return rc;
    rc = ldlt_csc_populate_row_adj(F, k + 1);
    if (rc != SPARSE_OK)
        return rc;

    ldlt_csc_clear_dense_col(ws);
    ldlt_csc_clear_dense_col_r(ws);
    *k_inout = k + 2;
    return SPARSE_OK;
}

/* Sprint 18 Days 3-4 ship a complete Bunch-Kaufman kernel directly on
 * CSC storage: scatter + cmod per column (handling both 1×1 and 2×2
 * priors via `ldlt_csc_cmod_unified`), four-criteria pivot selection
 * with an in-place symmetric swap for criteria 3 and 4, 1×1 divide
 * or 2×2 block factor with element-growth tracking against
 * `growth_bound = 1 / (100 * DROP_TOL)` matching the linked-list
 * reference, and gather through `chol_csc_gather`.
 *
 * F->perm is updated in place via the symmetric-swap helper at each
 * BK swap, so no separate "compose with fill-reducing perm" step is
 * needed — by the end of the loop F->perm holds the composition the
 * wrapper produces via its post-factor unpack.
 */
sparse_err_t ldlt_csc_eliminate_native(LdltCsc *F) {
    if (!F)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (n <= 0)
        return SPARSE_OK;

    /* Same structural input validation the wrapper performs. */
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    if (F->L->n != n || !F->L->col_ptr)
        return SPARSE_ERR_BADARG;
    if (F->L->col_ptr[0] != 0)
        return SPARSE_ERR_BADARG;
    idx_t l_nnz = F->L->col_ptr[n];
    if (l_nnz < 0)
        return SPARSE_ERR_BADARG;
    for (idx_t j = 0; j < n; j++) {
        idx_t col_start = F->L->col_ptr[j];
        idx_t col_end = F->L->col_ptr[j + 1];
        if (col_start < 0 || col_end < 0 || col_start > col_end || col_start > l_nnz ||
            col_end > l_nnz)
            return SPARSE_ERR_BADARG;
    }
    if (l_nnz > 0 && (!F->L->row_idx || !F->L->values))
        return SPARSE_ERR_NULL;
    for (idx_t p = 0; p < l_nnz; p++) {
        if (F->L->row_idx[p] < 0 || F->L->row_idx[p] >= n)
            return SPARSE_ERR_BADARG;
    }

    LdltCscWorkspace *ws = NULL;
    sparse_err_t err = ldlt_csc_workspace_alloc(n, &ws);
    if (err != SPARSE_OK)
        return err;

    /* Tolerances — match sparse_ldlt.c exactly so native / wrapper
     * decisions stay in lockstep on borderline matrices. */
    const double drop_tol = SPARSE_DROP_TOL;
    const double sing_tol = sparse_rel_tol(F->factor_norm, drop_tol);
    const double alpha_bk = ldlt_csc_bk_alpha();
    const double growth_bound = 1.0 / (100.0 * drop_tol);

    sparse_err_t rc = SPARSE_OK;

    idx_t k = 0;
    while (k < n) {
        rc = ldlt_csc_eliminate_one_step(F, ws, &k, drop_tol, sing_tol, alpha_bk, growth_bound);
        if (rc != SPARSE_OK)
            goto cleanup;
    }

cleanup:
    ldlt_csc_clear_dense_col(ws);
    ldlt_csc_clear_dense_col_r(ws);
    ldlt_csc_workspace_free(ws);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 9: LDL^T solve — forward / diagonal-block / backward sweeps
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t ldlt_csc_solve(const LdltCsc *F, const double *b, double *x) {
    if (!F || !b || !x)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (n == 0)
        return SPARSE_OK;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_BADARG;

    /* Workspace: y holds the permuted RHS → forward-solved vector;
     * z receives the diagonal-solved vector and is then overwritten
     * in place by the backward sweep. */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *y = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double));
    if (!y || !z) {
        free(y);
        free(z);
        return SPARSE_ERR_ALLOC;
    }

    /* Tolerance scaling matches sparse_ldlt_solve: 1x1 singularity
     * against ||A||_inf, 2x2 block singularity against the block's own
     * scale (|d11| + |d22| + |d21|)^2 to handle Schur complements
     * whose magnitude has drifted from the original matrix norm. */
    double solve_tol = SPARSE_DROP_TOL;
    double sing_tol = sparse_rel_tol(F->factor_norm, solve_tol);

    /* Phase 0: y[i] = b[perm[i]]  (apply P to b). */
    for (idx_t i = 0; i < n; i++)
        y[i] = b[F->perm[i]];

    /* Phase 1: Forward solve L*w = y.  L is unit lower triangular in
     * CSC: for each column j left-to-right, skip the unit diagonal
     * (first stored entry) and subtract L[i,j] * y[j] from every row
     * i > j present in column j's slot. */
    for (idx_t j = 0; j < n; j++) {
        idx_t start = F->L->col_ptr[j];
        idx_t end = F->L->col_ptr[j + 1];
        if (start == end || F->L->row_idx[start] != j) {
            /* Missing unit diagonal — Day 8's elimination guarantees it,
             * so this indicates a hand-corrupted CSC.  Fail safely. */
            free(y);
            free(z);
            return SPARSE_ERR_BADARG;
        }
        double yj = y[j];
        for (idx_t p = start + 1; p < end; p++) {
            idx_t i = F->L->row_idx[p];
            y[i] -= F->L->values[p] * yj;
        }
    }

    /* Phase 2: Diagonal solve D*z = w (w has overwritten y). */
    for (idx_t k = 0; k < n;) {
        if (F->pivot_size[k] == 1) {
            if (fabs(F->D[k]) < sing_tol) {
                free(y);
                free(z);
                return SPARSE_ERR_SINGULAR;
            }
            z[k] = y[k] / F->D[k];
            k++;
        } else {
            /* 2x2 block: [[d11, d21], [d21, d22]]; inv = 1/det * [[d22, -d21], [-d21, d11]].
             * `ldlt_csc_validate` guarantees pivot_size[k] == 2 implies
             * pivot_size[k+1] == 2, which in turn requires k+1 < n — but
             * that invariant isn't visible to clang-tidy's path analyser,
             * so we reject a malformed trailing 2x2 pivot here rather than
             * indexing past y/z. */
            if (k + 1 >= n) {
                free(y);
                free(z);
                return SPARSE_ERR_BADARG;
            }
            double d11 = F->D[k];
            double d22 = F->D[k + 1];
            double d21 = F->D_offdiag[k];
            double det = d11 * d22 - d21 * d21;
            double bscale = fabs(d11) + fabs(d22) + fabs(d21);
            double det_tol = (bscale > 0.0) ? solve_tol * bscale * bscale : sing_tol * sing_tol;
            if (fabs(det) < det_tol) {
                free(y);
                free(z);
                return SPARSE_ERR_SINGULAR;
            }
            double y_k1 = y[k + 1];
            z[k] = (d22 * y[k] - d21 * y_k1) / det;
            z[k + 1] = (d11 * y_k1 - d21 * y[k]) / det;
            k += 2;
        }
    }

    /* Phase 3: Backward solve L^T*v = z.  For each column j of L
     * right-to-left, the below-diagonal entries are exactly row j of
     * L^T; accumulate sum_{i>j} L[i,j] * z[i] and subtract from z[j]. */
    for (idx_t j = n - 1; j >= 0; j--) {
        idx_t start = F->L->col_ptr[j];
        idx_t end = F->L->col_ptr[j + 1];
        double sum = 0.0;
        for (idx_t p = start + 1; p < end; p++) {
            idx_t i = F->L->row_idx[p];
            sum += F->L->values[p] * z[i];
        }
        z[j] -= sum;
    }

    /* Phase 4: x[perm[i]] = z[i]  (apply P^T to z). */
    for (idx_t i = 0; i < n; i++)
        x[F->perm[i]] = z[i];

    free(y);
    free(z);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 19 Day 12: batched supernodal LDL^T — extract / writeback
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The implementations mirror the Cholesky helpers in sparse_chol_csc.c
 * (Sprint 18 Days 6 / 10).  See the design block at the top of this
 * file and the doc-comments on the declarations in
 * sparse_ldlt_csc_internal.h for the LDL^T-specific deltas (drop
 * threshold scale and D / D_offdiag / pivot_size handoff). */

sparse_err_t ldlt_csc_eliminate_supernodal(LdltCsc *F, idx_t min_size) {
    if (!F)
        return SPARSE_ERR_NULL;
    if (min_size < 1)
        return SPARSE_ERR_BADARG;

    idx_t n = F->n;
    if (n == 0)
        return SPARSE_OK;

    /* Same structural validation as `ldlt_csc_eliminate_native` so
     * misuse surfaces consistently across both entry points. */
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    if (F->L->n != n || !F->L->col_ptr)
        return SPARSE_ERR_BADARG;

    /* Detect supernodes from cached `F->pivot_size`. */
    idx_t *starts = malloc((size_t)n * sizeof(idx_t));
    idx_t *sizes = malloc((size_t)n * sizeof(idx_t));
    if (!starts || !sizes) {
        free(starts);
        free(sizes);
        return SPARSE_ERR_ALLOC;
    }
    idx_t super_count = 0;
    sparse_err_t err = ldlt_csc_detect_supernodes(F, min_size, starts, sizes, &super_count);
    if (err != SPARSE_OK) {
        free(starts);
        free(sizes);
        return err;
    }

    LdltCscWorkspace *ws = NULL;
    err = ldlt_csc_workspace_alloc(n, &ws);
    if (err != SPARSE_OK) {
        free(starts);
        free(sizes);
        return err;
    }

    const double drop_tol = SPARSE_DROP_TOL;
    const double sing_tol = sparse_rel_tol(F->factor_norm, drop_tol);
    const double alpha_bk = ldlt_csc_bk_alpha();
    const double growth_bound = 1.0 / (100.0 * drop_tol);

    sparse_err_t rc = SPARSE_OK;
    idx_t super_idx = 0;
    idx_t j = 0;
    while (j < n) {
        /* Skip past detected size-1 supernodes; the per-column scalar
         * branch below handles that one column.  The min_size >= 2
         * guard in the size check below is what gates the batched
         * path's structural-pattern requirements (no fill inside a
         * fundamental supernode). */
        if (super_idx < super_count && j == starts[super_idx] && sizes[super_idx] == 1) {
            super_idx++;
        }

        if (super_idx < super_count && j == starts[super_idx] && sizes[super_idx] >= 2) {
            /* ── Batched supernode at column j ───────────────────── */
            idx_t s_start = j;
            idx_t s_size = sizes[super_idx];
            idx_t panel_height = chol_csc_supernode_panel_height(F->L, s_start);
            /* `s_size >= 2` already enforced by the outer `if`; the
             * remaining defensive checks reject malformed `F->L` (e.g.
             * an empty column at `s_start` or a panel shorter than the
             * supernode's diagonal block). */
            if (panel_height < 1 || panel_height < s_size) {
                rc = SPARSE_ERR_BADARG;
                break;
            }
            if ((size_t)panel_height > SIZE_MAX / sizeof(idx_t) ||
                (size_t)s_size > SIZE_MAX / (size_t)panel_height) {
                rc = SPARSE_ERR_ALLOC;
                break;
            }
            size_t dense_cells = (size_t)panel_height * (size_t)s_size;
            if (dense_cells > SIZE_MAX / sizeof(double)) {
                rc = SPARSE_ERR_ALLOC;
                break;
            }
            double *dense = calloc(dense_cells, sizeof(double));
            idx_t *row_map = malloc((size_t)panel_height * sizeof(idx_t));
            double *D_block = malloc((size_t)s_size * sizeof(double));
            double *D_off_block = malloc((size_t)s_size * sizeof(double));
            idx_t *ps_block = malloc((size_t)s_size * sizeof(idx_t));
            if (!dense || !row_map || !D_block || !D_off_block || !ps_block) {
                free(dense);
                free(row_map);
                free(D_block);
                free(D_off_block);
                free(ps_block);
                rc = SPARSE_ERR_ALLOC;
                break;
            }

            idx_t ph_out = 0;
            rc = ldlt_csc_supernode_extract(F, s_start, s_size, dense, panel_height, row_map,
                                            &ph_out);
            if (rc == SPARSE_OK)
                rc = ldlt_csc_supernode_eliminate_diag(F, s_start, s_size, dense, panel_height,
                                                       row_map, panel_height, D_block, D_off_block,
                                                       ps_block, drop_tol);
            if (rc == SPARSE_OK) {
                idx_t panel_rows = panel_height - s_size;
                if (panel_rows > 0)
                    rc = ldlt_csc_supernode_eliminate_panel(dense, D_block, D_off_block, ps_block,
                                                            s_size, panel_height, dense + s_size,
                                                            panel_height, panel_rows);
            }
            if (rc == SPARSE_OK)
                rc = ldlt_csc_supernode_writeback(F, s_start, s_size, dense, panel_height, row_map,
                                                  panel_height, D_block, D_off_block, ps_block,
                                                  drop_tol);
            if (rc == SPARSE_OK) {
                /* Populate row-adjacency for every column in the
                 * supernode so subsequent scalar (or batched) columns
                 * can iterate `row_adj[col]` instead of `[0, k)`. */
                for (idx_t jj = 0; jj < s_size && rc == SPARSE_OK; jj++)
                    rc = ldlt_csc_populate_row_adj(F, s_start + jj);
            }

            free(dense);
            free(row_map);
            free(D_block);
            free(D_off_block);
            free(ps_block);
            if (rc != SPARSE_OK)
                break;

            j += s_size;
            super_idx++;
        } else {
            /* ── Scalar single-column step at j ───────────────────── */
            rc = ldlt_csc_eliminate_one_step(F, ws, &j, drop_tol, sing_tol, alpha_bk, growth_bound);
            if (rc != SPARSE_OK)
                break;
        }
    }

    ldlt_csc_clear_dense_col(ws);
    ldlt_csc_clear_dense_col_r(ws);
    ldlt_csc_workspace_free(ws);
    free(starts);
    free(sizes);
    return rc;
}
