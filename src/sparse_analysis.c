#include "sparse_analysis.h"
#include "sparse_alloc_internal.h"
#include "sparse_analysis_internal.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_lu.h"
#include "sparse_matrix_state_internal.h"
#include "sparse_reorder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Sprint 28 Days 6-10 — Item 4 (non-pipeline-level pivot): supernodal-etree
 * reordering post-pass.  See `docs/planning/EPIC_2/SPRINT_28/pivot_decision_day1.md`
 * for the chosen-pivot rationale.
 *
 * Day 6 landed the scaffolding (env-var parser + dispatch hook + failing-as-
 * expected test stub).  Day 7 lands the core algorithm: compose the etree
 * postorder with the AMD/ND permutation in place, then rebuild B and
 * recompute etree+postorder so colcount+symbolic Cholesky run on the final
 * (postorder-composed) ordering.  This is the classical Liu 1990 / Davis
 * 2006 §6.5 supernodal-etree reordering: a postorder of the elimination
 * tree maximises the number of consecutive columns that satisfy the
 * fundamental-supernode invariants of `chol_csc_detect_supernodes`.
 *
 * Reorder-agnostic: the post-pass operates on whatever permutation
 * `sparse_analyze` produced (AMD / RCM / COLAMD / ND), composing the
 * etree postorder of P*A*P^T into that perm.  The PR #36 review
 * (comment 3223500618) flagged the original `SPARSE_ND_SUPERNODAL_POSTORDER`
 * name as misleading — the algorithm has nothing ND-specific in it; the
 * `ND_` prefix was an artefact of the Day-1 framing as a Sprint 28 ND
 * fill-quality pivot.  The canonical env var is now `SPARSE_SUPERNODAL_POSTORDER`;
 * the legacy `SPARSE_ND_SUPERNODAL_POSTORDER` is still accepted for
 * back-compat (Sprint 28 captures + advisory recipes that already
 * shipped under the old name remain valid).
 *
 * The composition contract: for an input perm `perm_in` (AMD/ND output) and
 * the etree postorder `po` computed on B = P_in*A*P_in^T, the output perm
 * `perm_out` satisfies
 *
 *     perm_out[k] = perm_in[po[k]]
 *
 * i.e. the k-th column of the final reordered matrix is the column that the
 * etree postorder visits at position k in the AMD-permuted matrix's column
 * space, then chased back through `perm_in` to the original A-column index.
 *
 * Default-off (env var unset) keeps the Sprint 27 behaviour bit-identical;
 * env-var-on adds one extra etree+postorder pass (the second pass is
 * trivial on a postordered etree; total overhead is bounded by the cost
 * of one `sparse_etree_compute` + `sparse_etree_postorder` + one
 * `sparse_permute` call — see `non_pipeline_interim_day7.txt` for the
 * measured per-fixture wall delta). */
typedef enum {
    SUPERNODAL_POSTORDER_OFF = 0, /* Default — Sprint 27 behaviour preserved */
    SUPERNODAL_POSTORDER_ON = 1,  /* Day 7+ — Liu 1990 postorder composition */
} supernodal_postorder_mode_t;

static supernodal_postorder_mode_t parse_supernodal_postorder(void) {
    /* Canonical name: `SPARSE_SUPERNODAL_POSTORDER` (PR #36 review).
     * Legacy name `SPARSE_ND_SUPERNODAL_POSTORDER` is still accepted
     * for back-compat with Sprint 28 captures + advisory recipes that
     * shipped under the old name; the canonical name takes precedence
     * if both are set. */
    const char *env = getenv("SPARSE_SUPERNODAL_POSTORDER");
    if (!env || !*env)
        env = getenv("SPARSE_ND_SUPERNODAL_POSTORDER");
    if (env && strcmp(env, "on") == 0)
        return SUPERNODAL_POSTORDER_ON;
    /* Default + unrecognized + "off" all fall through. */
    return SUPERNODAL_POSTORDER_OFF;
}

/* Compose the etree postorder `po` into the caller's perm in place.
 *
 * Computes `perm[k] := perm_old[po[k]]` for each k ∈ [0, n) using an
 * O(n) scratch buffer.  `po` must be a permutation of [0, n) (the
 * standard postorder contract from `sparse_etree_postorder`); `perm`
 * must point to a length-n array; `n >= 0`.  Returns SPARSE_ERR_ALLOC
 * on scratch-buffer allocation failure.
 *
 * The composition direction matches `sparse_permute`'s perm[new]=old
 * convention: if `perm_in[i]` says "the i-th column of the AMD-permuted
 * matrix is original column perm_in[i]", then after applying the
 * postorder on top, the k-th column of the postorder-permuted matrix
 * is the AMD-permuted matrix's po[k]-th column, which corresponds to
 * original column perm_in[po[k]]. */
static sparse_err_t apply_supernodal_postorder(const idx_t *postorder, idx_t n, idx_t *perm) {
    size_t tmp_bytes = 0;
    if (n < 0)
        return SPARSE_ERR_BADARG;
    if (n == 0)
        return SPARSE_OK;
    if (!postorder || !perm)
        return SPARSE_ERR_NULL;

    idx_t *tmp = NULL;
    if (sparse_idx_count_bytes_overflow(n, sizeof(idx_t), &tmp_bytes) ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&tmp) != SPARSE_OK)
        return SPARSE_ERR_ALLOC;

    for (idx_t k = 0; k < n; k++) {
        idx_t j = postorder[k];
        if (j < 0 || j >= n) {
            free(tmp);
            return SPARSE_ERR_BADARG;
        }
        tmp[k] = perm[j];
    }
    memcpy(perm, tmp, tmp_bytes);
    free(tmp);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_analyze — compute symbolic analysis
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_analyze(const SparseMatrix *A, const sparse_analysis_opts_t *opts,
                            sparse_analysis_t *analysis) {
    if (!A || !analysis)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;
    if (sparse_matrix_require_original_row_col_state(A) != SPARSE_OK)
        return SPARSE_ERR_BADARG;

    /* Default options: Cholesky, no reordering */
    sparse_factor_type_t ftype = SPARSE_FACTOR_CHOLESKY;
    sparse_reorder_t reorder = SPARSE_REORDER_NONE;
    if (opts) {
        ftype = opts->factor_type;
        reorder = opts->reorder;
    }

    idx_t n = A->rows;
    sparse_analysis_free(analysis); /* free any prior contents */
    analysis->n = n;
    analysis->type = ftype;

    /* Cache ||A||_inf */
    analysis->analysis_norm = sparse_norminf_const(A);

    /* Compute fill-reducing permutation if requested */
    sparse_err_t err = SPARSE_OK;
    if (reorder != SPARSE_REORDER_NONE && n > 0) {
        if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&analysis->perm) != SPARSE_OK) {
            sparse_analysis_free(analysis);
            return SPARSE_ERR_ALLOC;
        }
        if (reorder == SPARSE_REORDER_RCM) {
            err = sparse_reorder_rcm(A, analysis->perm);
        } else if (reorder == SPARSE_REORDER_AMD) {
            err = sparse_reorder_amd(A, analysis->perm);
        } else if (reorder == SPARSE_REORDER_COLAMD) {
            /* Note: COLAMD is a column ordering, but sparse_analyze applies
             * it symmetrically (same perm for rows+cols) since the analysis
             * API is built around symmetric permutations. For column-only
             * application, use sparse_qr_factor_opts with COLAMD instead. */
            err = sparse_reorder_colamd(A, analysis->perm);
        } else if (reorder == SPARSE_REORDER_ND) {
            err = sparse_reorder_nd(A, analysis->perm);
        } else {
            sparse_analysis_free(analysis);
            return SPARSE_ERR_BADARG;
        }
        if (err != SPARSE_OK) {
            sparse_analysis_free(analysis);
            return err;
        }
    }

    /* Dispatch by factorization type */
    switch (ftype) {
    case SPARSE_FACTOR_CHOLESKY:
    case SPARSE_FACTOR_LDLT: {
        /* Validate symmetry early to avoid producing a meaningless etree */
        if (!sparse_is_symmetric(A, 1e-12)) {
            sparse_analysis_free(analysis);
            return SPARSE_ERR_NOT_SPD;
        }

        /* Symmetric path: etree + postorder + colcount + symbolic Cholesky.
         * If a permutation was computed, build a symmetrically permuted
         * copy using sparse_permute (perm[new] = old convention). */
        const SparseMatrix *B = A;
        SparseMatrix *B_perm = NULL;

        if (analysis->perm) {
            err = sparse_permute(A, analysis->perm, analysis->perm, &B_perm);
            if (err) {
                sparse_analysis_free(analysis);
                return err;
            }
            B = B_perm;
        }

        /* Allocate etree and postorder work arrays. */
        if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&analysis->etree) != SPARSE_OK ||
            sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&analysis->postorder) != SPARSE_OK) {
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return SPARSE_ERR_ALLOC;
        }

        err = sparse_etree_compute(B, analysis->etree);
        if (err) {
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return err;
        }

        err = sparse_etree_postorder(analysis->etree, n, analysis->postorder);
        if (err) {
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return err;
        }

        /* Sprint 28 Day 7: optional supernodal-etree reordering post-pass.
         * Compose the etree postorder into `analysis->perm` then rebuild
         * B + recompute etree/postorder on the composed ordering so the
         * downstream colcount + symbolic Cholesky run on the final layout
         * (and the cached `analysis->etree` / `analysis->postorder` stay
         * consistent with the perm exposed to callers).  Skipped when
         * `analysis->perm` is NULL (no reordering requested — there's
         * nothing to compose) or the env var is unset (Sprint 27
         * behaviour preserved bit-identically). */
        if (analysis->perm && parse_supernodal_postorder() == SUPERNODAL_POSTORDER_ON) {
            err = apply_supernodal_postorder(analysis->postorder, n, analysis->perm);
            if (err) {
                sparse_free(B_perm);
                sparse_analysis_free(analysis);
                return err;
            }
            /* Rebuild B under the composed perm and recompute etree +
             * postorder.  The recomputed postorder is the identity for a
             * postorder-permuted etree (Liu 1990 §3) but
             * `sparse_etree_postorder` still has to walk the tree, so we
             * run it for correctness rather than asserting identity. */
            sparse_free(B_perm);
            B_perm = NULL;
            err = sparse_permute(A, analysis->perm, analysis->perm, &B_perm);
            if (err) {
                sparse_analysis_free(analysis);
                return err;
            }
            B = B_perm;
            err = sparse_etree_compute(B, analysis->etree);
            if (err) {
                sparse_free(B_perm);
                sparse_analysis_free(analysis);
                return err;
            }
            err = sparse_etree_postorder(analysis->etree, n, analysis->postorder);
            if (err) {
                sparse_free(B_perm);
                sparse_analysis_free(analysis);
                return err;
            }
        }

        idx_t *cc = NULL;
        if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&cc) != SPARSE_OK) {
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return SPARSE_ERR_ALLOC;
        }

        err = sparse_colcount(B, analysis->etree, analysis->postorder, cc);
        if (err) {
            free(cc);
            sparse_free(B_perm);
            sparse_analysis_free(analysis);
            return err;
        }

        /* Compute symbolic structure */
        sparse_symbolic_t sym_internal;
        err = sparse_symbolic_cholesky(B, analysis->etree, analysis->postorder, cc, &sym_internal);
        free(cc);
        sparse_free(B_perm);

        if (err) {
            sparse_analysis_free(analysis);
            return err;
        }

        /* Copy internal symbolic to public struct (same layout) */
        analysis->sym_L.col_ptr = sym_internal.col_ptr;
        analysis->sym_L.row_idx = sym_internal.row_idx;
        analysis->sym_L.n = sym_internal.n;
        analysis->sym_L.nnz = sym_internal.nnz;
        break;
    }

    case SPARSE_FACTOR_LU: {
        /* Unsymmetric path: use sparse_symbolic_lu which computes
         * column etree of A^T*A and produces L and U bounds. */
        sparse_symbolic_t sym_L_int, sym_U_int;
        err = sparse_symbolic_lu(A, analysis->perm, &sym_L_int, &sym_U_int);
        if (err) {
            sparse_analysis_free(analysis);
            return err;
        }

        analysis->sym_L.col_ptr = sym_L_int.col_ptr;
        analysis->sym_L.row_idx = sym_L_int.row_idx;
        analysis->sym_L.n = sym_L_int.n;
        analysis->sym_L.nnz = sym_L_int.nnz;

        analysis->sym_U.col_ptr = sym_U_int.col_ptr;
        analysis->sym_U.row_idx = sym_U_int.row_idx;
        analysis->sym_U.n = sym_U_int.n;
        analysis->sym_U.nnz = sym_U_int.nnz;

        /* The etree/postorder are computed internally by sparse_symbolic_lu.
         * We don't expose them for the LU path since they're of the
         * symmetrized pattern, not A itself. */
        break;
    }

    default:
        sparse_analysis_free(analysis);
        return SPARSE_ERR_BADARG;
    }

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_analysis_free
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_analysis_free(sparse_analysis_t *analysis) {
    if (!analysis)
        return;
    free(analysis->perm);
    free(analysis->etree);
    free(analysis->postorder);
    free(analysis->sym_L.col_ptr);
    free(analysis->sym_L.row_idx);
    free(analysis->sym_U.col_ptr);
    free(analysis->sym_U.row_idx);
    memset(analysis, 0, sizeof(*analysis));
}

/* ═══════════════════════════════════════════════════════════════════════
 * Helper: build symmetrically permuted copy of A
 * ═══════════════════════════════════════════════════════════════════════ */

/* Reset permutation state on a working copy so the underlying solvers
 * don't apply stale reorder/pivot permutations from the original matrix. */
static SparseMatrix *sanitize_working_copy(SparseMatrix *B) {
    if (!B)
        return NULL;
    sparse_reset_perms(B);
    sparse_factor_state_replace_reorder_perm(B, NULL);
    return B;
}

static sparse_err_t build_permuted_copy(const SparseMatrix *A, const idx_t *perm,
                                        SparseMatrix **out) {
    SparseMatrix *B = NULL;
    sparse_err_t err;
    if (perm) {
        err = sparse_permute(A, perm, perm, &B);
        if (err != SPARSE_OK)
            return err;
    } else {
        B = sparse_copy(A);
        if (!B)
            return SPARSE_ERR_ALLOC;
    }
    *out = sanitize_working_copy(B);
    return *out ? SPARSE_OK : SPARSE_ERR_ALLOC;
}

static void sparse_factors_take_matrix_factor(sparse_factors_t *factors, SparseMatrix *factor);
static void sparse_factors_take_ldlt_factor(sparse_factors_t *factors, sparse_ldlt_t *ldlt);

static sparse_err_t factor_cholesky_with_analysis_csc(const SparseMatrix *A,
                                                      const sparse_analysis_t *analysis,
                                                      sparse_factors_t *factors) {
    CholCsc *L_csc = NULL;
    SparseMatrix *L = NULL;
    sparse_err_t err = chol_csc_from_sparse_with_analysis(A, analysis, &L_csc);
    if (err != SPARSE_OK)
        return err;

    err = chol_csc_eliminate_supernodal(L_csc, SPARSE_CSC_SUPERNODE_MIN_SIZE);
    if (err != SPARSE_OK) {
        chol_csc_free(L_csc);
        return err;
    }

    L = sparse_create(analysis->n, analysis->n);
    if (!L) {
        chol_csc_free(L_csc);
        return SPARSE_ERR_ALLOC;
    }

    err = sparse_factor_state_begin_cholesky(L);
    if (err != SPARSE_OK) {
        chol_csc_free(L_csc);
        sparse_free(L);
        return err;
    }

    /* Keep the shared analysis/factor surface in analysis coordinate space.
     * `analysis->perm` stays the published symmetric permutation, so the
     * factors matrix itself keeps a NULL reorder_perm like the old
     * REORDER_NONE delegated path did. */
    err = chol_csc_writeback_to_sparse(L_csc, L, NULL);
    chol_csc_free(L_csc);
    if (err != SPARSE_OK) {
        sparse_free(L);
        return err;
    }

    sparse_factors_take_matrix_factor(factors, L);
    return SPARSE_OK;
}

static int perm_matches_analysis_reorder(const idx_t *perm, const sparse_analysis_t *analysis) {
    idx_t n = analysis->n;
    if (analysis->perm) {
        for (idx_t i = 0; i < n; i++) {
            if (perm[i] != analysis->perm[i]) // NOLINT
                return 0;
        }
        return 1;
    }

    for (idx_t i = 0; i < n; i++) {
        if (perm[i] != i) // NOLINT
            return 0;
    }
    return 1;
}

static sparse_err_t factor_ldlt_with_analysis_csc(const SparseMatrix *A,
                                                  const sparse_analysis_t *analysis,
                                                  sparse_factors_t *factors) {
    LdltCsc *F_pre = NULL;
    LdltCsc *F_batched = NULL;
    LdltCsc *source = NULL;
    SparseMatrix *A_perm = NULL;
    sparse_analysis_t derived_analysis = {0};
    sparse_ldlt_t ldlt = {0};

    sparse_err_t err = ldlt_csc_from_sparse(A, analysis->perm, 2.0, &F_pre);
    if (err != SPARSE_OK)
        return err;

    err = ldlt_csc_eliminate_native(F_pre);
    if (err != SPARSE_OK) {
        ldlt_csc_free(F_pre);
        return err;
    }

    if (perm_matches_analysis_reorder(F_pre->perm, analysis)) {
        /* The scalar BK pre-pass did not introduce extra symmetric swaps
         * beyond the caller's fill-reducing reorder, so the caller's
         * symbolic analysis already matches the CSC repeated-run path. */
        err = ldlt_csc_from_sparse_with_analysis(A, analysis, &F_batched);
    } else {
        /* When BK adds extra swaps, rebuild the symbolic analysis on the
         * pre-permuted matrix so the batched CSC factor sees the complete
         * symmetric pattern in its final coordinate space. */
        err = sparse_permute(A, F_pre->perm, F_pre->perm, &A_perm);
        if (err == SPARSE_OK) {
            sparse_reset_perms(A_perm);

            sparse_analysis_opts_t an_opts = {
                .factor_type = SPARSE_FACTOR_LDLT,
                .reorder = SPARSE_REORDER_NONE,
            };
            err = sparse_analyze(A_perm, &an_opts, &derived_analysis);
        }
        if (err == SPARSE_OK)
            err = ldlt_csc_from_sparse_with_analysis(A_perm, &derived_analysis, &F_batched);
    }
    if (err != SPARSE_OK) {
        ldlt_csc_free(F_pre);
        ldlt_csc_free(F_batched);
        sparse_analysis_free(&derived_analysis);
        sparse_free(A_perm);
        return err;
    }

    for (idx_t k = 0; k < analysis->n; k++)
        F_batched->pivot_size[k] = F_pre->pivot_size[k];

    err = ldlt_csc_eliminate_supernodal(F_batched, /*min_size=*/2);
    if (err == SPARSE_OK) {
        for (idx_t i = 0; i < analysis->n; i++)
            F_batched->perm[i] = F_pre->perm[i];
        source = F_batched;
    } else {
        source = F_pre;
    }

    err = ldlt_csc_writeback_to_ldlt(source, SPARSE_DROP_TOL, &ldlt);
    if (err == SPARSE_OK)
        sparse_factors_take_ldlt_factor(factors, &ldlt);

    sparse_ldlt_free(&ldlt);
    ldlt_csc_free(F_batched);
    ldlt_csc_free(F_pre);
    sparse_analysis_free(&derived_analysis);
    sparse_free(A_perm);
    return err;
}

static void sparse_factors_init_payload(sparse_factors_t *factors, sparse_factor_type_t type,
                                        idx_t n) {
    factors->type = type;
    factors->n = n;
}

static void sparse_factors_take_matrix_factor(sparse_factors_t *factors, SparseMatrix *factor) {
    factors->F = factor;
    factors->factor_norm = sparse_factor_state_factor_norm(factor);
}

static void sparse_factors_take_ldlt_factor(sparse_factors_t *factors, sparse_ldlt_t *ldlt) {
    factors->F = ldlt->L;
    factors->factor_norm = ldlt->factor_norm;
    factors->D = ldlt->D;
    factors->D_offdiag = ldlt->D_offdiag;
    factors->pivot_size = ldlt->pivot_size;
    factors->ldlt_perm = ldlt->perm;

    ldlt->L = NULL;
    ldlt->D = NULL;
    ldlt->D_offdiag = NULL;
    ldlt->pivot_size = NULL;
    ldlt->perm = NULL;
    ldlt->n = 0;
    ldlt->factor_norm = 0.0;
    ldlt->tol = 0.0;
}

static void sparse_factors_make_ldlt_view(const sparse_factors_t *factors,
                                          sparse_ldlt_t *ldlt_view) {
    ldlt_view->L = factors->F;
    ldlt_view->D = factors->D;
    ldlt_view->D_offdiag = factors->D_offdiag;
    ldlt_view->pivot_size = factors->pivot_size;
    ldlt_view->perm = factors->ldlt_perm;
    ldlt_view->n = factors->n;
    ldlt_view->factor_norm = factors->factor_norm;
    ldlt_view->tol = SPARSE_DROP_TOL;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_factor_numeric
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_factor_numeric(const SparseMatrix *A, const sparse_analysis_t *analysis,
                                   sparse_factors_t *factors) {
    if (!A || !analysis || !factors)
        return SPARSE_ERR_NULL;
    if (A->rows != analysis->n || A->cols != analysis->n)
        return SPARSE_ERR_SHAPE;
    if (sparse_matrix_require_original_row_col_state(A) != SPARSE_OK)
        return SPARSE_ERR_BADARG;

    idx_t n = analysis->n;
    sparse_factors_t new_factors = {0};
    sparse_factors_init_payload(&new_factors, analysis->type, n);

    switch (analysis->type) {
    case SPARSE_FACTOR_CHOLESKY: {
        sparse_err_t err = SPARSE_OK;
        if (n >= SPARSE_CSC_THRESHOLD) {
            /* Avoid the extra `sparse_analyze(...)` hidden inside the CSC
             * one-shot wrapper on larger repeated-run Cholesky problems. */
            err = factor_cholesky_with_analysis_csc(A, analysis, &new_factors);
        } else {
            /* Keep the linked-list route unchanged for smaller problems. */
            SparseMatrix *L = NULL;
            err = build_permuted_copy(A, analysis->perm, &L);
            if (err != SPARSE_OK)
                return err;

            sparse_cholesky_opts_t chol_opts = {
                .reorder = SPARSE_REORDER_NONE,
            };
            err = sparse_cholesky_factor_opts(L, &chol_opts);
            if (err != SPARSE_OK) {
                sparse_free(L);
                return err;
            }

            sparse_factors_take_matrix_factor(&new_factors, L);
        }
        if (err != SPARSE_OK)
            return err;
        break;
    }

    case SPARSE_FACTOR_LU: {
        /* Build (optionally permuted) copy and factor with existing LU */
        SparseMatrix *LU = NULL;
        sparse_err_t err = build_permuted_copy(A, analysis->perm, &LU);
        if (err != SPARSE_OK)
            return err;

        err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
        if (err != SPARSE_OK) {
            sparse_free(LU);
            return err;
        }

        sparse_factors_take_matrix_factor(&new_factors, LU);
        break;
    }

    case SPARSE_FACTOR_LDLT: {
        sparse_err_t err = SPARSE_OK;
        if (n >= SPARSE_CSC_THRESHOLD) {
            err = factor_ldlt_with_analysis_csc(A, analysis, &new_factors);
        } else {
            /* Keep the linked-list route unchanged for smaller problems. */
            SparseMatrix *B = NULL;
            err = build_permuted_copy(A, analysis->perm, &B);
            if (err != SPARSE_OK)
                return err;

            sparse_ldlt_t ldlt;
            sparse_ldlt_opts_t ldlt_opts = {
                .reorder = SPARSE_REORDER_NONE,
            };
            err = sparse_ldlt_factor_opts(B, &ldlt_opts, &ldlt);
            sparse_free(B);
            if (err != SPARSE_OK)
                return err;

            sparse_factors_take_ldlt_factor(&new_factors, &ldlt);
            sparse_ldlt_free(&ldlt);
        }
        if (err != SPARSE_OK)
            return err;
        break;
    }

    default:
        return SPARSE_ERR_BADARG;
    }

    sparse_factor_free(factors);
    *factors = new_factors;
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_factor_solve
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_factor_solve(const sparse_factors_t *factors, const sparse_analysis_t *analysis,
                                 const double *b, double *x) {
    if (!factors || !analysis || !b || !x)
        return SPARSE_ERR_NULL;
    if (!factors->F)
        return SPARSE_ERR_BADARG;
    if (analysis->n != factors->n)
        return SPARSE_ERR_SHAPE;
    if (analysis->type != factors->type)
        return SPARSE_ERR_BADARG;

    idx_t n = factors->n;
    const idx_t *perm = analysis->perm;

    /* Permute b if a fill-reducing permutation was used.
     * perm[new] = old convention: b_perm[new_i] = b[perm[new_i]] */
    double *b_perm = NULL;
    const double *b_eff = b;
    if (perm) {
        if (sparse_malloc_idx_array(n, sizeof(double), (void **)&b_perm) != SPARSE_OK)
            return SPARSE_ERR_ALLOC;
        for (idx_t i = 0; i < n; i++)
            b_perm[i] = b[perm[i]];
        b_eff = b_perm;
    }

    sparse_err_t err;
    double *x_tmp = NULL;
    if (sparse_malloc_idx_array(n, sizeof(double), (void **)&x_tmp) != SPARSE_OK) {
        free(b_perm);
        return SPARSE_ERR_ALLOC;
    }

    switch (factors->type) {
    case SPARSE_FACTOR_CHOLESKY:
        err = sparse_cholesky_solve(factors->F, b_eff, x_tmp);
        break;
    case SPARSE_FACTOR_LU:
        err = sparse_lu_solve(factors->F, b_eff, x_tmp);
        break;
    case SPARSE_FACTOR_LDLT: {
        sparse_ldlt_t ldlt_tmp;
        sparse_factors_make_ldlt_view(factors, &ldlt_tmp);
        err = sparse_ldlt_solve(&ldlt_tmp, b_eff, x_tmp);
        break;
    }
    default:
        err = SPARSE_ERR_BADARG;
        break;
    }

    if (err != SPARSE_OK) {
        free(b_perm);
        free(x_tmp);
        return err;
    }

    /* Unpermute the solution: perm[new] = old, so x[old] = x_tmp[new] */
    if (perm) {
        for (idx_t i = 0; i < n; i++)
            x[perm[i]] = x_tmp[i];
    } else {
        memcpy(x, x_tmp, (size_t)n * sizeof(double));
    }

    free(b_perm);
    free(x_tmp);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_factor_free
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_factor_free(sparse_factors_t *factors) {
    if (!factors)
        return;
    sparse_free(factors->F);
    free(factors->D);
    free(factors->D_offdiag);
    free(factors->pivot_size);
    free(factors->ldlt_perm);
    memset(factors, 0, sizeof(*factors));
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_refactor_numeric
 *
 * Convenience wrapper around sparse_factor_numeric() using an existing
 * symbolic analysis. Performs a full numeric refactorization and does
 * not attempt to validate or reuse the previous numeric structure.
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_refactor_numeric(const SparseMatrix *A_new, const sparse_analysis_t *analysis,
                                     sparse_factors_t *factors) {
    if (!A_new || !analysis || !factors)
        return SPARSE_ERR_NULL;

    if (A_new->rows != analysis->n || A_new->cols != analysis->n)
        return SPARSE_ERR_SHAPE;

    /* Factor into a temporary first so old factors survive on error */
    sparse_factors_t new_factors;
    memset(&new_factors, 0, sizeof(new_factors));
    sparse_err_t err = sparse_factor_numeric(A_new, analysis, &new_factors);
    if (err != SPARSE_OK)
        return err;

    sparse_factor_free(factors);
    *factors = new_factors;
    return SPARSE_OK;
}
