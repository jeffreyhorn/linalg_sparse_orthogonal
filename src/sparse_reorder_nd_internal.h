#ifndef SPARSE_REORDER_ND_INTERNAL_H
#define SPARSE_REORDER_ND_INTERNAL_H

/**
 * @file sparse_reorder_nd_internal.h
 * @brief Nested-Dissection internal tunables — bench + test only.
 *
 * This header exists so in-tree benchmarks and unit tests have a
 * documented entry point for the otherwise file-local ND base-case
 * cutoff.  Two known consumers:
 *
 *   - `benchmarks/bench_reorder.c --nd-threshold N` (Sprint-22 Day-9
 *     threshold sweep).
 *   - `tests/test_reorder_nd.c` (drops the threshold around small
 *     fixtures so the partition step actually runs and the
 *     separator-last contract is exercised).
 *
 * The variable is *not* part of the public API and external library
 * consumers should not write to it.
 *
 * **Not thread-safe.**  The variable is a process-wide global; ND
 * calls running concurrently with a writer will see a torn /
 * inconsistent value.  Tests that mutate it should restore the
 * previous value on every exit path before the next test starts.
 *
 * **No ABI / API stability guarantee.**  The Sprint-23 follow-up
 * that splices quotient-graph AMD into each leaf is expected to
 * remove or rename this — at that point the threshold becomes a
 * real "stop recursing here, run AMD" cutover and the right
 * surface for tuning is an opts struct on `sparse_reorder_nd`
 * itself.
 */

#include "sparse_graph_internal.h"
#include "sparse_matrix.h"
#include "sparse_types.h"

/**
 * @brief ND base-case threshold (`n ≤ threshold` → leaf-AMD via
 *        `sparse_reorder_amd_qg`).
 *
 * Default 160 after Sprint 86 Day 6's reviewed-runtime re-sweep on
 * the current multilevel pipeline.  The bounded `bench_reorder
 * --skip-factor` sweep showed that t=160 materially reduces the
 * current ND reorder hotspot while preserving the present fill
 * contracts: Pres_Poisson 7 371.8 ms → 5 015.2 ms with nnz_L +0.5 %,
 * Kuu 5 972.7 ms → 2 964.4 ms with nnz_L -1.4 %, s3rmt3m3 4 896.7 ms
 * → 3 423.9 ms with nnz_L -0.6 %, and bcsstk14 464.6 ms → 377.5 ms
 * with nnz_L +1.7 %.  t=192 buys little extra runtime on
 * Pres_Poisson while pushing nnz_L higher there, so it remains an
 * opt-in rather than the default.
 *
 * Prior history: Sprint 27 Day 3 had raised the default to 128 under
 * a relaxed 2pp flip rule after Sprint 26 Day 5's strict-1pp t=96
 * choice. Sprint 22 Day 9's original t=32 came from a sweep where the
 * leaf path was natural ordering; Sprint 23 spliced quotient-graph
 * AMD into each leaf, which changed the cost shape and let larger
 * thresholds win.
 *
 * Per-fixture-class advisory: bimodal-degree solid-mechanics SPDs
 * (Kuu's CV=0.425 class) benefit monotonically from larger t —
 * t=256 produces -6.9 % nnz_L on Kuu vs t=96.  Such workloads can
 * opt in to a larger threshold via the `bench_reorder
 * --nd-threshold N` flag or by writing this variable directly.
 *
 * Defined in `src/sparse_reorder_nd.c`; declared here so in-tree
 * benches and tests can override it without an inline `extern`.
 *
 * @warning Internal bench/test hook only — not thread-safe; not
 *          part of the public API.
 */
extern idx_t sparse_reorder_nd_base_threshold;

/**
 * @brief Resolve the compatibility/default ND policy baseline.
 *
 * This is the shared internal owner for the legacy env-var compatibility path
 * and the internal default values used by both the direct
 * `sparse_reorder_nd(...)` path and the explicit analysis lifecycle.
 *
 * Callers that layer typed analysis options on top should start from this
 * baseline and then apply typed-field overrides, so typed values continue to
 * win over compatibility env vars exactly as shipped.
 */
sparse_graph_nd_policy_t sparse_reorder_nd_default_policy(void);

/**
 * @brief Policy-aware ND entry point used by `sparse_analyze(...)`.
 *
 * Preserves the public `sparse_reorder_nd(...)` compatibility surface while
 * allowing the explicit analysis lifecycle to pass resolved typed policy.
 */
sparse_err_t sparse_reorder_nd_with_policy(const SparseMatrix *A, idx_t *perm,
                                           const sparse_graph_nd_policy_t *policy);

/**
 * @brief Return whether ND profile tracing is currently enabled.
 *
 * The current-thread override wins when active; otherwise the legacy
 * `SPARSE_ND_PROFILE` compatibility env var controls the result.
 */
int sparse_reorder_nd_profile_current(void);

/**
 * @brief Override ND profile tracing for the current thread.
 *
 * Used by focused tests and internal call sites that need an explicit
 * precedence seam instead of ambient process env state. The begin/end calls
 * must be paired.
 */
void sparse_reorder_nd_profile_override_begin(int enabled);

/**
 * @brief Clear the current-thread ND profile override.
 */
void sparse_reorder_nd_profile_override_end(void);

#endif /* SPARSE_REORDER_ND_INTERNAL_H */
