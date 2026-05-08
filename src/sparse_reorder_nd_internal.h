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

#include "sparse_types.h"

/**
 * @brief ND base-case threshold (`n ≤ threshold` → leaf-AMD via
 *        `sparse_reorder_amd_qg`).
 *
 * Default 96 from the Sprint 26 Day 5 re-sweep across t ∈
 * {32, 48, 64, 96, 128} on the full corpus — the maximum
 * threshold satisfying the PROJECT_PLAN flip rule (≥ 5 %
 * Pres_Poisson wall improvement, no fixture nnz_L regression
 * past 1pp).  Result on Pres_Poisson: ND wall 38.1 s → 12.2 s
 * (-67.9 %) with nnz_L bit-stable (-0.21pp); see
 * `docs/planning/EPIC_2/SPRINT_26/nd_base_threshold_decision.md`
 * for the sweep matrix + flip-rule application.
 *
 * Sprint 22 Day 9's earlier default of 32 came from a sweep where
 * the leaf path was natural ordering (no AMD); Sprint 23 spliced
 * quotient-graph AMD into each leaf, which changed the cost shape
 * and let larger thresholds win.
 *
 * Defined in `src/sparse_reorder_nd.c`; declared here so in-tree
 * benches and tests can override it without an inline `extern`.
 *
 * @warning Internal bench/test hook only — not thread-safe; not
 *          part of the public API.
 */
extern idx_t sparse_reorder_nd_base_threshold;

#endif /* SPARSE_REORDER_ND_INTERNAL_H */
