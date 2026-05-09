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
 * Default 128 from the Sprint 27 Day 3 relaxed-flip-rule re-sweep
 * across t ∈ {96, 128, 192, 256} under the new Sprint 27 Day 2 HCC
 * + Kuu-safe default coarsening — the maximum threshold satisfying
 * the relaxed flip rule (≥ 5 % Pres_Poisson wall improvement, no
 * fixture nnz_L regression past 2pp).  Result on Pres_Poisson: ND
 * wall 8 826 ms → 7 079 ms (-19.8 %) with nnz_L +0.5 % (within
 * 2pp).  Bonus Kuu nnz_L -1.1 % win.  See
 * `docs/planning/EPIC_2/SPRINT_27/nd_base_threshold_decision.md`
 * for the sweep matrix + relaxed-flip-rule application.
 *
 * Prior history: Sprint 26 Day 5 picked t=96 under a strict 1pp
 * cap (t=128 was rejected by s3rmt3m3 +1.05pp).  Sprint 22 Day 9's
 * original t=32 came from a sweep where the leaf path was natural
 * ordering; Sprint 23 spliced quotient-graph AMD into each leaf,
 * which changed the cost shape and let larger thresholds win.
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

#endif /* SPARSE_REORDER_ND_INTERNAL_H */
