# Sprint 24 Day 1 — qg-AMD wall-time fix-candidate decision

## Decision

**Candidate (c): revert Sprint 23 Days 2-5** (element absorption, supervariable detection, approximate-degree formula, dense-row skip).  Restores Sprint 22's variable-only quotient-graph baseline.  Implementation begins on Day 2.

## Profile evidence

`SPARSE_QG_PROFILE=1 build/bench_amd_qg --only bcsstk14` captured on Day 1 (full output in `profile_day1_bcsstk14.txt`):

| Phase                                          | Time (ms) | % of total |
|------------------------------------------------|-----------|-----------:|
| qg_init                                        |     0.5   |      0.0 % |
| Initial all-vertices supervariable scan (Day 4) |     0.6   |      0.0 % |
| qg_pick_min_deg total                          |    13.0   |      0.3 % |
| qg_eliminate total                             |  4 310.8  |     99.7 % |
| ↳ qg_merge_supervariables (per-pivot, Day 4)   |    41.9   |      1.0 % |
| ↳ qg_recompute_deg (Days 3-4 element-side walk) |  4 114.1  |     95.1 % |

`qg_recompute_deg` is **95.1 %** of total wall time on bcsstk14.  Day 4's supervariable detection — the suspect from `SPRINT_23/bench_summary_day12.md "(b)"` — contributes 1.0 % across all per-pivot calls.

## Why the original candidates miss

`bench_summary_day12.md "(b)"`'s three candidates all targeted Day 4's supervariable detection.  The profile shows it's not the bottleneck:

- **(a) sorted-list compare in supervariable detection** — caps the speedup at ~42 ms (1 % of total).  Best case: bcsstk14 4 325 ms → 4 283 ms.  20× short of the 1.5× Sprint 22 baseline target (~210 ms).
- **(b) regularity-heuristic gating** — same ceiling.  Same gap.
- **(c) revert Days 2-5 entirely** — restores Sprint 22's variable-only quotient graph.  No elements ⇒ no element-side walk in `qg_compute_deg_exact` ⇒ no per-pivot O(adj-of-adj) cost.  Sprint 22 measured bcsstk14 qg-AMD at ~140 ms; reverting closes the regression directly.

A new candidate the profile surfaces:

- **(d) optimise `qg_compute_deg_exact`'s element-side walk** (e.g. cache element variable-sets, dedup at element level instead of vertex level, skip subset elements).  Possible but unproven; would require rewriting the recompute_deg path that Sprint 23 took as given.

## Why (c) over (d)

1. **Risk profile.**  (c) is a pure revert against a known-good Sprint 22 baseline measured at 140 ms on bcsstk14.  (d) is novel optimisation work whose final wall time is unknown; it could land anywhere from 200 ms (acceptable) to 1 000 ms (still failing the gate).

2. **Sprint 24 budget shape.**  PROJECT_PLAN.md item 2's 32-hour budget assumes a structural change like (a) / (b) / (c).  An optimisation in (d) territory would more realistically need 50-80 hours plus extensive validation.  (c) lands in 1-2 days of revert + conflict resolution; the sprint's other 50 % budget can flow to items 4-5.

3. **Days 2-5's algorithmic features don't pay their wall-time freight.**  The profile shows:
   - Day 3 element absorption: enables the recompute_deg adjacency-of-adjacency walk (the bottleneck) — net negative.
   - Day 4 supervariable detection: ~1 % of total cost; on irregular SPD, ~0 supervariables form so it's pure overhead.
   - Day 5 approximate-degree: opt-in only (`SPARSE_QG_USE_APPROX_DEG`), Day 6 measured it 4.9× *slower* than exact-degree on bcsstk14, so it never fires in production.
   - Day 6 dense-row cap: piggybacks on Day 5; same opt-in path.

4. **Sprint 23's Pres_Poisson 0.95× ND/AMD outcome is independent of Days 2-5.**  Sprint 23's headline win was from Day 11's multi-pass FM at the finest level (1.026 → 0.952×, the largest single jump per `SPRINT_23/RETROSPECTIVE.md` "Performance highlights").  Days 2-5 are fill-neutral by construction — reverting them doesn't lose the 0.95× ratio.

5. **Sprint 23 RETROSPECTIVE.md's "What surprised us" already flagged this.**  The retro noted Days 2-5's wall-time cost wasn't visible in per-day fixture suites and surfaced only at Day 12 — i.e. the algorithmic features were a misallocation in retrospect.  Reverting respects the lessons-learned.

## What survives the revert

The revert touches `src/sparse_reorder_amd_qg.c` (and its internal header) but leaves Sprint 23 Days 1, 6, 7, 9-11 intact:

| Sprint 23 work | Survives revert? | Why |
|---|---|---|
| Day 1 SPD synth fixture | ✓ | `tests/test_reorder_nd.c` test, unrelated to qg-AMD internals |
| Day 6 cap_fired_count probe | ✗ | Tied to approximate-degree path |
| Day 7 leaf-AMD splice (`nd_recurse`) | ✓ | `src/sparse_reorder_nd.c`, Sprint 22 qg-AMD also satisfies the leaf-AMD-callable contract |
| Days 9-10 gain-bucket FM | ✓ | `src/sparse_graph.c` + `sparse_graph_fm_buckets.h`, no qg-AMD dependency |
| Day 11 multi-pass FM at finest level | ✓ | Headline driver of Pres_Poisson 0.95× win |

So the revert closes gate (b) without losing gate (a)'s spirit (Pres_Poisson ND beats AMD).

## Tests that change behaviour

- `tests/test_reorder_amd_qg.c::test_qg_supervariable_synthetic` — supervariable detection no longer fires; this test asserts contiguous-leaves placement, which won't hold under variable-only AMD.  Day 2 deletes or repurposes.
- `tests/test_reorder_amd_qg.c::test_qg_approx_degree_*` (Days 5 / 6 parity tests) — approximate-degree code path is gone; tests delete.
- `tests/test_reorder_amd_qg.c::test_qg_supervariable_synthetic_corpus` (Day 13) — same as `_synthetic`; delete.
- `tests/test_reorder_amd_qg.c::test_qg_approx_degree_parity_corpus` (Day 13) — approximate-degree path gone; delete.
- `tests/test_reorder_amd_qg.c::test_qg_dense_row_completion` (Day 6) — dense-row cap is gone; delete.

The test_reorder_amd_qg suite shrinks from 14 tests / 2 209 assertions to ~7 tests (the corpus delegation tests, stress test, and workspace-extension regression test all survive — though the workspace-extension test's specific iw_size assertion needs a 5·nnz+6·n+1 update for the Sprint 22 size).

## Item 4 status (Davis §7.5.1 external-degree refinement)

Item 4 is conditional on item 2 retaining the approximate-degree code path (i.e. (a) or (b)).  With (c) chosen, item 4 is **N/A** — the approximate-degree formula no longer exists to refine.  Per PROJECT_PLAN.md Day 6's note: "Skip Day 6 entirely if Day 1 chose fix candidate (c).  In that case, recover the 8-hour budget for an item 5 head-start (move Day 8's item-5 work into Day 6)."

The 20-hour Day 6 + Day 7 budget recovers as item 5 head-start.

## Item-3 status (Pres_Poisson AMD parity test)

Item 3 was already hedged in PROJECT_PLAN.md to "skip cleanly if item 2 chooses the (c) revert path".  With (c) chosen, item 3 reduces to a no-op — the approximate-degree code path is gone, no parity test to run.  Day 5's 8-hour budget recovers as further item 5 head-start, or as a buffer against item-2 revert overrun.

Net effect: the (c) revert frees ~28 hours from items 3 + 4 that get redirected to item 5 (more time for the Pres_Poisson ND/AMD ≤ 0.85× target) and item 7 (closing tests + retro), with a 4-hour revert/test-cleanup overrun cushion.

## Day 2 plan

1. Identify the four Sprint 23 commits to revert: `336d74a` (Day 2), `aea071a` (Day 3), `6096840` (Day 4), `f0fe391` (Day 5).  Day 6 (`db36001`) and Day 13 (`439c472`) added Sprint-23 tests + the cap_fired_count probe — those revert too where they touch `sparse_reorder_amd_qg.c` / `test_reorder_amd_qg.c`.
2. Use `git revert <commits>` in reverse chronological order; resolve conflicts against Sprint 24 Day 1's wall-check + profile instrumentation (the profile fields drop, the wall-check stays).
3. Delete the obsolete Sprint 23 tests (`test_qg_supervariable_synthetic*`, `test_qg_approx_degree_*`, `test_qg_dense_row_completion`).
4. Re-run `make test`; corpus delegation parity tests (nos4 / bcsstk04 / bcsstk14) must produce bit-identical fill (637 / 3 143 / 116 071).
5. Re-run `make wall-check`; bcsstk14 qg-AMD wall-time should land in the Sprint 22 ~140 ms band.  The Day 1 baseline (7 000 ms) is generous; Day 4 will tighten it after the full revert lands.
