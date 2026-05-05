# Sprint 24 Plan: Ordering Follow-Ups (Sprint 23 Deferrals)

**Sprint Duration:** 14 days
**Goal:** Close the qg-AMD wall-time regression Sprint 23 introduced (62-199× vs Sprint 22 quotient-graph baseline on irregular SuiteSparse SPD; gate (b) hard-fail in `docs/planning/EPIC_2/SPRINT_23/bench_summary_day12.md`), tighten the Pres_Poisson ND/AMD ratio toward the literal Sprint 22 plan target (Sprint 23 landed at 0.952×; this sprint targets ≤ 0.85×), close the Sprint 23 Day 13 deferral that left the AMD parity test at bcsstk14 only (Pres_Poisson skipped because USE_APPROX would push the suite past 30 minutes on the pre-fix wall-time profile), and add per-day wall-time regression-check infrastructure to prevent similar regressions in future sprints.  Also lands the Davis 2006 §7.5.1 external-degree refinement (deferred from Sprint 23 Day 5) if Sprint 23's approximate-degree path is retained.

**Starting Point:** Sprint 23 (PR #31, merged at `3e2cba3`) shipped: the full Davis 2006 quotient-graph AMD with element absorption + supervariable detection + approximate-degree formula + dense-row skip (`src/sparse_reorder_amd_qg.c` Days 2-6); the per-leaf AMD splice in `nd_recurse`'s base case (`src/sparse_reorder_nd.c` Day 7); the FM gain-bucket structure (`src/sparse_graph_fm_buckets.h` + integration in `src/sparse_graph.c` Days 9-10); 3-pass FM at the finest uncoarsening level by default (Day 11); and the `make wall-check`-shaped bench reference at `bench_amd_qg.c` plus per-day captures (`docs/planning/EPIC_2/SPRINT_23/bench_*.{csv,txt}`).  Sprint 23's `bench_summary_day12.md` documents the headline outcomes: gate (a) Pres_Poisson ND/AMD = 0.952× (literal ≤ 0.7× missed but spirit met — ND beats AMD); gate (b) qg-AMD wall on bcsstk14 = 6 951 ms = 108× the Sprint 22 bitset baseline (HARD FAIL); gate (c) `bench_day14` nnz_L bit-identical-or-better on every fixture (PASS).  Sprint 24 inherits gates (a) and (b) per `SPRINT_23/RETROSPECTIVE.md` "Sprint 24 inputs".

**End State:** `sparse_reorder_amd_qg` runs with one of three candidate fixes from `SPRINT_23/bench_summary_day12.md "(b)"` — sorted-list compare in supervariable detection, regularity-heuristic gating, or a clean revert of Sprint 23 Days 2-5 — with wall time on bcsstk14 ≤ 1.5× of Sprint 22's quotient-graph baseline (~210 ms ceiling).  Fill correctness stays bit-identical to Sprint 22 + Sprint 23 across the full corpus + synthetic banded.  If the chosen fix retains the approximate-degree code path, the Davis 2006 §7.5.1 external-degree refinement either ships as default (≤ 5 % nnz divergence vs exact) or behind `SPARSE_QG_USE_EXTERNAL_DEG`.  ND fill on Pres_Poisson reaches ≤ 0.85× of AMD via deeper coarsening or smarter separator extraction.  A `make wall-check` target catches > 2× per-day wall-time regressions before commit.  `tests/test_reorder_amd_qg.c::test_qg_approx_degree_parity_corpus` extends to Pres_Poisson once item 2's wall-time fix lands.  Cross-corpus re-bench captures land in `docs/planning/EPIC_2/SPRINT_24/bench_*.{csv,txt}` plus a `bench_summary_day14.md` headline-gate verdict.  `SPRINT_24/RETROSPECTIVE.md` ships stubbed for the post-sprint write-up; `docs/algorithm.md` AMD subsection updates to reflect whatever Days 2-5 logic survived item 2; `SPRINT_22/PERF_NOTES.md` gets a "Sprint 24 closures" subsection.

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to ~130 hours — about 4 hours above the 126-hour PROJECT_PLAN.md estimate, providing a similar safety buffer to Sprints 22 / 23 (which estimated 124 / 88 hrs and shipped at ~134 / ~80 hrs).  Risk concentration is item 2 (32 hrs across Days 1-4): it touches a load-bearing kernel that every Cholesky / LDL^T / LU / QR call routes through and gates the affordability of items 3 + 4.  Item 5 (32 hrs across Days 8-11) is the Pres_Poisson ND/AMD ratio close — independent of item 2, can be reshuffled if item 2 over-runs.

---

## Day 1: Sprint Kickoff — Wall-Check Instrumentation & qg-AMD Profile

**Theme:** Land the `make wall-check` target up front so Days 2-5's fix-and-validate iterations run with the regression gate active, then profile `sparse_reorder_amd_qg` on bcsstk14 / Pres_Poisson under the Sprint 23 default path to pick the right fix candidate from `bench_summary_day12.md "(b)"`.

**Time estimate:** 10 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_23/RETROSPECTIVE.md` "Sprint 24 inputs" + `bench_summary_day12.md` "(b)" + Sprint 23 Day 6's `davis_notes.md` wall-time finding.  Pin the Sprint 22 quotient-graph baseline (~140 ms on bcsstk14) and the Sprint 23 default-path measurement (~6 951 ms) as the regression band item 2 has to close.
2. Implement `make wall-check` target in the top-level `Makefile`.  It runs `build/bench_amd_qg --only bcsstk14` + `build/bench_reorder --only Pres_Poisson --skip-factor`, parses the `reorder_ms` column from each CSV via `awk`, and exits non-zero if either bcsstk14 qg-AMD wall-time or Pres_Poisson AMD wall-time is > 2× a baseline file at `docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt` (committed alongside the Makefile change).  Initial baseline values: bcsstk14 qg-AMD = 7 000 ms (Sprint 23 ceiling, will fall as item 2 lands); Pres_Poisson AMD = 1 100 000 ms (Sprint 23 worst-case, also falls).  Document the gate semantics in a new `### Performance regression gates` subsection in `docs/algorithm.md` that explains the > 2× threshold + the same-machine-class caveat.
3. Profile `sparse_reorder_amd_qg` on bcsstk14 under the Sprint 23 default path.  Use `clock_gettime(CLOCK_MONOTONIC, ...)` instrumentation around the qg_init → elimination loop → qg_free flow, with per-pivot-step samples logged to stderr behind `SPARSE_QG_PROFILE=1`.  Capture flame-graph-style aggregates (cumulative time per phase: hash compute, supervariable compare, approximate-degree, element absorption).  Save the profile output to `docs/planning/EPIC_2/SPRINT_24/profile_day1_bcsstk14.txt` for the Day 2 fix-selection commit message to cite.
4. Pick the fix candidate from `bench_summary_day12.md "(b)"`: (a) sorted-list compare, (b) regularity heuristic gating, or (c) revert Days 2-5 entirely.  Default expectation: (a) wins if the supervariable hash compare is the dominant phase per the profile (likely; Days 2-5 added that step); (b) wins if the supervariable detection is mostly overhead (no merges firing on irregular SPD); (c) wins if the per-pivot cost is uniformly distributed across Days 2-5's additions.  Document the decision + the supporting profile evidence in `docs/planning/EPIC_2/SPRINT_24/fix_decision_day1.md`.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `make wall-check` target landed in `Makefile` with baseline file at `docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt`
- `docs/algorithm.md` "Performance regression gates" subsection
- `docs/planning/EPIC_2/SPRINT_24/profile_day1_bcsstk14.txt` profile capture
- `docs/planning/EPIC_2/SPRINT_24/fix_decision_day1.md` candidate selection + rationale
- All quality checks clean

### Completion Criteria
- `make wall-check` runs on the current tip of master and exits 0 (baseline is set to current measurement; future > 2× regressions fail the gate)
- Profile capture identifies a single dominant phase or set of phases (≥ 60 % of total wall time) so the Day 2 fix has a clear target
- Fix-decision doc names the chosen fix candidate (a / b / c) with the profile-derived rationale
- `make format && make lint && make test && make wall-check` clean

---

## Day 2: qg-AMD Wall-Time Fix — Implementation Start

**Theme:** Begin implementing the fix candidate chosen on Day 1.  Land the structural change (data-structure swap, gating heuristic, or revert sequence) but defer end-to-end validation to Day 3 — Day 2's gate is "library + tests build, no fill regression on smoke fixtures".

**Time estimate:** 10 hours

### Tasks
1. Implement the fix-candidate's structural change in `src/sparse_reorder_amd_qg.c`:
   - **Candidate (a) sorted-list compare:** replace the supervariable hash bucket's pairwise full-list compare with a sorted-list compare.  Within each hash bucket, sort by adjacency-list length first (cheap pre-filter — different lengths can't match), then for equal-length pairs do a lexicographic compare on the sorted adjacency.  Adjacency lists are already sorted by `qg_compact`'s invariant, so the compare is O(k) per pair instead of O(k²).
   - **Candidate (b) regularity gating:** add a startup-time regularity probe that samples N vertices' adjacency-list length distribution.  If standard deviation / mean exceeds a threshold (e.g., 0.5 — irregular structure with poor supervariable potential), skip Day 4's supervariable detection entirely and fall through to the Sprint 22 quotient-graph code path.
   - **Candidate (c) revert Days 2-5:** `git revert` the four sprint commits (`336d74a`, `aea071a`, `6096840`, `f0fe391`); resolve any conflicts with the bench-summary captures in `SPRINT_23/`; verify the test suite still passes.
2. Run the existing `tests/test_reorder_amd_qg.c` suite — fill should stay bit-identical (the fix shouldn't change pivot order on the corpus delegation tests).  If a test diverges, root-cause before moving on.
3. Capture an interim wall-time delta on bcsstk14 via `make wall-check` to gauge progress against the 1.5× Sprint-22-baseline ceiling.  If the delta is > 50 % of the way there (Sprint 23 default 6 951 ms → < 4 000 ms), the fix is on track; if it's < 25 % (still > 5 000 ms), pause and reconsider on Day 3.
4. Commit with a message that cites Day 1's profile + fix-decision doc.
5. Run `make format && make lint && make test && make wall-check` (the wall-check baseline file from Day 1 stays at the Sprint 23 ceilings; this commit may already pass the > 2× gate at the lower end, leaving room for later regressions to still trigger).

### Deliverables
- Day 2 commit implementing the structural change for the chosen fix candidate
- Existing `tests/test_reorder_amd_qg.c` corpus parity tests still pass (bit-identical fill on nos4 / bcsstk04 / bcsstk14)
- Interim bcsstk14 wall-time measurement captured (stderr or `docs/planning/EPIC_2/SPRINT_24/wall_day2_bcsstk14.txt`)
- All quality checks clean

### Completion Criteria
- Library compiles, full test suite passes, no fill regression on the corpus delegation tests
- `make wall-check` exits 0 against the Day 1 baseline
- bcsstk14 qg-AMD wall time has dropped at least 25 % from the Sprint 23 default measurement (else escalate on Day 3)

---

## Day 3: qg-AMD Wall-Time Fix — Polish & Synthetic-Banded Validation

**Theme:** Complete the fix's correctness work — bit-identical fill across the full corpus + synthetic banded, no determinism drift, no behaviour change behind `SPARSE_QG_USE_APPROX_DEG`.  Day 2 was structural; Day 3 cleans up edge cases the structural change exposed.

**Time estimate:** 10 hours

### Tasks
1. Run `tests/test_reorder_amd_qg.c::test_qg_approx_degree_parity_200` and `test_qg_approx_degree_upper_bound` to confirm `d_approx ≥ d_exact` still holds on the synthetic 50-vertex / 200-vertex fixtures under `SPARSE_QG_VERIFY_DEG=1`.  If candidate (a) was chosen and the sorted-list compare's tie-break differs from the hash-then-pairwise version, the merged supervariable IDs may differ — but `d_approx` should still bound `d_exact`.  If the bound breaks, root-cause before moving on.
2. Run `benchmarks/bench_amd_qg.c` (full corpus + synthetic banded) and verify nnz(L) is bit-identical to Sprint 23's `bench_day14_amd_qg.csv`.  If a single-fixture drift appears, decide: accept it as tie-breaking noise (≤ 1 nnz on a single fixture) or root-cause as a real regression.
3. For candidate (a) only: add a sorted-list-compare unit test in `tests/test_reorder_amd_qg.c` (`test_qg_supervariable_sorted_compare`) that builds a fixture with two vertices having identical adjacency in different insertion orders, verifies they merge into one supervariable.  For candidate (b): add `test_qg_regularity_gate_fires` that builds a regular fixture and verifies the gate enables supervariable detection, plus a heterogeneous fixture that verifies the gate disables it.  For candidate (c): no new test — the revert restores Sprint 22 behaviour, the existing parity tests cover it.
4. Re-run `make wall-check` and confirm bcsstk14 qg-AMD wall ≤ Sprint 22 baseline × 1.5 (~210 ms ceiling).  If still over, escalate decision: extend Day 4 by 4 hours into Day 5's slot, push item 3 to Day 6.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- Approximate-degree parity contracts pass on synthetic 50/200-vertex fixtures
- Full corpus + synthetic banded nnz(L) bit-identical to Sprint 23 (single-fixture ≤ 1 nnz drift acceptable if tie-break-traced)
- New unit test for the chosen fix candidate (a / b only)
- bcsstk14 qg-AMD wall time at or below 1.5× Sprint 22 baseline
- All quality checks clean

### Completion Criteria
- `make test` covers the new fix-candidate unit test
- bcsstk14 qg-AMD wall ≤ 210 ms (1.5 × Sprint 22 baseline of 140 ms)
- nnz(L) parity intact across SuiteSparse + synthetic banded
- `make format && make lint && make test && make wall-check` clean

---

## Day 4: qg-AMD Wall-Time Fix — Pres_Poisson Validation & Update wall-check Baseline

**Theme:** Confirm the fix scales to the largest fixture (Pres_Poisson, n = 14 822) before committing to the new wall-time baseline.  Update `wall_check_baseline.txt` to the post-fix measurement so Days 5-12 work against a tighter gate.

**Time estimate:** 8 hours

### Tasks
1. Run `build/bench_reorder --only Pres_Poisson --skip-factor` end-to-end and capture the AMD wall-time + nnz(L).  Sprint 23 default measured ~22 minutes (1 339 904 ms); fix target is ≤ 1.5× Sprint 22 baseline (~12 200 × 1.5 ≈ 18 300 ms = 18 seconds) — a ~70× speedup over Sprint 23.  Even partial closure (≤ 60 000 ms = 1 minute, ~22× speedup) is acceptable as a Day 4 gate; the > 2× regression-check gate doesn't require hitting the literal target, just preventing further drift.
2. Verify Pres_Poisson AMD nnz(L) = 2 668 793 bit-identical to Sprint 22 + Sprint 23 captures.  Mismatch means a real regression — root-cause before moving on.
3. Update `docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt` with the new bcsstk14 + Pres_Poisson AMD wall times.  Include a comment block at the top of the file explaining: which day landed the baseline, which fix candidate produced it, and what the previous values were (so future Sprint 24 days catch a regression by comparing to *the new* tighter ceiling, not the Sprint 23 worst-case).
4. Run `benchmarks/bench_amd_qg.c` full bench (qg vs bitset on the entire corpus + synthetic banded).  Capture to `docs/planning/EPIC_2/SPRINT_24/bench_day4_amd_qg.{csv,txt}`.  Sanity-check that the Sprint 22 Day 13 wall-time band (qg ≤ 1.8× bitset on SuiteSparse, qg ≥ 4× faster on banded n ≥ 5 000) is restored.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- Pres_Poisson AMD wall-time measured + captured
- Updated `wall_check_baseline.txt` reflecting post-fix ceilings
- `docs/planning/EPIC_2/SPRINT_24/bench_day4_amd_qg.{csv,txt}` post-fix qg-vs-bitset capture
- All quality checks clean

### Completion Criteria
- bcsstk14 qg-AMD ≤ 1.5× Sprint 22 baseline (= ≤ 210 ms)
- Pres_Poisson qg-AMD ≤ Sprint 23 default × 0.5 (= ≤ 670 000 ms; partial closure acceptable, full closure to ≤ 18 000 ms preferred)
- nnz(L) bit-identical to Sprint 22 + Sprint 23 across full corpus + synthetic banded
- New `wall_check_baseline.txt` committed
- `make format && make lint && make test && make wall-check` clean

---

## Day 5: AMD Parity Test on Pres_Poisson

**Theme:** Close Sprint 23 Day 13's deferral that left the AMD parity test at bcsstk14 only.  With Day 4's wall-time fix landed, USE_APPROX on Pres_Poisson should run in a few minutes instead of the 30-minute pre-fix profile.

**Time estimate:** 8 hours

### Tasks
1. Extend `tests/test_reorder_amd_qg.c::test_qg_approx_degree_parity_corpus` to add a Pres_Poisson branch.  Set both `SPARSE_QG_VERIFY_DEG=1` and `SPARSE_QG_USE_APPROX_DEG=1` (matching Sprint 23 Day 13's bcsstk14 pattern), load `tests/data/suitesparse/Pres_Poisson.mtx`, run `sparse_reorder_amd_qg`, and validate the resulting permutation.  Re-use the existing `env_snapshot_t` cleanup pattern; skip cleanly if Day 1's chosen fix candidate was (c) revert.
2. Time the new test under `make test`.  Target: full `tests/test_reorder_amd_qg.c` suite under 5 minutes total (Sprint 23 with bcsstk14-only ran at ~30 s; Pres_Poisson under USE_APPROX adds the bulk).  If over 10 minutes, gate the Pres_Poisson branch behind `SPARSE_QG_LONG_TESTS=1` env var so default `make test` stays fast.
3. Verify the conservative-bound contract (`d_approx ≥ d_exact`) holds across Pres_Poisson's full elimination — the per-pivot assert in `qg_recompute_deg` fires under `SPARSE_QG_VERIFY_DEG`.  Debug build catches any underestimate; release build still validates the call doesn't crash.
4. Update `tests/test_reorder_amd_qg.c::test_qg_approx_degree_parity_corpus`'s `printf` to log Pres_Poisson's pivot count + total wall time alongside the existing bcsstk14 message.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `test_qg_approx_degree_parity_corpus` extended to Pres_Poisson
- Test wall time documented (gated behind `SPARSE_QG_LONG_TESTS=1` if > 10 minutes)
- All quality checks clean

### Completion Criteria
- Pres_Poisson conservative-bound parity contract verified (assert holds across the full elimination)
- `tests/test_reorder_amd_qg.c` suite total time ≤ 5 minutes (or ≤ 30 s with `SPARSE_QG_LONG_TESTS=0` and ≤ 10 minutes with `=1`)
- Test fixture's hardcoded skip path on `(c) revert` returns cleanly (the approximate-degree code path is gone, no test to run)
- `make format && make lint && make test && make wall-check` clean

---

## Day 6: Davis 2006 §7.5.1 External-Degree Refinement — Design & Implementation Start

**Theme:** Conditional on Day 1 choosing fix candidate (a) or (b) — i.e. the approximate-degree code path is retained.  Davis 2006 §7.5.1 describes an "external degree" refinement that tightens the approximate-degree formula's bound by tracking which neighbours are external to the pivot's element-set vs internal.  Day 6 implements the data-structure side; Day 7 wires it into the formula and validates.

**Time estimate:** 8 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_23/davis_notes.md` "Day-1 reading" section that flagged this as an unimplemented refinement, plus Davis 2006 §7.5.1 itself.  Capture the refinement's contract: for each variable adjacent to the pivot, distinguish "external" (variable adjacent to a *different* element from the pivot's) vs "internal" (variable in the pivot's element-set only).  The refinement walks the pivot's element variable-set once per pivot to flag externals; the approximate-degree formula then sums only external contributions instead of all element-side adjacency.
2. Add the data-structure plumbing to `src/sparse_reorder_amd_qg.c`: a per-vertex `idx_t external_deg[]` slot in `qg_t`, allocated in `qg_init` and freed in `qg_free`.  Bump `iw_size` only if the new array fits inside the existing buffer; otherwise allocate it standalone (cheaper than reshuffling the iw layout).
3. Sketch the per-pivot walk in a comment block: pivot p with element-set V(p); for each variable v ∈ V(p), walk v's element-side (`elen[v]` entries); for each element e ≠ p in v's elements, mark v as external (sets `external_deg[v] = 1` boolean flag, or a counter if Davis describes a multi-element refinement).  No code yet — design block only.
4. Add a `SPARSE_QG_USE_EXTERNAL_DEG` env-var gate at the entry of `qg_compute_deg_approx` that switches between the Sprint 23 default (counts all element-side adjacency as external) and the new refinement.  Off by default — Day 7 measures the impact before flipping the default.
5. Skip Day 6 entirely if Day 1 chose fix candidate (c).  In that case, recover the 8-hour budget for an item 5 head-start (move Day 8's item-5 work into Day 6).  Record the skip + the budget shuffle in `docs/planning/EPIC_2/SPRINT_24/PLAN.md` as an inline note.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `qg_t::external_deg` slot allocated + freed
- Design block describing the per-pivot walk + flagging rule
- `SPARSE_QG_USE_EXTERNAL_DEG` env-var gate at `qg_compute_deg_approx` entry (off by default)
- All quality checks clean

### Completion Criteria
- Library compiles; existing tests pass (the gate's off-by-default branch is bit-identical to Sprint 23 behaviour)
- Design block describes how `external_deg[]` is populated per pivot and consumed in the formula
- If Day 1 chose (c) revert: this day is skipped + the budget shuffle is documented inline
- `make format && make lint && make test && make wall-check` clean

---

## Day 7: External-Degree Refinement — Wire Into Formula & Validate

**Theme:** Plug the Day-6 data-structure work into `qg_compute_deg_approx`'s formula, validate against the conservative-bound contract on synthetic + corpus fixtures, and decide whether to promote external-degree to default.

**Time estimate:** 12 hours

### Tasks
1. Implement the per-pivot walk in `qg_eliminate`: after the pivot's element is recorded but before the next pivot's degree-recompute pass, walk the pivot's element-set; for each variable v in the set, walk v's element-side and set `external_deg[v]` according to the §7.5.1 refinement.  Cost: O(|element_set(p)| × max_elen) per pivot.
2. Update `qg_compute_deg_approx` (under `SPARSE_QG_USE_EXTERNAL_DEG=1`) to consult `external_deg[]` instead of summing all element-side adjacency.  The formula becomes `d_approx(i) = |adj(i, V)| + Σ_{e ∈ adj(i, E)} external_deg_in_e_excluding_pivot`.
3. Run `tests/test_reorder_amd_qg.c::test_qg_approx_degree_upper_bound` (50-vertex synthetic) and `test_qg_approx_degree_parity_200` (200-vertex synthetic) under `SPARSE_QG_USE_EXTERNAL_DEG=1` + `SPARSE_QG_VERIFY_DEG=1`.  The conservative-bound contract (`d_approx ≥ d_exact`) must still hold — the refinement should produce a *tighter* upper bound, not a wrong one.  If `d_approx < d_exact` ever fires, root-cause; the refinement's correctness is the gate.
4. Measure pivot-order divergence vs the Sprint 23 default path (no external-degree) on bcsstk14 + Pres_Poisson.  Capture the resulting nnz(L); if external-degree gives a tighter ratio (≤ 5 % nnz(L) difference vs exact-degree), promote to default by flipping `SPARSE_QG_USE_EXTERNAL_DEG`'s default to "on".  Otherwise document the refinement as available behind the env var and leave default off.  Record the decision + numbers in `docs/planning/EPIC_2/SPRINT_24/external_deg_decision.md`.
5. Add a unit test `test_qg_external_degree_tighter_bound` in `tests/test_reorder_amd_qg.c` that builds a fixture with two distinct elements and verifies `external_deg[]` gets set correctly under the env var.  Skip if Day 1 chose (c) revert (this whole day is N/A).
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- External-degree walk integrated into `qg_eliminate`
- Approximate-degree formula updated under the env-var gate
- Pivot-order divergence measurement + decision doc
- New unit test pinning the external-degree contract
- All quality checks clean

### Completion Criteria
- Conservative-bound contract `d_approx ≥ d_exact` holds across all synthetic + corpus parity tests under `SPARSE_QG_USE_EXTERNAL_DEG=1`
- Decision doc records: kept off / promoted to default, with the per-fixture pivot-order numbers
- `tests/test_reorder_amd_qg.c::test_qg_external_degree_tighter_bound` passes (or is skipped on revert path)
- `make format && make lint && make test && make wall-check` clean

---

## Day 8: ND Fill-Quality Follow-Up — Design & Coarsening Floor Sweep

**Theme:** Sprint 23 Day 11's multi-pass FM landed Pres_Poisson at 0.952× of AMD; closing 0.95 → 0.85 needs deeper algorithmic work.  Day 8 evaluates option (a): deepen coarsening — current bottoms out at MAX(20, n/100), try MAX(20, n/200) or a fixed coarsening floor of 50.

**Time estimate:** 10 hours

### Tasks
1. Read `src/sparse_graph.c::sparse_graph_hierarchy_build` and the coarsening-stop heuristic.  Currently the hierarchy stops when `n_coarsest ≤ MAX(20, n_orig / 100)` — that's 148 vertices for Pres_Poisson (n_orig = 14 822).  Sprint 22 chose the n/100 ratio empirically; this sprint sweeps tighter floors to see if a smaller coarsest level produces a better bisection that propagates back through the FM uncoarsening to a tighter cut at the finest level.
2. Add a `SPARSE_ND_COARSEN_FLOOR_RATIO` env var that overrides the n/100 divisor.  Default 100 (current); sweep {200, 400, 800, ∞} via a small script that runs `sparse_reorder_nd` on Pres_Poisson and captures both nnz(L) and wall time.  ∞ means "always coarsen down to MAX(20)" — the most aggressive setting.  Save the sweep capture to `docs/planning/EPIC_2/SPRINT_24/coarsen_floor_sweep_pres_poisson.txt`.
3. Repeat the sweep on bcsstk14 + Kuu + nos4 to make sure the chosen ratio doesn't regress smaller fixtures.  bcsstk14 currently lands at 1.130× AMD; nos4 at 1.520×; Kuu at 2.275×.  Acceptable: any of these can drift up to 0.05× without flagging, since the Pres_Poisson close is the headline gate.
4. Pick the best ratio (or "best of n/200, n/400") that drops Pres_Poisson ND/AMD ≤ 0.90× without regressing the smaller fixtures past the noise threshold.  If no ratio reaches ≤ 0.90×, mark option (a) insufficient and prepare for Day 9's option (b) work.
5. Commit the env-var override (default unchanged) so Day 9 can build on top of it.  Record the sweep results + the ratio choice in `docs/planning/EPIC_2/SPRINT_24/nd_coarsen_floor_decision.md`.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `SPARSE_ND_COARSEN_FLOOR_RATIO` env var implemented
- Coarsen-floor sweep capture for Pres_Poisson + bcsstk14 + Kuu + nos4
- Decision doc recording the chosen ratio + per-fixture nnz(L) deltas
- All quality checks clean

### Completion Criteria
- Sweep capture documents 4+ ratio settings × 4 fixtures = 16+ measurements
- Decision doc names the winning ratio (or marks option (a) insufficient + escalates to Day 9 option (b))
- Pres_Poisson ND/AMD ≤ 0.90× under the chosen ratio (preferred) or ≤ 0.93× as a partial-close fallback
- `make format && make lint && make test && make wall-check` clean

---

## Day 9: ND Fill-Quality Follow-Up — Smarter Separator Extraction

**Theme:** Sprint 22 Day 4's edge-to-vertex separator extraction lifts the *smaller-side* boundary to the separator (METIS convention).  Day 9 explores option (b): a balanced-cost variant that lifts whichever side has the smaller *boundary* regardless of weight.

**Time estimate:** 12 hours

### Tasks
1. Read `src/sparse_graph.c::edge_to_vertex_separator` (Sprint 22 Day 4).  Current implementation: of the two sides of the edge cut, pick the one with smaller vertex weight, and lift its boundary vertices into the separator.  This minimises post-extraction imbalance but doesn't necessarily minimise *separator size* — if the smaller-weight side has a larger boundary, the separator gets bloated.
2. Implement a `SPARSE_ND_SEP_LIFT_STRATEGY` env-var-gated alternative: lift the side with smaller boundary count, regardless of vertex weight.  Track both sides' boundary count during the cut walk.  If the imbalance penalty is too high (post-lift ratio > 70/30 say), fall back to the smaller-weight strategy.
3. Run the cross-corpus bench (`bench_reorder.c`) under `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary` on every fixture.  Capture to `docs/planning/EPIC_2/SPRINT_24/sep_strategy_sweep.txt`.  Compare against the Day 8 result + Sprint 23 baseline.
4. Combine option (a) + (b): set both env vars and re-run on Pres_Poisson.  If the combined effect drops Pres_Poisson ND/AMD ≤ 0.85×, the sprint's stretch target is met; otherwise document the gap as Sprint-25 territory.
5. Pick the production default: keep the chosen Day 8 ratio; flip `SPARSE_ND_SEP_LIFT_STRATEGY` to "balanced_boundary" if it's a clear win on Pres_Poisson (≤ 0.90×) without regressing the smaller fixtures past 5 percentage points; else keep it off.  Record the decision in `docs/planning/EPIC_2/SPRINT_24/nd_sep_strategy_decision.md`.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `SPARSE_ND_SEP_LIFT_STRATEGY` env var implemented
- Sep-strategy sweep capture across the full corpus
- Combined Day 8 + Day 9 measurement on Pres_Poisson
- Decision doc + chosen production default
- All quality checks clean

### Completion Criteria
- Pres_Poisson ND/AMD ≤ 0.90× under combined Day 8 + Day 9 settings (target: ≤ 0.85×)
- bcsstk14, Kuu, nos4 ratios each within 5 percentage points of their Sprint 23 baselines
- `make format && make lint && make test && make wall-check` clean

---

## Day 10: ND Fill-Quality Follow-Up — Tuning & Determinism Re-Check

**Theme:** With Days 8-9's options characterised, tune the chosen production defaults for the smallest-fixture-regression / largest-Pres_Poisson-win trade-off, and verify the `test_partition_determinism_*` contracts still hold under the new strategies.

**Time estimate:** 8 hours

### Tasks
1. If Day 9's combined setting reached ≤ 0.85× on Pres_Poisson, this day reduces to verification: re-run all 39 `tests/test_graph.c` partition tests under the chosen defaults, confirm separator-size ranges still hold + determinism contracts still pass bit-identically.  Record any per-fixture drift in `docs/planning/EPIC_2/SPRINT_24/nd_tuning_day10.md`.
2. If Day 9 didn't reach ≤ 0.85×, spend the day exploring narrow tunings: combine the two options with a 3-pass FM at the *second-finest* level (currently single-pass), or vary the coarsening seed RNG to see if a different deterministic seed produces a tighter cut on Pres_Poisson.  Time-box at 6 hours; if no further win, accept the Day-9 gap and document.
3. Tighten the Pres_Poisson nnz_nd fixture-pin in `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` from the Sprint 23 `≤ 1.0× nnz_amd` to `≤ 0.85× nnz_amd` (or whatever the achieved ratio + 2-percentage-point noise margin allows).  Note inline that the bound was tightened by Sprint 24 with cross-references to Day 9's decision doc.
4. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- All 39 partition tests pass under chosen Day 8 + Day 9 defaults
- Determinism contracts (`test_partition_determinism_*`) pass bit-identically
- Tightened Pres_Poisson nnz fixture-pin
- Tuning notes captured in `docs/planning/EPIC_2/SPRINT_24/nd_tuning_day10.md`
- All quality checks clean

### Completion Criteria
- `tests/test_graph.c` 39/39 pass with the new defaults
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` asserts the tightened bound and passes
- `make format && make lint && make test && make wall-check` clean

---

## Day 11: ND Fill-Quality Follow-Up — Close & Document

**Theme:** Lock in the ND fill-quality win with the partition fixture-pin + algorithm.md updates.  Bench the 10×10 grid to make sure the corpus-wide effect is measurable on small fixtures too.

**Time estimate:** 6 hours

### Tasks
1. Tighten `tests/test_reorder_nd.c::test_nd_10x10_grid_matches_or_beats_amd_fill`'s bound from Sprint 23's `≤ 1.21× nnz_amd` to whatever Days 8-9's combined effect achieves on this 100-vertex fixture (likely 1.10×-1.15× — the 10×10 grid's small size limits how much coarsening / separator-strategy changes can move it).
2. Update `docs/algorithm.md`'s ND subsection to describe the new coarsening-floor + separator-extraction options.  Cross-reference Days 8-10's decision docs.  Drop the Sprint 23 caveat that called out the deferred 0.7× target — Sprint 24 closes most of the gap.
3. Run the full `tests/test_reorder_nd.c` suite + `tests/test_graph.c` under all combinations of the new env vars set / unset to confirm no test regresses across the matrix.
4. Capture the Day-11 wall-time on Pres_Poisson ND under the new defaults.  Goal: ≤ Sprint 23's 36.4 s (Day 14's measurement).  Days 8-9's algorithmic wins shouldn't regress wall time meaningfully — coarsening to a smaller floor adds a fixed overhead, but the tighter cut typically reduces downstream work.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- Tightened 10×10 grid bound in `tests/test_reorder_nd.c`
- Updated `docs/algorithm.md` ND subsection
- Pres_Poisson ND wall-time under new defaults captured
- All quality checks clean

### Completion Criteria
- All 12 tests in `tests/test_reorder_nd.c` pass with tightened bounds
- All 39 tests in `tests/test_graph.c` pass under all env-var combinations
- Pres_Poisson ND wall ≤ Sprint 23 baseline + 5 % drift
- `make format && make lint && make test && make wall-check` clean

---

## Day 12: Cross-Corpus Re-Bench

**Theme:** Re-run `benchmarks/bench_reorder.c` and `benchmarks/bench_amd_qg.c` against the Sprint 22 + Sprint 23 baselines.  Capture nnz(L) and wall time for every fixture × ordering combination.  This is the headline measurement day.

**Time estimate:** 12 hours

### Tasks
1. Run `benchmarks/bench_reorder.c` against the full SuiteSparse corpus (nos4, bcsstk04, Kuu, bcsstk14, s3rmt3m3, Pres_Poisson) with all five orderings (NONE / RCM / AMD / COLAMD / ND).  Capture stdout to `docs/planning/EPIC_2/SPRINT_24/bench_day12.csv` and a human-readable rendering to `bench_day12.txt`.  Use the same `column -t -s,` pipe Sprint 22 / 23 used so visual diffs are clean.
2. Run `benchmarks/bench_amd_qg.c` (qg vs bitset).  Capture to `bench_day12_amd_qg.{csv,txt}`.  This is the canonical wall-time check: bcsstk14 qg-AMD must be ≤ 1.5× the Sprint 22 bitset baseline (~140 ms × 1.5 = 210 ms ceiling) per the headline gate.
3. Side-by-side compare against `SPRINT_22/bench_day14.txt` and `SPRINT_23/bench_day14.txt`.  Build a markdown table in `docs/planning/EPIC_2/SPRINT_24/bench_summary_day12.md` showing nnz(L) and wall-time deltas per fixture × ordering across the three sprints.
4. Headline checks (Sprint 24's deliverable gates from PROJECT_PLAN.md item 6):
   - **(a)** qg-AMD wall on bcsstk14 ≤ 1.5× Sprint 22 quotient-graph baseline (~210 ms ceiling).
   - **(b)** qg-AMD nnz(L) bit-identical to Sprint 22 + Sprint 23 captures.
   - **(c)** Pres_Poisson ND/AMD ≤ 0.85× (item 5 stretch target).
   - **(d)** All `SPRINT_23/bench_day14.txt` nnz_L rows stay bit-identical or improve.
5. If any of (a)-(d) miss, root-cause and document in `bench_summary_day12.md`.  Day 13's budget covers the closing tests + retro stub; if a hard regression turns up here, flag it as a Sprint 25 follow-up rather than land Sprint 24 broken.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `bench_day12.{csv,txt}` cross-ordering capture
- `bench_day12_amd_qg.{csv,txt}` qg-vs-bitset capture
- `bench_summary_day12.md` side-by-side table vs Sprint 22 + Sprint 23 baselines
- All four headline checks (a)-(d) status documented
- All quality checks clean

### Completion Criteria
- Headline check (a): bcsstk14 qg-AMD ≤ 210 ms (1.5× Sprint 22 baseline of ~140 ms)
- Headline check (b): qg-AMD nnz_L bit-identical to Sprint 22 + Sprint 23 captures
- Headline check (c): Pres_Poisson ND/AMD ≤ 0.85× (or ≤ 0.90× partial close documented)
- Headline check (d): nnz_L bit-identical-or-better on every `bench_day14.txt` row
- `bench_summary_day12.md` committed with the deltas
- `make format && make lint && make test && make wall-check` clean

---

## Day 13: Closing Tests & Documentation Sweep

**Theme:** Add the closing tests called out in PROJECT_PLAN.md item 7, refresh `docs/algorithm.md`'s AMD subsection to reflect whichever Days 2-5 logic survived item 2, and append a "Sprint 24 closures" subsection to `SPRINT_22/PERF_NOTES.md` with the Day-12 numbers.

**Time estimate:** 8 hours

### Tasks
1. New tests in `tests/test_reorder_amd_qg.c` if Days 1-7 added user-visible behavior:
   - Day 2's regression test (sorted-list compare or regularity gate, depending on the chosen fix candidate)
   - Day 7's external-degree behind-flag test (already added, double-check it's enabled in `RUN_TEST`)
   - Day 5's Pres_Poisson AMD parity test (already added Day 5; double-check it's not gated behind `SPARSE_QG_LONG_TESTS=0` for the closing day)
2. Update `docs/algorithm.md`'s AMD subsection.  If Day 1 chose fix candidate (c) revert: drop the "Four mechanisms" prose down to whichever survived (the original Sprint 22 quotient-graph baseline plus Day 6's parity-test framework).  If (a) or (b): adjust the supervariable detection prose to reflect the sorted-list compare or regularity gate.  Cite Days 2-5 commits as the implementation history.
3. Append a "Sprint 24 closures" subsection to `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` (or open `SPRINT_24/PERF_NOTES.md` if the Sprint 22 file gets unwieldy).  Include the Day-12 bench-summary table + a 1-2 paragraph narrative explaining what Sprint 24 actually moved (qg-AMD wall time on bcsstk14, Pres_Poisson ND/AMD, AMD parity-test coverage) and what it didn't (the 0.7× literal target, the optional external-degree refinement if it didn't promote).
4. Stub `docs/planning/EPIC_2/SPRINT_24/RETROSPECTIVE.md` with the same eight-section structure as `SPRINT_23/RETROSPECTIVE.md`: Goal recap, DoD checklist, Final metrics, Performance highlights, What went well, What surprised us, What didn't go well, Items deferred, Lessons, Sprint 25 inputs, Acknowledgements, Day-by-day capsule, DoD verification.  Headers + DoD checklist filled; prose sections placeholders for Day 14.
5. Update `docs/planning/EPIC_2/PROJECT_PLAN.md`'s Sprint 24 section: add a "**Status: Complete**" line + the actual hours-spent total (from time-tracking accumulated through the sprint).
6. Decide on Sprint 24 design-rationale notes: if any in-tree scratch docs (profile capture, decision docs) need to live on past the sprint, retain them under `docs/planning/EPIC_2/SPRINT_24/`; if they're transient, delete them.  Document the call inline in the retro stub.
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- New tests for whichever Days 2-7 behavior changes shipped
- `docs/algorithm.md` AMD subsection refreshed
- `SPRINT_22/PERF_NOTES.md` Sprint-24-closures subsection appended
- `SPRINT_24/RETROSPECTIVE.md` stubbed
- `PROJECT_PLAN.md` Sprint 24 marked complete with actual hours
- All quality checks clean

### Completion Criteria
- New tests pass under `make test`
- `algorithm.md` AMD subsection no longer mentions the Sprint 23 simplifications that item 2 reverted (if (c) was chosen) or describes the new sorted-list / regularity-gate logic (if (a) / (b))
- `PERF_NOTES.md` closures subsection cites Day-12's numbers
- Retrospective stub has all 8 section headers
- `make format && make lint && make test && make wall-check` clean

---

## Day 14: Soak, Final Bench Capture & Sprint Retrospective

**Theme:** Final cross-ordering capture as the end-of-sprint headline, full corpus regression run, retrospective body, and PR open.  This is the day that gates the merge.

**Time estimate:** 8 hours

### Tasks
1. Final capture: re-run `benchmarks/bench_reorder.c` and `benchmarks/bench_amd_qg.c` once more.  Save to `bench_day14.{csv,txt}` / `bench_day14_amd_qg.{csv,txt}`.  Sanity-check that nothing regressed since Day 12 — should be bit-identical on the bench output since Day 13 was tests + docs only.
2. Run `make sanitize` against the full test suite (ASan + UBSan).  Sprint 23's sanitize pass had pre-existing infrastructure issues; confirm whether they're resolved in the Sprint 23 → master merge or persist.  Any new warnings since Sprint 22's clean baseline are a hard gate — investigate before the retro write-up.
3. Run `make tsan` against the OpenMP-parallelised tests.  Sprint 23 added thread-safety guards; verify Sprint 24's algorithmic changes don't introduce new races (they shouldn't — items 2-5 are single-threaded).
4. Fill in the Sprint 24 retrospective (`SPRINT_24/RETROSPECTIVE.md` body): for each of the 8 sections, write 1-2 paragraphs.  Headline material: did the AMD wall-time fix close gate (b) cleanly?  Did the ND fill-quality work reach 0.85× on Pres_Poisson?  What slipped (the 0.7× literal target if Day 9's combined effect didn't reach it)?  What surprised us (the wall-check gate's cumulative effect — caught any small regressions that would have compounded?  Or: external-degree refinement either tightened nnz meaningfully or didn't, against Davis 2006's promise).
5. Open the Sprint 24 PR (`gh pr create`) targeting `master`.  PR description summarises the seven items + the day-by-day commits + the headline numbers from `bench_day14.txt` vs Sprint 23's `bench_day14.txt`.
6. Run `make format && make lint && make test && make sanitize && make wall-check`.

### Deliverables
- `bench_day14.{csv,txt}` + `bench_day14_amd_qg.{csv,txt}` final captures
- `make sanitize` + `make tsan` clean (or pre-existing infrastructure gaps explicitly flagged in retro)
- `RETROSPECTIVE.md` body filled in (all 8 sections)
- Sprint 24 PR opened
- All quality checks clean

### Completion Criteria
- Final cross-ordering capture matches Day 12's output bit-identically (no regressions in Day 13's doc/test sweep)
- `make sanitize` + `make tsan` clean against the full test suite (or documented as Sprint-25 infrastructure follow-ups)
- Retrospective body written; all 8 sections have content (not stubs)
- PR opened; description references the headline qg-AMD wall-time + Pres_Poisson ND/AMD numbers
- `make format && make lint && make test && make sanitize && make wall-check` clean

---

## Sprint 24 Summary

**Total estimated hours:** 10 + 10 + 10 + 8 + 8 + 8 + 12 + 10 + 12 + 8 + 6 + 12 + 8 + 8 = 130 hours

**Item-to-day mapping:**

| Item | Days | Hours |
|------|------|-------|
| 1: Wall-time regression-check instrumentation | Day 1 | 6 |
| 2: qg-AMD wall-time root-cause + fix | Days 1-4 | 32 |
| 3: AMD parity test on Pres_Poisson | Day 5 | 8 |
| 4: Davis §7.5.1 external-degree refinement | Days 6-7 | 20 |
| 5: ND fill-quality follow-up — Pres_Poisson ≤ 0.85× | Days 8-11 | 32 |
| 6: Cross-corpus re-bench | Day 12 | 12 |
| 7: Tests + docs + retrospective | Days 13-14 | 16 |

**Headline gates (must pass on Day 14):**

- qg-AMD wall on bcsstk14 ≤ 1.5× Sprint 22 quotient-graph baseline (~210 ms ceiling) — closes Sprint 23 Day-12 gate (b) hard-fail
- qg-AMD nnz(L) bit-identical to Sprint 22 + Sprint 23 captures
- Pres_Poisson ND/AMD ≤ 0.85× (stretch target; partial close to ≤ 0.90× acceptable per `bench_summary_day12.md`)
- All `SPRINT_23/bench_day14.txt` nnz_L rows stay bit-identical or improve
- `make wall-check` exits 0 against the Day 4 updated baseline

**Risk flags:**

- Item 2 (32 hrs across Days 1-4) is the load-bearing kernel fix.  If the chosen fix candidate (a / b / c) doesn't reach the ≤ 1.5× ceiling on Day 4, options are: extend item 2 by 4 hours into Day 5's slot (push item 3 to Day 6), or fall back to candidate (c) revert.  Day 4's wall-check baseline update is the gate that triggers the escalation.
- Item 4 (20 hrs across Days 6-7) is conditional on Day 1's fix-candidate choice.  If (c) revert, Days 6-7's budget recovers as item 5 head-start (Day 8's work moves into Day 6).
- Item 5 (32 hrs across Days 8-11) targets ≤ 0.85× on Pres_Poisson.  PROJECT_PLAN.md acknowledges the literal 0.7× from Sprint 22 plan is out of scope; Sprint 23 missed 0.7× by a wide margin.  If neither option (a) deeper coarsening nor option (b) smarter separator extraction reaches 0.85×, document the partial close (≤ 0.90× target) and flag the rest as Sprint-25 territory.
- The 130-hour estimate has a 4-hour cushion above the 126-hour PROJECT_PLAN.md figure — tighter than Sprints 22 / 23 (which had ~10-hour buffers).  The risk concentration around item 2 + the conditional nature of item 4 mean this estimate could absorb a larger overrun by dropping item 4 entirely (saves 20 hours) without compromising the headline gates.
