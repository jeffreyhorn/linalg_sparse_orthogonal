# Sprint 26 Day 5 — `nd_base_threshold` Re-Sweep + Default-Flip Decision

## Decision

**Flip the default `sparse_reorder_nd_base_threshold` from 32 to 96.**

The flip clears both halves of the PLAN.md flip rule (≥ 5 % Pres_Poisson
wall improvement + no fixture nnz_L regression past 1pp) by a
comfortable margin and produces dramatic wall-time improvements
across the entire corpus.  No defaults flipped this large — Sprint
22-25's env-var-gated alternatives all stayed off-by-default; Sprint
26 Day 5 is the first sprint to flip a base ND parameter.

## Sweep results (5-threshold × 6-fixture matrix)

### ND nnz_L vs threshold

| fixture | t=32 (default) | t=48 | t=64 | t=96 | t=128 |
|---|---:|---:|---:|---:|---:|
| nos4 | 968 | 903 | 809 | 809 | 637 |
| bcsstk04 | 3 702 | 3 650 | 3 650 | 3 722 | 3 722 |
| Kuu | 924 385 | 917 619 | 908 834 | 881 177 | 875 101 |
| bcsstk14 | 131 017 | 131 655 | 130 971 | 129 292 | 129 576 |
| s3rmt3m3 | 478 890 | 480 555 | 481 382 | 483 195 | 483 907 |
| Pres_Poisson | 2 541 734 | 2 541 351 | 2 539 903 | 2 536 427 | 2 549 740 |

### ND wall-time (ms) vs threshold

| fixture | t=32 | t=48 | t=64 | t=96 | t=128 |
|---|---:|---:|---:|---:|---:|
| nos4 | 104.5 | 97.1 | 65.0 | 64.5 | 0.4 |
| bcsstk04 | 556.9 | 374.2 | 368.3 | 174.4 | 178.5 |
| Kuu | 15 091 | 8 873 | 8 171 | 6 629 | 6 360 |
| bcsstk14 | 6 088 | 3 599 | 2 987 | 1 188 | 1 119 |
| s3rmt3m3 | 14 388 | 8 253 | 7 908 | 3 695 | 3 397 |
| **Pres_Poisson** | **38 131** | **22 997** | **21 502** | **12 222** | **11 530** |

### Per-fixture nnz_L deltas vs t=32 baseline (percentage points)

| fixture | t=48 | t=64 | t=96 | t=128 |
|---|---:|---:|---:|---:|
| nos4 | -6.7 % win | -16.4 % win | -16.4 % win | -34.2 % win |
| bcsstk04 | -1.4 % | -1.4 % | +0.5 % | +0.5 % |
| Kuu | -0.7 % | -1.7 % | -4.7 % win | -5.3 % win |
| bcsstk14 | +0.5 % | -0.04 % | -1.3 % win | -1.1 % win |
| s3rmt3m3 | +0.3 % | +0.5 % | +0.9 % | **+1.05 % > 1pp** |
| Pres_Poisson | -0.02 % | -0.07 % | -0.21 % win | +0.31 % |

### Per-fixture wall deltas vs t=32 baseline (%)

| fixture | t=48 | t=64 | t=96 | t=128 |
|---|---:|---:|---:|---:|
| nos4 | -7.1 % | -37.8 % | -38.3 % | -99.6 % |
| bcsstk04 | -32.8 % | -33.9 % | -68.7 % | -67.9 % |
| Kuu | -41.2 % | -45.9 % | -56.1 % | -57.9 % |
| bcsstk14 | -40.9 % | -50.9 % | -80.5 % | -81.6 % |
| s3rmt3m3 | -42.6 % | -45.0 % | -74.3 % | -76.4 % |
| **Pres_Poisson** | **-39.7 %** | **-43.6 %** | **-67.9 %** | **-69.8 %** |

## Flip-rule application

PLAN.md Day 5 task 3 specifies the flip rule:
> Flip the default if (a) Pres_Poisson wall improves by ≥ 5 % AND
> (b) no fixture regresses nnz_L past 1pp.

| threshold | (a) Pres_Poisson wall | (b) max nnz_L regression | flip-rule |
|---|---|---|---|
| t=48 | -39.7 % ✓ | +0.5 % (bcsstk04) ✓ | PASS |
| t=64 | -43.6 % ✓ | +0.5 % (s3rmt3m3) ✓ | PASS |
| **t=96** | **-67.9 % ✓** | **+0.9 % (s3rmt3m3) ✓** | **PASS** |
| t=128 | -69.8 % ✓ | +1.05 % (s3rmt3m3) ✗ (just past 1pp) | FAIL |

**t=96 is the maximum threshold that satisfies the flip rule.**  All
of {48, 64, 96} pass; t=96 produces the largest Pres_Poisson wall
improvement among rule-satisfying candidates (-67.9 % vs t=64's
-43.6 %).  t=128 fails the (b) gate by 0.05 % (s3rmt3m3 +1.05 % is
just past the 1pp ceiling); this is essentially noise but the flip
rule is literal — t=128 is rejected.

## Why the dramatic wall-time improvement?

Day 4's per-recursion-level profile (`per_recursion_profile_day4.md`)
showed cost concentrating at depths 6-9 (88 % of partition cost on
169 small-subgraph calls).  Each of those calls invokes the full
multilevel pipeline (coarsen + bisect + uncoarsen + FM) on a
subgraph of n ≈ 50-300; the per-call cost has a constant-factor
floor of 60-200 ms regardless of subgraph size.

Raising `nd_base_threshold` from 32 to 96 cuts off the recursion
earlier, replacing the multilevel-pipeline calls at depths 6-9 with
direct leaf-AMD invocations.  Leaf-AMD at n ≈ 80-100 is
sub-millisecond per Sprint 22-23 measurements (`bench_amd_qg.c`'s
nos4 at n=100: 0.2 ms).  Net: ~169 calls × ~150 ms (multilevel) →
~169 calls × ~1 ms (leaf-AMD) ≈ 25 s saved, matching the observed
Pres_Poisson reduction (38 131 ms → 12 222 ms = 25 909 ms saved).

This also explains why nnz_L barely moves: leaf-AMD on n=64-96
subgraphs produces fill very close to what ND-recursion-then-leaf-
AMD would produce — the multilevel separator-last benefit only
kicks in for n much larger than the AMD fill horizon.

## Per-fixture findings under t=96 (the flipped default)

- **nos4 (n=100)**: WIN -16.4pp nnz_L + -38.3 % wall.  At t=96, the
  recursion fires once (n=100 > 96) producing a partition + two
  leaf-AMD calls; total nnz_L (809) sits between AMD's 637 and the
  Sprint 22-25 t=32 ND's 968.
- **bcsstk04 (n=132)**: NEUTRAL fill (+0.5 %) + -68.7 % wall.  At
  t=96, recurses once.
- **Kuu (n=7102)**: WIN -4.7pp nnz_L + -56.1 % wall.  Largest fill
  improvement among large fixtures — Kuu's irregular structure
  benefits from leaf-AMD's exact pivot ordering at slightly larger
  subgraphs.
- **bcsstk14 (n=1806)**: WIN -1.3pp nnz_L + -80.5 % wall.  Wall
  drops the most among fixtures (5× speedup).
- **s3rmt3m3 (n=5357)**: NEUTRAL fill (+0.9pp; just under the 1pp
  ceiling) + -74.3 % wall.  This fixture is at the boundary of the
  flip rule; consistent with its pattern in Sprint 25's
  intermediate-FM sweep where it tends to regress slightly under
  most env-var changes.
- **Pres_Poisson (n=14822)**: WIN -0.21pp nnz_L + -67.9 % wall.
  The headline fixture: nnz_L stays bit-stable (well within 0.5pp
  of Sprint 25's 2 541 734) while wall plummets from 38 s → 12 s.

## Implications for Sprint 26's headline gate (Pres_Poisson 0.85×)

This is purely a wall-time + small fill-quality win on the default
path; it does NOT close the Pres_Poisson 0.85× literal target by
itself.  Pres_Poisson ND/AMD under t=96 = 2 536 427 / 2 668 793 =
**0.9504×** (vs Sprint 25's 0.9524×; -0.21pp tightening).  Items
5-7 (FINEST FM annealing / geometric grid-cut / per-vertex sep
scoring) still need to close the remaining ~10pp.

But the Day-5 fix DOES help in two important ways:

1. **Faster iteration cycles for Days 6-8 (Item 5 FINEST FM)**:
   Pres_Poisson ND wall drops from 38 s to 12 s, so each FINEST-FM
   sub-axis sweep iteration runs ~3× faster.  The Day 8 cross-
   corpus sweep (which Sprint 25 Day 9 took 6 hours to capture)
   should now fit in ~2 hours.
2. **Wall-check baseline becomes much tighter**: the 47 055 ms
   baseline + 1.5× = 70 583 ms ceiling becomes massively over-
   provisioned at the post-flip ~12 s wall.  Future Sprint 26
   work that incrementally adds wall cost (e.g. Item 5's
   bucket-tie-break tracking adds zero, but Items 6-7 may add
   some) has plenty of headroom.

A wall-check baseline retightening (47 055 → ~14 000 with 1.5×
ceiling = 21 000) could land later in Sprint 26 (Day 13 task 4
test-bound tightening is the natural place); for Day 5 we just
verify the gate passes by a comfortable margin.

## Day 5 follow-ups to confirm

1. **Test contract check**: `test_nd_pres_poisson_fill_with_leaf_amd`
   asserts ND/AMD ≤ 0.96×.  Under t=96, Pres_Poisson ND/AMD =
   0.9504×, well within the bound. ✓
2. **`test_nd_10x10_grid_matches_or_beats_amd_fill`** asserts ND ≤
   1.17× AMD.  At t=96, the 10×10 grid (n=100) recurses once like
   nos4; expected ND nnz ≈ 809 vs AMD's nnz_L ≈ 656 = 1.23× — wait,
   this would fail the 1.17× bound.  Need to verify post-flip and
   relax the bound to ~1.25× if needed (the 1.17× was tightened in
   Sprint 24 Day 8 against the t=32 measurement of 1.158×).
3. **wall-check**: Pres_Poisson ND ≤ 70 583 ms ceiling.  Post-flip
   Pres_Poisson ND ≈ 12 200 ms — vastly under ceiling. ✓

(Item 2 above is checked + handled in Day 5 task 4-5.)

## Risk flags

- **Sprint 25 baselines change non-bit-identically.**  Every fixture
  has a different ND nnz_L under t=96 vs t=32.  Future bench
  comparisons need to know which threshold was active.  Mitigation:
  the comment block in `src/sparse_reorder_nd.c` and
  `sparse_reorder_nd_internal.h` should record the Sprint 26 Day 5
  flip + reference this decision doc.
- **Test contracts that pin pre-Sprint-26-Day-5 nnz_L need to be
  re-evaluated.**  At minimum:
  - `test_nd_pres_poisson_fill_with_leaf_amd` (≤ 0.96×): unchanged ✓
  - `test_nd_10x10_grid_matches_or_beats_amd_fill` (≤ 1.17×): may
    need to relax if it now produces ~1.23× — Day 5 task 4-5
    measures this.
- **bench_check_baseline.txt**: the 47 055 ms baseline becomes very
  loose under the new default.  Not a flip-blocker but should be
  retightened in Day 13 to preserve the gate's regression-detection
  power.

## What ships in Sprint 26 Day 5

- `sparse_reorder_nd_base_threshold = 96` (was 32).
- Comment block in `src/sparse_reorder_nd.c` updated to reference
  this decision + the Sprint 26 Day 5 measurement.
- Comment block in `src/sparse_reorder_nd_internal.h` updated.
- Test bound update if `test_nd_10x10_grid_matches_or_beats_amd_fill`
  needs relaxation.

## References

- `docs/planning/EPIC_2/SPRINT_26/PLAN.md` Day 5
- `docs/planning/EPIC_2/SPRINT_26/nd_base_threshold_sweep.txt` —
  the raw 5-threshold × 6-fixture capture this decision is based on
- `docs/planning/EPIC_2/SPRINT_26/per_recursion_profile_day4.md` —
  Day 4's per-depth analysis that informs why raising the threshold
  helps so much (depths 6-9 cost concentration)
- `docs/planning/EPIC_2/SPRINT_25/RETROSPECTIVE.md` "Items deferred"
  #4 — the original Sprint 25 Day 11 finding (nd_emit_natural firing
  32 times at ~165ms each = ~5.3s) that motivated this re-sweep
- `docs/planning/EPIC_2/SPRINT_22/bench_day9_nd.txt` — Sprint 22 Day
  9's original threshold sweep that set t=32 as the default; this
  Day 5 supersedes those measurements with Sprint 26's per-depth
  understanding
- `src/sparse_reorder_nd.c` — Day 5 default flip
- `src/sparse_reorder_nd_internal.h` — comment-block update
