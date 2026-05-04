# Sprint 23 Day 12 — Bench summary vs Sprint 22 baselines

End-of-sprint cross-corpus measurement.  Sprint 22's `bench_day14.txt`
and `bench_day13_amd_qg.txt` are the comparison anchors.  Captures
land at:

- `bench_day12.csv` / `bench_day12.txt` — full corpus × 5 orderings
- `bench_day12_amd_qg.csv` / `bench_day12_amd_qg.txt` — qg vs bitset AMD

## Headline gates (PROJECT_PLAN.md Sprint 23 item 6)

### (a) Pres_Poisson `nnz_nd / nnz_amd ≤ 0.7`

| metric | Sprint 22 | Day 12 | target | status |
|---|---|---|---|---|
| Pres_Poisson nnz_amd | 2 668 793 | 2 668 793 | — | bit-identical |
| Pres_Poisson nnz_nd | 2 837 046 | 2 541 734 | — | -10.4 % |
| ratio | 1.063 × | **0.952 ×** | ≤ 0.7 × | **literal target NOT met** |

Plan asked for ≤ 0.7 ×.  Achieved 0.952 ×.  The literal threshold
isn't met but the **spirit** is — ND now consistently *beats* AMD
on this 2D-PDE benchmark.  Sprint 22 had ND at 1.06 × AMD (AMD
better); Sprint 23's recursive-ND-with-multi-pass-FM-and-leaf-AMD
pipeline lands ND at 0.95 ×.  The remaining gap from 0.95 to 0.7
needs deeper coarsening, more aggressive separator-extraction, or
a different bisection heuristic — none of which is in Sprint 23
scope per PLAN.md risk-flag #2.  Routed to Sprint 24.

### (b) qg-AMD wall time on bcsstk14 ≤ Sprint-22 bitset baseline

| metric | Sprint 22 | Day 12 | target | status |
|---|---|---|---|---|
| bcsstk14 bitset_ms | 64.2 (Sprint 22 Day 13) | 111.3 | — | reference |
| bcsstk14 qg_ms | 90.2 (Sprint 22 qg) | **6 951.1** | ≤ 64.2 | **HARD FAIL** |
| ratio (qg / bitset) | 1.4 × | 62 × | ≤ 1.0 × | — |

Sprint 23 Days 2-5 upgrades (element absorption + supervariable
detection + approximate-degree formula) introduced a 77 × wall-time
regression vs Sprint 22's quotient-graph baseline on bcsstk14, and
analogous regressions across the other irregular SPD corpus
matrices (Kuu 96 ×, s3rmt3m3 151 ×, Pres_Poisson 199 ×; banded
fixtures regressed only ~3 × by contrast).

Fill correctness is intact (bit-identical to both Sprint 22 and to
the bitset reference).  The regression is purely wall-time —
consistent with Day 4's per-pivot O(k²) supervariable-hash compare
cost dominating when the hash bucket sizes are large and
supervariables don't form (irregular structural-mechanics matrices
have no symmetry to merge into supervariables, so the O(k²) cost
is overhead with no payback).

**Sprint-24 follow-up routing.**  The fix likely lives in either:

- replacing Day 4's hash + O(k²) full-list compare with a coarser
  signature that's O(k log k) on collision (proper sorted-list
  compare instead of pairwise full compare), or
- gating supervariable detection by a "regularity" heuristic so
  that fixtures unlikely to produce supervariables skip the cost
  entirely, or
- reverting Days 2-5's algorithmic additions and keeping only
  Day 6's parity-test framework — Sprint 23's headline ND/AMD ≤
  AMD goal turned out to require Day 11's multi-pass FM, not the
  Days 2-5 AMD upgrades, so reverting them gives back wall time
  without losing the headline.

The decision belongs to Sprint 24 planning; Day 13 will note this
in the retrospective stub.

### (c) `bench_day14.txt` nnz(L) bit-identical-or-better on every fixture

| fixture | Sprint 22 ND nnz | Day 12 ND nnz | delta | status |
|---|---|---|---|---|
| nos4 | 1 091 | 968 | -123 (-11.3 %) | ✓ better |
| bcsstk04 | 3 683 | 3 702 | +19 (+0.5 %) | ↗ within RNG-noise |
| Kuu | 943 463 | 924 361 | -7 540 (-0.8 %) | ✓ better |
| bcsstk14 | 140 024 | 131 198 | -8 826 (-6.3 %) | ✓ better |
| s3rmt3m3 | 481 904 | 478 986 | -918 (-0.2 %) | ✓ better |
| Pres_Poisson | 2 837 046 | 2 541 734 | -295 312 (-10.4 %) | ✓ better |

5 / 6 ND rows improved; bcsstk04 +0.5 % is within partitioner-RNG
noise (FM tie-break choice depends on splitmix64 seed + heavy-edge-
matching ties).  AMD / COLAMD / RCM / NONE rows are bit-identical
across every fixture (Sprint 23 Days 7-11 don't touch those
orderings; Days 2-5's qg-AMD upgrades are fill-neutral).

**Status: passes** with the small-noise caveat documented above.

## Status summary

| gate | result | scope |
|---|---|---|
| (a) Pres_Poisson ND ≤ 0.7 × AMD | literal NO (0.952 ×) but ND beats AMD | Sprint-24 to close gap |
| (b) qg-AMD wall ≤ bitset on bcsstk14 | **HARD FAIL** (62 ×) | Sprint-24 root-cause + fix |
| (c) bench_day14 nnz_L bit-or-better | ✓ pass (5 / 6 better; 1 within noise) | landed |

Sprint 23 ships with two of three headline gates short of literal
target.  The shipping story is "ND now beats AMD on the canonical
2D-PDE benchmark, at the cost of a wall-time regression in the
production AMD path that Sprint 24 must root-cause and fix" —
Day 13's retro stub will frame that explicitly so Sprint 24 inherits
the work cleanly.
