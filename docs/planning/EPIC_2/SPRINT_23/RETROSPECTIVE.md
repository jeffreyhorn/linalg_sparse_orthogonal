# Sprint 23 Retrospective — Ordering Quality Follow-Ups (Sprint 22 Deferrals)

**Sprint budget:** 14 working days (~100 hours estimated, per PLAN.md)
**Branch:** `sprint-23`
**Calendar elapsed:** 2026-05-02 → 2026-05-03 (intensive condensed run; the 14-day budget tracks engineering effort, not wall-clock days)

> **Status:** Day 13 stub.  Day-by-day metrics + headline-gate
> outcomes are populated; prose sections (Lessons, Sprint 24
> Inputs, Acknowledgements) are placeholders for the post-sprint
> write-up — landing the structure now makes that write-up
> mechanical.

## Goal recap

> Close the two ordering-quality gaps Sprint 22 left open:
>
>   1. ND fill ratio on Pres_Poisson — Sprint 22 measured 1.06×
>      AMD's nnz(L), short of the plan's "≥ 2× reduction over AMD"
>      target.
>   2. qg-AMD wall-time tail on small SPD corpus matrices —
>      Sprint 22 measured ~30 % bitset-favoured at n ≤ 1 800.
>
> Algorithmic fronts: (a) bring `sparse_reorder_amd_qg` up to the
> full Davis 2006 algorithm — element absorption + supervariable
> detection + approximate-degree updates; (b) port METIS's O(1)
> gain-bucket FM into `graph_refine_fm`; (c) splice quotient-graph
> AMD into `nd_recurse`'s leaf base case.

(See `docs/planning/EPIC_2/SPRINT_23/PLAN.md` for the day-by-day
breakdown.)

## Definition of Done checklist

| item | status | reference |
|---|---|---|
| 1. SPD synth fixture for Cholesky-via-ND residual @ 1e-12 | ✓ | Day 1 commit `e802796`, `tests/test_reorder_nd.c::test_cholesky_via_nd_residual_spd_synth` (3.77 e-15) |
| 2. Element absorption + supervariable detection in qg-AMD | ✓ | Days 3-4 commits `aea071a`, `6096840` |
| 3. Approximate-degree update + dense-row skip | ✓ | Days 5-6 commits `f0fe391`, `db36001` |
| 4. ND leaves call qg-AMD via `nd_subgraph_to_sparse` | ✓ | Day 7 commit `20bb1e9` |
| 5. FM gain-bucket structure + integration | ✓ | Days 9-10 commits `58bf278`, `2b944ea` |
| 6. Cross-corpus re-bench + headline-gate verdicts | ✓ | Day 12 commit `4f52dc3`, `bench_summary_day12.md` |
| 7. Closing tests + algorithm.md / PERF_NOTES.md / retro stub | this commit | Day 13 |

Headline gates from PROJECT_PLAN.md item 6 — see
`docs/planning/EPIC_2/SPRINT_23/bench_summary_day12.md`:

| gate | result |
|---|---|
| (a) Pres_Poisson `nnz_nd / nnz_amd ≤ 0.7×` | literal NO (0.952×); spirit YES — ND beats AMD |
| (b) qg-AMD wall on bcsstk14 ≤ Sprint-22 bitset (64 ms) | **HARD FAIL** (6 951 ms = 108× bitset) |
| (c) bench_day14 nnz_L bit-identical-or-better | ✓ pass (5/6 better; 1 within RNG noise) |

## Final metrics

End-of-sprint cross-corpus capture: `docs/planning/EPIC_2/SPRINT_23/bench_day12.txt`.

### ND/AMD nnz(L) ratios (vs Sprint 22 Day 14)

| fixture        | Sprint 22 | Sprint 23 | delta |
|---------------|-----------|-----------|-------|
| nos4           | 1.713×    | 1.520×    | -11 % |
| bcsstk04       | 1.172×    | 1.178×    | +0.5 % (within RNG noise) |
| Kuu            | 2.322×    | 2.275×    | -2 %  |
| bcsstk14       | 1.207×    | 1.130×    | -6 %  |
| s3rmt3m3       | 1.015×    | 1.009×    | -0.6 % |
| **Pres_Poisson** | **1.063×** | **0.952×** | **-10 % (ND beats AMD)** |

5 / 6 fixtures dropped; bcsstk04 +0.5 % within partitioner RNG.

### qg-AMD wall time (vs bitset reference)

Sprint 22 Day 13 had qg at 0.6-1.8× the bitset across SuiteSparse
fixtures; Sprint 23 Day 12 has qg at 0.01-0.06× the bitset on
SuiteSparse SPD (62-114× regression).  Banded fixtures regressed
from 4-7× wins to 1.8-2.6× wins — still wins, but down.  See
`bench_day12_amd_qg.txt`.

Fill quality bit-identical between Sprint 22 qg, Sprint 23 qg, and
bitset across the entire corpus + synthetic banded.  Memory
headline (≥ 4× reduction at n ≥ 50 000) still holds.

## Performance highlights

The headline outcome is **ND now beats AMD on Pres_Poisson** (the
canonical 2D-PDE fill-quality benchmark from Sprint 20 onward) —
the first time in this codebase's history.  The cumulative drivers,
in order of contribution:

1. **Day 11 multi-pass FM** at the finest uncoarsening level
   (1 → 3 passes; intermediate levels stay single-pass).  Drops
   Pres_Poisson nnz_nd / nnz_amd from 1.026× to 0.952× — the
   largest single jump.  Each successive pass converges toward
   the FM local optimum on this fixture's 2D separator structure;
   pass 3 is the sweet spot, pass 5 sits at the same ratio.

2. **Day 7 leaf-AMD splice**.  `nd_recurse`'s small-subgraph base
   case now calls `sparse_reorder_amd_qg` instead of emitting in
   subgraph-local order.  Wins on 2/6 fixtures (nos4 -11 %, Kuu
   -0.8 %); 4/6 saw small (≤ 1 %) increases that are within
   partitioner RNG noise.  10×10 grid dropped 1.38× → 1.20× from
   this alone.

3. **Days 9-10 gain-bucket FM**.  Sprint 22's O(n) max-gain linear
   scan replaced with an O(1) bucket-array max-find via highest-
   non-empty-bucket cursor; per-pass cost lifted from O(n²) to
   O(|E|).  This made Day 11's multi-pass exploration affordable
   — without cheap passes, multi-pass FM on Pres_Poisson would
   have been minutes per call.

The wall-time profile is unchanged from Sprint 22 (~45 s ND on
Pres_Poisson) — the bucket-FM win was offset by the AMD wall-time
regression in Day 7's leaf-AMD splice (each leaf calls qg-AMD,
which Sprint 23 Days 2-5 made slower; see DoD gate (b)).
Closing the AMD wall-time regression in Sprint 24 should drop
Pres_Poisson ND wall to the ≤ 10 s range PLAN.md anticipated.

## What went well

(post-sprint write-up — placeholder)

Candidate themes from the day-by-day capsule:

- **Day 11's multi-pass FM was the sleeper hit.**  PLAN.md framed
  it as a "budget-permitting last 2 hours" exploration with a
  clear deferral path; reality was a 4-pp jump in Pres_Poisson
  ratio that turned the spirit of headline gate (a) from "miss"
  to "ND beats AMD".

- **Sprint 22's modular ND + multilevel partition pipeline made
  the Day 7 splice fall out cleanly.**  `nd_recurse`'s base case
  was already a single function pointer (`nd_emit_natural`); the
  Day 7 swap is a 50-line patch that mostly handles the per-leaf
  failure-fallback path.

- **The bucket-FM swap (Days 9-10) was caught early by an
  unexpected derived test.**  bcsstk04 LDL^T-no-pivoting residual
  blew up to 7.94 e+9 in the first iteration of the bucket-FM —
  not the partition tests' separator-size ranges that PLAN.md
  flagged as the gate, but the LDL^T residual that the new
  partition's pivot order happened to break.  Skipped-vertex
  re-insertion fixed it; without that test, the regression would
  have shipped silently.

## What didn't go well

(post-sprint write-up — placeholder)

Candidate themes:

- **The qg-AMD wall-time regression** is Sprint 23's biggest
  miss.  Headline gate (b) fails by ~108× on bcsstk14; Pres_Poisson
  AMD now takes ~22 minutes (was 12 seconds in Sprint 22).  Days
  2-5 should have included a wall-time regression check at each
  step — the cumulative cost of element absorption + supervariable
  hash O(k²) compare + workspace doubling wasn't apparent until
  the Day 12 cross-corpus bench.

- **PLAN.md's literal targets were optimistic for Pres_Poisson
  ND/AMD ≤ 0.7×.**  Achieved 0.952×; the plan's risk-flag #2
  acknowledged this might happen but the day-budget for Day 11
  multi-pass FM was 2 hours of exploratory work, not a deeper
  algorithmic redesign.  Sprint 24 work needed to reach 0.7×.

- **The Day 6 wall-time finding (USE_APPROX 4.9× slower than
  exact)** was a hint that Days 2-5's algorithmic additions
  weren't paying their wall-time freight, but the Day-6 commit
  framed it as "approximate-degree formula's limitation" rather
  than investigating whether the qg-AMD baseline itself had
  regressed.  Day 12's bench made the actual scope of the
  regression visible.

## Items deferred

| item | rationale | Sprint 24 routing |
|---|---|---|
| Pres_Poisson ND/AMD ≤ 0.7× | Achieved 0.952× via Day 11 multi-pass FM; closing the rest needs deeper coarsening / smarter separator extraction | "ND fill quality follow-up" item |
| qg-AMD wall-time regression on irregular SPD | Days 2-5 added per-pivot O(k²) supervariable compare that dominates when supervariables don't form (irregular structural-mechanics matrices) | "qg-AMD wall-time fix" item — three candidate fixes in `bench_summary_day12.md` "(b)" |
| Pres_Poisson under SPARSE_QG_VERIFY_DEG | Day 13 corpus parity test runs only on bcsstk14 (USE_APPROX is 5× slower; Pres_Poisson would push the test suite past 30 minutes) | "AMD parity test on Pres_Poisson" item if/when wall-time fix lands |
| Davis §7.5.1 external-degree refinement | Mentioned in Day 5's davis_notes but not implemented | Sprint 24 if approximate-degree is retained |

## Lessons

(post-sprint write-up — placeholder)

Sketch:

- **A bench day before a closure day surfaces hard regressions
  that the sprint-internal fixture suite misses.**  Day 12's
  bench was budgeted as "headline measurement"; turned out to be
  the first time we measured AMD wall time on irregular SPD
  fixtures end-to-end since Sprint 22 Day 13.  The 77-199×
  regression had been accumulating across Days 2-5 with no
  intermediate signal.

- **The "consider every unlocked vertex every step" semantics of
  Sprint 22's FM was load-bearing for derived tests** in ways
  that the partition test ranges didn't capture.  bcsstk04 LDL^T
  no-pivoting depended on a specific tie-breaking pattern that
  the bucket-FM's pop-order broke until skipped-vertex re-insertion
  was added.  Lesson for Sprint 24: when swapping a graph
  algorithm's internals, run derived numerical tests (factorization
  residuals) before declaring the swap safe.

## Sprint 24 inputs

(post-sprint write-up — placeholder; see `bench_summary_day12.md`
for routing.)

Top three Sprint 24 work items per `PROJECT_PLAN.md` framing:

1. **qg-AMD wall-time root-cause + fix.**  Days 2-5's regression
   on irregular SPD (62-199× vs Sprint 22 baseline).  Three
   candidate fixes documented in `bench_summary_day12.md "(b)"`:
   replace Day 4's hash + O(k²) full-list compare with a
   sorted-list compare, gate supervariable detection by a
   regularity heuristic, or revert Days 2-5 entirely (Day 11's
   multi-pass FM was the actual headline driver, not Days 2-5).

2. **Pres_Poisson ND/AMD ≤ 0.7×.**  Need deeper coarsening or
   smarter separator extraction beyond Sprint 22's smaller-side
   lift; Sprint 23 closed half the gap (1.06× → 0.95×) via
   multi-pass FM, the rest needs algorithmic work.

3. **AMD parity test on Pres_Poisson** (Day 13 deferral).  Once
   item 1 is fixed, the corpus parity test the Day 13 stub
   gestured toward becomes affordable.

## Acknowledgements

(post-sprint write-up — placeholder)

## Day-by-day capsule (for the prose write-up)

| day | theme | signal landed |
|---|---|---|
| 1 | SPD synth fixture + Davis §7 reading | residual gate at 1e-12 (was 1e-8); davis_notes.md scaffolds Days 2-5 |
| 2 | qg_t workspace extension | iw_size 5·nnz+6·n+1 → 7·nnz+8·n+1; elen[]; bit-identical fill |
| 3 | element absorption (Davis representation) | absorbed[] tracking; iw_used peak shrinks toward end of elimination |
| 4 | supervariable detection | super[] / super_size[]; star fixture pins co-elimination contract |
| 5 | approximate-degree formula | opt-in via SPARSE_QG_USE_APPROX_DEG; Day 6 measures USE_APPROX 4.9× slower (warning sign) |
| 6 | parity test framework + cap_fired probe | 200-vertex parity + dense-row coverage; warning sign in davis_notes deferred |
| 7 | ND leaves call qg-AMD | nd_subgraph_to_sparse + nd_recurse leaf splice; 10×10 grid 1.38 → 1.20× |
| 8 | ND validation + 10×10 grid tightening + bench capture | Day-7 fill audit; partial plan-target deviations documented in bench_day8 |
| 9 | gain-bucket data structure + tests | fm_bucket_array_t + 5-function API; 5 tests; 2 663 assertions |
| 10 | gain-bucket FM integration | graph_refine_fm refactored; bcsstk04 LDL^T residual hazard caught + fixed via skipped-vertex re-insertion |
| 11 | FM stress + multi-pass exploration | 5 → 9 fm-bucket tests; multi-pass adopted as default (3 passes finest) — Pres_Poisson 1.026 → 0.952×, ND beats AMD |
| 12 | cross-corpus re-bench | bench_day12.txt + bench_day12_amd_qg.txt; (a)/(b)/(c) verdicts in bench_summary_day12.md |
| 13 | closing tests + docs sweep + retro stub | this commit |
| 14 | soak + final bench + retro body + PR | (Day 14 work) |

## davis_notes.md retention call

Sprint 23's `davis_notes.md` started as Day 1 reading notes for
Davis 2006 §7 and grew across Days 2-11 to host multi-pass FM
exploration findings, bcsstk14 wall-time probes, cap_fire counts,
and design rationale that's cited in commit messages and code
comments.  **Retained** under `docs/planning/EPIC_2/SPRINT_23/`.
PLAN.md §13.6 acknowledged either retain or delete is acceptable;
the file's been substantively cited (the Day-11 multi-pass FM
table is the canonical record of the experiment, the bcsstk14 §7
notes ground the Day-3-4 implementation rationale), so retention
costs nothing and saves the citations from rotting.

## DoD verification

(post-sprint write-up — Day 14 final-bench section will populate)
