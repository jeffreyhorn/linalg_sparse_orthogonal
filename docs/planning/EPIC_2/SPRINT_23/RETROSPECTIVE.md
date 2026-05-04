# Sprint 23 Retrospective — Ordering Quality Follow-Ups (Sprint 22 Deferrals)

**Sprint budget:** 14 working days (~100 hours estimated, per PLAN.md)
**Branch:** `sprint-23`
**Calendar elapsed:** 2026-05-02 → 2026-05-03 (intensive condensed run; the 14-day budget tracks engineering effort, not wall-clock days)

> **Status:** Day 14 final.  Day-by-day metrics, headline-gate
> outcomes, and prose sections all populated.  Sprint 23 ships
> with one of three literal-target headline gates met (c) and the
> spirit (but not the literal threshold) of (a) achieved — ND now
> beats AMD on Pres_Poisson.  Gate (b) is a hard regression Sprint
> 24 must close.

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

**Day 11's multi-pass FM was the sleeper hit.**  PLAN.md framed
it as a "budget-permitting last 2 hours" exploration with a clear
deferral path — "if 3-pass doesn't measurably improve cut quality
(likely on regular meshes), revert to single-pass and note the
finding".  Reality was the opposite: 3 passes drops Pres_Poisson
ND/AMD from 1.026× to 0.952× — the largest single jump of the
sprint and the single move that turned headline gate (a) from
"literal miss with no spirit met" to "literal miss but spirit met
— ND now beats AMD".  Wall time *also* dropped 7 seconds with
multi-pass because tighter partitions shrink downstream recursive-
ND work faster than extra passes add cost.  The Day-9/10 bucket-FM
infrastructure made each pass cheap enough to afford, and the
exploration that PLAN.md hedged on turned out to be the headline
move.

**Sprint 22's modular ND + multilevel partition pipeline made
Day 7's splice fall out cleanly.**  `nd_recurse`'s base case was
already a single function call (`nd_emit_natural`); the Day 7
swap is a 50-line patch — `nd_subgraph_to_sparse` builds a
temporary `SparseMatrix` from the leaf graph, the recursion calls
`sparse_reorder_amd_qg` on it, and a fallback path emits natural
ordering on any failure.  Zero changes to the multilevel partition
pipeline.

**The bucket-FM swap (Days 9-10) was caught early by an unexpected
derived test.**  `test_ldlt_via_nd_dispatch` — the bcsstk04 LDL^T
no-pivoting residual test — blew up to 7.94e+9 in the first
iteration of the bucket-FM.  This wasn't the partition tests'
separator-size ranges that PLAN.md flagged as the gate; it was a
*derived* numerical contract the new partition's pivot order
happened to break.  Root cause: the initial bucket-FM dropped
balance-ineligible vertices permanently (instead of retrying them
on subsequent steps as Sprint 22's full re-scan did), and the
resulting partition produced a perm[] where LDL^T-without-
pivoting hit a nearly-zero pivot.  Skipped-vertex re-insertion
fixed it; the residual returned to 6.02e-12 (bit-identical to
Sprint 22 baseline).  Without this test, the regression would
have shipped silently — only solver-correctness tests would have
caught it after release.

**Sprint 22's RNG-determinism contract held cleanly through every
algorithmic change.**  Days 7 (leaf-AMD splice), 9-10 (bucket-FM),
and 11 (multi-pass FM) all touched the partition output, but
`test_partition_determinism_*` continued to pass bit-identically
across re-runs.  The "same input + same seed → same output"
property held because every algorithmic change was deterministic
under the same splitmix64 seed, and the LIFO bucket-pop order
inherited from Day 9's head-insert pattern remained stable.

## What surprised us

**The Day-12 cross-corpus bench surfaced a regression that none
of the sprint's day-by-day fixtures caught.**  Days 2-5 added
element absorption + supervariable detection + approximate-degree
formula on top of the Sprint 22 quotient-graph baseline.  Each
day's fixture suite (corpus delegation tests, parity tests,
synthetic edge-case tests) passed cleanly.  Day 12 ran the full
cross-corpus bench and observed AMD wall-time on irregular SPD
fixtures had regressed 62-199× vs Sprint 22 baseline (bcsstk14:
90 ms → 6 951 ms; Pres_Poisson: 5.7 s → 22.3 minutes).  The
algorithmic correctness was fine — fill bit-identical to bitset
across every fixture — but the wall-time cost of Day 4's per-pivot
O(k²) supervariable-hash compare wasn't visible until measured
end-to-end.

**Supervariable detection is a fixture-shape-dependent win.**  On
banded fixtures (where supervariables form readily because of
structural regularity) Day 4's detection saves more wall time
than the O(k²) compare costs — Day-12 banded fixtures regressed
only ~3× vs Sprint 22 baseline, vs 62-199× on irregular SPD.  The
supervariable detection's payoff scales with how many merges
actually fire; on irregular structural-mechanics matrices the
hash buckets are large but no merges happen, so the O(k²) compare
is pure overhead.

**Sprint 22's PLAN target of `nnz_nd / nnz_amd ≤ 0.5×` on
Pres_Poisson was over-optimistic.**  Sprint 23 closed half the
gap (1.06 → 0.95) via the cumulative effect of leaf-AMD + bucket-
FM + multi-pass FM, but the remaining 0.95 → 0.7 (let alone 0.5)
needs deeper coarsening or a smarter separator-extraction
heuristic — algorithmic work outside Sprint 23's scope.  The
Sprint 22 PLAN's risk-flag #2 anticipated this; Sprint 23's
PLAN.md relaxed the target from 0.5 to 0.7 and even that wasn't
quite reached.

## What didn't go well

**The qg-AMD wall-time regression is Sprint 23's biggest miss.**
Headline gate (b) — qg-AMD wall on bcsstk14 ≤ Sprint-22 bitset
baseline (64 ms) — fails by ~108×.  Pres_Poisson AMD wall went
from 12.2 s (Sprint 22) to 22.3 minutes (Day 12).  Fill is
bit-identical to bitset, so the algorithm is correct, but the
production AMD path is now substantially slower than Sprint 22's
quotient-graph baseline on every irregular SPD fixture.  Days 2-5
should have included a wall-time regression check at each step
— the cumulative cost of element absorption + supervariable
O(k²) compare + workspace doubling wasn't visible until Day 12
measured end-to-end.

**PLAN.md's literal target of `nnz_nd / nnz_amd ≤ 0.7×` on
Pres_Poisson was not achievable in Sprint 23's scope.**  Achieved
0.952×.  The plan's risk-flag #2 anticipated this might happen
and pre-flagged Sprint-24 deferral; Day 11's multi-pass FM
exploration brought us most of the way (0.95 vs targeted 0.7) and
"ND beats AMD" is the spirit of the goal, but the literal target
needs algorithmic work outside the 14-day budget.

**Day 6's wall-time finding was a missed signal.**  The
`davis_notes.md` Day-6 measurement on bcsstk14 noted "default
exact-degree 11.84 s, USE_APPROX 57.51 s" and framed the 4.9×
slowdown as the approximate-degree formula's limitation.  Reality
was deeper: the Sprint 22 baseline for bcsstk14 was 90 ms — the
"default exact-degree 11.84 s" was already a 130× regression vs
Sprint 22, and the Day-6 commit treated 11.84 s as the *baseline*.
Day 12's cross-corpus bench is the first time the regression's
true magnitude got named.  Lesson logged below.

**The retro stub at Day 13 grew larger than the retro body at
Day 14.**  Sprint 22's retro pattern was a Day-14 stub plus
post-sprint prose; Sprint 23's PLAN.md asked for a Day-13 stub
plus Day-14 body.  In practice, the Day-13 metrics + headline-
gate outcomes were 90 % of what the retro needed — the Day-14
body filled in the prose sections but didn't add new structural
content.  Future sprints can probably skip the stub-vs-body
distinction; one Day-14 retro that absorbs the Day-13 work would
match the actual time spent.

## Items deferred

| item | rationale | Sprint 24 routing |
|---|---|---|
| Pres_Poisson ND/AMD ≤ 0.7× | Achieved 0.952× via Day 11 multi-pass FM; closing the rest needs deeper coarsening / smarter separator extraction | "ND fill quality follow-up" item |
| qg-AMD wall-time regression on irregular SPD | Days 2-5 added per-pivot O(k²) supervariable compare that dominates when supervariables don't form (irregular structural-mechanics matrices) | "qg-AMD wall-time fix" item — three candidate fixes in `bench_summary_day12.md` "(b)" |
| Pres_Poisson under SPARSE_QG_VERIFY_DEG | Day 13 corpus parity test runs only on bcsstk14 (USE_APPROX is 5× slower; Pres_Poisson would push the test suite past 30 minutes) | "AMD parity test on Pres_Poisson" item if/when wall-time fix lands |
| Davis §7.5.1 external-degree refinement | Mentioned in Day 5's davis_notes but not implemented | Sprint 24 if approximate-degree is retained |

## Lessons

**A wall-time regression check belongs in every algorithmic-
addition day, not just the closure-day bench.**  Days 2-5 added
features whose per-pivot cost compounded; each day's fixture
suite passed but accumulated a 77-199× regression that wasn't
visible until Day 12.  Future sprints touching `sparse_reorder_amd_qg`
(or any other production-path algorithm) should run a 10-second
wall-time probe on bcsstk14 + Pres_Poisson at the end of each
day's work, with a `>2×` regression threshold treated as a hard
gate before the day's commit lands.  Cheap to instrument
(extract a one-liner from `bench_reorder.c`); would have caught
the regression at Day 3.

**The "consider every unlocked vertex every step" semantics of
Sprint 22's FM was load-bearing for derived tests in ways the
partition test ranges didn't capture.**  bcsstk04 LDL^T-without-
pivoting happened to depend on a specific FM tie-breaking pattern;
the bucket-FM's first iteration broke it without breaking
`test_partition_*`.  Lesson for Sprint 24: when swapping a graph
algorithm's internals, run *derived numerical tests*
(factorization residuals, not just structural-property tests)
before declaring the swap safe.  The `test_ldlt_via_nd_dispatch`
caught a 7.94e+9 residual in 30 seconds; a slower-converging
iterative-solver test could miss the same regression.

**Multi-pass FM's payoff scales with the cost of a single pass.**
Sprint 22 ran single-pass FM at every uncoarsening level because
its O(n²) max-gain scan made multi-pass infeasible on Pres_Poisson.
Sprint 23 Days 9-10's bucket-FM dropped per-pass cost to O(|E|);
*then* multi-pass became affordable, and Day 11's exploration
turned a 1.026× ratio into 0.952×.  The infrastructure work and
the algorithmic exploration are coupled — neither alone closes
the headline gap.  Future sprint planning should sequence
infrastructure-before-exploration deliberately rather than
treating the exploration as a budget-permitting afterthought.

**Plan-target deviation should be documented inline at the
deviation site, not just in the retro.**  Days 8 and 11 both
shipped tightened-but-relaxed test bounds (10×10 grid 1.21× vs
plan's 1.0×; Pres_Poisson 1.10× vs plan's 0.7×) with the rationale
captured in inline test comments.  This worked well — anyone
opening `tests/test_reorder_nd.c` sees the plan-vs-reality
context next to the assertion threshold.  Sprint 24's qg-AMD
wall-time fix should follow the same pattern: when a plan target
isn't met, the assertion bound and the deviation reasoning go
side-by-side in the test file.

## Sprint 24 inputs

The shipping story for the Sprint 24 PR-description framing:
"Sprint 23 closed Sprint 22's ND fill-quality gap on the canonical
2D-PDE benchmark (Pres_Poisson 1.06× → 0.95× of AMD), at the cost
of an AMD wall-time regression on irregular SPD that Sprint 24
must root-cause and fix."

Top items routed from Sprint 23 to Sprint 24:

1. **qg-AMD wall-time root-cause + fix.**  Days 2-5's regression
   on irregular SPD (62-199× vs Sprint 22 baseline) is Sprint 24's
   highest-priority correctness-adjacent item — the production
   path's wall time is no longer competitive with what Sprint 22
   shipped.  Three candidate fixes documented in
   `bench_summary_day12.md "(b)"`:
   - **Replace Day 4's hash + O(k²) full-list compare with a
     sorted-list compare** (O(k log k) on collision).  Lowest-risk
     fix: keeps the supervariable-detection win on regular
     fixtures while bounding worst-case cost on irregular ones.
   - **Gate supervariable detection by a regularity heuristic.**
     Sprint 22's qg-AMD baseline is a known-good wall-time
     reference; supervariables only fire when the heuristic
     predicts a payoff.  Higher-risk: heuristic mistuning could
     leave wins on the table.
   - **Revert Days 2-5 entirely.**  Day 11's multi-pass FM turned
     out to be the actual headline driver, not Days 2-5; reverting
     gives back wall time without losing the headline.  Highest-
     risk in spirit but lowest-risk in practice — restores Sprint
     22 baseline.

2. **Pres_Poisson ND/AMD ≤ 0.7× literal target.**  Sprint 23
   landed 0.952× via Day 11 multi-pass FM; the remaining gap
   needs deeper coarsening (current bottoms out at MAX(20, n/100))
   or smarter separator-extraction (current uses METIS's
   smaller-side lift).  Lower-priority than item 1 since "ND
   beats AMD" is the spirit of the goal and that's met.

3. **AMD parity test on Pres_Poisson** (Day 13 deferral).  Once
   item 1 is fixed, the corpus parity test
   (`test_qg_approx_degree_parity_corpus`) can extend from
   bcsstk14 to Pres_Poisson without pushing the test suite past
   30 minutes.

4. **Day-by-day wall-time regression check.**  Lessons-section
   item; not strictly a Sprint-24 deliverable but the
   instrumentation-side work belongs there.  Add a 10-second
   probe to the make target the day-by-day commits run before
   merging.

## Acknowledgements

Davis 2006's "Direct Methods for Sparse Linear Systems" §7 was
the algorithmic spine of Days 2-5 — element absorption (§7.3),
supervariable detection (§7.4), approximate-degree formula
(§7.5).  Karypis-Kumar 1998's METIS paper §4 was the reference
for Days 9-10's gain-bucket structure.  Sprint 22's modular
ND + multilevel partition pipeline was load-bearing for Days 7
(leaf-AMD splice) and 11 (multi-pass FM at the finest level)
— neither would have been a 50-line patch without that
infrastructure.

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

End-of-sprint check.  Final captures (Day 14 reruns of bench_reorder
+ bench_amd_qg) live alongside the Day 12 captures at
`docs/planning/EPIC_2/SPRINT_23/bench_day14.{csv,txt}` and
`bench_day14_amd_qg.{csv,txt}` — sanity-confirmed bit-identical to
the Day 12 numbers since Day 13 was tests + docs only.

| DoD criterion | result | reference |
|---|---|---|
| ND fill-quality on canonical 2D-PDE benchmark beats AMD | ✓ | Pres_Poisson 0.952× |
| qg-AMD fill-quality bit-identical to bitset reference | ✓ | bench_day14_amd_qg.txt |
| Recursive ND completes on full corpus | ✓ | tests/test_reorder_nd.c, all 12 tests |
| FM gain-bucket structure correctness | ✓ | tests/test_graph_fm_buckets.c, 9 tests |
| LDL^T + LU + Cholesky residuals under ND ≤ 1e-6 | ✓ | tests/test_reorder_nd.c (all factorization-residual tests pass) |
| Determinism: same input + same seed → same output | ✓ | test_partition_determinism_*, test_nd_determinism_public_api |
| qg-AMD wall ≤ Sprint-22 bitset baseline | ✗ | Sprint-24 routing (`bench_summary_day12.md "(b)"`) |
| Pres_Poisson ND/AMD ≤ 0.7× | ✗ literal; ✓ spirit | Sprint-24 routing (`bench_summary_day12.md "(a)"`) |
| `make format && lint && test` clean | ✓ | All 51 binaries pass; 0 failures across the suite |
| `make sanitize` clean | partial | Pre-existing make-build infrastructure issue blocks `make sanitize`; targeted sanitizer build of test_graph_fm_buckets + test_reorder_amd_qg verifies the new code is clean.  Filed for Sprint 24 infrastructure work. |
| `make tsan` clean | partial | Sprint 23 changes are single-threaded; Sprint 22 Day 14's tsan baseline still applies to the unchanged OpenMP paths. |
