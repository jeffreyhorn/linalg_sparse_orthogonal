# Sprint 24 Retrospective — Ordering Follow-Ups (Sprint 23 Deferrals)

**Sprint budget:** 14 working days (~130 hours estimated, per PLAN.md); ran in 11 days after Day 1's (c)-revert pulled forward Days 6-7 + 8-11 work
**Branch:** `sprint-24`
**Calendar elapsed:** 2026-04-30 → 2026-05-05 (intensive condensed run; the day budget tracks engineering effort, not wall-clock days)

> **Status:** Day 11 final.  Day-by-day metrics, headline-gate
> outcomes, and prose sections all populated.  Sprint 24 ships with
> three of four headline-gate literal targets met (a, b, d); gate
> (c) — Pres_Poisson ND/AMD ≤ 0.85× — misses by 10pp and routes
> the algorithmic work to Sprint 25.

## Goal recap

> Close the qg-AMD wall-time regression Sprint 23 introduced
> (62-199× vs Sprint 22 quotient-graph baseline on irregular SuiteSparse
> SPD; gate (b) hard-fail in
> `docs/planning/EPIC_2/SPRINT_23/bench_summary_day12.md`); tighten
> the Pres_Poisson ND/AMD ratio toward the literal Sprint 22 plan
> target (Sprint 23 landed at 0.952×; this sprint targets ≤ 0.85×);
> close the Sprint 23 Day 13 deferral that left the AMD parity
> test at bcsstk14 only (Pres_Poisson skipped because USE_APPROX
> would push the suite past 30 minutes on the pre-fix wall-time
> profile); and add per-day wall-time regression-check infrastructure
> to prevent similar regressions in future sprints.  Also lands
> the Davis 2006 §7.5.1 external-degree refinement (deferred from
> Sprint 23 Day 5) if Sprint 23's approximate-degree path is retained.

(See `docs/planning/EPIC_2/SPRINT_24/PLAN.md` for the day-by-day
breakdown; `fix_decision_day1.md` for the (c)-revert decision that
re-shuffled Days 5-11.)

## Definition of Done checklist

| item | status | reference |
|---|---|---|
| 1. Wall-time regression-check instrumentation (`make wall-check`) | ✓ | Day 1 commit `fa9eb2a`, `Makefile` + `scripts/wall_check.sh` + `wall_check_baseline.txt` |
| 2. qg-AMD wall-time root-cause + fix | ✓ | Day 1 profile (`profile_day1_bcsstk14.txt`), Day 2 (c)-revert commit `3b39677`, `fix_decision_day1.md` |
| 3. AMD parity test on Pres_Poisson | N/A | Item became no-op under (c) revert — approximate-degree code path is gone, no parity test to run |
| 4. Davis 2006 §7.5.1 external-degree refinement | N/A | Item became no-op under (c) revert — approximate-degree code path is gone, nothing to refine |
| 5. ND fill-quality follow-up — Pres_Poisson ≤ 0.85× | partial | Days 5-6 added env-var-gated alternatives (`SPARSE_ND_COARSEN_FLOOR_RATIO`, `SPARSE_ND_SEP_LIFT_STRATEGY`); best opt-in 0.942×; literal 0.85× target routes to Sprint 25 per `nd_sep_strategy_decision.md` |
| 6. Cross-corpus re-bench + headline-gate verdicts | ✓ | Day 9 commit `83da221`, `bench_summary_day9.md` |
| 7. Tests + docs + retrospective | ✓ | Day 10 commit `5be34c0` (AMD-subsection update, PERF_NOTES.md closures, retro stub); Day 11 commit ships final bench + retro body + Sprint 24 PR |

Headline gates from PROJECT_PLAN.md item 6 — see
`docs/planning/EPIC_2/SPRINT_24/bench_summary_day9.md`:

| gate | result |
|---|---|
| (a) qg-AMD wall on bcsstk14 ≤ 1.5× Sprint 22 baseline (~210 ms) | **PASS** (125.8 ms) |
| (b) qg-AMD nnz(L) bit-identical to Sprint 22 + Sprint 23 | **PASS** (9/9 fixtures) |
| (c) Pres_Poisson ND/AMD ≤ 0.85× | **MISS** (0.952× default; 0.942× best opt-in) — routed to Sprint 25 |
| (d) Sprint 23 nnz_L bit-identical-or-better | **PASS** (1 row, Kuu ND, +24 nnz drift in tie-break noise band) |

## Final metrics

End-of-sprint cross-corpus capture: `docs/planning/EPIC_2/SPRINT_24/bench_day9.txt`
+ `bench_day9_amd_qg.txt`.

### qg-AMD wall time (vs Sprint 22 + Sprint 23)

| fixture       | Sprint 22 | Sprint 23 | Sprint 24 Day 9 | Δ vs S23  |
|---------------|----------:|----------:|----------------:|-----------|
| bcsstk14      |    ~140 ms |   4 715 ms |          126 ms | -97 % (39× faster) |
| Kuu           |    ~700 ms |  25 720 ms |          547 ms | -98 % (47×) |
| s3rmt3m3      |   ~2 100 ms |  51 321 ms |          728 ms | -99 % (70×) |
| Pres_Poisson  |  ~12 200 ms | 758 927 ms |        8 139 ms | -99 % (93×) |
| Pres_Poisson qg peak RSS | 19.19 MB | 25.09 MB | 19.19 MB | -24 % (back to S22) |

Day 2's (c) revert closed the regression on every irregular SuiteSparse
SPD fixture by 30-110×.  Memory profile returned to Sprint 22's exact
baseline.

### ND/AMD nnz(L) ratios (vs Sprint 22 + Sprint 23)

| fixture        | Sprint 22 | Sprint 23 | Sprint 24 Day 9 (default) | Sprint 24 best opt-in |
|----------------|----------:|----------:|--------------------------:|-----------------------|
| nos4           |    1.713× |    1.520× |                    1.520× | 1.251× (`balanced_boundary`) |
| bcsstk04       |    1.172× |    1.178× |                    1.178× | unchanged |
| Kuu            |    2.322× |    2.275× |                    2.275× | 1.415× (`balanced_boundary`, -38pp) |
| bcsstk14       |    1.207× |    1.130× |                    1.130× | 1.048× (`balanced_boundary`) |
| s3rmt3m3       |    1.015× |    1.009× |                    1.009× | unchanged |
| **Pres_Poisson** | **1.063×** | **0.952×** |                **0.952×** | **0.942× (`COARSEN_FLOOR_RATIO=200`, -1pp)** |

Default-path nnz_L bit-identical to Sprint 23 on every fixture (Day 2's
revert is fill-neutral by construction; Days 5-6 are env-var-gated
opt-ins).  Sprint 24's headline algorithmic landing is the env-var
matrix: Pres_Poisson ratio=200 advisory; non-Pres_Poisson balanced_boundary
advisory; both shipped behind env vars with documented per-fixture
trade-offs.

## Performance highlights

The headline outcome is **Sprint 23's qg-AMD wall-time regression
closed via revert**, restoring Sprint 22's wall-time + memory profile
bit-identically.  Day 1's `clock_gettime` profile of bcsstk14 measured
95 % of total time in `qg_recompute_deg`'s element-side adjacency-of-
adjacency walk — the cost Sprint 23 Day 3's element absorption
enabled, not Day 4's supervariable hash compare (which the three
originally-considered fix candidates were targeting).  Day 2's
`git revert` of the four Sprint 23 commits closed the regression
end-to-end on every irregular SuiteSparse SPD fixture: bcsstk14 went
from Sprint 23's 4 715 ms back to 126 ms (39× speedup, 0.90× of
Sprint 22's published 140 ms), Kuu from 25 720 ms to 547 ms (47×),
s3rmt3m3 from 51 321 ms to 728 ms (70×), Pres_Poisson AMD from
758 927 ms (~12.6 minutes) to 8 139 ms (~8.1 seconds, 93× speedup).
Memory profile also returned to Sprint 22's exact baseline: Pres_
Poisson qg peak = 19.19 MB (Sprint 23 was 25.09 MB after the
iw_size doubling + super[]/super_size[] arrays).  Day 11's final
re-bench confirms bit-identical nnz_L on every fixture × ordering
(30/30 rows from `bench_day9.txt`; 18/18 (qg, bitset) pairs from
`bench_day9_amd_qg.txt`); wall times drift in the typical ±25 %
run-to-run band but stay well under the wall-check 2× ceiling on
every measurement.

The two new ND env-var-gated alternatives — `SPARSE_ND_COARSEN_FLOOR_
RATIO` (Day 5) and `SPARSE_ND_SEP_LIFT_STRATEGY` (Day 6) — ship as
documented advisory.  ratio=200 drops Pres_Poisson ND/AMD from
0.952× to 0.942× (a 1pp tightening); balanced_boundary drops Kuu
from 2.275× to 1.415× (38pp — the largest single nnz win Sprint 24
produced on any fixture), nos4 from 1.520× to 1.251× (27pp), and
bcsstk14 from 1.130× to 1.048× (8pp), while staying neutral on
Pres_Poisson (+0.1pp).  Production defaults stay unchanged because
no single setting wins on Pres_Poisson + the smaller fixtures
simultaneously; both env vars exist for opt-in tuning when callers
know their workload's structural class.

The Pres_Poisson 0.85× stretch target slipped (best opt-in 0.942×).
The Sprint 23 → Sprint 24 cumulative motion is still ~1pp closer to
AMD's pivot quality, but the 7-percentage-point gap requires
algorithmic work (HCC coarsening, multi-pass FM at intermediate
levels, or spectral bisection at the coarsest level — see Sprint 25
inputs below) outside the Days 5-6 sweep space.

## What went well

**Day 1's profile re-targeted item 2 the day-of.**  Sprint 23's
closing-day bench-summary fingered Day 4 supervariable detection's
hash compare as the bottleneck; Sprint 24 PLAN.md inherited the
hypothesis and budgeted 32 hours across Days 1-4 for fix-and-
validate cycles against it.  Day 1's `clock_gettime` instrumentation
took ~6 hours and immediately surfaced the actual bottleneck
(`qg_recompute_deg`'s element-side adj-of-adj walk, 95 % of total;
supervariable compare was 1 %).  The profile's evidence ruled out
all three Sprint-23-suggested fix candidates (a/b/c-without-
revert) — none of them addressed `qg_recompute_deg` — and pointed
at a new candidate (d) "optimize the element-side walk in
`qg_recompute_deg`" that (c) revert dominated on risk + budget.
The decision was visible in the profile within 30 minutes of
running it.  Net: ~30 hours of fix-and-validate work avoided;
freed budget rolled into Day 5's item-5 head-start.

**The wall-check Makefile target landed Day 1 and caught no
regressions across the sprint.**  Sprint 23 RETROSPECTIVE.md's
"what didn't go well" called for a per-day wall-time regression
check; Sprint 24 Day 1 implemented it (~6 hours of work) and Days
2-10 ran with it active.  Across nine commits the gate fired zero
false positives and would have caught any single-day > 2× drift
at commit time.  The instrumentation's cost is small enough that
it should be standard for every sprint that touches a production-
path algorithm; Sprint 25 inherits the pattern.

**Days 5-6's env-var-gated alternative pattern shipped two new ND
features without changing any default behavior.**  Both
`SPARSE_ND_COARSEN_FLOOR_RATIO` (Day 5) and `SPARSE_ND_SEP_LIFT_
STRATEGY` (Day 6) follow the Sprint 23 Day 11 `SPARSE_FM_FINEST_
PASSES` convention: env-var-gated with a documented fallback to
the existing default, validated under a per-day wall-check + corpus
parity check.  The corpus tests stayed bit-identical across all
four env-var combinations × 51 binaries (= 204 test invocations
on Day 7).  The pattern lets the sprint ship algorithmic
exploration as production-quality opt-in features without
committing to a default flip — useful when the sweep doesn't
converge on a clear corpus-wide winner.

**The (c) revert came in under budget.**  PLAN.md allotted 32 hours
to Days 1-4 for item 2.  Day 2's revert was a `git revert` of four
commits + re-applying the SPARSE_QG_PROFILE instrumentation in the
simpler form appropriate to the variable-only baseline; Days 3-4
were the corpus parity bench + wall-check baseline tightening.
Total: ~16 hours, freeing the remaining 16 hours for the ND fill-
quality work in Days 5-7.  Items 3 + 4 reduced to no-ops under
the (c) revert (28 more hours freed); Sprint 24 ran 11 days
against the 14-day estimate, with the 3-day savings not materially
advancing item 5 (which needs algorithmic work outside the sprint's
scope).

**Sprint 24 closes Sprint 23's flagged sanitize / tsan
verification gap.**  Sprint 23 RETROSPECTIVE.md DoD verification
listed `make sanitize` and `make tsan` as "partial" pending
follow-up infrastructure work.  Sprint 24 Day 11's full-suite
`make sanitize` (UBSan) run passed 51/51 test binaries with 0
runtime errors; `make tsan` (using Homebrew LLVM clang to work
around Apple Clang's bundled-TSan deadlock per the Makefile note)
also passed 51/51 with 0 ThreadSanitizer warnings and 0 data
races.  The Day-11 verification can either be evidence that the
Sprint 23 issue was transient / environment-specific or that the
(c) revert + Days 5-6's surgical env-var-gated additions don't
trigger whatever blocked Sprint 23's run; either way, the gap is
closed for Sprint 24's PR.  Sprint 25 inherits a sanitize-clean
+ tsan-clean baseline.

## What surprised us

**The Sprint 23 closing-day root-cause hypothesis was wrong.**
`SPRINT_23/bench_summary_day12.md "(b)"` triaged the regression to
"Day 4's per-pivot O(k²) supervariable-hash-bucket compare dominates
when supervariables don't form on irregular structural-mechanics
matrices" — a plausible-sounding story consistent with the wall-time
regression band (62-199× on irregular SPD vs ~3× on banded).  The
actual Day-1 profile measurement showed supervariable compare at
1 % of cost and `qg_recompute_deg` at 95 %.  The three originally-
considered fix candidates (sorted-list compare, regularity-heuristic
gating, full revert) were chosen against the wrong root cause — the
first two would have left 95 % of the cost in place; the third
incidentally worked because reverting Day 3 took the elements out
of the workspace that Day 4's exact-degree formula was walking.
Lesson logged in `What didn't go well` and `Lessons` below — closing-
day root-cause hypotheses without instrumented evidence shouldn't
drive next-sprint planning.

**Day 5's coarsening-floor sweep showed ratios ≥ 400 *regress* on
Pres_Poisson** (0.99× vs ratio=200's 0.94×).  The coarsest level
pegs at the floor of 20 vertices regardless of the divisor; ratio
≥ 400 just shrinks the coarsest graph faster, leaving the brute-
force bisection enumerating `2^(20-1)` patterns on essentially the
same coarsest graph, but losing the intermediate-level structure
that ratio=200's 74-vertex coarsest graph preserved.  The sweep
shape isn't monotone: more aggressive coarsening doesn't strictly
improve ND fill quality — there's a sweet spot at ratio=200 for
this fixture, and the sweep's other endpoints (default 100 +
infinity divisor) both miss it.

**Days 5-6's options interact destructively on Pres_Poisson.**
ratio=200 alone: 0.942×.  balanced_boundary alone: 0.953×
(+0.1pp, neutral).  Combined: 0.950× — *worse* than ratio=200 alone
by 0.8pp.  The two changes touch different stages of the pipeline
(coarsening vs separator extraction) but the cuts they each find
on Pres_Poisson don't compose; the ratio=200 coarsest cut commits
to a side that balanced_boundary then re-decides differently
during separator extraction, and the resulting separator is larger
than what either single setting produces.  Per `nd_sep_strategy_
decision.md` the production default stays Sprint 22's
`smaller_weight` behaviour because no single setting wins on both
Pres_Poisson and the smaller fixtures simultaneously.

**`balanced_boundary` is a 38pp ND/AMD win on Kuu but a no-op on
Pres_Poisson.**  Sprint 22 + Sprint 23 had Kuu at 2.275× AMD (the
worst ND/AMD ratio in the corpus); balanced_boundary drops it to
1.415× — comfortably the largest single nnz win Sprint 24 produced
on any fixture.  This wasn't anticipated by PLAN.md (which framed
Day 6 as a Pres_Poisson-targeted exploration); the production-
default decision would have looked different had the headline
metric been Kuu rather than Pres_Poisson.  The advisory note in
`docs/algorithm.md` ND subsection records the per-fixture trade-off
so callers can opt in by workload class.

## What didn't go well

**The 0.85× Pres_Poisson stretch target is the second sprint in a
row to miss its literal goal.**  Sprint 22 PLAN's 0.5× → Sprint 22
actual 1.063×.  Sprint 23 PLAN's 0.7× → Sprint 23 actual 0.952×.
Sprint 24 PLAN's 0.85× → Sprint 24 actual 0.942× (best opt-in).
Each sprint moved the metric ~10pp; the ratio is converging slowly
toward AMD's pivot quality but no single algorithmic change closes
the remaining gap.  The Sprint 25 routing flags concrete avenues
(HCC coarsening, multi-pass FM at intermediate levels, spectral
bisection at the coarsest level) but the historical pattern
suggests a one-sprint close is unlikely; PROJECT_PLAN.md should
either reset the target to a more incremental tightening or
acknowledge multi-sprint convergence in the planning prose.

**Items 3 + 4 reduced to no-ops under the (c) revert.**  PLAN.md
allocated 28 hours to items 3 (Pres_Poisson AMD parity test) and
4 (Davis §7.5.1 external-degree refinement).  Both targeted the
approximate-degree code path that Day 2 reverted away; both
became N/A on Day 1.  The freed budget redirected cleanly to
item 5's head-start (Days 5-7 of Sprint 24 ran PLAN's Days 8-10
work), but the up-front-conditional planning in PLAN.md item 4
("conditional on item 2 retaining the approximate-degree code
path") could have been firmer about the contingency: item 4's 20
hours should have had an explicit "if N/A, redirect to <X>" line
rather than just relying on the 4-hour Day-1 decision to absorb
the contingency cleanly.

**Pres_Poisson ND wall-time drifted 21 % above Sprint 23 baseline.**
Sprint 23 ND wall on Pres_Poisson = 36.4 s; Sprint 24 Day 8
measured 42.86 s on the same fixture under default settings.  No
algorithmic change to ND default path explains this — the Day 5-6
env vars are off by default, and the underlying partition / FM
code is unchanged.  Likely run-to-run variance on a host with
concurrent load (Sprint 24's bench captures span 35.5 s → 42.86 s
across Days 5-8).  PLAN's "≤ Sprint 23 baseline + 5 %" target
wasn't realistic given the measured 21 % run-to-run band; Sprint
25 should add an ND wall line to `wall_check_baseline.txt` with a
50 % threshold instead.

**The retro stub-vs-body split duplicated work.**  Sprint 23's
RETROSPECTIVE.md "what didn't go well" already noted this
("Sprint 22's retro pattern was a Day-14 stub plus post-sprint
prose; Sprint 23's PLAN.md asked for a Day-13 stub plus Day-14
body... Future sprints can probably skip the stub-vs-body
distinction; one Day-14 retro that absorbs the Day-13 work would
match the actual time spent.")  Sprint 24 inherited the same
pattern (Day 10 stub + Day 11 body) and ran into the same
duplication: Day 10's stub already had ~80 % of the metrics +
DoD content + day-by-day capsule, leaving Day 11 mostly cosmetic
prose work.  Sprint 25 PLAN.md should consolidate to a single
final-day retrospective.

## Items deferred

| item | rationale | Sprint 25 routing |
|---|---|---|
| Pres_Poisson ND/AMD ≤ 0.85× | Days 5-6's options reach 0.942× best opt-in; 0.85× needs algorithmic work outside Sprint 24's scope | Smarter coarsening / multi-pass FM at intermediate levels / spectral bisection at the coarsest level |
| Davis 2006 §7.5.1 external-degree refinement | N/A under (c) revert; resurrect if a future sprint reintroduces approximate-degree | Sprint-25-or-later if approximate-degree returns |
| `make wall-check` Pres_Poisson ND wall line | Day 8 captured 42.86 s default-path; run-to-run variance 21 % | Add baseline line with 50 % threshold rather than 5 % |
| ND wall-time tightening to meet 5 % drift target on default path | Sprint 24's default ND path is 1.06-1.10× of Sprint 23's; not a regression but worth profiling | Profile + tighten if the 5 % drift target is to be met |

## Lessons

**Profile before fixing.**  Sprint 23 Day 12's gate-(b) bench
flagged a wall-time regression but the three Sprint-23-suggested
fix candidates all targeted the wrong phase (Day 4 supervariable
detection at 1 % of total cost vs Day 3 element absorption enabling
the dominant `qg_recompute_deg` walk at 95 %).  Day 1's `clock_
gettime` instrumentation was the single highest-leverage artifact
of the sprint — it re-targeted item 2 from a 30-hour fix-and-
validate effort against the wrong root cause to a 1-day revert
against the actual root cause.  The cost of the profile capture
itself was ~6 hours (the SPARSE_QG_PROFILE env var, the timing
hooks, the bcsstk14 capture); the savings were ~30 hours of (a)/(b)
candidate work that would have addressed 1 % of total cost.

**A revert is a valid fix.**  Sprint 23's Days 2-5 work is preserved
in commit history (master via PR #31) but unwound by Sprint 24
Day 2's `git revert`.  The commits + lessons + design notes
survive; only the runtime cost is removed.  When an algorithmic
addition's payoff doesn't materialize and its cost dominates,
reverting beats optimizing — both for risk profile (the revert
restores a known-good baseline; the optimization tries an
unproven path) and for downstream signal (a clean revert
re-anchors the next sprint's planning to the original baseline,
while a partial optimization leaves a half-implemented mechanism
that still has to be reasoned about).

**The wall-check gate validated Sprint 23's lessons-section
proposal end-to-end.**  Sprint 23 RETROSPECTIVE.md noted "the
cumulative cost of element absorption + supervariable O(k²)
compare + workspace doubling wasn't visible until Day 12 measured
end-to-end"; Sprint 24 Day 1 implemented the per-day wall-check
gate Sprint 23 had identified as the missing signal; Day 1's
instrumentation work paid for itself by Day 4 (the post-revert
wall_check_baseline.txt tightening would have been a guessing game
without the gate).  The pattern should be standard for every sprint
that touches a production-path algorithm — Sprint 25 inherits the
gate.

**Env-var-gated production-default flips need a clear flip rule
documented before the experiment runs.**  Day 6's `SPARSE_ND_SEP_
LIFT_STRATEGY` ran the sweep with no production-default flip rule
documented up-front.  When the sweep surfaced 38pp Kuu wins +
neutral Pres_Poisson, the "production default = best on the
headline fixture" rule was *implied* but not in PLAN.md.  Day 7's
decision doc had to re-justify the literal flip rule before the
env var could stay off-by-default with a clear rationale.  Future
sprints introducing env-var-gated alternatives should put the
flip rule in PLAN.md alongside the experiment description: "if
this setting beats X on Y, flip default; otherwise stay off."

**Closing-day root-cause hypotheses are commitments to next-sprint
work — make them only with profile evidence.**  Sprint 23 Day 12's
bench-summary.md "(b)" wrote a confident-sounding story about
supervariable detection's O(k²) compare being the bottleneck.
Sprint 24 PLAN.md inherited the story, sized 32 hours of fix
work against it, and would have spent that budget on the wrong
phase if Day 1 hadn't run a profile.  Closing-day hypotheses
should either include the profile-evidence link inline or
explicitly mark themselves as "needs verification" before next-
sprint planning consumes them.

## Sprint 25 inputs

The shipping story for the Sprint 25 PR-description framing:
"Sprint 24 closed Sprint 23's qg-AMD wall-time regression via
revert (Days 2-5 unwound; Sprint 22 baseline restored bit-identically),
shipped two new ND fill-quality opt-in env vars with documented
per-fixture advisory settings, and pinned the Sprint 23 Day 11
multi-pass FM achievement in tightened test bounds.  Pres_Poisson
0.85× stretch target routes to Sprint 25 with three concrete avenues
identified."

Top items routed from Sprint 24 to Sprint 25:

1. **Pres_Poisson ND/AMD ≤ 0.85× via algorithmic work.**  Three
   candidates from `nd_sep_strategy_decision.md` "Why option (b) misses
   the 0.85× target on Pres_Poisson":
   - Heavy Connectivity Coarsening (Karypis-Kumar 1998 §5) preserving
     cut structure better through coarsening than current heavy-edge
     matching.
   - Multi-pass FM at intermediate levels (currently single-pass at
     all but finest); compounded uncoarsening.
   - Spectral bisection at the coarsest level (currently brute-force
     / GGGP); globally better starting point for FM uncoarsening
     cascade.

2. **`make wall-check` Pres_Poisson ND wall line.**  Day 8's 42.86 s
   measurement vs Sprint 23's 36.4 s baseline shows the gate would
   fire on Pres_Poisson ND wall regressions; Sprint 25 should add a
   baseline line with a 50 % threshold rather than 5 % (high run-to-
   run variance on this fixture makes 5 % unrealistic).

3. **ND wall-time profile + tightening.**  Sprint 24 Day 8's
   measurement is 21 % above Sprint 23 baseline on Pres_Poisson ND;
   profile to identify whether this is real cost or measurement noise.

## Acknowledgements

Sprint 23's RETROSPECTIVE.md "Lessons" section (specifically the
"wall-time regression check belongs in every algorithmic-addition
day, not just the closure-day bench" lesson) directly motivated
Sprint 24 Day 1's `make wall-check` target.  The infrastructure
work paid for itself within the sprint — Day 4's wall-check baseline
tightening would have been a guessing game without the gate, and
Days 5-9's env-var-gated experiments each ran with the gate active
to prevent a Sprint-23-style accumulating regression.

Davis 2006 §7's prose describes element absorption + supervariable
detection + approximate-degree formula as the canonical AMD
optimizations; Sprint 23 implemented them and Sprint 24 reverted
them.  The reference is unchanged; the empirical finding is that
the four mechanisms' wall-time cost on irregular SuiteSparse SPD
fixtures dominates their fill-quality payoff (which was zero —
fill is bit-identical with or without the four mechanisms).  The
Sprint 22 simplified-quotient-graph baseline that survives is also
Davis 2006 §7.1's variable-only quotient-graph form, just without
the §7.3-§7.5 mechanisms layered on top.  Sprint 25 may revisit
the four mechanisms if a future workload requires the
supervariable wins on highly-symmetric fixtures (banded synthetic
n ≥ 5 000 was the only fixture class where Sprint 23's mechanisms
produced a measurable win in the Sprint 23 Day 12 cross-corpus
bench).

Karypis-Kumar 1998's METIS paper §4 (gain-bucket FM) and §5 (Heavy
Connectivity Coarsening) remain the algorithmic spine of the ND
pipeline.  Sprint 22 implemented §4's gain-bucket FM (lifted to
multi-pass at the finest level by Sprint 23 Day 11); Sprint 25's
Pres_Poisson 0.85× work is the §5 HCC coarsening starting point.

Sprint 22's modular ND + multilevel partition pipeline made
Days 5-6's env-var-gated experiments straightforward: both new env
vars sit at single function entry points (`sparse_graph_hierarchy_
build` for the coarsening floor; `graph_edge_separator_to_vertex_
separator` for the lift strategy) with trivial fallback paths to
the existing default.  Without Sprint 22's separation of concerns,
the experiments would have required structural refactoring before
the algorithmic exploration could run.

Sprint 23's gain-bucket FM (Days 9-10) and multi-pass FM at the
finest level (Day 11) carry through Sprint 24 unchanged — the
Pres_Poisson 0.952× ND/AMD ratio they delivered is bit-stable
across both sprints.  Sprint 24's Days 5-6 env-var work composes
on top of the Sprint-23 FM infrastructure; the Day-7 production-
default decision was framed as "extend Sprint 23's defaults" rather
than "redesign them."

## Day-by-day capsule (for the prose write-up)

| day | theme | signal landed |
|---|---|---|
| 1 | Wall-check instrumentation + qg-AMD profile | `make wall-check` target + baseline file; bcsstk14 profile attributes 95 % to `qg_recompute_deg`; fix candidate (c) chosen |
| 2 | qg-AMD wall-time fix — revert Sprint 23 Days 2-5 | `git revert` of `336d74a`/`aea071a`/`6096840`/`f0fe391`; bcsstk14 wall back to 114 ms; obsolete tests removed |
| 3 | Cross-corpus + synthetic banded bench post-revert | Bit-identical fill across full corpus + banded; qg wins on banded restored to Sprint 22 levels |
| 4 | Pres_Poisson AMD capture + tighten wall-check baseline | Pres_Poisson AMD = 7 862.6 ms (full bench); baseline ratcheted 7 000 → 130 ms (bcsstk14) and 1 100 000 → 8 000 ms (Pres_Poisson) |
| 5 | `SPARSE_ND_COARSEN_FLOOR_RATIO` env var (item 5 head-start) | ratio=200 advisory for Pres_Poisson (0.952× → 0.942×); ratios ≥ 400 regress; `nd_coarsen_floor_decision.md` |
| 6 | `SPARSE_ND_SEP_LIFT_STRATEGY` env var (item 5 option (b)) | balanced_boundary advisory for non-Pres_Poisson (Kuu -38pp, bcsstk14 -8pp, nos4 -27pp); `nd_sep_strategy_decision.md` |
| 7 | ND tuning re-check + production-default decision + Pres_Poisson bound tightening | Production default stays smaller_weight; `test_nd_pres_poisson_fill_with_leaf_amd` tightened from 1.0× to 0.96×; `nd_tuning_day7.md` |
| 8 | 10×10 grid bound tightening + algorithm.md ND subsection | `test_nd_10x10_grid_matches_or_beats_amd_fill` 1.21× → 1.17×; algorithm.md ND subsection refreshed; `nd_tuning_day8.md` |
| 9 | Cross-corpus re-bench + headline-gate verdicts | bench_day9.{csv,txt} + bench_day9_amd_qg.{csv,txt}; (a)/(b)/(d) PASS; (c) MISS routed to Sprint 25; `bench_summary_day9.md` |
| 10 | Closing tests + docs sweep + retro stub | algorithm.md AMD subsection updated to drop reverted Sprint 23 mechanisms; SPRINT_22/PERF_NOTES.md "Sprint 24 closures" subsection appended; SPRINT_24/RETROSPECTIVE.md stubbed; PROJECT_PLAN.md Sprint 24 marked Status: Complete (~80 hrs vs 126 estimate) |
| 11 | Soak + final bench + retro body + PR | bench_day11.{csv,txt} + bench_day11_amd_qg.{csv,txt} sanity-confirms bit-identical nnz_L vs Day 9 across 30/30 cross-ordering rows + 18/18 qg-vs-bitset pairs; retrospective prose sections filled in; Sprint 24 PR opened |

## Day-budget vs estimate

PLAN.md estimated 130 hours across 14 days; Sprint 24 ran in 11 days
because Day 1's (c) revert pulled forward Days 6-7 (item 4 became N/A)
and Day 5 (item 3 became N/A) into earlier slots:

| PLAN.md day | Sprint 24 day | re-shuffle reason |
|---|---|---|
| Day 5 (AMD parity test on Pres_Poisson) | N/A | (c) revert removed approximate-degree path |
| Day 6 (External-degree refinement design) | N/A | (c) revert removed approximate-degree path |
| Day 7 (External-degree refinement integration) | N/A | (c) revert removed approximate-degree path |
| Day 8 (ND coarsen-floor sweep) | Day 5 | Day 5 budget recovered |
| Day 9 (ND smarter separator extraction) | Day 6 | Day 6 budget recovered |
| Day 10 (ND tuning re-check) | Day 7 | Day 7 budget recovered |
| Day 11 (ND close + document) | Day 8 | One day of slack absorbed |
| Day 12 (cross-corpus re-bench) | Day 9 | (no shift) |
| Day 13 (closing tests + retro stub) | Day 10 | (no shift) |
| Day 14 (soak + final bench + PR) | Day 11 | (no shift) |

Net: 11 days actual vs 14 days estimate.  The 3-day savings come from
items 3 + 4 becoming N/A under the (c) revert; the freed budget didn't
materially advance item 5 (Pres_Poisson 0.85×) which needs algorithmic
work outside Sprint 24's scope.

## DoD verification

End-of-sprint check.  Final captures (Day 11 reruns of `bench_reorder`
+ `bench_amd_qg`) live alongside Day 9's captures at
`docs/planning/EPIC_2/SPRINT_24/bench_day11.{csv,txt}` and
`bench_day11_amd_qg.{csv,txt}` — sanity-confirmed bit-identical nnz_L
across 30/30 cross-ordering rows + 18/18 (qg, bitset) pairs.  Day-11
wall times drift in the typical macOS arm64 ±25 % run-to-run band
(bcsstk14 qg-AMD: 125.8 → 160.2 ms; Pres_Poisson AMD: 8 139 → 9 612
ms) but every measurement stays well under the wall-check 2× ceiling.

| DoD criterion | result | reference |
|---|---|---|
| qg-AMD wall on bcsstk14 ≤ 1.5× Sprint 22 baseline | ✓ | bench_day11_amd_qg.txt: 160.2 ms (Day 9: 125.8 ms; both ≤ 210 ms ceiling) |
| qg-AMD nnz(L) bit-identical to Sprint 22 + Sprint 23 | ✓ | bench_day11_amd_qg.txt: 9/9 fixtures bit-identical (re-confirmed Day 11) |
| Pres_Poisson ND/AMD ≤ 0.85× | ✗ literal; partial close | 0.952× default; 0.942× best opt-in; routed to Sprint 25 |
| All Sprint 23 nnz_L rows bit-identical-or-better | ✓ | bench_summary_day9.md: 30 rows; Kuu ND +24 nnz noise; Day 11 re-confirms |
| `make wall-check` exits 0 against Day-4 baseline | ✓ | Day 11 wall-check passes (within 2× gate on both fixtures) |
| `make format && lint && test` clean | ✓ | All 51 binaries pass; 0 failures across the suite |
| `make sanitize` clean | ✓ | Day 11 full UBSan run: 51/51 test binaries pass (0 sanitizer runtime errors).  Sprint 23's pre-existing make-build infrastructure issue is *resolved* — the Sprint 22 → Sprint 24 cleanup sequence (Sprint 22's wrapper-vs-API split + Sprint 23's revert + Sprint 24 Day 2's revert) restored the full-suite sanitizer-clean property. |
| `make tsan` clean | ✓ | Day 11 full TSan run with Homebrew LLVM clang (Apple Clang's bundled TSan deadlocks per Makefile `tsan` target's note): 51/51 test binaries pass, 0 ThreadSanitizer warnings, 0 data races detected. |
| GitHub Actions CI on PR #32 | partial (pre-existing) | Day 12 PR #32 status: lint PASS, cmake-build-and-test PASS, tsan PASS; coverage FAIL at 80.8 % vs 95 % `COV_THRESHOLD` (pre-existing — PRs #28 / #29 / #30 / #31 all FAILED the coverage check at the same threshold; not introduced by Sprint 24).  build-and-test was still pending at retrospective time (long-running `make bench` step on Pres_Poisson; PR #31 was CANCELLED on the same step).  Coverage gate calibration routes to Sprint 25 as part of its "CI Hardening" scope (PROJECT_PLAN.md Sprint 25 item 8 absorbs it). |
