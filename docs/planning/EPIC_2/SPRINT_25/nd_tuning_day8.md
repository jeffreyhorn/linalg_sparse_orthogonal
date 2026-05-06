# Sprint 25 Day 8 — Spectral Bisection Validation & Headline-Gate Escalation

## Summary

Day 8 lands the spectral path's edge-case validation + ships the
cross-corpus capture under `SPARSE_ND_COARSEST_BISECTION=spectral`.
**Spectral alone moves Pres_Poisson ND/AMD from 0.952× to 0.953×
— essentially unchanged**, the third Sprint 25 algorithmic axis to
miss the headline gate.  Per PLAN.md Day 8 task 4, this triggers
the escalation: "headline depends entirely on items 1+2+5 env-var
combinations" — but Day 9's combined-effect matrix sweep is
unlikely to find a constructive composition that closes the
remaining ~9pp gap.

Sprint 25's preliminary verdict (subject to Day 9 final
measurement): **Pres_Poisson 0.85× literal target slips again**;
the documented graceful-degradation path (partial close → Sprint 26
routing per Sprint 24 RETROSPECTIVE.md "What didn't go well" #1) is
the expected closure.

## Per-fixture spectral-only deltas (vs default)

| fixture | default ND | spectral ND | Δ | category |
|---|---|---|---|---|
| nos4 | 968 | 968 | 0 | = |
| bcsstk04 | 3 702 | 3 702 | 0 | = |
| Kuu | 924 385 | 708 650 | **-23.3 %** | huge WIN |
| bcsstk14 | 131 017 | 123 207 | **-6.0 %** | WIN |
| s3rmt3m3 | 478 890 | 485 822 | +1.4 % | noise |
| **Pres_Poisson** | **2 541 734** | **2 541 978** | **+0.01 %** | **= (escalate)** |

## Edge-case unit tests landed (Day 8)

Per PLAN.md Day 8 task 2:

| test | fixture | path exercised | status |
|---|---|---|---|
| `test_spectral_bisection_n1` | n=1 single-vertex graph | trivial-size early return | PASS |
| `test_spectral_bisection_n2` | n=2 two-vertex edge | trivial-size early return | PASS |
| `test_spectral_bisection_disconnected` | two K_3 components | `λ_1 - λ_0 < 1e-6` → GGGP fallback | PASS |
| `test_spectral_bisection_lanczos_failure` | n=5 path graph (smoke) | Lanczos-failure fallback (smoke; no fault injection) | PASS |

The Day 7 implementation already wired all four edge-case
fallbacks; Day 8 lands the unit tests pinning each path.  The
disconnected-graph test is the most-load-bearing of the four — it
exercises the actual production fallback firing in production-
realistic conditions; the n=1 / n=2 / Lanczos-failure tests pin
the structural contract.

## Headline-gate accounting after Day 8

| stage | Pres_Poisson ND/AMD |
|---|---|
| Sprint 24 baseline (master) | 0.952× |
| + Day 3 HCC alone | 0.937× (-1.5pp) |
| + Day 5 multi-pass FM intermediate (default passes=1) | 0.937× (no movement) |
| + Day 8 spectral alone | 0.953× (essentially unchanged from 0.952× default) |
| **Sprint 25 stretch target** | **≤ 0.85× (literal)** |
| **Remaining gap** | **~10pp** |

Three of three algorithmic axes (HCC + multi-pass FM intermediate
+ spectral bisection) have shown the same pattern on Pres_Poisson:
small / no movement of nnz_L despite producing structurally
different cuts at the coarsest level.  The downstream FM
uncoarsening polishes the partition to a near-identical local
optimum regardless of the coarsest-level choice.

## Why three independent axes all wash out on Pres_Poisson

**Hypothesis** (consolidated from Day 3 + Day 5 + Day 8 findings):

The 3-pass FM at the finest level (Sprint 23 Day 11) on a regular-
structure 2D Poisson grid converges to a strong local optimum that
dominates the final cut quality.  Pre-finest-level choices (matching
strategy, intermediate FM passes, coarsest bisection method)
influence the trajectory of the multilevel pipeline but not the
endpoint nnz_L:  whatever cut gets handed to the finest level, FM
polishes it to a similar geometric-cut-of-a-2D-grid solution.

Where the choices DO matter:
- **Irregular SPD fixtures** (Kuu, bcsstk14): finest-level FM
  doesn't fully converge in 3 passes, so the coarsest-level cut
  has lasting influence on the final nnz_L.  Spectral wins -23.3 %
  on Kuu; HCC wins -6.9 % on nos4 (Day 3); multi-pass intermediate
  wins -10.2 % on Kuu at passes=3 (Day 5).
- **Wall-time** on regular fixtures: spectral cuts close to the
  FM optimum, so fewer FM iterations are needed at each
  uncoarsening level.  Pres_Poisson ND wall drops 21× under
  spectral (37 369 → 1 741 ms) without changing nnz_L.

**Implication for Sprint 25's headline gate**: closing the
Pres_Poisson 0.85× literal target requires changing the FM at the
FINEST level, not the coarsest-level choices.  This is outside
Sprint 25's PLAN.md scope (item 2 multi-pass intermediate
explicitly carved out the second-finest + third-finest, leaving
the finest-level passes as Sprint 23's 3-pass default).  Sprint 26
inherits this finding.

## Day-9 + Day-10 implications

**Day 9 (cross-corpus combined-effect matrix sweep)**:
- Will measure all combinations of HCC + multi-pass FM intermediate
  + spectral × Sprint 24's two existing env vars.
- Expected outcome on Pres_Poisson: best combined-effect ≤ 0.94×
  (based on Day 3's HCC win of 1.5pp; ~0pp from items 2 + 3 alone;
  small constructive composition possible).
- 0.85× literal target unlikely to be reached — partial close
  documentation is the planned outcome.

**Day 10 (production-default decisions)**:
- `SPARSE_ND_COARSENING` (Day 1): stays `heavy_edge` (Kuu +8.5 %
  regression rules out HCC default).
- `SPARSE_FM_INTERMEDIATE_PASSES` (Day 4): stays 1 (Day 5 sweep
  found no flip target meeting the rule).
- `SPARSE_ND_COARSEST_BISECTION` (Day 6): stays `gggp` per Day 8
  finding (Pres_Poisson essentially unchanged + s3rmt3m3 small
  regression — neither side of the flip rule is met).
- All three Sprint 25 env vars ship behind their respective gates
  as advisory for irregular SPD workloads (Kuu / bcsstk14 see
  measurable wins from each of HCC, multi-pass FM intermediate,
  and spectral).

## Sprint 26 routing (preliminary)

Per Sprint 24 RETROSPECTIVE.md "What didn't go well" #1:
"the 0.85× Pres_Poisson stretch target is the second sprint in a
row to miss its literal goal" — Sprint 25 will be the third.
Sprint 26's "ND fill-quality follow-up" routing (if added) should
explore:

1. **Multi-pass FM at the FINEST level beyond 3 passes**: Sprint
   23 Day 11's exploration showed pass 5 hit the same 0.952× ratio
   as pass 3.  Sprint 26 could try (a) a different acceptance rule
   (e.g. annealing, accept gain-decreasing moves with probability),
   (b) a different bucket-FM tie-break for moves with the same
   gain, (c) thick-restart-style FM with global rollback.

2. **Direct geometric cut on Pres_Poisson** (not via FM): the
   2D Poisson grid has a known optimal cut (the median row or
   column).  A heuristic that detects regular-grid structure and
   substitutes the geometric cut would close the gap on Pres_Poisson
   without affecting other fixtures.  This is workload-specific
   but Pres_Poisson is the headline.

3. **Different separator-vertex selection** (post-edge-cut → vertex-
   separator extraction).  Sprint 22 Day 4's `smaller_weight` lift
   strategy + Sprint 24 Day 6's `balanced_boundary` alternative
   both choose a SIDE then lift its boundary; an alternative would
   choose VERTICES individually based on a "separator-suitability"
   score, similar to how AMD selects pivots.

## Quality verification (Day 8)

- All 4 new edge-case spectral tests PASS (smoke + structural).
- Day 7's eigenvalue-ordering + GGGP-fallback tests still PASS.
- Default-path corpus parity bit-identical to current master under
  unset env vars.
- No fixture regresses past 5pp under spectral alone (PLAN.md Day
  8 completion criterion: PASS).
- `make format && make lint && make test && make wall-check` all
  clean (Day 8 gates).

## References

- `docs/planning/EPIC_2/SPRINT_25/PLAN.md` Day 8
- `docs/planning/EPIC_2/SPRINT_25/spectral_bisection_design.md` —
  Day 6 design + Day 7 algorithm
- `docs/planning/EPIC_2/SPRINT_25/bench_day8_spectral_only.{csv,txt}` —
  the cross-corpus capture this assessment is based on
- `docs/planning/EPIC_2/SPRINT_25/nd_tuning_day3.md` — Day-3 HCC
  escalation (analogous wash-out finding)
- `docs/planning/EPIC_2/SPRINT_25/intermediate_fm_decision.md` —
  Day-5 multi-pass-intermediate decision (analogous wash-out
  finding)
- Sprint 24 `RETROSPECTIVE.md` "What didn't go well" #1 — historical
  pattern of Pres_Poisson 0.5× → 0.7× → 0.85× sprint targets all
  missing the literal goal.
