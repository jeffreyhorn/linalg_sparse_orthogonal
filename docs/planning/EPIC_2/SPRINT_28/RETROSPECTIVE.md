# Sprint 28 Retrospective — Non-Pipeline-Level Pres_Poisson Closure (Sprint 27 Deferrals)

**Sprint budget:** 14 working days (~144 hours estimated, per PLAN.md); ran 14 days as planned
**Branch:** `sprint-28`
**Calendar elapsed:** 2026-05-10 → 2026-05-11 (intensive condensed run; the day budget tracks engineering effort, not wall-clock days)

> **Status:** Day 14 — TODO fill in once Day 14 lands.  TL;DR: Sprint 28's Item-4 non-pipeline pivot (supernodal-etree post-pass via Liu 1990) ships infrastructure but is bit-equivalent to the default on Pres_Poisson nnz_L by symmetric-permutation invariance.  Three new advisory env vars (Items 1, 2, 4); zero production default flips.  The literal 0.85× Pres_Poisson target is formally RETIRED with Sprint 28's empirical evidence after 6 consecutive sprints of misses.

## Goal recap

> Close the Pres_Poisson ND/AMD ≤ 0.85× literal target Sprints 22-27 collectively missed (Sprint 27 default 0.9226×; -7.3pp gap; 5th consecutive sprint to miss) via non-pipeline-level interventions — Sprint 27's 24-setting × 6-fixture matrix delivered the strongest empirical evidence yet that pipeline-level interventions cannot move this fixture, so this sprint pivots to a fundamentally different approach.  Three candidate non-pipeline pivots from Sprint 27 RETROSPECTIVE.md "Items deferred" #1: (a) METIS-style multi-matching coarsening, (b) geometric domain decomposition, (c) supernodal reordering on the elimination tree.  Day 1 picks ONE based on cost / upside / fit study; if all three look infeasible, Day 1 falls back to formal calibration of the target to the empirical floor.

(See `docs/planning/EPIC_2/SPRINT_28/PLAN.md` for the day-by-day breakdown; `headline_summary.md` for the Day-13 verdict; the per-axis decision docs in `docs/planning/EPIC_2/SPRINT_28/`.)

## Definition of Done checklist

| item | status | reference |
|---|---|---|
| 1. Formal gain-noise variant of thick-restart FM (Sprint 27 Day 11 deferred) | ✓ SHIPPED — advisory only | Day 2 commit `f5f345c` (impl + interim sweep); Day 3 commit `709640a` (full sweep + advisory decision; Pres_Poisson +24pp regress) |
| 2. Multi-strategy FM ensemble (Sprint 27 parking-lot) | ✓ SHIPPED — advisory only | Day 3 commit `709640a` (design); Day 4 commit `d74dce0` (impl + interim); Day 5 commit `9b67f55` (close + advisory decision; Pres_Poisson +1.5pp regress) |
| 3. Non-pipeline-level pivot decision study | ✓ COMPLETED | Day 1 commit `b3b750f` (`pivot_decision_day1.md`; picked (c) supernodal-etree reordering over (a) METIS multi-matching + (b) geometric DD) |
| 4. Item-4 chosen pivot implementation: (c) supernodal-etree reordering | ✓ SHIPPED — advisory only | Day 6 commit `c5ba39c` (scaffolding); Day 7 commit `fc5f554` (Liu 1990 core algorithm); Day 8 commit `4ace325` (corpus-safety + edge cases); Day 9 commit `9353593` (24-cell sweep, verdict MISSED); Day 10 commit `d79e02d` (`non_pipeline_decision.md` + close-to-target stub) |
| 5. Cross-corpus re-bench + production-default decisions + test-bound calibration | ✓ COMPLETED | Day 12 commit `52bf9fa` (24-setting matrix + headline_summary started); Day 13 (this commit) finalises headline + production-default decisions + bound calibration |
| 6. Pres_Poisson ND wall further reduction (conditional) | ✗ NO-OP — trigger conditions did not fire | Day 11 commit `05df6db` (`wall_reduction_decision.md`; (c) is post-permutation so doesn't change partition wall; no real-world feedback) |
| 7. Tests + docs + retrospective | ⏳ IN-PROGRESS Day 13 + 14 | Day 13 this commit (algorithm.md draft + retrospective skeleton + headline_summary final); Day 14 final fill-in |

Headline gates from PROJECT_PLAN.md Sprint 28 + PLAN.md "Headline gates":

| gate | result |
|---|---|
| Pres_Poisson ND/AMD ≤ 0.85× literal target | ✗ **MISS** at 0.9226× (-7.3pp from target; 6th consecutive sprint; FORMALLY RETIRED) |
| Pres_Poisson < Sprint 27 default (0.9226×) | ✗ EQUAL at 0.9226× (Item-4 SUPERNODAL_POSTORDER=on is bit-equivalent by symmetric-permutation invariance) |
| Smaller-fixture corpus safety (< 5pp regress) | ✓ PASS (all 6 fixtures × {AMD, ND} × env on/off: nnz_L invariant; supernode-count delta ≤ ±3) |
| Item-1 (`SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal`) flip | ✗ STAY — advisory only (catastrophic Pres_Poisson regress) |
| Item-2 (`SPARSE_FM_FINEST_STRATEGY=ensemble`) flip | ✗ STAY — advisory only (1.5pp regress + 2-3× wall) |
| Item-4 (`SPARSE_ND_SUPERNODAL_POSTORDER=on`) flip | ✗ STAY — advisory only (nnz_L invariant; +6-15 % analyze wall) |
| `make wall-check` PASS | ✓ PASS (Pres_Poisson ND ~3-5s; baseline 47s; 1.5× gate 70.5s; ~14× headroom) |
| `make sanitize` + `make tsan` CLEAN | TODO Day 14 — re-run before PR |

## Final metrics

TODO Day 14 — extend the Sprint 27 ND/AMD nnz(L) ratio table with Sprint 28; copy Pres_Poisson ND wall + largest single-fixture improvement.

### ND/AMD nnz(L) ratios (Sprint 22 → Sprint 28)

TODO Day 14.

### Pres_Poisson ND wall (Sprint 25 vs Sprint 28)

TODO Day 14.

## Performance highlights

TODO Day 14 — Sprint 28 produced ZERO production default flips.  Three advisory paths shipped.  Largest single-fixture improvement: bit-equivalent (no new wins; Sprint 27's Kuu setting 18 remains the corpus-wide largest at -35.3 %).

## What went well

TODO Day 14.

## What surprised us

TODO Day 14 — primary surprise: the bit-equality of Item-4 SUPERNODAL_POSTORDER=on with the default on Pres_Poisson nnz_L was predicted in the Day-1 dossier but the empirical evidence on every cell of every measurement (Days 7, 8, 9, 12) is the cleanest confirmation possible.

## What didn't go well

TODO Day 14.

## Items deferred (route to Sprint 29+)

1. **Supernodal numeric-factor kernels** — the natural follow-up that gives the Sprint-28 Item-4 infrastructure measurable production value.  Day-1 dossier estimated 5-15 % numeric-factor wall reduction on supernodal-heavy fixtures.  Out of Sprint 28 scope.
2. **Pres_Poisson literal 0.85× target** — formally retired with Sprint 28's empirical evidence (6 sprints + non-pipeline pivot).  Sprint 29+ revisit only with fundamentally different machinery (METIS interop, geometric mesh-aware ordering, hybrid AMD-then-ND-on-separators).  None budgeted for Sprint 29.
3. **`bench_reorder` env-var integration** — the existing perm-pre-applied + REORDER_NONE path doesn't fire the SUPERNODAL_POSTORDER dispatch.  Low priority; Sprint 28 used ad-hoc /tmp helpers for Day 7, 9, 12 measurements.
4. TODO Day 14 — any items surfaced during Sprint 28 that weren't closed.

## Lessons (Sprint 28-specific)

TODO Day 14 — Sprint-28-specific lessons.  Likely candidates: "non-pipeline pivot decision-day pattern (Day 1) is a successful framing for high-uncertainty exploratory sprints"; "bit-equality of an intervention by mathematical invariance is the cleanest empirical evidence — preserve it"; "post-permutation post-passes ship infrastructure value even when the immediate metric doesn't move".

## Sprint 29 inputs

TODO Day 14 — concrete handoff items.  Initial seed:
1. Sprint 28 Item-4 infrastructure is in place; any future sprint that wires supernodal numeric-factor kernels can opt-in via `SPARSE_ND_SUPERNODAL_POSTORDER=on`.
2. The literal 0.85× Pres_Poisson target is retired; future sprints may use the gap-from-empirical-floor (Sprint 28's 0.923×) as the new baseline if they pursue a fundamentally different approach.
3. PROJECT_PLAN.md Sprint 29 section needs status update (Sprint 28 complete → wrap-up cycle).

## Day-by-day capsule

TODO Day 14 — one-line capsule per Day 1-14.

## Day-budget vs estimate

TODO Day 14 — actuals vs PLAN.md estimates (144 hrs total).

## DoD verification

TODO Day 14 — re-run `make format && make lint && make test && make sanitize && make tsan && make wall-check` before PR.

## Acknowledgements

TODO Day 14.
