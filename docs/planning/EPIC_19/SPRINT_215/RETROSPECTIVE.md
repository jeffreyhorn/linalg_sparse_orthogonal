# Sprint 215 Retrospective

**Sprint:** 215 - External Baseline And State-Of-The-Art Evidence Blueprint

**Duration:** 14 planned days on branch `sprint-215`

**Status:** Blueprint closed; selected external pilot blocked before execution

## Source And Definition Of Done

The [project plan](../PROJECT_PLAN.md) sets six items totaling 166 estimated
hours and makes a selected external pilot optional. The
[sprint plan](PLAN.md), [working notes](WORKING_NOTES.md),
[Day 14 audit](artifacts/day14-closeout-review.md) and 14 daily artifacts
are the evidence trail. Estimates are not measured hours.

- [x] Selected exact SuiteSparse/CHOLMOD version, commit, solver operation,
      matrix bytes and one local correctness-only cell; recorded excluded
      candidates and publication unknowns.
- [x] Predeclared identical inputs, conversion/ABI checks, independent
      numerical gates, optional timing protocol and provenance before a run.
- [x] Designed an isolated v1 external-evidence/status contract and
      parseable planning example without altering selected report rows.
- [x] Added opt-in pilot harness and synthetic checks; exercised a
      fail-closed preflight and documented the absent validator blocker.
- [x] Updated public and maintainer claim boundaries; added focused
      broad-claim mutation tests.
- [x] Passed the Day 13 local C, report, comparison, benchmark, docs and
      synthetic checks; completed blueprint, queue and Day 14 closeout.
- [ ] Accepted two-sided external correctness, timing, hosted or publication
      evidence. These were not earned by the planned blocker fallback.

## What Went Well

1. **Selection stayed reproducible and narrow.** Days 2-3 pinned CHOLMOD
   5.3.4/SuiteSparse `v7.12.2`, `symmetric_4.mtx` bytes, RHS, solver paths
   and exclusions before interpreting any output. This prevented the
   source-controlled Cholesky helper or `nos4` benchmark from becoming an
   accidental external baseline.
2. **Correctness and timing were separated.** Day 4 froze independent
   residual/oracle tests and factor statuses. Day 5 made timing a distinct,
   optional paired-sample study. The tiny selected pilot never acquired a
   speed or winner claim by implication.
3. **The runner failed closed.** Day 10 preserved a hash-checked preflight
   bundle but ran neither solver without the v1 schema/validator. Synthetic
   tests exercise changed package identity and malformed or failed outputs.
4. **Existing reporting stayed stable.** The proposed v1 document did not
   change selected manifests, report normalization, workflows, or Sprint
   212's threshold-free canonical row. Day 13 compatibility checks passed.
5. **Claims match evidence.** Public/maintainer docs and a focused mutation
   guard describe the blueprint and blocker without implying external
   solver parity, hosted timing or state-of-the-art performance.

## What Didn't Go Well

1. **The schema design was not implemented.** The v1 schema, semantic
   validator, canonical full-cell-ID vector and accepted record producer
   remain missing. A parseable planning JSON and a compiled helper are not
   substitutes. This blocked the intended pilot execution.
2. **Runtime and publication provenance remain incomplete.** Package
   discovery did not verify actual loaded dependencies, bottle/module build
   options, effective threads or ABI. Fixture attribution and CHOLMOD
   module-license review remain separate publication gates.
3. **No representative external evidence exists.** One 4 x 4 SPD cell would
   at most support a narrow local correctness observation after validation.
   Additional matrices, a second solver family, comparable timing and
   exact-branch Linux/macOS runs were not undertaken.
4. **The provisional and proposed record shapes differ.** The runner emits
   `attempt.json` with `provisional_status`; the planned schema uses
   `evidence_status` and `evidence.json`. A future explicit transformation
   and validation path is necessary; renaming a file would be misleading.

## Final Metrics And Claim

| Metric | Close state |
| --- | --- |
| Project-plan items reconciled | 6/6; item 215.4 closed via named blocked fallback, not executed pilot |
| Planned effort | 166 h; actual hours not tracked |
| Daily closeout artifacts | 14 day-numbered Markdown artifacts, plus supporting Day 3 and Day 7 files |
| Accepted external solver results / timing samples | 0 / 0 |
| External v1 schema / validator | 0 / 0 implemented |
| Retained Day 10 preflight raw files | 16/16 SHA-256 checks passed on Day 13 |
| Local selected comparison freshness | 46 source-controlled-helper rows; not CHOLMOD |
| Local C quality gate | `make format && make lint && make test` passed Day 13 |
| Focused pilot checks | 9 synthetic tests passed Day 13; opt-in helper compiled, not run |
| Hosted external pilot or cross-platform runs | 0 recorded |

The earned outcome is an exact external-baseline evidence blueprint,
bounded opt-in harness, blocked preflight record and explicit residual
handoff. It is **not** an external numerical pass, timing comparison,
cross-matrix or cross-platform parity result, public external-result
approval, or state-of-the-art claim. See the
[blueprint](STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md) for the conditional
rerun and [residual queue](RESIDUAL_QUEUE.md) for owners and gates.

## Daily Evidence

- [Day 1 intake](artifacts/day1-external-baseline-evidence-intake.md)
- [Day 2 candidates](artifacts/day2-candidate-baselines-and-matrices.md)
- [Day 3 selection](artifacts/day3-baseline-selection-decision.md) and
  [inputs](artifacts/day3-methodology-inputs.md)
- [Day 4 correctness](artifacts/day4-correctness-and-tolerance-contract.md)
- [Day 5 methodology](artifacts/day5-external-comparison-methodology.md)
- [Day 6 schema design](artifacts/day6-external-report-schema-design.md)
- [Day 7 validation design](artifacts/day7-schema-validation-design.md) and
  [planning JSON](artifacts/day7-external-evidence-blueprint.example.json)
- [Day 8 feasibility decision](artifacts/day8-pilot-feasibility-decision.md)
- [Day 9 harness](artifacts/day9-selected-pilot-harness.md)
- [Day 10 blocked outcome](artifacts/day10-selected-pilot-outcome.md)
- [Day 11 claim matrix](artifacts/day11-evidence-and-claim-matrix.md)
- [Day 12 documentation plan](artifacts/day12-claim-documentation-and-validation-plan.md)
- [Day 13 integrated validation](artifacts/day13-integrated-validation.md)
- [Day 14 closeout](artifacts/day14-closeout-review.md)
