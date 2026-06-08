# Sprint 59 Day 1 - scope and final quality/platform baseline

Date: 2026-06-08
Branch: `sprint-59`

## Scope

Start Sprint 59 from the actual Sprint 58 validated close state and the Epic 5
remaining quality/platform follow-through queue, then reduce the next work to a
bounded final productization and closeout package centered on the strongest
live validation, platform, and handoff surfaces.

## Authoritative baseline

Sprint 59 starts from a preserved reviewed validation baseline:

- strongest local reviewed baseline: `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

This means Sprint 59 is not a validation-recovery sprint. It is a final
quality/platform and closeout sprint.

## What Sprint 58 already proved

The following is already real before Sprint 59 begins:

- public-surface simplification already landed on the highest-signal README,
  tutorial, header, example, and benchmark surfaces
- no public API redesign was needed
- no workflow-boundary drift remained at Sprint 58 close
- the direct repeated-run, factor-many, iterative-handle, and eigensolver-
  handle stories remained validated
- Sprint 58 closed from:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`

Interpretation:

- Sprint 59 does not need to re-decide the public product surface
- Sprint 59 needs to close the remaining staged quality/platform queue and
  write the final Epic 5 handoff from that already-validated state

## What the Epic 5 review and todo list already fixed as the next queue

The Epic 5 review and todo notes already point to the same bounded remaining
problem:

- re-evaluate serialized dead-code execution
- reassess macOS dead-code staging
- reassess Windows reviewed-wrapper parity and dead-code exclusion
- reassess whether the current coverage contract should remain unchanged

The inherited review guidance remains concrete:

- the quality contract is already honest and strong
- the remaining gaps should now be treated as a bounded productization pass
- each staged or excluded surface should leave Sprint 59 with a fresh
  disposition:
  - fixed
  - still intentionally staged
  - or explicitly deferred again with current rationale

Interpretation:

- Sprint 59 should treat the quality/platform queue as still live
- the strongest remaining maintainability pressure is now platform-story
  truthfulness and residual disposition rather than core implementation work

## Actual Sprint 59 queue

The Sprint 59 project-plan items reduce to six bounded work classes:

1. quality/platform residual audit
2. bounded quality follow-through batch
3. final cross-surface compatibility sweep
4. full validation sweep
5. Epic 5 summary and handoff
6. project-plan / residual-journal finalization

The strongest architectural narrowing is:

- keep the work centered on measured follow-through and closeout truthfulness
- prefer explicit residual disposition over broad platform ambition
- preserve the Sprint 50-58 public and validation fence exactly
- do not broaden into feature work, API redesign, or CI-platform reinvention

## Main hotspots

Highest-value touched surfaces at sprint start:

- quality-contract and platform-story surfaces:
  - `README.md` = `973`
  - `Makefile` = `878`
  - `docs/maintainer_guide.md` = `294`
  - `.github/workflows/ci.yml` = `221`
- dead-code workflow and classification surfaces:
  - `scripts/deadcode_workflow.sh` = `219`
  - `scripts/deadcode_report.py` = `550`
- project-level planning and closeout surfaces:
  - `docs/planning/EPIC_5/PROJECT_PLAN.md` = `340`
  - `docs/planning/EPIC_5/SPRINT_58/RETROSPECTIVE.md` = `226`
  - `docs/planning/EPIC_5/SPRINT_58/artifacts/day13-full-validation-sweep.md` = `107`
  - `docs/planning/EPIC_5/SPRINT_58/artifacts/day14-closeout-and-handoff.md` = `125`

Interpretation:

- the strongest implementation-adjacent follow-through pressure is in
  `Makefile`, the dead-code workflow scripts, and the platform contract wording
- the strongest closeout-writing pressure is now in Epic-level summary and
  residual disposition surfaces rather than sprint-local implementation notes

## Preserved fence

Sprint 59 still inherits the controlling compatibility and non-goal boundary:

- no public API redesign
- no reopening of the validated lifecycle or handle contracts
- no solver-family expansion disguised as quality/platform work
- no broad platform redesign disguised as final polish
- no fake closure claims on staged surfaces without fresh measured evidence

## Conclusion

Day 1 fixes Sprint 59's real starting point:

- preserved reviewed baseline
- inherited validated public and workflow fence
- bounded remaining quality/platform and closeout queue
- named productization and handoff hotspots
- explicit non-goal fence against feature, API, or platform overreach

That is enough to move to the Day 2 validation and truthfulness-anchor recheck
without reopening Sprint 50-58 product-surface decisions.
