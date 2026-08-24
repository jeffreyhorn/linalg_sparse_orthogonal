# Sprint 178 Day 14: Sprint Closeout

## Scope

Day 14 closes Sprint 178 and prepares retrospective inputs. It does not widen
the allocation-failure claim beyond the selected `sparse_matmul()` workspace
proof validated on Day 12 and reconciled on Day 13.

## Item Closeout

| Item | Result | Evidence |
| --- | --- | --- |
| 178.1 Subsystem Selection Detail | Complete | `artifacts/day3-subsystem-selection.md` |
| 178.2 Cleanup Invariant Record | Complete | `artifacts/day4-cleanup-invariant.md` |
| 178.3 Harness Extension | Complete | `artifacts/day5-harness-design.md`, `artifacts/day6-harness-implementation.md` |
| 178.4 Regression Tests | Complete | `artifacts/day7-first-regression.md`, `artifacts/day8-coverage-expansion.md`, `artifacts/day9-cleanup-error-contracts.md` |
| 178.5 Focused Gate | Complete | `artifacts/day10-focused-gate.md` |
| 178.6 Claim Documentation and Validation | Complete | `artifacts/day11-scoped-claim-documentation.md`, `artifacts/day12-integrated-validation.md`, `artifacts/day13-claim-recalibration.md` |

## Artifact Inventory

- `PLAN.md`
- `WORKING_NOTES.md`
- `artifacts/day1-sprint-intake.md`
- `artifacts/day2-allocation-surface-inventory.md`
- `artifacts/day3-subsystem-selection.md`
- `artifacts/day4-cleanup-invariant.md`
- `artifacts/day5-harness-design.md`
- `artifacts/day6-harness-implementation.md`
- `artifacts/day7-first-regression.md`
- `artifacts/day8-coverage-expansion.md`
- `artifacts/day9-cleanup-error-contracts.md`
- `artifacts/day10-focused-gate.md`
- `artifacts/day11-scoped-claim-documentation.md`
- `artifacts/day12-integrated-validation.md`
- `artifacts/day13-claim-recalibration.md`
- `artifacts/day14-closeout.md`

## Final Earned Claim

Sprint 178 may claim only:

`sparse_matmul()` workspace allocation has deterministic allocation-failure
cleanup evidence for selected accumulator, nonzero-flag, and touched-column
workspace allocations. The proof covers stale-output suppression and
retry-after-reset behavior under the private allocation-failure hook.

## Protected Non-Claims

Sprint 178 does not prove broad allocation-failure cleanup for:

- matrix shell construction;
- insertion and product-flush allocation;
- matrix copy, transpose, CSR/CSC conversion, or build helpers;
- direct solvers, QR, LDLT, Cholesky, SVD, eigensolvers, graph routines, or
  reorder routines;
- package/install flows;
- generated-report tooling;
- public allocation-failure API.

## Validation Summary

Day 12 passed the required integrated validation:

- focused Make gate;
- standalone registration guard;
- CMake/CTest registration and selectors;
- docs terminology and whitespace hygiene;
- `make format && make lint && make test`.

Day 14 rechecked:

- artifact inventory placement under `docs/planning/EPIC_16/SPRINT_178/`;
- registration guard;
- broad allocation-failure non-claim wording;
- `git diff --check`.

## Retrospective Inputs

- The existing private fail-at-count allocation hook was sufficient for the
  selected matrix multiply workspace proof.
- The strongest outcome is the combination of tests, focused Make gate, CTest
  labels, registration guard, README wording, maintainer guidance, and
  validation record.
- The main risk remains claim creep. The final wording should stay scoped to
  selected `sparse_matmul()` workspace allocations.
- Future allocation-failure work should continue selecting one subsystem at a
  time and should not combine numerical-kernel allocation proof with generated
  tooling or package/install concerns.

## Sprint 179 Handoff

Sprint 179 should start from the Epic 16 project-plan section
`Sprint 179: Generated API HTML Publication Decision`.

Expected first actions:

- audit Doxygen inputs, ignored outputs, warnings, and page coverage;
- decide hosted publication, retained CI artifact, committed output, or
  enforced local-only status;
- add or tighten freshness/staging guards for the selected path;
- update navigation only after the product decision is explicit.

Sprint 178 leaves no generated API HTML implementation changes and no
allocation-failure blocker for Sprint 179.
