# Sprint 178 Day 13: Claim Recalibration And Residuals

## Scope

Day 13 reconciles the completed Sprint 178 evidence against Sprint 177 Gate 1
and the public/maintainer documentation updated on Day 11.

## Gate 1 Reconciliation

| Gate 1 field | Sprint 178 result | Status |
| --- | --- | --- |
| Target | Allocation-Failure Proof Batch 2. | Met |
| Required evidence | One additional subsystem has deterministic injected allocation-failure coverage, cleanup assertions, no stale public state, retry-after-reset proof, focused validation, and scoped docs. | Met |
| Validation | Focused Make gate, registration guard, CMake/CTest selectors, docs hygiene, and full C quality gate passed on Day 12. | Met |
| Fail definition | No nondeterministic failures, unasserted cleanup, stale public state publication, unproven retry behavior, or broad documentation wording remain in the selected proof. | Met |
| Claim boundary | One additional named subsystem has deterministic allocation-failure cleanup evidence. | Met |
| Protected non-claims | Broad allocation-failure coverage remains explicitly out of scope. | Met |

## Earned Claim

Sprint 178 may claim:

`sparse_matmul()` workspace allocation has deterministic allocation-failure
cleanup evidence for the selected accumulator, nonzero-flag, and touched-column
workspace allocations. The evidence includes stale-output suppression and
successful retry after resetting the private allocation-failure hook.

## Evidence Chain

| Evidence | Owner |
| --- | --- |
| Selected subsystem and failure sites | `artifacts/day3-subsystem-selection.md` |
| Cleanup and no-publication invariants | `artifacts/day4-cleanup-invariant.md` |
| Harness design and private-hook semantics | `artifacts/day5-harness-design.md` |
| Harness implementation | `artifacts/day6-harness-implementation.md` |
| First stale-output regression | `artifacts/day7-first-regression.md` |
| Remaining workspace failure coverage | `artifacts/day8-coverage-expansion.md` |
| Cleanup and error-contract review | `artifacts/day9-cleanup-error-contracts.md` |
| Focused Make/CTest registration | `artifacts/day10-focused-gate.md` |
| Public and maintainer wording | `artifacts/day11-scoped-claim-documentation.md` |
| Integrated validation | `artifacts/day12-integrated-validation.md` |

## Public Claim Check

The README and maintainer guide now name:

- `make iterative-allocation-failure-gate` for the Sprint 176 iterative
  repeated-run handle proof;
- `make matmul-allocation-failure-gate` for the Sprint 178 `sparse_matmul()`
  workspace proof.

The positive Sprint 178 claim names only `sparse_matmul()` workspace
allocation. It does not claim matrix-wide or library-wide allocation-failure
safety.

## Retained Non-Claims

Sprint 178 does not prove allocation-failure cleanup for:

- matrix shell construction;
- insertion and product-flush allocation;
- matrix copy, transpose, CSR/CSC conversion, and build helpers;
- direct solvers, QR, LDLT, Cholesky, SVD, eigensolvers, graph routines, or
  reorder routines;
- package/install flows;
- generated-report tooling;
- public allocation-failure test API.

## Residual Queue

| Residual | Status | Next action |
| --- | --- | --- |
| Matrix construction and conversion allocation failures | Retained | Select a future sprint only if this becomes the highest-value allocation gap. |
| Solver-family allocation failures beyond iterative handles | Retained | Treat each solver family as a separate selected-subsystem proof. |
| Generated tooling allocation failures | Retained | Do not mix with numerical-kernel allocation proofs. |
| Public allocation-failure API | Rejected | Keep the hook private/internal. |

## Retrospective Inputs

- Sprint 178 successfully widened allocation-failure evidence from one selected
  iterative-handle family to one selected matrix operation.
- The fail-at-count hook was sufficient; no public test-injection API was
  needed.
- The focused Make gate plus registration guard kept local validation easy to
  run and hard to drift.
- The most important closeout risk remains claim creep. Day 14 should keep the
  final records scoped to `sparse_matmul()` workspace allocation only.

## Validation

- `python3 tests/test_matmul_allocation_failure_gate_registration.py`
- `rg -n "Selected allocation-failure proofs|matmul-allocation-failure-gate|tests/test_matmul\\.c.*owns|matrix multiply allocation-failure proof" README.md docs/maintainer_guide.md`
- `rg -n "broad allocation-failure|not broad allocation-failure|does not establish broad allocation-failure" README.md docs/maintainer_guide.md docs/planning/EPIC_16/SPRINT_178/artifacts`
- `git diff --check`
