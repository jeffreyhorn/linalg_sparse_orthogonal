# Sprint 139 Day 1: QR Residual Intake

## Purpose

Day 1 establishes the Sprint 139 scope before selecting or implementing a QR
closure lane. It creates the working-notes baseline, artifact directory,
inherited QR evidence inventory, item-to-day owner map, initial closure
criteria, validation expectations, and non-claim register.

This is a documentation-only setup artifact. It does not change QR source,
tests, corpus rows, oracle commands, public solver wording, or support-tier
claims.

## Sprint 139 Scope

Sprint 139 implements seven project-plan items:

| Item | Project-plan name | Day owner |
| --- | --- | --- |
| 1 | QR Residual Reaudit | Days 1-3 |
| 2 | QR Fixture Batch | Days 3-5 |
| 3 | QR Oracle Comparison | Days 3, 6-7 |
| 4 | QR Proof-Owner Split | Days 3, 8-9 |
| 5 | Solver Docs Update | Days 10-11 |
| 6 | Validation | Day 12 |
| 7 | Closeout and Residuals | Days 11, 13-14 |

Day 1 completion means later Sprint 139 days can select, design, and implement
the QR closure without rediscovering Sprint 137/138 evidence contracts or
widening unsupported QR claims.

## Artifact Structure

| Path | Role |
| --- | --- |
| `docs/planning/EPIC_12/SPRINT_139/PLAN.md` | Day-by-day Sprint 139 execution plan. |
| `docs/planning/EPIC_12/SPRINT_139/WORKING_NOTES.md` | Sprint goal, constraints, inventories, day ownership, validation expectations, non-claims, and day notes. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day1-qr-residual-intake.md` | This Day 1 QR residual intake artifact. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day2-qr-residual-reaudit.md` | Planned Day 2 residual ranking and selected-priority decision. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day3-closure-design.md` | Planned Day 3 selected residual closure, fixture, oracle, and proof-owner design. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day4-fixture-batch-design.md` | Planned Day 4 deterministic QR fixture batch design. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day5-fixture-batch-implementation.md` | Planned Day 5 QR fixture batch implementation. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day6-oracle-comparison-design.md` | Planned Day 6 QR oracle comparison design. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day7-oracle-comparison-implementation.md` | Planned Day 7 QR oracle comparison implementation. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day8-proof-owner-design.md` | Planned Day 8 focused QR proof-owner design. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day9-proof-owner-implementation.md` | Planned Day 9 focused QR proof-owner implementation. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day10-solver-documentation-update.md` | Planned Day 10 solver documentation update. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day11-maintainer-guidance-residual-queue.md` | Planned Day 11 maintainer guidance and remaining QR residual queue. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day12-focused-validation.md` | Planned Day 12 focused validation evidence. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day13-claim-closure-handoff.md` | Planned Day 13 closed claim, non-claims, and Sprint 140 handoff. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day14-closeout-validation-summary.md` | Planned Day 14 Sprint 139 closeout and validation summary. |

## Inherited QR Evidence Inventory

| Source | Day 1 reading | Sprint 139 use |
| --- | --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` Sprint 139 | Requires closure of the selected QR residual with fixtures, oracle rows, proof ownership, docs, validation, and handoff. | Fixes the sprint deliverables and 168-hour budget. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day13-documentation-sprint139-handoff.md` | Selects `qr_rank_deficient_6x4_nullspace_v1`, its generator, row IDs, null vector, and tolerance. | Provides the first concrete closure lane. |
| `docs/planning/EPIC_12/SPRINT_138/RETROSPECTIVE.md` | States that Sprint 138 did not close solver-backed QR behavior and leaves it to Sprint 139. | Confirms the residual is intentionally open. |
| `tests/corpus/README.md` | Documents first-lane QR handoff and warns against raw-basis equality or support-tier promotion without reviewed evidence. | Controls closure interpretation and support wording. |
| `tests/corpus/manifests/fixtures.tsv` | Defines the first generated QR fixture: 6x4, 14 nonzeros, rank 3, nullity 1. | Supplies fixture metadata for proof and oracle work. |
| `tests/corpus/manifests/generators.tsv` | Defines deterministic generator hash and regeneration policy. | Supplies reproducibility metadata. |
| `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv` | Defines rank, nullity, and normalized null-vector residual expected rows. | Supplies expected-result row IDs and tolerance semantics. |
| `tests/test_qr.c` | Owns many QR factorization, rank, nullspace, projector, and scalar-boundary checks. | Candidate proof-owner source for the selected nullspace residual. |
| `tests/test_qr_solve.c` | Owns QR solve, rank-deficient residual, minimum-norm, and external-reference checks. | Candidate owner only if the selected closure expands to solve behavior. |
| `tests/test_qr_helpers.h` | Owns reusable QR fixture and residual helpers. | Candidate helper location for the corpus fixture builder. |
| `tests/qr_external_dense_reference.py` | Owns bounded dense-reference QR fixture computations. | Candidate reference pattern, not a broad external-library parity claim. |
| `README.md`, `docs/algorithm.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md` | Current public and maintainer QR wording. | Must be updated only after earned evidence exists. |

## Initial Closure Criteria

| Criterion | Required evidence |
| --- | --- |
| Selected QR residual remains bounded | Day 2 ranking confirms rank-deficient nullspace residual is still the best fully closable target. |
| Fixture facts are available | Corpus fixture and generator rows validate successfully. |
| Expected results are available | Expected rank, nullity, and normalized residual rows are `ready_for_oracle`. |
| Solver-backed QR rank is proven | Focused QR proof reports rank `3` on the selected fixture. |
| Solver-backed QR nullity is proven | Focused QR proof reports nullity `1` or one nullspace direction for the selected fixture. |
| Solver-backed QR residual is proven | Focused proof or oracle row shows normalized residual for the solver-produced nullspace direction within tolerance. |
| Proof owner is discoverable | The focused lane is either clearly extracted from `tests/test_qr.c` or added as a dedicated QR proof owner without weakening existing coverage. |
| Public wording is earned | Solver docs mention only the closed fixture-local behavior and preserve broader non-claims. |

## Initial Validation Expectation Register

| Touched surface | Required validation |
| --- | --- |
| Sprint 139 planning artifacts only | `git diff --check`, trailing-whitespace scan under `docs/planning/EPIC_12/SPRINT_139`, and focused Markdown link/path validation under `docs/planning/EPIC_12`. |
| Corpus rows or schemas | `python3 scripts/validate_corpus_schema.py`, TSV width checks, and focused row-claim review. |
| Oracle or Python reference scripts | `python3 -m py_compile <script>`, focused command execution, and generated-report provenance checks. |
| QR tests/helpers/build integration | Focused QR target plus Make/CMake source-list parity when new files are added. |
| QR `.c` or `.h` implementation/API files | Focused QR tests followed by `make format && make lint && make test`. |
| Public or maintainer docs | Markdown link/path validation and claim-boundary scan against broad QR non-claims. |

## Non-Claim Register

| Non-claim | Reason |
| --- | --- |
| Broad QR correctness | Sprint 139 closes one selected residual, not every QR path. |
| Raw QR basis parity | Equivalent nullspace bases can differ by sign, scaling, and orientation. |
| Broad rank-deficient solve behavior | The first closure lane targets rank/nullity/nullspace residual unless later evidence expands scope. |
| Global least-squares or minimum-norm behavior | Existing tests are bounded; this lane does not automatically close solve-wide behavior. |
| SuiteSparse or external corpus parity | Optional external rows remain disabled/skipped until reviewed evidence exists. |
| LAPACK, NumPy, or SciPy parity | Dense helper use remains fixture-local and does not imply broad external-library equivalence. |
| Partial-SVD correctness | Sprint 140 owns selected partial-SVD residual closure. |
| Platform, package, ABI, performance, or state-of-the-art status | QR residual closure does not prove product/runtime/platform claims. |

## Day 1 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 139 project-plan item has a day-level owner. | Complete | Scope table above and `WORKING_NOTES.md` day-level ownership table. |
| Inherited QR/corpus evidence is visible before fixture or code changes begin. | Complete | Inherited evidence inventory lists Sprint 137/138 handoff artifacts, corpus rows, QR tests, helper surfaces, and docs. |
| Closure criteria distinguish earned QR evidence from remaining non-claims. | Complete | Initial closure criteria and non-claim register define what can and cannot be claimed. |
