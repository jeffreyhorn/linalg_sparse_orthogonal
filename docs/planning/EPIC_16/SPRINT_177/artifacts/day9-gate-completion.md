# Sprint 177 Day 9: Acceptance Gate Completion

**Sprint:** 177 - Epic 16 Baseline, Evidence Matrix & Closure Gates
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_177/`
**Status:** Complete

## Purpose

Complete the acceptance gate set for the remaining Epic 16 closure targets:
additional external comparison, public header coherence, large review-surface
reduction, and final closeout. Together with the Day 8 gates, every selected
Sprint 178-186 target now has pass/fail criteria, owner files, validation
commands, documentation expectations, claim boundaries, and protected
non-claims.

## Gate 6: Additional Bounded External Comparison Family

**Target sprint:** 183
**Residual:** S177-R07
**Matrix rows:** ESM-007, ESM-013

| Field | Acceptance requirement |
| --- | --- |
| Owner files | `scripts/run_external_comparison.py`, selected reference helper, selected-target manifest, `tests/test_run_external_comparison.py`, `tests/test_normalize_report_index.py`, `Makefile`, maintainer guide, solver-selection docs, README, and any affected corpus/report metadata. |
| Required evidence | One additional comparison family has source-controlled fixtures, metrics, tolerances, expected rows, generated output, normalized report-index rows, selected-target manifest registration, and scoped docs. |
| Validation commands | `make report-index-comparison-freshness`; `python3 tests/test_run_external_comparison.py`; `python3 tests/test_normalize_report_index.py`; Python compile checks for changed scripts; relevant C tests if solver behavior changes; `git diff --check`. |
| Pass definition | The selected family regenerates deterministically, expected row counts match, duplicate or missing manifest rows fail clearly, unsupported optional dependencies defer instead of passing, and docs describe fixture-local comparison only. |
| Fail definition | The new family broadens parity claims, depends on unstable optional data without skip/defer semantics, lacks manifest ownership, leaves stale generated rows, or allows duplicate/missing rows to pass silently. |
| Claim boundary | One named fixture-local external comparison family is maintained and freshness-checked. |
| Protected non-claims | No LAPACK, NumPy, SciPy, SuiteSparse, MKL, Eigen, ecosystem parity, performance superiority, package/ABI support, platform parity, release readiness, or state-of-the-art claim. |
| Documentation updates | README report command notes, maintainer solver evidence table, solver-selection docs, normalized report handoff, and selected-target manifest notes. |
| Handoff artifact | Sprint 183 family-selection, report-integration, freshness-gate, and validation artifacts. |

## Gate 7: Public Header Coherence Batch 3

**Target sprint:** 184
**Residual:** S177-R09
**Matrix rows:** ESM-005, ESM-011

| Field | Acceptance requirement |
| --- | --- |
| Owner files | One selected public header family under `include/`, matching implementation and tests if comments reveal behavior gaps, `docs/api_reference.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, Doxygen checks, and declaration guard artifacts. |
| Required evidence | The selected header family has coherent lifecycle, ownership, error-code, option/result, tolerance, cancellation, workspace, and unsupported-claim wording without unintended declaration drift. |
| Validation commands | `make format && make lint && make test` if any C/header file changes; `make docs-check`; `make api-docs-freshness`; declaration-baseline or checksum checks if declarations are reorganized; `git diff --check`. |
| Pass definition | Public comments, declarations, examples, and docs agree; generated API coverage remains valid; declaration order or prototypes change only with explicit baseline proof; no unsupported package/ABI/platform claims are introduced. |
| Fail definition | Documentation contradicts header declarations, declarations drift without a guard, generated API coverage regresses, comments promise unsupported behavior, or validation omits required C/header quality checks. |
| Claim boundary | One selected public header family is clearer and documentation-coherent while preserving the supported API contract. |
| Protected non-claims | No whole-library API redesign, API freeze, stable ABI, dynamic ABI compatibility, shared-library support, generated HTML hosting, or broad solver correctness expansion. |
| Documentation updates | API reference, tutorial/cookbook examples if relevant, solver-selection guidance, maintainer API docs section, and Sprint 184 maintenance note. |
| Handoff artifact | Sprint 184 baseline, cleanup, declaration guard, and integrated validation artifacts. |

## Gate 8: Large Test/Source Review-Surface Reduction

**Target sprint:** 185
**Residual:** S177-R13
**Matrix rows:** ESM-014, ESM-013

| Field | Acceptance requirement |
| --- | --- |
| Owner files | One selected large source/test/tooling cluster from Day 4, build registration files, source-list metadata, focused tests, and a maintenance note. |
| Required evidence | One review surface is reduced through helper extraction, fixture relocation, proof-owner split, or equivalent no-behavior-change structure with build/test registration preserved. |
| Validation commands | Affected focused tests; `make source-list-check` if library sources change; `make format && make lint && make test` for C/header changes; CMake quality checks if registration changes; script tests if Python/shell tooling changes; `git diff --check`. |
| Pass definition | The selected file or cluster has a smaller or clearer review surface, new helpers have explicit ownership, Make/CMake registration remains synchronized, behavior is unchanged, and future contribution guidance is documented. |
| Fail definition | The refactor changes behavior without intended evidence, creates unregistered files, leaves duplicated ownership unclear, increases review surface, or omits full quality gates for C/header changes. |
| Claim boundary | One named review surface has improved maintainability and ownership without changing product behavior. |
| Protected non-claims | No whole-repository maintainability closure, no broad architecture cleanup, no solver-performance improvement claim, and no source-list drift tolerance. |
| Documentation updates | Sprint 185 maintenance note, relevant test/helper comments only where useful, and maintainer guide notes if ownership changes affect contributor workflow. |
| Handoff artifact | Sprint 185 cluster-selection, extraction-design, maintenance-note, and validation artifacts. |

## Gate 9: Final Validation, Claim Calibration, And Closeout

**Target sprint:** 186
**Residual:** S177-R14 and all selected rows
**Matrix rows:** ESM-001 through ESM-014

| Field | Acceptance requirement |
| --- | --- |
| Owner files | Evidence/status matrix updates, README, INSTALL, maintainer guide, report docs, benchmark docs, API docs, package docs, workflow comments, project-plan status, sprint retrospectives, and Epic 16 retrospective. |
| Required evidence | Completed Sprint 178-185 outcomes are reconciled against the matrix, earned claims are updated, unearned claims remain explicit non-claims, validation evidence is recorded, and residuals are prioritized. |
| Validation commands | Required quality gates selected from the Day 10 quality surface map; package checks; report freshness checks; generated API checks; workflow guards; `git diff --check`. |
| Pass definition | Public docs only claim evidence that landed, every selected target is complete or residualized with a reason, unsupported surfaces remain explicit, and the Epic 16 retrospective records outcomes, validation, residuals, and state-of-the-art assessment. |
| Fail definition | Planned but unearned claims are promoted, validation gaps are hidden, residuals lack owner/next-action notes, or closeout docs contradict package/platform/report/API evidence. |
| Claim boundary | Epic 16 public claims match earned evidence and explicit support tiers. |
| Protected non-claims | Broad state-of-the-art, external ecosystem parity, portable performance, broad package-manager support, shared-library support, dynamic ABI compatibility, runtime-loader behavior, broad Windows parity, and broad generated-report parity remain rejected unless a sprint explicitly earned evidence. |
| Documentation updates | README, INSTALL, maintainer guide, report docs, benchmark docs, API docs, project plan status, sprint closeouts, and `docs/planning/EPIC_16/EPIC_16_RETROSPECTIVE.md`. |
| Handoff artifact | Sprint 186 integrated validation, claim recalibration, retrospective, and next-epic handoff artifacts. |

## Prior Review-Comment Failure Modes To Preserve

These recurring review issues must remain visible in all gates:

- Do not let workflow guards validate the same lane that can remove them; run
  fast workflow guard tests from an earlier, independent job when feasible.
- Scope artifact upload assertions to the exact selected upload block, not to
  broad substrings elsewhere in a workflow file.
- Treat duplicate manifest/report rows as failures before constructing maps
  that would silently overwrite entries.
- Prefer explicit missing-row assertions over `next(...)` failures that hide
  the real missing row.
- Keep Windows CTest count expectations synchronized with CMake test
  registration when tests are promoted.
- Preserve static-first package wording when unrelated docs mention Windows
  CMake-first or package evidence.
- Keep "allocation-failure" terminology consistent with Make targets, labels,
  and public docs.
- Preserve NULL-handle public error contracts before validating other
  arguments in public functions.
- Run `make format && make lint && make test` whenever C or header files
  change, even when the primary task is documentation or governance.

## Gate Coverage Summary

| Sprint | Gate artifact | Status |
| --- | --- | --- |
| 178 | Allocation-failure proof batch 2 | Complete in Day 8 |
| 179 | Generated API HTML status | Complete in Day 8 |
| 180 | Package-manager provider decision | Complete in Day 8 |
| 181 | Selected report target manifest | Complete in Day 8 |
| 182 | Windows report freshness decision | Complete in Day 8 |
| 183 | Additional bounded comparison family | Complete in Day 9 |
| 184 | Public header coherence batch 3 | Complete in Day 9 |
| 185 | Large review-surface reduction | Complete in Day 9 |
| 186 | Final validation and claim calibration | Complete in Day 9 |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 178-186 target has an acceptance gate | Complete | Day 8 covers Sprints 178-182; Day 9 covers Sprints 183-186. |
| Gates define validation and documentation expectations | Complete | Each gate includes owner files, commands, pass/fail definition, docs updates, and handoff artifacts. |
| Previous review-comment failure modes are reflected | Complete | Review-trap list records workflow guard, manifest duplicate, CTest count, package wording, terminology, error-contract, and quality-gate traps. |
