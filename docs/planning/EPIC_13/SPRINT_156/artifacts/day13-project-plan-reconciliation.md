# Sprint 156 Day 13: Project Plan Reconciliation

## Purpose

Reconcile the Epic 13 project plan against completed Sprint 147-156 artifacts,
validation results, deferred work, and the Day 12 retrospective draft. This
artifact prepares the Day 14 final retrospective and handoff.

## Inputs Reviewed

- `docs/planning/EPIC_13/PROJECT_PLAN.md`
- Sprint 147-155 `RETROSPECTIVE.md` files
- Sprint 147-155 Day 13/Day 14 integrated validation and closeout artifacts
- Sprint 156 Day 1-12 artifacts
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day11-residual-queue-publication.md`
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day12-retrospective-draft.md`

## Status Labels

| Label | Meaning |
| --- | --- |
| Complete | Planned item was delivered with artifacts and validation appropriate to touched surfaces. |
| Complete with narrowed claim | Work completed by intentionally narrowing the public claim or preserving a non-claim. |
| Deferred | Work was intentionally left for a later epic and appears in the Day 11 residual queue. |
| Superseded | Planned framing was replaced by a clearer product decision or evidence boundary. |
| Pending PR evidence | Local/source-controlled work is complete, but current branch hosted PR CI is not yet available. |

## Sprint-Level Reconciliation

| Sprint | Planned goal | Delivered status | Evidence artifacts | Reconciliation |
| --- | --- | --- | --- | --- |
| 147 | Freeze baseline, select closure targets, and define gates. | Complete | Sprint 147 retrospective, Day 5 selected-gap register, Day 6 claim-target register, Day 12 quality map, Day 14 handoff. | Planning sprint delivered the gate structure used by later implementation sprints. |
| 148 | Promote or replace Windows-staged test surfaces. | Complete | Sprint 148 retrospective, Day 13 integrated validation, Day 14 closeout. | `test_threads`, `test_sprint4_integration`, and `test_fuzz` were promoted to reviewed Windows CMake scope; broader Windows parity stayed deferred. |
| 149 | Decide reviewed Windows install-validation support tier. | Complete with narrowed claim | Sprint 149 retrospective, Day 13 integrated evidence review, Day 14 closeout. | Promoted Windows CMake install/downstream validation only; Windows Makefile and Windows `pkg-config` parity remain residuals. |
| 150 | Expand maintained QR corpus family. | Complete with narrowed claim | Sprint 150 retrospective, Day 13 integrated validation, Day 14 closeout. | Closed selected QR rank-deficient and minimum-norm fixture families; reorder/COLAMD and broad QR behavior remain deferred. |
| 151 | Expand maintained partial-SVD corpus family. | Complete with narrowed claim | Sprint 151 retrospective, Day 13 integrated validation, Day 14 closeout. | Closed selected clustered/repeated, projector, sparse-output, fail-closed, and recovery families; broad vector, convergence, and optimality claims remain deferred. |
| 152 | Promote selected generated report freshness. | Complete with narrowed claim | Sprint 152 retrospective, Day 13 quality gate/residual review, Day 14 closeout. | Selected QR plus partial-SVD oracle freshness is maintained locally; benchmark, sentinel, coverage, dead-code, and hosted generated publication remain residuals. |
| 153 | Make shared-library ABI product decision. | Complete with narrowed claim | Sprint 153 retrospective, Day 13 quality gate/residual review, Day 14 handoff. | Product decision preserved static-first support and strengthened shared-library rejection instead of implementing dynamic support. |
| 154 | Build first direct comparison harness and narrow study. | Complete with narrowed claim | Sprint 154 retrospective, Day 13 validation/study publication, first narrow study, Day 14 handoff. | One local QR minimum-norm comparison study was published; external package and ecosystem parity remain residuals. |
| 155 | Align tutorial, headers, and API reference. | Complete with documented residuals | Sprint 155 retrospective, Day 13 integrated validation, Day 14 handoff. | Tutorial/API path and selected header cleanup landed; generated API HTML refresh and remaining headers remain residuals. |
| 156 | Validate final state, recalibrate claims, publish residuals, draft retrospective, and reconcile plan. | In progress, Days 1-13 complete | Sprint 156 Day 1-13 artifacts. | Day 14 remains to finalize the retrospective, final claim check, closeout artifact, commit/push/PR workflow. |

## Project-Plan Item Reconciliation

| Sprint | Plan item range | Status | Notes |
| --- | --- | --- | --- |
| 147 | Items 1-7 | Complete | Baseline inventory, residual selection, claim register, evidence templates, quality map, public claim freeze, and handoff were all published. |
| 148 | Items 1-7 | Complete | Staged tests were audited, designed, ported, promoted in CMake/CI policy, documented, and validated with required full C gate. |
| 149 | Items 1-7 | Complete with narrowed claim | Windows CMake package lane was promoted; Windows Makefile/`pkg-config`, package-manager, shared-library, loader, ABI, and broad Windows parity were intentionally not promoted. |
| 150 | Items 1-7 | Complete with narrowed claim | Selected QR families closed; reorder/COLAMD and broad QR families were intentionally deferred. |
| 151 | Items 1-7 | Complete with narrowed claim | Selected partial-SVD families closed; broad repeated-spectrum/raw-vector/convergence/sparse-output claims were intentionally deferred. |
| 152 | Items 1-7 | Complete with narrowed claim | Selected oracle freshness landed; generated artifact upload/retention and other generated families stayed deferred. |
| 153 | Items 1-7 | Complete with narrowed claim | Product decision chose stronger static-first deferral rather than shared-library implementation; this matches the plan's explicit either/or path. |
| 154 | Items 1-7 | Complete with narrowed claim | Harness and first study landed; optional NumPy/SciPy and ecosystem baselines remained defer rows/non-claims. |
| 155 | Items 1-7 | Complete with residuals | Tutorial, selected headers, API index, declaration preservation, and validation landed; generated Doxygen refresh is residual. |
| 156 | Items 1-5 | Complete | Evidence inventory, local/package/report/corpus/comparison validation, platform reconciliation, claim audit, and residual queue are complete. |
| 156 | Item 6 | Draft complete | Day 12 created the retrospective draft; Day 14 must publish the final Epic 13 retrospective. |
| 156 | Item 7 | Complete for Day 13 | This artifact reconciles the project plan; Day 14 must incorporate it into final closeout. |

## Completed, Deferred, And Superseded Work

### Completed

- Epic 13 evidence gates and claim target structure.
- Windows CMake promotion for all previously staged tests.
- Windows CMake install/downstream reviewed package confidence path.
- Maintained QR corpus-family expansion for selected fixtures.
- Maintained partial-SVD corpus-family expansion for selected fixtures.
- Selected local generated oracle freshness for combined QR plus partial-SVD
  rows.
- Static-first package decision with strengthened shared-library rejection.
- First narrow QR minimum-norm comparison harness and study.
- Tutorial/API adoption path and selected declaration-preserving public-header
  cleanup.
- Sprint 156 final evidence inventory, validation, platform reconciliation,
  claim audit, residual publication, retrospective draft, and plan
  reconciliation.

### Deferred

The deferred items are published in Day 11 as `E13-R01` through `E13-R18`.
The most important deferred groups are:

- Windows Makefile and Windows `pkg-config` parity.
- Package-manager distribution.
- Shared-library product support and dynamic ABI compatibility.
- Hosted promotion for local-only generated oracle/comparison rows.
- Broader QR and partial-SVD corpus/comparison breadth.
- Optional NumPy/SciPy and broader ecosystem baselines.
- Portable performance methodology.
- Generated API HTML refresh and remaining public-header cleanup.
- Broad state-of-the-art positioning.
- Typed runtime/backend control promotion and additional sentinel rows.

### Superseded Or Narrowed

| Planned framing | Final disposition |
| --- | --- |
| Windows install-validation parity | Narrowed to reviewed Windows CMake install/downstream validation; Unix Makefile/`pkg-config` parity was not generalized to Windows. |
| QR family selection including reorder/COLAMD paths | Narrowed to rank-deficient rectangular and minimum-norm families; reorder/COLAMD deferred because semantics are mixed with ordering, fill, optional SuiteSparse, and performance-adjacent concerns. |
| Partial-SVD family selection including broad repeated/vector semantics | Narrowed to selected subspace-safe and fail-closed families; broad raw-vector and convergence claims remain deferred. |
| Generated report freshness publication | Narrowed to selected oracle freshness; other generated families remain advisory or future owned. |
| Shared-library ABI decision | Resolved as stronger static-first deferral rather than shared implementation. |
| External comparison harness | Narrowed to one QR fixture and one source-controlled dense helper baseline; optional package and ecosystem baselines remain deferred. |
| API reference plan | Delivered as source-header-first API index and generated freshness policy; generated HTML refresh deferred. |
| State-of-practice / state-of-the-art assessment | Narrowed to explicit non-claim; evidence supports maturity improvements only. |

## Estimate And Validation Variance

| Area | Plan expectation | Observed variance | Reconciliation |
| --- | --- | --- | --- |
| Windows CTest counts | Counts would change only after evidence. | Counts drifted as tests were promoted and later corpus tests added; CI count guards surfaced required updates. | Expected-count brittleness is useful but must be updated deliberately with each CMake test addition. |
| Fixed installed header counts | Package lanes check installed header count. | Header count remained `19`; future public headers will break fixed-count checks until updated. | Keep fixed counts as intentional contract checks, but document updates with public-header additions. |
| Hosted CI evidence | Needed for platform claims. | Several sprint closeouts had hosted proof pending until PR workflows; Sprint 156 currently has master baseline only. | Treat current branch CI as PR-time pending until PR runs complete. |
| Generated report freshness | Plan allowed selected generated families. | Only selected oracle and comparison freshness became claim-bearing; other generated families stayed advisory/deferred. | Correct narrowing; prevents generated local rows from being overread. |
| Shared-library ABI | Plan allowed implementation or stronger deferral. | Stronger deferral was selected. | Complete under the plan because exact blockers and guard tests now exist. |
| External comparison | Plan asked for first narrow study. | The study used a source-controlled helper rather than an external package. | Complete for first harness/study; external package parity remains residual. |
| API docs | Plan included API index or generated publication plan. | API index and policy landed; generated HTML refresh deferred. | Complete with residual; avoids large generated-output churn during Sprint 155. |

## Retrospective Draft Reconciliation

The Day 12 retrospective draft matches this project-plan status with these
required Day 14 edits:

- Keep Sprint 156 status as final only after Day 14 closeout artifact exists.
- Preserve "pending PR evidence" wording for current branch hosted CI until PR
  workflows run.
- Keep shared-library support under static-first deferral, not partial
  implementation.
- Keep comparison wording scoped to one QR fixture and one source-controlled
  dense helper.
- Keep generated API HTML refresh in residuals.
- Reference the Day 11 residual queue as the final residual source of truth.

No earned claim in the Day 12 draft needs removal based on the project-plan
reconciliation. The draft should not be promoted to final until Day 14 runs
the final docs-only hygiene and claim/non-claim check.

## Next-Epic Handoff Updates

The next epic should prefer complete closure candidates from Day 11:

| Priority | Candidate | Required first decision |
| --- | --- | --- |
| 1 | Generated API HTML refresh | Decide Doxygen input scope, generated `sparse_version.h` handling, and warning/page-coverage gate. |
| 2 | Hosted selected oracle/comparison promotion | Decide whether selected local-only report gates become reviewed hosted evidence and how artifacts are retained. |
| 3 | One bounded QR comparison expansion | Select exactly one QR fixture family and metrics before implementation. |
| 4 | One bounded partial-SVD comparison publication | Select subspace-safe metrics and baseline helper policy before implementation. |
| 5 | Windows Makefile or Windows `pkg-config` parity decision | Decide product scope before adding CI/toolchain complexity. |
| 6 | Next public-header cleanup batch | Select headers and declaration-preservation gate before editing. |

Long-horizon items should stay deferred unless product direction changes:
package-manager distribution, shared-library product support, dynamic ABI
compatibility, broad ecosystem parity, portable performance superiority,
broad state-of-the-art positioning, and typed runtime/backend API promotion.

## Completion Criteria Check

- Project-plan status matches real artifacts.
- Completed, deferred, narrowed, and superseded work is separated.
- Deferred work appears in the Day 11 residual queue.
- Estimate and validation surprises are recorded.
- The Day 12 retrospective draft is reconciled and ready for Day 14 final
  publication.
- Next-epic handoff is grounded in complete-gap closure rather than broad
  partial progress.
