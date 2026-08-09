# Sprint 147 Day 4 Epic 12 Residual Intake

## Purpose

Day 4 reconciles the Sprint 146 published residual queue into Epic 13 intake
work. This is not the final Day 5 gap-selection decision. It assigns every
Epic 12 residual an initial Epic 13 disposition, owner surface, dependency
shape, duplicate boundary, and promotion gate reminder.

Residuals remain non-claims until their promotion gates pass.

## Intake Sources

| Source | Intake Value |
| --- | --- |
| `docs/planning/EPIC_12/SPRINT_146/artifacts/day11-published-residual-queue.md` | Canonical R1-R14 residual queue, owners, priorities, non-claims, and promotion gates. |
| `docs/planning/EPIC_12/EPIC_12_RETROSPECTIVE.md` | Epic 12 earned claims, non-claims, future-epic candidate grouping, and state-of-the-art assessment. |
| `docs/planning/EPIC_13/reviews/review-codex-2026-08-09.md` | Fresh code/documentation/platform review and current gap assessment. |
| `docs/planning/EPIC_13/reviews/todo-codex-2026-08-09.md` | Stepwise Epic 13 closure recommendation and acceptance criteria. |
| `docs/planning/EPIC_13/PROJECT_PLAN.md` | Sprint 147-156 allocation that turns selected residual groups into implementation sprints. |

## Intake Classification

| Classification | Meaning |
| --- | --- |
| Candidate | Active Epic 13 input that may be selected on Day 5 and has a plausible sprint owner. |
| Blocked | Valuable but cannot be promoted before a prerequisite decision or evidence gate exists. |
| Duplicate | Covered by a broader candidate; retain as a subcase, not independent scope. |
| Deferred | Explicitly left outside the likely Epic 13 closure path unless Day 5 selects it. |
| Rejected | Should not be pursued because it would widen unsupported claims or conflict with the current product contract. |

## Residual Grouping

| Group | Residuals | Epic 13 Sprint Surface | Notes |
| --- | --- | --- | --- |
| Platform and hosted CI | R1, R2, R3 | Sprints 148, 149, 156 | R2 and R3 are direct closure candidates; R1 is a recurring evidence-publication input for final reconciliation. |
| Package and ABI | R4, R14 | Sprint 153 | R4 is the product decision center. R14 depends on release/package-manager mechanics and should not lead ABI work. |
| Numerical corpus | R5, R6 | Sprints 150, 151 | These are high-value corpus expansion candidates and depend on Day 8 corpus evidence gates. |
| Generated reports | R7 | Sprint 152 | Depends on selected claim-bearing families from QR, partial-SVD, platform, package, or comparison work. |
| Adoption and API documentation | R8, R9 | Sprint 155 | Should follow earned claim changes rather than invent new support language. |
| Runtime/backend governance | R10, R11 | Candidate only if selected; no dedicated Epic 13 implementation sprint currently allocated | Strong overlap with report/performance work, but lower priority than Windows/corpus/package/comparison closure. |
| Competitive positioning | R12, R13 | Sprints 154, 156 | R12 is prerequisite evidence. R13 remains rejected as a broad claim unless R12 earns a narrow claim. |

## R1-R14 Intake Disposition

| ID | Residual | Initial Epic 13 Disposition | Owner Surface | Prerequisite Evidence | Promotion Gate Reminder |
| --- | --- | --- | --- | --- | --- |
| R1 | Branch-specific hosted CI reconciliation for Sprint 146 | Duplicate | CI/report owner | Current branch/PR workflow run IDs and conclusions when available. | Fold into Sprint 156 final hosted evidence reconciliation; do not count as a separate feature gap. |
| R2 | Windows staged test portability closure | Candidate | Platform/test owner | Day 7 Windows gate; current staged blockers for `test_threads`, `test_sprint4_integration`, and `test_fuzz`; Windows CMake reviewed lane. | Hosted Windows CMake lane intentionally registers and executes promoted staged coverage with updated docs/report rows. |
| R3 | Windows reviewed install-validation parity decision | Candidate | Platform/package owner | Sprint 148 platform outcome; current supplemental Windows CMake install/downstream proof; Linux/macOS package proof. | Product decision promotes or rejects reviewed Windows install-validation parity with workflow, report, docs, and hosted proof aligned. |
| R4 | Shared-library ABI productization | Candidate | Package/ABI owner | Sprint 149 package decision; public symbol/header inventory; static-first proof remains green. | Shared build/install/export support or stronger static-first deferral is backed by ABI policy, loader/package tests, metadata, and docs. |
| R5 | Broad QR residual expansion | Candidate | QR/corpus owner | Day 8 corpus evidence gate; Sprint 139 fixture-local closure; corpus schema and oracle/report commands. | Multiple QR fixture families have metadata, expected rows, focused proof owners, validation commands, generated-local classification, and bounded wording. |
| R6 | Broad partial-SVD residual expansion | Candidate | SVD/corpus owner | Day 8 corpus evidence gate; Sprint 140 fixture-local closure; subspace-safe comparison contract. | Multiple partial-SVD fixture families have metadata, expected rows, focused proof owners, validation commands, and bounded wording. |
| R7 | Generated benchmark, sentinel, coverage, dead-code, and guardrail refresh package | Candidate | Report/benchmark owner | Day 9 freshness gate; selected generated families that support concrete claims. | `normalize_report_index.py --require-generated <family> --check-freshness` passes for selected families without turning local rows into hosted proof. |
| R8 | Tutorial alignment with first-use ladder | Candidate | Documentation owner | Earned claim updates from Sprints 148-154; Sprint 145 first-use ladder. | Tutorial matches build, first solve, data input, solver choice, diagnostics, install/downstream, and public claim scan. |
| R9 | Broader public-header cleanup | Candidate | API/documentation owner | Header cleanup selection; declaration-preservation scan; no ABI/signature drift. | Remaining selected headers receive cleanup and `make format && make lint && make test` passes if headers change. |
| R10 | Runtime/backend typed-control promotion review | Deferred | Runtime/backend owner | Explicit selected control, API design, ABI review, tests, and docs. | Keep maintainer controls non-API unless Day 5 selects a complete typed-control promotion path. |
| R11 | Additional runtime/backend sentinel rows | Deferred | Runtime/backend and benchmark owners | Selected sentinel metric, runtime budget, row semantics, variance policy, and report freshness decision. | Keep as future runtime/report work unless it directly supports Sprint 152 generated freshness closure. |
| R12 | External-library parity study | Candidate | Numerical lead | Selected comparison target, pinned dependencies, fixture set, metrics, tolerances, platform/compiler context, and skip/defer semantics. | Narrow comparative study names libraries, versions, fixtures, metrics, caveats, and non-claims before any parity wording is promoted. |
| R13 | State-of-the-art competitive decision | Blocked | Epic owner | R12 direct comparative evidence plus final Epic 13 evidence inventory. | Broad state-of-the-art remains rejected; only a narrow evidence-backed claim can be considered in Sprint 156. |
| R14 | Package-manager distribution | Deferred | Package/distribution owner | R4 ABI/package product decision, release/versioning policy, recipe ownership, install/update/uninstall proof. | Do not select unless shared/static package product mechanics are stable enough for package-manager support. |

## Dependency Map

| Dependency | Affects | Reason |
| --- | --- | --- |
| Windows staged portability before install parity | R2 -> R3 | Reviewed Windows package parity is clearer after the CMake test surface and staged exclusions are settled. |
| Package parity before shared-library decision | R3 -> R4 | Shared-library productization must preserve or explicitly revise static-first install/export guarantees. |
| Shared-library product decision before package-manager distribution | R4 -> R14 | Package-manager recipes need stable artifact, ABI, versioning, loader, and update/uninstall semantics. |
| Corpus evidence gates before broad solver claims | Day 8 -> R5, R6 | QR and partial-SVD expansion must define fixture metadata, expected rows, comparison semantics, proof owners, and report boundaries first. |
| QR/partial-SVD expansion before generated freshness selection | R5, R6 -> R7 | Freshness gates should require only generated families that support actual claim-bearing work. |
| Corpus expansion before external comparison | R5 or R6 -> R12 | Comparison fixtures and tolerances should come from maintained corpus families, not one-off studies. |
| External comparison before competitive decision | R12 -> R13 | State-of-the-art or parity language requires direct comparative evidence. |
| Earned claims before tutorial/header finalization | R2-R7, R12 -> R8, R9 | Adoption docs should reflect what implementation and evidence actually earned. |

## Duplicate And Overlap Notes

| Residual | Overlap | Fence |
| --- | --- | --- |
| R1 | Overlaps Sprint 156 cross-platform/CI reconciliation and Sprint 152 report freshness policy. | Treat as evidence publication/reconciliation, not a standalone implementation sprint. |
| R3 | Overlaps R2 because both affect Windows support-tier wording. | Keep install-validation parity separate from staged pthread/POSIX test portability. |
| R4 | Overlaps R14 package-manager work. | Make ABI/shared-library product decision before distribution recipes. |
| R5 | Overlaps R12 external comparison if QR is the comparison target. | Corpus fixtures and proof owners must land before comparison claims. |
| R6 | Overlaps R12 external comparison if partial-SVD is the comparison target. | Subspace-safe comparison semantics must land before external parity language. |
| R7 | Overlaps benchmark, sentinel, coverage, dead-code, guardrail, corpus, and comparison reports. | Require only families tied to selected claims; keep other generated rows advisory. |
| R8 | Overlaps public claim wording in every implementation sprint. | Use tutorial alignment as final adoption cleanup, not as evidence creation. |
| R9 | Overlaps ABI work because public headers define API surface. | Header cleanup must preserve declarations and avoid ABI/signature drift. |
| R10 | Overlaps R4 if typed controls become ABI-visible. | Keep maintainer controls internal unless a complete API/ABI gate is selected. |
| R11 | Overlaps R7 because sentinel rows are generated report rows. | Add sentinel rows only with row semantics, freshness policy, and performance non-claims. |
| R13 | Overlaps R12 and all claim-bearing work. | Competitive decision is a final claim audit output, not an implementation shortcut. |

## Initial Epic 13 Candidate Set

Strong candidates for Day 5 selection:

- R2: Windows staged test portability closure.
- R3: Windows reviewed install-validation parity decision.
- R4: Shared-library ABI productization or stronger deferral.
- R5: Broad QR maintained corpus expansion.
- R6: Broad partial-SVD maintained corpus expansion.
- R7: Selected generated report freshness publication.
- R8/R9: Tutorial and header adoption closure.
- R12: First narrow external comparison harness.

Deferred or blocked by default:

- R10 and R11 remain deferred unless Day 5 explicitly swaps them into scope
  because they support a selected runtime/report claim.
- R13 remains blocked on R12 and final evidence.
- R14 remains deferred behind R4 because package-manager support needs a
  stable product contract first.
- R1 is a duplicate evidence-reconciliation input for Sprint 156, not a
  separate implementation target.

## Day 5 Questions

1. Should Sprint 147 select the Epic 13 project-plan scope as written, or
   substitute runtime/backend follow-through for one of the current
   Windows/corpus/package/comparison/adoption tracks?
2. Should R4 aim for implemented shared-library support, or a stronger
   static-first deferral with exact blockers and tests?
3. Which solver family should own the first external comparison target after
   corpus expansion: QR or partial-SVD?
4. Which generated report families are claim-bearing enough to justify
   `--require-generated` freshness gates?
5. Should package-manager distribution stay deferred for Epic 13 unless the
   ABI decision unexpectedly closes early?

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Epic 12 residual has an Epic 13 disposition. | Complete | R1-R14 intake disposition table assigns candidate, blocked, duplicate, or deferred status. |
| Duplicates are explicit. | Complete | Duplicate and overlap notes fence R1, R3/R2, R4/R14, R5/R12, R6/R12, R7, R8/R9, R10/R4, R11/R7, and R13/R12 interactions. |
| No residual becomes a claim without a gate. | Complete | Each row retains a promotion gate reminder and preserves the Sprint 146 non-claim boundary. |
