# Sprint 196 Day 1 Artifact: Closeout Intake

**Date:** 2026-09-03
**Sprint item coverage:** 196.1, 196.2, 196.3, 196.4, 196.5, 196.6
**Day 1 goal:** Establish the Sprint 196 evidence map before any claim,
project-plan status, retrospective, or residual-queue edits begin.

## Intake Summary

Day 1 created the Sprint 196 working-notes scaffold and mapped the closeout
surfaces needed for Epic 17 final validation. The sprint now has an initial
item checklist, evidence-source inventory, claim-surface inventory,
validation-gate map, risk register, and Day 2 reconciliation questions.

No claim or status surfaces were edited during Day 1. That preserves the
planned ordering: evidence reconciliation first, then claim calibration and
project-plan status updates.

## Sprint Item Trace

| Item | Day 1 result |
| --- | --- |
| 196.1 Evidence Reconciliation | Identified Sprint 187-195 plans, working notes, retrospectives, daily artifacts, Day 14 closeouts, validation records, review comments, and residual tables as the reconciliation corpus. |
| 196.2 Claim Recalibration | Identified README, INSTALL, maintainer guide, benchmark docs, API docs, solver/user docs, corpus docs, report-index schema docs, planning docs, and public headers as claim surfaces. |
| 196.3 Project Plan Status | Marked `docs/planning/EPIC_17/PROJECT_PLAN.md` as the Day 7 status target, with status edits deferred until reconciliation evidence exists. |
| 196.4 Integrated Validation | Mapped candidate gate families for docs, C/header, CMake, Windows/PowerShell, report freshness, performance, reliability, review-surface, and package proof changes. |
| 196.5 Epic Retrospective | Marked the Epic retrospective as a later artifact dependent on claim/status calibration and final validation evidence. |
| 196.6 Residual Queue | Seeded the residual-queue process from Sprint 187-195 residual tables and Day 14 handoff artifacts. |

## Evidence Anchors

| Sprint | Closeout anchor | Current Day 1 reading |
| --- | --- | --- |
| 187 | `docs/planning/EPIC_17/SPRINT_187/artifacts/day14-closeout-summary.md` | Baseline and acceptance-gate sprint completed. |
| 188 | `docs/planning/EPIC_17/SPRINT_188/artifacts/day14-closeout-summary.md` | Homebrew proof completed with guarded package residual. |
| 189 | `docs/planning/EPIC_17/SPRINT_189/artifacts/day14-sprint-closeout.md` | PowerShell validation ownership completed with hosted evidence pending at closeout. |
| 190 | `docs/planning/EPIC_17/SPRINT_190/artifacts/day14-sprint-closeout.md` | Windows selected report freshness residual narrowed; hosted evidence and promotion pending at closeout. |
| 191 | `docs/planning/EPIC_17/SPRINT_191/artifacts/day14-closeout-and-handoff.md` | One bounded local-only comparison family landed with explicit residuals. |
| 192 | `docs/planning/EPIC_17/SPRINT_192/artifacts/day14-closeout-and-handoff.md` | One selected performance evidence lane landed with methodology-bound limits. |
| 193 | `docs/planning/EPIC_17/SPRINT_193/artifacts/day14-closeout.md` | Selected QR external-reference review surface reduced and guarded. |
| 194 | `docs/planning/EPIC_17/SPRINT_194/artifacts/day14-closeout-handoff.md` | Adoption/API coherence guidance simplified and calibrated. |
| 195 | `docs/planning/EPIC_17/SPRINT_195/artifacts/day14-closeout-review-package.md` | Selected symbolic Cholesky allocation-failure owner proved. |

## Claim Surfaces For Later Calibration

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`
- `docs/api_reference.md`
- `include/*.h`
- `docs/solver_selection.md`
- `docs/cookbook.md`
- `tests/corpus/README.md`
- `tests/corpus/schemas/report_index_fields.md`
- `docs/planning/EPIC_17/PROJECT_PLAN.md`
- `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md`

## Initial Validation Gate Families

- Repository hygiene: `git diff --check`, `make source-list-check`,
  `make format-check`.
- C/header quality: `make format`, `make lint`, `make test`,
  `make quality-review-compile`.
- CMake/reviewed build: `make quality-review-cmake-compile`,
  `make quality-review-cmake`, selected hosted CI evidence.
- Docs/API: `make docs-check`, `make api-docs-validate`,
  `make api-docs-freshness`, `make qr-header-docs-guard`.
- Windows/PowerShell: `make windows-powershell-validate`,
  `make windows-powershell-guard`, hosted Windows workflow evidence.
- Report freshness: `make report-index-oracle-freshness`,
  `make report-index-comparison-freshness`, selected target manifest checks.
- Performance: `make bench-canonical-report-freshness`,
  `make bench-canonical-report-freshness-tests`, `make performance-sentinels`.
- Reliability: `make symbolic-allocation-failure-gate`,
  `make iterative-allocation-failure-gate`, `make matmul-allocation-failure-gate`.
- Review-surface reduction: `make qr-external-ref-helper-guard`,
  `make ldlt-csc-helper-guard`.
- Package proof: install/CMake/Homebrew proof scripts and package guard targets
  identified from the Sprint 188 evidence set.

## Risks Carried Into Day 2

| Risk | Day 2 handling |
| --- | --- |
| State-of-the-art or release claims exceed earned evidence. | Require every claim to cite Sprint 187-195 evidence or remain a non-claim. |
| Prior residuals are duplicated or obscured. | Merge duplicate residuals only after preserving exact owner conditions. |
| Hosted and local evidence are conflated. | Keep local command results, unavailable local environments, and hosted CI proof separate. |
| Final validation scope grows because code/header files are touched. | Recompute mandatory gates from actual changed files before Day 13/14 closeout. |
| Package-manager readiness is overstated. | Keep provider-specific package support guarded until proof and metadata residuals are closed. |

## Day 2 Handoff

Day 2 should convert this inventory into an outcome ledger for Sprints 187-195.
The ledger should classify each outcome as complete, narrowed, guarded, pending,
superseded, deferred, or residualized; link the supporting artifact; and note
the specific claim/status edits required later in Sprint 196.

## Validation

- `git status --short --branch`
- `sed -n '1,90p' docs/planning/EPIC_17/SPRINT_196/PLAN.md`
- `sed -n '338,380p' docs/planning/EPIC_17/PROJECT_PLAN.md`
- `find docs/planning/EPIC_17/SPRINT_18* docs/planning/EPIC_17/SPRINT_19* -maxdepth 2 -type f | sort | rg 'SPRINT_18[7-9]|SPRINT_19[0-5]'`
- `rg -n "Status:|Outcome:|Residual|Validation|Complete" docs/planning/EPIC_17/SPRINT_18[7-9]/RETROSPECTIVE.md docs/planning/EPIC_17/SPRINT_19[0-5]/RETROSPECTIVE.md`

Day 1 changed planning documentation only. No `.c` or `.h` files were modified,
so the full C quality gate is not required for this day.

