# Sprint 156 Day 1 Closeout Baseline

## Purpose

Day 1 establishes the Sprint 156 final-closeout baseline. It identifies the
Epic 13 evidence sources, sprint artifact structure, claim stop conditions, and
validation surfaces that later Sprint 156 days must reconcile before the Epic
13 retrospective and next-epic handoff.

## Project-Plan Scope

Sprint 156 is `Epic 13 Final Validation, Claim Recalibration & Closeout` from
`docs/planning/EPIC_13/PROJECT_PLAN.md`. Its project-plan deliverables are:

- final Epic 13 evidence inventory;
- final validation package;
- public claim and non-claim audit;
- residual queue with promotion gates;
- Epic 13 retrospective;
- next-epic handoff.

Day 1 does not run the final quality baseline and does not edit public product
claims. It creates the source map and stop-condition register that constrain
the remaining closeout work.

## Source Inventory

| Sprint | Focus | Primary Closeout Sources | Day 1 Status |
| --- | --- | --- | --- |
| Sprint 147 | Epic 13 baseline, claim targets, and evidence gates | `PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, Day 1-14 artifacts | Available |
| Sprint 148 | Windows staged test portability closure | `PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, Day 1-14 artifacts | Available |
| Sprint 149 | Windows install-validation parity decision | `PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, Day 1-14 artifacts | Available |
| Sprint 150 | QR maintained corpus family expansion | `PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, Day 1-14 artifacts | Available |
| Sprint 151 | Partial-SVD maintained corpus family expansion | `PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, Day 1-14 artifacts | Available |
| Sprint 152 | Generated report freshness publication | `PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, Day 1-14 artifacts, `sprint153-abi-package-handoff.md` | Available |
| Sprint 153 | Shared-library ABI product decision | `PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, Day 1-14 artifacts | Available |
| Sprint 154 | External comparison harness and first narrow study | `PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, Day 1-14 artifacts, first narrow study artifact | Available |
| Sprint 155 | Tutorial, header cleanup, and API reference coherence | `PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, Day 1-14 artifacts, declaration-preservation scans | Available |

Inventory check: Sprints 147 through 155 each have a plan, working notes,
retrospective, and day artifacts. Sprint 152 and Sprint 154 include additional
handoff/study artifacts. Sprint 155 includes header declaration preservation
evidence and a direct Sprint 156 handoff.

## Closeout Scope

Sprint 156 owns final reconciliation for:

- platform support evidence across Linux, macOS, Windows, reviewed,
  supplemental, staged, local-only, and deferred lanes;
- QR and partial-SVD maintained corpus families;
- generated report freshness and normalized report index meaning;
- static-first package, install, CMake, `pkg-config`, and downstream consumer
  evidence;
- shared-library ABI product decision and package non-claims;
- external comparison harness and first narrow `qr-minnorm` study;
- tutorial, cookbook, README, INSTALL, solver-selection, API reference,
  maintainer, benchmark/report, and public-header claim wording;
- final residual queue and next-epic handoff.

Sprint 156 does not need to close every deferred technical gap. It must make
remaining gaps explicit, assign promotion gates, and prevent unsupported
claims from being treated as completed work.

## Non-Scope And Stop Conditions

The final closeout must stop and correct wording, evidence tables, or handoff
text if it introduces or implies:

- unqualified state-of-the-art sparse linear algebra status;
- broad QR, partial-SVD, SVD, eigensolver, direct-solver, Windows, package,
  external-library, or ecosystem parity;
- broad LAPACK, NumPy, SciPy, SuiteSparse, CHOLMOD, ARPACK, PETSc, Trilinos,
  Eigen, or package-manager parity;
- portable performance superiority or backend superiority from local
  benchmark, sentinel, report, or comparison rows;
- shared-library product support;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager distribution;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad Windows platform parity;
- generated report freshness without the selected freshness gate;
- generated API HTML freshness or completeness without the documented
  publication policy;
- optional external dependency skips or external service outages as pass
  evidence.

## Validation Surfaces Touched During Epic 13

| Surface | Evidence Source | Day 1 Closeout Use |
| --- | --- | --- |
| Local quality gates | Sprint 147-155 validation artifacts and retrospectives | Day 3-4 validation matrix and local baseline |
| Windows staged portability | Sprint 148 artifacts and retrospective | Day 6 platform reconciliation |
| Windows install/downstream | Sprint 149 artifacts and retrospective | Day 5 package validation and Day 6 platform reconciliation |
| QR corpus | Sprint 150 artifacts, tests, reports, and retrospective | Day 7 corpus/report validation |
| Partial-SVD corpus | Sprint 151 artifacts, tests, reports, and retrospective | Day 7 corpus/report validation |
| Generated report freshness | Sprint 152 artifacts and maintainer/report docs | Day 7 report validation and Day 10 claim audit |
| Package and ABI | Sprint 153 artifacts, `INSTALL.md`, package metadata, and maintainer docs | Day 5 package validation and Day 10 claim audit |
| External comparison | Sprint 154 artifacts, comparison scripts, report rows, and maintainer docs | Day 8 comparison reconciliation |
| Adoption and API docs | Sprint 155 artifacts, tutorial, API reference, cookbook, README, and public headers | Day 9 adoption/API reconciliation and Day 10 claim audit |
| Public claims and non-claims | README, INSTALL, docs, support-tier docs, headers, and retrospectives | Day 10 final claim/non-claim audit |

## Initial Evidence Questions For Day 2

Day 2 should convert this source inventory into a matrix that answers:

- Which final Epic 13 claims are backed by reviewed evidence?
- Which evidence is supplemental, local-only, staged, generated, or deferred?
- Which report rows require freshness checks before they can be cited?
- Which package/install claims depend on platform-specific proof?
- Which comparison claims are limited to the narrow `qr-minnorm` study?
- Which adoption/API improvements are documentation coherence work rather than
  new solver or package evidence?
- Which residuals already have owners and gates, and which still need them?

## Day 1 Completion Check

- Sprint 156 working notes exist.
- Sprint 156 artifact directory exists.
- Day 1 closeout baseline exists.
- Sprint 147-155 source inventory is recorded.
- Final closeout scope and non-scope are explicit.
- Stop conditions are recorded before claim audit work begins.
- Validation surfaces touched during Epic 13 are identified.
- Day 2 evidence-inventory questions are defined.
