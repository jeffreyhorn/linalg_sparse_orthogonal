# Sprint 166 Day 1: Sprint Intake And Evidence Map

## Purpose

Day 1 establishes the Sprint 166 artifact package, ties the sprint to the
current Epic 14 project plan, and maps the final Epic 14 evidence categories
that will drive validation, claim recalibration, project-plan reconciliation,
the Epic 14 retrospective, and the residual queue.

## Source Plan

The active source plan is
`docs/planning/EPIC_14/PROJECT_PLAN.md`, section "Sprint 166: Epic 14 Final
Validation, Claim Recalibration & Closeout". The prompt referenced
`docs/planning/EPIC_12/PROJECT_PLAN.md` and the older title "Sprint 166: Final
Validation, Claim Calibration & Closeout"; that path/title mismatch is
recorded in `WORKING_NOTES.md` and `PLAN.md`.

## Sprint 166 Scope

Sprint 166 closes Epic 14 by:

- inventorying final Epic 14 evidence;
- running the strongest feasible local validation baseline and supplemental
  generated docs/report/package/performance checks;
- reconciling hosted CI evidence against local-only and advisory evidence;
- auditing public claims and non-claims;
- reconciling Epic 14 project-plan items as complete, narrowed, or
  residualized;
- drafting and publishing the Epic 14 retrospective if evidence is sufficient;
- publishing the final residual queue and next-epic candidates.

## Handoff Inputs Reviewed

| Sprint | Evidence package | Day 1 reading |
| --- | --- | --- |
| Sprint 157 | `docs/planning/EPIC_14/SPRINT_157/` | Epic 14 baseline, evidence freeze, and claim-target inputs for final reconciliation. |
| Sprint 158 | `docs/planning/EPIC_14/SPRINT_158/` | Generated API HTML publication closure and generated-reference boundaries. |
| Sprint 159 | `docs/planning/EPIC_14/SPRINT_159/` | Hosted oracle/comparison freshness promotion evidence. |
| Sprint 160 | `docs/planning/EPIC_14/SPRINT_160/` | QR comparison family closure and hosted/local evidence split. |
| Sprint 161 | `docs/planning/EPIC_14/SPRINT_161/` | Partial-SVD comparison publication closure and fixture-scoped comparison evidence. |
| Sprint 162 | `docs/planning/EPIC_14/SPRINT_162/` | Windows package parity decision and staged package non-claims. |
| Sprint 163 | `docs/planning/EPIC_14/SPRINT_163/` | Methodology-bound performance publication and non-superiority boundaries. |
| Sprint 164 | `docs/planning/EPIC_14/SPRINT_164/` | Declaration-preserving public-header/API coherence and generated API policy. |
| Sprint 165 | `docs/planning/EPIC_14/SPRINT_165/` | Static-first package boundary hardening and Sprint 166 package closeout handoff. |

## Final Evidence Category Map

| Category | Evidence to inventory | Claim boundary to preserve |
| --- | --- | --- |
| Generated API HTML publication | Sprint 158 artifacts, generated docs checks, API docs policy, maintainer guide | Generated API output must not imply hosted publication, release evidence, broad platform proof, package proof, or dynamic ABI support unless separately validated. |
| Hosted oracle/comparison freshness | Sprint 159 and 160 artifacts, report-index freshness checks, hosted workflow evidence | Hosted evidence must be separated from local-only rows and source-controlled advisory metadata. |
| QR comparison family | Sprint 160 artifacts, QR comparison rows, solver docs | Fixture-scoped QR comparison evidence must not become broad QR correctness, external-library parity, performance, or state-of-the-art proof. |
| Partial-SVD comparison family | Sprint 161 artifacts, partial-SVD comparison rows, solver docs | Fixture-scoped partial-SVD evidence must not become broad SVD parity, convergence-rate, vector identity, performance, or state-of-the-art proof. |
| Windows package decision | Sprint 162 artifacts and Windows CI package lane | Windows evidence remains CMake-first unless a later product decision adds Makefile or `pkg-config` command parity. |
| Performance publication | Sprint 163 artifacts, benchmark reports, performance sentinels | Local methodology-bound rows must not become portable performance, backend superiority, release, hosted, package, ABI, or state-of-the-art proof. |
| Public header/API coherence | Sprint 164 artifacts, public headers, generated docs checks | Header/doc cleanup must remain declaration-preserving unless an explicit reviewed API change is made. |
| Static-first package boundary | Sprint 165 artifacts, install scripts, package docs, static deferral guard | Static archive package proof must not imply shared-library, dynamic ABI, runtime-loader, package-manager, or broad platform package support. |
| Final validation | Sprint 166 validation artifacts and selected command matrix | Validation claims must identify local, hosted, reviewed, supplemental, advisory, and source-controlled evidence levels. |
| Epic 14 closeout | Project-plan reconciliation, Epic 14 retrospective, residual queue | Completed claims, narrowed claims, retained non-claims, and residual product decisions must be explicit. |

## Initial Non-Goal Register

| Non-goal | Reason |
| --- | --- |
| Broad state-of-the-art sparse linear algebra claim | Epic 14 final assessment must map positive claims to recurring evidence and reject unsupported broad claims. |
| Broad external-library parity | QR and partial-SVD comparisons are bounded by selected fixture families and evidence rows. |
| Portable performance superiority | Sprint 163 performance publication is methodology-bound and local, not portable superiority evidence. |
| Hosted proof from local generated rows | Local generated rows and normalized indexes are navigation/evidence metadata unless hosted CI owns them. |
| Shared-library support | Sprint 165 intentionally retained static-first package support and fail-closed shared-library deferral. |
| Dynamic ABI compatibility | Exact package version metadata is not a dynamic ABI policy or binary compatibility guarantee. |
| Runtime-loader behavior | No loader metadata, installed shared consumers, or runtime-loader validation has been selected. |
| Package-manager distribution | Package-manager distribution requires provider-specific packaging and install/upgrade proof. |
| Windows Makefile parity | Sprint 162 and Sprint 165 retain Windows package proof as CMake-first. |
| Windows `pkg-config` command parity | Windows `sparse.pc` inspection remains metadata-only unless provider-backed execution proof is added. |
| Broad platform parity | Reviewed/supplemental/local-only evidence levels must remain distinct. |

## Stop-Condition Register

Stop and revise if any Sprint 166 work:

1. converts bounded solver comparison evidence into broad solver parity;
2. converts local/advisory generated rows into hosted or release proof;
3. converts performance rows into portable performance or backend superiority
   claims;
4. converts static package proof into shared-library, dynamic ABI,
   runtime-loader, package-manager, Windows Makefile, or Windows `pkg-config`
   support;
5. marks an Epic 14 item complete without evidence links;
6. hides deferred product decisions by omitting residual owners, blockers,
   prerequisites, or promotion gates;
7. edits `.c` or `.h` files without running
   `make format && make lint && make test`.

## Day 2 Handoff

Day 2 should start the final evidence inventory with generated documentation,
API publication, hosted oracle/comparison freshness, and report-index evidence.
It should use the category map above to separate hosted evidence from
local-only generated output and advisory source-controlled rows.

## Validation Notes

Day 1 changed only Sprint 166 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 166 scope is tied to the Epic 14 project plan. | Complete | Source plan section and stale prompt path/title mismatch are recorded. |
| All Epic 14 evidence categories are identified. | Complete | Final evidence category map covers generated API, hosted oracle/comparison, QR, partial-SVD, Windows package, performance, public header/API, static package, validation, and closeout. |
| Unsupported claims are separated from final validation work. | Complete | Non-goal and stop-condition registers preserve unsupported package, ABI, platform, performance, external-parity, generated-report, and state-of-the-art boundaries. |
