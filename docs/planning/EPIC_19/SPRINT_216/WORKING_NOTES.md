# Sprint 216 Working Notes

**State:** Days 1-14 evidence reconciliation, claim audit, prior-sprint status,
residual priority selection, focused local validation, installed-static gates,
and [sprint retrospective](RETROSPECTIVE.md) complete. All six Sprint 216 selected deliverables are
closed locally on this branch, subject to review and merge; broader product
capabilities remain unearned.
**Plan:** [Sprint 216 plan](PLAN.md). **Epic scope:**
[project plan](../PROJECT_PLAN.md),
[review](../reviews/review-codex-2026-09-20.md), and
[todo](../reviews/todo-codex-2026-09-20.md).

The Sprint 207-215 plans, working notes, retrospectives, decision artifacts,
validation records, and closeout reviews are present on merged `master` as
PRs #230-#238. The merge record proves the changes landed; it does not by
itself re-run hosted jobs or validate a broader product claim. The Day 1
[intake](artifacts/day1-epic-evidence-intake.md) records observed boundaries.

## Source Index

Each `S###` directory has `PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`,
and daily `artifacts/`. The links below point to the decision, integrated
validation, and final closeout records that must be checked before assigning
item status. `R###`, `D###`, and `V###` are evidence keys used in the ledger.

| Sprint / merged PR | Decision and outcome | Validation and closeout |
| --- | --- | --- |
| 207 / [#230](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/230) | [R207], [D207] | [V207], [C207] |
| 208 / [#231](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/231) | [R208], [D208] | [V208], [C208] |
| 209 / [#232](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/232) | [R209], [D209] | [V209], [C209] |
| 210 / [#233](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/233) | [R210], [D210] | [V210], [C210] |
| 211 / [#234](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/234) | [R211], [D211] | [V211], [C211] |
| 212 / [#235](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/235) | [R212], [D212] | [V212], [C212] |
| 213 / [#236](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/236) | [R213], [D213] | [V213], [C213] |
| 214 / [#237](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/237) | [R214], [D214] | [V214], [C214] |
| 215 / [#238](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/238) | [R215], [D215] | [V215], [C215] |

The [Sprint 214 residual queue](../SPRINT_214/RESIDUAL_QUEUE.md) and
[Sprint 215 residual queue](../SPRINT_215/RESIDUAL_QUEUE.md) are explicit
handoffs; older residuals also appear in each retrospective. The
[Day 8 crosswalk](artifacts/day8-residual-queue-intake.md) deduplicated them
against the Epic review/todo without discarding unresolved gates.

## Day 2 Reconciliation

The [Day 2 cross-sprint record](artifacts/day2-outcome-validation-reconciliation.md)
compares all nine decisions with their integrated checks, closeout reviews,
merged PRs, platform limits and residual dependencies. Proposed sprint-level
dispositions are: 207 provider decision complete/public distribution
residualized; 208-209 Windows selected freshness re-deferred; 210 one LDLT
allocation owner complete/broader reliability residualized; 211 one helper
cluster complete/broader refactor residualized; 212 threshold-free benchmark
policy complete/timing gate deferred; 213 local-only API policy complete/
publication deferred; 214 ABI/release design complete/shared and release
proof deferred; 215 blueprint and blocked fallback complete/external result
residualized. These are **not** final item statuses or new support claims.

Day 2 discrepancies D2-1 through D2-6 include the stale project-plan
`215-216` pending row, historical validation records preceding PR review
changes, Sprint 215's historical nine-test count versus 13 current local
synthetic tests, missing external v1 schema/validator, unknown current hosted
artifact status, and design-only outcomes. Day 6-7 owns plan status fixes;
Days 9-14 own final-tree validation.

## Day 3 Evidence Freeze

The [Day 3 claim inventory](artifacts/day3-claim-inventory.md) freezes the
Sprint 207-215 decision, retrospective, validation, closeout, merged-PR, and
current contract source set for item 216.1. This is a historical evidence-map
freeze, not a current-tree gate pass or a capability promotion. D2-1 through
D2-6 remain named corrections, blockers, or residuals. New hosted evidence
must be recorded as a dated delta with exact run/job/artifact identity.

The Day 3 matrix covers README, INSTALL, maintainer, benchmark, API, corpus,
and Epic 19 planning text. It identifies C01-C02 as unqualified performance
language, C06-C07 as missing/stale external-baseline context in the support
matrix, C09/C14 as superseded API policy ownership, and C16 as the stale
Sprint 215/216 snapshot. C03-C05, C08, C10-C13, C15, and C17 already express
bounded claims and should retain those negative boundaries. Days 4-5 own
public-doc revisions; Days 6-7 own plan statuses. Neither the blueprint nor
the blocked pilot is accepted external evidence.

## Day 4 User And Maintainer Calibration

The [Day 4 edit record](artifacts/day4-user-maintainer-claims.md) tracks README
performance wording, INSTALL static/platform/release and blocked external
baseline context, and the maintainer guide's latest local-only API owner and
current-checkout validation instructions. D4-1 adds the formerly unqualified
README LU complexity note to the claim inventory. A focused documentation
regression now covers INSTALL's blocked-pilot boundary alongside README.
These are wording and guard changes only; they do not promote product support.

## Day 5 Remaining Claim Surfaces

The [Day 5 completed claim matrix](artifacts/day5-benchmark-api-corpus-claims.md)
reconciles the benchmark guide, API reference, corpus README, report-index
schema text, and Sprint 215 planning section. D5-1 removes ambiguous
benchmark "publication row" wording; D5-2 separates optional-data policy,
source-controlled selected targets, and the blocked external `attempt.json`.
The external-claim guard now covers those evidence docs and accepts explicit
non-claims. At Day 5, C16's stale status snapshot remained assigned to item
216.3 on Days 6-7. No selected report row,
v1 evidence validator, hosted API publication, or timing gate was created.

## Day 6 Early Item Status

The [Day 6 item-status ledger](artifacts/day6-project-plan-status-207-211.md)
assigns one `complete` disposition to each of the 30 Sprint 207-211 items,
with merged decision, implementation, validation, or closeout evidence beside
each. The status closes the selected deliverable, not a stronger product
capability. Public provider distribution, selected Windows Cholesky/QR
freshness, broad allocation reliability, and broader review-surface reduction
remain deferred or residualized with explicit next acceptance targets.
At Day 6, Sprint 212-215 item statuses and the stale 215-216 snapshot row
remained Day 7 work; no current hosted or full quality rerun was inferred.

## Day 7 Final Prior-Sprint Status

The [Day 7 item-status ledger](artifacts/day7-project-plan-status-212-215.md)
assigns one evidence-backed `complete` disposition to each Sprint 212-215
selected deliverable (24 items). Together with Day 6, all 54 Sprint 207-215
items are covered. The project-plan snapshot now separates merged Sprint 215
from in-progress Sprint 216. C16/D2-1 is corrected, while historical PR-review
chronology, external v1 validation, current hosted evidence, and final-tree
checks remain open as named gates. Sprint 216's six item statuses are
unassigned placeholders until integrated validation and Day 14 review.

## Day 8 Residual Intake

The [Day 8 intake crosswalk](artifacts/day8-residual-queue-intake.md) maps
Sprint 207-215 residuals and Epic 19 review/todo tracks to the
[draft next-epic queue](RESIDUAL_QUEUE.md). Its 21 ranked entries separate
external evidence integrity, bounded closure candidates, and long-horizon
product/research decisions. Each entry has an owner role, dependency,
acceptance evidence, stop condition, and claim gate. The Sprint 211 local
full-CTest gap is not carried forward as missing because Sprint 214 recorded
60/60 local CTest; hosted candidate proof remains separate. The queue is a
Day 8 draft, not a product decision or completed item 216.6; Day 9 owns final
priority and validation-matrix reconciliation.

## Day 9 Queue Decision And Validation Selection

The [prioritized queue](RESIDUAL_QUEUE.md) now fixes the Day 8 first order
for 21 closure targets, subject to a recorded Day 14 evidence delta. The
[Day 9 decision and validation matrix](artifacts/day9-residual-decision-validation-matrix.md)
reconciles completed selected items with residual capabilities, records the
explicit **no state-of-the-art claim** decision, and selects executable
current-tree checks for Days 10-14. A bounded external correctness statement
requires validated two-sided results and publication approval; a broad
assessment requires representative, pre-registered matrix/family/platform
coverage and controlled timing, not merely the Sprint 215 blueprint.
No C/header change was found in the Day 9 worktree inventory. Install checks
that invoke `make clean` must run serially after report/benchmark checks, with
a rebuild before any final freshness rerun. Hosted and external v1 gaps stay
named blockers, not passing local checks.

## Day 10 Focused Validation

The [Day 10 execution record](artifacts/day10-focused-validation.md) reports
exit codes for every selected focused claim, API, corpus, comparison,
benchmark, pilot-synthetic, and docs check. All passed locally on Darwin
`24.6.0` x86_64. Generated comparison and benchmark rows were inspected for
status, provenance, dependency deferral, and explicit non-claims; neither
local freshness nor synthetic pilot hashes establish CHOLMOD, hosted,
portable-timing, release, or state-of-the-art evidence. No C/header file was
changed; Day 11 later ran the serial installed-static/package gates.

## Day 11 Quality Gate And Retrospective Outline

The [Day 11 gate record and outline](artifacts/day11-quality-gate-and-retro-outline.md)
reports six serial static/package commands exiting `0`, including 21 static
contract tests, the developer-mode local Homebrew proof boundary, 26 Make
install checks, and 30 CMake install checks. No C/header file changed, so the
full source chain was not triggered. The Make install test cleaned the build
tree; Day 10's generated comparison artifacts are no longer present and
must be regenerated before any later freshness claim. The outline maps Epic
outcomes, negative evidence, validation, unsupported claims and residuals
to primary artifacts for Day 12 drafting. Hosted and external v1 gates remain
unrun; Sprint 216 statuses remain provisional.

## Day 12 Epic Outcomes Draft

The [Epic 19 retrospective draft](../EPIC_19_RETROSPECTIVE.md) and
[Day 12 evidence cross-check](artifacts/day12-epic-outcomes-draft.md) group
Sprint 207-215 outcomes by capability, cite primary decisions and local
validation, preserve the blocked external and hosted gaps, and route each
unearned capability to the prioritized queue. All 54 prior-sprint items are
complete only for their selected deliverables; Sprint 216 items remain
unassigned pending Day 13 claim review and Day 14 integrated validation.
No C/header file changed for this documentation draft.

## Day 13 State-Of-The-Art And Claim Review

The [Day 13 retrospective audit](artifacts/day13-retrospective-review.md)
cross-checks every outcome group, validation statement and lesson against
the Day 3/5 claim matrices, Day 6/7 ledgers, residual queue, and Day 10/11
current-tree checks. The [review-ready retrospective](../EPIC_19_RETROSPECTIVE.md)
now distinguishes CHOLMOD discovery from an executed baseline: zero accepted
external cells, no two-sided solver result, no controlled timing, no hosted
external matrix, and no publication permission. RQ-01 through RQ-03 would
unlock only an exact-cell correctness statement; RQ-11/RQ-12/RQ-17/RQ-18
define a separate broad-assessment entry bar, not automatic ranking.
Reviewer questions and the final non-claim list are in the audit. Sprint 216
statuses remain unassigned until Day 14 integrated validation and closeout.

## Day 14 Integrated Validation And Closeout

The [Day 14 record](artifacts/day14-integrated-validation-closeout.md) reruns
the affected claim, API, corpus, selected manifest, comparison, benchmark,
pilot-synthetic and docs checks on the current tree. Local results pass;
regenerated comparison rows remain source-controlled-helper `local_only` and
the canonical benchmark remains threshold-free. The 54 prior-sprint item
audit and 21-entry queue audit pass. No C/header file changed, so the full
source chain was not required. Hosted Windows, actual CHOLMOD execution,
external v1 validation, shared ABI and candidate release remain unrun or
unavailable, with exact queue gates. Items 216.1-216.6 are `complete` for
their selected deliverables on this branch, subject to review and merge.

## Item-To-Evidence Ledger

The Day 6 and Day 7 ledgers assign actual 207-215 dispositions, while the
[project-plan final table](../PROJECT_PLAN.md#final-item-status-on-branch)
assigns 216.1-216.6 from Day 14 results. An item can close a decision or design
without proving the capability considered by that decision. Owner entries
name review roles, not assigned individuals. These began as Day 1 acceptance
tests; Days 2-14 added evidence and local validation. Day 13 reviewed item
216.5 against those ledgers and the queue without changing their selected
dispositions.

| Item | Owner role | Evidence | Provisional acceptance check |
| --- | --- | --- | --- |
| 207.1 | Package owner | [D207] | Provider choice and excluded tiers stated. |
| 207.2 | Package owner | [R207] | Formula, metadata, archive and static surface audited. |
| 207.3 | Package owner | [V207] | Local proof path runs with its environment/cleanup limits. |
| 207.4 | Package owner | [V207] | Guards reject unsupported provider claims. |
| 207.5 | Docs owner | [R207] | User/maintainer wording says developer-mode local proof only. |
| 207.6 | Quality owner | [V207] | Required package/install/docs checks passed; gaps retained. |
| 208.1 | Windows owner | [R208] | Exact hosted Cholesky artifact inspected and scoped. |
| 208.2 | Windows owner | [D208] | Promotion or re-deferral justified by generated rows. |
| 208.3 | Manifest owner | [R208] | Metadata/absence guards match decision. |
| 208.4 | Report owner | [V208] | Windows-path and stale/missing row regressions pass. |
| 208.5 | Docs owner | [R208] | Selected Windows freshness remains appropriately qualified. |
| 208.6 | Quality owner | [V208] | Manifest, workflow, report and docs gates recorded. |
| 209.1 | Windows/QR owner | [R209] | Exact MSVC probe and six-file artifact contract defined. |
| 209.2 | CI owner | [R209] | Bounded collection lane exists without promotion. |
| 209.3 | Report owner | [V209] | Artifact/path/filter regressions pass. |
| 209.4 | Manifest owner | [D209] | Hosted artifact absence keeps selected QR re-deferred. |
| 209.5 | Docs owner | [R209] | Windows QR claim guards and docs match re-deferral. |
| 209.6 | Quality owner | [V209] | QR, workflow, manifest, docs checks recorded. |
| 210.1 | Solver owner | [D210] | One allocation-failure owner selected explicitly. |
| 210.2 | Solver owner | [R210] | Cleanup, stale-output, input and retry invariants frozen. |
| 210.3 | Solver owner | [R210] | Deterministic hooks reach selected LDLT allocations. |
| 210.4 | Test owner | [V210] | Failure sweep and lifecycle assertions execute. |
| 210.5 | Quality/docs owner | [R210] | Focused Make/CTest gate and claim boundary exist. |
| 210.6 | Quality owner | [V210] | Focused, registration, source and C gates recorded. |
| 211.1 | Review owner | [D211] | Candidate ranking selects one bounded cluster. |
| 211.2 | Review owner | [R211] | Behavior and registration invariants stated. |
| 211.3 | C/test owner | [R211] | Helper boundary and dependencies specified. |
| 211.4 | C/test owner | [R211] | Selected test bodies moved without claim expansion. |
| 211.5 | Quality owner | [V211] | Ownership/order and behavior regressions pass. |
| 211.6 | Quality owner | [V211] | Full C and CMake parity evidence recorded. |
| 212.1 | Benchmark owner | [R212] | Runner, fields and selected-row evidence inventoried. |
| 212.2 | Benchmark owner | [D212] | Threshold-free decision tied to missing methodology. |
| 212.3 | Benchmark owner | [R212] | Freshness/metadata guards implement chosen policy. |
| 212.4 | Test owner | [V212] | Negative metadata and claim regressions pass. |
| 212.5 | Docs owner | [R212] | No hosted timing threshold is implied. |
| 212.6 | Quality owner | [V212] | Benchmark, manifest and docs checks recorded. |
| 213.1 | API/docs owner | [R213] | Four publication options compared. |
| 213.2 | API/docs owner | [D213] | Local-only policy and rejected options explicit. |
| 213.3 | CI owner | [R213] | Workflow/staging guard implements local-only choice. |
| 213.4 | Test owner | [V213] | Routing, link and workflow regressions pass. |
| 213.5 | Docs owner | [R213] | User/maintainer text excludes hosted HTML. |
| 213.6 | Quality owner | [V213] | API docs freshness/docs checks recorded. |
| 214.1 | Release owner | [D214] | Design/deferral selected, not shared proof. |
| 214.2 | ABI owner | [R214] | Provisional export/compatibility/loader gates specified. |
| 214.3 | Build owner | [R214] | Unsupported shared modes rejected; static contract kept. |
| 214.4 | Install owner | [V214] | Installed static consumers and failures exercised. |
| 214.5 | Release owner | [R214] | Checklist names tag, archive, provenance and CI gates. |
| 214.6 | Quality owner | [V214] | Install, guards and applicable full gates recorded. |
| 215.1 | Baseline owner | [R215] | Exact CHOLMOD cell and exclusions selected. |
| 215.2 | Numerical owner | [R215] | Correctness and timing protocols separate. |
| 215.3 | Report owner | [R215] | Isolated v1 schema design; implementation gap visible. |
| 215.4 | Pilot owner | [D215] | Blocked fallback evidenced; no solver pass inferred. |
| 215.5 | Docs owner | [R215] | Blueprint/pilot/non-claim wording distinguished. |
| 215.6 | Quality owner | [V215] | Focused/local gates and unrun hosted lanes recorded. |
| 216.1 | Epic owner | [Day 3 freeze](artifacts/day3-claim-inventory.md); [Day 14 closeout](artifacts/day14-integrated-validation-closeout.md) | Complete for nine-sprint evidence reconciliation; external/hosted gaps remain named. |
| 216.2 | Docs owner | [Day 5 matrix](artifacts/day5-benchmark-api-corpus-claims.md); [Day 13 audit](artifacts/day13-retrospective-review.md); [Day 14 closeout](artifacts/day14-integrated-validation-closeout.md) | Complete for claim calibration on named surfaces, not feature promotion. |
| 216.3 | Epic owner | [Day 6 ledger](artifacts/day6-project-plan-status-207-211.md); [Day 7 ledger](artifacts/day7-project-plan-status-212-215.md); [Day 14 closeout](artifacts/day14-integrated-validation-closeout.md) | Complete for 54 prior-item and six current-item dispositions. |
| 216.4 | Quality owner | [Day 10 focused validation](artifacts/day10-focused-validation.md); [Day 11 quality gate](artifacts/day11-quality-gate-and-retro-outline.md); [Day 14 closeout](artifacts/day14-integrated-validation-closeout.md) | Complete for locally required changed-surface gates; no C/header change; hosted/external/release proof absent. |
| 216.5 | Epic owner | [Retrospective](../EPIC_19_RETROSPECTIVE.md); [Day 13 audit](artifacts/day13-retrospective-review.md); [Day 14 closeout](artifacts/day14-integrated-validation-closeout.md) | Complete for selected retrospective and explicit state-of-the-art non-claim. |
| 216.6 | Next-epic owner | [Prioritized queue](RESIDUAL_QUEUE.md); [Day 9 decision](artifacts/day9-residual-decision-validation-matrix.md); [Day 14 closeout](artifacts/day14-integrated-validation-closeout.md) | Complete for 21 ranked acceptance targets; those future capabilities remain unearned. |

## Claim Register

| Claim surface | Current evidence boundary | Current disposition |
| --- | --- | --- |
| Package distribution | Sprint 207 local developer-mode Homebrew static proof; no public provider tier. | Day 4 INSTALL retains static-first matrix and provider non-claim; no new provider path. |
| Windows Cholesky | Sprint 208 hosted selected files exist, but generated support-tier contradictions prevent freshness promotion. | Re-deferred. |
| Windows QR | Sprint 209 evidence-collection lane exists; no inspected hosted QR artifact at closeout. | Re-deferred. |
| Allocation reliability | Sprint 210 selected no-reorder linked-list LDLT 25-site proof. | Selected owner only. |
| Review surface | Sprint 211 selected LDLT CSC native-parity helper extraction. | No solver behavior claim. |
| Benchmark | Sprint 212 selected canonical row remains threshold-free. | Day 4 README narrows C01-C02 and D4-1; Day 5 guide fixes D5-1, retains C03/C12 and local S5/S6 split. |
| Generated API | Sprint 213 HTML remains local-only ignored output. | Day 4 maintainer guide corrects C09; Day 5 API reference corrects C14 and retains C08/C15. |
| ABI/release | Sprint 214 static-first deferral/design and local installed static consumers. | Day 4 INSTALL/maintainer retain C05/C10 checklist-as-gates, not shared/release proof. |
| External baseline | Sprint 215 blueprint and blocked preflight; no accepted two-sided solver result. | Day 4 INSTALL adds C06-C07; Day 5 benchmark/corpus/schema/planning text and guard close D5-2; no accepted parity or speed. |
| Epic planning | Sprint 215 merged #238; Sprint 216 selected work closed on branch, review/merge pending. | Day 7 corrected C16; Day 14 assigned 216 statuses from local validation and retained C17 non-goals. |

## Validation Matrix

Day 1 is a documentation intake, not a rerun of historical or hosted proof.
The following commands are candidates for Day 9 selection, not passing Day 1
results. Days 10-14 must record exact platform, exit status and logs.

| Changed/evidence surface | Candidate focused check | Boundary |
| --- | --- | --- |
| Claim docs | `make docs-check`; relevant Python claim-guard tests | Text consistency only. |
| Package/static install | `bash tests/test_install.sh`; static package and deferral guards | Local install proof is not a public provider release. |
| Selected manifests/reports | `make report-index-comparison-freshness`; selected manifest tests | Generated-local freshness, not broad parity. |
| Benchmark | `make bench-canonical-report-freshness-tests` | Threshold-free row only. |
| Generated API | `make api-docs-freshness` | Local-only generation and routing. |
| External pilot | `python3 -m unittest tests.test_run_cholmod_pilot` | Synthetic/blocker coverage, not solver execution. |
| C/header changes | `make format && make lint && make test` | Mandatory if any `.c` or `.h` file changes. |
| Hosted evidence | Inspect exact run, job, artifact and status | Workflow registration alone is not execution. |

## Decision Log

| Day | Decision | Rationale / next point |
| --- | --- | --- |
| 1 | Do not promote any support or state-of-the-art claim during intake. | Day 3 claim inventory and Day 14 final gate decide wording. |
| 1 | Treat merged retrospectives as historical evidence, not fresh execution. | Day 2 reconciles logs and hosted artifacts; Day 9 selects reruns. |
| 1 | Leave project-plan statuses unchanged today. | Snapshot still marks 215-216 pending; Day 6-7 status work must correct 215 using merged PR #238 and keep 216 provisional. |
| 3 | Freeze the Sprint 207-215 historical source set, not a global passing verdict. | Discrepancies D2-1 through D2-6 remain visible; new hosted/current-tree evidence needs an explicit dated delta. |
| 3 | Approve no stronger public claim yet. | C01-C17 classify existing wording; Days 4-7 own changes, Days 9-14 own final validation. |
| 4 | Treat README performance labels as descriptions of paths, not comparative results. | C01-C02 and new D4-1 narrowed; selected benchmark remains threshold-free. |
| 4 | Put the blocked external pilot in the public support matrix. | C06-C07 now distinguish blueprint and blocked preflight from accepted correctness/timing; add textual regression without calling it solver proof. |
| 5 | Keep benchmark freshness, sentinel thresholds, and external acceptance in separate evidence lanes. | D5-1/D5-2 fixed in docs; provisional attempt and optional-data skips cannot become selected pass rows. |
| 5 | Close the claim-audit matrix while leaving the status rewrite open. | C16 remains an item 216.3 task for Days 6-7, not an implicit pass. |
| 6 | Mark each Sprint 207-211 selected item `complete`, not its unearned capability. | Merged sprint closeouts support all 30 bounded deliverables; public provider, selected Windows, broad allocation, and broad refactor proof remain separate future targets. |
| 7 | Complete prior-sprint item ledger and split Sprint 215/216 snapshot. | All 54 items have selected-deliverable dispositions; blocked pilot is not external evidence, and Sprint 216 items remain unassigned until final gates. |
| 8 | Draft one deduplicated queue by acceptance gate, not by repeated sprint wording. | P0 external v1/cell/publication prerequisites; P1 bounded closures; P2 long-horizon decisions with revisit triggers. Day 9 may revise ranks after validation-matrix review. |
| 9 | Finalize residual priority baseline without promoting any capability. | RQ-01/02/03 stay prerequisite-ordered; P1 selected closures and P2 scope decisions retain stop conditions. No state-of-the-art claim is earned. |
| 9 | Select focused current-tree gates and record build-state interactions. | Docs/claim/API/corpus/report/benchmark/pilot and static install checks scheduled; source gate becomes mandatory only if `.c`/`.h` changes. |
| 10 | Treat generated comparison and benchmark outputs as local-only evidence. | Focused checks pass, but all comparison rows use source-controlled helpers; optional package rows defer, benchmark threshold is `n/a`, and pilot tests do not execute CHOLMOD. |
| 13 | Decline state-of-the-art and stronger support wording at retrospective review. | Zero accepted external cells, no two-sided or controlled timing results, no publication approval; bounded and broad gates remain in the residual queue. Day 14 owns final integrated checks and item status assignment. |
| 14 | Close all six Sprint 216 selected deliverables on the branch without promoting residual capabilities. | Affected local gates, 54-item prior ledger and 21-entry queue audits pass; C chain not triggered, hosted/external/release lanes remain unrun. Review and merge are pending. |

## Risk And Gap Register

| ID | Gap or risk | Impact | Next check |
| --- | --- | --- | --- |
| G1 | `PROJECT_PLAN.md` snapshot said Sprints 215-216 pending. | Stale planning claim after #238 merge. | Resolved Day 7: separate merged 215 and in-progress 216 rows; verify at Day 14 closeout. |
| G2 | Sprint 215 v1 schema and validator absent; pilot blocked. | No accepted external correctness/timing evidence. | Day 9 RQ-01/02/03 prerequisite order; no solver/publication claim until accepted evidence. |
| G3 | Windows selected freshness remains re-deferred. | Hosted lane existence may be mistaken for promotion. | Day 2 compare generated rows, manifest and artifact records. |
| G4 | Public package, shared ABI, release and hosted API claims unearned. | Docs may overstate support. | Day 4 user/maintainer calibration; Day 5 remaining surfaces. |
| G5 | Historical validation records are platform/scenario-specific. | Reusing pass counts as current proof would overclaim. | Day 10 focused, Day 11 installed-static, and Day 14 changed-surface local reruns recorded; exact current hosted evidence remains residualized. |
| G6 | Older residuals duplicate Sprint 214/215 handoffs. | Repeated wording can hide distinct acceptance gates or inflate the queue. | Resolved at planning level: Day 8 crosswalk and Day 9 queue decision retain 21 distinct gates; review again at Day 14. |
| G7 | Some sprint integrated checks predate merged PR review changes. | Historical pass logs may be mistaken for final-tree coverage. | Days 9-14 rerun current affected gates. |

## Changed-Surface Tracker

| Day | Paths | C/header changed? | Required checks | Result |
| --- | --- | --- | --- | --- |
| 1 | `SPRINT_216/PLAN.md`, `WORKING_NOTES.md`, `artifacts/day1-epic-evidence-intake.md` | No | Link/whitespace and focused docs check | Local links, whitespace, and `make docs-check` passed; no hosted or C gate run. |
| 2 | `WORKING_NOTES.md`, `artifacts/day2-outcome-validation-reconciliation.md` | No | Local links, docs check, focused pilot suite | Links, whitespace, `make docs-check`, and 13 pilot synthetic tests passed locally; no hosted or C gate run. |
| 3 | `WORKING_NOTES.md`, `artifacts/day3-claim-inventory.md` | No | Local links, whitespace, docs check | Local links, whitespace, and `make docs-check` passed; no hosted or C gate run. |
| 4 | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `tests/test_external_baseline_claim_docs.py`, `WORKING_NOTES.md`, `artifacts/day4-user-maintainer-claims.md` | No | Documentation, performance/package/API claim guards; links and whitespace | `make docs-check`, support docs, external/selected-performance tests, package-manager/static deferral guards, `make api-docs-freshness`, links, whitespace, and `git diff --check` passed locally. No hosted or C gate run. |
| 5 | `benchmarks/README.md`, `docs/api_reference.md`, `tests/corpus/README.md`, `tests/corpus/schemas/report_index_fields.md`, `PROJECT_PLAN.md`, `tests/test_external_baseline_claim_docs.py`, `WORKING_NOTES.md`, `artifacts/day5-benchmark-api-corpus-claims.md` | No | Focused claim, corpus schema, API and docs checks; links/whitespace | PASS: external-baseline claim unit tests (5), selected-performance docs, corpus schema, canonical report freshness, API docs freshness, support-docs guard, selected-report manifest, local links, trailing whitespace, and `git diff --check`. |
| 6 | `PROJECT_PLAN.md`, `WORKING_NOTES.md`, `artifacts/day6-project-plan-status-207-211.md` | No | Link, whitespace, docs checks and item-count/status audit | PASS: 30 unique item dispositions; local links, `make docs-check`, trailing whitespace, and `git diff --check`. Historical/hosted gates not rerun. |
| 7 | `PROJECT_PLAN.md`, `WORKING_NOTES.md`, `artifacts/day7-project-plan-status-212-215.md` | No | 54-item status audit, link/contradiction/whitespace, docs and claim guards | PASS: 54 unique prior-sprint dispositions; local links, `make docs-check`, external claim unit tests (5), support-docs guard, selected-performance docs, trailing whitespace, and `git diff --check`. No hosted or C gate run. |
| 8 | `RESIDUAL_QUEUE.md`, `PROJECT_PLAN.md`, `WORKING_NOTES.md`, `artifacts/day8-residual-queue-intake.md` | No | Source crosswalk, queue ID/field/link audit, docs and whitespace checks | PASS: 21 unique entries with owner/dependency/acceptance/stop/claim fields; local links, `make docs-check`, support-docs guard, trailing whitespace, and `git diff --check`. No capability/hosted/C gate run. |
| 9 | `RESIDUAL_QUEUE.md`, `PROJECT_PLAN.md`, `WORKING_NOTES.md`, `artifacts/day9-residual-decision-validation-matrix.md` | No | Queue ID/field/status/link audit, docs/claim guards and whitespace | PASS: 21 queue entries and 54 prior-sprint dispositions audited; local links, external claim unit tests (5), selected-performance docs, support-docs guard, `make docs-check` (18 public headers), trailing whitespace, and `git diff --check`. No hosted or C gate run. |
| 10 | `WORKING_NOTES.md`, `PROJECT_PLAN.md`, `artifacts/day10-focused-validation.md` | No | Day 9 focused claim/API/corpus/comparison/benchmark/pilot/docs matrix; links and whitespace | PASS: all 14 focused commands exited 0; six local comparison bundles (40 passing local-only rows), canonical threshold-free report, 13 pilot synthetic tests, and 18-header docs coverage inspected. Install/package, hosted, CHOLMOD, and C gate not run. |
| 11 | `WORKING_NOTES.md`, `PROJECT_PLAN.md`, `artifacts/day11-quality-gate-and-retro-outline.md` | No | Static/package guards, serial installed consumers, conditional C gate decision, docs/claim guards, links/whitespace | PASS: static contract unit tests (21), static deferral, package-manager unit guard, local Homebrew boundary, Make install (26/26), CMake install (30/30), external claim tests (5), support-docs guard, and `make docs-check` (18 public headers); source gate not triggered. Generated Day 10 reports removed by `make clean`; no hosted/external proof. |
| 12 | `EPIC_19_RETROSPECTIVE.md`, `PROJECT_PLAN.md`, `WORKING_NOTES.md`, `artifacts/day12-epic-outcomes-draft.md` | No | Primary-evidence cross-check, local links, claim/docs guards, whitespace | PASS: 54 prior-item and 21 queue-ID audits; local links, external claim tests (5), selected-performance docs, support-docs guard, `make docs-check` (18 public headers), trailing whitespace and `git diff --check`. No hosted, external-solver, installed-package or C gate rerun. |
| 13 | `EPIC_19_RETROSPECTIVE.md`, `PROJECT_PLAN.md`, `WORKING_NOTES.md`, `artifacts/day13-retrospective-review.md` | No | Claim/ledger/queue audit, links, docs and focused claim guards, whitespace | PASS: local links, external claim tests (5), selected-performance docs, support-docs guard, `make docs-check` (18 public headers), trailing whitespace and `git diff --check`. No hosted, external-solver, release or C gate rerun. |
| 14 | `EPIC_19_RETROSPECTIVE.md`, `PROJECT_PLAN.md`, `WORKING_NOTES.md`, `artifacts/day14-integrated-validation-closeout.md` | No | Final affected claim/API/corpus/report/benchmark/pilot/docs checks, 60-item/21-entry status/queue audit, links and whitespace | PASS: Day 14 focused checks and 54 prior-item/21 queue-entry audits; final documentation and whitespace checks recorded in the Day 14 closeout. No hosted, external-solver, release or C gate run. |

## Next Actions

1. Review and merge `sprint-216` only with the selected-scope status and
   [Day 14 record](artifacts/day14-integrated-validation-closeout.md) intact.
2. Carry RQ-01 through RQ-21 into next-epic planning by their exact
   acceptance gates, beginning with the external v1 validator, two-sided
   cell and publication review. Do not infer new hosted results or claims.

[R207]: ../SPRINT_207/RETROSPECTIVE.md
[D207]: ../SPRINT_207/artifacts/day5-provider-decision.md
[V207]: ../SPRINT_207/artifacts/day12-integrated-validation.md
[C207]: ../SPRINT_207/artifacts/day14-closeout-review.md
[R208]: ../SPRINT_208/RETROSPECTIVE.md
[D208]: ../SPRINT_208/artifacts/day5-promotion-decision.md
[V208]: ../SPRINT_208/artifacts/day13-integrated-validation.md
[C208]: ../SPRINT_208/artifacts/day14-closeout-review.md
[R209]: ../SPRINT_209/RETROSPECTIVE.md
[D209]: ../SPRINT_209/artifacts/day8-manifest-decision.md
[V209]: ../SPRINT_209/artifacts/day13-integrated-validation.md
[C209]: ../SPRINT_209/artifacts/day14-closeout-review.md
[R210]: ../SPRINT_210/RETROSPECTIVE.md
[D210]: ../SPRINT_210/artifacts/day2-owner-ranking.md
[V210]: ../SPRINT_210/artifacts/day12-integrated-validation.md
[C210]: ../SPRINT_210/artifacts/day14-closeout-review.md
[R211]: ../SPRINT_211/RETROSPECTIVE.md
[D211]: ../SPRINT_211/artifacts/day2-candidate-ranking.md
[V211]: ../SPRINT_211/artifacts/day12-integrated-validation.md
[C211]: ../SPRINT_211/artifacts/day14-closeout-review.md
[R212]: ../SPRINT_212/RETROSPECTIVE.md
[D212]: ../SPRINT_212/artifacts/day5-product-policy-decision.md
[V212]: ../SPRINT_212/artifacts/day12-integrated-validation.md
[C212]: ../SPRINT_212/artifacts/day14-closeout-review.md
[R213]: ../SPRINT_213/RETROSPECTIVE.md
[D213]: ../SPRINT_213/artifacts/day5-product-policy-decision.md
[V213]: ../SPRINT_213/artifacts/day13-integrated-validation.md
[C213]: ../SPRINT_213/artifacts/day14-closeout-review.md
[R214]: ../SPRINT_214/RETROSPECTIVE.md
[D214]: ../SPRINT_214/artifacts/day2-abi-release-scope-decision.md
[V214]: ../SPRINT_214/artifacts/day13-integrated-validation.md
[C214]: ../SPRINT_214/artifacts/day14-closeout-review.md
[R215]: ../SPRINT_215/RETROSPECTIVE.md
[D215]: ../SPRINT_215/artifacts/day10-selected-pilot-outcome.md
[V215]: ../SPRINT_215/artifacts/day13-integrated-validation.md
[C215]: ../SPRINT_215/artifacts/day14-closeout-review.md
