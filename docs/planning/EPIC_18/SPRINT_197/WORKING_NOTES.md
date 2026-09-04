# Sprint 197 Working Notes: Epic 18 Final Validation, Claim Calibration & Closeout

**Sprint:** 197
**Status:** Day 14 complete
**Goal:** Reconcile Epic 18 outcomes, run final validation, calibrate claims,
publish the retrospective and residual queue, and decide whether any stronger
support claims are earned.

## Numbering Note

The requested Sprint 197 day plan intentionally uses the final-validation scope
from the cited Epic 18 project-plan lines. In the merged
`docs/planning/EPIC_18/PROJECT_PLAN.md`, that final-validation section is
labeled Sprint 206, while Sprint 197 is the baseline, residual-ledger, and
closure-selection sprint. These working notes preserve the requested
`SPRINT_197` path but track the final-validation item numbers 206.1 through
206.6 to avoid losing traceability to the cited project-plan section.

## Sprint Item Checklist

| Item | Name | Day 1 closeout surface | Status |
| --- | --- | --- | --- |
| 206.1 | Evidence Reconciliation | Epic 18 review, todo, project plan, future Sprint 197-205 artifacts, validation records, PR review comments, and residual queues | Day 3 conflict review complete |
| 206.2 | Claim Recalibration | README, INSTALL, maintainer guide, benchmark docs, API docs, corpus docs, generated-report docs, and planning docs | Public and maintainer/API recalibration no-ops recorded |
| 206.3 | Project Plan Status | `docs/planning/EPIC_18/PROJECT_PLAN.md` item and sprint status notes with evidence links | Interim status snapshot and item ledger recorded |
| 206.4 | Integrated Validation | Focused gates by changed surface plus full C gates when `.c` or `.h` files change | Focused and required full-gate decisions recorded |
| 206.5 | Epic Retrospective | `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` outcomes, evidence, non-claims, residuals, and state-of-the-art assessment | Day 14 closeout draft complete |
| 206.6 | Residual Queue | Prioritized next-epic residual queue with exact closure targets and long-horizon deferrals | Residual queue and final closeout review complete |

## Day 1: Closeout Intake

### Scope Trace

| Final-validation item | Day 1 interpretation | Evidence to collect before edits |
| --- | --- | --- |
| 206.1 Evidence Reconciliation | Build a consolidated ledger of what Epic 18 planned to close and what evidence will be needed once sprint artifacts exist. | Epic 18 review, todo, project plan, future Sprint 197-205 plans/notes/retrospectives, CI runs, PR reviews, generated reports, and residual queues. |
| 206.2 Claim Recalibration | Identify public, maintainer, API, benchmark, corpus, and planning claim surfaces before editing wording. | Current user docs, maintainer docs, support/readiness wording, benchmark methodology docs, API docs, package docs, Windows docs, and report-index docs. |
| 206.3 Project Plan Status | Prepare to annotate project-plan items with final status and evidence links. | Epic 18 project plan plus future sprint closeout artifacts proving each status. |
| 206.4 Integrated Validation | Map required gates to the surfaces likely to change during final closeout. | Makefile targets, Python validation scripts, C/header gates, docs/API guards, package guards, Windows guards, report freshness guards, and performance guards. |
| 206.5 Epic Retrospective | Identify retrospective structure and evidence sections needed for final closeout. | Existing Epic 17 retrospective plus Sprint 197-205 artifacts once available. |
| 206.6 Residual Queue | Start residual categories without declaring closure prematurely. | Epic 18 review/todo gaps, Epic 17 residual queue, project-plan non-goals, and future sprint deferrals. |

### Evidence Source Inventory

| Source | Topic | Primary files | Day 1 status signal |
| --- | --- | --- | --- |
| Epic 17 closeout | Pre-Epic-18 baseline, completed closures, and residuals | `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md`, `docs/planning/EPIC_17/EPIC_17_RESIDUAL_QUEUE.md` | Merged on `master`; primary historical baseline. |
| Epic 18 review | Current gap assessment and state-of-the-art readiness review | `docs/planning/EPIC_18/reviews/review-codex-2026-09-04.md` | Present; source for gap categories and claim limits. |
| Epic 18 todo | Step-by-step gap closure plan | `docs/planning/EPIC_18/reviews/todo-codex-2026-09-04.md` | Present; source for closure sequencing and evidence expectations. |
| Epic 18 project plan | Ten sprint plan for Sprints 197-206 | `docs/planning/EPIC_18/PROJECT_PLAN.md` | Present; source for item estimates, prerequisites, and deliverables. |
| Sprint 197 plan | Requested day-by-day final-validation plan | `docs/planning/EPIC_18/SPRINT_197/PLAN.md` | Present on this branch; maps the cited final-validation scope into daily work. |
| Sprint 198-205 artifacts | Package, Windows, benchmark, comparison, API, reliability, review-surface, and adoption evidence | Future `docs/planning/EPIC_18/SPRINT_198` through `SPRINT_205` directories | Missing at Day 1 because those sprints have not been executed on this branch. |
| Generated reports | Comparison, benchmark, oracle, and selected freshness evidence | `tests/corpus/`, report-index docs, generated report artifacts when present | Inventory only; final reconciliation must separate checked-in evidence from generated local output. |
| CI and PR review records | Hosted platform proof and review-comment resolutions | GitHub PRs, Actions runs, review comments, branch commits | Inventory only; later days must link specific run URLs or comment IDs when used as evidence. |

### Future Sprint Evidence Categories

| Sprint | Expected topic | Evidence needed for final closeout |
| --- | --- | --- |
| 197 | Baseline, residual ledger, closure selection, and acceptance gates | Gap ledger, deduplicated residual queue, closure selection record, acceptance-gate map, and claim-surface map. |
| 198 | Homebrew license metadata and formula proof closure | Approved license metadata decision, formula proof output, package guards, install docs, and retained non-claims. |
| 199 | Selected Windows Cholesky freshness promotion | Hosted Windows evidence, manifest decision, path-normalization/freshness tests, workflow guard updates, and docs calibration. |
| 200 | Additional allocation-failure owner proof | Selected owner invariant record, deterministic failure-injection tests, focused reliability gate, cleanup/retry evidence, and docs. |
| 201 | Additional review-surface reduction | Candidate ranking, selected cluster invariant record, extraction diff, registration guard, focused regression, and validation record. |
| 202 | Hosted selected benchmark freshness on one additional platform | Platform/row decision, methodology metadata, hosted workflow evidence, freshness tests, and non-portable performance wording. |
| 203 | Windows QR incompatible comparison promotion | MSVC/CMake proof or explicit re-deferral, generator fixes, manifest decision, normalizer/workflow tests, and selected-claim docs. |
| 204 | Generated API publication decision | Product decision, publication or local-only guard implementation, freshness/link checks, routing docs, and claim-boundary guard. |
| 205 | Support matrix and adoption quick-reference consolidation | Public doc audit, quick-reference table, support truth consolidation, diagnostics vocabulary, claim guards, and validation record. |

### Claim Surface Inventory

| Surface | Why it needs review | Day 1 disposition |
| --- | --- | --- |
| `README.md` | Top-level user-facing capability, support, package, quality, platform, and performance claims. | Inventory only; defer edits until evidence reconciliation. |
| `INSTALL.md` | Active install/support/readiness matrix and package-manager guidance. | Inventory only; compare later against package and platform evidence. |
| `docs/maintainer_guide.md` | Maintainer ownership for package, Windows, freshness, validation, support tiers, and non-claim semantics. | Inventory only; compare later against guard expectations and residuals. |
| `benchmarks/README.md` | Performance methodology, selected evidence scope, benchmark freshness, and non-portable claim boundaries. | Inventory only; compare later against Sprint 202 evidence and retained Sprint 192 baseline. |
| `docs/api_reference.md` | API documentation coverage, generated API routing, and publication semantics. | Inventory only; compare later against Sprint 204 decision. |
| `include/*.h` | Public API contracts and public-header documentation. | Inventory only; any header edit triggers full C quality gates. |
| `docs/solver_selection.md` | User guidance for choosing solvers and interpreting capability/support boundaries. | Inventory only; compare later against Sprint 205 adoption updates. |
| `docs/cookbook.md` and `docs/tutorial.md` | User workflow examples and diagnostics interpretation. | Inventory only; compare later against support matrix and quick-reference consolidation. |
| `tests/corpus/README.md` and `tests/corpus/schemas/report_index_fields.md` | Report-index, selected target, comparison, and freshness semantics. | Inventory only; compare later against Sprints 199, 202, and 203. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Sprint status, item disposition, evidence links, and final closeout consistency. | Defer edits until outcome ledger exists. |
| `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` | Final Epic 18 outcomes, validation evidence, residuals, and state-of-the-art assessment. | To be created late in the sprint. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Prioritized next-epic residuals and long-horizon deferrals. | To be created after residual triage. |

### Initial Validation Gate Map

| Gate family | Candidate commands or evidence | Applies when |
| --- | --- | --- |
| Repository hygiene | `git diff --check`, `git status --short`, source-list checks | Baseline and before final closeout. |
| C/header quality | `make format`, `make lint`, `make test`, `make quality-review-compile` | Any `.c` or `.h` source/header change. |
| CMake/reviewed build | `make quality-review-cmake-compile`, `make quality-review-cmake`, selected hosted CI evidence | CMake, install/export, package, or downstream consumer changes. |
| Docs/API | `make docs-check`, API docs coverage/freshness targets, Markdown/link checks if added | User docs, API docs, generated docs, or public-header wording changes. |
| Windows/PowerShell | `make windows-powershell-validate`, `make windows-powershell-guard`, hosted Windows workflow evidence | Windows claim, workflow, PowerShell, or selected freshness semantics change. |
| Report freshness | Report-index oracle/comparison freshness targets and selected target manifest validation | Report index, comparison target, manifest, or freshness wording changes. |
| Performance evidence | Benchmark freshness tests, selected benchmark report freshness, and performance sentinel guards | Benchmark methodology, selected performance lane, or report claims change. |
| Reliability proof | Allocation-failure focused gates and stale-output/retry tests | Reliability harness, allocation-failure proof, or cleanup/retry semantics change. |
| Review-surface reduction | Source-list guards and helper registration guards | Helper movement, source-list ownership, or review-surface claims change. |
| Package proof | Package/static install guards and provider-specific proof scripts | Package-manager or install support claims change. |

### Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Requested Sprint 197 path uses the final-validation scope that the project plan labels Sprint 206. | Later artifacts may appear out of sequence or reference unavailable Sprint 198-205 evidence. | Keep the numbering note explicit and treat missing future sprint artifacts as a Day 1 risk, not as completed evidence. |
| Final docs overclaim state-of-the-art status. | Public claims could exceed Epic 18 evidence. | Tie every stronger claim to completed evidence and keep non-claims explicit. |
| Residuals are hidden by broad "complete" wording. | Future owners lose exact closure targets. | Preserve residual tables with owner conditions and required evidence. |
| Local validation cannot reproduce hosted Windows/macOS evidence. | Final validation record could conflate local and hosted proof. | Separate local command results from hosted CI evidence and record unavailable environments as residuals. |
| Claim wording duplicates across docs. | Docs drift after closeout. | Assign each claim type a primary source and make secondary docs point to it where safe. |
| C/header edits unexpectedly expand validation scope. | Closeout may require the full C quality gate. | Avoid code/header edits unless required; if touched, run `make format && make lint && make test`. |
| Package-manager residual is represented as support instead of readiness evidence. | Users may infer unsupported distribution guarantees. | Keep package-manager claims guarded until provider-specific proof and license metadata close the residual. |
| Windows/PowerShell evidence semantics are too broad. | Hosted validation ownership may be mistaken for general Windows parity. | Preserve selected-lane and hosted-evidence boundaries. |
| Generated API publication wording drifts. | Users may expect hosted generated docs or committed generated HTML without a policy decision. | Route API publication claims through the Sprint 204 decision record. |

### Day 2 Reconciliation Questions

1. Which Epic 18 artifacts exist now, and which Sprint 198-205 artifacts are
   intentionally absent because those sprints have not run?
2. Which Epic 17 residuals remain inputs to Epic 18, and which have already
   been selected in the Epic 18 project plan?
3. Which claim surfaces currently mention package-manager support, broad
   Windows parity, portable performance, ABI/shared-library support, release
   readiness, or state-of-the-art status?
4. Which validation gates are mandatory for the files that change during this
   sprint?
5. Which hosted evidence will require PR/CI links instead of local command
   output?

### Day 1 Validation

- Reviewed Sprint 197 Day 1 plan requirements.
- Reviewed the cited Epic 18 final-validation project-plan section.
- Reviewed the full Epic 18 project plan and current Epic 18 file inventory.
- Reviewed recent Epic 17 working-note format for consistency.
- Created the Day 1 artifact
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day1-closeout-intake.md`.
- No production code, public header, generated API output, README, INSTALL, or
  maintainer-guide claims were edited on Day 1.

## Day 2: Outcome Reconciliation Ledger

### Method

Day 2 reconciles the evidence that exists at the start of this sprint with the
Epic 18 project-plan intent. Because this branch is the first Sprint 197 branch
created after Epic 18 planning merged, Sprint 198 through Sprint 205 artifacts
do not exist yet. Their rows below are therefore classified as planned,
future-missing evidence rather than complete, failed, or deferred outcomes.

### Current Epic 18 Artifact Ledger

| Artifact | Status | Evidence role | Day 2 disposition |
| --- | --- | --- | --- |
| `docs/planning/EPIC_18/reviews/review-codex-2026-09-04.md` | Present | Current project review and gap assessment. | Available evidence input. |
| `docs/planning/EPIC_18/reviews/todo-codex-2026-09-04.md` | Present | Step-by-step gap-closure plan. | Available evidence input. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Present | Sprint 197-206 scope, estimates, prerequisites, and deliverables. | Available planning source. |
| `docs/planning/EPIC_18/SPRINT_197/PLAN.md` | Present on this branch | Requested day-by-day final-validation plan from the cited scope. | Available sprint execution source. |
| `docs/planning/EPIC_18/SPRINT_197/WORKING_NOTES.md` | Present on this branch | Running evidence ledger and day-by-day execution notes. | Updated through Day 2. |
| `docs/planning/EPIC_18/SPRINT_197/artifacts/day1-closeout-intake.md` | Present on this branch | Day 1 scope trace, source map, claim surfaces, gates, and risks. | Available evidence anchor. |
| `docs/planning/EPIC_18/SPRINT_197/artifacts/day2-outcome-ledger.md` | Present on this branch | Day 2 reconciled artifact and planned-outcome ledger. | Created for item 206.1. |
| `docs/planning/EPIC_18/SPRINT_198` through `SPRINT_205` | Missing | Future sprint implementation evidence. | Not available; cannot support final claims yet. |
| `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` | Missing | Final Epic 18 outcome and state-of-the-art assessment. | To be created late in this sprint. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Missing | Prioritized next-epic residual queue. | To be created after residual triage. |

### Sprint 197-205 Planned Outcome Ledger

| Sprint | Topic | Current status on Day 2 | Evidence anchor | Claim boundary |
| --- | --- | --- | --- | --- |
| 197 | Baseline, residual ledger, closure selection, and acceptance gates | Planned in `PROJECT_PLAN.md`; this branch is executing a final-validation plan under the requested `SPRINT_197` path. | `docs/planning/EPIC_18/PROJECT_PLAN.md`; `docs/planning/EPIC_18/SPRINT_197/PLAN.md`; Day 1 and Day 2 artifacts. | Treat current branch artifacts as planning/closeout scaffolding only; do not claim implementation closures. |
| 198 | Homebrew license metadata and formula proof closure | Future-missing evidence. | Project-plan Sprint 198 section; Epic 17 residual `E17-RQ-001`; Epic 18 todo Phase 2. | No Homebrew, Homebrew/core, bottle, Linuxbrew, public tap, or broad package-manager support claim is earned. |
| 199 | Selected Windows Cholesky freshness promotion | Future-missing evidence. | Project-plan Sprint 199 section; Epic 17 residual `E17-RQ-005`; Epic 18 todo Phase 3. | No promoted Windows selected Cholesky freshness claim is earned until hosted evidence and manifest metadata agree. |
| 200 | Additional allocation-failure owner proof | Future-missing evidence. | Project-plan Sprint 200 section; Epic 17 residual `E17-RQ-022`; Epic 18 todo Phase 4. | No broad reliability or allocation-failure coverage claim is earned; prior proof remains selected-owner scoped. |
| 201 | Additional review-surface reduction | Future-missing evidence. | Project-plan Sprint 201 section; Epic 17 residual `E17-RQ-016`; Epic 18 todo Phase 5. | No broad test-suite or QR/direct-solver maintainability cleanup claim is earned. |
| 202 | Hosted selected benchmark freshness on one additional platform | Future-missing evidence. | Project-plan Sprint 202 section; Epic 17 residual `E17-RQ-013`; Epic 18 todo Phase 6. | No portable performance, hosted multi-platform benchmark freshness, or timing-threshold claim is earned. |
| 203 | Windows QR incompatible comparison promotion | Future-missing evidence. | Project-plan Sprint 203 section; Epic 17 residual `E17-RQ-006`; Epic 18 todo Phase 7. | No Windows QR incompatible freshness, broad QR parity, or broad Windows comparison claim is earned. |
| 204 | Generated API publication decision | Future-missing evidence. | Project-plan Sprint 204 section; Epic 18 todo Phase 8; Epic 17 residual `E17-RQ-025`. | No hosted generated API publication or committed generated HTML claim is earned. |
| 205 | Support matrix and adoption quick-reference consolidation | Future-missing evidence. | Project-plan Sprint 205 section; Epic 18 todo Phase 9 inputs. | No simplified support matrix or quick-reference outcome can be cited until implemented and guarded. |

### Baseline Residual Reconciliation

| Residual | Epic 18 destination | Current disposition | Evidence required before closure |
| --- | --- | --- | --- |
| E17-RQ-001 Package-manager/Homebrew support blocker | Sprint 198 | Selected near-term candidate, not closed. | Approved license metadata, exact formula license identifier, passing Homebrew proof, package guards, install checks, and calibrated docs. |
| E17-RQ-005 Selected Cholesky Windows freshness promotion | Sprint 199 | Selected near-term candidate, not closed. | Hosted Windows selected comparison pass, artifact inspection, selected manifest promotion, normalizer/workflow tests, PowerShell guards, and calibrated docs. |
| E17-RQ-022 Additional allocation-failure owner | Sprint 200 | Selected near-term candidate, not closed. | Chosen owner, invariant record, deterministic failure/retry tests, focused gate, registration guard, docs, and full C gate if code changes. |
| E17-RQ-016 Additional QR review-surface cluster | Sprint 201 | Selected near-term candidate, not closed. | Candidate ranking, behavior-preservation invariant, extraction, helper/source registration guard, focused regressions, and C gates if applicable. |
| E17-RQ-013 Windows/macOS selected benchmark freshness | Sprint 202 | Selected near-term candidate, not closed. | One hosted platform/row decision, methodology metadata, workflow lane, freshness tests, hosted evidence review, and non-portable docs. |
| E17-RQ-006 Windows QR incompatible freshness | Sprint 203 | Selected near-term candidate, not closed. | MSVC/CMake probe, generator/normalizer fixes if needed, manifest decision, hosted artifact review, QR focused tests, and selected-claim docs. |
| E17-RQ-025 Hosted generated API publication | Sprint 204 | Selected decision candidate, not closed. | Product decision plus publication implementation or strengthened local-only guard, freshness/link checks, and routing docs. |
| E17-RQ-020 Markdown link-check target | Possible Sprint 204 or 205 support work | Not selected as a standalone sprint closure. | Dedicated link-check target, fixtures, exclusions, failure semantics, and docs guard integration. |
| E17-RQ-002 Shared-library packaging and dynamic ABI support | Long-horizon residual | Explicitly outside Epic 18 unless later scope changes. | ABI policy, shared build/install support, versioning rules, compatibility tests, and release governance. |
| E17-RQ-003 Broad Windows parity | Long-horizon residual | Explicit non-goal. | Broad Windows solver/test/packaging/report matrix and hosted evidence across claimed surfaces. |
| E17-RQ-012 Portable performance evidence | Long-horizon residual | Explicit non-goal. | Multi-platform methodology, thresholds or interpretation policy, variance data, hardware/compiler metadata, and release-quality artifacts. |
| E17-RQ-026 Unqualified state-of-the-art status | Long-horizon residual | Explicit non-goal. | Broad comparative correctness, performance, robustness, package, ABI, release, and ecosystem evidence. |

### Evidence-Link Table By Topic

| Topic | Current evidence | Missing future evidence | Claim boundary for later days |
| --- | --- | --- | --- |
| Package/Homebrew | Epic 17 residual queue and Epic 18 plan/todo identify the blocker. | Sprint 198 artifacts and approved license metadata. | Keep package-manager support unclaimed until proof exits successfully with approved metadata. |
| Windows Cholesky freshness | Epic 17 residual narrowed the selected Cholesky lane; Epic 18 selects it for promotion or re-deferral. | Sprint 199 hosted evidence, manifest decision, workflow/normalizer tests. | Distinguish guarded workflow existence from promoted freshness evidence. |
| Allocation-failure reliability | Sprint 195 baseline proof exists for selected symbolic Cholesky behavior. | Sprint 200 selected additional owner proof. | Do not turn one selected proof into broad allocation-failure or OOM coverage. |
| Review-surface maintainability | Sprint 193 baseline reduced one QR helper cluster. | Sprint 201 additional selected cluster evidence. | Claim only selected review-surface reductions, not broad test-suite simplification. |
| Benchmark evidence | Sprint 192 baseline provides one Linux selected hosted benchmark lane. | Sprint 202 one additional hosted platform/row. | Keep performance claims methodology-bound, threshold-free, and non-portable. |
| QR external comparison | Sprint 191 baseline provides local-only QR incompatible comparison evidence. | Sprint 203 MSVC/CMake and hosted Windows selected evidence. | Do not claim broad QR parity or Windows QR freshness before promotion evidence. |
| Generated API docs | Current `make docs-check` owner exists and generated HTML remains local artifact output. | Sprint 204 publication decision and policy implementation. | Do not imply hosted publication, committed generated HTML, ABI completeness, or release docs. |
| Adoption/support docs | Epic 17 improved support/readiness routing. | Sprint 205 quick-reference and support matrix consolidation. | Avoid simplifying language in ways that widen support or release claims. |

### Completion, Deferral, And Supersession List

| Classification | Items | Day 2 interpretation |
| --- | --- | --- |
| Completed before this sprint | Epic 17 closeout, residual queue, Epic 18 review, Epic 18 todo, Epic 18 project plan, Sprint 197 plan, Day 1 intake artifact. | Available as planning and baseline evidence. |
| Completed during Day 2 | Outcome ledger, residual reconciliation, evidence-link table, completion/deferral/supersession separation. | Supports item 206.1 intake only, not implementation closure. |
| Future-missing | Sprint 198-205 plans, working notes, retrospectives, CI evidence, and implementation artifacts. | Required before final closeout can claim any selected Epic 18 closure. |
| Deferred by explicit non-goal | Broad state-of-the-art status, shared-library/dynamic ABI support, release readiness, broad Windows parity, broad package-manager distribution, broad external-library parity, portable performance. | Must remain non-claims unless future sprint evidence intentionally changes scope. |
| Superseded or mismatched | Requested Sprint 197 final-validation path versus project-plan Sprint 206 label. | Preserve requested path but reference final-validation items as 206.1-206.6 for traceability. |

### Day 2 Acceptance Evidence For 206.1

- Current Epic 18 artifacts have a reconciled status and evidence role.
- Sprint 197-205 planned outcomes are classified by topic, current evidence,
  missing evidence, and claim boundary.
- Epic 17 residuals selected by Epic 18 are mapped to their planned sprint
  destinations and closure evidence.
- Deferred and future-missing items are separated from completed artifacts.
- The Day 2 artifact
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day2-outcome-ledger.md`
  duplicates this reconciliation in PR-reviewable form.

### Day 2 Validation

- Reviewed Sprint 197 Day 2 plan requirements.
- Reviewed the Epic 18 project plan, review, todo, and current Sprint 197
  artifacts.
- Reviewed the Epic 17 residual queue as the baseline residual source.
- No production code, public headers, README, INSTALL, maintainer-guide,
  generated report, or support-claim files were edited on Day 2.

## Day 3: Evidence Conflict and Gap Review

### Method

Day 3 compares the Day 2 ledger against current public documentation, the Epic
18 project plan, the Epic 18 review/todo, Epic 17 residuals, and the absence
of Sprint 198-205 implementation artifacts. The goal is not to edit claims yet;
it is to classify contradictions, stale-risk areas, environment-only evidence,
and exact wording constraints before Day 4 public claim auditing.

### Evidence Conflict Matrix

| Area | Current evidence | Potential conflict | Classification | Day 3 resolution |
| --- | --- | --- | --- | --- |
| Sprint numbering | `SPRINT_197/PLAN.md` uses the requested final-validation scope; `PROJECT_PLAN.md` labels that scope Sprint 206. | Readers could expect Sprint 197 baseline work while this branch performs closeout intake. | Human-review-required planning mismatch. | Preserve the requested path, keep the numbering note, and cite final-validation items as 206.1-206.6. |
| Future sprint evidence | Only Epic 18 review/todo/project plan and Sprint 197 branch artifacts exist. | Sprint 198-205 planned closures could be accidentally treated as completed proof. | Missing future evidence. | Mark all Sprint 198-205 outcomes future-missing until their artifacts, validation, and PR evidence exist. |
| Package/Homebrew | `INSTALL.md` and Epic 17 residuals keep package-manager support unclaimed. | The Sprint 198 plan could be read as current Homebrew support. | Checked-in non-claim plus future-missing proof. | Require approved license metadata and passing proof before any package-manager support promotion. |
| Windows selected Cholesky | Current docs describe a guarded workflow path, not promoted freshness. | Sprint 199 could be read as already promoted Windows selected freshness. | Checked-in non-claim plus hosted-evidence dependency. | Require hosted artifact review, selected metadata, support tier, and claim contract before promotion. |
| Windows/PowerShell | `docs/maintainer_guide.md` separates PowerShell validation ownership from report freshness. | Hosted PowerShell checks could be conflated with Windows report proof. | Hosted/local evidence boundary. | Keep PowerShell parse/ownership evidence separate from CMake, CTest, report generation, and freshness evidence. |
| Benchmark evidence | Current docs describe Linux selected hosted performance freshness as methodology-bound and threshold-free. | Sprint 202 could be read as portable performance or timing-threshold work. | Checked-in non-claim plus future platform evidence. | Require one selected platform/row proof and preserve non-portable, threshold-free wording. |
| QR external comparison | Current docs describe Linux/macOS selected comparison evidence and QR incompatible local-only boundaries. | Sprint 203 could be read as broad QR or Windows parity. | Selected-evidence boundary. | Require MSVC/CMake proof and hosted artifacts for the exact QR incompatible target before any Windows claim. |
| Generated API | Current docs treat generated API HTML as local-only. | Sprint 204 could be read as hosted publication. | Product-decision dependency. | Require explicit publication or local-only policy decision before changing generated API publication wording. |
| Allocation failure | Epic 17 proved selected symbolic Cholesky allocation behavior only. | Sprint 200 could imply broad allocation-failure, OS OOM, or concurrency guarantees. | Selected-owner boundary. | Require a new owner-specific invariant, failure/retry tests, and focused gate before any additional reliability claim. |
| Review-surface reduction | Epic 17 reduced one QR helper cluster. | Sprint 201 could imply broad QR/test-suite maintainability cleanup. | Selected-cluster boundary. | Require one selected cluster, behavior-preservation record, extraction proof, and guard before any new maintainability claim. |
| Support/adoption docs | `INSTALL.md`, solver selection, tutorial, cookbook, README, and maintainer guide have overlapping support wording. | Later simplification could remove needed caveats or create inconsistent support truth. | Stale/duplication risk. | Treat `INSTALL.md#support-readiness-matrix` as the primary public support truth until Sprint 205 changes it. |

### Evidence Type Classification

| Evidence type | Examples | Use in closeout | Required wording constraint |
| --- | --- | --- | --- |
| Checked-in planning evidence | Epic 18 review, todo, project plan, Sprint 197 plan, Day 1/Day 2 artifacts | Supports planning traceability and selected closure intent. | Cannot be cited as implementation proof. |
| Checked-in user documentation | README, INSTALL, solver selection, tutorial, cookbook, benchmark/API docs | Defines current public claims and non-claims. | Must not be widened without implementation and validation evidence. |
| Checked-in maintainer guidance | `docs/maintainer_guide.md`, report schema docs, guard docs | Defines evidence ownership and guard semantics. | Must separate local, hosted, generated, selected, and advisory evidence. |
| Local command evidence | `make docs-check`, `git diff --check`, focused Make/Python gates | Supports local validation for changed surfaces. | Must not be represented as hosted platform proof. |
| Hosted CI evidence | GitHub Actions Windows/macOS/Linux jobs and uploaded artifacts | Supports hosted platform claims only for exact jobs and artifacts. | Must include platform, job, target, artifact, and support-tier boundaries. |
| Generated local artifacts | Doxygen output, benchmark CSVs, comparison reports under ignored build paths | Useful for local inspection and freshness commands. | Must be treated as stale/advisory unless selected freshness gates require current output. |
| Optional dependency evidence | Homebrew, `pwsh`, optional external reference tooling | Supports claims only when available and recorded. | Missing optional tools are environment residuals, not passes. |
| Human-review-required evidence | License metadata decisions, hosted artifact interpretation, publication policy choices | Blocks support promotion until approved. | Must remain explicit deferral or decision-needed wording. |

### Gap and Stale-Risk List

| Gap or stale-risk area | Current status | Risk if ignored | Next action |
| --- | --- | --- | --- |
| Sprint 198-205 artifacts unavailable | Future-missing. | Final closeout could fabricate implementation evidence. | Keep future sprint rows blocked until actual artifacts exist. |
| Package license metadata | Human-review-required. | Homebrew proof could invent unsupported legal metadata. | Require approved standalone root license metadata before Sprint 198 closure. |
| Windows selected Cholesky promotion | Hosted-evidence dependency. | Guarded workflow could be overstated as promoted freshness. | Require hosted artifact inspection and manifest support-tier update. |
| Windows QR incompatible promotion | Hosted MSVC/CMake dependency. | Local QR comparison could be overstated as Windows QR parity. | Require exact target MSVC/CMake evidence and selected artifact review. |
| Additional benchmark platform | Hosted-evidence dependency. | Selected Linux evidence could be overstated as portable performance. | Select one additional platform/row and keep threshold-free interpretation. |
| Generated API publication | Product-decision dependency. | Local Doxygen generation could be mistaken for hosted docs. | Decide publication policy before editing API publication wording. |
| Support matrix duplication | Checked-in docs have many overlapping caveats. | Future edits could diverge or weaken boundaries. | Audit public surfaces on Day 4 and prefer central support truth. |
| Broad state-of-the-art posture | Explicit non-goal. | Planning could become marketing claim drift. | Keep the state-of-the-art claim unearned unless broad evidence exists. |

### Claim-Boundary Notes By Area

| Area | Allowed current wording | Forbidden without future evidence |
| --- | --- | --- |
| Package/Homebrew | Local formula proof material exists; package-manager support remains unclaimed. | Homebrew support, Homebrew/core readiness, bottles, Linuxbrew, public tap, package-manager distribution. |
| Windows | Selected guarded workflow paths and validated MSVC CMake/install slices where documented. | Broad Windows parity, Windows Makefile parity, Windows `pkg-config` parity, broad Windows report freshness. |
| PowerShell | Ownership/parsing validation for selected workflow snippets. | CMake execution, report generation, package proof, or freshness promotion. |
| Benchmarks | Methodology-bound selected hosted/local measurement evidence. | Portable performance, timing thresholds, backend superiority, release benchmark proof, state-of-the-art performance. |
| Comparisons | Fixture-local selected comparison rows for named targets and platforms with evidence. | Broad external-library parity, broad QR/LU/Cholesky correctness, unselected families, cross-platform inheritance. |
| Generated API | Local-only Doxygen generation and coverage against checked-in public headers. | Hosted API publication, committed generated HTML, ABI completeness, release documentation guarantee. |
| Reliability | Selected allocation-failure owner behavior when a focused proof exists. | Broad allocation-failure coverage, OS OOM guarantees, concurrent hook behavior, package/install reliability. |
| Release/state-of-the-art | Experimental/static-first project with selected evidence and explicit non-claims. | Release readiness, ecosystem parity, unqualified state-of-the-art sparse linear algebra status. |

### Day 3 Acceptance Evidence For 206.1

- Contradictions, missing evidence, stale-risk areas, environment-only proof,
  and human-review-required decisions are classified.
- Hosted CI evidence, local command evidence, checked-in docs, generated
  artifacts, and planning artifacts are separated.
- Claim-boundary notes are available for Day 4 public documentation audit.
- The Day 3 artifact
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day3-evidence-conflicts.md`
  duplicates this review in PR-reviewable form.

### Day 3 Validation

- Reviewed Sprint 197 Day 3 plan requirements.
- Compared the Day 2 ledger against Epic 18 project-plan, review, todo, Epic
  17 residual queue, and current claim-sensitive public/maintainer docs.
- No production code, public headers, README, INSTALL, maintainer-guide,
  generated report, or support-claim files were edited on Day 3.

## Day 4: Public Claim Surface Audit

### Method

Day 4 audits public-facing documentation against the Day 2 outcome ledger and
Day 3 evidence boundaries. It does not recalibrate wording yet. The audit
identifies current claim owners, overclaim risk, duplicated caveats, missing
future evidence links, and the edits that should happen only after later
evidence exists.

### Public Claim-Surface Audit

| Surface | Current claim posture | Audit finding | Day 4 disposition |
| --- | --- | --- | --- |
| `README.md` | Short front door with capability inventory, workflow routing, selected evidence boundaries, and support/readiness links. | Claims are mostly calibrated, but the document is broad enough that later Epic 18 promotions must update it carefully. | Keep as the top-level adoption route; future edits should link to `INSTALL.md` for support truth and avoid repeating full caveats. |
| `INSTALL.md` | Primary user-facing support/readiness matrix. | Strong current owner for static-first install, package-manager deferral, Windows CMake validation, selected comparison freshness, selected performance freshness, allocation-failure proof, generated API local-only status, and broad non-claims. | Treat `INSTALL.md#support-readiness-matrix` as public support truth until Sprint 205 changes it. |
| `examples/README.md` | First-run examples and diagnostics handoff. | Useful adoption surface; must not become a support matrix, benchmark claim, or package proof owner. | Later edits should only route users to README/INSTALL/solver-selection for support boundaries. |
| `docs/cookbook.md` | Data-format and workflow recipes. | Good location for first-use route updates, but risky if recipes imply support beyond `INSTALL.md`. | Later edits should keep recipes operational and point support claims back to `INSTALL.md`. |
| `docs/tutorial.md` | Guided API usage. | Likely low overclaim risk, but examples can drift if support wording is duplicated. | Later edits should avoid package/platform/performance assertions except as links. |
| `docs/solver_selection.md` | Problem-shape decision tree and selected comparison/oracle notes. | Contains detailed selected evidence and non-claims; high value but verbose. | Future edits should update named target/platform evidence only when Sprints 199, 203, or 205 produce proof. |
| `benchmarks/README.md` | Benchmark commands, CSV interpretation, and methodology caveats. | Clearly says benchmark rows are local/methodology-bound and not portable performance proof. | Preserve as benchmark interpretation owner; Sprint 202 edits must stay threshold-free unless evidence changes. |
| `docs/api_reference.md` | API reference routing and generated docs source-of-truth. | Needs review after Sprint 204, but current generated API status is local-only. | Do not imply hosted publication or generated HTML availability before a product decision. |
| `tests/corpus/README.md` | Corpus/report evidence interpretation. | Claim-sensitive for selected oracle/comparison rows and freshness. | Later target promotions must update corpus docs together with manifest and normalizer tests. |
| `tests/corpus/schemas/report_index_fields.md` | Report-index field semantics and selected freshness policy. | Strong owner for row-level support tiers, claim scopes, freshness policies, and non-claims. | Treat as schema/policy owner for selected report changes; avoid user-facing support claims without INSTALL alignment. |

### Overclaim and Stale-Claim Table

| Claim area | Current public state | Overclaim or stale risk | Required evidence before edits |
| --- | --- | --- | --- |
| Package-manager/Homebrew | Package-manager distribution is not claimed; Homebrew proof material is blocker evidence. | A future Sprint 198 plan or formula template could be misread as supported Homebrew installation. | Approved license metadata, exact formula license, passing proof, package guards, install checks, and docs update. |
| Windows selected Cholesky freshness | Windows workflow is guarded but promotion remains blocked on evidence and metadata review. | The existing workflow path could be described as promoted selected freshness too early. | Hosted artifact inspection, selected target rows, manifest support tier, PowerShell guard, and normalizer/freshness tests. |
| Windows QR incompatible comparison | QR incompatible comparison is selected/local-only; Windows promotion is future work. | Solver-selection or corpus docs could inherit Linux/macOS selected evidence onto Windows. | MSVC/CMake proof, hosted Windows artifacts, manifest promotion, workflow tests, and QR focused checks. |
| Benchmark performance | Benchmark docs keep timing local, methodology-bound, threshold-free, and non-portable. | Additional hosted platform freshness could be overstated as portable performance. | Selected platform/row metadata, hosted artifact, freshness tests, and benchmark docs retaining non-portable interpretation. |
| Generated API docs | Generated API HTML is local-only. | Public docs could imply hosted API docs, committed generated HTML, or release documentation. | Sprint 204 product decision and publication/local-only guard implementation. |
| Allocation-failure reliability | Selected allocation-failure proofs are listed with explicit non-claims. | A new owner proof could be treated as broad OOM or all-allocation coverage. | Owner-specific invariant, deterministic failure/retry tests, focused gate, and calibrated reliability docs. |
| Shared library and ABI | Static-first only; shared-library/dynamic ABI support deferred. | CMake/install improvements could imply dynamic ABI or runtime-loader support. | Explicit shared-library design, ABI policy, build/install support, compatibility tests, and docs. |
| State-of-the-art/readiness | Public docs route to selected evidence and retain broad non-claims. | Epic closeout language could read like release or state-of-the-art readiness. | Broad comparative correctness, performance, package, ABI, platform, ecosystem, release, and reliability evidence. |

### Duplicate Caveat and Routing Findings

| Duplicate topic | Current locations | Routing recommendation |
| --- | --- | --- |
| Support/readiness status | README, INSTALL, solver-selection, maintainer guide, corpus docs. | Keep `INSTALL.md#support-readiness-matrix` as the public source; other docs should link there for support interpretation. |
| Performance non-claims | README, INSTALL, solver-selection, benchmark docs, maintainer guide. | Keep benchmark methodology details in `benchmarks/README.md`; public docs should summarize and link. |
| Selected comparison/freshness caveats | README, INSTALL, solver-selection, corpus docs, schema docs, maintainer guide. | Keep row semantics in corpus/schema docs and user support impact in INSTALL. |
| Package-manager deferral | README, INSTALL, Homebrew docs, maintainer guide, planning docs. | Keep operational user guidance in INSTALL and provider-proof details in packaging/maintainer docs. |
| Generated API local-only status | INSTALL, API reference, maintainer guide. | Keep user routing in `docs/api_reference.md` and support status in INSTALL until Sprint 204 decides publication policy. |

### Public Documentation Edit Plan

| Trigger | Public docs to edit | Required Day 4 guardrail |
| --- | --- | --- |
| Sprint 198 closes Homebrew proof | README, INSTALL, packaging Homebrew README, possibly examples route if install path changes. | Promote only the exact provider/support tier earned by proof; preserve unsupported providers and shared-library non-claims. |
| Sprint 199 promotes selected Windows Cholesky freshness | INSTALL, README support summary, solver-selection selected comparison note, corpus README/schema docs. | Limit claim to exact target, workflow, platform, artifact, support tier, and freshness policy. |
| Sprint 202 adds a hosted benchmark platform | Benchmarks README, INSTALL support matrix, README benchmark note. | Preserve methodology-bound and non-portable wording; do not add timing thresholds unless explicitly proven. |
| Sprint 203 promotes Windows QR incompatible comparison | Solver-selection, corpus docs, INSTALL, README selected comparison summary. | Do not broaden to QR parity, Windows report parity, or external-library parity. |
| Sprint 204 decides generated API publication | `docs/api_reference.md`, INSTALL, README docs route. | Match the chosen hosted/publication/local-only policy exactly. |
| Sprint 205 consolidates support/adoption | README, INSTALL, solver-selection, cookbook, tutorial, examples README. | Keep one public support truth and avoid removing necessary retained non-claims. |

### Evidence-Link Requirements

| Claim type | Minimum evidence link before public promotion |
| --- | --- |
| Package-manager support | License metadata decision, proof script output, package guards, install checks, and provider docs. |
| Windows selected freshness | Hosted workflow run, uploaded artifact identity, selected manifest row, normalizer/freshness test, and PowerShell guard. |
| Benchmark platform freshness | Hosted workflow run, canonical bundle, methodology metadata, freshness checker, and benchmark docs. |
| Comparison promotion | Generator command, target key, report rows, selected manifest metadata, normalizer tests, and hosted/local scope. |
| Generated API publication | Product decision, publication workflow or local-only guard, docs freshness, and link/coverage evidence. |
| Reliability proof | Owner invariant, deterministic failure injection path, retry/cleanup tests, focused gate, and full C gate when code changes. |
| Release or state-of-the-art readiness | Broad evidence package across correctness, performance, platform, package, ABI, release policy, and external-library parity. |

### Day 4 Acceptance Evidence For 206.2

- Public claim surfaces are identified and classified by claim owner,
  overclaim risk, and routing expectation.
- Overclaim and stale-claim risks are separated from current accurate
  non-claims.
- Duplicate caveats have a consolidation plan that preserves
  `INSTALL.md#support-readiness-matrix` as the current support owner.
- Future public documentation edits have evidence-link requirements.
- The Day 4 artifact
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day4-public-claim-audit.md`
  duplicates this audit in PR-reviewable form.

### Day 4 Validation

- Reviewed Sprint 197 Day 4 plan requirements.
- Searched public and maintainer docs for package, Windows, PowerShell,
  performance, benchmark, comparison, freshness, support, ABI, release,
  generated API, and state-of-the-art wording.
- Reviewed high-signal README, INSTALL, benchmark, solver-selection, corpus,
  and API documentation sections.
- No production code, public headers, public claim docs, generated reports, or
  support matrix wording were edited on Day 4.

## Day 5: Maintainer, API, and Planning Claim Audit

### Method

Day 5 audits maintainer-facing, API-reference, corpus/report-schema, and
planning-adjacent surfaces. It compares those sources against the Day 2 ledger
and Day 3 evidence-type rules. The audit prepares future item 206.2 claim
recalibration and item 206.3 project-plan status edits, but does not change
maintainer or API docs yet.

### Maintainer/API Claim Audit

| Surface | Current owner role | Audit finding | Later edit trigger |
| --- | --- | --- | --- |
| `docs/maintainer_guide.md` selected comparison freshness section | Maintainer owner for selected comparison generation, manifest authority, hosted lane interpretation, Windows Cholesky workflow boundary, and selected non-claims. | Current wording correctly separates Linux/macOS hosted selected comparison freshness from the bounded Windows Cholesky workflow path and keeps Windows promotion blocked on evidence and metadata review. | Sprint 199 or 203 changes selected target metadata, Windows workflow scope, artifact names, or promoted support tier. |
| `docs/maintainer_guide.md` Windows PowerShell validation section | Maintainer owner for PowerShell snippet validation and hosted `--require-pwsh` interpretation. | Current wording correctly states PowerShell validation does not run CMake, CTest, report generators, uploads, package proofs, or freshness commands. | Any Windows workflow, PowerShell parser, selected report artifact, or Windows support interpretation change. |
| `docs/maintainer_guide.md` generated API reference section | Maintainer owner for generated API local-only policy and freshness semantics. | Current wording still references prior Sprint 179/Sprint 186 decisions but remains accurate: generated HTML is ignored, local-only, not hosted, not artifact-published, and not release evidence. | Sprint 204 product decision changes publication policy, artifact retention, Pages hosting, local-only guard, or Doxygen input ownership. |
| `docs/api_reference.md` | Source-controlled API reference index and generated HTML routing. | Current wording matches maintainer guidance: public headers are the source of truth and generated HTML is local-only. | Sprint 204 publication decision or any public-header documentation/source-of-truth change. |
| `tests/corpus/schemas/report_index_fields.md` | Schema owner for selected target identity, support tier, freshness policy, workflow scope, claim scope, non-claims, and manifest authority. | Current wording correctly warns not to add fake deferral rows or widen `workflow_platforms` without reviewed metadata. | Any selected target row promotion, expected row count/ID change, artifact path change, support-tier change, or workflow platform promotion. |
| `tests/corpus/README.md` | Corpus evidence interpretation and report-index handoff. | Must remain synchronized with schema and manifest when selected freshness targets are promoted. | Sprint 199, 202, or 203 report/freshness changes. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Planning source for sprint item statuses, prerequisites, deliverables, and total estimates. | Current file is planning-only and should not be edited to final status until implementation evidence exists. | Day 8 project-plan status update after outcome evidence is available. |
| `docs/planning/EPIC_18/SPRINT_197/WORKING_NOTES.md` | Branch-local running ledger for requested closeout work. | Accurate place to record numbering mismatch, missing future artifacts, and validation decisions. | Every remaining day should update status and evidence rather than relying on memory. |

### Planning Status Edit Inventory

| Project-plan area | Status edit needed later | Evidence required before edit |
| --- | --- | --- |
| Sprint 197 baseline and closure selection | Mark as complete, narrowed, or superseded only if this branch creates enough baseline/ledger artifacts to satisfy the planned baseline scope. | Sprint 197 artifacts, gap ledger, residual selection, acceptance-gate map, claim-surface map, validation notes. |
| Sprint 198 Homebrew proof | Mark complete only with provider metadata and proof output; otherwise residualize explicitly. | License decision, formula metadata, proof script output, package guards, docs checks. |
| Sprint 199 Windows Cholesky freshness | Mark promoted only if hosted evidence and manifest metadata agree; otherwise re-defer. | Hosted Windows run, artifact inspection, selected manifest row, normalizer tests, PowerShell guard. |
| Sprint 200 allocation-failure owner | Mark complete only for the exact selected owner. | Invariant record, deterministic failure/retry tests, focused gate, registration guard, full C gate if code changed. |
| Sprint 201 review-surface reduction | Mark complete only for one selected cluster. | Candidate ranking, extraction diff, behavior-preservation notes, focused tests, guard. |
| Sprint 202 benchmark platform freshness | Mark complete only for one exact platform/row. | Hosted workflow, canonical bundle, methodology metadata, freshness test, benchmark docs. |
| Sprint 203 Windows QR incompatible comparison | Mark promoted only if MSVC/CMake evidence proves the exact target. | Windows generation run, selected rows, manifest promotion, normalizer/workflow tests, docs. |
| Sprint 204 generated API publication decision | Mark complete once publication/local-only decision is implemented and guarded. | Decision record, workflow/guard implementation, docs/API freshness and link checks. |
| Sprint 205 support/adoption consolidation | Mark complete only after public docs and guards agree. | Public doc audit, quick reference, support truth update, diagnostics vocabulary, claim guards. |
| Sprint 206 final closeout | Mark complete only after final validation, retrospective, and residual queue exist. | Integrated validation log, retrospective, residual queue, final claim decision table. |

### Generated API Publication Boundary Notes

| Boundary | Current state | Required future handling |
| --- | --- | --- |
| Source-controlled API truth | `docs/api_reference.md` and checked-in public headers under `include/`. | Preserve unless Sprint 204 changes the API routing policy. |
| Generated HTML location | `docs/api/html/` generated locally by Doxygen and ignored. | Keep local-only unless a publication decision adds hosted/artifact/committed output. |
| Freshness proof | `make docs-check` and `make api-docs-freshness` validate local generated output and staging policy. | Add or update checks if publication policy changes. |
| Installed generated header | `sparse_version.h` remains generated from `VERSION` and `include/sparse_version.h.in`; it is not an expected Doxygen page. | Keep install validation separate from Doxygen coverage. |
| Non-claims | No hosted API publication, source-controlled generated HTML, artifact-published generated HTML, dynamic ABI, package-manager distribution, or completeness beyond Doxyfile inputs. | Preserve unless future evidence explicitly closes the boundary. |

### Validation-Owner Update Plan

| Change area | Maintainer/API owner files | Validation owners |
| --- | --- | --- |
| Package/Homebrew support | `docs/maintainer_guide.md`, `INSTALL.md`, packaging Homebrew docs, selected package guards | Homebrew proof script, package-manager/static deferral guards, install tests, docs checks. |
| Windows selected freshness | `docs/maintainer_guide.md`, `INSTALL.md`, corpus README, schema docs, selected target manifest | Windows PowerShell guard, selected workflow tests, normalizer tests, report freshness command, hosted Windows run. |
| Benchmark platform freshness | `docs/maintainer_guide.md`, `benchmarks/README.md`, `INSTALL.md`, selected target manifest | Benchmark freshness tests, canonical report freshness, hosted platform run, docs checks. |
| Generated API publication | `docs/api_reference.md`, `docs/maintainer_guide.md`, `INSTALL.md`, Doxygen policy files | `make docs-check`, `make api-docs-freshness`, publication/link checks if added. |
| Reliability proof | `docs/maintainer_guide.md`, README/INSTALL reliability rows, focused test registration docs | Focused allocation-failure gate, CTest label guard, `make format && make lint && make test` when C changes. |
| Review-surface reduction | `docs/maintainer_guide.md`, helper guard docs, source-list ownership | Source-list check, helper guard, focused tests, CMake parity if registration changes, full C gate when needed. |

### Day 5 Acceptance Evidence For 206.2 and 206.3

- Maintainer, API, corpus/schema, and planning owner surfaces are audited.
- Generated API publication remains explicitly local-only until Sprint 204
  decides otherwise.
- Project-plan status edits are inventoried with evidence requirements.
- Validation-owner updates are mapped for package, Windows, benchmark,
  generated API, reliability, and review-surface changes.
- The Day 5 artifact
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day5-maintainer-api-claim-audit.md`
  duplicates this audit in PR-reviewable form.

### Day 5 Validation

- Reviewed Sprint 197 Day 5 plan requirements.
- Reviewed maintainer-guide selected comparison, Windows PowerShell, generated
  API, benchmark, and report-index sections.
- Reviewed `docs/api_reference.md` generated HTML and claim-boundary sections.
- Reviewed report-index field schema policy around selected targets,
  workflow platforms, support tiers, freshness, and non-claims.
- No production code, public headers, maintainer docs, API docs, report schema
  docs, public claim docs, generated reports, or project-plan status rows were
  edited on Day 5.

## Day 6: Public Documentation Recalibration

### Method

Day 6 applies the Day 4 public claim audit to the current evidence ledger. It
checks whether public documentation should be edited now or whether the current
public wording already matches earned evidence. Because the Day 2 ledger shows
that the implementation evidence expected from Sprints 198 through 205 is not
available on this branch, Day 6 records a no-promotion decision rather than
turning future plan intent into public support language.

### Public Documentation Decision

No public documentation edits were made on Day 6.

The current public docs already preserve the necessary support boundaries:

- `INSTALL.md#support-readiness-matrix` remains the public support/readiness
  source of truth.
- `README.md` remains the front door and routes detailed install, support,
  benchmark, generated API, corpus, and claim-boundary questions to owner docs.
- `benchmarks/README.md` keeps benchmark evidence methodology-bound and
  platform-specific.
- `docs/api_reference.md` keeps generated API HTML local-only and checked-in
  public headers as the source of truth.
- `docs/solver_selection.md` keeps selected comparison/oracle guidance scoped to
  fixtures rather than package, ABI, platform, performance, or state-of-the-art
  claims.
- `tests/corpus/README.md` and
  `tests/corpus/schemas/report_index_fields.md` retain selected target,
  freshness, manifest, and row-semantics boundaries.

### Retained Public Non-Claims

| Topic | Day 6 retained boundary |
| --- | --- |
| Package managers | No Homebrew/core, Linuxbrew, bottle, tap, vcpkg, Conan, pkgsrc, distro/system package, or broad provider support is claimed. |
| Shared libraries and ABI | Shared-library packaging, dynamic ABI compatibility, runtime-loader behavior, SONAME/install-name/RPATH, DLL/import-library behavior, and static/shared selectors remain deferred. |
| Windows support | Windows remains bounded to the reviewed MSVC/CMake static-first path; Windows Makefile parity, Windows `pkg-config` execution parity, runtime-loader behavior, and broad Windows parity are not claimed. |
| Windows report freshness | The selected Cholesky workflow remains guarded, not promoted, until hosted evidence, selected metadata, support tier, and claim contract are reviewed together. Broad Windows report freshness, Windows selected oracle freshness, Windows selected benchmark freshness, and unselected Windows comparison families remain unclaimed. |
| Benchmarks | Benchmark rows are methodology-bound and do not claim portable timing thresholds, speedup guarantees, release benchmark readiness, or state-of-the-art performance. |
| Generated API HTML | Generated HTML remains local-only and not hosted, artifact-published, source-controlled release evidence, or complete beyond checked-in public headers selected by Doxyfile. |
| Reliability | Allocation-failure proof remains selected and local-only, not broad OOM, platform, generated-tooling, package/install, or state-of-the-art reliability proof. |
| Ecosystem parity | Broad SuiteSparse, PETSc, Trilinos, Eigen, SciPy, NumPy, LAPACK, platform, release, performance, and state-of-the-art parity remain unclaimed. |

### Public Surface Edit Triggers

| Surface | Keep as-is until |
| --- | --- |
| `README.md` | A later sprint promotes an exact support tier, removes a residual, or changes top-level capability routing. |
| `INSTALL.md` | Package, platform, Windows freshness, shared-library, ABI, or support/readiness evidence changes. |
| `benchmarks/README.md` | A reviewed hosted benchmark row/platform is added or methodology metadata changes. |
| `docs/api_reference.md` | The generated API publication/local-only policy changes. |
| `docs/solver_selection.md` | Adoption simplification changes solver guidance, diagnostics vocabulary, or selected evidence routing. |
| `tests/corpus/README.md` | Selected target manifest rows, report freshness scope, or corpus row semantics change. |
| `tests/corpus/schemas/report_index_fields.md` | Manifest fields, workflow platforms, support tiers, artifact paths, or freshness policy change. |

### Day 6 Acceptance Evidence For 206.2

- Public docs were checked against the Day 2 outcome ledger and Day 3 evidence
  classification.
- Existing public non-claims were retained because no new implementation sprint
  evidence exists on this branch.
- The support/readiness routing table remains discoverable from public docs
  without requiring users to read sprint notes.
- The Day 6 artifact
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day6-public-recalibration.md`
  records the public no-promotion decision in PR-reviewable form.

### Day 6 Validation

- Reviewed Sprint 197 Day 6 plan requirements.
- Re-scanned README, INSTALL, benchmark, API, solver-selection, corpus, and
  report-index schema docs for support/readiness, Windows, PowerShell,
  Homebrew, benchmark, generated API, ABI, package, and state-of-the-art
  wording.
- No production code, public headers, public docs, generated reports, report
  schema files, or project-plan status rows were edited on Day 6.

## Day 7: Maintainer and API Documentation Recalibration

### Method

Day 7 applies the Day 5 maintainer/API claim audit to final-validation item
206.2. It reviews maintainer, API, corpus, schema, benchmark-adjacent, generated
report, and planning-owner surfaces for evidence routing and claim boundaries.
Because the implementation evidence expected from Sprints 198 through 205 is
still future-missing on this branch, Day 7 avoids editing owner docs and records
the no-promotion basis explicitly.

### Maintainer/API Documentation Decision

No maintainer, API, corpus, schema, benchmark-adjacent, generated-report, or
project-plan status docs were edited on Day 7.

The current owner docs already identify the relevant gates and artifacts:

- `docs/maintainer_guide.md` owns selected comparison freshness, Windows
  PowerShell validation, generated API local freshness, benchmark freshness,
  package support, reliability proof, review-surface reduction, and residual
  interpretation.
- `docs/api_reference.md` owns checked-in public-header API truth and generated
  HTML local-only policy.
- `tests/corpus/README.md` owns report-family and selected-target evidence
  interpretation.
- `tests/corpus/schemas/report_index_fields.md` owns manifest field semantics,
  support tiers, freshness policy, claim scope, non-claims, workflow artifacts,
  and platform promotion rules.
- `docs/planning/EPIC_18/PROJECT_PLAN.md` remains a planning surface until Day
  8 applies outcome/status annotations.

### Evidence-Owner Routing

| Change area | Owner route retained |
| --- | --- |
| Package/Homebrew support | Maintainer guide, INSTALL, `packaging/homebrew/`, Homebrew proof command, package deferral guards, and install checks. |
| Windows selected Cholesky freshness | Maintainer guide, INSTALL, corpus README, selected target manifest, Windows workflow, PowerShell guard, normalizer target filtering, and hosted Windows evidence. |
| Windows QR incompatible comparison | Selected comparison manifest, report generator, Windows-safe generation path, normalizer/workflow tests, and explicit re-deferral until MSVC/CMake evidence exists. |
| Selected benchmark freshness | Benchmark README, selected manifest benchmark row, canonical benchmark bundle, benchmark freshness checker, hosted platform evidence, and methodology metadata. |
| Generated API publication | API reference, maintainer guide, Doxyfile, local generated HTML ignore policy, `make docs-check`, and `make api-docs-freshness`. |
| Allocation-failure reliability proof | Focused allocation-failure gate, owner-specific tests, registration guards, cleanup/retry invariants, and full C quality gate when implementation changes. |
| Review-surface reduction | Source-list checks, helper registration guards, focused owner tests, extraction invariants, and full C quality gate when source or headers change. |

### Retained Maintainer/API Non-Claims

| Topic | Day 7 retained boundary |
| --- | --- |
| PowerShell validation | Hosted PowerShell parsing is workflow validation ownership only, not selected report freshness or artifact publication evidence. |
| Windows selected Cholesky | The bounded Windows path remains guarded until hosted evidence, manifest metadata, support tier, and claim contract are reviewed together. |
| Windows QR incompatible comparison | The QR target remains outside promoted Windows selected freshness until a future sprint adds MSVC/CMake evidence and reviewed metadata. |
| Generated API HTML | Generated HTML remains local-only and not source-controlled, hosted, artifact-published, or release evidence. |
| Benchmarks | Freshness remains selected and methodology-bound, without raw timing, unselected row, portable speedup, platform parity, or state-of-the-art performance claims. |
| Package and ABI | Package-manager support, shared libraries, dynamic ABI compatibility, runtime-loader behavior, and broad ecosystem parity remain unclaimed. |
| Planning status | Project-plan status cannot promote planned work without artifact, validation, or explicit residual evidence. |

### Day 7 Acceptance Evidence For 206.2

- Maintainer/API docs were checked against the Day 2 outcome ledger, Day 3
  evidence classification, and Day 5 owner-surface audit.
- Existing evidence-owner routing remains sufficient for the available evidence.
- No unsupported support tier, freshness platform, package proof, API
  publication policy, reliability proof, or review-surface closure was promoted.
- The Day 7 artifact
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day7-maintainer-api-recalibration.md`
  records the maintainer/API no-promotion decision in PR-reviewable form.

### Day 7 Validation

- Reviewed Sprint 197 Day 7 plan requirements.
- Re-scanned maintainer guide, API reference, corpus README, report-index schema
  docs, and Epic 18 project plan for selected comparison, PowerShell, generated
  API, Homebrew, support tier, freshness, Windows, publication, local-only, and
  state-of-the-art wording.
- No production code, public headers, maintainer/API docs, corpus/schema docs,
  generated reports, benchmark docs, public docs, or project-plan status rows
  were edited on Day 7.

## Day 8: Project Plan Status Update

### Method

Day 8 addresses final-validation item 206.3 by adding project-plan status notes
without overstating completion. The status pass follows the Epic 16 and Epic 17
closeout pattern of recording evidence-linked dispositions in
`PROJECT_PLAN.md`, but it labels the update as an interim snapshot because the
requested `SPRINT_197` branch is executing the cited Sprint 206
final-validation scope.

### Project Plan Update

`docs/planning/EPIC_18/PROJECT_PLAN.md` now includes a
"Sprint 197 Day 8 Interim Status Snapshot" after the overview. The snapshot
records:

- the numbering caveat between requested `SPRINT_197` artifacts and the plan's
  Sprint 206 final-validation section;
- Sprint 197 as in progress with a numbering caveat;
- Sprints 198 through 205 as pending future execution because no
  sprint-specific artifact directories, validation records, or PR evidence
  exist on this branch;
- Sprint 206 as partially in progress through the requested `SPRINT_197`
  final-validation artifacts;
- `SPRINT_197/artifacts/day8-project-plan-status.md` as the full item-level
  status ledger.

### Item-Level Disposition Summary

| Scope | Disposition | Evidence |
| --- | --- | --- |
| Sprint 197 items 197.1-197.6 | In progress with numbering caveat | Day 1-8 `SPRINT_197` artifacts provide final-validation intake, reconciliation, audits, recalibration, and interim status evidence, not a completed baseline sprint closeout. |
| Sprint 198 items 198.1-198.6 | Pending future execution | No Homebrew/license metadata, proof execution, guard promotion, docs promotion, or validation artifacts exist on this branch. |
| Sprint 199 items 199.1-199.6 | Pending future execution | No hosted Windows Cholesky evidence review, manifest decision, normalizer hardening, workflow update, docs calibration, or validation artifacts exist on this branch. |
| Sprint 200 items 200.1-200.6 | Pending future execution | No additional allocation-failure owner, invariants, harness, regressions, focused gate, docs, or validation artifacts exist on this branch. |
| Sprint 201 items 201.1-201.6 | Pending future execution | No review-surface candidate ranking, selection, extraction, guard, regression, docs, or validation artifacts exist on this branch. |
| Sprint 202 items 202.1-202.6 | Pending future execution | No additional hosted benchmark platform/row selection, methodology metadata, workflow lane, freshness tests, docs, or validation artifacts exist on this branch. |
| Sprint 203 items 203.1-203.6 | Pending future execution | No Windows QR MSVC/CMake proof, generator fix, manifest decision, normalizer/workflow tests, docs, or validation artifacts exist on this branch. |
| Sprint 204 items 204.1-204.6 | Pending future execution | No generated API publication decision, guard implementation, freshness/link checks, API routing docs, claim guard, or validation artifacts exist on this branch. |
| Sprint 205 items 205.1-205.6 | Pending future execution | No adoption quick reference, support consolidation, diagnostics vocabulary, claim guard, docs, or validation artifacts exist on this branch. |
| Sprint 206 items 206.1-206.3 | Partial final-validation evidence | Day 1-8 artifacts reconcile evidence, audit claims, record no-promotion decisions, and add interim project-plan status notes. |
| Sprint 206 items 206.4-206.6 | Pending final-validation work | Integrated validation, Epic 18 retrospective, and residual queue remain planned for later days. |

### Supersession and Residualization Record

No Sprint 197-205 item was superseded, completed, narrowed, deferred, or
residualized on Day 8. The only status promotion is for item 206.3 on the
requested final-validation path: it moves from drafted status-edit inventory to
partial completion through an interim project-plan snapshot and full item-level
ledger.

Future days may mark work deferred or residualized only after creating the
corresponding residual queue entry or final retrospective evidence. Future
sprints may mark items complete only with their own sprint artifacts,
validation records, and PR evidence.

### Day 8 Acceptance Evidence For 206.3

- `PROJECT_PLAN.md` now points readers to the interim status snapshot and full
  item-level ledger.
- Every Sprint 197-205 planned item has a current Day 8 disposition in
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day8-project-plan-status.md`.
- Partial outcomes are not marked complete.
- No package, Windows, benchmark, comparison, API publication, release, ABI, or
  state-of-the-art claim was promoted.

### Day 8 Validation

- Reviewed Sprint 197 Day 8 plan requirements.
- Reviewed Epic 16 and Epic 17 project-plan closeout status table conventions.
- Reviewed the Epic 18 project plan and Sprint 197 Day 1-7 artifacts.
- Updated `docs/planning/EPIC_18/PROJECT_PLAN.md` and Sprint 197 planning
  artifacts only.
- No production code, public headers, generated reports, public docs,
  maintainer/API docs, benchmark docs, or report schema files were edited on
  Day 8.

## Day 9: Focused Validation Planning

### Method

Day 9 converts the current branch diff and evidence-owner routing into a
validation command matrix for final-validation item 206.4. The day is
planning-only: costly generated-report, benchmark, hosted Windows, and full C
quality gates are prepared for Day 10 or later execution rather than run as
part of the planning pass.

### Changed Surface Validation Owners

| Changed surface | Validation owner |
| --- | --- |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | `git diff --check`; `make docs-check`; manual review that interim statuses do not promote future work. |
| `docs/planning/EPIC_18/SPRINT_197/PLAN.md` | `git diff --check`; `make docs-check`; sprint-plan structure review. |
| `docs/planning/EPIC_18/SPRINT_197/WORKING_NOTES.md` | `git diff --check`; `make docs-check`; item checklist and evidence-ledger review. |
| `docs/planning/EPIC_18/SPRINT_197/artifacts/*.md` | `git diff --check`; `make docs-check`; artifact completeness review. |
| Public docs | No Day 6 edits; no targeted public-doc guard required unless later edits touch README/INSTALL/benchmark/API/corpus surfaces. |
| Maintainer/API/corpus/schema docs | No Day 7 edits; owner guards remain optional confidence unless later edits touch those files. |
| C source and headers | No changes through Day 9; full C gate remains not required unless future `*.c` or `*.h` edits occur. |

### Focused Gate Command List

| Gate family | Command | Day 9 decision |
| --- | --- | --- |
| Patch hygiene | `git diff --check` | Run on Day 9 and after every later edit. |
| Docs/API generation | `make docs-check` | Run on Day 9 and after every later docs edit. |
| Generated API local-only policy | `make api-docs-freshness` | Prepare for Day 10; required if API/publication policy changes. |
| Windows PowerShell ownership | `make windows-powershell-guard` | Prepare for Day 10; required if Windows workflow, selected report wording, or PowerShell snippets change. |
| Package-manager non-claims | `bash scripts/package_manager_deferral_check.sh` | Prepare for Day 10; required if package docs/support wording changes. |
| Shared-library/dynamic ABI non-claims | `bash scripts/static_package_deferral_check.sh` | Prepare for Day 10; required if package/install/ABI wording changes. |
| Selected comparison freshness | `make report-index-comparison-freshness` | Prepare as optional focused final evidence; generated output stays local unless hosted/promoted. |
| Selected oracle freshness | `make report-index-oracle-freshness` | Prepare as optional focused final evidence; generated output stays local unless hosted/promoted. |
| Selected benchmark freshness | `make bench-canonical-report-freshness` | Prepare as optional focused final evidence; benchmark output remains methodology-bound. |
| Source registration | `make source-list-check` | Required only if source registration changes; optional final confidence otherwise. |
| Helper ownership | `make ldlt-csc-helper-guard`; `make qr-external-ref-helper-guard`; `make qr-header-docs-guard` | Required only if the corresponding helper/header surfaces change. |
| Full C quality | `make format && make lint && make test` | Required if any `*.c` or `*.h` file changes; not required for current planning-only diff. |

### Full-Gate Trigger Decision

- Current branch changes through Day 9 are planning Markdown and one
  project-plan Markdown update.
- No `.c` or `.h` files have changed, so `make format && make lint &&
  make test` is not required for Day 9.
- If a later day edits C source or any public/internal header, the full C gate
  becomes mandatory before proceeding.
- If a later day edits generated API publication policy, run
  `make docs-check` and `make api-docs-freshness`.
- If a later day edits Windows workflow, selected target manifest, or Windows
  claim wording, run `make windows-powershell-guard` and treat hosted Windows
  CI as the actual platform evidence owner.

### Environment Residuals

| Evidence | Day 9 residual |
| --- | --- |
| Hosted Windows MSVC/CMake comparison proof | Not locally reproduced; required before Windows freshness promotion. |
| Hosted PowerShell `--require-pwsh` proof | Local missing `pwsh`, if encountered, is an environment residual rather than pass evidence. |
| Homebrew formula proof | Requires approved license metadata and a suitable local Homebrew proof environment. |
| Hosted benchmark platform freshness | Requires hosted runner artifact and methodology metadata; local timing rows are not portable performance evidence. |
| Generated report freshness | Local regeneration commands may create artifacts, but they do not create hosted support claims. |

### Validation Log Template

The Day 10 validation log should record command, surface owner, result, evidence
path or output summary, and follow-up for each focused gate. The template lives
in `docs/planning/EPIC_18/SPRINT_197/artifacts/day9-integrated-validation-matrix.md`.

### Day 9 Acceptance Evidence For 206.4

- Every changed surface has a validation owner.
- Focused gates are listed with execution timing and required triggers.
- Full C gate requirements are explicit and traceable to `*.c`/`*.h` changes.
- Hosted-only and local-environment residuals are recorded before validation
  execution.

### Day 9 Validation

- Reviewed Sprint 197 Day 9 plan requirements.
- Reviewed Makefile validation targets for docs, API freshness, selected report
  freshness, Windows PowerShell ownership, source-list checks, helper guards,
  reviewed quality paths, format, lint, and tests.
- Updated Sprint 197 planning artifacts only.
- No production code, public headers, generated reports, public docs,
  maintainer/API docs, benchmark docs, or report schema files were edited on
  Day 9.

## Day 10: Focused Validation Execution

### Method

Day 10 executes the non-mutating focused gates selected in the Day 9 validation
matrix. The current branch diff remains planning Markdown plus the Epic 18
project-plan interim status snapshot, so generated report freshness, benchmark
freshness, helper-specific C guards, and full C gates are skipped unless their
triggering surfaces change later.

### Executed Gates

| Command | Result | Evidence summary |
| --- | --- | --- |
| `git diff --check` | Pass | No patch whitespace, conflict marker, or diff hygiene issues. |
| `make api-docs-freshness` | Pass | Doxygen generation and API coverage passed; local-only generated API guard confirmed ignored, untracked, unstaged, and non-published generated HTML. |
| `make windows-powershell-guard` | Pass | Workflow wiring, selected Cholesky guarded path, selected manifest references, Windows deferral record, required snippets, and claim boundaries passed. The target includes an intentional negative `--require-pwsh` test; local `pwsh` is unavailable and remains an environment residual, but the overall guard exited successfully. |
| `bash scripts/package_manager_deferral_check.sh` | Pass | Package-manager deferral record, provider recipe absence, selected Homebrew local proof boundary, metadata neutrality, and public non-claims passed. |
| `bash scripts/static_package_deferral_check.sh` | Pass | Static package contract, shared-library/dynamic ABI deferral, metadata neutrality, support wording, and Windows package non-claims passed. |
| `make source-list-check` | Pass | Source-list guard passed with 49 library sources. |

### Skipped Gates

| Command | Day 10 disposition |
| --- | --- |
| `make report-index-comparison-freshness` | Skipped because no comparison generator, selected manifest, normalizer, report docs, workflow artifact, or generated comparison source changed; command would regenerate local artifacts. |
| `make report-index-oracle-freshness` | Skipped because no oracle generator, selected oracle metadata, report docs, or generated oracle source changed; command would regenerate local artifacts. |
| `make bench-canonical-report-freshness` | Skipped because no benchmark code, benchmark docs, manifest row, methodology metadata, or freshness checker changed; command would regenerate benchmark artifacts. |
| `make bench-canonical-report-freshness-tests` | Skipped because benchmark freshness logic and manifest semantics were unchanged. |
| `make ldlt-csc-helper-guard` | Skipped because LDLT CSC helper/test surfaces and maintainer guard wording were unchanged. |
| `make qr-external-ref-helper-guard` | Skipped because QR external-reference helper/test surfaces and maintainer guard wording were unchanged. |
| `make qr-header-docs-guard` | Skipped because QR public headers and QR header docs were unchanged. |
| `make format && make lint && make test` | Skipped because no `.c` or `.h` files changed. |

### Environment Residuals

| Evidence | Residual |
| --- | --- |
| Local PowerShell parse execution | Local `pwsh` was unavailable during `make windows-powershell-guard`; structural checks passed and missing local PowerShell remains an environment residual, not pass evidence. |
| Hosted Windows MSVC/CMake evidence | Not reproduced locally; hosted CI remains required for Windows platform proof. |
| Homebrew formula proof | Not run because this branch has no package-support promotion or license metadata decision. |
| Hosted benchmark platform freshness | Not run locally; platform promotion requires hosted artifact evidence and methodology metadata. |
| Generated report freshness | Local report regeneration was skipped for unchanged report surfaces. |

### Fixes

No fixes were required. All executed focused gates passed.

### Day 10 Acceptance Evidence For 206.4

- Focused docs/API, Windows ownership, package non-claim, static package/ABI
  non-claim, source-list, and patch hygiene gates passed.
- Skipped generated-report, benchmark, helper, and full C gates have explicit
  trigger-based reasons.
- Environment residuals were recorded before final validation closeout.
- The Day 10 artifact
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day10-focused-validation-log.md`
  captures the command results and follow-up rules in PR-reviewable form.

### Day 10 Validation

- Reviewed Sprint 197 Day 10 plan requirements.
- Executed the focused gates selected from the Day 9 matrix.
- Updated Sprint 197 planning artifacts only.
- No production code, public headers, generated reports, public docs,
  maintainer/API docs, benchmark docs, or report schema files were edited on
  Day 10.

## Day 11: Full Quality Gate Execution

### Method

Day 11 verifies the current changed surfaces and records the full-gate decision
for item 206.4. The branch remains documentation/planning-only, so Day 11 runs
the full documentation/planning checks required by the changed surfaces and
records why `make format && make lint && make test` is not required.

### Changed Surface Evidence

| Check | Result | Interpretation |
| --- | --- | --- |
| `git diff --name-only` | `docs/planning/EPIC_18/PROJECT_PLAN.md` | The only tracked modified file outside the new Sprint 197 directory is the Epic 18 project-plan interim status snapshot. |
| `git diff --name-only -- '*.c' '*.h'` | No output | No C source, public header, or internal header changed. |
| `git diff --cached --name-only` | No output | No files are staged. |
| `git ls-files docs/api build cmake-build scripts/__pycache__` | No output | Generated API/build/cache paths are not tracked. |
| `git status --short --ignored` | Modified `PROJECT_PLAN.md`, untracked `SPRINT_197/`, ignored `.claude/`, `.swp`, `archive/sparse_lu`, `build/`, `cmake-build/`, and `docs/api/` | Generated Doxygen output remains ignored local noise; no tracked generated artifact was introduced. |

### Required Gates Executed

| Command | Result | Evidence summary |
| --- | --- | --- |
| `git diff --check` | Pass | No trailing whitespace, conflict marker, or patch hygiene issue. |
| `make docs-check` | Pass | Doxygen generation succeeded; API docs coverage reported 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. |

### Full C Gate Decision

| Command | Day 11 status | Reason |
| --- | --- | --- |
| `make format` | Not required | No `*.c` or `*.h` files changed. |
| `make lint` | Not required | No C source or header changes require strict compile/lint validation. |
| `make test` | Not required | No implementation or header behavior changed. |
| `make format && make lint && make test` | Not required | The sprint rule triggers this sequence only when C/header files change; `git diff --name-only -- '*.c' '*.h'` had no output. |

### Final Validation Command List With Results

| Command | Latest result | Day |
| --- | --- | --- |
| `git diff --check` | Pass | Day 11 |
| `make docs-check` | Pass | Day 11 |
| `make api-docs-freshness` | Pass | Day 10 |
| `make windows-powershell-guard` | Pass with local `pwsh` unavailable residual captured by the guard | Day 10 |
| `bash scripts/package_manager_deferral_check.sh` | Pass | Day 10 |
| `bash scripts/static_package_deferral_check.sh` | Pass | Day 10 |
| `make source-list-check` | Pass | Day 10 |
| `make format && make lint && make test` | Not required for current docs-only diff | Day 11 |

### Skipped Generated Artifact Gates

| Command | Reason skipped |
| --- | --- |
| `make report-index-comparison-freshness` | No comparison generator, selected comparison manifest, normalizer, workflow, report docs, or generated comparison source changed. |
| `make report-index-oracle-freshness` | No oracle generator, selected oracle metadata, report-index docs, or generated oracle source changed. |
| `make bench-canonical-report-freshness` | No benchmark code, benchmark docs, selected manifest row, methodology metadata, or freshness checker changed. |

### Fixes

No fixes were required. Required Day 11 gates passed.

### Day 11 Acceptance Evidence For 206.4

- Required full gates for the current changed surfaces passed.
- Full C gate requirements are tied to the absence or presence of `*.c` and
  `*.h` diffs.
- No tracked generated artifacts or staged generated files were introduced.
- The Day 11 artifact
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day11-full-quality-gate-log.md`
  records the full quality-gate decision and clean tracked-worktree notes.

### Day 11 Validation

- Reviewed Sprint 197 Day 11 plan requirements.
- Ran required documentation/planning checks for the current branch state.
- Verified no `.c` or `.h` files changed.
- Verified generated API/build/cache paths are not tracked or staged.
- Updated Sprint 197 planning artifacts only.

## Day 12: Epic Retrospective Draft

### Method

Day 12 drafts `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` for
final-validation item 206.5. The draft follows the Epic 17 retrospective shape
while preserving the current evidence boundary: requested `SPRINT_197`
artifacts execute the project plan's Sprint 206 final-validation scope, and
Sprints 198 through 205 have no branch-local implementation evidence yet.

### Retrospective Draft Contents

| Section | Day 12 coverage |
| --- | --- |
| Epic objective | Records Epic 18's selected-closure intent and the numbering caveat. |
| Sprint outcomes | Marks Sprint 197 as in progress with numbering caveat, Sprints 198-205 as pending future execution, and Sprint 206 as partial final-validation evidence. |
| Major outcomes | Summarizes evidence reconciliation, public claim calibration, maintainer/API calibration, project-plan status, focused validation, full-gate decision, and claim governance. |
| Project-plan status | Counts interim dispositions without marking any Epic 18 item finally complete. |
| Validation evidence | Captures Day 10-11 focused/full validation results and skipped-gate boundaries. |
| Changed surface | Records planning/docs-only edits and no C/header changes. |
| Earned claims | Limits earned claims to final-validation planning, no-promotion decisions, and focused validation evidence. |
| Non-claims | Keeps package, Windows, benchmark, comparison, generated API, release, ABI, ecosystem parity, and state-of-the-art claims unpromoted. |
| Residual queue draft | Seeds Day 13 residual categories with closure targets. |
| State-of-the-art assessment | States that Epic 18 does not currently earn unqualified state-of-the-art status. |

### Outcome and Non-Claim Summary

The retrospective draft treats the current branch as a closeout governance
increment, not as implementation completion. It records that the branch has
earned evidence for planning, reconciliation, claim audits, no-promotion
decisions, project-plan interim status, and validation gating. It does not
claim Homebrew/package-manager support, promoted Windows freshness, additional
allocation-failure proof, additional review-surface reduction, additional
hosted benchmark freshness, Windows QR comparison promotion, generated API
publication, release readiness, shared-library/dynamic ABI support, portable
performance, broad ecosystem parity, or state-of-the-art sparse linear algebra
status.

### Cross-Links

| Retrospective topic | Evidence link |
| --- | --- |
| Evidence intake and numbering caveat | `SPRINT_197/artifacts/day1-closeout-intake.md` |
| Outcome ledger | `SPRINT_197/artifacts/day2-outcome-ledger.md` |
| Evidence conflicts | `SPRINT_197/artifacts/day3-evidence-conflicts.md` |
| Public claim audit and recalibration | `SPRINT_197/artifacts/day4-public-claim-audit.md`; `SPRINT_197/artifacts/day6-public-recalibration.md` |
| Maintainer/API audit and recalibration | `SPRINT_197/artifacts/day5-maintainer-api-claim-audit.md`; `SPRINT_197/artifacts/day7-maintainer-api-recalibration.md` |
| Project-plan status | `SPRINT_197/artifacts/day8-project-plan-status.md`; `docs/planning/EPIC_18/PROJECT_PLAN.md` |
| Validation planning and execution | `SPRINT_197/artifacts/day9-integrated-validation-matrix.md`; `SPRINT_197/artifacts/day10-focused-validation-log.md`; `SPRINT_197/artifacts/day11-full-quality-gate-log.md` |
| Retrospective draft record | `SPRINT_197/artifacts/day12-retrospective-draft.md` |

### Day 12 Acceptance Evidence For 206.5

- `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` now exists as a draft.
- The draft includes outcome, evidence, validation, non-claim, residual, and
  state-of-the-art assessment sections.
- Claims are backed by the current evidence ledger and explicitly identify
  absent Sprint 198-205 evidence.
- Residuals are seeded for Day 13 without being presented as completed work.

### Day 12 Validation

- Reviewed Sprint 197 Day 12 plan requirements.
- Reviewed the Epic 17 retrospective format.
- Reviewed Sprint 197 Day 1-11 artifacts and working notes.
- Created the Epic 18 retrospective draft and Day 12 draft record.
- Updated only planning documentation.

## Day 13: Residual Queue and Claim Decision

### Method

Day 13 completes final-validation item 206.6 for the current branch state. It
publishes a prioritized residual queue with closure criteria, owner surfaces,
expected evidence, validation commands, and claim boundaries. It also records
the final claim decision before Day 14 closeout review.

### Published Residual Queue

The prioritized residual queue is
`docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md`.

| Priority | Residual ID | Theme | Closure target |
| ---: | --- | --- | --- |
| 1 | E18-RQ-001 | Homebrew/package-manager support blocker | Add approved license metadata, run the selected Homebrew proof, update guards, install checks, and promote only the earned support tier. |
| 2 | E18-RQ-002 | Selected Windows Cholesky freshness promotion | Review hosted Windows artifacts, align manifest metadata, rerun normalizer/workflow/PowerShell guards, and promote or re-defer explicitly. |
| 3 | E18-RQ-003 | Additional allocation-failure owner proof | Select one owner, record invariants, extend deterministic failure tests, add a focused gate, update docs, and run full C validation if implementation changes. |
| 4 | E18-RQ-004 | Additional review-surface reduction | Select one high-risk cluster, extract/refactor behavior-preserving helpers, add ownership guards, and run focused/full validation. |
| 5 | E18-RQ-005 | Additional hosted selected benchmark freshness | Select one platform/row, add methodology metadata, create hosted artifact evidence, and keep performance claims methodology-bound. |
| 6 | E18-RQ-006 | Windows QR incompatible comparison promotion | Add MSVC/CMake proof, Windows-safe generation/path behavior, exact manifest metadata, selected tests, hosted evidence review, and calibrated docs. |
| 7 | E18-RQ-007 | Generated API publication policy | Decide hosted/artifact/committed/local-only policy and implement matching guards, link checks, and docs. |
| 8 | E18-RQ-008 | Adoption and diagnostics simplification | Add a quick reference, consolidate support truth, normalize diagnostics vocabulary, and preserve claim guards. |
| 9 | E18-RQ-009 | Release, shared-library, and dynamic ABI readiness | Define release criteria, ABI policy, shared-library metadata, loader validation, package selectors, and public claim review. |
| 10 | E18-RQ-010 | State-of-the-art evidence program | Define external baselines, methodology, platform matrix, reliability semantics, package provenance, thresholds, and reviewed hosted evidence. |

### Final Claim Decision

| Claim area | Day 13 decision |
| --- | --- |
| Sprint 197 final-validation governance | Earned as interim branch evidence. |
| Evidence reconciliation | Earned as partial final-validation evidence. |
| Public claim calibration | Earned as no-promotion evidence. |
| Maintainer/API claim calibration | Earned as no-promotion evidence. |
| Project-plan status | Earned as interim status evidence. |
| Focused/full validation | Earned for the current docs/planning diff. |
| Epic retrospective | Drafted and updated with residual queue links; Day 14 final review complete for the current branch state. |
| Residual queue | Published for the current closeout state. |
| Homebrew/package-manager support | Not earned. |
| Selected Windows Cholesky freshness promotion | Not earned. |
| Additional allocation-failure owner proof | Not earned. |
| Additional review-surface reduction | Not earned. |
| Additional hosted benchmark freshness | Not earned. |
| Windows QR incompatible comparison promotion | Not earned. |
| Generated API publication | Not earned; current policy remains local-only. |
| Adoption/support simplification | Not earned. |
| Release readiness | Not earned. |
| State-of-the-art status | Not earned. |

### Long-Horizon Deferrals

- Shared-library packaging and dynamic ABI compatibility.
- Runtime-loader behavior and platform-specific binary packaging.
- Release readiness and release benchmark policy.
- Broad Windows parity.
- Portable performance and hosted timing thresholds.
- Broad external-library parity.
- Hosted generated API publication if product policy selects it.
- Broad allocation-failure, OS OOM, and concurrency semantics.
- Unqualified state-of-the-art sparse linear algebra positioning.

### Day 14 Review Checklist

| Check | Day 14 expectation |
| --- | --- |
| Artifact completeness | Confirm `PLAN.md`, `WORKING_NOTES.md`, Day 1-13 artifacts, retrospective, residual queue, and project-plan snapshot are present. |
| Internal consistency | Confirm numbering caveat, pending Sprint 198-205 status, no-promotion decisions, and residual queue agree across artifacts. |
| Claim calibration | Confirm public, maintainer, retrospective, residual, and project-plan text does not promote unsupported package, Windows, benchmark, API, release, ABI, or state-of-the-art claims. |
| Validation currency | Re-run lightweight checks after Day 13 edits and record results. |
| C/header trigger | Confirm whether any `*.c` or `*.h` files changed; run `make format && make lint && make test` only if triggered. |
| Generated artifact noise | Confirm generated API/build/cache artifacts remain ignored and untracked. |
| PR summary inputs | Prepare summary of changed docs, validation evidence, non-claims, residual queue, and numbering caveat. |

### Day 13 Acceptance Evidence For 206.6

- `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` now exists.
- Residuals have priorities, closure criteria, owner surfaces, expected
  evidence, validation commands, and claim boundaries.
- Long-horizon deferrals are separated from near-term residuals.
- Final claim decisions are explicit and evidence-linked.
- `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md`,
  `docs/planning/EPIC_18/PROJECT_PLAN.md`, and the Day 8 item-level status
  ledger were updated to reference the residual queue and Day 13 claim
  decision.

### Day 13 Validation

- Reviewed Sprint 197 Day 13 plan requirements.
- Reviewed the Epic 17 residual queue format.
- Created the Epic 18 residual queue and Day 13 claim-decision artifact.
- Updated the retrospective, project-plan snapshot, Day 8 item-level status,
  and working notes.
- Updated only planning documentation.

## Day 14: Final Closeout Review

### Method

Day 14 performs the final coherence review and handoff preparation for the
requested Sprint 197 final-validation branch. It verifies artifact
completeness, cross-artifact consistency, claim calibration, validation
currency, generated-artifact hygiene, and PR summary inputs.

### Artifact Completeness Review

| Artifact family | Day 14 result |
| --- | --- |
| Sprint plan | `docs/planning/EPIC_18/SPRINT_197/PLAN.md` is present. |
| Working notes | `docs/planning/EPIC_18/SPRINT_197/WORKING_NOTES.md` is present and updated through Day 14. |
| Day artifacts | Day 1 through Day 14 artifacts are present under `docs/planning/EPIC_18/SPRINT_197/artifacts/`. |
| Project-plan status | `docs/planning/EPIC_18/PROJECT_PLAN.md` contains the interim status snapshot and points to Sprint 197 evidence. |
| Retrospective | `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` exists as a Day 14 closeout draft with explicit residuals. |
| Residual queue | `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` exists and contains prioritized near-term and long-horizon residuals. |

### Internal Consistency Review

| Check | Day 14 result |
| --- | --- |
| Numbering caveat | Consistent across plan, working notes, project-plan snapshot, retrospective, and closeout artifacts. |
| Sprint 198-205 status | Consistently pending future execution because no branch-local implementation artifacts, validation records, or PR evidence exist. |
| Claim calibration | Public and maintainer/API audits and no-promotion decisions agree with retrospective, residual queue, and project-plan status. |
| Validation records | Day 9-11 validation planning/execution/full-gate decision records agree with current docs/planning-only diff. |
| Residual handoff | Day 13 artifact, `EPIC_18_RESIDUAL_QUEUE.md`, and retrospective residual section agree on priorities and claim boundaries. |
| State-of-the-art assessment | Retrospective and residual queue both keep unqualified state-of-the-art status unearned. |

### Final Validation Summary

| Command | Latest result | Evidence |
| --- | --- | --- |
| `git diff --check` | Pass | Day 14 final validation. |
| `make docs-check` | Pass | Day 14 final validation. |
| `make api-docs-freshness` | Pass | Day 10 focused validation. |
| `make windows-powershell-guard` | Pass | Day 10 focused validation, with local `pwsh` unavailable residual recorded. |
| `bash scripts/package_manager_deferral_check.sh` | Pass | Day 10 focused validation. |
| `bash scripts/static_package_deferral_check.sh` | Pass | Day 10 focused validation. |
| `make source-list-check` | Pass | Day 10 focused validation. |
| `make format && make lint && make test` | Not required | No `.c` or `.h` files changed through Day 14. |

### Known Residuals

The prioritized handoff lives in
`docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md`. The top near-term residuals
are Homebrew/package-manager support, selected Windows Cholesky freshness,
additional allocation-failure owner proof, additional review-surface reduction,
additional hosted benchmark freshness, Windows QR incompatible comparison,
generated API publication policy, and adoption/diagnostics simplification.
Long-horizon deferrals include release readiness, shared-library/dynamic ABI
support, broad Windows parity, portable performance, broad external-library
parity, hosted generated API publication if selected by product policy, broad
allocation/OS OOM/concurrency semantics, and unqualified state-of-the-art
positioning.

### PR Summary Inputs

| Topic | Summary input |
| --- | --- |
| Changed files | Epic 18 project plan snapshot, Epic 18 retrospective draft, Epic 18 residual queue, Sprint 197 plan, working notes, and Day 1-14 artifacts. |
| Validation | `git diff --check`; `make docs-check`; Day 10 focused gates for API local-only, Windows PowerShell, package-manager deferral, static package deferral, and source-list checks. |
| Non-claims | No package-manager, broad Windows, benchmark portability, generated API publication, release, ABI, ecosystem parity, or state-of-the-art claim is promoted. |
| Numbering caveat | Requested `SPRINT_197` artifacts execute the final-validation scope that `PROJECT_PLAN.md` labels Sprint 206. |
| C gate | No `.c` or `.h` files changed, so `make format && make lint && make test` was not required. |

### Day 14 Acceptance Evidence

- All requested sprint deliverables are present for the current branch state.
- The project plan, retrospective, residual queue, working notes, validation
  logs, and claim-decision artifacts agree.
- Final claims are calibrated to exact evidence.
- Remaining work is captured in the prioritized residual queue.
- The Day 14 artifact
  `docs/planning/EPIC_18/SPRINT_197/artifacts/day14-final-closeout-review.md`
  records the final closeout review and PR summary inputs.

### Day 14 Validation

- Reviewed Sprint 197 Day 14 plan requirements.
- Reviewed artifact completeness and cross-artifact consistency.
- Updated retrospective, project-plan snapshot, Day 8 status ledger, and
  working notes for Day 14 closeout state.
- Re-ran final lightweight checks after Day 14 edits.
- Verified no `.c` or `.h` files changed.
- Updated only planning documentation.
