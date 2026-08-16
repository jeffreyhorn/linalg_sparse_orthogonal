# Day 1 Sprint Intake

## Scope

Day 1 establishes the Sprint 157 planning baseline. It does not select final
Epic 14 targets yet. It creates the sprint artifact structure, identifies the
source inputs, maps Sprint 157 project-plan items to days, defines evidence
capture categories, records branch state, and sets stop conditions that later
Sprint 157 days must honor.

## Prompt Reconciliation

The requested branch and output path are `sprint-157` and
`docs/planning/EPIC_14/SPRINT_157/PLAN.md`. In
`docs/planning/EPIC_14/PROJECT_PLAN.md`, Sprint 157 is:

- `Sprint 157: Epic 14 Baseline, Evidence Freeze & Claim Targets`
- lines 22-58 in the current file

The prompt's quoted title and line range point to the later Epic 14 closeout
sprint. Day 1 therefore follows the authoritative Sprint 157 section by sprint
number, branch, and output path.

## Branch Baseline

| Field | Value |
| --- | --- |
| Branch | `sprint-157` |
| Starting commit | `5b370dc33c1775205d839f99f0ef8ab8eaf7c3bd` |
| Starting summary | `5b370dc3 Merge pull request #174 from jeffreyhorn/planning/epic-14` |
| Upstream | current `master` after PR #174 merge |
| New Sprint 157 files at Day 1 start | `PLAN.md`, `WORKING_NOTES.md`, `artifacts/day1-sprint-intake.md` |

## Source Inputs

| Input | Purpose |
| --- | --- |
| `docs/planning/EPIC_14/PROJECT_PLAN.md` | Authoritative Sprint 157 item list, duration, goal, prerequisites, deliverables, and total estimate. |
| `docs/planning/EPIC_14/reviews/review-codex-2026-08-14.md` | Epic 14 review findings and state-of-the-art assessment. |
| `docs/planning/EPIC_14/reviews/todo-codex-2026-08-14.md` | Step-by-step closure plan that turns review gaps into implementation candidates. |
| `docs/planning/EPIC_13/EPIC_13_RETROSPECTIVE.md` | Epic 13 earned claims, non-claims, validation snapshot, and next-epic candidates. |
| `docs/planning/EPIC_13/SPRINT_156/artifacts/day11-residual-queue-publication.md` | Final Epic 13 residual queue with owners, blockers, prerequisites, and promotion gates. |
| `README.md`, `INSTALL.md`, `docs/api_reference.md`, `docs/maintainer_guide.md` | Public and maintainer claim boundaries for docs, package, API, generated evidence, platform support, and non-claims. |
| `Makefile`, `CMakeLists.txt`, `.github/workflows/*.yml` | Build, package, CI, and platform support evidence. |
| `tests/corpus/**`, `scripts/run_corpus_oracle.py`, `scripts/run_external_comparison.py`, `scripts/normalize_report_index.py` | Corpus, oracle, comparison, generated report, and freshness ownership. |
| `Doxyfile`, `include/`, `docs/api_reference.md` | Generated API reference and public-header source-of-truth inputs. |

## Baseline Categories

| Category | Day Owner | What To Capture | Evidence Boundary |
| --- | --- | --- | --- |
| Source/public API | Day 2 | File counts, line counts, public headers, installed headers, examples, benchmarks, scripts, source-list duplication. | Maintainability and API inventory only; not correctness proof. |
| Tests/CI | Day 3 | C tests, script tests, corpus tests, install tests, sanitizers, dead-code, CI lanes, Windows CTest count. | Reviewed/supplemental/staged distinctions must remain explicit. |
| Documentation/claims | Day 4 | README, install, tutorial, cookbook, API, solver-selection, benchmark, maintainer, corpus, example, and header wording. | Public claims require evidence owners; non-claims remain non-claims. |
| Generated artifacts | Day 5 | Doxygen HTML, corpus/oracle reports, comparison reports, benchmark/sentinel reports, coverage, dead-code, large-matrix reports. | Source-controlled metadata is not generated pass evidence. |
| Package/ABI/platform | Day 6 | Static-first install/export, Windows CMake downstream validation, Windows parity deltas, shared-library and ABI blockers. | Static-first support does not imply shared-library or dynamic ABI support. |
| Residuals | Day 7 | Epic 13 residuals and Epic 14 review gaps grouped by claim surface. | Residuals stay rejected or deferred until promotion gates pass. |
| Targets | Day 8 | Selected complete-gap targets and explicit non-goals mapped to Sprints 158-166. | A selected target must be closeable with a binary artifact, proof, or decision. |
| Evidence contracts | Day 9 | Templates for API docs, hosted reports, QR/partial-SVD comparison, Windows package, performance, and header cleanup. | Templates must distinguish pass evidence from advisory output. |
| Quality | Day 10 | Validation commands by touched surface. | C/header changes require full C quality gates. |
| Claims | Day 11 | Accepted claim register, explicit non-claim register, evidence owners, docs update checklist. | State-of-the-art and broad parity remain rejected unless recurring proof exists. |
| Risks/handoff | Day 12 | Risk register, mitigations, stop conditions, Sprint 158 generated API docs handoff. | Handoff must not broaden generated API docs claims. |
| Reconciliation/closeout | Days 13-14 | Artifact consistency, residuals, open questions, validation, Sprint 158 final handoff. | Sprint 158 must start from concrete generated API docs prerequisites. |

## Day-Level Owner Map

| Day | Owner Area | Primary Output |
| ---: | --- | --- |
| 1 | Sprint intake | Working notes, artifact structure, baseline categories, item owner map, stop conditions. |
| 2 | Code and public surface inventory | Source/header/example/benchmark/script inventory and largest-file risks. |
| 3 | Test and CI baseline | Local and hosted validation surfaces plus Windows reviewed-count snapshot. |
| 4 | Documentation and claims | Public/support docs inventory, positive claims, non-claims, and claim owners. |
| 5 | Generated artifact baseline | Generated-family inventory and source-controlled vs ignored-output boundaries. |
| 6 | Package, ABI, and platform baseline | Static-first proof, Windows package deltas, and ABI/shared-library blockers. |
| 7 | Residual consolidation | Claim-oriented residual register with owners, blockers, prerequisites, and gates. |
| 8 | Epic 14 target selection | Selected targets, explicit non-goals, and target-to-sprint map. |
| 9 | Evidence contracts | Reusable evidence templates for selected Epic 14 work. |
| 10 | Quality surface map | Validation matrix by touched surface and stop conditions. |
| 11 | Claim target register | Accepted claims, rejected broad claims, evidence owners, and docs update list. |
| 12 | Risk and Sprint 158 handoff | Risk register and generated API docs handoff draft. |
| 13 | Baseline reconciliation | Reconciled artifact index and residual updates. |
| 14 | Sprint closeout | Final Sprint 157 artifacts, validation, residuals, and Sprint 158 handoff. |

## Evidence Capture Format

Later Sprint 157 artifacts should record evidence in this format:

| Field | Requirement |
| --- | --- |
| Evidence source | File, command, CI job, report row, generated artifact, retrospective source, or review finding. |
| Claim supported | Narrow claim, support-tier statement, non-claim boundary, or residual. |
| Status | `supported`, `selected`, `candidate`, `deferred`, `rejected`, `residual`, or `blocked`. |
| Validation required | Exact local command, hosted CI requirement, generated freshness check, package proof, declaration-preservation check, or manual audit. |
| Boundary | What the evidence does not prove. |
| Owner | Maintainer surface responsible for future updates. |
| Handoff | Next day or sprint that consumes the evidence. |

## Stop Conditions

| Stop Condition | Required Action |
| --- | --- |
| Candidate claim lacks concrete evidence. | Keep it as residual/non-claim and do not add public wording. |
| State-of-the-art, broad external parity, or portable performance wording lacks direct comparative evidence. | Reject the claim and record it in the non-claim register. |
| Platform support wording lacks hosted platform proof. | Keep the lane reviewed, supplemental, staged, local-only, advisory, or deferred as appropriate. |
| Package/ABI wording lacks downstream consumer proof. | Preserve static-first or deferred wording. |
| Windows CMake package evidence is used to imply Windows Makefile or `pkg-config` parity. | Correct the wording and preserve the parity residual until its own gate passes. |
| Generated local rows are treated as source-controlled pass evidence. | Correct artifact wording and require the selected freshness gate before promotion. |
| Doxygen output is described as current without a documented `make docs` run, warning triage, and page-coverage check. | Treat generated HTML as stale or local-only until Sprint 158 closes the gap. |
| C/header changes occur. | Require `make format && make lint && make test` before closeout. |
| Documentation-only changes fail whitespace or claim scans. | Fix wording or stop for clarification. |
| Review feedback or validation failure is unclear. | Stop and ask before committing a claim or fix. |

## Sprint 158 Handoff Seed

Sprint 158 should begin from the generated API documentation gap. Day 1
identifies these critical prerequisite surfaces:

- `Doxyfile`
- `include/`
- `include/sparse_version.h.in`
- `docs/api_reference.md`
- `docs/maintainer_guide.md`
- `README.md`
- `docs/tutorial.md`
- `docs/cookbook.md`
- `docs/solver_selection.md`
- `Makefile` target `docs`

Until Sprint 158 closes the publication decision, generated API HTML remains a
known residual and the checked-in public headers remain the source of truth for
exact declarations.

## Day 1 Completion Check

- Sprint 157 scope is tied to the authoritative Sprint 157 project-plan
  section.
- Artifact directory structure exists.
- Branch baseline is recorded.
- Source inputs, baseline categories, day owners, evidence format, and stop
  conditions are documented.
- Sprint 158 generated API docs handoff seed is identified.
