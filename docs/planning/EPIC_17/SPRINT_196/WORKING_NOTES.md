# Sprint 196 Working Notes: Epic 17 Final Validation, Claim Calibration & Closeout

**Sprint:** 196
**Status:** Complete
**Goal:** Reconcile all Epic 17 work, run final validation, calibrate public
claims, and publish the Epic 17 retrospective and residual queue.

## Sprint Item Checklist

| Item | Name | Day 1 closeout surface | Status |
| --- | --- | --- | --- |
| 196.1 | Evidence Reconciliation | Sprint 187-195 plans, notes, retrospectives, closeout artifacts, validation records, review comments, and residuals | Complete |
| 196.2 | Claim Recalibration | README, INSTALL, maintainer guide, benchmark docs, API docs, planning docs, and generated evidence docs | Day 13 final claim review complete |
| 196.3 | Project Plan Status | `docs/planning/EPIC_17/PROJECT_PLAN.md` Sprint 187-196 item status and evidence links | Complete |
| 196.4 | Integrated Validation | Focused gates by touched surface plus final full quality gates required by code/header/doc changes | Day 12 focused/full-quality validation complete |
| 196.5 | Epic Retrospective | `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` outcomes, evidence, non-claims, residuals, and state-of-the-art assessment | Day 13 final review complete |
| 196.6 | Residual Queue | Prioritized next-epic residual queue with closure targets and long-horizon deferrals | Complete |

## Day 1: Closeout Intake

### Scope Trace

| Sprint 196 item | Day 1 interpretation | Evidence to collect before edits |
| --- | --- | --- |
| 196.1 Evidence Reconciliation | Build a consolidated ledger of what Sprints 187-195 actually closed, narrowed, deferred, or left residualized. | Prior sprint retrospectives, Day 14 closeout artifacts, working notes, validation commands, CI failures, and PR review fixes. |
| 196.2 Claim Recalibration | Identify public and maintainer claim surfaces before editing wording. | Current user docs, maintainer docs, benchmark methodology docs, API docs, package docs, Windows docs, and report-index docs. |
| 196.3 Project Plan Status | Prepare to annotate project-plan items with final status and evidence links. | Epic 17 project plan plus sprint-specific artifacts proving each status. |
| 196.4 Integrated Validation | Map required gates to the surfaces likely to change during Sprint 196. | Makefile targets, Python validation scripts, C/header gates, docs/API guards, package guards, Windows guards, report freshness guards, and performance guards. |
| 196.5 Epic Retrospective | Identify the retrospective structure and evidence sections needed for final closeout. | Existing Epic retrospectives plus Sprint 187-195 retrospective patterns. |
| 196.6 Residual Queue | Start a residual queue seed without declaring closure prematurely. | Residual tables from Sprint 187-195 retrospectives and Day 14 handoffs. |

### Evidence Source Inventory

| Sprint | Topic | Primary files | Day 1 status signal |
| --- | --- | --- | --- |
| 187 | Epic 17 baseline, gap ledger, and acceptance gates | `docs/planning/EPIC_17/SPRINT_187/PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, `artifacts/day14-closeout-summary.md` | Complete baseline and gate-selection sprint. |
| 188 | Homebrew proof completion | `docs/planning/EPIC_17/SPRINT_188/PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, `artifacts/day14-closeout-summary.md` | Complete with guarded package residual around approved standalone root license metadata. |
| 189 | PowerShell validation ownership | `docs/planning/EPIC_17/SPRINT_189/PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, `artifacts/day14-sprint-closeout.md` | Complete with hosted Windows evidence pending PR CI at closeout. |
| 190 | Windows selected report freshness decision | `docs/planning/EPIC_17/SPRINT_190/PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, `artifacts/day14-sprint-closeout.md` | Complete with residual narrowed; hosted Windows evidence and manifest promotion pending at closeout. |
| 191 | Bounded external comparison family | `docs/planning/EPIC_17/SPRINT_191/PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, `artifacts/day14-closeout-and-handoff.md` | Complete; one bounded local-only comparison family landed with residuals documented. |
| 192 | Methodology-bound performance evidence lane | `docs/planning/EPIC_17/SPRINT_192/PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, `artifacts/day14-closeout-and-handoff.md` | Complete; one threshold-free hosted selected performance evidence lane landed with explicit limits. |
| 193 | Selected large review-surface reduction | `docs/planning/EPIC_17/SPRINT_193/PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, `artifacts/day14-closeout.md` | Complete; selected QR external-reference helper surface reduced without production behavior change. |
| 194 | Adoption and API coherence simplification | `docs/planning/EPIC_17/SPRINT_194/PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, `artifacts/day14-closeout-handoff.md` | Complete; adoption/API guidance simplified and support/readiness truth consolidated. |
| 195 | Selected reliability and failure-path proof | `docs/planning/EPIC_17/SPRINT_195/PLAN.md`, `WORKING_NOTES.md`, `RETROSPECTIVE.md`, `artifacts/day14-closeout-review-package.md` | Complete; one selected symbolic Cholesky allocation-failure owner proved with deterministic cleanup evidence. |

### Artifact Inventory

Each Sprint 187-195 directory contains a plan, working notes, retrospective,
daily artifacts, and a Day 14 closeout/handoff artifact. The Day 14 artifacts
are the first-pass evidence anchors for Day 2 reconciliation:

- `docs/planning/EPIC_17/SPRINT_187/artifacts/day14-closeout-summary.md`
- `docs/planning/EPIC_17/SPRINT_188/artifacts/day14-closeout-summary.md`
- `docs/planning/EPIC_17/SPRINT_189/artifacts/day14-sprint-closeout.md`
- `docs/planning/EPIC_17/SPRINT_190/artifacts/day14-sprint-closeout.md`
- `docs/planning/EPIC_17/SPRINT_191/artifacts/day14-closeout-and-handoff.md`
- `docs/planning/EPIC_17/SPRINT_192/artifacts/day14-closeout-and-handoff.md`
- `docs/planning/EPIC_17/SPRINT_193/artifacts/day14-closeout.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day14-closeout-handoff.md`
- `docs/planning/EPIC_17/SPRINT_195/artifacts/day14-closeout-review-package.md`

### Claim Surface Inventory

| Surface | Why it needs review | Day 1 disposition |
| --- | --- | --- |
| `README.md` | Top-level user-facing capability, support, package, quality, and performance claims. | Inventory only; defer wording edits until evidence reconciliation. |
| `INSTALL.md` | Active install/support/readiness matrix and package-manager guidance. | Inventory only; compare against Sprint 188 and Sprint 194 evidence. |
| `docs/maintainer_guide.md` | Maintainer ownership for package, Windows, freshness, validation, and non-claim semantics. | Inventory only; compare against guard expectations and residuals. |
| `benchmarks/README.md` | Performance methodology, selected evidence scope, and non-portable benchmark claim boundaries. | Inventory only; compare against Sprint 192 evidence. |
| `docs/api_reference.md` | API documentation coverage and public surface consistency. | Inventory only; compare against Sprint 194 API/header work. |
| `include/*.h` | Public API contracts and simplified user-facing header narratives. | Inventory only; C/header changes would trigger the full C quality gate. |
| `docs/solver_selection.md` | User guidance for choosing solvers and interpreting support/readiness. | Inventory only; compare against adoption simplification outcomes. |
| `docs/cookbook.md` | User workflow examples and diagnostic expectations. | Inventory only; compare against adoption and package guidance. |
| `tests/corpus/README.md` and `tests/corpus/schemas/report_index_fields.md` | Report-index and selected target manifest semantics. | Inventory only; compare against Sprints 190-192. |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Final item status for Sprint 187-196. | Defer edits until Day 7 status evidence exists. |
| `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` | Final Epic 17 outcome and residual narrative. | Create later after claim/status calibration. |

### Initial Validation Gate Map

| Gate family | Candidate commands or evidence | Applies when |
| --- | --- | --- |
| Repository hygiene | `git diff --check`, `make source-list-check`, `make format-check` | Baseline and before final closeout. |
| C/header quality | `make format`, `make lint`, `make test`, `make quality-review-compile` | Any `.c` or `.h` code/header change. |
| CMake/reviewed build | `make quality-review-cmake-compile`, `make quality-review-cmake`, selected CI workflow evidence | CMake, package, install, or downstream consumer changes. |
| Docs/API | `make docs-check`, `make api-docs-validate`, `make api-docs-freshness`, `make qr-header-docs-guard` | User docs, API docs, generated docs, or public-header wording changes. |
| Windows/PowerShell | `make windows-powershell-validate`, `make windows-powershell-guard`, hosted Windows workflow evidence | Windows claim, workflow, PowerShell, or selected freshness semantics change. |
| Report freshness | `make report-index-oracle-freshness`, `make report-index-comparison-freshness`, selected target manifest validation | Report index, comparison target, manifest, or freshness wording changes. |
| Performance evidence | `make bench-canonical-report-freshness`, `make bench-canonical-report-freshness-tests`, `make performance-sentinels` | Benchmark methodology, selected performance lane, or report claims change. |
| Reliability proof | `make symbolic-allocation-failure-gate`, `make iterative-allocation-failure-gate`, `make matmul-allocation-failure-gate` | Reliability harness, allocation-failure proof, or stale-output/retry semantics change. |
| Review-surface reduction | `make qr-external-ref-helper-guard`, `make ldlt-csc-helper-guard` | Large helper movement, source-list ownership, or review-surface claims change. |
| Package proof | Install/CMake/Homebrew proof scripts and package guard targets identified by the package artifacts | Package-manager or install support claims change. |

### Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Final docs overclaim state-of-the-art status. | Public claims could exceed Epic 17 evidence. | Tie claims to completed evidence and keep non-claims explicit. |
| Residuals are hidden by broad "complete" wording. | Future owners lose exact closure targets. | Preserve residual tables with owner conditions and evidence required to close. |
| Sprint status differs between plan intent and closeout outcome. | Project-plan status could become inaccurate. | Use retrospectives and Day 14 artifacts as source of truth. |
| Local validation cannot reproduce hosted Windows/macOS evidence. | Final validation record could conflate local and hosted proof. | Separate local command results from hosted CI evidence and record unavailable environments as residuals. |
| Claim wording duplicates across docs. | Docs drift after closeout. | Assign each claim type a primary source and make secondary docs point to it. |
| C/header edits accidentally expand the validation burden. | Final closeout may require the full C gate. | Avoid C/header edits unless required; if touched, run full requested C gates. |
| Package-manager residual is represented as support instead of readiness evidence. | Users may infer unsupported distribution guarantees. | Keep package-manager support guarded until provider-specific proof closes the residual. |
| Windows/PowerShell evidence semantics are too broad. | Hosted validation ownership may be mistaken for general Windows parity. | Preserve selected-lane and hosted-evidence boundaries. |

### Day 2 Reconciliation Questions

1. Which Sprint 187-195 outcomes are fully closed, narrowed, guarded, pending,
   superseded, or deferred?
2. Which residuals from prior sprints are duplicates and should be merged into
   one next-epic queue item?
3. Which public docs currently conflict with the final evidence state?
4. Which hosted evidence remains pending or local-unavailable and needs explicit
   non-claim wording?
5. Which validation gates are mandatory for Sprint 196 based on the files that
   will actually change?

### Day 1 Validation

- Reviewed Sprint 196 plan and Epic 17 project-plan items 196.1-196.6.
- Inventoried Sprint 187-195 plans, working notes, retrospectives, daily
  artifacts, and Day 14 closeout artifacts.
- Identified initial claim surfaces and validation-gate families.
- No production code, public header, generated API output, README, INSTALL, or
  maintainer-guide claims were edited on Day 1.

## Day 2: Sprint Outcome Reconciliation

### Outcome Ledger

| Sprint | Topic | Reconciled status | Evidence anchor | Residual or pending state |
| --- | --- | --- | --- | --- |
| 187 | Epic 17 baseline, gap ledger, and acceptance gates | Complete | `docs/planning/EPIC_17/SPRINT_187/artifacts/day14-closeout-summary.md`; `docs/planning/EPIC_17/SPRINT_187/RETROSPECTIVE.md` | Future sprint choices were selected as handoffs; broad state-of-the-art, ABI, package, platform, parity, and performance claims remained non-goals. |
| 188 | Homebrew proof completion | Complete with guarded residual | `docs/planning/EPIC_17/SPRINT_188/artifacts/day14-closeout-summary.md`; `docs/planning/EPIC_17/SPRINT_188/RETROSPECTIVE.md` | Missing approved standalone root license metadata blocks proof exit `0` and any Homebrew install support claim. |
| 189 | PowerShell validation ownership | Complete with hosted evidence pending at sprint closeout | `docs/planning/EPIC_17/SPRINT_189/artifacts/day14-sprint-closeout.md`; `docs/planning/EPIC_17/SPRINT_189/RETROSPECTIVE.md` | Local `pwsh` absence remains environment residual; hosted `powershell-validation` pass evidence was PR-CI-owned. |
| 190 | Windows selected report freshness decision | Complete with residual narrowed | `docs/planning/EPIC_17/SPRINT_190/artifacts/day14-sprint-closeout.md`; `docs/planning/EPIC_17/SPRINT_190/RETROSPECTIVE.md` | Bounded selected Cholesky Windows workflow path exists, but manifest promotion and hosted artifact evidence were still pending at closeout. |
| 191 | Bounded external comparison family | Complete | `docs/planning/EPIC_17/SPRINT_191/artifacts/day14-closeout-and-handoff.md`; `docs/planning/EPIC_17/SPRINT_191/RETROSPECTIVE.md` | `qr-incompatible-ls` is local-only selected evidence; Windows promotion, optional package baselines, broader QR parity, and generated local artifacts remain residual/future evidence. |
| 192 | Methodology-bound performance evidence lane | Complete | `docs/planning/EPIC_17/SPRINT_192/artifacts/day14-closeout-and-handoff.md`; `docs/planning/EPIC_17/SPRINT_192/RETROSPECTIVE.md` | One Linux hosted selected benchmark bundle is methodology-bound and threshold-free; portable performance, timing thresholds, Windows/macOS freshness, and release benchmark claims remain residual. |
| 193 | Selected large review-surface reduction | Complete | `docs/planning/EPIC_17/SPRINT_193/artifacts/day14-closeout.md`; `docs/planning/EPIC_17/SPRINT_193/RETROSPECTIVE.md` | Selected QR helper extraction closed one review-surface claim only; economy/sparse/refinement QR clusters, helper dependency caveat, and helper-size follow-ups remain residual. |
| 194 | Adoption and API coherence simplification | Complete | `docs/planning/EPIC_17/SPRINT_194/artifacts/day14-closeout-handoff.md`; `docs/planning/EPIC_17/SPRINT_194/RETROSPECTIVE.md` | Support/readiness truth was consolidated; package-manager distribution, shared library/dynamic ABI, broad Windows parity, portable performance, link-check target, and local PowerShell remain residual. |
| 195 | Selected reliability and failure-path proof | Complete | `docs/planning/EPIC_17/SPRINT_195/artifacts/day14-closeout-review-package.md`; `docs/planning/EPIC_17/SPRINT_195/RETROSPECTIVE.md` | Selected `sparse_symbolic_cholesky()` allocation-failure owner is proved; other symbolic/analyze/helper/direct/matrix reliability paths, OS OOM, concurrency, and hosted gate ownership remain residual. |

### Item-Level Disposition Summary

| Sprint | Completed items | Guarded, residual, or pending items | Supersession notes |
| --- | --- | --- | --- |
| 187 | 187.1, 187.2, 187.3, 187.4, 187.5, 187.6 | None inside Sprint 187; selected implementation work was intentionally handed to Sprints 188-195. | Sprint 187 plan intent remains valid as the baseline, but later sprint retrospectives supersede handoff assumptions where implementation found narrower proof. |
| 188 | 188.1, 188.3, 188.4, 188.5, 188.6 | 188.2 remains residual because authoritative root license metadata was not available. | Any earlier plan wording implying metadata implementation should be read as superseded by the no-invented-license decision. |
| 189 | 189.1, 189.2, 189.3 in source, 189.4, 189.5, 189.6 | Hosted pass evidence and local `pwsh` exit `0` were pending/unavailable at sprint closeout. | PowerShell validation ownership does not supersede or imply Windows report freshness; Sprint 190 owns that decision. |
| 190 | 190.1, 190.2, 190.3, 190.4, 190.5, 190.6 | Manifest Windows metadata promotion and hosted artifact evidence remained pending at sprint closeout. | Sprint 190 narrows the previous Windows freshness deferral to one selected Cholesky lane, but does not close broad Windows report freshness. |
| 191 | 191.1, 191.2, 191.3, 191.4, 191.5, 191.6 | Windows selected QR incompatible freshness, optional NumPy/SciPy baselines, generated local artifacts, and broader QR parity remain residual. | The new QR incompatible family supersedes candidate-family uncertainty only for the selected local fixture. |
| 192 | 192.1, 192.2, 192.3, 192.4 as threshold-free decision, 192.5, 192.6 | Hosted timing threshold, portable performance, Windows/macOS freshness, unselected CSV publication, and release benchmark claim remain residual. | The threshold-free policy supersedes any implied Sprint 192 timing-threshold or performance-superiority expectation. |
| 193 | 193.1, 193.2, 193.3, 193.4, 193.5, 193.6 | Remaining QR clusters, helper dependency tracking, large helper size, and unrelated warning hygiene remain residual. | The selected extraction supersedes broad large-test cleanup only for the QR rank/nullspace/threshold cluster. |
| 194 | 194.1, 194.2, 194.3, 194.4, 194.5, 194.6 | Link-check target, package distribution, shared library/dynamic ABI, broad Windows parity, portable performance, and further header detail cleanup remain residual. | `INSTALL.md` becomes the support/readiness owner; duplicated older support wording should be calibrated against it. |
| 195 | 195.1, 195.2, 195.3, 195.4, 195.5, 195.6 | Other reliability owners, OS OOM, concurrent allocation hook behavior, and hosted CI ownership remain residual. | The symbolic Cholesky proof supersedes only the selected owner gap, not broad allocation-failure reliability. |

### Evidence-Link Table By Topic

| Topic | Earned evidence | Primary links | Claim boundary for later days |
| --- | --- | --- | --- |
| Package/Homebrew | Local proof path and guards are hardened, but Homebrew support remains blocked by approved license metadata. | Sprint 188 closeout and retrospective. | Do not claim package-manager install support, Homebrew/core readiness, bottles, taps, Linuxbrew, or provider registry support. |
| Windows/PowerShell | PowerShell workflow ownership exists and a bounded selected Cholesky Windows workflow path exists. | Sprint 189 and 190 closeouts/retrospectives. | Distinguish hosted evidence from local `pwsh` absence; do not claim broad Windows parity or broad Windows report freshness. |
| External comparison | One additional bounded local-only `qr-incompatible-ls` selected comparison family landed. | Sprint 191 closeout and retrospective. | Keep claims fixture-local, reference-local, and local-only unless future hosted/platform evidence promotes a target. |
| Performance | One selected Linux hosted benchmark bundle exists with complete methodology metadata and threshold-free semantics. | Sprint 192 closeout and retrospective. | Do not claim portable performance superiority, timing thresholds, or release benchmark status. |
| Review-surface maintainability | One selected QR external-reference helper extraction is complete and guardable. | Sprint 193 closeout and retrospective. | Do not describe broad QR cleanup or solver behavior change. |
| Adoption/API coherence | User docs, install support matrix, diagnostics wording, and selected public-header narrative were simplified. | Sprint 194 closeout and retrospective. | Use `INSTALL.md` support/readiness matrix as claim owner; do not widen deferred support. |
| Reliability/failure path | One selected symbolic Cholesky allocation-failure path is deterministic, guarded, and documented. | Sprint 195 closeout and retrospective. | Keep proof scoped to selected `sparse_symbolic_cholesky()` output allocation behavior. |
| Epic governance | Sprint 187 created the gap ledger, acceptance gates, quality map, and handoffs. | Sprint 187 closeout and retrospective. | Final Epic claims must remain evidence-bound and explicitly separate retained non-goals. |

### Conflict And Supersession List

| Area | Earlier wording or expectation to watch | Reconciled Day 2 interpretation |
| --- | --- | --- |
| Homebrew metadata | Sprint 188 planned possible metadata implementation. | Superseded by the license strategy decision: no root license metadata or Homebrew license identifier can be invented without project approval. |
| Homebrew support | Local formula material could be mistaken for user install support. | Guarded residual only; expected proof unavailable result remains exit `2` until metadata evidence exists. |
| PowerShell ownership | Hosted job wiring could be read as Windows report freshness. | Sprint 189 owns PowerShell validation only; Sprint 190 owns selected report freshness. |
| Windows selected Cholesky | A workflow path could be read as completed Windows metadata promotion. | Sprint 190 narrowed the residual but left hosted evidence review and manifest promotion pending at closeout. |
| QR incompatible comparison | Generated rows could be read as broad QR external parity. | Sprint 191 closes one local-only selected comparison fixture; broader QR parity and optional package baselines remain future work. |
| Benchmark lane | Hosted CSV publication could be read as a performance threshold or portable claim. | Sprint 192 publishes methodology-bound measurement evidence only. |
| QR helper extraction | Moving one helper cluster could be read as broad test-suite simplification. | Sprint 193 reduced one selected QR review surface only. |
| Support matrix | Consolidated support wording could be read as new support. | Sprint 194 simplifies and routes truth; it does not add package, ABI, platform, or performance support. |
| Symbolic reliability proof | One deterministic allocation-failure proof could be read as broad OOM/reliability coverage. | Sprint 195 proves only selected `sparse_symbolic_cholesky()` output allocation behavior on bounded fixtures. |
| Epic 17 outcome | Completing selected closures could be read as state-of-the-art status. | Epic 17 improves evidence and coherence but does not earn unqualified state-of-the-art, ecosystem parity, release, or portable performance claims. |

### Day 2 Acceptance Evidence For 196.1

- Every Sprint 187-195 outcome now has a reconciled status and evidence anchor
  in this working-notes ledger.
- Deferred, guarded, pending, and residual outcomes are separated from
  completed outcomes.
- Conflicts between original sprint intent and final closeout interpretation
  are recorded before claim or status edits.
- The Day 2 artifact
  `docs/planning/EPIC_17/SPRINT_196/artifacts/day2-outcome-ledger.md`
  duplicates this reconciliation in PR-reviewable form.

### Day 2 Validation

- Re-read Sprint 196 Day 2 plan requirements.
- Reviewed Sprint 187-195 project-plan item rows.
- Reviewed Sprint 187-195 retrospectives for status and residual sections.
- Reviewed Sprint 187-195 Day 14 closeout/handoff artifacts as evidence
  anchors.
- No public claim files, project-plan status rows, production code, or public
  headers were edited on Day 2.

## Day 3: Residual and Deferred Work Triage

### Triage Method

Day 3 treats Sprint 187-195 retrospective residual tables as the primary source
of truth, then checks working notes and Day 14 closeout artifacts for
handoff-only residuals. Repeated residuals are merged when they point to the
same missing proof owner, validation lane, or claim boundary.

### Consolidated Residual Queue

| Queue ID | Consolidated residual | Source sprints | Classification | Priority | Owner condition | Closure evidence |
| --- | --- | --- | --- | --- | --- | --- |
| E17-RQ-001 | Package-manager/Homebrew support remains unclaimed because approved standalone root license metadata and exact Homebrew license identifier are missing. | 188, 194 | Next-epic candidate | High | Future package-provider/product owner | Add approved root license metadata, set exact Homebrew license identifier, rerun local proof to exit `0`, rerun package/static guards, update user and maintainer docs to only the earned support level. |
| E17-RQ-002 | Shared-library packaging and dynamic ABI support remain deferred. | 187, 194 | Long-horizon deferral | High | Future ABI/package owner | Add shared-library build/install artifacts, symbol visibility policy, ABI/versioning policy, loader validation, package metadata, static/shared selection semantics, and consumer tests. |
| E17-RQ-003 | Broad Windows parity remains unclaimed beyond selected hosted/CMake evidence. | 187, 190, 194 | Long-horizon deferral | High | Future Windows platform owner | Add reviewed Windows evidence for each desired surface with exact workflow owners, artifacts, docs, tests, and non-claim updates. |
| E17-RQ-004 | Local PowerShell execution is unavailable on this machine. | 189, 190, 194 | Validation/tooling follow-up | Medium | Local developer environment or hosted CI owner | Install `pwsh` and rerun `make windows-powershell-validate` for exit `0`, or continue recording local exit `2` separately from hosted `--require-pwsh` evidence. |
| E17-RQ-005 | Sprint 190 selected Cholesky Windows freshness promotion depends on hosted evidence review and manifest metadata promotion. | 190 | Next-epic candidate | High | Selected report target manifest/Windows CI owner | Observe hosted `selected-comparison-freshness` pass, inspect exact uploaded Cholesky bundle, promote only `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` to Windows metadata, and rerun coupled manifest/normalizer/generator/PowerShell tests. |
| E17-RQ-006 | Windows selected QR incompatible freshness remains deferred. | 191 | Next-epic candidate | Medium | Hosted Windows comparison owner | Add MSVC/CMake proof for `qr-incompatible-ls`, inspect hosted artifacts, update exact selected target metadata, and preserve broader QR parity non-claims. |
| E17-RQ-007 | Optional NumPy/SciPy package baselines remain deferred advisory rows. | 191 | Long-horizon deferral | Low | External comparison baseline owner | Select package-backed baselines, define dependency policy, add package availability checks, update comparison rows/docs, and avoid package-manager support implications. |
| E17-RQ-008 | Broader QR least-squares or external-library parity remains unproved. | 187, 191 | Long-horizon deferral | Medium | Future comparison-family owner | Add additional bounded fixtures one at a time with exact reference, metrics, tolerances, row IDs, freshness diagnostics, and claim calibration. |
| E17-RQ-009 | Generated local comparison artifacts remain ignored and must be regenerated or reviewed from CI uploads before use as evidence. | 191 | Validation/tooling follow-up | Medium | Reviewer/CI evidence owner | Regenerate `build/comparison/qr_incompatible_ls/` locally or inspect CI-uploaded artifacts; do not cite ignored generated rows without fresh evidence. |
| E17-RQ-010 | Selected comparison review volume may grow as target families accumulate. | 191 | Validation/tooling follow-up | Low | Test/report infrastructure owner | Extract shared selected-target constants only if row identity, diagnostics, and reviewability stay explicit. |
| E17-RQ-011 | Hosted timing thresholds are not defined. | 192 | Long-horizon deferral | Medium | Performance-governance owner | Add reviewed baseline, variance model, machine-class policy, flake budget, failure wording, and conservative enforcement semantics before any timing gate. |
| E17-RQ-012 | Portable performance evidence remains unclaimed. | 187, 192, 194 | Long-horizon deferral | High | Benchmark methodology owner | Add multi-platform, multi-machine, repeated, variance-aware evidence with compiler, CPU, threading, backend, matrix, and environment context. |
| E17-RQ-013 | Windows/macOS selected benchmark freshness is not owned. | 192 | Next-epic candidate | Medium | Platform CI owner | Add hosted platform lanes and selected artifact validation without broadening current Linux selected benchmark claims. |
| E17-RQ-014 | Unselected canonical CSV publication remains local-only. | 192 | Documentation/tooling follow-up | Low | Benchmark publication owner | Select, document, guard, and publish each promoted row explicitly before treating an unselected CSV as review evidence. |
| E17-RQ-015 | Release benchmark claims remain undefined. | 192 | Long-horizon deferral | Medium | Release engineering owner | Define release fixtures, reproducible environments, archived artifacts, acceptance criteria, and docs before release-performance claims. |
| E17-RQ-016 | Additional QR review-surface clusters remain in `tests/test_qr.c`. | 193 | Next-epic candidate | Medium | Future QR review-surface owner | Select economy, sparse-mode, or refinement cluster explicitly, define helper boundary, preserve registration, add guard coverage, and rerun focused/full validation. |
| E17-RQ-017 | Header-only focused rebuild caveat remains for QR helper edits. | 193 | Validation/tooling follow-up | Medium | Test/build owner | Add dependency tracking for helper headers or keep forced rebuild guidance before focused QR execution. |
| E17-RQ-018 | Large helper size may still add review burden. | 193 | Documentation/tooling follow-up | Low | Test-structure owner | Split helpers only if the change reduces review burden without source-list or proof-owner ambiguity. |
| E17-RQ-019 | Existing unrelated warning hygiene remains outside Sprint 193 closure. | 193 | Out-of-scope historical note | Low | Future warning-hygiene owner | Review the named warning separately if it still reproduces under current gates. |
| E17-RQ-020 | No dedicated Markdown link-check target exists. | 194 | Validation/tooling follow-up | Medium | Documentation tooling owner | Add a link-check target, fixtures, documented failure semantics, and maintainable exclusions before making link integrity a local gate. |
| E17-RQ-021 | Public headers still contain declaration-adjacent detailed API contracts. | 194 | Documentation-only follow-up | Low | API documentation owner | Move only broad workflow narrative when generated API coverage and docs routing remain valid; keep exact declaration contracts near declarations. |
| E17-RQ-022 | Additional allocation-failure owners remain unproved. | 195 | Next-epic candidate | High | Future symbolic/analysis/solver/matrix reliability owners | Select one owner per sprint, record invariants, wrapper-control selected allocations if needed, add cleanup/stale-output/retry tests, focused gate, docs, and full validation. |
| E17-RQ-023 | OS OOM and concurrent allocation-hook behavior remain unclaimed. | 195 | Long-horizon deferral | Medium | Allocator/platform owner | Define allocator policy, concurrency semantics, platform evidence, and stress/sanitizer validation before documenting claims. |
| E17-RQ-024 | No hosted CI lane owns the symbolic allocation-failure gate. | 195 | Validation/tooling follow-up | Medium | Future CI owner | Add a reviewed hosted lane for `make symbolic-allocation-failure-gate` or keep the gate explicitly local-only in support/readiness wording. |
| E17-RQ-025 | Hosted generated API publication remains unselected. | 187 | Long-horizon deferral | Low | Product/docs publication owner | Select hosted publication, define freshness, artifact retention, deployment, versioning, and claim ownership before advertising hosted generated API docs. |
| E17-RQ-026 | Unqualified state-of-the-art sparse linear algebra status remains unearned. | 187, 192, 194 | Long-horizon deferral | High | Product/research/performance owner | Close broad algorithmic, ecosystem, performance, portability, packaging, reliability, and documentation evidence gaps before making any state-of-the-art claim. |

### Deduplication Decisions

| Merged queue ID | Source residuals merged | Reason |
| --- | --- | --- |
| E17-RQ-001 | Sprint 188 missing license metadata; Sprint 194 package-manager distribution remains unclaimed | Both block package-manager support promotion on provider-proof and metadata evidence. |
| E17-RQ-003 | Sprint 187 broad Windows non-goal; Sprint 194 broad Windows parity residual | Both require per-surface Windows evidence rather than selected-lane inference. |
| E17-RQ-004 | Sprint 189 local `pwsh` absent; Sprint 190 local `pwsh` absent; Sprint 194 local PowerShell unavailable | Same local environment residual and same accepted exit `2` semantics. |
| E17-RQ-005 | Sprint 190 `R186-WIN-REPORT-FRESHNESS`; source manifest Windows metadata; hosted selected artifact evidence | Same selected Cholesky promotion decision and evidence review. |
| E17-RQ-012 | Sprint 187 portable performance non-goal; Sprint 192 portable performance residual; Sprint 194 portable performance residual | Same missing multi-platform, repeated, variance-aware methodology proof. |
| E17-RQ-022 | Sprint 195 symbolic LU, analyze, etree helper, direct solver, and matrix construction residuals | Same selected-owner reliability process, but future planning should split them into one owner at a time. |

### Priority Rationale

High-priority residuals either affect prominent user/support claims
(package-manager, Windows, dynamic ABI, portable performance), complete an
already narrowed selected lane (Windows selected Cholesky freshness), prove a
high-risk reliability family, or govern the state-of-the-art non-claim.

Medium-priority residuals are valuable but more bounded: selected QR Windows
freshness, platform benchmark freshness, QR review-surface cleanup, local
PowerShell evidence, Markdown link checking, hosted allocation-failure gate
ownership, and allocator/concurrency semantics.

Low-priority residuals are retained but should not drive near-term planning
unless adjacent work is already changing the same surface: optional package
baselines, unselected CSV publication, helper-size cleanup, declaration-adjacent
header narrative, hosted generated API publication, and historical warning
hygiene.

### Day 3 Inputs To 196.6

- Next-epic candidates are: E17-RQ-001, E17-RQ-005, E17-RQ-006, E17-RQ-013,
  E17-RQ-016, and E17-RQ-022.
- Validation/tooling follow-ups are: E17-RQ-004, E17-RQ-009, E17-RQ-010,
  E17-RQ-017, E17-RQ-020, and E17-RQ-024.
- Documentation-only follow-ups are: E17-RQ-014, E17-RQ-018, and E17-RQ-021.
- Long-horizon deferrals are: E17-RQ-002, E17-RQ-003, E17-RQ-007,
  E17-RQ-008, E17-RQ-011, E17-RQ-012, E17-RQ-015, E17-RQ-023, E17-RQ-025,
  and E17-RQ-026.
- Out-of-scope historical notes are: E17-RQ-019.

### Day 3 Validation

- Re-read Sprint 196 Day 3 plan requirements.
- Extracted residuals from Sprint 187-195 retrospective residual sections.
- Cross-checked Day 14 closeout artifacts for handoff-only residuals and
  future-work notes.
- Searched Sprint 187-195 working notes for repeated residual, deferral,
  non-claim, and pending-evidence terms.
- No public claim files, project-plan status rows, production code, or public
  headers were edited on Day 3.

## Day 4: Claim Surface Audit

### Audit Method

Day 4 compared active user, maintainer, benchmark, API, corpus, packaging, and
public-header wording against the Day 2 outcome ledger and Day 3 residual
queue. The audit classified wording as accurate, too broad, stale, duplicated,
underclaimed, or requiring an evidence link before Sprint 196 claim edits.

### Claim-Surface Audit

| Surface | Audience | Current assessment | Evidence or residual tie | Day 5+ action |
| --- | --- | --- | --- | --- |
| `README.md` | Public first-read | Mostly accurate and evidence-bounded, but report freshness, hosted performance, Windows, package, and install paragraphs are dense and repeated. One Windows sentence near the CMake install section has awkward line wrapping that could hide the boundary. | Day 2 Windows/package/performance ledger; E17-RQ-001, E17-RQ-003, E17-RQ-005, E17-RQ-012, E17-RQ-026. | Tighten public wording, preserve selected-lane boundaries, and route support detail to `INSTALL.md#support-readiness-matrix`. |
| `INSTALL.md` | Public install/support truth | Strongest current support/readiness owner. Matrix accurately separates supported, validated, hosted-evidence, local-only, deferred, and not-claimed surfaces. | Sprint 194 adoption/API closeout; Day 3 package, Windows, ABI, performance, and state-of-the-art residuals. | Preserve as canonical user support surface; update only if Day 5 finds stale hosted/promotion language. |
| `docs/maintainer_guide.md` | Maintainer evidence interpretation | Accurate and detailed, but high duplication with README, INSTALL, corpus, and benchmark docs. It should remain the detailed owner for validation interpretation, not public workflow prose. | Sprint 188-195 evidence owners; E17-RQ-004, E17-RQ-005, E17-RQ-009, E17-RQ-017, E17-RQ-024. | Keep detailed gates here; later edits should reduce public-doc duplication rather than remove maintainer specificity. |
| `benchmarks/README.md` | Public/maintainer benchmark interpretation | Accurate threshold-free selected-performance wording. It clearly rejects portable speed and broad hosted performance claims. | Sprint 192 closeout; E17-RQ-011, E17-RQ-012, E17-RQ-013, E17-RQ-014, E17-RQ-015. | Preserve methodology-bound wording; add or tighten links to residual queue only if Day 5 touches benchmark claims. |
| `docs/api_reference.md` | Public API index | Accurate local-only generated API and non-claim wording. | Sprint 179/194 docs evidence; E17-RQ-021, E17-RQ-025. | Preserve local-only generated API boundary; avoid claiming hosted API publication or completeness beyond selected headers. |
| `docs/solver_selection.md` | Public solver workflow routing | Accurate but dense in selected comparison sections; repeatedly lists target-specific non-claims. | Sprint 191 comparison evidence; Sprint 194 diagnostics coherence; E17-RQ-006, E17-RQ-008. | Consider link-based consolidation only if it does not hide fixture-local claim boundaries. |
| `docs/cookbook.md` | Public task routing | Accurate and concise. It routes support/readiness to INSTALL and treats reports/benchmarks as evidence tools. | Sprint 194 adoption/API evidence. | No required Day 5 edit unless README/INSTALL wording changes need matching user-doc routing. |
| `docs/tutorial.md` | Public learning path | Accurate. It separates local tutorial usage from installed support and benchmark/report evidence. | Sprint 194 adoption/API evidence; E17-RQ-012, E17-RQ-026. | No required Day 5 edit unless public support wording changes. |
| `examples/README.md` | Public example routing | Accurate. It routes installed-consumer setup to INSTALL and benchmark interpretation to benchmark docs. | Sprint 194 installed-consumer tutorial evidence. | No required Day 5 edit unless support matrix anchors change. |
| `tests/corpus/README.md` | Maintainer/report owner | Accurate and detailed for selected target manifest, hosted lane, and freshness non-claims. | Sprints 190-192; E17-RQ-005, E17-RQ-006, E17-RQ-009, E17-RQ-013. | Keep as report-target authority; later public docs should link here rather than copy row details. |
| `tests/corpus/schemas/report_index_fields.md` | Maintainer/schema owner | Accurate for selected manifest semantics and release-proof non-claim. | Sprints 181/190/192 report-index work. | No immediate edit required. |
| `packaging/homebrew/README.md` | Maintainer/package proof owner | Needs Day 5/6 review only if package wording is recalibrated; it is the package-proof detail owner, not a public install promise. | Sprint 188 closeout; E17-RQ-001. | Preserve proof-only language and missing-license blocker. |
| `include/*.h` | Public declaration-adjacent contracts | Mostly accurate declaration contracts. Headers still contain some detailed support caveats by design; changing them would trigger full C/header validation. | Sprint 194 header narrative cleanup; E17-RQ-021. | Avoid header edits unless a concrete overclaim is found; if touched, run full C quality gates. |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Planning/status owner | Not yet updated for final item status. | Day 2 outcome ledger; Day 7 planned project-plan status work. | Defer edits until Day 7. |
| `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` | Final epic closeout owner | Does not exist yet. | Day 2/3 ledgers and future validation record. | Create after claim/status calibration and validation evidence exist. |

### Overclaim, Underclaim, Stale, And Duplication Table

| Class | Surface | Finding | Required calibration |
| --- | --- | --- | --- |
| Overclaim risk | README Windows/report freshness paragraphs | Wording is bounded, but dense phrasing could be read as broader Windows report evidence if users skim past the hosted-evidence/manifest-promotion caveat. | Shorten and make the selected Cholesky-only caveat more visible. |
| Overclaim risk | README hosted selected-performance paragraph | Accurate, but close proximity to benchmark commands can still invite portable-performance inference. | Keep `threshold-free`, `selected row only`, and `not portable speed evidence` in the same paragraph. |
| Overclaim risk | INSTALL Windows selected Cholesky row | The status `hosted-evidence` is acceptable only if paired with selected-lane and non-broad freshness language. | Confirm whether Sprint 190/PR follow-up hosted evidence and manifest status justify current status; otherwise downgrade wording. |
| Overclaim risk | Public headers | Header comments mention support and selected behavior in declaration-adjacent text. | Avoid additional support claims in headers; move only workflow narrative if future edits touch them. |
| Underclaim risk | README reliability section | Selected symbolic allocation-failure proof is present but can be hard to find relative to broader test descriptions. | Preserve selected-owner scope while making the proof discoverable from support/readiness or validation sections. |
| Underclaim risk | README/INSTALL selected Cholesky Windows path | If hosted evidence and metadata promotion landed after Sprint 190, current "guarded path" wording may understate the final state. | Day 5 must verify current manifest/workflow/docs before either retaining guarded wording or claiming reviewed selected Windows evidence. |
| Stale risk | README Sprint 182 deferral reference | README still references the Sprint 182 deferral record for all other Windows report freshness. This is likely accurate but must be reconciled with Sprint 190 and PR review fixes. | Keep only if selected Cholesky remains the sole exception and manifest metadata confirms it. |
| Stale risk | Maintainer guide Sprint 180/Sprint 182/Sprint 190 cross-references | Detailed sprint provenance is useful but can drift from active support truth. | Preserve provenance only where it explains active guard behavior; route current user truth through INSTALL. |
| Duplication | README, INSTALL, maintainer guide, corpus README | Selected report freshness and Windows non-claims are repeated across several docs. | Keep manifest/corpus/maintainer detail as authority; make README summary shorter. |
| Duplication | README, benchmarks README, maintainer guide | Selected performance methodology is repeated at several levels. | Keep benchmark docs as methodology owner; README should summarize and link. |
| Duplication | README, INSTALL, packaging Homebrew README, maintainer guide | Package-manager proof-only and missing-license blocker are repeated. | Keep INSTALL public support status and packaging/maintainer proof detail; README should route rather than restate. |
| Accurate | INSTALL support/readiness matrix | Matrix matches Day 2 and Day 3 boundaries for package, Windows, comparison, performance, reliability, API, ABI, and state-of-the-art. | Preserve as canonical public support owner. |
| Accurate | Benchmarks README | Selected performance lane is threshold-free and non-portable. | Preserve. |
| Accurate | API reference | Generated API HTML remains local-only and not hosted/release evidence. | Preserve. |
| Accurate | Corpus schema docs | Selected target manifest does not widen support claims or release proof. | Preserve. |

### Documentation Edit Plan

| Phase | Target files | Edit intent | Validation expectation |
| --- | --- | --- | --- |
| Day 5 public calibration | `README.md`, possibly `INSTALL.md`, `benchmarks/README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `examples/README.md` | Shorten public claim text, preserve selected non-claims, route support truth to INSTALL, and avoid widening package/Windows/performance/reliability claims. | `git diff --check`; docs/API guards selected by touched files; no full C gate unless headers/code change. |
| Day 6 maintainer/API calibration | `docs/maintainer_guide.md`, `docs/api_reference.md`, `tests/corpus/README.md`, `tests/corpus/schemas/report_index_fields.md`, `packaging/homebrew/README.md` | Keep maintainer evidence details accurate and synchronized with public wording without making maintainer docs the public front door. | `make docs-check`, relevant Python schema/guard tests if report/package wording changes. |
| Day 7 project-plan status | `docs/planning/EPIC_17/PROJECT_PLAN.md` | Mark Sprint 187-196 statuses with evidence links and residual states. | `git diff --check`; targeted grep for all sprint item IDs. |
| Day 8-10 retrospective/residual docs | Epic retrospective and residual queue artifacts | Publish final outcomes, non-claims, residual priorities, and state-of-the-art assessment. | Documentation hygiene plus claim-surface grep. |
| Day 11-14 validation closeout | Validation artifacts and retrospectives | Record final command results and closeout evidence. | Surface-driven focused gates plus full gates if actual changes require them. |

### Evidence-Link Requirements By Document

| Document | Must link or route to | Must not imply |
| --- | --- | --- |
| `README.md` | `INSTALL.md#support-readiness-matrix`, benchmark docs, corpus manifest/docs, maintainer guide for proof interpretation. | Package-manager support, broad Windows report freshness, portable performance, broad parity, release readiness, state-of-the-art status. |
| `INSTALL.md` | Support/readiness matrix, static install, CMake install, package proof details, Windows support interpretation. | Shared-library support, dynamic ABI, Windows `pkg-config` execution parity, provider/package-manager distribution. |
| `docs/maintainer_guide.md` | Exact gate owners, selected target manifest, proof scripts, validator scripts, sprint evidence. | Public support promotion without matching user-doc support matrix and guard evidence. |
| `benchmarks/README.md` | Selected benchmark target ID, methodology metadata, threshold-free policy, generated-output locations. | Portable speed, timing threshold, release benchmark proof, broad hosted performance. |
| `docs/api_reference.md` | Public headers and local Doxygen freshness. | Hosted generated API publication or completeness beyond selected checked-in headers. |
| `tests/corpus/README.md` | Selected target manifest and freshness command ownership. | Broad report freshness, unselected family pass evidence, or broad platform support. |
| `packaging/homebrew/README.md` | Local proof script and missing-license blocker. | Homebrew/core, bottle, tap, Linuxbrew, or broad provider support. |

### Day 4 Acceptance Evidence For 196.2

- Claim recalibration targets are tied to Day 2 evidence and Day 3 residual
  IDs.
- Public surfaces and maintainer/report/proof-owner surfaces are separated by
  audience.
- No state-of-the-art, performance, platform, package, or reliability claim was
  widened during audit.
- The Day 4 artifact
  `docs/planning/EPIC_17/SPRINT_196/artifacts/day4-claim-surface-audit.md`
  records the audit in reviewable form.

### Day 4 Validation

- Re-read Sprint 196 Day 4 plan requirements.
- Searched maintained public, maintainer, benchmark, API, corpus, packaging,
  and public-header surfaces for support, package, Windows, PowerShell,
  comparison, performance, reliability, release, parity, and state-of-the-art
  wording.
- Inspected high-risk README, INSTALL, maintainer guide, and benchmark sections
  directly.
- No public claim files, project-plan status rows, production code, or public
  headers were edited on Day 4.

## Day 5: Public Documentation Recalibration

### Public Claim Edits

| File | Edit | Evidence basis | Retained non-claims |
| --- | --- | --- | --- |
| `README.md` | Reframed the Windows selected Cholesky language as one guarded workflow path, not promoted selected freshness while selected-target manifest metadata still omits Windows platform promotion. | Day 2 Sprint 190 outcome ledger; Day 3 E17-RQ-005; current selected target manifest row for `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`. | No broad Windows report freshness, selected oracle freshness on Windows, selected benchmark freshness on Windows, or broad platform parity. |
| `README.md` | Split the normalized report-index paragraph so manifest ownership and selected Windows caveats are easier to see. | Day 4 README overclaim-risk audit. | No release proof, no unselected local-only family promotion, no generated report freshness from PowerShell validation. |
| `README.md` | Reflowed the CMake install support paragraph so Windows CMake-first, PowerShell validation ownership, guarded selected Cholesky workflow, and Sprint 182 residual scope are readable. | Day 4 stale/underclaim risk for awkward wrapping. | No Windows Makefile parity, Windows `pkg-config` execution parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. |
| `INSTALL.md` | Changed the support/readiness matrix row from `Windows selected Cholesky comparison freshness` / `hosted-evidence` to `Windows selected Cholesky comparison workflow` / `guarded-workflow`. | Current manifest still lists Cholesky comparison support as `local_only` with `linux;macos` workflow platforms; Windows workflow exists separately. | No promoted Windows selected freshness until hosted evidence, selected metadata, support tier, and claim contract are reviewed together. |
| `INSTALL.md` | Reworded the Windows platform row to describe a guarded selected Cholesky workflow path and separate selected freshness promotion from hosted evidence plus manifest metadata. | Day 2 Sprint 190 residual-narrowed state and Day 3 E17-RQ-005. | No broad Windows report freshness or broad Windows parity. |

### Public Docs Left Unchanged

| File | Reason |
| --- | --- |
| `benchmarks/README.md` | Existing selected-performance wording is already threshold-free, selected-row scoped, and non-portable. |
| `docs/solver_selection.md` | Dense but accurate selected comparison caveats remain better handled during maintainer/report calibration unless a concrete public overclaim appears. |
| `docs/cookbook.md` | Already routes support/readiness to INSTALL and treats benchmarks/reports as evidence tools. |
| `docs/tutorial.md` | Already separates local tutorial usage from install/support and report evidence. |
| `examples/README.md` | Already points installed consumers to INSTALL and benchmark interpretation to benchmark docs. |
| `include/*.h` | No concrete header overclaim was found, and avoiding header edits keeps Day 5 out of full C/header gate scope. |

### Support/Readiness Interpretation Update

The public support matrix now distinguishes a `guarded-workflow` from
`hosted-evidence`. `guarded-workflow` means the workflow path exists and has
structural/command ownership, but the selected report target manifest has not
yet promoted the platform support tier and claim contract. This preserves the
Sprint 190 narrowed residual instead of turning it into support by wording.

### Day 5 Retained Non-Claims

- Package-manager distribution and Homebrew install support remain unclaimed.
- Shared-library packaging and dynamic ABI support remain deferred.
- Windows support remains CMake-first and selected-lane bounded.
- Windows selected Cholesky comparison remains a guarded workflow path until
  hosted evidence and manifest promotion are reviewed together.
- Selected comparison and benchmark evidence remain fixture/row scoped.
- Hosted selected performance remains threshold-free and non-portable.
- Selected allocation-failure proof remains selected-owner scoped.
- Release readiness and unqualified state-of-the-art status remain unclaimed.

### Day 5 Validation Results

Because Day 5 changed public docs that are checked by claim-boundary guards,
the required validation is broader than whitespace-only hygiene but still does
not require the full C quality gate unless `.c` or `.h` files change.

- `git diff --check`: passed.
- `make windows-powershell-guard`: passed after preserving exact
  claim-boundary marker phrases while keeping the new guarded-workflow
  interpretation.
- `bash scripts/package_manager_deferral_check.sh`: passed after keeping the
  lowercase `package-manager support` README non-claim marker.
- `bash scripts/static_package_deferral_check.sh`: passed.
- `python3 tests/test_selected_report_targets_manifest.py`: passed.
- `python3 tests/test_selected_performance_docs.py`: passed.
- `python3 tests/test_normalize_report_index.py`: passed.
- `make docs-check`: passed; generated `scripts/__pycache__/` files were
  removed after the run.

No production code or public headers were edited on Day 5.

## Day 6: Maintainer and API Documentation Recalibration

### Maintainer/API Claim Edits

| File | Edit | Evidence basis | Retained boundary |
| --- | --- | --- | --- |
| `docs/maintainer_guide.md` | Added an Epic 17 evidence ownership table covering package, Windows/PowerShell, selected comparison, selected performance, review-surface, adoption/API, and reliability proof surfaces. | Day 2 outcome ledger; Day 3 residual queue; Sprint 187-195 closeout artifacts. | Maintainer guide interprets evidence owners; executable truth remains in Makefile, scripts, workflows, tests, and headers. |
| `docs/maintainer_guide.md` | Updated normalized-report and selected-comparison sections to describe Sprint 190 Windows Cholesky as guarded workflow evidence until hosted evidence, selected metadata, support tier, and claim contract are reviewed together. | Day 5 public-doc calibration; current selected target manifest keeps Cholesky comparison `local_only` with `linux;macos` workflow platforms. | No broad Windows report freshness, selected oracle freshness on Windows, selected benchmark freshness on Windows, release proof, or state-of-the-art status. |
| `tests/corpus/README.md` | Aligned selected report freshness wording with the guarded Windows Cholesky workflow path and manifest-promotion requirement. | Day 5 INSTALL support/readiness matrix; Day 3 E17-RQ-005. | Corpus docs remain the report/manifest authority and do not turn workflow presence into selected platform promotion. |
| `docs/api_reference.md` | Audited and left unchanged. | Existing generated HTML section already says generated API HTML is local-only and not hosted/source-controlled/release evidence. | No hosted generated API publication, package-manager distribution, dynamic ABI compatibility, broad Windows parity, or completeness beyond checked-in public headers selected by `Doxyfile`. |

### Maintainer Gate-Owner Map

| Evidence family | Owner surfaces | Required/focused gates |
| --- | --- | --- |
| Package/Homebrew proof | `INSTALL.md`, `packaging/homebrew/README.md`, package scripts, Sprint 188 artifacts | `bash scripts/package_manager_deferral_check.sh`, `bash scripts/static_package_deferral_check.sh`, `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh` |
| Windows/PowerShell ownership | `.github/workflows/windows-ci.yml`, `scripts/validate_windows_powershell.py`, selected target manifest, README/INSTALL/maintainer/corpus markers | `make windows-powershell-guard`, `make windows-powershell-validate`, hosted Windows `--require-pwsh` job |
| Selected comparison freshness | selected target manifest, corpus docs, runner/normalizer scripts, comparison tests | `make report-index-comparison-freshness`, `python3 tests/test_selected_report_targets_manifest.py`, `python3 tests/test_run_external_comparison.py`, `python3 tests/test_normalize_report_index.py` |
| Selected performance evidence | benchmark docs, selected target manifest, canonical freshness checker | `make bench-canonical-report-freshness`, `python3 tests/test_selected_performance_docs.py`, `python3 tests/test_bench_canonical_freshness.py` |
| Review-surface reduction | QR proof owner, helper header, helper guard scripts/tests | `make qr-external-ref-helper-guard`, `python3 tests/test_qr_external_ref_helper_guard.py`, full C gate after header/test changes |
| Adoption/API coherence | README, INSTALL, tutorial/cookbook/solver-selection docs, examples README, API reference, public headers | `make docs-check`, `make api-docs-freshness`, `make qr-header-docs-guard`, install checks; full C gate after header edits |
| Reliability/failure-path proof | selected implementation owner, focused tests, Make/CTest labels, README/INSTALL/maintainer wording | `make symbolic-allocation-failure-gate`, `python3 tests/test_symbolic_allocation_failure_gate_registration.py`, full C gate after code/header changes |

### Planning-Adjacent Notes

- Day 6 did not update `docs/planning/EPIC_17/PROJECT_PLAN.md`; Day 7 owns
  final item-status annotations.
- Day 6 did not create `EPIC_17_RETROSPECTIVE.md`; Day 8 starts retrospective
  outline and metrics after public/maintainer claim calibration.
- Day 6 did not edit public headers; no `.h` claim drift required a header
  change.

### Day 6 Validation Results

- `git diff --check`: passed.
- `make windows-powershell-guard`: passed; README, INSTALL, maintainer guide,
  and corpus README claim-boundary markers remain synchronized.
- `python3 tests/test_selected_report_targets_manifest.py`: passed.
- `python3 tests/test_normalize_report_index.py`: passed.
- `make docs-check`: passed.
- `make api-docs-freshness`: passed; generated API HTML remains ignored and
  local-only.
- Generated `scripts/__pycache__/` files from Python validation imports were
  removed after the run.
- No production code or public headers were edited on Day 6.

## Day 7: Project Plan Status Pass

### Project-Plan Status Edits

Day 7 updated `docs/planning/EPIC_17/PROJECT_PLAN.md` with closeout-status
tables for every Epic 17 sprint. The status pass follows the Epic 16 closeout
pattern: each sprint now has a table after its total estimate, and each item
row links to the most specific Sprint artifact or retrospective available.

| Sprint | Status summary | Evidence basis |
| --- | --- | --- |
| 187 | All six items marked Complete. | Sprint 187 closeout summary and retrospective. |
| 188 | Two package metadata/proof items marked Residualized or Deferred; proof hardening, guards, docs, and validation marked Complete or Complete with guarded residual. | Sprint 188 closeout summary and retrospective; Day 3 E17-RQ-001. |
| 189 | PowerShell ownership items marked Complete, with hosted CI wiring explicitly Complete with hosted evidence pending at closeout. | Sprint 189 closeout artifact and retrospective; Day 3 E17-RQ-004. |
| 190 | Selected Windows Cholesky work marked Complete, Narrowed, or Residualized depending on whether it was workflow evidence, manifest promotion, or claim calibration. | Sprint 190 closeout artifact and retrospective; Day 3 E17-RQ-005. |
| 191 | All six selected `qr-incompatible-ls` comparison items marked Complete. | Sprint 191 closeout and retrospective. |
| 192 | Five items marked Complete and regression policy marked Narrowed because the lane intentionally stayed threshold-free. | Sprint 192 closeout and retrospective; Day 3 E17-RQ-011 and E17-RQ-012. |
| 193 | All six selected QR review-surface reduction items marked Complete. | Sprint 193 closeout and retrospective. |
| 194 | All six adoption/API coherence items marked Complete. | Sprint 194 closeout handoff and retrospective. |
| 195 | All six selected symbolic Cholesky allocation-failure items marked Complete. | Sprint 195 closeout review package and retrospective. |
| 196 | At Day 7, items 196.1 and 196.3 were marked Complete; 196.2 was complete through current claim passes; 196.4 and 196.5 were pending; 196.6 was in progress pending final residual publication. | Sprint 196 Day 2-7 artifacts and current working notes. |

### Status Vocabulary

| Status | Meaning in Day 7 pass |
| --- | --- |
| Complete | The planned item landed with evidence and no retained blocker for the selected scope. |
| Complete with guarded residual | The selected guard/doc/proof work landed, but a clearly named blocker prevents broader support or promotion. |
| Complete with hosted evidence pending at closeout | Source-controlled workflow ownership landed, but hosted PR-CI evidence remained the promotion owner at that sprint closeout. |
| Complete with residual narrowed | The sprint narrowed a broader gap to one guarded path but did not close the broader support claim. |
| Narrowed | The final evidence intentionally covers less than the original item wording could imply. |
| Deferred | The implementation or promotion was not performed because a prerequisite decision or evidence source was missing. |
| Residualized | The item produced an explicit residual with closure conditions instead of final support/promotion. |
| Pending | Sprint 196 closeout item scheduled for later Day 8-14 work. |
| In progress | Partial Sprint 196 evidence exists, but final closeout publication is not complete. |

### Day 7 Item Count Snapshot

| Status family | Count |
| --- | ---: |
| Complete | 46 |
| Complete through public and maintainer/API passes | 1 |
| Complete with guarded residual | 2 |
| Complete with hosted evidence pending at closeout | 1 |
| Complete with residual narrowed | 2 |
| Narrowed | 2 |
| Deferred | 1 |
| Residualized | 2 |
| Pending | 2 |
| In progress | 1 |

The Day 7 count included all 60 Epic 17 project-plan rows as they stood before
Day 10 residual queue publication. Day 10 updates the current status by
marking 196.6 Complete; final Sprint 196 counts will need a Day 14 update
after integrated validation and retrospective finalization.

### Residual Boundaries Preserved

- Sprint 188 status keeps Homebrew/package-manager support blocked by
  approved license metadata and exact formula license identifier.
- Sprint 190 status keeps Windows selected Cholesky as guarded workflow
  evidence until hosted evidence, selected metadata, support tier, and claim
  contract are promoted together.
- Sprint 192 status keeps selected performance evidence threshold-free and
  non-portable.
- Sprint 195 status keeps reliability proof scoped to selected
  `sparse_symbolic_cholesky()` output allocation.
- Sprint 196 status does not pre-claim final integrated validation or Epic
  retrospective completion.

### Day 7 Validation Results

- `git diff --check`: passed.
- `python3 tests/test_selected_report_targets_manifest.py`: passed.
- `python3 tests/test_selected_performance_docs.py`: passed.
- `make docs-check`: passed.
- No `.c` or `.h` files were edited on Day 7, so the full C quality gate is
  not required for this day.

## Day 8: Epic Retrospective Outline and Metrics

### Retrospective Structure

Day 8 created a draft outline at
`docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md`. The outline follows the Epic
15 and Epic 16 retrospective structure while preserving draft markers for
sections that require Day 9-14 evidence before final wording.

Planned final sections:

- Epic objective.
- Sprint outcomes.
- Major outcomes.
- Project-plan status.
- Validation evidence.
- Changed surface.
- Earned claims.
- Non-claims.
- Residual queue.
- State-of-the-art assessment.
- What went well.
- Could be better.
- Key deliverables.

### Initial Metrics Sourced

| Metric | Source |
| --- | --- |
| Sprint 187-196 plan coverage | `PROJECT_PLAN.md`; Sprint 196 Day 7 status pass. |
| Sprint 187-195 retrospective coverage | `find docs/planning/EPIC_17 -maxdepth 2 -name RETROSPECTIVE.md`. |
| Current project-plan status snapshot | `SPRINT_196/artifacts/day7-project-plan-status.md`, corrected on Day 8 to total 60 project-plan rows and updated on Day 10 for completed residual publication. |
| 49 library sources | Sprint 193, 194, and 195 retrospective validation tables. |
| 18 checked-in public headers, 18 generated reference pages, 18 generated source pages | Sprint 194 retrospective and Sprint 196 Day 6-7 `make docs-check` output. |
| 54 selected oracle freshness rows | Sprint 194 retrospective. |
| 46 selected comparison freshness rows | Sprint 191 and Sprint 194 retrospectives. |
| One selected Linux hosted performance lane | Sprint 192 retrospective. |
| One selected C implementation owner changed for reliability proof | Sprint 195 retrospective. |
| Six public headers changed for adoption/API cleanup | Sprint 194 retrospective. |

### Missing Evidence Checklist

| Missing or not final yet | Owner day |
| --- | --- |
| First complete retrospective prose replacing TODO markers | Day 9 |
| Integrated validation result table for Sprint 196 changed surfaces | Day 11 |
| Final state-of-the-art assessment after residual queue publication | Day 12 |
| Final key deliverable links and closeout summary | Day 13 complete |
| Final project-plan status/count reconciliation after all closeout work | Day 14 complete |

### Day 8 Validation Results

- `git diff --check`: passed.
- `make docs-check`: passed.
- No `.c` or `.h` files were edited on Day 8, so the full C quality gate is
  not required for this day.

## Day 9: Epic Retrospective Draft

### Retrospective Draft Work

Day 9 replaced the Day 8 retrospective outline with a complete draft at
`docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md`. The draft now has prose and
tables for the Epic objective, sprint outcomes, major outcomes, current
project-plan status, validation evidence, changed surface, earned claims,
non-claims, residual queue preview, state-of-the-art assessment,
retrospective themes, open questions, and key deliverables.

The draft remains explicitly non-final because Sprint 196 still has scheduled
Day 10-14 work for residual queue publication, integrated validation, full
validation decision, final key-deliverable links, and project-plan count
reconciliation.

### Evidence Used

| Evidence source | Retrospective section informed |
| --- | --- |
| `SPRINT_196/artifacts/day2-outcome-ledger.md` | Epic objective, sprint outcomes, major outcomes, earned claims, non-claims. |
| `SPRINT_196/artifacts/day3-residual-triage.md` | Residual queue preview, non-claims, state-of-the-art assessment, open questions. |
| `SPRINT_196/artifacts/day4-claim-surface-audit.md` | Public and maintainer claim-boundary wording. |
| `SPRINT_196/artifacts/day5-public-claim-recalibration.md` | Package, Windows, selected performance, selected comparison, and public support boundaries. |
| `SPRINT_196/artifacts/day6-maintainer-api-recalibration.md` | Maintainer evidence ownership, API local-only boundary, and gate ownership. |
| `SPRINT_196/artifacts/day7-project-plan-status.md` | Project-plan status table and current status counts. |
| `SPRINT_196/artifacts/day8-retrospective-outline-and-metrics.md` | Retrospective structure and initial metric sources. |
| Sprint 187-195 retrospectives | Validation anchors, changed-surface metrics, closed-claim language, residuals, and retrospective themes. |

### Open Questions Preserved

| Question | Owner day |
| --- | --- |
| Which focused gates should form the Sprint 196 integrated validation bundle after the residual queue is added? | Day 11 |
| Is any full C quality gate required by Sprint 196 final edits, or are all remaining changes documentation-only? | Day 12 |
| What final project-plan status counts should replace the current draft counts once 196.4-196.6 are complete? | Day 13 complete |
| Should the final retrospective keep any draft caveat language after final validation passes? | Day 13 complete |

### Day 9 Validation Results

- `git diff --check`: passed.
- `make docs-check`: passed.
- No `.c` or `.h` files were edited on Day 9, so the full C quality gate is
  not required for this day.

## Day 10: Prioritized Residual Queue

### Residual Queue Publication

Day 10 published the Epic 17 residual handoff in
`docs/planning/EPIC_17/EPIC_17_RESIDUAL_QUEUE.md`. The queue converts the Day
3 triage into a next-epic planning artifact with priorities, owner surfaces,
closure targets, expected evidence, validation commands, claim boundaries,
validation/tooling follow-ups, documentation-only follow-ups, long-horizon
deferrals, and an out-of-scope historical note.

The Day 10 pass also linked the residual queue from:

- `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md`;
- `docs/planning/EPIC_17/PROJECT_PLAN.md`.

### Current Project-Plan Status Snapshot

| Status family | Count |
| --- | ---: |
| Complete | 50 |
| Complete with guarded residual | 2 |
| Complete with hosted evidence pending at closeout | 1 |
| Complete with residual narrowed | 2 |
| Narrowed | 2 |
| Deferred | 1 |
| Residualized | 2 |

The current count includes all 60 Epic 17 project-plan rows after marking
196.2, 196.4, 196.5, and 196.6 Complete. Day 14 closeout packaging did not
change item status categories.

### Priority Rationale

| Priority | Residual ID | Rationale |
| ---: | --- | --- |
| 1 | E17-RQ-001 | Package-manager/Homebrew support is user-visible and blocked by one explicit product/legal metadata decision. |
| 2 | E17-RQ-005 | Selected Cholesky Windows freshness was already narrowed to one guarded workflow path and needs hosted evidence/manifest promotion review. |
| 3 | E17-RQ-022 | Additional allocation-failure owner proof has a reusable Sprint 195 pattern and high reliability value. |
| 4 | E17-RQ-016 | Additional QR review-surface reduction is actionable if one cluster is selected and guarded. |
| 5 | E17-RQ-013 | Windows/macOS selected benchmark freshness extends a selected lane but requires hosted platform evidence. |
| 6 | E17-RQ-006 | Windows QR incompatible freshness is useful but depends on MSVC/CMake proof and exact metadata promotion. |

### Publication Boundaries

- Long-horizon deferrals are listed but not treated as implementation-ready
  sprint work.
- Documentation-only follow-ups are separated from next-epic implementation
  candidates.
- Validation/tooling follow-ups identify exact owner conditions and avoid
  reclassifying environment residuals as passes.
- The residual queue does not promote package-manager, Windows, performance,
  ABI, release, generated API, or state-of-the-art claims.

### Day 10 Validation Results

- `git diff --check`: passed.
- `make docs-check`: passed.
- No `.c` or `.h` files were edited on Day 10, so the full C quality gate is
  not required for this day.

## Day 11: Integrated Focused Validation

### Focused Gate Results

Day 11 ran the focused Epic 17 evidence-owner gates across package/install,
Windows/PowerShell, selected report targets, selected comparison freshness,
selected oracle freshness, selected performance freshness, review-surface
guards, reliability guards, corpus/schema checks, and docs/API freshness.

| Gate | Result | Notes |
| --- | --- | --- |
| `bash scripts/package_manager_deferral_check.sh` | Passed | Package-manager support non-claims and selected Homebrew proof boundary stayed intact. |
| `bash scripts/static_package_deferral_check.sh` | Passed | Static-first package and shared-library/dynamic ABI non-claims stayed intact. |
| `bash tests/test_install.sh` | Passed | 23 install checks passed, 0 failures. |
| `bash tests/test_cmake_install.sh` | Passed | 27 CMake install checks passed, 0 failures, 0 skips. |
| `make windows-powershell-guard` | Passed | Structural and claim-boundary checks passed; local `pwsh` unavailable path remained residual evidence, not pass evidence. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus manifest/schema validation passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target manifest validation passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Selected comparison workflow guard passed. |
| `python3 tests/test_normalize_report_index.py` | Passed | Report-index normalizer regression tests passed. |
| `python3 tests/test_run_external_comparison.py` | Passed | External comparison runner tests passed. |
| `make report-index-comparison-freshness` | Passed | Local-only generated comparison freshness passed with 46 rows. |
| `make report-index-oracle-freshness` | Passed | Local-only generated oracle freshness passed with 54 rows. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Selected performance docs guard passed. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Selected benchmark freshness regression tests passed. |
| `make bench-canonical-report-freshness` | Passed on rerun | First parallel run failed before `build/include/sparse_version.h` existed; after normal generated-header creation, rerun passed. |
| `python3 tests/test_qr_external_ref_helper_guard.py` | Passed | QR external-reference helper guard tests passed. |
| `make qr-external-ref-helper-guard` | Passed | QR helper ownership guard passed. |
| `make qr-header-docs-guard` | Passed | QR header/docs coherence guard passed. |
| `make ldlt-csc-helper-guard` | Passed | LDLT CSC helper ownership guard passed. |
| `python3 tests/test_symbolic_allocation_failure_gate_registration.py` | Passed | Symbolic allocation-failure registration guard passed. |
| `make symbolic-allocation-failure-gate` | Passed on rerun | First parallel run failed before `build/include/sparse_version.h` existed; after normal generated-header creation, rerun passed with 101 tests, 0 failures, 0 skips, 1262 assertions. |
| `make api-docs-freshness` | Passed | Generated API coverage and local-only guard passed. |
| `git diff --check` | Passed | Whitespace hygiene passed. |

### Precondition Failure Handling

The first parallel run of `make bench-canonical-report-freshness`,
`make report-index-comparison-freshness`, and
`make symbolic-allocation-failure-gate` failed immediately with:

`fatal error: 'sparse_version.h' file not found`

The failure was a clear generated-header build precondition exposed by running
multiple Make gates concurrently from the current build state. The normal
generated header was then present under `build/include/sparse_version.h`, and
all three gates passed on rerun. No source or documentation edits were needed
to fix the failure.

### Environment Residuals

- Local `pwsh` is unavailable. `make windows-powershell-guard` passed because
  the guard validates structural ownership and explicitly keeps local
  unavailable PowerShell separate from pass evidence.
- Hosted Windows evidence remains outside this local Day 11 validation pass
  and remains a future promotion owner for selected Windows freshness claims.
- Generated API HTML and generated report outputs remain ignored local
  artifacts, not source-controlled evidence.

### Day 11 Validation Results

- Focused Epic 17 evidence-owner gates passed as listed above.
- `git diff --check`: passed.
- No `.c` or `.h` files were edited on Day 11, so the full C quality gate is
  not required for this day. Day 12 still owns the final full-quality decision
  for Sprint 196 closeout.

## Day 12: Full Quality Gate and Decision

### Full-Quality Decision

Day 12 ran the full-quality gates required for the current Sprint 196 changed
surface. The branch has no `.c` or `.h` diffs, and `make format` did not
create any `.c` or `.h` changes. Therefore `make test` is not required by the
user's stated quality-check rule for this documentation/planning-only sprint
branch.

| Gate | Result | Notes |
| --- | --- | --- |
| `git diff --name-only -- '*.c' '*.h'` | Passed | No C implementation, C test, or header files are modified in the branch diff. |
| `make format` | Passed | Clang-format completed across configured C/header sources and produced no `.c`/`.h` diff. |
| `make lint` | Passed | Built tooling/example binaries, ran strict warning syntax checks, completed clang-tidy across 49 library sources, and completed cppcheck across 109 source/test paths. |
| `make test` | Not required | No `.c` or `.h` files changed; Day 11 focused reliability, report, install, docs, and ownership gates already passed for changed evidence surfaces. |

### Environment and Generated Output Notes

- `build/` contains generated local build, benchmark, comparison, and corpus
  outputs from Day 11-12 validation and remains ignored.
- `docs/api/html/` contains generated Doxygen output from docs/API validation
  and remains ignored.
- Local `pwsh` is still unavailable; this remains the same environment
  residual recorded on Day 11 and in the residual queue.

### Day 12 Validation Results

- `git diff --check`: passed.
- `make format`: passed.
- `make lint`: passed.
- `make docs-check`: passed after Day 12 documentation updates.
- No `.c` or `.h` files were edited on Day 12, so `make test` was not
  required.

## Day 13: Final Claim and Retrospective Review

### Review Scope

Day 13 re-read the current public, maintainer, planning, retrospective,
residual, benchmark, corpus, and API-adjacent documentation for consistency:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`
- `docs/api_reference.md`
- `tests/corpus/README.md`
- `docs/planning/EPIC_17/PROJECT_PLAN.md`
- `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md`
- `docs/planning/EPIC_17/EPIC_17_RESIDUAL_QUEUE.md`

### Claim Review Results

The high-risk claim review checked package-manager, Homebrew, broad Windows,
external-library parity, portable performance, reliability, release, ABI,
shared-library, generated API, and state-of-the-art language. The reviewed
matches were either bounded earned claims or explicit retained non-claims.

| Claim area | Day 13 result |
| --- | --- |
| Package-manager and Homebrew | Support remains unclaimed until approved root license metadata, exact formula license metadata, successful proof output, guards, and docs land together. |
| Windows | The docs retain a validated MSVC CMake install/downstream path and one guarded selected Cholesky workflow path without broad Windows parity or promoted selected freshness. |
| External comparisons | Selected comparison language remains fixture/target scoped and does not claim broad SuiteSparse, PETSc, Trilinos, Eigen, SciPy, LAPACK, NumPy, or ecosystem parity. |
| Performance | The selected hosted benchmark lane remains threshold-free and methodology-bound, without portable performance, superiority, release, or state-of-the-art claims. |
| Reliability | Allocation-failure proof remains selected to named owners and does not claim broad OOM, concurrency, direct-solver, generated-tooling, package, or install reliability. |
| API and ABI | Public headers remain declaration sources; generated HTML is local-only; shared-library and dynamic ABI support remain deferred. |
| Retrospective and residuals | The retrospective now links key deliverables, records current status counts, and points future work to the prioritized residual queue. |

### Fixes Made

- Marked project-plan items 196.2 and 196.5 complete with Day 13 evidence.
- Updated the Epic 17 retrospective status from draft to final claim-review
  complete.
- Replaced the last pending project-plan count with current Day 13 counts.
- Finalized retrospective key-deliverable links.
- Added `SPRINT_196/artifacts/day13-final-claim-retrospective-review.md`.

### Day 13 Validation Results

- `git diff --check`: passed.
- `make docs-check`: passed after Day 13 documentation updates.
- `git diff --name-only -- '*.c' '*.h'`: passed with no output.
- No `.c` or `.h` files were edited on Day 13, so `make test` was not
  required.

## Day 14: Epic Closeout and Review Package

### Closeout Summary

Day 14 packaged Sprint 196 and Epic 17 evidence for review. The pass confirmed
that Sprint 196 items 196.1 through 196.6 are complete with evidence links,
the Epic 17 retrospective and residual queue are published, and the remaining
Epic 17 limitations are explicit non-claims or residual queue items rather
than hidden support promises.

### Item-To-Evidence Traceability

| Item | Final status | Primary evidence |
| --- | --- | --- |
| 196.1 Evidence Reconciliation | Complete | `SPRINT_196/artifacts/day2-outcome-ledger.md` |
| 196.2 Claim Recalibration | Complete | `SPRINT_196/artifacts/day4-claim-surface-audit.md`, `day5-public-claim-recalibration.md`, `day6-maintainer-api-recalibration.md`, `day13-final-claim-retrospective-review.md` |
| 196.3 Project Plan Status | Complete | `SPRINT_196/artifacts/day7-project-plan-status.md`, `PROJECT_PLAN.md` closeout status tables |
| 196.4 Integrated Validation | Complete | `SPRINT_196/artifacts/day11-integrated-focused-validation.md`, `day12-full-quality-decision.md` |
| 196.5 Epic Retrospective | Complete | `EPIC_17_RETROSPECTIVE.md`, `day8-retrospective-outline-and-metrics.md`, `day9-epic-retrospective-draft.md`, `day13-final-claim-retrospective-review.md` |
| 196.6 Residual Queue | Complete | `EPIC_17_RESIDUAL_QUEUE.md`, `SPRINT_196/artifacts/day10-prioritized-residual-queue.md` |

### Review Checklist

| Review area | Day 14 result |
| --- | --- |
| Evidence reconciliation | Sprint 187-195 outcomes are reconciled and linked from the project plan and retrospective. |
| Claim recalibration | Public, maintainer, planning, retrospective, residual, benchmark, corpus, and API-adjacent surfaces retain bounded earned claims. |
| Project-plan status | Sprint 187-196 closeout status rows are present; Sprint 196 has no pending item rows. |
| Integrated validation | Day 11 focused gates and Day 12 full-quality decision are recorded with command results and residual context. |
| Epic retrospective | `EPIC_17_RETROSPECTIVE.md` records outcomes, validation evidence, changed surfaces, earned claims, non-claims, residuals, and state-of-the-art assessment. |
| Residual queue | `EPIC_17_RESIDUAL_QUEUE.md` prioritizes near-term candidates and separates validation/tooling, documentation-only, long-horizon, and historical residuals. |
| Non-claims | Package-manager support, broad Windows parity, external-library parity, portable performance, release readiness, shared-library/dynamic ABI support, hosted generated API publication, broad allocation-failure coverage, and unqualified state-of-the-art status remain unclaimed. |

### Final Residual Handoff

The near-term residual queue remains:

1. Package-manager/Homebrew support blocker.
2. Selected Cholesky Windows freshness promotion.
3. Additional allocation-failure owner.
4. Additional QR review-surface cluster.
5. Windows/macOS selected benchmark freshness.
6. Windows QR incompatible freshness.

Long-horizon residuals remain shared-library/dynamic ABI support, broad
Windows parity, optional package baselines, broader QR/external-library
parity, hosted timing thresholds, portable performance evidence, release
benchmark claims, OS OOM/concurrent allocation-hook behavior, hosted generated
API publication, and unqualified state-of-the-art sparse linear algebra
status.

### Day 14 Validation Results

- `git diff --check`: passed.
- `make docs-check`: passed after Day 14 documentation updates.
- `git diff --name-only -- '*.c' '*.h'`: passed with no output.
- No `.c` or `.h` files were edited on Day 14, so `make test` was not
  required.
