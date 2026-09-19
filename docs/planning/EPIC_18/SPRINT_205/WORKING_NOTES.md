# Sprint 205 Working Notes: Support Matrix and Adoption Quick-Reference Consolidation

**Sprint:** 205 - Support Matrix and Adoption Quick-Reference Consolidation  
**Branch:** `sprint-205`  
**Base commit:** `9a2b4ff6`  
**Plan:** [PLAN.md](./PLAN.md)  
**Epic source:** [EPIC_18/PROJECT_PLAN.md](../PROJECT_PLAN.md)

## Sprint Goal

Reduce public documentation friction by centralizing support truth and adding a
compact problem-shape quick reference without weakening claim boundaries.

## Item Checklist

| Item | Description | Status | Evidence path |
| --- | --- | --- | --- |
| 205.1 | Public Doc Audit | Complete for audit evidence | Day 1 intake surface map; [Day 2 public documentation audit](./artifacts/day2-public-doc-audit.md); [Day 3 maintainer/report audit](./artifacts/day3-maintainer-report-audit.md) |
| 205.2 | Quick Reference Design | Implemented | [Day 4 quick-reference design](./artifacts/day4-quick-reference-design.md); [Day 6 quick-reference implementation](./artifacts/day6-quick-reference-implementation.md) |
| 205.3 | Support Truth Consolidation | Complete | [Day 5 support-truth architecture](./artifacts/day5-support-truth-architecture.md); [Day 7 consolidation batch](./artifacts/day7-support-truth-consolidation.md); [Day 8 example/workflow routing](./artifacts/day8-example-workflow-routing.md) |
| 205.4 | Diagnostics Vocabulary | Complete | [Day 9 diagnostics vocabulary design](./artifacts/day9-diagnostics-vocabulary-design.md); [Day 10 diagnostics vocabulary implementation](./artifacts/day10-diagnostics-vocabulary-implementation.md) |
| 205.5 | Claim Guard Updates | Complete | [Day 11 claim guard design](./artifacts/day11-claim-guard-design.md); [Day 12 claim guard implementation](./artifacts/day12-claim-guard-implementation.md) |
| 205.6 | Validation | Complete | [Day 13 integrated validation](./artifacts/day13-integrated-validation.md); [Day 14 closeout review](./artifacts/day14-closeout-review.md) |

## Day Status Ledger

| Day | Title | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Support Intake | Complete | [day1-support-intake.md](./artifacts/day1-support-intake.md) |
| 2 | Public Audit | Complete | [day2-public-doc-audit.md](./artifacts/day2-public-doc-audit.md) |
| 3 | Maintainer Audit | Complete | [day3-maintainer-report-audit.md](./artifacts/day3-maintainer-report-audit.md) |
| 4 | Quick Reference Design | Complete | [day4-quick-reference-design.md](./artifacts/day4-quick-reference-design.md) |
| 5 | Support Truth Design | Complete | [day5-support-truth-architecture.md](./artifacts/day5-support-truth-architecture.md) |
| 6 | Quick Reference Implementation | Complete | [day6-quick-reference-implementation.md](./artifacts/day6-quick-reference-implementation.md) |
| 7 | Support Consolidation | Complete | [day7-support-truth-consolidation.md](./artifacts/day7-support-truth-consolidation.md) |
| 8 | Example Routing | Complete | [day8-example-workflow-routing.md](./artifacts/day8-example-workflow-routing.md) |
| 9 | Diagnostics Design | Complete | [day9-diagnostics-vocabulary-design.md](./artifacts/day9-diagnostics-vocabulary-design.md) |
| 10 | Diagnostics Implementation | Complete | [day10-diagnostics-vocabulary-implementation.md](./artifacts/day10-diagnostics-vocabulary-implementation.md) |
| 11 | Guard Design | Complete | [day11-claim-guard-design.md](./artifacts/day11-claim-guard-design.md) |
| 12 | Guard Implementation | Complete | [day12-claim-guard-implementation.md](./artifacts/day12-claim-guard-implementation.md) |
| 13 | Integrated Validation | Complete | [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md) |
| 14 | Closeout Review | Complete | [day14-closeout-review.md](./artifacts/day14-closeout-review.md) |

## Item-To-Surface Traceability

| Sprint item | Primary surfaces | Planned evidence |
| --- | --- | --- |
| 205.1 Public Doc Audit | `README.md`, `INSTALL.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/api_reference.md`, `examples/README.md`, `benchmarks/README.md` | Day 2 public audit plus Day 3 maintainer/report audit |
| 205.2 Quick Reference Design | README routing, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `INSTALL.md` support/readiness matrix | Day 4 design matrix and acceptance criteria |
| 205.3 Support Truth Consolidation | `INSTALL.md#support-readiness-matrix`, README package/support paragraphs, `docs/api_reference.md`, `docs/maintainer_guide.md`, examples and benchmark docs | Day 5 support-truth architecture, Day 7 consolidation diff, and Day 8 example/workflow routing |
| 205.4 Diagnostics Vocabulary | Direct solver docs, iterative solver docs, QR/SVD docs, eigensolver docs, benchmark/report-index docs, public examples | Day 9 vocabulary map and Day 10 wording updates |
| 205.5 Claim Guard Updates | Existing docs guards, install docs checks, API docs guards, benchmark/selected-performance docs tests, Windows claim validators | Day 11 guard design and Day 12 guard updates |
| 205.6 Validation | `make docs-check`, `make api-docs-freshness`, install docs checks, focused Python/shell guards, `git diff --check`, full C gate if `.c` or `.h` files change | Day 13 integrated validation and Day 14 closeout |

## Initial Public Surface Inventory

| Surface | Day 1 role | Initial notes |
| --- | --- | --- |
| `README.md` | Primary project entry point and support claim summary. | Needs audit for whether support/readiness links are easy to find and not duplicated. |
| `INSTALL.md` | Current support/readiness authority. | Owns static package story, local generated API status, platform/package/ABI boundaries, and install validation routing. |
| `docs/tutorial.md` | New-user learning path. | Already routes install/support questions to `INSTALL.md`; audit whether common adoption questions still require too many jumps. |
| `docs/cookbook.md` | Short task-oriented usage recipes. | Candidate destination or sibling for a compact problem-shape quick reference. |
| `docs/solver_selection.md` | Solver workflow selection guide. | Already contains detailed problem-shape tables and many claim-boundary paragraphs; likely needs a shorter entry route. |
| `docs/api_reference.md` | Source-controlled API entry point. | Must retain Sprint 204 local-only generated API policy and avoid ABI/package support implications. |
| `examples/README.md` | Example discovery and user workflow hints. | Should point users to the quick reference and support/readiness authority without duplicating caveats. |
| `benchmarks/README.md` | Benchmark command and report-index interpretation. | Must remain local/selected-evidence focused, not a portable performance claim. |
| `docs/maintainer_guide.md` | Maintainer support, claim, and validation authority. | Day 3 audit owner for non-claim vocabulary, guard expectations, and residual interpretation. |

## Inherited Support Decisions From Sprints 198-204

| Sprint | Day 1 inherited decision | Sprint 205 implication |
| --- | --- | --- |
| 198 | Closed a bounded developer-mode local Homebrew static source formula proof. Broad Homebrew, Homebrew/core, bottles, Linuxbrew, public tap, package-manager distribution, shared-library package support, dynamic ABI, and runtime-loader claims remain unclaimed. | Quick-reference/support wording must not say Homebrew or package-manager support is available. |
| 199 | Closed Windows Cholesky evidence as a re-deferral rather than promoted selected Windows freshness; generated metadata and non-claim wording remained unpromoted. | Support matrix wording must distinguish guarded workflow evidence from support promotion. |
| 200 | Closed selected `sparse_symbolic_lu()` allocation-failure owner proof only. Broad allocation-failure reliability remains unclaimed. | Diagnostics wording may cite selected owner proof only with scope; no broad reliability support claim. |
| 201 | Closed selected SVD rank, pseudoinverse, and dense low-rank helper extraction. Broad SVD behavior, review-surface cleanup, API/ABI, package, and platform claims remain unclaimed. | SVD quick-reference wording must stay workflow-oriented and avoid broad parity or support claims. |
| 202 | Closed Linux plus macOS hosted selected-performance freshness for the selected benchmark row. Portable performance, timing thresholds, broad benchmark publication, package/ABI support, release proof, and state-of-the-art claims remain unclaimed. | Benchmark routing can mention selected freshness only as bounded evidence, not portable performance support. |
| 203 | Re-deferred Windows QR incompatible selected freshness after local generator/freshness and guard work. Hosted Windows/MSVC proof and hosted artifact inspection remain absent. | QR support wording must not imply Windows selected QR freshness or broad Windows report freshness. |
| 204 | Closed stronger local-only generated API policy. Generated HTML remains ignored local output; hosted publication, retained artifacts, committed HTML, ABI/shared-library/package support, release evidence, broad platform evidence, portable performance, and state-of-the-art claims remain unclaimed. | API quick-reference and support matrix must route to source-controlled API docs and local freshness checks only. |

## Initial Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Support wording becomes simpler but broader. | Users may infer package, ABI, platform, performance, or state-of-the-art support that is not proven. | Keep `INSTALL.md` as support truth; update or add claim guards before closeout. |
| Quick reference duplicates detailed solver-selection tables. | Future docs drift between problem-shape guidance and detailed solver docs. | Design quick reference as a routing table with links, not a second full solver manual. |
| Repeated caveats are removed without replacement links. | Documentation becomes easier to read but loses claim-boundary context. | Replace repeated caveats with links to authoritative support and non-claim sections. |
| Diagnostics terms vary by solver family. | Users may misinterpret residuals, convergence, selected report statuses, or local evidence. | Build a vocabulary ledger before editing diagnostics wording. |
| Guard updates overfit current wording. | Future legitimate wording edits fail for incidental reasons, or unsupported claims slip through. | Prefer marker coverage tied to claim boundaries and targeted regression fixtures. |
| Full C gate is skipped after header edits. | Header or API-reference changes could miss compile-quality regressions. | Day 13 must inventory changed `.c`/`.h` files and run `make format && make lint && make test` if any changed. |

## Validation Matrix

| Command | Owner | When required | Day 1 status |
| --- | --- | --- | --- |
| `git diff --check` | Whitespace sanity | After any docs/script/test edits | Planned after Day 1 files |
| `make docs-check` | Doxygen/docs generation and API coverage | After public docs/API docs changes where applicable | Planned for validation days or earlier if API docs change |
| `make api-docs-freshness` | Generated API local-only freshness, staging, and routing | After API reference or generated API policy wording changes | Planned if API docs/routing wording changes |
| Install docs checks | Install/support/readiness consistency | After `INSTALL.md`, README install/support, or package wording changes | To identify exact target on Day 3-Day 5 |
| Existing claim/support Python and shell guards | Claim-boundary protection | After support wording or guard edits | Day 11-Day 12 owner |
| Focused docs tests | Guard regression coverage | After adding or changing docs guards | Day 12-Day 13 owner |
| `make format && make lint && make test` | Full C quality gate | Required if `.c` or `.h` files change | Not required on Day 1; no C/header edits |

## Open Questions

| Question | Owner day | Initial handling |
| --- | --- | --- |
| Should the compact quick reference live in README, `docs/cookbook.md`, `docs/solver_selection.md`, or a new doc? | Day 4 | Resolved for implementation: add the compact quick reference to `docs/cookbook.md` near the first-use ladder, add short routes from README/tutorial/examples, and keep `docs/solver_selection.md` as the detailed solver decision owner. |
| Which document should become the single support/readiness authority for public users? | Day 5 | Resolved: keep `INSTALL.md#support-readiness-matrix` as public support truth; keep `docs/maintainer_guide.md`, report schema/manifest, `benchmarks/README.md`, and `docs/api_reference.md` as interpretation/detail owners. |
| Which repeated caveats can be replaced by links without losing claim boundaries? | Day 7 | Day 5 classifies conversion candidates: README package/Windows/benchmark/generated-API paragraphs, tutorial/cookbook handoff caveats, examples benchmark/support caveats, and solver-selection benchmark/package reminders can be shortened only when the replacement points to the authoritative owner. |
| Which diagnostics terms need canonical definitions? | Day 9 | Build vocabulary across direct, iterative, QR/SVD, eigensolver, and report docs. |
| Which existing guard should own simplified support wording? | Day 11 | Reuse existing install/API/benchmark/Windows claim guards where possible. |

## Explicit Non-Goals

Sprint 205 does not claim or implement:

- new solver behavior, numerical algorithm changes, or tolerance policy changes;
- public API or ABI expansion;
- shared-library support or dynamic ABI compatibility;
- package-manager distribution, Homebrew/core readiness, bottles, Linuxbrew,
  public tap maintenance, vcpkg, Conan, pkgsrc, distro packages, or broad
  package support;
- broad Windows support, Windows Makefile parity, Windows `pkg-config`
  parity, or broad Windows report freshness;
- hosted generated API publication, retained generated-doc artifacts, or
  committed generated HTML;
- portable performance, timing thresholds, broad benchmark publication,
  backend superiority, or state-of-the-art sparse linear algebra evidence;
- release readiness or external-library parity.

## Day 1 Notes

Day 1 completed Sprint 205 intake and scaffolding only. The branch now has the
Sprint 205 plan, working-notes scaffold, item-to-surface traceability, inherited
decision inventory, risk register, validation matrix, open questions, and Day 1
artifact. No user-facing docs, source code, public headers, workflows, guards,
or generated outputs were changed on Day 1.

## Day 2 Notes

Day 2 completed the public documentation audit for README, INSTALL, tutorial,
cookbook, solver selection, API reference, examples, and benchmarks. The audit
found that the current public wording is mostly claim-safe and well linked, but
support caveats are repeated enough that users must often read several long
documents to answer basic adoption questions. The strongest consolidation path
is to keep `INSTALL.md#support-readiness-matrix` as support truth, keep
`docs/solver_selection.md` and `benchmarks/README.md` as deep evidence owners,
and add a compact quick-reference/routing surface that links out instead of
copying all proof caveats.

No public docs were edited on Day 2. The audit is recorded in
[day2-public-doc-audit.md](./artifacts/day2-public-doc-audit.md).

## Day 3 Notes

Day 3 completed the maintainer, benchmark/report, selected-target manifest,
schema, and planning-adjacent audit. The audit found that maintainer and report
surfaces are mostly internally consistent, but they use dense vocabulary that
must not be copied into the public quick reference. The source-of-truth split is
clear:

- `INSTALL.md#support-readiness-matrix` should remain the public support truth;
- `tests/corpus/manifests/selected_report_targets.tsv` and
  `tests/corpus/schemas/report_index_fields.md` should remain selected report
  contract truth;
- `benchmarks/README.md` should remain benchmark/report user interpretation;
- `docs/maintainer_guide.md` should remain proof interpretation and guard
  policy;
- Epic 18 planning artifacts should remain evidence and residual routing, not
  first-use documentation.

Day 3 also recorded diagnostics vocabulary conflicts for Day 9: `status`,
`support_tier`, `claim_boundary`, `residual_norm`, `fresh`, `pass/fail`,
`defer`, `skip`, `local_only`, and `hosted_selected` need context-specific
definitions before wording consolidation.

No public docs, maintainer docs, source code, headers, scripts, workflows, or
generated outputs were edited on Day 3. The audit is recorded in
[day3-maintainer-report-audit.md](./artifacts/day3-maintainer-report-audit.md).

## Day 4 Notes

Day 4 completed the compact quick-reference design. The selected placement is
`docs/cookbook.md` because it is already the data-first user workflow surface
and can host a short problem-shape-to-workflow table without turning README
into a larger evidence ledger or duplicating the detailed solver-selection
guide. README, tutorial, and examples should link to the quick reference after
implementation. `docs/solver_selection.md`, `INSTALL.md`, `docs/api_reference.md`,
and `benchmarks/README.md` remain the detailed owner docs.

The designed quick reference uses compact user labels rather than raw
manifest/schema vocabulary. Each row has a first workflow, runnable example or
guide, support/readiness route, and retained boundary. The design explicitly
excludes unearned package-manager, ABI, broad Windows, portable performance,
release, external-library parity, and state-of-the-art claims.

No public docs were edited on Day 4. The implementation design is recorded in
[day4-quick-reference-design.md](./artifacts/day4-quick-reference-design.md).

## Day 5 Notes

Day 5 completed the support-truth architecture. The selected model keeps
`INSTALL.md#support-readiness-matrix` as the single public support/readiness
authority while preserving detail ownership in `docs/api_reference.md`,
`benchmarks/README.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md`,
and `tests/corpus/manifests/selected_report_targets.tsv`.

The Day 5 conversion plan classifies repeated caveats into three groups:

- safe to shorten when replaced by links to owner docs;
- must stay inline near high-risk support claims;
- maintainer-only evidence that should not become public first-use routing.

No public docs were edited on Day 5. The architecture is recorded in
[day5-support-truth-architecture.md](./artifacts/day5-support-truth-architecture.md).

## Day 6 Notes

Day 6 implemented the compact adoption quick reference in
`docs/cookbook.md#problem-shape-quick-reference` and added discovery links from
README, tutorial, and examples. The new table routes common problem shapes to
the first workflow, runnable example or owner doc, and retained boundary while
keeping support/readiness ownership in `INSTALL.md#support-readiness-matrix`.

Changed public docs:

- `docs/cookbook.md`
- `README.md`
- `docs/tutorial.md`
- `examples/README.md`

Claim-boundary summary:

- no package-manager, Homebrew/core, shared-library, dynamic ABI, broad
  Windows, portable performance, release, or state-of-the-art claim was added;
- benchmark/report rows route to `benchmarks/README.md` and explicitly retain
  the no-portable-performance boundary;
- API rows route to `docs/api_reference.md` and explicitly retain local-only
  generated HTML wording;
- installed-consumer rows route to INSTALL and explicitly retain static-first
  and no package-manager/shared-library/dynamic ABI wording.

The implementation evidence is recorded in
[day6-quick-reference-implementation.md](./artifacts/day6-quick-reference-implementation.md).

## Day 7 Notes

Day 7 performed the support-truth consolidation batch. It shortened repeated
README generated API, report/Windows, and installation/package caveats by
routing to the owner docs while preserving inline warnings for local-only API
HTML, unpromoted Windows selected freshness, static-first install, no dynamic
ABI/shared-library support, and no package-manager/Homebrew support.

Changed docs:

- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- `docs/maintainer_guide.md`

Support-truth routing now has an explicit maintainer-guide section that records
`INSTALL.md#support-readiness-matrix` as public support truth and
`docs/cookbook.md#problem-shape-quick-reference` as the compact user routing
surface. No `.c` or `.h` files changed.

The consolidation evidence is recorded in
[day7-support-truth-consolidation.md](./artifacts/day7-support-truth-consolidation.md).

## Day 8 Notes

Day 8 completed the example and workflow routing pass for item 205.3. The pass
kept examples as runnable local workflow references, linked installed-consumer
interpretation back to the support/readiness matrix, and aligned the
solver-selection example handoff with the same route model.

Changed docs:

- `examples/README.md`
- `docs/solver_selection.md`

Route interpretation now distinguishes:

- build-tree teaching examples from installed downstream consumers;
- input-format examples from solver-support decisions;
- benchmark measurement handoffs from example usage;
- support/readiness status from example-local behavior.

No package-manager, shared-library, dynamic ABI, broad Windows, hosted API,
portable performance, release, or state-of-the-art support claim was added. No
`.c` or `.h` files changed.

The routing evidence is recorded in
[day8-example-workflow-routing.md](./artifacts/day8-example-workflow-routing.md).

## Day 9 Notes

Day 9 completed the diagnostics vocabulary design for item 205.4. The design
groups terminology by direct solvers, iterative solvers, QR/SVD, eigensolvers,
benchmark/report metadata, and validation guard states, then separates
public-facing workflow wording from maintainer/report schema wording.

The preferred public vocabulary now emphasizes:

- problem-local residuals and run-local convergence diagnostics;
- QR/SVD/eigensolver diagnostics scoped to the workflow that produced them;
- benchmark rows as local or selected evidence, not portable performance
  proof;
- `skip` and `defer` as scope states rather than pass/fail evidence;
- `local_only` and `hosted_selected` as maintainer/report metadata rather than
  first-use documentation vocabulary.

No public wording, source code, headers, scripts, workflows, manifests,
generated outputs, or validation commands were changed on Day 9. The design is
recorded in
[day9-diagnostics-vocabulary-design.md](./artifacts/day9-diagnostics-vocabulary-design.md).

## Day 10 Notes

Day 10 implemented the diagnostics vocabulary selected on Day 9 across the
selected user-facing and maintainer documentation surfaces. The pass updated
README, tutorial, cookbook, solver-selection, examples, benchmark, API
reference, and maintainer-guide wording while preserving exact API names,
schema fields, commands, and evidence row identities.

Changed docs:

- `README.md`
- `docs/cookbook.md`
- `docs/solver_selection.md`
- `examples/README.md`
- `docs/tutorial.md`
- `benchmarks/README.md`
- `docs/api_reference.md`
- `docs/maintainer_guide.md`

The wording now consistently distinguishes problem-local residuals,
run-local convergence fields, QR-local and SVD-local diagnostics, Ritz
residuals, local measurement artifacts, current generated-output diagnostics,
selected-target evidence, and skip/defer scope states. Maintainer/report
schema vocabulary remains available in benchmark and maintainer sections where
the surrounding text defines it.

No source code, public headers, scripts, workflows, manifests, schemas,
generated outputs, or validation commands were changed on Day 10. The
implementation evidence is recorded in
[day10-diagnostics-vocabulary-implementation.md](./artifacts/day10-diagnostics-vocabulary-implementation.md).

## Day 11 Notes

Day 11 completed the claim-guard design for item 205.5. The design inventories
the existing package-manager, static-package, API-docs local-only/routing,
selected-performance, Windows/PowerShell, selected-manifest, and Makefile docs
validation surfaces, then assigns Sprint 205 quick-reference, support-truth,
example-routing, diagnostics-vocabulary, generated-API, benchmark, package,
ABI, Windows, and performance boundaries to guard owners.

The selected Day 12 approach is:

- add one focused Python guard for Sprint 205 cross-document quick-reference,
  support-truth, and diagnostics-vocabulary markers;
- reuse existing specialized guards for package-manager, static package,
  generated API local-only/routing, selected performance, Windows, and
  selected-target manifest boundaries;
- update existing marker expectations only where Sprint 205 intentionally
  changed wording;
- keep guard failures semantic rather than formatting-sensitive.

No guard behavior, tests, scripts, Makefile targets, source code, headers,
workflows, manifests, schemas, generated outputs, or validation commands were
changed on Day 11. The design is recorded in
[day11-claim-guard-design.md](./artifacts/day11-claim-guard-design.md).

## Day 12 Notes

Day 12 implemented executable guard coverage for item 205.5. The implementation
adds `tests/test_support_quick_reference_docs.py` and a standalone
`make support-docs-guard` target. The guard checks Sprint 205 quick-reference,
support-truth, route-interpretation, diagnostics-vocabulary, benchmark,
generated-API, and maintainer guidance markers, and includes regression
fixtures for missing quick-reference routes, missing support-truth routes,
missing examples route interpretation, missing diagnostics vocabulary, package
overclaims, portable-performance overclaims, and hosted generated API
overclaims.

Changed guard surfaces:

- `tests/test_support_quick_reference_docs.py`
- `Makefile`
- `scripts/check_api_docs_routing.py`
- `tests/test_api_docs_routing.py`

Focused validation:

- `python3 tests/test_support_quick_reference_docs.py` - passed
- `make support-docs-guard` - passed
- `python3 tests/test_selected_performance_docs.py` - passed
- `python3 tests/test_api_docs_routing.py` - passed

No production source, public headers, CMake install behavior, CI workflows,
selected target manifests, report schemas, generated outputs, package
metadata, API behavior, or support/readiness status changed on Day 12. The
implementation evidence is recorded in
[day12-claim-guard-implementation.md](./artifacts/day12-claim-guard-implementation.md).

## Day 13 Notes

Day 13 completed integrated validation for item 205.6. The validation pass
covered changed-file inventory, generated-output tracking state, Sprint 205
support quick-reference guards, selected-performance docs, generated API
freshness/routing/local-only checks, package-manager deferral, static package
deferral, Windows/PowerShell claim boundaries, and whitespace checks.

Focused validation:

- `git diff --name-only -- '*.c' '*.h'` - passed with no changed C or header
  files.
- `git ls-files --others --exclude-standard -- '*.c' '*.h'` - passed with no
  untracked C or header files.
- `git diff --check` - passed.
- `python3 tests/test_support_quick_reference_docs.py` - passed.
- `make support-docs-guard` - passed.
- `python3 tests/test_selected_performance_docs.py` - passed.
- `python3 tests/test_api_docs_routing.py` - passed.
- `python3 tests/test_api_docs_local_only_guard.py` - passed.
- `make api-docs-freshness` - passed.
- `bash scripts/package_manager_deferral_check.sh` - passed after marker
  alignment to current README wording.
- `bash scripts/static_package_deferral_check.sh` - passed after marker
  alignment to current README wording.
- `python3 tests/test_validate_windows_powershell.py` - passed after marker
  alignment to current README wording.

No `.c` or `.h` files changed, so `make format && make lint && make test` was
not required for Day 13. Doxygen output exists locally under ignored
`docs/api/`; `git status --ignored --short docs/api` reports `!! docs/api/`,
and no generated API output is staged or tracked.

The integrated validation evidence is recorded in
[day13-integrated-validation.md](./artifacts/day13-integrated-validation.md).

## Day 14 Notes

Day 14 completed Sprint 205 closeout review. The closeout reviewed every
Sprint 205 item, artifact, public documentation edit, maintainer documentation
edit, guard change, validation result, generated-output state, and Epic 18
status surface.

Closeout updates:

- `docs/planning/EPIC_18/PROJECT_PLAN.md` now records Sprint 205 as closed
  with support matrix and adoption quick-reference consolidation evidence.
- `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` now counts Sprint 205 as
  closed in the Epic 18 implementation status summary.
- `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` now treats E18-RQ-008 as
  a closed selected baseline with only future adoption UX residual options.
- `WORKING_NOTES.md` marks Day 14 and item 205.6 complete.
- `day14-closeout-review.md` records completed, narrowed, deferred, residual,
  validation, generated-output, and retrospective-input outcomes.

No new support, package, ABI, Windows, hosted generated API, portable
performance, release, or state-of-the-art claims were added. The closeout
decision is that Sprint 205 is ready for retrospective creation and PR review.
