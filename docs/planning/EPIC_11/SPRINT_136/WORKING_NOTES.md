# Sprint 136 Working Notes

## Sprint Goal

Validate the final Epic 11 outcome package, compare earned evidence against
the state-of-the-art target and explicit non-claims, clean unsupported public
wording, publish residual queues, and close Epic 11 with Sprint 136 and
Epic 11 retrospectives plus a final handoff.

Sprint 136 is the Epic 11 closeout sprint. It must treat prior sprint
artifacts as evidence inputs, not as automatic public claims. Generated report
indexes, benchmark rows, package proofs, platform CI lanes, documentation
navigation, and residual queues only support the bounded claims recorded by
their owner artifacts.

## Starting Constraints

- Treat Sprint 131 report-index policy as the report baseline: generated
  indexes are traceability and freshness evidence, not broad correctness,
  coverage-completeness, release, or performance proof.
- Treat Sprint 133 as the static-first package baseline: shared-library
  packaging, dynamic ABI compatibility, runtime-loader behavior, and
  package-manager support remain deferred non-claims.
- Treat Sprint 134 as the platform-tier baseline: Linux owns the reviewed
  static-first package-contract CI lane; macOS install/export confidence is
  supplemental; Windows install/downstream confidence is supplemental; staged
  Windows pthread/POSIX tests remain staged until source portability and hosted
  MSVC proof exist.
- Treat Sprint 135 adoption docs as navigation and documentation
  productization, not new solver behavior, package support, normalized report
  schema, portable benchmark performance, or platform parity.
- Treat the end-of-epic deferred QR residual queue as future-epic triage input
  with promotion criteria, not immediate Sprint 136 implementation scope.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check` and a focused
  trailing-whitespace scan over touched Markdown surfaces.

## Input Artifact Inventory

| Input | Role in Sprint 136 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 136 | Defines final closeout items for evidence inventory, validation design/execution, competitive recalibration, claim cleanup, residual publication, and retrospectives. |
| `docs/planning/EPIC_11/SPRINT_136/PLAN.md` | Provides day-level execution order and 164-hour budget. |
| `docs/planning/EPIC_11/SPRINT_118` through `SPRINT_130` artifacts | Provide earlier Epic 11 source/test, solver evidence, oracle, residual, and closeout history. |
| `docs/planning/EPIC_11/SPRINT_131/artifacts/day14-closeout-report-index-handoff.md` | Provides report-index, freshness, corpus taxonomy, coverage, dead-code, guardrail, and report non-claim boundaries. |
| `docs/planning/EPIC_11/SPRINT_132/RETROSPECTIVE.md` and artifacts | Provide performance sentinel and runtime-governance residual context. |
| `docs/planning/EPIC_11/SPRINT_133/artifacts/day14-closeout-package-abi-handoff.md` | Provides static-first package, ABI, shared-library, package-manager, and package-proof boundaries. |
| `docs/planning/EPIC_11/SPRINT_134/artifacts/day14-platform-tier-closeout-handoff.md` | Provides Linux, macOS, Windows, staged-test, and supplemental platform support tiers. |
| `docs/planning/EPIC_11/SPRINT_135/artifacts/day14-adoption-closeout-handoff.md` | Provides adoption-surface, cookbook, algorithm-reference, historical-appendix, and report-index documentation handoff. |
| `README.md` | Front-door public feature, build, adoption, package, platform, and report-index wording. |
| `INSTALL.md` | Static-first install, downstream-consumer, and platform support truth. |
| `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/matrix_market.md` | First-use, compressed-first, solver-choice, and Matrix Market adoption surfaces. |
| `docs/algorithm.md`, `docs/algorithm_history.md` | Current algorithm reference and historical measurement appendix surfaces. |
| `docs/maintainer_guide.md` | Maintainer support-tier, validation ownership, package/platform, report, benchmark, and non-claim policy surface. |
| `examples/README.md`, `examples/*.c` | Maintained example discovery and example-source adoption evidence. |
| `benchmarks/README.md`, `benchmarks/*.c`, generated report paths | Benchmark workflow, local-measurement, sentinel, canonical-report, and guardrail evidence surfaces. |

## Day-Level Ownership

| Day | Owner focus | Project-plan items |
| --- | --- | --- |
| 1 | Closeout intake, inherited evidence map, artifact baseline, item ownership, and claim fences | Items 1-7 |
| 2 | Final source/test/oracle/performance/package/platform/docs/residual evidence inventory | Item 1 |
| 3 | Reviewed and supplemental validation architecture | Item 2 |
| 4 | Executable validation command plan and stop conditions | Item 2 |
| 5 | Reviewed validation batch 1: docs, source-list, package/static proof | Item 3 |
| 6 | Reviewed validation batch 2: CMake, tests, package/install proof, local quality as needed | Item 3 |
| 7 | Supplemental benchmark/report/package validation and generated evidence reconciliation | Item 3 |
| 8 | Competitive evidence baseline against Epic 11 goals and state-of-the-art non-claims | Item 4 |
| 9 | Final competitive claim recalibration decisions | Item 4 |
| 10 | Unsupported public/support claim audit | Item 5 |
| 11 | Unsupported public/support claim cleanup and focused validation | Item 5 |
| 12 | Post-Epic-11 residual queue and deferred QR residual publication | Item 6 |
| 13 | Sprint 136 and Epic 11 retrospective drafts plus handoff synthesis | Item 7 |
| 14 | Final Sprint 136 retrospective, Epic 11 retrospective, validation summary, and closeout handoff | Item 7 |

## Validation Expectations

| Change type | Required validation |
| --- | --- |
| Sprint 136 planning artifacts only | `git diff --check` and focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_136`. |
| Public documentation wording | `git diff --check`, focused trailing-whitespace scan, local link/path checks for touched docs, and claim-boundary scan. |
| Package/install/package metadata wording | Claim scan against Sprint 133 static-first support truth and Sprint 134 platform tiers. |
| Benchmark/report wording | Claim scan against Sprint 131 report-index semantics and Sprint 132 performance sentinel/runtime-governance boundaries. |
| Generated report execution | Inspect report freshness, owner paths, row meanings, and generated-versus-curated boundaries before using as evidence. |
| Workflow, script, CMake, or build-system edits | Syntax or dry-run validation for touched tooling plus support-tier scan. |
| `.c` or `.h` edits | `make format && make lint && make test` after focused implementation validation. |

## Inherited Claim Fences

| Claim family | Sprint 136 boundary |
| --- | --- |
| Source/test ownership | Source and test changes must be tied to explicit owner files, commands, and quality gates; ownership does not imply complete solver-family coverage. |
| Oracle evidence | External-reference helpers and oracle rows remain helper-specific; no broad LAPACK, NumPy, SciPy, SuiteSparse, dense-library, backend, or ecosystem parity claim follows from helper output alone. |
| Report indexes | Generated indexes provide artifact maps, row interpretation, and freshness context; they are not broad correctness, release, coverage-completeness, or performance proof. |
| Coverage/dead-code reports | Coverage is supplemental and tree-mutating; dead-code reports are conservative report-completeness evidence, not zero-findings or removal-ready proof. |
| Package/ABI | Maintained package support is static-first; shared-library packaging, dynamic ABI compatibility, runtime-loader behavior, package-manager recipes, and static/shared selectors remain deferred non-claims. |
| Platform support | Linux is the strongest reviewed package-contract CI owner; macOS install/export confidence is supplemental; Windows install/downstream confidence is supplemental; Windows staged tests remain staged. |
| Adoption docs | Cookbook, tutorial, solver-selection, and algorithm docs improve navigation and explanation; they do not create new solver behavior, package support, report schema, or platform parity. |
| Benchmarks/performance | Benchmark rows and sentinel reports are local measurement evidence with freshness context, not portable speed, scalability, backend parity, or correctness-over-time guarantees. |
| Competitive positioning | State-of-the-art comparison must be phrased as bounded evidence and explicit non-claims; no best-in-class, parity, broad performance, or ecosystem superiority claim is earned without direct proof. |
| QR residuals | Deferred QR residuals remain future-epic candidates with promotion criteria; Sprint 136 may publish and classify them but should not silently implement or promote them. |

## Day 1 Notes

- Created the Sprint 136 working-notes baseline and artifact directory.
- Re-read the Sprint 136 section of `docs/planning/EPIC_11/PROJECT_PLAN.md`.
- Mapped Sprint 136 Items 1-7 to day-level owners across Days 1-14.
- Reviewed Sprint 131 closeout as the inherited report-index, freshness,
  corpus, coverage, dead-code, and guardrail baseline.
- Reviewed Sprint 133 closeout as the inherited static-first package, ABI,
  shared-library, package-manager, and package-proof baseline.
- Reviewed Sprint 134 closeout as the inherited Linux/macOS/Windows platform
  support-tier and staged-test baseline.
- Reviewed Sprint 135 closeout as the inherited adoption documentation,
  cookbook, algorithm-reference, historical-appendix, and report-index
  navigation baseline.
- Recorded final claim fences for source/test ownership, oracle evidence,
  report indexes, package/ABI, platform support, adoption docs, benchmarks,
  competitive positioning, and QR residuals.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 1 beyond Sprint 136 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 2 Notes

- Wrote the final evidence inventory artifact.
- Inventoried Epic 11 sprint retrospectives and closeout artifacts from
  Sprint 118 through Sprint 135.
- Grouped source/test evidence into eigensolver movement, direct/iterative
  oracle split, SVD/QR/rank-deficient fixtures, residual QR evidence,
  partial-SVD solver-selection gates, performance/runtime governance,
  package/install proof, platform CI tiers, and adoption documentation.
- Identified current owner surfaces for source, headers, tests, examples,
  benchmarks, scripts, workflows, public docs, maintainer docs, and generated
  report paths.
- Separated generated report and validation artifacts from public claims:
  report rows, benchmark rows, freshness metadata, package proofs, and CI lanes
  remain evidence with support-tier context, not broad claims.
- Grouped initial residuals into QR residuals, report/corpus architecture,
  performance/runtime governance, package/ABI/distribution, platform staging,
  documentation automation, and competitive/non-claim cleanup.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 2 beyond Sprint 136 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 3 Notes

- Wrote the validation architecture artifact.
- Classified final validation lanes as reviewed, local reviewed-equivalent,
  supplemental, hosted, staged, deferred, or unsupported.
- Mapped command owners for docs hygiene, link/path checks, source-list checks,
  Make quality, CMake registration, package/install proofs, static deferral
  proof, benchmark/report generation, dead-code, coverage, and platform CI.
- Defined validation requirements by touched surface so documentation-only,
  public-doc, package, benchmark/report, workflow/script, CMake/build, and
  C/header changes have explicit gates before Day 4 command planning.
- Preserved inherited support tiers: Linux package-contract CI is reviewed;
  macOS and Windows install/downstream lanes remain supplemental; Windows
  pthread/POSIX tests remain staged; coverage remains supplemental and
  tree-mutating; generated reports remain freshness and traceability evidence.
- Recorded full-validation risks and stop conditions for code changes, claim
  wording drift, package/platform widening, generated-report absence, hosted
  runner-only proof, and supplemental-lane promotion.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 3 beyond Sprint 136 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 4 Notes

- Wrote the validation command-plan artifact.
- Converted the Day 3 validation architecture into an executable Day 5-7
  command sequence for documentation hygiene, source/header diff decisions,
  package/static proofs, CMake/install proofs, generated report commands, and
  supplemental/deferred lanes.
- Defined artifact capture paths under
  `docs/planning/EPIC_11/SPRINT_136/validation/` for Day 5-7 summaries and
  generated report metadata snapshots.
- Defined pass/fail interpretation for every planned command family before
  execution begins.
- Marked intentionally skipped or deferred lanes: full C quality gate unless
  `.c`/`.h` files change, coverage unless coverage wording changes, hosted
  Linux/macOS/Windows CI proof until branch push/CI, shared-library/dynamic
  ABI/package-manager proof, and immediate QR residual implementation.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 4 beyond Sprint 136 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 5 Notes

- Ran reviewed validation batch 1 and wrote the Day 5 validation summary.
- Ran `git diff --check`: passed.
- Ran the focused Sprint 136 trailing-whitespace scan: passed.
- Confirmed no tracked or untracked `.c` or `.h` files changed, so the full
  C quality gate remains not required.
- Captured touched-surface status: only
  `docs/planning/EPIC_11/SPRINT_136/` is currently untracked/changed.
- Ran package proof script syntax preflight:
  `bash -n tests/test_install.sh tests/test_cmake_install.sh
  scripts/static_package_deferral_check.sh`: passed.
- Ran `python3 scripts/check_library_sources.py`: passed with 49 library
  sources.
- Ran `bash scripts/static_package_deferral_check.sh`: passed, confirming
  shared-build rejection, static target declaration, absence of shared
  export/ABI metadata, no package static/shared selector, and deferred support
  wording.
- Ran a package/platform/performance/parity claim-scan baseline across Sprint
  136 and public/support docs. Findings were expected non-claim/support-tier
  wording, not a Day 5 blocker.
- Wrote the initial validation skip/defer register for Day 6-7 and hosted CI
  lanes.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 5 beyond Sprint 136 planning artifacts and validation
  records.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 6 Notes

- Ran reviewed validation batch 2 and wrote the Day 6 validation summary.
- Confirmed no tracked or untracked `.c` or `.h` files changed, so
  `make format && make lint && make test` remains not required.
- Ran local CMake configure:
  `cmake -S . -B build-sprint136-cmake`: passed with AppleClang.
- Ran local CMake build:
  `cmake --build build-sprint136-cmake`: passed.
- Ran local CTest registration:
  `ctest --test-dir build-sprint136-cmake -N`: passed with 57 registered
  tests.
- Ran local CTest execution:
  `ctest --test-dir build-sprint136-cmake --output-on-failure`: passed,
  57/57 tests, 0 failed, 740.85 seconds total real time.
- Reconciled the local CTest count with Sprint 134 platform-tier notes: 57 is
  the local non-Windows count; Windows reviewed CTest count remains 54 after
  staged pthread/POSIX exclusions.
- Ran CMake install/export package proof:
  `bash tests/test_cmake_install.sh`: passed, 21 checks, 0 failures, 0 skips.
- Removed generated local validation build directories after capture so the
  branch remains documentation-only.
- Updated the validation skip/defer register after Day 6.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 6 beyond Sprint 136 planning artifacts and validation
  records.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 7 Notes

- Ran supplemental/report validation and wrote the Day 7 validation summary.
- Ran `make bench-canonical-report`: passed and wrote
  `build/bench-reports/canonical/` with four measurement rows.
- Ran `make performance-sentinels`: passed and wrote
  `build/bench-reports/sentinels/` with 11 sentinel rows.
- Ran `make large-matrix-guardrails`: passed and wrote
  `build/bench-reports/large-matrix-guardrails/` with six guardrail rows.
- Inspected generated manifests and indexes for freshness, row counts, commit,
  branch, platform, compiler, support-tier notes, and claim-boundary notes.
- Ran local Make install/`pkg-config` package proof:
  `bash tests/test_install.sh`: passed, 22 checks, 0 failures.
- Recorded that generated benchmark/report evidence is local and
  freshness-scoped: it does not create portable performance, release,
  platform-parity, or broad correctness claims.
- Updated the validation skip/defer register after Day 7.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 7 beyond Sprint 136 planning artifacts and validation
  records.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 8 Notes

- Wrote the competitive evidence baseline artifact.
- Re-read the Epic 11 project-plan goal: Epic 11 improves product maturity,
  source/test ownership, oracle breadth, performance governance,
  package/ABI/platform decisions, and adoption surface, but still should not
  claim unqualified state-of-the-art status.
- Compared final source/test/oracle evidence against solver-family claim
  classes and classified them as earned internal/local evidence with bounded
  external-reference and residual non-claims.
- Compared Day 7 benchmark/report evidence against performance wording and
  classified it as local/freshness-scoped evidence, not portable performance,
  scalability, memory, backend parity, or state-of-the-art proof.
- Compared package/platform/install evidence against adoption/support wording:
  static-first package support is earned locally and through Linux reviewed CI
  ownership; macOS/Windows install/downstream confidence remains supplemental;
  shared-library, dynamic ABI, runtime-loader, and package-manager support
  remain deferred non-claims.
- Classified adoption documentation as earned navigation/productization
  evidence, not behavior, package, report-schema, or platform expansion.
- Published a state-of-the-art non-claim register and Day 9 recalibration
  inputs.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 8 beyond Sprint 136 planning artifacts and validation
  records.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 9 Notes

- Wrote the competitive claim recalibration artifact.
- Converted Day 8 evidence classifications into final claim decisions for
  public docs, maintainer docs, package/install docs, platform wording,
  benchmark/report wording, solver-selection wording, residuals, and
  competitive positioning.
- Marked earned public wording as limited to evidence ownership, local
  validation discipline, static-first install/export, tiered platform support,
  report freshness, and adoption navigation.
- Marked solver/oracle claims as maintainer-facing or bounded public workflow
  guidance only; broad external parity and state-of-the-art wording remain
  non-claims.
- Preserved package/platform decisions: static-first package support may be
  described; shared-library, dynamic ABI, runtime-loader, package-manager,
  reviewed macOS install parity, reviewed Windows install parity, and Windows
  staged-test promotion must not be claimed.
- Preserved benchmark/report decisions: generated reports may be described as
  local freshness-scoped evidence, not portable performance, scalability,
  memory, release, correctness, or competitive superiority proof.
- Built the Day 10 unsupported-claim audit queue from high-risk wording
  categories and owner surfaces.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 9 beyond Sprint 136 planning artifacts and validation
  records.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 10 Notes

- Wrote the unsupported-claim audit artifact.
- Scanned public, maintainer, benchmark, package, adoption, algorithm, and
  Sprint 136 planning surfaces for claim drift against the Day 8-9 evidence
  and claim decisions.
- Checked package/platform wording for shared-library, dynamic ABI,
  runtime-loader, package-manager, reviewed macOS/Windows install parity, and
  Windows staged-test promotion drift.
- Checked benchmark/report wording for portable performance, scalability,
  memory, speed superiority, universal reorder/fill, and report-index
  overclaiming.
- Checked competitive, coverage, and external-parity wording for
  state-of-the-art, broad ecosystem parity, every-solver-family coverage,
  normalized cross-report proof, release proof, and correctness-proof drift.
- Found no P0 public-doc cleanup blockers; positive scan hits were already
  fenced as explicit non-claims, local/fixture/historical evidence, algorithm
  context, or prerequisite/toolchain instructions.
- Built the Day 11 verification queue for package/platform support tiers,
  benchmark/report local-measurement fences, competitive non-claims, and
  algorithm explanatory wording.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 10 beyond Sprint 136 planning artifacts and validation
  records.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 11 Notes

- Wrote the unsupported-claim cleanup artifact.
- Re-ran focused claim-boundary scans across the Day 10 P1 surfaces for
  competitive overclaims, performance/report guarantees, package/platform
  support drift, coverage-completeness wording, and broad external parity.
- Confirmed no P0 unsupported public/support wording required editing.
- Recorded no-op cleanup decisions for package/platform support tiers,
  benchmark/report local-measurement fences, competitive and external-parity
  non-claims, and algorithm explanatory wording.
- Preserved existing evidence-owner surfaces for static-first package support,
  platform tiers, benchmarks/generated reports, solver/oracle boundaries, and
  competitive non-claims.
- Published the remaining non-claim register for Day 12 residual publication.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 11 beyond Sprint 136 planning artifacts and validation
  records.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 12 Notes

- Wrote the residual queue publication artifact.
- Consolidated residuals from Sprint 118-135 retrospectives, recent closeout
  handoffs, Day 8-11 Sprint 136 claim artifacts, and validation records.
- Classified residuals as future-epic candidates, evidence-blocked,
  metadata-blocked, optional-local work, or explicit non-claims.
- Published a post-Epic-11 residual queue covering QR residual expansion,
  partial-SVD residual expansion, corpus/report indexes, coverage/dead-code,
  runtime/backend sentinels, package/ABI/distribution, platform promotions,
  Windows staged tests, and documentation maintenance.
- Published a dedicated deferred QR residual queue with current baselines,
  blockers, and promotion criteria for compatible zero-residual, wide
  residual-only, nullspace/subspace, threshold, SuiteSparse, minimum-norm, and
  QR-vs-SVD cross-check work.
- Preserved the explicit post-Epic-11 non-claim register so Day 13-14
  retrospectives and handoff language do not turn residuals into earned
  claims.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 12 beyond Sprint 136 planning artifacts and validation
  records.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 13 Notes

- Wrote the retrospective draft and handoff synthesis artifact.
- Drafted Sprint 136 retrospective inputs from validation, generated report,
  claim recalibration, unsupported-claim cleanup, residual publication, and
  closeout evidence.
- Drafted the Epic 11 retrospective structure around objectives, major
  outcomes, validation evidence, earned claims, non-claims, residuals, and
  closeout assessment.
- Synthesized final handoff sections for evidence, validation, claims,
  residuals, package/platform support, benchmark/report context, adoption
  documentation, and non-claims.
- Reconciled proposed closeout language against Day 5-7 validation, Day 8-9
  claim decisions, Day 10-11 cleanup decisions, and the Day 12 residual queue.
- Published the Day 14 gap list and finalization checklist.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 13 beyond Sprint 136 planning artifacts and validation
  records.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 14 Notes

- Wrote the final Sprint 136 retrospective.
- Wrote the Epic 11 retrospective.
- Wrote the final Epic 11 closeout handoff artifact.
- Reconciled Sprint 136 deliverables against the Day 1-13 artifact package,
  Day 5-7 validation evidence, Day 8-9 claim decisions, Day 10-11 cleanup
  decisions, and Day 12 residual queue.
- Preserved final support-tier boundaries:
  - static-first package support only;
  - shared-library, dynamic ABI, runtime-loader, and package-manager support
    remain non-claims;
  - Linux remains the strongest reviewed package-contract owner;
  - macOS and Windows package/install confidence remains supplemental;
  - Windows pthread/POSIX-backed tests remain staged;
  - benchmark/report evidence remains local and freshness-scoped.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 14 beyond Sprint 136 planning artifacts, final
  retrospectives, and closeout handoff records.
- Final Day 14 validation passed:
  - package/platform claim-boundary scan;
  - performance/report claim-boundary scan;
  - competitive/parity claim-boundary scan;
  - coverage/report-index/support-tier claim-boundary scan;
  - `git diff --check`;
  - Sprint 136 trailing-whitespace scan;
  - local Sprint 136 markdown link/path validation;
  - C/header change check.
- No `.c` or `.h` files changed, so the full C quality gate was not required.
