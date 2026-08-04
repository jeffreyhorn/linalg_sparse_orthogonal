# Sprint 137 Working Notes

## Sprint Goal

Freeze the post-Epic-11 baseline, select the gaps Epic 12 will close
completely, and create the evidence contracts that govern later implementation
sprints.

Sprint 137 is the Epic 12 intake and contract sprint. It must treat Epic 11
residuals as candidate inputs, not as preselected scope. Later sprints may only
promote corpus, QR, partial-SVD, report, runtime, package, platform, adoption,
or state-of-the-art wording after the required evidence, validation, and claim
gates are present.

## Starting Constraints

- Treat Epic 11 as closed on bounded evidence, not on broad state-of-the-art
  parity.
- Treat the Epic 11 residual queue as the source of candidate gaps for Epic 12.
- Prefer complete closure of selected gaps over partial progress on many gaps.
- Preserve static-first packaging, explicit platform tiers, local benchmark
  interpretation, generated-report boundaries, and state-of-the-art non-claims
  unless future sprints earn stronger proof.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only Sprint 137 changes require `git diff --check` and
  focused Markdown hygiene/link checks.

## Input Artifact Inventory

| Input | Role in Sprint 137 |
| --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` Sprint 137 | Defines the baseline, residual reconciliation, gap selection, evidence templates, quality map, claim freeze, and closeout items. |
| `docs/planning/EPIC_12/SPRINT_137/PLAN.md` | Provides day-level execution order and 166-hour budget. |
| `docs/planning/EPIC_12/reviews/review-codex-2026-08-03.md` | Provides the current code review, measured signals, state-of-the-art assessment, and highest-value gap closures. |
| `docs/planning/EPIC_12/reviews/todo-codex-2026-08-03.md` | Provides the Epic 12 execution sequence and completion definition. |
| `docs/planning/EPIC_11/EPIC_11_RETROSPECTIVE.md` | Provides Epic 11 earned claims, non-claims, validation summary, and future-epic candidates. |
| `docs/planning/EPIC_11/SPRINT_136/artifacts/day12-residual-queue-publication.md` | Provides the post-Epic-11 residual queue, explicit non-claim register, QR residual queue, and promotion criteria. |
| `README.md` | Public front door for capabilities, workflow selection, support tiers, package/install claims, CI claims, and report interpretation. |
| `INSTALL.md` | Static-first install, downstream-consumer, package, ABI, and platform support truth. |
| `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/matrix_market.md` | Adoption, compressed-first, solver-selection, and Matrix Market documentation surfaces. |
| `docs/algorithm.md`, `docs/algorithm_history.md` | Current algorithm reference and historical measurement surfaces. |
| `docs/maintainer_guide.md` | Maintainer support-tier, validation, package/platform, report, benchmark, and non-claim policy surface. |
| `.github/workflows/*.yml` | Hosted Linux, macOS, Windows, package, sanitizer, dead-code, benchmark, and platform support-tier evidence. |
| `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `tests/test_install.sh`, `tests/test_cmake_install.sh` | Build, install, package, pkg-config, CMake export, and downstream-consumer proof surfaces. |
| `scripts/*.sh`, `scripts/*.py`, `benchmarks/README.md`, `benchmarks/*.c` | Report, benchmark, sentinel, guardrail, dead-code, and performance-governance evidence surfaces. |
| `src/`, `include/`, `tests/`, `examples/` | Source, public API, proof-owner, and adoption-example surfaces for Day 2 metrics and later gap selection. |

## Day-Level Ownership

| Day | Owner focus | Project-plan items |
| --- | --- | --- |
| 1 | Scope setup, artifact baseline, inherited input map, item ownership, validation expectations | Items 1-7 |
| 2 | Source/test/benchmark/example size and maintainability baseline | Item 1 |
| 3 | Build, package, CI, report, benchmark, and support-tier baseline | Item 1 |
| 4 | Epic 11 residual intake and candidate grouping | Item 2 |
| 5 | Residual owners, dependencies, promotion gates, and non-goals | Item 2 |
| 6 | Gap-selection scoring rubric and complete-closure criteria | Item 3 |
| 7 | Epic 12 selected gap decision for Sprints 138-146 | Item 3 |
| 8 | Corpus fixture, generated-matrix, optional-data, and oracle row templates | Item 4 |
| 9 | Report-index, normalized metadata, and stale-report templates | Item 4 |
| 10 | Package/ABI, platform promotion, downstream proof, and public claim templates | Item 4 |
| 11 | Quality surface map by touched documentation, scripts, build, CI, package, report, and C/header changes | Item 5 |
| 12 | Public claim freeze across README, INSTALL, docs, examples, benchmarks, and maintainer guide | Item 6 |
| 13 | Handoff synthesis and Sprint 138 corpus-readiness package | Item 7 |
| 14 | Sprint 137 closeout, residuals, validation summary, and working-notes completion | Item 7 |

## Initial Validation Expectations

| Change type | Required validation |
| --- | --- |
| Sprint 137 planning artifacts only | `git diff --check` plus focused Markdown link/path validation under `docs/planning/EPIC_12`. |
| Public documentation wording | `git diff --check`, focused Markdown link/path validation, and claim-boundary scan against Epic 11/Epic 12 non-claims. |
| Script or report-generator edits | Syntax validation for the touched script, focused report command where feasible, and support-tier/freshness semantics review. |
| Makefile, CMake, pkg-config, install, or package edits | Relevant package/install/CMake proof commands plus static/shared support-boundary review. |
| CI workflow edits | Workflow syntax or structural review plus hosted-runner support-tier notes; do not treat unrun hosted lanes as passed local evidence. |
| Benchmark or generated report execution | Capture command, platform, compiler/configuration, source commit, row meaning, freshness, support tier, and skip/defer status. |
| `.c` or `.h` edits | `make format && make lint && make test` after any focused tests needed for the touched implementation. |

## Inherited Claim Fences

| Claim family | Sprint 137 boundary |
| --- | --- |
| State of the art | Epic 12 starts from an explicit non-claim; no broad state-of-the-art wording is earned by planning artifacts. |
| Numerical corpus | Corpus work is a future maintained lane; current evidence remains fixture-local unless Sprint 138+ adds row semantics and proof. |
| QR evidence | Deferred QR residuals need trust value, output semantics, tolerance, rank/nullity, and support-tier metadata before promotion. |
| Partial SVD | Partial-SVD residuals need edge-case, convergence-budget, ordering, vector/subspace, tolerance, and skip semantics before promotion. |
| Report indexes | Generated reports are freshness and traceability evidence, not broad correctness, release, coverage, or performance proof. |
| Runtime/backend | Benchmark and sentinel rows are local measurement evidence; no portable performance, backend parity, OpenMP speedup, scalability, or memory claim is earned. |
| Package/ABI | Static-first remains maintained; shared-library packaging, dynamic ABI, runtime-loader behavior, and package-manager support remain non-claims. |
| Platform support | Linux is strongest; macOS and Windows install/export confidence remain supplemental; Windows pthread/POSIX tests remain staged. |
| Adoption docs | Navigation improvements do not create new solver behavior, package support, platform parity, or report schema claims. |
| Coverage/dead-code | Coverage remains supplemental; dead-code output is triage/report-completeness evidence, not removal-ready proof. |

## Day 1 Notes

- Created the Sprint 137 working-notes baseline and artifact directory.
- Re-read the Sprint 137 section of `docs/planning/EPIC_12/PROJECT_PLAN.md`.
- Re-read the Epic 12 review and gap-closure todo.
- Re-read the Epic 11 retrospective and Sprint 136 residual queue publication.
- Mapped Sprint 137 Items 1-7 to day-level owners across Days 1-14.
- Recorded inherited inputs before Day 2 metrics and Day 4 residual intake.
- Recorded validation expectations for documentation-only, public-doc,
  script/report, build-system, CI, benchmark/report, and C/header changes.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 1 beyond Sprint 137 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 2 Notes

- Wrote the source, test, and maintainability baseline artifact.
- Captured 191 C/header/template files under `src`, `include`, `tests`,
  `benchmarks`, and `examples`, totaling 123,352 lines.
- Captured current surface counts: 49 implementation `.c` files, 20 private
  implementation headers, 19 public headers/templates, 58 test `.c` files, 11
  test helper headers, 16 benchmark `.c` files, and 15 example `.c` files.
- Ranked largest implementation owners, led by `src/sparse_ldlt_csc.c`,
  `src/sparse_lu_csr.c`, `src/sparse_ldlt.c`,
  `src/sparse_iterative.c`, `src/sparse_qr.c`, `src/sparse_eigs.c`, and
  `src/sparse_svd.c`.
- Ranked largest proof owners, led by `tests/test_qr.c`,
  `tests/test_ldlt_csc.c`, `tests/test_integration.c`,
  `tests/test_svd.c`, `tests/test_ldlt.c`, `tests/test_etree.c`, and
  `tests/test_iterative.c`.
- Confirmed source-list ownership signals: `build-metadata/library_sources.txt`,
  Makefile `LIB_SRCS`, and CMake `add_library(sparse_lu_ortho STATIC ...)`
  all list 49 library sources.
- Recorded test ownership signals: Makefile lists 57 main test binaries while
  the tree has 58 test `.c` files; CMake has 54 `add_sparse_test(...)` lines
  plus conditional POSIX/fuzz gates, preserving the Windows staged-test
  boundary.
- Tied maintainability risks to Epic 12 candidate gaps for QR, partial-SVD,
  corpus/oracle architecture, report normalization, runtime/backend
  governance, package/platform promotion, and adoption simplification.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 2 beyond Sprint 137 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 3 Notes

- Wrote the build, package, CI, and report baseline artifact.
- Reviewed Makefile, CMake, pkg-config, package, install, static-deferral,
  benchmark/report, dead-code, coverage, and hosted CI surfaces.
- Recorded the current static-first package proof map: Make install and
  `pkg-config`, CMake install/export and `find_package(Sparse)`, exact-version
  metadata, downstream consumers, no shared artifacts, and
  `BUILD_SHARED_LIBS=ON` rejection.
- Recorded CI lane ownership: Linux reviewed Make/CMake/dead-code/package
  proof plus supplemental runtime/sanitizer/coverage/benchmark lanes; macOS
  reviewed Apple Clang lane plus supplemental package/compiler lanes; Windows
  reviewed CMake subset plus supplemental CMake install/downstream lane.
- Preserved platform asymmetries: Linux remains strongest, macOS install/export
  remains supplemental, Windows install/downstream remains supplemental, and
  `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain staged on
  Windows.
- Inventoried report families: canonical benchmarks, performance sentinels,
  large-matrix guardrails, dead-code, coverage, and package proof logs.
- Confirmed no generated `build/bench-reports`, `build/deadcode`, or
  `coverage` outputs were present in the worktree during the baseline pass, so
  Day 3 records commands and output paths rather than fresh report contents.
- Recorded package and report non-claims so generated reports are not read as
  broad correctness, release, coverage-completeness, portable-performance, or
  platform-parity proof.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 3 beyond Sprint 137 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 4 Notes

- Wrote the Epic 11 residual intake artifact.
- Extracted residuals from the Epic 11 retrospective and the Sprint 136 Day 12
  residual queue publication.
- Reviewed Sprint 130-136 retrospectives for carry-forward detail behind
  partial-SVD, corpus/report, runtime/backend, package/ABI, platform,
  adoption, and closeout residuals.
- Classified residuals as candidates, duplicates/consolidated items,
  already-covered context, optional-local work, or explicit non-claims.
- Grouped active candidates by Epic 12 workstream: corpus/oracle architecture,
  QR residual closure, partial-SVD residual closure, report
  normalization/freshness, runtime/backend governance, package/ABI
  productization, platform promotion/staged tests, adoption/documentation, and
  maintainability support.
- Fenced already-covered Epic 11 work so Sprint 137 does not duplicate Sprint
  130 accepted partial-SVD lanes, Sprint 131 taxonomy/index policy, Sprint 132
  runtime vocabulary, Sprint 133 static-first package decision, Sprint 134
  Linux package-contract promotion, Sprint 135 adoption baseline, or Sprint
  136 unsupported-claim cleanup.
- Recorded unresolved Day 5 questions for package/ABI direction, selected QR
  residual, selected partial-SVD residual, report-family normalization scope,
  platform promotion choice, and adoption timing.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 4 beyond Sprint 137 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 5 Notes

- Wrote the residual owner and non-goal map artifact.
- Assigned every active Day 4 residual candidate to one primary Epic 12 owner
  workstream with supporting owners where dependencies cross boundaries.
- Defined owner workstreams for corpus/oracle, QR, partial-SVD, report index,
  runtime/backend, package/ABI, platform, adoption/docs, maintainability, and
  closeout work.
- Recorded dependency order from baseline evidence through residual intake,
  owner assignment, Day 6 selection criteria, Day 7 gap decision, Days 8-10
  evidence templates, Days 11-12 quality/claim gates, and Days 13-14 handoff.
- Published an explicit Epic 12 non-goal register covering unqualified
  state-of-the-art status, broad ecosystem parity, GPU/distributed support,
  broad package-manager distribution, portable performance superiority, full
  giant-test decomposition, unsupported platform parity, coverage-completeness,
  dead-code removal-readiness, and generated-report overclaims.
- Defined promotion gates for corpus/oracle, QR, partial-SVD, report/freshness,
  runtime/backend, package/ABI, platform, and adoption claims.
- Recorded stop conditions for overbroad residuals, missing owners, weak
  promotion gates, flattened report schemas, unsupported ABI/platform claims,
  raw-basis parity assumptions, and docs-only support widening.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 5 beyond Sprint 137 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 6 Notes

- Wrote the gap-selection criteria artifact.
- Defined the complete-closure requirements for scope boundary, ownership,
  implementation path, evidence source, validation command, row/output
  semantics, support tier, documentation alignment, non-claim preservation, and
  residual disposition.
- Published a 21-point scoring rubric across user value, state-of-the-art
  relevance, dependency readiness, testability/proofability, platform/package
  risk, documentation/adoption impact, and complete-closure feasibility.
- Added score interpretation bands so Day 7 can distinguish strong candidates,
  narrow candidates, deferral/splitting candidates, and rejected Epic 12 scope.
- Added candidate pre-screen rules for corpus/oracle, QR, partial-SVD, report,
  runtime/backend, package/ABI, platform, adoption, and maintainability work.
- Recorded anti-goals that block shallow residual coverage, unsupported
  fixture/report/package/platform promotion, docs-only claim expansion, and
  broad state-of-the-art wording.
- Added a claim gate matrix covering corpus/oracle, QR, partial-SVD,
  report/freshness, runtime/backend, package/ABI, platform, and adoption
  claims.
- Reviewed feasibility against the remaining Sprint 138-146 budget and
  constrained later selection toward one complete closure target per
  implementation sprint unless proof ownership is shared.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 6 beyond Sprint 137 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 7 Notes

- Wrote the Epic 12 gap-selection decision artifact.
- Applied the Day 6 scoring rubric to the Day 5 residual owner map and scored
  candidate families across user value, state-of-the-art relevance, dependency
  readiness, testability/proofability, platform/package risk,
  documentation/adoption impact, and complete-closure feasibility.
- Selected one primary target for every later Epic 12 sprint:
  Sprint 138 corpus/oracle contract, Sprint 139 QR rank-deficient
  nullspace/subspace closure, Sprint 140 partial-SVD repeated/clustered
  spectrum closure with convergence-budget semantics, Sprint 141 report
  normalization/freshness, Sprint 142 runtime/backend precedence plus one
  sentinel lane, Sprint 143 static-first package/ABI follow-through,
  Sprint 144 Windows CMake install/downstream platform lane, Sprint 145
  adoption front door, and Sprint 146 closeout.
- Explicitly deferred or rejected non-selected residuals, including broad
  SuiteSparse/minimum-norm expansion, lower-priority QR/SVD residuals, extra
  sentinel lanes, macOS reviewed promotion, Windows pthread/POSIX staged-test
  promotion, documentation automation, shared-library packaging, dynamic ABI,
  package-manager support, unqualified state-of-the-art status, GPU support,
  distributed support, and broad external-library parity.
- Recorded dependency-ordered handoff notes from corpus/oracle through solver
  residuals, reports, runtime/backend, package/ABI, platform, adoption, and
  closeout.
- Recorded claim boundaries for every selected target so later sprints cannot
  widen solver, report, runtime, package, platform, adoption, or
  state-of-the-art wording without earned evidence.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 7 beyond Sprint 137 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.
