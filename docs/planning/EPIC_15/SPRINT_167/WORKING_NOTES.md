# Sprint 167 Working Notes

## Sprint Goal

Establish the Epic 15 baseline and define exact evidence gates for
performance, ABI, package, API, comparison, and platform claims.

## Source Artifact Note

The Sprint 167 request referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but
the active merged Sprint 167 planning source is
`docs/planning/EPIC_15/PROJECT_PLAN.md`, section "Sprint 167: Epic 15
Baseline, Evidence Ledger & Claim Gate".

## Branch Baseline

- Branch: `sprint-167`
- Starting point: current `master` after PR #185 merge.
- Epic 14 status: complete through PR #184, with residuals intentionally
  carried into Epic 15 planning.
- Epic 15 planning status: review, gap-closure todo, and project plan are
  present under `docs/planning/EPIC_15/`.
- Sprint 167 status: day-by-day plan exists at
  `docs/planning/EPIC_15/SPRINT_167/PLAN.md`.

## Initial Evidence Categories

| Category | Scope | Initial owner candidates |
| --- | --- | --- |
| Claims | Current user-facing statements about solver capability, adoption, package support, reports, and validation. | `README.md`, `INSTALL.md`, docs indexes, report indexes, sprint/epic retrospectives |
| Non-claims | Explicitly unsupported or deferred claims that must not drift into public docs. | README non-claim sections, package docs, ABI/package retrospectives |
| Reports | Generated, normalized, source-controlled, local-only, hosted, and advisory report outputs. | `reports/`, `docs/reports/`, report-index files, normalizer scripts |
| CI | Hosted reviewed and supplemental evidence for Linux, macOS, and Windows. | `.github/workflows/*.yml` |
| Package | Static-first install, uninstall, CMake package, pkg-config metadata, package-manager support, and ABI boundaries. | `Makefile`, `CMakeLists.txt`, `cmake/`, `tests/test_install.sh`, install docs |
| API | Public headers, generated API docs, examples, tutorials, and declaration coherence. | `include/`, `docs/api_reference.md`, `docs/tutorial.md`, examples |
| Platform | Linux, macOS, Windows, local-only, hosted-only, reviewed, supplemental, and staged support. | CI workflows, install scripts, platform docs, retrospectives |
| Performance | Benchmark rows, methodology metadata, sentinels, hosted/local status, and superiority non-claims. | `bench/`, `scripts/`, reports, performance docs |
| Comparison | External oracle/comparison fixtures, runner scripts, tolerances, generated rows, and freshness checks. | comparison scripts, corpus manifests, generated reports |
| Failure behavior | Allocation failure, cleanup invariants, OOM behavior, and partial-construction safety. | allocation helpers, solver setup code, targeted tests |

## Initial Evidence Status Labels

| Label | Meaning |
| --- | --- |
| Supported | The claim has source-controlled evidence and a matching validation command or reviewed CI lane. |
| Partially supported | Evidence exists but is narrower than the claim surface or limited to selected fixtures/platforms. |
| Hosted-only | Evidence depends on GitHub Actions or another hosted lane rather than local reproduction. |
| Local-only | Evidence is reproducible locally but is not hosted or release-published. |
| Advisory | Evidence helps navigation or planning but is not a hard support claim. |
| Deferred | The project intentionally leaves the capability for a later sprint or epic. |
| Unsupported | The project must not claim this capability. |

## Explicit Non-Goals

Sprint 167 does not add or imply:

- a broad state-of-the-art sparse linear algebra claim;
- broad external-library parity with SuiteSparse, Eigen, SciPy, PETSc,
  Trilinos, LAPACK, or vendor sparse libraries;
- portable performance superiority or backend superiority;
- shared-library support;
- dynamic ABI stability;
- runtime-loader behavior;
- package-manager distribution;
- broad Windows, macOS, or Linux parity beyond reviewed evidence;
- hosted publication for local-only generated artifacts;
- API behavior changes;
- source or header implementation changes.

## Working Assumptions

- Day 1 is planning and artifact setup only.
- The active source plan is Epic 15 even though the prompt references Epic 12.
- Later days will determine exact evidence owners before any claim wording is
  changed.
- Local generated reports, normalized indexes, and advisory rows are not
  hosted proof unless a hosted workflow owns them.
- If `.c` or `.h` files change during Sprint 167, run
  `make format && make lint && make test`.
- If only documentation/planning artifacts change, run at least
  `git diff --check` and targeted consistency scans.

## Stop Conditions

Stop and revise before proceeding if a change:

- converts a fixture-scoped solver result into broad solver correctness,
  external-library parity, or state-of-the-art evidence;
- treats local-only generated reports as hosted or published evidence;
- describes methodology-bound benchmark rows as portable performance
  superiority;
- treats static-first package proof as shared-library, dynamic ABI,
  runtime-loader, package-manager, or broad platform package support;
- edits public claims without an evidence-ledger row;
- changes source or header files without running the full quality gate;
- leaves generated build, coverage, cache, install, or report output staged
  for commit unintentionally.

## Daily Log

### Day 1: Sprint Intake And Artifact Setup

- Re-read the Sprint 167 section of
  `docs/planning/EPIC_15/PROJECT_PLAN.md`.
- Reviewed the Epic 15 Codex review and gap-closure todo artifacts.
- Created the Sprint 167 artifact structure.
- Recorded the prompt path mismatch and active Epic 15 source artifact.
- Defined initial evidence categories for claims, non-claims, reports, CI,
  package, API, platform, performance, comparison, and failure behavior.
- Defined evidence status labels, explicit non-goals, working assumptions, and
  stop conditions.
- Created `artifacts/day1-sprint-intake.md`.

### Day 2: Prior Epic Residual Audit

- Reviewed `docs/planning/EPIC_13/EPIC_13_RETROSPECTIVE.md` for deferred
  work, retained non-claims, and next-epic candidates.
- Reviewed `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md` for completed
  closures, narrowed outcomes, retained non-claims, and the final residual
  queue.
- Classified Epic 13 priority residuals as closed, narrowed, or still open
  after Epic 14.
- Identified Epic 14 residuals as the primary Sprint 167 evidence-ledger input
  for Epic 15.
- Preserved long-horizon deferrals for broad external parity, portable
  performance superiority, broad state-of-the-art positioning, dynamic ABI,
  shared-library support, package-manager distribution, broad generated-report
  parity, and Windows Makefile/`pkg-config` parity.
- Created `artifacts/day2-prior-epic-residual-audit.md`.

### Day 3: Residual Risk And Value Classification

- Classified residual IDs R167-01 through R167-10 by claim risk, user value,
  closure feasibility, dependencies, and recommended Epic 15 handling.
- Ranked hosted performance publication, shared-library ABI product design,
  package-manager readiness, broader public-header cleanup, additional
  bounded comparison coverage, cross-platform report freshness, generated API
  publication status, and allocation-failure evidence as the closeable Epic 15
  candidate set.
- Kept broad state-of-the-art/external parity as a high-risk explicit
  non-claim rather than a closeable Sprint 167 selection target.
- Identified dependency chains among ABI/package decisions, generated
  docs/header cleanup, comparison/report freshness, and performance/CI
  publication.
- Created `artifacts/day3-residual-risk-value-classification.md`.

### Day 4: Source And Header Surface Inventory

- Inventoried current source and public-header files under `src/` and
  `include/`.
- Identified large implementation hotspots led by `sparse_ldlt_csc.c`,
  `sparse_lu_csr.c`, `sparse_ldlt.c`, `sparse_iterative.c`, `sparse_qr.c`,
  `sparse_eigs.c`, `sparse_svd.c`, `sparse_chol_csc.c`, `sparse_matrix.c`,
  and `sparse_lu.c`.
- Identified public-header cleanup candidates led by QR, SVD, LDLT, LU,
  LU CSR, ILU, IC, Cholesky, CSR, reorder, and analysis headers.
- Mapped implementation families to public headers and maintained examples
  where obvious.
- Ranked allocation-failure candidate subsystems based on allocation density
  and bounded testability, with LU CSR, LDLT CSC, QR, LDLT, and partial SVD as
  leading candidates.
- Recorded that future `.c` or `.h` edits must trigger
  `make format && make lint && make test`.
- Created `artifacts/day4-source-header-surface-inventory.md`.

### Day 5: Test And Corpus Surface Inventory

- Inventoried the tracked test surface: 59 `tests/test_*.c` proof-owner tests,
  13 tracked test helper headers, 20 tracked corpus metadata/expected files,
  five external dense reference helpers, and six tracked
  corpus/report/comparison scripts.
- Mapped major solver-family tests for matrix/core, LU/LU CSR, Cholesky/LDLT,
  QR, SVD/partial SVD, iterative solvers, eigensolvers, graph/reorder, package
  install, and report/comparison infrastructure.
- Reviewed maintained corpus manifests and report-family contracts for QR,
  partial SVD, selected oracle rows, selected comparison rows, package rows,
  CI rows, benchmark rows, sentinel rows, and advisory report families.
- Distinguished source-controlled advisory metadata from generated local
  evidence and reviewed hosted Linux selected oracle/comparison freshness.
- Identified candidate Epic 15 comparison families: LU CSR dense solve, LDLT
  SPD/semidefinite solve, Cholesky CSC SPD solve, QR rank-deficient
  least-squares follow-up, and partial-SVD clustered/repeated follow-up.
- Recorded platform/test-scope boundaries, including Windows CMake CTest
  count `59`, reviewed Linux selected oracle/comparison freshness, and retained
  non-claims for broad platform/report parity.
- Created `artifacts/day5-test-corpus-surface-inventory.md`.

### Day 6: CI And Workflow Inventory

- Reviewed `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, and
  `.github/workflows/windows-ci.yml`.
- Mapped Linux reviewed and supplemental lanes: Makefile compile-quality,
  CMake parity, dead-code completeness, static package contract, selected
  oracle/comparison freshness, direct runtime, sanitizer, fast benchmark,
  ThreadSanitizer, and coverage.
- Mapped macOS reviewed and supplemental lanes: Apple Clang reviewed
  Make/CMake paths, wall-check, sanitizer, Homebrew GCC supplemental build/test,
  Make install/`pkg-config`, CMake install/export, and static-first package
  deferral.
- Mapped Windows reviewed lanes: CMake configure/build, CTest registration and
  execution with expected count `59`, and CMake install/downstream package
  validation.
- Identified local-only or advisory checks without hosted proof, including
  canonical benchmark publication, performance sentinels, generated API HTML,
  broad generated-report freshness, optional data, and package-manager
  distribution.
- Recorded CI brittleness notes for expected test counts, generated artifact
  paths, hosted service/action availability, runner image/toolchain pinning,
  and platform-specific path/shell behavior.
- Created `artifacts/day6-ci-workflow-inventory.md`.

### Day 7: Package And Install Evidence Inventory

- Reviewed Makefile install/uninstall behavior, CMake install/export behavior,
  `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, install validation scripts,
  static deferral guard, README package wording, `INSTALL.md`, and package
  guidance in `docs/maintainer_guide.md`.
- Mapped the supported package contract as static-first: static library,
  public headers, generated version header, pkg-config metadata, CMake package
  metadata, downstream consumers, exact version checks, and uninstall cleanup.
- Separated Unix Make/pkg-config execution proof from Windows metadata-only
  `sparse.pc` inspection and Windows CMake-first downstream proof.
- Recorded ABI and package non-claims for shared libraries, dynamic ABI,
  runtime-loader behavior, static/shared selectors, package-manager
  distribution, Windows Makefile parity, and Windows `pkg-config` execution
  parity.
- Identified package-manager readiness candidates for Epic 15: formal
  deferral, source-package archive proof, vcpkg manifest proof, Homebrew
  formula proof, or CPack/archive installer proof.
- Created `artifacts/day7-package-install-evidence-inventory.md`.

### Day 8: Documentation And Claim Surface Inventory

- Reviewed README, INSTALL, benchmark, API reference, tutorial, cookbook,
  solver-selection, maintainer-guide, corpus README, corpus schemas, report
  family manifests, and planning artifacts for claim ownership.
- Classified current user-facing documents, maintainer interpretation docs,
  source-controlled evidence metadata, generated local artifacts, hosted CI
  logs, and historical planning artifacts.
- Mapped authoritative claim owners for installation/package support,
  generated API HTML, report freshness, selected comparison rows, performance
  reports, platform tiers, public API, solver guidance, and state-of-the-art
  non-claims.
- Identified stale or ambiguous wording candidates for Day 9/Day 10 ledger
  review, especially Epic 12 path references in current Epic 15 sprint
  prompts, generated API local-only status, benchmark publication wording,
  package-manager support, and historical planning artifacts.
- Created `artifacts/day8-documentation-claim-surface-inventory.md`.

### Day 9: Evidence Ledger Draft

- Created the first Epic 15 evidence ledger draft in
  `artifacts/day9-evidence-ledger-draft.md`.
- Added ledger rows for build/test quality, Linux/macOS/Windows platform tiers,
  static-first package support, shared-library/ABI non-claims,
  package-manager readiness, generated API HTML, public API/header coherence,
  solver correctness, maintained corpus/oracle rows, selected external
  comparison rows, benchmark/performance evidence, report freshness, and
  allocation/failure-path evidence.
- Classified each row as supported, partially supported, local-only,
  hosted-only, advisory, deferred, or unsupported.
- Attached source files, commands, report owners, CI lanes, future sprint
  owners, and retained non-claims to the draft rows.
- Identified missing evidence links and unclear owners for Day 10 correction,
  including hosted performance publication, package-manager distribution,
  shared-library ABI policy, generated API publication, broad report platform
  parity, allocation-failure proof, and PR #184 hosted-result reconciliation.

### Day 10: Evidence Ledger Review And Corrections

- Reviewed the Day 9 ledger against public docs, install/package metadata,
  CI workflows, report-index ownership, and prior-epic residuals.
- Corrected ledger posture where selected or functional evidence could be
  over-read as broader support: generated API HTML is local-only, broad
  generated-report freshness is unsupported beyond selected rows, and
  allocation/failure-path evidence is deferred until deterministic
  failure-injection proof exists.
- Kept Linux, macOS, and Windows platform rows scoped to named hosted lanes
  rather than broad platform parity.
- Preserved static-first source install as supported only for maintained
  source install/export paths, with shared-library support, dynamic ABI
  stability, runtime-loader behavior, and package-manager distribution left as
  explicit non-claims.
- Added explicit non-claim rows for state-of-the-art status, broad
  external-library parity, portable performance superiority, shared-library
  support, dynamic ABI stability, package-manager distribution, broad platform
  parity, Windows Makefile/`pkg-config` parity, generated API HTML
  publication, broad report freshness, broad allocation-failure guarantees,
  and solver correctness beyond maintained fixtures.
- Assigned each high-risk row to a future Sprint 168-176 owner or retained
  deferral label.
- Created `artifacts/day10-evidence-ledger-review.md`.

### Day 11: Gap Selection Gate

- Compared the ranked Day 3 residuals against the reviewed Day 10 evidence
  ledger.
- Selected a finite Epic 15 closure set: hosted methodology-bound performance
  publication, shared-library ABI product decision, package-manager readiness
  or formal deferral, one public-header coherence batch, generated API HTML
  publication status, one bounded external comparison family, one
  cross-platform report freshness promotion or deferral, one deterministic
  allocation-failure proof, and final claim recalibration.
- Mapped selected gaps to Sprint 168 through Sprint 176 owners.
- Deferred broad state-of-the-art status, broad external-library parity,
  portable performance superiority, broad platform parity, ecosystem-wide
  package-manager distribution, broad dynamic ABI stability, broad all-family
  report freshness, broad solver correctness, broad allocation-failure
  guarantees, and Windows Makefile/`pkg-config` parity.
- Recorded selection dependencies so later sprints keep ABI/package,
  performance/methodology, header/API publication, comparison/report, and
  final claim recalibration work in a coherent order.
- Created `artifacts/day11-gap-selection-gate.md`.

### Day 12: Acceptance Criteria And Stop Conditions

- Converted the selected Day 11 gaps into objective acceptance criteria for
  hosted performance publication, shared-library ABI decision,
  package-manager readiness, public-header coherence, generated API HTML
  status, bounded external comparison expansion, cross-platform report
  freshness, deterministic allocation-failure proof, and final claim
  recalibration.
- Defined required local validation commands and hosted-evidence requirements
  for each selected gap.
- Added stop conditions for ambiguous evidence, over-broad public wording,
  missing full C quality gates after source/header edits, performance
  methodology gaps, package/ABI overclaims, package-manager support drift,
  header/API behavior drift, generated API publication ambiguity, comparison
  tolerance ambiguity, platform freshness overclaims, allocation-failure
  nondeterminism, and final closeout drift.
- Added an implementation handoff template for Sprints 168-176.
- Linked evidence ledger rows to the new acceptance gates.
- Created `artifacts/day12-claim-gates.md`.

### Day 13: Sprint Reconciliation And Sprint 168 Handoff

- Reconciled the Sprint 167 artifact set from intake through claim gates.
- Confirmed the selected gap set remains consistent across the residual
  ranking, evidence ledger, gap-selection gate, and claim-gate artifact.
- Reaffirmed the final Sprint 167 claim posture: local quality is command
  supported; platform evidence is tiered and hosted-job scoped; static-first
  source install is supported for maintained paths; shared-library support,
  dynamic ABI stability, package-manager distribution, broad platform parity,
  broad external parity, portable performance superiority, broad report
  freshness, broad solver correctness, broad allocation-failure guarantees,
  and unqualified state-of-the-art status remain non-claims.
- Prepared a Sprint 168 handoff recommending the existing
  `bench_refactor_csc` canonical report path as the first hosted
  performance-publication candidate, subject to runtime and methodology
  validation.
- Recorded alternative performance candidates, Sprint 168 prerequisites,
  evidence boundaries, initial tasks, open residuals, and hosted-evidence
  needs.
- Created `artifacts/day13-sprint-reconciliation.md`.

### Day 14: Final Validation And Closeout

- Created the final Sprint 167 closeout artifact.
- Confirmed the Sprint 167 artifact set includes the plan, working notes, and
  Day 1 through Day 14 artifacts.
- Mapped Sprint 167 project-plan items 167.1 through 167.6 to completed
  artifacts.
- Reconciled the final evidence posture across quality, platform, package,
  ABI, package-manager, generated API, public header, corpus/oracle,
  comparison, performance, report freshness, allocation/failure behavior, and
  state-of-the-art claim surfaces.
- Confirmed Sprint 168 can begin with `bench_refactor_csc` through
  `make bench-canonical-report` as the recommended hosted
  performance-publication candidate, with portable superiority and broad
  backend/platform claims retained as non-claims.
- Recorded skipped checks: `make format`, `make lint`, and `make test` were
  not required because Sprint 167 changed only planning artifacts.
- Created `artifacts/day14-sprint-closeout.md`.
