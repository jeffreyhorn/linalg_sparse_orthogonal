# Sprint 147 Working Notes

## Goal

Sprint 147 freezes the post-Epic-12 baseline, selects Epic 13 closure targets,
and defines evidence gates for platform, corpus, report, ABI, and comparison
work before implementation sprints begin.

## Starting Evidence

- Epic 12 closed with bounded QR and partial-SVD fixture-local proof, report
  governance, static-first package proof, platform support-tier clarity, and a
  final residual queue.
- Epic 13 review identifies the highest-value remaining gaps: Windows parity,
  broader maintained QR and partial-SVD corpus coverage, generated report
  freshness, shared-library ABI/productization, external comparison evidence,
  tutorial/header adoption cleanup, and final claim recalibration.
- Epic 13 todo recommends complete gap closure over shallow progress across
  every residual.
- Sprint 147 owns baseline capture, selected-gap decisions, claim target
  boundaries, evidence gate templates, quality surface mapping, public claim
  freeze, and Sprint 148 Windows prerequisites.

## Baseline Categories

| Category | Primary Sources | Day 1 Capture Rule |
| --- | --- | --- |
| Source/test size | `include/`, `src/`, `tests/`, `benchmarks/`, `examples/`, `scripts/` | Capture reproducible counts and largest-file risks on Day 2. |
| Build/package | `Makefile`, `CMakeLists.txt`, `cmake/`, `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Record static-first package proof and source-list drift risks on Day 2. |
| CI/platform | `.github/workflows/*.yml`, `INSTALL.md`, `docs/maintainer_guide.md` | Record Linux/macOS/Windows reviewed, supplemental, staged, local-only, hosted-only, and deferred lanes on Days 2 and 7. |
| Corpus | `tests/corpus/**`, `scripts/validate_corpus_schema.py`, `scripts/run_corpus_oracle.py` | Separate source-controlled metadata from generated local proof on Day 3. |
| Report | `tests/corpus/manifests/report_families.tsv`, `tests/corpus/schemas/report_index_fields.md`, `scripts/normalize_report_index.py` | Preserve row-meaning and freshness-policy boundaries on Day 3 and Day 9. |
| Documentation | `README.md`, `INSTALL.md`, `docs/*.md`, `benchmarks/README.md`, public headers | Audit unsupported claim wording on Day 13. |
| Residuals | Epic 12 retrospective and Sprint 146 residual queue | Convert residuals R1-R14 into selected gaps, non-goals, and duplicate fences on Days 4-5. |
| Claims | Epic 13 review/todo, Epic 12 non-claims, public docs | Define candidate earned claims and rejected claims on Day 6. |

## Item-To-Day Owner Map

| Sprint 147 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: Baseline Inventory | Days 1-3 | Day 1 sets categories; Days 2-3 capture technical, corpus, report, package, and support baselines. |
| Item 2: Residual Selection | Days 4-5 | Day 4 reconciles Epic 12 residuals; Day 5 selects Epic 13 gaps and duplicate fences. |
| Item 3: Claim Target Register | Day 6 | Day 6 defines candidate earned claims, required evidence, rejected claims, and rollback rules. |
| Item 4: Evidence Gate Templates | Days 7-11 | Days 7-11 define gates for Windows parity, corpus families, generated freshness, ABI/package decisions, and external comparison. |
| Item 5: Quality Surface Map | Day 12 | Day 12 maps required checks by touched surface and defines stop conditions. |
| Item 6: Public Claim Freeze | Day 13 | Day 13 audits public and support surfaces and records wording fixes or no-fix rationale. |
| Item 7: Sprint Closeout | Day 14 | Day 14 validates artifacts, publishes Sprint 148 prerequisites, and prepares retrospective input notes. |

## Stop Conditions

- A proposed Epic 13 claim cannot be tied to concrete implementation,
  validation, report, package, CI, documentation, or comparison evidence.
- A residual is silently promoted without its owner, blocker, prerequisite, and
  promotion gate.
- A state-of-the-art or broad external-parity claim lacks direct comparative
  evidence.
- A platform claim lacks hosted platform evidence.
- A package or ABI claim lacks downstream consumer proof.
- Generated report rows are treated as source-controlled pass evidence.
- A Day 13 public-claim scan finds unsupported wording that cannot be fixed
  within sprint scope.
- Required quality checks fail or their scope is unclear.

## Daily Log

### Day 1: Baseline Intake

- Re-read the Sprint 147 project-plan section and converted Items 1-7 into
  day-level owners.
- Reviewed Epic 12 retrospective, Sprint 146 residual queue, and Epic 13
  review/todo inputs.
- Created the Sprint 147 artifact directory under
  `docs/planning/EPIC_13/SPRINT_147/artifacts/`.
- Established baseline categories for source/test size, build/package, CI,
  corpus, report, documentation, residuals, and claims.
- Recorded stop conditions for unsupported claim promotion, unclear evidence,
  platform/package overclaiming, generated-report confusion, and failed
  validation.
- Day 2 handoff: capture current source/test/build/package/CI baseline numbers
  and largest-file maintainability risks using reproducible commands.

### Day 2: Technical Baseline

- Captured reproducible file-type counts across implementation, tests,
  scripts, workflows, CMake/package files, and documentation.
- Recorded current selected-root totals: 2,986 files, 140 C sources, 53
  headers, 11 Python scripts/helpers, 2,206 Markdown docs, 3 workflows, and 2
  CMake-related files counted by the baseline command.
- Identified the largest maintainability risks, led by `tests/test_qr.c`
  (3,970 lines), `tests/test_ldlt_csc.c` (3,915 lines),
  `tests/test_integration.c` (3,279 lines), and `src/sparse_ldlt_csc.c`
  (2,095 lines).
- Mapped Make and CMake ownership surfaces and noted source-list/test-list
  drift risk.
- Recorded the Windows reviewed CMake lane: `EXPECTED_WINDOWS_CTEST_COUNT=56`
  with `test_threads`, `test_sprint4_integration`, and `test_fuzz` staged out
  for pthread/POSIX blockers.
- Inventoried package proof commands: `tests/test_install.sh`,
  `tests/test_cmake_install.sh`, and `scripts/static_package_deferral_check.sh`
  plus Linux, macOS, and Windows workflow owners.
- Day 3 handoff: capture corpus/report rows, generated-local boundaries, QR
  and partial-SVD proof owners, and freshness validation commands.

### Day 3: Evidence Baseline

- Inventoried the maintained corpus metadata after Epic 12: two fixture rows,
  two generator rows, two expected-result files with eleven ready-for-oracle
  rows, and one disabled optional-data policy row.
- Separated source-controlled corpus inputs from generated-local outputs under
  `build/` and `coverage/`.
- Captured the report-family baseline from
  `tests/corpus/manifests/report_families.tsv`, including source-controlled,
  generated-local, documentation, hosted-CI, and optional-data freshness
  policies.
- Mapped the QR fixture-local closure to `tests/test_qr_corpus.c` and
  `python3 scripts/run_corpus_oracle.py --include-solver-qr`.
- Mapped the partial-SVD fixture-local closure to
  `tests/test_svd_partial_corpus.c` and
  `python3 scripts/run_corpus_oracle.py --include-partial-svd`.
- Recorded validation commands for schema checks, normalized report-index
  checks, freshness diagnostics, and fixture-local proof lanes.
- Day 4 handoff: reconcile Epic 12 residuals against this baseline without
  treating generated-local rows, skip rows, or source-controlled expected rows
  as broad pass evidence.

### Day 4: Residual Intake

- Re-read the Sprint 146 published residual queue and Epic 12 retrospective.
- Grouped R1-R14 by platform/hosted CI, package/ABI, numerical corpus,
  generated reports, adoption/API docs, runtime/backend governance, and
  competitive positioning.
- Assigned every residual an initial Epic 13 disposition: candidate, blocked,
  duplicate, or deferred.
- Identified dependency order from Windows staged portability through install
  parity, shared-library/product decisions, corpus expansion, generated
  freshness, external comparison, adoption cleanup, and final claim
  recalibration.
- Fenced duplicate and overlapping work so R1 remains final evidence
  reconciliation, R13 remains blocked by external comparison evidence, and
  R14 remains deferred behind the ABI/product decision.
- Preserved promotion gates for each residual so none becomes a claim by
  appearing in the intake queue.
- Day 5 handoff: select the final Epic 13 gap set, explicit non-goals, and
  duplicate fences from the Day 4 candidate/deferred/blocked map.

### Day 5: Gap Selection

- Ranked R1-R14 by product value, feasibility, evidence maturity, and closure
  risk.
- Selected the Epic 13 project-plan scope as the closure set: Windows staged
  test portability, Windows install-validation parity decision, QR corpus
  expansion, partial-SVD corpus expansion, generated report freshness,
  shared-library ABI product decision, first narrow external comparison,
  tutorial/header/API coherence, and final claim recalibration.
- Deferred R10 and R11 unless a selected report/freshness gate requires
  runtime/backend sentinel work.
- Deferred R14 package-manager distribution behind the Sprint 153 ABI/package
  product decision.
- Kept R1 as a Sprint 156 hosted evidence reconciliation input rather than a
  standalone implementation gap.
- Kept R13 blocked as a broad state-of-the-art claim unless Sprint 154 earns a
  narrow comparison-backed statement.
- Added duplicate fences for Windows support wording, Windows package parity,
  ABI/distribution, corpus/comparison, generated freshness, adoption docs, and
  competitive claims.
- Day 6 handoff: convert selected gaps into candidate earned claims, required
  evidence, rollback rules, and explicit rejected claims.

### Day 6: Claim Targets

- Converted selected Epic 13 gaps into candidate claim IDs C1-C9 covering
  Windows staged portability, Windows install-validation parity decision, QR
  corpus expansion, partial-SVD corpus expansion, generated freshness,
  shared-library ABI product decision, external comparison, adoption/API
  coherence, and final closeout.
- Mapped each candidate claim to required implementation, validation, report,
  CI, package, comparison, and documentation evidence.
- Preserved rejected claims for broad state-of-the-art status, broad external
  parity, broad QR/partial-SVD correctness, raw basis/vector identity, portable
  performance, generated freshness from source-controlled rows alone, and
  unsupported Windows/package-manager surfaces.
- Marked shared-library ABI support as conditional on Sprint 153 implementation
  and validation; otherwise only stronger static-first deferral can be earned.
- Recorded promotion rules and rollback rules so later sprints can remove or
  narrow claims when local, hosted, package, corpus, report, comparison, or
  docs evidence fails.
- Day 7 handoff: define the Windows evidence gate for candidate claims C1 and
  C2, including staged-test promotion, install-validation parity, CTest count,
  hosted logs, docs, and report rows.

### Day 7: Windows Gate

- Inventoried the current Windows tiers: reviewed MSVC CMake
  configure/build/CTest subset, supplemental CMake install/downstream
  confidence, staged pthread/POSIX tests, deferred Makefile/`pkg-config`
  parity, and unsupported shared/ABI/package-manager surfaces.
- Recorded the current reviewed Windows CTest count as `56`.
- Mapped staged test blockers: `test_threads` and
  `test_sprint4_integration` use pthread APIs directly, and `test_fuzz`
  depends on POSIX temp-file behavior.
- Defined promotion options for each staged test: direct portable source port,
  Windows-native equivalent, split proof owner, or explicit rejection.
- Defined the Windows install-validation parity gate for Sprint 149, keeping it
  separate from Sprint 148 staged-test portability and from Windows Makefile or
  `pkg-config` parity.
- Defined CTest expected-count rules, hosted Windows log requirements,
  documentation/report-row updates, Sprint 148 prerequisites, and stop
  conditions.
- Day 8 handoff: define QR and partial-SVD corpus-family evidence gates while
  preserving fixture-family boundaries and excluding raw-basis/raw-vector
  identity claims.

### Day 8: Corpus Gates

- Defined source-controlled row requirements for fixture, generator,
  expected-result, optional-data, and report-family rows.
- Defined Sprint 150 QR corpus-family promotion gates for rank-deficient
  rectangular solve, underdetermined minimum-norm, and reorder/COLAMD-influenced
  QR candidates.
- Defined QR comparison semantics for exact rank/nullity, residual norms,
  nullspace/subspace checks, minimum-norm metrics, and ordering diagnostics
  without raw QR basis identity.
- Defined Sprint 151 partial-SVD corpus-family promotion gates for repeated or
  clustered spectra, rank-deficient rectangular matrices, sparse low-rank
  output, and convergence/fail-closed behavior.
- Defined partial-SVD comparison semantics for singular values, projector
  distances, triplet residuals, orthogonality, status rows, and diagnostics
  without raw singular-vector identity.
- Recorded oracle/report row requirements, optional-data skip/defer rules,
  validation commands, promotion rules, and stop conditions.
- Day 9 handoff: define generated report freshness gates for selected
  claim-bearing families without turning advisory local rows into pass
  evidence.

### Day 9: Freshness Gates

- Inventoried generated report families relevant to Epic 13 claims: oracle,
  benchmark, sentinel, guardrail, dead-code, coverage, and missing-generated
  report-index rows.
- Selected generated `oracle` rows as the default required-generated target
  after Sprints 150-151 add broader QR and partial-SVD corpus families.
- Kept benchmark, advisory sentinel, coverage, dead-code, and missing-generated
  rows advisory unless a later selected claim explicitly requires them.
- Marked hard-gate sentinel, guardrail, and external comparison rows as
  conditional required-generated candidates only when a sprint selects them as
  claim-bearing.
- Defined freshness metadata requirements for command, artifact path, commit,
  branch, timestamp, platform, compiler, configuration, support tier, status,
  claim scope, and non-claims.
- Defined missing, stale, advisory, skipped, deferred, unsupported, and failing
  row semantics.
- Drafted the CI artifact policy: generated local reports remain ignored
  outputs unless a later sprint records hosted run metadata and artifact
  retention rules.
- Day 10 handoff: define the ABI/package gate using the same source-controlled
  metadata versus executable proof boundary.

### Day 10: ABI Gate

- Inventoried the current static-first package baseline: CMake
  `BUILD_SHARED_LIBS=ON` rejection, static package deferral guard, Make
  install/`pkg-config` proof, CMake install/export proof, Linux reviewed
  package lane, macOS reviewed package lanes, and Windows supplemental
  CMake-first install/downstream confidence.
- Defined Sprint 153 product decision options: implement supported
  shared-library behavior, strengthen tested static-first deferral, and keep
  package-manager distribution deferred behind ABI/release mechanics.
- Defined shared-library implementation evidence for public symbols, headers,
  visibility/export policy, version/ABI policy, build rules, install/export
  metadata, loader proof, downstream consumers, hosted platform proof, and
  documentation.
- Defined stronger static-first deferral evidence so shared support cannot be
  accidentally implied if Sprint 153 rejects implementation.
- Recorded package metadata, downstream consumer, platform validation,
  documentation, non-claim, and stop-condition requirements.
- Day 11 handoff: define the external comparison evidence gate with the same
  direct-proof standard for external parity and state-of-practice wording.

### Day 11: Comparison Gate

- Defined candidate Sprint 154 comparison targets from QR maintained corpus
  families, partial-SVD maintained corpus families, existing direct-solver
  dense-reference patterns, and benchmark/sentinel rows.
- Chose a single QR or partial-SVD maintained corpus family as the default
  bounded target shape, depending on which Sprint 150 or Sprint 151 family
  lands with stronger evidence.
- Defined dependency and optional-data policy for external library/tool name,
  version, installation method, invocation, availability, skip/defer reason,
  platform support, and license/terms.
- Defined a comparison row schema covering fixture key, project command,
  external library/version/command, metric, tolerance, status, platform,
  compiler, configuration, commit, timestamp, claim scope, and non-claims.
- Defined metric rules for QR, partial-SVD, performance, and platform evidence.
- Recorded acceptable narrow wording and rejected wording for broad
  state-of-the-art, ecosystem parity, portable performance, raw basis/vector
  identity, and cross-platform overclaiming.
- Day 12 handoff: turn Windows, corpus, freshness, ABI/package, and comparison
  gates into a touched-surface quality map with stop conditions.

### Day 12: Quality Map

- Mapped validation owners for C implementation, headers, proof-owner tests,
  scripts, Makefile, CMake, CI workflows, package/install metadata, corpus
  metadata, report indexes, documentation, benchmarks, generated artifacts, and
  external comparison work.
- Defined the full C quality gate trigger: any `.c` or `.h` change requires
  `make format && make lint && make test` before commit or review-response
  closure.
- Marked build-registration, source-list, package-metadata, and
  platform-support changes as strong full-gate candidates even when their
  immediate diffs are not C source files.
- Tied supplemental checks to the Day 7-11 gates: Windows hosted CMake evidence,
  corpus schema/oracle checks, generated report freshness checks, static package
  install/export checks, shared ABI decision proof, and external comparison
  dependency/version proof.
- Recorded stop conditions for failing required checks, unclear review feedback,
  missing hosted proof, CTest count drift, stale generated rows, incomplete
  corpus metadata, unsupported package/ABI wording, and unsupported public
  claims.
- Seeded Sprint 156's final validation package with code-quality, corpus,
  platform, package, external comparison, public-claim, and residual-register
  expectations.
- Day 13 handoff: use the quality map to audit README, INSTALL, benchmark docs,
  maintainer guide, tutorials, solver-selection docs, and public headers for
  unsupported widened claims before implementation sprints begin.

### Day 13: Claim Freeze

- Scanned README, INSTALL, maintainer guide, solver-selection, cookbook,
  tutorial, benchmark docs, and public headers for wording around
  state-of-the-art claims, external parity, shared-library ABI,
  package-manager support, Windows parity, performance, and generated report
  freshness.
- Classified current wording as supported narrow claims, explicit non-claims,
  or residual boundaries; found no clear unsupported public claim requiring a
  documentation fix.
- Froze the current baseline: QR and partial-SVD claims remain fixture-local,
  package/install support remains static-first, Windows claims remain tiered,
  benchmark rows remain local measurement evidence, and generated report rows
  remain freshness/navigation diagnostics unless a later sprint selects them as
  required-generated proof.
- Recorded implementation-sprint warnings for Sprints 148-156 so evidence work
  does not widen into broad platform parity, package-manager support,
  shared-library ABI, portable performance, external-library parity, or
  state-of-the-art wording.
- Day 14 handoff: use the frozen claim baseline while preparing Sprint 147
  closeout, artifact index, validation summary, and Sprint 148 Windows
  prerequisite checklist.

### Day 14: Closeout Handoff

- Reviewed the Sprint 147 artifact set and confirmed Day 1-13 deliverables
  cover baseline intake, technical baseline, corpus/report baseline, residual
  intake, selected gaps, claim targets, Windows gates, corpus gates, generated
  freshness, ABI/package, external comparison, quality map, and public claim
  freeze.
- Published the Sprint 147 closeout artifact with deliverable status, artifact
  index, handoff map, selected-gap index, residual/non-goal register,
  validation summary, and retrospective input notes.
- Published the Sprint 148 Windows prerequisite checklist: current reviewed
  Windows workflow/job, `EXPECTED_WINDOWS_CTEST_COUNT=56`, staged test
  surfaces, blockers, promotion rules, hosted evidence requirements, and
  Windows non-claims.
- Confirmed Sprint 147 remains planning/documentation-only; no `.c` or `.h`
  files were changed, so the full C quality gate is not required for Day 14.
- Sprint 148 handoff: begin with staged-test source and CMake gate audit before
  changing expected CTest count or support wording.
