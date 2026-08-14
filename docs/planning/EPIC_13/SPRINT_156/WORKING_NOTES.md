# Sprint 156 Working Notes

## Goal

Sprint 156 closes Epic 13 by validating the final evidence state, recalibrating
public claims and non-claims, publishing residuals, writing the Epic 13
retrospective, and preparing a next-epic handoff grounded in reviewed evidence
rather than aspiration.

## Starting Evidence

- Sprint 147 established Epic 13 baseline gaps, claim targets, and evidence
  gates.
- Sprint 148 promoted Windows staged test portability work and recorded staged
  blockers.
- Sprint 149 made the Windows install-validation parity decision and promoted
  reviewed CMake install/downstream validation while preserving Windows
  Makefile and `pkg-config` non-claims.
- Sprint 150 expanded the maintained QR corpus family and report evidence.
- Sprint 151 expanded the maintained partial-SVD corpus family and report
  evidence.
- Sprint 152 published generated report freshness policy and selected
  freshness gates.
- Sprint 153 preserved the static-first package contract and deferred
  shared-library ABI support as a product decision.
- Sprint 154 created the first narrow external comparison harness and
  `qr-minnorm` study without broad ecosystem parity claims.
- Sprint 155 aligned the tutorial, selected public headers, and API reference
  entry point while preserving declaration and claim boundaries.

## Item-To-Day Owner Map

| Sprint 156 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: Final Evidence Inventory | Days 1-2 | Day 1 inventories source structure and stop conditions; Day 2 builds the final evidence matrix. |
| Item 2: Full Quality Baseline | Days 3-5 | Day 3 designs validation scope; Day 4 runs local baseline; Day 5 validates install/package surfaces. |
| Item 3: Cross-Platform/CI Reconciliation | Day 6 | Day 6 reconciles Linux, macOS, Windows, reviewed, supplemental, staged, local-only, and deferred evidence. |
| Item 4: Claim And Non-Claim Audit | Days 7-10 | Day 7 validates corpus/report evidence; Day 8 validates comparison evidence; Day 9 validates adoption/API evidence; Day 10 audits public claims. |
| Item 5: Residual Queue Publication | Day 11 | Day 11 consolidates residuals with owners, blockers, prerequisites, and promotion gates. |
| Item 6: Epic 13 Retrospective | Day 12 | Day 12 drafts the Epic 13 retrospective from evidence, claims, non-claims, validation, and residuals. |
| Item 7: Final Project Plan Reconciliation | Days 13-14 | Day 13 reconciles plan status; Day 14 finalizes closeout and next-epic handoff. |

## Stop Conditions

- Any public documentation presents the project as an unqualified
  state-of-the-art sparse linear algebra library.
- Narrow QR, partial-SVD, direct-solver, eigensolver, comparison, benchmark, or
  corpus evidence is converted into broad external-library or ecosystem parity.
- Local benchmark, report, sentinel, or comparison rows are treated as portable
  performance proof.
- Static-first package metadata is interpreted as shared-library support,
  dynamic ABI compatibility, runtime-loader compatibility, package-manager
  distribution, or static/shared selector support.
- Windows reviewed CMake-first support is widened into Windows Makefile parity,
  Windows `pkg-config` execution parity, or broad Windows platform parity.
- Generated report rows or generated API HTML are treated as fresh evidence
  without the selected freshness command or publication policy.
- Optional external dependencies, external-service outages, skipped CI lanes,
  staged blockers, or local-only checks are counted as passing support evidence.
- `.c` or public `.h` edits land without `make format && make lint && make
  test`.
- Public header edits land without declaration-preservation evidence.

## Validation Surfaces

- local formatting, lint, and test gates;
- Make install, uninstall, and `pkg-config` package proof;
- CMake install/export and downstream consumer proof;
- Windows reviewed CMake-first CTest and install/downstream proof;
- macOS and Linux reviewed/supplemental package lanes;
- QR and partial-SVD maintained corpus tests and report rows;
- generated report freshness normalization commands;
- external comparison harness and `qr-minnorm` report family;
- README, INSTALL, tutorial, cookbook, solver-selection, benchmark/report,
  maintainer, support-tier, API reference, and selected public-header wording;
- Sprint 147-155 plans, working notes, artifacts, retrospectives, and handoffs.

## Daily Log

### Day 1: Sprint Intake And Closeout Baseline

- Re-read the Sprint 156 section of
  `docs/planning/EPIC_13/PROJECT_PLAN.md`.
- Confirmed Sprint 156 is the Epic 13 final validation and claim
  recalibration sprint, even though the initiating prompt referenced the stale
  Epic 12 project-plan path.
- Created the Sprint 156 artifact directory.
- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day1-closeout-baseline.md`.
- Inventoried Sprint 147 through Sprint 155 evidence sources:
  - all nine prior Epic 13 sprint directories have a plan, working notes,
    retrospective, and day artifacts;
  - Sprint 152 and Sprint 154 include additional cross-sprint/report study
    artifacts;
  - Sprint 155 includes declaration-preservation scans and the explicit Sprint
    156 handoff.
- Recorded closeout stop conditions for state-of-the-art, parity, ABI,
  package-manager, platform, performance, report freshness, and generated API
  reference claims.
- Identified validation surfaces touched during Epic 13.
- Day 2 handoff: build the final evidence inventory matrix across platform,
  corpus, report, ABI/package, comparison, adoption, and validation surfaces.

### Day 2: Final Evidence Inventory

- Re-read the Day 2 plan tasks from
  `docs/planning/EPIC_13/SPRINT_156/PLAN.md`.
- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day2-evidence-inventory.md`.
- Inventoried the final Epic 13 evidence by deliverable family:
  - Sprint 147 baseline, gap selection, claim targets, and evidence gates;
  - Sprint 148 Windows staged portability and promoted CMake test count;
  - Sprint 149 Windows reviewed CMake install/downstream validation;
  - Sprint 150 QR maintained corpus and generated-local report rows;
  - Sprint 151 partial-SVD maintained corpus and generated-local report rows;
  - Sprint 152 selected oracle generated report freshness policy;
  - Sprint 153 static-first package contract and shared-library ABI deferral;
  - Sprint 154 first narrow `qr-minnorm` comparison study;
  - Sprint 155 adoption/API/header documentation coherence.
- Classified evidence as reviewed, supplemental, local, generated-local,
  source-controlled, staged, or deferred so later claim audit work can avoid
  treating all evidence as equivalent.
- Recorded missing or stale evidence for Day 3 through Day 10 follow-up:
  hosted CI reconciliation, final local quality baseline, package validation,
  corpus/report freshness checks, comparison freshness, adoption/API scan,
  generated API HTML refresh deferral, remaining public-header cleanup, and
  final public claim audit.
- Day 3 handoff: turn the evidence inventory into a concrete validation matrix
  with commands, required gates, skip/defer semantics, and escalation rules.

### Day 3: Validation Matrix Design

- Re-read the Sprint 147 Day 12 quality surface map and the current Makefile,
  install, CMake, report-index, corpus, and comparison validation targets.
- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day3-validation-matrix.md`.
- Defined the final validation matrix by surface:
  - documentation-only closeout changes require whitespace and claim-evidence
    checks;
  - `.c` or public `.h` changes require `make format && make lint && make
    test`;
  - strongest local code/build baseline is `make quality-review-full`;
  - static-first package proof uses `bash tests/test_install.sh`,
    `bash tests/test_cmake_install.sh`, and
    `bash scripts/static_package_deferral_check.sh`;
  - corpus/report proof uses schema validation, selected oracle generation,
    and `make report-index-oracle-freshness`;
  - comparison proof uses `python3 scripts/run_external_comparison.py
    --self-check` and `make report-index-comparison-freshness`;
  - platform proof remains a hosted CI reconciliation task, not something local
    commands can substitute for.
- Recorded skip/defer semantics for unavailable hosted CI, optional external
  dependencies, generated API HTML refresh, staged Windows lanes, and
  source-controlled report rows.
- Day 4 handoff: run the selected local baseline from the matrix and record
  exact command results, environment details, failures, skips, and deferrals.

### Day 4: Full Local Quality Baseline

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day4-local-baseline.md`.
- Ran Day 4 intake checks:
  - `git status --short --branch`;
  - `git diff --check`;
  - `git diff --name-only master...HEAD`;
  - local environment capture with `uname`, `cc --version`,
    `cmake --version`, and `make --version`.
- Confirmed the Sprint 156 branch delta before Day 4 artifact edits was
  documentation-only under `docs/planning/EPIC_13/SPRINT_156/`.
- Ran the strongest local reviewed baseline:
  - `make quality-review-full`.
- Result: passed.
  - Makefile reviewed path passed: format-check, lint, full `make test`, and
    deadcode-check.
  - CMake reviewed path passed: configure, clean serial build, `ctest -N`,
    Makefile/CMake test-count parity, and full CTest execution.
  - CMake registered `59` tests, matching the Makefile test count.
  - CTest passed `59/59` tests with `0` failures.
- Recorded expected/known limitations:
  - evidence is local macOS evidence only, not hosted Linux/macOS/Windows proof;
  - full C quality gate was not required by the Sprint 156 file delta because
    no `.c` or `.h` files changed, but the stronger local reviewed baseline
    was run for final closeout confidence;
  - CMake build emitted one non-fatal AppleClang warning in
    `tests/test_svd_partial_corpus.c` for `INFINITY` double promotion; the
    reviewed target still passed.
- Day 5 handoff: validate package/install proof with Make install,
  CMake install/export, downstream consumer checks, and static package
  deferral guard.

### Day 5: Package And Install Validation

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day5-package-validation.md`.
- Re-ran the maintained static package guard:
  - `bash scripts/static_package_deferral_check.sh`;
  - result: passed, including `BUILD_SHARED_LIBS=ON` rejection, explicit
    static target checks, install metadata checks, no export/ABI metadata, no
    static/shared selector, and support wording checks.
- Re-ran the maintained Make install and `pkg-config` proof:
  - `bash tests/test_install.sh`;
  - result: passed with `23` checks, `0` failures;
  - installed static archive, no shared artifacts, `19` headers, `sparse.pc`,
    exact version `2.2.0`, `pkg-config` cflags/libs/static libs, downstream
    compile/link/run, maintained example compile/run, and uninstall proof.
- Re-ran the maintained CMake install/export and downstream proof:
  - `bash tests/test_cmake_install.sh`;
  - result: passed with `27` checks, `0` failures, `0` skips;
  - installed static archive, no shared artifacts, `19` headers, CMake package
    files, static imported target metadata, source/build tree path absence,
    downstream `find_package(Sparse)`, exact-version consumer, mismatched
    version rejection, and `pkg-config` version `2.2.0`.
- Re-ran package report-index checks:
  - `python3 scripts/normalize_report_index.py --family package --check`;
  - `python3 scripts/normalize_report_index.py --family package
    --check-freshness`;
  - `python3 scripts/normalize_report_index.py --family runtime_backend
    --check-freshness`;
  - results: package structure passed with `6` rows, package freshness passed
    with `6` source-controlled rows, runtime-backend freshness passed with `1`
    source-controlled row.
- Recorded package evidence boundaries:
  - local Unix/macOS package proof does not prove Windows Makefile or Windows
    `pkg-config` execution parity;
  - static-first package proof does not prove shared-library support, dynamic
    ABI compatibility, runtime-loader behavior, package-manager distribution,
    or static/shared selectors.
- Day 6 handoff: reconcile hosted Linux, macOS, Windows, reviewed,
  supplemental, staged, and deferred platform evidence separately from this
  local package proof.

### Day 6: Platform Reconciliation

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day6-platform-reconciliation.md`.
- Reviewed the maintained platform contract in:
  - `.github/workflows/ci.yml`;
  - `.github/workflows/macos-ci.yml`;
  - `.github/workflows/windows-ci.yml`;
  - `README.md`;
  - `INSTALL.md`;
  - `docs/maintainer_guide.md`.
- Reconciled the platform lanes:
  - Linux remains the strongest reviewed source of truth, with reviewed
    Makefile compile-quality, CMake parity, dead-code, and static-first
    package-contract lanes plus supplemental runtime, benchmark, TSan, and
    coverage signals;
  - macOS carries reviewed Apple Clang Makefile/CMake/wall/sanitizer proof,
    reviewed static-first Make install/`pkg-config` and CMake install/export
    proof, and supplemental Homebrew GCC second-compiler proof;
  - Windows carries reviewed MSVC 2022 CMake-first proof with
    `EXPECTED_WINDOWS_CTEST_COUNT=59`, full hosted CTest, and reviewed CMake
    install/downstream validation.
- Verified recent hosted master merge evidence with `gh run list`:
  - CI run `31723661978` succeeded for PR #172 merge commit
    `c7981c6ef7fa887e87575279009113b1dcf3a630`;
  - macOS CI run `31723661840` succeeded for the same merge commit;
  - Windows CI run `31723661771` succeeded for the same merge commit.
- Separated support labels into reviewed hosted proof, supplemental hosted
  proof, local-only proof, source-controlled policy, deferred non-claim,
  PR-time pending, and external outage.
- Recorded that Sprint 148 closed the prior Windows staged-test exclusions by
  promoting `test_threads`, `test_sprint4_integration`, and `test_fuzz` into
  the reviewed CMake subset.
- Preserved remaining Windows deferred non-claims:
  Windows Makefile parity, Windows `pkg-config` execution parity,
  package-manager support, shared-library support, dynamic ABI support,
  runtime-loader behavior, and broad Windows platform parity.
- Classified hosted setup/download failures such as GitHub Actions
  action-resolution `Service Unavailable` errors as external outages unless a
  rerun reaches repository command execution and reproduces a command failure.
- Recorded that Sprint 156 branch CI is still PR-time pending because the
  current branch has not yet had hosted PR runs.
- Day 7 handoff: validate corpus/report evidence without converting generated
  rows or local proof into broader hosted, platform, or performance claims.

### Day 7: Corpus And Report Validation

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day7-corpus-report-validation.md`.
- Reviewed corpus/report ownership and freshness policy in:
  - `docs/maintainer_guide.md`;
  - `README.md`;
  - `tests/corpus/README.md`;
  - `tests/corpus/manifests/report_families.tsv`;
  - `tests/corpus/manifests/fixtures.tsv`;
  - `tests/corpus/manifests/generators.tsv`;
  - `tests/corpus/expected/*.tsv`;
  - `tests/corpus/schemas/report_index_fields.md`.
- Ran corpus schema validation:
  - `python3 scripts/validate_corpus_schema.py`;
  - result: passed, reporting `tests/corpus ok`.
- Ran focused QR corpus proof:
  - `make build/test_qr_corpus`;
  - `./build/test_qr_corpus`;
  - result: passed with `14` tests, `0` failures, `0` skips, and `258`
    assertions.
- Ran focused partial-SVD corpus proof:
  - `make build/test_svd_partial_corpus`;
  - `./build/test_svd_partial_corpus`;
  - result: passed with `10` tests, `0` failures, `0` skips, and `247`
    assertions.
- Ran selected local oracle freshness:
  - `make report-index-oracle-freshness`;
  - result: passed, regenerating ignored local outputs under `build/` and
    validating required oracle freshness with `54` normalized oracle rows.
- Ran normalized corpus/oracle index checks:
  - `python3 scripts/normalize_report_index.py --family corpus --family oracle
    --check`;
  - result: passed with `128` rows.
- Recorded selected combined oracle output:
  - `52` generated rows total;
  - `23` QR solver-backed rows;
  - `26` partial-SVD solver-backed rows;
  - `3` generated-reference rows;
  - all generated oracle rows had `comparison_status=pass`;
  - selected fixture-key count was `10`.
- Confirmed the current partial-SVD-only documentation count of `105`
  normalized corpus/oracle rows remains interpretable as the partial-SVD-only
  run shape, not the selected combined QR plus partial-SVD gate.
- Preserved local-only boundaries:
  generated corpus/oracle rows are not hosted CI, broad solver correctness,
  external-library parity, platform, package, ABI, performance, release, or
  state-of-the-art proof.
- Published the deferred corpus-family queue with owners and promotion
  criteria for broad QR rank-threshold policy, broad QR solve/minimum-norm
  behavior, QR reordered/COLAMD corpus behavior, SuiteSparse/external corpus,
  broad partial-SVD repeated-spectrum/raw-vector behavior, sparse-output
  optimality, convergence-rate/partial-result semantics, and hosted
  corpus/report evidence.
- Day 8 handoff: validate external comparison evidence without widening the
  selected `qr-minnorm` study into broad QR, external-library, platform, or
  performance parity.

### Day 8: Comparison Study Reconciliation

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day8-comparison-reconciliation.md`.
- Reviewed Sprint 154 comparison artifacts:
  - `day3-comparison-target-selection.md`;
  - `day11-report-integration-implementation.md`;
  - `day13-integrated-validation-and-study-publication.md`;
  - `first-narrow-qr-minnorm-comparison-study.md`.
- Reviewed comparison ownership in:
  - `scripts/run_external_comparison.py`;
  - `scripts/normalize_report_index.py`;
  - `tests/qr_external_dense_reference.py`;
  - `tests/corpus/manifests/report_families.tsv`;
  - `README.md`;
  - `docs/maintainer_guide.md`.
- Ran comparison harness validation:
  - `python3 scripts/run_external_comparison.py --self-check`;
  - result: passed.
- Ran selected comparison freshness:
  - `make report-index-comparison-freshness`;
  - result: passed, regenerating ignored local comparison outputs and
    validating required comparison freshness with `7` normalized rows.
- Re-ran the selected freshness target with `PYTHONDONTWRITEBYTECODE=1` after
  clearing transient Python bytecode so the generated manifest recorded
  `worktree_state=clean`.
- Ran focused normalized comparison checks:
  - `python3 scripts/normalize_report_index.py --family comparison
    --require-generated comparison --check-freshness`;
  - `python3 scripts/normalize_report_index.py --family comparison --output
    build/report-index/day8-comparison-normalized.tsv`;
  - results: passed with `7` rows.
- Recorded current selected study provenance:
  - target `qr-minnorm`;
  - fixture `qr_underdetermined_minnorm_2x4`;
  - baseline `source-controlled-dense-qr-reference`;
  - baseline helper `tests/qr_external_dense_reference.py`;
  - source commit `c00a349b9cab7edd58be79c0f6496c9f1097261b`;
  - branch `sprint-156`;
  - platform `darwin-x86_64`;
  - support tier `local_only`.
- Recorded selected generated comparison rows:
  - `project_status`: pass;
  - `baseline_status`: pass;
  - `residual_norm`: delta `1.5700924586837752e-16`, tolerance `1e-10`;
  - `solution_norm`: delta `1.1102230246251565e-16`, tolerance `1e-10`;
  - `solution_values`: delta `1.1102230246251565e-16`, tolerance `1e-10`;
  - `project_vs_baseline_max_abs_delta`: delta
    `1.1102230246251565e-16`, tolerance `1e-10`;
  - all six generated rows were `pass`.
- Confirmed dependency status:
  - `python3` and `tests/qr_external_dense_reference.py` were required pass
    dependencies;
  - `numpy` and `scipy` remained deferred optional package baselines and were
    not counted as proof.
- Audited comparison wording in README, maintainer guide, and the Sprint 154
  publication artifact; no public wording change was needed because the
  comparison lane already remains fixture-local and local-only.
- Published the residual comparison queue with owners and promotion criteria
  for broader QR targets, optional NumPy/SciPy, ecosystem baselines,
  raw-basis/order metrics, partial-SVD comparison publication, performance,
  hosted CI, package-manager, shared-library, loader, and ABI lanes.
- Day 9 handoff: reconcile tutorial, API reference, cookbook, examples,
  selected headers, and adoption-surface wording against the final evidence
  boundaries.

### Day 9: Adoption And API Surface Reconciliation

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day9-adoption-api-reconciliation.md`.
- Reviewed Sprint 155 adoption and API artifacts:
  - `day5-tutorial-alignment-summary.md`;
  - `day11-api-reference-guidance-implementation.md`;
  - `day12-preservation-and-reconciliation.md`;
  - `day13-integrated-validation.md`;
  - `day14-closeout-sprint156-handoff.md`.
- Reviewed current adoption/API surfaces:
  - `README.md`;
  - `docs/tutorial.md`;
  - `docs/cookbook.md`;
  - `docs/solver_selection.md`;
  - `docs/api_reference.md`;
  - `examples/README.md`;
  - `INSTALL.md`;
  - `benchmarks/README.md`;
  - `docs/maintainer_guide.md`.
- Confirmed the coherent first-use route:
  README front door -> examples first run -> cookbook data-first path ->
  solver-selection problem-shape route -> install only for downstream
  consumers -> benchmarks/reports after workflow selection -> API reference and
  public headers for exact declarations.
- Rechecked API/adoption link targets:
  - `docs/api_reference.md`;
  - `include/`;
  - `docs/tutorial.md`;
  - `docs/cookbook.md`;
  - `docs/solver_selection.md`;
  - `examples/README.md`;
  - result: passed for checked-in targets.
- Reclassified generated API HTML:
  - `docs/api/html/index.html` is generated by `make docs`, ignored by Git,
    and not a checked-in repository link target;
  - any locally present `docs/api/html/` output is local generated output, not
    source-controlled evidence.
- Rechecked Sprint 155 declaration-preservation evidence:
  - Day 8 normalized declaration diff: `0` bytes;
  - Day 9 normalized declaration diff: `0` bytes;
  - Day 12 normalized declaration diff: `0` bytes.
- Reconciled generated API HTML inventory:
  - `18` checked-in public headers;
  - no checked-in generated API HTML tree under `docs/api/html/`;
  - generated API HTML publication remains a future refresh/publish decision.
- Ran stale/unsupported phrase scan across README, tutorial, cookbook,
  solver-selection, API reference, examples README, install docs, maintainer
  guide, and public headers; matches were explicit non-claim wording only.
- Recorded that Day 9 made no `.c` or public `.h` edits, so no new
  declaration scan or full C quality gate was required.
- Published residuals for generated API HTML refresh/publication, generated
  installed `sparse_version.h` Doxygen coverage, remaining public-header
  cleanup, and Day 10 final public claim audit.
- Day 10 handoff: perform the full public claim/non-claim audit across
  state-of-the-art, parity, performance, ABI, package, platform, report, and
  support-tier wording.

### Day 10: Public Claim And Non-Claim Audit

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day10-claim-audit.md`.
- Audited public and support documentation:
  - `README.md`;
  - `INSTALL.md`;
  - `docs/tutorial.md`;
  - `docs/cookbook.md`;
  - `docs/solver_selection.md`;
  - `docs/api_reference.md`;
  - `examples/README.md`;
  - `benchmarks/README.md`;
  - `tests/corpus/README.md`;
  - `docs/maintainer_guide.md`;
  - public headers under `include/`.
- Ran focused public claim scans for unsupported state-of-the-art, ecosystem
  parity, external-library parity, portable performance, shared-library,
  dynamic ABI, runtime-loader, package-manager, Windows Makefile, Windows
  `pkg-config`, broad Windows, and broad platform wording.
- Ran focused support/report scans for reviewed/supplemental support-tier,
  local-only, freshness, report-index, generated-report, benchmark, and
  performance-claim wording.
- Inspected `INSTALL.md` package and support-tier sections directly:
  - static-first package wording remains the maintained package claim;
  - Linux remains the strongest reviewed source of truth;
  - macOS reviewed Apple Clang plus reviewed static-first install/export proof
    remains bounded, with Homebrew GCC supplemental;
  - Windows remains reviewed CMake-first with reviewed CMake install/downstream
    validation;
  - shared-library, dynamic ABI, runtime-loader, package-manager, Windows
    Makefile, Windows `pkg-config`, and broad Windows parity remain explicit
    non-claims.
- Confirmed public corpus/report/comparison wording remains local-only and
  fixture-bound:
  - QR corpus/report evidence maps to Day 7;
  - partial-SVD corpus/report evidence maps to Day 7;
  - the selected QR minimum-norm comparison maps to Day 8 only;
  - benchmark/report rows remain local measurement and freshness diagnostics,
    not portable performance proof.
- Confirmed API/reference wording remains bounded:
  - checked-in public headers remain the declaration source of truth;
  - generated API HTML remains a convenience view with documented refresh and
    missing-page residuals from Day 9.
- No public documentation patch was required because all broad phrase matches
  were already evidence-bound or explicit non-claims.
- Day 10 made no `.c` or public `.h` edits, so no full C quality gate was
  required.
- Day 11 handoff: reconcile final residuals and risk register against the Day
  10 claim audit before final closeout validation.

### Day 11: Residual Queue Publication

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day11-residual-queue-publication.md`.
- Consolidated residual sources from:
  - Sprint 147-155 retrospectives;
  - Sprint 147-155 closeout and handoff artifacts;
  - Sprint 156 Day 5 package validation;
  - Sprint 156 Day 6 platform reconciliation;
  - Sprint 156 Day 7 corpus/report validation;
  - Sprint 156 Day 8 comparison reconciliation;
  - Sprint 156 Day 9 adoption/API reconciliation;
  - Sprint 156 Day 10 claim audit.
- Deduplicated residuals into final categories:
  - platform/package;
  - ABI/package;
  - reports/CI and report tooling;
  - QR corpus and comparison;
  - partial-SVD corpus and comparison;
  - external parity;
  - performance;
  - API/docs;
  - claims/product;
  - runtime/backend.
- Published `18` final residuals with owner role, blocker, prerequisites, and
  promotion gate fields.
- Prioritized next-epic candidates by complete-gap closure potential:
  - generated API HTML refresh;
  - hosted promotion for selected local-only oracle/comparison rows;
  - one bounded QR comparison expansion;
  - one bounded partial-SVD comparison publication;
  - Windows package parity decision;
  - next public-header cleanup batch.
- Separated long-horizon deferrals:
  - package-manager distribution;
  - shared-library product support;
  - dynamic ABI compatibility;
  - broad ecosystem parity;
  - portable performance superiority;
  - broad state-of-the-art positioning;
  - typed runtime/backend API promotion unless tied to selected API and ABI
    scope.
- Preserved non-claim language for state-of-the-art, broad ecosystem parity,
  broad numerical correctness, raw basis/vector identity, portable
  performance, generated report pass evidence, shared-library/ABI/loader,
  package-manager, Windows Makefile, Windows `pkg-config`, and broad Windows
  parity claims.
- Day 11 made no `.c` or public `.h` edits, so no full C quality gate was
  required.
- Day 12 handoff: draft the Epic 13 retrospective using the Day 11 residual
  queue as the closeout residual source of truth.

### Day 12: Epic 13 Retrospective Draft

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day12-retrospective-draft.md`.
- Reviewed prior epic retrospective structure, using Epic 12 as the closest
  format reference for:
  - epic objective;
  - sprint outcomes;
  - major outcomes;
  - validation evidence;
  - earned claims;
  - non-claims;
  - residuals and future-epic candidates;
  - state-of-the-art assessment;
  - what went well / what did not go well.
- Drafted Sprint 147-156 outcomes:
  - Sprint 147 evidence gates and claim targets;
  - Sprint 148 Windows staged CMake test promotion;
  - Sprint 149 Windows CMake install/downstream validation;
  - Sprint 150 QR maintained corpus expansion;
  - Sprint 151 partial-SVD maintained corpus expansion;
  - Sprint 152 selected generated oracle freshness;
  - Sprint 153 static-first/shared-library deferral product decision;
  - Sprint 154 first narrow comparison study;
  - Sprint 155 tutorial/API/header coherence;
  - Sprint 156 final validation, claim audit, and residual queue.
- Summarized validation evidence from Sprint 156 Days 4-11 and the sprint
  retrospectives for Sprints 147-155.
- Carried Day 11 residual queue into the draft as the next-epic handoff source
  of truth.
- Kept branch-specific Sprint 156 hosted CI as PR-time pending rather than
  claiming hosted branch success before PR workflows run.
- Preserved non-claims for broad state-of-the-art, ecosystem parity, broad
  numerical correctness, raw basis/vector identity, portable performance,
  hosted proof for local-only generated rows, generated API HTML completeness,
  shared-library/dynamic ABI/loader/package-manager support, Windows Makefile
  parity, Windows `pkg-config` parity, and broad Windows platform parity.
- Day 12 made no `.c` or public `.h` edits, so no full C quality gate was
  required.
- Day 13 handoff: reconcile this retrospective draft against the Epic 13
  project plan and the completed artifact set before final Day 14 publication.

### Day 13: Project Plan Reconciliation

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day13-project-plan-reconciliation.md`.
- Reviewed `docs/planning/EPIC_13/PROJECT_PLAN.md` against:
  - Sprint 147-155 retrospectives;
  - Sprint 147-155 Day 13/Day 14 validation and closeout artifacts;
  - Sprint 156 Day 1-12 artifacts;
  - Day 11 final residual queue;
  - Day 12 Epic 13 retrospective draft.
- Reconciled Sprint 147-156 status:
  - Sprints 147-148 complete as planned;
  - Sprints 149-155 complete with intentionally narrowed claims or documented
    residuals;
  - Sprint 156 Days 1-13 complete, with Day 14 final closeout remaining.
- Marked Sprint 153's shared-library item complete under the plan's
  implementation-or-deferral branch: static-first support was preserved and
  shared-library support now has exact blocker diagnostics and guard checks.
- Marked generated report freshness complete with narrowed scope:
  selected oracle and comparison freshness landed; benchmark, sentinel,
  coverage, dead-code, and broader generated report families remain residuals.
- Reconciled estimate and validation variances:
  - Windows CTest count guard drift;
  - fixed installed-header counts;
  - PR-time hosted evidence boundaries;
  - local-only generated report freshness;
  - source-controlled comparison helper rather than external package baseline;
  - deferred generated API HTML refresh.
- Confirmed the Day 12 retrospective draft needs no earned-claim removals but
  must remain draft/pending until Day 14 final docs-only hygiene and
  claim/non-claim checks are complete.
- Updated next-epic handoff priorities around complete-gap closure:
  generated API HTML refresh, hosted selected oracle/comparison promotion, one
  bounded QR comparison expansion, one bounded partial-SVD comparison
  publication, Windows parity decision, and next header cleanup batch.
- Day 13 made no `.c` or public `.h` edits, so no full C quality gate was
  required.
- Day 14 handoff: finalize Sprint 156 artifacts, publish the final Epic 13
  retrospective, run final docs-only hygiene and claim checks, and prepare the
  branch for PR closeout.

### Day 14: Final Closeout And Handoff

- Created
  `docs/planning/EPIC_13/EPIC_13_RETROSPECTIVE.md`.
- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day14-closeout-handoff.md`.
- Promoted the Day 12 retrospective draft to final root-level retrospective
  wording after applying the Day 13 project-plan reconciliation:
  - Sprint 156 is now marked complete;
  - Day 13 project-plan reconciliation is included in validation evidence;
  - final Day 14 docs-only checks are included;
  - branch-specific Sprint 156 hosted PR evidence remains explicitly pending
    until PR workflows run.
- Published the final Sprint 156 artifact index for Days 1-14.
- Re-ran the final claim/non-claim scan across the Day 11 residual queue, Day
  12 retrospective draft, Day 13 project-plan reconciliation, and final Day 14
  retrospective wording.
- Preserved final non-claims for:
  - unqualified state-of-the-art sparse linear algebra status;
  - broad ecosystem or external-library parity;
  - broad QR, SVD, or partial-SVD correctness;
  - raw QR basis and raw singular-vector identity parity;
  - portable performance or backend superiority;
  - generated report pass evidence from source-controlled rows alone;
  - hosted proof for local-only generated rows;
  - generated API HTML completeness;
  - shared-library, dynamic ABI, runtime-loader, static/shared selector, and
    package-manager support;
  - Windows Makefile parity, Windows `pkg-config` parity, and broad Windows
    platform parity.
- Confirmed the next-epic handoff should start from
  `day11-residual-queue-publication.md` and prioritize bounded complete-gap
  closure.
- Day 14 made no `.c` or public `.h` edits, so no full C quality gate was
  required.
