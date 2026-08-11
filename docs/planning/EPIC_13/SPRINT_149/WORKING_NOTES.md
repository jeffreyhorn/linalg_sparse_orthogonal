# Sprint 149 Working Notes

## Goal

Sprint 149 decides and implements the reviewed Windows install-validation
support tier without confusing it with Unix Makefile, Windows `pkg-config`, or
shared-library parity.

## Starting Evidence

- Sprint 148 closed the Windows staged-test portability gap for
  `test_threads`, `test_sprint4_integration`, and `test_fuzz`.
- The Windows reviewed lane is still CMake-first: hosted MSVC 2022 configure,
  build, `ctest -N` enumeration with `EXPECTED_WINDOWS_CTEST_COUNT=59`, and
  full hosted CTest execution.
- `.github/workflows/windows-ci.yml::install-and-downstream` is explicitly
  supplemental CMake-first static package confidence, not a reviewed Windows
  install-validation lane.
- Linux CI carries the reviewed static-first package contract in
  `.github/workflows/ci.yml::package-contract`, running
  `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh`.
- macOS CI carries reviewed static-first install/export proof in
  `.github/workflows/macos-ci.yml::install-and-pkgconfig` and
  `.github/workflows/macos-ci.yml::cmake-install-export`.
- Public and maintainer docs currently preserve these Windows non-claims:
  Windows Makefile parity, Windows `pkg-config` parity, separate reviewed
  Windows install-validation parity, package-manager support, shared-library
  support, dynamic ABI support, and broad Windows parity.

## Support-Tier Baseline

| Tier | Current Surface | Sprint 149 Rule |
| --- | --- | --- |
| Reviewed Linux | `.github/workflows/ci.yml::package-contract` | Treat as the strongest package-contract comparison point: Make install/`pkg-config`, CMake install/export, and static deferral guard. |
| Reviewed macOS | `.github/workflows/macos-ci.yml::install-and-pkgconfig` and `::cmake-install-export` | Treat as reviewed static-first install/export proof for the maintained static archive package contract. |
| Reviewed Windows | `.github/workflows/windows-ci.yml::build-and-test` | Preserve as CMake-first configure/build/CTest proof; do not infer install-validation parity from CTest promotion. |
| Supplemental Windows | `.github/workflows/windows-ci.yml::install-and-downstream` | Audit for possible promotion, split, rename, or explicit retained-supplemental status. |
| Local Unix validation | `tests/test_install.sh` and `tests/test_cmake_install.sh` | Use as local direct proof-owner scripts, but do not treat Unix shell/`pkg-config` behavior as Windows parity. |
| Unsupported | shared-library ABI, dynamic ABI, runtime-loader behavior, package-manager distribution, Windows Makefile parity, Windows `pkg-config` execution parity | Keep explicit non-claims unless a later sprint creates a separate product contract and evidence lane. |

## Item-To-Day Owner Map

| Sprint 149 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: Windows Package Audit | Days 1-2 | Day 1 captures the baseline; Day 2 compares Windows proof to Linux/macOS reviewed lanes. |
| Item 2: Promotion Criteria | Days 3-4 | Day 3 defines criteria; Day 4 applies them to the product decision. |
| Item 3: Workflow Implementation | Days 5-6 | Day 5 designs workflow edits; Day 6 implements the selected lane status. |
| Item 4: Package Metadata Checks | Days 7-8 | Day 7 designs stronger checks; Day 8 implements package metadata assertions. |
| Item 5: Downstream Consumer Proof | Days 9-10 | Day 9 designs consumer proof; Day 10 implements normal, exact-version, and mismatch-version proof. |
| Item 6: Documentation Alignment | Day 11 | Day 11 aligns README, INSTALL, maintainer guidance, and report rows. |
| Item 7: Validation And Closeout | Days 12-14 | Day 12 validates locally, Day 13 integrates evidence, and Day 14 closes with Sprint 150 handoff. |

## Stop Conditions

- A reviewed Windows install-validation claim lacks hosted Windows proof.
- A Windows package claim implies Unix Makefile parity or Windows
  `pkg-config` execution parity.
- CMake package confidence is described as package-manager support,
  shared-library ABI support, dynamic ABI support, or runtime-loader support.
- A workflow rename or promotion weakens the current reviewed Windows CMake
  subset.
- Static-first package metadata checks stop rejecting shared-library or
  unsupported package wording.
- Exact-version and mismatch-version consumer behavior is claimed without an
  installed-package `find_package(Sparse)` proof.
- Required local checks or full C quality gates fail.

## Daily Log

### Day 1: Install Intake

- Re-read the Sprint 149 section of
  `docs/planning/EPIC_13/PROJECT_PLAN.md`.
- Reviewed Sprint 148 Day 12 documentation alignment and Day 14 closeout
  handoff.
- Created the Sprint 149 artifact directory and Day 1 install baseline
  artifact.
- Captured the Linux reviewed package-contract lane:
  `.github/workflows/ci.yml::package-contract`, including
  `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh`.
- Captured the macOS reviewed install/export lanes:
  `.github/workflows/macos-ci.yml::install-and-pkgconfig` and
  `.github/workflows/macos-ci.yml::cmake-install-export`.
- Captured the Windows supplemental lane:
  `.github/workflows/windows-ci.yml::install-and-downstream`, named
  `Windows supplemental CMake install/downstream confidence path`.
- Defined install-validation evidence fields for package files, downstream
  consumers, version behavior, and unsupported claims.
- Day 2 handoff: compare each Windows supplemental assertion against the
  Linux/macOS reviewed proof stack, separating CMake-equivalent evidence from
  Unix-only Makefile and `pkg-config` proof.

### Day 2: Package Audit

- Created the Windows package audit artifact in
  `artifacts/day2-windows-package-audit.md`.
- Compared the Windows supplemental CMake install/downstream lane against the
  Linux reviewed `package-contract` lane and the macOS reviewed
  `install-and-pkgconfig` plus `cmake-install-export` lanes.
- Classified Windows CMake configure/build/install, static `.lib` install,
  header count, CMake package-file presence, CMake metadata shared-artifact
  rejection, installed CMake consumer proof, exact-version consumer proof, and
  mismatch-version rejection as CMake-equivalent evidence.
- Classified Windows `sparse.pc` file-presence and text checks as package
  metadata evidence only, not Windows `pkg-config` execution parity.
- Classified Makefile install/uninstall, `pkg-config --exists`, variable
  resolution, cflags/libs resolution, `pkg-config` downstream compile/link/run,
  and shell-script install behavior as intentionally non-parity on Windows.
- Recorded proof gaps: no Windows execution of the static package deferral
  guard, no explicit source/build path leak check in the Windows CMake package
  metadata, no explicit static imported-target type token check, no uninstall
  cleanup proof, no static `.lib` archive path check inside generated
  `SparseTargets-*.cmake`, and no basic non-example downstream consumer.
- Day 3 handoff: promotion criteria should decide whether those gaps block
  reviewed Windows CMake install-validation promotion, require a narrowed
  reviewed lane, or justify retaining supplemental status.

### Day 3: Criteria Gate

- Created the promotion criteria artifact in
  `artifacts/day3-promotion-criteria.md`.
- Defined four candidate outcomes: promote a reviewed Windows CMake
  install-validation lane, split reviewed metadata/consumer proof from retained
  supplemental extras, retain supplemental-only status, or explicitly reject
  Windows install-validation promotion for this sprint.
- Selected the narrow promotable claim shape for criteria purposes:
  "reviewed Windows CMake install/downstream validation" can be considered
  only if all required hosted MSVC checks pass, while Windows Makefile and
  Windows `pkg-config` execution remain non-goals.
- Marked positive static imported-target metadata, installed include-prefix
  metadata, installed static `.lib` target-location metadata, source/build path
  leak rejection, hosted CMake configure/build/install, static archive/header
  checks, no DLL/shared metadata, CMake downstream run, exact-version run, and
  mismatch-version rejection as must-pass promotion rows.
- Marked static package deferral guard execution as a promotion criterion
  unless Day 4 explicitly keeps it under Linux/macOS reviewed package-contract
  ownership with narrower Windows wording.
- Defined failure semantics for missing package files, unexpected DLL/shared
  metadata, unsupported `sparse.pc` wording, source/build path leaks, failed
  downstream consumers, exact-version failures, and mismatch-version false
  positives.
- Day 4 handoff: apply these criteria to the current lane and choose
  promotion-after-fixes, split reviewed/supplemental, retained supplemental, or
  rejection before editing workflows.

### Day 4: Product Decision

- Created the Windows install-validation product decision artifact in
  `artifacts/day4-product-decision.md`.
- Applied the Day 3 WIV criteria to the current Windows supplemental lane and
  selected conditional promotion after implementation, not immediate
  promotion.
- Chose the narrow product outcome: Sprint 149 will promote a reviewed Windows
  CMake install/downstream validation lane for the maintained static-first
  installed package once the missing CMake metadata checks and hosted evidence
  pass.
- Kept Windows Makefile install/uninstall parity, Windows `pkg-config`
  execution parity, package-manager support, shared-library support, dynamic
  ABI support, runtime-loader behavior, and broad Windows parity as explicit
  non-claims.
- Resolved WIV-19 by retaining
  `scripts/static_package_deferral_check.sh` under Linux/macOS reviewed
  package-contract ownership for this sprint; Windows promotion must still
  preserve static-first text checks and absence of shared artifacts, but it
  will not claim to run the Unix shell deferral guard on Windows.
- Required Days 5-10 to add positive `STATIC IMPORTED`, install include-prefix,
  installed `.lib` target-location, and source/build path leak checks before
  any reviewed Windows install-validation wording lands.
- Day 5 handoff: design workflow edits for the conditional promotion path,
  including job/step names, comments, metadata checks, and hosted evidence
  requirements.

### Day 5: Workflow Design

- Created the workflow implementation design artifact in
  `artifacts/day5-workflow-design.md`.
- Inspected `.github/workflows/windows-ci.yml` ownership, triggers, job names,
  step names, current comments, and PowerShell structure for the
  `install-and-downstream` lane.
- Chose a single-lane promotion design instead of a split reviewed/supplemental
  design: the existing `install-and-downstream` job remains the owner but is
  renamed after checks land to `Windows reviewed CMake install/downstream
  validation path`.
- Planned the step rename from `Run maintained supplemental CMake
  install/downstream proof` to `Run reviewed CMake install/downstream
  validation proof`.
- Designed WIV-09 through WIV-12 insertion immediately after the current
  installed package-file loop, before `sparse.pc` metadata checks and
  downstream consumer checks.
- Defined PowerShell assertion owners for `STATIC IMPORTED`, install-prefix
  include metadata, installed `.lib` imported-location metadata, and
  source/build path leak rejection.
- Recorded hosted-only evidence expectations: final promotion remains pending
  until the PR's Windows install/downstream job passes on `windows-2022`.
- Day 6 handoff: implement workflow comments, job/step names, and the
  criteria-backed checks exactly as designed, without changing source files or
  the Windows CTest lane.

### Day 6: Workflow Update

- Updated `.github/workflows/windows-ci.yml` according to the Day 5 design.
- Renamed the install/downstream job from
  `Windows supplemental CMake install/downstream confidence path` to
  `Windows reviewed CMake install/downstream validation path`.
- Renamed the main install step from `Run maintained supplemental CMake
  install/downstream proof` to `Run reviewed CMake install/downstream
  validation proof`.
- Updated top-level workflow comments and the CTest inspection message to
  describe reviewed Windows install validation as CMake install/downstream
  scoped while preserving no Windows Makefile parity and no `pkg-config`
  execution parity.
- Added WIV-09 through WIV-12 checks: positive `STATIC IMPORTED` target
  metadata, install-prefix include metadata, installed static `.lib` imported
  location metadata, and source/build path leak rejection.
- Added `SparseTargets-release.cmake` to the required installed package-file
  list.
- Preserved existing static `.lib`, no-DLL, header-count, `sparse.pc` metadata,
  maintained example, exact-version consumer, and mismatch-version rejection
  checks.
- Day 7 handoff: design any additional package metadata refinements from this
  workflow implementation, especially if hosted Windows later shows CMake
  target file naming or path formatting differences.

### Day 7: Metadata Design

- Created the package metadata check design artifact in
  `artifacts/day7-metadata-check-design.md`.
- Reviewed the Day 6 Windows workflow metadata assertions against
  `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, `sparse.pc.in`,
  `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh`.
- Confirmed that the Windows Visual Studio install proof should check
  `SparseTargets-release.cmake`, while the Unix local CMake script keeps
  `SparseTargets-noconfig.cmake`.
- Designed Day 8 refinements for `sparse.pc` metadata: require `Name: sparse`,
  exact `Version: $version`, `Cflags: -I${includedir}`, and a static archive
  `Libs:` line without claiming Windows `pkg-config` execution.
- Designed explicit installed version-header presence checking through
  `include/sparse/sparse_version.h` while retaining the fixed 19-header count
  for Sprint 149.
- Kept downstream consumer proof separate from package-file inspection:
  metadata failures should occur before example/exact-version consumer build
  steps, and consumer failures should keep their existing configure/build/run
  labels.
- Day 8 handoff: implement the additional metadata assertions in
  `.github/workflows/windows-ci.yml` without changing `.c` or `.h` files.

### Day 8: Metadata Checks

- Updated `.github/workflows/windows-ci.yml` with the Day 7 package metadata
  refinements.
- Added an explicit installed version-header check for
  `include/sparse/sparse_version.h` so version metadata is not hidden inside
  aggregate header count.
- Strengthened installed `sparse.pc` metadata checks by requiring
  `Name: sparse`, exact `Version: $version`, includedir-based `Cflags`, and
  static archive `Libs` metadata in addition to the existing static archive
  description and unsupported-wording rejection.
- Kept all `sparse.pc` checks as text-only metadata inspection; the workflow
  still does not execute Windows `pkg-config` or claim Windows `pkg-config`
  parity.
- Preserved the Day 6 CMake package checks for positive `STATIC IMPORTED`,
  install-prefix include metadata, installed `.lib` imported location,
  source/build path leak rejection, and absence of shared imported metadata.
- Day 9 handoff: design the downstream consumer proof around the now-stronger
  installed package metadata surface, keeping consumer configure/build/run
  diagnostics separate from package-file checks.

### Day 9: Consumer Design

- Created the downstream consumer proof design artifact in
  `artifacts/day9-consumer-proof-design.md`.
- Reviewed `examples/cmake_example/CMakeLists.txt`,
  `examples/cmake_example/main.c`, the Windows workflow consumer blocks, and
  the Unix `tests/test_cmake_install.sh` comparison path.
- Confirmed the maintained normal consumer proof should remain
  `examples/cmake_example` configured with `CMAKE_PREFIX_PATH="$prefix"`,
  built in Release, executed, and checked for `Sparse library version`,
  `Solution:`, and `OK`.
- Defined exact-version proof as a generated CMake project under
  `$env:RUNNER_TEMP` that uses `find_package(Sparse $version EXACT REQUIRED)`,
  links `Sparse::sparse_lu_ortho`, builds, runs, and checks the same output
  tokens.
- Defined mismatch-version proof as a generated project using a lower
  same-major version that must fail CMake configure; a successful configure is
  a hard failure.
- Designed Day 10's optional enhancement: add a generated non-versioned basic
  CMake consumer separate from the maintained example if Sprint 149 wants a
  direct analog to the Unix generated consumer without invoking `pkg-config`.
- Day 10 handoff: implement the generated basic CMake consumer only if it stays
  low-risk and preserves current maintained example, exact-version, and
  mismatch-version behavior.

### Day 10: Consumer Proof

- Updated `.github/workflows/windows-ci.yml` with a generated non-versioned
  basic CMake consumer that builds against the installed package through
  `CMAKE_PREFIX_PATH="$prefix"`.
- Added `$basicConsumerSrc` under `$env:RUNNER_TEMP` so generated source files
  stay outside the repository.
- Added `$basicConsumerBuild` as a separate build directory from the maintained
  example, exact-version consumer, and mismatch-version configure proof.
- Generated a minimal CMake project using `find_package(Sparse REQUIRED)` and
  `target_link_libraries(basic_consumer PRIVATE Sparse::sparse_lu_ortho)`.
- Generated a minimal C source that includes installed sparse headers, creates
  a 3x3 sparse matrix, inserts one entry, prints `sparse version:`, prints
  `nnz: 1`, frees the matrix, and prints `OK`.
- Added configure, build, run, `$LASTEXITCODE`, multiline output normalization,
  and output-token checks for the basic generated consumer.
- Preserved the maintained example, exact-version generated consumer, and
  mismatch-version fail-closed proof.
- Day 11 handoff: align README, INSTALL, maintainer guidance, and report rows
  with the reviewed Windows CMake install/downstream lane and the added basic
  consumer proof.

### Day 11: Docs Alignment

- Updated `README.md` CI summary so Windows is still CMake-first but now names
  reviewed CMake install/downstream validation for the static-first package
  surface.
- Updated `INSTALL.md` maintained-install and supported-platform wording to
  describe the reviewed Windows CMake install/downstream lane and preserve
  Windows Makefile, `pkg-config` execution, package-manager, shared-library,
  dynamic ABI, runtime-loader, and broad Windows parity non-claims.
- Updated `docs/maintainer_guide.md` package/platform ownership guidance to
  identify Windows reviewed CMake install/downstream validation and list the
  exact metadata, consumer, version, and no-shared-artifact checks it owns.
- Updated `tests/corpus/manifests/report_families.tsv` CI reviewed-lanes row
  so report interpretation includes Windows reviewed CMake install/downstream
  validation while preserving non-claims for Windows Makefile and `pkg-config`
  execution parity.
- Created the documentation alignment artifact in
  `artifacts/day11-docs-alignment.md`.
- Day 12 handoff: run local workflow/report/documentation checks and record
  hosted Windows proof as pending until the PR runs the reviewed Windows
  install/downstream validation job.

### Day 12: Local Validation

- Created the local validation artifact in
  `artifacts/day12-local-validation.md`.
- Ran `git diff --check`; no whitespace errors were reported.
- Ran a targeted trailing-whitespace scan across the Windows workflow, updated
  public docs, report manifest row, and Sprint 149 artifacts; no matches were
  found.
- Parsed `.github/workflows/windows-ci.yml` with Ruby YAML loading; the file
  parsed successfully.
- Ran corpus/report checks:
  `python3 scripts/validate_corpus_schema.py`,
  `python3 scripts/normalize_report_index.py --family ci --check`, and
  `python3 scripts/normalize_report_index.py --family package --check`; all
  passed.
- Ran affected local package checks:
  `bash tests/test_cmake_install.sh` passed with 26 checks,
  `bash tests/test_install.sh` passed with 23 checks, and
  `bash scripts/static_package_deferral_check.sh` passed.
- Ran focused public-doc/workflow searches for stale supplemental/reviewed
  install-validation wording; no stale matches were found.
- Ran focused unsupported-claim searches for Windows Makefile, `pkg-config`,
  package-manager, shared-library, dynamic ABI, runtime-loader, and broad
  Windows parity wording; matches were explicit non-claims only.
- Confirmed no repository `.c` or `.h` files changed through Day 12, so the
  full `make format && make lint && make test` gate is not required by the
  review-comment rule.
- Day 13 handoff: wait for hosted Windows CI evidence for the
  `Windows reviewed CMake install/downstream validation path` job and record
  whether the promoted lane passes in the PR environment.

### Day 13: Integrated Evidence Review

- Created the integrated evidence review artifact in
  `artifacts/day13-integrated-evidence-review.md`.
- Compared the Day 4 conditional-promotion decision against the final workflow
  job name, step name, workflow comments, README, INSTALL guide, maintainer
  guide, and CI report-family row.
- Confirmed the project-facing claim remains narrow: reviewed Windows CMake
  install/downstream validation for the maintained static-first package
  surface.
- Confirmed the public docs and workflow continue to preserve non-claims for
  Windows Makefile parity, Windows `pkg-config` execution parity,
  package-manager support, shared-library support, dynamic ABI support,
  runtime-loader behavior, and broad Windows parity.
- Ran `gh pr view` for the current branch; no pull request exists yet for
  `sprint-149`.
- Ran `gh run list --branch sprint-149 --limit 10`; no hosted workflow runs
  exist yet for the branch.
- Reran focused stale-wording searches for supplemental/reviewed install labels
  and unsupported Windows parity wording; public docs and workflow are clean,
  with remaining matches limited to historical sprint-artifact context or
  recorded search command text.
- Recorded hosted Windows validation as pending until a PR or branch run
  executes the `Windows reviewed CMake install/downstream validation path` job.
- Day 14 handoff: prepare Sprint 149 closeout with the local evidence complete
  and the hosted Windows job explicitly marked as PR-time evidence.

### Day 14: Closeout

- Created the closeout and Sprint 150 handoff artifact in
  `artifacts/day14-closeout-handoff.md`.
- Reviewed the full Sprint 149 artifact package and confirmed Days 1-14 have
  artifacts under `docs/planning/EPIC_13/SPRINT_149/artifacts/`.
- Finalized the Sprint 149 support boundary: reviewed Windows CMake
  install/downstream validation for the maintained static-first package
  surface, subject to hosted MSVC proof in PR CI.
- Preserved explicit non-claims for Windows Makefile install/uninstall parity,
  Windows `pkg-config` execution and downstream parity, package-manager
  resolver behavior, shared-library packaging, dynamic ABI compatibility,
  runtime-loader behavior, and broad Windows parity.
- Reviewed the Sprint 150 project-plan scope:
  `QR Maintained Corpus Family Expansion`, including QR family selection,
  fixture metadata, oracle semantics, proof-owner tests, report integration,
  documentation alignment, and validation.
- Recorded Sprint 150 handoff guidance so QR corpus work remains independent
  from unresolved Windows package-lane residuals.
- Final local closeout checks:
  `git diff --check` passed, workflow YAML parsing passed, targeted
  trailing-whitespace scan passed, and stale-reference searches were limited to
  historical artifact context or recorded search text.
- Confirmed no repository `.c` or `.h` files changed during Sprint 149, so the
  full `make format && make lint && make test` gate is not required by the
  review-comment rule.
- Sprint 149 closeout status: local implementation and local validation are
  complete; hosted Windows evidence remains PR-time pending until the branch is
  pushed and CI runs.
