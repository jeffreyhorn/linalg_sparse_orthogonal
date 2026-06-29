# Sprint 97 Day 14: Sprint Closeout And Handoff

## Purpose

Day 14 closes Sprint 97 from the validated Day 13 state. The sprint reduced a
high-cost build-topology drift risk, clarified the static-first package
contract, strengthened installed consumer proof, calibrated workflow/platform
claims, and leaves a ranked Sprint 98 handoff queue.

## What Changed

### Duplication Reduced

Sprint 97 selected the library source list as the first build-topology
convergence target.

Implemented surfaces:

- `build-metadata/library_sources.txt`
- `scripts/check_library_sources.py`
- `make source-list-check`
- `make quality-review-compile` now runs:
  1. `format-check`
  2. `source-list-check`
  3. `lint`

Result:

- Make `LIB_SRCS`, CMake `add_library(sparse_lu_ortho STATIC ...)`, and the
  manifest remain readable and reviewable.
- Silent library-source drift is now mechanically detected.
- The checker passed with `42` library sources.

### Package-Surface Decision

Sprint 97 preserves the static-first package contract.

Maintained package surface:

- static archive install
- public headers
- generated version header
- `pkg-config` metadata
- CMake package export through `Sparse::sparse_lu_ortho`
- exact-version package metadata
- local Make and CMake install/export proof

Deferred package surface:

- shared-library build output
- shared-library install/export metadata
- dynamic ABI guarantee
- Windows DLL/import-library package story
- reviewed shared-library consumer tests
- package-manager integration

### Consumer Proof Follow-Through

The install/export proof scripts now assert the static-first package shape in
both directions:

- the static archive is installed
- no `.so`, `.so.*`, `.dylib`, or `.dll` artifact appears in the temporary
  install prefix

Observed Day 13 proof:

- `tests/test_install.sh`: 14 passed, 0 failed
- `tests/test_cmake_install.sh`: 16 passed, 0 failed, 0 skipped

### Workflow Coherence

Local and public command descriptions now agree:

- `README.md`
- `INSTALL.md`
- `Makefile`

All describe `quality-review-compile` as:

```text
format-check + source-list-check + lint
```

CI workflow YAML did not need broad restructuring. Linux and macOS inherit the
source-list guard through the existing `make quality-review-compile` command.

### Cross-Platform Calibration

Platform story after Sprint 97:

- Linux remains the strongest reviewed source of truth.
- macOS remains narrower than Linux and carries supplemental static-first
  Make install/`pkg-config` confidence.
- Windows remains the reviewed CMake-first consumer subset.
- Windows expected CTest count remains `51`.
- Windows staged exclusions remain visible:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`

The active Windows workflow no longer carries stale sprint-history wording for
the fuzz/property exclusion.

## Validation Evidence

Day 13 validation passed:

```sh
python3 scripts/check_library_sources.py
make quality-review-compile
make quality-review-cmake-compile
bash tests/test_install.sh
bash tests/test_cmake_install.sh
git diff --check
rg -n "[ \t]+$" .github/workflows/windows-ci.yml README.md INSTALL.md docs/maintainer_guide.md tests/test_install.sh tests/test_cmake_install.sh docs/planning/EPIC_9/SPRINT_97 build-metadata scripts/check_library_sources.py Makefile CMakeLists.txt
```

Observed highlights:

- source-list checker: `source-list-check: PASS (42 library sources)`
- reviewed Make compile-quality path: passed
- reviewed CMake compile/parity path: passed
- CMake tests: 54
- Makefile tests: 54
- Make/CMake test-count parity: passed
- Make install/`pkg-config` proof: 14 passed, 0 failed
- CMake install/export proof: 16 passed, 0 failed, 0 skipped
- whitespace hygiene: passed

No `.c` or `.h` files were modified during Sprint 97. The full
`make format && make lint && make test` chain was therefore not required by
the sprint validation rule, but the reviewed compile-quality wrapper and CMake
compile/parity wrapper were run because build, script, and package proof
surfaces changed.

## Artifact Index

- `PLAN.md` - 14-day Sprint 97 plan
- `WORKING_NOTES.md` - day-by-day execution notes
- `artifacts/day1-authoritative-inputs.txt` - authoritative starting inputs
- `artifacts/day1-build-topology-baseline.md` - build/workflow baseline
- `artifacts/day2-build-topology-duplication-audit.md` - ranked duplication map
- `artifacts/day3-convergence-architecture-design.md` - convergence design
- `artifacts/day4-source-list-boundary-freeze.md` - source-list landing plan
- `artifacts/day5-source-list-reduction-batch1.md` - implementation notes
- `artifacts/day6-source-list-reduction-closeout.md` - reduction closeout
- `artifacts/day7-package-surface-decision-audit.md` - package decision audit
- `artifacts/day8-package-surface-decision.md` - static-first decision
- `artifacts/day9-consumer-proof-follow-through.md` - consumer proof update
- `artifacts/day10-workflow-coherence-follow-through.md` - workflow coherence
- `artifacts/day11-cross-platform-calibration.md` - platform calibration
- `artifacts/day12-cross-platform-product-follow-through.md` - platform follow-through
- `artifacts/day13-validation-and-residual-queue.md` - validation and residual queue
- `artifacts/day14-sprint-closeout-and-handoff.md` - closeout and handoff

## Sprint 98 Handoff Queue

Ranked candidates:

1. Test registration convergence or a stronger parity guard that preserves
   platform exclusions, local Make/CMake parity, and Windows expected-count
   assertions.
2. Benchmark registration reduction if a manifest/checker pattern remains
   simple and keeps benchmark subsets visible.
3. Example registration reduction if it lowers review cost without obscuring
   CMake target names.
4. Optional checker self-test fixture for `scripts/check_library_sources.py` if
   parser complexity grows.
5. Possible Windows CMake install-validation lane only after explicit design,
   proof ownership, and CI cost review.

Package/product non-claims to preserve:

- no shared-library package contract
- no dynamic ABI guarantee
- no package-manager claim
- no reviewed Windows install-validation lane
- no Windows Makefile parity

Platform residuals to preserve as explicit:

- Windows `test_threads`
- Windows `test_sprint4_integration`
- Windows `test_fuzz`
- Windows dead-code flow
- macOS dead-code lane
- macOS full reviewed install/export parity

## PR-Ready State

Sprint 97 is ready for retrospective and PR preparation:

- artifacts cover every sprint day
- closeout links the implementation to validation evidence
- Sprint 98 residuals are ranked and evidence-backed
- package and platform non-claims are explicit
- no hidden build/package/platform contradiction is known in the touched
  surfaces

## Day 14 Result

Sprint 97 closes from evidence rather than aspiration. The branch now contains
a validated build/package/product convergence package and an explicit handoff
queue for the next sprint.
