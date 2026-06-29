# Sprint 97 Day 12: Cross-Platform Product Follow-Through

## Purpose

Day 12 applies the Day 11 platform calibration to active product, workflow,
and support surfaces. The goal is to remove stale platform wording while
preserving the current proof boundaries.

## Reviewed Surfaces

Surfaces checked:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- Sprint 97 artifacts and working notes

## Update Batch

Day 12 updates `.github/workflows/windows-ci.yml`.

The Windows workflow already had the correct proof boundary:

- reviewed CMake configure/build
- reviewed `ctest -N`
- reviewed full `ctest`
- expected CTest count `51`
- staged exclusions for `test_threads`, `test_sprint4_integration`, and
  `test_fuzz`
- no reviewed Makefile parity
- no separate reviewed install-validation lane

The only stale product-surface wording was the historical `Sprint 68` label on
the active fuzz/property exclusion. Day 12 removes that sprint-history label
from the workflow comment and runtime log message while preserving the same
exclusion.

## Residual Queue

Platform residuals that should remain visible for Sprint 98 or later:

- Windows Makefile reviewed wrappers
- Windows dead-code flow
- Windows `test_threads`
- Windows `test_sprint4_integration`
- Windows `test_fuzz`
- Windows separate install-validation lane
- macOS dead-code lane
- macOS full reviewed install/export parity
- shared-library package lane on every platform
- possible future CMake install validation on Windows, only if explicitly
  designed and proven

## Verification

Focused Day 12 checks:

```sh
python3 scripts/check_library_sources.py
git diff --check
rg -n "[ \t]+$" .github/workflows/windows-ci.yml docs/planning/EPIC_9/SPRINT_97
rg -n "Sprint 68|install-consumer lane" .github/workflows/windows-ci.yml docs/maintainer_guide.md README.md INSTALL.md
```

Observed results:

- source-list checker passed: `source-list-check: PASS (42 library sources)`
- `git diff --check` passed
- trailing-whitespace scan passed with no matches
- active platform surfaces no longer carry the stale Sprint 68 workflow label
- no active maintainer/front-door surface uses `install-consumer lane`

No `.c` or `.h` files are modified by this workflow/docs calibration, so the
full `make format && make lint && make test` chain is not required.

## Day 12 Result

The active platform product surfaces now tell the same story: Linux remains the
strongest reviewed source of truth, macOS remains narrower with supplemental
static-first install confidence, and Windows remains the reviewed CMake-first
consumer subset with explicit staged exclusions and no historical sprint label
in the active workflow.
