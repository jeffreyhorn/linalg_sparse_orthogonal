# Sprint 66 Day 9: Post-Landing Audit and Rerank

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Reassess the Sprint 66 queue after the Day 8 packaging/productization landing
and determine whether any real packaging contradiction still justifies a second
core implementation batch, or whether the remaining work has narrowed to
workflow/install-contract reconciliation.

## Audit Result

Day 8 closed the strongest packaging contradiction.

The maintained package surface now reads coherently across the live build and
docs surfaces:

- `CMakeLists.txt` states that the maintained package surface remains
  static-first even when `BUILD_SHARED_LIBS=ON` is requested
- `INSTALL.md` treats `make install`, `cmake --install`, `pkg-config`, and
  `find_package(Sparse)` as one intentional static archive distribution story
- `README.md` states the same compact top-level package contract directly
- `docs/maintainer_guide.md` owns the narrow ABI and platform interpretation
- `tests/test_cmake_install.sh` now verifies installed package version against
  the repo `VERSION` file

That means Sprint 66 no longer has a first-order contradiction around whether
the repo has a real install/export surface or what release shape that surface
implies.

## What Did Not Reopen

The post-landing reread did not uncover a second unresolved build/install
contradiction of comparable weight:

- the release shape still intentionally stays static-first
- the version metadata chain still stays coherent and single-sourced from
  `VERSION`
- downstream `pkg-config` and `find_package(Sparse)` consumption still point at
  the same maintained archive surface
- the repo still does not imply a broader shared-library or dynamic-ABI
  guarantee

So Day 10 should not invent a second packaging batch just to make the sprint
look symmetrical. A broader shared-library or wider ABI move remains a separate
product decision, not normal Sprint 66 cleanup.

## Reranked Remaining Queue

The strongest remaining residual is now above the package-shape machinery:

- workflow comments and job labels still need to read as one coherent reviewed
  platform contract
- install/package regression ownership should stay explicit:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- top-level and maintainer wording should stay aligned with the platform fence:
  - Linux strongest reviewed source of truth
  - macOS reviewed quality plus supplemental install/`pkg-config`
  - Windows reviewed CMake subset only

That makes the next highest-value Sprint 66 target:

- workflow/CI/install-contract reconciliation around the shipped packaging and
  platform truth story

Likely touched surfaces:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

Support only if the landing proves it is required:

- `Makefile`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

## Validation

This was a docs-only audit/rerank day, so no reviewed baseline rerun was
required.

Targeted sanity checks used:

- reread of the Day 8 artifact and working-notes landing record
- targeted `rg` checks across the live packaging, install, maintainer,
  workflow, and focused regression surfaces
- direct reread of the current top-level packaging summary in `README.md`
- direct reread of the focused version-source-of-truth path in
  `tests/test_cmake_install.sh`
- `git diff --stat master...HEAD` to reconfirm the current bounded branch shape

## Exit State

Sprint 66 Day 9 closes with:

- one confirmed closed Day 8 packaging contradiction
- one reranked remaining queue centered on workflow/install-contract
  reconciliation
- one explicit Day 10 target set with a bounded support surface
