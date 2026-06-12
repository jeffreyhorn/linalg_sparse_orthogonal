# Sprint 66 Day 7: Exact Landing Fence and Regression Plan

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Fix the exact touched-file fence, proof plan, and validation order before any
Sprint 66 packaging or platform changes land.

## Required First-Batch Surface

The exact required first-batch surface is:

- `CMakeLists.txt`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`

This keeps the first implementation batch centered on the highest-value
build/install/docs contract instead of widening immediately into workflows or
regression scripts.

## Optional Support Surface

Optional support surfaces only if the proof burden forces them:

- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `.github/workflows/ci.yml`
- `Makefile`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

These are not part of the first batch by default. They only become valid Sprint
66 touches if the landed packaging contract materially changes the install or
platform truth surface.

## Explicit Non-Touch Set

The explicit non-touch set for Sprint 66 is:

- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- broad dead-code artifact topology
- Windows-specific Makefile wrapper support
- broad shared-library enablement
- broad ABI guarantee widening
- macOS dead-code enforcement
- Windows dead-code enforcement
- new platform-specific benchmark or solver validation lanes

## Proof Plan

The proof plan for the remaining sprint is:

- substantial packaging/platform work:
  - `make quality-review-full`
- focused install/package regression checks when install/export behavior or
  contract wording moves materially:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- workflow truth checks when workflow comments/job labels or platform-claim
  wording moves:
  - direct review of `.github/workflows/ci.yml`
  - direct review of `.github/workflows/macos-ci.yml`
  - direct review of `.github/workflows/windows-ci.yml`

This keeps proof attached to existing maintained surfaces instead of inventing a
new platform harness.

## Day 8-12 Sequence

The remaining implementation order is:

1. Day 8
   - first packaging/productization batch on the required build/install/docs
     surfaces
2. Day 9
   - post-landing audit and rerank of remaining contradictions
3. Day 10
   - second bounded batch only if the Day 8 landing leaves one real
     contract-level contradiction unresolved
4. Day 11
   - workflow/CI/contract reconciliation plus focused install/package
     regression support only where the landed contract requires it
5. Day 12
   - docs and maintainer-story follow-through on the converged contract

## Validation Order

For later `*.c` / `*.h` changes, the required minimum remains:

- `make format`
- `make lint`
- `make test`

For substantial packaging/platform/build/workflow changes, the maintained
default remains:

- `make quality-review-full`

For docs-only landing or reconciliation days:

- targeted sanity checks only

## Exit State

Sprint 66 Day 7 closes with:

- one exact touched-file fence
- one concrete proof and validation plan
- one explicit Day 8-12 landing order
- one bounded implementation map for the rest of the sprint
