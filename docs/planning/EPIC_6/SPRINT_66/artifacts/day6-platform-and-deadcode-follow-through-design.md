# Sprint 66 Day 6: Platform and Dead-Code Follow-Through Design

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Convert the remaining platform and dead-code residual queue into one bounded
implementation plan that stays inside the reviewed truth fence and names the
later proof surfaces precisely.

## What Moves in Sprint 66

The bounded platform/dead-code set that may move in Sprint 66 is:

- wording alignment across:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - relevant workflow comments and job labels
- install/package regression ownership where the packaging batch changes the
  shipped contract
- narrow Makefile or workflow follow-through only if the packaging landing
  materially changes the reviewed command story

This keeps the platform lane centered on truthful contract convergence rather
than on broad platform expansion.

## What Stays Deferred

The deferred set remains explicit:

- Windows Makefile reviewed-wrapper parity
- Windows dead-code enforcement
- macOS dead-code enforcement
- broad dead-code topology redesign
- broad wrapper redesign beyond the audited seams
- fake cross-platform closure beyond reviewed evidence

Sprint 66 should keep these items visible without silently absorbing them into
the first implementation batch.

## Follow-Through Proof Contract

Each bounded follow-through batch should prove one of the following:

- reviewed workflow truthfulness
  - comments, job names, and docs still match what Linux, macOS, and Windows
    actually enforce
- install/package regression truth
  - Make/pkg-config install path still works through `tests/test_install.sh`
  - CMake install/export/find-package path still works through
    `tests/test_cmake_install.sh`
- bounded operational cleanup
  - only where the packaging batch changes the touched command or workflow
    story directly

This keeps proof attached to concrete regression surfaces rather than creating
a new generic platform harness.

## Exact Implementation Fence

Required platform/dead-code follow-through surfaces:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`

Likely support only if the landing proves they must move:

- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `.github/workflows/ci.yml`
- `Makefile`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

Explicit non-touch set for this lane:

- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- broad dead-code artifact topology
- Windows-specific Makefile wrapper support
- new platform-specific benchmark or solver validation lanes

## Regression-Coverage Shortlist

The focused later regression shortlist is now explicit:

- `make quality-review-full`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- platform-truth sanity checks on:
  - `.github/workflows/windows-ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/ci.yml`

These are the concrete proof homes that later Sprint 66 implementation should
use if the packaging or workflow contract changes.

## Exit State

Sprint 66 Day 6 closes with:

- one bounded platform/dead-code implementation plan
- one explicit deferred residual list
- one concrete install/package regression shortlist
- one clear Day 7 starting point for the exact touched-file fence
