# Sprint 66 Day 11: CI and Command-Surface Reconciliation

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Align the remaining maintained command and CI truth surfaces with the landed
Sprint 66 packaging/platform contract, remove stale sprint-era commentary that
now obscures the reviewed lane model, and fix the exact Day 12-14 queue.

## Landed Reconciliation

The Day 11 batch landed on:

- `README.md`
- `INSTALL.md`

### README

The top-level CI summary was still too generic for the current Sprint 66
contract.

It now states directly that:

- Linux is the strongest reviewed source of truth
- macOS enforces the Apple Clang reviewed path and carries supplemental GCC and
  static-first Make install/`pkg-config` verification
- Windows enforces the reviewed CMake subset and backs the CMake-first consumer
  story

### INSTALL

The supported-platform table was still leaning on sprint-history notes rather
than the live lane model.

It now describes the current platform truth directly:

- Linux gcc lane owns the strongest reviewed source-of-truth surface
- Linux clang lane is the supplemental TSan/OpenMP lane
- macOS Apple Clang is the reviewed macOS lane and the workflow also carries
  supplemental static-first install/`pkg-config` verification
- macOS Homebrew GCC is supplemental second-compiler direct coverage
- Windows is the reviewed CMake subset only, supporting the CMake-first
  consumer story rather than a separate reviewed install-validation lane

## What Did Not Need to Move

The Day 11 reread did not uncover a remaining contradiction that required:

- `Makefile` command-surface changes
- workflow job behavior changes
- install/export implementation changes

This remained a wording and operator-map reconciliation pass, not another
implementation batch.

## Reranked Remaining Queue

The strongest remaining Day 12 proof gap is now explicit:

- `tests/test_cmake_install.sh` already checks installed `pkg-config` version
  against the repo `VERSION` file
- `tests/test_install.sh` still only proves that `pkg-config --modversion
  sparse` is non-empty, not that it matches the same source of truth

That makes the next highest-value target:

- focused install/package regression tightening on the Unix-side Make install
  proof path

Likely touched Day 12 surfaces:

- `tests/test_install.sh`

Support only if the proof burden requires it:

- `tests/test_cmake_install.sh`
- `INSTALL.md`
- `README.md`

## Validation

This was a docs-only reconciliation day, so no reviewed baseline rerun was
required.

Targeted sanity checks used:

- reread of the landed Day 10 contract state across docs and workflows
- direct reread of the maintained command surface in `Makefile`
- targeted `rg` checks across the command/docs/workflow surfaces for install,
  package, and platform-contract terminology
- branch-diff review against `master...HEAD`

## Exit State

Sprint 66 Day 11 closes with:

- one reconciled command-facing CI/platform summary story
- one explicit Day 12 proof gap on the Unix-side Make install regression path
- one fixed Day 12-14 close sequence
