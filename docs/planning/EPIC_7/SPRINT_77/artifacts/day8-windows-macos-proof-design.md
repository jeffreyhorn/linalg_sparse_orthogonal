# Sprint 77 Day 8 Artifact: Windows/macOS Proof Design

Date: 2026-06-17
Branch: sprint-77

## Purpose

Define the bounded platform-confidence follow-through batch now justified by
the Day 7 rerank, with one exact macOS/Windows proof seam and one explicit
fence against widening the reviewed platform claim.

## Main Result

Sprint 77 now has one explicit second implementation fence:

- required Day 9 center:
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- strongest support only if the batch truly forces it:
  - `docs/maintainer_guide.md`
- lower-value support only:
  - `README.md`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

## Why This Is the Right Day 9 Center

The best Day 9 move is not to invent a new reviewed install-validation lane.

It is to make the existing workflow-level proof split read more explicitly and
more symmetrically:

- macOS already has:
  - reviewed Apple Clang path
  - supplemental Make install/`pkg-config` verification
- Windows already has:
  - reviewed CMake subset
  - CMake-first consumer story
  - explicit staged exclusions

That means the strongest bounded payoff is workflow-level proof clarification,
not broad new platform coverage.

## Preserved Truthfulness Checklist

The Day 9 batch must preserve:

- no broader reviewed-platform claim than current evidence supports
- no fake Windows Makefile parity claim
- no disguised shared-library maturity claim
- no promotion of local Unix-side install scripts into reviewed macOS or
  Windows parity
- Linux remaining the strongest reviewed source of truth

## Day 9 Intended Shape

The intended Day 9 shape is:

- clarify the macOS supplemental install-path status directly in the workflow
  surface
- clarify the Windows reviewed CMake-only consumer status directly in the
  workflow surface
- tighten asymmetry reading without claiming new review scope
- move maintainer-policy follow-through only if the workflow wording would
  otherwise drift from the authoritative package/platform reading

## Explicit Non-Goals

Day 9 explicitly should not:

- add a new reviewed Windows install-validation job
- add a reviewed macOS install/export parity claim
- widen the Windows reviewed subset into Makefile parity
- widen the package story into ABI or shared-library claims
- reopen the Day 6 install-guide lane unless absolutely forced

## Exit State

Sprint 77 now has one explicit proof-design center:

- workflow-level macOS/Windows proof clarification first
- maintainer-policy follow-through only if truly forced
- support surfaces kept narrow in advance
