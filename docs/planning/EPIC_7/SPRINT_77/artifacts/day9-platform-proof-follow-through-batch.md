# Sprint 77 Day 9 Artifact: Platform Proof Follow-Through Batch

Date: 2026-06-17
Branch: sprint-77

## Purpose

Land the bounded macOS/Windows workflow-level proof clarification batch so the
existing platform asymmetry reads more explicitly without widening the reviewed
platform claim.

## Main Result

The Day 9 result stayed inside the Day 8 fence:

- `macos-ci.yml` now states more explicitly that the install/`pkg-config` job
  is confidence-building supplemental package verification, not reviewed macOS
  install/export parity
- the macOS supplemental job and step names now read more directly as
  package-path and consumer-confidence proof
- `windows-ci.yml` now states more explicitly that the reviewed lane is
  CMake-first consumer proof only
- the Windows job and `ctest -N` inspection step now read more directly as
  reviewed consumer-scope proof, and the log output now restates the non-claim
  about Makefile parity and separate reviewed install validation

## What Landed

The landed cleanup makes the workflow-layer proof split read more directly:

- macOS:
  - supplemental install-path verification remains real
  - but it now reads more explicitly as confidence-building support rather than
    reviewed parity
- Windows:
  - reviewed CMake subset remains real
  - but it now reads more explicitly as consumer proof only, not broader
    reviewed install validation or Makefile parity

## Preserved Fence

The preserved truthfulness fence stayed intact:

- no broader reviewed-platform claim was introduced
- no fake Windows Makefile parity claim was introduced
- no new reviewed Windows or macOS install-validation lane was implied
- no shared-library, ABI, or product-claim widening was introduced

## Support Surfaces Not Needed

No support-only follow-through was actually needed:

- `docs/maintainer_guide.md`
- `README.md`
- `CMakeLists.txt`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

## Validation

This was a docs/workflow-only landing, so the Day 9 check stayed on the
targeted sanity path:

- diff review
- terminology/alignment reread
- branch-state verification

## Exit State

Sprint 77 now has one landed bounded platform-confidence batch:

- clearer macOS supplemental package-proof reading
- clearer Windows reviewed consumer-proof reading
- no widened reviewed-platform claim beyond maintained evidence
