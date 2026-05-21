# Sprint 37 Day 13 Full Validation Sweep

**Date:** 2026-05-21  
**Branch:** `sprint-37`

## Objective

Run the full maintained validation surface after the Day 12 focused
reconciliation pass and capture the authoritative pre-close Sprint 37 state.

This Day 13 sweep covers:

- direct maintained gates
- reviewed local wrapper paths
- reviewed CMake parity paths

## Validation Results

### Direct maintained gates

- `make format`
  - passed
  - `real 2.74`
- `make lint`
  - passed
  - `real 235.65`
- `make test`
  - passed
  - `real 111.30`

Interpretation:

- formatting stayed clean
- compile/static-analysis quality stayed clean
- the default runtime suite stayed clean

## Reviewed local wrapper paths

- `make quality-review-compile`
  - passed
  - `real 256.69`
- `make quality-review`
  - passed
  - `real 313.09`

What this reconfirms:

- `quality-review-compile` still means:
  - `format-check`
  - `lint`
- `quality-review` still means:
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`

The serial dead-code contract remained sound inside the reviewed wrapper.

## Reviewed CMake parity paths

- `make quality-review-cmake-compile`
  - passed
  - `real 47.31`
- `make quality-review-cmake`
  - passed
  - `real 210.24`

Parity details:

- `ctest -N`: `53`
- Makefile/CMake test-count parity: `53` vs `53`
- full `ctest`: `53 / 53` passed
- `Total Test time (real) = 156.66 sec`

Interpretation:

- the Sprint 36 reviewed CMake parity baseline is still exact
- Sprint 37 maintainability work did not introduce hidden Makefile/CMake
  divergence

## Day 13 End State

The full maintained validation surface is green.

No new reconciliation queue was created:

- no fallout from the Day 5 test-helper consolidation
- no fallout from the Day 6 benchmark-helper consolidation
- no operator-path drift from the Day 7 target-normalization work
- no dead-code report/check regression from the Day 9 refactor
- no workflow-contract drift from the Day 11 wording cleanup

The remaining known operational constraints are unchanged:

- dead-code remains authoritative only in serial mode because of shared-path
  execution
- tree-mutating instrumentation modes still require `make clean` before
  returning to the normal direct/reviewed path

## Conclusion

Sprint 37 reaches Day 14 from a validated baseline:

- direct maintained gates passed
- reviewed local wrapper paths passed
- reviewed CMake parity paths passed
- test-count parity remained exact

The sprint is ready for closeout without an additional fix cycle.
