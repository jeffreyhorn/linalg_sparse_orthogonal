# Sprint 78 Day 2 - Validation Baseline and Hotspot Truth Recheck

Date: 2026-06-17  
Branch: sprint-78

## Purpose
Reconfirm the implementation-day validation contract and the highest-signal rerun surfaces Sprint 78 must preserve before any large-source decomposition or giant-test architecture work lands.

## Main Result
Sprint 78's code-day validation and truth-surface contract is now explicit before any maintainability batch lands.

The strongest local reviewed baseline is still:
- `make quality-review-full`

Reviewed CMake parity remains the main truthfulness anchor:
- `ctest -N --test-dir build/quality-review-cmake` = `53`

The Sprint 78 authority split is now fixed explicitly:
- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial maintainability or proof-architecture batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

## Live Proof-Surface Split
The reviewed CMake tree currently owns the key Sprint 78 proof surfaces most likely to be stressed:
- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt_csc`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_etree`
- `./build/quality-review-cmake/test_graph`
- `./build/quality-review-cmake/test_iterative`
- `./build/quality-review-cmake/test_ldlt`
- `./build/quality-review-cmake/test_svd`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_reorder_nd`

Representative example surfaces remain:
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`

The strongest current policy-owned proof split remains:
- `tests/test_reorder_nd.c` = shared ND compatibility/default-policy owner
- `tests/test_chol_csc.c` = family-local large-`n` Cholesky CSC handoff and publish-back owner
- `tests/test_integration.c` = public one-shot vs repeated-run Cholesky parity and matrix-shell reset owner
- `tests/test_fuzz.c` = bounded seeded generative follow-through for the large-`n` CSC lifecycle parity lane

## Preserved Validation Reading
Sprint 78 keeps the same reviewed/truthfulness reading fixed earlier in Epic 7:
- reviewed CMake parity remains the stable anchor
- code-day batches require the normal format/lint/test gate
- larger maintainability or proof-architecture batches should default to the stronger full reviewed path
- giant-test work must not invent new proof owners by accident

## Highest-Signal Sprint 78 Rerun Set
The likely rerun set for later Sprint 78 code days is now fixed to:
- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt_csc`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_etree`
- `./build/quality-review-cmake/test_graph`
- `./build/quality-review-cmake/test_iterative`
- `./build/quality-review-cmake/test_ldlt`
- `./build/quality-review-cmake/test_svd`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_reorder_nd`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`

## Exit State
- No rerun ambiguity remains around the likely touched maintainability seams.
- The proof-owner and example-support split is explicit before source or giant-test work begins.
- Sprint 78 can now move into a ranked source-hotspot audit from a fixed validation baseline.
