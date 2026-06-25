# Sprint 89 Day 8: External Comparison Sweep Design

## Purpose

Freeze the exact bounded external comparison protocol and reporting shape so
Sprint 89 can execute one explicit final evidence package before deciding
whether any last-mile implementation batch is still necessary.

## Main Result

Sprint 89 now has one exact Day 9 comparison-execution contract:

- required execution owners:
  - `tests/test_chol_csc.c`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `Makefile` through `make bench-reorder-sprint86`
- directly forced support-only comparison surfaces only if the execution
  exposes a real contradiction:
  - `tests/chol_external_dense_reference.py`
  - `benchmarks/bench_reorder.c`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`

## Fixed Comparison Protocol

The exact bounded comparison protocol is now fixed:

- maintained correctness lane:
  - execute `./build/quality-review-cmake/test_chol_csc`
  - treat the retained external-dense-reference checks as the maintained
    correctness owner on the bounded SPD lane
  - capture the explicit fixture-level outputs for:
    - `nos4`
    - `bcsstk04`
  - interpret agreement through:
    - `max|x-x_ref|`
    - retained in-repo residual strength
- maintained package-shape lane:
  - execute:
    - `bash tests/test_install.sh`
    - `bash tests/test_cmake_install.sh`
  - capture:
    - pass totals
    - fail totals
    - skip totals where applicable
  - interpret the lane through the maintained static-first consumer and export
    contract rather than through broad shared-library or cross-platform claims
- bounded runtime-reference support lane:
  - execute `make bench-reorder-sprint86`
  - capture the bounded Sprint 86 slice outputs for:
    - `bcsstk14`
    - `Pres_Poisson`
  - preserve the emitted comparison fields:
    - reorder name
    - `nnz_L`
    - `reorder_ms`
  - interpret this lane as:
    - branch-local touched-runtime evidence
    - not a portable timing proof
    - not a broad product-superiority claim

## Accepted Reporting Shape

The accepted Day 9 reporting shape is now fixed:

- correctness agreement:
  - one explicit statement for `nos4`
  - one explicit statement for `bcsstk04`
  - each statement must include the external agreement metric and the retained
    residual reading
- package/consumer shape alignment:
  - exact totals from `tests/test_install.sh`
  - exact totals from `tests/test_cmake_install.sh`
  - one explicit interpretation of whether the maintained installed-package
    contract still matches the shipped surface
- bounded runtime observations:
  - one explicit AMD-vs-ND comparison on `bcsstk14`
  - one explicit AMD-vs-ND comparison on `Pres_Poisson`
  - one bounded interpretation of whether the retained Sprint 86 runtime
    narrative still reads as truthful

## Good-Enough-to-Close Criteria

The strongest "good enough to close" comparison criteria are now explicit:

- the maintained SPD external lane shows no correctness mismatch on `nos4` or
  `bcsstk04`
- both install/export proof scripts pass without exposing a package-shape or
  consumer-contract contradiction
- the Sprint 86 reorder slice remains interpretable as bounded mixed runtime
  evidence and does not expose one new touched contradiction large enough to
  justify a final implementation batch on its own
- any remaining difference must be classifiable as:
  - bounded and acceptable
  - or an explicit residual item for the next planning cycle

## Forced Spillover Rule

The strongest forced-spillover rule is now fixed:

- Day 10 and Day 11 should only move into a real final fix batch if the Day 9
  comparison execution exposes:
  - an SPD correctness disagreement
  - a local install/export contradiction
  - a touched reorder/ND contradiction still large enough to justify one last
    bounded fix
  - or an unavoidable support-surface wording contradiction created by the
    evidence
- the comparison package should not force movement just because:
  - external ecosystems are broader elsewhere
  - a fixture shows mixed rather than uniformly dominant runtime behavior
  - the repo remains intentionally bounded on capability, platform, or product
    shape

## Preserved Fence

The preserved Day 8 fence is explicit:

- no new mandatory external dependency stack
- no promotion of advisory METIS-class or wider ecosystem comparison into the
  maintained contract
- no broad timing pass/fail policy
- no canonical reporting rewrite before the bounded comparison lane runs
- no blind final-fix batch before the retained evidence package is executed

## Exit State

- Sprint 89 now has one exact bounded external comparison protocol and
  reporting shape.
- Day 9 can execute the retained correctness, package-shape, and bounded
  runtime lanes without ad hoc framing.
- Any final implementation batch remains gated by explicit comparison outcomes
  rather than by generic endgame pressure.
