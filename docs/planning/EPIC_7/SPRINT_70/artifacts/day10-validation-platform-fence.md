# Sprint 70 Day 10: Validation and Platform Contract II

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Convert the Day 9 preservation map into one exact Epic 7 truthfulness fence so
later implementation sprints have an explicit non-goal and preservation
checklist before any product, capability, performance, or cleanup work lands.

## Maintained Reviewed Truth vs Supplemental Proof vs Deferred Ambition

The Day 10 split is now explicit.

Maintained reviewed truth:

- `make quality-review-full`
- Linux reviewed Makefile compile-quality path
- Linux reviewed CMake parity path
- Linux enforced dead-code report/check path
- macOS reviewed Apple Clang compile/CMake/wall-check/sanitize lane
- Windows reviewed CMake subset

Supplemental-but-truthful proof:

- Linux direct runtime, TSan, coverage, and `bench-fast`
- macOS Homebrew GCC direct build/test/wall-check
- macOS static-first Make install/`pkg-config`
- local Unix-side install/package regression scripts
- threshold-free canonical benchmark report artifacts

Explicitly deferred platform/release ambitions:

- shared-library maturity claims
- Windows reviewed Makefile parity
- Windows reviewed dead-code
- macOS reviewed dead-code
- broad reviewed install-validation parity across every platform

## Explicit Non-Goals

Sprint 70 Day 10 fixes the following contract non-goals for later Epic 7 work:

- no fake shared-library or dynamic-ABI maturity claims
- no inflated Windows parity claims beyond the reviewed CMake subset
- no install/package story broader than:
  - local Unix-side proof scripts
  - supplemental macOS Make install/`pkg-config`
  - reviewed Windows CMake-consumer path
- no reinterpretation of `make bench-canonical-report` as a pass/fail or
  portable timing gate
- no benchmark or example wording that steals test-owned regression, oracle,
  or property guarantees
- no silent promotion of staged dead-code lanes into reviewed platform claims

## Later-Sprint Preservation Checklist

Later Epic 7 implementation work should preserve:

1. strongest local reviewed baseline still reads as:
   - `make quality-review-full`
2. reviewed CMake parity claims still route through:
   - `make quality-review-cmake`
   - `ctest -N --test-dir build/quality-review-cmake`
3. platform wording still preserves:
   - Linux strongest reviewed truth
   - macOS narrower reviewed lane plus supplemental install proof
   - Windows reviewed CMake subset only
4. install/package wording still preserves:
   - static-first release shape
   - local Unix-side script proof ownership
   - no broad reviewed install-validation claim
5. benchmark wording still preserves:
   - canonical maintained surface is bounded
   - `bench-canonical-report` is threshold-free artifact reporting
6. dead-code wording still preserves:
   - Linux enforced
   - macOS staged
   - Windows staged
   - serialized execution limit remains explicit

## Deferred Residuals

The following remain explicitly deferred unless a later touched lane truly
forces them:

- reviewed shared-library packaging claims
- Windows Makefile reviewed-wrapper parity
- Windows dead-code enablement
- macOS dead-code enablement
- broad reviewed install-validation convergence
- new timing-threshold governance beyond the currently justified surfaces

## Exit State

Sprint 70 Day 10 closes with one explicit truthfulness fence:

1. maintained reviewed truth:
   - strongest local reviewed baseline plus current reviewed platform lanes
2. supplemental-but-truthful proof:
   - direct runtime, extra compiler/sanitizer lanes, local install scripts,
     and threshold-free benchmark artifacts
3. explicitly deferred ambitions:
   - shared-library maturity, reviewed Windows Makefile/dead-code parity, and
     broad install-validation parity

That gives Day 11 one exact job:

- synthesize the product, capability, surface, and contract audits into one
  explicit Epic 7 state-of-the-art target and success rubric
