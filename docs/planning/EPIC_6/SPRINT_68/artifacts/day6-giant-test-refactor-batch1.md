# Sprint 68 Day 6: Giant-Test Refactor Batch 1

Date: 2026-06-13
Branch: `sprint-68`

## Purpose

Land the first bounded helper-extraction batch inside the Day 5 fence by
making `test_chol_csc.c` read more like the canonical family-local proof owner
and less like a mixed support-helper bucket.

## Landed Batch

Touched test surfaces:

- `tests/test_chol_csc.c`
- `tests/test_chol_csc_supernodal_helpers.h`

No proof-owner spillover, new test binary, or implementation widening was
required for this first batch.

## `tests/test_chol_csc_supernodal_helpers.h`: Expanded the Family-Local Support Seam

The landed batch moved four helpers out of the main giant test and into the
existing family-local helper header:

- `day7_chol_csc_get(...)`
- `day8_chol_csc_match(...)`
- `day10_factored_matches(...)`
- `day10_roundtrip_check(...)`

This consolidates support scaffolding that primarily serves:

- supernode diagonal-block reference comparisons
- scalar-vs-batched factored CSC equality checks
- writeback round-trip comparison plumbing

The helpers remain narrow and CSC-family-specific, so the landing stays bounded
to the existing local seam instead of widening into a new shared testing
abstraction.

## `tests/test_chol_csc.c`: Preserved the Canonical Proof Owner While Shrinking Local Support Clutter

The main file still owns:

- scenario assertions
- proof bodies
- family-local coverage intent
- explicit `RUN_TEST(...)` ordering

What it no longer needs to carry inline is the same amount of supernodal and
writeback support code for:

- factored-entry lookup
- batched-vs-scalar factor matching
- writeback round-trip comparison scaffolding

That means the landing reduced local maintenance pressure without splitting the
proof owner into multiple binaries or introducing opaque registration
machinery.

## Explicit Non-Widening Result

The first landed batch did not widen into:

- new `tests/test_chol_csc_*.c` binaries
- `tests/test_integration.c`
- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- any implementation `src/` file
- benchmark or maintained-doc truth surfaces

That matters because Sprint 68 still has real later lanes, and the Day 6
success condition was a bounded `test_chol_csc` helper extraction, not broader
test churn.

## Validation

Because `*.c` / `*.h` changed, the required validation set was run:

- `make format`
- `make lint`
- `make test`

And because this was substantial giant-test architecture work, the stronger
reviewed path was also run:

- `make quality-review-full`

The reviewed CMake parity anchor remained:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

No reviewed CMake failed-test log was left behind after the run.

## Exit State

Sprint 68 Day 6 now hands off one concrete first landing result:

1. `tests/test_chol_csc.c`
   - remains the canonical Cholesky CSC family-local proof owner
2. `tests/test_chol_csc_supernodal_helpers.h`
   - now owns more of the narrow supernodal/writeback support scaffolding
3. the batch stayed inside the exact two-file landing fence
4. validation and the stronger reviewed path completed from the landed state

That gives Day 7 one exact follow-through job:

- rerank the residual giant-test and assurance queue after the first landed
  `test_chol_csc` helper extraction
