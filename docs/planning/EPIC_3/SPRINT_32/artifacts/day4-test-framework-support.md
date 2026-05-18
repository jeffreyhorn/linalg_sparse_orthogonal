# Sprint 32 Day 4 Test-Framework Support

**Date:** 2026-05-17  
**Branch:** `sprint-32`

## Objective

Implement the minimum real framework support needed to represent slow or experimental tests honestly, without changing the repo’s fundamental one-binary-per-test-file execution model.

## What Shipped

### `tests/test_framework.h`

Day 4 adds three concrete capabilities:

1. explicit skipped-test accounting
2. a body-level skip primitive
3. opt-in wrappers for non-default tests

The new surface is:

- `SKIP_TEST("reason")`
- `RUN_TEST_SLOW(fn)` gated by `SPARSE_TEST_SLOW`
- `RUN_TEST_EXPERIMENTAL(fn)` gated by `SPARSE_TEST_EXPERIMENTAL`

The framework summary now reports:

- `Tests run`
- `Tests failed`
- `Tests skipped`
- `Assertions`

That closes the main honesty gap from Day 3: a deliberate non-default test is no longer forced to masquerade as either an active test or a commented-out stub.

### `tests/test_framework_optin.c`

Day 4 also adds a focused self-check binary that validates the new framework behavior directly.

It proves:

- ordinary `RUN_TEST(...)` behavior is unchanged
- `SKIP_TEST(...)` records a skip and suppresses the pass line
- disabled slow/experimental wrappers print visible `[SKIP]` lines and do not execute the test body
- enabled wrappers execute normally

### Build-system wiring

The new self-check is registered in both:

- `Makefile`
- `CMakeLists.txt`

So the support path is available from both local validation entry points used elsewhere in the repo.

## Why This Is The Right Size

Day 3’s policy explicitly rejected a large test-runner redesign. The shipped Day 4 support keeps that promise:

- no new external runner
- no category-specific CTest topology
- no xfail layer
- no special case for `test_reorder_nd.c`

Instead, the category choice stays local and reviewable at the exact call site in `main()`:

- active: `RUN_TEST(...)`
- slow opt-in: `RUN_TEST_SLOW(...)`
- experimental opt-in: `RUN_TEST_EXPERIMENTAL(...)`

That is enough for Sprint 32’s concrete need and small enough to be maintainable.

## Validation

### Makefile path

Validated commands:

- `make format`
- `make build/test_framework_optin build/test_reorder_nd`
- `./build/test_framework_optin`
- `./build/test_reorder_nd`

Results:

- `test_framework_optin` passed with `8` executed tests and `3` counted skips
- `test_reorder_nd` still passed all `23` active tests under the modified framework

### CMake / CTest path

Validated commands:

- `cmake -S . -B build/sprint32-day1-cmake`
- `cmake --build build/sprint32-day1-cmake --parallel 1 --target test_framework_optin test_reorder_nd`
- `ctest --test-dir build/sprint32-day1-cmake --output-on-failure -R '^test_framework_optin$'`

Results:

- `test_framework_optin` passed in CTest
- `test_reorder_nd` rebuilt cleanly in the same build tree and passed on its CTest rerun

Note:

- one combined `ctest -R 'test_framework_optin|test_reorder_nd'` invocation started before the just-linked `test_framework_optin` binary was visible on disk and marked it `Not Run`
- rerunning `test_framework_optin` after the build completed passed cleanly
- this was a timing artifact in the validation sequence, not a framework defect

## Implication For Day 5

The new support exists, but Day 2 and Day 3 already showed that the dormant `test_reorder_nd.c` trio is not a good fit for it:

- the three commented-out ND stubs are historical or retired target claims
- they are not live current contracts worth preserving as opt-in coverage

So Day 5 should use this support as the repository’s new truthful mechanism going forward, while still deleting the dormant ND trio from suite code and leaving their evidence in sprint docs.
