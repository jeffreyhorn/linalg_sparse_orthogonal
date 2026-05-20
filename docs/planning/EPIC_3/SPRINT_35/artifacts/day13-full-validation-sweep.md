# Sprint 35 Day 13: Full Validation Sweep

## Scope

Re-run the maintained reviewed-quality and CMake-parity flows after the Sprint
35 public-doc rewrite, and capture the final validated baseline before
closeout.

This day is the proof that Sprint 35 changed the public-facing guidance
without disturbing the enforced Sprint 34 operator and parity paths.

## Commands Run

### Direct maintained gates

- `/usr/bin/time -p make format`
- `/usr/bin/time -p make lint`
- `/usr/bin/time -p make test`

### Reviewed Makefile wrappers

- `/usr/bin/time -p make quality-review-compile`
- `/usr/bin/time -p make quality-review`

### Reviewed CMake parity wrappers

- `/usr/bin/time -p make quality-review-cmake-compile`
- `/usr/bin/time -p make quality-review-cmake`

### Explicit full-suite timing on the maintained CMake tree

- `/usr/bin/time -p ctest --test-dir build/quality-review-cmake --output-on-failure`

## Main Result

All intended Day 13 validation flows passed.

Sprint 35 closes Day 13 with the public-doc rewrite in place and the full
Sprint 34 reviewed-quality baseline still green:

- direct maintained gates passed
- reviewed Makefile wrappers passed
- reviewed CMake parity wrappers passed
- dead-code checks remained green
- active CTest suite count remained `53`

## Timings

### Direct maintained gates

- `make format`
  - passed
  - `real 5.16`
- `make lint`
  - passed
  - `real 432.50`
- `make test`
  - passed
  - `real 106.16`

### Reviewed Makefile wrappers

- `make quality-review-compile`
  - passed
  - `real 381.24`
- `make quality-review`
  - passed
  - `real 560.29`

### Reviewed CMake parity

- `make quality-review-cmake-compile`
  - passed
  - `real 73.68`
- `ctest --test-dir build/quality-review-cmake --output-on-failure`
  - `53 / 53` passed
  - `Total Test time (real) = 173.23 sec`
  - `/usr/bin/time -p` reported `real 173.27`

## What Was Reconfirmed

### 1. The direct Sprint 34 quality baseline still holds

The direct operator gates remained green after the Sprint 35 doc rewrite:

- formatting still passes
- strict compile/static-analysis still passes
- the maintained Makefile test suite still passes

Because `make lint` still includes the compile-only benchmark/example gate,
this also reconfirms that the example-facing compile surface named by Sprint 35
remains clean.

### 2. The reviewed Makefile wrapper paths still work end to end

Both maintained wrapper targets passed:

- `quality-review-compile = format-check + lint`
- `quality-review = format-check + lint + test + deadcode-check`

That means Sprint 35 did not regress the operator-facing entry points that
Sprint 34 established as the reviewed local workflow.

### 3. The maintained CMake parity contract is still exact

`make quality-review-cmake-compile` passed and kept the test-count parity
check exact:

- `ctest -N`: `53`
- Makefile/CMake parity: `53` vs `53`

The full suite also remained green on the maintained CMake tree:

- `53 / 53` tests passed

High-signal confirmations from the final suite:

- `test_framework_optin`
  - passed with `8` run / `0` failed / `3` skipped
- `test_reorder_nd`
  - passed
  - remained the dominant long test at `100.57 sec` in the wrapper log
- `test_reorder_amd_qg`
  - passed

### 4. Sprint 35 carries no new drift into closeout

Day 13 surfaced no new mismatch in:

- public type names
- public option-struct snippet style
- README/tutorial command naming
- reviewed quality-wrapper behavior
- CMake parity expectations

So Day 14 can stay a true closeout/handoff pass rather than another cleanup
cycle.

## Bottom Line

Sprint 35 improved public documentation truthfulness while preserving the
reviewed local quality and CMake parity flows exactly.

The validated closeout baseline is:

- direct gates: green
- reviewed Makefile wrappers: green
- reviewed CMake parity wrappers: green
- CTest count: `53`
- full CTest suite: `53 / 53` passed
- dead-code sibling path: green
