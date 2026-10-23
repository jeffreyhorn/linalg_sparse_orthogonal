# Sprint 217 Day 3: Finite Extreme-Scale Norm Fixtures

**Scope:** [Day 3 plan](../PLAN.md), [Day 2 contract](day2-numerical-behavior-contract.md).
This is a before-fix regression record. The only C file changed was
`tests/test_sparse_vector.c`; `src/sparse_vector.c` remains unchanged. The selected vector test
suite is intentionally red until the norm implementation is repaired.

## Added Cases

| Test | Input and independent expectation | Pre-fix observation | State |
| --- | --- | --- | --- |
| `test_norm2_large_finite` | `[1e200, -1e200]`; `hypot` = `1.414213562373095e+200` | `inf` | Expected fail: representable result overflowed while squaring |
| `test_norm2_tiny_finite` | `[1e-200, -1e-200]`; `hypot` = `1.414213562373095e-200` | `0` | Expected fail: both component squares underflowed |
| `test_norm2_mixed_scales` | `[1e200, 1e-200, -1e200, -1e-200]`; `hypot` of large pair = `1.414213562373095e+200` at double precision | `inf` | Expected fail: mixed magnitudes and signs must not alter representable norm |
| `test_norm2_near_equal_opposite_signs` | `[1e154, -1e154*(1+DBL_EPSILON)]`; `hypot` = `1.4142135623730953e+154` | `inf` | Expected fail: squares fit separately but their sum overflows |
| `test_norm2_finite_singleton_and_zero_controls` | `[-1e100]` -> `1e100`; `[0,-0]` and `n=-1` -> `+0` | All pass | Control: ordinary singleton, signed zero, nonpositive length |
| `test_norminf_finite_extremes` | `[DBL_MAX,-1e-200,0]` -> `DBL_MAX`; `[-1e-200]` -> `1e-200` | Both pass | Control: finite maximum is unaffected |

All nonzero 2-norm checks use `isfinite(actual)` and relative error
`fabs(actual/expected - 1) <= 8*DBL_EPSILON`; the tiny case additionally
requires `actual > 0`. Expected paired norms come from the C library's
`hypot`, not the implementation's square/sum recurrence. The mixed-scale
oracle omits the tiny pair because it cannot change the correctly rounded
double result at the `1e200` scale. Opposite signs are deliberate: a norm
must aggregate magnitudes rather than cancel entries.

The existing `test_norm2_basic`, `test_norm2_single`, `test_norm2_zero`, and
`test_norm2_null` remain registered and passing. New zero and infinity-norm
controls explicitly require finite results because the test framework's
`ASSERT_NEAR` comparison alone would not fail on NaN.

## Local Before-Fix Run

Date: 2026-10-20. Branch: `sprint-217`. Base commit:
`bde1d873c55a5fbda9fd39531f4ca2c05c7ee3f7`. Toolchain:
Apple clang 17.0.0 (`clang-1700.6.4.2`),
`x86_64-apple-darwin24.6.0`, CommandLineTools. These are local observations,
not a hosted platform result.

| Command | Exit | Evidence |
| --- | ---: | --- |
| `clang-format --dry-run --Werror tests/test_sparse_vector.c` | 0 | New C test formatting accepted |
| `make build/test_sparse_vector` | 0 | Test binary rebuilt with strict compiler warnings enabled by Makefile |
| `./build/test_sparse_vector` | 1 | 30 run, 4 failed, 0 skipped, 61 assertions |

Exact selected output from the test run:

```text
large finite norm2: actual=inf expected=1.414213562373095e+200
tiny finite norm2: actual=0 expected=1.414213562373095e-200
mixed finite norm2: actual=inf expected=1.414213562373095e+200
near-equal signed norm2: actual=inf expected=1.4142135623730953e+154
Tests run:    30
Tests failed: 4
Tests skipped: 0
Assertions:   61
SOME TESTS FAILED
```

The nonzero exit is required evidence, not a passing quality gate. Do not
remove, skip, suppress, or weaken these fixtures to regain green status.
Days 5-7 must fix `vec_norm2`; rerun the exact command afterward and require
all four formerly failing cases to pass. Day 4 adds nonfinite and solver
fixtures separately. A full `make test` run has not been claimed for Day 3.
