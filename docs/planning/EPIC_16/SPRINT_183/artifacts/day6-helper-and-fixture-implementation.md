# Sprint 183 Day 6: Helper And Fixture Implementation

## Purpose

Implement the source-controlled helper and focused helper tests needed for the
Sprint 183 Cholesky selected comparison fixture before extending the comparison
runner.

## Implemented Helper Behavior

`tests/chol_external_dense_reference.py` now supports the Day 5 fixture key:

```text
cholesky_spd_tridiag_5
```

The helper builds the deterministic 5x5 SPD tridiagonal matrix directly:

```text
A = [[ 4, -1,  0,  0,  0],
     [-1,  4, -1,  0,  0],
     [ 0, -1,  4, -1,  0],
     [ 0,  0, -1,  4, -1],
     [ 0,  0,  0, -1,  4]]
```

It uses the existing helper RHS policy, so `x_true = [1, 2, 3, 4, 5]` and
`rhs = A * x_true = [2, 4, 6, 8, 16]`. The dense Cholesky solve returns values
within the Day 5 `1e-10` tolerance.

## Compatibility Preserved

The helper still accepts Matrix Market path arguments for existing Cholesky CSC
external dense-reference checks. Missing `.mtx` paths continue to emit:

```text
SKIP matrix file not found
```

with exit code 0. Unknown non-path fixture keys now fail closed:

```text
ERROR unknown fixture not_a_fixture
```

with nonzero exit status.

## Focused Tests Added

Added `tests/test_chol_external_dense_reference.py` with focused coverage for:

- exact fixture matrix values;
- RHS derived from `[1, 2, 3, 4, 5]`;
- dense Cholesky solution agreement within `1e-12`;
- CLI success for `cholesky_spd_tridiag_5`;
- unknown fixture diagnostics;
- preserved missing Matrix Market skip behavior.

The test file has a standalone `main()` entry point, matching nearby helper
test style.

## C Fixture Decision

No C code changed on Day 6. A new C fixture test is not required for the helper
implementation because production Cholesky behavior is unchanged and
`tests/test_cholesky.c` already includes the 5x5 SPD tridiagonal factor/solve
proof. Day 8 may add or adjust C validation only if the runner implementation
touches production Cholesky behavior, which is not expected.

## Day 7 Handoff

The runner extension can rely on this helper invocation:

```text
python3 tests/chol_external_dense_reference.py cholesky_spd_tridiag_5
```

Expected output shape:

```text
OK 5
<solution_0>
<solution_1>
<solution_2>
<solution_3>
<solution_4>
```

The current implementation emits roundoff-level values for some entries, for
example `3.0000000000000009`, which is well inside the selected `1e-10`
tolerance. Runner parsing should treat values as floats and compare by
tolerance, not by string identity.

## Validation

Day 6 validation:

| Command | Status | Notes |
| --- | --- | --- |
| `python3 tests/test_chol_external_dense_reference.py` | Pass | Focused helper contract suite passed. |
| `python3 tests/chol_external_dense_reference.py cholesky_spd_tridiag_5` | Pass | Emitted `OK 5` and five solution values. |
| `python3 tests/chol_external_dense_reference.py not_a_fixture` | Pass | Failed closed with unknown fixture diagnostic. |
| `python3 tests/chol_external_dense_reference.py tests/data/symmetric_4.mtx` | Pass | Existing Matrix Market path behavior preserved. |
| `python3 tests/chol_external_dense_reference.py missing_fixture.mtx` | Pass | Existing missing-file skip behavior preserved. |
| `git diff --check` | Pass | Final whitespace validation passed after notes update. |

## Completion Criteria Review

| Criterion | Status | Notes |
| --- | --- | --- |
| Selected fixture can be evaluated by source-controlled project and baseline logic. | Complete | Baseline helper supports the selected fixture key; project-side runner logic is Day 7/8 scope. |
| Helper diagnostics are clear for missing optional dependencies or invalid inputs. | Complete | Unknown keys fail closed; missing `.mtx` paths keep skip behavior. Optional package rows remain runner-owned defer rows. |
| Any C changes have an identified validation path. | Complete | No C code changed; existing Cholesky C fixture coverage remains the proof path. |
