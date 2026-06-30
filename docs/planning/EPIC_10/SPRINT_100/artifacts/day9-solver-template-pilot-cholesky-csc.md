# Solver Comparison Evidence Pilot: Cholesky CSC External Dense Reference

## Summary

| field | value |
|---|---|
| comparison family | direct solver |
| solver path | Cholesky CSC solve |
| artifact type | pilot-filled example for Day 9 template |
| current owner | `tests/test_chol_csc.c` |
| external oracle owner | `tests/chol_external_dense_reference.py` |
| validation command | `make build/test_chol_csc && ./build/test_chol_csc` |
| current claim state | earned, bounded |

## Claim Evaluated

Bounded claim:

> Cholesky CSC solves agree with an external-process dense Cholesky reference
> on named SuiteSparse SPD fixtures.

## Fixtures

| fixture | source | class | reorder | backend |
|---|---|---|---|---|
| `nos4` | `tests/data/suitesparse/nos4.mtx` | SuiteSparse SPD Matrix Market fixture | `SPARSE_REORDER_NONE` | CSC |
| `bcsstk04` | `tests/data/suitesparse/bcsstk04.mtx` | SuiteSparse SPD Matrix Market fixture | `SPARSE_REORDER_AMD` | CSC |

## Problem Construction

| field | current behavior |
|---|---|
| matrix load | C harness loads Matrix Market fixture with `sparse_load_mm` |
| RHS | C harness sets `x_true[i] = i + 1`, then computes `b = A*x_true` |
| solver result | C harness factors a copy with `sparse_cholesky_factor_opts`, then solves with `sparse_cholesky_solve` |
| oracle result | Python helper loads dense Matrix Market data, builds the same RHS, runs dense Cholesky, then solves |

## Oracle Behavior

| oracle state | meaning |
|---|---|
| `OK n` followed by `n` values | oracle succeeded and returned reference solution |
| `SKIP reason` | fixture/helper unavailable; C harness records a skipped test |
| `ERROR reason` or non-zero exit | oracle failure; C harness treats as a failing comparison |

## Acceptance Criteria

| metric | purpose |
|---|---|
| max solution difference against dense reference | checks `x` against `x_ref` |
| relative residual | checks `A*x = b` consistency |
| tolerance | set by C harness per fixture helper invocation |

## Unsupported Cases

| case | current behavior |
|---|---|
| Windows external helper path | C harness skips with `external dense reference helper is not enabled on Windows` |
| non-SPD matrix | Python dense Cholesky raises an error; not part of this bounded lane |
| missing Matrix Market fixture | Python helper emits `SKIP matrix file not found` |

## Correctness Evidence

Current maintained tests:

- `test_external_dense_reference_nos4_csc`
- `test_external_dense_reference_bcsstk04_amd_csc`

Current maintained validation command:

```sh
make build/test_chol_csc && ./build/test_chol_csc
```

Day 9 did not rerun this focused command; the pilot records the current lane
shape for template design.

## Timing Evidence

None. This lane is an oracle/correctness lane, not a benchmark lane.

## Non-Claims

This evidence does not claim:

- broad Cholesky ecosystem parity;
- external factorization parity;
- proof of internal CSC layout equivalence;
- every SPD Matrix Market fixture is covered;
- Windows external dense-reference parity;
- portable performance superiority.

## Follow-Up Candidates

- add fixture taxonomy before broadening beyond `nos4` and `bcsstk04`;
- decide whether future Cholesky CSC comparisons need more conditioning,
  scale, reorder, or matrix-size variation;
- extract reusable oracle helper patterns only if they reduce giant-test
  ownership without hiding family-specific tolerance logic.

