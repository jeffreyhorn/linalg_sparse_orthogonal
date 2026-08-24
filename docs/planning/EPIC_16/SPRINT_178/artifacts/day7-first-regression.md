# Sprint 178 Day 7: First Failure Regression

## Scope

Day 7 adds the first dedicated deterministic failure regression for the
selected Sprint 178 subsystem: `sparse_matmul()` workspace allocation.

The first selected ownership path is the accumulator workspace allocation
(`acc`), reached with the existing private allocation hook at fail-after count
`6`.

## Implemented Regression

| Test | File | Failure site |
| --- | --- | --- |
| `test_matmul_acc_allocation_failure_clears_stale_output` | `tests/test_matmul.c` | `acc` workspace allocation |

## Public-State Setup

The regression intentionally starts with a non-`NULL` caller output pointer:

1. Build valid `A` and `B` inputs.
2. Build a separate one-entry stale matrix.
3. Set `C` to the stale matrix before invoking `sparse_matmul()`.
4. Enable the private allocation hook for the first selected workspace
   allocation.

The stale matrix is retained through a separate local pointer so the test can
free it after `sparse_matmul()` clears `C`.

## Assertions

The regression asserts:

- `sparse_matmul(A, B, &C)` returns `SPARSE_ERR_ALLOC` when `acc` allocation
  is injected to fail.
- `C == NULL` after the failure, proving no stale public output remains
  observable through the public output parameter.
- the separate stale matrix still contains its original sentinel value and
  remains caller-owned.
- after `sparse_alloc_test_reset()`, the same `A` and `B` inputs retry
  successfully.
- retry output has the expected dimensions, nonzero count, and numeric
  product.

## Boundary

This is the first failure-path regression only. It does not broaden Sprint 178
to matrix shell allocation, `sparse_insert()` product-flush allocation, or
other matrix/solver subsystems.

## Validation

- `make build/test_matmul && ./build/test_matmul`
- `make format`
- `make lint`
- `make test`

## Handoff

Day 8 should apply the same stale-output/no-publication pattern to the
remaining selected workspace allocation sites: `nz_flag` and `touched`.
