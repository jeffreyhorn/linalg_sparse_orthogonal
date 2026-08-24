# Sprint 178 Day 6: Harness Implementation

## Scope

Day 6 implements the minimal test-side harness for the selected Sprint 178
subsystem: `sparse_matmul()` workspace allocation.

The implementation does not add public API, does not change production
allocation-hook semantics, and does not expand Sprint 178 beyond the selected
matrix-multiply workspace family.

## Implemented Files

| File | Change |
| --- | --- |
| `tests/test_matmul.c` | Adds a private-hook-backed allocation-failure harness for `sparse_matmul()` workspace arrays. |

## Selected Failure Sites

The test harness targets the three workspace allocations that happen after
`sparse_matmul()` has validated inputs and created its temporary output
matrix:

| Site | Allocation | Fail-after count |
| --- | --- | --- |
| accumulator workspace | `acc` | `6` |
| nonzero flag workspace | `nz_flag` | `7` |
| touched-column workspace | `touched` | `8` |

The counts preserve the existing one-shot countdown behavior from Sprint 176:
positive values count down through successful wrapped allocations, `0` fails
the next wrapped allocation once, and the hook then resets to normal behavior.

## Fixture

The harness builds both input matrices before enabling allocation-failure
injection:

`A` is `2 x 3`:

- `A(0,0) = 1`
- `A(0,2) = 2`
- `A(1,1) = 3`

`B` is `3 x 2`:

- `B(0,0) = 4`
- `B(2,1) = 5`
- `B(1,0) = 6`

The retry product is expected to be:

- `C(0,0) = 4`
- `C(0,1) = 10`
- `C(1,0) = 18`
- `C(1,1) = 0`

## Assertions

For each selected workspace allocation, the harness:

1. Resets the private allocation hook.
2. Sets the selected fail-after count.
3. Calls `sparse_matmul(A, B, &C)`.
4. Expects `SPARSE_ERR_ALLOC`.
5. Asserts `C == NULL`, proving no output publication on selected workspace
   allocation failure.
6. Resets the hook.
7. Calls `sparse_matmul(A, B, &C)` again.
8. Expects `SPARSE_OK`.
9. Verifies dimensions, nonzero count, and the expected numeric product.

## Boundaries Preserved

- No public allocation-failure test API was added.
- No production allocation behavior was redesigned.
- No broad allocation-failure coverage claim was introduced.
- The harness only covers `sparse_matmul()` workspace arrays.
- Output shell allocation and `sparse_insert()` allocation paths remain out of
  Sprint 178 Day 6 scope.

## Validation

- `make build/test_matmul`
- `./build/test_matmul`
- `make format`
- `make lint`
- `make test`

## Handoff

Day 7 should review the new harness against the Day 4 cleanup invariants and
decide whether additional no-stale-state assertions are needed before the
focused gate is added later in Sprint 178.
