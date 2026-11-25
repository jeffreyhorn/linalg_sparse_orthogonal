# Day 4 Add Permutation Enforcement and Caller Audit

## Outcome

`sparse_add` and `sparse_add_inplace` now enforce the Day 2 original-state contract through the existing `sparse_matrix_require_original_state` seam. Both operands must be unfactored and have identity forward and inverse row and column maps.

Validation is placed after pointer and shape checks and before result allocation, cache invalidation, factor-state clearing, scaling, insertion, or value mutation. In-place self alias is rejected after state validation and before mutation. No public type, function signature, exported symbol, or index-width behavior changed.

`sparse_matmul` remains unchanged for Day 5. Its nine permutation combinations remain an explicit experimental failure rather than being conflated with the completed add work.

## Validation order

| API | Enforced order | Rejection state |
| --- | --- | --- |
| `sparse_add` | non-NULL A/B/C_out; clear `*C_out`; equal shape; original A; original B; allocate | state rejection returns `SPARSE_ERR_BADARG`, leaves `C_out=NULL`, and preserves both inputs |
| `sparse_add_inplace` | non-NULL A/B; equal shape; original A; original B; reject `A==B`; invalidate and mutate | state or alias rejection returns `SPARSE_ERR_BADARG` and preserves A and B, including cache/factor metadata |

The tests separately prove that shape errors precede state errors and that a NULL A or B leaves valid `C_out` storage unchanged under the pre-existing `sparse_add` pointer contract.

## Regression coverage

The always-on arithmetic suite now covers:

- add and in-place add with left, right, and both operands selected;
- row-only, column-only, and combined row-plus-column valid maps;
- an inverse-row-only add violation and inverse-column-only in-place violation;
- factored A for out-of-place add and factored B for in-place add;
- shape mismatch combined with nonidentity state;
- in-place `A==B` with a populated norm cache;
- stale output ownership across NULL input rejection;
- an armed allocation failure that remains unconsumed by state rejection, then fires at the first valid add allocation and permits a successful retry.

Snapshots compare logical values, all four maps, `nnz`, cached norm, factor norm, and factored compatibility state, then verify row/column links and counts. The malformed inverse-only fixture intentionally bypasses map-validity checking only when comparing unchanged bytes/fields; linked storage must still remain valid.

Focused result after formatting: 32 tests passed, one Day 5 matmul test skipped, 388 assertions, zero failures.

## Caller audit

`rg` over checked-in C and header files found no production caller of `sparse_add` or `sparse_add_inplace`; all current call sites outside their definitions are tests. Existing identity-state call patterns include distinct add operands, out-of-place same-input add, rectangular add, cancellation, solver integration, and in-place scaling. They remain supported and pass unchanged.

The new failure applies only to calls that already violated the public warning against physical arithmetic on nonidentity/factored state, plus the newly frozen in-place self-alias rule. There is no internal caller migration or compatibility wrapper to update.

## Evidence commands

| Command | Result |
| --- | --- |
| `make format` | Pass |
| `make lint` | Pass: strict compile, clang-tidy, and cppcheck (112/112) |
| `make test` | Pass: full runtime suite |
| `make docs-check` | Pass: all 18 checked-in public headers covered |
| Strict standalone compile of `tests/test_sparse_arith.c` with `-Werror` | Pass |
| `cmake --build build/s227-day1-cmake --target test_sparse_arith --parallel 4` | Pass |
| `build/s227-day1-cmake/test_sparse_arith` | Pass: 32 tests, one Day 5 skip, 388 assertions |
| `SPARSE_TEST_EXPERIMENTAL=1 build/s227-day1-cmake/test_sparse_arith` | Expected Day 5-only failure: nine matmul status mismatches and nine unexpected outputs; all Day 4 tests pass |
| `git diff --check` | Pass |

## Boundary and handoff

Day 4 establishes pre-allocation/pre-mutation enforcement only for add and in-place add. Day 5 must apply the same seam to matmul, promote its pending test, complete public contract wording, and design the sorted-add output builder. No performance or logical-permutation-support claim is made.
