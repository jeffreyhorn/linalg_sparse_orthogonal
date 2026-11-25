# Day 9 Add Failure, Alias, and Randomized Closure

## Outcome

The out-of-place `sparse_add` contract now has executable coverage for every documented rejection/output state, allocation behavior for both dense output and complete cancellation, caller output-variable alias hazards, repeated calls, same-input success, and replayable wide rectangular arithmetic. The public header now explains that `C_out` is an output storage location rather than an owning smart pointer.

No arithmetic or ownership behavior changed. Day 9 closes tests and documentation around the frozen contract.

## Exact input snapshots

The Sprint 227 snapshot helper now records and compares:

- dimensions, `nnz`, logical values, cached norm, factor norm, and factored state;
- all forward and inverse permutation values and their backing-array addresses;
- row/column header-array addresses and every header pointer;
- factor-state and reorder-permutation ownership pointers;
- pool head, current slab, free list, slab count, and current-slab use count; and
- every node pointer, coordinate, value, right link, and down link in row order.

This upgrades failure assertions from value/validity checks to exact input topology and metadata preservation for compact fixtures.

## Failure and output ownership matrix

One stale-output-owner fixture checks all documented out-of-place rejection classes:

| Call | Status | Output storage | Inputs and stale owner |
| --- | --- | --- | --- |
| NULL `C_out` | `SPARSE_ERR_NULL` | No storage to update | Exact A/B snapshots retained |
| NULL A or B | `SPARSE_ERR_NULL` | Existing pointer remains unchanged | Available input and stale owner exact |
| Shape mismatch | `SPARSE_ERR_SHAPE` | Cleared to NULL | A, mismatched input, and stale owner exact |
| Original-state rejection | `SPARSE_ERR_BADARG` | Cleared to NULL | A/B and stale owner exact |
| First allocation failure | `SPARSE_ERR_ALLOC` | Cleared to NULL | A/B and stale owner exact |

The stale owner is retained separately and remains valid, proving that clearing output storage neither frees nor mutates its former pointee.

## Cancellation-only allocation sweep

A dense 6x6 pair with `B = -A` emits no output nodes. Its allocation surface therefore ends after the result object, six shell arrays, and column-tail workspace:

- fail indices 0–7 return `SPARSE_ERR_ALLOC` and NULL output;
- every failure preserves exact A/B snapshots and is followed by a successful retry;
- indices 8 and 9 succeed, covering the final and one-beyond-final boundaries; and
- every success returns a structurally valid empty matrix with zero slabs and `nnz=0`.

Together with Day 8's two-slab sweep, this covers both maximum publication and zero-node publication paths.

## Output-variable aliases and repeated calls

The alias regression proves:

- a caller variable currently storing A may also be passed as `C_out` when a separate A owner is retained; success replaces only that variable and leaves A exact;
- allocation failure clears the aliased caller variable to NULL but leaves the separately retained A exact; resetting injection permits retry;
- a second call may reuse a variable holding the first result, but the first result is not freed or mutated and must remain separately owned;
- `sparse_add(A, A, 2, -1, &slot)` succeeds with a fresh result equal to A while A remains exact; and
- when an input pointer itself is NULL, pointer validation occurs before output clearing, even if output storage currently holds the other input.

The public `sparse_add` documentation now states these storage and ownership rules directly.

## Replayable wide rectangular corpus

Seed `0x22709add` drives 48 deterministic cases with:

- 2–8 rows;
- 33–97 columns, always rectangular and wide;
- sparse integer inputs in `[-8, 8]`;
- integer coefficients in `[-3, 3]`;
- exact independent dense-oracle arithmetic and the frozen `1e-15` output threshold; and
- exact post-call A/B logical values, output values, copy values, `nnz`, and row/column structure.

The prior seed `0x2270add7` remains as the 96-case compact corpus. Both seeds are fixed literals in the always-on suite for direct replay.

## Evidence

| Command | Result |
| --- | --- |
| `make format` | Pass |
| Strict `-Werror` syntax check of `tests/test_sparse_arith.c` | Pass |
| Focused 32-bit arithmetic suite | 46 tests, 79,486 assertions, zero failures/skips |
| Focused 64-bit arithmetic suite | 46 tests, 79,486 assertions, zero failures/skips |
| Focused ASan/UBSan arithmetic suite | 46 tests, 79,486 assertions, zero failures/skips |
| `make lint` | Pass: strict warnings, clang-tidy, and cppcheck |
| `make test` | Pass: complete repository test suite |
| `make docs-check` | Pass: all 18 checked-in public headers covered |
| `git diff --check` | Pass |

