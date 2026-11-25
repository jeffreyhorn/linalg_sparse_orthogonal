# Day 6 Sorted Addition Append Construction

## Outcome

`sparse_add` now emits its already sorted two-pointer merge through a file-local `SparseSortedOutputBuilder`. The builder links each retained node directly to the current row tail and its column tail. The add path no longer calls `sparse_insert`, so it does not restart row or column scans for each output node.

The arithmetic contract is unchanged: pointer, shape, and original-state validation still precede allocation; finite scaling and overlap combination retain the same accumulation order; magnitudes below `1e-15` are dropped; exact cancellation produces no node; inputs remain borrowed and unchanged; and success returns a fresh caller-owned original-state matrix.

## Builder implementation

The implementation follows the Day 5 design:

- `sparse_sorted_output_init` creates an injectable matrix shell and a zeroed column-tail workspace;
- `sparse_sorted_output_begin_row` enforces consecutive ascending rows and resets the row tail;
- `sparse_sorted_output_append` enforces strictly ascending columns, checks column-row monotonicity and `IDX_MAX`, allocates and initializes one injectable node, publishes row and column links, then increments `nnz`;
- `sparse_sorted_output_finish` frees workspace and transfers the sole matrix owner;
- `sparse_sorted_output_abort` frees workspace and the partially linked matrix and is safe after any pre-finish failure.

The builder is deliberately file-local and used only by out-of-place addition. Matmul retains its existing append helper, and in-place addition retains its cursor/update path. No public symbol, layout, function signature, threshold, or permutation contract changed.

## Structural evidence

The representative 4x5 regression includes all initial implementation patterns in one deterministic output:

| Row | Merge pattern | Expected retained columns |
| --- | --- | --- |
| 0 | overlap, A-only, B-only, and scaled combination | 0, 1, 2, 4 |
| 1 | disjoint A/B entries | 0, 1, 3 |
| 2 | exact overlap cancellation | none |
| 3 | empty A row and B-only entries | 2, 4 |

The test checks every dense value, per-row node count, strict row ordering, global row/column link validity, duplicate freedom through strict ordering, total `nnz=9`, identity maps, invalid/uncomputed norm metadata, unfactored output state, and unchanged input snapshots. Successful destruction is exercised by normal cleanup.

## Allocation evidence

The result shell and all six shell arrays are injectable. Focused failures additionally cover:

- fail-after 7: column-tail workspace allocation;
- fail-after 8: first output-node slab allocation.

Both return `SPARSE_ERR_ALLOC`, leave output NULL, preserve both inputs, and clean the partial builder. Resetting injection then produces a structurally valid four-node result. Day 7 remains responsible for the complete shell/workspace sweep and later slab boundaries.

## Complexity boundary

For `r` rows, `c` columns, input nonzeros `nnz(A)+nnz(B)`, and retained output `nnz(C)`, construction now performs:

- one linear two-pointer pass over both inputs;
- O(1) row-tail and column-tail link work per retained output node;
- O(c) temporary column-tail storage;
- no output-coordinate search and no per-node row/column restart.

This establishes an implementation complexity property, not a measured speed claim. Performance evidence remains reserved for the preregistered later sprint days.

## Evidence

| Command or inspection | Result |
| --- | --- |
| `make format` | Pass |
| Strict `-Werror` syntax check of implementation and arithmetic tests | Pass |
| Focused CMake build of `test_sparse_arith` | Pass |
| Focused arithmetic binary | 39 tests, 595 assertions, zero failures/skips |
| Source inspection | No `sparse_insert` call remains inside `sparse_add` |
| `make lint` | Pass: strict compile, clang-tidy, and cppcheck (112/112) |
| `make test` | Pass: full runtime suite |
| `make docs-check` | Pass: all 18 checked-in public headers covered |
| `git diff --check` | Pass |

