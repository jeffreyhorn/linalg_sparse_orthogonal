# Day 6: Builder Review And Dense Oracle

**State:** Item 218.2's ordered output builder is complete. The independent
small-matrix oracle is in place for the first Day 6 shapes; Day 7 extends
contribution and cancellation coverage. Day 2 remains the pre-change timing
baseline, not evidence of a post-change speedup.

## Builder Review

`sparse_matmul` traverses each physical input row and accumulates products
in a one-row dense workspace. It sorts only touched column indexes, then
emits each retained coordinate through `matmul_append_row_node`. No output
path calls `sparse_insert` or searches a row or column head per value.
Because output rows advance monotonically and touched columns are sorted,
the local row tail and persistent per-column tails maintain strictly ordered
orthogonal chains. The output-link cost is O(result nnz), separate from
input contribution traversal and per-row touched-column sorts. Tail storage
is O(output columns); the accumulator, flags, and touched list are also
O(output columns). The output shell and pooled nodes remain independently
owned by the new matrix.

`sparse_create` checks positive output dimensions and its header and
permutation allocations. All four matmul work arrays use checked index
allocation helpers. The append helper checks the `nnz` limit and node
allocation before linking; there is no fallible operation between linking
the row and column chains. On any subsequent failure, the single cleanup
path releases all work arrays and the entire private matrix. `*C` stays
NULL until construction finishes. The public API, accumulation order, and
drop predicate are unchanged. There is no broader arithmetic or memory
improvement claim from this structural review.

## Independent Oracle

`tests/test_matmul.c` now computes each expected product with a direct dense
three-loop multiplication over caller-defined input arrays. It constructs
the sparse inputs separately, then checks the output against the dense
result without using the sparse builder to derive expectations. All fixtures
are small, finite, integer-valued matrices, so exact comparisons distinguish
construction errors without tolerance or non-finite policy questions. The
oracle verifies result dimensions, expected nonzero count, every retained
value, absent zero coordinates, strictly increasing row and column chains,
and pointer identity between both representations. Traversals are bounded
so malformed cycles fail the test instead of looping forever.

| Fixture | Scope |
| --- | --- |
| 3x2 times 2x4 | Rectangular shape and sparse row/column ordering. |
| Identity on either side | Both identity directions and empty structural coordinates. |
| Empty left and right operands | Zero-result ownership, headers, and `nnz`. |
| 4x3 times 3x5 with empty output row and columns | Structural zero rows and columns with other populated chains. |
| 2x3 times 3x2 | Fully dense result and all coordinates retained. |

Zero-dimensional matrices are not valid inputs to `sparse_create`; the
zero-row/column fixture refers to all-zero structural rows and columns.
Sub-threshold cancellation and non-finite behavior are intentionally left
to the existing cases and Day 7/8 work rather than copied into the oracle.

## Focused Verification

| Command | Result |
| --- | --- |
| `make matmul-allocation-failure-gate` | Exit 0; 27 tests, 0 failed/skipped, 6846 assertions. |
| `cmake --build build/s218-baseline --target test_matmul --parallel 2` | Exit 0; changed sources rebuilt. |
| `ctest --test-dir build/s218-baseline -R '^test_matmul$' --output-on-failure` | Exit 0; 1/1 passed. |
| `make format-check` | Exit 0. |
| `make source-list-check` | Exit 0; 49 library sources. |
| `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. |
| `make test` | Exit 0; full Make suite ended with `All tests passed.` |
