# Day 10: Matmul Rollback, Retry, And Instrumentation

**State:** The registered matmul failure gate sweeps every later slab boundary
in a three-row product, immediately retries each failure, and checks the
result against an independently specified dense-value formula. The output is
private until success, so a failed call must leave `*C == NULL` without
freeing the caller's stale output pointer.

## Failure Sweep

The fixture multiplies a `3 x 1` matrix with row values `1, 2, 3` by a
`1 x (NODES_PER_SLAB + 2)` matrix with values `j % 7 + 1`. The expected
result at `(i, j)` is `(i + 1) * (j % 7 + 1)`. This gives a wide first row
and two later rows with shared output columns. For every slab boundary after
the first, the test injects failure at that slab's allocation, requires
`SPARSE_ERR_ALLOC` and a NULL result, and checks stale output ownership and
unchanged input values and `nnz`. The fault hook is then reset and the same
inputs are multiplied immediately. Each retry checks the expected shape,
`nnz`, all output values, and ordered row and column chains before freeing
the result.

On the default 4096-node slab build, failures occur at node indices 4096
(row 0, column 4096), 8192 (row 1, column 4094), and 12288 (row 2,
column 4092). A build with `SPARSE_NODES_PER_SLAB=1` fails at every later
node, indices 1 through 8; this covers mid-row, row-transition, and
already-linked-column rollback without requiring thousands of separate
processes. The Day 9 gate continues to cover the shell, arrays, workspaces,
and first slab.

## Public Errors

The precedence regression checks a NULL output pointer before input or
shape validation, either NULL input with a valid output pointer, and shape
failure with a stale pointer. In all cases where the output pointer itself
is valid, the callee clears it but does not free caller-owned stale storage.
The public constructor rejects zero rows or zero columns, so there is no
valid zero-dimensional matrix to pass to `sparse_matmul` through this API.

## Instrumentation And Limits

| Command | Result |
| --- | --- |
| `make matmul-allocation-failure-gate` | Exit 0; 41 tests, 0 failed/skipped, 372262 assertions. |
| `cmake -S . -B build/s218-day10-sanitize -DSANITIZE=all -DCMAKE_C_FLAGS=-DSPARSE_NODES_PER_SLAB=1` | Exit 0. |
| `cmake --build build/s218-day10-sanitize --target test_matmul --parallel 4` | Exit 0. |
| `ctest --test-dir build/s218-day10-sanitize -R '^test_matmul$' --output-on-failure` | Exit 0; 1/1 passed under ASan and UBSan. |
| `cmake -S . -B build/s218-day10-leaks -DCMAKE_C_FLAGS=-DSPARSE_NODES_PER_SLAB=1` | Exit 0. |
| `cmake --build build/s218-day10-leaks --target test_matmul --parallel 4` | Exit 0. |
| `leaks --atExit -- build/s218-day10-leaks/test_matmul` | Exit 0; 41 tests passed; 0 leaks, 0 leaked bytes. |
| `leaks --atExit -- build/test_matmul` | Exit 0; 41 tests passed; 0 leaks, 0 leaked bytes with default slab size. |
| `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. |
| `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. |
| `make test` | Exit 0; final output `All tests passed.` |

On this macOS host, the repository's Make ASan target disables ASan leak
detection. The separate native `leaks` runs provide leak observations for
the same registered fixtures; ASan/UBSan covers memory access and undefined
behavior in the one-node-slab build. This is local instrumentation, not a
claim about every possible allocator or platform.

Day 11 owns final allocation-failure closure and the measurement setup.
