# Day 8 Add Allocation-Failure Sweep

## Outcome

The sorted `sparse_add` builder now has an exhaustive deterministic allocation-failure sweep from its first result allocation through one boundary beyond its final allocation. Every injected failure returns `SPARSE_ERR_ALLOC`, leaves the caller output NULL, preserves both inputs, frees all partial ownership, and is followed immediately by a successful structurally valid retry.

No production change was required on Day 8. The Day 6 finish/abort ownership design survived every shell, workspace, first-slab, and later-slab boundary.

## Allocation map

The sweep uses a `(SPARSE_NODES_PER_SLAB + 1)` by 1 checkerboard fixture. A and B own alternating rows, so their union emits exactly one output node per row and necessarily allocates two output slabs under any positive supported slab size.

| Fail index | Allocation | Expected result |
| ---: | --- | --- |
| 0 | Result `SparseMatrix` object | Clean `SPARSE_ERR_ALLOC` |
| 1 | Row-header array | Clean `SPARSE_ERR_ALLOC` |
| 2 | Column-header array | Clean `SPARSE_ERR_ALLOC` |
| 3 | Forward row permutation | Clean `SPARSE_ERR_ALLOC` |
| 4 | Inverse row permutation | Clean `SPARSE_ERR_ALLOC` |
| 5 | Forward column permutation | Clean `SPARSE_ERR_ALLOC` |
| 6 | Inverse column permutation | Clean `SPARSE_ERR_ALLOC` |
| 7 | Builder column-tail workspace | Clean `SPARSE_ERR_ALLOC` |
| 8 | First output-node slab | Clean `SPARSE_ERR_ALLOC` |
| 9 | Second output-node slab after a full first slab | Clean `SPARSE_ERR_ALLOC` |
| 10 | No remaining allocation | Success |
| 11 | One boundary beyond the final allocation | Success |

This establishes ten injectable allocations for the selected two-slab result and proves the success boundary is not inferred from a single lucky index.

## Assertions per boundary

For every failing index 0–9, the regression checks:

- exact `SPARSE_ERR_ALLOC` status;
- NULL output;
- unchanged dimensions, `nnz`, identity maps, values, row links, column links, and shared node identity in A and B;
- a reset followed by successful reconstruction;
- exact output dimensions, `nnz`, values, identity maps, and two-slab ownership;
- one node per row, monotonic single-column traversal, and pointer identity between every row head and its corresponding column node; and
- successful output destruction before the next failure index.

Indices 10 and 11 independently produce the same valid result and are destroyed normally.

## Ownership and leak evidence

The focused ASan/UBSan run passes the complete sweep without an invalid access, use-after-free, or double free. Because the local Apple AddressSanitizer runtime does not support LeakSanitizer, the focused binary was also run under macOS `leaks`; it reports 0 leaks and 0 leaked bytes.

The result follows directly from the builder ownership boundary:

- shell-allocation failure is cleaned by `sparse_create_impl`;
- workspace failure enters `sparse_sorted_output_abort` with one matrix owner;
- slab failure leaves every previously published node owned by the matrix pool;
- abort frees the workspace and the complete matrix pool exactly once; and
- finish frees only workspace, transfers the matrix, clears the builder pointer, and disables abort ownership.

## No-repeated-scan inspection

The `sparse_add` path still performs one two-pointer row merge and calls only `sparse_sorted_output_append` for retained values. Append uses the current row tail and indexed column tail directly. It contains no output row walk, output column walk, coordinate search, or `sparse_insert` call. Therefore the failure-hardening work did not regress the Day 6 O(1)-per-retained-node linking criterion.

## Evidence

| Command or inspection | Result |
| --- | --- |
| `make format` | Pass |
| Strict `-Werror` syntax check of `tests/test_sparse_arith.c` | Pass |
| Focused 32-bit arithmetic suite | 42 tests, 8,613 assertions, zero failures/skips |
| Focused 64-bit arithmetic suite | Pass |
| Focused ASan/UBSan arithmetic suite | Pass |
| macOS `leaks --atExit` focused arithmetic suite | 0 leaks, 0 leaked bytes |
| Source-path inspection | `sparse_add` retains one input merge and direct tail append; no output scan or `sparse_insert` call |
| `make lint` | Pass: strict production compile, clang-tidy, and cppcheck (112/112) |
| `make test` | Pass: full runtime suite |
| `make docs-check` | Pass: all 18 checked-in public headers covered |
| `git diff --check` | Pass |

