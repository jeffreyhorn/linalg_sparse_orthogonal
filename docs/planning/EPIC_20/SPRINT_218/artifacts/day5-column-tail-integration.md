# Day 5: Column-Tail Integration

**State:** The matmul output builder now links both dimensions without a
per-entry head search. Day 2 remains the pre-change timing baseline; no
speedup or memory claim follows from this structural change alone.

## Construction And Ownership

`sparse_matmul` retains its row accumulator, touched-column sort, and
`fabs(acc[col]) >= 1e-15` emission rule. The private append helper now uses
the current row tail and a zero-initialized `B->cols`-element column-tail
array. Output rows are visited in increasing order, so the next node for a
column always follows its tail; sorted touched columns similarly keep each
row chain ordered. Each retained coordinate creates one pool node, shared
by the row and column chains. Linking is O(1) per emitted node, after the
existing accumulation and per-row sort.

`sparse_create` checks output shell dimensions and its header/permutation
buffers. The new tail array uses `sparse_calloc_idx_array`, which checks
index-to-size conversion and byte multiplication before allocating. The
append helper checks `nnz == IDX_MAX` and node allocation before linking.
There is no fallible operation between the two links or before `nnz++`.
Both success and failure paths release all four work arrays. Failure also
destroys the private output, including any nodes already emitted; `*C`
remains NULL. The result keeps `sparse_create`'s identity permutations,
invalid norm cache, and unfactored state.

## Focused Coverage

`test_matmul_column_links_and_consumers` builds a 4x5 product with an empty
middle row, three empty columns, and repeated nonzeros in two columns. It
checks shape, `nnz`, default metadata, row and column order, pointer identity
across both chains, and NULL termination. It then exercises `sparse_get`,
`sparse_matvec`, CSR and CSC conversion, and `sparse_free`. The allocation
harness now fails the new column-tail workspace allocation, verifies NULL
output and an unchanged caller-owned stale matrix, then retries successfully.
The existing reverse-touch, cancellation, and non-finite cases continue to
pin the previous accumulator behavior.

## Verification

| Command | Result |
| --- | --- |
| `clang-format --dry-run --Werror src/sparse_matrix.c tests/test_matmul.c` | Exit 0. |
| `make matmul-allocation-failure-gate` | Exit 0; registration passed, 22 tests, 0 failed/skipped, 6487 assertions. |
| `cmake --build build/s218-baseline --target test_matmul --parallel 2` | Exit 0. |
| `ctest --test-dir build/s218-baseline -R '^test_matmul$' --output-on-failure` | Exit 0; 1/1 passed. |
| `make format-check` | Exit 0. |
| `make source-list-check` | Exit 0; 49 library sources. |
| `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. |
| `make test` | Exit 0; full Make suite ended with `All tests passed.` |

The independent dense oracle and broad failure-site coverage remain Days
6-10 work. A matched post-change benchmark comparison remains Days 11-12.
