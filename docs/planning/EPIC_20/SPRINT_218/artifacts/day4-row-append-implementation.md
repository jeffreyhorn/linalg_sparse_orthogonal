# Day 4: Ordered Row Emission

**State:** Row-head rescans removed from `sparse_matmul`; column linking is
still the existing sorted head search and is assigned to Day 5. This is an
intermediate, structurally valid implementation, not the final linear-time
output builder. Day 2's measurements remain the **pre-change** baseline;
no speedup claim is made here.

## Change

`src/sparse_matrix.c` now uses a private `matmul_append_row_node` after the
existing touched-column sort and drop predicate. A local `row_tail` resets
for each physical output row. Every retained column gets exactly one
`sparse_matrix_make_node` allocation; the new node is linked at the current
row tail in O(1), while the column chain is searched from its head and
linked in ascending physical row order. `out->nnz` increases only after both
links exist. The output is freshly created, so `cached_norm=-1`, identity
permutations, and unfactored state already match the old `sparse_insert`
result. It is private until all rows are complete, so no output mutex is
needed during construction.

The contribution traversal and arithmetic are unchanged. The original
`fabs(acc[col]) >= 1e-15` predicate still drops exact cancellation,
sub-threshold values, and NaN accumulators, while retaining infinities.
Each touched accumulator and flag is reset after consideration. Empty rows
leave NULL row headers and do not carry a stale tail into the next row.

One `fail` path now frees the three work arrays and the entire private
output (including all previously emitted pool nodes); `*C` remains NULL.
The append helper returns before linking anything if `out->nnz` is already
`IDX_MAX` or a pool allocation fails. There is no error-producing operation
between row and column linking. The existing checked workspace failure
positions and retry behavior remain unchanged. The planned checked
column-tail allocation and its additional failure position are Day 5/9-10
work, not hidden in this change.

## Focused Regression Matrix

| Case | What it proves |
| --- | --- |
| `test_matmul_wide_row_sorted_after_reverse_touches` | 1x1024 product touches columns in descending order; output row chain contains exactly 1024 ascending, unit-valued nodes. |
| `test_matmul_row_append_duplicate_cancellation_and_empty_row` | Multiple contributions sum once per coordinate, cancellation drops a touched column, and an empty middle row stays empty. |
| `test_matmul_nan_drops_and_infinity_is_retained` | Current non-finite comparison behavior is preserved without a new numeric policy. |
| Existing matmul allocation/retry and integration cases | On failure, output is NULL and retry produces the same values; existing shape, identity, rectangular, and SuiteSparse controls still run. |

## Verification

| Command | Result |
| --- | --- |
| `clang-format --dry-run --Werror src/sparse_matrix.c tests/test_matmul.c` | Exit 0. |
| `make matmul-allocation-failure-gate` | Exit 0; registration passed, 21 tests passed, 0 failed/skipped, 6373 assertions. |
| `cmake --build build/s218-baseline --target test_matmul --parallel 4` | Exit 0; rebuilt changed library and test source. |
| `ctest --test-dir build/s218-baseline -R '^test_matmul$' --output-on-failure` | Exit 0; 1/1 passed. |
| `make format-check` | Exit 0. |
| `make source-list-check` | Exit 0; 49 library sources. |
| `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. |
| `make test` | Exit 0; full Make suite ended with `All tests passed.` |

The current column head scan remains O(sum `c_j(c_j-1)/2`) across output
columns. Day 5 replaces it with persistent column tails and adds column-link
consumer/structure checks. Day 6 will review the final builder against the
independent dense oracle; Days 9-10 extend failure-site coverage.
