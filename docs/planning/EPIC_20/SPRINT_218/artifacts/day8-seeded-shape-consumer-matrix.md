# Day 8: Seeded Shape And Consumer Matrix

**State:** The 218.3 correctness fixtures are implemented. This record
separates the tested behavior from a broader numerical API promise.

## Oracle And Consumers

The deterministic generator uses a local 32-bit state and four fixed seeds,
not `rand()` or shared test state. It produces small positive integer inputs
so the independent dense triple-loop product and matmul output can be
compared exactly. Explicit support masks prevent a generator change from
silently changing the sparse/dense coverage. The shared oracle checks shape,
`nnz`, every row and column link in order, row/column pointer identity,
`sparse_get` at every coordinate, matvec against the retained dense result,
CSR and CSC pointers/indices/values, and both unchanged input matrices.
Finite checks precede near comparisons so NaN cannot pass through the test
framework's comparison macro.

| Fixture | Shape | Expected support | Structural feature |
| --- | --- | --- | --- |
| Seed `0x2180801` tall sparse | 5x3 times 3x4 | 12/20 | Empty `B` row 1, empty output row 2 and column 3; nonzero `A` column 1 still reaches the empty intermediate row. |
| Seed `0x2180802` tall dense | 5x2 times 2x3 | 15/15 | Every output coordinate retained. |
| Seed `0x2180803` wide sparse | 2x5 times 5x6 | 6/12 | Odd output columns structurally empty; both rows retain even columns. |
| Seed `0x2180804` wide dense | 2x5 times 5x6 | 12/12 | Every output coordinate retained. |
| Large finite control | 1x2 times 2x2 | 2/2 | Each retained result is `1e300`; matvec remains finite. |
| Non-finite accumulation control | 1x2 times 2x3 | 2/3 | `+inf + -inf` drops as NaN; `DBL_MAX + DBL_MAX` retains positive infinity; `1e150 + 1e150` remains finite. |

The non-finite control pins the observed pre-change flush predicate, not a
new guarantee for arbitrary IEEE exceptions. Earlier Day 6/7 fixtures cover
rectangular products, identity, empty operands and structural rows/columns,
duplicate contributions, exact cancellation, `1e-15` threshold boundaries,
and repeated wide-row ordering. Day 8 extends their consumer checks too.

## Correctness Ledger

| Surface | Evidence | Discrepancy |
| --- | --- | --- |
| Tall/wide, sparse/dense seeded shapes | Four fixed-seed masks and independent dense values. | None observed. |
| Empty intermediate and output structures | Tall sparse fixture checks row and column absence in both link directions and compressed pointers. | None observed. |
| Retained-value consumers and unchanged inputs | `sparse_get`, matvec, CSR, CSC, and source rereads for every finite oracle fixture. | None observed. |
| Finite large and non-finite controls | `1e300` remains finite; NaN accumulator drops, infinity retains, following the frozen Day 1 predicate. | None observed. |
| Cancellation, duplicates, and ordering | Day 7 explicit support masks and three-run row/column traversal. | None observed. |

No correctness discrepancy is being traded for speed. Allocation-site
failure expansion remains assigned to Days 9-10; post-change performance
comparison remains assigned to Days 11-12.

## Verification

| Command | Result |
| --- | --- |
| `make matmul-allocation-failure-gate` | Exit 0; 37 tests, 0 failed/skipped, 29632 assertions. |
| `cmake --build build/s218-baseline --target test_matmul test_csr --parallel 2` | Exit 0; both focused targets built. |
| `ctest --test-dir build/s218-baseline -R '^(test_matmul\|test_csr)$' --output-on-failure` | Exit 0; 2/2 passed. |
| `make format-check` | Exit 0. |
| `make source-list-check` | Exit 0; 49 library sources. |
| `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. |
| `make test` | Exit 0; full Make suite ended with `All tests passed.` |
