# Day 3 Permutation and Oracle Fixtures

## Outcome

Day 3 makes the frozen original-state arithmetic contract executable without changing production behavior. The normal arithmetic suite now enforces fixture construction, forward/inverse permutation consistency, independent dense arithmetic, logical snapshots, node counts, sorted row/column links, and mutation detection. A temporary experimental regression enforces the pending rejection contract across all three APIs and intentionally fails against the Day 1 implementation.

The reusable helper is `tests/arithmetic_contract_fixture.h`. It is test-only and does not add a public or library symbol.

## Corpus

The checked-in [regression corpus](day3-permutation-regression-corpus.tsv) contains 30 rows:

- 27 unsupported-state rejection rows: three APIs × left/right/both operand positions × row-only/column-only/row-plus-column maps;
- three already-conforming identity-state rows tied to independent dense arithmetic;
- rectangular `2x3` addition and `2x3 * 3x2` multiplication fixtures;
- exact expected status, output/destination state, and unchanged-state checks for every rejection.

Permutation constructors accept logical-to-physical arrays and derive the inverse map from that array. They reject out-of-range and duplicate physical indices, preventing the fixture itself from creating a corrupt map.

## Independent checks

The helper computes expected addition and multiplication with direct dense loops. It does not call `sparse_add`, `sparse_add_inplace`, or `sparse_matmul`. Its structural validator independently walks both linked dimensions and checks:

- strictly increasing physical columns in every row;
- strictly increasing physical rows in every column;
- node coordinates within matrix bounds;
- row-walk count, column-walk count, and stored `nnz` agreement;
- forward/inverse row and column map agreement.

Snapshots retain logical values, all four maps, `nnz`, cached norm, factor norm, and the compatibility factored flag. Rejection tests therefore detect output allocation, value mutation, map mutation, count changes, cache/factor metadata changes, and broken row/column structure.

## Test-first evidence

Dedicated tree: `build/s227-day1-cmake`, configured from base commit `fcdeaaee856821e8c801d805c63b5fd40eb89b2f` and rebuilt with the Day 3 test changes.

| Command | Result | Meaning |
| --- | --- | --- |
| `cmake --build build/s227-day1-cmake --target test_sparse_arith --parallel 4` | Pass | The focused test binary compiles with the reusable fixture layer. |
| `build/s227-day1-cmake/test_sparse_arith` | Pass: 26 tests, 1 pending-contract skip, 152 assertions | All helper/oracle self-tests and existing conforming arithmetic cases remain green. |
| `SPARSE_TEST_EXPERIMENTAL=1 build/s227-day1-cmake/test_sparse_arith` | Expected failure: 1 aggregate test, 54 failed assertions among 416 total assertions | Every one of the 27 unsupported-state calls returns `SPARSE_OK` instead of `SPARSE_ERR_BADARG`; all 18 out-of-place calls allocate output, and all nine in-place calls mutate `A`. |
| `make format && make lint && make test` | Pass | Formatting, strict compilation, clang-tidy, cppcheck, and the complete default runtime suite pass; the pending contract remains excluded only by its explicit experimental wrapper. |
| `make docs-check` | Pass | Doxygen generation and all 18 checked-in public-header coverage checks pass. |

The experimental test is intentionally opt-in until the shared production guard lands. Day 4/5 must promote it to the normal suite after all assertions pass; it is not an accepted permanent skip.

## Boundary

This artifact establishes fixture validity and the exact pre-repair failures. It does not claim that the production contract is implemented, that factored/inverse-only/alias/failure-injection coverage is complete, or that arithmetic performance changed.
