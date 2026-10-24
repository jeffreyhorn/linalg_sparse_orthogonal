# Day 7: Cancellation And Duplicate Contributions

**State:** The Day 7 finite-arithmetic and ordering regressions are registered
in `tests/test_matmul.c`. The Day 1 `fabs(value) >= 1e-15` output predicate is
unchanged.

## Oracle And Support Fixtures

The small dense oracle multiplies caller-provided arrays independently of
the sparse output builder. Day 7 supplies a separate expected support mask;
the test checks `nnz`, every present and absent coordinate, sorted row and
column chains, and shared row/column node identity. The value comparison is
separate from the support decision.

| Fixture | Expected result and support | Value tolerance |
| --- | --- | --- |
| Two contributions per coordinate in a 1x2 times 2x6 product | Column 0 sums to 5 and is present once; column 1 cancels exactly; columns 2 and 4 have magnitudes `2^-50 < 1e-15` and drop; columns 3 and 5 have magnitudes `2^-49 > 1e-15` and remain, including the negative residual. Support is `{0,3,5}`. | Zero: all products and residuals are exact binary values. |
| One contribution at each side of the threshold | `nextafter(1e-15, 0)` drops; `1e-15` and `nextafter(1e-15, +inf)` remain. Support is `{1,2}`. | Zero: multiplication by one leaves each represented value unchanged. |
| Three fractional contributions to one output entry | `0.1 + 0.2 + 0.3` produces one entry. | `8 * DBL_EPSILON` absolute; this permits reassociation of three small rounded decimal values without relaxing support. |

The wide ordering case multiplies a 3x257 matrix with a reverse-column
permutation matrix. Each of three independent runs must produce exactly 771
entries. Every row chain is checked from column 0 through 256 and every
column chain from row 0 through 2, with exact integer values and no extra
nodes. This verifies deterministic structure and values by traversal rather
than random lookups.

## Focused Verification

| Command | Result |
| --- | --- |
| `make matmul-allocation-failure-gate` | Exit 0; 31 tests, 0 failed/skipped, 27275 assertions. |
| `cmake --build build/s218-baseline --target test_matmul --parallel 2` | Exit 0; focused target rebuilt. |
| `ctest --test-dir build/s218-baseline -R '^test_matmul$' --output-on-failure` | Exit 0; 1/1 passed. |
| `make format-check` | Exit 0. |
| `make source-list-check` | Exit 0; 49 library sources. |
| `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. |
| `make test` | Exit 0; full Make suite ended with `All tests passed.` |

These fixtures freeze finite accumulation, threshold, and ordered output
behavior; they do not assess non-finite arithmetic or performance. Day 8
continues structural and edge-case coverage, while Days 11-12 compare
post-change timings against the Day 2 baseline.
