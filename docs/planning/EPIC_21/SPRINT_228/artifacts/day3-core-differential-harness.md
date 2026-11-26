# Sprint 228 Day 3: Core Differential Harness

## Outcome

Day 3 adds a maintained differential harness in `tests/test_svd.c` for the two explicit sparse low-rank reconstruction routes:

- `SPARSE_SVD_LOWRANK_OUTER=off`: dense `m*n` compatibility accumulator;
- `SPARSE_SVD_LOWRANK_OUTER=on`: per-cell outer-product reconstruction.

Each route is evaluated independently against an analytical diagonal reconstruction oracle. The test does not use either production route as the expected result for the other. The ordinary default remains unchanged; these tests select only the two explicit compatibility values.

## Deterministic replay matrix

The three Day 2 core fixtures supply exact singular values and canonical singular vectors. Each is replayed at rank one, the pinned intermediate rank, and the maximum accepted rank.

| Replay ID | Shape | Diagonal data | Requested rank | Drop tolerance | Expected `nnz` |
| --- | --- | --- | ---: | ---: | ---: |
| `diag-square-32-r1` | 32×32 | `10,5,1,0.1,0,...` | 1 | `1e-8` | 1 |
| `diag-square-32-r4` | 32×32 | `10,5,1,0.1,0,...` | 4 | `1e-8` | 4 |
| `diag-square-32-r32` | 32×32 | `10,5,1,0.1,0,...` | 32 | `1e-8` | 4 |
| `diag-tall-9x5-r1` | 9×5 | `8,4,2,1,0` | 1 | `0` | 1 |
| `diag-tall-9x5-r3` | 9×5 | `8,4,2,1,0` | 3 | `0` | 3 |
| `diag-tall-9x5-r5` | 9×5 | `8,4,2,1,0` | 5 | `0` | 4 |
| `diag-wide-5x9-r1` | 5×9 | `8,4,2,1,0` | 1 | `1e-12` | 1 |
| `diag-wide-5x9-r3` | 5×9 | `8,4,2,1,0` | 3 | `1e-12` | 3 |
| `diag-wide-5x9-r5` | 5×9 | `8,4,2,1,0` | 5 | `1e-12` | 4 |

For each replay ID, the harness executes `off` and `on`, yielding 18 independently judged route cells. The zero terminal singular value in the rectangular maximum-rank cases also proves that a maximum accepted rank does not create an explicit zero.

## Assertions

Every route cell checks:

1. `SPARSE_OK` and a non-null output;
2. exact logical dimensions and expected `nnz`;
3. every logical value against the analytical diagonal oracle with a cell bound of `64 * DBL_EPSILON * sigma_0`;
4. relative Frobenius error no greater than `1e-12`;
5. finite, nonzero stored values;
6. strictly increasing column indices in every row and row indices in every column;
7. exactly one column-chain reference for every row-chain node, excluding duplicate or disconnected entries;
8. identity row/column permutations, fresh factor metadata, and consistent row/column counts;
9. unchanged input `nnz` and logical values after each route;
10. restoration of the caller's prior selector environment after each invocation.

The internal node walk is intentionally test-only. It validates sparse representation invariants that the public logical getter alone cannot distinguish.

## Observed results

All 18 route cells returned `SPARSE_OK`. Every cell reported:

- `max_abs = 0.000e+00`;
- `rel_frob = 0.000e+00`;
- the exact expected `nnz` from the replay matrix.

The focused SVD binary completed 117 tests and 15,669 assertions with zero failures or skips. Existing low-rank, SVD, condition-estimation, and corpus tests remain in the same executable.

## Verification

| Command | Result |
| --- | --- |
| `make format` | Passed |
| `make build/test_svd && ./build/test_svd` | Passed: 117 tests, 15,669 assertions, 0 failures, 0 skips |
| `make lint` | Passed: strict compile, clang-tidy, and cppcheck |
| `make test` | Passed: all maintained test binaries |

Day 3 changes test and planning evidence only. The production default, selector parser, reconstruction implementation, and public API are unchanged.
