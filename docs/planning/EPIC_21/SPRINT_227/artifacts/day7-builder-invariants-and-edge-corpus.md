# Day 7 Builder Invariants and Edge Corpus

## Outcome

The sorted output builder added on Day 6 is now covered by a deterministic edge corpus, a fixed-seed randomized dense oracle, stronger cross-link identity checks, both supported index widths, and a focused sanitizer run. No production algorithm or public contract changed on Day 7.

## Edge corpus

The always-on arithmetic suite now exercises:

- two structurally empty inputs and an empty output;
- empty rows before and after populated rows;
- disjoint row patterns and overlapping columns;
- identical coordinates with complete cancellation;
- both scalar coefficients equal to zero and one scalar equal to zero;
- a rectangular 2x7 result;
- a 3x64 matrix with a dense first row, periodic cancellation, an empty middle row, and far-apart final-row entries;
- output copying, logical reads, structural validation, and destruction for every deterministic case; and
- 96 fixed-seed cases with dimensions from 1x1 through 6x6, sparse integer inputs, coefficients in `[-2, 2]`, and exact comparison with the independent dense add oracle after applying the frozen `1e-15` drop rule.

The fixed seed is `0x2270add7`. The generated values and coefficients are small integers, so expected arithmetic is exactly representable and does not hide accumulation-order differences behind a tolerance.

## Structural invariants

`s227_links_are_valid` now proves more than sorted coordinates and matching row/column counts. For every row-linked node, it requires that the identical pointer occur exactly once in the corresponding column chain; it applies the reciprocal check to every column-linked node. Together with strict coordinate ordering and exact `nnz` counts, this checks:

- ascending row and column chains;
- correct row and column coordinates;
- tail maintenance across empty and populated rows;
- node uniqueness and shared row/column node identity; and
- agreement between logical reads, row traversal, column traversal, and `nnz`.

The helper no longer limits map validation to six rows or columns, allowing the 64-column fixture to use the same structural proof as the compact oracle cases. Snapshot storage remains deliberately bounded to six by six.

## Size and ownership audit

| Concern | Audit result |
| --- | --- |
| Column-tail byte count | `sparse_calloc_idx_array(cols, sizeof(Node *), ...)` checks signed-index conversion and `size_t` multiplication before allocation. |
| Matrix shell arrays | `sparse_create_impl` routes all six dimension-derived arrays through checked index-array allocators and releases partial shell buffers on failure. |
| Row sequencing | `begin_row` accepts only `row == next_row`, requires bounds, and advances at most from `rows - 1` to `rows`; no signed increment exceeds `IDX_MAX`. |
| Column sequencing | `append` requires a valid column strictly greater than the prior row column and verifies that an existing column tail has a smaller row. |
| Nonzero count | `append` rejects `nnz == IDX_MAX` before allocating or incrementing. |
| Partial node ownership | Every fallible validation and count check occurs before node allocation. Once allocated, a node is immediately owned by the output pool and linked into both chains without another fallible operation. |
| Workspace ownership | The builder alone owns `col_tails`; finish frees it exactly once and transfers only the matrix, while abort frees it and the entire partial matrix. |
| Output transfer | `finish` requires all rows to have begun, clears the builder's matrix pointer, and marks the builder inactive after assigning the caller output. |

No unchecked dimension multiplication or ambiguous partial-node owner remains in the builder. Day 8 retains the planned exhaustive allocation-failure sweep.

## Evidence

| Command | Result |
| --- | --- |
| `make format` | Pass |
| Focused 32-bit CMake build and `test_sparse_arith` | 41 tests, 4,447 assertions, zero failures/skips |
| Clean CMake build with `-DSPARSE_IDX_BITS=64` and `test_sparse_arith` | 41 tests, 4,447 assertions, zero failures/skips |
| ASan/UBSan focused build and `test_sparse_arith` | Pass with no sanitizer finding; LeakSanitizer is unsupported by the local Apple runtime and was disabled |
| Strict `-Werror` syntax check of `tests/test_sparse_arith.c` | Pass |
| `make lint` | Pass: strict production compile, clang-tidy, and cppcheck (112/112) |
| `make test` | Pass: full runtime suite |
| `make docs-check` | Pass: all 18 checked-in public headers covered |
| `git diff --check` | Pass |

