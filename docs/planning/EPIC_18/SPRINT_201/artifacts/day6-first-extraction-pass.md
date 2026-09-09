# Sprint 201 Day 6: First Extraction Pass

## Summary

Day 6 implemented the first narrowly scoped extraction for the selected Sprint
201 SVD review-surface reduction. The rank subgroup moved from
`tests/test_svd.c` into `tests/test_svd_helpers.h` while preserving
`tests/test_svd.c` as the proof-owner binary.

Changed code files:

- `tests/test_svd.c`
- `tests/test_svd_helpers.h`

No production source, public header, Makefile, CMake, source-list, or CI files
were changed.

## Extraction Implemented

| Existing Test Wrapper | Helper-Owned Implementation |
| --- | --- |
| `test_svd_rank_full` | `tf_svd_test_rank_full` |
| `test_svd_rank_deficient` | `tf_svd_test_rank_deficient` |
| `test_svd_rank_nearly_singular` | `tf_svd_test_rank_nearly_singular` |
| `test_svd_rank_diagonal_threshold_fixture` | `tf_svd_test_rank_diagonal_threshold_fixture` |
| `test_svd_qr_rank_dependent_row_fixture` | `tf_svd_test_qr_rank_dependent_row_fixture` |
| `test_svd_rank_null` | `tf_svd_test_rank_null` |

The original `RUN_TEST(...)` registrations remain in `tests/test_svd.c` and
retain their existing order.

## Dependency And Visibility Changes

`tests/test_svd_helpers.h` now includes:

- `sparse_qr.h`, required by the moved SVD/QR rank cross-check helper;
- `<stdio.h>`, required by the moved `printf(...)` diagnostics.

All moved implementations use `static inline` visibility and the existing
`tf_svd_` helper naming family.

## Behavior Preservation

The extraction preserved:

- rank fixture dimensions and values;
- rank tolerances, including `0.0`, `1e-14`, `1e-12`, `1e-10`, and `1e-6`;
- duplicate-column and dependent-row fixture semantics;
- QR rank cross-check behavior;
- error-code expectations for null inputs;
- diagnostic `printf(...)` text;
- cleanup and early-return behavior;
- selected `RUN_TEST(...)` names and order.

## Review-Surface Result

Post-format line counts:

| File | Lines |
| --- | ---: |
| `tests/test_svd.c` | 2914 |
| `tests/test_svd_helpers.h` | 394 |

The first pass removed the rank implementation bodies from the large proof-owner
file and replaced them with thin wrappers. Pseudoinverse and dense low-rank
implementations remain for Day 7.

## Deviations From Day 5 Design

No material deviations.

The only dependency addition was expected by the Day 5 design:

- `sparse_qr.h` became necessary because the selected rank subgroup includes the
  dependent-row SVD/QR rank cross-check.

## Validation

Focused validation:

```sh
make build/test_svd
./build/test_svd
```

Focused result:

- `make build/test_svd` passed.
- `./build/test_svd` passed: 114 tests run, 0 failed, 0 skipped.

Required quality gate because `.c` and `.h` files changed:

```sh
make format
make lint
make test
```

Required gate result:

- `make format` passed.
- `make lint` passed.
- `make test` passed.

## Day 6 Completion Evidence

Day 6 completes the first extraction pass for item 201.3. The rank subgroup now
has helper ownership, local reachability through `test_svd`, and full quality
gate coverage.
