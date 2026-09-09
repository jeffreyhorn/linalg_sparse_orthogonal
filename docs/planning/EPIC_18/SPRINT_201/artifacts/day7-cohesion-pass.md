# Sprint 201 Day 7 Cohesion Pass

## Summary

Day 7 completed the second extraction pass for the selected SVD review-surface
cluster by moving the pseudoinverse and dense low-rank tests into
`tests/test_svd_helpers.h` while preserving the registered test wrappers in
`tests/test_svd.c`.

## Changed Files

- `tests/test_svd.c`
- `tests/test_svd_helpers.h`

No production source, public header, build registration, or test registration
order changed.

## Extracted Helper Groups

Day 6 established the rank-helper extraction boundary. Day 7 extended the same
pattern to the remaining selected core cluster:

- pseudoinverse tests;
- dense low-rank tests;
- helper-owned dependencies needed by those tests.

## Final Wrapper Map

Rank wrappers retained in `tests/test_svd.c`:

| Registered test wrapper | Helper-owned implementation |
| --- | --- |
| `test_svd_rank_full` | `tf_svd_test_rank_full` |
| `test_svd_rank_deficient` | `tf_svd_test_rank_deficient` |
| `test_svd_rank_nearly_singular` | `tf_svd_test_rank_nearly_singular` |
| `test_svd_rank_diagonal_threshold_fixture` | `tf_svd_test_rank_diagonal_threshold_fixture` |
| `test_svd_qr_rank_dependent_row_fixture` | `tf_svd_test_qr_rank_dependent_row_fixture` |
| `test_svd_rank_null` | `tf_svd_test_rank_null` |

Pseudoinverse wrappers moved on Day 7:

| Registered test wrapper | Helper-owned implementation |
| --- | --- |
| `test_pinv_diagonal` | `tf_svd_test_pinv_diagonal` |
| `test_pinv_moore_penrose` | `tf_svd_test_pinv_moore_penrose` |
| `test_pinv_null` | `tf_svd_test_pinv_null` |
| `test_pinv_rectangular` | `tf_svd_test_pinv_rectangular` |
| `test_pinv_underdetermined_minnorm_solution` | `tf_svd_test_pinv_underdetermined_minnorm_solution` |

Dense low-rank wrappers moved on Day 7:

| Registered test wrapper | Helper-owned implementation |
| --- | --- |
| `test_lowrank_diagonal` | `tf_svd_test_lowrank_diagonal` |
| `test_lowrank_error_bound` | `tf_svd_test_lowrank_error_bound` |
| `test_lowrank_errors` | `tf_svd_test_lowrank_errors` |

## Dependency Notes

`tests/test_svd_helpers.h` now owns the dependencies needed by the extracted
cluster:

- `sparse_qr.h` remains for the dependent-row SVD/QR rank cross-check moved on
  Day 6;
- `sparse_vector.h` was added for `vec_norm2(...)` in the underdetermined
  minimum-norm pseudoinverse fixture;
- `<stdio.h>` remains for moved diagnostic output.

No helper dependency leaked into production code or public installation
surfaces.

## Behavior-Preservation Notes

The Day 7 move preserved:

- diagonal pseudoinverse expected values;
- Moore-Penrose reconstruction residual checks;
- null-input error-code expectations;
- rectangular pseudoinverse fixture dimensions and tolerances;
- underdetermined minimum-norm solution checks and `vec_norm2(...)` norm
  comparison;
- dense low-rank reconstruction values, error-bound logic, and bad-argument
  assertions;
- diagnostic text;
- allocation cleanup and early-return behavior;
- `RUN_TEST(...)` names and order.

## Review-Surface Result

Post-format line counts:

| File | Lines |
| --- | ---: |
| `tests/test_svd.c` | 2657 |
| `tests/test_svd_helpers.h` | 677 |

Relative to the Day 6 post-format baseline, Day 7 removed 257 lines from
`tests/test_svd.c` and moved the selected test body ownership into the helper
header.

## Validation

Focused commands:

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

## Completion

Day 7 completes the core extraction pass for item 201.3. The selected SVD rank,
pseudoinverse, and dense low-rank cluster now has helper-owned implementations,
stable registered wrappers, focused SVD coverage, and full quality-gate
evidence.
