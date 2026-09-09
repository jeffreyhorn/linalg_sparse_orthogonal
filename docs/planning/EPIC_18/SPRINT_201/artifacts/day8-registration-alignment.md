# Sprint 201 Day 8 Registration Alignment

## Summary

Day 8 added a selected SVD helper registration guard and confirmed that the
extracted helper cluster remains tied to the registered `test_svd` proof-owner
binary.

The extraction did not create, remove, or rename a C source file. No Makefile,
CMake, or library source-list registration changes were needed for the helper
header itself. The only build-surface change is a focused Make target for the
new registration guard.

## Registration Decision

`tests/test_svd_helpers.h` remains a header-only test helper included by
`tests/test_svd.c`.

| Surface | Decision |
| --- | --- |
| Make test registration | Keep `$(TESTDIR)/test_svd.c` as the proof-owner binary. |
| CMake test registration | Keep `add_sparse_test(test_svd)` as the proof-owner binary. |
| Library source manifest | Do not list `tests/test_svd_helpers.h`; it is not a library source. |
| Standalone test target | Do not add `test_svd_helpers`; helper stays included-only. |
| Guard target | Add `make svd-helper-guard`. |

## Guard Added

Day 8 added `scripts/check_svd_helper_guard.sh` and wired it through:

```sh
make svd-helper-guard
```

The guard checks:

- required files exist;
- `test_svd.c` remains registered in the Makefile test source list;
- `test_svd` remains registered in CMake;
- `tests/test_svd.c` includes `test_svd_helpers.h` exactly once;
- the selected moved helper implementations remain in
  `tests/test_svd_helpers.h`;
- the selected `RUN_TEST(...)` registrations remain in `tests/test_svd.c`
  exactly once;
- `test_svd_helpers.h` is absent from Makefile source registration, CMake
  registration, and `build-metadata/library_sources.txt`;
- no standalone `test_svd_helpers` CMake test is introduced accidentally.

## Selected Cluster Guard Markers

The guard covers the selected rank, pseudoinverse, and dense low-rank helper
cluster:

- `tf_svd_test_rank_full`
- `tf_svd_test_rank_deficient`
- `tf_svd_test_rank_nearly_singular`
- `tf_svd_test_rank_diagonal_threshold_fixture`
- `tf_svd_test_qr_rank_dependent_row_fixture`
- `tf_svd_test_rank_null`
- `tf_svd_test_pinv_diagonal`
- `tf_svd_test_pinv_moore_penrose`
- `tf_svd_test_pinv_null`
- `tf_svd_test_pinv_rectangular`
- `tf_svd_test_pinv_underdetermined_minnorm_solution`
- `tf_svd_test_lowrank_diagonal`
- `tf_svd_test_lowrank_error_bound`
- `tf_svd_test_lowrank_errors`

## Validation

Registration guard:

```sh
make svd-helper-guard
```

Result:

```text
svd-helper-guard: required files ok
svd-helper-guard: proof-owner registration ok
svd-helper-guard: helper boundary ok
svd-helper-guard: selected cluster ownership ok
svd-helper-guard: header-only registration ok
svd-helper-guard: passed
```

Source-list check:

```sh
make source-list-check
```

Result:

```text
source-list-check: PASS (49 library sources)
```

CMake registration configure check:

```sh
cmake -S . -B build/sprint201-day8-registration-check
```

Result:

```text
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/jeff/experiments/linalg_sparse_orthogonal/build/sprint201-day8-registration-check
```

Generated-artifact hygiene:

```sh
git status --short --ignored build/sprint201-day8-registration-check
```

Result:

```text
!! build/
```

The CMake check wrote only under ignored `build/`; no generated build artifacts
were staged.

## Residual Registration Risks

No residual build-registration risk is known for the selected SVD helper
cluster. The helper remains header-only by design, `test_svd.c` remains the
proof-owner binary, and the new guard fails clearly if the selected helper
implementations, proof-owner registrations, or header-only boundary drift.

Day 9 can build on this by documenting the ownership semantics and adding any
extra drift-sensitive guard coverage that proves useful without broadening the
selected-cluster scope.
