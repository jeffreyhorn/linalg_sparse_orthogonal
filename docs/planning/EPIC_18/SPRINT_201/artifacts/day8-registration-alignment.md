# Sprint 201 Day 8 Registration Alignment

## Summary

Day 8 added a selected SVD helper registration guard and confirmed that the
extracted helper cluster remains tied to the registered `test_svd` proof-owner
binary.

The extraction did not create, remove, or rename a C source file. No CMake or
library source-list registration changes were needed for the helper headers
themselves. PR #223 review follow-up added explicit Makefile prerequisites for
both SVD helper headers on `build/test_svd` so helper-only edits rebuild the
proof-owner binary.

## Registration Decision

`tests/test_svd_selected_helpers.h` remains a proof-owner-only header included
by `tests/test_svd.c`. `tests/test_svd_helpers.h` remains shared fixture
support.

| Surface | Decision |
| --- | --- |
| Make test registration | Keep `$(TESTDIR)/test_svd.c` as the proof-owner binary. |
| CMake test registration | Keep `add_sparse_test(test_svd)` as the proof-owner binary. |
| Make helper prerequisites | Keep `tests/test_svd_helpers.h` and `tests/test_svd_selected_helpers.h` as explicit `build/test_svd` prerequisites. |
| Library source manifest | Do not list either SVD helper header; they are not library sources. |
| Standalone test target | Do not add standalone SVD helper tests; helpers stay included-only. |
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
- `tests/test_svd.c` includes `test_svd_helpers.h` and
  `test_svd_selected_helpers.h` exactly once;
- the selected moved helper implementations remain in
  `tests/test_svd_selected_helpers.h` and absent from `tests/test_svd.c` and
  `tests/test_svd_helpers.h`;
- the selected `RUN_TEST(...)` registrations remain in `tests/test_svd.c`
  exactly once and in frozen order;
- selected-helper `sparse_qr.h`, `sparse_svd.h`, and `sparse_vector.h`
  dependencies remain explicit;
- both SVD helper headers remain explicit Makefile prerequisites for
  `build/test_svd`;
- both SVD helper headers are absent from CMake registration and
  `build-metadata/library_sources.txt`;
- no standalone SVD helper CMake test is introduced accidentally.

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
