# Sprint 201 Day 9 Ownership Guard

## Summary

Day 9 hardened the Sprint 201 SVD helper guard with fixture-based negative
tests. The guard now has direct evidence that selected ownership drift fails
clearly before the extracted rank, pseudoinverse, and dense low-rank cluster can
silently move back into `tests/test_svd.c` or become an unintended build/test
registration surface.

## Guard Rules

The selected SVD helper guard enforces these ownership rules:

| Rule | Guarded Surface |
| --- | --- |
| `tests/test_svd.c` remains the proof-owner binary. | Makefile `TEST_SRCS` and CMake `add_sparse_test(test_svd)`. |
| `tests/test_svd_helpers.h` remains included-only. | Helper is absent from Makefile, CMake, and `build-metadata/library_sources.txt`. |
| The helper header remains the selected implementation owner. | Moved `tf_svd_test_*` definitions must remain in `tests/test_svd_helpers.h` and absent from `tests/test_svd.c`. |
| Registered test names remain stable. | Selected `RUN_TEST(...)` markers must remain in `tests/test_svd.c` exactly once. |
| Helper dependencies stay explicit. | `sparse_svd.h` and `sparse_vector.h` must remain included by the helper. |

## Negative Coverage Added

Day 9 added `tests/test_svd_helper_guard.py`, following the existing QR helper
guard test pattern. The Python test creates temporary minimal repo fixtures and
checks that the shell guard fails clearly for representative drift cases:

- missing `#include "test_svd_helpers.h"` in `tests/test_svd.c`;
- missing `sparse_svd.h` helper dependency;
- missing `sparse_vector.h` helper dependency;
- moved selected helper definition appearing in `tests/test_svd.c`;
- missing selected `RUN_TEST(...)` proof-owner registration;
- missing Makefile registration for `test_svd.c`;
- missing CMake registration for `test_svd`;
- accidental Makefile registration of `test_svd_helpers.h`;
- accidental library-source manifest registration of `test_svd_helpers.h`.

The test also verifies the current tree and a clean minimal fixture both pass
the guard.

## Validation

New SVD guard regression:

```sh
python3 tests/test_svd_helper_guard.py
```

Result:

- Passed.

Focused guard command:

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

Adjacent guard regression:

```sh
python3 tests/test_qr_external_ref_helper_guard.py
```

Result:

- Passed.

## Scope Boundary

The guard wording stays selected-cluster scoped. It does not claim broad SVD
correctness, broad review-surface reduction across the repository, public API
change, CMake install behavior change, or performance improvement.

## Completion

Day 9 completes the drift-sensitive ownership-guard portion of item 201.4 for
the selected SVD helper cluster. Day 10 can now focus on behavior-preservation
evidence without needing additional registration-surface changes.
