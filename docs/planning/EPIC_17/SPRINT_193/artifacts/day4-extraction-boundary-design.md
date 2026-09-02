# Sprint 193 Day 4: Extraction Boundary Design

## Decision

Sprint 193 will use a header-only QR helper boundary:

`tests/test_qr_external_ref_helpers.h`

The helper will own selected external dense-reference rank/nullspace/threshold
test bodies and their local reader helpers. `tests/test_qr.c` remains the
registered proof-owner binary and keeps `main` plus every `RUN_TEST(...)`
entry.

## Boundary Map

| Stays in `tests/test_qr.c` | Moves to `tests/test_qr_external_ref_helpers.h` |
| --- | --- |
| `_POSIX_C_SOURCE` feature-test macro block. | `read_qr_basis_external_reference`. |
| Production/test includes and `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`. | `read_qr_threshold_external_reference`. |
| `main` and all `RUN_TEST(...)` registration. | Selected external nullspace/projector tests. |
| General QR, economy, sparse-mode, reorder, and refinement tests. | Selected external rank-threshold tests. |
| `test_qr_external_dense_reference_economy_projector_5x3`. | `make_rankdef_wide_3x5`, which supports only the selected wide nullspace test. |

## Include Design

Planned include order in `tests/test_qr.c`:

```c
#include "test_qr_helpers.h"
#define TF_ENABLE_EXTERNAL_REFERENCE_HELPER
#include "test_solver_helpers.h"
#include "test_qr_external_ref_helpers.h"
```

The selected helper header should use:

```c
#ifndef TEST_QR_EXTERNAL_REF_HELPERS_H
#define TEST_QR_EXTERNAL_REF_HELPERS_H
...
#endif
```

The header should not become a standalone compilation unit. Its static helper
and test bodies are compiled as part of `tests/test_qr.c`, preserving the
existing proof-owner model.

## Cleanup and State Contract

The movement must preserve:

- `TF_EXTERNAL_REFERENCE_SKIP` and `SKIP_TEST(reason)` behavior;
- external-reference failure diagnostics and early returns;
- command-overflow diagnostics;
- every `sparse_qr_free`, `sparse_free`, and `free` cleanup path;
- `tf_qr_insert_or_free` cleanup semantics;
- Windows skip branches and messages;
- fixture keys and Python command string;
- rank, nullity, threshold, perturbation, projector, residual, and
  orthogonality expectations.

No selected test mutates process-global overrides or environment variables.
Day 7 should still re-audit cleanup after movement.

## Guard Design

Add a focused guard after implementation:

| Guard surface | Expected check |
| --- | --- |
| Script | `scripts/check_qr_external_ref_helper_guard.sh`. |
| Make target | `qr-external-ref-helper-guard`. |
| Proof owner | `tests/test_qr.c` exists and remains in `Makefile` `TEST_SRCS`. |
| CMake owner | `CMakeLists.txt` still has `add_sparse_test(test_qr)`. |
| Helper ownership | `tests/test_qr.c` includes `test_qr_external_ref_helpers.h` exactly once. |
| Header-only boundary | Helper header is absent from Make/CMake/library source registration and no standalone CMake test is added. |
| Registration preservation | Selected `RUN_TEST(...)` entries remain in `tests/test_qr.c`. |
| Scope boundary | Moved selected test definitions are absent from `tests/test_qr.c`; economy external-reference test remains there. |

## Source-List Decision

No Make/CMake/library source-list changes are planned for the extraction
itself. The only expected Makefile addition is the future guard target. If Day
5 or later requires a compiled source or new proof-owner binary, Sprint 193
must pause and update the invariant contract before proceeding.

## Review Checkpoints

1. Add and compile the empty helper scaffold first.
2. Move reader helpers before selected test bodies.
3. Move nullspace/projector tests before threshold tests.
4. Run focused `test_qr` validation after meaningful movement steps.
5. Add the guard only after the final helper boundary is stable.
6. Run full `make format && make lint && make test` after C/H changes.

## Validation

Commands run:

```sh
git status --short --branch
sed -n '430,620p' docs/planning/EPIC_17/SPRINT_193/WORKING_NOTES.md
sed -n '135,198p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_193/artifacts/day3-selected-cluster-contract.md
rg -n "tf_qr_make_|tf_qr_insert_or_free|vec_norm2|sparse_qr_|sparse_matvec|sparse_create|sparse_insert|sparse_free|ASSERT_|REQUIRE_|SKIP_TEST|TF_FAIL_|tf_read_external_reference_vector|snprintf|strcmp|sqrt|fabs" tests/test_qr.c | sed -n '1,180p'
sed -n '1,90p' tests/test_ldlt_csc.c
sed -n '619,626p' Makefile
rg -n "test_qr_helpers|test_qr_external|test_qr.c|test_qr\)" Makefile CMakeLists.txt build-metadata/library_sources.txt docs/maintainer_guide.md tests/*.h tests/*.c | head -n 120
```

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
