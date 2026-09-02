# Sprint 193 Day 3: Selected Cluster and Invariant Contract

## Decision

Sprint 193 selects exactly one implementation target:

`tests/test_qr.c` external dense-reference rank/nullspace/threshold block.

The selected cluster is a test-only review-surface reduction. The planned
implementation path is a family-local helper header, not a new test binary and
not a production source extraction.

## Selection Rationale

| Criterion | Evidence |
| --- | --- |
| Review-surface payoff | `tests/test_qr.c` is the largest remaining C test file at 3970 lines. |
| Cohesive ownership | The selected functions share QR external dense-reference fixture keys, Python reference execution, Windows skip semantics, nullspace projector checks, and rank-threshold checks. |
| Behavior-preservation confidence | The selected tests already run through the existing `test_qr` proof-owner binary and can keep their names, ordering, fixtures, tolerances, diagnostics, and cleanup behavior. |
| Low registration risk | Header-only extraction can preserve existing Make and CMake test registration. |
| Guard fit | The Sprint 185 helper guard pattern can be adapted for QR helper-header ownership and registration boundaries. |

## Selected Symbols

| Symbol | Current owner | Day 4 action |
| --- | --- | --- |
| `read_qr_basis_external_reference` | `tests/test_qr.c` | Move to selected helper header if include dependencies are clean. |
| `read_qr_threshold_external_reference` | `tests/test_qr.c` | Move to selected helper header if include dependencies are clean. |
| `test_qr_external_dense_reference_rank1_4x3_nullspace_projector` | `tests/test_qr.c` | Move body to helper header. |
| `test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector` | `tests/test_qr.c` | Move body to helper header. |
| `test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector` | `tests/test_qr.c` | Move body to helper header. |
| `make_rankdef_wide_3x5` | `tests/test_qr.c` | Move with the wide nullspace test. |
| `test_qr_external_dense_reference_rankdef_wide_3x5_nullspace_subspace` | `tests/test_qr.c` | Move body to helper header. |
| `test_qr_external_dense_reference_rank_threshold_diag4_family` | `tests/test_qr.c` | Move body to helper header. |
| `test_qr_external_dense_reference_rank_threshold_diag4_scaled_family` | `tests/test_qr.c` | Move body to helper header. |
| `test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family` | `tests/test_qr.c` | Move body to helper header. |
| `test_qr_external_dense_reference_rank_threshold_dependent_row_4x3_perturbed_family` | `tests/test_qr.c` | Move body to helper header. |

## Explicitly Out Of Scope

- `test_qr_external_dense_reference_economy_projector_5x3` remains in
  `tests/test_qr.c`.
- General QR, economy-mode, sparse-mode, reorder, and refinement tests remain
  in `tests/test_qr.c`.
- `src/sparse_qr.c`, `include/sparse_qr.h`, and public QR APIs are not selected
  for Sprint 193.
- No new QR proof-owner binary is planned.
- No new production `.c` source or library source-list entry is planned.

## No-Behavior-Change Contract

The implementation must preserve:

- every selected test function name;
- every `RUN_TEST(...)` entry and its order in `tests/test_qr.c`;
- fixture keys passed to `tests/qr_external_dense_reference.py`;
- the command string `python3 tests/qr_external_dense_reference.py %s`;
- Windows skip branches and skip messages;
- expected rank, nullity, dimension, threshold, perturbation, projector,
  residual, and orthogonality values;
- `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` behavior;
- `_POSIX_C_SOURCE` feature-test macro behavior;
- `ASSERT_*`, `REQUIRE_*`, `TF_FAIL_`, and `printf` diagnostic wording;
- cleanup order for `sparse_qr_free`, `sparse_free`, and `free`;
- Makefile `$(TESTDIR)/test_qr.c` registration;
- CMake `add_sparse_test(test_qr)` registration;
- absence of QR helper headers from `build-metadata/library_sources.txt`.

## Planned Boundary

| Stays in `tests/test_qr.c` | Moves to selected helper header |
| --- | --- |
| Feature-test macro block and main include order. | External basis and threshold reader helpers. |
| Existing `main` and `RUN_TEST(...)` registration. | Selected external rank/nullspace projector test bodies. |
| Non-selected economy external-reference test body. | Selected external rank-threshold test bodies. |
| General QR/economy/sparse/reorder/refinement tests. | Wide rank-deficient fixture helper used only by selected wide nullspace test. |

## Acceptance Criteria

- Day 4 records a concrete helper-header name, include contract, and guard
  design before code movement.
- Day 5 adds a compile-clean scaffold before large test bodies move.
- The final extraction reduces `tests/test_qr.c` review surface while keeping
  `test_qr` as the proof owner.
- Focused validation runs `make build/test_qr` and `./build/test_qr`.
- Because `.c`/`.h` files will change after Day 4, final validation must
  include `make format && make lint && make test`.

## Validation

Commands run:

```sh
git status --short --branch
sed -n '260,430p' docs/planning/EPIC_17/SPRINT_193/WORKING_NOTES.md
sed -n '91,151p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
rg -n "test_qr.c|test_qr\)|test_qr_helpers|qr_external" Makefile CMakeLists.txt tests/test_qr.c tests/test_qr_helpers.h tests/qr_external_dense_reference.py
sed -n '1,260p' tests/qr_external_dense_reference.py
sed -n '2460,2565p' tests/test_qr.c
rg -n "read_qr_basis_external_reference|read_qr_threshold_external_reference|make_rankdef_wide_3x5|test_qr_external_dense_reference" tests/test_qr.c
sed -n '3896,3925p' tests/test_qr.c
```

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
