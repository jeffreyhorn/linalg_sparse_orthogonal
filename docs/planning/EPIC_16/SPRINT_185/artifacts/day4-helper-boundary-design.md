# Sprint 185 Day 4: Helper Boundary Design

## Purpose

Define the extracted files and helper ownership for the selected Sprint 185
cluster before any code movement begins.

## Selected Cluster

`tests/test_ldlt_csc.c`

The extraction design keeps the existing `test_ldlt_csc` binary as the proof
owner. The default plan is header-only helper extraction, with no new Makefile,
CMake, source-list, or production-code registration.

## Existing Helper Pattern Reviewed

| Existing helper | Relevant pattern |
| --- | --- |
| `tests/test_qr_helpers.h` | Include-guarded test helper header with family-prefixed helper names and static inline helpers. |
| `tests/test_svd_helpers.h` | Matrix fixture builders and dense numeric helpers live in a self-contained family helper header. |
| `tests/test_direct_solver_helpers.h` | Shared direct-solver assertions remain narrow and reusable. |
| `tests/test_chol_csc_supernodal_helpers.h` | Closest precedent: family-local CSC helper header with structural comparison, residual, and supernodal helpers. |

## Proposed Files

| File | Scope | First planned day | Registration impact |
| --- | --- | --- | --- |
| `tests/test_ldlt_csc_supernode_helpers.h` | Supernode detection, extract/writeback, dense panel indexing, snapshot, and supernodal factor-state comparison helpers. | Day 6 | Header-only include; no Make/CMake registration. |
| `tests/test_ldlt_csc_fixtures.h` | KKT, scaled-KKT, random-indefinite, external-reference fixture state, and two-pass factor setup helpers. | Day 7 candidate | Header-only include; no Make/CMake registration. |
| `tests/test_ldlt_csc_oracle_helpers.h` | Dense lower-triangle oracle helpers, symmetric-swap dense oracle, triple-builder fixtures, native-wrapper comparison helpers. | Day 8 candidate | Header-only include; no Make/CMake registration. |

Only `tests/test_ldlt_csc_supernode_helpers.h` is approved for the first
mechanical pass. The fixture and oracle headers are designed seams that should
be revisited after first-pass validation.

## First-Pass Helper Boundary

Day 6 should start with the lowest-risk helper group:

- `build_dense_ldlt_with_pivots`
- `cm_idx`
- `snapshot_supernode_state`
- `ldlt_csc_factor_state_matches`

These helpers support local supernode tests without changing `RUN_TEST(...)`
ownership. If dependency order is still straightforward during implementation,
Day 6 may also move `build_dense_spd`; otherwise it should remain local until
the fixture-header pass.

## Deferred Helper Boundaries

| Boundary | Deferred reason |
| --- | --- |
| External dense-reference helpers | They depend on `_POSIX_C_SOURCE`, `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`, Windows skip behavior, `popen`/Python command semantics, and fixture state cleanup. Move only after the first header extraction validates cleanly. |
| Symmetric-swap dense oracle helpers | They are cohesive but support a later proof section; moving them first would touch a larger span than needed. |
| Native wrapper comparison helpers | They rely on process-global kernel override behavior and should move only with a focused validation checkpoint. |
| Production LDLT CSC sources | Source extraction is out of Day 4 scope and remains deferred for Sprint 185 unless a later artifact explicitly changes direction. |

## Include And Ownership Model

- Keep helper headers self-contained with include guards.
- Include `sparse_chol_csc_internal.h`, `sparse_ldlt.h`,
  `sparse_ldlt_csc_internal.h`, `sparse_matrix.h`, `sparse_reorder.h`,
  `sparse_types.h`, `test_framework.h`, and C library headers only when the
  moved helpers need them.
- Include new helper headers from `tests/test_ldlt_csc.c` after
  `test_solver_helpers.h`.
- Preserve `_POSIX_C_SOURCE` placement before system headers.
- Preserve `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` before
  `test_solver_helpers.h`.
- Prefer preserving current helper names during movement; add a prefix only
  when Day 8 cleanup shows a real ambiguity.
- Keep moved helpers `static` so inclusion in the original test translation
  unit preserves internal linkage.

## Must Remain In `tests/test_ldlt_csc.c`

- `main` and every existing `RUN_TEST(...)` call.
- Test body functions unless a later proof-owner split is explicitly approved.
- Chronological proof-owner section comments that make the file's solver
  history readable.
- Test-specific assertion thresholds, random seeds, fixture constants, and
  external-reference skip messages.
- Global setup macros, especially `_POSIX_C_SOURCE` and
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`.

## First-Pass Validation

After Day 6 code movement:

```sh
make build/test_ldlt_csc && ./build/test_ldlt_csc
make format && make lint && make test
```

If Day 6 unexpectedly adds a new test binary, also run:

```sh
make quality-review-cmake-compile
```

If Day 6 unexpectedly adds a library source file, also run:

```sh
make source-list-check
```

## Day 5 Handoff

Day 5 should confirm that the planned Day 6 header-only extraction needs no
Make/CMake/source-list update, identify which existing guard catches the new
header in format/lint/test workflows, and record rollback criteria if the
header extraction exposes dependency or ordering risk.

## Validation

Day 4 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.
