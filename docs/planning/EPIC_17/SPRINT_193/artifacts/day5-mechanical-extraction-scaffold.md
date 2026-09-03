# Sprint 193 Day 5: Mechanical Extraction Scaffold

## Summary

Day 5 added the QR external-reference helper scaffold and proved the existing
`test_qr` proof-owner binary still builds and passes. No selected test bodies
or helper implementations moved yet.

## Changes

| File | Change |
| --- | --- |
| `tests/test_qr_external_ref_helpers.h` | New header-only scaffold with `TEST_QR_EXTERNAL_REF_HELPERS_H`. |
| `tests/test_qr.c` | Includes `test_qr_external_ref_helpers.h` after `test_solver_helpers.h`. |

## Behavior Movement

None. This day intentionally moved no tests, fixtures, readers, assertions,
diagnostics, cleanup paths, or registration entries.

## Registration and Source Lists

| Surface | Result |
| --- | --- |
| `Makefile` `TEST_SRCS` | Unchanged; `test_qr.c` remains registered. |
| `CMakeLists.txt` | Unchanged; `add_sparse_test(test_qr)` remains registered. |
| `build-metadata/library_sources.txt` | Unchanged. |
| Library sources | Unchanged. |
| Test binaries | No new binary added. |

## Focused Validation

Command:

```sh
make build/test_qr && ./build/test_qr
```

Result:

| Metric | Value |
| --- | --- |
| Build | passed |
| Tests run | 77 |
| Failures | 0 |
| Skips | 0 |
| Assertions | 960 |
| Runtime | 4.384 s |

## Review-Surface State

| File | Lines |
| --- | ---: |
| `tests/test_qr.c` | 3971 |
| `tests/test_qr_external_ref_helpers.h` | 9 |

The scaffold creates the target boundary. Actual review-surface reduction
begins on Day 6 when the selected reader helpers and external-reference test
bodies move.

## Day 6 Handoff

Move the selected QR external-reference logic in this order:

1. Reader helpers.
2. Selected nullspace/projector tests.
3. `make_rankdef_wide_3x5` with the wide nullspace test.
4. Selected rank-threshold tests.

Keep `main`, all `RUN_TEST(...)` entries, and
`test_qr_external_dense_reference_economy_projector_5x3` in `tests/test_qr.c`.

## Validation

Commands run:

```sh
git status --short --branch
sed -n '1,40p' tests/test_qr.c
sed -n '620,760p' docs/planning/EPIC_17/SPRINT_193/WORKING_NOTES.md
sed -n '183,236p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
make build/test_qr && ./build/test_qr
wc -l tests/test_qr.c tests/test_qr_external_ref_helpers.h
git diff -- tests/test_qr.c tests/test_qr_external_ref_helpers.h
```

Day 5 changed `.c` and `.h` files. Focused scaffold validation passed. The
full C quality gate remains required before Sprint 193 closeout.
