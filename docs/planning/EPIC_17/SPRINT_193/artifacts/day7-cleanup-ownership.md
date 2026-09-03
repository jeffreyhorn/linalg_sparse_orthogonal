# Sprint 193 Day 7: Cleanup and Error-Path Ownership

## Summary

Day 7 audited cleanup, early-return, and process-global state behavior in the
moved QR external-reference helper boundary. No behavior cleanup was required.
The only code change was updating the helper header's ownership comment so it
matches the post-movement state.

## Audit Findings

| Area | Finding |
| --- | --- |
| `REQUIRE_*` macros | None in `tests/test_qr_external_ref_helpers.h`; no moved selected test can bypass cleanup through a `REQUIRE_*` early return. |
| External-reference skips | Existing `TF_EXTERNAL_REFERENCE_SKIP` and `SKIP_TEST(reason)` behavior remains unchanged. |
| External-reference failures | Existing `TF_FAIL_` diagnostics and immediate returns remain unchanged. |
| Reader argument failures | Invalid arguments, unsupported fixture keys, and command overflow return structured error statuses. |
| Matrix allocation/insert failures | Existing `ASSERT_NOT_NULL` checks and `tf_qr_insert_or_free` cleanup semantics remain unchanged. |
| QR ownership | Failure paths after QR ownership is established still call `sparse_qr_free(&qr)` and `sparse_free(A)` before returning. |
| Heap ownership | Stack buffers dominate the moved block; existing heap paths retain their `free` ordering. |
| Environment state | No selected moved test uses `tf_setenv`, `tf_unsetenv`, `setenv`, or `unsetenv`. |
| Process-global overrides | No selected moved test mutates kernel overrides or global registration state. |

## Code Cleanup

| File | Change |
| --- | --- |
| `tests/test_qr_external_ref_helpers.h` | Replaced the stale scaffold comment with a current boundary comment describing selected rank/nullspace/threshold dense-reference ownership and retained `tests/test_qr.c` proof-owner registration. |

## Focused Validation

The first focused command confirmed the Makefile header-dependency caveat:

```text
make: `build/test_qr' is up to date.
```

Day 7 then forced a rebuild of only the focused proof-owner binary:

```sh
find build -maxdepth 1 -type f -name test_qr -delete && make build/test_qr && ./build/test_qr
```

Result:

| Metric | Value |
| --- | --- |
| Build | passed after forced rebuild |
| Tests run | 77 |
| Failures | 0 |
| Skips | 0 |
| Assertions | 960 |
| Runtime | 4.995 s |

## Day 8 Handoff

Day 8 should add the QR external-reference helper guard and encode these
ownership rules:

- helper header exists and has `TEST_QR_EXTERNAL_REF_HELPERS_H`;
- `tests/test_qr.c` includes it exactly once;
- selected moved test definitions remain outside `tests/test_qr.c`;
- selected `RUN_TEST(...)` registrations remain inside `tests/test_qr.c`;
- the economy external-reference test body remains in `tests/test_qr.c`;
- `test_qr` remains registered in Make and CMake;
- helper header remains absent from standalone Make/CMake/library source
  registration.

## Validation

Commands run:

```sh
git status --short --branch
sed -n '236,298p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
sed -n '760,940p' docs/planning/EPIC_17/SPRINT_193/WORKING_NOTES.md
sed -n '1,220p' tests/test_qr_external_ref_helpers.h
rg -n "return;|return NULL|return TF_|SKIP_TEST|TF_FAIL_|ASSERT_|REQUIRE_|sparse_qr_free|sparse_free|free\(|tf_setenv|tf_unsetenv|setenv|unsetenv|override|kernel|global" tests/test_qr_external_ref_helpers.h
rg -n "static void test_qr_external_dense_reference|static int read_qr_|make_rankdef_wide_3x5|#ifdef _WIN32|#endif" tests/test_qr_external_ref_helpers.h
tail -n 80 tests/test_qr_external_ref_helpers.h
rg -n "REQUIRE_|tf_setenv|tf_unsetenv|setenv|unsetenv|override|kernel" tests/test_qr_external_ref_helpers.h tests/test_qr.c
make build/test_qr && ./build/test_qr
find build -maxdepth 1 -type f -name test_qr -delete && make build/test_qr && ./build/test_qr
```

Day 7 changed `.c` and `.h` files only through the helper-header ownership
comment. Focused QR validation passed after a forced rebuild. The full C
quality gate remains required before Sprint 193 closeout.
