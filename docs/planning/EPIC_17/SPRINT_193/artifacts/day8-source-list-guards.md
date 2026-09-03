# Day 8 Source-List Guard Artifact

## Scope

Day 8 added a cluster-specific guard for the Sprint 193 QR external-reference
helper extraction:

- Guard script: `scripts/check_qr_external_ref_helper_guard.sh`
- Make target: `make qr-external-ref-helper-guard`
- Guard tests: `tests/test_qr_external_ref_helper_guard.py`

The guard is intentionally narrow. It protects the selected
rank/nullspace/threshold external-reference helper boundary without creating a
general repository source-list policy.

## Guarded Ownership Contract

`tests/test_qr_external_ref_helpers.h` must:

- Exist with include guard `TEST_QR_EXTERNAL_REF_HELPERS_H`.
- Be included exactly once by `tests/test_qr.c`.
- Own the moved selected rank/nullspace/threshold reader and test definitions.
- Stay absent from Makefile, CMake, and library-source registration.

`tests/test_qr.c` must:

- Remain the registered `test_qr` proof-owner in Make and CMake.
- Retain the selected `RUN_TEST(...)` registrations for the extracted tests.
- Retain the economy external-reference test body, which remains outside the
  selected extraction scope.
- Not regain the moved selected-cluster definitions.

## Failure Modes Covered

The focused Python guard tests cover:

- Positive validation against the current tree.
- Positive validation against a minimal fixture tree.
- Missing helper include in `tests/test_qr.c`.
- Moved definition drifting back into `tests/test_qr.c`.
- Economy external-reference body drifting into the helper.
- Header-only helper being named in Makefile registration.

## Validation Note

Because Day 7 confirmed the focused Make path can treat included test helper
headers as stale dependencies, QR behavior validation for helper changes should
force-rebuild `build/test_qr` before running the binary.
