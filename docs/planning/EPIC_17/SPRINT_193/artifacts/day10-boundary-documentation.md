# Day 10 Boundary Documentation Artifact

## Scope

Day 10 documented the Sprint 193 QR external-reference helper boundary in
`docs/maintainer_guide.md`.

## Maintainer-Facing Contract

The guide now records:

- `tests/test_qr_external_ref_helpers.h` owns the selected QR
  rank/nullspace/threshold external-reference readers, moved selected test
  bodies, and reader failure-path behavior tests.
- `tests/test_qr.c` remains the registered QR proof-owner binary.
- `tests/test_qr.c` keeps `main`, selected `RUN_TEST(...)` registrations,
  `_POSIX_C_SOURCE`, `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`, and the economy
  external-reference test body.
- `tests/test_qr.c` must define `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` before
  including `test_qr_external_ref_helpers.h`; the helper header includes
  `test_qr_helpers.h` and `test_solver_helpers.h` itself so formatter-driven
  include sorting cannot hide either the QR-specific helper declarations or the
  external-reference reader API.
- `tests/test_qr_external_ref_helpers.h` remains family-local and header-only.
- `make qr-external-ref-helper-guard` is the focused ownership/source-list/docs
  guard after helper-layout changes.
- Focused QR behavior validation after helper-header edits should force-rebuild
  `build/test_qr` before running `./build/test_qr`.

## Non-Goals

The documentation explicitly frames the extraction as a no-behavior-change
review-surface reduction. It does not claim:

- new QR algorithm capability
- numerical tolerance changes
- performance improvement
- platform expansion
- broader external parity

## Docs Guard Coverage

`scripts/check_qr_external_ref_helper_guard.sh` now checks maintainer-guide
markers for:

- the Sprint 193 QR helper boundary
- the helper owner file
- the registered QR proof-owner file
- the guard command
- the no-behavior-change interpretation

`tests/test_qr_external_ref_helper_guard.py` includes a negative case proving a
missing maintainer-guide guard-command marker fails clearly.
