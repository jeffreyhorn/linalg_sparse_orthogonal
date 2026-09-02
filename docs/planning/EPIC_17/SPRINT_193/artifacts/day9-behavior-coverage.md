# Day 9 Behavior Coverage Artifact

## Scope

Day 9 added focused behavior regression coverage for the Sprint 193 QR
external-reference helper extraction. The work targets extraction-sensitive
helper-reader behavior rather than broad QR algorithm behavior.

## Coverage Added

The extracted helper now owns two behavior tests:

- `test_qr_external_reference_readers_reject_invalid_arguments`
- `test_qr_external_reference_readers_reject_unsupported_fixtures`

These tests cover:

- Basis reader NULL fixture-key rejection.
- Basis reader NULL output-buffer rejection.
- Basis reader NULL reason-buffer rejection.
- Threshold reader NULL fixture-key rejection.
- Threshold reader NULL output-buffer rejection.
- Threshold reader NULL reason-buffer rejection.
- Unsupported basis fixture diagnostics.
- Unsupported threshold fixture diagnostics.

## Assertion Preservation

The existing success-path numerical tests remain registered through
`tests/test_qr.c` and continue to verify:

- Rank-1 nullspace projector agreement.
- Duplicate-column rank-deficient projector agreement.
- Dependent-row rank-deficient projector agreement.
- Wide rank-deficient nullspace subspace agreement.
- Diagonal threshold family rank decisions.
- Scaled diagonal threshold family rank decisions.
- Perturbed duplicate-family threshold decisions.
- Perturbed dependent-row threshold decisions.

No existing assertions were removed or weakened. The new tests add
diagnostic/status preservation checks around the reader boundary that moved into
`tests/test_qr_external_ref_helpers.h`.

## Validation

Focused validation commands:

```sh
python3 tests/test_qr_external_ref_helper_guard.py
make qr-external-ref-helper-guard
find build -maxdepth 1 -type f -name test_qr -delete && make build/test_qr && ./build/test_qr
git diff --check
```

Observed proof after Day 9:

- Guard regression tests pass.
- The QR helper guard passes.
- `test_qr` includes the new behavior tests.
- The focused QR binary reports 79 tests, 0 failures, 0 skips, 976 assertions,
  and 4.693 s runtime.
- `git diff --check` passes.
