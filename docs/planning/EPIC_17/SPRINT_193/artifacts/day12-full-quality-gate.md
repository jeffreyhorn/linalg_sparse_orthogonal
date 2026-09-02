# Day 12 Full Quality Gate Artifact

## Scope

Day 12 ran the required full C quality gate for the Sprint 193 QR
external-reference helper extraction.

## Formatter-Stable Fix

The first `make test` attempt after `make format` exposed a formatter-stability
issue: clang-format sorted `test_qr_external_ref_helpers.h` before
`test_solver_helpers.h`, hiding the `TF_EXTERNAL_REFERENCE_*` declarations that
the extracted helper uses.

The fix made `tests/test_qr_external_ref_helpers.h` include its own
`test_solver_helpers.h` dependency after ensuring
`TF_ENABLE_EXTERNAL_REFERENCE_HELPER` is defined. The maintainer documentation
and Day 10 artifact were updated to describe that formatter-stable dependency
contract.

## Required Gate

Final required gate:

```sh
make format && make lint && make test
```

Result: passed.

Key observed details:

- `make format` completed without introducing unrelated tracked-file churn.
- `make lint` completed strict warnings, clang-tidy, and cppcheck.
- `make test` ended with `All tests passed.`
- `test_qr` reported 79 tests, 0 failures, 0 skips, 976 assertions.
- `test_reorder_nd` reported 35 tests, 0 failures, 1 skip.
- `test_framework_optin` reported 8 tests, 0 failures, 3 skips.

## Follow-Up Focused Checks

After the required gate, Day 12 also reran:

```sh
make source-list-check
python3 tests/test_qr_external_ref_helper_guard.py && make qr-external-ref-helper-guard
git diff --check
```

Results:

- `make source-list-check`: passed with 49 library sources.
- QR helper guard regression tests: passed.
- `make qr-external-ref-helper-guard`: passed, including maintainer-doc
  markers.
- `git diff --check`: passed.

## Residuals

No Day 12 quality-gate residual remains. Day 13 still owns the review-surface
audit and Day 14 still owns final closeout/handoff.
