# Sprint 193 Day 6: Helper Movement and Call-Site Preservation

## Summary

Day 6 moved the selected QR external dense-reference rank/nullspace/threshold
logic into `tests/test_qr_external_ref_helpers.h`. The movement was mechanical:
test names, fixture keys, thresholds, tolerances, diagnostics, cleanup paths,
and `RUN_TEST(...)` registration stayed behavior-preserving.

## Moved Symbols

| Symbol | New owner |
| --- | --- |
| `read_qr_basis_external_reference` | `tests/test_qr_external_ref_helpers.h` |
| `read_qr_threshold_external_reference` | `tests/test_qr_external_ref_helpers.h` |
| `test_qr_external_dense_reference_rank1_4x3_nullspace_projector` | `tests/test_qr_external_ref_helpers.h` |
| `test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector` | `tests/test_qr_external_ref_helpers.h` |
| `test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector` | `tests/test_qr_external_ref_helpers.h` |
| `make_rankdef_wide_3x5` | `tests/test_qr_external_ref_helpers.h` |
| `test_qr_external_dense_reference_rankdef_wide_3x5_nullspace_subspace` | `tests/test_qr_external_ref_helpers.h` |
| `test_qr_external_dense_reference_rank_threshold_diag4_family` | `tests/test_qr_external_ref_helpers.h` |
| `test_qr_external_dense_reference_rank_threshold_diag4_scaled_family` | `tests/test_qr_external_ref_helpers.h` |
| `test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family` | `tests/test_qr_external_ref_helpers.h` |
| `test_qr_external_dense_reference_rank_threshold_dependent_row_4x3_perturbed_family` | `tests/test_qr_external_ref_helpers.h` |

## Preserved Proof Owner

`tests/test_qr.c` still owns:

- `main`;
- every selected `RUN_TEST(...)` registration;
- the non-selected economy external-reference test body;
- general QR, economy, sparse-mode, reorder, and refinement tests;
- the feature-test macro block and helper enablement include order.

No Make/CMake test registration changed, and no production source-list entry
was added.

## Review-Surface Metrics

| File | Before Day 6 | After Day 6 | Change |
| --- | ---: | ---: | ---: |
| `tests/test_qr.c` | 3971 | 3038 | -933 |
| `tests/test_qr_external_ref_helpers.h` | 9 | 947 | +938 |

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
| Runtime | 5.421 s |

## Day 7 Handoff

Day 7 should perform the cleanup and error-path audit after movement. The
highest-value checks are external-reference skip/failure returns, allocation
failure cleanup, QR factor cleanup, and confirmation that the selected moved
block does not mutate process-global state.

## Validation

Commands run:

```sh
git status --short --branch
sed -n '1,115p' tests/test_qr.c
sed -n '1140,1675p' tests/test_qr.c
sed -n '1865,2235p' tests/test_qr.c
python3 - <<'PY'
from pathlib import Path
...
PY
sed -n '1,140p' tests/test_qr.c
sed -n '1,120p' tests/test_qr_external_ref_helpers.h
rg -n "read_qr_basis_external_reference|read_qr_threshold_external_reference|make_rankdef_wide_3x5|test_qr_external_dense_reference" tests/test_qr.c tests/test_qr_external_ref_helpers.h
wc -l tests/test_qr.c tests/test_qr_external_ref_helpers.h
make build/test_qr && ./build/test_qr
git diff --stat
git diff -- tests/test_qr.c tests/test_qr_external_ref_helpers.h | sed -n '1,220p'
git diff --check
```

Day 6 changed `.c` and `.h` files. Focused QR validation passed. The full C
quality gate remains required before Sprint 193 closeout.
