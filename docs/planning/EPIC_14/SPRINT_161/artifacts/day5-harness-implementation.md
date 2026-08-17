# Day 5 Harness Implementation

## Summary

Day 5 implements the selected `partial-svd-diag6-k2` comparison target in the
external comparison runner and extends the focused runner test. The new target
generates the ten selected rows defined by the Day 3 metric contract, while the
existing QR targets keep their six-row contracts.

## Files Changed

| File | Change |
| --- | --- |
| `scripts/run_external_comparison.py` | Added the `partial-svd-diag6-k2` target descriptor, partial-SVD C probe source, SVD baseline parser, partial-SVD observation rows, partial-SVD study rows, selected row IDs, target-specific non-claims, and manifest metadata. |
| `tests/test_run_external_comparison.py` | Added target expectations for `partial-svd-diag6-k2`, per-target metric sets, row-ID assertions, and unsupported-target coverage for the new CLI target. |
| `docs/planning/EPIC_14/SPRINT_161/WORKING_NOTES.md` | Added Day 5 log entry. |

## Implemented Target

| Field | Value |
| --- | --- |
| CLI target | `partial-svd-diag6-k2` |
| Fixture key | `partial_svd_diag6_k2` |
| Subfamily | `partial_svd_diag6_k2` |
| Operation | `partial_svd` |
| Output directory | `build/comparison/partial_svd_diag6_k2` |
| Rank | `2` |
| Reference helper | `tests/svd_external_dense_reference.py` |
| Support tier | `local_only` |
| Claim scope | fixture-local partial-SVD diagonal top-k comparison only |

## Generated Row Set

The new target emits exactly ten selected rows:

| Row ID | Metric |
| --- | --- |
| `comparison_partial_svd_diag6_k2_project_status_v1` | `project_status` |
| `comparison_partial_svd_diag6_k2_baseline_status_v1` | `baseline_status` |
| `comparison_partial_svd_diag6_k2_singular_value_0_v1` | `singular_value_0` |
| `comparison_partial_svd_diag6_k2_singular_value_1_v1` | `singular_value_1` |
| `comparison_partial_svd_diag6_k2_singular_values_max_abs_delta_v1` | `singular_values_max_abs_delta` |
| `comparison_partial_svd_diag6_k2_residual_norm_v1` | `residual_norm` |
| `comparison_partial_svd_diag6_k2_u_orthogonality_v1` | `u_orthogonality` |
| `comparison_partial_svd_diag6_k2_v_orthogonality_v1` | `v_orthogonality` |
| `comparison_partial_svd_diag6_k2_u_projector_diag_v1` | `u_projector_diag` |
| `comparison_partial_svd_diag6_k2_v_projector_diag_v1` | `v_projector_diag` |

## Focused Local Run

Command:

```sh
python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2 --output-dir "$tmp"
```

Result:

```text
external-comparison: partial-svd-diag6-k2 project-vs-baseline comparison passed
```

The generated study rows all passed. The singular-value comparison rows had
zero project-vs-baseline delta for values `9` and `6`; residual,
orthogonality, and projector diagnostic rows also passed their upper-bound
tolerances on the local run.

## Preservation Of Existing QR Behavior

The runner keeps QR-specific paths for:

- `qr-minnorm`
- `qr-compatible-ls`
- QR dense-reference helper parsing
- QR solution/residual study rows
- QR six-row selected row contracts

The partial-SVD path is selected by target descriptor metadata and does not
change the QR target descriptors or expected metrics.

## Validation

Commands run:

```sh
python3 -m py_compile scripts/run_external_comparison.py tests/test_run_external_comparison.py
python3 scripts/run_external_comparison.py --self-check
python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2 --output-dir "$tmp"
python3 tests/test_run_external_comparison.py
```

Results:

- Python compile check passed.
- Runner self-check passed.
- Focused partial-SVD target generation passed.
- Focused runner test passed.

No `.c` or `.h` source files were modified.

## Day 6 Handoff

Day 6 should add source-controlled report-family metadata for
`partial_svd_diag6_k2`, verify dependency status semantics, and prepare the row
set for normalizer freshness integration without treating optional dependency
defers as pass evidence.
