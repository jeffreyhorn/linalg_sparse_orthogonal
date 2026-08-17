# Day 13 Evidence And Claim Review

## Summary

Day 13 reviewed the selected QR comparison evidence end to end: fixture,
generated row, freshness gate, tests, documentation, support tier, skip/defer
semantics, and Sprint 161 handoff.

The review found and fixed one stale diagnostic path in the required-generated
comparison missing-family message. That diagnostic now names both selected
study artifacts.

## Claim-To-Evidence Trace

| Claim | Fixture | Generated Rows | Tests | Documentation |
| --- | --- | --- | --- | --- |
| Fixture-local QR minimum-norm comparison passes against the source-controlled dense helper. | `qr_underdetermined_minnorm_2x4` | Six rows under `build/comparison/qr_minnorm/study.tsv`. | `tests/test_run_external_comparison.py`; `tests/test_normalize_report_index.py`; `make report-index-comparison-freshness`. | `README.md`; `docs/maintainer_guide.md`; `docs/solver_selection.md`; `tests/corpus/README.md`. |
| Fixture-local QR compatible least-squares comparison passes against the source-controlled dense helper. | `qr_overdetermined_compatible_5x3` | Six rows under `build/comparison/qr_compatible_ls/study.tsv`. | `tests/test_run_external_comparison.py`; `tests/test_normalize_report_index.py`; `make report-index-comparison-freshness`. | `README.md`; `docs/maintainer_guide.md`; `docs/solver_selection.md`; `tests/corpus/README.md`. |
| Selected comparison freshness is current only when all 12 generated rows are present, unique, current, and pass. | Both selected fixtures | 12 selected generated rows plus two source-controlled contract rows. | `tests/test_normalize_report_index.py`; `make report-index-comparison-freshness`. | `docs/maintainer_guide.md`; `tests/corpus/README.md`. |
| Optional NumPy/SciPy rows are not pass evidence. | Not selected | `dependency_status.tsv` rows with `status=defer`. | `tests/test_run_external_comparison.py`. | `docs/maintainer_guide.md`; `tests/corpus/README.md`. |

## Selected Row Set

The selected generated comparison row set remains exactly 12 rows:

- six `comparison_qr_underdetermined_minnorm_2x4_*_v1` rows;
- six `comparison_qr_overdetermined_compatible_5x3_*_v1` rows.

Each family contributes:

- `project_status`
- `baseline_status`
- `residual_norm`
- `solution_norm`
- `solution_values`
- `project_vs_baseline_max_abs_delta`

## Support-Tier Consistency

Both selected generated comparison families remain `local_only`.

Reviewed hosted execution, when present, means only that the selected gate ran
and passed on that hosted Linux surface. It does not promote the generated
rows into:

- broad QR proof;
- external-library parity;
- platform portability proof;
- package-manager proof;
- shared-library ABI proof;
- performance proof;
- release proof;
- state-of-the-art evidence.

## Skip/Defer Interpretation

Skip and defer states remain non-proof context.

Evidence checked:

- `scripts/run_external_comparison.py` emits NumPy and SciPy dependency rows as
  `defer`, `optional_package_baseline_not_selected`, `required=no`, and
  `deferred rows are not pass evidence`.
- `tests/test_run_external_comparison.py` asserts this optional dependency
  wording.
- `tests/test_normalize_report_index.py` rejects selected comparison rows with
  `defer` status under required selected freshness.
- Public and maintainer documentation states optional NumPy/SciPy defers
  cannot create pass evidence.

## Day 13 Diagnostic Fix

During the evidence scan, the required-generated comparison missing-family
diagnostic still named only:

```text
build/comparison/qr_minnorm/study.tsv
```

This was fixed to report both selected artifacts:

```text
artifacts=build/comparison/qr_minnorm/study.tsv,build/comparison/qr_compatible_ls/study.tsv
```

The focused normalizer test now asserts this diagnostic for the missing-family
case, row-set mismatch case, unexpected-row case, and non-pass selected-row
case.

## Sprint 161 Partial-SVD Handoff

Sprint 161 should reuse the Sprint 160 pattern for the first bounded
partial-SVD comparison publication.

Recommended first target:

| Candidate | Why |
| --- | --- |
| `partial_svd_diag6_k2` | Source-controlled dense helper exists; diagonal values are deterministic; top-k singular values are stable; raw vector identity can be avoided. |

Required Sprint 161 setup:

1. Select one target family before implementation.
2. Define selected row IDs and tolerances before code changes.
3. Prefer descriptor-backed runner configuration.
4. Add `report_families.tsv` metadata before interpreting generated rows.
5. Add focused runner tests for dispatch, output files, row IDs, metadata, and
   optional dependency context.
6. Add normalizer tests for complete, missing, unexpected, duplicate, stale,
   fail, and defer row states.
7. Keep C proof-owner tests unchanged unless implementation or fixture-helper
   behavior changes.
8. Preserve `local_only` support tier unless later hosted-product wording is
   explicitly earned.

Sprint 161 should avoid claims for:

- broad partial-SVD correctness;
- raw singular-vector identity;
- vector sign/order identity;
- repeated-spectrum vector ordering;
- convergence-rate superiority;
- partial-result guarantees after fail-closed outcomes;
- broad sparse-output/drop-tolerance optimality;
- NumPy/SciPy/LAPACK parity;
- platform, package, ABI, performance, release, or state-of-the-art evidence.

## Validation

Day 13 touched Python normalizer diagnostics and documentation, so focused
validation is:

```sh
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
git diff --check
```

No `.c` or `.h` files changed.

## Day 14 Handoff

Day 14 should perform final targeted checks, update closeout artifacts, review
stale paths/non-claims one last time, and prepare retrospective inputs.
