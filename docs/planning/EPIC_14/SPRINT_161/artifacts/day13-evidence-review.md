# Day 13 Evidence Review

Day 13 reviewed the selected `partial_svd_diag6_k2` comparison surface end to
end: source-controlled contract, generated rows, freshness behavior, tests,
documentation, and next-sprint handoff.

## Claim-To-Evidence Trace

| Claim Boundary | Evidence Owner | Review Result |
| --- | --- | --- |
| One selected partial-SVD fixture is compared against the source-controlled dense SVD helper. | `tests/corpus/manifests/report_families.tsv`, `scripts/run_external_comparison.py --target partial-svd-diag6-k2` | The contract row is `comparison/partial_svd_diag6_k2`; the generator target emits split artifacts under `build/comparison/partial_svd_diag6_k2/`. |
| Project and helper runs completed successfully. | `build/comparison/partial_svd_diag6_k2/study.tsv` | `comparison_partial_svd_diag6_k2_project_status_v1` and `comparison_partial_svd_diag6_k2_baseline_status_v1` are selected rows with `pass` status. |
| Top-k singular values match the helper for the diagonal fixture. | Generated rows and `tests/test_run_external_comparison.py` | `singular_value_0`, `singular_value_1`, and `singular_values_max_abs_delta` rows pass with `1e-10` absolute tolerance. |
| Residual and orthogonality diagnostics are within tolerance. | Generated rows, runner tests, normalizer tests | `residual_norm`, `u_orthogonality`, and `v_orthogonality` pass as upper-bound rows. |
| Diagonal projector diagnostics are within tolerance. | Generated rows, runner tests, normalizer tests | `u_projector_diag` and `v_projector_diag` pass as upper-bound rows. |
| Selected comparison freshness requires all selected rows. | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py` | The selected row set includes ten `partial_svd_diag6_k2` rows plus the two selected QR families; missing, unexpected, duplicate, stale, fail, skip, and defer cases are tested. |
| Documentation describes the evidence narrowly. | `README.md`, `docs/maintainer_guide.md`, `docs/solver_selection.md`, `tests/corpus/README.md`, `tests/corpus/schemas/report_index_fields.md` | Public and maintainer docs now describe selected QR plus partial-SVD comparison freshness and preserve local-only non-claims. |

## Generated Row Checklist

The selected `partial_svd_diag6_k2` generated family contains exactly ten
selected rows:

- `comparison_partial_svd_diag6_k2_project_status_v1`
- `comparison_partial_svd_diag6_k2_baseline_status_v1`
- `comparison_partial_svd_diag6_k2_singular_value_0_v1`
- `comparison_partial_svd_diag6_k2_singular_value_1_v1`
- `comparison_partial_svd_diag6_k2_singular_values_max_abs_delta_v1`
- `comparison_partial_svd_diag6_k2_residual_norm_v1`
- `comparison_partial_svd_diag6_k2_u_orthogonality_v1`
- `comparison_partial_svd_diag6_k2_v_orthogonality_v1`
- `comparison_partial_svd_diag6_k2_u_projector_diag_v1`
- `comparison_partial_svd_diag6_k2_v_projector_diag_v1`

Day 12 validation regenerated these rows and
`make report-index-comparison-freshness` reported
`normalize-report-index: freshness ok (25 rows)`.

## Support-Tier Consistency

| Surface | Expected Support Tier | Observed |
| --- | --- | --- |
| Source-controlled report-family contract | `local_only` | `tests/corpus/manifests/report_families.tsv` uses `local_only`. |
| Generated `study.tsv` rows | `local_only` | All selected partial-SVD comparison rows carry `local_only`. |
| Documentation | local generated evidence only | README, maintainer guide, solver-selection docs, corpus README, and schema docs all avoid hosted/release/platform/package/ABI/performance claims. |
| Freshness diagnostics | advisory freshness for generated local rows | Selected rows report `fresh` against current `HEAD`, but remain local generated evidence. |

## Skip/Defer Interpretation

The dependency rows for `partial_svd_diag6_k2` keep optional package baselines
as non-proof context:

- `python3`: `pass`, required interpreter availability only.
- `tests/svd_external_dense_reference.py`: `pass`, required source-controlled
  helper availability only.
- `numpy`: `defer`, optional package baseline not selected.
- `scipy`: `defer`, optional package baseline not selected.

The normalizer tests confirm selected comparison rows with `skip` or `defer`
status fail the required selected comparison freshness gate. Optional
dependency defers cannot be read as pass evidence and cannot create NumPy,
SciPy, LAPACK, SuiteSparse, Eigen, or external-library ecosystem parity.

## Wording Audit

The reviewed docs preserve these non-claims:

- no broad SVD correctness;
- no broad partial-SVD correctness;
- no raw singular-vector identity;
- no vector sign or orientation identity;
- no repeated-spectrum ordering claim;
- no external-library ecosystem parity;
- no hosted CI proof from the local generated rows;
- no release proof;
- no platform portability proof;
- no package-manager proof;
- no shared-library ABI proof;
- no performance or state-of-the-art claim.

The positive wording is bounded to one fixture-local diagonal top-k comparison
against a source-controlled dense SVD reference helper.

## Sprint 162 Windows Package Handoff

Sprint 162 should start from a separate product boundary:

1. Compare current Windows CMake install/downstream proof with Linux/macOS
   Make install and `pkg-config` proof.
2. Decide whether Windows `pkg-config`, Windows Makefile parity, both, or
   neither will be promoted.
3. If retained as non-claim, add stronger unsupported-surface checks so Windows
   package wording cannot imply `pkg-config`, Makefile, package-manager,
   shared-library ABI, or runtime-loader support.
4. Keep static-first package metadata and exact-version CMake downstream proof
   separate from Windows `pkg-config` or Makefile parity.
5. Update CI lane names, support-tier docs, INSTALL/README/maintainer wording,
   and downstream package checks only for the selected product decision.

The Sprint 161 comparison work does not affect Windows package parity. Its
handoff is only that package claims must remain evidence-bound and cannot
borrow proof from local generated solver comparisons.

## Review Conclusion

The selected partial-SVD comparison evidence is reviewable end to end. The
implemented surface supports only a fixture-local `partial_svd_diag6_k2`
diagonal top-k comparison with local generated freshness. No unsupported broad
SVD, partial-SVD, external parity, package, platform, ABI, performance,
release, or state-of-the-art claim was found in the reviewed Sprint 161
wording.
