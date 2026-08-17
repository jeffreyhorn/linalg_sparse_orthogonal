# Day 2 QR Target Family Selection

## Selected Family

Sprint 160 selects `qr_overdetermined_compatible_5x3` as the new bounded QR
comparison family.

| Field | Value |
| --- | --- |
| Selected target label | `qr-compatible-ls` |
| Fixture key | `qr_overdetermined_compatible_5x3` |
| Fixture shape | overdetermined compatible least-squares, 5 rows by 3 columns |
| Existing fixture owner | `tests/qr_external_dense_reference.py` |
| Existing C proof owner | `tests/test_qr_solve.c` |
| Proposed comparison output root | `build/comparison/qr_compatible_ls/` |
| Proposed generator target | `qr-compatible-ls` |
| Proposed row ID prefix | `comparison_qr_overdetermined_compatible_5x3_` |
| Initial support tier | local generated until freshness and hosted artifact behavior are implemented |
| Claim scope | fixture-local QR compatible least-squares comparison only |

## Selection Evidence

The selected fixture is already implemented in the source-controlled dense
reference helper and returns an exact compatible solution:

```text
$ python3 tests/qr_external_dense_reference.py qr_overdetermined_compatible_5x3
OK 4
1
-2
0.5
0
```

This gives Day 3 a stable initial metric contract:

| Metric | Expected interpretation |
| --- | --- |
| project status | project probe succeeds |
| baseline status | source-controlled dense helper succeeds |
| residual norm | project and baseline residuals are both near zero |
| solution norm | project and baseline solution norms agree |
| solution values | project and baseline solution vectors agree componentwise |
| max absolute delta | maximum solution component delta is within tolerance |

## Why This Target

`qr_overdetermined_compatible_5x3` is the smallest useful expansion beyond the
current `qr_underdetermined_minnorm_2x4` comparison family:

- it exercises a compatible overdetermined least-squares path rather than a
  minimum-norm underdetermined path;
- it has exact source-controlled reference values;
- it can be compared through residual and solution metrics without raw QR
  basis identity;
- it already has focused C proof ownership in `tests/test_qr_solve.c`;
- it can reuse the existing generated comparison artifact model of project
  observations, baseline observations, dependency status, study rows, summary,
  and manifest;
- it does not require optional NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or
  external package baselines.

## Rejected Or Deferred Candidates

| Candidate | Decision | Reason |
| --- | --- | --- |
| `qr_overdetermined_incompatible_4x2` | Deferred | The helper emits a meaningful nonzero least-squares residual (`1.7320508075688772`). That is useful, but Day 2 keeps Sprint 160 on the simpler exact compatible family; nonzero residual semantics deserve their own metric contract to avoid confusing residual agreement with exact solve behavior. |
| `qr_rankdef_duplicate_5x4_residual_only` | Deferred | Residual-only rank-deficient comparison could be read as broad rank-deficient solve or rank-policy evidence unless the row contract is tighter than Day 2 needs. |
| `qr_rankdef_dependent_row_4x3_residual_only` | Deferred | Dependent-row residual comparison risks rank-threshold or rank-deficient solve interpretation. It remains a later candidate after compatible least-squares comparison closure. |
| optional NumPy/SciPy/LAPACK comparison | Rejected for Sprint 160 | Sprint 160 should reuse the source-controlled dense helper and must not promote optional package availability into pass evidence. |
| raw QR basis or Q/R comparison | Rejected | Basis identity, sign, orientation, and ordering are explicitly non-claims. |

## Fixture And Owner Map

| Owner | Responsibility |
| --- | --- |
| `tests/qr_external_dense_reference.py` | Source-controlled baseline fixture builder and dense least-squares reference. |
| `tests/test_qr_solve.c` | Existing C proof owner for bounded compatible least-squares behavior. |
| `scripts/run_external_comparison.py` | Future generator extension point for `qr-compatible-ls`. |
| `scripts/normalize_report_index.py` | Future selected-row freshness and row-set policy owner. |
| `tests/test_normalize_report_index.py` | Future focused stale/missing/duplicate/unexpected/failing/defer coverage owner. |
| `tests/corpus/manifests/report_families.tsv` | Future report-family metadata owner for row meaning, support tier, artifact pattern, claim scope, and non-claims. |
| `docs/maintainer_guide.md` | Future maintainer command and interpretation guidance owner. |
| `docs/solver_selection.md` and `README.md` | Future public claim-boundary wording owners. |

## Initial Row Sketch

Day 3 may revise these row IDs, but Day 2 defines the starting row set:

- `comparison_qr_overdetermined_compatible_5x3_project_status_v1`
- `comparison_qr_overdetermined_compatible_5x3_baseline_status_v1`
- `comparison_qr_overdetermined_compatible_5x3_residual_norm_v1`
- `comparison_qr_overdetermined_compatible_5x3_solution_norm_v1`
- `comparison_qr_overdetermined_compatible_5x3_solution_values_v1`
- `comparison_qr_overdetermined_compatible_5x3_project_vs_baseline_max_abs_delta_v1`

The row set mirrors the existing QR minimum-norm comparison family so freshness
semantics can stay predictable. Day 3 must still define exact tolerances,
claim-bearing fields, diagnostic fields, skip/defer behavior, and stale/missing
failure behavior before implementation.

## Basis-Identity Non-Claim Notes

The selected family compares observable solve outputs only. It does not compare
or claim:

- raw Q basis vectors;
- Q sign or orientation;
- Q/R ordering or pivot identity;
- internal Householder state;
- global rank-threshold policy;
- broad rank-deficient solve behavior;
- broad QR parity against LAPACK, NumPy, SciPy, SuiteSparse, Eigen, or any
  external-library ecosystem.

## Completion Check

- One target family is selected and narrow enough to close in one sprint.
- Selection criteria are evidence-based and reproducible from source-controlled
  helper output.
- Rejected and deferred candidates have documented blockers.
- Day 3 can proceed to metric contract design without needing CI or code edits
  first.
