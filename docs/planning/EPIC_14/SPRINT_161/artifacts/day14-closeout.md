# Day 14 Closeout

Day 14 closes Sprint 161 with final validation, selected/deferred row notes,
claim review, retrospective inputs, and the Sprint 162 handoff.

## Final Validation Record

| Command | Result | Output Summary |
| --- | --- | --- |
| `make report-index-comparison-freshness` | Passed | Regenerated selected QR and `partial_svd_diag6_k2` comparison reports; `normalize-report-index: freshness ok (25 rows)`. |
| `make report-index-oracle-freshness` | Passed | Regenerated selected QR/partial-SVD oracle output; `normalize-report-index: freshness ok (54 rows)`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | `tests/corpus ok`. |
| `python3 scripts/normalize_report_index.py --family corpus --family oracle --family comparison --check` | Passed | `normalize-report-index: 153 rows ok`. |
| `python3 tests/test_normalize_report_index.py` | Passed | `test-normalize-report-index: ok`. |
| `python3 tests/test_run_external_comparison.py` | Passed | `test-run-external-comparison: ok`. |
| `python3 -m py_compile scripts/normalize_report_index.py scripts/run_external_comparison.py tests/test_normalize_report_index.py tests/test_run_external_comparison.py scripts/validate_corpus_schema.py` | Passed | No syntax errors. |
| `git diff --check` | Passed | No whitespace errors. |

No `.c` or `.h` files are modified on the branch, so `make format`,
`make lint`, and `make test` were not required by the Sprint 161 changed-file
surface.

## Selected Row Closeout

Sprint 161 closes with one selected partial-SVD comparison family:
`partial_svd_diag6_k2`.

The selected generated rows are:

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

The rows are generated-local, `local_only`, and bounded to one fixture-local
diagonal top-k comparison against `tests/svd_external_dense_reference.py`.

## Deferred Row Closeout

Optional dependency rows remain non-proof context:

- `numpy`: `defer`, optional package baseline not selected.
- `scipy`: `defer`, optional package baseline not selected.

Required selected comparison freshness rejects selected rows that are skipped
or deferred. Optional dependency defers cannot be read as pass evidence and do
not create NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or external-library
ecosystem parity.

## Claim Review

The closeout scan reviewed README, maintainer guide, solver-selection docs,
corpus docs, report-index schema docs, Sprint 161 artifacts, comparison
scripts, normalizer tests, report-family rows, and the Makefile target.

The positive claim remains:

> selected fixture-local partial-SVD diagonal top-k comparison for
> `partial_svd_diag6_k2` against the selected source-controlled dense SVD
> reference helper.

The Sprint 161 surface preserves these non-claims:

- no broad SVD correctness;
- no broad partial-SVD correctness;
- no raw singular-vector identity;
- no vector sign or orientation identity;
- no repeated-spectrum ordering claim;
- no NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or external-library ecosystem
  parity;
- no hosted CI proof from local generated rows;
- no release proof;
- no platform portability proof;
- no package-manager proof;
- no shared-library ABI proof;
- no performance superiority;
- no state-of-the-art claim.

## Retrospective Inputs

Use these artifacts as the Sprint 161 retrospective input set:

- `day1-sprint-intake.md`
- `day2-target-selection.md`
- `day3-metric-contract.md`
- `day4-harness-design.md`
- `day5-harness-implementation.md`
- `day6-expected-rows.md`
- `day7-test-design.md`
- `day8-focused-tests.md`
- `day9-report-design.md`
- `day10-report-integration.md`
- `day11-docs-alignment.md`
- `day12-validation.md`
- `day13-evidence-review.md`
- `day14-closeout.md`
- `WORKING_NOTES.md`

Likely retrospective themes:

- Narrow fixture-local evidence can be promoted without changing solver code.
- Comparison freshness is strongest when generator output, normalizer row
  sets, tests, docs, and Makefile targets share the same selected artifact
  contract.
- Optional dependency rows need explicit non-proof wording or they can be
  mistaken for ecosystem parity.
- Dirty-worktree provenance in generated local artifacts is acceptable only as
  local evidence and should never be used as release proof.

## Sprint 162 Handoff

Sprint 162 should start with the Windows package parity decision from the Epic
14 plan. The handoff is:

1. Audit Windows CMake install/downstream proof separately from Unix Make
   install and `pkg-config` proof.
2. Decide whether to promote Windows `pkg-config`, Windows Makefile parity,
   both, or neither.
3. If retaining a non-claim, add stronger checks and docs so Windows package
   wording cannot imply unsupported `pkg-config`, Makefile, package-manager,
   shared-library ABI, or runtime-loader support.
4. Keep static-first package metadata and exact-version CMake downstream proof
   separate from Windows `pkg-config` and Makefile parity.
5. Update CI lane names, support-tier docs, INSTALL/README/maintainer wording,
   and downstream package checks only for the selected product decision.

Sprint 161 comparison evidence must not be reused as Windows package proof.

## Closeout Status

Sprint 161 deliverables are complete and traceable. The selected
`partial_svd_diag6_k2` comparison family is implemented, indexed, tested,
documented, validated, and bounded by explicit non-claims. Sprint 162 is ready
to begin from the Windows package parity handoff.
