# Day 4 Harness Design

## Summary

Day 4 designs the comparison-runner, metadata, expected-row, diagnostics, and
validation changes needed to implement the selected `partial-svd-diag6-k2`
target. The design reuses the Sprint 160 descriptor-backed runner shape and
keeps the first partial-SVD comparison family scoped to the Day 3 metric
contract.

## Runner Extension Point

`scripts/run_external_comparison.py` should gain one new target descriptor:

| Field | Design |
| --- | --- |
| CLI target | `partial-svd-diag6-k2` |
| Fixture key | `partial_svd_diag6_k2` |
| Subfamily | `partial_svd_diag6_k2` |
| Operation | `partial_svd` |
| Output directory | `build/comparison/partial_svd_diag6_k2` |
| Requested rank | `2` |
| Expected singular values | `9`, `6` |
| Singular-value tolerance | `1e-10` |
| Residual tolerance | `1e-10` |
| Orthogonality tolerance | `1e-10` |
| Projector tolerance | `1e-10` |
| Claim scope | fixture-local partial-SVD diagonal top-k comparison only |
| Success message | `external-comparison: partial-svd-diag6-k2 project-vs-baseline comparison passed` |

The target should not change existing QR descriptor behavior, output schemas,
or self-check semantics.

## Project Probe Design

The project probe should compile a temporary C program against
`libsparse_lu_ortho.a`, following the existing QR probe pattern. The probe
should:

1. Build the selected diagonal 6x6 fixture in source.
2. Run the public partial-SVD API for `k=2`.
3. Emit key-value observations for:
   - `status`
   - `singular_value_0`
   - `singular_value_1`
   - `singular_values_max_abs_delta`
   - `residual_norm`
   - `u_orthogonality`
   - `v_orthogonality`
   - `u_projector_diag`
   - `v_projector_diag`
4. Avoid emitting raw U/V component comparisons as selected evidence.
5. Treat build, link, allocation, API, and malformed-output failures as
   project probe failures, not as skip/defer evidence.

If a helper function for residual, orthogonality, or projector calculations
would require changing C/H proof-owner files, implementation should stop and
reassess before broadening the change.

## Baseline Probe Design

The baseline probe should call:

```sh
python3 tests/svd_external_dense_reference.py partial_svd_diag6_k2
```

The parser should require:

- first line: `OK 2`
- second line: finite numeric value `9`
- third line: finite numeric value `6`
- no reliance on NumPy, SciPy, LAPACK, or any external package
- no raw singular-vector baseline

Baseline parse failures should be explicit `baseline_reference_parse_failed`
or equivalent diagnostics, not a passing row.

## Generated Artifacts

The target should produce the same artifact set as the existing comparison
runner targets:

| Artifact | Path |
| --- | --- |
| Project observations | `build/comparison/partial_svd_diag6_k2/project_observations.tsv` |
| Baseline observations | `build/comparison/partial_svd_diag6_k2/baseline_observations.tsv` |
| Dependency status | `build/comparison/partial_svd_diag6_k2/dependency_status.tsv` |
| Study rows | `build/comparison/partial_svd_diag6_k2/study.tsv` |
| Summary | `build/comparison/partial_svd_diag6_k2/summary.md` |
| Manifest | `build/comparison/partial_svd_diag6_k2/manifest.tsv` |

The manifest should record target, fixture key, study path, source commit,
branch, worktree state, generated timestamp, compiler, platform, support tier,
and library path in the same style as the existing QR targets.

## Selected Row Map

Implementation should emit exactly these selected study rows for the new
target:

| Row ID | Metric | Row Kind |
| --- | --- | --- |
| `comparison_partial_svd_diag6_k2_project_status_v1` | `project_status` | `metric_comparison` |
| `comparison_partial_svd_diag6_k2_baseline_status_v1` | `baseline_status` | `dependency_status` |
| `comparison_partial_svd_diag6_k2_singular_value_0_v1` | `singular_value_0` | `metric_comparison` |
| `comparison_partial_svd_diag6_k2_singular_value_1_v1` | `singular_value_1` | `metric_comparison` |
| `comparison_partial_svd_diag6_k2_singular_values_max_abs_delta_v1` | `singular_values_max_abs_delta` | `metric_comparison` |
| `comparison_partial_svd_diag6_k2_residual_norm_v1` | `residual_norm` | `metric_comparison` |
| `comparison_partial_svd_diag6_k2_u_orthogonality_v1` | `u_orthogonality` | `metric_comparison` |
| `comparison_partial_svd_diag6_k2_v_orthogonality_v1` | `v_orthogonality` | `metric_comparison` |
| `comparison_partial_svd_diag6_k2_u_projector_diag_v1` | `u_projector_diag` | `metric_comparison` |
| `comparison_partial_svd_diag6_k2_v_projector_diag_v1` | `v_projector_diag` | `metric_comparison` |

`expected_study_row_ids()` and runner self-checks should cover the new target
without weakening QR expectations.

## Source-Controlled Metadata Row

`tests/corpus/manifests/report_families.tsv` should add one comparison row:

| Field | Design |
| --- | --- |
| `report_family` | `comparison` |
| `subfamily` | `partial_svd_diag6_k2` |
| `row_meaning` | `external_process_dense_reference_comparison` |
| `row_origin` | `generated_local` |
| `status` | `unknown` |
| `support_tier` | `local_only` |
| `freshness_policy` | `generated_compare_inputs` |
| `generator_command` | `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2` |
| `artifact_pattern` | `build/comparison/partial_svd_diag6_k2/study.tsv` |
| `claim_scope` | generated comparison rows record one local fixture-level partial-SVD diagonal top-k comparison for `partial_svd_diag6_k2` against the selected source-controlled dense reference helper |
| `owner` | `Report maintainer` |
| `introduced_in` | `Sprint 161 Day 6` |

The non-claims should include no broad SVD or partial-SVD correctness, no raw
singular-vector identity, no vector sign/order identity, no repeated-spectrum
ordering claim, no NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity, no hosted CI
proof, no release proof, no platform portability proof, no package-manager
proof, no shared-library ABI proof, no performance superiority, and no
state-of-the-art claim.

## Failure Diagnostic Matrix

| Failure Class | Expected Diagnostic Behavior |
| --- | --- |
| Missing static library | Fail before generation with the current missing-library diagnostic style. |
| C probe compile/link failure | Fail as `project_probe_failed` with command output. |
| Partial-SVD API failure | Emit or raise project-status failure; selected freshness must fail. |
| Project output malformed | Fail as malformed project observation, naming the missing or invalid metric. |
| Baseline helper failure | Fail as baseline probe failure with command output. |
| Baseline output malformed | Fail as baseline parse failure, naming expected `OK 2` and value rows. |
| Singular-value tolerance miss | Fail the matching singular-value row and aggregate delta row. |
| Residual tolerance miss | Fail `comparison_partial_svd_diag6_k2_residual_norm_v1`. |
| Orthogonality tolerance miss | Fail the matching diagnostic selected row. |
| Projector tolerance miss | Fail the matching diagnostic selected row. |
| Missing selected row | Fail validation and name the row ID. |
| Duplicate selected row | Fail validation and name the duplicate row ID. |
| Unexpected selected-family row | Normalizer freshness should fail until metadata and tests are updated. |
| Stale source commit | Normalizer freshness should fail as stale generated evidence. |
| Defer or skip selected row | Fail selected freshness; defer/skip rows are non-proof context only. |
| Convergence/fail-closed behavior required | Stop implementation; this first target does not publish convergence or fail-closed claims. |

## Touched Surface Plan

| Surface | Expected Change | Validation |
| --- | --- | --- |
| `scripts/run_external_comparison.py` | Add target descriptor, project probe path, baseline parser, study-row builder, expected-row IDs, and self-check support. | `python3 -m py_compile scripts/run_external_comparison.py`; runner self-check if available; focused target run. |
| `tests/test_run_external_comparison.py` | Add target expectations, required metric set, row-count expectation, and optional dependency assertions for the new target. | `python3 tests/test_run_external_comparison.py`. |
| `tests/corpus/manifests/report_families.tsv` | Add source-controlled partial-SVD comparison metadata row. | `python3 scripts/validate_corpus_schema.py` if schema covers this manifest; focused metadata assertions. |
| `scripts/normalize_report_index.py` | Add selected partial-SVD comparison row set only after generated row contract is implemented. | `python3 tests/test_normalize_report_index.py`; `make report-index-comparison-freshness` after integration. |
| `tests/test_normalize_report_index.py` | Add complete, missing, unexpected, duplicate, stale, fail, and defer cases. | `python3 tests/test_normalize_report_index.py`. |
| `Makefile` | Add the new target to `report-index-comparison-freshness` once normalizer selection is ready. | `make report-index-comparison-freshness`. |
| Docs | Update README, maintainer guide, solver-selection docs, and corpus docs after generated evidence exists. | Documentation hygiene and targeted freshness checks. |

## Day 5 Implementation Guardrails

Day 5 may begin with runner implementation, but it should stop if:

- the partial-SVD probe cannot produce all ten selected metrics;
- producing projector or orthogonality rows requires changing public solver
  behavior;
- raw singular-vector component comparison becomes necessary;
- any selected row would need `skip` or `defer` status to pass;
- the implementation would broaden the claim beyond fixture-local diagonal
  top-k comparison.

## Validation

Day 4 is documentation-only. Validation is limited to Markdown hygiene checks
for Sprint 161 planning files.
