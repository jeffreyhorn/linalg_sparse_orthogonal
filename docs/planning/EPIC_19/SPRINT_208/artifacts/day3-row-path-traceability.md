# Sprint 208 Day 3: Row And Path Traceability

## Purpose

Trace selected Windows Cholesky row IDs, target identity, artifact paths,
manifest metadata, workflow upload scope, and normalizer behavior before the
Sprint 208 promotion criteria are set.

## Selected Target Identity

| Field | Source-controlled value |
| --- | --- |
| `target_id` | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| `family` | `comparison` |
| `subfamily` | `cholesky_spd_tridiag_5` |
| `target_key` | `cholesky-spd-tridiag-5` |
| `row_meaning` | selected Cholesky SPD tridiagonal solve comparison freshness |
| `selection_scope` | `reviewed_cross_platform_selected` |
| `freshness_policy` | `generated_compare_inputs` |
| `generator_command` | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` |
| `artifact_pattern` | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| `expected_rows` | `6` |
| `owner` | Report maintainer |

The hosted Windows workflow uses the same target key in both the generator
command and the freshness command.

## Expected Row IDs

| Expected row ID | Day 2 hosted status | Day 2 hosted reason |
| --- | --- | --- |
| `comparison_cholesky_spd_tridiag_5_project_status_v1` | `pass` | `project_status_match` |
| `comparison_cholesky_spd_tridiag_5_baseline_status_v1` | `pass` | `baseline_status_success` |
| `comparison_cholesky_spd_tridiag_5_residual_norm_v1` | `pass` | `project_baseline_residual_delta_within_tolerance` |
| `comparison_cholesky_spd_tridiag_5_solution_norm_v1` | `pass` | `project_baseline_solution_norm_delta_within_tolerance` |
| `comparison_cholesky_spd_tridiag_5_solution_values_v1` | `pass` | `project_baseline_solution_values_delta_within_tolerance` |
| `comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1` | `pass` | `project_baseline_max_abs_delta_within_tolerance` |

The hosted row set matches the manifest `expected_row_ids` exactly.

## Artifact Path Traceability

| Surface | Path or artifact value | Interpretation |
| --- | --- | --- |
| Manifest `artifact_pattern` | `build/comparison/cholesky_spd_tridiag_5/study.tsv` | Source-controlled forward-slash artifact pattern. |
| Hosted generated row `artifact_path` | `build\comparison\cholesky_spd_tridiag_5\study.tsv` | Windows backslash path from hosted artifact. |
| Workflow upload root | `build/comparison/cholesky_spd_tridiag_5/` | Exact selected Cholesky upload scope. |
| Uploaded artifact name | `sprint190-windows-selected-comparison-cholesky` | Windows selected Cholesky artifact name. |
| Uploaded files | six selected Cholesky files | Matches manifest `required_files`. |

`scripts/normalize_report_index.py` normalizes artifact paths by replacing
backslashes with forward slashes before selected-artifact matching. It accepts
exact matches and suffix matches for absolute Windows paths, and current tests
reject near-match siblings and backup-path variants.

## Normalizer Coverage

| Coverage area | Existing regression owner | Day 3 assessment |
| --- | --- | --- |
| Backslash, mixed-separator, and absolute Windows suffix paths | `test_selected_comparison_generated_rows_match_windows_artifact_paths` | Covered for Cholesky artifact matching. |
| Near-match artifact rejection | `test_selected_comparison_generated_rows_reject_near_match_artifact_paths` | Covered for Cholesky near matches. |
| Windows-path stale rows | `test_selected_comparison_target_freshness_rejects_windows_path_stale_rows` | Covered for Cholesky freshness failure diagnostics. |
| Stale or failed selected Cholesky rows | `test_selected_comparison_target_freshness_rejects_cholesky_stale_or_failed` | Covered. |
| Wrong target row set | `test_selected_comparison_target_freshness_rejects_wrong_target_rows` | Covered. |
| Selected-target CLI misuse | `test_selected_target_requires_freshness_check` and unknown-target coverage | Covered. |
| Workflow broad upload paths | `test_windows_selected_cholesky_broad_upload_fails_clearly` and PowerShell validation tests | Covered. |
| Future Windows manifest artifact drift | `test_future_windows_metadata_rejects_wrong_artifact` | Covered for artifact-name mismatch. |

Day 3 does not identify a blocking path-normalization gap for the current
selected Cholesky evidence. Day 9-Day 10 should still review whether promotion
needs stronger exact support-tier/non-claim assertions and direct duplicate or
extra-row Cholesky cases.

## Manifest To Hosted Evidence Comparison

| Field | Manifest value | Hosted evidence value | Day 3 status |
| --- | --- | --- | --- |
| Target key | `cholesky-spd-tridiag-5` | `cholesky-spd-tridiag-5` | Match. |
| Subfamily | `cholesky_spd_tridiag_5` | `cholesky_spd_tridiag_5` | Match. |
| Expected rows | `6` | `6` | Match. |
| Expected row IDs | Six selected Cholesky IDs | Same six IDs | Match. |
| Required files | six selected files | same six files | Match. |
| Artifact pattern | forward-slash `build/comparison/.../study.tsv` | backslash `build\comparison\...\study.tsv` | Path form differs, but normalizer coverage handles this. |
| Workflow file | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` | `.github/workflows/windows-ci.yml` generated evidence exists | Manifest metadata blocker. |
| Workflow artifact | Linux/macOS selected artifacts | `sprint190-windows-selected-comparison-cholesky` | Manifest metadata blocker. |
| Workflow platforms | `linux;macos` | `windows-amd64` row platform | Manifest metadata blocker. |
| Support tier | `local_only` | `local_only` | Consistent but blocks stronger hosted claim. |
| Non-claims | includes `no Windows report freshness` | includes `no hosted CI proof`; `no Windows report freshness` | Claim-semantics blocker. |

## Blocker Classification

### Evidence Availability

No current evidence availability blocker was found:

- latest Windows CI run `35731703320` completed successfully;
- selected Cholesky job completed successfully;
- artifact `10696020870` is present, unexpired, and downloadable;
- artifact file set matches selected `required_files`;
- six selected row IDs are present and pass;
- row source commit matches `d75118349269c6805654070eb44c9c57608b3e47`,
  which is current `master` and this branch base.

### Path Normalization

No current path-normalization blocker was found for the selected Cholesky
artifact pattern:

- generated Windows row path uses backslashes;
- manifest pattern uses forward slashes;
- normalizer converts backslashes before selected-artifact matching;
- existing tests cover backslash, mixed separator, absolute Windows suffix,
  near-match rejection, stale rows, and wrong-target rows.

### Manifest And Claim Semantics

Promotion remains blocked until these surfaces are reconciled:

- selected manifest metadata does not list Windows workflow file, job,
  artifact, or platform;
- generated rows still use `support_tier=local_only`;
- generated rows and summary still say `no hosted CI proof`;
- generated rows and summary still say `no Windows report freshness`;
- public and maintainer docs still describe the lane as guarded workflow
  evidence, not promoted selected Windows freshness.

## Day 4 Handoff

Day 4 should define objective promotion criteria that separate:

1. artifact evidence requirements already satisfied by Day 2;
2. path-normalization requirements that appear covered but need final review;
3. manifest metadata that must change if promotion is selected;
4. generated support-tier and non-claim wording that must change or force
   re-deferral;
5. public and maintainer docs that must align with the final decision.

