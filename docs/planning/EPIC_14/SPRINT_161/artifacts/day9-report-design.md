# Day 9 Report Integration Design

## Summary

Day 9 defines how the selected `partial_svd_diag6_k2` comparison family should
appear in generated comparison artifacts and normalized report-index output.
The implementation already emits and checks the rows locally; this design
locks the report interpretation, evidence tier, reviewer inspection path, and
documentation targets before the Day 10/Day 11 follow-through.

## Evidence-Tier Decision

| Field | Decision |
| --- | --- |
| Report family | `comparison` |
| Subfamily | `partial_svd_diag6_k2` |
| Row origin | `generated_local` |
| Freshness policy | `generated_compare_inputs` |
| Support tier | `local_only` |
| Hosted classification | Not hosted proof |
| Supplemental classification | Not supplemental hosted proof |
| Release classification | Not release proof |
| Claim scope | fixture-local partial-SVD diagonal top-k comparison only |

The selected rows may support only this bounded local claim when every selected
row is present, current, unique, and passing.

## Normalized Row Design

Each generated `study.tsv` row normalizes into a report-index row with:

| Normalized Field | Expected Partial-SVD Value |
| --- | --- |
| `report_family` | `comparison` |
| `subfamily` | `partial_svd_diag6_k2` |
| `native_row_id` | source `comparison_row_id` from `study.tsv` |
| `row_origin` | `generated_local` |
| `row_meaning` | `external_process_dense_reference_comparison` |
| `status` | source row `status`, required to be `pass` for selected freshness |
| `support_tier` | `local_only` |
| `freshness_status` | `fresh` when source commit matches current HEAD |
| `freshness_reason` | generated row source commit matches current HEAD |
| `artifact_path` | `build/comparison/partial_svd_diag6_k2/study.tsv` |
| `generator_command` | `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2` |
| `claim_scope` | fixture-local partial-SVD diagonal top-k comparison only |

The source-controlled contract row normalizes separately as an advisory
`source_controlled` row and remains non-pass evidence.

## Selected Row Set

The selected partial-SVD rows are:

| Row ID | Metric | Interpretation |
| --- | --- | --- |
| `comparison_partial_svd_diag6_k2_project_status_v1` | `project_status` | Project-side partial-SVD probe completed successfully. |
| `comparison_partial_svd_diag6_k2_baseline_status_v1` | `baseline_status` | Source-controlled SVD helper produced reference top-k values. |
| `comparison_partial_svd_diag6_k2_singular_value_0_v1` | `singular_value_0` | Largest singular value matches helper value `9` within tolerance. |
| `comparison_partial_svd_diag6_k2_singular_value_1_v1` | `singular_value_1` | Second singular value matches helper value `6` within tolerance. |
| `comparison_partial_svd_diag6_k2_singular_values_max_abs_delta_v1` | `singular_values_max_abs_delta` | Aggregate top-k singular-value delta is within tolerance. |
| `comparison_partial_svd_diag6_k2_residual_norm_v1` | `residual_norm` | Selected triplet residual diagnostic is within tolerance. |
| `comparison_partial_svd_diag6_k2_u_orthogonality_v1` | `u_orthogonality` | Left-vector orthogonality diagnostic is within tolerance. |
| `comparison_partial_svd_diag6_k2_v_orthogonality_v1` | `v_orthogonality` | Right-vector orthogonality diagnostic is within tolerance. |
| `comparison_partial_svd_diag6_k2_u_projector_diag_v1` | `u_projector_diag` | Diagonal-fixture left projector diagnostic is within tolerance. |
| `comparison_partial_svd_diag6_k2_v_projector_diag_v1` | `v_projector_diag` | Diagonal-fixture right projector diagnostic is within tolerance. |

Together with the two selected QR families, selected comparison freshness now
requires `22` generated rows.

## Freshness Requirement

The selected comparison freshness gate is:

```sh
make report-index-comparison-freshness
```

The gate should:

1. Regenerate `qr-minnorm`.
2. Regenerate `qr-compatible-ls`.
3. Regenerate `partial-svd-diag6-k2`.
4. Run normalized comparison freshness with required generated comparison rows.

Freshness fails if any selected row is missing, unexpected, duplicate, stale,
malformed, skipped, deferred, or non-passing. Optional dependency rows are
context only and cannot substitute for selected rows.

## Reviewer Inspection Path

Reviewers should be able to inspect the selected family without rerunning the
command by reading:

| Artifact | Purpose |
| --- | --- |
| `build/comparison/partial_svd_diag6_k2/study.tsv` | Selected generated comparison rows, statuses, tolerances, provenance, support tier, and non-claims. |
| `build/comparison/partial_svd_diag6_k2/summary.md` | Human-readable local summary and non-claim boundary. |
| `build/comparison/partial_svd_diag6_k2/manifest.tsv` | Source commit, branch, worktree state, platform, compiler, helper, commands, and artifact paths. |
| `build/comparison/partial_svd_diag6_k2/project_observations.tsv` | Project probe metric observations before study-row comparison. |
| `build/comparison/partial_svd_diag6_k2/baseline_observations.tsv` | Source-controlled SVD helper observations. |
| `build/comparison/partial_svd_diag6_k2/dependency_status.tsv` | Required helper and optional dependency context. |
| `build/report-index/normalized-index.tsv` | Normalized rows when the freshness target writes or when maintainers request report output. |

Generated `build/` artifacts remain ignored local evidence unless separately
published by an explicit reviewed lane.

## Non-Claim Boundary

Report wording must preserve:

- no broad SVD correctness;
- no broad partial-SVD correctness;
- no raw singular-vector identity;
- no vector sign or orientation identity;
- no repeated-spectrum ordering claim;
- no external-library ecosystem parity;
- no hosted CI proof;
- no release proof;
- no platform portability proof;
- no package-manager proof;
- no shared-library ABI proof;
- no performance superiority;
- no state-of-the-art claim.

## Documentation Targets

The following public docs still need Day 11 alignment because they describe
selected comparison freshness as QR-only:

| File | Required Update |
| --- | --- |
| `README.md` | Change quick validation wording from selected QR comparison freshness to selected QR plus partial-SVD comparison freshness. |
| `docs/maintainer_guide.md` | Update selected comparison freshness instructions, expected target list, expected row count, and SVD evidence table. |
| `docs/solver_selection.md` | Add bounded partial-SVD comparison wording under SVD guidance without broad parity claims. |
| `tests/corpus/README.md` | Update selected comparison freshness section from two QR families/six rows each to QR plus partial-SVD with ten partial-SVD rows. |
| `tests/corpus/schemas/report_index_fields.md` | Add selected comparison freshness notes if schema docs should mirror the selected oracle gate section. |

## Day 10 Handoff

Day 10 should verify the generated normalized rows and freshness wiring end to
end, then record the generated local validation output. Day 11 should update
the public docs listed above with earned, local-only wording.
