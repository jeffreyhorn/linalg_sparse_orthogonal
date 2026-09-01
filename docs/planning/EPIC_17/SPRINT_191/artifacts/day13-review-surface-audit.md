# Sprint 191 Day 13: Review Surface Audit

## Summary

Day 13 reviewed the Sprint 191 diff for consistency, unnecessary churn,
overbroad claims, brittle tests, accidental generated output, and Windows
scope drift.

The selected `qr-incompatible-ls` family remains reviewable as one bounded
change: one target, one fixture, one subfamily, one artifact directory, six
selected rows, and Linux/macOS selected comparison freshness only.

## Consistency Checklist

| Check | Result |
| --- | --- |
| Target key | `qr-incompatible-ls` is consistent across runner, manifest, Makefile, tests, workflows, and docs. |
| Subfamily | `qr_incompatible_ls` is consistent across runner output, report-family metadata, selected target metadata, normalizer diagnostics, and docs. |
| Fixture | `qr_overdetermined_incompatible_4x2` is consistently named as the only selected fixture. |
| Expected rows | The selected manifest row has `expected_rows=6`, matching `runner.expected_study_row_ids()`. |
| Row IDs | Manifest row IDs match the runner-generated row IDs in order. |
| Artifact path | `build/comparison/qr_incompatible_ls/study.tsv` is the sole selected study artifact path. |
| Required files | The selected manifest names the six generated files: project observations, baseline observations, dependency status, study, summary, and manifest. |
| Linux workflow | Upload scope includes the six exact QR incompatible artifact paths. |
| macOS workflow | Upload scope includes the six exact QR incompatible artifact paths. |
| Windows workflow | No `qr-incompatible-ls`, `qr_incompatible_ls`, or `qr_overdetermined_incompatible_4x2` selected-target references. |
| Documentation | Current docs describe fixture-local evidence and retain broad QR, broad least-squares, external parity, platform, package, ABI, performance, release, and state-of-the-art non-claims. |

## Cleanup

Day 13 updated one live maintainer trust-boundary table that still described
selected QR generated comparisons as only:

- `qr_underdetermined_minnorm_2x4`;
- `qr_overdetermined_compatible_5x3`.

It now includes:

- `qr_overdetermined_incompatible_4x2`.

The same table now explicitly includes the broad least-squares non-claim.

## No-Change Decisions

| Topic | Decision |
| --- | --- |
| Repeated test constants | Kept. The explicit constants make selected row identity obvious to reviewers. |
| Production abstraction | No new abstraction added. The optional `expected_residual_norm` field is sufficient and keeps existing zero-residual targets stable. |
| Generated artifact tests | No new tests added. Runner tests already assert required files, manifest fields, row IDs, metrics, support tier, and dependency rows. |
| Windows metadata | Not promoted. QR incompatible least-squares has no Windows selected freshness metadata until a future MSVC proof is reviewed. |
| Historical planning files | Not edited. Historical artifacts remain prior-state records. |

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `python3 tests/test_normalize_report_index.py` | Pass |
| `bash scripts/check_qr_header_docs_guard.sh` | Pass |
| `python3 tests/test_selected_comparison_workflow.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 scripts/validate_corpus_schema.py` | Pass |
| `make report-index-comparison-freshness` | Pass, 46 normalized rows |
| `python3 -m py_compile tests/test_run_external_comparison.py tests/test_normalize_report_index.py scripts/run_external_comparison.py scripts/normalize_report_index.py` | Pass |
| active-doc stale wording scan | Pass |

No `.c` or `.h` files changed, so the full C quality gate is not required for
Day 13.

## Day 14 Handoff

Day 14 should perform the final closeout validation, inspect generated
`qr_incompatible_ls` summary and manifest output one last time, confirm the
worktree contains no generated cache/output churn, and write the sprint
closeout artifact with residuals and PR-ready evidence.
