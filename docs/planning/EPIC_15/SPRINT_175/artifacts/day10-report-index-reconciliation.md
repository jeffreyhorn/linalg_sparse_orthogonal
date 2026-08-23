# Day 10: Report Index And Manifest Reconciliation

## Purpose

Reconcile selected report rows, generated comparison artifacts, manifest
ownership, and freshness expectations after Sprint 175 promoted reviewed macOS
hosted selected comparison freshness.

## Report Index Review

Reviewed `scripts/normalize_report_index.py` selected comparison definitions:

- selected comparison row ids already include all four selected families;
- selected comparison artifacts already include:
  - `build/comparison/qr_minnorm/study.tsv`;
  - `build/comparison/qr_compatible_ls/study.tsv`;
  - `build/comparison/partial_svd_diag6_k2/study.tsv`;
  - `build/comparison/lu_nonsym_square_5/study.tsv`;
- strict comparison freshness already fails closed for missing, duplicate,
  unexpected, stale, failed, skipped, or deferred selected comparison rows;
- selected comparison generated row count remains 28 generated rows plus four
  source-controlled contract rows in the normalized comparison index.

No `scripts/normalize_report_index.py` code change was required.

## Manifest Reconciliation

Updated `tests/corpus/manifests/report_families.tsv`:

- the CI reviewed-lanes row now names Linux selected oracle/comparison
  freshness and macOS selected comparison freshness;
- CI metadata non-claims now explicitly preserve no local report freshness
  proof from CI metadata alone, no Windows report freshness, no unsupported
  platform closure, no Windows Makefile parity, no Windows `pkg-config`
  execution parity, and no benchmark release claim;
- selected comparison rows remain `row_origin=generated_local`,
  `support_tier=local_only`, and
  `freshness_policy=generated_compare_inputs`;
- selected comparison row non-claims now say there is no hosted CI proof from
  generated-local row metadata, rather than implying the workflow-artifact
  lanes do not exist;
- selected comparison rows now explicitly retain no broad platform portability
  proof and no Windows report freshness.

This keeps the distinction clear:

- generated comparison TSV rows remain local generated metadata;
- hosted Linux/macOS workflow artifacts are reviewed selected-artifact
  evidence;
- CI lane metadata identifies the reviewed hosted checks but is not itself
  local generated proof.

## Test Updates

Updated `tests/test_normalize_report_index.py` with
`test_selected_comparison_manifest_support_tiers_remain_bounded`.

The new test verifies:

- all four selected comparison manifest rows exist;
- each selected comparison manifest row remains generated-local and local-only;
- artifact patterns match `build/comparison/<subfamily>/study.tsv`;
- non-claims preserve generated-local hosted-boundary wording, broad platform
  non-claims, Windows report freshness non-claims, package-manager and
  shared-library ABI non-claims, performance non-claims, and state-of-the-art
  non-claims;
- the CI reviewed-lanes row names Linux selected oracle/comparison freshness
  and macOS selected comparison freshness while preserving Windows and release
  non-claims.

## Support-Tier Result

| Surface | Reconciled State |
| --- | --- |
| Generated selected comparison rows | `generated_local`, `local_only`, strict freshness checked. |
| Hosted Linux selected comparison artifacts | Reviewed workflow artifacts, not committed generated rows. |
| Hosted macOS selected comparison artifacts | Reviewed workflow artifacts, not committed generated rows. |
| CI lane manifest row | Reviewed cross-platform metadata identifying hosted lanes. |
| Windows report freshness | Still unpromoted and explicitly non-claimed. |
| Broad report-index freshness | Still unpromoted. |

## Validation Results

| Check | Result |
| --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | Passed. |
| `bash scripts/package_manager_deferral_check.sh` | Passed. |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `git diff --check` | Passed. |

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 10.
