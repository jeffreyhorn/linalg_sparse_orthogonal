# Sprint 199 Day 5: Windows Path Normalization Tests

## Purpose

Harden selected comparison artifact filtering for Windows path separators
before any selected Windows Cholesky freshness manifest promotion.

Day 5 focuses on path matching only. Missing rows, stale rows, wrong target,
and wrong platform diagnostics remain Day 6 work.

## Changed File

| File | Change |
| --- | --- |
| `tests/test_normalize_report_index.py` | Expanded selected comparison generated-row path matching tests and added near-match rejection coverage. |

No production code changed on Day 5 because
`selected_comparison_generated_rows()` already normalizes backslashes to
forward slashes before exact or suffix artifact matching.

## Positive Path Cases

`test_selected_comparison_generated_rows_match_windows_artifact_paths` now
checks that the selected artifact pattern
`build/comparison/cholesky_spd_tridiag_5/study.tsv` matches all of these
generated row paths:

| Case | Path |
| --- | --- |
| Relative POSIX path | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| Relative Windows path | `build\comparison\cholesky_spd_tridiag_5\study.tsv` |
| Mixed separator path | `build/comparison\cholesky_spd_tridiag_5/study.tsv` |
| Absolute Windows suffix path | `D:\a\linalg_sparse_orthogonal\linalg_sparse_orthogonal\build\comparison\cholesky_spd_tridiag_5\study.tsv` |

These cases cover the hosted artifact shape observed in Day 2 and the
absolute Windows path form used by hosted MSVC/CMake project commands.

## Negative Path Cases

`test_selected_comparison_generated_rows_reject_near_match_artifact_paths`
checks that selected filtering rejects:

| Case | Path |
| --- | --- |
| Similar target with extra digit | `build/comparison/cholesky_spd_tridiag_50/study.tsv` |
| Similar target prefix | `build/comparison/not_cholesky_spd_tridiag_5/study.tsv` |
| Similar target suffix | `build/comparison/cholesky_spd_tridiag_5_extra/study.tsv` |
| Artifact file extension suffix | `build/comparison/cholesky_spd_tridiag_5/study.tsv.bak` |
| Absolute Windows near match | `D:\a\repo\build\comparison\cholesky_spd_tridiag_50\study.tsv` |

The negative cases preserve exact/suffix semantics: a path can match only if
the normalized artifact path is exactly the selected artifact pattern or ends
with a directory separator plus that exact selected artifact pattern.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed |

## Promotion Impact

Day 5 reduces the Day 4 path-normalization blocker for selected generated-row
filtering. It does not by itself promote Windows in the selected target
manifest because these blockers remain:

- missing-row diagnostics need selected Windows coverage;
- stale-artifact diagnostics need selected Windows coverage;
- wrong-target and wrong-platform rows need negative coverage;
- generated `support_tier=local_only` still needs semantic resolution;
- generated summary/non-claim text still needs alignment before public
  promotion.

## Day 6 Handoff

Day 6 should add target-specific diagnostics for missing selected rows, stale
rows, wrong target keys, wrong platforms, and unavailable versus stale
evidence. Diagnostics should name `cholesky-spd-tridiag-5`, the selected
artifact path, and the remediation command.
