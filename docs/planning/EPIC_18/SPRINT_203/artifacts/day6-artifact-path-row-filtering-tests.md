# Day 6: Artifact Path And Row Filtering Tests

## Purpose

Day 6 hardened selected QR incompatible comparison freshness coverage for
Windows-style artifact paths. The change adds target-specific QR path matching
fixtures to complement existing stale-row, dependency-only, duplicate, and
wrong-target diagnostics.

## Code Change

| File | Change |
| --- | --- |
| `tests/test_normalize_report_index.py` | Added QR incompatible generated-row path matching and near-match rejection tests, and invoked both from `main()`. |

## New Coverage

| Fixture | Expected behavior |
| --- | --- |
| `build/comparison/qr_incompatible_ls/study.tsv` | Matches the selected QR incompatible artifact. |
| `build\comparison\qr_incompatible_ls\study.tsv` | Matches after backslash normalization. |
| `build/comparison\qr_incompatible_ls/study.tsv` | Matches after mixed-separator normalization. |
| `D:\a\linalg_sparse_orthogonal\linalg_sparse_orthogonal\build\comparison\qr_incompatible_ls\study.tsv` | Matches by normalized suffix under an absolute Windows path. |
| `build/comparison/qr_incompatible_ls_extra/study.tsv` | Rejected as a near match. |
| `build/comparison/not_qr_incompatible_ls/study.tsv` | Rejected as a near match. |
| `build/comparison/qr_incompatible_lss/study.tsv` | Rejected as a near match. |
| `build/comparison/qr_incompatible_ls/study.tsv.bak` | Rejected as a backup/stale-like suffix. |
| `D:\a\repo\build\comparison\qr_incompatible_ls_extra\study.tsv` | Rejected as an absolute near match. |

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `python3 -m py_compile tests/test_normalize_report_index.py` | Passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed; six selected QR incompatible rows fresh, `46` comparison rows total. |

## Promotion Boundary

This is guard coverage only. It does not promote Windows QR incompatible
freshness, does not add Windows manifest metadata, and does not change
workflow YAML. Hosted Windows/MSVC evidence is still required before any
promotion decision.
