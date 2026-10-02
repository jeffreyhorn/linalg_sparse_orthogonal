# Sprint 212 Day 13: Review Hardening

## Summary

Day 13 reviewed the Sprint 212 changed surfaces for guard bypasses, stale
wording, missing exact-field checks, and claim-boundary gaps. Two concrete
hardening items were found and fixed.

## Findings And Fixes

| Finding | Fix |
| --- | --- |
| `methodology_notes` forbidden-token checks did not trim token whitespace before comparison. A token such as ` selected_timing_threshold ` could bypass exact forbidden-token matching. | `scripts/check_bench_canonical_freshness.py` now strips each semicolon-delimited methodology-note token before required and forbidden token enforcement. |
| The selected performance docs guard matched `timing threshold` but not the hyphenated `timing-threshold` form for selected canonical benchmark overclaims. | `tests/test_selected_performance_docs.py` now matches `timing[- ]threshold` and includes a hyphenated overclaim regression. |

## Regression Coverage

| Test | Purpose |
| --- | --- |
| `test_selected_methodology_notes_reject_spaced_threshold_promotion` | Confirms whitespace around forbidden methodology-note tokens does not bypass the freshness checker. |
| `test_forbidden_selected_timing_threshold_overclaim_fails_clearly` | Confirms hyphenated selected timing-threshold wording is rejected by the docs guard. |

## Review Sweep

The review sweep checked active docs, corpus docs, schema docs, and Sprint 212
planning artifacts for selected performance overclaims and timing-threshold
wording. Matches were limited to non-claims, guard descriptions, and planning
examples of rejected claims.

## Changed-File Snapshot

| Path | Lines |
| --- | ---: |
| `README.md` | 1132 |
| `INSTALL.md` | 618 |
| `benchmarks/README.md` | 839 |
| `docs/maintainer_guide.md` | 2231 |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 440 |
| `scripts/check_bench_canonical_freshness.py` | 546 |
| `tests/test_bench_canonical_freshness.py` | 746 |
| `tests/test_selected_performance_docs.py` | 236 |
| `tests/test_selected_report_targets_manifest.py` | 1292 |

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_performance_docs.py` | Passed. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 scripts/validate_corpus_schema.py` | Passed. |
| Selected overclaim text sweep with `rg` | Reviewed; matches were non-claims, guard descriptions, or planning rejected-claim examples. |
| `git diff --check` | Passed. |
| `git status --short` | Passed; Day 7-13 docs/script/test edits are tracked and the Sprint 212 planning directory remains untracked until commit time. |
| `git diff --name-only -- '*.c' '*.h'` | Passed; no C/header files changed. |

## Day 13 Outcome

The branch has been reviewed for common Sprint 212 bypass classes. The two
found gaps now have focused regressions and passing validation. No new
performance, platform, package, release, or state-of-the-art claim was added.
