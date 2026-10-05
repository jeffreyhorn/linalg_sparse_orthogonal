# Sprint 212 Day 8: Tooling Implementation Batch Two

## Summary

Day 8 completes the selected benchmark freshness tooling path for the
threshold-free deferral policy. The implementation adds exact positive
metadata coverage, expands forbidden methodology-token coverage to the full
checker tuple, and adds a manifest methodology drift regression.

## Changed Files

| Path | Change |
| --- | --- |
| `tests/test_bench_canonical_freshness.py` | Adds exact selected threshold-free metadata assertions, full forbidden methodology-token coverage, and a manifest methodology mismatch fixture. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | Records Day 8 tooling completion, fixtures, scope boundary, and validation evidence. |

## Completed Tooling Behavior

The selected canonical benchmark freshness path now has focused coverage for:

- selected workload identity and CSV/index agreement;
- exact threshold-free fields: `status=measurement`, `baseline=n/a`,
  `threshold=n/a`, `warmup=none_configured`,
  `variance=not_computed_single_sample`, and
  `repeat_semantics=configured_repeat_1`;
- local and hosted selected claim-boundary behavior;
- unselected rows remaining `local_only` and `local_threshold_free`;
- required `not_portable_performance_claim` methodology marker;
- rejection of the full forbidden methodology-token set;
- `index.tsv` and `manifest.txt` methodology-note agreement.

No hosted timing threshold, numeric baseline, numeric threshold, portable
performance claim, broad benchmark claim, release benchmark claim, or
state-of-the-art claim was added.

## Regression Fixtures

| Test | Purpose |
| --- | --- |
| `test_positive_local_report_records_exact_threshold_free_methodology` | Proves the generated local selected row records the exact threshold-free metadata and agrees with manifest methodology notes. |
| `test_selected_methodology_notes_reject_threshold_promotion` | Iterates every forbidden methodology token from `FORBIDDEN_METHODOLOGY_NOTES` and verifies a token-specific checker failure. |
| `test_manifest_methodology_notes_must_match_selected_row` | Proves manifest methodology drift fails with a clear row-versus-manifest diagnostic. |

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_bench_canonical_freshness.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `git diff --check` | Passed. |
| `git status --short` | Passed; Day 7-8 script/test edits are tracked and the Sprint 212 planning directory remains untracked until commit time. |
| `git diff --name-only -- '*.c' '*.h'` | Passed; no C/header files changed. |

## Day 8 Outcome

Item 212.3 is complete for the freshness tooling path. The generated selected
benchmark report cannot silently switch to a timing-threshold, portable
performance, superiority, or mismatched-manifest methodology claim through the
freshness checker path.
