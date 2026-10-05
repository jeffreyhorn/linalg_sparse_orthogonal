# Sprint 212 Day 7: Tooling Implementation Batch One

## Summary

Day 7 implements the first threshold-free guard batch for selected canonical
benchmark freshness. The update strengthens `methodology_notes` validation and
adds focused regression fixtures for missing non-claim metadata, threshold
promotion wording, and unselected claim-boundary drift.

## Changed Files

| Path | Change |
| --- | --- |
| `scripts/check_bench_canonical_freshness.py` | Adds a required methodology-note constant and a forbidden methodology-token list for performance or threshold promotion wording. |
| `tests/test_bench_canonical_freshness.py` | Adds three negative fixtures covering missing non-portable methodology note, threshold-promoting methodology note, and unselected hosted claim-boundary drift. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | Records Day 7 implementation scope, fixtures, validation, and remaining handoff. |

## Guard Behavior

The selected canonical benchmark row remains threshold-free. The checker still
requires the selected values established by earlier sprints:

- `status=measurement`
- `baseline=n/a`
- `threshold=n/a`
- `warmup=none_configured`
- `variance=not_computed_single_sample`
- `repeat_semantics=configured_repeat_1`

Day 7 adds explicit methodology-note boundaries:

- `not_portable_performance_claim` is required;
- `portable_performance_claim` is rejected;
- `performance_superiority_claim` is rejected;
- `state_of_the_art_claim` is rejected;
- `hosted_timing_gate` is rejected;
- `timing_threshold_gate` is rejected;
- `portable_speed_claim` is rejected;
- `cross_platform_performance_claim` is rejected;
- `selected_timing_threshold` is rejected;
- `regression_threshold` is rejected.

Unselected canonical rows remain local-only and threshold-free. Day 7 adds a
companion regression for the existing unselected `claim_boundary` enforcement.

## Regression Fixtures

| Test | Purpose |
| --- | --- |
| `test_selected_methodology_notes_require_non_portable_boundary` | Proves a selected row cannot omit the non-portable performance boundary marker. |
| `test_selected_methodology_notes_reject_threshold_promotion` | Proves selected methodology notes cannot carry threshold-promotion wording. |
| `test_unselected_rows_cannot_claim_hosted_threshold_free_boundary` | Proves unselected rows cannot inherit the hosted selected claim boundary. |

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_bench_canonical_freshness.py` | Passed. |
| `git diff --check` | Passed. |
| `git status --short` | Passed; Day 7 script/test edits are tracked and the Sprint 212 planning directory remains untracked until commit time. |
| `git diff --name-only -- '*.c' '*.h'` | Passed; no C/header files changed. |

## Day 7 Outcome

Day 7 completes the first methodology tooling batch without adding any hosted
timing threshold or performance claim. The selected benchmark freshness policy
is still a threshold-free freshness and methodology guard, now with stronger
metadata-level regression coverage.
