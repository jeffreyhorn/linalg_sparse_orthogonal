# Sprint 212 Day 11: Maintainer Documentation

## Summary

Day 11 completes the maintainer-facing documentation calibration for the
Sprint 212 threshold-free selected benchmark policy. The maintainer guide now
documents exact repair commands, methodology-note requirements, forbidden
methodology-note promotion tokens, and the evidence required before any future
timing-threshold promotion.

## Changed Files

| Path | Change |
| --- | --- |
| `docs/maintainer_guide.md` | Adds selected benchmark methodology-note requirements, forbidden-token interpretation, future threshold prerequisites, and repair workflow. |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | Marks Sprint 212 in progress with Day 1-11 evidence instead of pending future execution. |
| `tests/test_selected_performance_docs.py` | Requires maintainer repair and future threshold-prerequisite markers. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | Records Day 11 status, changed-file snapshot, and validation evidence. |

## Maintainer Guidance Added

The selected performance section now says:

- `methodology_notes` must include `not_portable_performance_claim`;
- methodology-note tokens that promote portable performance, performance
  superiority, state-of-the-art status, hosted timing gates, timing threshold
  gates, portable speed, cross-platform performance, selected timing
  thresholds, or regression thresholds are rejected;
- future timing-threshold promotion requires stable runner-class evidence,
  compiler evidence, repeat policy, warmup policy, variance rule, baseline
  provenance, threshold value, retained-artifact policy, and updated non-claim
  evidence;
- repair should start with `make bench-canonical-report-freshness`, then use
  the focused freshness, selected manifest, and selected performance docs
  regression suites to locate the drift.

## Planning Status

`docs/planning/EPIC_19/PROJECT_PLAN.md` now records Sprint 212 as in progress
with Day 1-11 artifacts and explicitly states that Sprint 212 selected
threshold-free deferral hardening for `SRT-BENCH-REFACTOR-CSC-NOS4`.

Sprints 213-216 remain pending future execution.

## Changed-File Snapshot

| Path | Lines |
| --- | ---: |
| `README.md` | 1132 |
| `INSTALL.md` | 618 |
| `benchmarks/README.md` | 839 |
| `docs/maintainer_guide.md` | 2231 |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 440 |
| `scripts/check_bench_canonical_freshness.py` | 546 |
| `tests/test_bench_canonical_freshness.py` | 729 |
| `tests/test_selected_performance_docs.py` | 220 |
| `tests/test_selected_report_targets_manifest.py` | 1292 |

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_performance_docs.py` | Passed. |
| `git diff --check` | Passed. |
| `git status --short` | Passed; Day 7-11 docs/script/test edits are tracked and the Sprint 212 planning directory remains untracked until commit time. |
| `git diff --name-only -- '*.c' '*.h'` | Passed; no C/header files changed. |

## Day 11 Outcome

Item 212.5 is complete for both user-facing and maintainer-facing
documentation. Maintainers now have exact commands and claim-boundary rules
for repairing selected benchmark freshness without weakening the
threshold-free policy.
