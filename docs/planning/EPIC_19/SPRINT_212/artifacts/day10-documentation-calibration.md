# Sprint 212 Day 10: Documentation Calibration Batch One

## Summary

Day 10 updates user-facing selected benchmark documentation so the Sprint 212
threshold-free policy is visible in the main README, INSTALL support matrix,
and benchmark methodology guide. It also extends the selected performance docs
guard so the new prerequisite wording cannot drift silently.

## Changed Files

| Path | Change |
| --- | --- |
| `README.md` | Adds selected canonical benchmark timing-threshold promotion prerequisites while preserving no-threshold and no-portable-performance wording. |
| `INSTALL.md` | Adds threshold-promotion prerequisites to the Linux/macOS selected performance freshness support/readiness row. |
| `benchmarks/README.md` | Adds selected methodology field markers, required non-portable methodology note wording, and future timing-threshold prerequisite wording. |
| `tests/test_selected_performance_docs.py` | Requires the new markers and adds a missing future-threshold-prerequisite regression. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | Records Day 10 documentation calibration and validation evidence. |

## Documentation Boundary

The user-facing selected benchmark path now states that
`make bench-canonical-report-freshness` remains threshold-free for
`SRT-BENCH-REFACTOR-CSC-NOS4` and selected `bench_refactor_csc` on
`tests/data/suitesparse/nos4.mtx --repeat 1`.

Any future selected canonical timing-threshold promotion must first record:

- stable runner-class evidence;
- compiler evidence;
- repeat policy;
- warmup policy;
- variance rule;
- baseline provenance;
- threshold value;
- retained-artifact policy;
- updated non-claim evidence.

## Guard Coverage

`tests/test_selected_performance_docs.py` now requires:

- README future threshold-prerequisite wording;
- INSTALL support/readiness prerequisite wording;
- benchmark README selected methodology markers for `warmup`, `variance`, and
  `methodology_notes`;
- benchmark README future threshold-prerequisite wording.

The test suite also adds
`test_missing_future_threshold_prerequisite_marker_fails_clearly`, proving the
benchmark README prerequisite text is required.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_performance_docs.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed. |
| `git diff --check` | Passed. |
| `git status --short` | Passed; Day 7-10 docs/script/test edits are tracked and the Sprint 212 planning directory remains untracked until commit time. |
| `git diff --name-only -- '*.c' '*.h'` | Passed; no C/header files changed. |

## Day 10 Outcome

Item 212.5 is underway for user-facing documentation. The selected benchmark
freshness path now explains both the current threshold-free boundary and the
evidence required before any future timing-threshold promotion.
