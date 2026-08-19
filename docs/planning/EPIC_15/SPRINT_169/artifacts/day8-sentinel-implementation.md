# Sprint 169 Day 8: Regression Sentinel Implementation

## Purpose

Day 8 implements the Day 7 selected-lane regression sentinel. The change adds
a narrow `S6` local smoke ceiling for the selected
`bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` lane while
keeping the canonical selected performance publication row threshold-free.

## Implementation Summary

| Area | Change |
| --- | --- |
| `scripts/performance_sentinels.sh` | Added `S6` as a selected-lane local large-regression gate for `refactor_csc_ms`. |
| `scripts/performance_sentinels.sh` | Added `bench_refactor_csc_nos4.csv` as the raw selected-lane artifact in the sentinel bundle. |
| `scripts/performance_sentinels.sh` | Added `SPARSE_SELECTED_REFACTOR_CSC_MS_CEILING`, defaulting to `500.0` ms, as a positive numeric override for validation and future maintainer experiments. |
| `scripts/performance_sentinels.sh` | Added clear pass/fail/skip behavior for missing binaries, missing fixtures, parse failures, benchmark failures, and ceiling breaches. |
| `scripts/normalize_report_index.py` | Classified `local_selected_regression_gate` as a hard sentinel boundary alongside `local_wall_gate`. |
| `tests/test_normalize_report_index.py` | Added synthetic `S6` coverage to prove normalized report rows preserve the new hard-gate boundary. |
| `Makefile` | Updated the `performance-sentinels` comment to include the selected refactor CSC smoke ceiling. |
| `benchmarks/README.md` | Documented the S6 artifact, row semantics, hard-gate boundary, and non-claim interpretation. |
| `docs/maintainer_guide.md` | Documented S6 maintainer interpretation and narrowed the performance-sentinel non-claim guidance. |

## Generated Row Contract

The generated `S6` row uses:

| Field | Value |
| --- | --- |
| `sentinel_id` | `S6` |
| `status` | `pass`, `fail`, or `skip` |
| `support_tier` | `reviewed_thresholded` |
| `claim_boundary` | `local_selected_regression_gate` |
| `command` | `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` |
| `matrix_or_fixture` | `nos4.mtx` |
| `metric` | `refactor_csc_ms` |
| `baseline` | default ceiling value, currently `500.0` |
| `threshold` | default ceiling value, currently `500.0` |
| `artifact` | `bench_refactor_csc_nos4.csv` |
| `baseline_provenance` | `sprint169_selected_nos4_local_smoke_ceiling` |
| `repeat_semantics` | `configured_repeat_1` |
| `warmup` | `none_configured` |
| `variance` | `not_computed_single_sample` |
| `methodology_notes` | `selected_local_large_regression_gate;not_portable_performance_claim` |

The row deliberately reports backend fields as `n/a` because the default
selected SPD/Cholesky refactor benchmark row does not own the LDLT fallback
context that S3 preserves.

## Pass/Fail Behavior

`S6` passes when the selected benchmark runs, the selected row is parseable,
and `refactor_csc_ms` is less than or equal to the configured ceiling.

`S6` fails when:

- the selected benchmark runs but `refactor_csc_ms` cannot be parsed;
- the selected benchmark runs and `refactor_csc_ms` exceeds the configured
  ceiling.

`S6` skips when:

- `bench_refactor_csc` is missing;
- `tests/data/suitesparse/nos4.mtx` is missing;
- the selected benchmark command itself fails before a parseable row exists.

The script exits nonzero on an `S6` fail, matching the Day 7 design. Skips are
recorded as environment gaps, not passing evidence.

## Failure Output Example

The forced-threshold validation produced the expected failure message:

```text
performance-sentinels: FAIL S6 selected refactor_csc_ms=0.072 ms > 0.000001 ms local smoke ceiling for nos4.mtx --repeat 1
```

## Publication Boundary

This implementation does not change the selected canonical publication row.
The selected canonical report remains:

- `claim_boundary=hosted_selected_threshold_free` in hosted mode;
- `baseline=n/a`;
- `threshold=n/a`;
- `warmup=none_configured`;
- `variance=not_computed_single_sample`.

`S6` is local regression governance only. An `S6` pass must not be described
as portable performance evidence, hosted benchmark evidence, external-library
parity, state-of-the-art performance, or release-quality speed proof.

## Validation

Day 8 changed shell, Python, documentation, tests, Makefile comments, and
planning artifacts. No `.c` or `.h` files were modified, so the full C quality
gate is not required for this day.

Validation run:

```sh
bash -n scripts/performance_sentinels.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/normalize_report_index.py \
  tests/test_normalize_report_index.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_normalize_report_index.py
PYTHONDONTWRITEBYTECODE=1 make performance-sentinels
SPARSE_SELECTED_REFACTOR_CSC_MS_CEILING=0.000001 \
  PYTHONDONTWRITEBYTECODE=1 make performance-sentinels
PYTHONDONTWRITEBYTECODE=1 make performance-sentinels
git diff --check
```

Results:

- shell syntax check passed;
- Python compile check passed;
- `tests/test_normalize_report_index.py` passed;
- normal `make performance-sentinels` passed and emitted `S6` with
  `refactor_csc_ms=0.068`, `baseline=500.0`, and `threshold=500.0`;
- forced-ceiling validation failed as expected with the S6 failure output;
- final normal `make performance-sentinels` passed again;
- `git diff --check` passed.

## Day 8 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sentinel behavior is bounded and reproducible enough for its stated scope. | Complete | `S6` runs one selected `nos4.mtx --repeat 1` command and records stable metadata plus a broad local ceiling. |
| Selected report publication remains threshold-free. | Complete | Canonical report script/checker were not changed for thresholds; S6 is emitted only by `performance-sentinels`. |
| Focused sentinel validation passes or deferral is justified. | Complete | Normal sentinel and normalizer validation passed; forced failure path produced the expected S6 message. |
