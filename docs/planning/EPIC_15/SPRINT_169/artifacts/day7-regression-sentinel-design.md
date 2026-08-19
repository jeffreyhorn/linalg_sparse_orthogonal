# Sprint 169 Day 7: Regression Sentinel Design

## Purpose

Day 7 designs a bounded regression sentinel for the selected Sprint 168/169
performance lane. The goal is to catch large local regressions in the selected
`bench_refactor_csc` path without changing the selected canonical publication
row from threshold-free evidence into a portable timing claim.

## Existing Sentinel Surface Reviewed

| Surface | Current command | Current meaning | Day 7 decision |
| --- | --- | --- | --- |
| Wall-check gate | `make wall-check` through `make performance-sentinels` | Existing thresholded local pass/fail gate for reorder wall-time regressions. | Keep unchanged as `S5`. |
| Performance sentinel bundle | `make performance-sentinels` | Emits `sentinels.tsv`, `manifest.txt`, raw wall-check output, and threshold-free S2/S3 benchmark context. | Extend this bundle rather than creating a new report family. |
| Canonical selected report | `make bench-canonical-report-freshness` | Methodology-bound freshness check for selected `bench_refactor_csc` publication row. | Keep threshold-free with `baseline=n/a` and `threshold=n/a`. |

The current split is intentional: `S5` may fail as a calibrated local gate,
while S2/S3 are report-only rows. Day 7 preserves that interpretation and
adds a separate selected-lane sentinel rather than putting thresholds into the
canonical report index.

## Selected Design

Add a new `S6` selected-lane large-regression sentinel to
`scripts/performance_sentinels.sh` in the implementation day.

| Field | Planned value |
| --- | --- |
| Sentinel ID | `S6` |
| Support tier | `reviewed_thresholded` |
| Claim boundary | `local_selected_regression_gate` |
| Command | `bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` |
| Fixture | `nos4.mtx` |
| Metric | `refactor_csc_ms` from the CSV row |
| Repeat semantics | `configured_repeat_1` |
| Warmup | `none_configured` |
| Variance | `not_computed_single_sample` |
| Baseline provenance | Sprint 169 checked-in sentinel policy, calibrated only as a broad local smoke ceiling for the selected `nos4.mtx` lane |
| Threshold form | A deliberately broad absolute ceiling, recorded in-row as milliseconds |
| Failure behavior | Fail `make performance-sentinels` only when selected `refactor_csc_ms` exceeds the broad local ceiling or the selected row cannot be parsed after the benchmark runs |

The threshold should be wide enough to avoid reading normal runner variance as
a product claim, but narrow enough to catch a class of large regressions such
as disabled sparse-path reuse, accidental dense fallback, or pathological
extra work in the selected CSC refactor path.

## Baseline Provenance Policy

`S6` should not use a fresh per-run generated baseline. Its row should carry a
stable baseline provenance string that makes the limitation explicit, for
example:

```text
sprint169_selected_nos4_local_smoke_ceiling
```

The row must also preserve:

- the exact command and fixture;
- `build_mode`;
- `OMP_NUM_THREADS`;
- `SPARSE_CHOL_DENSE_BACKEND` and `SPARSE_LDLT_DENSE_BACKEND` context where
  already available through the sentinel bundle;
- `repeat_semantics=configured_repeat_1`;
- `warmup=none_configured`;
- `variance=not_computed_single_sample`;
- methodology notes that say the row is not a portable performance claim.

The baseline/threshold values are therefore local governance metadata, not
speed evidence.

## Runtime Budget

The sentinel should reuse the existing `performance-sentinels` Makefile
surface and the existing `bench_refactor_csc` binary dependency.

Implementation constraints:

- one selected benchmark invocation only;
- no extra broad benchmark sweep;
- no external dependency or downloaded fixture;
- expected local runtime should remain seconds-scale;
- generated raw output should stay in `build/bench-reports/sentinels/`;
- missing binary or missing fixture should emit a skip row, not a pass.

This keeps the lane compatible with local maintainer checks and possible
hosted supplemental checks without converting it into hosted publication
evidence.

## Failure Output

When the benchmark runs but the selected metric breaches the ceiling, the
implementation should print a clear failure before exiting nonzero, using the
same narrow language as the generated row:

```text
performance-sentinels: FAIL S6 selected refactor_csc_ms=<actual> ms > <threshold> ms local smoke ceiling for nos4.mtx --repeat 1
```

If the benchmark runs but the expected row or `refactor_csc_ms` field cannot
be parsed, fail with a parser-contract error. Missing binary or fixture should
remain a skip row because those are environment gaps rather than evidence that
the selected path regressed.

## Publication Versus Sentinel Boundary

The selected canonical publication row remains threshold-free:

- `support_tier=hosted_selected` only in hosted mode;
- `claim_boundary=hosted_selected_threshold_free`;
- `baseline=n/a`;
- `threshold=n/a`;
- `warmup=none_configured`;
- `variance=not_computed_single_sample`.

`S6` is separate local regression governance:

- `support_tier=reviewed_thresholded`;
- `claim_boundary=local_selected_regression_gate`;
- non-portable machine/runtime caveats;
- pass/fail semantics only for the selected local sentinel row.

Documentation must not describe an `S6` pass as a portable speed result,
state-of-the-art result, hosted benchmark result, external-library parity
result, or release-quality performance guarantee.

## Deferral Criteria

Defer implementation rather than adding `S6` if any of these are true on the
implementation day:

- `bench_refactor_csc` output cannot be parsed deterministically by column
  name or stable CSV position;
- the selected fixture is unavailable in the maintained repository;
- a broad ceiling cannot be expressed without inviting portable performance
  interpretation;
- runtime no longer fits the bounded local sentinel budget;
- implementation would require changing `.c` or `.h` files solely to support
  the sentinel.

## Day 8 Implementation Plan

1. Extend `scripts/performance_sentinels.sh` with an `S6` selected
   `bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` row.
2. Emit raw selected output as a separate generated artifact, for example
   `bench_refactor_csc_nos4.csv`.
3. Parse `refactor_csc_ms` into one pass/fail row and fail only on parse
   errors or threshold breach.
4. Record the `S6` command and metadata in `manifest.txt`.
5. Update documentation so `S5` and `S6` are hard local gates while S2/S3 and
   canonical publication rows stay threshold-free.
6. Run shell syntax, sentinel generation, focused report checks, and
   `git diff --check`.

## Day 7 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| One bounded sentinel path is selected or explicitly deferred. | Complete | `S6` selected-lane local smoke ceiling is selected for Day 8 implementation. |
| Sentinel wording cannot be read as universal speed evidence. | Complete | The design requires local-only claim boundary, explicit machine/runtime caveats, and non-portable methodology notes. |
| Runtime budget is compatible with local and hosted checks. | Complete | The design adds one selected benchmark invocation and no broad sweep or external fixture. |
