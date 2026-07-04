# Sprint 104 Day 11 Benchmark Reporting Alignment

## Purpose

Day 11 applies the Day 10 benchmark reporting audit to selected public,
maintainer, and generated-report surfaces. The changes preserve existing
benchmark execution behavior while making the documentation and report labels
match Sprint 104's backend/runtime evidence contract.

## Changed Surfaces

| surface | alignment change |
|---|---|
| `README.md` | adds `make performance-sentinels` to the workflow summary, Make target list, and performance section as a bounded local sentinel bundle |
| `benchmarks/README.md` | documents the sentinel bundle artifacts, S5 hard-gate scope, S2 threshold-free scope, skip behavior, backend/thread context, and non-claims |
| `docs/maintainer_guide.md` | adds `performance-sentinels` to benchmark governance and keeps `wall-check` as the only current hard timing gate |
| `docs/algorithm.md` | adds a short note connecting the historical wall-check gate to the newer Sprint 104 sentinel bundle |
| `scripts/bench_canonical_report.sh` | changes generated canonical report category from `proof` to `measurement` |

## Before and After Reporting Examples

| area | before | after |
|---|---|---|
| canonical report metadata | `surface=canonical`, `category=proof` | `surface=canonical`, `category=measurement` |
| README command list | benchmark commands listed `make bench` and `make bench-canonical-report` | command list also includes `make performance-sentinels` as wall-check hard gate plus threshold-free Cholesky CSC context |
| benchmark guide sentinel docs | sentinel bundle documented only in Makefile/script comments | benchmark guide documents `sentinels.tsv`, `manifest.txt`, `wall_check.txt`, `bench_chol_csc_nos4.csv`, skip rows, and non-claims |
| maintainer governance | canonical/report lanes documented; no Day 9 sentinel bundle | governance names `performance-sentinels`, its runtime context fields, and its no-portable-performance-evidence boundary |
| algorithm wall-check notes | historical wall-check gate only | wall-check remains the hard gate; Sprint 104 sentinel bundle adds threshold-free Cholesky CSC local context |

## Preserved Behavior

- `make bench-canonical-report` still runs the same four maintained benchmark
  commands and writes the same artifact set.
- `make performance-sentinels` still uses the Day 9 implementation:
  - S5 existing `wall-check` rows are the only hard timing gate.
  - S2 Cholesky CSC rows are threshold-free local report context.
- No benchmark CSV schema emitted by a C benchmark binary changed.
- No public API or source behavior changed.

## Backend and Runtime Disclosure Alignment

The updated wording follows these rules:

- builtin kernels remain the portable baseline;
- optional dense backend env requests are context fields, not product-wide
  vendor backend guarantees;
- Cholesky CSC sentinel rows must be interpreted with the recorded dense
  backend env values and selected dense-kernel fields;
- OpenMP timing must be read with the recorded build mode and
  `OMP_NUM_THREADS`;
- speedup, residual, and agreement columns remain local measurement context
  unless a focused test or oracle artifact owns the correctness claim.

## Validation Plan

Because Day 11 changes documentation and shell script report text, but no
`.c` or `.h` files, validation should cover:

- shell syntax for `scripts/bench_canonical_report.sh`;
- focused execution of `make bench-canonical-report` to verify generated
  metadata uses `category=measurement`;
- focused execution of `make performance-sentinels` to verify documented
  sentinel artifacts still generate;
- `git diff --check`;
- trailing-whitespace scan on touched docs, script, and Sprint artifacts.

## Completion Check

| criterion | status |
|---|---|
| Day 10 wording rules applied to selected docs | complete |
| benchmark output label aligned where misleading | complete |
| existing benchmark execution behavior preserved | complete |
| before/after reporting examples recorded | complete |
| validation plan defined | complete |
