# Sprint 104 Day 9 Performance Sentinel Batch

## Purpose

Day 9 implements the first bounded local performance sentinel batch from the
Day 8 design. The batch intentionally keeps hard pass/fail behavior limited to
the existing `wall-check` lane and adds threshold-free Cholesky CSC reporting
for local before/after comparison.

## Implemented Scope

| surface | change | claim boundary |
|---|---|---|
| `scripts/performance_sentinels.sh` | new maintainer wrapper that records context, runs S5 wall-check, and captures S2 Cholesky CSC report rows | local regression evidence only |
| `Makefile` | new `performance-sentinels` target with benchmark binary dependencies | focused operator target, not part of default quality gate |
| `build/bench-reports/sentinels/` | generated report directory for local runs | untracked generated output |

No benchmark CSV schema, public API, library source behavior, OpenMP schedule,
or timing threshold changed.

## Sentinel Lanes Implemented

| ID | status | command | behavior |
|---|---|---|---|
| S5 | hard pass/fail | `make wall-check` via `scripts/wall_check.sh` | existing qg-AMD / Pres_Poisson AMD / Pres_Poisson ND threshold gate |
| S2 | threshold-free report | `build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1` | captures Cholesky linked-list, CSC scalar, and CSC supernodal timing plus descriptor context |

S1, S3, and S4 remain selected but deferred until a later pass wires a local
baseline or threshold-free canonical report bundle into the sentinel wrapper.

## Output Contract

The wrapper writes:

- `build/bench-reports/sentinels/sentinels.tsv`
- `build/bench-reports/sentinels/manifest.txt`
- `build/bench-reports/sentinels/wall_check.txt`
- `build/bench-reports/sentinels/bench_chol_csc_nos4.csv`

`sentinels.tsv` has the Day 8 fields:

- `sentinel_id`
- `status`
- `command`
- `build_mode`
- `omp_num_threads`
- `matrix_or_fixture`
- `metric`
- `value`
- `baseline`
- `threshold`
- `notes`

## Representative Local Output

Representative run on branch `sprint-104`:

```text
performance-sentinels: wrote build/bench-reports/sentinels
  - sentinels.tsv
  - manifest.txt
  - wall_check.txt
  - bench_chol_csc_nos4.csv
```

Representative `sentinels.tsv` rows:

```text
S5 pass bcsstk14 qg_amd_reorder_ms value=68.6 baseline=130 threshold=2x
S5 pass Pres_Poisson amd_reorder_ms value=4437.4 baseline=8000 threshold=2x
S5 pass Pres_Poisson nd_reorder_ms value=4110.4 baseline=47055 threshold=1.5x
S2 report nos4.mtx factor_ll_ms value=0.313 threshold=n/a
S2 report nos4.mtx factor_csc_ms value=0.381 threshold=n/a
S2 report nos4.mtx factor_csc_sn_ms value=0.357 threshold=n/a
S2 report nos4.mtx speedup_csc value=0.82 threshold=n/a
S2 report nos4.mtx speedup_csc_sn value=0.88 threshold=n/a
```

Representative context:

```text
build_mode=serial
omp_num_threads=unset
sparse_chol_dense_backend=unset
sparse_ldlt_dense_backend=unset
```

## Skip and Failure Behavior

- Missing S2 binary or fixture emits `S2 skip` rows instead of a false pass.
- Missing S5 binaries or baseline emits `S5 skip` rows.
- A real `wall-check` threshold failure writes the report bundle and exits
  non-zero.
- S2 benchmark runtime failure emits an `S2 skip` row and keeps S5 as the only
  hard gate.

## Limitations and Non-Claims

This batch does not claim:

- S2 timings are portable;
- S2 timing rows are pass/fail quality gates;
- optional dense acceleration is present;
- OpenMP speedup is measured;
- benchmark residuals replace tests;
- S1, S3, and S4 have hard thresholds.

Generated report artifacts under `build/` are local evidence and are not
committed.

## Validation Plan

Required focused validation for this script/Makefile batch:

1. `bash -n scripts/performance_sentinels.sh`
2. `make performance-sentinels`
3. inspect `build/bench-reports/sentinels/sentinels.tsv`
4. inspect `build/bench-reports/sentinels/manifest.txt`
5. `make lint` because the Makefile/build-tooling surface changed
6. `git diff --check`
7. trailing-whitespace scan on touched script, Makefile, and Sprint 104 docs

## Completion Check

| criterion | status |
|---|---|
| bounded local sentinel wrapper added | complete |
| compact structured output added | complete |
| hard gate limited to existing wall-check | complete |
| S2 threshold-free backend-aware report added | complete |
| runtime context recorded | complete |
| skip behavior defined in implementation | complete |
| limitations and non-claims recorded | complete |
