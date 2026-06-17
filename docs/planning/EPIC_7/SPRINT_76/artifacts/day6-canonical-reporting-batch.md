# Sprint 76 Day 6 Artifact: Canonical Reporting Batch

Date: 2026-06-17
Branch: sprint-76

## Purpose

Land the first bounded canonical reporting batch on the maintained report
workflow without widening the canonical benchmark surface or introducing
timing-threshold policy.

## Main Result

The Day 6 landing stayed inside the Day 5 fence:

- `scripts/bench_canonical_report.sh` now emits stronger bundle-level
  longitudinal metadata
- `Makefile` now exposes a bounded report-label override seam through
  `BENCH_CANONICAL_REPORT_LABEL`
- the canonical report still runs through the same public command:
  - `make bench-canonical-report`
- the same four canonical maintained emitters still define the numeric report
  surface:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`

## Landed Behavior

The canonical report bundle now preserves the four CSV outputs and adds bounded
cross-run metadata:

- `manifest.txt` now carries:
  - generated timestamp
  - report directory
  - report label
  - git commit
  - git branch
  - exact command mapping
  - explicit artifact inventory
- `index.tsv` now carries one structured row per canonical emitted artifact
  with:
  - `surface`
  - `category`
  - `report_label`
  - `generated_at_utc`
  - `git_commit`
  - `git_branch`
  - `artifact`
  - `relative_path`
  - `command`

## Preserved Truthfulness Fence

The Day 6 batch preserved the key Sprint 76 guarantees:

- one CSV per canonical emitter remains the numeric artifact surface
- benchmark binaries still own CSV row semantics and proof fields
- `make bench-canonical-report` remains threshold-free
- the report bundle remains a comparison aid, not a pass/fail portability
  claim
- the canonical maintained benchmark surface did not widen

## Explicit Non-Landings

The batch did not widen into:

- timing thresholds or pass/fail benchmark gates
- runtime or exploratory benchmark capture
- canonical benchmark driver edits
- benchmark-row schema rewrites inside the benchmark binaries
- doc or policy follow-through in:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `README.md`

## Smoke Verification

The bounded smoke verification was:

```bash
make bench-canonical-report \
  BENCH_CANONICAL_REPORT_DIR=build/bench-reports/canonical-day6-smoke \
  BENCH_CANONICAL_REPORT_LABEL=day6-smoke
```

The generated bundle contained:

- `bench_refactor_csc.csv`
- `bench_chol_csc.csv`
- `bench_iterative_reuse.csv`
- `bench_eigs_reuse.csv`
- `index.tsv`
- `manifest.txt`

The smoke metadata confirmed the intended bounded contract:

- `report_label=day6-smoke`
- `git_commit=cd54ba2` at run time
- `git_branch=sprint-76`

## Exit State

Sprint 76 now has one stronger canonical report bundle:

- the workflow is still cheap and threshold-free
- the canonical benchmark face is unchanged
- cross-run and cross-branch artifact comparison is now easier and more
  explicit
