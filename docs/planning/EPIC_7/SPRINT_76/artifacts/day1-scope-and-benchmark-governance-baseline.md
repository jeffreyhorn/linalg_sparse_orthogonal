# Sprint 76 Day 1 - Scope and Benchmark Governance Baseline

Date: 2026-06-17  
Branch: sprint-76

## Purpose
Establish a precise Sprint 76 starting baseline for benchmark governance, profiling, and longitudinal reporting using the live tree rather than broad Epic 7 review language.

## Main Result
Sprint 76 now starts from a precise benchmark-governance and reporting queue, not from another planning reset and not from another backend or capability landing.

The strongest local reviewed baseline is still:
- `make quality-review-full`

Reviewed CMake parity was re-materialized live and remains explicit:
- `ctest -N --test-dir build/quality-review-cmake` = `53`

The highest-value Sprint 76 pressure is now clearly narrowed to:
- benchmark-governance re-audit
- canonical reporting and longitudinal-comparison design
- maintained benchmark workflow clarification
- profiling and threshold-policy truthfulness
- benchmark/proof-owner alignment
- validation and closeout

## Preserved Fence
Sprint 76 must preserve the current benchmark-governance truth:
- `make bench-canonical-report` is the threshold-free maintained reporting surface.
- Canonical maintained benchmark proof remains centered on:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- Benchmark artifacts remain reporting and interpretation surfaces, not portable timing gates.
- Narrower exploratory or thresholded lanes such as `bench-fast`, `wall-check`, `bench_reorder`, and `bench_amd_qg` must not silently broaden into the canonical proof contract.
- Maintained docs must keep examples, tests, benchmarks, and reviewed validation ownership distinct.

## Strongest Likely Sprint 76 Touch Surfaces
Maintained benchmark-governance, workflow, and reporting surfaces:
- `README.md` = `1045`
- `benchmarks/README.md` = `377`
- `docs/maintainer_guide.md` = `677`
- `Makefile` = `897`
- `scripts/bench_canonical_report.sh` = `56`

Canonical maintained benchmark emitters:
- `benchmarks/bench_refactor_csc.c` = `611`
- `benchmarks/bench_chol_csc.c` = `423`
- `benchmarks/bench_iterative_reuse.c` = `395`
- `benchmarks/bench_eigs_reuse.c` = `278`

Support or narrower benchmark surfaces:
- `benchmarks/bench_reorder.c` = `321`
- `benchmarks/bench_amd_qg.c` = `332`

## High-Signal Live Reading
The live tree already fixes several important Sprint 76 truths:
- `README.md` carries the compact caller-facing benchmark and reporting summary.
- `benchmarks/README.md` distinguishes maintained canonical surfaces from narrower fast or thresholded lanes.
- `docs/maintainer_guide.md` is the deepest benchmark-governance policy authority.
- `scripts/bench_canonical_report.sh` is still the threshold-free canonical report generator and currently emits the four maintained CSV families.
- `Makefile` preserves `bench-canonical-report` as a report-oriented workflow surface rather than a pass/fail timing gate.
- `benchmarks/bench_chol_csc.c` already exposes the Sprint 75 backend measurability seam through `csc_supernodal_panel_solver`, so Sprint 76 starts from a stronger measurable reporting surface than prior benchmark-governance passes had.

## Interpretation
The strongest Day 1 narrowing is now explicit:
- Sprint 76 is not primarily about inventing new benchmark binaries.
- It is primarily about clarifying which maintained benchmark rows, report fields, comparisons, and workflow commands are canonical, longitudinal, exploratory, or thresholded.
- The highest-value early work is therefore governance and reporting ownership, not another numeric-kernel or public-API batch.

## Exit State
- The maintained reviewed baseline is rechecked.
- The reviewed CMake parity anchor is explicit and current.
- The live benchmark-governance owners, report surfaces, and hotspot map are fixed in writing.
- Sprint 76 can now move into a ranked governance and reporting audit from a precise Day 1 baseline.
