# Sprint 75 Day 1: Scope and Performance Backend Baseline

Date: 2026-06-17
Branch: `sprint-75`

## Purpose

Turn the Sprint 75 project-plan scope plus the Sprint 70, Sprint 64, and
Sprint 74 handoff into one bounded backend/performance sprint, with the
strongest live hotspot families and non-goal fence fixed before deeper audit
begins.

## Main Result

Sprint 75 now starts from a precise backend/performance queue, not from
another planning reset and not from another capability or public-surface
cleanup wave.

The strongest next Epic 7 queue is explicitly:

- hotspot re-audit
- backend/policy design
- kernel integration batch
- callback/runtime follow-through
- benchmark proof refresh
- regression/fallback proof
- validation and closeout

## Preserved Fence

The Sprint 70 and Sprint 64 backend/performance truthfulness fence remains
explicit:

- no fake external-backend or shared-library maturity claim
- no backend widening that weakens the self-contained default build
- no benchmark timing thresholds masquerading as portable proof
- no widened reviewed-platform claim detached from maintained evidence

## Live Baseline Anchors

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

## Strongest Likely Sprint 75 Touch Surfaces

Raw Day 1 `wc -l` counts from the live tree:

### Maintained public and policy surfaces

- `README.md` = `1044`
- `docs/maintainer_guide.md` = `670`
- `benchmarks/README.md` = `370`
- `include/sparse_cholesky.h` = `220`
- `include/sparse_qr.h` = `385`
- `include/sparse_svd.h` = `257`

### Backend/performance implementation seams

- `src/sparse_dense.c` = `597`
- `src/sparse_qr.c` = `1563`
- `src/sparse_svd.c` = `1319`
- `src/sparse_chol_csc_supernodal.c` = `507`
- `src/sparse_chol_csc.c` = `1564`
- `src/sparse_eigs.c` = `1534`

### Strongest proof and reporting surfaces

- `tests/test_chol_csc.c` = `4664`
- `tests/test_qr.c` = `3197`
- `tests/test_svd.c` = `2766`
- `tests/test_integration.c` = `2448`
- `tests/test_eigs.c` = `1560`
- `benchmarks/bench_chol_csc.c` = `407`
- `benchmarks/bench_refactor_csc.c` = `611`
- `benchmarks/bench_eigs_reuse.c` = `278`
- `benchmarks/bench_svd.c` = `180`
- `examples/example_analysis.c` = `210`

## Backend/Performance Hotspot Families

The live backend/performance pressure remains concentrated in these families:

### Kernel ownership and backend-aware implementation seams

Centered on the current backend-aware and dense-kernel implementation lanes:

- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_chol_csc.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`

### Callback and runtime parity

Centered on the maintained progress/cancellation and backend-observability
story:

- `include/sparse_types.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `docs/maintainer_guide.md`

### Canonical proof and reporting

Centered on the maintained benchmark and reporting surfaces:

- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_eigs_reuse.c`
- `benchmarks/bench_svd.c`
- `README.md`
- `benchmarks/README.md`

## Interpretation

The live tree says Sprint 75 should start here:

- the strongest next backend-aware landing is still bounded, not framework-wide
- callback/runtime parity remains a real second lane, not a separate rewrite
- benchmark proof needs to stay measurable and reviewable without becoming a
  pass/fail timing gate
- proof cost remains concentrated in permanent solver tests, benchmark
  surfaces, and maintained interpretation docs rather than in new synthetic
  proof owners

## Exit State

Sprint 75 Day 1 closes with:

1. one backend/performance starting queue
2. one explicit non-goal fence
3. one live reviewed baseline anchor
4. one ranked live backend/performance hotspot map
