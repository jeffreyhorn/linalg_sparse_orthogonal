# Sprint 75 Day 11 Artifact: Benchmark Proof Refresh

Date: 2026-06-17
Branch: sprint-75

## Purpose

Refresh the maintained `bench_chol_csc` proof surface so the Sprint 75 Day 7
kernel landing is directly measurable, while keeping the Sprint 75 Day 10
public callback/runtime semantics clearly test-owned.

## Main Result

Sprint 75 Day 11 landed one bounded benchmark-proof batch across:

- `benchmarks/bench_chol_csc.c`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- `README.md`

The main benchmark result is:

- `bench_chol_csc` now emits `csc_supernodal_panel_solver`
- the new field reports whether the active dense-kernel descriptor exposes the
  required batched `solve_panel` callback for the supernodal lane
- the docs now state directly that benchmark proof owns path/backend
  measurability, while `tests/test_integration.c` still owns the public
  callback/cancel runtime contract

## Landed Proof Field

The new stable CSV field is:

- `csc_supernodal_panel_solver`

Current interpretation:

- `batched_panel`
  - the active dense-kernel descriptor exposes the required `solve_panel`
    callback
- `missing`
  - the callback is absent, so the supernodal lane would fail through the
    narrow backend-contract boundary rather than silently claiming the Day 7
    path

That makes the Day 7 backend-aware landing measurable without introducing a
timing threshold or a broader backend-policy claim.

## Ownership Split

The benchmark-side ownership is now explicit:

- `bench_chol_csc`
  - path identity
  - dense-kernel descriptor identity
  - panel-solve capability identity
  - timing and residual proof

The test-side ownership stays explicit too:

- `tests/test_integration.c`
  - public progress/cancel callback semantics
  - CSC cancel-before-writeback preservation
  - wrapper-owned runtime truth

So the Sprint 75 Day 10 runtime contract remains test-owned even after the
Day 11 benchmark refresh.

## Validation

Because `benchmarks/bench_chol_csc.c` changed, I ran:

- `make format`
- `make lint`
- `make test`

All passed.

I also ran one live benchmark row:

- `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative retained output:

- header includes `csc_supernodal_panel_solver`
- row includes `builtin,batched_panel`
- residuals stayed in the `1e-16` lane:
  - `res_ll = 7.06e-16`
  - `res_csc = 5.89e-16`
  - `res_csc_sn = 5.89e-16`

## Exit State

Day 11 closes with:

- one new stable benchmark proof field for the Day 7 panel-solve seam
- one explicit benchmark-vs-test ownership split for runtime semantics
- one validated live benchmark row proving the field reports `batched_panel`
- one bounded proof refresh without benchmark-governance widening
