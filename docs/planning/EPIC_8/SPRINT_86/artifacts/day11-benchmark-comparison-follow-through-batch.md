# Sprint 86 Day 11: Benchmark / Comparison Follow-Through Batch

## Purpose

Land the bounded runtime-evidence package fixed on Day 10 so the touched ND
lane has one explicit cheap rerun surface and one benchmark-local command
contract without widening the canonical maintained benchmark face.

## Main Result

The Day 11 landing stayed inside the Day 10 fence:

- required implementation center:
  - `benchmarks/bench_reorder.c`
- directly forced support surfaces actually needed:
  - `Makefile`
  - `benchmarks/README.md`
- not needed in the batch:
  - `scripts/bench_canonical_report.sh`
  - canonical maintained benchmark binaries
  - `docs/maintainer_guide.md`
  - `README.md`
  - proof-owner tests
  - ND / graph implementation owners

## Landed Surface

The landed measurement package introduced one bounded runtime-lane rerun seam:

- `bench_reorder --sprint86-slice`
  - restricts the benchmark run to:
    - `bcsstk14`
    - `Pres_Poisson`
  - preserves the existing emitted CSV schema:
    - `matrix`
    - `n`
    - `reorder`
    - `nnz_L`
    - `reorder_ms`
    - `factor_ms`

The landed command contract added one narrow Make wrapper:

- `make bench-reorder-sprint86`
  - expands to:
    - `bench_reorder --sprint86-slice --skip-factor`

This keeps the bounded Sprint 86 rerun:

- cheap enough for local before/after checks
- outside the canonical maintained benchmark surface
- threshold-free and comparison-oriented

## Measured Slice Output

The bounded Sprint 86 slice emitted the exact touched-corpus comparison the
design called for:

- `bcsstk14`
  - `none`: `nnz_L=190791`, `reorder_ms=0.0`
  - `rcm`: `nnz_L=178311`, `reorder_ms=8.9`
  - `amd`: `nnz_L=116071`, `reorder_ms=99.5`
  - `colamd`: `nnz_L=146037`, `reorder_ms=131.9`
  - `nd`: `nnz_L=132634`, `reorder_ms=366.9`
- `Pres_Poisson`
  - `none`: `nnz_L=5061932`, `reorder_ms=0.0`
  - `rcm`: `nnz_L=3187081`, `reorder_ms=101.8`
  - `amd`: `nnz_L=2668793`, `reorder_ms=6531.9`
  - `colamd`: `nnz_L=3415793`, `reorder_ms=10972.0`
  - `nd`: `nnz_L=2474435`, `reorder_ms=4986.5`

The bounded reading is explicit:

- `Pres_Poisson`: ND still beats AMD on both fill and reorder wall time in the
  skip-factor Sprint 86 slice
- `bcsstk14`: AMD still beats ND on both fill and reorder wall time in the
  same bounded slice

## Strongest Clarification

The useful Day 11 clarification is explicit:

- this is a measurement-surface improvement, not a correctness-owner change
- the touched ND lane now has a stable branch-local rerun entry point
- the canonical maintained benchmark face stayed unchanged
- the slice output is local comparison evidence, not portable performance
  proof and not a pass/fail timing gate

## Preserved Fence

The Day 10 bounded fence held:

- no widening of `bench-canonical-report`
- no new threshold gate
- no reopening of `tests/test_reorder_nd.c`
- no reopening of ND / graph implementation code
- no user-facing README or maintainer-guide timing-claim churn

## Validation

The landed batch passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `make bench-reorder-sprint86`

Reviewed parity remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`

The validated reviewed baseline on this run was:

- reviewed `test_reorder_nd` = `125.12 sec`
- reviewed CMake total = `225.38 sec`

Because Day 11 touched only the measurement surface, those lower observed
times are retained as the current validated baseline, not claimed as causal
wins from the benchmark follow-through itself.

## Exit State

- Sprint 86 now has one landed bounded benchmark/comparison follow-through
  batch.
- The touched ND runtime seam has an explicit local rerun target that stays in
  the runtime lane and outside the canonical maintained benchmark surface.
- The next Sprint 86 seam is later CI/reviewed-path alignment and closeout,
  not more unbounded benchmark-governance churn.
