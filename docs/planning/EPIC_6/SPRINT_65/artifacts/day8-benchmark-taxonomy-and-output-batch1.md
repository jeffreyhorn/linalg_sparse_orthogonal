# Sprint 65 Day 8: Benchmark Taxonomy and Output Batch 1

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Land the first bounded taxonomy/output normalization slice on the direct
canonical benchmark surfaces without widening into the later iterative,
eigensolver, or solver-efficiency batches.

## Landed Scope

This batch intentionally stays limited to:

- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/README.md`

It intentionally does not widen into:

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- solver implementation files
- public headers
- build wiring
- broad benchmark-governance docs outside the benchmark-local surface

## Output Normalization Landed

Both direct canonical CSV surfaces now start with the same stable governance
fields:

- `benchmark`
- `category`
- `matrix`
- `scenario`

### `bench_refactor_csc`

The normalized per-row identity is now:

- `benchmark = bench_refactor_csc`
- `category = proof`
- `scenario = chol_spd | ldlt_kkt`

The existing timing, speedup, and residual fields remain stable:

- `analyze_ms`
- `refactor_public_ms`
- `refactor_csc_ms`
- `solve_public_ms`
- `solve_csc_ms`
- `speedup_refactor`
- `res_public`
- `res_csc`

### `bench_chol_csc`

The normalized per-row identity is now:

- `benchmark = bench_chol_csc`
- `category = proof`
- `scenario = chol_backend_compare`

The existing path/timing/residual fields remain stable:

- `csc_scalar_path`
- `csc_supernodal_path`
- `csc_supernodal_dense_kernel`
- `factor_ll_ms`
- `factor_csc_ms`
- `factor_csc_sn_ms`
- `solve_ll_ms`
- `solve_csc_ms`
- `solve_csc_sn_ms`
- `speedup_csc`
- `speedup_csc_sn`
- `res_ll`
- `res_csc`
- `res_csc_sn`

## Why This Batch Is Bounded

The first direct canonical surfaces were already structurally close to the Day
5 normalization contract:

- timing fields were already `_ms`
- path/backend identity was already explicit where relevant
- speedup and residual fields were already stable and truthful

So the first Day 8 batch only needed to add the missing governance identity
fields, not redesign the entire schema.

## Benchmark-Local Documentation Follow-Through

`benchmarks/README.md` now reflects the new local schema truth for the direct
canonical surfaces:

- `benchmark`
- `category`
- `scenario`

This keeps the documentation change bounded to the touched benchmark-local
surface while broader canonical-surface consolidation remains a Day 9 task.

## Validation and Retained Output

Because benchmark `*.c` files changed, the Day 8 validation gate was:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 686.65 sec`

The retained benchmark-proof spot checks were:

- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative normalized rows are now:

- `bench_refactor_csc,proof,nos4.mtx,chol_spd,100,594,0.622,0.340,0.205,0.014,0.009,1.66,8.24e-16,7.06e-16`
- `bench_chol_csc,proof,nos4.mtx,chol_backend_compare,100,594,scalar,supernodal,builtin,3.252,1.076,0.597,0.017,0.007,0.006,3.02,5.45,7.06e-16,5.89e-16,5.89e-16`

## Day 8 Exit State

Sprint 65 now has:

- the first normalized canonical output slice landed on the direct benchmark
  lane
- stable `benchmark` / `category` / `scenario` identity fields on both direct
  canonical CSV surfaces
- benchmark-local documentation aligned to the new fields
- a clean handoff into the Day 9 canonical-surface consolidation batch
