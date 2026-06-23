# Sprint 86 Day 6: ND Runtime Reduction Batch

## Purpose

Land one bounded ND runtime-reduction batch inside `src/sparse_reorder_nd.c`
that materially improves the authoritative reviewed-runtime long pole without
widening into proof-surface rebalancing or graph-family rewrite.

## Main Result

Sprint 86's first implementation landing stayed inside the Day 5 fence:

- required implementation center:
  - `src/sparse_reorder_nd.c`
- directly forced support follow-through actually needed:
  - `src/sparse_reorder_nd_internal.h`
  - `src/sparse_graph.c`
  - `benchmarks/bench_reorder.c`
  - `tests/test_reorder_nd.c`
- not needed in the batch:
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
  - `tests/test_graph.c`
  - `tests/test_reorder.c`
  - `docs/maintainer_guide.md`
  - `README.md`

## Landed Surface

The kept runtime win was not the first implementation idea attempted.

Two leaf-glue-oriented `src/sparse_reorder_nd.c` experiments were tried and
discarded after validation because they did not improve the authoritative
reviewed path. The final kept landing instead came from the ND
orchestration/policy seam:

- `sparse_reorder_nd_base_threshold`
  - `128 -> 160`

The touched runtime-history and helper comments were aligned in:

- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_nd_internal.h`
- `src/sparse_graph.c`
- `benchmarks/bench_reorder.c`

## Why This Seam Won

The decisive Day 6 clarification came from focused profiling:

- `SPARSE_ND_PROFILE=1 ./build/test_reorder_nd`

That profile showed the current ND hotspot is dominated by partition work, not
leaf-AMD glue:

- `partition = 23022.473 ms`
- `leaf_amd = 155.773 ms`
- `subgraph = 55.253 ms`
- `total = 23482.393 ms`

That is why the kept win came from threshold policy rather than deeper
leaf-side helper surgery.

## Threshold Re-Sweep Evidence

The bounded `bench_reorder --skip-factor` sweep fixed the final default:

- Pres_Poisson headline:
  - `t=128`: `nnz(L)=2462201`, `reorder wall=7371.8 ms`
  - `t=160`: `nnz(L)=2474435`, `reorder wall=5015.2 ms`
  - `t=192`: `nnz(L)=2499686`, `reorder wall=4687.5 ms`

The retained default is therefore:

- `128 -> 160`

Reason:

- `160` materially reduces the current reviewed-runtime hotspot while
  preserving the current fill-quality proof contract
- `192` buys comparatively little extra runtime on Pres_Poisson while pushing
  fill higher there, so it remains opt-in instead of the default

The multi-fixture sweep stayed inside the current proof tolerances:

- `nos4`
  - unchanged at `nnz(L)=637`
- `bcsstk04`
  - `3722 -> 3143`
  - `135.2 ms -> 2.5 ms`
- `Kuu`
  - `764664 -> 753755`
  - `5972.7 ms -> 2964.4 ms`
- `bcsstk14`
  - `130422 -> 132634`
  - `464.6 ms -> 377.5 ms`
- `s3rmt3m3`
  - `487832 -> 484890`
  - `4896.7 ms -> 3423.9 ms`
- `Pres_Poisson`
  - `2462201 -> 2474435`
  - `7371.8 ms -> 5015.2 ms`

## Proof Follow-Through

The only proof-owner movement the kept runtime seam truly forced was inside
`tests/test_reorder_nd.c`:

- the Pres_Poisson fill commentary now reflects the current default-path
  ratio:
  - `0.923 -> 0.927`
- the fixed-`k` differentiation fixture moved from `bcsstk04` to `bcsstk14`
  because `bcsstk04` becomes a pure leaf-AMD case at the new threshold and no
  longer exercises the separator-lift seam
- the retained fixed-`k` differentiation values are:
  - `hybrid=284058`
  - `balance=195336`
  - `degree=267391`

## Validation

The landed batch passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed parity and the authoritative runtime anchors remained explicit:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- reviewed `test_reorder_nd` = `138.68 sec`
- reviewed CMake `Total Test time (real)` = `234.05 sec`

Relative to the validated Sprint 85 close anchor:

- reviewed `test_reorder_nd`
  - `283.53 sec -> 138.68 sec`
- reviewed CMake total real time
  - `404.15 sec -> 234.05 sec`

## Exit State

- Sprint 86 now has one landed bounded ND runtime-reduction batch.
- The real first win came from the ND threshold/policy seam rather than a
  deeper graph rewrite or proof-surface redistribution.
- The authoritative reviewed long pole moved materially while correctness
  proof quality and reviewed parity stayed intact.
