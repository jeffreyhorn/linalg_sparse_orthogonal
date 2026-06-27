# Sprint 93 Day 12: Proof and Runtime Evidence Follow-Through Batch

## Purpose

Land one bounded runtime-evidence follow-through batch on the retained reorder
benchmark owner so the touched Sprint 93 ND lane reports enough local context
to stay interpretable after the Day 7 runtime reduction and Day 10
runtime-control cleanup.

## Main Result

The Day 12 landing stayed inside the Day 11 fence:

- required center:
  - `benchmarks/bench_reorder.c`
- directly forced support follow-through:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- not needed:
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `scripts/bench_canonical_report.sh`

## Landed Batch

The touched reorder benchmark now emits a smaller but more self-describing CSV
shape for the bounded Sprint 93 runtime lane:

- preserved core fields:
  - `matrix`
  - `n`
  - `reorder`
  - `nnz_L`
  - `reorder_ms`
  - `factor_ms`
- added bounded evidence fields:
  - `reorder_path`
  - `fixture_slice`
  - `nd_base_threshold`

The landed interpretation is intentionally narrow:

- `reorder_path`
  - `direct` or `analyze`
- `fixture_slice`
  - `sprint86` or `all`
- `nd_base_threshold`
  - the live ND base-threshold value used by the emitted row

This keeps the touched runtime lane readable across reruns without widening
into generic benchmark governance or broader public runtime claims.

## Runtime Evidence

The focused bounded reruns now read directly from the emitted rows:

- `./build/bench_reorder --sprint86-slice --skip-factor`
  - header:
    - `matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold`
  - representative rows:
    - `bcsstk14,1806,nd,132634,300.4,skip,direct,sprint86,160`
    - `Pres_Poisson,14822,nd,2474435,4318.9,skip,direct,sprint86,160`
- `./build/bench_reorder --sprint86-slice --skip-factor --reorder-via-analyze`
  - representative rows:
    - `bcsstk14,1806,nd,132634,322.1,skip,analyze,sprint86,160`
    - `Pres_Poisson,14822,nd,2474435,4665.0,skip,analyze,sprint86,160`

The bounded Day 12 evidence call is now explicit:

- the touched Sprint 86 runtime slice remains mixed by matrix, not broad-claim
  oriented
- the row shape now records whether timing came from:
  - direct reorder entry
  - analyze-driven reorder entry
- the row shape also records the bounded fixture scope and live ND threshold
  that produced that evidence

## Validation

Because `benchmarks/bench_reorder.c` changed, the required queue ran and
passed:

- `make format`
- `make lint`
- `make test`

Focused runtime-evidence reruns also passed:

- `./build/bench_reorder --sprint86-slice --skip-factor`
- `./build/bench_reorder --sprint86-slice --skip-factor --reorder-via-analyze`

## Exit State

- Sprint 93 now has one landed bounded runtime-evidence follow-through batch.
- The touched reorder benchmark rows carry enough local context to stay
  interpretable after the Sprint 93 runtime batches.
- Proof-owner movement and canonical-reporting widening were not required by
  the Day 12 evidence lane.
