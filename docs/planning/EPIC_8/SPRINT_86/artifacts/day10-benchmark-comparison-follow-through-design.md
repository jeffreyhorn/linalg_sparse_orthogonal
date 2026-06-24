# Sprint 86 Day 10: Benchmark / Comparison Follow-Through Design

## Purpose

Define the bounded evidence package Sprint 86 should land next so the touched
ND runtime seam has a smaller, clearer measurement surface without becoming a
broad benchmark-governance rewrite.

## Main Result

Sprint 86 now has one explicit third implementation contract:

- required Day 11 center:
  - `benchmarks/bench_reorder.c`
- directly forced support-only follow-through only if the evidence batch truly
  needs it:
  - `benchmarks/README.md`
  - `Makefile`
- maintainer/user wording only if the landed evidence surface truly changes
  operator guidance:
  - `docs/maintainer_guide.md`
  - `README.md`
- lower-value non-touch surfaces:
  - `scripts/bench_canonical_report.sh`
  - canonical maintained benchmark binaries
  - `benchmarks/bench_fillin.c`
  - proof-owner tests including `tests/test_reorder_nd.c`
  - ND / graph implementation owners

## Exact Day 11 Center

The exact Day 11 implementation center is now fixed to one bounded
`bench_reorder` follow-through package, not a canonical benchmark rewrite and
not a new timing gate.

The decisive Day 10 reason is explicit:

- the Sprint 86 touched runtime seam is the ND/reorder lane
- `bench_reorder` already owns the corresponding runtime/comparison semantics:
  - `matrix`
  - `reorder`
  - `nnz_L`
  - `reorder_ms`
  - `factor_ms`
- the canonical maintained benchmark surface is deliberately smaller and should
  stay untouched unless a later sprint justifies widening it
- Day 6 and Day 9 already supply the before/after reviewed anchors, so the
  next value is cleaner branch-local evidence, not another benchmark-policy
  layer

## Best Evidence Lane

The strongest Day 10 evidence lane is now fixed to:

- keep the evidence owned by `bench_reorder`
- add one bounded reviewed-runtime slice around the actually touched Sprint 86
  fixtures:
  - `bcsstk14`
  - `Pres_Poisson`
  - `Kuu` only if one bounded safety comparison is truly needed after the
    Day 6 threshold shift
- make the Day 11 surface easy to rerun locally without touching correctness
  ownership
- interpret the emitted runtime rows against the already-recorded Day 6 and
  Day 9 reviewed anchors rather than marketing single-run numbers as portable
  truth

The highest-value Day 11 output package is therefore:

- one narrow `bench_reorder` surface for the reviewed ND runtime slice
- explicit touched-corpus comparison output
- bounded branch-local interpretation notes only if the rerun contract really
  changes

## Support-Only Follow-Through

The strongest support-only follow-through is now:

- `benchmarks/README.md`
- `Makefile`
- `docs/maintainer_guide.md`
- `README.md`

Current reading:

- `benchmarks/README.md` should move only if Day 11 adds or changes one small
  `bench_reorder` rerun surface that needs benchmark-local explanation
- `Makefile` should move only if one bounded helper target materially improves
  rerun consistency for the touched runtime slice
- maintainer/user docs should stay untouched unless the landed evidence batch
  truly changes command guidance

## Preserved Fence

The bounded Day 10 fence is explicit:

- no widening of the canonical maintained benchmark surface
- no reinterpretation of `bench-canonical-report` as a pass/fail timing gate
- no reopening of `tests/test_reorder_nd.c`
- no reopening of ND / graph implementation code
- no broad benchmark-local schema rewrite beyond the touched `bench_reorder`
  seam
- no branch-local timing numbers promoted into portable user-facing claims

## Exit State

- Sprint 86 now has one exact third implementation contract.
- Day 11 can stay bounded to `benchmarks/bench_reorder.c` plus only directly
  forced benchmark-local follow-through.
- Canonical benchmark governance, correctness proof owners, and later
  CI/reviewed-path alignment remain explicitly later.
