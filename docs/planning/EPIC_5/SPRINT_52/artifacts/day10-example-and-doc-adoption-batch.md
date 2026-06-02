# Sprint 52 Day 10: Example and Doc Adoption Batch

## Purpose

Day 10 aligns the two highest-value caller-facing repeated-run direct surfaces
with the stronger Sprint 52 Phase 2 behavior:

- `README.md`
- `examples/example_analysis.c`

The goal is not broad documentation churn. The goal is to make the strongest
public summary surface and the strongest shipped repeated-run direct example say
the same truthful thing about the live contract.

## Main Day 10 Conclusion

Sprint 52 now has a bounded adoption batch that reflects the stronger Phase 2
repeated-run direct lifecycle without widening scope:

- `README.md` now states the repeated-run direct workflow explicitly
- `example_analysis` now teaches what is actually reused and what is not
- one-shot direct APIs remain first-class and visible
- the batch does not broaden into tutorial rewrite or mass example conversion

## Touched Surfaces

### `README.md`

Day 10 adds a compact repeated-run direct workflow section to the top-level
README.

The updated README now makes the public direct repeated-run shape explicit:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`
- `sparse_analysis_free(...)`
- `sparse_factor_free(...)`

It also now states the intended lifecycle directly:

1. zero-init analysis/factors objects
2. analyze once
3. factor / solve
4. refactor / solve many
5. free explicitly

The key Phase 2 boundaries are now visible in one compact place:

- one-shot LU / Cholesky / LDL^T remain first-class peer entry points
- repeated direct reuse preserves symbolic/permutation setup, not stale
  numeric factor contents
- `sparse_refactor_numeric(...)` is the public same-pattern numeric refresh
  path rather than a generic rebuild-anything entry
- the library rejects obvious gross-structure drift cheaply, but does not
  promise a full structural-pattern verifier

That keeps the README truthful without trying to duplicate all of
`include/sparse_analysis.h`.

### `examples/example_analysis.c`

Day 10 keeps the mechanics of the strongest shipped repeated-run direct example
unchanged, but improves how it teaches the contract.

The example now states more clearly:

- reuse preserves symbolic/permutation setup
- reuse does not preserve old numeric factor contents
- later matrices must keep the same sparsity pattern
- rebuilding a fresh same-pattern matrix is the safest high-signal example
  discipline for callers

This shows up in three places:

- file-level comments
- the refactor-loop comments
- runtime output printed by the example itself

That makes the example a better companion to the README and
`include/sparse_analysis.h` without changing the actual workflow it
demonstrates.

## Explicit Non-Landings

Day 10 intentionally does **not** do these:

- rewrite `docs/tutorial.md`
- convert multiple small one-shot examples into repeated-run examples
- broaden `benchmarks/README.md` beyond the Day 8 benchmark-proof work
- redesign the repeated-run direct contract itself
- reopen LU as anything other than the intentionally bounded special-case seam

## Validation

Because `examples/example_analysis.c` changed, the full required code-day gate
was run:

- `make format`
- `make lint`
- `make test`

All passed.

## Focused Follow-On

The strongest shipped repeated-run direct example also ran cleanly:

- `./build/example_analysis`

Representative live output:

- `Reuse preserves symbolic/permutation setup from analysis.`
- `Refactor expects new values on the same sparsity pattern.`
- `Fresh same-pattern matrices keep that contract explicit.`
- timing summary now ends with:
  - `Reused state: symbolic/permutation setup only`
  - `Not reused: stale numeric factor contents`

The example's solve residual remained `4.44e-16`.

## Day 10 Operational Result

Sprint 52 now has the highest-value caller-facing adoption surfaces aligned
with the stronger Phase 2 lifecycle story:

1. the README says the repeated-run direct contract compactly and truthfully
2. the strongest shipped example teaches the same contract in comments and live
   output
3. one-shot direct usage remains visible as the simpler/default path for
   one-off solves

That closes the bounded adoption batch cleanly enough for the next day to
focus on regression expansion rather than caller-surface drift.
