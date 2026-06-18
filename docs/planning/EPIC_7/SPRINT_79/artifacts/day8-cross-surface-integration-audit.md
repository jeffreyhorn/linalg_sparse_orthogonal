# Sprint 79 Day 8 - Cross-Surface Integration Audit

Date: 2026-06-18  
Branch: sprint-79

## Purpose
Re-read the integrated post-Day-6 support and policy surfaces so the remaining
Sprint 79 support problem is reduced to one bounded contradiction map instead
of a generic final docs sweep.

## Main Result
Sprint 79’s broad integration problem is now reduced to one ranked
support-surface contradiction map.

The strongest current contradiction is:

- `docs/maintainer_guide.md`

The strongest second contradiction is:

- `README.md`

## Why The Maintainer Guide Now Leads
`docs/maintainer_guide.md` is the strongest remaining contradiction center
because it is still the authoritative policy surface for:

- direct-family lifecycle interpretation
- proof ownership
- platform-confidence reading
- deferred direct-usability framing

But its current direct-family proof-ownership section still reads as if the
large-`n` lifecycle assurance story stops at the earlier Cholesky-heavy state:

- it names the Cholesky lifecycle/property owners directly
- it does not yet name the new Day 6 LDL^T repeated-run lifecycle oracle and
  bounded seeded large-`n` property owners with the same directness
- it therefore now lags the integrated proof package even though the tests do
  not

## Why README Now Ranks Second
`README.md` is the strongest second seam because its compact repeated-run proof
split is broadly truthful but still under-describes the integrated post-Day-6
state.

What already stays true:

- the callback/cancel and family-local caveat reading is still accurate
- the repeated-run benchmark proof split is still accurate
- `bench_refactor_csc --indefinite-kkt` is still the bounded benchmark-side
  LDL^T repeated-run proof surface

What now lags:

- the surrounding proof-owner split still reads more like “benchmark proof
  plus Cholesky-owned oracle/property context” than the integrated current
  state after the new LDL^T public oracle and bounded seeded property landing

## Headers Are Support-Only
The direct-solver headers do not currently justify becoming required Day 9
surfaces.

`include/sparse_ldlt.h` already reads truthfully enough because it:

- states the family-local owned-factor surface clearly
- points stable-pattern repeated direct runs to the shared
  `sparse_analysis.h` lifecycle
- keeps the current callback/cancel limitation truthful

`include/sparse_cholesky.h` already reads truthfully enough because it:

- keeps the one-shot vs repeated-run split explicit
- keeps the CSC callback caveat and matrix-preservation language accurate
- does not claim broader proof ownership than it should

So both headers are now support-only unless the Day 9 wording truly forces a
narrow follow-through.

## Lower-Ranked Support Surfaces
The weaker remaining support surfaces are now explicit:

- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`

Why they rank lower:

- `docs/tutorial.md` already keeps regression/oracle/property ownership with
  the maintained test surfaces
- `examples/README.md` already keeps example-side workflow teaching separate
  from proof ownership
- `benchmarks/README.md` already names the benchmark-vs-test ownership split
  for the repeated-run direct lane clearly enough

## Exact Day 9 Fence
Required Day 9 reconciliation surfaces:

- `docs/maintainer_guide.md`
- `README.md`

Support only if wording truly forces it:

- `include/sparse_ldlt.h`
- `include/sparse_cholesky.h`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`

## Exit State
- Sprint 79’s integration problem is now reduced to one exact contradiction
  map led by the maintainer guide and README.
- The required Day 9 reconciliation batch is fixed explicitly to those two
  surfaces.
- The direct-solver headers and nearby support docs are now bounded
  support-only context rather than assumed touches.
