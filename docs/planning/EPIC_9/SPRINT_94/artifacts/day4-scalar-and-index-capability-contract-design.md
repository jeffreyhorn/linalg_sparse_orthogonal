# Sprint 94 Day 4: Scalar and Index Capability Contract Design

## Purpose

Define the bounded scalar/index capability contract so Sprint 94 can widen one
real capability seam without blurring wider-index maturity, solver-family
breadth, or explicit non-claims.

## Main Result

Sprint 94 now has one explicit scalar/index capability contract:

- public scalar-contract debt:
  - means the repo has already widened naming and some caller-facing buffer
    seams through `sparse_scalar_t`, but still ships a real-only `double`
    product truth
  - remains the strongest first-class implementation target

- index/ABI-maturity debt:
  - means compile-time `SPARSE_IDX_BITS` and `idx_t` are real public features,
    but touched-path formatting, width-aware proof, and consumer
    interpretation are not yet as mature as the contract wording invites
  - remains real Sprint 94 work, but sequenced behind the first scalar seam
    unless directly forced

- solver-family breadth debt:
  - means some solver-family public buffers already route through
    `sparse_scalar_t`, while deeper dense/solver owners still read as bounded
    real-only implementations
  - remains real Sprint 94 work, but only where a first scalar/index widening
    actually needs it

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- Sprint 94 should not treat all remaining capability debt as one generic
  "support more numeric types" problem
- it should not treat wider-index maturity as equivalent to broad scalar
  widening
- it should first widen one bounded scalar-contract seam, then tighten
  touched index/ABI maturity and solver-family breadth only where the same
  widened seam still depends on them

## Preserved Product Truth

The bounded product interpretation is now fixed:

- builtin reviewed defaults stay authoritative
- public wording may widen one bounded scalar lane without claiming broad
  complex or mixed-precision maturity
- touched 64-bit maturity must read as stronger ABI/consumer correctness on
  touched paths, not as a claim that every product surface is fully 64-bit
  battle-hardened

## Explicit Non-Claim Fence

Sprint 94 will not claim:

- full-library complex support
- broad mixed-precision maturity
- templated-everywhere numeric genericity
- broad package/platform symmetry detached from the touched scalar and index
  seams

## Strongest Owner Split

The strongest direct-owner reading is now explicit:

- first-center public and implementation owners:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - the matching strongest shared implementation seam behind that public
    scalar owner

- second-center index/ABI owners if truly forced:
  - touched width-aware public headers
  - touched formatting, allocation, and consumer-proof owners

- third-center solver-family owners only if the widened scalar/index seam
  truly requires them:
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_qr.h`
  - `include/sparse_dense.h`
  - the matching implementation and proof owners

- later proof-only or support-only owners unless the first landing forces
  movement:
  - `benchmarks/bench_svd.c`
  - `benchmarks/bench_eigs.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`

## Deferred From The First Capability Landing

The first batch now explicitly defers:

- fake whole-library complex support
- fake broad mixed-precision maturity
- generic family-wide numeric rewriting
- benchmark-only widening detached from reviewed proof owners
- package/workflow narrative churn detached from one real scalar/index seam

## Exit State

- Sprint 94 has one explicit scalar/index capability contract before code
  movement.
- Day 5 is fixed to the bounded public scalar-contract seam with touched
  index-width maturity and solver-family breadth explicitly later unless the
  first landing forces movement.
- Later proof, benchmark, and support work remains sequenced behind the first
  capability landing.
