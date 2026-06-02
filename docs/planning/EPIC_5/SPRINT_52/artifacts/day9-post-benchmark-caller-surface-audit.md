# Sprint 52 Day 9: Post-Benchmark Caller-Surface Audit

## Purpose

Day 9 narrows the remaining adoption work after the Sprint 52 integration and
benchmark-proof batches. The goal is not to start editing every visible docs
surface; it is to identify which caller-facing surfaces actually need a Phase 2
refresh and which ones are already aligned enough to leave alone.

## Main Day 9 Conclusion

The remaining queue is smaller than the generic Sprint 52 plan implied:

- `README.md` is the strongest remaining user-facing adoption target
- `examples/example_analysis.c` is the strongest remaining shipped-example
  adoption target
- `examples/README.md` and `benchmarks/README.md` are already aligned enough
  that they do not justify broadening Day 10 by default
- tutorial-scale or sweeping example conversion remains out of scope

## Audited Surfaces

The Day 9 caller-surface audit focused on:

- `README.md`
- `examples/example_analysis.c`
- `examples/README.md`
- `benchmarks/README.md`

To keep the audit honest against the live public contract, the following
contract homes were also rechecked:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

## Surface-by-Surface Findings

### `README.md`

Current state:

- already advertises the analyze-once / factor-many workflow
- already lists the main lifecycle functions in the API overview
- already keeps one-shot APIs as first-class user-facing entries

Remaining drift:

- the repeated-run direct story is still terse relative to the stronger Sprint
  52 Phase 2 behavior
- it does not yet foreground the key Phase 2 boundary in one compact place:
  - analyze once
  - factor / solve
  - refactor / solve many
  - reuse preserves symbolic/permutation setup, not old numeric factor state
  - the library now rejects obvious gross-structure drift cheaply, but does
    not promise a full structural-pattern verifier

Day 10 assessment:

- primary target

### `examples/example_analysis.c`

Current state:

- already demonstrates the correct public lifecycle:
  - zero-init analysis/factors objects
  - analyze once
  - factor / solve
  - refactor / solve many
- already rebuilds fresh matrices with the same pattern and different values,
  which matches the public contract well

Remaining drift:

- its explanatory framing still reads more like the Sprint 50/51 contract than
  the stronger Sprint 52 Phase 2 wording
- it should explain more clearly:
  - what state is actually reused
  - why fresh same-pattern matrices remain the safe example discipline
  - that reuse does not mean reusing stale numeric factor contents

Day 10 assessment:

- primary target

### `examples/README.md`

Current state:

- already names `example_analysis` explicitly
- already describes it as the strongest repeated-run direct example
- already keeps one-shot examples as first-class simpler entry points

Remaining drift:

- no important Phase 2 drift surfaced
- a tiny supporting tweak could be justified later only if the example source
  gets a stronger explanatory angle that should be mirrored here

Day 10 assessment:

- leave alone by default
- optional tiny supporting touch only if needed

### `benchmarks/README.md`

Current state:

- Day 8 already aligned it with the real same-pattern value-changing repeated-
  run proof
- already distinguishes `bench_refactor` from `bench_refactor_csc`
- already states the measured-output contract clearly enough

Remaining drift:

- none that justifies more adoption-surface work in Sprint 52 Day 10

Day 10 assessment:

- leave alone

## Day 10 Adoption Boundary

The strongest bounded Day 10 package is now clear:

- primary targets:
  - `README.md`
  - `examples/example_analysis.c`
- optional tiny supporting touch:
  - `examples/README.md`

The following should stay out of scope:

- broad tutorial rewrite
- sweeping conversion of one-shot examples into repeated-run examples
- benchmark framework or benchmark README expansion beyond tiny supporting
  wording
- changing the lifecycle contract itself instead of reflecting it
- reopening LU as anything other than the intentionally bounded special-case
  seam

## Deferred / Secondary Surfaces

These surfaces are not blocked and do not need Sprint 52 Day 10 by default:

- `examples/README.md`
- `benchmarks/README.md`
- `docs/tutorial.md`
- the smaller one-shot examples

They can stay secondary unless a small supporting clarification becomes
necessary while landing the two primary Day 10 targets.

## Day 9 Operational Result

Sprint 52 now has a concrete adoption queue instead of a generic one:

1. the benchmark proof and shared repeated-run direct contract are already in
   good shape
2. the highest-value remaining public adoption work is concentrated in the
   top-level README and `example_analysis`
3. the rest of the caller-facing surfaces are aligned enough to avoid
   accidental scope creep

That gives Day 10 a tight, high-signal boundary rather than a diffuse docs
cleanup task.
