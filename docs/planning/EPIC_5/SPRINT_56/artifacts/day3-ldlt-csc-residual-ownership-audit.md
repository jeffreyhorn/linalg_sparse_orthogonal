# Sprint 56 Day 3 - LDLT CSC residual ownership audit

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Reduce `src/sparse_ldlt_csc.c` to a concrete extraction map before code
movement begins.

## Live ownership bands

The current `src/sparse_ldlt_csc.c` function map separates into five real
ownership bands:

1. lifecycle / storage / structural conversion
   - alloc / free
   - row-adjacency growth
   - supernode detection
   - sparse-to-CSC conversion
   - analysis-aware sparse-to-CSC conversion
   - CSC-to-sparse conversion
   - writeback to `sparse_ldlt_t`
   - validation
2. wrapper / compatibility path
   - linked-list expansion helper
   - wrapper elimination path
3. scalar/native LDLT CSC kernel core
   - symmetric swap
   - workspace alloc/free
   - Bunch-Kaufman scan helpers
   - scatter / lookup / cmod helpers
   - one-step elimination
   - native elimination driver
   - solve path
4. supernodal LDLT CSC helper cluster
   - supernode extract
   - supernode dense writeback
   - diagonal-block eliminate
   - panel eliminate
   - supernodal elimination driver
5. small local support seams
   - row-map binary search
   - dense-column clear helpers

Interpretation:

- the file is large, but it is no longer ambiguous
- the real question is which of these ownership bands should move first

## Strongest first extraction target

The strongest first extraction target is the supernodal LDLT CSC helper
cluster:

- it is already grouped contiguously near the end of the file
- it has its own vocabulary and dense-workflow identity
- it offers meaningful line-count and readability relief
- it avoids immediately reopening the scalar/native Bunch-Kaufman core

Why it outranks the scalar/native kernel for Batch 1:

- lower coupling to the public compatibility-facing entry surface
- clearer proof boundary in the CSC-specific tests
- better maintainability gain per unit risk

## Strongest second extraction target

The scalar/native elimination kernel is the strongest second seam:

- it contains the largest remaining ownership mass
- it is behaviorally important
- it is more intertwined than the supernodal cluster

Why it should come second rather than first:

- it couples symmetric swap, cmod, row-adjacency, elimination, and solve logic
- a first-batch split there would carry a higher behavior-drift risk
- it becomes easier to reason about once the supernodal cluster is removed

## Lower-priority seams

### Conversion / validation / writeback

This is real ownership, but it is a weaker first-batch target:

- less numerically cohesive than the supernodal cluster
- less line-count relief where the file currently feels heaviest
- more risk of a low-value mechanical split

### Wrapper / compatibility path

This remains intentionally secondary:

- already bounded
- important for compatibility and A/B proof
- not where most maintainability weight currently lives

### Small helper cleanup

This is later cleanup work, not a first extraction target:

- useful only after larger ownership bands move
- too small to justify leading the sprint

## Proof-surface implications

The proof surfaces reinforce the CSC-native ownership boundaries:

- `tests/test_ldlt_csc.c` is still the main file-level proof surface
- `tests/test_integration.c` still proves the public repeated-run direct path
- `benchmarks/bench_refactor_csc.c` directly names the CSC completion seam and
  carries both SPD and indefinite repeated-run evidence

That means the best extraction seams are the ones those proof surfaces already
imply exist. Utility-first slicing would be harder to validate and harder to
explain.

## Ranked extraction order

1. supernodal LDLT CSC helper cluster
2. scalar/native elimination kernel cluster
3. conversion / validation / writeback cluster
4. wrapper / compatibility cluster
5. small residual helper cleanup

## Recommended Batch 1 boundary

Start Sprint 56 implementation work by extracting the supernodal LDLT CSC
helper cluster into an owned file.

Keep in `src/sparse_ldlt_csc.c` for the first batch:

- lifecycle/conversion entry points
- wrapper compatibility path
- scalar/native elimination kernel
- solve path

## Conclusion

Day 3 turns the LDLT CSC large-file problem into a concrete decomposition map
with one clear first target and an explicit reason not to start with the
largest residual kernel band.
