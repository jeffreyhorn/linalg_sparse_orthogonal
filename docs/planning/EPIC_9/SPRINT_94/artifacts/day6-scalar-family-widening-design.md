# Sprint 94 Day 6: Scalar-Family Widening Design

## Purpose

Define the bounded implementation contract for Sprint 94's first scalar-family
widening seam so Day 7 can land one real capability step without widening
into broad solver-family, index, or support churn.

## Main Result

Sprint 94 now has one explicit scalar-widening implementation contract:

- exact first implementation center:
  - `include/sparse_types.h`
  - `src/sparse_matrix.c`

- exact widening target:
  - finish the first public-to-implementation scalar seam on the shared
    matrix-shell helper path so the public `sparse_scalar_t` contract stops
    reading as a naming-only preparation layer on this owner
  - keep the widening bounded to the matrix-shell helper, arithmetic, and
    storage seam rather than reopening every solver-family implementation
    owner at once

## Preserved Invariants

The preserved invariants are now fixed:

- the default reviewed build remains real-only and continues to use `double`
  semantics unless the widened contract explicitly says otherwise
- public API clarity must improve, not blur; callers should be able to see
  that the widened seam is real on the touched owner without inferring fake
  broad numeric genericity elsewhere
- touched width and ABI interpretation remains unchanged in Day 7; wider
  index maturity is still a later batch unless the scalar landing truly
  forces it
- touched proof owners stay deterministic and auditable

## Directly Forced Follow-Through

The strongest directly forced follow-through is now fixed to:

- `include/sparse_matrix.h`
- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`

## Explicitly Deferred From Day 7

The strongest explicitly deferred work is now fixed:

- `include/sparse_dense.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_qr.h`
- the matching solver-family implementation owners
- `docs/maintainer_guide.md` and broader public/support wording unless the
  first scalar landing truly changes the contract reading

## Strongest Clarification

The useful Day 6 clarification is now explicit:

- Day 7 should make the shared matrix-shell scalar seam real enough to
  support one credible capability widening step
- it should not be a broad complex-support claim
- it should not become a generic family-wide numeric rewrite

## Exit State

- Sprint 94 has one exact Day 7 implementation center fixed to
  `include/sparse_types.h` plus the shared matrix-shell implementation seam.
- The widened scalar target and preserved invariants are explicit before code
  moves.
- Later index/ABI and solver-family work remains clearly sequenced behind the
  first scalar landing.
