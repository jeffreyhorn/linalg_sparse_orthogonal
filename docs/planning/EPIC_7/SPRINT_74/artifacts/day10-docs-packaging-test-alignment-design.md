# Sprint 74 Day 10: Docs / Packaging / Test Alignment Design

## Objective

Fix the smallest maintained-surface follow-through actually required by the
Day 6 width-contract landing and the Day 9 scalar-surface landing, without
turning Sprint 74 into a generic docs or packaging cleanup pass.

## Inputs Re-read

- `docs/planning/EPIC_7/SPRINT_74/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/WORKING_NOTES.md`
- `docs/planning/EPIC_7/SPRINT_74/artifacts/day6-index-width-integration-batch1.md`
- `docs/planning/EPIC_7/SPRINT_74/artifacts/day9-scalar-surface-preparation-batch.md`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `include/sparse_types.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

## Design Result

### 1. The required follow-through is capability wording, not install/package wording

The landed Sprint 74 contract changed in two exact ways:

- width selection is now a named compile-time contract through
  `SPARSE_IDX_BITS`
- the strongest public real-only scalar seam now routes through
  `sparse_scalar_t`

That moves the capability wording, but it does not move the install/export or
reviewed-platform contract.

### 2. `README.md` is the required public follow-through surface

`README.md` still owns the strongest caller-facing capability summary.

It now needs to say directly that:

- the default reviewed build still ships the 32-bit `idx_t` lane
- wider indices are the bounded compile-time modernization seam, not a manual
  typedef-edit story
- the current shipped scalar lane remains real-only
- `sparse_scalar_t` is the public dense-scalar owner on the touched iterative
  and eigs seam

### 3. `docs/maintainer_guide.md` is the required policy follow-through surface

The maintainer guide should own the narrower Sprint 74 interpretation:

- current shipped support remains narrower than the broader Epic 7 target
- the landed width seam is compile-time and bounded
- the landed scalar seam is public-contract preparation only
- later scalar breadth and later algorithm-family widening remain deferred

### 4. `INSTALL.md` and the touched headers are support-only, not Day 11 centers

No install or packaging claim moved in the landed code:

- no platform-lane interpretation changed
- no install/export behavior changed
- no ABI claim widened

The touched public headers already read truthfully:

- `include/sparse_types.h` states the compile-time width and scalar owners
- `include/sparse_iterative.h` and `include/sparse_eigs.h` already use
  `sparse_scalar_t` on the touched public seam

### 5. Proof-owner follow-through belongs to Day 12 unless a contradiction appears

The touched proof owners are already clear:

- `tests/test_sparse_matrix.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`

That proof map may need explicit maintained-surface alignment later, but it is
not the Day 11 batch center.

## Exact Day 11 Touch Set

Required:

- `README.md`
- `docs/maintainer_guide.md`

Support only if wording truly forces it:

- `INSTALL.md`
- `include/sparse_types.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

Explicit non-touch set:

- implementation `src/` files
- proof-owner test files
- benchmark docs and benchmark binaries
- platform/install workflow files
- later capability headers such as `include/sparse_svd.h`

## Preserved Truthfulness Checklist

Day 11 must preserve:

- reviewed builds still default to the 32-bit `idx_t` lane
- `SPARSE_IDX_BITS` is a bounded compile-time widening seam, not proof that
  the whole repo is now broadly 64-bit-modernized
- current shipped scalar support remains real-only
- `sparse_scalar_t` is a bounded public preparation seam, not a broader
  generic-scalar or complex-support claim
- install/export and reviewed-platform claims stay unchanged
- proof ownership stays with the focused test surfaces, not with docs or
  examples

## Bottom Line

Sprint 74 Day 10 narrows the maintained follow-through batch to one public
surface and one policy surface:

- `README.md`
- `docs/maintainer_guide.md`

Everything else is support-only unless the Day 11 wording pass proves
otherwise.
