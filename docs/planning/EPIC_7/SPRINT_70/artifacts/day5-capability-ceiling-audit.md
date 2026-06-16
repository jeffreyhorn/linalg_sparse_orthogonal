# Sprint 70 Day 5: Capability Ceiling Audit I

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Reduce the broad Epic 7 "capability expansion" question to one ranked live
ceiling map so later modernization work starts from the strongest actual
product constraints rather than from a vague feature wishlist.

## Evidence Base

This audit is grounded in the current public capability surfaces and the Epic 7
review baseline:

- `include/sparse_types.h`
- `README.md`
- `include/sparse_eigs.h`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `docs/planning/EPIC_7/reviews/review-codex-2026-06-15.md`

## Ranked Capability Ceilings

### 1. First ceiling: 32-bit index width

The strongest current capability ceiling is still the global `idx_t` model:

- `typedef int32_t idx_t;`
- `IDX_MAX = INT32_MAX`

This is the strongest first ceiling because it affects all three layers at
once:

- public product caveat:
  - matrix dimensions and nnz cap at roughly 2.1 billion
- implementation-local assumption:
  - dimensions, permutations, symbolic structures, and many workspaces are
    defined in terms of `idx_t`
- compatibility/package implication:
  - widening this is not a hidden refactor; it changes the public typedef
    exported to downstream consumers

Why it ranks first:

- it is the broadest width limit in the current product
- it affects essentially every sparse family, not just one subsystem
- it is easier to isolate conceptually than broad scalar-type generalization

### 2. Second ceiling: real-only scalar support

The second strongest ceiling is the repo-wide real-only, double-precision
contract.

That ceiling is visible directly in public APIs:

- solver inputs and outputs use `double *`
- iterative callbacks use `double *`
- eigensolver outputs use `double *`
- factor/result structs store `double` payloads directly

Why it matters:

- it excludes complex-valued sparse workflows
- it excludes precision-product variants such as single precision or mixed
  precision
- it makes the library less competitive against broader sparse numerical
  platforms even where the current algorithms are otherwise solid

Why it ranks second instead of first:

- it is broader and more invasive than index-width widening
- it would touch nearly every public numerical contract simultaneously
- the proof, packaging, and migration burden is therefore higher

### 3. Third ceiling: symmetric-only sparse eigensolver scope

The current public eigensolver surface is explicitly symmetric:

- `sparse_eigs_sym(...)`
- grow-m Lanczos
- thick-restart Lanczos
- explicit LOBPCG
- shift-invert composition through LDL^T for symmetric problems

This is a real state-of-the-art positioning limit because:

- the library now has a credible symmetric sparse eigensolver story
- but it still has no public unsymmetric sparse eigensolver story

Why it ranks third:

- it is narrower than the global width and scalar-type ceilings
- it affects one major capability family instead of the entire product model
- the current symmetric path is already comparatively mature within its lane

## Public Caveats vs Internal Assumptions vs Compatibility Events

The Day 5 split is now explicit.

Public caveats:

- 32-bit matrix dimensions and nnz
- real-only double-precision numerics
- symmetric-only sparse eigensolver contract

Implementation-local assumptions:

- pervasive `idx_t` use in dimensions, nnz, permutations, and workspaces
- pervasive `double`-typed solver, iterative, and eigensolver storage
- eigensolver naming and result contracts specialized to symmetric problems

Compatibility/package implications:

- index-width widening is a public typedef and downstream rebuild event
- scalar-type widening is a larger API/ABI/product-line event still
- eigensolver-family widening is less global than scalar generalization, but
  still expands the public supported capability promise

This distinction matters because Epic 7 should not treat:

- type-width modernization
- scalar-model modernization
- algorithm-family expansion

as if they were one interchangeable "capability" batch.

## First Modernization Shortlist After Day 5

Sprint 70 now has one concrete capability shortlist:

1. strongest first modernization candidate:
   - 32-bit index-width ceiling
2. strongest second modernization candidate:
   - real-only scalar ceiling
3. narrower but still important capability candidate:
   - unsymmetric sparse eigensolver gap
4. later/deferred or support context:
   - wider precision-product ambitions beyond the first two ceilings
   - broader capability wishlist not yet justified by the current product
     contract

## Exit State

Sprint 70 Day 5 closes with one ranked capability map instead of a generic
expansion bucket:

- first:
  - index width
- second:
  - scalar model
- third:
  - eigensolver-family breadth

That gives Day 6 one exact job:

- separate the first realistic Epic 7 modernization lane from the larger
  deferred ambitions without blurring width, scalar, and algorithm expansion
  into one fake implementation target
