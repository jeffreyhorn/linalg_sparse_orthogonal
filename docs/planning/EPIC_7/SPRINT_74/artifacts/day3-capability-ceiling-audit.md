# Sprint 74 Day 3: Capability Ceiling Audit

Date: 2026-06-16
Branch: `sprint-74`

## Purpose

Reduce the broad Sprint 74 "capability modernization" question to one ranked
live ceiling map so the sprint works from the strongest current product
constraints rather than from a vague feature wishlist.

## Evidence Base

This audit is grounded in the current public capability surfaces, the live
implementation seams, and the Epic 7 review baseline:

- `README.md`
- `include/sparse_types.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_svd.h`
- `include/sparse_analysis.h`
- `src/sparse_types.c`
- `src/sparse_matrix.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_svd.c`
- `docs/planning/EPIC_7/reviews/review-codex-2026-06-15.md`

## Ranked Capability Ceilings

### 1. First ceiling: 32-bit index width

The strongest current capability ceiling is still the global `idx_t` model:

- `typedef int32_t idx_t;`
- `IDX_MAX = INT32_MAX`

This remains the strongest first ceiling because it affects all three layers
at once:

- public product caveat:
  - matrix dimensions and nnz cap at roughly 2.1 billion
- implementation-local assumption:
  - dimensions, permutations, and many workspace/allocation calculations are
    defined in terms of `idx_t`
- compatibility/package implication:
  - widening this is a public typedef and downstream rebuild event

The strongest Day 3 clarification is now explicit:

- the first Sprint 74 question is not "can the repo become 64-bit everywhere
  at once?"
- it is "can the repo make the 32-bit ceiling non-permanent through one real
  bounded width-modernization seam?"

### 2. Second ceiling: real-only scalar support

The second strongest ceiling is still the repo-wide real-only, `double`
contract.

That ceiling is visible directly in the live public and implementation
surfaces:

- iterative callback signatures still expose `const double *` and `double *`
- eigensolver outputs and working arrays remain `double`-typed
- SVD options, results, and dense internal accumulators remain real-only

Why it matters:

- it excludes complex-valued sparse workloads
- it excludes broader precision-product variants
- it keeps the library materially narrower than broader state-of-the-art
  sparse numerical platforms

Why it ranks second instead of first:

- it is broader and more invasive than width modernization
- it touches nearly every public numerical contract simultaneously
- the proof, packaging, and migration burden is therefore still higher

The useful narrowing is that the strongest scalar-preparation center is not
the entire repo. It is the public callback/result and dense-kernel surfaces
where the real-only contract is most explicit and most reused.

### 3. Third ceiling: symmetric-only sparse eigensolver breadth

The current public eigensolver surface is still explicitly symmetric:

- `sparse_eigs_sym(...)`
- symmetric-only backend and repeated-run documentation

This remains a real state-of-the-art positioning limit because the library now
has a credible symmetric eigensolver story, but still no public unsymmetric
sparse eigensolver story.

Why it ranks third:

- it is narrower than the global width and scalar ceilings
- it affects one major capability family instead of the entire product model
- the current symmetric path is already comparatively mature within its lane

## Public Caveats vs Internal Assumptions vs Compatibility Events

Public product caveats:

- 32-bit matrix dimensions and nnz
- real-only double-precision numerics
- symmetric-only sparse eigensolver contract

Implementation-local assumptions:

- pervasive `idx_t` use in dimensions, permutations, and workspace sizing
- pervasive `double`-typed callbacks, vectors, result arrays, and dense
  kernels in iterative/eigs/SVD lanes
- eigensolver naming and result contracts specialized to symmetric problems

Compatibility/package implications:

- index-width widening is a public typedef and downstream rebuild event
- scalar-surface widening is a larger API/ABI/product-line event
- eigensolver-family widening expands the public supported capability promise
  without solving the broader width or scalar ceilings

## First Modernization Shortlist After Day 3

Sprint 74 now has one concrete capability shortlist:

1. strongest first modernization candidate:
   - index-width path
2. strongest second modernization candidate:
   - scalar-surface preparation and later broadening
3. narrower but still important capability candidate:
   - unsymmetric sparse eigensolver breadth
4. later/deferred or support context:
   - broader precision-product ambitions beyond the first two ceilings
   - wider algorithm-family wishlist not yet justified by the current product
     contract

## Exit State

Sprint 74 Day 3 closes with one ranked capability map instead of a generic
modernization bucket:

- first:
  - index width
- second:
  - scalar model
- third:
  - eigensolver-family breadth

That gives Day 4 one exact job:

- separate the first realistic Sprint 74 modernization fence from the larger
  deferred capability ambitions without blurring width, scalar, and
  algorithm-family expansion into one fake implementation target
