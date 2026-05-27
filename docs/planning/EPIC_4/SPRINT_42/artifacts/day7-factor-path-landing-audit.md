# Sprint 42 Day 7 Artifact: Factor-Path Landing Audit

## Purpose

Choose the bounded factor-entry paths for the main Sprint 42 normalization
batches after the Day 5 internal factor-state seam and Day 6 shared
matrix-state guards are both live.

The goal is not to reopen every factorization family at once. The goal is to
decide where the new seams already make a safe next landing possible and where
Sprint 42 still needs a small bridge adapter first.

## Inputs Reviewed

This audit is grounded in the current live code and the prior Sprint 42
artifacts:

- `docs/planning/EPIC_4/SPRINT_42/artifacts/day2-lifecycle-seam-refresh-inventory.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day3-internal-handle-scaffolding-design.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day4-matrix-state-guard-helper-design.md`
- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `src/sparse_analysis.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`

## Current Seam Status After Days 5-6

### What is already in place

Sprint 42 now has two real internal seams:

1. private LU / Cholesky factor-state ownership publication
2. shared matrix-state validation helpers

That means the remaining question is no longer "where should handles exist at
all?" It is "which factor-entry families can now adopt those seams cleanly
without forcing broader public or cross-family redesign?"

### What is still intentionally not in place

Sprint 42 still does **not** have:

- a public explicit-handle rollout
- a redesigned `sparse_factors_t` public shape
- a unified factor payload model across LU / Cholesky / LDLT / QR / SVD
- a CSC-to-private-factor-state bridge for every backend

That keeps the main normalization queue smaller than a full lifecycle rewrite.

## Factor-Path Classification

### 1. Ready for direct normalization

These paths can take the next Sprint 42 batch directly.

#### LU one-shot matrix path

Primary files:

- `src/sparse_lu.c`

Why ready:

- Day 5 already inserted the private factor-state seam
- Day 6 already normalized solve-side factored-state checks
- the public API is still one-shot and matrix-centered, so internal ownership
  cleanup can land without touching installed headers

Main Day 8/9 opportunity:

- reduce remaining direct compatibility-field publication drift
- keep the matrix as wrapper surface while making the private factor-state seam
  more authoritative internally

#### Cholesky one-shot matrix path

Primary files:

- `src/sparse_cholesky.c`

Why ready:

- Day 5 already inserted the private Cholesky factor-state seam
- Day 6 already normalized original-state and solve-side factored-state gates
- the linked-list path is still the cleanest direct proof surface for Sprint 42

Main Day 8/9 opportunity:

- tighten the linked-list factor entry and compatibility publication path
- reconcile the CSC writeback path with the Day 5 factor-state seam where that
  can be done safely

### 2. Bridge paths needing minor local adapters

These are the main Sprint 42 bridge candidates: real lifecycle pressure is
present, but the path still needs a narrow adapter layer rather than a direct
internal-handle landing everywhere at once.

#### `sparse_factors_t` analyze-once bridge

Primary files:

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`

Why this is the main bridge seam:

- `sparse_factors_t` is already the public compatibility bridge
- the current payload is still matrix-centric:
  - `SparseMatrix *F`
  - plus LDLT-specific side arrays
- `sparse_factor_numeric()` still delegates to one-shot factorization routines,
  then repackages their results
- `sparse_factor_solve()` still reconstructs an LDLT wrapper ad hoc on the
  solve side

Why it is not yet a direct handle landing:

- the public bridge type is installed API
- Sprint 42 can normalize what it owns internally, but should not widen into a
  public redesign batch

Safe Sprint 42 starting point:

- normalize the private/implementation-side interpretation of factor payload
  ownership
- reduce ad hoc bridge assembly in `sparse_factor_numeric()`,
  `sparse_factor_solve()`, and `sparse_factor_free()`
- keep the public `sparse_factors_t` shape stable

#### LDLT analyze-once / CSC bridge seam

Primary files:

- `src/sparse_ldlt.c`
- `src/sparse_ldlt_csc.c`

Why this is a bridge case:

- LDLT is already an explicit factor-handle API at the public surface
- but its reordered / CSC / analysis-assisted workflows still cross several
  bridge layers:
  - temporary permuted matrix copies
  - CSC factor working formats
  - unpack / transplant steps back into `sparse_ldlt_t`

Why this is only a minor-adapter target for Sprint 42:

- LDLT is not the hidden matrix-as-factor-handle problem Day 5 was designed
  to address
- the real Sprint 42 value is making the analyze-once bridge and helper
  boundaries cleaner, not re-architecting LDLT itself

Safe Sprint 42 starting point:

- keep LDLT public semantics unchanged
- only take small bridge/ownership cleanups that help the
  `sparse_factors_t` path or the Day 10 contract-normalization pass

#### Cholesky CSC writeback seam

Primary files:

- `src/sparse_chol_csc.c`

Why this matters:

- the CSC path still republishes factor state directly into matrix
  compatibility fields
- it currently sits adjacent to, but not fully on, the Day 5 private
  factor-state seam

Why it is a minor-adapter case rather than a separate major batch:

- the public path is still the same Cholesky API
- the CSC writeback path is a backend-specific publication seam
- Sprint 42 only needs to reconcile it with the new internal state model, not
  redesign the CSC backend

### 3. Guard-complete or lower-priority lifecycle paths

These families are important, but they are not the highest-value Day 8/9
normalization targets.

#### QR

Primary files:

- `src/sparse_qr.c`

Current state:

- original-state guard adoption is already done
- QR already externalizes its factor/result state into `sparse_qr_t`

Interpretation:

- QR was an important Day 6 guard-adoption target
- it is not the main Day 8/9 ownership-normalization target

#### SVD

Primary files:

- `src/sparse_svd.c`

Current state:

- original-state guard adoption is already done
- SVD already externalizes results into `sparse_svd_t`

Interpretation:

- SVD is lifecycle-sensitive, but not matrix-handle-overloaded in the same way
  as LU / Cholesky
- its best Sprint 42 role is to stay stable while the direct factor and bridge
  seams get cleaned up

#### Analysis symbolic entry path

Primary files:

- `src/sparse_analysis.c`

Current state:

- symbolic analyze guards are already normalized

Interpretation:

- the symbolic entry itself is not the main problem anymore
- the numeric/result bridge (`sparse_factor_numeric` and friends) is the part
  that still matters for Sprint 42

## `sparse_factors_t` Readiness Assessment

### What is safe now

Sprint 42 can safely begin bounded bridge normalization around
`sparse_factors_t` because:

- the public bridge object already exists
- the Day 5 LU / Cholesky factor-state seam is live
- the Day 6 lifecycle-state guard seam is live
- the current bridge logic is concentrated in one implementation file:
  `src/sparse_analysis.c`

### What should remain out of scope

Sprint 42 should **not** attempt:

- public-field removal from `sparse_factors_t`
- a new installed-header bridge type
- unifying LU / Cholesky / LDLT payload layout under one public object
- deep QR / SVD bridge coupling just for symmetry

### Safe first normalization pressure

The best Sprint 42 bridge work is therefore:

- implementation-side cleanup of factor payload assembly
- clearer ownership handoff in factor/free/solve flows
- reduced ad hoc rebuilding of temporary wrapper state

## Day 8-10 Migration Order

### Day 8: direct matrix-mutating factor paths

Primary target set:

- `src/sparse_lu.c`
- `src/sparse_cholesky.c`

Focus:

- use the Day 5 internal factor-state seam more consistently
- remove remaining bespoke compatibility-publication drift where the live seam
  already supports cleanup
- keep behavior identical

Secondary bounded follow-on if the batch stays small:

- touched CSC writeback/publication alignment in `src/sparse_chol_csc.c`

### Day 9: analyze-once bridge normalization

Primary target set:

- `src/sparse_analysis.c`

Focus:

- start bounded `sparse_factors_t` bridge normalization
- reduce matrix-centric payload assembly drift in:
  - `sparse_factor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_factor_free(...)`
  - `sparse_refactor_numeric(...)`

Bounded secondary follow-on if needed:

- small LDLT bridge adapters that make the analyze-once bridge cleaner without
  reopening the LDLT public API

### Day 10: cancellation and mutation contract normalization

Primary target set:

- LU / Cholesky factor paths
- touched analyze-once bridge paths

Focus:

- make cancellation and mutation expectations line up with the now-live
  ownership seams
- keep public semantics stable
- avoid broad docs churn

## Explicit Deferred / Later Paths

These are intentionally not the main Sprint 42 factor-path landing set:

- broad QR ownership refactoring
- broad SVD ownership refactoring
- deep CSC backend redesign
- public explicit-handle rollout
- installed-header lifecycle rewrites

Those remain later Epic 4 lifecycle-phase work, not the bounded Sprint 42
normalization queue.

## Day 7 Conclusions

1. LU and Cholesky are the highest-value direct normalization targets because
   the private factor-state seam is already live there.
2. `sparse_factors_t` is the main Sprint 42 bridge seam, not QR or SVD.
3. LDLT is mostly a bridge-adapter follow-on, not a primary ownership-rewrite
   target.
4. QR and SVD are already "guard-complete enough" for Sprint 42's next
   batches.
5. Days 8-10 now have a concrete landing order:
   - Day 8: LU / Cholesky direct path normalization
   - Day 9: `sparse_factors_t` bridge normalization
   - Day 10: cancellation / mutation contract cleanup
