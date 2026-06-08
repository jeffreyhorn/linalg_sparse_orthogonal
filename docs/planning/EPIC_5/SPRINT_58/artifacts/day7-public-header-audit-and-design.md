# Sprint 58 Day 7 - public header audit and design

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Reduce the public-header cleanup problem to a bounded offender list by auditing
the strongest API-adjacent narrative hotspots directly, separating the real
cleanup classes, ranking the touched headers by caller visibility and risk, and
fixing the exact Day 8 header set plus wording invariants before any permanent
header edits land.

## Audited headers

Primary public-header surfaces:

- `include/sparse_eigs.h` = `687`
- `include/sparse_iterative.h` = `765`
- `include/sparse_analysis.h` = `375`
- `include/sparse_lu.h` = `337`
- `include/sparse_cholesky.h` = `204`
- `include/sparse_ldlt.h` = `334`

These were audited using:

- targeted `rg` scans for sprint-history, future-work, repeated-run, and
  support-boundary wording
- direct reads of the highest-signal overview and lifecycle sections
- comparison against the simplified README/tutorial wording landed in Days 5-6

## Drift shapes

### `include/sparse_eigs.h`

Live cleanup classes:

- stale sprint chronology
- stale future-work wording
- overlong lifecycle explanation
- terminology mismatch with the current README/tutorial story

Key fact:

- this is the strongest public-header narrative offender in the repo and the
  clearest Day 8 first target

### `include/sparse_iterative.h`

Live cleanup classes:

- light stale sprint chronology
- overlong repeated-run lifecycle explanation
- terminology mismatch around the support boundary

Key fact:

- this is the strongest companion surface because it exposes the repeated-run
  handle boundary most visibly to callers

### `include/sparse_analysis.h`

Live cleanup classes:

- overlong lifecycle explanation
- smaller terminology alignment opportunity

Key fact:

- this is a plausible third target, but it is lower-risk and more deferrable
  than the two stronger surfaces

### Direct-family headers

Observed surfaces:

- `include/sparse_cholesky.h`
- `include/sparse_lu.h`
- `include/sparse_ldlt.h`

Key fact:

- these now read comparatively better and should stay deferred unless the Day 8
  cleanup exposes a real contradiction

## Ranked cleanup order

### Rank 1: `include/sparse_eigs.h`

Why it ranks first:

- highest overlap of cleanup classes
- strongest stale sprint-history burden
- strongest future-work and tuning-local commentary burden
- strongest mismatch with the newly simplified top-level docs

### Rank 2: `include/sparse_iterative.h`

Why it ranks second:

- carries the main repeated-run handle support boundary
- should align tightly with README/tutorial wording
- easier bounded companion target than reopening more direct-family headers

### Rank 3: `include/sparse_analysis.h`

Why it ranks third:

- still visibly verbose at the public-header layer
- useful if the Day 8 batch can stay bounded
- not necessary if it would cause the batch to sprawl

## Selected Day 8 set

Selected touched-header set:

1. `include/sparse_eigs.h`
2. `include/sparse_iterative.h`
3. `include/sparse_analysis.h` only if the batch remains tight and clearly
   aligned to the README/tutorial wording

## Invariants

The Day 8 header cleanup must preserve:

- API semantics
- ownership truth
- concise behavioral comments that still matter to callers
- support-boundary wording aligned with the current README/tutorial story

The Day 8 header cleanup should remove:

- stale sprint-history narrative
- stale future-sprint wording
- overlong narrative better suited to planning or deeper docs

## Conclusion

The public-header cleanup problem is now concrete:

- `include/sparse_eigs.h` is the strongest first target
- `include/sparse_iterative.h` is the strongest repeated-run companion target
- `include/sparse_analysis.h` is an optional third target if the batch stays
  bounded
- direct-family headers remain intentionally deferred unless a contradiction
  appears during the Day 8 landing

That is enough to move to Day 8 implementation without vague header scope.
