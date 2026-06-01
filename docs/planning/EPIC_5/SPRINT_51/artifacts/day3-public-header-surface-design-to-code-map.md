# Sprint 51 Day 3: Public Header Surface Design-To-Code Map

## Purpose

Day 3 converts the Sprint 50 direct-solver lifecycle contract into a concrete
header-edit map for Sprint 51 Phase 1. The goal is to identify exactly what
must land in `sparse_analysis.h`, `sparse_lu.h`, `sparse_cholesky.h`, and
`sparse_ldlt.h`, while keeping the shared repeated-run contract and the
family-local one-shot wording boundaries explicit before the first header patch
lands.

## Main Day 3 Conclusion

The Phase-1 direct-lifecycle header landing is smaller than a generic “touch
the direct headers” batch:

- `sparse_analysis.h` remains the shared repeated-run direct contract home
- `sparse_lu.h` and `sparse_cholesky.h` remain one-shot-first, mutable-matrix
  family headers
- `sparse_ldlt.h` remains the owned-factor family header
- the family headers should add bounded relationship wording and
  cross-references rather than duplicate the full repeated-run lifecycle prose

That keeps Sprint 51 aligned with the Sprint 50 fence:

- no broad direct-handle redesign
- no demotion of one-shot APIs
- no raw internal storage exposure
- no repeated-run overstatement that reuse preserves old numeric factor state

## Header-By-Header Landing Map

### `include/sparse_analysis.h`

This header is already the public repeated-run direct anchor. It should own the
shared Phase-1 lifecycle wording:

- zero/init of `sparse_analysis_t` and `sparse_factors_t`
- analyze once
- factor / solve
- refactor / solve many
- explicit free
- same-pattern reuse semantics
- object ownership boundaries:
  - analysis owns symbolic/permutation setup
  - factors own numeric factor state
  - neither object owns the source matrix

Recommended Day 4 emphasis:

- sharpen the shared repeated-run wording where it is currently implicit
- keep the existing workflow example as the highest-signal public pattern
- avoid broadening this into backend-layout or internal-kernel promises

### `include/sparse_lu.h`

This header should remain family-local and one-shot-first. It already carries
the important compatibility truths:

- factorization is in-place on a copied matrix
- callers should use `sparse_copy()` to preserve the original
- one-shot factor / solve remains the simple/default path

Recommended Day 4 changes:

- add a bounded cross-reference to `sparse_analysis.h` for stable-pattern
  repeated runs
- keep the copy-before-factor guidance explicit
- do not restate the full shared repeated-run lifecycle in LU-local prose

### `include/sparse_cholesky.h`

This header should also remain family-local and one-shot-first, with even more
care around matrix mutation:

- lower triangle is overwritten with `L`
- upper triangle entries are removed
- reordered one-shot factorization remains a first-class entry point

Recommended Day 4 changes:

- add a bounded repeated-run direct-path cross-reference
- preserve the visible mutable-matrix / copy-before-factor guidance
- do not hide the one-shot SPD path behind shared lifecycle abstractions

### `include/sparse_ldlt.h`

This header already exposes a distinct owned-factor object model through
`sparse_ldlt_t`. That family-local shape should remain intact.

Recommended Day 4 changes:

- clarify the relationship between:
  - the family-local `sparse_ldlt_t` path
  - the shared `sparse_analysis_t` / `sparse_factors_t` repeated-run path
- keep backend, tolerance, inertia, and solve details local
- avoid replacing the existing LDL^T factor-object surface with generic shared
  handle language

## Shared vs Family-Local Contract Boundary

### Shared repeated-run contract belongs in `sparse_analysis.h`

Keep these concepts centralized there:

- analyze once
- factor / solve
- refactor / solve many
- same-pattern reuse
- explicit free
- analysis/factors object ownership boundaries

### Family-local truth belongs in LU / Cholesky / LDL^T headers

Keep these concepts local:

- in-place mutation and copy-before-factor guidance
- LU pivoting and reorder options
- Cholesky SPD and lower-triangular overwrite behavior
- LDL^T factor-object ownership and inertia helpers
- backend / telemetry family details
- one-shot convenience path wording

## True Phase-1 Header Batch Boundary

The real first header batch is:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

The following are intentionally later documentation-only follow-ons, not Day 4
scope:

- `README.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/tutorial.md`

## Day 3 Operational Result

Sprint 51 no longer needs to rediscover what the first header batch means.
Phase 1 is now reduced to:

- one shared repeated-run contract home
- three family-local relationship adjustments
- explicit preservation of one-shot compatibility wording

## Highest-Value Day 3 Conclusions

1. `sparse_analysis.h` is the shared repeated-run direct vocabulary home and
   should stay that way.
2. LU and Cholesky headers should stay one-shot-first and visibly mutation-aware.
3. LDL^T already has an owned-factor surface; its Sprint 51 work is mainly
   relationship wording, not lifecycle invention.
4. The first header batch is bounded enough to implement without reopening
   README, benchmark-doc, or tutorial scope.
