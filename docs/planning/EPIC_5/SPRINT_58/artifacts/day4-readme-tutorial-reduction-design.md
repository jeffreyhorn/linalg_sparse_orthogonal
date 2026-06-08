# Sprint 58 Day 4 - README/tutorial reduction design

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Freeze the first bounded README/tutorial simplification boundary by selecting
the exact top-level workflow sections Sprint 58 should reduce first, defining
the wording invariants those edits must preserve, and recording the non-goal
fence before any permanent prose changes land.

## Selected first landing

### Primary target: `README.md`

The first README landing should cover:

- top-level feature and workflow summaries near the front of the file
- repeated-run versus one-shot positioning where the public caller story is
  summarized
- brief benchmark/example summary wording that points users toward the stable
  product surfaces rather than sprint history
- explicit exclusions or bounded support statements that should remain visible
  but read more like product guidance

The first README landing should intentionally defer:

- deep CSC Cholesky / LDL^T historical performance narratives
- deep eigensolver backend chronology
- large benchmark-history sections
- large test-history inventories

### Paired target: `docs/tutorial.md`

The first tutorial landing should cover:

- concise workflow alignment around one-shot-first guidance
- clearer pointers to:
  - repeated direct lifecycle
  - iterative-handle opt-in paths
  - eigensolver-handle opt-in paths
- wording that keeps the tutorial aligned with the final shipped example and
  header story

The first tutorial landing should intentionally defer:

- broad structural reordering
- large new sections
- feature-deep expansion that duplicates README or examples

## Invariants

The Day 5 docs reduction must preserve:

- truthful workflow claims
  - one-shot APIs remain first-class
  - repeated-run paths remain bounded opt-in workflows
  - supported exclusions stay visible where they matter
- alignment with validated example and benchmark behavior
  - `example_analysis` remains the strongest direct repeated-run example
  - iterative-handle support remains `CG`, `GMRES`, `MINRES`
  - eigensolver-handle support remains grow-m, thick-restart, and explicit
    `LOBPCG`
  - benchmark workflow groupings remain anchored in the current drivers
- stable top-level navigability
  - the README remains the top-level product map
  - the tutorial remains the practical getting-started guide

## Cleanup policy

For the first docs batch:

- remove stale sprint-history narrative
- keep product-level guidance
- keep concise support-boundary caveats that matter to callers
- prefer shorter workflow wording over richer implementation commentary

## Explicit non-goals

Not part of Day 5:

- broad tutorial rewrite
- benchmark taxonomy rewrite
- example README cleanup
- public-header cleanup
- repo-wide normalization of every historical README section

## Landing checklist

1. Reduce the touched README front-door workflow sections first.
2. Keep tutorial edits smaller and clearly paired to the README changes.
3. Reconcile touched wording against:
   - `examples/example_analysis.c`
   - `examples/example_iterative.c`
   - `examples/example_ic_minres.c`
   - `examples/example_eigs.c`
   - `benchmarks/README.md`
4. Avoid touching deep historical sections whose cleanup belongs to later days.
5. Run targeted docs sanity checks after the patch lands.

## Conclusion

The first Sprint 58 docs reduction boundary is now explicit:

- primary target:
  - top-level `README.md` workflow framing
- paired alignment target:
  - bounded `docs/tutorial.md` workflow wording
- preserved invariants:
  - truthful workflow claims
  - example/benchmark alignment
  - stable top-level navigability
- explicit non-goals:
  - no broad rewrite
  - no benchmark/examples/header work yet

That is enough to move to Day 5 implementation without vague cleanup scope.
