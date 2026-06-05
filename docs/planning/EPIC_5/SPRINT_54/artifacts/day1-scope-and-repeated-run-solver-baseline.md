# Sprint 54 Day 1 - scope and repeated-run solver baseline

Date: 2026-06-03
Branch: `sprint-54`

## Scope

Start Sprint 54 from the actual Epic 4 public repeated-run handle close state
and the Sprint 53 validated repo state, then reduce the next work to a bounded
repeated-run solver-lifecycle completion queue.

## Authoritative baseline

Sprint 54 starts from a preserved reviewed validation baseline:

- strongest local reviewed baseline: `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

This means Sprint 54 is not a validation-recovery sprint. It is a
support-boundary and solver-lifecycle completion sprint.

## What Epic 4 and Sprint 53 already proved

The following is already real before Sprint 54 begins:

- explicit public repeated-run iterative handles already exist
- explicit public repeated-run eigensolver handles already exist
- one-shot iterative/eigensolver APIs remain first-class supported entry points
- handle reuse already preserves allocation capacity/setup, not stale
  numerical Krylov / Ritz / search state
- the direct-solver lifecycle and CSC follow-through package already closed
  from a reviewed validated baseline in Sprint 53
- the repo already has repeated-run benchmark surfaces in:
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- the repo already has caller-facing repeated-run documentation in:
  - `README.md`
  - `examples/README.md`

Interpretation:

- Sprint 54 does not need to prove that public repeated-run support exists
- Sprint 54 needs to decide and align the final supported repeated-run solver
  boundary across the remaining families

## Actual Sprint 54 queue

The Sprint 54 project-plan items reduce to six bounded work classes:

1. public solver lifecycle audit
2. steady-state inclusion/exclusion decisions for remaining iterative and
   advanced solver families
3. iterative handle expansion where justified
4. eigensolver lifecycle tightening
5. repeated-run benchmark alignment to the final public support set
6. targeted regression, example, README, and validation closeout work

The strongest architectural narrowing is:

- keep the work centered on the existing public handle model
- complete or tighten solver-family support only where the lifecycle story
  stays clear and supportable
- document intentional exclusions honestly where support should stay bounded
- do not broaden into a generic solver-API redesign or a large tutorial
  rewrite

## Main hotspots

Highest-value touched surfaces at sprint start:

- public headers:
  - `include/sparse_iterative.h` = `718`
  - `include/sparse_eigs.h` = `680`
- main implementations:
  - `src/sparse_iterative.c` = `2361`
  - `src/sparse_eigs.c` = `3233`
  - `src/sparse_iterative_workspace_internal.c` = `215`
  - `src/sparse_eigs_workspace_internal.c` = `267`
- proof surfaces:
  - `tests/test_iterative.c` = `2865`
  - `tests/test_eigs.c` = `1329`
  - `tests/test_eigs_lobpcg.c` = `1196`
  - `benchmarks/bench_iterative_reuse.c` = `250`
  - `benchmarks/bench_eigs_reuse.c` = `202`
- caller-facing adoption:
  - `examples/example_iterative.c` = `144`
  - `examples/example_eigs.c` = `285`
  - `README.md` = `972`
  - `examples/README.md` = `116`
  - `docs/maintainer_guide.md` = `294`

Interpretation:

- the strongest risk seams cluster in the iterative public header/impl pair
  plus the repeated-run proof and docs/example surfaces around them
- the eigensolver side remains important, but the likely Sprint 54 work there
  is lifecycle tightening and support-boundary agreement rather than a broad
  new public shape

## Preserved fence

Sprint 54 still inherits the controlling compatibility and non-goal boundary:

- one-shot iterative/eigensolver APIs remain first-class peer entry points
- explicit repeated-run handles remain opt-in lifecycle surfaces
- handle reuse preserves allocation capacity/setup, not stale numerical state
- the direct-solver lifecycle fence from Sprint 50 remains intact
- no raw internal workspace layout exposure
- no broad solver-API redesign
- no broad tutorial/example corpus rewrite

## Conclusion

Day 1 fixes Sprint 54's real starting point:

- preserved reviewed baseline
- validated inherited public repeated-run handle state
- bounded repeated-run solver completion queue
- named iterative/eigensolver code, test, benchmark, and doc hotspots
- preserved compatibility and non-goal fence

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening Epic 4 or Sprint 50-53 architecture decisions.
