# Sprint 55 Day 1 - scope and large-source decomposition baseline

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Start Sprint 55 from the actual Sprint 54 repeated-run solver close state and
the Epic 5 large-source review queue, then reduce the next work to a bounded
large-source decomposition package centered on the two highest-value remaining
solver translation units.

## Authoritative baseline

Sprint 55 starts from a preserved reviewed validation baseline:

- strongest local reviewed baseline: `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

This means Sprint 55 is not a validation-recovery sprint. It is a
maintainability and ownership sprint.

## What Sprint 54 already proved

The following is already real before Sprint 55 begins:

- explicit steady-state repeated-run solver support boundaries already exist
- public iterative handles already exist for:
  - `CG`
  - `GMRES`
  - `MINRES`
- public eigensolver handles already exist and are directly proved for:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- one-shot solver APIs remain first-class supported entry points
- reuse already preserves allocation/setup capacity, not stale numerical state
- the repo already has repeated-run benchmark surfaces in:
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- the repo already has caller-facing repeated-run documentation in:
  - `README.md`
  - `examples/README.md`
  - `docs/tutorial.md`

Interpretation:

- Sprint 55 does not need to re-decide the public repeated-run support surface
- Sprint 55 needs to improve internal implementation ownership while preserving
  that already-validated public contract

## What the Epic 5 review and todo list already fixed as the next queue

The Epic 5 review and todo notes already point to the same bounded
maintainability problem:

- `src/sparse_eigs.c` remains a top-tier large-source hotspot
- `src/sparse_iterative.c` remains a top-tier large-source hotspot
- the right improvement shape is:
  - split by stable ownership seams
  - separate helper logic from orchestration
  - reduce stale sprint-history narrative in permanent implementation files

The live repo state now confirms that the review queue is still current:

- `src/sparse_eigs.c` = `3233` lines
- `src/sparse_iterative.c` = `2377` lines

Interpretation:

- Sprint 55 should treat the Epic 5 review as still live, not historical
- `src/sparse_iterative.c` has actually grown beyond the review snapshot and is
  now an even stronger extraction target than the original review implied

## Actual Sprint 55 queue

The Sprint 55 project-plan items reduce to seven bounded work classes:

1. `sparse_eigs.c` seam audit
2. eigensolver decomposition batch 1
3. eigensolver decomposition batch 2
4. `sparse_iterative.c` seam audit
5. iterative decomposition batch 1
6. historical comment reduction on touched permanent implementation files
7. validation and closeout

The strongest architectural narrowing is:

- keep the work centered on the two largest remaining solver translation units
- prefer helper-vs-orchestration ownership splits over generic mechanical file
  splits
- preserve the Sprint 54 public support boundary exactly
- do not broaden into public API redesign, new solver-family exposure, or
  large documentation rewrites

## Main hotspots

Highest-value touched surfaces at sprint start:

- public headers:
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `687`
- main implementations:
  - `src/sparse_iterative.c` = `2377`
  - `src/sparse_eigs.c` = `3233`
  - `src/sparse_iterative_workspace_internal.c` = `215`
  - `src/sparse_eigs_workspace_internal.c` = `267`
  - `src/sparse_iterative_internal.h` = `26`
  - `src/sparse_eigs_internal.h` = `620`
- proof surfaces:
  - `tests/test_iterative.c` = `2993`
  - `tests/test_eigs.c` = `1522`
  - `tests/test_eigs_lobpcg.c` = `1196`
  - `benchmarks/bench_iterative_reuse.c` = `370`
  - `benchmarks/bench_eigs_reuse.c` = `253`
- caller-facing adoption:
  - `examples/example_iterative.c` = `144`
  - `examples/example_eigs.c` = `285`
  - `README.md` = `987`
  - `docs/maintainer_guide.md` = `294`

Interpretation:

- the strongest implementation risk seams remain concentrated in the two large
  solver translation units
- the current small workspace-helper files are not yet enough decomposition on
  their own
- the proof surfaces have grown enough that extraction work must preserve
  benchmark/test parity deliberately

## Preserved fence

Sprint 55 still inherits the controlling compatibility and non-goal boundary:

- one-shot solver APIs remain first-class peer entry points
- repeated-run handles remain bounded opt-in lifecycle surfaces
- the supported iterative handle set remains:
  - `CG`
  - `GMRES`
  - `MINRES`
- the supported eigensolver handle set remains:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows remain intentionally outside the
  public repeated-run handle set
- no broad solver-API redesign
- no raw internal workspace-layout exposure
- no broad documentation rewrite

## Conclusion

Day 1 fixes Sprint 55's real starting point:

- preserved reviewed baseline
- inherited validated repeated-run solver support fence
- bounded large-source decomposition queue
- named eigensolver/iterative implementation and proof hotspots
- explicit non-goal fence against public API expansion

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening Sprint 54's public solver-lifecycle decisions.
