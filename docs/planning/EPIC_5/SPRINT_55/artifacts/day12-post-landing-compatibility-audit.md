# Sprint 55 Day 12 - post-landing compatibility audit

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Audit the landed Sprint 55 decomposition work against two explicit questions:

1. Did Sprint 55 preserve the validated public solver/lifecycle fence?
2. Did Sprint 55 improve source ownership in a real, defensible way?

## Compatibility result

The landed branch still matches the preserved Sprint 54 repeated-run solver
contract:

- one-shot solver APIs still read as first-class
- repeated-run handles still read as bounded opt-in paths
- iterative public repeated-run handles remain intentionally limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver public repeated-run handles remain intentionally limited to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- retained exclusions still read as intentional exclusions, not accidental
  omissions:
  - `BiCGSTAB`
  - block iterative workflows

The tutorial, examples README, benchmark README, top-level README, and the
public iterative/eigensolver headers all still agree on that boundary.

## Ownership result

The decomposition work now yields real source ownership improvements:

### Eigensolver side

- retained shared/front-door file:
  - `src/sparse_eigs.c` = `1534`
- extracted backend-owned files:
  - `src/sparse_eigs_lobpcg.c` = `401`
  - `src/sparse_eigs_thick_restart.c` = `914`
- retained shared internal declaration surface:
  - `src/sparse_eigs_internal.h` = `631`

Relative to the Day 1 baseline:

- `src/sparse_eigs.c`: `3233` -> `1534`

### Iterative side

- retained shared/front-door file:
  - `src/sparse_iterative.c` = `1985`
- extracted backend-owned file:
  - `src/sparse_iterative_minres.c` = `308`
- retained shared internal declaration surface:
  - `src/sparse_iterative_internal.h` = `79`

Relative to the Day 1 baseline:

- `src/sparse_iterative.c`: `2377` -> `1985`

Interpretation:

- Sprint 55 is no longer just conceptual decomposition
- the eigensolver and iterative ownership seams now exist as real permanent
  source files
- the retained main files are more orchestration-focused than the Day 1
  baseline

## Build-surface check

The build system agrees with the landed ownership split:

- `Makefile` includes:
  - `src/sparse_iterative_minres.c`
  - `src/sparse_eigs_lobpcg.c`
  - `src/sparse_eigs_thick_restart.c`
- `CMakeLists.txt` includes the same three files

This confirms the new ownership boundaries are maintained consistently across
both supported local build paths.

## Residual queue

The remaining follow-up work is real but not Sprint 55-blocking:

- later iterative decomposition:
  - `GMRES`
  - shared block-wrapper scaffolding
- later eigensolver cleanup:
  - more trimming of the retained `src/sparse_eigs.c`
  - possible future private-header taxonomy cleanup if it clearly improves
    maintainability
- still intentionally out of scope:
  - broad public API redesign
  - reopening the public repeated-run support boundary
  - turning `BiCGSTAB` into a Sprint 55 public-handle topic

## Day 13 checklist

Run final validation from the landed Day 12 state:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Targeted follow-ons:

- `./build/test_iterative`
- `./build/test_minres`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/example_iterative`
- `./build/example_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

## Conclusion

Sprint 55 Day 12 confirms that the landed branch preserved the public
solver/lifecycle fence and produced real ownership improvements rather than a
mechanical file shuffle.

No blocker-level compatibility drift remains before Day 13 validation.
