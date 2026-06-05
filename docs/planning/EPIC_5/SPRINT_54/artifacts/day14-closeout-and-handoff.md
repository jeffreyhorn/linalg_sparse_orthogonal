# Sprint 54 Day 14 - Closeout and handoff

Date: 2026-06-03
Branch: `sprint-54`

## Summary

Sprint 54 closes the public repeated-run solver lifecycle completion work from
a measured validated baseline rather than from partial support, partial proof,
or documentation intent.

The sprint started from the Sprint 53 validated post-CSC-follow-through state
and ended with an explicit steady-state repeated-run solver support boundary,
public MINRES handle support, tightened eigensolver handle proof across all
supported backends, aligned reuse benchmarks, and caller-facing docs that now
match the real public lifecycle surface.

## Delivered package

Sprint 54 leaves behind one coherent repeated-run solver package:

- explicit steady-state support-boundary decisions in:
  - `docs/planning/EPIC_5/SPRINT_54/WORKING_NOTES.md`
  - `docs/planning/EPIC_5/SPRINT_54/artifacts/day4-solver-surface-decision-batch.md`
- iterative repeated-run handle completion in:
  - `include/sparse_iterative.h`
  - `src/sparse_iterative.c`
  - `tests/test_iterative.c`
- tightened eigensolver repeated-run proof in:
  - `include/sparse_eigs.h`
  - `tests/test_eigs.c`
- aligned reuse-benchmark proof in:
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
  - `benchmarks/README.md`
- reconciled caller-facing repeated-run wording in:
  - `README.md`
  - `examples/README.md`
  - `docs/tutorial.md`

## Delivered repeated-run state

Sprint 54 closes the main Sprint 54 seams in a bounded way:

- the supported iterative public repeated-run handle set is now explicit and
  implemented as:
  - `CG`
  - `GMRES`
  - `MINRES`
- the supported eigensolver public repeated-run handle set is now explicit and
  directly proved as:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- the public reuse benchmarks now match those real supported sets
- the top-level README, examples README, and tutorial no longer imply a
  narrower or fuzzier repeated-run support surface than the code actually
  offers

## Preserved contract

Sprint 54 preserved the bounded solver-lifecycle fence established earlier in
Epic 4 and Epic 5:

- one-shot solver APIs remain first-class peer entry points
- repeated-run handles remain bounded opt-in paths rather than universal
  replacements
- reuse preserves allocation/setup capacity, not stale numerical Krylov, Ritz,
  or search-direction state
- `BiCGSTAB` remains intentionally outside the public repeated-run handle set
- block iterative workflows remain intentionally outside the public repeated-run
  handle set
- no broad solver-API redesign or public workspace-layout exposure was
  introduced

## Validation close state

Sprint 54 closes from the Day 13 validated baseline:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained truthfulness anchors:

- reviewed CMake parity = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `make quality-review-full` reviewed CMake total time = `144.25 sec`

Targeted Sprint 54 follow-ons also passed:

- `./build/test_iterative`
- `./build/test_minres`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Representative direct results:

- `bench_iterative_reuse`
  - `cg-tridiag-300 1.12x`
  - `gmres-unsym-220 0.85x`
  - `minres-kkt-42 1.28x`
- `bench_eigs_reuse`
  - `growm-nos4-k5 1.00x`
  - `thick-bcsstk14-k5 0.99x`
  - `lobpcg-diag40-k3 1.00x`
  - all three kept exact eigenvalue parity:
    - `|lambda|max diff = 0.000e+00`
- `example_eigs`
  - explicit `LOBPCG` on `bcsstk04`
  - `3 / 3` smallest eigenpairs
  - `62` outer iterations
  - residual `8.808e-09`

## Handoff to Sprint 55

Sprint 55 no longer needs to decide what the steady-state public repeated-run
solver support surface actually is.

The next bounded queue can therefore focus on real post-Sprint-54 work such
as:

- larger caller-teaching modernization if a later sprint wants explicit
  repeated-run examples beyond the current bounded docs updates
- any later public-handle expansion beyond the bounded Sprint 54 support set
- broader benchmark or caller-surface evolution built on the now-explicit
  solver-lifecycle fence

## Project-plan impact

Sprint 54 does not require a `PROJECT_PLAN.md` update.

Reason:

- the sprint closed from the planned Day 13 validated baseline
- the delivered package still matches the Epic 5 Sprint 54 intent
- no blocker or replanning queue surfaced during closeout

## Conclusion

Sprint 54 is complete. It hands off a validated repeated-run solver lifecycle
package with an explicit steady-state support boundary, public MINRES handle
support, fully aligned eigensolver-handle proof, matched public reuse
benchmarks, preserved one-shot-first compatibility, honest bounded exclusions,
and stable reviewed-baseline truthfulness anchors.
