# Sprint 54 Day 2 - validation baseline and touched-surface recheck

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Reconfirm the maintained reviewed validation baseline Sprint 54 must preserve,
then define the smallest authoritative rerun set for later iterative and
eigensolver lifecycle code days.

## Reviewed baseline remains exact

The maintained Sprint 54 reviewed baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The authority split is still explicit and unchanged:

- `make quality-review-full`
  - strongest local reviewed baseline
- `make quality-review`
  - reviewed Makefile path
- `make quality-review-cmake`
  - reviewed CMake parity path
- `make deadcode-check`
  - report-completeness gate, not a zero-findings gate

Interpretation:

- Sprint 54 should preserve the exact reviewed wording and counts already used
  by the live repo
- substantial public lifecycle batches should continue to treat reviewed CMake
  parity as a truthfulness anchor

## Code-day validation rule

The mandatory gate for later `*.c` / `*.h` solver-lifecycle work remains:

- `make format`
- `make lint`
- `make test`

The stronger default for substantial public repeated-run API or solver-family
integration batches remains:

- `make quality-review-full`

This is enough. Sprint 54 does not need a custom validation regime.

## Main repeated-run follow-on binaries

The highest-signal rerun binaries for the already-supported repeated-run public
handle paths are present in `build/`:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/example_iterative`
- `./build/example_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Interpretation:

- these are the right default targeted reruns when Sprint 54 touches the
  current public handle story

## Remaining-family decision surfaces

The families most likely to be included, excluded, or clarified in Sprint 54
already have live proof/adoption surfaces too:

- `./build/test_minres`
- `./build/test_bicgstab`
- `./build/bench_bicgstab`
- `./build/example_ic_minres`

Current surface sizes:

- `tests/test_minres.c` = `1588`
- `tests/test_bicgstab.c` = `1586`
- `benchmarks/bench_bicgstab.c` = `173`
- `examples/example_ic_minres.c` = `232`

Interpretation:

- MINRES and BiCGSTAB are not just abstract design questions
- if Sprint 54 changes their public lifecycle story, these binaries belong in
  the targeted rerun set

## Authoritative Sprint 54 rerun list

The authoritative rerun boundary for later Sprint 54 work is:

- reviewed anchors:
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
- mandatory code-day gate:
  - `make format`
  - `make lint`
  - `make test`
- supported public-handle follow-ons:
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
  - `./build/example_iterative`
  - `./build/example_eigs`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`
- remaining-family decision follow-ons:
  - `./build/test_minres`
  - `./build/test_bicgstab`
  - `./build/bench_bicgstab`
  - `./build/example_ic_minres`

## Conclusion

Day 2 closes with a simple validation contract:

- keep the reviewed baseline wording and parity anchors exact
- use the normal `make format` / `make lint` / `make test` gate on code days
- default to `make quality-review-full` on substantial public lifecycle
  batches
- rerun both the already-supported public handle surfaces and the
  MINRES/BiCGSTAB decision surfaces when later work justifies them

That is enough to start the Day 3 public solver lifecycle audit from a clean,
explicit validation and touched-surface boundary.
