# Sprint 58 Day 2 - validation baseline and touched-surface recheck

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Reconfirm the reviewed validation baseline and the exact example/benchmark and
public-surface rerun set Sprint 58's later docs/header/example cleanup batches
must preserve.

## Maintained reviewed baseline

The maintained Sprint 58 baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The authority split also remains unchanged:

- `make quality-review-full`
  - strongest local reviewed baseline
- `make quality-review`
  - reviewed Makefile path
- `make quality-review-cmake`
  - reviewed CMake parity path
- `make deadcode-check`
  - report-completeness gate, not a zero-findings gate

Interpretation:

- Sprint 58 should keep using the exact `strongest local reviewed baseline`
  wording
- the reviewed CMake count and Makefile/CMake parity remain the authoritative
  truthfulness anchors for the sprint

## Code-day validation boundary

The mandatory gate for later `*.c` / `*.h` public-header or example days
remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial shipped-surface batches remains:

- `make quality-review-full`

Interpretation:

- docs-only audit/design/summary days do not need the full code-day gate
- public-header or example code-touch days still do
- substantial shipped-surface batches should continue to use the stronger
  reviewed baseline path too

## Quality-contract wording recheck

The quality-contract wording remains aligned across:

- `README.md`
  - strongest local reviewed baseline command map
  - explicit `deadcode-check` completeness-gate meaning
- `docs/maintainer_guide.md`
  - maintainer-facing authority framing
  - reviewed CMake parity anchor
  - dead-code interpretation boundary
- `Makefile`
  - executable reviewed-target authority
  - rerun guidance
  - test-count parity checks
- GitHub workflows
  - reviewed CMake execution path
  - deadcode-check execution path

Interpretation:

- Sprint 58 does not need to reopen any quality-contract documentation work on
  Day 2
- the maintained baseline language is already stable enough to carry forward
  unchanged

## Authoritative Sprint 58 rerun set

The main Sprint 58 follow-on binaries already present in `build/` are:

- `./build/example_analysis`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

These are the default high-signal reruns for Sprint 58 implementation days.

Interpretation:

- the sprint can keep its validation focus on the examples and benchmark
  surfaces that directly teach or summarize the final public workflow story
- no broader default rerun set is required on Day 2

## Conclusion

Day 2 leaves Sprint 58 with an explicit validation and rerun contract:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit `*.c` / `*.h` code-day gate
- explicit stronger reviewed-baseline default
- authoritative example/benchmark rerun set from the live build tree

That is enough to move to the Day 3 public docs audit without validation
ambiguity.
