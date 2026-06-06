# Sprint 57 Day 2 - validation baseline and touched-surface recheck

Date: 2026-06-06
Branch: `sprint-57`

## Scope

Reconfirm the reviewed validation baseline and the exact giant-test, benchmark,
and caller-story rerun set Sprint 57's later refactor and regression batches
must preserve.

## Maintained reviewed baseline

The maintained Sprint 57 baseline remains:

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

- Sprint 57 should keep using the exact `strongest local reviewed baseline`
  wording
- the reviewed CMake count and Makefile/CMake parity remain the authoritative
  truthfulness anchors for the sprint

## Code-day validation boundary

The mandatory gate for later `*.c` / `*.h` giant-test refactor and regression
days remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial proof-surface batches remains:

- `make quality-review-full`

Interpretation:

- docs-only audit/design/summary days do not need the full code-day gate
- substantial test refactor and regression-expansion days should continue to
  run the stronger reviewed baseline path too

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

Interpretation:

- Sprint 57 does not need to reopen any quality-contract documentation work on
  Day 2
- the maintained baseline language is already stable enough to carry forward
  unchanged

## Authoritative Sprint 57 rerun set

The main Sprint 57 follow-on binaries already present in `build/` are:

- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/test_svd`
- `./build/test_qr`
- `./build/test_iterative`
- `./build/test_integration`
- `./build/bench_refactor_csc`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`
- `./build/example_analysis`

These are the default high-signal reruns for Sprint 57 implementation days.

Interpretation:

- the sprint can keep its validation focus on the direct-solver giant tests,
  large repeated-run solver proofs, and caller-story surfaces actually touched
  by the planned refactor and regression work
- no broader default rerun set is required on Day 2

## Live proof/adoption surface sizes

The current high-signal proof/adoption surfaces are:

- `tests/test_chol_csc.c` = `4643`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_svd.c` = `3746`
- `tests/test_qr.c` = `3197`
- `tests/test_iterative.c` = `2993`
- `tests/test_integration.c` = `1803`
- `benchmarks/bench_refactor_csc.c` = `611`
- `benchmarks/bench_iterative_reuse.c` = `370`
- `benchmarks/bench_eigs_reuse.c` = `253`
- `examples/example_analysis.c` = `210`

Interpretation:

- Sprint 57 refactor work should treat proof-surface parity and caller-story
  stability as first-class concerns
- the rerun set is a real guard against behavior drift while ownership and
  coverage structure change

## Conclusion

Day 2 leaves Sprint 57 with an explicit validation and rerun contract:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit `*.c` / `*.h` code-day gate
- explicit stronger reviewed-baseline default
- authoritative giant-test and caller-story rerun set from the live build tree

That is enough to move to the Day 3 direct-solver giant-test audit without
validation ambiguity.
