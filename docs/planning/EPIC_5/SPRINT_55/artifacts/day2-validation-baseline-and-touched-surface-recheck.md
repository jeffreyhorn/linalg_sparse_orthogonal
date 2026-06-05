# Sprint 55 Day 2 - validation baseline and touched-surface recheck

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Reconfirm the reviewed validation baseline and the exact iterative/eigensolver
rerun set Sprint 55's later extraction batches must preserve.

## Maintained reviewed baseline

The maintained Sprint 55 baseline remains:

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

- Sprint 55 should keep using the exact `strongest local reviewed baseline`
  wording
- the reviewed CMake count and Makefile/CMake parity remain the authoritative
  truthfulness anchors for the sprint

## Code-day validation boundary

The mandatory gate for later `*.c` / `*.h` extraction days remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial implementation ownership batches
remains:

- `make quality-review-full`

Interpretation:

- docs-only audit/design/summary days do not need the full code-day gate
- extraction batches that materially reshape implementation ownership should
  continue to run the stronger reviewed baseline path too

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

- Sprint 55 does not need to reopen any quality-contract documentation work on
  Day 2
- the maintained baseline language is already stable enough to carry forward
  unchanged

## Authoritative Sprint 55 rerun set

The main Sprint 55 follow-on binaries already present in `build/` are:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/test_minres`
- `./build/example_iterative`
- `./build/example_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

These are the default high-signal reruns for Sprint 55 extraction days.

Interpretation:

- the sprint can keep its validation focus on the iterative/eigensolver family
  surfaces actually touched by the decomposition work
- no broader default rerun set is required on Day 2

## Live proof/adoption surface sizes

The current high-signal proof/adoption surfaces are:

- `tests/test_minres.c` = `1588`
- `tests/test_eigs.c` = `1522`
- `tests/test_eigs_lobpcg.c` = `1196`
- `benchmarks/bench_iterative_reuse.c` = `370`
- `benchmarks/bench_eigs_reuse.c` = `253`
- `examples/example_iterative.c` = `144`
- `examples/example_eigs.c` = `285`

Interpretation:

- Sprint 55 extraction work should treat proof-surface parity as a first-class
  concern
- the rerun set is a real guard against behavior drift while ownership moves

## Conclusion

Day 2 leaves Sprint 55 with an explicit validation and rerun contract:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit `*.c` / `*.h` code-day gate
- explicit stronger reviewed-baseline default
- authoritative iterative/eigensolver rerun set from the live build tree

That is enough to move to the Day 3 `sparse_eigs.c` seam audit without
validation ambiguity.
