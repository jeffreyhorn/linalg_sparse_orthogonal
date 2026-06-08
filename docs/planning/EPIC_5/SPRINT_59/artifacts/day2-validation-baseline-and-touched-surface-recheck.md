# Sprint 59 Day 2 - validation baseline and touched-surface recheck

Date: 2026-06-08
Branch: `sprint-59`

## Scope

Reconfirm the reviewed validation baseline, the exact code-day gate, and the
targeted final-sprint rerun set Sprint 59's later quality/platform and Epic 5
closeout batches must preserve.

## Maintained reviewed baseline

The maintained Sprint 59 baseline remains:

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

- Sprint 59 should keep using the exact `strongest local reviewed baseline`
  wording
- the reviewed CMake count and Makefile/CMake parity remain the authoritative
  truthfulness anchors for the sprint

## Code-day validation boundary

The mandatory gate for later `*.c` / `*.h` quality/platform days remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial quality/platform follow-through
remains:

- `make quality-review-full`

Interpretation:

- docs-only audit/design/summary days do not need the full code-day gate
- any later Sprint 59 code-touching day still does
- substantial quality/platform batches should continue using the stronger
  reviewed baseline path too

## Quality-contract wording recheck

The quality-contract wording remains aligned across:

- `README.md`
  - strongest local reviewed baseline command map
  - explicit dead-code completeness-gate meaning
  - platform/status truthfulness wording
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
  - dead-code report/check execution path
  - coverage-contract execution path

Interpretation:

- Sprint 59 does not need to reopen the baseline language on Day 2
- the maintained quality contract is already stable enough to carry forward
  unchanged while the residual audit focuses on staged/excluded follow-through

## Authoritative Sprint 59 rerun set

The main Sprint 59 follow-on binaries already present in `build/` are:

- `./build/test_integration`
- `./build/test_iterative`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/example_analysis`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

These are the default high-signal reruns for Sprint 59 implementation and
final-validation days.

Interpretation:

- the sprint can keep its rerun focus on direct/public lifecycle proof,
  representative examples, representative benchmark drivers, and the reviewed
  parity anchors
- no broader default rerun set is required on Day 2

## Conclusion

Day 2 leaves Sprint 59 with an explicit validation and rerun contract:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit `*.c` / `*.h` code-day gate
- explicit stronger reviewed-baseline default
- authoritative final-sprint rerun set from the live build tree

That is enough to move to the Day 3 quality/platform residual audit without
validation ambiguity.
