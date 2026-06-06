# Sprint 56 Day 2 - validation baseline and touched-surface recheck

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Reconfirm the reviewed validation baseline and the exact CSC direct-solver and
SVD rerun set Sprint 56's later extraction batches must preserve.

## Maintained reviewed baseline

The maintained Sprint 56 baseline remains:

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

- Sprint 56 should keep using the exact `strongest local reviewed baseline`
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

- Sprint 56 does not need to reopen any quality-contract documentation work on
  Day 2
- the maintained baseline language is already stable enough to carry forward
  unchanged

## Authoritative Sprint 56 rerun set

The main Sprint 56 follow-on binaries already present in `build/` are:

- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_svd`
- `./build/test_integration`
- `./build/bench_refactor_csc`
- `./build/example_analysis`

These are the default high-signal reruns for Sprint 56 extraction days.

Interpretation:

- the sprint can keep its validation focus on the CSC direct-solver and SVD
  family surfaces actually touched by the decomposition work
- no broader default rerun set is required on Day 2

## Live proof/adoption surface sizes

The current high-signal proof/adoption surfaces are:

- `tests/test_chol_csc.c` = `4643`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_svd.c` = `3746`
- `tests/test_integration.c` = `1803`
- `benchmarks/bench_refactor_csc.c` = `611`
- `examples/example_analysis.c` = `210`

Interpretation:

- Sprint 56 extraction work should treat proof-surface parity as a first-class
  concern
- the rerun set is a real guard against behavior drift while ownership moves

## Conclusion

Day 2 leaves Sprint 56 with an explicit validation and rerun contract:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit `*.c` / `*.h` code-day gate
- explicit stronger reviewed-baseline default
- authoritative CSC direct-solver and SVD rerun set from the live build tree

That is enough to move to the Day 3 `sparse_ldlt_csc.c` residual ownership
audit without validation ambiguity.
