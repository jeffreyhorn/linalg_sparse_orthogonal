# Epic 3 Summary

## Purpose

Summarize the stable end state of Epic 3 after the final audit work in Sprint
39 so later feature work can start from the maintained quality contract rather
than reconstructing the full sprint-by-sprint cleanup history.

## Enforced Baseline

Epic 3 leaves behind a concrete maintained quality baseline:

- direct maintained gates:
  - `make format`
  - `make lint`
  - `make test`
- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity baseline:
  - `ctest -N --test-dir build/quality-review-cmake`
  - full `ctest` through `make quality-review-cmake`
- dead-code evidence/completeness path:
  - `make deadcode-report`
  - `make deadcode-check`

Current measured reviewed CMake parity truth:

- `53` registered tests

## Warning-Clean Final State

Epic 3 finished the repository-wide warning cleanup and now keeps warning
claims under an explicit evidence hierarchy:

- strongest routine local reviewed baseline:
  - `make quality-review-full`
- authoritative repository-wide warning proof:
  - `make warning-workflow WARNING_WORKFLOW_LABEL=label`
  - Apple Clang CMake full-tree inventory
- narrower supporting cross-check:
  - Makefile `all`

Important closeout rule:

- `make quality-review-full` is the strongest routine local reviewed command
- it is **not** the same claim as repository-wide warning inventory proof

## Dead-Code Final State

Epic 3 also finished the first honest dead-code maturation pass:

- compile-db benchmark/example coverage gap:
  - `0`
- definitely-unused internal cleanup queue:
  - `0`

Residual report buckets remain, but they are no longer an active removal batch:

- `public-surface-review = 4`
  - already-audited justified keeps
- `secondary-candidate-signal = 35`
  - supporting `cppcheck` evidence only
- `non-deadcode-static-analysis-noise = 6`
  - appendix/noise context only

Important closeout rule:

- `deadcode-check` is a report-completeness gate
- it is **not** a zero-findings gate
- authoritative execution remains serialized because the workflow still uses
  shared `build/deadcode-cmake` / `build/deadcode/` paths

## Test-Truthfulness Final State

Epic 3 left the active test surface in an explicit and auditable state:

- no commented-out `RUN_TEST(...)` scaffolding remains in the active suite
- live opt-in semantics remain executable truth in `tests/test_framework.h`:
  - `RUN_TEST_SLOW(...)`
  - `RUN_TEST_EXPERIMENTAL(...)`
  - `SKIP_TEST(...)`
- public and maintainer docs now point historical or retired test evidence back
  to `docs/planning/` artifacts rather than dormant active-suite scaffold

## Public Docs And Example Contract

Epic 3 ended with a clear ownership model for long-term docs:

- `README.md`
  - concise operator command map
  - dead-code contract
  - cross-platform CI contract
  - readiness checklist
  - maintainer standards
- tutorial and installed headers
  - authoritative public API usage/behavior teaching
- `docs/planning/EPIC_3/**`
  - historical engineering evidence

Maintainer expectations now made explicit:

- public non-default examples should use designated initializers
- dormant or historical test evidence belongs in `docs/planning/` artifacts
- Sprint 30 warning authority docs remain the source of truth for
  repository-wide warning claims

## Cross-Platform Contract

Epic 3 ended with a truthful cross-platform model rather than fake symmetry:

- Linux
  - strongest enforced reviewed baseline
  - dead-code enforced
- macOS
  - Apple Clang reviewed path enforced
  - Homebrew GCC leg supplemental
  - dead-code staged
- Windows
  - reviewed CMake subset enforced
  - local Makefile reviewed-wrapper parity staged
  - dead-code excluded

Important closeout rule:

- reviewed CMake parity is the strongest shared reviewed baseline across
  platforms

## Coverage And Readiness

Coverage and readiness wording is now truthful:

- coverage is supplemental, not part of the reviewed baseline
- current enforced coverage threshold:
  - `80%` line coverage on `src/` in the Linux coverage path
- readiness now has one concise maintained checklist in `README.md`

## Residual Risks And Intentional Limits

Epic 3 does **not** close with zero limitations. The remaining real limits are:

- dead-code shared-path execution remains serialized
- residual dead-code content buckets remain supporting/closeout context rather
  than an active cleanup queue
- macOS dead-code remains staged
- Windows local Makefile reviewed-wrapper parity remains staged
- Windows dead-code remains excluded

These are the intended final closeout limits, not newly discovered regressions.

## Start-Next-Work Guidance

Later feature work should treat the following as the maintained starting point:

1. Use `make quality-review-full` as the strongest routine local reviewed
   baseline.
2. Use the Sprint 30 warning workflow for repository-wide warning claims.
3. Use the dead-code path as a serialized completeness/reporting tool, not a
   zero-findings assertion.
4. Treat the `README.md` command/contract sections and `tests/test_framework.h`
   as the stable maintainer/operator truth surfaces.
5. Preserve the current enforced/staged/excluded cross-platform boundaries
   unless a later feature explicitly broadens them.

## Key References

- [README.md](../../../../../README.md)
- [Compile Hygiene Playbook](../../SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md)
- [Rebuild Workflow](../../SPRINT_30/REBUILD_WORKFLOW.md)
- [Sprint 38 Handoff](../../SPRINT_38/HANDOFF.md)
- [Sprint 39 Working Notes](../WORKING_NOTES.md)
