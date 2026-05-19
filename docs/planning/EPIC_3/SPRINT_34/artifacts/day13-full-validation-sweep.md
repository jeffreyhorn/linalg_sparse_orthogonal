# Sprint 34 Day 13: full validation sweep

## Goal

Prove that Sprint 34’s local reviewed paths, CMake parity path, and CI-equivalent enforcement commands all pass together on the current branch state.

## Direct local quality gates

- `make format`: passed
- `make lint`: passed
  - real time: `244.14 s`
- `make test`: passed
  - real time: `62.34 s`

## Reviewed wrapper / CI-equivalent local commands

- `make quality-review-compile`: passed
  - real time: `314.65 s`
- `make quality-review`: passed
  - real time: `438.34 s`
- `make deadcode-report`: passed
  - real time: `0.39 s`
- `make deadcode-check`: passed
  - real time: `0.56 s`
- `make quality-review-cmake-compile`: passed
  - `ctest -N` reported `53` registered tests
  - real time: `63.14 s`
- `make quality-review-cmake`: passed
  - full `ctest`: `53 / 53` passed
  - CTest reported real test time: `181.55 s`
  - wrapper wall time: `239.97 s`

## Truthfulness and dead-code invariants

### Sprint 32 truthfulness surface

- `tests/test_reorder_nd.c`
  - active `RUN_TEST(...)` registrations: `23`
- `tests/test_framework_optin.c` / `tests/test_framework.h`
  - `SKIP_TEST(...)`: present
  - `RUN_TEST_SLOW(...)`: present
  - `RUN_TEST_EXPERIMENTAL(...)`: present

### Sprint 33 dead-code artifact flow

Regenerated artifacts present:

- `build/deadcode/report.md`
- `build/deadcode/report.tsv`
- `build/deadcode/coverage-notes.txt`
- `build/deadcode/cppcheck.txt`
- `build/deadcode/xunused.txt`

Refreshed during the Day 13 run:

- `coverage-notes.txt`: `12:05`
- `cppcheck.txt`: `12:06`
- `report.md`: `12:06`
- `report.tsv`: `12:06`
- `xunused.txt`: `12:06`

## Result

Sprint 34’s reviewed enforcement work is currently green across:

- direct local gates
- reviewed Makefile wrappers
- reviewed CMake parity wrappers
- dead-code report/check flow

The earlier Sprint 32 and Sprint 33 invariants remain intact on the same branch state.
