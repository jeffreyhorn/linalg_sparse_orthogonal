# Sprint 33 Day 13: Full Validation Sweep

## Goal

Revalidate the full Sprint 33 end state across both supported validation
surfaces:

- normal Makefile quality gates
- dedicated CMake/CTest path

Also confirm that the Sprint 33 dead-code workflow still runs after the cleanup
pass and that Sprint 32's live opt-in test contract remains intact.

## Commands Run

### Normal quality path

1. `make format`
2. `/usr/bin/time -p make lint`
3. `/usr/bin/time -p make test`

### Dedicated CMake/CTest path

4. `/usr/bin/time -p cmake --build build/sprint33-day1-cmake --parallel 1 --clean-first`
5. `ctest -N --test-dir build/sprint33-day1-cmake`
6. `/usr/bin/time -p ctest --test-dir build/sprint33-day1-cmake --output-on-failure`

### Dead-code workflow

7. `/usr/bin/time -p make deadcode-report`
8. `/usr/bin/time -p make deadcode-check`

### Truthfulness / opt-in spot checks

9. `rg -n "RUN_TEST\\(|RUN_TEST_SLOW\\(|RUN_TEST_EXPERIMENTAL\\(" tests/test_framework_optin.c tests/test_reorder_nd.c`
10. `cat build/deadcode/coverage-notes.txt`
11. `python3 - <<'PY' ... summarize bucket counts from build/deadcode/report.tsv ... PY`

## Validation Results

### Makefile path

- `make format`: passed
- `make lint`: passed
  - wall time: `473.25 s`
- `make test`: passed
  - wall time: `132.22 s`

`make test` reconfirmed the live opt-in surface:

- `test_framework_optin`: passed
- summary inside that binary:
  - `Tests run: 8`
  - `Tests failed: 0`
  - `Tests skipped: 3`

## CMake / CTest Path

- clean serialized rebuild of `build/sprint33-day1-cmake`: passed
  - wall time: `53.22 s`
- `ctest -N`: passed
  - registered tests: `53`
- full `ctest --output-on-failure`: passed
  - result: `53 / 53` passed
  - CTest reported real test time: `178.31 s`
  - wrapper wall time: `178.35 s`

## Dead-Code Workflow

- `make deadcode-report`: passed
  - wall time: `83.23 s`
- `make deadcode-check`: passed
  - wall time: `104.00 s`

The post-cleanup report remains stable:

- `coverage-gap`: `7`
- `public-surface-review`: `4`
- `secondary-candidate-signal`: `35`
- `non-deadcode-static-analysis-noise`: `6`
- `definitely-unused-internal-candidate`: `0`

The current compile-db coverage-gap list is unchanged:

- missing benchmark:
  - `bench_svd`
- missing examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

## Truthfulness / Opt-In State

The active/default and opt-in registration model remains live and explicit:

- `tests/test_framework_optin.c` still contains:
  - plain `RUN_TEST(...)` coverage
  - `RUN_TEST_SLOW(...)`
  - `RUN_TEST_EXPERIMENTAL(...)`
- `tests/test_reorder_nd.c` still has `23` active `RUN_TEST(...)` registrations
  and no dormant commented-out registration queue was reintroduced during Sprint
  33

## End State

Sprint 33 enters closeout with:

- normal quality gates green
- CMake/CTest parity green
- dead-code report/check green
- no definitely-unused internal candidate queue left
- Sprint 32 opt-in truthfulness conventions still live and validated

## Observations For Handoff

- The dead-code workflow remains intentionally serial/operator-invoked.
- Day 10 / Day 11's shared-build-tree race did not reappear during the serialized
  Day 13 run.
- The remaining dead-code backlog is not "cleanup-ready" code removal; it is the
  already-documented mix of compile-db coverage gaps, audited public keeps, and
  secondary/noise buckets.
