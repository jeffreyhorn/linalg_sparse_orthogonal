# Sprint 185 Day 12: Focused Cluster Validation

## Purpose

Record focused validation for the selected Sprint 185 LDLT CSC review-surface
extraction before the Day 13 full quality gate.

## Focused Scope

| Field | Value |
| --- | --- |
| Selected proof owner | `tests/test_ldlt_csc.c` / `test_ldlt_csc` |
| Helper headers | `tests/test_ldlt_csc_fixtures.h`, `tests/test_ldlt_csc_oracle_helpers.h`, `tests/test_ldlt_csc_supernode_helpers.h` |
| Selected-cluster guard | `make ldlt-csc-helper-guard` |
| Source-list guard | `make source-list-check` |

## Current Review-Surface Size

| Path | Current lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3469 |
| `tests/test_ldlt_csc_fixtures.h` | 145 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 149 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 134 |

The selected proof-owner file remains reduced from the Day 3 baseline of 3915
lines to 3469 lines.

## Focused Validation

Validation completed:

```sh
if [ -e build/test_ldlt_csc ]; then rm build/test_ldlt_csc; fi
make build/test_ldlt_csc
./build/test_ldlt_csc
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

Results:

- `make build/test_ldlt_csc`: passed after forcing the stale binary out of the
  build tree.
- `./build/test_ldlt_csc`: passed with 100 tests, 0 failures, 0 skips, and
  3556 assertions.
- `make ldlt-csc-helper-guard`: passed.
- `make source-list-check`: PASS, 49 library sources.
- `git diff --check`: passed.

## Diff Review

Reviewed the accumulated Sprint 185 diff for the selected cluster and required
docs/guards.

| Surface | Day 12 finding |
| --- | --- |
| `tests/test_ldlt_csc.c` | Adds three helper-header includes and removes helper definitions moved into family-local headers. |
| `tests/test_ldlt_csc_fixtures.h` | Contains extracted KKT/scaled-KKT/two-pass setup helpers. |
| `tests/test_ldlt_csc_oracle_helpers.h` | Contains extracted dense-oracle and native-wrapper comparison helpers. |
| `tests/test_ldlt_csc_supernode_helpers.h` | Contains extracted supernode fixture/snapshot/factor-state helpers. |
| `Makefile` | Adds only the `ldlt-csc-helper-guard` target. |
| `scripts/check_ldlt_csc_helper_guard.sh` | Adds the selected-cluster guard. |
| `docs/maintainer_guide.md` | Adds helper ownership and guard guidance. |

No `RUN_TEST(...)` ordering changes, public test-body changes, solver source
changes, production API changes, fixture-value changes, or registration
broadening were found. Tolerance-bearing helper signatures remain part of the
moved helper bodies and are covered by the focused test pass.

## Day 13 Full Validation Command List

Run the full C quality gate and retained focused checks:

```sh
make format
make lint
make test
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

If Day 13 finds any formatting, lint, test, or guard failure, fix the issue and
rerun the failing command before recording the full-gate result.

## Day 13 Handoff

- Run the full quality gate.
- Re-run the selected-cluster guard and source-list check.
- Record final validation notes and any cleanup required for review readiness.
