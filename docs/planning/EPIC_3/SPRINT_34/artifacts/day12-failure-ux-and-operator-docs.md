# Sprint 34 Day 12: failure UX and operator docs

## Goal

Make Sprint 34’s local and CI quality paths easier to operate by clarifying:

- which reviewed path is running
- which lower-level command to rerun when a phase fails
- which artifacts to inspect when dead-code report completeness passes but findings still exist

## Shipped UX improvements

### Makefile wrappers

The reviewed wrapper targets now provide three things more explicitly:

1. path identity
2. direct rerun commands
3. final pass summaries

Improved targets:

- `quality-review-compile`
- `quality-review`
- `quality-review-cmake-compile`
- `quality-review-cmake`
- `deadcode-report`
- `deadcode-check`

Notable change in the dead-code flow:

- `deadcode-check` no longer stops at “report completeness checks passed”
- it now also tells the operator that findings may still exist and points directly to:
  - `build/deadcode/report.md`
  - `build/deadcode/report.tsv`

### README

The maintained docs now include an explicit operator command map for:

- compile-quality wrapper
- full reviewed local path
- CMake parity compile path
- full CMake parity path
- dead-code report generation
- dead-code completeness gate

The README also now documents direct rerun commands for:

- `format-check`
- `lint`
- `test`
- `deadcode-check`
- CMake configure/build/`ctest -N`/full `ctest`
- dead-code artifact inspection

### CI workflow step names

The primary Linux workflow now says more about what each reviewed step actually does:

- reviewed CMake path:
  - clean rebuild + `ctest -N` + `ctest`
- reviewed Makefile path:
  - `format-check` + `lint`
- dead-code report generation:
  - report files plus raw evidence
- dead-code validation:
  - completeness, not “zero findings”
- artifact upload:
  - on failure or success

## Validation

Validated the message surfaces directly:

- `make -n quality-review-compile`
- `make -n quality-review`
- `make -n quality-review-cmake-compile`
- `make -n quality-review-cmake`
- `make deadcode-report`
- `make deadcode-check`

Confirmed CI step-name anchors in `.github/workflows/ci.yml` for:

- reviewed CMake path
- reviewed Makefile path
- dead-code report generation
- dead-code report validation
- dead-code artifact upload

## Result

Sprint 34’s quality paths now explain the next operator action more clearly without changing the underlying enforcement contract:

- same quality gates
- better rerun guidance
- better artifact guidance
- clearer CI step meaning
