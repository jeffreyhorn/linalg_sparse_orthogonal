# Sprint 36 Day 10: Cross-Platform CMake / Makefile / CI Parity Report

## Scope

Summarize the Sprint 36 cross-platform quality contract in one compact map:

- which reviewed-quality checks are available from Make
- which are available from CMake
- which are enforced, staged, or supplemental in CI

This report is grounded in the shipped Day 5-9 workflow and portability work,
not in aspirational future parity.

## Status Vocabulary

- **Enforced**: part of the maintained contract on that platform today
- **Staged**: relevant surface, but not yet enforced there
- **Supplemental**: useful extra signal, but not the core reviewed contract
- **Excluded**: intentionally outside the current platform contract

## Local Reviewed Paths

### Maintained Local Commands

| Surface | Linux | macOS | Windows |
|--------|-------|-------|---------|
| `make quality-review-compile` | Enforced baseline | Enforced baseline on Apple Clang; supplemental GCC leg stays direct | Staged |
| `make quality-review` | Enforced baseline | Available locally but not CI-enforced; dead-code remains staged in platform contract | Staged |
| `make quality-review-cmake-compile` | Enforced baseline | Enforced baseline | Enforced subset via direct workflow steps |
| `make quality-review-cmake` | Enforced baseline | Enforced baseline | Staged as named wrapper; enforced behavior is direct CMake `ctest -N` + `ctest` |
| `make deadcode-report` | Enforced baseline | Staged | Excluded |
| `make deadcode-check` | Enforced baseline | Staged | Excluded |

### Interpretation

- The reviewed **CMake** path is the only fully honest cross-platform reviewed
  baseline.
- The reviewed **Makefile** path remains Linux/macOS-maintainer oriented.
- Dead-code remains intentionally non-portable as a full platform contract in
  Sprint 36.

## CI Contract By Platform

| Platform | Enforced | Staged | Supplemental / Excluded |
|--------|---------|---------|---------------------------|
| Linux | `make quality-review-compile`; `make quality-review-cmake`; `make deadcode-report`; `make deadcode-check` | none inside the maintained reviewed baseline | direct runtime + `bench-fast`; TSan; coverage |
| macOS | Apple Clang: `make quality-review-compile`; `make quality-review-cmake`; `make wall-check`; `make sanitize` | dead-code (`make deadcode-report`, `make deadcode-check`) | Homebrew GCC direct `make` + `make test` + `make wall-check`; install/pkg-config validation |
| Windows | reviewed CMake configure/build; `ctest -N`; full `ctest` | `make quality-review-compile`; `make quality-review`; dead-code | excluded tests: `test_threads`, `test_sprint4_integration`, `test_fuzz` |

## Surface Notes

### Reviewed CMake parity

- Linux: enforced via `make quality-review-cmake`
- macOS: enforced on the Apple Clang leg via `make quality-review-cmake`
- Windows: enforced as the narrower direct reviewed CMake subset

This is the strongest shared parity surface in Sprint 36.

### Reviewed Makefile compile-quality

- Linux: enforced
- macOS: enforced on Apple Clang only
- Windows: staged

This remains the clearest cross-platform gap after Sprint 36 Day 8.

### Dead-code

- Linux: enforced
- macOS: staged
- Windows: excluded

Reasons remain unchanged:

- `xunused` setup complexity
- shared `build/deadcode-cmake` / `build/deadcode/` execution model
- compile-db coverage gap (`bench_svd` + six examples)

### Unix-maintainer helper flows

These are not part of the reviewed cross-platform parity contract:

- `wall-check`
- `warning-workflow`
- coverage helpers

They remain:

- useful on Unix maintainer paths
- supplemental or excluded rather than fake all-platform reviewed gates

## Day 11 Follow-On Queue

The report narrows the final fix batch to small consistency work only:

1. Reconcile any remaining wording mismatch between:
   - local target descriptions
   - workflow comments
   - this parity report
2. Tighten any residual staged/deferred wording where it is still less explicit
   than the report.
3. Avoid reopening:
   - dead-code maturity work
   - fake Windows Makefile parity
   - broad public-doc cleanup outside Sprint 36 scope

## Conclusion

Sprint 36 now has a compact parity map rather than scattered assumptions.

The main outcome is clear:

- Linux is still the strongest enforced reviewed baseline
- reviewed CMake parity is the real cross-platform anchor
- macOS and Windows now state their narrower contracts truthfully
- dead-code and Unix-maintainer helper flows remain explicitly staged or
  excluded where that is the honest current state
