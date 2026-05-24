# Sprint 40 Day 3 Artifact: Support Baseline

## Purpose

Capture the validated support baseline beyond the direct local wrappers before
later Epic 4 architecture and lifecycle work begins.

## Reviewed CMake Parity Baseline

The current reviewed CMake parity path remains the strongest shared reviewed
baseline across platforms:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- `make quality-review-cmake-compile` owns:
  - configure
  - clean rebuild
  - `ctest -N`
  - Makefile/CMake test-count parity
- `make quality-review-cmake` owns:
  - `make quality-review-cmake-compile`
  - full `ctest`

This remains a support baseline adjacent to the reviewed Makefile path, not a
replacement for Makefile-owned formatting, static-analysis, or dead-code
policy.

## Dead-Code Baseline

Current dead-code report state:

- compile-db translation-unit counts:
  - `src = 25`
  - `tests = 53`
  - `benchmarks = 14`
  - `examples = 12`
- benchmark/example compile-db coverage gap:
  - closed
- residual report buckets:
  - `coverage-gap = 0`
  - `definitely-unused-internal-candidate = 0`
  - `public-surface-review = 4`
  - `secondary-candidate-signal = 35`
  - `non-deadcode-static-analysis-noise = 6`

Current semantics:

- `make deadcode-report`
  - refreshes evidence and writes `report.md` / `report.tsv`
- `make deadcode-check`
  - validates report completeness
  - is not a zero-findings gate

Current known limit:

- authoritative dead-code execution remains serialized because `deadcode*`
  still shares:
  - `build/deadcode-cmake`
  - `build/deadcode/`

## Cross-Platform Baseline

### Linux

Enforced:

- `make quality-review-compile`
- `make quality-review-cmake`
- `make deadcode-report`
- `make deadcode-check`

Supplemental:

- direct runtime path
- `bench-fast`
- TSan
- coverage

### macOS

Enforced:

- Apple Clang `make quality-review-compile`
- Apple Clang `make quality-review-cmake`
- `make wall-check`
- `make sanitize`

Staged:

- `make deadcode-report`
- `make deadcode-check`

Supplemental:

- Homebrew GCC direct `make`
- Homebrew GCC direct `make test`
- Homebrew GCC `make wall-check`
- install/pkg-config validation

### Windows

Enforced:

- reviewed CMake configure/build
- `ctest -N`
- full `ctest`
- expected registered test count = `50`

Staged:

- `make quality-review-compile`
- `make quality-review`
- dead-code

Excluded:

- `test_threads`
- `test_sprint4_integration`
- `test_fuzz`

## Benchmark / Example / Tooling Support Baseline

Current support targets that later Epic 4 refactors must not silently break:

- `make tooling-build`
  - benchmark/example compile-only support surface
- `make bench-build`
  - benchmark compile-only support surface
- `make examples-build`
  - example compile-only support surface
- `make wall-check`
  - maintained performance-regression support gate
- `make coverage`
  - supplemental coverage gate
  - current threshold remains `80%`

## Day 3 Conclusion

Sprint 40 now has a broad enough starting baseline to move into structural
inventory work without later ambiguity about what “the inherited support
surface” meant:

- reviewed local command baseline
- reviewed CMake parity baseline
- dead-code reporting/completeness baseline
- cross-platform enforced/staged/excluded baseline
- benchmark/example/tooling support baseline
