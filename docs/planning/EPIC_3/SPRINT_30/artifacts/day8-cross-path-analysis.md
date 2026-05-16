# Sprint 30 Day 8 Cross-Path Warning Analysis

**Date:** 2026-05-16  
**Branch:** `sprint-30`

## Objective

Re-run the post-Day-5 warning baseline on the primary Apple Clang CMake path and the Makefile library-build path, then separate shared warning debt from path-specific or path-scope-only findings.

## Validation Run

Ran:

- `make warning-workflow WARNING_WORKFLOW_LABEL=day8-crosspath`
- `make format`
- `make lint`
- `make test`

This used the serialized Day 8 workflow default:

- `WARNING_WORKFLOW_JOBS=1`

That serialization matters because it keeps compiler stderr non-interleaved, which makes area/file attribution stable.

## Measured Results

### Apple Clang CMake path

- configure warnings: `0`
- full-build warnings: `112`
- `ctest`: `52/52` passed
- total CTest time: `159.95 sec`

By area:

- `tests`: `98`
- `benchmarks`: `13`
- `examples`: `1`
- `src`: `0`

By warning class:

- `-Wmissing-field-initializers`: `72`
- `-Wdouble-promotion`: `34`
- `-Wunused-function`: `3`
- `-Wimplicit-function-declaration`: `2`
- `-Wswitch`: `1`

### Makefile `all` path

- warnings: `0`

### End-of-day validation

- `make format`: passed
- `make lint`: passed
- `make test`: passed

## Shared vs Path-Specific Findings

### Shared warnings across both measured paths

- none

### CMake-only remaining warnings

All `112` remaining warnings are present only on the CMake path in this comparison.

Important interpretation:

- this does **not** mean those warnings are “false” or “toolchain noise”
- it means the current Makefile `all` path does not compile the same repository scope as the CMake full-tree build

Current CMake-only warning scope:

- tests: initializer drift, double-promotion debt, and dormant unused-function scaffolding
- benchmarks: initializer drift, `snprintf` implicit-declaration portability debt, and stale `switch` coverage in `bench_main`
- examples: public-facing initializer drift

### Makefile-only warnings

- none

## Day 8 Interpretation

Day 8 clarified the remaining warning debt:

- the core library is clean on both build paths
- there is no warning class currently shared by the authoritative CMake full-tree build and the Makefile library-only cross-check
- the remaining warning volume is auxiliary-code debt that is invisible to the default Makefile `all` scope

This means the right conclusion is:

- the library-proper cleanup succeeded
- the repository is **not** yet warning-clean
- the remaining work should be scheduled by auxiliary area, not dismissed as a compiler-path anomaly

## Day 8 Surprise And Resolution

The first Day 8 rerun exposed a workflow issue: parallel CMake warning capture could interleave stderr lines and distort area/file attribution while leaving total warning counts unchanged.

Day 8 resolved that by updating the workflow to default to:

- `WARNING_WORKFLOW_JOBS=1`

This keeps future baseline artifacts parse-stable without changing the underlying measured warning totals.

## Day 8 Conclusion

Day 8 completed the cross-path reproduction goal:

- the post-core-fix warning state was reproduced cleanly
- the remaining warnings were shown to be CMake-path-only because they live outside the Makefile `all` scope
- and the warning workflow itself was hardened so future comparisons use stable, non-interleaved logs
