# Remediation Plan

**Date:** 2026-05-15  
**Derived from:** `review-codex-2026-05-15.md`

## Goal

Address the shortcomings in the current codebase without adding new library features.

## Plan

1. Establish a warning-baseline pass for the whole tree.
   - Reproduce the current warning set from a clean CMake build and group it by `src/`, `tests/`, `benchmarks/`, and `examples/`.
   - Treat core-library warnings first, then benchmarks/examples/tests.
   - Record the baseline in the sprint notes so “warning count went up/down” is measurable.

2. Remove core-library compile warnings.
   - Fix the `INFINITY`-related warning sites in `src/sparse_lu.c`, `src/sparse_ldlt.c`, `src/sparse_qr.c`, and `src/sparse_svd.c`.
   - Rebuild after each batch to confirm the warning count drops rather than shifts.
   - Keep behavior unchanged; this task is hygiene, not a semantic rewrite.

3. Clean up benchmark portability warnings.
   - Fix the benchmark files that currently trigger implicit-declaration warnings for `snprintf`.
   - Audit `_POSIX_C_SOURCE` usage in benchmark files and raise or refactor feature-test handling where needed.
   - Re-run macOS-oriented builds to confirm the benchmark layer is warning-clean on supported toolchains.

4. Bring `bench_main` in sync with the current reorder API.
   - Update usage text to advertise all supported reorder modes that the tool should expose.
   - Extend `reorder_name()` to cover every valid `sparse_reorder_t` value.
   - Extend CLI parsing to accept `colamd` and `nd`.
   - Verify behavior against the existing benchmark helpers so benchmark outputs are internally consistent.

5. Replace brittle positional option initialization in public-facing code.
   - Update header examples in `include/` to use designated initializers.
   - Update `examples/` and `benchmarks/` to do the same for public options structs.
   - After that, update tests opportunistically in the same style where warnings are currently being emitted.

6. Resolve dormant test scaffolding in `tests/test_reorder_nd.c`.
   - Decide which dormant bodies are historical documentation and which are intended future checks.
   - For historical artifacts: move the evidence into planning/benchmark docs and delete the dead test code.
   - For intended checks: convert them into explicit skip/xfail-style tests or gate them behind a dedicated slow/experimental test mechanism.
   - Eliminate the current unused-function warnings from that file.

7. Tighten build-quality enforcement once the tree is clean.
   - Add a compile-quality step that fails CI when warnings are introduced in the supported targets being built.
   - Scope it carefully: at minimum, enforce warning-free `src/`, `benchmarks/`, `examples/`, and the normal `tests/` set under the primary CI compilers.
   - Avoid turning on `-Werror` globally until the warning baseline is genuinely clean.

8. Re-run the full validation matrix after cleanup.
   - Rebuild with CMake from a clean tree.
   - Re-run `ctest`.
   - Re-run the existing CI-equivalent quality steps that are practical locally.
   - Confirm that documentation/examples still compile or remain consistent after the initializer changes.

9. Document the final engineering standard.
   - Add a short maintainer note covering designated initializers for public option structs.
   - Document the expectation that dormant experimental assertions do not live in the normal test suite.
   - Document the expectation that developer tools stay in sync with the public enums they expose.

## Exit Criteria

- Clean CMake build with no new warnings in the reviewed targets.
- `bench_main` supports the reorder modes the library currently supports.
- Dormant ND test scaffolding is either removed or formalized.
- Public docs/examples use designated initializers for evolving option structs.
- Full `ctest` run still passes.
