# Sprint 119 Day 13 Validation Parity Package

## Purpose

Package the Day 13 validation evidence for the Sprint 119 eigensolver source
boundary work. The evidence covers the selection/lifting source extraction,
build-system source-list parity, focused eigensolver behavior, CMake reviewed
CTest registration, and the full required Makefile quality lane for the
branch's `.c` movement.

## Scope

- In scope:
  - `src/sparse_eigs_selection_internal.c` selection/lifting helper extraction.
  - `src/sparse_eigs.c` removal of the extracted helper implementations.
  - Makefile, CMake, and library source-list registration for the new private
    source.
  - Focused eigensolver behavior across grow-m, thick-restart, and LOBPCG
    surfaces.
- Out of scope:
  - Shift-invert setup/conversion source movement. Day 11 intentionally
    deferred this because setup, factor lifetime, telemetry, transformed
    eigenvalue conversion, backend dispatch, and cleanup remain tightly coupled.
  - Public API changes.
  - Public documentation or product-claim changes.
  - Benchmark execution.

## Command Evidence

| Command | Result | Evidence |
| --- | --- | --- |
| `make source-list-check` | Pass | `source-list-check: PASS (49 library sources)` |
| `make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg` | Pass | Focused eigensolver binaries were up to date. |
| `./build/test_eigs` | Pass | 43 tests, 0 failed, 955 assertions. |
| `./build/test_eigs_thick_restart` | Pass | 23 tests, 0 failed, 384 assertions. |
| `./build/test_eigs_lobpcg` | Pass | 29 tests, 0 failed, 287 assertions. |
| `cmake -S . -B build-cmake-review && cmake --build build-cmake-review && ctest --test-dir build-cmake-review -N` | Pass | Clean CMake build compiled `src/sparse_eigs_selection_internal.c`; `ctest -N` reported `Total Tests: 54`. |
| `rm -rf build-cmake-review && make format && make lint && make test` | Pass | Full required Make quality lane completed and ended with `All tests passed.` |

## CTest Membership

- Expected reviewed local CMake CTest count: 54.
- Observed reviewed local CMake CTest count: 54.
- Membership includes:
  - `test_eigs`;
  - `test_eigs_thick_restart`;
  - `test_eigs_lobpcg`;
  - the full local reviewed CTest surface through `test_reorder_amd_qg`.

## Skipped Supplemental Lanes

- Windows CTest-count enforcement was not run locally; the CI Windows lane owns
  that platform-specific reviewed subset.
- Benchmark execution was skipped because Sprint 119 Day 13 validates source
  ownership and parity, not performance claims.
- Package/install validation was skipped because no package export, install, or
  ABI surface changed on Day 13.
- Public documentation claim validation was limited to diff review because the
  branch did not modify public docs or product claims.

## Conclusions

- The new selection/lifting private source is registered in all local source
  inventories that govern Make and CMake builds.
- Focused eigensolver behavior remains unchanged across grow-m,
  thick-restart, and LOBPCG tests.
- CMake CTest registration remains stable at 54 tests.
- The full required Make quality chain passes for the branch's `.c` movement.
- The shift-invert source-boundary deferral remains explicit and validated; it
  is not silently mixed into the completed selection/lifting extraction.

## Completion Criteria

| Criterion | Status |
| --- | --- |
| Source-list evidence captured | Complete |
| Makefile quality evidence captured | Complete |
| CMake parity evidence captured | Complete |
| CTest count evidence captured | Complete |
| Focused eigensolver evidence captured | Complete |
| Skipped-lane rationale captured | Complete |
| Temporary CMake review build removed | Complete |
