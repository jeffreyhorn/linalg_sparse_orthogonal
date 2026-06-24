# Sprint 87 Day 12: Support-Surface Alignment & Validation Queue Freeze

## Purpose

Reconcile the touched Sprint 87 package, consumer, and workflow surfaces
against the landed boundaries and freeze the exact Day 13 validation queue.

## Main Result

No new support-only edit is needed before the full validation sweep.

The final Sprint 87 touched-surface truth map is now explicit:

- adopted package/product center:
  - `CMakeLists.txt`
- adopted consumer-proof center:
  - `tests/test_install.sh`
- adopted workflow/platform center:
  - `.github/workflows/macos-ci.yml`
- retained adjacent package/export proof owner:
  - `tests/test_cmake_install.sh`
- retained downstream consumer surface:
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
- support-only surfaces that do not need new movement before Day 13:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `.github/workflows/ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`

## Final Proof-Owner Map

The final Sprint 87 proof-owner split is now fixed:

- strongest reviewed baseline owners:
  - `tests/test_reorder_nd.c`
  - `tests/test_reorder.c`
  - `tests/test_reorder_amd_qg.c`
  - `tests/test_graph.c`
- representative reviewed examples:
  - `example_analysis`
  - `example_basic_solve`
- maintained local Make/pkg-config install/export and consumer proof owner:
  - `tests/test_install.sh`
- maintained local CMake install/export and consumer proof owner:
  - `tests/test_cmake_install.sh`
- bounded workflow/platform evidence owner:
  - `.github/workflows/macos-ci.yml`
- retained narrower reviewed CMake-first consumer scope owner:
  - `.github/workflows/windows-ci.yml`

## Support-Surface Boundary

The support-surface reading stayed fixed:

- `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` already remain
  truthful about the static-first package contract, exact-version CMake export
  behavior, the strengthened local consumer proof, and the narrower
  macOS/Windows workflow claims
- canonical maintained reporting remains command/script-owned through:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
- Sprint 87 did not reopen reviewed correctness ownership, benchmark
  ownership, or Windows workflow scope beyond the already-maintained
  CMake-first consumer reading

## Frozen Day 13 Queue

The exact Day 13 queue is now fixed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- focused reviewed proof owners:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
- representative reviewed examples:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- package/consumer proof reruns:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- retained maintained reporting follow-on:
  - `make bench-canonical-report`

## Strongest Clarification

The useful Day 12 clarification is explicit now:

- Sprint 87 does not need another package, consumer, workflow, or docs batch
  before validation
- the authoritative correctness owners for the sprint close remain the
  unchanged reviewed baseline tests and examples
- the authoritative Sprint 87 package/consumer owners are the two local
  install/export proof scripts already fixed above
- the macOS workflow follow-through is real evidence, but it does not redefine
  the stronger reviewed baseline or widen Windows scope

## Exit State

- no support-only drift remains before Day 13
- the final validation queue is explicit and unambiguous
- Day 13 can execute from a fixed package/consumer/workflow truth map rather
  than re-deciding Sprint 87 scope
