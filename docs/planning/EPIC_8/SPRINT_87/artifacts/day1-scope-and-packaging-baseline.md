# Sprint 87 Day 1: Scope and Packaging Baseline

## Purpose

Turn the Sprint 87 project-plan section and the Sprint 86 validated closeout
into one bounded package / ABI / consumer execution package before any build,
install/export, or workflow-aware change lands.

## Starting Truth

Sprint 87 begins from a validated Sprint 86 close state, not from another
generic Epic 8 reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Sprint 86 already moved the strongest prior contradiction:

- one bounded ND runtime reduction landed
- one bounded proof-owner/runtime-surface rebalance landed
- one bounded benchmark/comparison follow-through package landed

That means Sprint 87 can start from the next real Epic 8 contradiction center:

- the current package / ABI / install-export / downstream-consumer ceiling on
  the highest-value maintained build, proof, and workflow surfaces

## Sprint 87 Workstreams

The highest-value Sprint 87 package is now fixed explicitly around:

- release / package gap audit
- product-matrix design
- packaging batch
- consumer-proof expansion
- workflow / platform follow-through
- support-surface alignment
- validation and closeout

## Strongest Packaging Starting Point

The live maintained package contract is already narrower and clearer than a
generic cross-platform binary-product claim:

- the shipped install/export surface is real and maintained
- the maintained release shape is intentionally static-first
- downstream `pkg-config` and `find_package(Sparse)` both describe that same
  installed static archive surface
- version metadata is single-sourced from `VERSION`
- current wording explicitly does not promise a broad shared-library or
  dynamic-ABI guarantee

Sprint 87 therefore does not begin from "make packaging exist." It begins from
one explicit truthfulness question:

- whether the repo remains permanently static-first with better export and
  consumer clarity, or earns one bounded shared-library / ABI lane with real
  maintained proof

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 87 surfaces:

- build/package owners:
  - `CMakeLists.txt`
  - `Makefile`
  - `sparse.pc.in`
  - `cmake/SparseConfig.cmake.in`
- maintained local proof owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
- workflow/platform evidence surfaces:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- support and contract wording:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`

## Preserved Fence

Sprint 87 is explicitly bounded against:

- promising a broad shared-library product lane before the product matrix is
  designed and proved
- making a wide ABI-compatibility guarantee the repo does not review
- treating workflow coverage as stronger than the maintained local proof
- drifting into generic build-system churn detached from a real packaging seam
- broadening platform claims beyond what Linux, macOS, and Windows actually
  maintain today

## Day 1 Result

Sprint 87 now starts from one precise package / ABI / consumer execution
package rather than from a generic "improve packaging" bucket. The strongest
likely touch surfaces, preserved non-goals, and maintained package baseline
are fixed in writing before the validation/proof recheck begins.
