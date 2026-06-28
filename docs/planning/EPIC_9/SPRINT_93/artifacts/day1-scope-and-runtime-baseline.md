# Sprint 93 Day 1: Scope and Runtime Baseline

## Purpose

Turn the Sprint 93 project-plan section and the Sprint 92 validated closeout
into one bounded runtime-scalability, threading, and ND-convergence execution
package before any runtime audit, contract design, or implementation lands.

## Starting Truth

Sprint 93 begins from a validated Sprint 92 close state, not from another
generic backend or benchmark reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Sprint 92 already moved the strongest prior backend contradiction:

- the shared dense owner now has one bounded builtin-vs-portable backend seam
- LDLT now converges onto the same bounded backend reading instead of a
  separate family-local acceleration pocket
- benchmark-side backend observability is explicit enough that Sprint 93 does
  not need to reopen backend truthfulness first

That means Sprint 93 can start from the next real Epic 9 contradiction center:

- reviewed runtime concentration, threading/runtime contract sharpness, and
  ND-convergence follow-through

## Sprint 93 Workstreams

The highest-value Sprint 93 package is now fixed explicitly around:

- reviewed runtime audit
- threading/runtime contract design
- ND runtime reduction design and batch
- runtime-control cleanup
- proof-surface rebalancing
- runtime evidence follow-through

## Strongest Runtime Starting Point

The live maintained runtime story is already more disciplined than a generic
"make graph and reorder faster" claim:

- the reviewed lane already exposes the main long pole clearly enough to rank
  work honestly
- graph and ND owners are concentrated enough to target surgically
- runtime and threading proof owners already exist and can be reused rather
  than invented
- workflow and install/export surfaces are maintained, but they are not the
  right first implementation center for Sprint 93

Sprint 93 therefore does not begin from "scale the whole library." It begins
from one explicit truthfulness question:

- where can the repo earn one bounded runtime-scalability and threading
  improvement on the reviewed graph/ND lane without overclaiming broad
  cross-platform scaling maturity

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 93 surfaces:

- strongest runtime and reordering implementation owners:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_reorder_amd_qg.c`
- strongest proof-owner tests likely to matter:
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `tests/test_threads.c`
  - `tests/test_omp.c`
- strongest benchmark and runtime-evidence owners:
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_amd_qg.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `scripts/bench_canonical_report.sh`
- strongest support, build, and workflow surfaces if runtime work truly forces
  follow-through:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

## Preserved Fence

Sprint 93 is explicitly bounded against:

- reopening the Sprint 92 backend lane as the first owner again
- promising broad runtime scalability beyond the maintained reviewed lane
  before touched runtime proof improves
- widening into capability-surface or packaging-product work before the
  runtime concentration seam is reduced
- treating benchmark timing alone as stronger than reviewed executable truth,
  install/export proof, or maintained workflow surfaces
- drifting into generic concurrency churn detached from one real highest-value
  reviewed runtime seam

## Day 1 Result

Sprint 93 now starts from one precise runtime-scalability, threading, and
ND-convergence execution package rather than from a generic "speed up graph
and reorder" bucket. The strongest likely touch surfaces, preserved
non-goals, and maintained reviewed starting truth are fixed in writing before
the validation and maintained-surface recheck begins.
