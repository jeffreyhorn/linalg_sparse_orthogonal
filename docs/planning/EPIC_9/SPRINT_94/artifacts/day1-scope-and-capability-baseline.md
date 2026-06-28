# Sprint 94 Day 1: Scope and Capability Baseline

## Purpose

Turn the Sprint 94 project-plan section and the Sprint 93 validated closeout
into one bounded capability-surface modernization execution package before any
capability rerank, scalar/index contract design, or implementation lands.

## Starting Truth

Sprint 94 begins from a validated Sprint 93 close state, not from another
generic runtime or backend reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Sprint 91-93 already moved the strongest prior enabling contradictions:

- compressed-first public and direct-workflow entry seams are materially
  clearer
- the shared dense owner now has one bounded builtin-vs-portable backend seam
- the strongest reviewed runtime long pole is smaller and more explicitly
  measured

That means Sprint 94 can start from the next real Epic 9 contradiction center:

- bounded capability breadth beyond the current real-only scalar contract,
  compile-time-selected index-width contract, and narrow solver-family reading
  on the touched public seams

## Sprint 94 Workstreams

The highest-value Sprint 94 package is now fixed explicitly around:

- capability rerank
- scalar/index architecture design
- scalar-family widening design and batch
- index/ABI maturity follow-through
- solver-family breadth follow-through
- proof/docs/package alignment

## Strongest Capability Starting Point

The live maintained capability story is already more disciplined than a
generic "make the whole library numerically broader" claim:

- the public type contract already exposes one bounded scalar-preparation seam
  through `sparse_scalar_t`
- the repo already carries compile-time-selected index width, but touched-path
  consumer maturity and proof interpretation still need sharper treatment
- the strongest direct, iterative, SVD, and eigensolver owners are explicit
  enough to target surgically rather than widen every family at once
- workflow, package, and install/export surfaces are maintained, but they are
  not the right first implementation center for Sprint 94

Sprint 94 therefore does not begin from "generic feature expansion." It
begins from one explicit truthfulness question:

- where can the repo earn one bounded, valuable capability-breadth
  improvement on the scalar/index seam without overclaiming broad full-library
  complex, mixed-precision, or solver-family maturity

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 94 surfaces:

- strongest public scalar/index contract owners:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `include/sparse_dense.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_eigs.h`
  - `include/sparse_iterative.h`
- strongest shared implementation owners likely to matter:
  - `src/sparse_types.c`
  - `src/sparse_dense.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_eigs.c`
  - `src/sparse_iterative.c`
  - `src/sparse_matrix.c`
- strongest proof-owner tests likely to matter:
  - `tests/test_dense.c`
  - `tests/test_qr.c`
  - `tests/test_svd.c`
  - `tests/test_eigs.c`
  - `tests/test_iterative.c`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
- strongest benchmark and capability-evidence owners:
  - `benchmarks/bench_svd.c`
  - `benchmarks/bench_eigs.c`
  - `benchmarks/bench_iterative_reuse.c`
- strongest support, build, and workflow surfaces if capability work truly
  forces follow-through:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

## Preserved Fence

Sprint 94 is explicitly bounded against:

- reopening the Sprint 93 runtime lane as the first owner again
- promising broad full-library complex or mixed-precision maturity before one
  bounded widened capability seam is actually proved
- treating compile-time index-width configurability alone as equivalent to
  finished touched-path ABI maturity
- widening into broad package/platform claims before the touched scalar and
  index seams are materially stronger
- drifting into generic solver-family expansion detached from one real
  scalar/index contract improvement

## Day 1 Result

Sprint 94 now starts from one precise capability-surface modernization
execution package rather than from a generic "add more numerical features"
bucket. The strongest likely touch surfaces, preserved non-goals, and
maintained reviewed starting truth are fixed in writing before the validation
and maintained-surface recheck begins.
