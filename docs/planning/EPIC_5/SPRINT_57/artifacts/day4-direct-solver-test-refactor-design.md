# Sprint 57 Day 4 - direct-solver test refactor design

Date: 2026-06-06
Branch: `sprint-57`

## Scope

Freeze the first bounded direct-solver giant-test refactor boundary before
editing permanent proof surfaces, using the Day 3 `test_chol_csc.c`
supernodal-first ranking as the design anchor.

## Selected first refactor seam

Sprint 57 Batch 1 should target:

- `tests/test_chol_csc.c`

The first owned seam should be:

- the family-local helper layer that supports the supernodal / writeback /
  dispatch proof cluster

Recommended new helper file:

- `tests/test_chol_csc_supernodal_helpers.h`

This keeps the test binary build shape unchanged while still creating a real
owned seam inside the largest direct-solver giant test.

## Why the first batch should use a helper header instead of a new test binary

The live build pattern is still:

- `test_chol_csc` is built from one source file
- `test_ldlt_csc` is built from one source file
- `test_integration` is built from one source file

The repo already has precedent for narrow local helper headers:

- `tests/test_solver_helpers.h`
- `benchmarks/bench_backend_compare_helpers.h`
- `examples/example_alloc_helpers.h`

Maintainability conclusion:

- keep Day 5 build-neutral
- avoid Makefile/CMake target-shape churn in the first test refactor batch
- create an owned helper seam first, then re-evaluate proof-block splitting
  after the main hotspot is simpler

## Exact ownership split

### Move into `tests/test_chol_csc_supernodal_helpers.h`

Family-local helpers that serve only the selected supernodal/writeback/dispatch
proof family, including:

- `detect_supernodes_alloc(...)`
- `day8_count_supernodes(...)`
- `day9_assert_batched_matches_scalar(...)`
- `day11_build_spd(...)`
- any small adjacent helper needed only by the same proof family

### Keep in `tests/test_chol_csc.c`

- alloc/grow smoke tests
- conversion / round-trip / permutation / analysis / validate groups
- workspace and scalar elimination groups
- solve / factor-solve groups
- the supernodal, writeback, and dispatch tests themselves in Batch 1
- `main()` and current grouped `RUN_TEST(...)` ordering

## Why the tests themselves stay in the main file in Batch 1

The selected family is large enough to justify a real seam, but the first step
should stay narrower:

- helper ownership is the clearest low-risk first extraction
- the supernodal tests remain grouped and readable in the existing runner
- keeping test bodies in place minimizes churn in:
  - line ordering
  - section structure
  - `RUN_TEST(...)` grouping
  - review diff size

Phase-1 maintainability conclusion:

- extract the family-local helper seam first
- keep proof bodies in `test_chol_csc.c` for the first landing
- use Day 6 to decide whether a second owned include-style proof block is worth
  it

## Bounded non-goal fence

Batch 1 should not:

- create a new multi-file test binary topology
- widen `tests/test_solver_helpers.h` into a broad CSC test framework
- move `test_ldlt_csc.c` or `test_integration.c` logic
- normalize the whole `test_chol_csc.c` comment body
- reopen lifecycle or factor-many contract decisions

## Invariants the first batch must preserve

### Binary and runner invariants

- the `test_chol_csc` binary remains the same test target
- `main()` remains in `tests/test_chol_csc.c`
- `RUN_TEST(...)` ordering and test names remain stable

### Proof invariants

- scalar-path and supernodal-path proof meanings remain unchanged
- writeback and dispatch proof meanings remain unchanged
- scalar↔batched parity checks remain unchanged
- residual thresholds remain unchanged

### Fixture and caller-story invariants

- SuiteSparse fixture coverage remains intact:
  - `nos4`
  - `bcsstk04`
  - `bcsstk14`
  - `Kuu`
- one-shot compatibility and dispatch proof boundaries remain intact
- repeated-run caller-story parity remains unchanged relative to:
  - `benchmarks/bench_refactor_csc.c`
  - `examples/example_analysis.c`

This is an ownership change, not a behavior change.

## Minimal comment policy

Batch 1 should:

- preserve durable proof commentary
- preserve comments that explain:
  - residual thresholds
  - structural expectations
  - corpus-safety meaning
- trim stale sprint-history narrative only where touched inside the moved
  helper seam

Batch 1 should not:

- perform a whole-file historical comment purge
- rewrite retained scalar-path sections that are outside the moved seam

## Expected Day 5 touched files

Primary expected touched set:

- `tests/test_chol_csc.c`
- `tests/test_chol_csc_supernodal_helpers.h` (new)

Secondary touch only if genuinely needed:

- `Makefile`
- `CMakeLists.txt`

Avoid by default:

- `tests/test_ldlt_csc.c`
- `tests/test_integration.c`
- `tests/test_solver_helpers.h`
- `benchmarks/bench_refactor_csc.c`
- `examples/example_analysis.c`

## Validation checklist

Required code-day validation:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Primary proof surfaces:

- `./build/test_chol_csc`
- `./build/test_integration`

Secondary parity surfaces:

- `./build/test_cholesky`
- `./build/bench_refactor_csc`
- `./build/example_analysis`

## Conclusion

Day 4 fixes the first Sprint 57 direct-test refactor boundary explicitly:

- target the largest direct-solver giant test:
  - `tests/test_chol_csc.c`
- extract the supernodal family’s local helper seam into:
  - `tests/test_chol_csc_supernodal_helpers.h`
- keep the test binary shape, runner, and main proof bodies stable in Batch 1
- preserve proof meaning, corpus coverage, and caller-story parity exactly

That gives Day 5 a precise, low-risk, build-neutral landing plan.
