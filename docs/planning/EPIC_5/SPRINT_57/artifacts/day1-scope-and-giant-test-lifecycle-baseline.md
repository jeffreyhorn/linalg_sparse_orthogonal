# Sprint 57 Day 1 - scope and giant-test / lifecycle baseline

Date: 2026-06-06
Branch: `sprint-57`

## Scope

Start Sprint 57 from the actual Sprint 56 validated close state and the Epic 5
remaining giant-test and regression-proof queue, then reduce the next work to
a bounded test-maintainability and lifecycle/factor-many coverage package
centered on the largest live proof surfaces.

## Authoritative baseline

Sprint 57 starts from a preserved reviewed validation baseline:

- strongest local reviewed baseline: `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

This means Sprint 57 is not a validation-recovery sprint. It is a
maintainability and regression-proof sprint.

## What Sprint 56 already proved

The following is already real before Sprint 57 begins:

- bounded CSC direct-solver and SVD decomposition already landed
- the main CSC/SVD production hotspots were materially reduced:
  - `src/sparse_ldlt_csc.c`: `2723 -> 2127`
  - `src/sparse_chol_csc.c`: `2194 -> 1532`
  - `src/sparse_svd.c`: `1728 -> 1319`
- no public header/API redesign was needed
- one-shot solver APIs remain first-class supported entry points
- the repeated direct-run analysis/factors lifecycle remains the validated
  direct-solver contract
- the repeated-run iterative/eigensolver support boundary remains the
  validated Sprint 54 shape
- build-surface parity was already preserved across:
  - `Makefile`
  - `CMakeLists.txt`

Interpretation:

- Sprint 57 does not need to re-decide the public solver or lifecycle surface
- Sprint 57 needs to improve test ownership and proof clarity while preserving
  that already-validated public contract

## What the Epic 5 review and todo list already fixed as the next queue

The Epic 5 review and todo notes already point to the same bounded
maintainability and proof problem:

- several core test binaries are now giant enough to be maintainability
  hotspots
- the primary named targets are:
  - `test_chol_csc`
  - `test_svd`
  - `test_ldlt_csc`
  - `test_qr`
  - `test_etree`
  - `test_iterative`
- direct lifecycle coverage still needs final public-model proof shaping
- benchmark and example parity checks should stay coupled to any regression
  expansion where they prove caller stories directly

The live repo state confirms that the queue is still current:

- `tests/test_chol_csc.c` = `4643` lines
- `tests/test_svd.c` = `3746` lines
- `tests/test_ldlt_csc.c` = `3680` lines
- `tests/test_qr.c` = `3197` lines
- `tests/test_iterative.c` = `2993` lines
- `tests/test_etree.c` = `2962` lines
- `tests/test_integration.c` = `1803` lines

Interpretation:

- Sprint 57 should treat the Epic 5 giant-test queue as still live, not
  historical
- the strongest remaining maintainability pressure is now in proof surfaces
  rather than production implementations

## Actual Sprint 57 queue

The Sprint 57 project-plan items reduce to six bounded work classes:

1. large-test audit
2. direct-solver test refactor batch
3. iterative / eigensolver test refactor batch
4. lifecycle regression expansion
5. factor-many / compatibility regression expansion
6. validation and closeout

The strongest architectural narrowing is:

- keep the work centered on giant proof surfaces first
- prefer helper-vs-proof-group ownership splits over generic mechanical test
  splits
- preserve the Sprint 50-56 public and lifecycle fence exactly
- do not broaden into public API redesign, solver-family expansion, or helper-
  driven coverage churn

## Main hotspots

Highest-value touched surfaces at sprint start:

- giant tests:
  - `tests/test_chol_csc.c` = `4643`
  - `tests/test_svd.c` = `3746`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_qr.c` = `3197`
  - `tests/test_iterative.c` = `2993`
  - `tests/test_etree.c` = `2962`
  - `tests/test_integration.c` = `1803`
- benchmark surfaces:
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_iterative_reuse.c` = `370`
  - `benchmarks/bench_eigs_reuse.c` = `253`
- caller-facing examples:
  - `examples/example_eigs.c` = `285`
  - `examples/example_analysis.c` = `210`
  - `examples/example_iterative.c` = `144`
- truthfulness / maintainer docs:
  - `README.md` = `987`
  - `docs/maintainer_guide.md` = `294`

Interpretation:

- the strongest direct-solver test seams are now in:
  - `test_chol_csc`
  - `test_ldlt_csc`
  - `test_integration`
- the strongest non-direct seam candidates are:
  - `test_svd`
  - `test_iterative`
  - then `test_qr` / `test_etree` as secondary queue
- the benchmark and example surfaces remain small enough to stay as parity and
  caller-story truthfulness anchors rather than as primary refactor targets

## Preserved fence

Sprint 57 still inherits the controlling compatibility and non-goal boundary:

- one-shot APIs remain first-class peer entry points
- the analysis/factors repeated direct-run path remains the validated direct
  lifecycle shape
- repeated-run iterative/eigensolver handles remain the validated support set
- no broad public API redesign
- no reopening of the direct-solver lifecycle contract
- no solver-family expansion disguised as test work
- no helper-driven coverage churn that weakens behavior-level confidence

## Conclusion

Day 1 fixes Sprint 57's real starting point:

- preserved reviewed baseline
- inherited validated decomposition and public-contract fence
- bounded remaining giant-test maintainability queue
- named test, benchmark, and example hotspots
- explicit non-goal fence against public API or feature expansion

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening Sprint 50-56 public contract decisions.
