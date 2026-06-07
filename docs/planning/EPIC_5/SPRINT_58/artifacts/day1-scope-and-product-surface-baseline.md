# Sprint 58 Day 1 - scope and product-surface baseline

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Start Sprint 58 from the actual Sprint 57 validated close state and the Epic 5
remaining public-surface cleanup queue, then reduce the next work to a bounded
documentation/examples/benchmark simplification package centered on the
strongest live caller-facing surfaces.

## Authoritative baseline

Sprint 58 starts from a preserved reviewed validation baseline:

- strongest local reviewed baseline: `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

This means Sprint 58 is not a validation-recovery sprint. It is a
product-surface simplification sprint.

## What Sprint 57 already proved

The following is already real before Sprint 58 begins:

- bounded giant-test maintainability improvement already landed where the
  cleanest proof-family seams existed
- public direct repeated-run lifecycle proof was tightened
- factor-many / one-shot compatibility proof was tightened
- no public header/API redesign was needed
- one-shot APIs remain first-class supported entry points
- the repeated direct-run analysis/factors lifecycle remains the validated
  direct-solver contract
- the repeated-run iterative/eigensolver support boundary remains the
  validated Sprint 54 shape
- benchmark/example workflow shape remained stable through Sprint 57

Interpretation:

- Sprint 58 does not need to re-decide the public solver or lifecycle surface
- Sprint 58 needs to make the public story easier to scan while preserving
  that already-validated contract

## What the Epic 5 review and todo list already fixed as the next queue

The Epic 5 review and todo notes already point to the same bounded
product-surface simplification problem:

- remove stale sprint-history framing from permanent public headers and README
  sections
- keep planning chronology in `docs/planning/` instead of public API surfaces
- normalize lifecycle guidance across:
  - README
  - tutorial
  - examples
  - benchmark docs
  - public headers
- keep the one-shot-first story where appropriate, but make the advanced
  lifecycle story equally clear

The inherited review guidance remains concrete:

- `README.md` is still strong but larger and more detailed than the final
  product surface likely needs
- `include/sparse_eigs.h` was called out directly as carrying stale
  sprint-history framing
- `benchmarks/README.md` and parts of the main README still describe benchmark
  surfaces in sprint-local terms rather than product-level terms
- `examples/README.md` remains a high-value public entry surface for workflow
  alignment

Interpretation:

- Sprint 58 should treat the Epic 5 docs/examples/benchmark cleanup queue as
  still live
- the strongest remaining maintainability pressure is now caller-facing wording
  and workflow framing rather than implementation ownership

## Actual Sprint 58 queue

The Sprint 58 project-plan items reduce to six bounded work classes:

1. public docs audit
2. README/tutorial reduction
3. public-header narrative cleanup
4. example modernization
5. benchmark taxonomy cleanup
6. sanity sweep and closeout

The strongest architectural narrowing is:

- keep the work centered on stable workflow guidance first
- prefer reduction and simplification over broader explanatory expansion
- preserve the Sprint 50-57 public and lifecycle fence exactly
- do not broaden into public API redesign, solver-family expansion, or
  benchmark/framework redesign

## Main hotspots

Highest-value touched surfaces at sprint start:

- top-level docs:
  - `README.md` = `987`
  - `docs/tutorial.md` = `415`
- public headers:
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `687`
  - `include/sparse_analysis.h` = `375`
  - `include/sparse_lu.h` = `337`
  - `include/sparse_ldlt.h` = `334`
  - `include/sparse_cholesky.h` = `204`
- example docs and examples:
  - `examples/README.md` = `134`
  - `examples/example_eigs.c` = `285`
  - `examples/example_ic_minres.c` = `232`
  - `examples/example_analysis.c` = `210`
  - `examples/example_iterative.c` = `144`
  - `examples/example_svd_lowrank.c` = `120`
- benchmark docs and benchmark surfaces:
  - `benchmarks/README.md` = `235`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_iterative_reuse.c` = `370`
  - `benchmarks/bench_refactor.c` = `303`
  - `benchmarks/bench_eigs_reuse.c` = `253`

Interpretation:

- the strongest top-level docs reduction pressure is `README.md` first, then
  `docs/tutorial.md`
- the strongest public-header narrative cleanup pressure is in
  `include/sparse_iterative.h` and `include/sparse_eigs.h`
- the example and benchmark docs are smaller, but they are still high-value
  because they directly teach the final caller workflow story

## Preserved fence

Sprint 58 still inherits the controlling compatibility and non-goal boundary:

- one-shot APIs remain first-class peer entry points
- the analysis/factors repeated direct-run path remains the validated direct
  lifecycle shape
- repeated-run iterative/eigensolver handles remain the validated support set
- no broad public API redesign
- no reopening of the direct-solver lifecycle contract
- no solver-family expansion disguised as docs or example work
- no benchmark/framework redesign disguised as taxonomy cleanup

## Conclusion

Day 1 fixes Sprint 58's real starting point:

- preserved reviewed baseline
- inherited validated public-contract fence
- bounded remaining public docs/header/example/benchmark cleanup queue
- named caller-facing hotspots
- explicit non-goal fence against public API or feature expansion

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening Sprint 50-57 public contract decisions.
