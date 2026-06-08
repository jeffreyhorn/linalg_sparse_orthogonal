# Sprint 58 Day 14 - closeout and handoff

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Package Sprint 58 as one coherent documentation/examples/benchmark
simplification handoff from the Day 13 validated baseline.

## Final deliverables

### Public docs reduction

- `README.md`
- `docs/tutorial.md`

Result:

- top-level workflow framing is shorter and more product-level
- one-shot vs repeated-run positioning is clearer
- the highest-signal tutorial guidance now aligns with the simplified README

### Public-header narrative cleanup

- `include/sparse_eigs.h`
- `include/sparse_iterative.h`

Result:

- stale sprint/future-work framing is reduced
- repeated-run support-boundary wording is normalized to the final product
  story
- public semantics and ABI shape remain unchanged

### Example modernization

- `examples/example_eigs.c`
- `examples/README.md`

Result:

- the strongest remaining shipped example-side narrative offender is now
  normalized to the stable product story
- the example docs still preserve the one-shot-first posture and explicit
  repeated-run-handle boundary

### Benchmark taxonomy cleanup

- `benchmarks/README.md`

Result:

- the benchmark story now reads as stable workflow groups:
  - one-shot compatibility/comparison
  - direct repeated-run lifecycle
  - iterative public-handle reuse
  - eigensolver public-handle reuse

## Measured touched-surface end state

- `README.md`: `973`
- `docs/tutorial.md`: `453`
- `include/sparse_eigs.h`: `646`
- `include/sparse_iterative.h`: `765`
- `examples/README.md`: `134`
- `examples/example_eigs.c`: `287`
- `benchmarks/README.md`: `248`

## Preserved compatibility fence

Sprint 58 closes with the steady-state workflow fence intact:

- one-shot APIs remain first-class/default workflows
- repeated-run direct solves remain an analyze-once / factor-many path
- repeated-run iterative handles remain limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- repeated-run eigensolver handles remain limited to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows remain one-shot compatibility
  surfaces

## Final validated baseline

Carried forward from Day 13:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 481.74 sec`

## Deferred queue

Explicit future-facing residuals after Sprint 58:

- deeper long-form `README.md` chronology/performance-history cleanup
- any lower-priority public-header follow-through only if a later contradiction
  appears
- broader docs-density reduction outside the bounded Sprint 58 target set

## Project-plan check

`docs/planning/EPIC_5/PROJECT_PLAN.md` does not need a Sprint 58 correction.

## Conclusion

Sprint 58 closes as one coherent simplified public-surface package:

- top-level docs are clearer
- public header narrative drift is reduced
- the highest-value example surface is modernized
- benchmark taxonomy is more stable and workflow-first
- the reviewed validation baseline remains explicit
- the residual queue is conscious future work rather than hidden drift
