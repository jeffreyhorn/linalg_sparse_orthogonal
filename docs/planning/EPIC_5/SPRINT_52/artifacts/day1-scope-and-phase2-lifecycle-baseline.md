# Sprint 52 Day 1 - scope and Phase 2 lifecycle baseline

Date: 2026-06-01
Branch: `sprint-52`

## Scope

Start Sprint 52 from the actual Sprint 51 public direct-lifecycle Phase 1 end
state and reduce the next work to a bounded deeper-integration queue.

## Authoritative baseline

Sprint 52 starts from a preserved reviewed validation baseline:

- strongest local reviewed baseline: `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

This means Sprint 52 is not a baseline-repair sprint. It is a deeper
integration sprint.

## What Sprint 51 already proved

The following is already real before Sprint 52 begins:

- shared/family public direct header contract refresh
- bounded LU lifecycle integration through the shared path where the default
  option surface fit
- shared Cholesky lifecycle routing through the public
  analysis/factor path
- shared LDL^T lifecycle routing through the public analysis/factor path
- focused lifecycle regression proof in `tests/test_integration.c`
- aligned repeated-run adoption/docs in `examples/README.md` and
  `benchmarks/README.md`

Interpretation:

- Sprint 52 does not need to prove the first public direct repeated-run path
  exists
- Sprint 52 needs to make that path behave more like the real first-class
  repeated-run workflow

## Actual Phase 2 queue

The Sprint 52 project-plan items reduce to seven bounded work classes:

1. analysis contract audit
2. numeric reuse integration
3. refactor path tightening
4. factor-many benchmark proof
5. example/doc adoption
6. regression coverage expansion
7. validation and closeout

The strongest architectural narrowing is:

- keep the work centered on `sparse_analysis_t` / `sparse_factors_t`
- reduce avoidable fallback to one-shot symbolic work
- strengthen same-pattern refactor behavior
- measure the factor-many story explicitly
- do not broaden into a new generic direct abstraction

## Main hotspots

Highest-value touched surfaces at sprint start:

- public contract:
  - `include/sparse_analysis.h` = `355`
  - `include/sparse_lu.h` = `337`
  - `include/sparse_cholesky.h` = `204`
  - `include/sparse_ldlt.h` = `320`
- implementation:
  - `src/sparse_analysis.c` = `626`
  - `src/sparse_lu.c` = `1040`
  - `src/sparse_cholesky.c` = `514`
  - `src/sparse_ldlt.c` = `1494`
- proof/adoption:
  - `tests/test_integration.c` = `1314`
  - `benchmarks/bench_refactor.c` = `159`
  - `benchmarks/bench_refactor_csc.c` = `388`
  - `examples/example_analysis.c` = `191`

Interpretation:

- the strongest risk seams still cluster around shared lifecycle routing and
  direct-family integration
- the strongest proof surfaces remain integration tests, refactor benchmarks,
  and the main repeated-run direct example

## Preserved fence

Sprint 52 still inherits the controlling compatibility boundary:

- one-shot LU / Cholesky / LDL^T remain first-class peer entry points
- one-shot usage remains the simple/default path for one-off solves
- repeated direct runs remain analysis/factors-centric
- reuse preserves symbolic/permutation/setup state, not old numeric factor
  contents
- no raw internal CSC/native storage exposure
- no broad generic direct-handle redesign
- no structural-pattern verifier redesign

## Conclusion

Day 1 fixes Sprint 52’s real starting point:

- preserved reviewed baseline
- implemented Sprint 51 Phase 1 handoff
- bounded Phase 2 deeper-integration queue
- named code/test/benchmark/example hotspots
- preserved compatibility and non-goal fence

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening earlier sprint design decisions.
