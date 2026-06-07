# Sprint 57 Day 12 - post-expansion compatibility audit

Date: 2026-06-06 19:46:12 CDT  
Branch: `sprint-57`

## Goal

Audit the landed Sprint 57 branch after the giant-test refactors and lifecycle
regression additions, then record:

- whether any proof-surface or wording drift remains
- what is intentionally deferred
- the exact Day 13 validation checklist from the landed state

## Audit basis

Reviewed surfaces:

- direct proof surfaces
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_integration.c`
- solver-family proof surfaces
  - `tests/test_svd.c`
  - `tests/test_iterative.c`
  - extracted helper seams in
    - `tests/test_chol_csc_supernodal_helpers.h`
    - `tests/test_svd_partial_helpers.h`
    - `tests/test_iterative_handle_helpers.h`
- caller-facing wording
  - `README.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `docs/tutorial.md`
- branch diff shape
  - `git diff --stat master...HEAD`

## Main audit result

The landed branch still matches the intended steady-state contract:

- no public API redesign
- no support-boundary drift
- no benchmark/example workflow drift
- no hidden solver-behavior expansion

The strongest compatibility fact is structural:

- `master...HEAD` still has no `include/` changes at all

That keeps Sprint 57 firmly in the proof-surface / maintainability category
rather than turning it into an untracked product-surface sprint.

## What is now intentionally deferred

### Direct-solver proof density

- `tests/test_ldlt_csc.c` is now the strongest deferred direct-solver
  giant-test seam
- `tests/test_integration.c` remains intentionally dense because it is the
  main public direct lifecycle / factor-many caller story

### Solver-family proof density

- `tests/test_qr.c` remains deferred
- the retained density in `tests/test_svd.c` and `tests/test_iterative.c` is
  now more intentional after the helper extractions

## Caller-facing wording check

No blocker-level wording drift surfaced:

- README still states the repeated-run direct contract in the intended order:
  analyze once, factor / solve, refactor / solve many on same-pattern values
- one-shot LU / Cholesky / LDL^T wording remains first-class
- `benchmarks/README.md` still matches the live benchmark proof surfaces
- `examples/README.md` and `docs/tutorial.md` still describe shipped examples
  as one-shot-first while naming the bounded repeated-run support surfaces

## Final validation checklist

Day 13 should validate the landed branch with:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`

Targeted follow-ons:

- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/test_svd`
- `./build/test_iterative`
- `./build/test_integration`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

## Conclusion

No blocker-level contract drift remains before the final validation sweep.

Sprint 57 is now cleanly positioned for Day 13-14 as:

- giant-test maintainability follow-through
- direct lifecycle proof tightening
- factor-many / one-shot compatibility proof tightening

with the residual queue reduced to consciously deferred proof-density work
rather than unresolved branch drift.
