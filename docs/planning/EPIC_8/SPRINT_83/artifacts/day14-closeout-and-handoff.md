# Sprint 83 Day 14: Closeout and Handoff

## Purpose

Close Sprint 83 from the validated Day 13 baseline and leave one explicit
handoff queue for Sprint 84 and the later Epic 8 implementation sprints.

## Closeout State

Sprint 83 now closes as one coherent Epic 8 capability-surface modernization
package across:

- capability re-rank
- bounded scalar/index architecture contract
- Day 6 shared matrix-shell scalar-surface expansion
- Day 9 shared scalar/index vocabulary reconciliation
- Day 11 bounded QR public-header widening
- validated Day 13 close baseline

The preserved fence stayed intact:

- the shipped scalar contract still remains real-only `double`
- Sprint 83 widened the public owner reading through `sparse_scalar_t` and
  `idx_t`, not broad numeric genericity
- no SVD public-header widening was reopened inside Sprint 83
- no Cholesky or LDL^T public-header capability widening was reopened inside
  Sprint 83
- no true complex-scalar or mixed-precision claim was widened
- no package, install, export, or runtime-package claim was widened beyond
  the untouched mechanics

## Project-Plan Recheck

`docs/planning/EPIC_8/PROJECT_PLAN.md` does not need a Sprint 83 correction.

The landed Sprint 83 package still supports the intended Epic 8 execution
order:

1. Sprint 84: stronger external differential, seeded-property, and
   failure-path assurance on the touched shared/direct lanes
2. Sprint 85: maintainability work after the widened capability reading and
   assurance surface are stable
3. later SVD/direct-family capability widening, true complex support, mixed
   precision, and broader package/ABI/runtime maturity only where bounded
   evidence justifies the next move

## Validated Baseline

Sprint 83 closes from the Day 13 validated baseline:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 446.47 sec`
- `./build/quality-review-cmake/test_sparse_matrix` -> `59 / 59`
- `./build/quality-review-cmake/test_qr` -> `73 / 73`
- `./build/quality-review-cmake/test_svd` -> `97 / 97`
- `./build/quality-review-cmake/test_chol_csc` -> `149 / 149`
- `./build/quality-review-cmake/test_ldlt` -> `87 / 87`
- `./build/quality-review-cmake/test_integration` -> `53 / 53`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_svd tests/data/suitesparse/nos4.mtx`
- `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `make bench-canonical-report`

This means Sprint 83 hands off from one measured capability-surface baseline
rather than from design intent alone.

## Handoff Queue

The ranked carry-forward queue from Sprint 83 is now fixed explicitly:

1. stronger external differential, seeded-property, and failure-path
   assurance on the touched shared/direct lanes
2. maintainability work after the widened capability reading and assurance
   surface are stable
3. later SVD and direct-family public capability widening only where bounded
   evidence justifies widening
4. later true complex support, mixed precision, and broader package/ABI/runtime
   maturity from the remaining Epic 8 queue

## Bottom Line

Sprint 83 achieved its purpose: the project now has one proof-backed shared
scalar-owner widening on the matrix shell, one reconciled shared scalar/index
vocabulary owner, one bounded QR public-header widening, and one validated
close baseline with the shipped real-only scalar contract still intact.
Sprint 84 can now widen numerical assurance on top of a clearer capability
contract instead of reopening the same owner contradictions first.
