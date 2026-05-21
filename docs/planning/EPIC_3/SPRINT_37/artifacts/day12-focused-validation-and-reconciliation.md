# Sprint 37 Day 12 Focused Validation and Reconciliation

**Date:** 2026-05-21  
**Branch:** `sprint-37`

## Objective

Re-run the highest-signal Sprint 37 maintainability surfaces before the full
Day 13 sweep:

- the Day 5 shared test-helper cluster
- the Day 6 benchmark-helper pair
- the Day 7 through Day 11 reviewed/dead-code/operator support paths
- the inherited Sprint 36 sanitizer/reset caveat

The goal was to confirm that the refactors stayed behavior-preserving and that
no hidden reconciliation batch remained.

## Validation Scope

### 1. Day 5 test-helper cluster

Built and executed directly:

- `test_iterative`
- `test_bicgstab`
- `test_ilu`
- `test_minres`
- `test_sprint5_integration`
- `test_sprint10_integration`
- `test_sprint12_integration`
- `test_sprint13_integration`

Result:

- all eight binaries passed

Interpretation:

- `tests/test_solver_helpers.h` remains a safe narrow shared header for the
  iterative/preconditioner/integration residual-helper cluster
- no new ownership or behavioral drift surfaced

### 2. Day 6 benchmark-helper pair

Built and executed directly:

- `bench_chol_csc --small-corpus --repeat 1`
- `bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `bench_ldlt_csc --dispatch --repeat 1`

Result:

- all direct benchmark checks passed

Interpretation:

- `bench_backend_compare_helpers.h` remains an appropriate narrow shared layer
  for the Cholesky/LDLT backend-comparison pair
- the refactor did not reopen the Sprint 31 benchmark-contract boundaries

### 3. Sanitizer/reset reconciliation

Commands:

- `make sanitize`
- `make clean`

Result:

- `make sanitize` completed successfully
- the tree was then reset explicitly with `make clean`

Why this matters:

- this confirms the Sprint 36 Day 13 caveat is an operator workflow issue,
  not a latent regression in Sprint 37 code
- tree-mutating modes still require an explicit reset before returning to the
  normal direct/reviewed path
- `make clean` remains the correct canonical reset

### 4. Reviewed/dead-code/operator support path recheck

Validated directly:

- workflow YAML parse:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- `bash -n scripts/deadcode_workflow.sh`
- `python3 -m py_compile scripts/deadcode_report.py`
- serial dead-code path:
  - `make deadcode-report && make deadcode-check`
- reviewed local compile path:
  - `make quality-review-compile`
- reviewed CMake compile parity path:
  - `make quality-review-cmake-compile`

Results:

- all YAML/script checks passed
- serial dead-code report/check passed
- reviewed Makefile compile wrapper passed
- reviewed CMake compile wrapper passed
- `ctest -N` remained `53`
- Makefile/CMake test-count parity remained `53` vs `53`

Important constraint reaffirmed:

- dead-code validation is still authoritative only in serial mode because the
  shared `build/deadcode-cmake` and `build/deadcode/` paths can still race
  under concurrent invocation

## Reconciliation Outcome

Day 12 did not uncover a new implementation batch.

What is now revalidated:

- the helper consolidations from Days 5 and 6 stayed behavior-preserving
- the quality-target normalization and reviewed wrapper guidance from Day 7
  stayed correct
- the structural/reporting cleanup from Day 9 stayed sound
- the wording cleanup from Day 11 did not change the workflow contract
- the sanitizer/reset guidance inherited from Sprint 36 is still the correct
  operator rule

## Residual Queue

No new Day 12 reconciliation queue was created.

Still true:

- dead-code remains serial-only until later shared-path work lands
- tree-mutating instrumentation modes still require `make clean` before
  returning to the normal direct/reviewed path
- larger auxiliary surfaces such as `bench_main.c`, `bench_eigs.c`, and the
  giant feature-owner tests remain explicit non-targets for this sprint

## Day 12 Conclusion

Sprint 37’s highest-signal maintainability changes held up under focused rerun.
The branch remains in the expected state for Day 13:

- no new helper-refactor fallout
- no reviewed-path reconciliation patch required
- no change to the known operational caveats

Day 13 can therefore be the authoritative full validation sweep rather than a
continuation of cleanup.
