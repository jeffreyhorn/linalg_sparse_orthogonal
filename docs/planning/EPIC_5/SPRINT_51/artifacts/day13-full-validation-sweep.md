# Sprint 51 Day 13 - Full validation sweep

Date: 2026-06-01
Branch: `sprint-51`

## Scope

Run the full validation sweep for the Sprint 51 public direct-solver lifecycle
Phase 1 landing and record the maintained reviewed-baseline truthfulness
anchors.

## Commands run

Full gate:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Truthfulness anchor recheck:

- `ctest -N --test-dir build/quality-review-cmake`

Sprint 51 targeted follow-ons:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

## Results

### 1. Required full gate passed

All required code-day validation commands passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

This confirms that the Day 4-10 header/source/test/adoption work closes from a
clean validation state.

### 2. Reviewed baseline remained exact

The strongest local reviewed baseline remains:

- `make quality-review-full`

The maintained reviewed parity anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 500.45 sec`

### 3. Sprint 51 targeted follow-ons passed

All targeted direct-lifecycle follow-ons passed:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Representative direct results:

- `example_analysis`
  - initial solve residual: `4.44e-16`
  - sampled refactor residuals remained `4.44e-16`
- `bench_refactor`
  - `tridiag-200`: `1.01x`
  - `tridiag-500`: `1.04x`
  - `bcsstk04`: `1.04x`
  - `nos4`: `0.76x`
- `bench_refactor_csc`
  - `nos4`: `2.34x`
  - `bcsstk04`: `2.81x`
  - `bcsstk14`: `5.36x`
  - `s3rmt3m3`: `7.87x`
  - `Kuu`: `6.33x`
  - `Pres_Poisson`: `12.14x`
- family regression reruns:
  - `test_cholesky`: `21 / 21`
  - `test_ldlt`: `83 / 83`
  - `test_etree`: `97 / 97`
  - `test_chol_csc`: `137 / 137`
  - `test_ldlt_csc`: `95 / 95`

## Interpretation

The Sprint 51 public direct-lifecycle Phase 1 landing is now validated from
three angles at once:

- repository-wide code-day gates stayed green
- strongest reviewed Makefile/CMake parity stayed exact
- direct repeated-run analysis/factor/refactor surfaces still behaved correctly
  on the strongest example, benchmark, and direct-factor family follow-ons

The one-shot direct APIs remain compatible, the repeated-run direct path remains
analysis/factors-centric, and no new blocker surfaced in the validation sweep.

## Conclusion

Day 13 validation is complete. Sprint 51 is ready for Day 14 closeout and
handoff.
