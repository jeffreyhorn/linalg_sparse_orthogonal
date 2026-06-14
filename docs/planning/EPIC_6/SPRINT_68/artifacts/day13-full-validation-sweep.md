# Sprint 68 Day 13: Full Validation Sweep

## Goal

Validate the landed Sprint 68 branch from the strongest reviewed baseline and
the touched giant-test/oracle/property surfaces before Sprint 68 closeout.

## Validation Commands

### Standard code-day gate

```bash
make format
make lint
make test
```

### Strongest reviewed baseline

```bash
make quality-review-full
```

### Targeted Sprint 68 follow-ons

```bash
./build/test_integration
./build/test_chol_csc
./build/test_fuzz
./build/test_framework_optin
./build/test_reorder_nd
./build/example_analysis
./build/example_basic_solve
./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1
./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1
./build/bench_iterative_reuse
./build/bench_eigs_reuse
```

## Results

### Standard gate

- `make format` passed
- `make lint` passed
- `make test` passed

### Reviewed baseline

- `make quality-review-full` passed
- reviewed CMake parity stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - full reviewed CMake `ctest` = `53 / 53`
  - `Total Test time (real) = 465.15 sec`

### Touched Sprint 68 proof surfaces

- `./build/test_integration` -> `47 / 47`
- `./build/test_chol_csc` -> `145 / 145`
- `./build/test_fuzz` -> `25 / 25`
- `./build/test_framework_optin` -> `8` run, `3` skipped, `0` failed
- `./build/test_reorder_nd` -> `34 / 34`

## Representative Retained Outputs

### Examples

- `example_analysis`
  - solve residual stayed `4.44e-16`
- `example_basic_solve`
  - residual stayed `0.00e+00`

### Giant-test / assurance signals

- `test_fuzz`
  - retained:
    - `large-n CSC lifecycle property: 3/3 passed`
- `test_reorder_nd`
  - retained:
    - `Pres_Poisson ND/AMD = 0.923`
    - `bcsstk14 ND/AMD = 1.124`

### Maintained benchmark surfaces

- `bench_refactor_csc nos4`
  - retained row:
    - `bench_refactor_csc,proof,nos4.mtx,chol_spd,...,1.85,8.24e-16,7.06e-16`
- `bench_chol_csc nos4`
  - retained row:
    - `bench_chol_csc,proof,nos4.mtx,chol_backend_compare,...,scalar,supernodal,builtin,...,0.91,0.98,7.06e-16,5.89e-16,5.89e-16`
- `bench_iterative_reuse`
  - retained rows:
    - `cg-tridiag-300` -> `1.05x`
    - `gmres-unsym-220` -> `1.02x`
    - `minres-kkt-42` -> `1.02x`
- `bench_eigs_reuse`
  - retained rows:
    - `growm-nos4-k5` -> `1.12x`
    - `thick-bcsstk14-k5` -> `1.02x`
    - `lobpcg-diag40-k3` -> `1.06x`
    - `lambda_max_diff = 0.000e+00`

## Non-Blocking Note

The reviewed CMake path was still dominated by `test_reorder_nd`:

- `test_reorder_nd = 320.42 sec`
- total reviewed CMake `ctest` time = `465.15 sec`

That remains a runtime note only. The full reviewed path completed cleanly and
all parity anchors stayed exact.

## Bottom Line

Sprint 68 Day 13 closes with the landed branch fully validated:

1. the standard code-day gate passed
2. the strongest reviewed baseline passed with exact parity anchors
3. the touched Sprint 68 owner surfaces and representative examples/benchmarks
   all retained the expected proof signals
