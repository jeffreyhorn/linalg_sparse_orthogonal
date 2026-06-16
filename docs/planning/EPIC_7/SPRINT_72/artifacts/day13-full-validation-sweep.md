# Sprint 72 Day 13: Full Validation Sweep

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Validate the landed Sprint 72 branch from the strongest reviewed baseline and
the touched ownership-boundary surfaces before Sprint 72 closeout.

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

### Targeted Sprint 72 follow-ons

```bash
./build/quality-review-cmake/test_sparse_matrix
./build/quality-review-cmake/test_integration
./build/quality-review-cmake/test_chol_csc
./build/quality-review-cmake/example_analysis
./build/quality-review-cmake/example_basic_solve
./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1
./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1
bash tests/test_install.sh
bash tests/test_cmake_install.sh
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
  - `Total Test time (real) = 334.55 sec`

### Touched Sprint 72 proof surfaces

- `./build/quality-review-cmake/test_sparse_matrix` -> `56 / 56`
- `./build/quality-review-cmake/test_integration` -> `48 / 48`
- `./build/quality-review-cmake/test_chol_csc` -> `146 / 146`

### Install/package proof surfaces

- `bash tests/test_install.sh` -> `11 / 11`
- `bash tests/test_cmake_install.sh` -> `13 / 13`

## Representative Retained Outputs

### Examples

- `example_analysis`
  - solve residual stayed `4.44e-16`
- `example_basic_solve`
  - residual stayed `0.00e+00`

### Ownership-boundary proof signals

- `test_integration`
  - retained:
    - `test_reset_perms_invalidates_permuted_lu_shell`
- `test_chol_csc`
  - retained:
    - `test_writeback_publishes_solve_ready_factored_shell`

### Maintained benchmark surfaces

- `bench_refactor_csc nos4`
  - retained row:
    - `bench_refactor_csc,proof,nos4.mtx,chol_spd,100,594,0.266,0.186,0.110,0.009,0.005,1.69,8.24e-16,7.06e-16`
- `bench_chol_csc nos4`
  - retained row:
    - `bench_chol_csc,proof,nos4.mtx,chol_backend_compare,100,594,scalar,supernodal,builtin,0.247,0.312,0.393,0.005,0.004,0.004,0.79,0.63,7.06e-16,5.89e-16,5.89e-16`

### Install/package signals

- `test_install.sh`
  - installed `pkg-config` version stayed `2.2.0`
- `test_cmake_install.sh`
  - installed `pkg-config` version stayed `2.2.0`

## Non-Blocking Note

The reviewed CMake path was still dominated by `test_reorder_nd`:

- `test_reorder_nd = 240.93 sec`
- total reviewed CMake `ctest` time = `334.55 sec`

That remains a runtime note only. The full reviewed path completed cleanly and
all parity anchors stayed exact.

## Bottom Line

Sprint 72 Day 13 closes with the landed branch fully validated:

1. the standard code-day gate passed
2. the strongest reviewed baseline passed with exact parity anchors
3. the touched Sprint 72 ownership proof surfaces retained the expected
   regression signals
4. representative examples, maintained benchmarks, and install/package proof
   scripts all stayed clean
