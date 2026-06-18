# Sprint 79 Day 13 - Full Validation Sweep

Date: 2026-06-18  
Branch: sprint-79

## Purpose
Run the full Sprint 79 closeout validation queue from the integrated tree,
confirm that the new public LDL^T oracle/property coverage survives the full
reviewed baseline, and verify that the maintained reporting and install/export
surfaces still pass from the same post-fix state.

## Main Result
Sprint 79 Day 13 validation completed cleanly, but only after fixing one real
Makefile dependency bug uncovered by the install-proof rerun.

The surfaced issue was not in the landed Sprint 79 assurance work itself. It
was in the build/install dependency graph:

- `tests/test_install.sh` exposed that `make install` after `make clean` could
  compile library objects before generating `build/include/sparse_version.h`
- the library object rule now depends explicitly on `$(GENERATED_VERSION)`
- after that fix, the full validation queue and all follow-ons passed from a
  coherent post-fix state

## Landed Fix
The only code change in the Day 13 sweep is in `Makefile`:

- library object compilation now depends directly on the generated version
  header
- this makes the clean install path truthful and stable for translation units
  that include `sparse_version.h`

## Full Validation Baseline
The final validated baseline is now explicit:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 451.58 sec`

## Focused Follow-Ons
The touched Sprint 79 proof-owner and example follow-ons also all passed:

- `./build/quality-review-cmake/test_integration` -> `51 / 51`
- `./build/quality-review-cmake/test_fuzz` -> `26 / 26`
- `./build/quality-review-cmake/test_chol_csc` -> `147 / 147`
- `./build/quality-review-cmake/test_ldlt` -> `84 / 84`
- `./build/quality-review-cmake/test_ldlt_csc` -> `96 / 96`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `make bench-canonical-report`
- `bash tests/test_install.sh` -> `11 / 11`
- `bash tests/test_cmake_install.sh` -> `13 / 13`

## Representative Retained Outputs
The highest-signal retained outputs stayed clean:

- `test_fuzz` retained:
  - `large-n LDLT CSC lifecycle property: 3/3 passed`
- `test_chol_csc` retained:
  - `tests/data/suitesparse/bcsstk14.mtx: n=1806, rel_residual=1.080e-15`
- `test_ldlt` retained:
  - `KKT 500x500: relres=4.465e-17, nnz(L)=1298`
  - `bcsstk04 LDL^T vs Cholesky: max|diff| = 1.427e-14`
- `test_ldlt_csc` retained:
  - `tridiag indefinite n=10: rel_res = 0.000e+00`
  - `arrow 6x6 indefinite (AMD): rel_res = 9.869e-17`
- `example_analysis` residual stayed `4.44e-16`
- `example_basic_solve` residual stayed `0.00e+00`

The canonical report bundle also regenerated cleanly and retained the expected
proof-facing rows:

- `bench_refactor_csc nos4`:
  - `speedup_refactor = 1.01`
  - residuals `8.24e-16` / `7.06e-16`
- `bench_chol_csc nos4`:
  - `csc_supernodal_panel_solver = batched_panel`
  - residuals `7.06e-16`, `5.89e-16`, `5.89e-16`
- `bench_iterative_reuse gmres-unsym-220`:
  - speedup `1.07`
- `bench_iterative_reuse minres-kkt-42`:
  - speedup `1.24`
- `bench_eigs_reuse thick-bcsstk14-k5`:
  - speedup `0.95`
- `bench_eigs_reuse lobpcg-diag40-k3`:
  - speedup `1.04`
- both install regressions retained installed `pkg-config` version `2.2.0`

## Non-Blocking Note
One runtime note is now explicit for closeout:

- reviewed CMake `test_reorder_nd` still dominated runtime at `315.52 sec` out
  of the `451.58 sec` total
- despite that runtime skew, the full reviewed path completed cleanly and all
  maintained parity anchors stayed exact

## Exit State
- Sprint 79 now closes toward Day 14 from a real post-fix validated baseline.
- The new public LDL^T oracle/property package survived:
  - full gates
  - reviewed parity
  - representative examples
  - canonical reporting
  - install/export proof
- The Day 13 sweep therefore improved the tree in one real way beyond
  validation alone: it removed a latent clean-install dependency bug before
  Epic 7 closeout.
