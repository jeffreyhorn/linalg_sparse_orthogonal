# Sprint 65 Day 2: Validation Baseline and Touched-Surface Recheck

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Freeze the validation and truthfulness baseline that later Sprint 65
benchmark-governance, solver-efficiency, and regression-reporting work must
preserve before the sprint moves into the deeper benchmark-role audit.

## Rechecked Surfaces

- reviewed count anchor:
  - `ctest -N --test-dir build/quality-review-cmake`
- reviewed wrapper expansion:
  - `make -n quality-review-full`
- user-facing quality/truthfulness wording:
  - `README.md`
- maintainer-policy interpretation:
  - `docs/maintainer_guide.md`
- exact target semantics:
  - `Makefile`
- platform-truthfulness wording:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- targeted Sprint 65 rerun-set presence:
  - `build/`

## Day 2 Baseline Conclusions

### 1. The strongest local reviewed baseline is still `make quality-review-full`

Sprint 65 inherits the same authoritative local validation command as the
Sprint 64 close state:

- `make quality-review-full`

That remains the strongest local reviewed baseline because it preserves both:

- the reviewed Makefile path
- the reviewed CMake parity path

This should remain the top-level local trust anchor unless a later Epic 6
implementation sprint proves that the contract itself must change.

### 2. The reviewed CMake parity count is still the main numerical truthfulness anchor

The current reviewed CMake inventory remains:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

That count still matters because it is the simplest exact proof that:

- the reviewed CMake path still sees the maintained local full test surface
- Makefile/CMake parity has not drifted silently

### 3. The current code-day gate versus stronger reviewed baseline split is stable

The maintained split is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial benchmark-governance, solver-efficiency, or
  regression-reporting work:
  - `make quality-review-full`
- docs-only days:
  - no automatic code-quality gate required
  - use targeted sanity checks instead

This remains consistent with the repo’s current Sprint 64 close discipline and
does not need reinterpretation on Sprint 65 Day 2.

### 4. The current quality/platform story is coherent across README, maintainer guide, Makefile, and workflows

The main maintained surfaces still agree on the current contract:

- Linux remains the enforced reviewed source-of-truth path
- macOS remains reviewed but narrower, with dead-code still staged
- Windows keeps the reviewed CMake subset enforced while the broader Makefile
  reviewed wrappers stay staged
- coverage remains a supplemental signal, not an active reviewed-baseline
  residual
- dead-code remains serialized and separate from the core format/lint/test
  gate

That means Sprint 65 can proceed from a stable truthfulness contract rather
than needing a wording-reconciliation batch just to start benchmark taxonomy
or solver-efficiency work.

### 5. The targeted Sprint 65 rerun set is present and aligned to the actual benchmark and solver-risk surface

The confirmed rerun set is:

- direct lifecycle and CSC proof surfaces:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_sparse_lu`
- adjacent dense-kernel and spectral sentinels:
  - `./build/test_qr`
  - `./build/test_svd`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_svd_lowrank`
- representative workflow benchmarks:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_ldlt_csc`
  - `./build/bench_eigs_reuse`
  - `./build/bench_iterative_reuse`

That is already strong enough to support:

- benchmark-role and output/taxonomy normalization on the maintained benchmark
  proof surface
- representative direct/CSC proof after behavior-affecting efficiency edits
- repeated-run benchmark follow-through without pretending every exploratory
  bench binary must become authoritative
- adjacent regression verification so Sprint 65 does not widen unrelated
  backend or platform claims by accident

## Authoritative Day 2 Validation Boundary

- docs-only days:
  - use targeted sanity checks, not the full code-day gate by default
- bounded `*.c` / `*.h` days:
  - run:
    - `make format`
    - `make lint`
    - `make test`
- substantial benchmark-governance, solver-efficiency, or regression-reporting
  code days:
  - prefer:
    - `make quality-review-full`
  - and refresh representative benchmark/example/proof surfaces as needed

## Day 2 Exit State

Sprint 65 now has a written validation baseline that matches the live repo:

- strongest local reviewed baseline unchanged
- reviewed CMake parity anchor unchanged
- rerun set fixed from the current build tree around maintained benchmark
  proof, direct/CSC proof, and adjacent solver sentinels
- docs-only versus code-day versus stronger-review path split fixed explicitly
- no contradiction across the main quality/truthfulness surfaces
