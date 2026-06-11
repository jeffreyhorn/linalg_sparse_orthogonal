# Sprint 63 Day 2: Validation Baseline and Touched-Surface Recheck

Date: 2026-06-10
Branch: `sprint-63`

## Purpose

Freeze the validation and truthfulness baseline that later Sprint 63
direct-lifecycle implementation work must preserve before the sprint moves
into the deeper LU follow-through, CSC repeated-run uniformity, and
solve/refactor semantics audit.

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
- targeted Sprint 63 rerun-set presence:
  - `build/`

## Day 2 Baseline Conclusions

### 1. The strongest local reviewed baseline is still `make quality-review-full`

Sprint 63 inherits the same authoritative local validation command as the
Sprint 62 close state:

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
- stronger default for substantial direct-lifecycle or CSC-sensitive work:
  - `make quality-review-full`
- docs-only days:
  - no automatic code-quality gate required
  - use targeted sanity checks instead

This remains consistent with the repo’s current Sprint 62 close discipline and
does not need reinterpretation on Sprint 63 Day 2.

### 4. The current quality/platform story is coherent across README, maintainer guide, Makefile, and workflows

The main maintained surfaces still agree on the current contract:

- Linux remains the enforced reviewed source-of-truth path
- macOS remains reviewed but narrower, with dead-code still staged
- Windows keeps the reviewed CMake subset enforced while the Makefile reviewed
  wrappers and dead-code flow stay staged
- coverage remains a supplemental signal, not an active reviewed-baseline
  residual
- dead-code remains operationally serialized and separate from `lint` and
  `test`

That means Sprint 63 can proceed from a stable truthfulness contract rather
than needing a wording-reconciliation batch just to start lifecycle
implementation work.

### 5. The targeted Sprint 63 rerun set is present and aligned to the actual direct lifecycle-risk surface

The confirmed rerun set is:

- direct lifecycle and integration proofs:
  - `./build/test_integration`
- direct solver family proofs:
  - `./build/test_sparse_lu`
  - `./build/test_cholesky`
  - `./build/test_chol_csc`
  - `./build/test_ldlt`
  - `./build/test_ldlt_csc`
- adjacent repeated-run solver proofs:
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
- representative workflow benchmarks:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

That is already strong enough to support:

- LU repeated-run follow-through
- CSC lifecycle and dispatch verification
- representative direct example sanity checks after behavior-affecting edits
- adjacent repeated-run regression verification so Sprint 63 does not widen
  solver-support boundaries by accident

## Authoritative Day 2 Validation Boundary

- docs-only days:
  - use targeted sanity checks, not the full code-day gate by default
- bounded `*.c` / `*.h` days:
  - run:
    - `make format`
    - `make lint`
    - `make test`
- substantial LU/CSC lifecycle or semantics-sensitive code days:
  - prefer:
    - `make quality-review-full`
  - and refresh representative proof/benchmark/example surfaces as needed

## Day 2 Exit State

Sprint 63 now has a written validation baseline that matches the live repo:

- strongest local reviewed baseline unchanged
- reviewed CMake parity anchor unchanged
- rerun set fixed from the current build tree around direct lifecycle, LU,
  CSC, and adjacent repeated-run regression surfaces
- docs-only versus code-day versus stronger-review path split fixed explicitly
- no contradiction across the main quality/truthfulness surfaces
