# Sprint 60 Day 2: Validation Baseline and Touched-Surface Recheck

Date: 2026-06-08
Branch: `sprint-60`


## Purpose

Freeze the validation and truthfulness baseline that later Epic 6
implementation work must preserve before the sprint moves into the deeper
productization and architecture audits.

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
- targeted Sprint 60 rerun-set presence:
  - `build/`

## Day 2 Baseline Conclusions

### 1. The strongest local reviewed baseline is still `make quality-review-full`

Sprint 60 inherits the same authoritative local validation command as the Epic
5 close state:

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

- the reviewed CMake path still sees the maintained full test surface
- Makefile/CMake parity has not drifted silently

### 3. The current code-day gate versus stronger reviewed baseline split is stable

The maintained split is:

- bounded code days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial implementation work:
  - `make quality-review-full`
- docs-only days:
  - no automatic code-quality gate required
  - use targeted sanity checks instead

This is consistent with the repo’s actual Sprint 50-59 operating discipline and
does not need reinterpretation on Sprint 60 Day 2.

### 4. The current quality/platform story is coherent across README, maintainer guide, and Makefile

The main maintained surfaces still agree on the current contract:

- dead-code remains separate from `lint` and `test`
- dead-code remains operationally serialized
- Linux remains the enforced reviewed source-of-truth path
- macOS dead-code remains staged
- Windows keeps the reviewed CMake subset enforced while broader reviewed
  Makefile wrappers and dead-code stay staged

This means Sprint 60 can proceed from a stable truthfulness contract rather
than needing an immediate cleanup batch just to align baseline wording.

### 5. The targeted Sprint 60 rerun set is present and already matches the likely Epic 6 work bands

The confirmed rerun set is:

- tests:
  - `./build/test_integration`
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- examples:
  - `./build/example_analysis`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
- benchmarks:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

That is already a strong enough rerun set to support:

- direct-solver usability work
- CSC/direct repeated-run work
- iterative/eigensolver workflow work
- performance-governance and benchmark-story work

## Authoritative Day 2 Validation Boundary

- docs-only days:
  - use targeted sanity checks, not the full code-day gate by default
- bounded `*.c` / `*.h` days:
  - run:
    - `make format`
    - `make lint`
    - `make test`
- substantial architecture/performance/cross-surface code days:
  - prefer:
    - `make quality-review-full`
  - and refresh representative proof/benchmark/example surfaces as needed

## Day 2 Exit State

Sprint 60 now has a written validation baseline that matches the live repo:

- strongest local reviewed baseline unchanged
- reviewed CMake parity anchor unchanged
- rerun set fixed from the current build tree
- docs-only versus code-day versus stronger-review path split fixed explicitly
- no contradiction across the main quality/truthfulness surfaces
