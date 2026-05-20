# Sprint 36 Day 1 Cross-Platform Baseline

**Date:** 2026-05-20  
**Branch:** `sprint-36`

## Objective

Turn the Sprint 34 and Sprint 35 closeout state into a concrete Sprint 36
starting inventory by confirming the inherited reviewed-quality baseline,
auditing the current Linux/macOS/Windows quality surfaces, and naming the
first implementation targets for parity and portability work.

## Baseline Summary

Sprint 36 starts from the Sprint 34/Sprint 35 closeout exactly as intended:

- no inherited warning-debt queue in reviewed local/CMake targets
- no inherited public-doc cleanup queue
- active `ctest` registry remains `53`
- reviewed-quality wrapper contract already exists:
  - `make quality-review-compile`
  - `make quality-review`
  - `make quality-review-cmake-compile`
  - `make quality-review-cmake`
- dead-code workflow already exists and is part of the Linux reviewed CI shape:
  - `make deadcode-report`
  - `make deadcode-check`

Current branch head during the Day 1 baseline capture:

- `6ff786e`

This means Sprint 36 is not a cleanup sprint. It is a parity sprint:
cross-platform reviewed-path alignment, portability of helper scripts/targets,
and truthful reporting of what each platform actually enforces.

## Current Cross-Platform Quality Surface

### Local reviewed contract

Existing maintained quality targets:

- `format`
- `format-check`
- `lint`
- `test`
- `check`
- `tooling-build`
- `quality-review-compile`
- `quality-review`
- `quality-review-cmake-compile`
- `quality-review-cmake`
- `deadcode-compile-db`
- `deadcode`
- `deadcode-report`
- `deadcode-check`
- `wall-check`

Important current behavior:

- `quality-review-compile` is the reviewed local compile-quality path
- `quality-review` is the reviewed local end-to-end quality path
- `quality-review-cmake-compile` is the reviewed CMake parity path with test
  count parity
- `quality-review-cmake` extends that path with full `ctest`
- `.NOTPARALLEL` already protects the reviewed and dead-code wrapper targets

### CMake parity baseline

Current authoritative CMake validation view:

- `ctest -N --test-dir build/quality-review-cmake`: `53` tests

Interpretation:

- the active-suite count remains the main parity anchor
- Sprint 36 should preserve that count unless a deliberate test-scope change is
  made later

### CI surface by platform

#### Linux (`ci.yml`)

Current reviewed/path-aligned coverage:

- `make quality-review-compile`
- `make quality-review-cmake`
- `make deadcode-report`
- `make deadcode-check`
- additional Linux-only surfaces:
  - `make test`
  - `make sanitize`
  - `make asan`
  - `make bench-build`
  - `make bench-fast`
  - TSan jobs
  - coverage job

#### macOS (`macos-ci.yml`)

Current coverage:

- direct `make`
- direct `make test`
- `make wall-check`
- Apple Clang `make sanitize`
- Homebrew GCC matrix leg
- install/pkg-config validation

Interpretation:

- macOS already has substantial real coverage
- but it does not yet express the Sprint 34 reviewed wrapper contract directly

#### Windows (`windows-ci.yml`)

Current coverage:

- CMake configure
- CMake build
- CMake `ctest`

Interpretation:

- Windows already has a real CMake build/test path
- but it is the furthest from the reviewed local/Linux wrapper story

## Local Tooling Baseline

Day 1 local tool availability:

- `cppcheck`: present
- `clang-tidy`: present
- `xunused`: present
- `ctest`: present

Implication:

- Sprint 36 can start directly from parity/portability auditing
- no local prerequisite-install day is needed

## Inherited Constraints From Sprint 34 / Sprint 35

### 1. Dead-code compile-db coverage gap remains real

The current compilation-database exclusion list still contains:

- benchmark:
  - `bench_svd`
- examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Implication:

- Sprint 36 should preserve this gap truthfully in reporting and portability
  discussions
- Sprint 36 should not claim dead-code parity is “fully portable” while this
  exclusion list remains explicit

### 2. Dead-code execution model is still serial-first

Current dead-code targets still share:

- `build/deadcode-cmake`
- `build/deadcode/`

Implication:

- Sprint 36 portability work should preserve serialized execution semantics
  unless it intentionally isolates those paths

### 3. Public-doc contract from Sprint 35 is now part of the baseline

Sprint 35 closed the public-surface consistency queue and established:

- headers as the authoritative API contract
- `README.md` as the concise entrypoint
- `docs/tutorial.md` as the fuller teaching surface

Implication:

- Sprint 36 should not casually regress command names or platform caveats in
  support docs when aligning cross-platform quality expectations

## First Implementation Surfaces

### 1. CI parity and workflow naming

Primary files:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### 2. Reviewed target portability

Primary files:

- `Makefile`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`

### 3. Platform-parity reporting

Primary surfaces:

- Sprint 36 artifact/report docs
- workflow step names and expectation wording

## Day 1 Conclusion

Sprint 36 starts from a strong validated Linux/local reviewed baseline but only
a partially aligned cross-platform expression of that baseline:

- Linux already speaks in reviewed wrapper terms
- macOS and Windows still exercise real build/test surfaces
- but they do not yet map cleanly to the same reviewed-path contract

That makes the next step clear: Day 2 and Day 3 should turn this into explicit
macOS and Windows audit queues before Sprint 36 changes CI or helper logic.
