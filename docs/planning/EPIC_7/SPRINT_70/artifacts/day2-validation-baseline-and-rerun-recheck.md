# Sprint 70 Day 2: Validation Baseline and Rerun Recheck

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Reconfirm the reviewed baseline and the targeted rerun set that Sprint 70
planning and later Epic 7 implementation sprints must preserve before deeper
product-model, capability, and contract audits continue.

## Authoritative Rechecks

- `ctest -N --test-dir build/quality-review-cmake`
- `make -n quality-review-full`
- direct existence recheck of the targeted Sprint 70 rerun set across:
  - `build/quality-review-cmake/`
  - `build/`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

## Day 2 Validation Conclusions

### 1. The strongest local reviewed baseline is unchanged

Sprint 70 still starts from:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

That remains the authoritative Day 2 truth surface for later implementation
and architecture days.

### 2. The validation split is now explicit before any code or architecture movement

The validation contract for Sprint 70 is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial architecture, capability,
  benchmark-governance, or platform work:
  - `make quality-review-full`
- docs-only days:
  - targeted sanity checks only

This matches the maintained repo contract instead of inventing a lighter
Sprint-70-specific rule set.

### 3. The high-signal Sprint 70 rerun set is now fixed around the actual Epic 7 risk surface

The targeted rerun set present in the reviewed CMake tree is:

- cross-family/orchestration and direct-family proof owners:
  - `build/quality-review-cmake/test_integration`
  - `build/quality-review-cmake/test_chol_csc`
  - `build/quality-review-cmake/test_ldlt_csc`
  - `build/quality-review-cmake/test_reorder_nd`
- assurance and broader numerical proof support:
  - `build/quality-review-cmake/test_fuzz`
  - `build/quality-review-cmake/test_framework_optin`
  - `build/quality-review-cmake/test_iterative`
  - `build/quality-review-cmake/test_eigs`
  - `build/quality-review-cmake/test_graph`
  - `build/quality-review-cmake/test_qr`
  - `build/quality-review-cmake/test_svd`
- representative examples:
  - `build/quality-review-cmake/example_analysis`
  - `build/quality-review-cmake/example_basic_solve`

The maintained benchmark/reporting surfaces currently present in the root
`build/` tree are:

- `build/bench_refactor_csc`
- `build/bench_chol_csc`
- `build/bench_iterative_reuse`
- `build/bench_eigs_reuse`

The maintained install/package proof surfaces remain script-owned:

- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

### 4. The local branch currently has a split binary surface, and that split is still truthful

Day 2 recheck showed:

- the reviewed CMake tree currently holds the authoritative local test and
  example binaries
- the root `build/` tree currently holds the maintained benchmark binaries
- the install/package regressions remain shell-script proof surfaces rather
  than prebuilt executables

That split is acceptable for Sprint 70 as long as later notes and artifacts do
not confuse:

- reviewed proof-owner test binaries
- maintained benchmark/reporting binaries
- install/package script-owned proof

### 5. Sprint 70’s likely touched-surface class remains narrower than the full reviewed suite

Day 2 confirms the most likely Sprint 70 touched lane is concentrated in:

- maintained public product and policy surfaces
- product-model, capability, and configuration audit seams
- proof/adoption/reporting surfaces only where architecture or contradiction
  analysis genuinely points to them
- project-level Epic 7 planning and review surfaces

So the sprint should stay bounded to baseline, audit, and contract work rather
than widening into generic repo churn.

## Day 2 Exit State

Sprint 70 now has one explicit validation contract before deeper audit and
architecture work:

- strongest local reviewed baseline is still `make quality-review-full`
- reviewed CMake parity remains explicit at `53`
- bounded code-touching days must run `make format`, `make lint`, and
  `make test`
- substantial architecture, capability, benchmark, or platform work should
  default to `make quality-review-full`
- the high-signal Sprint 70 rerun set is fixed around the reviewed CMake proof
  tree, maintained benchmark binaries, and install/package regression scripts
