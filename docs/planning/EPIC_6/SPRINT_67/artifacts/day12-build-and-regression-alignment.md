# Sprint 67 Day 12 Artifact: Build and regression alignment

Date: 2026-06-13
Branch: `sprint-67`

## Scope

Docs-only Sprint 67 Day 12 alignment batch across:

- `README.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`

Non-goals:

- no source-list or target-list rewiring
- no implementation changes
- no test-code churn
- no reopened packaging/platform/build-system work

## Problem

After the Day 6-11 maintainability landings, the build graph itself was still
truthful:

- `Makefile` library and test source lists already matched the live branch
- `CMakeLists.txt` source topology already matched the landed decomposition

The residual contradiction sat one layer higher:

- maintained docs still did not say clearly which regression surfaces now own
  the Sprint 67 graph/reorder and large-`n` Cholesky maintainability seams
- `README.md` still understated the live CSC Cholesky / CSC LDL^T suite sizes

That meant the repo had the right tests, but the maintained explanation of test
ownership lagged behind the landed boundaries.

## Landing

### 1. `README.md`

Updated the Cholesky CSC repeated-run section so it now distinguishes:

- benchmark-side repeated-run workflow proof:
  - `bench_refactor`
  - default SPD mode in `bench_refactor_csc`
- family-local CSC helper proof:
  - `tests/test_chol_csc.c`
- public one-shot vs explicit repeated-run parity/error-path proof:
  - `tests/test_integration.c`

Updated the test inventory so the CSC direct-family suite counts reflect the
live branch instead of older Sprint 17-era values:

- CSC Cholesky: `145`
- CSC LDL^T: `96`

### 2. `docs/maintainer_guide.md`

Added the maintained proof-ownership split that now matters after Sprint 67:

- `tests/test_reorder_nd.c` owns the shared ND compatibility/default-policy
  convergence proof lane
- `tests/test_chol_csc.c` owns the family-local large-`n` analysis-backed
  Cholesky CSC handoff proof lane
- `tests/test_integration.c` owns the public one-shot vs explicit repeated-run
  Cholesky parity and failure-preservation lane
- benchmark surfaces remain benchmark-side workflow/performance proof, not
  replacements for those regression owners

### 3. `benchmarks/README.md`

Tightened the `bench_refactor_csc` contract so it now states plainly:

- failed refactor preservation stays owned by `tests/test_integration.c`
- family-local large-`n` analysis-backed CSC helper parity stays owned by
  `tests/test_chol_csc.c`

## Validation

This was a docs-only alignment batch, so no compile/test rerun was required.

Targeted sanity set used instead:

- `git diff -- README.md docs/maintainer_guide.md benchmarks/README.md`
- terminology/alignment `rg`
- touched-surface `wc -l`
- branch status recheck

## Result

Sprint 67 Day 12 closes the real remaining alignment gap from the Day 6-11
maintainability work:

- build and source lists were already correct
- maintained regression-surface ownership is now correct too
- the repo now says clearly which proof lanes own:
  - shared ND policy convergence
  - family-local large-`n` Cholesky CSC handoff parity
  - public repeated-run Cholesky parity and failure-preservation
