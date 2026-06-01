# Sprint 51 Day 9: Focused Regression Expansion Design & Inventory

## Objective

Audit the live direct-solver lifecycle regression surface after the Day 4-8
implementation work so Day 10 only lands the smallest remaining high-signal
coverage additions.

## Surfaces Reviewed

Primary regression surfaces:

- `tests/test_integration.c`
- `tests/test_etree.c`
- `tests/test_cholesky.c`
- `tests/test_ldlt.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt_csc.c`

Adoption / caller-story surfaces:

- `examples/example_analysis.c`
- `examples/README.md`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/README.md`

## Findings

### 1. The lifecycle core is already better-covered than the original Day 9 placeholder implied

The live tree already has direct lifecycle coverage for:

- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`
- one-shot vs explicit-analysis parity for LU / Cholesky / LDL^T
- wrapper vs default-options parity for LU / Cholesky / LDL^T where the
  wrapper contract is phase-1-relevant

That coverage is distributed mainly across:

- `tests/test_etree.c`
- `tests/test_integration.c`

So Day 10 should not behave as if Sprint 51 still lacks a basic direct
lifecycle regression story.

### 2. The strongest remaining test gap is small public-surface sequencing/ownership clarity

What is still worth adding is narrower:

- small focused checks that make the public sequence more obvious:
  - zero-init `sparse_analysis_t`
  - zero-init `sparse_factors_t`
  - analyze → factor → solve → refactor → solve
- direct invalid-sequence rejection where the public contract already defines
  it

This is different from broad parity expansion:

- large family-by-family equivalence work is already present
- the remaining value is in making the public lifecycle contract easier to
  see and harder to drift

### 3. LU is not a Day 10 wrapper-routing target

Day 8 confirmed that pushing `sparse_lu_factor(...)` through the current
options/lifecycle route creates a real recursion seam through
`sparse_factor_numeric(..., SPARSE_FACTOR_LU)`.

That means:

- LU wrapper routing is a later routing-refactor problem
- Day 10 should not try to “complete” LU wrapper routing under a regression
  label

### 4. The strongest later adoption targets remain unchanged

The best Day 11 adoption surfaces are still:

- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`

Those files already own the strongest shipped repeated-run direct story:

- analyze once
- factor / solve
- refactor many
- timing and residual reporting

### 5. The two concrete carried-forward docs drifts still remain

The adoption-side drifts identified in Sprint 50 are still live:

- `benchmarks/README.md`
  - still describes `bench_refactor` as LDL^T re-factor with cached symbolic
  - the live driver is the Cholesky analyze-once / factor-many benchmark
- `examples/README.md`
  - still omits `example_analysis`

These are natural Day 11 follow-ons if the surrounding files are touched, not
urgent Day 10 blockers.

## Day 10 Landing Map

### Mandatory targets

- small focused lifecycle sequencing/ownership regressions

### Best landing surface

- `tests/test_integration.c`

Why:

- it already carries the small public-surface parity checks added in Sprint 51
- it is a better fit for bounded lifecycle-sequencing tests than expanding the
  already-large `tests/test_etree.c` sweep further without a specific gap

### Explicit non-goals

- no new LU wrapper-routing attempt
- no broad `tests/test_etree.c` expansion
- no benchmark/example adoption yet
- no tutorial churn

## Bottom Line

Sprint 51’s remaining queue is now smaller than the original plan placeholder
suggested:

- the lifecycle core is already well-covered
- the strongest remaining Day 10 work is small public-surface
  sequencing/ownership coverage
- the strongest later Day 11 work remains `example_analysis`,
  `bench_refactor*`, and the two carried-forward README drifts
