# Sprint 38 Day 4 Dead-Code Workflow Maturation Audit

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Audit the current dead-code workflow artifacts, report buckets, staged/excluded
surfaces, and residual `cppcheck` evidence so Sprint 38 can choose the next
maturity step without overstating readiness.

## Current Dead-Code Ground Truth

### Current report buckets

Live `build/deadcode/report.tsv` bucket counts:

- `coverage-gap` = `7`
- `public-surface-review` = `4`
- `secondary-candidate-signal` = `35`
- `non-deadcode-static-analysis-noise` = `6`
- `definitely-unused-internal-candidate` = `0`

Current dispositions:

- `defer-until-compile-db-expanded` = `7`
- `keep-public-api-day8-audited` = `4`
- `summarize-only-supporting-evidence` = `35`
- `appendix-only-not-cleanup-candidate` = `6`

### Current report meaning

The report still does the following correctly:

- names compile-db coverage gaps explicitly
- keeps exported public-surface findings out of the cleanup queue
- preserves `cppcheck` secondary signals as supporting evidence only
- preserves static-analysis noise as appendix-only summary data
- shows there is currently no definitely-unused internal cleanup queue

### Current enforced check boundary

`make deadcode-check` still validates:

- report generation succeeded
- every `xunused` finding was categorized
- the coverage-gap section exists

It does **not** mean:

- zero findings
- zero residual `cppcheck` noise
- full benchmark/example compile-db coverage
- concurrency-safe execution

## Comparison To The Sprint 33 Baseline

Sprint 33 close handed off:

- `coverage-gap` = `7`
- `public-surface-review` = `4`
- `secondary-candidate-signal` = `35`
- `non-deadcode-static-analysis-noise` = `6`
- `definitely-unused-internal-candidate` = `0`

Current Sprint 38 Day 4 state:

- same bucket counts
- same staged/deferred interpretation
- same shared-path serialization limitation

Interpretation:

- the workflow has stayed stable and truthful
- but it has not yet crossed into a meaningfully stronger readiness class

## Ranked Residual Dead-Code Maturity Queue

### 1. Compile-db coverage gap remains the largest structural limit

Still excluded from dead-code compile-db/reporting coverage:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

Why this matters:

- scanner silence on those paths still means nothing
- this limits how strong any future dead-code content-based gate can be

Classification:

- `staged`

Priority:

- highest

### 2. Shared-path serialization still limits stronger repeated/local CI assumptions

Current shared paths:

- `build/deadcode-cmake`
- `build/deadcode/`

Why this matters:

- concurrent `deadcode*` invocations can still race
- stronger enforcement claims remain unsafe unless the workflow stays serialized
  or the paths are isolated

Classification:

- `staged`

Priority:

- highest

### 3. `cppcheck` secondary signals remain too noisy for direct cleanup gating

Current supporting-evidence bucket:

- `35` rows

Why this matters:

- these rows still aggregate `staticFunction` / `unusedFunction` signals without
  enough review context to become removal instructions
- they are useful as prioritization hints, not as content-based pass/fail rules

Classification:

- `actionable for report refinement`

Priority:

- medium-high

### 4. Static-analysis noise summary remains useful but not gate-worthy

Current appendix-only noise bucket:

- `6` rows summarizing:
  - `constVariablePointer`
  - `normalCheckLevelMaxBranches`
  - `variableScope`
  - `constParameterPointer`
  - `constVariable`
  - `unreadVariable`

Why this matters:

- the summary is still useful to explain why raw `cppcheck` output is not being
  treated as direct dead-code evidence
- but it should remain informational, not promotable to failure logic

Classification:

- `keep summarized`

Priority:

- medium

### 5. Public-surface reviewed keeps are no longer a maturity risk

Current audited keeps:

- `givens_apply_right`
- `sparse_print_dense`
- `sparse_print_entries`
- `sparse_print_info`

Why this matters:

- the workflow no longer needs more public-API ambiguity work here
- these rows should stay visible, but they are not the current maturity blocker

Classification:

- `keep`

Priority:

- low

## Actionable vs Staged vs Defer

### Actionable in Sprint 38

- improve report wording/structure around the current bucket meanings
- tighten operator-facing explanation of what `deadcode-check` proves
- align README/report wording with the now-narrower compile-gap story from Day 3
- make the next-action guidance more explicit now that there is no current
  internal cleanup queue

### Still staged

- compile-db coverage expansion for the seven named excluded files
- shared-path isolation or other concurrency-safe topology work
- any stronger content-based failure rule over `cppcheck` secondary signals

### Defer

- treating appendix-only noise as cleanup work
- converting the dead-code report into a broad multi-platform parity gate
- using dead-code content to block on public-surface reviewed keeps

## Day 7 Design Direction

The safest next dead-code maturity step is now bounded:

1. keep the current completeness gate intact
2. improve signal quality inside the current staged model
3. make the report and docs say more directly:
   - no current cleanup-ready internal queue
   - public rows are audited keeps
   - secondary `cppcheck` rows are supporting evidence only
   - coverage-gap rows are a bounded compile-db scope limitation
4. avoid any design that implies concurrent-safe or fully covered dead-code
   enforcement before the two structural limits are closed

That is the highest-value Sprint 38 dead-code direction grounded in the current
evidence.
