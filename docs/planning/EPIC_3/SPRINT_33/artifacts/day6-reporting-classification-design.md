# Sprint 33 Day 6 Reporting & Classification Design

**Date:** 2026-05-18  
**Branch:** `sprint-33`

## Objective

Define how Sprint 33 should convert the Day 5 raw dead-code artifacts into an auditable report without overstating what the tools prove. The main design problem is not how to print the findings; it is how to separate true cleanup candidates from public-surface review items, compile-db blind spots, and broad static-analysis noise.

## Day 6 Input Reality

The Day 5 workflow produces three distinct signals:

1. compile-db coverage gaps
2. narrow `xunused` reachability candidates
3. broad `cppcheck` static-analysis output

These signals do not deserve equal trust.

### Coverage gaps

Current known compile-db omissions:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

Interpretation:

- silence from `xunused` on those surfaces means very little
- the report must preserve this caveat prominently

### `xunused`

Current findings:

- internal/private candidate:
  - `chol_csc_dump_supernodes`
- exported/public review items:
  - `givens_apply_right`
  - `sparse_print_dense`
  - `sparse_print_entries`
  - `sparse_print_info`

Interpretation:

- `xunused` is the highest-signal dead-code input currently available
- it still cannot skip the public-surface review step

### `cppcheck`

Top recurring ids:

- `constVariablePointer`: `106`
- `staticFunction`: `90`
- `unusedFunction`: `80`
- `normalCheckLevelMaxBranches`: `23`

Interpretation:

- only part of this output overlaps with dead-code concerns
- the report must not present the entire raw `cppcheck` log as a deletion queue

## Chosen Category Model

Day 6 classification buckets:

### 1. `coverage-gap`

Definition:

- source is absent from the current `compile_commands.json`
- tool silence cannot be interpreted as absence of candidates

Examples:

- `bench_svd`
- the six example programs missing from the Day 5 compile-db

Handling:

- always shown near the top of the report
- never enters the cleanup queue

### 2. `definitely-unused-internal-candidate`

Definition:

- internal-only implementation/declaration surface
- no installed-header or documented outward surface involved
- signal is strong enough to consider Day 9 cleanup batching

Current example:

- `chol_csc_dump_supernodes`

Handling:

- primary cleanup queue for later Sprint 33 deletion work

### 3. `public-surface-review`

Definition:

- touches `include/`, documented examples, documented benchmark binaries, or otherwise outward-facing contract surface

Current examples:

- `givens_apply_right`
- `sparse_print_dense`
- `sparse_print_entries`
- `sparse_print_info`

Handling:

- explicitly deferred to Day 8
- not eligible for automatic cleanup batching

### 4. `secondary-candidate-signal`

Definition:

- evidence that may help prioritize inspection but is not cleanup-ready by itself

Current source:

- `cppcheck` `unusedFunction`
- `cppcheck` `staticFunction`

Handling:

- summarized by file and checker id
- does not become a line-by-line primary finding list

### 5. `non-deadcode-static-analysis-noise`

Definition:

- findings that are valid static-analysis observations but not dead-code evidence

Current source:

- `constVariablePointer`
- `variableScope`
- `normalCheckLevelMaxBranches`
- other style-only ids

Handling:

- count only
- appendix/summary only
- excluded from cleanup queues

## Chosen `deadcode-report` Output Shape

Day 7 should generate two report artifacts in `build/deadcode/`.

### Human summary

Proposed path:

- `build/deadcode/report.md`

Purpose:

- maintainer-readable summary
- conservative explanation of what is actionable now
- explicit preservation of coverage and public-surface caveats

Proposed section order:

1. run metadata
2. coverage-gap section
3. definitely-unused internal candidates
4. public-surface review items
5. aggregated secondary `cppcheck` signals
6. deferred noise summary
7. current sprint next-action queue

### Stable findings table

Proposed path:

- `build/deadcode/report.tsv`

Purpose:

- reproducible per-finding normalization
- easier future comparison or CI parsing

Proposed columns:

- `bucket`
- `tool`
- `symbol`
- `path`
- `line`
- `detail`
- `disposition`

Why both formats:

- Markdown is the right primary surface for human review
- TSV is the right stable surface for later diffability

## Chosen `deadcode-check` Contract

Day 6 decision:

- Sprint 33 `deadcode-check` should validate report completeness, not repository cleanliness

Proposed semantics:

1. invoke `deadcode-report`
2. fail if the report build fails
3. fail if any `xunused` finding is uncategorized
4. fail if the coverage-gap section is missing
5. pass when categorized candidates still remain

Why this is the right Sprint 33 scope:

- Day 8 still needs to audit public-surface items
- Day 9 through Day 11 still need to batch and remove internal candidates
- a fail-on-any-finding gate would fight the planned sprint sequence

This still creates a useful invariant:

- every high-signal finding must be accounted for
- the report cannot silently drop blind spots or ambiguous items

## Day 6 Conclusion

The report layer should be conservative and explicit:

- `xunused` drives the actionable queue
- `cppcheck` remains supporting evidence, not oracle
- compile-db blind spots stay visible
- public-surface findings stay out of the first cleanup batch until audited

That design keeps Sprint 33 aligned with the Sprint 32 truthfulness model while giving Day 7 a concrete implementation target.
