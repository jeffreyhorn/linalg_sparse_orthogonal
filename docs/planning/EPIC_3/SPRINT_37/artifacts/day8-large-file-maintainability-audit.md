# Sprint 37 Day 8 Large-File Maintainability Audit

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Identify the large auxiliary files whose size and structure now hurt
maintainability most, separate them from files that are merely large but still
cohesive, and define the highest-value one-or-two-file cleanup batch for Day 9.

## Executive Summary

The largest files in the repo are mostly tests, but the best refactor targets
are **not** the longest tests.

The strongest first large-file maintainability batch is:

1. `Makefile`
2. `scripts/deadcode_report.py`

Reason:

- both are mixed-concern auxiliary surfaces
- both concentrate multiple responsibilities in one file
- both impose large reread/review costs on routine maintenance
- both can be structurally improved without reopening feature semantics

By contrast, the giant test files are large but heavily sectioned feature-owner
files under a one-binary-per-source test build model. They remain important
residual debt, but they are not the best Day 9 target.

## Raw Size Leaders

### Tests

- `tests/test_chol_csc.c` = `4,643`
- `tests/test_svd.c` = `3,712`
- `tests/test_ldlt_csc.c` = `3,637`
- `tests/test_qr.c` = `3,259`
- `tests/test_etree.c` = `2,890`
- `tests/test_iterative.c` = `2,796`

### Benchmarks

- `benchmarks/bench_eigs.c` = `958`
- `benchmarks/bench_main.c` = `774`
- `benchmarks/bench_ldlt_csc.c` = `516`

### Auxiliary / tooling

- `README.md` = `852`
- `Makefile` = `845`
- `scripts/deadcode_report.py` = `472`
- `scripts/deadcode_workflow.sh` = `189`

## Cohesive Large Files Vs Mixed-Concern Large Files

### Large but still cohesive: giant tests

Representative structure signals:

- `tests/test_chol_csc.c`
  - `155` static functions
  - `138` section/marker lines
- `tests/test_svd.c`
  - `99` static functions
  - `115` section/marker lines
- `tests/test_ldlt_csc.c`
  - `123` static functions
  - `142` section/marker lines

Interpretation:

- these files are long, but they are strongly sectioned
- they map closely to broad feature-owner test coverage
- under the repo’s one-binary-per-test-source model, splitting them
  mechanically is not obviously a maintainability win

Conclusion:

- keep them in the residual queue
- do not use line count alone to drive the first large-file refactor

### Large and mixed-concern: `Makefile`

Key shape:

- `845` lines
- `39` `.PHONY` declarations
- roughly `101` target-like rule lines

Why it is a strong refactor target:

- it combines:
  - build rules
  - direct gates
  - reviewed wrappers
  - helper plumbing
  - dead-code flow
  - sanitizer / OMP / coverage modes
  - install / pkg-config logic
- even after Sprint 37 Day 7’s category signaling pass, it remains a large
  operational command map plus implementation file
- small target edits still require large-context rereads

Conclusion:

- highest-value Day 9 target

### Large and mixed-concern: `scripts/deadcode_report.py`

Key shape:

- `472` lines
- `15` top-level functions

Current responsibilities in one file:

- parse coverage notes
- parse `xunused`
- parse `cppcheck`
- classify xunused/public/internal buckets
- build TSV rows
- group/summarize findings
- render markdown
- validate the generated report

Why it is a strong refactor target:

- several phases of one workflow are layered together
- the file has weak visual sectioning compared with the larger tests
- capability accreted across Sprint 33 / 34 / 36, so rereads now cross parsing,
  policy, rendering, and validation concerns together

Conclusion:

- second-highest-value Day 9 target

### Not a first-pass target: `scripts/deadcode_workflow.sh`

Key shape:

- `189` lines

Why it is lower priority:

- still relatively cohesive as a workflow runner
- responsibility boundaries are simpler:
  - prerequisites
  - compile-db validation
  - coverage-note generation
  - `cppcheck`
  - `xunused`

Conclusion:

- keep in residual queue
- not the best first large-file refactor target

### Large benchmark owners: defer

Representative benchmark files:

- `benchmarks/bench_eigs.c` = `958`
- `benchmarks/bench_main.c` = `774`

Why they are not first:

- they are behavior-owner CLIs
- their size reflects real sweep/report/mode breadth
- refactoring them safely is more semantic than structural

Conclusion:

- keep in residual queue
- do not displace `Makefile` / `deadcode_report.py` as the first structural batch

## Ranked Day 9 Candidates

### Tier 1: Chosen batch

1. `Makefile`
2. `scripts/deadcode_report.py`

### Tier 2: Residual queue

- `scripts/deadcode_workflow.sh`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_main.c`
- `tests/test_chol_csc.c`
- `tests/test_svd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_qr.c`
- `tests/test_etree.c`
- `tests/test_iterative.c`

## Recommended Day 9 Cleanup Shape

For `Makefile`:

- improve structural grouping further
- tighten local helper/layout boundaries
- reduce reread cost around major operational surfaces
- preserve behavior and target names

For `scripts/deadcode_report.py`:

- separate parsing/classification/render/validation structure more clearly
- add stronger internal sectioning or helper extraction
- preserve CLI and generated-output semantics

Avoid:

- mechanical test splitting
- benchmark CLI behavior churn
- docs-only line-count cleanup

## Day 8 Conclusion

The best first large-file maintainability targets are not the repo’s largest
tests. They are the mixed-concern auxiliary surfaces that impose the highest
routine reread and review cost:

- `Makefile`
- `scripts/deadcode_report.py`

That gives Day 9 a concrete, bounded, behavior-preserving cleanup batch.
