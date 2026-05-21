# Sprint 37 Day 3 Benchmark-Helper Consolidation Audit

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Audit the maintained benchmark tree for duplicated helper logic, separate true
shared-helper opportunities from benchmark-local behavior, and define a bounded
first-pass cleanup queue that preserves the Sprint 31 benchmark contract.

## Executive Summary

Sprint 37 does have a real benchmark-helper consolidation queue, but it is much
smaller and more structurally concentrated than the test-helper queue.

The strongest first target is clear:

- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`

Those two files are nearly mirror-image backend-comparison harnesses and repeat
the same timer, residual, result-struct, and dispatch patterns.

By contrast:

- `bench_main.c`
- `bench_reorder.c`
- `bench_eigs.c`

are larger behavior owners whose CLI/reorder/backend contracts should mostly
stay local in the first pass.

## Current Benchmark Support Floor

At Day 3 there is still no shared benchmark support layer:

- no `bench_helpers.h`
- no `bench_common.c`
- no shared timer/result/fixture helper substrate

Current structure is:

- standalone benchmark executables
- explicit executable lists in `Makefile` and `CMakeLists.txt`
- behavior contract documentation in `benchmarks/README.md`

Interpretation:

- the benchmark tree has intentionally stayed flat so far
- Sprint 37 should add shared support only where it materially reduces repeated
  maintenance burden

## Density Hotspots

Highest helper-heavy maintained benchmarks:

| File | Lines | Static functions |
|---|---:|---:|
| `benchmarks/bench_eigs.c` | `958` | `19` |
| `benchmarks/bench_chol_csc.c` | `446` | `10` |
| `benchmarks/bench_main.c` | `774` | `9` |
| `benchmarks/bench_ldlt_csc.c` | `579` | `9` |
| `benchmarks/bench_amd_qg.c` | `332` | `9` |
| `benchmarks/bench_reorder.c` | `321` | `8` |
| `benchmarks/bench_convergence.c` | `421` | `7` |

Interpretation:

- helper density is real, but only a few files dominate the queue
- structural similarity matters more than file size alone

## Real Consolidation Candidate Families

### 1. Timer helper family

Strongest breadth signal in the benchmark tree.

Observed variants:

- `wall_time`
- `now_ms`

Observed across:

- `bench_bicgstab.c`
- `bench_chol_csc.c`
- `bench_convergence.c`
- `bench_eigs.c`
- `bench_ldlt_csc.c`
- `bench_main.c`
- `bench_refactor.c`
- `bench_refactor_csc.c`
- `bench_scaling.c`
- `bench_svd.c`
- `bench_amd_qg.c`
- `bench_reorder.c`

Assessment:

- **Candidate strength:** medium to high
- **Why:** broad duplication with very low semantic complexity
- **Risk:** low, but payoff alone is modest unless bundled with stronger nearby
  harness cleanup

Recommended treatment:

- consolidate opportunistically alongside the stronger backend-comparison batch
- do not pursue timer-only abstraction as an isolated first patch

### 2. Backend-comparison harness family

Observed strongest in:

- `bench_chol_csc.c`
- `bench_ldlt_csc.c`

Repeated pieces:

- `wall_time()` helper
- residual helper with nearly identical contract
- `bench_result_t` result struct shape
- linked-list baseline path
- CSC comparison path(s)
- matrix runner / per-matrix dispatch shape

Assessment:

- **Candidate strength:** very high
- **Why:** strongest like-for-like structural duplication in the benchmark tree
- **Risk:** medium, but bounded to a closely related benchmark pair

Recommended landing shape:

- small shared benchmark helper header for the pair
- or a tightly scoped `bench_backend_compare_*` helper layer used only by this
  cluster initially

### 3. Residual helper family

Observed variants:

- `rel_residual`
- `compute_rel_residual`

Observed across:

- `bench_chol_csc.c`
- `bench_ldlt_csc.c`
- `bench_refactor_csc.c`
- `bench_bicgstab.c`
- `bench_convergence.c`

Assessment:

- **Candidate strength:** high inside the backend-comparison / iterative bench
  clusters
- **Why:** repeated numerical post-check logic with minor naming drift
- **Risk:** low to medium if signatures stay explicit

Recommended landing shape:

- cluster helper, not a repo-wide benchmark numerical utilities layer

### 4. Matrix-runner / dispatch helper family

Observed variants:

- `bench_matrix`
- `bench_matrix_impl`
- `run_one`
- `run_scaling`
- `run_comparison`

Assessment:

- **Candidate strength:** medium
- **Why:** clear repeated naming, but semantics diverge quickly by benchmark
  mode
- **Risk:** medium to high

Recommended treatment:

- keep most runner logic local
- consolidate only in the `bench_chol_csc` / `bench_ldlt_csc` pair where the
  dispatch structure is genuinely parallel

### 5. Small CLI/path/report helpers

Observed variants:

- `ends_with`
- `symbolic_nnz_L`
- `run_one_via_analyze`
- `emit_header` / `emit_row`

Assessment:

- **Candidate strength:** low to medium
- **Why:** some duplication or near-duplication exists, but much of this logic
  is bound to benchmark-specific CLI/reporting contracts
- **Risk:** medium

Recommended treatment:

- defer until after the stronger harness pair is addressed

## Strongest First Extraction Target: `bench_chol_csc` + `bench_ldlt_csc`

Why this pair stands out:

- same backend-comparison purpose
- same broad CLI shape (`--repeat`, one-matrix/default-corpus style)
- same timing-helper style
- same residual-checking pattern
- same result-struct pattern
- same per-matrix runner / dispatch pattern

Why it is safer than broader benchmark sharing:

- semantic domain is closely aligned
- file-local behavior is already parallel enough to review side by side
- it does not risk blurring Sprint 31 reorder ownership

Expected first-batch scope:

- timer helper
- residual helper
- result struct / reporting support
- matrix-dispatch helper scaffolding

## Keep Local: Not Good First Extraction Targets

### `bench_main.c`

Reasons to keep local:

- broad CLI and mode surface:
  - LU / `--cholesky`
  - SpMV
  - iterative
  - file vs directory flows
- Sprint 31 intentionally narrowed its reorder contract to
  `none|rcm|amd|nd`

Implication:

- small utility cleanup is fine
- broad helper extraction is not the right first move

### `bench_reorder.c`

Reasons to keep local:

- owns the explicit fixture table
- owns the explicit reorder table
- owns the analyze-vs-preapply comparison behavior
- Sprint 31 explicitly left it as the broad reorder comparison harness

Implication:

- preserve local ownership of reorder-surface behavior

### `bench_eigs.c`

Reasons to keep local:

- multiple backend modes
- compare/sweep/matrix CLI submodes
- specialized result-shaping/reporting logic

Implication:

- large, but not a good first shared-helper extraction target

### `bench_amd_qg.c`

Reasons to keep local:

- specialized bitset and symbolic-fill helper logic
- low overlap with the broader benchmark family beyond timing utilities

## Build-System Constraint

Current benchmark build model:

- Makefile:
  - explicit `BENCH_SRCS`
  - one binary per benchmark source
- CMake:
  - one `add_executable(bench_...)` per benchmark
  - several POSIX-only benchmarks remain gated behind `NOT WIN32`

Implication:

- safest landing shapes are:
  - small shared headers
  - narrowly scoped cluster helpers
  - local helper cleanup and sectioning
- riskiest first move is a broad benchmark support `.c` layer that touches all
  build surfaces at once

## Sprint 31 Contract Constraints

The current benchmark contract from Sprint 31 remains load-bearing:

- `bench_main` keeps the solver-harness reorder set:
  - `none`
  - `rcm`
  - `amd`
  - `nd`
- `bench_reorder` owns the broader comparison surface:
  - `none`
  - `rcm`
  - `amd`
  - `colamd`
  - `nd`
- `bench_colamd` remains the QR/COLAMD-specific comparison tool
- `bench_chol_csc` and `bench_ldlt_csc` remain backend-comparison tools, not
  reorder sweep tools

Implication:

- helper consolidation must not flatten these distinct behavior owners
- shared support is most appropriate for timing and result plumbing, not for
  CLI contract ownership

## Ranked First-Pass Queue

### Priority A

`bench_chol_csc.c` + `bench_ldlt_csc.c`

Why first:

- strongest structural duplication
- bounded risk
- good maintainability payoff

### Priority B

Residual helper cleanup across:

- `bench_chol_csc.c`
- `bench_ldlt_csc.c`
- `bench_refactor_csc.c`
- `bench_bicgstab.c`
- `bench_convergence.c`

Why second:

- repeated numerical post-check logic
- likely easy to align once the pairwise harness shape is addressed

### Priority C

Light timer-helper normalization where it falls out naturally from Priority A
or B.

### Deferred / opportunistic

- `bench_main.c` CLI/reporting structure
- `bench_reorder.c` fixture/reorder ownership
- `bench_eigs.c` backend-comparison/reporting structure
- `bench_amd_qg.c` specialized bitset helpers

## Day 3 Conclusion

Sprint 37 has a real, bounded benchmark-helper consolidation queue:

- the first meaningful win is the backend-comparison pair
  `bench_chol_csc.c` / `bench_ldlt_csc.c`
- timer and residual helpers are the main shared-support themes
- behavior-rich CLI/reorder owners should remain local in the first pass

That gives Day 6 a concrete low-risk starting point:

- consolidate the backend-comparison pair first
- use that batch to clean up timer/residual duplication
- avoid broad benchmark framework work until there is stronger evidence
