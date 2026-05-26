# Sprint 41 Day 10 Artifact: Auxiliary-Surface Alignment Audit

## Purpose

Turn the post-Day-9 residual auxiliary queue into a concrete Day 11
implementation map by:

- separating easy helper-alignment wins from surfaces that should wait for
  later public-facing work
- keeping Sprint 40's public-migration-risk boundary explicit
- avoiding accidental drift into benchmark-harness redesign or script-side
  maintainability work that does not match Sprint 41's allocation-helper goal

## Starting Point After Day 9

Day 9 completed the main broader `src/` migration pass for Sprint 41:

- first-wave hotspot set complete
- broader easy `src/` pair complete
- `src/sparse_iterative.c` migrated

That means Day 10 is no longer about the core library tree. It is about the
residual auxiliary surfaces that still contain raw allocation/count logic:

- examples
- benchmarks
- scripts

The question is not "where does allocation code still exist?" The question is
"which auxiliary surfaces are honest Sprint 41 helper-alignment targets, and
which ones should remain deferred because they are public-teaching or
specialized harness surfaces?"

## Auxiliary Queue Classification

### 1. Easy alignment now

These are the best Day 11 candidates because they are small, still use simple
`n`-based allocations, and do not carry large behavior-owner or public-API
teaching risk.

#### `examples/example_iterative.c`

Current signs:

- simple `calloc((size_t)n, sizeof(double))` arrays:
  - `b`
  - `x`
  - `ones`
- no large custom workspace packing
- no broad CLI or harness logic

Why it is a good next migration:

- the safety seam is narrow and obvious
- the example remains readable after helper adoption
- it aligns directly with the Day 4 helper layer without changing the example's
  teaching purpose

Expected Day 11 shape:

- adopt `sparse_calloc_array(...)`
- keep example flow and teaching comments unchanged

#### `examples/example_matrix_free.c`

Current signs:

- simple `calloc((size_t)n, sizeof(double))` arrays:
  - `b`
  - `x`
  - `x_exact`
- narrow matrix-free teaching surface
- no specialized packed workspace logic

Why it is a good next migration:

- helper substitution is straightforward
- the public example remains easy to read
- it exercises the helper layer on a second example family without widening the
  batch

Expected Day 11 shape:

- adopt `sparse_calloc_array(...)`
- preserve matrix-free/operator-teaching semantics exactly

#### `examples/example_colamd.c`

Current signs:

- direct `malloc((size_t)n * sizeof(idx_t))` for:
  - `perm`
  - `id_perm`
- compact one-file demo with narrow reorder/fill comparison scope

Why it is still a plausible Day 11 candidate:

- the allocation pattern is simple and local
- the example's public-teaching message does not depend on manual byte math

Why it is lower priority than the two iterative examples:

- the current Sprint 41 seam is already proven well by the simpler `double`
  array examples
- Day 11 does not need to widen unless the batch remains obviously bounded

Recommended handling:

- include only if the Day 11 example batch still stays small and mechanical

### 2. Defer until later public-facing work

These files do contain raw allocation/count logic, but they are not the right
Sprint 41 target because changing them is more likely to overlap with future
public-surface, lifecycle, or usability work than with the narrow shared-helper
objective.

#### `examples/example_eigs.c`

Why defer:

- large behavior-owner example
- multiple demos/backends in one file
- strong public-teaching role for eigensolver composition and preconditioning

Interpretation:

- allocation alignment is possible later
- Sprint 41 should not turn this into a mixed helper + public-example rewrite

#### `examples/example_ic_minres.c`

Why defer:

- larger public-teaching example
- lifecycle/preconditioner composition is part of the example's value
- more likely to intersect later public migration/doc reconciliation work

Interpretation:

- not a good "quick alignment" candidate even though helper duplication exists

#### `examples/example_analysis.c`

Why defer for now:

- allocation sites are still straightforward, but the example is a direct
  public-teaching surface for analyze-once/factor-many workflows
- Sprint 40 already identified analyze/factor bridge semantics as a migration
  risk area

Interpretation:

- this is better treated with the later bridge/lifecycle work than as a routine
  Sprint 41 helper-consolidation edit

### 3. Keep local / specialized / later bounded work

These surfaces still contain raw allocation/count logic, but Sprint 41 should
not pull them into the routine auxiliary-alignment path because the real
maintainability issue is broader than helper substitution.

#### `benchmarks/bench_main.c`

Why keep local for now:

- large mixed-concern benchmark harness
- several benchmark modes and orchestration paths
- many allocations tied to benchmark semantics rather than a single narrow seam

Interpretation:

- a helper-alignment edit here would blur quickly into benchmark-harness
  cleanup, CLI normalization, or broader maintainability refactoring

#### `benchmarks/bench_eigs.c`

Why keep local for now:

- large benchmark driver
- specialized eigensolver benchmark semantics
- more custom byte-count/workspace logic than the simple example surfaces

Interpretation:

- better handled in a later benchmark-focused or maintainability-focused pass

#### Broader benchmark cluster

Other benchmark files still carrying raw allocation/count logic include:

- `benchmarks/bench_convergence.c`
- `benchmarks/bench_bicgstab.c`
- `benchmarks/bench_scaling.c`
- `benchmarks/bench_reorder.c`
- `benchmarks/bench_amd_qg.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_ldlt_csc.c`

Interpretation:

- these are real later candidates
- Sprint 41 does not need them to prove the shared helper layer on auxiliary
  surfaces

#### Scripts

Representative script hotspots by size remain:

- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

Why they are not the Day 10 target:

- they are not C allocation-helper duplication problems
- their maintainability issues are already known and belong to different later
  workstreams

Interpretation:

- Day 10 should not widen Sprint 41 into script refactoring

## Recommended Day 11 Batch

### Primary targets

- `examples/example_iterative.c`
- `examples/example_matrix_free.c`

Reason:

- both are small
- both are mechanically aligned with the Day 4 helper layer
- both preserve Sprint 40's public-risk boundary cleanly

### Optional bounded add-on

- `examples/example_colamd.c`

Reason:

- only if the live batch remains obviously small and mechanical

### Explicitly deferred from the routine Sprint 41 auxiliary path

- `examples/example_eigs.c`
- `examples/example_ic_minres.c`
- `examples/example_analysis.c`
- `benchmarks/bench_main.c`
- `benchmarks/bench_eigs.c`
- broader benchmark/support-script refactors

## Day 10 Conclusion

Sprint 41's auxiliary alignment queue is real, but it is narrower than a
generic examples/benchmarks/scripts cleanup pass.

The honest next move is:

- small example-focused helper alignment
- preserve public-teaching and lifecycle-risk surfaces for later work
- keep benchmark and script maintainability concerns out of the routine Sprint
  41 migration path

That gives Day 11 a bounded, low-risk auxiliary implementation batch rather
than a mixed helper/benchmark/doc cleanup pass.
