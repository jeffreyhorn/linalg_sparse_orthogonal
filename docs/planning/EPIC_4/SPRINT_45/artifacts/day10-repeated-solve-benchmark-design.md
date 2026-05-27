# Sprint 45 Day 10 Artifact: Repeated-Solve Benchmark Design

## Purpose

Define the smallest useful benchmark slice that can show Sprint 45's iterative
workspace reuse in a direct repeated-solve setting, without turning the sprint
into a broad benchmark framework or CLI redesign.

## Main Day 10 Conclusion

The right Sprint 45 benchmark evidence is a narrow A/B repeated-solve
comparison modeled after `bench_refactor.c`, centered on the migrated scalar
iterative paths:

- scalar CG
- scalar GMRES
- optional block CG only if the landing stays obviously small

That is the cleanest claim-safe Day 11 shape.

## Current Benchmark Surface

### 1. Existing iterative benchmarks are still one-shot oriented

The live benchmark surfaces currently do:

- `bench_convergence.c`
  - convergence tables
  - residual-vs-iteration histories
  - one-shot calls through public iterative APIs
- `bench_main --iterative`
  - one-shot timing summaries through public iterative APIs
- `bench_bicgstab.c`
  - solver-family comparison, not reusable-workspace evidence

Interpretation:

- these are valid baselines and supporting context
- they are not yet direct measurements of Sprint 45's reusable-workspace seam

### 2. `bench_refactor.c` is the best structural precedent

`bench_refactor.c` already shows the right comparison model:

- Approach A:
  - repeated one-shot calls
- Approach B:
  - repeated use of reusable internal state on a stable problem shape
- bounded wall-clock reporting
- no broad new harness abstraction

Interpretation:

- Day 11 should reuse this structure
- it should avoid broader benchmark/CLI churn

## Recommended Day 11 Target Set

### Primary targets

- scalar CG repeated-solve case
- scalar GMRES repeated-solve case

Reasons:

- both are already migrated onto the shared iterative workspace seam
- both already have visible one-shot benchmark precedent in the repo
- both are easier to interpret than wrapper-only or later-specialized paths

### Optional add-on only if it stays small

- block CG repeated-solve case

Reason:

- block CG is the only block path that represents a real direct workspace
  migration in Sprint 45

### Explicit non-targets for Day 11

- block GMRES
- block MINRES
- block BiCGSTAB
- MINRES scalar workspace migration
- benchmark CLI redesign
- broad `bench_main` mode expansion

Interpretation:

- the benchmark batch should stay aligned with the actual Sprint 45 landed
  workspace seams

## Comparison Model

The benchmark comparison should use:

- stable matrix/problem dimensions
- stable solver options/tolerances
- stable preconditioner/operator context when applicable
- repeated loop of:
  - one-shot public solve path
  - reusable-workspace-backed internal path

It should report:

- wall-clock time
- iterations / convergence summary
- concise interpretation of the repeated-solve comparison

It should avoid:

- universal speedup claims
- machine-independent performance claims
- broad allocator statistics unless added in a bounded trustworthy way

## Likely Implementation Shape

The cleanest Day 11 landing is:

- one small dedicated repeated-solve benchmark source

Acceptable fallback:

- a very small dedicated repeated-solve mode inside an existing iterative
  benchmark file

Least desirable Day 11 move:

- broadening `bench_main` CLI parsing and usage text during Sprint 45

Interpretation:

- Sprint 47 is the better home for bigger benchmark CLI modernization
- Sprint 45 should optimize for a narrow, honest measurement slice

## Recommended Day 11 Sequence

1. choose one SPD matrix/repeated-CG case
2. choose one nonsymmetric or general repeated-GMRES case
3. optionally add one block-CG case only if the batch remains very small
4. implement the A/B repeated-call comparison using the `bench_refactor`
   structure
5. record measured outputs in the Day 11 artifact

## Bottom Line

Day 10 fixes the Day 11 shape clearly:

- compare repeated one-shot solves vs repeated reusable-workspace-backed solves
- center the batch on scalar CG and scalar GMRES
- treat block CG as optional only if it stays obviously small
- keep the implementation small, dedicated, and claim-safe

That is the right benchmark-design handoff for Sprint 45.
