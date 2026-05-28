# Sprint 46 Day 10 Artifact: Repeated-Run Benchmark Design

## Purpose

Define the smallest honest repeated-run benchmark slice for the migrated
eigensolver workspace/state model so Sprint 46 can add measured reuse evidence
without broadening into benchmark-framework churn or over-claiming performance
results.

## Main Day 10 Conclusion

The right Day 11 benchmark target is **not** a broad rewrite of
`benchmarks/bench_eigs.c`.

It is a narrow Sprint 45-style A/B repeated-run comparison driver that measures:

- one-shot path
- reusable internal path
- repeated stable-dimension calls
- wall-time median comparison
- iteration/convergence/residual parity

That is the important Day 10 narrowing.

## Current Benchmark Surface Assessment

### 1. `bench_eigs.c` is already a broad backend/corpus sweep driver

The live `benchmarks/bench_eigs.c` already owns:

- multi-backend sweeps
- multiple `which` modes
- SuiteSparse/KKT corpus selection
- preconditioner sweeps
- human-readable and CSV modes
- backend comparison output

Interpretation:

- it is a valuable permanent benchmark surface
- it is too broad to serve as Sprint 46’s first repeated-run reuse proof

### 2. Sprint 45’s repeated-run benchmark is the right design model

`benchmarks/bench_iterative_reuse.c` provides the cleaner target shape:

- explicit one-shot vs reusable comparison
- repeated stable-dimension calls
- narrow selected cases
- modest claim scope
- behavior-level parity in addition to timing

Interpretation:

- Sprint 46 should mirror this pattern for eigensolvers
- repeated-run evidence should remain a separate narrow driver, not a broad
  benchmark-framework rewrite

## Selected Day 11 Benchmark Shape

### Required cases

Day 11 should include:

- grow-m Lanczos repeated-run case
- thick-restart Lanczos repeated-run case

### Optional only if the batch stays small

- LOBPCG repeated-run case

Interpretation:

- the required cases already prove repeated-run reuse across the migrated
  Lanczos-family paths
- LOBPCG is valuable, but only if it does not drag Day 11 into preconditioner
  or block-size experiment churn

## Proposed Stable Benchmark Cases

### Case A: grow-m Lanczos

- matrix:
  - `nos4`
- backend:
  - explicit `SPARSE_EIGS_BACKEND_LANCZOS`
- `which`:
  - `SPARSE_EIGS_LARGEST`
- `k`:
  - `5`

Why this case:

- small and stable
- already part of the eigensolver corpus
- good proof of repeated one-shot vs reusable internal grow-m execution

### Case B: thick-restart Lanczos

- matrix:
  - `bcsstk14`
- backend:
  - explicit `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`
- `which`:
  - `SPARSE_EIGS_LARGEST`
- `k`:
  - `5`

Why this case:

- exercises the larger bounded-memory Lanczos path
- aligns with the existing thick-restart corpus and benchmark notes
- gives the repeated-run comparison a materially different migrated backend

### Optional Case C: LOBPCG

- matrix:
  - `bcsstk04`
- backend:
  - explicit `SPARSE_EIGS_BACKEND_LOBPCG`
- `which`:
  - `SPARSE_EIGS_SMALLEST`
- `k`:
  - `3`
- preconditioner:
  - IC(0)

Why this case:

- consistent with the existing example/test corpus
- exercises the migrated block path
- still bounded if kept to one explicit preconditioned shape

## Measurement Rules

Day 11 should measure:

- one-shot wall time
- reusable-path wall time
- median comparison across repeated runs
- last-run iteration count
- convergence parity
- residual/output parity

Day 11 should **not** claim:

- universal speedups
- backend superiority in general
- corpus-wide gains
- asymptotic improvements beyond reduced repeated allocation churn

Interpretation:

- the benchmark should be behavior-first and modest
- Sprint 46 evidence should stay tightly coupled to what the migration actually
  changed

## Implementation Guidance for Day 11

The best Day 11 implementation shape is:

- a new narrow repeated-run benchmark driver
- explicit A/B case structure
- minimal local helper surface
- reuse of existing public entry points plus internal reusable seams where
  appropriate

It should **not** do:

- broad `bench_eigs.c` CLI redesign
- broad CSV/reporting redesign
- broad corpus enumeration
- extra benchmark framework abstraction

## Bottom Line

Day 10 defines the right Sprint 46 benchmark target:

- a narrow repeated-run A/B driver
- required grow-m + thick-restart cases
- optional bounded LOBPCG add-on
- modest measurement and claim scope
- no broad `bench_eigs` churn

That is the right design handoff for Sprint 46 Day 11.
