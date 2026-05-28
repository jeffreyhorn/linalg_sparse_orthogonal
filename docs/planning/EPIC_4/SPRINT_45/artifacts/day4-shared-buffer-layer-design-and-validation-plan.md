# Sprint 45 Day 4 Artifact: Shared Buffer Layer Design and Validation Plan

## Purpose

Define the common iterative buffer layer that Sprint 45 implementation days
will land, and fix the validation contract for those implementation batches
before code edits begin.

## Main Day 4 Conclusion

Sprint 45 should introduce one small internal buffer layer, not a second
iterative subsystem.

That layer should own:

- contiguous storage ownership
- checked reserve/grow behavior
- typed view preparation
- narrow reset/zero helpers

It should not own:

- solver stopping logic
- callback semantics
- preconditioner invocation
- operator ownership
- residual-history policy

This keeps the reusable-workspace work bounded to allocation/reuse structure
rather than widening into algorithm redesign.

## Shared Layer Ownership

### The shared layer should own

#### 1. Storage ownership

- internal owner initialization
- internal owner destruction
- safe capacity growth
- storage reuse across repeated stable-dimension solves

#### 2. Checked sizing

- use Sprint 41 shared allocation helpers
- centralize count/bytes math for:
  - graph-sized vector bundles
  - Hessenberg/Arnoldi bundles
  - block `n * nrhs` bundles
  - optional extra preconditioner vectors

#### 3. Typed preparation

Expose typed prepare helpers for:

- CG
- GMRES
- block CG
- later MINRES extension

Each prepare helper should:

- validate shape inputs
- ensure capacity
- return typed slices with stable member names

#### 4. Narrow reset helpers

The common layer may own:

- zeroing fresh/required work slices
- clearing reused contiguous storage when a solver depends on `calloc`-style
  semantics
- resetting reusable support backing when the owner retains its storage across
  solves

## What Stays Solver-Local

Keep local to solver bodies or wrapper logic:

- `stag_tracker_t` update/check behavior
- `reshist_t`
- verbose/progress reporting
- CG recurrence updates
- GMRES restart/Arnoldi control
- MINRES Lanczos / QR sequencing
- block convergence aggregation
- per-column wrapper dispatch loops

Interpretation:

- these are algorithm-flow and policy helpers
- they are not part of the shared contiguous-storage seam

## Zeroing and Reset Rules

### The common layer should guarantee

- valid typed slices after prepare
- safe zeroed storage when a solver requires clean initialization
- explicit fresh-solve semantics for reused storage

### The common layer should not guarantee

- preservation of old numerical state
- resumable Krylov iterations
- automatic history retention across solves
- hidden reconstruction of solver-specific control variables

Interpretation:

- reuse is about allocation/capacity reuse
- not about continuation semantics

## Shared vs Local Helper Boundary

### Shared helper candidates

- owner init/free
- owner ensure-capacity/reserve
- typed view prepare helpers
- checked packed-count calculation
- narrow zero/reset helpers for owned storage

### Keep-local helpers

- early true-residual checks
- solver-specific breakdown handling
- callback reporting/cancellation flow
- preconditioner application choreography
- block-wrapper per-column orchestration

## Validation Plan for Implementation Days

### Mandatory floor for all `*.c` / `*.h` changes

```bash
make format
make lint
make test
```

### Stronger default for substantial shared-layer or migration batches

```bash
make quality-review-full
```

### Targeted iterative follow-on checks

Run the touched iterative binaries as justified:

- `./build/test_iterative`
- `./build/test_block_solvers`
- `./build/test_minres`
- `./build/test_bicgstab`
- `./build/test_stagnation`

### Example follow-on checks when needed

If `examples/` change:

- `./build/example_iterative`
- `./build/example_matrix_free`

### Benchmark follow-on checks when needed

When the repeated-solve benchmark batch lands:

- run the touched iterative benchmark binary or binaries
- keep that scope bounded to repeated-solve evidence

## Benchmark-Surface Guidance

The strongest Sprint 45 repeated-solve benchmark direction is:

- iterative repeated solves with stable dimensions
- one-shot repeated calls vs reusable-workspace repeated calls

Use:

- `benchmarks/bench_convergence.c` as the strongest likely iterative
  comparison surface

Treat:

- `benchmarks/bench_refactor.c` mainly as a repeated-run comparison precedent,
  not as a direct iterative implementation target

## Fixed Implementation Order

The implementation order is now explicit:

1. shared internal buffer layer
2. scalar CG / matrix-free CG migration
3. GMRES / matrix-free GMRES migration
4. block CG migration
5. wrapper normalization
6. repeated-solve benchmark batch
7. optional MINRES extension only if the shared layer remains clean

## Bottom Line

Sprint 45 now has:

- a bounded common buffer-layer design
- an explicit shared-vs-local helper boundary
- a narrow reset/zero contract
- a concrete validation plan
- a fixed first implementation order

That is the right Day 4 handoff into the first code batch.
