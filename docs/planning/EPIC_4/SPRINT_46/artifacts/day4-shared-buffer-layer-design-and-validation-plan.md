# Sprint 46 Day 4 Artifact: Shared Buffer Layer Design and Validation Plan

## Purpose

Bound the common eigensolver buffer-backed helper layer and define the
implementation-day validation shape before Sprint 46 begins code changes.

## Shared Buffer Layer Scope

The shared eigensolver layer should stay narrow and capacity-oriented.

### It should own

- checked sizing for common eigensolver buffer shapes
- contiguous internal storage ownership
- reserve/grow behavior
- typed prepare helpers for:
  - grow-m Lanczos
  - thick-restart Lanczos
  - LOBPCG
- narrow reset/zero helpers for slices that require clean-state reuse

### It should not own

- eigensolver convergence policy
- Ritz extraction policy
- dense spectral math kernels
- restart orchestration
- soft-lock policy
- preconditioner invocation
- shift-invert factor ownership
- result emission/reporting semantics

This keeps the shared seam about memory ownership, capacity, and typed slicing,
not about collapsing the eigensolver algorithms into one generic engine.

## Shared vs Local Helper Boundary

### Shared-owner candidates

- owner init/free
- checked reserve/grow helpers
- typed prepare helpers
- packed-count calculation
- narrow reset/zero helpers

### Keep solver-local

- `lanczos_iterate_op(...)`
- `s20_ritz_pairs(...)`
- `s20_select_indices(...)`
- arrowhead assembly / dense Jacobi helpers
- thick-restart lock/restart-state choreography
- LOBPCG RR-step sequencing
- soft-lock logic
- shift-invert / preconditioner composition

## Reuse Model

The shared layer must support:

- repeated stable-dimension runs without reallocating
- bounded internal growth when later runs exceed current capacity

It must not imply:

- preservation of old Krylov state
- preservation of old Ritz data
- preservation of old restart/search-direction mathematical state

Sprint 46 reuse is capacity reuse, not eigensolver-history reuse.

## Validation Plan

### Mandatory full C/C-header gate

For any `*.c` / `*.h` change, always run:

- `make format`
- `make lint`
- `make test`

### Strong reviewed baseline for substantial batches

For shared-layer landings or multi-family eigensolver migrations, the default
stronger proof should be:

- `make quality-review-full`

### Targeted eigensolver follow-ons

Run the touched-binary follow-ons when the changed surface justifies them:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`
- `./build/bench_eigs`

### Additional notes

- example and benchmark reruns are targeted, not universal
- dead-code remains a separate serialized sibling path and is not part of the
  default Sprint 46 eigensolver code-day gate unless Sprint 46 changes
  dead-code surfaces directly

## First Code Landing Order

The bounded implementation order is:

1. shared eigensolver owner + typed prepare helpers
2. grow-m Lanczos migration
3. thick-restart Lanczos migration
4. LOBPCG migration
5. wrapper, benchmark, and memory-contract closeout

This preserves the internal-first rollout pattern already proven in Sprint 45
while keeping the first code batch intentionally narrow.
