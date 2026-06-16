# Sprint 71 Day 8: Public Header Narrative Cleanup Design

Date: 2026-06-16
Branch: `sprint-71`

## Purpose

Define the exact Sprint 71 cleanup contract for `include/sparse_cholesky.h`
before any header edits land.

## Header-Batch Design

### Keep in the header

The Cholesky header should keep only API-local truth:

- one-shot Cholesky usage and mutation semantics
- repeated-run handoff to `sparse_analysis.h`
- backend selector meaning
- `used_csc_path` meaning
- local progress/cancellation caveats
- public error-code semantics
- factorization and solve contract details

### Compress or remove from the header

The strongest removable density is:

- Sprint-number chronology
- ABI-history detail beyond the caller-facing compatibility point
- benchmark-reference spill
- broader maintainer-policy commentary that is not required for local API use

### Preserve exactly

The Day 9 batch must preserve:

- Cholesky as a one-shot public direct entry point
- the repeated-run lifecycle handoff to `sparse_analysis.h`
- `SPARSE_CHOL_BACKEND_AUTO`, `LINKED_LIST`, and `CSC` semantics
- `used_csc_path` as chosen-path telemetry
- invalid reorder/backend rejection before mutation
- reordered temporary-working-copy publication semantics
- local cancellation/progress caveats
- `SPARSE_ERR_BACKEND_CONTRACT` as a narrow CSC supernodal backend-contract
  error

## Support-Surface Follow-Through Map

No support surface should move with the header batch by default.

Current support authorities remain:

- `docs/tutorial.md`
  - teaching flow
- `examples/README.md`
  - adoption/example-side handoff
- `benchmarks/README.md`
  - workflow/performance proof interpretation
- `docs/maintainer_guide.md`
  - deeper policy and deferred reading

Support follow-through should happen only if the landed header wording would
otherwise create a contradiction.

## Day 9 Non-Touch Set

The exact non-touch set for the header batch is:

- `README.md`
- `INSTALL.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- other public headers
- implementation `src/` files
- permanent proof-owner test files
- platform/install workflow files

## Exit State

Sprint 71 Day 8 closes with one exact Day 9 design:

1. `include/sparse_cholesky.h` keeps API-local truth only
2. chronology and cross-surface spill are the main removal targets
3. support surfaces remain non-moving unless the header edit forces
   follow-through
4. the batch stays bounded to one public header
