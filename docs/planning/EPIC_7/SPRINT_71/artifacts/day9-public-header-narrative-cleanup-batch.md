# Sprint 71 Day 9: Public Header Narrative Cleanup Batch

Date: 2026-06-16
Branch: `sprint-71`

## Purpose

Land the bounded Sprint 71 header cleanup on `include/sparse_cholesky.h`
without weakening local contract truth or widening into support surfaces.

## Landed Batch

The landed cleanup in `include/sparse_cholesky.h` compressed:

- Sprint-number chronology around backend dispatch
- ABI-history detail beyond the caller-facing compatibility point
- benchmark-history references
- broader maintainer-policy spill inside callback commentary

The header now reads more directly as:

- a one-shot Cholesky entry-point reference
- a repeated-run handoff to `sparse_analysis.h`
- a backend/telemetry reference
- a local mutation/cancellation/error reference

## Preserved Reference Truth

The batch preserved:

- Cholesky as a one-shot public direct entry point
- the repeated-run handoff to `sparse_analysis.h`
- `SPARSE_CHOL_BACKEND_AUTO`, `LINKED_LIST`, and `CSC` semantics
- `used_csc_path` as chosen-path telemetry
- invalid reorder/backend rejection before mutation
- reordered temporary-working-copy publication semantics
- local progress/cancellation caveats
- `SPARSE_ERR_BACKEND_CONTRACT` as a narrow CSC supernodal backend-contract
  error

## Non-Widening Result

No support-surface follow-through was needed:

- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`

The batch remained bounded to one public header.

## Validation

Because `include/sparse_cholesky.h` changed, I ran:

- `make format`
- `make lint`
- `make test`

All passed.

Touched-surface raw `wc -l` count after the landing:

- `include/sparse_cholesky.h` = `216`

## Exit State

Sprint 71 Day 9 closes with the strongest remaining header/reference
contradiction materially cleaner:

1. `include/sparse_cholesky.h` now reads more like a local API reference
2. support surfaces remained untouched
3. local backend, cancellation, and error semantics stayed intact
4. the bounded code-day validation gate passed
