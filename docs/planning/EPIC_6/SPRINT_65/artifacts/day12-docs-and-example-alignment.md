# Sprint 65 Day 12: Docs and Example Alignment

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Align the high-signal user-facing workflow pages so Sprint 65's benchmark
governance model reads coherently:

- examples teach API workflow and ownership
- benchmarks prove retained workflow/performance behavior
- `make bench-canonical-report` captures threshold-free canonical snapshots

## Landed Scope

This batch intentionally stays limited to:

- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`

It intentionally does not widen into:

- implementation files
- benchmark binaries
- example code
- CI workflow changes
- broader maintainer-only policy expansion

## Landed Alignment

The Day 12 wording now aligns across the touched docs:

- `README.md`
  - explicitly separates example teaching from benchmark proof
  - exposes `make bench-canonical-report` as the threshold-free canonical
    snapshot surface
- `docs/tutorial.md`
  - keeps `example_analysis.c` as the repeated-run direct teaching example
  - hands off to `bench_refactor` / `bench_refactor_csc` and
    `make bench-canonical-report` for retained proof/reporting
- `examples/README.md`
  - keeps `example_analysis` as the adoption example
  - explicitly hands off to:
    - `bench_refactor`
    - `bench_refactor_csc`
    - `bench_iterative_reuse`
    - `bench_eigs_reuse`
    - `make bench-canonical-report`
- `benchmarks/README.md`
  - now states directly that examples are the API-adoption teaching surface
    while benchmarks are the workflow/performance proof surface

## Why This Batch Matters

Interpretation:

- users now hit the same ownership model in the top-level workflow docs, the
  repeated-run tutorial lane, the examples index, and the benchmark index
- the repeated-run direct workflow now has a cleaner path:
  - learn in `example_analysis`
  - prove in `bench_refactor_csc`
  - snapshot in `make bench-canonical-report`

## Docs-Only Sanity Checks

The targeted Day 12 sanity set was:

- `git diff -- README.md docs/tutorial.md examples/README.md benchmarks/README.md`
- terminology/alignment `rg`
- touched-surface `wc -l`
- branch status recheck

Measured touched-surface result:

- `README.md`: `998 -> 1000`
- `docs/tutorial.md`: `469 -> 477`
- `examples/README.md`: `147 -> 157`
- `benchmarks/README.md`: `347 -> 349`

## Day 12 Exit State

Sprint 65 now has:

- one coherent user-facing split between example teaching and benchmark proof
- one explicit handoff from repeated-run workflow adoption to canonical
  threshold-free reporting
- one bounded docs-only closeout state ahead of Day 13 validation
