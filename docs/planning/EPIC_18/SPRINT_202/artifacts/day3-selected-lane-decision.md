# Sprint 202 Day 3: Selected Platform And Row Decision

## Summary

Day 3 selects exactly one Sprint 202 platform/row pair:

- platform: macOS hosted runner;
- selected row: `SRT-BENCH-REFACTOR-CSC-NOS4`;
- benchmark: `bench_refactor_csc`;
- workload: `tests/data/suitesparse/nos4.mtx --repeat 1`;
- artifact: `build/bench-reports/canonical/bench_refactor_csc.csv`.

This completes the Sprint 202 item 202.1 planning decision. Workflow,
validator, docs, and hosted evidence implementation remain for later days.

## Decision Rationale

macOS is the selected additional hosted platform because it adds non-Linux
freshness evidence while keeping the implementation close to the existing
Linux selected benchmark lane:

- POSIX-style shell and artifact paths match the current generator model;
- macOS CI already exists and carries selected comparison freshness precedent;
- the existing selected benchmark row can be reused without adding a new
  canonical benchmark row;
- the selected artifact upload can stay limited to `bench_refactor_csc.csv`,
  `index.tsv`, and `manifest.txt`;
- the lane can remain threshold-free and claim-safe.

Windows remains valuable, but is deferred for this sprint's first additional
benchmark freshness lane because it has higher path, shell, CMake configuration,
and public non-claim coordination risk.

## In Scope

| Surface | Selected scope |
| --- | --- |
| Workflow file | `.github/workflows/macos-ci.yml`. |
| Workflow lane | One reviewed macOS selected benchmark freshness job. |
| Freshness checker | `scripts/check_bench_canonical_freshness.py --mode hosted`. |
| Selected artifact | `build/bench-reports/canonical/bench_refactor_csc.csv`. |
| Companion files | `build/bench-reports/canonical/index.tsv`, `build/bench-reports/canonical/manifest.txt`. |
| Metadata | Platform-bound hosted selected threshold-free benchmark metadata. |
| Docs | Public and maintainer wording for Linux and macOS selected benchmark freshness only. |

## Out Of Scope

| Surface | Status |
| --- | --- |
| Windows selected benchmark freshness | Deferred. |
| New selected benchmark rows | Deferred. |
| Broad canonical benchmark publication | Deferred. |
| Timing thresholds or regression baselines | Deferred. |
| Portable Linux/macOS performance claims | Deferred. |
| Benchmark superiority claims | Deferred. |
| Package-manager, ABI, runtime-loader, and OpenMP speedup claims | Deferred. |

## Required Claim Boundary

The selected lane may eventually claim only that the macOS hosted runner
regenerated and freshness-checked the selected `bench_refactor_csc` report
artifact with required threshold-free metadata. It must not claim that macOS
performance is comparable to Linux, that timing thresholds are enforced, or
that the library has broad performance portability.

## Day 4 Handoff

Day 4 should define the metadata contract for this selected macOS lane before
any workflow or checker edits. The contract must preserve:

- `support_tier=hosted_selected`;
- `claim_boundary=hosted_selected_threshold_free`;
- `status=measurement`;
- `baseline=n/a`;
- `threshold=n/a`;
- `warmup=none_configured`;
- `variance=not_computed_single_sample`;
- `methodology_notes` containing `not_portable_performance_claim`.
