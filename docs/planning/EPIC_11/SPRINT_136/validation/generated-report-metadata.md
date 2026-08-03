# Sprint 136 Generated Report Metadata

## Scope

Day 7 generated and inspected the canonical benchmark, performance sentinel,
and large-matrix guardrail report bundles.

These reports are generated evidence with freshness and support-tier context.
They are not public release artifacts, portable performance proof, broad
correctness proof, platform parity proof, or state-of-the-art comparison
evidence by themselves.

## Report Bundle Inventory

| Report family | Command | Output directory | Status |
| --- | --- | --- | --- |
| Canonical benchmark report | `make bench-canonical-report` | `build/bench-reports/canonical/` | Passed |
| Performance sentinels | `make performance-sentinels` | `build/bench-reports/sentinels/` | Passed |
| Large-matrix guardrails | `make large-matrix-guardrails` | `build/bench-reports/large-matrix-guardrails/` | Passed |

## Freshness Summary

| Report family | Generated at UTC | Branch | Commit | Platform | Compiler |
| --- | --- | --- | --- | --- | --- |
| Canonical benchmark report | `2026-07-26T00:09:46Z` | `sprint-136` | `b178de48` | Darwin `x86_64` | Apple clang `11.0.0` |
| Performance sentinels | `2026-07-26T00:10:01Z` | `sprint-136` | `b178de48` | Darwin `x86_64` | Apple clang `11.0.0` |
| Large-matrix guardrails | `2026-07-26T00:11:53Z` | `sprint-136` | `b178de48` | Darwin `x86_64` | Apple clang `11.0.0` |

## Row Counts And Interpretation

| Report family | Index file | Data rows | Interpretation |
| --- | --- | ---: | --- |
| Canonical benchmark report | `build/bench-reports/canonical/index.tsv` | 4 | Threshold-free local snapshot of maintained benchmark surfaces. |
| Performance sentinels | `build/bench-reports/sentinels/sentinels.tsv` | 11 | Three S5 reviewed thresholded local wall-check rows passed; eight S2 threshold-free local report rows recorded. |
| Large-matrix guardrails | `build/bench-reports/large-matrix-guardrails/index.tsv` | 6 | Four reviewed guardrail rows passed; two supplemental opt-in rows were skipped. |

## Artifact Lists

### Canonical Benchmark Report

- `bench_refactor_csc.csv`
- `bench_chol_csc.csv`
- `bench_iterative_reuse.csv`
- `bench_eigs_reuse.csv`
- `index.tsv`
- `manifest.txt`

### Performance Sentinels

- `sentinels.tsv`
- `manifest.txt`
- `wall_check.txt`
- `bench_chol_csc_nos4.csv`

### Large-Matrix Guardrails

- `index.tsv`
- `manifest.txt`
- `test_graph.txt`
- `test_reorder_nd.txt`
- `test_reorder_amd_qg.txt`
- `bench_reorder_sprint86.csv`

## Claim Boundaries

Preserved report boundaries:

- canonical benchmark rows are threshold-free local measurement snapshots;
- sentinel S5 rows are the existing local wall-check gate, while S2 rows are
  threshold-free local report context;
- large-matrix guardrail reviewed rows are bounded structural/report evidence;
- skipped supplemental guardrail rows remain opt-in and are not reviewed
  evidence;
- generated report timestamps and commit metadata provide freshness context,
  not CI/release guarantees;
- no generated row creates portable timing, scalability, memory, backend
  parity, platform parity, broad correctness, or state-of-the-art claims.
