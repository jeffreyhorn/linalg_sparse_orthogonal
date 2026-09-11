# Sprint 202 Day 1: Benchmark Freshness Intake

## Summary

Day 1 established the Sprint 202 scope for adding one hosted selected benchmark
freshness lane on one additional platform. The intake confirms that Sprint 202
starts from the Sprint 192 selected Linux lane and must not broaden performance,
platform, release, package, ABI, or state-of-the-art claims.

## Selected Performance Baseline

| Field | Current baseline |
| --- | --- |
| Selected target id | `SRT-BENCH-REFACTOR-CSC-NOS4` |
| Benchmark artifact | `build/bench-reports/canonical/bench_refactor_csc.csv` |
| Workload | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| Current hosted platform | Linux only |
| Current hosted job | `.github/workflows/ci.yml` / `hosted-performance-freshness` |
| Current local proof command | `make bench-canonical-report-freshness` |
| Hard selected checker | `scripts/check_bench_canonical_freshness.py` |
| Regression owner | `tests/test_bench_canonical_freshness.py` |
| Claim guard owner | `tests/test_selected_performance_docs.py` |

## Threshold-Free Contract

The selected benchmark freshness path remains methodology evidence only:

- `status=measurement`;
- `baseline=n/a`;
- `threshold=n/a`;
- `warmup=none_configured`;
- `variance=not_computed_single_sample`;
- `methodology_notes` includes `not_portable_performance_claim`.

Sprint 202 must preserve that contract unless a future sprint separately owns a
baseline, variance model, tolerance, and same-machine comparison policy.

## Initial Candidate Platforms

| Candidate | Day 1 interpretation |
| --- | --- |
| macOS hosted selected benchmark freshness | Likely strongest candidate because macOS hosted workflows already exist for selected comparison freshness, but benchmark runtime and claim wording still need review. |
| Windows hosted selected benchmark freshness | Valuable evidence gap, but path, shell, CMake configuration, executable location, and existing Windows selected benchmark non-claims increase implementation risk. |
| Linux second benchmark row | Lower fit because Sprint 202 asks for one additional platform, not another Linux row. |
| Local sentinel promotion | Adjacent but probably out of scope because Sprint 202 is about hosted selected freshness, not timing thresholds. |

## Day 1 Non-Claims

Sprint 202 Day 1 does not claim:

- portable performance;
- benchmark superiority;
- timing threshold or regression-baseline support;
- release benchmark readiness;
- broad platform parity;
- Windows selected benchmark freshness;
- macOS selected benchmark freshness;
- package-manager, ABI, runtime-loader, OpenMP speedup, backend superiority, or
  state-of-the-art evidence.

Those claims remain unavailable until the later Sprint 202 platform selection,
workflow implementation, tests, docs calibration, and hosted evidence review
are complete.

## Day 2 Handoff

Day 2 should rank platform/row pairs using:

- evidence value;
- hosted runtime budget;
- compiler and build-system fit;
- path and artifact normalization risk;
- existing workflow guard reuse;
- freshness diagnostic coverage;
- public claim-safety cost.

The default candidate row is the existing selected `bench_refactor_csc`
`nos4.mtx --repeat 1` workload unless Day 2 records a stronger reason to select
another row.

