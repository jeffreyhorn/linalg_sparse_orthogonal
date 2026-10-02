# Sprint 212 Day 5: Product Policy Decision

## Decision

Sprint 212 will implement **threshold-free deferral hardening** for selected
hosted canonical benchmark freshness.

Sprint 212 will not add a hosted selected timing threshold for
`SRT-BENCH-REFACTOR-CSC-NOS4`. The current evidence supports freshness,
artifact, manifest, and methodology validation for the selected
`bench_refactor_csc` row, but it does not support a hosted pass/fail timing
gate.

## Selected Scope

| Field | Selected policy |
| --- | --- |
| Selected target | `SRT-BENCH-REFACTOR-CSC-NOS4` |
| Benchmark row | `bench_refactor_csc` |
| Workload | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| Hosted lanes | Linux `sprint168-selected-performance-freshness`; macOS `sprint202-macos-selected-performance-freshness` |
| Policy | Threshold-free freshness and methodology hardening |
| Canonical status | `measurement` |
| Baseline | `n/a` |
| Threshold | `n/a` |
| Warmup | `none_configured` |
| Variance | `not_computed_single_sample` |
| Local claim boundary | `local_threshold_free` |
| Hosted claim boundary | `hosted_selected_threshold_free` |

## Rationale

The Day 4 thresholded criteria require runner stability, repeat/sample policy,
warmup, variance, baseline provenance, a threshold value, artifact retention,
and claim-boundary wording. The current branch has exact selected row identity
and useful hosted metadata, but it does not satisfy the threshold criteria:

- retained Linux hosted samples changed CPU model under the same runner label;
- retained macOS hosted samples include a visible timing outlier;
- selected rows use `configured_repeat_1`;
- selected rows use `warmup=none_configured`;
- selected rows use `variance=not_computed_single_sample`;
- canonical selected freshness uses `baseline=n/a` and `threshold=n/a`;
- hosted artifacts retain for seven days, so threshold failures would not have
  a durable evidence trail unless a separate baseline policy exists.

Threshold-free deferral hardening is therefore the most truthful product
policy for Sprint 212.

## Rejected Alternatives

| Alternative | Reason rejected |
| --- | --- |
| Hosted Linux selected threshold | CPU variability, single sample, no warmup, no variance, no baseline, and no threshold. |
| Hosted macOS selected threshold | Timing outlier, single sample, no warmup, no variance, no baseline, and no threshold. |
| Combined Linux/macOS threshold | Cross-platform timing comparability is explicitly unclaimed and would overstate the evidence. |
| Promote S6 local smoke ceiling as hosted selected freshness | S6 is local sentinel governance and must remain separate from hosted selected canonical freshness. |
| Broad benchmark threshold policy | Out of scope; Sprint 212 owns one selected methodology decision only. |

## Required Non-Claims

The implementation must not claim:

- portable performance;
- hosted selected timing threshold;
- release benchmark status;
- algorithmic superiority;
- platform parity;
- Windows selected benchmark freshness;
- package-manager distribution;
- package, ABI, shared-library, or runtime-loader support;
- broad benchmark-family publication;
- backend superiority;
- OpenMP speedup;
- state-of-the-art performance.

## Implementation Direction

Days 6-11 should implement and document the threshold-free policy rather than
a hosted threshold:

| Surface | Direction |
| --- | --- |
| Freshness checker | Preserve exact selected identity, hosted/local boundary, and threshold-free methodology fields; add fail-closed checks only where Day 6 finds gaps. |
| Freshness tests | Add negative fixtures for stale threshold/baseline promotion, missing methodology fields, unsupported threshold carryover, and hosted/local claim drift as needed. |
| Manifest tests | Preserve selected benchmark non-claims, Linux/macOS-only hosted evidence, and no Windows selected benchmark freshness. |
| Docs guard | Require wording that keeps canonical selected freshness threshold-free and separates it from local sentinel thresholds. |
| Benchmark docs | Clarify future threshold prerequisites and sentinel/canonical separation if current wording is not already testable enough. |
| Maintainer guide | Record the Day 5 decision and future evidence required before threshold promotion. |

## Day 5 Outcome

Item 212.2 is complete. Sprint 212 now has a documented product policy:
threshold-free deferral hardening for selected hosted canonical benchmark
freshness. The implementation surface is bounded to methodology fields,
guards, manifest checks, and documentation calibration. Unsupported threshold
expansion and portable performance claims are explicitly rejected.

## Validation

Day 5 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
