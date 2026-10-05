# Sprint 212 Day 4: Threshold Decision Criteria

## Summary

Day 4 defines the acceptance gate for the Sprint 212 product decision. The
criteria intentionally separate a real thresholded benchmark gate from a
threshold-free freshness/methodology policy. A thresholded gate is acceptable
only if it has enough runner, compiler, repeat, warmup, variance, baseline,
threshold, retention, and claim-boundary evidence to be enforceable without
implying portable performance.

## Decision Rule

Day 5 must choose one of these policies:

1. **Thresholded selected gate**: allowed only when every threshold criterion
   below is satisfied and enforceable by tests.
2. **Threshold-free deferral hardening**: required when any threshold criterion
   is missing, ambiguous, hosted-only, unretained, or overclaim-prone.

Given the Day 1-3 evidence, hosted selected benchmark thresholding is blocked.
The only threshold-adjacent candidate is a deliberately local smoke ceiling
such as the existing S6 lane, and that must remain separate from hosted
selected canonical freshness.

## Thresholded-Gate Acceptance Criteria

| Criterion | Must be true before implementation | Current Day 4 status |
| --- | --- | --- |
| Selected target identity | The policy targets exactly `SRT-BENCH-REFACTOR-CSC-NOS4`, `bench_refactor_csc`, `nos4.mtx`, `--repeat 1`, and one selected row. | Present. |
| Runner scope | The policy defines same-machine comparison, stable machine class, or a local-only smoke ceiling. | Missing for hosted selected lanes. |
| Compiler/build/thread scope | Compiler, build flags, build mode, and thread count are recorded and constrained. | Present as context; not yet a threshold policy. |
| Repeat/sample policy | The policy has repeated samples or explicitly says the gate is a non-statistical smoke ceiling. | Missing for hosted thresholding. |
| Warmup policy | The policy has warmup evidence or a documented no-warmup smoke-only rationale. | Missing for hosted thresholding. |
| Variance/outlier rule | The policy defines variance handling, outlier handling, or non-statistical smoke-only semantics. | Missing for hosted thresholding. |
| Baseline provenance | The policy has an auditable baseline owner, value, date, runner scope, and update process. | Missing for canonical selected freshness. |
| Threshold value | The policy has a numeric threshold or allowed regression rule with units. | Missing for canonical selected freshness. |
| Artifact retention | Evidence remains inspectable after review or baseline is source-controlled. | Weak for hosted artifacts: 7-day retention. |
| Claim boundary | Docs and tests prevent portable performance, platform parity, release, package/ABI, or state-of-the-art claims. | Present for threshold-free wording; threshold wording would need new guards. |

## Threshold-Free Deferral Acceptance Criteria

| Criterion | Required proof |
| --- | --- |
| Freshness remains selected | The checker continues to validate only the exact selected row and artifacts. |
| Threshold-free fields remain exact | `baseline=n/a`, `threshold=n/a`, `status=measurement`, `warmup=none_configured`, and `variance=not_computed_single_sample` remain enforced for canonical selected freshness. |
| Hosted/local boundary remains explicit | Hosted selected rows use `hosted_selected_threshold_free`; local rows use `local_threshold_free`; unselected rows remain local-only. |
| Stale threshold wording is rejected | Documentation guards reject selected hosted timing-gate, portable speed, broad benchmark publication, and cross-platform timing comparability wording. |
| Sentinel thresholds remain separate | S5/S6 local sentinel behavior remains local regression governance, not hosted selected canonical publication. |
| Missing metadata fails closed | Missing selected artifacts, missing manifest agreement, hosted local placeholders, stale support tiers, and stale claim boundaries fail validation. |
| Future threshold prerequisites are documented | Maintainer guidance states what evidence is required before any future threshold promotion. |

## Stop Conditions

Stop and ask for direction instead of implementing if:

- hosted artifacts contradict the selected manifest;
- a threshold is requested without baseline provenance or update policy;
- a threshold is requested for combined Linux/macOS timing values;
- the policy requires solver or benchmark-behavior changes instead of
  methodology validation;
- the policy depends on hosted history that expires before review;
- documentation cannot preserve non-portable, non-superiority wording;
- required quality checks fail.

## Evidence-To-Test Mapping

| Criterion | Enforcement path |
| --- | --- |
| Exact selected row identity | `scripts/check_bench_canonical_freshness.py`; `tests/test_bench_canonical_freshness.py`; selected manifest tests. |
| Hosted metadata validity | Hosted mode freshness checks and hosted regression fixtures. |
| Threshold-free field exactness | Freshness checker fixtures for baseline, threshold, status, warmup, variance, and methodology notes. |
| Non-claim wording | `tests/test_selected_performance_docs.py` plus README, INSTALL, benchmark README, maintainer guide, corpus README, and schema markers. |
| Manifest claim scope | `tests/test_selected_report_targets_manifest.py`. |
| Sentinel/canonical separation | Benchmark README and maintainer guide markers; add docs-guard markers if Day 5 selects threshold-free hardening. |
| Future threshold prerequisites | Maintainer guide wording and selected performance docs markers. |

## Day 4 Outcome

Item 212.2 now has explicit criteria before the Day 5 decision. The criteria
make hosted thresholding fail on current evidence and make threshold-free
deferral testable rather than vague. Day 5 should use these criteria to choose
the product policy and define the implementation scope.

## Validation

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
