# Sprint 212 Day 6: Methodology Schema Design

## Summary

Day 6 converts the Day 5 threshold-free policy decision into an
implementation-ready schema design. The design keeps selected canonical
benchmark freshness threshold-free and aligns the selected target manifest,
generated report fields, checker behavior, tests, and documentation wording.

## Schema Layers

| Layer | Owner | Responsibility |
| --- | --- | --- |
| Selected target authority | `tests/corpus/manifests/selected_report_targets.tsv` | Owns `SRT-BENCH-REFACTOR-CSC-NOS4`, hosted Linux/macOS workflow scope, claim scope, non-claims, and owner. |
| Generated report authority | `scripts/bench_canonical_report.sh` output: `index.tsv`, `manifest.txt`, selected CSV | Owns generated methodology values such as status, support tier, claim boundary, baseline, threshold, warmup, variance, and methodology notes. |
| Validation authority | `scripts/check_bench_canonical_freshness.py` and tests | Fails closed on selected identity drift, missing metadata, stale threshold fields, hosted/local claim drift, and selected CSV mismatch. |
| Interpretation authority | Benchmark docs, README, INSTALL, maintainer guide, corpus docs, report-index schema | Explains that the selected row is freshness/methodology evidence, not portable performance or a timing threshold. |

## Field Contract

| Field | Required value or rule | Validation owner |
| --- | --- | --- |
| `target_id` | `SRT-BENCH-REFACTOR-CSC-NOS4` | Manifest and freshness tests. |
| `family` / `subfamily` | `benchmark` / `canonical` | Manifest and freshness tests. |
| `target_key` / `artifact` | `bench_refactor_csc` | Freshness checker/tests. |
| `relative_path` | `bench_refactor_csc.csv` | Freshness checker/tests. |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` | Freshness checker/tests. |
| `fixture_or_workload` | `nos4.mtx` | Freshness checker/tests. |
| `matrix_size` | `n=100`, matching selected CSV `n` | Freshness checker/tests. |
| `status` | `measurement` | Freshness checker/tests; docs guard. |
| `support_tier` | Hosted selected row: `hosted_selected`; local/unselected rows stay bounded by checker policy. | Freshness checker/tests; manifest tests. |
| `claim_boundary` | Hosted selected row: `hosted_selected_threshold_free`; local selected row: `local_threshold_free`; unselected rows: `local_threshold_free`. | Freshness checker/tests; docs guard. |
| `repeat_semantics` | `configured_repeat_1` | Freshness checker/tests. |
| `warmup` | `none_configured` | Freshness checker/tests; docs guard. |
| `variance` | `not_computed_single_sample` | Freshness checker/tests; docs guard. |
| `baseline` | `n/a` | Freshness checker/tests; docs guard. |
| `threshold` | `n/a` | Freshness checker/tests; docs guard. |
| `backend_context` | `n/a` | Freshness checker/tests. |
| `methodology_notes` | Must include `not_portable_performance_claim`; should preserve threshold-free measurement meaning. | Freshness checker/tests; docs guard. |
| `workflow_platforms` | `linux;macos` | Manifest tests. |
| `non_claims` | Must preserve the Day 5 required non-claims, including no Windows selected benchmark freshness. | Manifest tests; docs guard. |

## Allowed And Forbidden Values

Allowed canonical selected policy values are:

- `status=measurement`
- `baseline=n/a`
- `threshold=n/a`
- `warmup=none_configured`
- `variance=not_computed_single_sample`
- `repeat_semantics=configured_repeat_1`
- `backend_context=n/a`
- `claim_boundary=local_threshold_free`
- `claim_boundary=hosted_selected_threshold_free`

Forbidden policy drift includes:

- numeric canonical selected `baseline` or `threshold`;
- `status=pass` for canonical selected freshness;
- hosted selected performance described as a timing gate;
- selected performance described as portable, cross-platform, superior, or
  state-of-the-art;
- S6 local sentinel smoke ceiling described as hosted selected canonical
  freshness;
- Windows selected benchmark freshness described as present.

## Fixture Plan

| Fixture | Expected behavior |
| --- | --- |
| Mutate selected `baseline` to `100.0` | Checker fails. Existing freshness fixture covers this. |
| Mutate selected `threshold` to `200.0` | Checker fails. Existing freshness fixture covers this. |
| Mutate selected `status` to `pass` | Checker fails. Existing freshness fixture covers this. |
| Mutate selected `warmup` away from `none_configured` | Checker fails. Existing fixture covers `not_recorded`; add another only if implementation changes warrant it. |
| Mutate selected `variance` away from `not_computed_single_sample` | Checker fails. Existing fixture covers `not_recorded`; add another only if implementation changes warrant it. |
| Remove `not_portable_performance_claim` from `methodology_notes` | Checker fails clearly. Add or confirm fixture on Day 7. |
| Add threshold-promoting methodology note | Checker should fail if Day 7 adds forbidden-token enforcement. |
| Change unselected row claim boundary away from `local_threshold_free` | Checker fails. Add companion fixture if missing. |
| Drop selected benchmark non-claim from manifest | Manifest test fails. Add if current coverage is not exact enough. |
| Add docs wording that hosted selected performance is a timing gate | Docs guard fails. Existing docs guard covers this. |
| Add docs wording that S6 is hosted selected canonical freshness | Docs guard should fail if Day 10-11 adds exact sentinel/canonical marker. |
| Remove future threshold-prerequisite wording | Docs guard should fail after documentation calibration adds that marker. |

## Implementation Priorities

Day 7 should start with existing tooling/tests and add only missing enforcement:

1. confirm or add `methodology_notes` omission coverage;
2. confirm or add unselected claim-boundary drift coverage for both
   `support_tier` and `claim_boundary`;
3. add exact benchmark non-claim manifest coverage if it is not already exact;
4. leave prose calibration to Days 10-11 unless a guard marker must be added
   earlier.

## Day 6 Outcome

Item 212.3 now has an implementation-ready design for the selected
threshold-free policy. Every required methodology field has an owner, allowed
values are explicit, stale threshold wording is identified, and the fixture
plan gives Days 7-9 a bounded implementation target.

## Validation

Day 6 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
