# Sprint 192 Day 9: Regression Policy Decision

## Summary

Day 9 keeps the hosted selected performance lane threshold-free. The lane
checks hosted freshness, selected row identity, methodology metadata, artifact
scope, and claim boundaries for `bench_refactor_csc` on
`tests/data/suitesparse/nos4.mtx --repeat 1`; it does not enforce a hosted
timing threshold or publish a performance pass claim.

## Decision

| Question | Decision |
| --- | --- |
| Add a hosted benchmark timing threshold now? | No. |
| Keep selected hosted row threshold-free? | Yes. |
| Selected row status | `measurement` |
| Selected row baseline | `n/a` |
| Selected row threshold | `n/a` |
| Selected row variance | `not_computed_single_sample` |
| Selected row warmup | `none_configured` |
| Hard hosted gate | Freshness, metadata, selected artifact presence, and claim boundary only. |
| Local sentinels | Remain separate local regression governance. |

## Rationale

The selected hosted lane runs on GitHub-hosted Linux where CPU assignment and
load can vary across runs. A single `--repeat 1` benchmark row is useful for
methodology-bound freshness evidence, but it is not a statistically defensible
runtime baseline.

Adding a hosted timing threshold now would create a brittle signal and invite
overclaims. A defensible threshold policy would need a recorded hosted baseline,
runner class, repeat count, warmup policy, variance model, tolerance, and
same-machine comparison semantics.

## Enforcement

`tests/test_bench_canonical_freshness.py` now includes explicit regression
policy tests:

- selected manifest non-claims must include no portable performance, release
  benchmark, algorithmic superiority, platform parity, state-of-the-art, or
  package/ABI claim;
- selected `baseline` must remain `n/a`;
- selected `threshold` must remain `n/a`;
- selected `status` must remain `measurement` and cannot become `pass`.

These tests make unsupported threshold broadening fail at the selected
benchmark freshness boundary.

## Separated Surfaces

| Surface | Interpretation |
| --- | --- |
| Hosted selected performance freshness | Methodology-bound freshness and selected artifact evidence only. |
| `make performance-sentinels` | Local sentinel governance with its own baseline and machine context. |
| `wall-check` | Existing local hard threshold surface. |
| Canonical benchmark CSV timings | Local measurement context, not portable pass/fail evidence. |

Local sentinel thresholds do not supply the hosted selected-performance
baseline. Hosted selected performance freshness must not inherit local sentinel
pass/fail meaning.

## Required Future Work Before Threshold Promotion

A later sprint can revisit hosted timing enforcement only after defining:

- hosted baseline source and update procedure;
- runner class and machine-context policy;
- repeat count and warmup policy;
- variance model and tolerance;
- same-run or same-machine comparison semantics;
- artifact fields for baseline provenance;
- documentation that distinguishes threshold evidence from performance
  superiority.

## Validation

Commands run:

```sh
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile tests/test_bench_canonical_freshness.py tests/test_selected_comparison_workflow.py scripts/check_bench_canonical_freshness.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected benchmark freshness tests passed, including the new policy
  regressions;
- workflow guard tests passed;
- report-index normalization regression tests passed;
- selected target schema validation passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 9.

## Day 10 Inputs

Day 10 claim calibration should carry forward the threshold-free policy:
freshness is not a runtime improvement claim, and local sentinel thresholds do
not broaden hosted selected-performance evidence.
