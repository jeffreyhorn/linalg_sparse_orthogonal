# Sprint 192 Day 10: Claim Calibration

## Summary

Day 10 calibrated documentation for the selected methodology-bound performance
lane and added a dedicated docs guard. The active documentation now names the
selected benchmark row, its threshold-free policy, and the non-claims that
prevent hosted freshness from being interpreted as performance superiority.

## Documentation Surfaces

| Surface | Day 10 update |
| --- | --- |
| `tests/corpus/README.md` | Added the exact selected performance target, workload, threshold-free values, and non-claims. |
| `tests/corpus/schemas/report_index_fields.md` | Added the selected benchmark target policy fields for normalized report-index interpretation. |
| `tests/test_selected_performance_docs.py` | Added a docs guard for required selected-performance markers and forbidden overclaims. |

Existing README, benchmark README, and maintainer-guide wording already carried
the main selected-performance boundaries; the new guard makes those markers
explicitly testable.

## Required Interpretation

The selected performance lane covers only:

```text
SRT-BENCH-REFACTOR-CSC-NOS4
bench_refactor_csc
tests/data/suitesparse/nos4.mtx --repeat 1
```

Required policy fields remain:

```text
status=measurement
baseline=n/a
threshold=n/a
warmup=none_configured
variance=not_computed_single_sample
support_tier=hosted_selected
claim_boundary=hosted_selected_threshold_free
```

## Required Non-Claims

Docs must not convert selected performance freshness into:

- portable performance;
- release benchmark proof;
- algorithmic superiority;
- platform parity;
- package or ABI support;
- runtime-loader support;
- external-library parity;
- OpenMP speedup evidence;
- backend superiority;
- state-of-the-art status.

## Docs Guard

`tests/test_selected_performance_docs.py` validates required markers in:

- `README.md`;
- `benchmarks/README.md`;
- `docs/maintainer_guide.md`;
- `tests/corpus/README.md`;
- `tests/corpus/schemas/report_index_fields.md`.

It also includes negative coverage for:

- missing required selected-performance markers;
- explicit unsupported selected-performance portable-performance claims.

## Validation

Commands run:

```sh
python3 tests/test_selected_performance_docs.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_comparison_workflow.py
python3 scripts/validate_corpus_schema.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile tests/test_selected_performance_docs.py tests/test_bench_canonical_freshness.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py scripts/check_bench_canonical_freshness.py scripts/normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected-performance docs guard passed;
- selected benchmark freshness tests passed;
- selected workflow guard tests passed;
- selected target schema validation passed;
- report-index normalization tests passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 10.

## Day 11 Inputs

Day 11 can extend the same guard style to additional failure and drift modes,
especially missing methodology fields, malformed generated rows, and any
workflow/docs drift that would broaden selected performance claims.
