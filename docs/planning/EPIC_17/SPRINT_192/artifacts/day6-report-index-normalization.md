# Sprint 192 Day 6: Report Index Normalization

## Summary

Day 6 integrated the selected methodology-bound benchmark lane with report-index
normalization coverage. The production normalizer already ingests canonical
benchmark report rows as advisory local measurements, so the implementation
kept hard selected benchmark freshness in the dedicated
`scripts/check_bench_canonical_freshness.py` gate and added tests proving the
normalizer preserves the metadata that downstream report review needs.

## Scope

Changed source surface:

| Surface | Day 6 change |
| --- | --- |
| `tests/test_normalize_report_index.py` | Expanded the synthetic canonical benchmark fixture to include selected-lane methodology metadata, asserted normalized field preservation, and added required benchmark artifact missing-output coverage. |

No production script changes were required on Day 6. The normalizer's existing
benchmark behavior is intentionally advisory, while the selected benchmark
freshness contract remains owned by the benchmark-specific checker.

## Normalized Benchmark Row Semantics

The canonical benchmark fixture now models a selected methodology-bound row for
`bench_refactor_csc` using `tests/data/suitesparse/nos4.mtx --repeat 1`.

The normalized row is expected to keep:

- `report_family=benchmark`;
- `subfamily=canonical`;
- `native_row_id=bench_refactor_csc`;
- `status=advisory`;
- `status_reason=measurement`;
- `source_commit=abc123`;
- `source_branch=sprint-141`;
- `platform=linux-x86_64`;
- `compiler=cc`;
- `freshness_status=generated_present_unchecked`;
- `freshness_reason=benchmark_row_loaded;stale_rules_deferred_to_days10_11`.

The `configuration` field is expected to preserve selected methodology details:

- `surface=canonical`;
- `category=measurement`;
- `report_label=test`;
- `build_mode=serial`;
- `omp_num_threads=unset`;
- `command=tests/data/suitesparse/nos4.mtx --repeat 1`;
- `relative_path=bench_refactor_csc.csv`;
- `row_report_family=benchmark`;
- `row_status=measurement`;
- `row_support_tier=hosted_selected`;
- `claim_boundary=hosted_selected_threshold_free`;
- `fixture_or_workload=nos4.mtx`;
- `matrix_size=n=100`;
- `repeat_semantics=configured_repeat_1`;
- `warmup=none_configured`;
- `variance=not_computed_single_sample`;
- `baseline=n/a`;
- `threshold=n/a`;
- `backend_context=n/a`;
- `methodology_notes=threshold_free_local_measurement%3Bnot_portable_performance_claim`.

This preserves the selected-lane evidence context without converting local
benchmark measurements into portable performance claims.

## Freshness Ownership

| Concern | Day 6 decision |
| --- | --- |
| Hard selected benchmark freshness | Owned by `scripts/check_bench_canonical_freshness.py`. |
| Report-index normalization | Advisory aggregation plus metadata preservation. |
| Missing required benchmark artifacts | Covered by the normalizer's generic required-family diagnostic. |
| Stale benchmark rows | Deferred to the dedicated checker and later hosted freshness work. |
| Production normalizer change | Not needed for Day 6 after tests confirmed existing field preservation. |

## Tests Added

`test_runtime_report_rows_preserve_boundaries()` now asserts that normalized
benchmark rows keep source identity and selected methodology metadata from the
canonical benchmark index.

`test_required_benchmark_freshness_reports_missing_artifacts()` verifies that
`--require-generated benchmark --check-freshness` fails clearly when benchmark
artifacts are absent:

```text
required generated family missing: benchmark
```

## Validation

Commands run:

```sh
python3 tests/test_normalize_report_index.py
python3 tests/test_bench_canonical_freshness.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 scripts/validate_corpus_schema.py
python3 -m py_compile tests/test_normalize_report_index.py scripts/normalize_report_index.py tests/test_bench_canonical_freshness.py scripts/check_bench_canonical_freshness.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- report-index normalization regression tests passed;
- benchmark canonical freshness regression tests passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- corpus schema validation passed;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 6.

## Day 7 Inputs

The Day 7 hosted lane design can rely on:

- selected benchmark metadata already flowing through the generated canonical
  index;
- normalized report rows retaining selected methodology fields for review;
- the dedicated benchmark checker remaining the hard local freshness owner;
- report-index normalization remaining advisory until hosted artifact
  publication semantics are defined.
