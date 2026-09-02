# Sprint 192 Day 5: Methodology Metadata Hardening

## Summary

Day 5 hardened the selected benchmark methodology metadata path with focused
tests. No benchmark C source or generated CSV schema change was needed.

Changed source surface:

| Surface | Day 5 change |
| --- | --- |
| `tests/test_bench_canonical_freshness.py` | Added selected CSV-to-index fixture coherence coverage and TSV control-character rejection coverage for environment-provided methodology metadata. |

The existing generator already emits the Day 3 required methodology fields, so
Day 5 focused on preventing drift in the selected fixture/metadata contract.

## Fixture Coherence Test

Added `test_selected_benchmark_csv_matches_index_fixture_contract()` to verify
that the generated selected CSV row and selected `index.tsv` row agree without
making timing values deterministic.

The test asserts:

| Contract field | Expected value |
| --- | --- |
| selected index `artifact` | `bench_refactor_csc` |
| selected CSV `benchmark` | `bench_refactor_csc` |
| selected index `fixture_or_workload` | `nos4.mtx` |
| selected CSV `matrix` | `nos4.mtx` |
| selected index `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| selected index `matrix_size` | `n=100` |
| selected CSV `n` | `100` |
| selected index `repeat_semantics` | `configured_repeat_1` |
| selected CSV `scenario` | `chol_spd` |
| selected CSV `nnz` | `594` |
| selected CSV backend fields | `n/a` for LDLT request, selection, and fallback |

The test intentionally does not assert exact timing values, residual values, or
`speedup_refactor` because those are measurement context and can vary by run.

## Metadata Control-Character Test

Added `test_generator_rejects_tsv_control_characters_in_methodology_metadata()`
to verify that environment-provided methodology metadata cannot inject TSV
control characters into generated report rows.

The test runs:

```sh
BENCH_CANONICAL_REPORT_LABEL='bad<TAB>label' make bench-canonical-report
```

and expects a clear generator failure:

```text
BENCH_CANONICAL_REPORT_LABEL must not contain tabs or newlines
```

This covers the most visible hosted/local report label field. The generator
already applies the same `reject_tsv_control_chars()` path to support tier,
claim boundary, runner context, build flags, CPU model, build mode override,
thread setting, methodology notes, and emitted index-row fields.

## Helper Additions

Day 5 added two small test helpers:

- `selected_index_row(report_dir)`;
- `selected_benchmark_csv_row(report_dir)`.

They keep fixture-coherence assertions local to the benchmark freshness test
file and avoid adding a new production abstraction before there is a broader
need.

## Implementation Decisions

| Question | Decision |
| --- | --- |
| Centralize selected benchmark constants? | Defer production changes. Existing checker constants remain the authority, and the new tests make selected row drift visible. |
| Parse `matrix_size` from the CSV in production? | Defer. The selected row uses `matrix_size=n=100` as a dimension label, and the new test verifies it agrees with CSV `n=100`. |
| Add benchmark CSV schema columns? | No. Existing CSV fields provide enough workload, matrix, backend, timing, speedup-context, and residual-context data. |
| Edit `benchmarks/bench_refactor_csc.c`? | No. No C benchmark defect was found. |
| Promote timing assertions? | No. Timing values remain measurement context, not deterministic test inputs or performance pass/fail evidence. |

## Generated Artifact Policy

Day 5 validation regenerated ignored benchmark artifacts under:

```text
build/bench-reports/canonical/
```

No generated report artifacts were added to source control.

## Day 5 Validation

Commands run:

```sh
python3 tests/test_bench_canonical_freshness.py
make bench-canonical-report-freshness
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile tests/test_bench_canonical_freshness.py scripts/check_bench_canonical_freshness.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- benchmark freshness regression tests passed, including the new
  CSV-to-index fixture coherence and TSV control-character rejection tests;
- selected canonical benchmark local freshness passed;
- corpus schema validation passed;
- normalized benchmark freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 5, so `make format && make lint &&
  make test` is not required.

## Day 6 Handoff

Day 6 should focus on report-index normalization:

1. verify whether normalized benchmark rows preserve all fields required by the
   selected benchmark checker;
2. decide whether `normalize_report_index.py --family benchmark
   --check-freshness` should call or mirror the selected checker contract;
3. add tests for missing selected benchmark artifacts, malformed metadata, or
   selected row drift if the normalizer is hardened;
4. avoid duplicating selected benchmark constants in another location unless
   the duplication is guarded by tests.
