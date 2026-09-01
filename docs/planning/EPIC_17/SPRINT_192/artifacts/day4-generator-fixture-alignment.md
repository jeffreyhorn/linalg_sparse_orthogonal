# Sprint 192 Day 4: Generator and Fixture Alignment

## Summary

Day 4 aligns the selected `bench_refactor_csc` generator, fixture, report
writer, and freshness paths with the Day 3 methodology contract.

The current generator can emit the required methodology fields for the selected
lane. No benchmark CSV schema change is required before Day 5. The main Day 5
implementation need is to reduce duplicated selected-lane constants and add
focused tests that make the selected fixture metadata contract harder to drift.

## Selected Fixture Ownership

| Field | Owner | Day 4 decision |
| --- | --- | --- |
| Fixture path | `scripts/bench_canonical_report.sh` | Keep the selected command as `tests/data/suitesparse/nos4.mtx --repeat 1`. |
| Fixture file | `tests/data/suitesparse/nos4.mtx` | Source-controlled Matrix Market fixture remains the workload input. |
| Benchmark row identity | `benchmarks/bench_refactor_csc.c` | CSV row emits `benchmark=bench_refactor_csc`, `matrix=nos4.mtx`, `scenario=chol_spd`, `n=100`, and `nnz=594`. |
| Selected report row identity | `scripts/bench_canonical_report.sh` and `scripts/check_bench_canonical_freshness.py` | Keep selected `index.tsv` identity as `artifact=bench_refactor_csc`, `fixture_or_workload=nos4.mtx`, `matrix_size=n=100`, and `repeat_semantics=configured_repeat_1`. |
| Selected manifest authority | `tests/corpus/manifests/selected_report_targets.tsv` | Keep `SRT-BENCH-REFACTOR-CSC-NOS4` as the selected benchmark target authority. |

The Matrix Market header for `nos4.mtx` records `100 100 347` stored
symmetric entries. The selected benchmark CSV reports `n=100` and `nnz=594`
after project-side loading/expansion. Day 4 therefore keeps
`matrix_size=n=100` as a dimension label in the selected report row and does
not reinterpret it as nonzero-count evidence.

## Generated CSV Contract

The selected generated CSV currently emits one data row with these fields:

| Field | Day 4 observed role |
| --- | --- |
| `benchmark` | Confirms selected binary identity: `bench_refactor_csc`. |
| `category` | Benchmark-local category: `proof`. |
| `matrix` | Fixture basename: `nos4.mtx`. |
| `scenario` | Selected path: `chol_spd`. |
| `n` | Selected matrix dimension: `100`. |
| `nnz` | Project-loaded sparse nonzero count: `594`. |
| `ldlt_dense_backend_request` | `n/a` for selected SPD/Cholesky path. |
| `ldlt_dense_backend_selected` | `n/a` for selected SPD/Cholesky path. |
| `ldlt_dense_backend_fallback` | `n/a` for selected SPD/Cholesky path. |
| `analyze_ms` | Timing context only. |
| `refactor_public_ms` | Timing context only. |
| `refactor_csc_ms` | Timing context only. |
| `solve_public_ms` | Timing context only. |
| `solve_csc_ms` | Timing context only. |
| `speedup_refactor` | Descriptive ratio only, not a speedup claim. |
| `res_public` | Correctness context for the benchmark run. |
| `res_csc` | Correctness context for the benchmark run. |

The CSV has enough row-level workload and backend context for the selected
methodology contract. It does not need new columns for Day 5.

## Metadata Capture Alignment

| Metadata area | Current source | Day 4 alignment |
| --- | --- | --- |
| Report label | `BENCH_CANONICAL_REPORT_LABEL` | Keep as environment-provided; hosted mode must set it. |
| Support tier | `SPARSE_CANONICAL_SUPPORT_TIER` | Keep environment-provided for selected row only. |
| Claim boundary | `SPARSE_CANONICAL_CLAIM_BOUNDARY` | Keep environment-provided for selected row only. |
| Runner context | `SPARSE_CANONICAL_RUNNER_CONTEXT` | Keep environment-provided; hosted mode must not be `local`. |
| Build flags | `SPARSE_CANONICAL_BUILD_FLAGS` | Keep environment-provided; hosted mode must not be `not_recorded`. |
| CPU model | `SPARSE_CANONICAL_CPU_MODEL` | Keep environment-provided; `unknown` remains acceptable. |
| Build mode | `SPARSE_CANONICAL_BUILD_MODE` or runtime detection | Keep override/detection behavior. |
| Thread count | `OMP_NUM_THREADS` | Keep as explicit environment context. |
| Timestamp | `date -u` | Keep generator-owned timestamp. |
| Commit/branch | `git rev-parse` | Keep generator-owned source context. |
| Compiler | `${CC:-cc} --version` | Keep generator-owned compiler context. |
| Warmup/variance | generator constants | Keep as explicit limitations. |
| Baseline/threshold | generator constants | Keep `n/a` until Day 9 changes policy. |
| Methodology notes | `SPARSE_CANONICAL_METHODOLOGY_NOTES` or default | Keep required `not_portable_performance_claim` token. |

Control-character validation currently covers all environment-provided
metadata and all index-row values emitted through `emit_index_row()`. Day 4
does not identify a missing TSV-control-character check in the selected
metadata path.

## Schema Change Decision

No benchmark CSV schema change is required for Day 5.

Small report-generator/checker hardening remains in scope if Day 5 chooses to
implement it:

1. centralize selected-lane constants so command, fixture, artifact,
   matrix-size, repeat, warmup, variance, baseline, threshold, support tier,
   and claim boundary drift is easier to test;
2. add tests that compare `bench_refactor_csc.csv` row context with the
   selected `index.tsv` row contract;
3. add tests for environment-provided methodology values with TSV control
   characters;
4. add a clearer assertion that unselected canonical rows remain context even
   when hosted metadata is supplied for the selected row.

## Normalizer Alignment

`scripts/normalize_report_index.py --family benchmark --check-freshness`
loads benchmark rows and reports advisory freshness, but it does not currently
enforce the full selected benchmark methodology contract owned by
`scripts/check_bench_canonical_freshness.py`.

Day 4 records the Day 6 implementation path:

- keep `scripts/check_bench_canonical_freshness.py` as the authoritative
  selected benchmark checker;
- either call its contract logic from benchmark normalization or add
  normalizer tests that prove normalized benchmark rows preserve the fields
  needed by the dedicated checker;
- avoid creating a second divergent list of selected benchmark constants.

## Artifact Scope Alignment

The selected manifest requires only:

- `bench_refactor_csc.csv`;
- `index.tsv`;
- `manifest.txt`.

The hosted workflow currently uploads those files plus three contextual
canonical CSVs. Day 4 does not change workflow scope. Day 7/Day 8 must decide
whether to narrow the upload or explicitly guard the contextual-bundle
interpretation.

## Day 5 Implementation Checklist

Day 5 should focus on low-risk methodology metadata hardening:

1. add or centralize selected benchmark constants in the existing benchmark
   freshness test/checker path;
2. add fixture coherence tests that compare `bench_refactor_csc.csv` row fields
   to selected `index.tsv` fields;
3. add a regression for TSV control-character rejection in at least one
   environment-provided methodology field;
4. avoid editing `benchmarks/bench_refactor_csc.c` unless a real CSV defect is
   found, because any benchmark C edit triggers `make format && make lint &&
   make test`;
5. keep generated `build/bench-reports/canonical/` artifacts ignored.

## Day 4 Validation

Commands run:

```sh
head -n 20 tests/data/suitesparse/nos4.mtx
sed -n '360,430p' benchmarks/bench_refactor_csc.c
sed -n '620,690p' benchmarks/bench_refactor_csc.c
sed -n '70,185p' scripts/bench_canonical_report.sh
sed -n '740,790p' scripts/normalize_report_index.py
python3 - <<'PY'
import csv
from pathlib import Path
p = Path('build/bench-reports/canonical/bench_refactor_csc.csv')
rows = list(csv.DictReader(p.open(newline='')))
print('rows', len(rows))
for row in rows:
    for key in ('benchmark','category','matrix','scenario','n','nnz','ldlt_dense_backend_request','ldlt_dense_backend_selected','ldlt_dense_backend_fallback','speedup_refactor','res_public','res_csc'):
        print(f'{key}={row[key]}')
PY
python3 - <<'PY'
from pathlib import Path
p = Path('tests/data/suitesparse/nos4.mtx')
for line in p.read_text().splitlines():
    if not line.startswith('%'):
        rows, cols, stored = line.split()[:3]
        print(f'matrix_market_rows={rows}')
        print(f'matrix_market_cols={cols}')
        print(f'matrix_market_stored_entries={stored}')
        print('symmetry=real symmetric')
        break
PY
rg -n "reject_tsv_control_chars|SPARSE_CANONICAL|BENCH_CANONICAL_REPORT_LABEL|emit_index_row|selected_matrix_size|artifacts:" scripts/bench_canonical_report.sh
rg -n "matrix_size|warmup|variance|baseline|threshold|runner_context|build_flags|report_label|methodology_notes|selected_matrix_size" tests/test_bench_canonical_freshness.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- `nos4.mtx` is source-controlled and declares a 100-by-100 symmetric Matrix
  Market fixture with 347 stored entries;
- generated `bench_refactor_csc.csv` exposes one selected row with
  `benchmark=bench_refactor_csc`, `matrix=nos4.mtx`, `scenario=chol_spd`,
  `n=100`, `nnz=594`, `n/a` backend fields, timing context, speedup context,
  and residual context;
- generator metadata fields align with the Day 3 contract;
- no benchmark CSV schema change is required for Day 5;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 4, so `make format && make lint &&
  make test` is not required.
