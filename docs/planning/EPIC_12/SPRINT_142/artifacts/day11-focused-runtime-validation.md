# Day 11 Focused Runtime Validation

## Purpose

Day 11 validates the Sprint 142 runtime/backend governance changes before the
full Day 12 quality gate. The scope is focused: backend dispatch precedence,
typed-control behavior, dense-helper fallback, sentinel generation, normalized
report-index ingestion, freshness diagnostics, and generated-output hygiene.

## Focused Test Evidence

| Command | Result | Evidence |
| --- | --- | --- |
| `make build/test_chol_csc build/test_ldlt_backend_dispatch build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_reorder_nd build/test_ldlt` | Passed | Focused backend/precedence binaries built; only `build/test_ldlt` required a rebuild. |
| `./build/test_chol_csc` | Passed | 92 tests passed; Cholesky backend AUTO, forced, invalid, CSC path, and dense-reference evidence remained intact. |
| `./build/test_ldlt_backend_dispatch` | Passed | 22 tests passed; LDLT dispatch and adjacent eigensolver backend selection coverage remained intact. |
| `./build/test_eigs_thick_restart` | Passed | 23 tests passed; thick-restart backend, AUTO dispatch, and parity coverage remained intact. |
| `./build/test_eigs_lobpcg` | Passed | 29 tests passed; LOBPCG, preconditioner, AUTO dispatch, and explicit override coverage remained intact. |
| `./build/test_reorder_nd` | Passed | 35 tests passed, 1 known skip; analysis typed-vs-env precedence and reorder policy coverage remained intact. |
| `./build/test_ldlt` | Passed | 89 tests passed; LDLT dense-helper default, explicit, external, and invalid-env fallback coverage remained intact. |

## Sentinel And Report-Index Evidence

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed | Synthetic sentinel S3 row parsing and normalized backend request/selected/fallback preservation passed. |
| `bash -n scripts/performance_sentinels.sh` | Passed | Shell syntax remained valid after Day 9 script changes. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus/report schema validation passed after Day 10 schema wording update. |
| `make performance-sentinels` | Passed | Generated `sentinels.tsv`, `manifest.txt`, `wall_check.txt`, `bench_chol_csc_nos4.csv`, and `bench_refactor_csc_kkt.csv`. |
| `python3 scripts/normalize_report_index.py --family sentinel --output build/report-index/normalized-index.tsv` | Passed | Wrote 21 normalized sentinel rows. |
| `python3 scripts/normalize_report_index.py --family sentinel --check-freshness` | Passed | `S2` and `S3` advisory rows were fresh; `S5` hard-gate rows stayed distinguishable. |
| `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --family guardrail --check-freshness` | Passed | 25 rows checked; missing benchmark/guardrail generated rows stayed advisory/warning without turning into pass evidence. |

## Generated-Output Hygiene

Generated benchmark and report-index files stayed under ignored `build/`
paths. `git ls-files --others --exclude-standard` initially showed one Python
cache file from schema validation:

- `scripts/__pycache__/validate_corpus_schema.cpython-314.pyc`

The cache was removed. After cleanup, the only untracked files were the
source-controlled Sprint 142 planning artifacts.

## Scoped Repairs

No runtime/backend behavior repairs were needed on Day 11. The only cleanup was
removing the generated Python cache file described above.

## Remaining Issues And Owners

| Issue | Owner | Stop Condition |
| --- | --- | --- |
| Full repository quality gate has not yet run after the Sprint 142 script/docs/test batch. | Day 12 full quality gate | Run the planned full gate and repair any scoped failures before sprint closeout. |
| Benchmark and guardrail generated report families were not regenerated in the combined freshness check. | Report/benchmark maintainer | Advisory/warning rows are acceptable because Day 11 changed sentinel rows only; regenerate those families only when their evidence is needed. |
| Sentinel `S5` hard-gate freshness remains `generated_present_unchecked` rather than strict fresh in diagnostics. | Report-index owner | Existing normalizer semantics treat hard-gate rows through generated-compare-input policy and keep failures hard; no Day 11 repair needed unless a future sprint tightens strict comparison semantics. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Focused tests pass or failures are repaired before broad validation. | Complete | All focused C tests and Python checks passed. |
| Generated outputs remain ignored unless intentionally source-controlled. | Complete | Generated `build/` outputs were ignored; Python cache was removed. |
| Any remaining issue has an owner and stop condition. | Complete | Remaining items are assigned above for Day 12 or future report-index ownership. |
