# Day 9 Sentinel Implementation

## Purpose

Day 9 implements the Day 8 runtime/backend sentinel expansion by adding an
advisory LDLT KKT sentinel row family to the existing `make
performance-sentinels` bundle. The implementation preserves the existing
hard-gate/advisory split and keeps all timing rows local-only.

## Implementation Summary

| Area | Change |
| --- | --- |
| `scripts/performance_sentinels.sh` | Added `bench_refactor_csc` as a maintained binary input and emits `S3` advisory rows from `bench_refactor_csc --indefinite-kkt --repeat 1`. |
| `Makefile` | Added `$(BUILDDIR)/bench_refactor_csc` to the `performance-sentinels` target dependencies and command arguments. |
| `tests/test_normalize_report_index.py` | Added a synthetic `S3` sentinel fixture and assertions that normalized output preserves advisory status plus backend request/selected/fallback configuration. |
| `benchmarks/README.md` and `docs/maintainer_guide.md` | Documented `S3`, `bench_refactor_csc_kkt.csv`, generated-output paths, and the local-only non-claim boundary. |

The sentinel manifest now records the full source commit SHA so freshness
checks compare generated sentinel rows against the same commit identity used by
`scripts/normalize_report_index.py`.

## Generated Rows

`S3` rows are emitted into:

- `build/bench-reports/sentinels/sentinels.tsv`

The raw source artifact is:

- `build/bench-reports/sentinels/bench_refactor_csc_kkt.csv`

The source command is:

```sh
bench_refactor_csc --indefinite-kkt --repeat 1
```

The generated `S3` metrics are threshold-free advisory rows:

- `analyze_ms`
- `refactor_public_ms`
- `refactor_csc_ms`
- `solve_public_ms`
- `solve_csc_ms`
- `speedup_refactor`
- `res_public`
- `res_csc`

Each `S3` row maps the benchmark CSV backend columns into the existing
sentinel schema:

- `ldlt_dense_backend_request` -> `backend_request`
- `ldlt_dense_backend_selected` -> `backend_selected`
- `ldlt_dense_backend_fallback` -> `backend_fallback`

## Preserved Semantics

`S5` remains the only hard local gate. `S2` and `S3` remain threshold-free
advisory rows. Missing binaries, fixtures, or failed advisory benchmark
commands emit `skip` rows where practical, and skips are not passes.

No normalized report-family schema was added. The existing sentinel normalizer
continues to separate hard-gate rows from advisory rows by `claim_boundary` and
preserves backend request/selected/fallback details in `configuration`.

## Generated-Output Policy

Generated sentinel outputs remain local artifacts under ignored `build/`
paths. Maintainers should regenerate them with:

```sh
make performance-sentinels
python3 scripts/normalize_report_index.py --family sentinel \
  --output build/report-index/normalized-index.tsv
python3 scripts/normalize_report_index.py --family sentinel --check-freshness
```

Do not commit generated `build/bench-reports/sentinels/` or
`build/report-index/` outputs unless a future sprint explicitly promotes a
specific artifact.

## Non-Claims

The new `S3` rows do not claim:

- portable LDLT performance;
- package, BLAS, Accelerate, or backend availability;
- state-of-the-art status;
- platform parity;
- broad solver correctness beyond the row-local residual context.

## Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed | Covers synthetic `S3` normalized advisory rows and backend field preservation. |
| `make build/bench_chol_csc build/bench_refactor_csc build/bench_amd_qg build/bench_reorder` | Passed | Built all binaries consumed by `make performance-sentinels`. |
| `make performance-sentinels` | Passed | Generated `sentinels.tsv`, `manifest.txt`, `wall_check.txt`, `bench_chol_csc_nos4.csv`, and `bench_refactor_csc_kkt.csv`. |
| `python3 scripts/normalize_report_index.py --family sentinel --output build/report-index/normalized-index.tsv` | Passed | Wrote 21 normalized sentinel rows including `S3`. |
| `python3 scripts/normalize_report_index.py --family sentinel --check-freshness` | Passed | Fresh advisory `S2`/`S3` rows and existing `S5` hard-gate rows remained distinguishable. |
| `make format && make lint` | Passed | Full format/lint gate completed after script/test changes. |
| `bash -n scripts/performance_sentinels.sh` | Passed | Shell syntax remained valid after final documentation alignment. |
| `git diff --check` | Passed | No whitespace errors. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| New sentinel rows appear deterministically in normalized output. | Complete | Synthetic `S3` fixture and assertions added to `tests/test_normalize_report_index.py`. |
| Hard gates and advisory rows are distinguishable. | Complete | `S5` remains `local_wall_gate`; `S2` and `S3` use `local_threshold_free`. |
| Local measurements are not described as portable performance evidence. | Complete | Script comments, benchmark docs, and this artifact preserve local-only wording. |
