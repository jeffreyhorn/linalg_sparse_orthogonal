# Day 8 Runtime Sentinel Design

## Purpose

Day 8 designs the Sprint 142 runtime/backend sentinel expansion. The design
uses the existing `make performance-sentinels` bundle and normalized report
index instead of creating a new report family. The goal is local regression
visibility for selected backend decisions without portable timing,
state-of-the-art, platform-parity, or backend-policy claims.

## Existing Sentinel Surface

| Surface | Maintained Command | Generated Artifact | Current Meaning | Day 8 Decision |
| --- | --- | --- | --- | --- |
| Wall-check sentinel | `make performance-sentinels` wrapping `make wall-check` | `build/bench-reports/sentinels/sentinels.tsv` rows with `sentinel_id=S5` | Thresholded local hard gate for existing reorder wall checks. | Keep as the only hard-gate sentinel row. |
| Cholesky CSC advisory sentinel | `make performance-sentinels` running `bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1` | `build/bench-reports/sentinels/sentinels.tsv` rows with `sentinel_id=S2` plus `bench_chol_csc_nos4.csv` | Threshold-free local Cholesky CSC backend/path visibility. | Keep as advisory and preserve backend/dense-kernel fields. |
| Normalized sentinel index | `python3 scripts/normalize_report_index.py --family sentinel` | `build/report-index/normalized-index.tsv` | Preserves hard-gate rows separately from advisory rows. | Reuse unchanged row-family semantics. |

## Selected Sentinel Rows

| Sentinel ID | Classification | Command | Fixture | Metrics | Backend Fields | Threshold | Freshness Policy | Non-Claims |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `S5` | Hard local gate | `make wall-check` through `make performance-sentinels` | `bcsstk14`, `Pres_Poisson` | `qg_amd_reorder_ms`, `amd_reorder_ms`, `nd_reorder_ms` | `n/a` | Existing `wall_check.sh` baselines and multipliers | `generated_compare_inputs` | No portable performance, throughput, state-of-the-art, or backend-policy closure claim. |
| `S2` | Advisory threshold-free row | `bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1` through `make performance-sentinels` | `nos4.mtx` | Cholesky factor/solve timings and speedup fields already emitted by `bench_chol_csc` | `backend_request`, `backend_selected`, `backend_fallback`, `dense_kernel`, `panel_solver` | None | `generated_local_advisory` | No hard performance gate, release benchmark proof, platform parity, or broad dense-backend claim. |
| `S3` | Advisory threshold-free row | `bench_refactor_csc --indefinite-kkt --repeat 1` through `make performance-sentinels` | generated `kkt-150` | LDLT analyze, refactor, solve, speedup, and residual fields from the CSV row | `ldlt_dense_backend_request`, `ldlt_dense_backend_selected`, `ldlt_dense_backend_fallback` mapped into existing sentinel backend columns | None | `generated_local_advisory` | No portable LDLT performance claim, no package/backend availability claim, no state-of-the-art claim, and no solver-correctness proof beyond the row-local residual context. |

## Explicit Deferrals

| Candidate | Day 8 Decision | Reason |
| --- | --- | --- |
| Eigensolver AUTO backend route sentinel | Defer | `bench_eigs_reuse` reports `backend_used`, but the current sentinel schema does not preserve eigensolver-specific convergence fields without either overloading backend columns or adding a wider generated row contract. |
| Shift-invert LDLT route snapshot | Defer | The available evidence is test-owned; adding a sentinel row would require a dedicated maintained report command before it can be normalized cleanly. |
| LDLT dense-helper requested/selected/fallback standalone row | Fold into `S3` only | A separate row would duplicate the `bench_refactor_csc --indefinite-kkt` backend fields without adding regression value. |
| OpenMP runtime sentinel | Defer | `build_mode` and `OMP_NUM_THREADS` are already report context; adding a runtime thread-policy row would imply library-owned thread control that Sprint 142 explicitly rejected. |
| Package/CMake/pkg-config rows | Defer to Sprint 143 | These are adoption/package proof surfaces, not runtime backend sentinel rows. |

## Row Semantics

`S5` remains the only hard gate. If any generated `sentinel_hard_gate` row
reports `fail`, `scripts/normalize_report_index.py --family sentinel
--check-freshness` should continue to report an error through the existing
freshness path.

`S2` and `S3` are advisory threshold-free rows. Their status should normalize
to `advisory` when the underlying benchmark row runs successfully, `skip` when
the binary or fixture is missing, and `unknown` only when a malformed or
unexpected generated row reaches the normalizer.

All selected rows must retain:

- `sentinel_id`
- `claim_boundary`
- `support_tier`
- `build_mode`
- `omp_num_threads`
- `matrix_or_fixture`
- `metric`
- `value`
- `baseline`
- `threshold`
- `backend_request`
- `backend_selected`
- `backend_fallback`

`S2` additionally retains `dense_kernel` and `panel_solver`. `S3` may report
those fields as `n/a` unless a future LDLT dense-kernel descriptor becomes a
maintained benchmark field.

## Report-Index Integration Plan

The existing `sentinel_generated_rows()` normalizer path is sufficient for the
selected rows because it already:

- separates `sentinel_hard_gate` rows from advisory rows by `claim_boundary`;
- maps `report` and `measurement` generated statuses to normalized
  `advisory`;
- preserves backend request, selected backend, and fallback in the normalized
  `configuration` field;
- loads branch, commit, platform, compiler, and generated timestamp from
  `build/bench-reports/sentinels/manifest.txt`;
- reports hard-gate failures as freshness errors while leaving advisory local
  measurements advisory.

Day 9 should therefore update only the generated `sentinels.tsv` producer and
the focused synthetic normalizer fixture if `S3` is implemented.

## Day 9 Implementation Notes

The planned Day 9 implementation should:

1. Extend `scripts/performance_sentinels.sh` to accept
   `bench_refactor_csc` as an additional maintained binary argument.
2. Update the `Makefile` `performance-sentinels` target dependency and command
   to pass `$(BUILDDIR)/bench_refactor_csc`.
3. Run `bench_refactor_csc --indefinite-kkt --repeat 1` into
   `build/bench-reports/sentinels/bench_refactor_csc_kkt.csv`.
4. Parse the single CSV data row into advisory `S3` rows for stable metrics
   such as `analyze_ms`, `refactor_public_ms`, `refactor_csc_ms`,
   `solve_public_ms`, `solve_csc_ms`, `speedup_refactor`, `res_public`, and
   `res_csc`.
5. Map the CSV backend fields into the existing sentinel columns:
   `backend_request`, `backend_selected`, and `backend_fallback`.
6. Emit `skip` rows when the binary is missing or the benchmark command fails.
7. Extend `tests/test_normalize_report_index.py` with a synthetic `S3` row so
   normalized output preserves the new advisory row and backend fields
   deterministically.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected sentinel rows are tied to maintained commands or fixtures. | Complete | `S5`, `S2`, and planned `S3` all use existing `make performance-sentinels` and maintained benchmark binaries. |
| Portable performance and platform claims remain explicit non-claims. | Complete | Each selected row is local-only and carries explicit non-claims. |
| Hard gates are separated from advisory rows. | Complete | `S5` keeps `local_wall_gate`; `S2` and `S3` use `local_threshold_free`. |
| Report-index integration is deterministic. | Complete | The design reuses existing `sentinel_generated_rows()` field mapping and adds only one synthetic fixture extension for Day 9. |
