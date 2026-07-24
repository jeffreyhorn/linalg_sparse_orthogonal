# Sprint 132 Day 11 - Focused Runtime Validation

## Purpose

Run the focused validation required for the Sprint 132 script/report metadata
changes and record reproducible evidence for closeout.

This validation covers touched script and generated-report surfaces only. It
does not run supplemental guardrails, broad benchmark sweeps, or full C quality
checks because no `.c` or `.h` files changed.

## Focused Validation Command Log

| Command | Result | Notes |
| --- | --- | --- |
| `bash -n scripts/performance_sentinels.sh` | Passed | Shell syntax check for the touched sentinel script. |
| `bash -n scripts/bench_canonical_report.sh` | Passed | Shell syntax check for the touched canonical report script. |
| `make performance-sentinels` | Passed | Regenerated `build/bench-reports/sentinels/` and preserved S5 hard wall-check behavior. |
| `make bench-canonical-report` | Passed | Regenerated `build/bench-reports/canonical/` with canonical CSVs, `index.tsv`, and manifest. |
| Sentinel TSV width check | Passed | Header has 20 tab-separated fields and 11 data rows; all rows match header width. |
| Canonical index width check | Passed | Header has 13 tab-separated fields and 4 data rows; all rows match header width. |
| Sentinel status/support-tier scan | Passed | S5 rows are `pass` and `reviewed_thresholded`; S2 rows are `report` and `reviewed_threshold_free`. |
| Manifest freshness scan | Passed | Sentinel and canonical manifests record `git_commit=d348b6ca` and `git_branch=sprint-132`. |

## Benchmark and Sentinel Report Results

| Report | Result | Evidence |
| --- | --- | --- |
| Sentinel S5 wall-check rows | Passed | Three S5 rows report `pass`, `reviewed_thresholded`, and `local_wall_gate`. |
| Sentinel S2 Cholesky CSC rows | Report-only | Eight S2 rows report `reviewed_threshold_free` and `local_threshold_free`. |
| Sentinel backend fields | Passed | S5 rows use `n/a`; S2 rows report selected backend `builtin`, dense kernel `builtin`, and panel solver `batched_panel`. |
| Canonical report rows | Passed | Four canonical artifact rows regenerated for `bench_refactor_csc`, `bench_chol_csc`, `bench_iterative_reuse`, and `bench_eigs_reuse`. |
| Canonical runtime context | Passed | Canonical index and manifest include platform, compiler, `build_mode=serial`, and `omp_num_threads=unset`. |

## Backend and Runtime Check Results

| Check | Result | Interpretation |
| --- | --- | --- |
| `SPARSE_CHOL_DENSE_BACKEND` context | Passed | Sentinel manifest records `sparse_chol_dense_backend=unset`; S2 rows preserve request `unset` and selected backend `builtin`. |
| `SPARSE_LDLT_DENSE_BACKEND` context | Passed | Sentinel manifest records `sparse_ldlt_dense_backend=unset`; no LDLT sentinel lane was added. |
| OpenMP build context | Passed | Sentinel and canonical outputs report `build_mode=serial`. |
| Thread-count context | Passed | Sentinel and canonical outputs report `omp_num_threads=unset`. |
| Backend non-claim boundary | Passed | Backend fields remain row context only; no backend parity or optional availability claim is emitted. |
| Threshold boundary | Passed | S5 remains hard-gated; S2 and canonical rows remain threshold-free. |

## Skipped-Check Rationale

| Skipped check | Rationale | Future owner |
| --- | --- | --- |
| `make format && make lint && make test` | No `.c` or `.h` files changed during Day 11 or the Day 8 metadata batch. | Required immediately if future C/header edits occur. |
| `make large-matrix-guardrails` | Guardrail scripts and docs were not changed or promoted in Day 8-11. Existing build artifact remains historical/stale from Sprint 131. | `large-matrix-guardrails` if Day 13 selects a refresh. |
| Supplemental large-matrix mode | Supplemental lanes remain opt-in and host-sensitive. | `large-matrix-guardrails`. |
| Broad `make bench` | Not required for script metadata changes and would expand runtime beyond the focused validation surface. | Benchmark owners when benchmark behavior changes. |
| OpenMP `make omp` | OpenMP behavior and build flags were not changed; metadata records serial mode for this run. | Runtime governance owner if OpenMP behavior changes. |
| LDLT/iterative/eigs/SVD focused binaries | These lanes were explicitly deferred or reused as existing canonical artifacts; no new lane was implemented. | Direct/backend, iterative, eigensolver, and SVD benchmark owners. |

## Blocker or Pass Summary

No blockers were found.

The touched script/report surfaces are reproducible on the current branch and
preserve Sprint 132 boundaries:

- S5 remains the only hard local timing gate.
- S2 remains threshold-free Cholesky CSC report context.
- Canonical reports remain threshold-free generated snapshots.
- Backend fields remain observability, not parity or availability claims.
- OpenMP/thread fields remain runtime context, not public thread-control API.

## Day 12 Handoff

Day 12 should publish the performance non-claim register using the validated
metadata and runtime evidence from Days 8-11. It should classify non-claims by
portable performance, backend parity, OpenMP/thread control, scalability,
memory, benchmark correctness, freshness, and supplemental promotion.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| All required checks pass before proceeding. | Complete | Script syntax checks, focused report targets, schema width checks, status/tier scans, and manifest freshness scans passed. |
| Skipped checks are justified by support tier or untouched surfaces. | Complete | Skipped-check rationale documents why full C quality, guardrails, supplemental mode, broad benchmarks, OpenMP, and deferred benchmark binaries were not run. |
| Validation evidence is reproducible for closeout. | Complete | Command log records exact commands and generated artifact paths for rerun. |
