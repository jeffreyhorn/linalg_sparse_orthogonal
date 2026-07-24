# Sprint 132 Day 5 - Backend Metadata Design

## Purpose

Translate the Day 4 backend/runtime contract into report-family metadata rules
that can guide later Sprint 132 implementation without changing benchmark
claims.

This design is intentionally report-facing. It defines which fields should be
required, optional, deferred, or omitted for canonical reports, performance
sentinels, large-matrix guardrails, and benchmark-local evidence.

## Metadata Design Principles

- Preserve request, selection, and fallback as separate concepts.
- Record OpenMP as build/runtime context, not as library-owned runtime policy.
- Treat missing backend metadata as `unknown`, not as builtin.
- Treat absent backend seams as `n/a`, not as missing data.
- Keep supplemental, reviewed, thresholded, threshold-free, generated,
  experimental, and deferred evidence distinguishable.
- Gate hard thresholds only when baseline, threshold, fixture, command,
  backend state, build mode, thread context, and host class are all explicit.
- Do not add fields that imply backend parity, optional backend portability,
  OpenMP speedup, memory portability, or broad benchmark governance.

## Proposed Common Metadata Fields

| Field | Type | Status | Applies to | Semantics |
| --- | --- | --- | --- | --- |
| `report_family` | string | Required | indexed reports | One of `canonical`, `sentinel`, `large_matrix_guardrail`, or `benchmark_local`. |
| `lane_id` | string | Required where indexed | sentinels, guardrails | Stable row identifier such as `S2`, `S5`, `G1`, `S1`, or future lane id. |
| `status` | string | Required where indexed | sentinels, guardrails | `pass`, `fail`, `report`, `skip`, or `error`; `report` means threshold-free evidence. |
| `support_tier` | string | Required for promoted rows | all promoted report families | `reviewed`, `supplemental`, `experimental`, `generated`, or `deferred`. |
| `command` | string | Required | all report families | Exact command or benchmark invocation that produced the row or artifact. |
| `artifact` | string | Required where indexed | canonical, sentinels, guardrails | File that contains the row or raw output. |
| `relative_path` | string | Required where indexed | canonical, sentinels, guardrails | Path relative to the report directory. |
| `generated_at_utc` | timestamp string | Required for generated artifacts | canonical, sentinels, guardrails | Generation time for freshness checks. |
| `git_commit` | string | Required when available | canonical, sentinels, guardrails | Commit that produced the artifact, or `unknown`. |
| `git_branch` | string | Required when available | canonical, sentinels, guardrails | Branch that produced the artifact, `detached`, or `unknown`. |
| `platform` | string | Required for runtime reports | sentinels, guardrails; deferred for canonical index | Host context; local-comparison evidence only. |
| `compiler` | string | Required for runtime reports | sentinels, guardrails; deferred for canonical index | Compiler string used for local interpretation. |
| `build_mode` | string | Required for runtime reports | sentinels; deferred for canonical and guardrails | `serial`, `openmp`, or explicit `unknown`. |
| `omp_num_threads` | string | Required for runtime reports | sentinels; deferred for canonical and guardrails | Raw `OMP_NUM_THREADS` value or `unset`; not a library API field. |
| `matrix_or_fixture` | string | Required for metric rows | sentinels, benchmark-local rows | Fixture name, corpus row, or `n/a` for non-fixture checks. |
| `metric` | string | Required for metric rows | sentinels, benchmark-local rows | Metric name such as `factor_csc_sn_ms` or `qg_amd_reorder_ms`. |
| `value` | string or number | Required for metric rows | sentinels, benchmark-local rows | Observed local value or `n/a`. |
| `baseline` | string or number | Required only for thresholded rows | sentinels | Baseline used for pass/fail timing comparison. |
| `threshold` | string or number | Required only for thresholded rows | sentinels | Threshold multiplier or bound. |
| `backend_request` | string | Required where backend-owned | direct/backend rows; deferred in sentinel TSV | Normalized request value, `unset`, or `n/a`. |
| `backend_selected` | string | Required where backend-owned | direct/backend rows; deferred in sentinel TSV | Selected backend, `builtin`, optional backend name, `unknown`, or `n/a`. |
| `backend_fallback` | string | Required where backend-owned | LDLT rows; deferred in sentinel TSV | `yes`, `no`, `n/a`, or `unknown`. |
| `dense_kernel` | string | Required for Cholesky CSC rows | Cholesky CSC benchmark and S2 sentinel note | Active Cholesky dense-kernel descriptor. |
| `panel_solver` | string | Required for Cholesky CSC rows | Cholesky CSC benchmark and S2 sentinel note | Active supernodal panel-solver descriptor. |
| `claim_boundary` | string | Required for curated/indexed design docs; deferred in generated TSVs | docs, report indexes | Short non-claim label such as `local_threshold_free` or `local_wall_gate`. |
| `freshness` | string | Required by report indexes that aggregate generated artifacts | report indexes | `fresh`, `stale`, `missing`, `regenerated`, or `unknown`. |

## Report-Family Field Matrix

| Report family | Current fields | Required now | Optional now | Deferred | Intentionally omitted |
| --- | --- | --- | --- | --- | --- |
| `performance-sentinels` TSV | `sentinel_id`, `status`, `command`, `build_mode`, `omp_num_threads`, `matrix_or_fixture`, `metric`, `value`, `baseline`, `threshold`, `notes` | Keep all current TSV fields. Interpret S5 as thresholded and S2 as threshold-free. Preserve backend details in `notes` until a later script change adds structured backend columns. | `dense_kernel`, `panel_solver`, `chol_env`, `ldlt_env` in S2 notes. | Structured `backend_request`, `backend_selected`, `backend_fallback`, `support_tier`, `claim_boundary`, and `artifact` TSV columns. | Portable timing verdicts, backend parity labels, OpenMP speedup labels. |
| `performance-sentinels` manifest | `generated_at_utc`, `report_dir`, `git_commit`, `git_branch`, `platform`, `compiler`, `build_mode`, `omp_num_threads`, backend env vars, commands, artifacts, notes | Keep all current provenance and runtime fields. | Backend env vars remain request context, not selected-backend proof. | Explicit support-tier and claim-boundary manifest keys. | Host-portable performance or backend-availability claims. |
| `bench-canonical-report` index | `surface`, `category`, `report_label`, `generated_at_utc`, `git_commit`, `git_branch`, `artifact`, `relative_path`, `command` | Keep current fields and require artifact-level command provenance. | Report label remains optional user context. | `platform`, `compiler`, `build_mode`, `omp_num_threads`, support tier, and claim-boundary fields if canonical metadata is expanded. | Hard thresholds and pass/fail timing status. |
| `bench-canonical-report` CSV artifacts | Benchmark-specific CSV columns | Preserve benchmark-owned backend fields already emitted by `bench_refactor_csc` and `bench_chol_csc`. | Benchmark-local comments or headers may explain row semantics. | Shared CSV-side runtime fields across all canonical benchmarks. | Cross-benchmark backend fields for rows without backend seams. |
| `large-matrix-guardrails` index | `lane_id`, `status`, `category`, `command`, `artifact`, `notes` | Keep reviewed versus supplemental category split and explicit skip rows. | Notes may continue to carry local interpretation details. | `generated_at_utc`, `git_commit`, `git_branch`, `platform`, `compiler`, `support_tier`, and `claim_boundary` in index rows if report indexing is widened. | Dense backend fields; this family does not own dense backend seams. |
| `large-matrix-guardrails` manifest | `generated_at_utc`, `report_dir`, `git_commit`, `git_branch`, `platform`, `compiler`, `supplemental`, lane lists, artifacts, notes | Keep provenance, compiler, platform, and supplemental flag. | Supplemental lane notes for max-RSS and threshold-free timing. | `build_mode` and `omp_num_threads` only if future guardrail runs become OpenMP-sensitive. | Backend request/selection/fallback fields. |
| `benchmark-local` runs | Binary-specific stdout and optional saved CSV | Require command, fixture, repeat count, backend context when backend-sensitive, build mode when OpenMP-sensitive, and local note provenance in sprint artifacts. | Raw output files may be referenced from artifacts instead of normalized. | Promotion into generated indexes. | Any pass/fail timing claim without accepted baseline and variance policy. |

## Backend Row Semantics

| Row situation | `backend_request` | `backend_selected` | `backend_fallback` | Row interpretation |
| --- | --- | --- | --- | --- |
| No backend environment variable is set and builtin is selected. | `unset` | `builtin` | `no` or `n/a` | Default self-contained path was used locally. |
| `SPARSE_*_DENSE_BACKEND=builtin`. | `builtin` | `builtin` | `no` | Explicit builtin request was honored locally. |
| Optional backend requested and selected. | optional backend name | optional backend name | `no` | Optional path was selected for this bounded lane only. |
| Optional backend requested but builtin selected. | optional backend name | `builtin` | `yes` | Local fallback occurred; do not infer optional backend correctness or availability. |
| Invalid backend request falls back. | invalid normalized value | `builtin` | `yes` | Fallback truthfulness row; not a public tuning guarantee for invalid values. |
| Optional backend cannot be probed or linked. | optional backend name | `builtin`, `unavailable`, or `unknown` | `yes` or `unknown` | Report as unavailable/fallback context; skip only if the lane explicitly requires the optional backend. |
| Metadata field is missing. | `unknown` | `unknown` | `unknown` | Row is not backend-comparable and cannot support backend-specific thresholds. |
| Path has no backend seam. | `n/a` | `n/a` | `n/a` | Backend metadata is intentionally not applicable. |

## OpenMP and Thread Row Semantics

| Row situation | `build_mode` | `omp_num_threads` | Row interpretation |
| --- | --- | --- | --- |
| Serial/default build. | `serial` | `unset` or raw environment value | Default product path; OpenMP runtime is not active because the binary is serial. |
| OpenMP-linked build. | `openmp` | raw `OMP_NUM_THREADS` value or `unset` | Runtime context only; does not claim speedup or a library thread policy. |
| Build mode cannot be detected. | `unknown` | raw `OMP_NUM_THREADS` value or `unset` | Row is not suitable for OpenMP-sensitive comparisons. |
| Caller uses nested parallelism outside the library. | `openmp` or `serial` | raw environment value | Out-of-scope unless the command and artifact document caller-owned nested runtime setup. |

## Implementation Touch Points

| Touch point | Likely change | Validation expectation | Notes |
| --- | --- | --- | --- |
| `scripts/performance_sentinels.sh` | Add structured backend and support-tier columns, or keep notes-only while documenting the limitation. | `bash -n scripts/performance_sentinels.sh`; focused `make performance-sentinels`; generated TSV and manifest inspection. | Highest-value implementation target because it owns local sentinel rows. |
| `scripts/bench_canonical_report.sh` | Add platform/compiler/build/runtime context to `index.tsv` and manifest only if canonical comparison needs it. | `bash -n scripts/bench_canonical_report.sh`; focused `make bench-canonical-report`; inspect generated index and manifest. | Should remain threshold-free. |
| `scripts/large_matrix_guardrails.sh` | Add support-tier/claim-boundary fields to index, or leave dense backend fields omitted by design. | `bash -n scripts/large_matrix_guardrails.sh`; `make large-matrix-guardrails`; inspect index and manifest. | Do not add backend columns unless a future backend-aware guardrail exists. |
| `benchmarks/bench_refactor_csc.c` | Keep LDLT backend fields stable; avoid widening rows without a new schema bump. | If changed, focused benchmark plus `make format && make lint && make test`. | Current CSV already owns request/selected/fallback for LDLT KKT mode. |
| `benchmarks/bench_chol_csc.c` | Keep Cholesky dense-kernel and panel-solver fields stable. | If changed, focused benchmark plus `make format && make lint && make test`. | Current CSV already owns dense-kernel and panel-solver descriptors. |
| `benchmarks/README.md` | Document field semantics if generated output schemas change. | Docs hygiene; full C quality only if source files also change. | Keep non-claim wording close to command descriptions. |
| `docs/maintainer_guide.md` | Update maintainer policy when metadata implementation lands. | Docs hygiene. | Keep API/runtime boundaries explicit. |

## Metadata Deferral Queue

| Deferred item | Reason | Promotion condition |
| --- | --- | --- |
| Structured sentinel backend columns | Current S2 notes already preserve Cholesky env/dense-kernel/panel context; schema widening should be batched with tests and docs. | Add columns when Day 7-8 implementation selects sentinel metadata as the touched surface. |
| Canonical platform/compiler/build fields | Canonical reports are threshold-free snapshots and currently emphasize artifact identity. | Add when canonical comparisons need host/build stratification in generated indexes. |
| Guardrail `build_mode` and `omp_num_threads` | Current reviewed guardrails are structural and CSV-shape focused; OpenMP is not the central claim. | Add only if guardrail lanes become OpenMP-sensitive. |
| Cross-family `claim_boundary` generated column | Useful, but touches multiple report scripts and docs. | Add when there is a single report-index consumer that reads it. |
| Shared report schema validator | Would reduce drift, but adds implementation and maintenance cost. | Add after at least two report families adopt the same structured metadata columns. |
| Backend availability probe row | Could clarify optional backend state, but may imply a capability contract if not bounded. | Add only with explicit supported/unavailable semantics and no portability claim. |
| Benchmark-local normalized metadata header | Helpful for ad hoc runs, but broad benchmark changes are high churn. | Add only for selected promoted benchmark-local lanes. |

## Day 6 Handoff

Day 6 should use this metadata design to choose sentinel candidates that are
cheap, bounded, and interpretable:

- keep S5 as the only current hard local timing gate
- keep S2 Cholesky CSC rows threshold-free unless a baseline is tied to exact
  runtime state
- prefer metadata completeness before adding new backend-sensitive thresholds
- reject sentinel candidates whose backend, OpenMP, fixture, repeat, support
  tier, or claim boundary is still `unknown`
- keep supplemental large-matrix lanes separate from reviewed recurring gates

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Metadata fields trace to Day 4 contract decisions. | Complete | Common fields and report-family matrix map directly to Day 4 request/selection/fallback, OpenMP, provenance, support-tier, and non-claim rules. |
| No metadata field creates a backend parity or portable timing claim. | Complete | Design principles, row semantics, and intentionally omitted fields preserve local-only interpretation. |
| Implementation touch points and blockers are explicit. | Complete | Touch-point table and deferral queue identify script, benchmark, and documentation owners plus validation expectations. |
