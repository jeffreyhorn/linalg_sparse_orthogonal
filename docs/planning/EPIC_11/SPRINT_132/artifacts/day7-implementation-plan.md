# Sprint 132 Day 7 - Sentinel Implementation Plan

## Purpose

Choose the Sprint 132 sentinel/report implementation batch and define the
exact edit scope, validation commands, rollback criteria, and deferral rules
for Day 8.

This plan selects a low-churn metadata implementation path. It does not add a
new hard timing threshold and does not require benchmark C or public API
changes.

## Selected Implementation Batch

| Batch item | Scope | Reason selected | Claim boundary |
| --- | --- | --- | --- |
| S6 sentinel metadata completeness | Add structured metadata columns to `performance-sentinels` rows for support tier, claim boundary, backend request/selection/fallback, dense kernel, panel solver, and artifact where available. | `performance-sentinels` is the highest-value local report surface and already records most runtime context. | Schema and interpretability improvement only; S5 remains the only hard timing gate. |
| C1 canonical report runtime context | Add platform, compiler, build mode, and `OMP_NUM_THREADS` to canonical manifest and `index.tsv`. | Canonical reports already index artifacts but lack host/build context needed for safe local comparisons. | Threshold-free report metadata only; no pass/fail timing status. |
| Documentation alignment | Update benchmark and maintainer docs to describe the widened generated metadata narrowly. | Script output schema changes need nearby interpretation rules. | Local report evidence, not portable performance, backend parity, or OpenMP speedup. |

## Explicit Non-Selection

| Candidate | Decision | Reason |
| --- | --- | --- |
| L1 LDLT backend/runtime report-only lane | Defer code/script integration; rely on existing `bench_refactor_csc` CSV fields for now. | The existing benchmark already emits `ldlt_dense_backend_request`, `ldlt_dense_backend_selected`, and `ldlt_dense_backend_fallback`; adding a new generated lane would increase runtime and schema churn. |
| I1 iterative convergence/BiCGSTAB | Design-only. | Fixture, tolerance, metric, runtime, and variance policy remain unresolved. |
| E1 eigensolver backend slice | Design-only. | Narrow backend/preconditioner slice and OpenMP policy remain unresolved. |
| V1 SVD/bidiag | Design-only. | Bounded fixture and metric semantics remain unresolved. |
| G5 supplemental large-matrix recurrence | Keep supplemental and opt-in. | Host sensitivity and recurring runtime policy remain unresolved. |
| New hard backend timing threshold | Do not implement. | Backend availability, fallback, build mode, thread context, and host class are not baseline-stabilized. |

## Touched-File Forecast

| File | Change type | Required check |
| --- | --- | --- |
| `scripts/performance_sentinels.sh` | Script schema change. Add TSV columns and structured values for report family, support tier, claim boundary, artifact, backend request/selection/fallback, dense kernel, and panel solver. | `bash -n scripts/performance_sentinels.sh`; `make performance-sentinels`; inspect generated TSV and manifest. |
| `scripts/bench_canonical_report.sh` | Script metadata change. Add platform, compiler, build mode, and `OMP_NUM_THREADS` to generated index and manifest. | `bash -n scripts/bench_canonical_report.sh`; `make bench-canonical-report`; inspect generated index and manifest. |
| `benchmarks/README.md` | Documentation update for generated report schema and interpretation. | Docs hygiene; command-output references match generated fields. |
| `docs/maintainer_guide.md` | Maintainer policy update for Sprint 132 metadata boundaries. | Docs hygiene; non-claim wording preserved. |
| `docs/planning/EPIC_11/SPRINT_132/WORKING_NOTES.md` | Sprint notes update. | `git diff --check`; focused whitespace scan. |
| `docs/planning/EPIC_11/SPRINT_132/artifacts/day8-*` | Day 8 implementation artifact after work lands. | `git diff --check`; focused whitespace scan. |

No `.c` or `.h` changes are planned. If Day 8 discovers that benchmark source
changes are required, stop and re-scope before editing C files.

## Day 8 Edit Checklist

1. Re-read `scripts/performance_sentinels.sh` and current generated TSV header.
2. Change `performance_sentinels.sh` so every row includes:
   - `report_family`
   - `support_tier`
   - `claim_boundary`
   - `artifact`
   - `backend_request`
   - `backend_selected`
   - `backend_fallback`
   - `dense_kernel`
   - `panel_solver`
3. Keep S5 rows backend fields as `n/a` unless a future backend-aware wall
   lane exists.
4. Keep S2 rows threshold-free with Cholesky dense-kernel and panel-solver
   values parsed from `bench_chol_csc`.
5. Preserve existing S5 pass/fail behavior and final exit status.
6. Change `bench_canonical_report.sh` so `index.tsv` and manifest include:
   - `platform`
   - `compiler`
   - `build_mode`
   - `omp_num_threads`
7. Keep canonical report rows threshold-free with no pass/fail timing status.
8. Update `benchmarks/README.md` for the new generated metadata fields.
9. Update `docs/maintainer_guide.md` to keep backend/runtime interpretation
   aligned with the script output.
10. Run validation commands in order and inspect generated artifacts.

## Validation Command Plan

| Step | Command | Required result |
| --- | --- | --- |
| Script syntax | `bash -n scripts/performance_sentinels.sh` | Exit 0. |
| Script syntax | `bash -n scripts/bench_canonical_report.sh` | Exit 0. |
| Sentinel generation | `make performance-sentinels` | Exit 0 or stop if S5 hard wall-check fails. Generated TSV includes the new metadata columns. |
| Canonical generation | `make bench-canonical-report` | Exit 0. Generated index and manifest include platform/compiler/build/thread context. |
| Artifact inspection | `sed -n '1,5p' build/bench-reports/sentinels/sentinels.tsv` and `sed -n '1,8p' build/bench-reports/canonical/index.tsv` | Headers and representative rows match the planned schema. |
| Docs hygiene | `git diff --check` | Exit 0. |
| Sprint markdown whitespace | `if rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_132; then exit 1; fi` | Exit 0. |

If Day 8 unexpectedly changes any `.c` or `.h` file, the validation plan
expands to include:

```sh
make format && make lint && make test
```

## Rollback Criteria

Rollback or re-scope the script changes if any of these occur:

- `make performance-sentinels` becomes flaky or substantially exceeds the
  planned local runtime budget.
- S5 no longer exits nonzero when the existing wall-check gate fails.
- S2 threshold-free rows are accidentally converted into pass/fail timing
  rows.
- TSV column order changes in a way that breaks simple downstream parsing
  without a corresponding documentation update.
- Backend fields require new C benchmark output instead of parsing existing
  columns or using `n/a`.
- Canonical reports start reporting pass/fail timing status.
- Generated metadata implies backend availability, backend parity, OpenMP
  speedup, or portable timing.

## Deferral Criteria

Defer a candidate instead of implementing it when:

- backend request, selection, fallback, or dense-kernel state would be
  `unknown` for the proposed row
- OpenMP build mode or `OMP_NUM_THREADS` cannot be captured without brittle
  detection
- the lane requires broad benchmark CSV churn
- the lane needs a new fixture, tolerance, baseline, or variance policy
- the lane requires optional backend availability on the local host
- validation would require full benchmark sweeps rather than focused report
  commands
- docs cannot state a narrow local-only claim boundary

## Future Owners for Deferred Lanes

| Deferred lane | Blocker | Dependency | Future owner |
| --- | --- | --- | --- |
| L1 recurring LDLT backend sentinel | Needs decision on whether an extra generated row is worth runtime and schema cost. | Existing `bench_refactor_csc` KKT backend fields; optional future sentinel schema. | Direct/backend benchmark owner. |
| I1 iterative convergence/BiCGSTAB | Needs stable fixture, tolerance, metric, and variance policy. | Iterative benchmark owner selects bounded command. | Iterative benchmark owner. |
| E1 eigensolver backend slice | Needs narrow backend/preconditioner slice and OpenMP policy. | Eigensolver benchmark owner measures focused runtime. | Eigensolver benchmark owner. |
| V1 SVD/bidiag | Needs bounded fixture, rank/tolerance, and metric semantics. | SVD benchmark owner maps Sprint 130 correctness evidence to a report-only metric. | SVD benchmark owner. |
| G5 supplemental large-matrix recurrence | Needs cadence, runtime budget, host-sensitivity policy, and owner. | Guardrail owner decides whether supplemental lanes become reviewed. | `large-matrix-guardrails`. |
| New hard backend timing gate | Needs accepted baseline by backend/runtime state and host class. | Repeated local baseline collection and variance policy. | Runtime governance owner and benchmark owner. |

## Day 8 Handoff

Day 8 should implement only the selected script/docs metadata batch unless a
new blocker appears. The expected implementation sequence is:

1. Update `performance_sentinels.sh`.
2. Update `bench_canonical_report.sh`.
3. Run focused script syntax checks.
4. Run `make performance-sentinels` and inspect sentinel artifacts.
5. Run `make bench-canonical-report` and inspect canonical artifacts.
6. Update docs with the final generated field names.
7. Re-run docs hygiene.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Implementation scope is small enough to validate in Sprint 132. | Complete | Selected batch is limited to two report scripts plus docs, with no planned C/header edits. |
| Every potential code/script/docs touch point has a required check. | Complete | Touched-file forecast and validation command plan list required checks per surface. |
| Deferred lanes have blocker, dependency, and future owner. | Complete | Future-owner table records blockers, dependencies, and owners for L1, I1, E1, V1, G5, and new hard backend gates. |
