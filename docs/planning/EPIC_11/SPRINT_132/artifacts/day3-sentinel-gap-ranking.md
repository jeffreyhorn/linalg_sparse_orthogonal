# Sprint 132 Day 3 - Sentinel Coverage Gap Ranking

## Purpose

Day 3 ranks missing or weak sentinel coverage by runtime risk, user-facing
workflow value, backend sensitivity, OpenMP sensitivity, corpus availability,
report metadata readiness, and claim impact.

This is a documentation-only ranking artifact. It does not change benchmark
code, sentinel scripts, report scripts, Makefile targets, generated artifacts,
maintainer wording, or public performance claims.

## Runtime-Risk Rubric

| Rank | Criteria | Suitable evidence shape | Threshold posture |
| --- | --- | --- | --- |
| High | Public workflow, recurring maintainer interest, existing benchmark owner, bounded fixture, stable runtime cost, and high regression or backend sensitivity. | Local sentinel or generated report row with manifest context. | Hard threshold only with accepted baseline and variance policy; otherwise threshold-free report. |
| Medium | Important implementation path but broad runtime variance, partial metadata, expensive fixture, optional backend, or unclear recurring cost. | Threshold-free report, canonical snapshot, or design-only sentinel candidate. | Avoid hard threshold until Day 4-6 contract and runtime budget are explicit. |
| Low | Exploratory benchmark, broad sweep, high flake cost, no stable fixture, or low user-facing claim impact. | Benchmark-local or supplemental report only. | No hard threshold in Sprint 132. |

Risk factors:

- `user_workflow`: direct solve/refactor, backend dispatch, iterative reuse,
  eigensolver reuse, SVD workflow, reorder/guardrail workflow.
- `runtime_cost`: bounded local command, expensive full sweep, or unknown.
- `regression_risk`: likelihood that a local slowdown would affect maintainers.
- `backend_sensitivity`: dense backend request/selected/fallback, solver
  backend dispatch, or optional backend availability.
- `openmp_sensitivity`: build mode, `OMP_NUM_THREADS`, nested runtime, or
  runtime-owned scheduling behavior.
- `corpus_availability`: checked-in small fixture, checked-in expensive
  fixture, generated input, optional data, or broad corpus sweep.
- `metadata_readiness`: whether command, backend, build mode, thread count,
  fixture, metric, freshness, and support tier can be recorded.
- `claim_impact`: risk that a row could be misread as portable performance,
  backend parity, scalability, memory, or correctness evidence.

## Sentinel Gap Ranking

| Rank | Gap | Current coverage | Risk basis | Recommended Sprint 132 posture | Owner |
| --- | --- | --- | --- | --- | --- |
| 1 | Backend/runtime observability contract | Partial metadata in canonical reports and sentinel manifests | Cross-cuts all sentinel/report rows; backend and OpenMP context are needed before safe implementation. | Design first in Day 4-5; no hard timing threshold. | Runtime governance owner and `report-index-owner`. |
| 2 | Canonical backend metadata completeness | `bench-canonical-report` has manifest and per-artifact rows; backend fields live unevenly inside CSVs | High report-index value; metadata affects direct/backend and canonical comparison interpretation. | Design metadata field matrix; implement only if fields can be generated without claim drift. | `benchmark-report-owner` and `report-index-owner`. |
| 3 | LDLT CSC backend/runtime sentinel | `bench_ldlt_csc` is benchmark-local; `bench_refactor_csc` has partial LDLT backend fields | User-facing direct symmetric-indefinite workflow and backend fallback interpretation; no compact sentinel row yet. | Candidate threshold-free sentinel/report lane after backend contract. | Direct/backend benchmark owner. |
| 4 | Iterative convergence and BiCGSTAB sentinel | `bench_convergence` and `bench_bicgstab` only under broad `make bench` | Iterative workflow is user-facing, but convergence timing can be fixture- and tolerance-sensitive. | Candidate threshold-free or soft-local sentinel after stable fixture and runtime budget. | Iterative benchmark owner. |
| 5 | Eigensolver backend sweep sentinel | `bench_eigs` broad backend sweep; `bench_eigs_reuse` canonical reuse report | Backend-rich but potentially broad/slow; OpenMP and preconditioner sensitivity can be high. | Report-only or narrow design candidate; avoid full-sweep threshold. | Eigensolver benchmark owner. |
| 6 | SVD/bidiag sentinel | `bench_svd` only under broad `make bench` | SVD is important after Sprint 130, but current benchmark has no bounded sentinel row or metadata contract. | Design-only candidate until fixture, metric, runtime, and non-claim boundary are explicit. | SVD benchmark owner. |
| 7 | Supplemental large-matrix recurring validation | Guardrail `S1`/`S2` skipped unless opt-in | Useful maintainer breadth, but runtime and platform variance are high. | Keep supplemental; decide policy before recurring validation. | `large-matrix-guardrails`. |
| 8 | Direct repeated-run `bench_refactor` visibility | `bench_refactor_csc` canonical; `bench_refactor` benchmark-local | Public repeated-run lifecycle is already represented by canonical CSC path; broadening may duplicate claims. | Defer unless Day 6 identifies a distinct low-cost lane. | Direct/backend benchmark owner. |

## Threshold-Suitability Notes

| Candidate | Hard threshold suitability | Reason |
| --- | --- | --- |
| Existing S5 wall-check | Suitable as-is | Already has accepted machine-class baseline and narrow wall-check scope. |
| Canonical backend metadata completeness | Not a timing threshold | Metadata quality is schema/freshness validation, not runtime pass/fail. |
| LDLT CSC backend/runtime | Threshold-hostile for now | Backend fallback, optional dense runtime, KKT fixture choice, and platform variance need Day 4-6 policy. |
| Iterative convergence/BiCGSTAB | Threshold-hostile for now | Iteration counts, convergence tolerances, stochastic or matrix-sensitive behavior, and CPU variance complicate hard thresholds. |
| Eigensolver backend sweep | Threshold-hostile for full sweep | Backend/preconditioner matrix is broad and can be slow; use scoped report rows first. |
| SVD/bidiag | Threshold-hostile for now | Need bounded fixture and metric policy; Sprint 130 evidence was correctness/claim-oriented, not timing baseline. |
| Supplemental large-matrix S1/S2 | Not suitable by default | Opt-in, threshold-free, platform-local reports; promotion needs runtime/support-tier policy. |
| Cholesky CSC S2 | Threshold-free by design | Current S2 rows are local report context with backend env and dense-kernel metadata. |

## Candidate Owner Map

| Candidate | Candidate owner | Validation command if implemented | Current blocker |
| --- | --- | --- | --- |
| Backend/runtime observability contract | Runtime governance owner | Docs hygiene; focused report checks only if scripts change | Need contract for builtin/optional backends, OpenMP, thread counts, fallback, and unknown states. |
| Canonical backend metadata completeness | `benchmark-report-owner` plus `report-index-owner` | `make bench-canonical-report` if script changes | Need field matrix and claim-boundary policy. |
| LDLT CSC backend/runtime lane | Direct/backend benchmark owner | Focused `build/bench_ldlt_csc` or `make bench-canonical-report` if included; full C quality if C changes | Need bounded fixture, metric, backend fields, and report-only versus threshold decision. |
| Iterative convergence/BiCGSTAB lane | Iterative benchmark owner | Focused `build/bench_convergence` or `build/bench_bicgstab`; full C quality if C changes | Need stable fixture, metric, runtime budget, and variance policy. |
| Eigensolver backend lane | Eigensolver benchmark owner | Focused `build/bench_eigs` or `build/bench_eigs_reuse`; full C quality if C changes | Need narrow backend slice and runtime budget. |
| SVD/bidiag lane | SVD benchmark owner | Focused `build/bench_svd`; full C quality if C changes | Need bounded SVD fixture, metric, and non-claim boundary. |
| Supplemental large-matrix policy | `large-matrix-guardrails` | `make large-matrix-guardrails`; supplemental mode only if policy touches opt-in lanes | Need recurring validation policy and runtime cost decision. |

## Residual Hot-Path Queue

| Residual | Support tier | Claim impact | Blocker | Future owner |
| --- | --- | --- | --- | --- |
| Backend/runtime contract missing | Deferred governance | Without contract, metadata rows can be over-read as backend parity or runtime control. | Day 4 contract not yet written. | Runtime governance owner. |
| Canonical report backend metadata uneven | Deferred metadata | Cross-branch comparisons can miss backend/fallback context. | Day 5 field matrix needed before script edits. | `report-index-owner`. |
| LDLT CSC sentinel not yet scoped | Deferred/report-only candidate | Could imply backend performance parity if hard-thresholded too early. | Need bounded metric and backend/fallback row semantics. | Direct/backend benchmark owner. |
| Iterative convergence sentinel not yet scoped | Deferred/report-only candidate | Could imply convergence-rate or solver superiority claims. | Need stable fixture and variance policy. | Iterative benchmark owner. |
| Eigensolver backend sentinel not yet scoped | Deferred/report-only candidate | Could imply broad backend or preconditioner superiority. | Need narrow low-cost slice. | Eigensolver benchmark owner. |
| SVD/bidiag sentinel not yet scoped | Deferred/report-only candidate | Could imply broad SVD performance or parity after Sprint 130 correctness work. | Need bounded fixture and metric. | SVD benchmark owner. |
| Supplemental large-matrix recurring validation undecided | Supplemental | Could imply portable scalability or memory proof. | Need runtime/support-tier policy before promotion. | `large-matrix-guardrails`. |

## Day 4 Handoff

Day 4 should define the backend/runtime contract before any implementation
batch is selected. The contract should settle:

- builtin versus optional dense backend states;
- backend request, selected, fallback, unavailable, and unknown semantics;
- OpenMP build mode versus runtime-owned thread behavior;
- `OMP_NUM_THREADS` and nested-runtime boundaries;
- which fields must be visible in sentinel, canonical, guardrail, and
  benchmark-local reports;
- which backend/runtime states block hard thresholds and which are safe as
  threshold-free report context.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sentinel gaps are ranked by risk, not convenience. | Complete | Ranking prioritizes cross-cutting backend/runtime observability, canonical metadata, and user-facing backend/solver workflows before easy exploratory lanes. |
| Threshold-hostile paths are marked as report-only or supplemental. | Complete | Threshold-suitability table marks LDLT, iterative, eigensolver, SVD, and supplemental large-matrix paths as threshold-hostile or report-only until blockers are resolved. |
| Every high-priority gap has an owner or explicit blocker. | Complete | Candidate owner map and residual queue assign owners and blockers for each ranked gap. |
