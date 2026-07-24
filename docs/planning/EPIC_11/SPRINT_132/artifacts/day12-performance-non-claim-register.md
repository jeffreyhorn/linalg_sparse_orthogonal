# Sprint 132 Day 12 - Performance Non-Claim Register

## Purpose

Publish the Sprint 132 non-claim register for local performance, backend,
runtime, sentinel, report-index, and supplemental evidence.

This artifact consolidates Sprint 131 report-index boundaries, the Day 4
backend/runtime contract, the Day 6 sentinel policy, the Day 8 implementation
batch, the Day 9 documentation cleanup, and the Day 10-11 validation results.

## Performance Non-Claim Register

| Non-claim | Current evidence | Owner | Promotion trigger |
| --- | --- | --- | --- |
| Local benchmark rows do not prove portable performance. | `make bench-canonical-report`, S2 sentinel rows, and benchmark CSVs carry local command/runtime context. | `benchmark-report-owner` | Repeated baseline policy across host classes, compiler families, fixtures, repeat counts, and variance windows. |
| Canonical reports are not pass/fail timing gates. | Canonical `index.tsv` has no pass/fail status and docs describe threshold-free use. | `benchmark-report-owner` and `report-index-owner` | Explicit threshold design, accepted baseline, failure meaning, and maintainer-guide update. |
| S2 Cholesky CSC rows are not hard performance gates. | S2 rows use `status=report`, `support_tier=reviewed_threshold_free`, and `claim_boundary=local_threshold_free`. | `performance-sentinels` owner | Baseline tied to exact fixture, backend state, build mode, thread context, host class, and variance policy. |
| S5 wall-check is not a broad benchmark suite. | S5 covers existing reorder wall-check metrics only. | `wall-check` owner | New lane-specific baseline and separate report claim for any additional metric. |
| Benchmark speedup fields are not solver superiority claims. | CSV speedups remain local workflow measurements. | Benchmark owners | Broader matrix set, statistical design, independent correctness evidence, and claim-gate review. |
| Generated metadata is not correctness proof. | Day 8-11 validated schema, status, and freshness only. | `report-index-owner` | Separate tests or oracle-backed validation for behavioral claims. |

## Backend and Runtime Non-Claim Table

| Non-claim | Current evidence | Owner | Promotion trigger |
| --- | --- | --- | --- |
| Backend request does not prove backend selection. | Sentinel and benchmark design separate request, selected, fallback, and `n/a` states. | Runtime governance owner | Selector-visible selected/fallback evidence for each promoted lane. |
| Builtin and optional backends are not parity-proven. | S2 reports builtin selected locally; LDLT optional rows remain benchmark-owned and not promoted. | Direct/backend benchmark owner | Backend-specific correctness tests plus comparable fixtures, tolerances, and local baselines. |
| Optional backend availability is not guaranteed. | Unavailable and unknown states remain explicit; no public availability probe was added. | Runtime governance owner | Supported probe contract, skip/fallback semantics, and platform support policy. |
| Fallback does not prove optional backend correctness or failure. | Day 4 fallback policy defines fallback as safe local selector resolution. | Runtime governance owner | Selector diagnostics that distinguish unavailable, unsupported, invalid, failed probe, and declined backend states. |
| OpenMP build mode does not prove speedup. | Outputs record `build_mode=serial` for the validated run; docs treat OpenMP as context. | Runtime governance owner | OpenMP-specific benchmark design with baseline, thread policy, and nested-runtime validation. |
| `OMP_NUM_THREADS` is not a library thread-control API. | Reports record `omp_num_threads=unset`; docs preserve runtime-owned interpretation. | Runtime governance owner | Public API proposal, implementation, tests, and maintainer-guide acceptance. |
| Nested runtime safety is not proven by indirect OpenMP reachability. | OpenMP owners remain SpMV and eigs paths; no new outer parallel regions were added. | Runtime governance owner | Focused nested-parallelism and oversubscription validation plan. |
| Backend fields in sentinel rows do not widen QR, SVD, eigensolver, or graph backend claims. | Backend metadata is scoped to S2 Cholesky CSC and existing direct benchmark CSVs. | Affected solver benchmark owners | Solver-specific backend design and proof surface for each family. |

## Runtime, Scalability, Memory, Corpus, and Freshness Non-Claims

| Category | Non-claim | Owner | Promotion trigger |
| --- | --- | --- | --- |
| Runtime scalability | No generated report proves scaling across thread counts, cores, or platforms. | Runtime governance owner | Multi-thread run matrix, host labels, variance policy, and nonportable scope wording. |
| Memory | No sentinel or canonical row proves portable memory behavior or max-RSS limits. | Benchmark owners and `large-matrix-guardrails` | Stable memory metric, host policy, fixture set, threshold, and failure meaning. |
| Corpus breadth | The current sentinels do not prove broad SuiteSparse, Matrix Market, or generated corpus coverage. | Corpus taxonomy owner | Reviewed fixture taxonomy, oracle ownership, support tier, and recurring validation command. |
| Solver coverage | Deferred iterative, eigensolver, and SVD lanes are not implemented evidence. | Iterative, eigensolver, and SVD benchmark owners | Bounded command, fixture, metric, runtime budget, and non-claim wording. |
| Report freshness | Freshness anchors do not imply CI, release, or support guarantees. | `report-index-owner` | Release policy that defines current/stale windows and required regeneration commands. |
| Stale artifacts | Historical build artifacts are planning context, not current validation. | Report family owners | Regeneration on current branch plus matching manifest commit/branch. |
| Supplemental reports | Opt-in supplemental lanes are not reviewed recurring gates. | Supplemental report owners | Runtime budget, owner, support-tier promotion, and claim-boundary review. |

## Supplemental-to-Reviewed Promotion Criteria

A local report row may be promoted from supplemental, experimental, deferred,
or threshold-free evidence only when all criteria below are satisfied:

1. Stable owner is assigned for the command, artifacts, schema, and docs.
2. Command has a bounded runtime budget that fits recurring local validation.
3. Fixture, corpus slice, repeat count, tolerance, and metric are fixed.
4. Status vocabulary and failure meaning are explicit.
5. Freshness anchors include generated time plus git commit/branch when
   available.
6. Platform, compiler, build mode, and `OMP_NUM_THREADS` are recorded when
   runtime-sensitive.
7. Backend request, selected backend, fallback, unavailable, unknown, and
   `n/a` states are recorded when backend-sensitive.
8. Baseline and variance policy exist before any timing threshold is added.
9. Documentation states what the row proves and what it does not prove.
10. Focused validation passes without requiring broad benchmark sweeps.
11. Supplemental mode or optional backend availability is not required unless
    the support policy explicitly accepts that host dependency.
12. Maintainer-guide wording is updated only after implementation and
    validation evidence support the stronger status.

## Future Owner Queue

| Queue item | Trigger | Future owner |
| --- | --- | --- |
| Canonical normalized support tier and claim boundary | A cross-report index consumer needs row-level normalized fields. | `report-index-owner` and `benchmark-report-owner` |
| Canonical direct backend row extraction | Backend-aware canonical comparisons need row-level selected/fallback context outside CSVs. | Direct/backend benchmark owner |
| LDLT recurring report-only lane | Existing KKT backend fields need recurring sentinel/report visibility. | Direct/backend benchmark owner |
| Iterative convergence/BiCGSTAB sentinel | Stable fixture, tolerance, metric, and runtime budget are selected. | Iterative benchmark owner |
| Eigensolver backend slice | Narrow backend/preconditioner slice and OpenMP policy are selected. | Eigensolver benchmark owner |
| SVD/bidiag report lane | Bounded fixture, rank/tolerance, and metric semantics are selected. | SVD benchmark owner |
| Large-matrix supplemental promotion | Supplemental runtime and host-sensitivity policy is accepted. | `large-matrix-guardrails` |
| Automated stale-report scanner | Common metadata contract exists across enough report families. | `report-index-owner` |
| Optional backend availability probe | Public or maintainer support policy needs explicit unavailable-state rows. | Runtime governance owner |
| New hard timing threshold | Baseline, host class, variance, backend, OpenMP, fixture, and failure meaning are accepted. | Runtime governance owner plus affected benchmark owner |

## Maintainer Wording Decision

No Day 12 maintainer-guide update is needed.

Rationale:

- Day 9 already updated benchmark governance wording for canonical
  threshold-free rows, sentinel support tiers, claim boundaries, backend state
  preservation, and supplemental promotion boundaries.
- Day 11 validation confirmed the touched report surfaces match that wording.
- Day 12 adds a planning non-claim register, not a new behavior, schema,
  threshold, backend, OpenMP, CI, or public API change.

Future maintainer wording should change only when a future sprint promotes a
specific row, adds a hard threshold, changes report schema, changes backend
availability semantics, or changes OpenMP/runtime behavior.

## Day 13 Handoff

Day 13 should run the final affected validation batch and publish the residual
runtime, sentinel, backend, and report queue. It should use this non-claim
register as the checklist for claim-drift review.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every performance/backend/runtime non-claim has owner and trigger. | Complete | Non-claim tables and future-owner queue assign owners and promotion triggers. |
| Promotion criteria are stricter than local measurement existence. | Complete | Promotion checklist requires owner, bounded runtime, fixture/metric policy, freshness, backend/runtime context, baseline/variance for thresholds, docs, and validation. |
| Maintainer wording is updated only when evidence supports it. | Complete | No Day 12 maintainer update was made because Day 9 wording already matches Day 11 validation evidence. |
