# Sprint 132 Day 6 - Sentinel Design Policy

## Purpose

Define which Sprint 132 sentinel candidates are ready for implementation,
which remain threshold-free report lanes, and which stay design-only until
their runtime, fixture, metadata, and non-claim boundaries are complete.

This policy uses the Day 3 gap ranking, Day 4 backend/runtime contract, and Day
5 metadata design. It does not add new hard timing gates.

## Sentinel Design Rules

- Keep S5 as the only current hard local wall-check gate.
- Treat S2 Cholesky CSC rows as threshold-free report context.
- Prefer metadata completeness checks before adding new backend-sensitive
  benchmark rows.
- Require every sentinel row to name its command, fixture, metric, support
  tier, backend/runtime context, and claim boundary.
- Treat `unknown` backend, build mode, thread count, fixture, or support tier
  as a blocker for hard thresholds.
- Treat `unavailable` optional backends as explicit skip/fallback context, not
  silent success.
- Keep supplemental large-matrix rows opt-in until a future sprint promotes
  them with runtime and review policy.
- Do not let benchmark rows imply backend parity, OpenMP speedup, portable
  timing, memory portability, solver superiority, or correctness proof.

## Candidate Sentinel Lane Table

| Lane | Candidate | Command or source | Metric | Support tier | Runtime budget | Threshold posture | Sprint 132 decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S5 | Existing wall-check gate | `make wall-check` through `make performance-sentinels` | `qg_amd_reorder_ms`, `amd_reorder_ms`, `nd_reorder_ms` | reviewed | <= 3 minutes local target budget | Hard local gate using existing baselines | Keep as-is. No schema or threshold widening needed. |
| S2 | Existing Cholesky CSC report lane | `build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1` through `make performance-sentinels` | Cholesky factor/solve timings and speedups | reviewed report | <= 2 minutes local target budget | Threshold-free | Keep as threshold-free; candidate for structured metadata only. |
| S6 | Sentinel metadata completeness | `make performance-sentinels` generated `sentinels.tsv` and `manifest.txt` | Required-field presence and non-claim notes | reviewed schema check | <= 30 seconds after sentinel generation | Non-timing pass/fail schema check | Implementation-ready if Day 7 selects script/docs changes. |
| C1 | Canonical backend metadata completeness | `make bench-canonical-report` generated `index.tsv`, manifest, and direct CSVs | Artifact provenance plus backend fields where emitted | generated report check | <= 5 minutes local target budget | Threshold-free schema/report check | Implementation-ready as report validation, not a timing gate. |
| L1 | LDLT CSC backend/runtime report | `build/bench_refactor_csc --indefinite-kkt --repeat 1` or canonical direct CSV row | backend request/selected/fallback and local LDLT timing values | experimental report | <= 2 minutes focused local run | Threshold-free | Candidate implementation only if existing benchmark output is reused without new C schema churn. |
| I1 | Iterative convergence/BiCGSTAB report | `build/bench_convergence` or `build/bench_bicgstab` bounded invocation | convergence timing or iteration summary | deferred | <= 3 minutes target, not yet proven | Threshold-free only | Design-only until fixture, metric, and variance policy are explicit. |
| E1 | Eigensolver backend slice | `build/bench_eigs` or `build/bench_eigs_reuse` narrow invocation | eigensolver timing/reuse/convergence context | deferred | <= 3 minutes target, not yet proven | Threshold-free only | Design-only until narrow backend/preconditioner slice is selected. |
| V1 | SVD/bidiag bounded report | `build/bench_svd` bounded invocation | SVD/bidiag local timing or residual-expansion metric | deferred | <= 3 minutes target, not yet proven | Threshold-free only | Design-only until fixture and metric semantics are explicit. |
| G5 | Large-matrix supplemental recurrence check | `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 make large-matrix-guardrails` | supplemental reorder/qg-AMD report availability | supplemental | <= 10 minutes target, host-sensitive | Threshold-free | Keep supplemental and opt-in; no recurring reviewed gate in Sprint 132. |

## Metric and Threshold Policy

| Metric family | Allowed metrics | Required context | Threshold rule | Non-claim semantics |
| --- | --- | --- | --- | --- |
| Existing wall-check timing | `qg_amd_reorder_ms`, `amd_reorder_ms`, `nd_reorder_ms` | existing baseline file, command, fixture, platform, compiler, build mode, `OMP_NUM_THREADS` | Keep existing hard thresholds only | Local regression gate for bounded reorder lanes, not portable performance. |
| Cholesky CSC report timing | `factor_ll_ms`, `factor_csc_ms`, `factor_csc_sn_ms`, solve timings, speedups | command, `nos4`, repeat count, dense kernel, panel solver, backend env vars, build mode, `OMP_NUM_THREADS` | No hard threshold | Local report context for Cholesky CSC path visibility. |
| Metadata completeness | required field presence, recognized status vocabulary, explicit skip/report/pass/fail semantics | generated artifact, schema owner, report family, support tier, claim boundary | Pass/fail allowed because it is schema, not timing | Validates interpretability, not performance. |
| LDLT backend report | backend request, selected, fallback, KKT fixture identity, local refactor/solve timings | command, repeat count, backend fields, build mode, `OMP_NUM_THREADS` | No hard threshold in Sprint 132 | Bounded backend observability for retained LDLT lane only. |
| Iterative or BiCGSTAB report | convergence summary, iterations, local timing, residual where emitted | stable fixture, tolerance, repeat count, command, build mode, `OMP_NUM_THREADS` | No hard threshold until variance policy exists | Local solver workflow evidence, not solver superiority. |
| Eigensolver report | reuse timing, backend/preconditioner slice, convergence/residual context | matrix, backend/preconditioner choice, command, build mode, `OMP_NUM_THREADS` | No hard threshold until narrow slice is accepted | Local eigensolver evidence, not broad backend or preconditioner parity. |
| SVD/bidiag report | bounded timing, residual, or expansion metric | matrix, rank/tolerance, command, build mode, `OMP_NUM_THREADS` | No hard threshold until fixture and metric policy exist | Local SVD workflow evidence, not broad SVD performance. |
| Supplemental large-matrix report | fill rows, reorder timing, qg-AMD rows, max-RSS where emitted | supplemental flag, platform, compiler, command, fixture slice | No hard threshold by default | Maintainer context only; no memory or scalability proof. |

## Reviewed Versus Supplemental Split

| Tier | Meaning | Allowed status values | Promotion rule |
| --- | --- | --- | --- |
| reviewed | Recurring local quality evidence with explicit owner and validation command. | `pass`, `fail`, `report`, `skip` | May run by default when runtime is bounded and semantics are stable. |
| reviewed thresholded | Existing local gate with accepted baseline and threshold. | `pass`, `fail`, `skip` | Limited to S5 unless a future sprint accepts a new baseline contract. |
| reviewed threshold-free | Recurring report evidence with stable fields but no pass/fail timing claim. | `report`, `skip`, `error` | May be included by default if skips are explicit and rows carry runtime context. |
| supplemental | Useful but opt-in evidence with higher runtime, host sensitivity, or broad corpus scope. | `report`, `skip`, `error` | Requires future owner, runtime budget, and claim-boundary promotion before becoming reviewed. |
| experimental | Focused sprint evidence used to evaluate a candidate lane. | `report`, `skip`, `error` | Requires stable command, fixture, metric, metadata, and runtime before recurring use. |
| deferred | Candidate is not ready for generated sentinel/report implementation. | `deferred` in design artifacts | Requires blocker resolution before implementation. |

## Skip, Unavailable, and Stale Behavior

| Condition | Required behavior | Interpretation |
| --- | --- | --- |
| Required binary missing | Emit `skip` when practical, or fail early for commands whose purpose is binary validation. | Infrastructure is incomplete; not a pass. |
| Required fixture missing | Emit `skip` with fixture name and command. | Evidence is absent; not a performance result. |
| Optional backend unavailable | Emit fallback/unavailable context when the lane can still run builtin; skip only when optional backend is required by lane policy. | Optional backend availability is host-local. |
| Backend metadata unknown | Mark as `unknown` or include explicit notes. | Blocks backend-specific comparisons and hard thresholds. |
| OpenMP build mode unknown | Mark as `unknown`. | Blocks OpenMP-sensitive comparisons. |
| `OMP_NUM_THREADS` unset | Record `unset`. | Valid runtime context, not a library default. |
| Supplemental mode disabled | Emit explicit supplemental `skip` rows. | Maintains visibility without promoting opt-in work. |
| Generated report stale | Mark stale in indexes or require regeneration before using as evidence. | Stale artifacts are historical context, not current validation. |
| Thresholded lane fails | Stop the command and preserve raw output. | Local regression signal; do not proceed as if threshold-free. |

## Implementation-Ready Lane List

| Candidate | Ready scope | Files likely touched | Validation if implemented |
| --- | --- | --- | --- |
| S6 sentinel metadata completeness | Add or validate structured support-tier/claim-boundary/backend context for `performance-sentinels` output. | `scripts/performance_sentinels.sh`, `benchmarks/README.md`, `docs/maintainer_guide.md` | `bash -n scripts/performance_sentinels.sh`; `make performance-sentinels`; inspect `build/bench-reports/sentinels/sentinels.tsv` and manifest. |
| C1 canonical backend metadata completeness | Add report-index metadata for host/build context only if low churn; otherwise document current CSV-owned backend fields. | `scripts/bench_canonical_report.sh`, `benchmarks/README.md`, `docs/maintainer_guide.md` | `bash -n scripts/bench_canonical_report.sh`; `make bench-canonical-report`; inspect canonical `index.tsv`, manifest, and direct CSV headers. |
| L1 LDLT backend/runtime report-only lane | Reuse existing `bench_refactor_csc` LDLT KKT backend fields as threshold-free evidence. | Prefer script/docs only; C changes only if current fields are insufficient. | Focused `build/bench_refactor_csc --indefinite-kkt --repeat 1`; full C quality only if C/header files change. |

## Design-Only Deferral List

| Candidate | Deferral reason | Required before implementation |
| --- | --- | --- |
| I1 iterative convergence/BiCGSTAB sentinel | Stable fixture, metric, tolerance, runtime, and variance policy are not yet defined. | Pick one bounded command, one metric family, repeat/tolerance policy, and non-claim wording. |
| E1 eigensolver backend slice | Full backend/preconditioner sweep is too broad and OpenMP-sensitive for a local sentinel. | Pick one narrow fixture and backend/preconditioner slice with a measured runtime budget. |
| V1 SVD/bidiag report | Current Sprint 130 evidence was correctness/claim-oriented; sentinel metric and fixture are still unset. | Define matrix, rank/tolerance, metric, runtime budget, and report-only claim boundary. |
| G5 supplemental large-matrix recurrence | Supplemental lanes are opt-in and platform-sensitive. | Define recurring cadence, owner, max runtime, and proof that rows remain non-gating. |
| New hard backend timing threshold | Backend fallback and optional availability vary by host. | Accepted baseline tied to exact backend request/selection/fallback, build mode, thread context, fixture, command, repeat count, and host class. |

## Day 7 Handoff

Day 7 should choose the implementation batch from the implementation-ready
lanes. The lowest-risk batch is script/docs metadata work around
`make performance-sentinels` and `make bench-canonical-report`, with no C
changes unless an existing benchmark field is proven insufficient.

Day 7 should also define rollback and deferral criteria:

- stop if `make performance-sentinels` becomes flaky or too slow
- stop if structured metadata requires broad benchmark CSV churn
- defer any lane whose backend, OpenMP, fixture, metric, support tier, or claim
  boundary remains unknown
- preserve S5 as the only hard timing gate

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every proposed sentinel has explicit metric and non-claim semantics. | Complete | Candidate table and metric policy define command, metric, support tier, threshold posture, and claim boundary for each lane. |
| Threshold-free lanes are not treated as hard performance gates. | Complete | S2, C1, L1, I1, E1, V1, and G5 are report-only, supplemental, experimental, or deferred. |
| Implementation candidates have validation commands and runtime budgets. | Complete | Implementation-ready table names focused commands, validation expectations, and bounded runtime budgets. |
