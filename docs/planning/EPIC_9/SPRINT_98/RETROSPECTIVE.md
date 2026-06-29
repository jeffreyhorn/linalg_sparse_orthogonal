# Sprint 98 Retrospective

**Sprint:** 98 - Assurance, External Comparison & Coverage Architecture Phase 3
**Duration:** 14 days (Days 1-14 landed on this branch)
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 98 started from the Epic 9 project-plan section and the live
      post-Sprint-97 assurance/comparison surface
- [x] external correctness and runtime/fill comparison candidates were reranked
      before implementation
- [x] a bounded proof/comparison architecture was written before new evidence
      lanes landed
- [x] the highest-value correctness expansion landed as an LDLT CSC external
      dense-reference lane on deterministic KKT fixtures
- [x] the runtime/fill lane landed as a bounded `bench-reorder-sprint86`
      artifact with `nnz_L` as the primary fill field
- [x] maintainer proof ownership was updated for the new LDLT CSC lane
- [x] benchmark-governance guardrails were added for the Sprint 98
      reorder/fill artifact
- [x] coverage and workflow topology were audited without widening reviewed
      scope
- [x] a compact Sprint 98 assurance-topology snapshot was added to the
      maintainer guide
- [x] CI, public docs, benchmark docs, and maintainer guidance were reconciled
      against the widened assurance model
- [x] the full required validation chain passed after the C test change:
  - `make format`
  - `make lint`
  - `make test`
- [x] Sprint 98 closed with a ranked Sprint 99 handoff queue

## What Went Well

1. **The sprint reranked evidence before adding it.**
   Day 2 separated external correctness, runtime/fill evidence, coverage
   topology, workflow ownership, and public claims. That kept Sprint 98 from
   turning a broad "competitive proof" goal into an unbounded benchmark or
   oracle expansion.

2. **The LDLT CSC lane stayed family-local and deterministic.**
   The new `tests/ldlt_external_dense_reference.py` helper uses fixture keys
   `kkt5` and `kkt10` and emits dense reference solutions without mirroring
   internal factor, pivot, permutation, or CSC layout details. The C harness in
   `tests/test_ldlt_csc.c` asserts user-visible solve agreement and residual
   strength.

3. **The correctness proof model now extends beyond SPD Cholesky.**
   Sprint 98 added the first maintained external correctness lane beyond the
   existing Cholesky CSC SPD proof. That strengthens direct-family assurance
   while explicitly avoiding broad LDLT ecosystem or every-solver-family
   claims.

4. **The runtime/fill lane reused an existing bounded command.**
   `make bench-reorder-sprint86` already named the small two-fixture slice.
   Sprint 98 used that surface to produce an artifact instead of adding a new
   benchmark harness, timing threshold, workflow lane, or canonical report
   expansion.

5. **The maintainer guide now has a compact topology map.**
   The Sprint 98 assurance-topology snapshot ties together LDLT CSC external
   correctness, reorder/fill calibration, coverage topology, and workflow
   topology. That improves discoverability without moving proof ownership out
   of the family-local tests.

6. **Support-surface alignment avoided public overclaiming.**
   README, INSTALL, benchmark docs, workflow comments, and maintainer guidance
   were reviewed against the new evidence. The sprint kept public docs
   high-level and left maintainer-only proof detail in the maintainer guide and
   planning artifacts.

7. **The branch closed from a full validation anchor.**
   Day 13 reran focused helper checks, focused `test_ldlt_csc`, the bounded
   reorder/fill benchmark, stale-claim scans, whitespace hygiene, and the full
   `make format && make lint && make test` chain.

## What Didn't Go Well

1. **The new LDLT proof still lives in a large test owner.**
   Keeping the harness in `tests/test_ldlt_csc.c` was the right locality choice
   for this sprint, but the file remains large and still deserves a future
   extraction boundary.

2. **The runtime/fill artifact is manual planning evidence.**
   Day 8 captured a useful bounded artifact, but repeated future captures may
   justify a small generated report target. Sprint 98 deliberately avoided
   adding that surface before repeated need was proven.

3. **The benchmark timing values remain environment-local.**
   The artifact includes `reorder_ms` for context, but those numbers cannot be
   interpreted as portable thresholds. The sprint had to repeat that guardrail
   in several places to prevent claim drift.

4. **Coverage topology was clarified rather than improved.**
   The sprint audited coverage and confirmed that it remains supplemental and
   tree-mutating. That is useful, but it does not expand coverage evidence or
   reduce coverage workflow cost.

5. **Workflow alignment stayed intentionally conservative.**
   No CI workflow lane was added for `bench-reorder-sprint86`. That preserves
   scope, but it leaves future classification work if the artifact becomes
   important enough for CI capture.

## Final Metrics

### Validation

| Metric | Sprint 98 close state |
|---|---:|
| helper positive fixtures | `kkt5` and `kkt10` passed |
| helper unknown-fixture behavior | `ERROR unknown fixture nope`, exit `1` |
| focused LDLT CSC test | `98` passed, `0` failed, `0` skipped |
| `kkt5` external-reference metric | `max|x-x_ref| = 0.000e+00`, `rel_residual = 0.000e+00` |
| `kkt10` external-reference metric | `max|x-x_ref| = 3.553e-15`, `rel_residual = 2.292e-16` |
| focused runtime/fill command | `make bench-reorder-sprint86` passed |
| full branch-level gate | `make format && make lint && make test` passed |
| final full-test summary | `All tests passed.` |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on touched code, helper, Sprint 98 docs, and maintainer guide |
| stale-claim scan | only negative guardrails and boundary language found |

### Sprint 98 Artifact Package

| Metric | Sprint 98 close state |
|---|---:|
| total artifact files under `SPRINT_98/artifacts/` | `15` |
| baseline/rerank/design artifacts | `4` |
| correctness expansion artifacts | `3` |
| runtime/fill artifacts | `3` |
| topology/support/validation/closeout artifacts | `5` |

Notes:

- baseline/rerank/design artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-assurance-baseline.md`
  - `day2-comparison-surface-rerank.md`
  - `day3-proof-comparison-architecture-design.md`
- correctness expansion artifacts:
  - `day4-external-correctness-boundary-freeze.md`
  - `day5-external-correctness-expansion-batch1.md`
  - `day6-correctness-expansion-closeout.md`
- runtime/fill artifacts:
  - `day7-runtime-fill-boundary-freeze.md`
  - `day8-runtime-fill-comparison-batch1.md`
  - `day9-runtime-fill-comparison-closeout.md`
- topology/support/validation/closeout artifacts:
  - `day10-coverage-topology-audit.md`
  - `day11-coverage-topology-cleanup.md`
  - `day12-ci-support-surface-alignment.md`
  - `day13-validation-and-residual-queue.md`
  - `day14-closeout-and-handoff.md`

### Landed Assurance Package

| Metric | Sprint 98 close state |
|---|---:|
| new external reference helpers | `1` |
| new LDLT CSC external-reference tests | `2` |
| new maintained correctness fixtures | `2` fixture keys |
| maintainer-guide proof/topology sections updated | `2` |
| runtime/fill artifact lanes captured | `1` |
| workflow files changed | `0` |
| Makefile or benchmark C files changed | `0` |
| public README/INSTALL files changed | `0` |

Notes:

- new helper:
  - `tests/ldlt_external_dense_reference.py`
- new tests:
  - `test_s98_external_dense_reference_kkt_5x5`
  - `test_s98_external_dense_reference_kkt_10x10`
- fixture keys:
  - `kkt5`
  - `kkt10`
- maintainer guide updates:
  - LDLT CSC external proof ownership
  - Sprint 98 assurance topology snapshot and runtime/fill guardrail

## Residual Deferred Debt

Sprint 98 deliberately stopped after one bounded correctness lane, one
runtime/fill artifact lane, topology cleanup, support alignment, and full
validation.

Most important carry-forward work:

- design broader LDLT CSC Matrix Market or indefinite corpus coverage before
  adding fixtures
- design iterative solver external comparison around convergence semantics
- design eigensolver/LOBPCG external comparison with explicit cluster,
  tolerance, and runtime limits
- decide whether repeated reorder/fill captures need a generated report target
- decide whether `bench_amd_qg` remains adjacent support evidence or becomes a
  separate bounded artifact lane
- classify any future `bench-reorder-sprint86` CI use as reviewed,
  supplemental, or artifact-only before adding it

Still consciously constrained rather than silently solved:

- no broad LDLT external proof across indefinite matrices
- no every-solver-family external proof
- no external factorization or pivot-layout parity claim
- no portable timing threshold
- no cross-platform timing parity
- no canonical benchmark report expansion
- no workflow widening
- no coverage threshold or target change
- no public-doc product claim for the new maintainer-only proof detail

Not carried forward as unresolved Sprint 98 debt:

- comparison-surface rerank
- proof/comparison architecture design
- LDLT CSC external correctness boundary
- LDLT CSC external helper and C harness implementation
- maintainer proof-owner wording for the new correctness lane
- reorder/fill runtime boundary
- bounded runtime/fill artifact capture
- benchmark-governance guardrail for the Sprint 98 artifact
- coverage/proof-owner topology audit
- maintainer-guide topology snapshot
- CI/support-surface alignment audit
- final validation and residual queue

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-assurance-baseline.md](./artifacts/day1-assurance-baseline.md)
- [day2-comparison-surface-rerank.md](./artifacts/day2-comparison-surface-rerank.md)
- [day3-proof-comparison-architecture-design.md](./artifacts/day3-proof-comparison-architecture-design.md)
- [day4-external-correctness-boundary-freeze.md](./artifacts/day4-external-correctness-boundary-freeze.md)
- [day5-external-correctness-expansion-batch1.md](./artifacts/day5-external-correctness-expansion-batch1.md)
- [day6-correctness-expansion-closeout.md](./artifacts/day6-correctness-expansion-closeout.md)
- [day7-runtime-fill-boundary-freeze.md](./artifacts/day7-runtime-fill-boundary-freeze.md)
- [day8-runtime-fill-comparison-batch1.md](./artifacts/day8-runtime-fill-comparison-batch1.md)
- [day9-runtime-fill-comparison-closeout.md](./artifacts/day9-runtime-fill-comparison-closeout.md)
- [day10-coverage-topology-audit.md](./artifacts/day10-coverage-topology-audit.md)
- [day11-coverage-topology-cleanup.md](./artifacts/day11-coverage-topology-cleanup.md)
- [day12-ci-support-surface-alignment.md](./artifacts/day12-ci-support-surface-alignment.md)
- [day13-validation-and-residual-queue.md](./artifacts/day13-validation-and-residual-queue.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 98 achieved its goal:

- maintained external correctness evidence now extends beyond the bounded SPD
  Cholesky lane to a bounded LDLT CSC KKT lane
- runtime/fill comparison evidence is captured for the selected
  `bench-reorder-sprint86` workload
- proof ownership and benchmark-governance boundaries are clearer
- coverage and workflow topology were audited without unsupported widening
- public docs remain bounded while maintainer-only proof detail is documented
  where it belongs
- the branch validates cleanly under focused checks and the full quality chain
- Sprint 99 receives a ranked assurance/comparison queue instead of an
  unbounded proof expansion backlog
