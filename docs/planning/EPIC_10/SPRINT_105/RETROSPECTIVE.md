# Sprint 105 Retrospective

**Sprint:** 105 - Reordering, Graph & Large-Matrix Scalability
**Duration:** 14 days (Days 1-14 landed on branch `sprint-105`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 105 started from the Epic 10 project-plan scope and Sprint 104's
      benchmark/runtime claim-boundary handoff.
- [x] reorder, graph, fill, runtime, memory, fixture, and reviewed-status
      surfaces were audited before implementation.
- [x] a canonical fill/fixture/reporting contract was defined for reorder and
      graph evidence.
- [x] named-matrix evidence was refreshed for the committed `bench_reorder`
      fixture set.
- [x] generated-family evidence was refreshed for graph, ND, and qg-AMD proof
      owners.
- [x] `make large-matrix-guardrails` landed as a deterministic reviewed
      structural guardrail bundle with supplemental lanes kept opt-in.
- [x] qg-AMD proof-owner comments were rewritten around current maintained
      behavior rather than historical sprint chronology.
- [x] user, benchmark, and maintainer docs now explain how to interpret
      reorder/fill metrics, local timing, max-RSS context, reviewed lanes, and
      supplemental reports.
- [x] final evidence reconciliation passed parser checks for full and bounded
      `bench_reorder` outputs.
- [x] final validation passed:
  - `bash -n scripts/large_matrix_guardrails.sh`
  - `make large-matrix-guardrails`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scans
- [x] Sprint 106 handoff items are explicit and bounded.

## What Went Well

1. **The sprint started with evidence boundaries.**
   Day 1 and Day 4 kept the work grounded in reviewed, supplemental,
   local-only, and non-claim categories. That prevented benchmark rows from
   becoming accidental product claims.

2. **The metric contract gave later work a stable vocabulary.**
   Day 3 defined fill, runtime, memory, fixture, skip, threshold, and
   reviewed-status fields before any target implementation. That made Day 9,
   Day 11, and Day 12 easier to reconcile.

3. **Named-matrix evidence stayed useful without becoming overbroad.**
   Day 5 kept the bounded two-fixture slice small and parser-checkable, while
   Day 6 recorded the full committed `bench_reorder --skip-factor` set as
   supplemental/report evidence.

4. **Generated-family evidence stayed test-owned.**
   Day 7 tied paths, grids, meshes, clique bridges, banded matrices, graph
   partitions, ND behavior, and qg-AMD behavior to focused tests instead of
   treating generated fixtures as broad workload coverage.

5. **The guardrail target landed with the right default shape.**
   Day 9 added `make large-matrix-guardrails` and
   `scripts/large_matrix_guardrails.sh`. The default run covers reviewed
   structural lanes `G1` through `G4`, while `S1` and `S2` remain explicit
   supplemental skips unless opted in.

6. **Docs caught up with the evidence model.**
   Day 11 aligned `docs/algorithm.md`, `benchmarks/README.md`, and
   `docs/maintainer_guide.md` so users and maintainers can read `nnz_L`,
   `reorder_ms`, `peak_rss_mb`, reviewed lanes, and supplemental lanes
   without overreading them.

7. **The final validation gate matched the branch risk.**
   Because Sprint 105 touched `tests/test_reorder_amd_qg.c`, Day 13 reran the
   full `make format && make lint && make test` gate after focused guardrail
   validation.

## What Didn't Go Well

1. **The `sprint86` fixture-slice label remains confusing.**
   The label is documented as historical compatibility, but it still appears
   in current `bench_reorder` output. A future schema migration or alias could
   improve readability if consumer impact is acceptable.

2. **Large-matrix memory evidence is still mostly report-only.**
   `bench_amd_qg --skip-bitset` can provide useful max-RSS context, but
   Sprint 105 correctly kept it supplemental because max-RSS is
   platform-local and allocator-sensitive.

3. **Graph/reorder ownership cleanup was intentionally narrow.**
   Day 10 cleaned `tests/test_reorder_amd_qg.c`, but larger owners such as
   `tests/test_graph.c`, `tests/test_reorder_nd.c`, `src/sparse_graph.c`, and
   `src/sparse_reorder_nd.c` still contain history-heavy comments. Broad
   cleanup was deferred to avoid risky churn.

4. **`bench_fillin` arrow evidence remains outside the package.**
   The generated arrow family is still useful LU fill context, but it lacks the
   structured reporting contract needed to promote it into Sprint 105's
   reviewed evidence surface.

5. **Timing thresholds remain deliberately scarce.**
   Sprint 105 did not add new hard timing thresholds beyond the existing
   `wall-check` model. That is the right result, but it means future runtime
   guardrails still need fresh baseline design before becoming pass/fail gates.

## Final Metrics

### Validation

| Metric | Sprint 105 close state |
|---|---:|
| full branch-level gate | `make format && make lint && make test` passed |
| guardrail target | `make large-matrix-guardrails` passed |
| guardrail script syntax | `bash -n scripts/large_matrix_guardrails.sh` passed |
| reviewed guardrail lanes | `G1`-`G4` pass rows |
| supplemental guardrail lanes | `S1` and `S2` default skip rows |
| full named-matrix CSV contract | parser check passed |
| bounded two-fixture CSV contract | parser check passed |
| guardrail `index.tsv` contract | parser check passed |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scans | passed on touched docs, script, Makefile, and qg-AMD test file |

### Sprint 105 Artifact Package

| Metric | Sprint 105 close state |
|---|---:|
| artifact files under `SPRINT_105/artifacts/` | 15 |
| baseline/audit/contract artifacts | 5 |
| evidence/guardrail implementation artifacts | 5 |
| cleanup/docs/reconciliation/closeout artifacts | 5 |

Notes:

- baseline, audit, and contract artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-scalability-baseline.md`
  - `day2-reorder-graph-surface-audit.md`
  - `day3-fill-fixture-contract.md`
  - `day4-evidence-boundary.md`
- evidence and guardrail implementation artifacts:
  - `day5-reorder-fill-reporting-batch1.md`
  - `day6-named-matrix-evidence.md`
  - `day7-generated-graph-family-evidence.md`
  - `day8-large-matrix-guardrail-design.md`
  - `day9-scalability-guardrail-implementation.md`
- cleanup, docs, reconciliation, validation, and closeout artifacts:
  - `day10-graph-reorder-ownership-cleanup.md`
  - `day11-reporting-and-documentation-alignment.md`
  - `day12-integrated-evidence-reconciliation.md`
  - `day13-final-validation-and-residual-queue.md`
  - `day14-closeout-and-handoff.md`

### Landed Surface

| Metric | Sprint 105 close state |
|---|---:|
| new Make targets | 1 |
| new helper scripts | 1 |
| test files touched | 1 |
| public/maintainer docs touched | 3 |
| benchmark docs touched | 1 |
| new Sprint 105 planning artifacts | 15 |

## Residual Deferred Debt

Most important carry-forward work:

- decide whether `fixture_slice=sprint86` should gain a clearer alias or schema
  migration;
- design a structured LU fill schema before promoting `bench_fillin` arrow
  generated-family context;
- keep `S1` full named-matrix guardrails opt-in unless a future baseline
  promotes them;
- keep `S2` qg-AMD/max-RSS evidence opt-in and platform-local;
- continue graph/reorder comment cleanup only where touched or where it
  materially improves ownership clarity;
- require fresh baseline, threshold source, and machine-class assumptions
  before any new hard timing threshold;
- update CMake or Windows reviewed counts only with explicit scope decisions if
  future guardrail lanes change registration.

Still consciously constrained rather than silently solved:

- no portable timing superiority claim;
- no cross-platform max-RSS threshold;
- no supplemental large-matrix lanes as reviewed gates;
- no global sparse-workload coverage claim from generated families;
- no claim that `sprint86` is a new product slice;
- no broad graph/reorder ownership cleanup beyond the touched qg-AMD proof
  owner;
- no `bench_fillin` arrow rows as canonical Sprint 105 evidence.

Not carried forward as unresolved Sprint 105 debt:

- reorder/graph audit;
- fill and fixture metric contract;
- named-matrix evidence refresh;
- generated-family evidence refresh;
- large-matrix guardrail design;
- large-matrix guardrail implementation;
- qg-AMD proof-owner cleanup;
- reporting and documentation alignment;
- evidence reconciliation;
- final validation;
- closeout and Sprint 106 handoff.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scalability-baseline.md](./artifacts/day1-scalability-baseline.md)
- [day2-reorder-graph-surface-audit.md](./artifacts/day2-reorder-graph-surface-audit.md)
- [day3-fill-fixture-contract.md](./artifacts/day3-fill-fixture-contract.md)
- [day4-evidence-boundary.md](./artifacts/day4-evidence-boundary.md)
- [day5-reorder-fill-reporting-batch1.md](./artifacts/day5-reorder-fill-reporting-batch1.md)
- [day6-named-matrix-evidence.md](./artifacts/day6-named-matrix-evidence.md)
- [day7-generated-graph-family-evidence.md](./artifacts/day7-generated-graph-family-evidence.md)
- [day8-large-matrix-guardrail-design.md](./artifacts/day8-large-matrix-guardrail-design.md)
- [day9-scalability-guardrail-implementation.md](./artifacts/day9-scalability-guardrail-implementation.md)
- [day10-graph-reorder-ownership-cleanup.md](./artifacts/day10-graph-reorder-ownership-cleanup.md)
- [day11-reporting-and-documentation-alignment.md](./artifacts/day11-reporting-and-documentation-alignment.md)
- [day12-integrated-evidence-reconciliation.md](./artifacts/day12-integrated-evidence-reconciliation.md)
- [day13-final-validation-and-residual-queue.md](./artifacts/day13-final-validation-and-residual-queue.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)
- [Benchmark guide](../../../../benchmarks/README.md)
- [Algorithm guide](../../../algorithm.md)
- [Maintainer guide](../../../maintainer_guide.md)
- [`scripts/large_matrix_guardrails.sh`](../../../../scripts/large_matrix_guardrails.sh)

## Bottom Line

Sprint 105 achieved its goal:

- reorder/fill evidence now has a documented metric and fixture contract;
- named-matrix and generated-family evidence is refreshed and bounded;
- large-matrix structural guardrails are runnable through
  `make large-matrix-guardrails`;
- supplemental report lanes remain opt-in and explicitly non-review gates;
- qg-AMD test ownership is clearer;
- users and maintainers have aligned docs for interpreting structural fill,
  local timing, memory context, and guardrail outputs;
- final validation passed before closeout;
- Sprint 106 receives a concrete queue rather than hidden residual work.
