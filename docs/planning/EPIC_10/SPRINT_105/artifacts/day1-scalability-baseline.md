# Sprint 105 Day 1 Scalability Baseline

## Purpose

Day 1 turns the Sprint 105 project-plan section into a bounded reorder, graph,
fill, and large-matrix scalability package. The baseline records workstream
ownership, evidence rules inherited from Sprint 100, Sprint 104 handoff
constraints, validation expectations, and claim boundaries before source,
benchmark, script, or workflow changes begin.

## Scope Summary

Sprint 105 owns scalability evidence for:

- AMD, COLAMD, nested-dissection, quotient-graph, graph partition, and
  separator surfaces;
- fill metrics, fill ratios, runtime context, memory proxies, and fixture
  naming fields for reorder artifacts;
- selected named matrices already suitable for maintained evidence;
- deterministic generated graph families;
- bounded large-matrix guardrails for memory, runtime, overflow, recursion,
  and pathological-fill risks;
- graph/reorder proof-owner cleanup where touched files have accumulated
  history-heavy comments or duplicated helpers;
- user and maintainer documentation for interpreting reorder/fill reports.

Sprint 105 does not own broad solver replacement claims, portable timing
superiority, vendor-package parity, GPU/distributed/out-of-core graph
processing, or unbounded large-matrix scalability.

## Workstream Ownership

| project-plan item | day-level ownership | primary output |
|---|---|---|
| Reorder/Graph Audit | Days 1-2 | owner inventory, ranked reorder/graph/fill gap queue |
| Fill Metrics Contract | Days 3-5 | canonical fill, runtime, memory, and fixture field contract |
| Named Matrix Expansion | Days 4-6 | selected named-matrix evidence and focused validation |
| Scalability Guardrails | Days 8-9 | deterministic large-matrix guardrail design and implementation |
| Graph Ownership Cleanup | Days 10 and 13 | helper/comment cleanup and focused validation |
| Reporting and Docs | Days 11-12 | reporting guidance, maintainer docs, integrated evidence package |
| Validation and Closeout | Days 12-14 | validation reconciliation, final fix batch, closeout, Sprint 106 handoff |

## Day-by-Day Traceability

| day | focus | project-plan item coverage |
|---:|---|---|
| 1 | scope and scalability baseline | all items mapped to sprint artifacts and validation rules |
| 2 | reorder and graph surface audit | Item 1 |
| 3 | fill and fixture contract design | Item 2 |
| 4 | evidence boundary and matrix selection | Items 2 and 3 |
| 5 | reorder/fill reporting batch 1 | Item 2 |
| 6 | named-matrix evidence expansion | Item 3 |
| 7 | generated graph-family expansion | Items 3 and 4 |
| 8 | large-matrix guardrail design | Item 4 |
| 9 | scalability guardrail implementation | Item 4 |
| 10 | graph/reorder ownership cleanup | Item 5 |
| 11 | reporting and documentation alignment | Item 6 |
| 12 | integrated evidence reconciliation | Items 3, 4, 6, and 7 |
| 13 | final fix batch and validation sweep | Items 5 and 7 |
| 14 | closeout and handoff | Item 7 |

## Sprint 100 Evidence Rules Carried Forward

Sprint 100 requires future benchmark, coverage, and performance artifacts to
record:

- exact command and owner;
- fixture or source scope;
- machine, compiler, backend, and thread context where relevant;
- emitted metrics and units;
- artifact paths and output format;
- threshold state, if any;
- reviewed versus supplemental status;
- unsupported or skipped cases;
- explicit non-claims.

The distinction between evidence types remains mandatory:

| evidence type | Sprint 105 interpretation |
|---|---|
| fill report | structural measurement artifact, not correctness proof by itself |
| local runtime row | command/fixture/backend/thread-context evidence only |
| memory proxy | bounded guardrail or diagnostic unless a contract defines stronger meaning |
| generated graph family | deterministic structural coverage, not a proxy for every sparse workload |
| performance sentinel | narrow regression gate, not benchmark superiority |
| benchmark residual or agreement field | context only unless a test/oracle owns correctness |

## Sprint 104 Handoff Constraints

Sprint 104 leaves Sprint 105 these constraints:

- benchmark and sentinel claim boundaries must be preserved when touching
  reorder, graph, or large-matrix evidence;
- local timing must remain tied to command, fixture, backend, thread context,
  and platform;
- `performance-sentinels` hard-fail behavior remains limited to S5 unless a
  new baseline design justifies more thresholds;
- new test additions should account for POSIX CMake registration and Windows
  expected CTest count decisions;
- OpenMP ownership wording should stay near any new parallel region or
  benchmark interpretation;
- optional dense backend language must not widen without tests, docs, and
  platform-scope updates.

## Reorder and Graph Questions for Day 2

The Day 2 audit should answer:

1. Which source, test, benchmark, and docs files own AMD, COLAMD, nested
   dissection, quotient graph, graph partition, and separator behavior?
2. Which current outputs report fill counts, fill ratios, runtime, memory, or
   skipped lanes?
3. Which fixture names are stable enough for aggregation across report and
   skip rows?
4. Which named matrices and generated families already have maintained owners?
5. Which graph/reorder files contain history-heavy comments or duplicated
   helpers worth cleaning only if touched?
6. Which large-matrix risks are deterministic enough for reviewed guardrails?
7. Which lanes must remain supplemental or local-only?

## Validation Expectations

| touched surface | validation expectation |
|---|---|
| Sprint planning docs only | `git diff --check`; trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_105` |
| public docs only | `git diff --check`; trailing-whitespace scan on touched docs |
| benchmark docs/scripts | focused benchmark/report command where runnable; docs hygiene |
| benchmark C files only | focused benchmark build or run; code-touch gate if `.c` is modified |
| tests only | focused affected test; `make format && make lint && make test` when `.c` or `.h` changes |
| library source or public headers | focused affected tests; `make format && make lint && make test` |
| build/CMake/workflow files | focused build/configure/workflow-equivalent command plus any code-touch gate |

## Initial Artifact Structure

```text
docs/planning/EPIC_10/SPRINT_105/
├── PLAN.md
├── WORKING_NOTES.md
└── artifacts/
    ├── day1-authoritative-inputs.txt
    └── day1-scalability-baseline.md
```

## Day 1 Completion Check

| criterion | status |
|---|---|
| every Sprint 105 project-plan item has day-level ownership | complete |
| Sprint 100 evidence-template rules are visible in working notes | complete |
| Sprint 104 runtime/benchmark handoff constraints are captured | complete |
| validation expectations are explicit before audit work begins | complete |
| artifacts directory exists | complete |

## Day 2 Starting Point

Day 2 should begin with the reorder and graph surface audit. The first
inventory pass should inspect source, tests, benchmarks, generated fixtures,
documentation, and planning artifacts for AMD, COLAMD, nested dissection,
quotient graph, graph partition, separator, fill-report, runtime, memory, and
large-matrix guardrail ownership.
