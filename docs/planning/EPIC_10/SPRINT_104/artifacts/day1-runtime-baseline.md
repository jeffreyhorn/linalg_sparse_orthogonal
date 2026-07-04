# Sprint 104 Day 1 Runtime Baseline

## Purpose

Day 1 turns the Sprint 104 project-plan section into a bounded
backend/runtime modernization package. The baseline records workstream
ownership, evidence rules inherited from Sprint 100, Sprint 102-103 handoff
constraints, validation expectations, and claim boundaries before source,
benchmark, or workflow changes begin.

## Scope Summary

Sprint 104 owns backend and runtime modernization for:

- builtin dense kernels as the portable fallback baseline;
- optional acceleration and backend selection behavior;
- backend descriptor, status, and observability surfaces;
- OpenMP and thread-control behavior;
- bounded local performance sentinels;
- benchmark/reporting wording that discloses backend and runtime context;
- validation and closeout artifacts.

Sprint 104 does not own broad package-parity, portable timing superiority,
GPU/distributed backend support, or universal vendor backend parity.

## Workstream Ownership

| project-plan item | day-level ownership | primary output |
|---|---|---|
| Backend Consumer Audit | Days 1-2 | backend consumer inventory, fallback map, optional acceleration point list |
| Runtime Contract Design | Days 3-4 | runtime contract design and descriptor boundary artifacts |
| Backend Descriptor Batch | Day 5 | descriptor or selection-surface implementation plus focused validation |
| OpenMP and Threading Cleanup | Days 6-7 | threading audit, cleanup patch or no-change decision, runtime-control docs |
| Performance Sentinel Batch | Days 8-9 | sentinel design and bounded local regression sentinel implementation |
| Benchmark Reporting Alignment | Days 10-11 | benchmark reporting audit and wording/output alignment |
| Validation and Closeout | Days 12-14 | cross-platform review, validation reconciliation, closeout, Sprint 105 handoff |

## Day-by-Day Traceability

| day | focus | project-plan item coverage |
|---:|---|---|
| 1 | scope and runtime baseline | all items mapped to sprint artifacts and validation rules |
| 2 | backend consumer audit | Item 1 |
| 3 | runtime contract design | Item 2 |
| 4 | descriptor surface boundary | Items 2 and 3 |
| 5 | backend descriptor batch | Item 3 |
| 6 | OpenMP and threading audit | Item 4 |
| 7 | OpenMP and threading cleanup | Item 4 |
| 8 | performance sentinel design | Item 5 |
| 9 | performance sentinel batch | Item 5 |
| 10 | benchmark reporting audit | Item 6 |
| 11 | benchmark reporting alignment | Item 6 |
| 12 | cross-platform runtime review | Items 2, 4, 6, and 7 |
| 13 | validation reconciliation | Item 7 |
| 14 | closeout and handoff | Item 7 |

## Sprint 100 Evidence Rules Carried Forward

Sprint 100 requires future benchmark and sentinel work to record:

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

| evidence type | Sprint 104 interpretation |
|---|---|
| benchmark report | local measurement artifact unless a later artifact defines stronger scope |
| performance sentinel | narrow regression gate, not benchmark superiority |
| residual field in benchmark output | context only unless a test/oracle owns correctness |
| optional backend timing | backend- and machine-local evidence only |
| OpenMP timing | thread-context-specific evidence only |

## Sprint 102-103 Handoff Constraints

Sprint 102 and Sprint 103 leave Sprint 104 these constraints:

- comparison evidence must stay tied to named fixtures, commands, validation
  records, and non-claims;
- internal consistency checks must not be described as independent external
  oracles;
- iteration counts and local timings are diagnostics unless a future benchmark
  or sentinel artifact defines a threshold and machine class;
- optional-helper or optional-backend availability must have explicit skip or
  fallback behavior;
- public wording should cite maintained evidence owners and avoid broad
  ecosystem parity language.

## Runtime Contract Questions for Day 3

The Day 2 audit should feed Day 3 answers for:

1. What is the exact portable builtin fallback contract?
2. Which optional backends are selectable, observable, or intentionally hidden?
3. What happens when a requested backend is unavailable?
4. Which runtime choices belong in public diagnostics versus benchmark-only
   output?
5. How should serial builds, OpenMP builds, nested parallelism, and thread
   overrides behave?
6. Which state is process-global, thread-local, environment-controlled, or
   option-local?
7. Which behavior is API contract, implementation detail, or maintainer-only
   diagnostic?

## Validation Expectations

| touched surface | validation expectation |
|---|---|
| Sprint planning docs only | `git diff --check`; trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_104` |
| public docs only | `git diff --check`; trailing-whitespace scan on touched docs |
| benchmark docs/scripts | focused benchmark/report command where runnable; docs hygiene |
| benchmark C files only | focused benchmark build or run; code-touch gate if `.c` is modified |
| tests only | focused affected test; `make format && make lint && make test` when `.c` or `.h` changes |
| library source or public headers | focused affected tests; `make format && make lint && make test` |
| build/CMake/workflow files | focused build/configure/workflow-equivalent command plus any code-touch gate |

## Initial Artifact Structure

```text
docs/planning/EPIC_10/SPRINT_104/
├── PLAN.md
├── WORKING_NOTES.md
└── artifacts/
    ├── day1-authoritative-inputs.txt
    └── day1-runtime-baseline.md
```

## Day 1 Completion Check

| criterion | status |
|---|---|
| every Sprint 104 project-plan item has day-level ownership | complete |
| Sprint 100 evidence-template rules are visible in working notes | complete |
| Sprint 102-103 handoff constraints are captured | complete |
| validation expectations are explicit before implementation days | complete |
| artifacts directory exists | complete |

## Day 2 Starting Point

Day 2 should begin with the backend consumer audit. The first inventory pass
should inspect source, test, benchmark, example, and documentation surfaces for
dense kernels, backend descriptors, optional acceleration, OpenMP controls,
fallback behavior, and benchmark output fields.
