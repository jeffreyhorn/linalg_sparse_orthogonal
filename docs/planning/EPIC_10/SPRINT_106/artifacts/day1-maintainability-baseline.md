# Sprint 106 Day 1 Maintainability Baseline

## Purpose

Day 1 converts the Sprint 106 project-plan section and prior Epic 10 handoffs
into a bounded extraction package. The sprint should reduce large-source and
giant-test risk by improving ownership and failure localization, not by
rewriting solver families or widening product claims.

## Sprint 106 Workstreams

| workstream | project-plan item | day ownership | primary output |
|---|---|---|---|
| extraction target re-rank | Item 1 | Days 1-2 | live ranked source/test owner queue |
| LDLT/CSC source extraction | Item 2 | Days 3-5 | extracted direct CSC helper owner and proof follow-through |
| LU/QR/eigs/iterative extraction | Item 3 | Days 6-8 | one or two focused secondary solver helper owners |
| giant-test fixture extraction | Item 4 | Days 9-11 | reusable direct, graph, integration, or oracle test helpers |
| source-list and CMake follow-through | Item 5 | Days 3-12 | exact Make/CMake/source-list parity after extraction |
| maintainability documentation | Item 6 | Days 13-14 | maintainer guidance and before/after ownership metrics |
| validation and closeout | Item 7 | Days 1-14 | required checks, closeout artifact, and Sprint 107 handoff |

Every project-plan item has day-level ownership. Day 2 owns the live re-rank
before implementation. Days 3-12 own source, test, and build-system
extraction. Days 13-14 own metrics, documentation, validation, and handoff.

## Starting Evidence

Sprint 100 recorded the initial maintainability baseline:

| category | highest-pressure owners from Sprint 100 |
|---|---|
| largest test owner | `tests/test_ldlt_csc.c` |
| next giant tests | `tests/test_integration.c`, `tests/test_qr.c`, `tests/test_ldlt.c`, `tests/test_etree.c`, `tests/test_graph.c`, `tests/test_iterative.c`, `tests/test_svd.c`, `tests/test_chol_csc.c`, `tests/test_chol_csc_supernodal.c`, `tests/test_reorder_nd.c` |
| largest source owner | `src/sparse_ldlt_csc.c` |
| next source hotspots | `src/sparse_lu_csr.c`, `src/sparse_qr.c`, `src/sparse_ldlt.c`, `src/sparse_eigs.c`, `src/sparse_iterative.c`, `src/sparse_matrix.c`, `src/sparse_svd.c`, `src/sparse_chol_csc.c`, `src/sparse_lu.c` |
| source membership drift risk | `build-metadata/library_sources.txt`, `Makefile`, `CMakeLists.txt`, and `scripts/check_library_sources.py` must remain synchronized |

Day 2 must refresh these rankings from the live tree because Sprint 101-105
changed source, tests, documentation, scripts, and build surfaces.

## Prior Sprint Handoff Constraints

### Sprint 102 Direct Solver Handoff

- Shared direct-solver helper and external-reference parsing work already
  exists; Sprint 106 should reuse or respect it instead of creating parallel
  helper conventions.
- LDLT CSC and linked-list LU have fixture-named oracle evidence; extraction
  must not weaken those proof owners.
- QR, SVD, LU CSR, and dispatch oracle expansion remain deferred comparison
  work, not automatic Sprint 106 scope.

### Sprint 103 Comparison Handoff

- Iterative, eigensolver, and SVD comparison evidence is bounded to named
  fixtures and maintained test owners.
- Source extraction in these families must preserve residual, tolerance, and
  non-claim wording.
- External package parity remains deferred and should not be implied by helper
  movement.

### Sprint 104 Runtime and Build Handoff

- Builtin fallback behavior remains the portable baseline.
- Optional backend and OpenMP wording must stay bounded by maintained tests and
  docs.
- Adding or moving tests requires explicit CTest registration and reviewed
  surface awareness.
- Source extraction must keep Make/CMake/source-list parity exact.

### Sprint 105 Reorder/Graph Handoff

- Reorder/fill, graph, and large-matrix evidence is separated into reviewed,
  supplemental, local-only, and non-claim lanes.
- Remaining graph/reorder history-heavy owners include `tests/test_graph.c`,
  `tests/test_reorder_nd.c`, `src/sparse_graph.c`, and
  `src/sparse_reorder_nd.c`.
- `make large-matrix-guardrails` and its supplemental lanes should remain
  bounded if graph/reorder tests are refactored.

## Day 2 Audit Starting Queue

Day 2 should re-rank, at minimum, these candidate areas:

| candidate | why it starts in scope | Day 2 decision needed |
|---|---|---|
| `src/sparse_ldlt_csc.c` | largest implementation owner in Sprint 100 and explicit Sprint 106 item | choose a narrow CSC helper seam for Days 3-5 |
| `tests/test_ldlt_csc.c` | largest proof owner and closest proof surface to the CSC extraction | decide which helpers should remain local and which can be shared |
| `src/sparse_lu_csr.c` | large LU CSR implementation owner and residual direct-solver comparison candidate | decide whether source extraction is lower risk than test/oracle expansion |
| `src/sparse_qr.c` and `tests/test_qr.c` | large QR implementation and proof owner with deferred QR oracle work | decide whether Sprint 106 should touch QR source, fixtures, or defer |
| `src/sparse_eigs.c` and spectral tests | large eigensolver orchestration owner with Sprint 103 evidence | decide if helper extraction can preserve comparison evidence cleanly |
| `src/sparse_iterative.c` and `tests/test_iterative.c` | large iterative owner and giant test with convergence/failure evidence | decide if extraction improves failure localization without claim drift |
| graph/reorder tests and sources | Sprint 105 residual history-heavy cleanup queue | decide whether fixture extraction materially improves maintainability |
| integration/oracle helpers | large integration owner and cross-family helper pressure | decide helper boundaries that keep call-site intent readable |

## Validation Matrix

| day type | required checks |
|---|---|
| docs-only planning day | `git diff --check`; trailing-whitespace scan on touched planning files |
| source extraction day | focused affected tests; `python3 scripts/check_library_sources.py`; `make format && make lint && make test` |
| header extraction day | focused affected tests; source-list check if library membership changes; `make format && make lint && make test` |
| test helper extraction day | focused affected test binaries; CTest registration check if registration changes; `make format && make lint && make test` |
| build-system follow-through day | source-list checker plus focused Make/CMake configure or build check; full C gate if code changed |
| mixed extraction day | all focused family checks plus full C quality gate |
| closeout day after code changes | full required gate, source-list/CMake reconciliation, docs hygiene, and before/after metric reconciliation |

## Non-Goals

Sprint 106 does not attempt to:

- redesign LDLT, LU, QR, eigensolver, iterative, graph, or reorder algorithms;
- introduce new public solver APIs as a side effect of extraction;
- change benchmark, runtime, or external-comparison claims;
- make all large files small in one sprint;
- add unplanned test registration changes or widen reviewed CTest scope;
- replace prior oracle, benchmark, or guardrail evidence with line-count
  metrics.

## Day 1 Exit Criteria

Day 1 is complete when:

- the Sprint 106 artifacts directory exists;
- authoritative inputs are recorded;
- workstream ownership maps every project-plan item to days;
- Sprint 102-105 handoff constraints are visible before extraction;
- validation expectations are explicit before Day 2 audit work begins;
- planning-document hygiene checks pass.
