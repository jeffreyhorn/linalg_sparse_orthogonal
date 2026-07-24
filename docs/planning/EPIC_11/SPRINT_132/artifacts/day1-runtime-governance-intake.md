# Sprint 132 Day 1 - Runtime Governance Intake

## Purpose

Day 1 establishes Sprint 132 scope, artifact structure, validation lanes, and
non-claim fences around local performance, backend, OpenMP, and sentinel
evidence.

This is a documentation-only intake artifact. It does not change benchmark
code, sentinel scripts, report scripts, Makefile targets, generated report
schemas, maintainer wording, or public claims.

## Project-Plan Item Map

| Item | Sprint 132 project-plan item | Day owner |
| --- | --- | --- |
| 1 | Hot Path Inventory | Days 1-3 |
| 2 | Backend Runtime Contract | Days 4-5 |
| 3 | Sentinel Design | Days 3 and 6-7 |
| 4 | Sentinel Implementation Batch | Days 7-8 and 10-11 |
| 5 | Benchmark Docs Cleanup | Days 9 and 12 |
| 6 | Validation | Days 10-13 |
| 7 | Closeout | Days 12-14 |

## Authoritative Inputs

| Input | Role |
| --- | --- |
| Sprint 132 project-plan section | Defines sprint goal, seven work items, deliverables, and 168-hour budget. |
| Sprint 132 `PLAN.md` | Defines 14-day execution path and validation expectations. |
| Sprint 131 report-index artifacts | Preserve generated-versus-curated decisions, freshness labels, stale/missing semantics, and first-index guardrail boundaries. |
| Sprint 131 ownership and residual artifacts | Preserve owner labels, support-tier boundaries, supplemental-to-reviewed promotion criteria, and Sprint 132 handoff candidates. |
| `docs/maintainer_guide.md` | Current maintainer truth for OpenMP/runtime control, backend-aware surfaces, canonical benchmarks, sentinels, and guardrails. |
| `benchmarks/README.md` | Current benchmark command, CSV schema, backend context, sentinel, and non-claim documentation. |
| `Makefile` | Current build/report target owner for benchmark, sentinel, guardrail, OpenMP, quality, coverage, and dead-code flows. |

## Source-Area Intake

| Source area | Current role | Sprint 132 interpretation |
| --- | --- | --- |
| `scripts/performance_sentinels.sh` | Emits `sentinels.tsv`, manifest, wall-check output, and threshold-free Cholesky CSC rows. | Primary local sentinel bundle owner; local regression evidence only. |
| `scripts/wall_check.sh` | Existing bounded wall-check threshold runner. | Hard gate owner for current wall-check lane only. |
| `scripts/bench_canonical_report.sh` | Emits threshold-free canonical benchmark CSV bundle, `index.tsv`, and manifest. | Canonical local/CI-friendly snapshot owner; not pass/fail timing proof. |
| `scripts/large_matrix_guardrails.sh` | Emits large-matrix guardrail `index.tsv`, manifest, reviewed logs, and supplemental rows. | Guardrail report owner; supplemental lanes stay opt-in unless future policy promotes them. |
| `benchmarks/bench_refactor_csc.c` | CSC refactor and LDLT dense-backend request/selection/fallback benchmark. | Direct/backend hot-path candidate with existing backend observability fields. |
| `benchmarks/bench_chol_csc.c` | Cholesky CSC linked-list, CSC, supernodal, dense-kernel, and panel-solver benchmark. | Direct/backend sentinel candidate; current S2 threshold-free report source. |
| `benchmarks/bench_iterative_reuse.c` | Iterative reuse benchmark. | Iterative/preconditioner hot-path candidate. |
| `benchmarks/bench_eigs_reuse.c` and `benchmarks/bench_eigs.c` | Eigensolver reuse and backend benchmark surfaces. | Eigensolver backend/runtime hot-path candidates. |
| `benchmarks/bench_svd.c` | SVD benchmark surface. | SVD/partial-SVD hot-path candidate. |
| `benchmarks/bench_reorder.c` and `benchmarks/bench_amd_qg.c` | Reorder, qg-AMD, wall-check, sentinel, and guardrail benchmark surfaces. | Reorder/large-matrix sentinel and guardrail owners. |
| `benchmarks/bench_backend_compare_helpers.h` | Shared backend benchmark helper and residual measurement helper. | Shared helper owner for backend comparison benchmarks. |
| `Makefile` OpenMP/report targets | Owns compile-time OpenMP flags, benchmark report targets, sentinels, guardrails, and quality gates. | Runtime build-mode and validation target owner. |
| `docs/maintainer_guide.md` and `benchmarks/README.md` | Maintainer and benchmark interpretation docs. | Update only when accepted evidence supports wording changes. |

## Initial Validation Lanes

| Lane | Command | When required |
| --- | --- | --- |
| Docs hygiene | `git diff --check` and Sprint 132 markdown whitespace scan | Every Sprint 132 documentation-only day. |
| Sentinel report | `make performance-sentinels` | Sentinel script, report schema, metadata, or sentinel docs changes. |
| Canonical report | `make bench-canonical-report` | Canonical report script/schema/docs changes. |
| Guardrail report | `make large-matrix-guardrails` | Guardrail script/schema/docs changes or supplemental policy decisions. |
| Focused benchmark binary | Specific `build/bench_*` target and command | Benchmark C behavior, CLI, metadata, or CSV schema changes. |
| Full C quality | `make format && make lint && make test` | Any `.c` or `.h` file change. |
| OpenMP/runtime | `make omp` or documented unavailable-runtime check | OpenMP target, runtime, pragma, or thread-observability changes. |

## Duplicate Fences and Non-Claims

Sprint 132 must preserve these boundaries:

- local timing rows are not portable performance claims;
- canonical benchmark reports are threshold-free local snapshots;
- performance sentinels are local regression evidence, with hard pass/fail
  limited to existing accepted threshold lanes;
- backend request, selection, fallback, and dense-kernel labels are
  observability fields, not backend parity proof;
- OpenMP build mode and `OMP_NUM_THREADS` context do not create a public
  thread-control API;
- supplemental large-matrix lanes remain opt-in report context unless a future
  artifact defines runtime budget, support tier, and promotion policy;
- freshness labels mean report traceability, not CI, release, platform, or
  support guarantees;
- benchmark docs should explain interpretation, not imply scalability, memory,
  ecosystem, or state-of-the-art claims.

## Day 2 Handoff

Day 2 should inventory hot paths by source family and current report/sentinel
coverage. The first pass should include:

- direct/backend paths: Cholesky CSC, LDLT CSC, CSC refactor, dense backend
  request/selection/fallback;
- compressed/reorder paths: qg-AMD, reorder ND, graph/guardrail lanes;
- iterative paths: iterative reuse, BiCGSTAB, convergence surfaces;
- eigensolver paths: backend and reuse benchmarks;
- SVD paths: SVD benchmark coverage and partial-SVD residual handoff context;
- runtime context: OpenMP build mode, `OMP_NUM_THREADS`, dense backend env
  vars, compiler, platform, and report freshness.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 132 project-plan item has a day-level owner. | Complete | Item map and working-notes day-level ownership table map Items 1-7 to Days 1-14. |
| Sprint 131 report-index and non-claim boundaries are preserved. | Complete | Authoritative inputs and duplicate fences carry forward Sprint 131 report, freshness, owner, and non-claim boundaries. |
| Benchmark, sentinel, backend, OpenMP, and guardrail surfaces are visible before design or implementation begins. | Complete | Source-area intake and validation-lane tables identify the current owners and commands for those surfaces. |
