# Sprint 106 Retrospective

**Sprint:** 106 - Large-Source & Giant-Test Maintainability Phase 7
**Duration:** 14 days (Days 1-14 landed on branch `sprint-106`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 106 started from the Epic 10 project-plan scope and Sprint
      100-105 maintainability, evidence, runtime, and scalability handoffs.
- [x] large source and giant-test owners were re-ranked from the live tree
      before extraction work started.
- [x] the highest-value CSC direct-family extraction boundary was frozen before
      source edits.
- [x] LDLT CSC row-adjacency construction moved into a focused private source
      owner without changing public APIs.
- [x] LDLT CSC proof follow-through preserved direct solver behavior and kept
      the new helper private.
- [x] secondary extraction scope selected QR Householder helpers and LU CSR
      structural helper ownership as narrow, reviewable seams.
- [x] QR Householder helper construction moved into a focused private source
      owner.
- [x] LU CSR structural allocation and cleanup moved into a focused private
      source owner.
- [x] giant-test fixture boundaries were defined before moving shared graph,
      direct-solver, and integration helpers.
- [x] graph, reorder, direct-solver, and integration test helpers were split
      into reusable test-only owners without changing reviewed test
      registration.
- [x] Makefile, CMake, and source-list checker ownership were reconciled after
      extracted library source files landed.
- [x] maintainer guidance now documents the Sprint 106 source and test helper
      ownership layout.
- [x] final validation passed:
  - `make format && make lint && make test`
  - `python3 scripts/check_library_sources.py`
  - `make quality-review-cmake-compile`
  - `make large-matrix-guardrails`
  - `git diff --check`
  - trailing-whitespace scans
- [x] Sprint 107 handoff items are explicit and bounded.

## What Went Well

1. **The sprint treated extraction as a proof problem.**
   Day 2 re-ranked live owners and Day 3 froze the CSC boundary before edits.
   That kept extraction work tied to failure-localization value instead of line
   count alone.

2. **The direct-solver splits stayed private and narrow.**
   LDLT row-adjacency, QR Householder construction, and LU CSR structural
   helpers moved into focused owners while public headers and install surfaces
   stayed unchanged.

3. **Build-system follow-through stayed exact.**
   The source-list checker moved from 42 to 45 tracked library sources, and
   Make/CMake reviewed test registration stayed at 54 tests on both surfaces.

4. **Test helper extraction reduced large proof-owner pressure.**
   Graph, reorder, LU CSR, and integration owners all got smaller while helper
   files remained test-only and did not add compiled helper targets or public
   install headers.

5. **Maintainer guidance caught up with the new ownership model.**
   Day 13 updated `docs/maintainer_guide.md` so future maintainers know where
   row-adjacency, QR Householder, LU CSR structural, graph fixture, direct
   solver helper, and integration fixture changes should go.

6. **Final validation matched the branch risk.**
   Because Sprint 106 touched C sources, headers, build lists, tests, and docs,
   Day 14 reran the full C gate plus source-list, CMake compile/parity, and
   large-matrix guardrail validation.

## What Didn't Go Well

1. **Several largest owners remain large.**
   Sprint 106 reduced selected owners, but `tests/test_ldlt_csc.c`,
   `tests/test_qr.c`, `tests/test_iterative.c`, `tests/test_svd.c`,
   `src/sparse_eigs.c`, and `src/sparse_matrix.c` still need careful future
   boundaries.

2. **The source reductions are intentionally modest.**
   The sprint prioritized low-risk private seams over broad rewrites. That
   means the largest implementation owners improved, but none became small
   modules in one sprint.

3. **Test helper extraction required conservative naming and ownership.**
   Shared fixtures are useful, but broad fixture movement can hide assertion
   intent. Sprint 106 kept helper movement narrow and left several broader
   fixture cleanups deferred.

4. **Every extracted library source increases build-list maintenance.**
   Adding three private `.c` files improved ownership, but it also expanded the
   Make/CMake/source-list surface that future maintainers must keep in sync.

5. **Source extraction did not address central matrix-shell risk.**
   `src/sparse_matrix.c` remains too central for opportunistic cleanup and
   needs API/compatibility review before any future split.

## Final Metrics

### Validation

| Metric | Sprint 106 close state |
|---|---:|
| full branch-level gate | `make format && make lint && make test` passed |
| source-list checker | `source-list-check: PASS (45 library sources)` |
| reviewed CMake compile/parity | passed |
| reviewed CMake tests registered | 54 |
| reviewed Makefile tests counted | 54 |
| large-matrix guardrails | passed |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scans | passed on touched docs and Sprint 106 artifacts |

### Source Owner Movement

| owner | Sprint 106 baseline | final | delta |
|---|---:|---:|---:|
| `src/sparse_ldlt_csc.c` | 2,174 | 2,095 | -79 |
| `src/sparse_qr.c` | 1,563 | 1,448 | -115 |
| `src/sparse_lu_csr.c` | 1,665 | 1,594 | -71 |
| `src/sparse_ldlt_csc_rowadj.c` | 0 | 82 | +82 |
| `src/sparse_qr_householder.c` | 0 | 79 | +79 |
| `src/sparse_qr_internal.h` | 0 | 16 | +16 |
| `src/sparse_lu_csr_struct.c` | 0 | 57 | +57 |
| `src/sparse_lu_csr_internal.h` | 0 | 9 | +9 |

### Test Owner Movement

| owner | Sprint 106 baseline | final | delta |
|---|---:|---:|---:|
| `tests/test_graph.c` | 2,925 | 2,758 | -167 |
| `tests/test_reorder_nd.c` | 2,340 | 2,304 | -36 |
| `tests/test_lu_csr.c` | 1,899 | 1,806 | -93 |
| `tests/test_integration.c` | 3,421 | 3,279 | -142 |
| `tests/test_graph_fixtures.h` | 0 | 195 | +195 |
| `tests/test_direct_solver_helpers.h` | 0 | 93 | +93 |
| `tests/test_integration_fixtures.h` | 0 | 140 | +140 |

### Build and Review Surfaces

| surface | baseline | final |
|---|---:|---:|
| library sources tracked by source-list checker | 42 | 45 |
| reviewed Makefile test binaries | 54 | 54 |
| reviewed CMake tests | 54 | 54 |
| new compiled test helper targets | 0 | 0 |
| new public/install headers | 0 | 0 |

### Sprint 106 Artifact Package

| Metric | Sprint 106 close state |
|---|---:|
| artifact files under `SPRINT_106/artifacts/` | 15 |
| baseline/ranking/boundary artifacts | 4 |
| implementation and proof-follow-through artifacts | 4 |
| fixture/build/docs/validation artifacts | 7 |

Notes:

- baseline, ranking, and boundary artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-maintainability-baseline.md`
  - `day2-extraction-target-rerank.md`
  - `day3-ldlt-csc-extraction-boundary.md`
- implementation and proof-follow-through artifacts:
  - `day4-ldlt-csc-extraction-batch.md`
  - `day5-ldlt-csc-proof-follow-through.md`
  - `day6-secondary-extraction-boundary.md`
  - `day7-qr-householder-extraction.md`
- fixture, build, documentation, validation, and closeout artifacts:
  - `day8-lu-csr-structural-extraction.md`
  - `day9-giant-test-fixture-boundary.md`
  - `day10-direct-graph-fixture-extraction.md`
  - `day11-integration-oracle-fixture-extraction.md`
  - `day12-source-list-cmake-reconciliation.md`
  - `day13-maintainer-guidance-and-metrics.md`
  - `day14-validation-closeout.md`

## Residual Deferred Debt

Most important carry-forward work:

- `tests/test_ldlt_csc.c` remains the largest direct CSC proof owner; extract
  only one row-adjacency assertion or residual/oracle helper after a narrow
  boundary artifact.
- `tests/test_qr.c` still has broad proof fixtures; extract repeated
  matrix/vector builders without hiding solve and reconstruction intent.
- `tests/test_iterative.c` still needs convergence-sensitive helper cleanup;
  start with reusable matrix/RHS builders only.
- `tests/test_svd.c` needs a dedicated SVD proof-owner cleanup with focused
  validation before moving rank/oracle helpers.
- `src/sparse_eigs.c` remains tied to Sprint 103 comparison surfaces and needs
  a fresh boundary before splitting workspace or dispatch helpers.
- `src/sparse_matrix.c` remains central API/compatibility territory and should
  not be split opportunistically.

Still consciously constrained rather than silently solved:

- no public API or install-header change;
- no new compiled test helper target;
- no reviewed test-count change;
- no broad direct-solver rewrite;
- no broad QR, iterative, eigensolver, or SVD proof-owner cleanup;
- no central sparse matrix shell extraction.

Not carried forward as unresolved Sprint 106 debt:

- live maintainability re-rank;
- LDLT CSC row-adjacency extraction;
- QR Householder helper extraction;
- LU CSR structural helper extraction;
- graph and reorder fixture extraction;
- direct-solver helper extraction;
- integration fixture extraction;
- source-list and CMake reconciliation;
- maintainer guidance update;
- final validation and closeout.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-maintainability-baseline.md](./artifacts/day1-maintainability-baseline.md)
- [day2-extraction-target-rerank.md](./artifacts/day2-extraction-target-rerank.md)
- [day3-ldlt-csc-extraction-boundary.md](./artifacts/day3-ldlt-csc-extraction-boundary.md)
- [day4-ldlt-csc-extraction-batch.md](./artifacts/day4-ldlt-csc-extraction-batch.md)
- [day5-ldlt-csc-proof-follow-through.md](./artifacts/day5-ldlt-csc-proof-follow-through.md)
- [day6-secondary-extraction-boundary.md](./artifacts/day6-secondary-extraction-boundary.md)
- [day7-qr-householder-extraction.md](./artifacts/day7-qr-householder-extraction.md)
- [day8-lu-csr-structural-extraction.md](./artifacts/day8-lu-csr-structural-extraction.md)
- [day9-giant-test-fixture-boundary.md](./artifacts/day9-giant-test-fixture-boundary.md)
- [day10-direct-graph-fixture-extraction.md](./artifacts/day10-direct-graph-fixture-extraction.md)
- [day11-integration-oracle-fixture-extraction.md](./artifacts/day11-integration-oracle-fixture-extraction.md)
- [day12-source-list-cmake-reconciliation.md](./artifacts/day12-source-list-cmake-reconciliation.md)
- [day13-maintainer-guidance-and-metrics.md](./artifacts/day13-maintainer-guidance-and-metrics.md)
- [day14-validation-closeout.md](./artifacts/day14-validation-closeout.md)
