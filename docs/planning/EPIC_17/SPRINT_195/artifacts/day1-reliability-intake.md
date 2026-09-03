# Sprint 195 Day 1: Reliability Intake

## Purpose

Establish the Sprint 195 scope, map project-plan items to owner surfaces,
inventory allocation-heavy and failure-prone candidates, and identify reusable
deterministic failure-injection patterns before selecting one owner.

## Scope Trace

| Epic item | Day 1 intake interpretation |
| --- | --- |
| 195.1 Owner Selection | Rank one allocation-heavy or failure-prone owner by allocation density, cleanup complexity, user impact, deterministic hook fit, and current test gaps. |
| 195.2 Invariant Record | Define cleanup ownership, output publication, stale-output behavior, retry semantics, global-state restoration, and unsupported breadth before implementation. |
| 195.3 Harness Extension | Reuse `sparse_alloc_test_fail_after(...)` and `sparse_alloc_test_reset()` where possible, or add a narrow owner-local deterministic fail-at-count mechanism. |
| 195.4 Regression Tests | Add selected-owner tests for allocation failure, cleanup, stale-output suppression, and successful retry behavior. |
| 195.5 Focused Gate And Docs | Add a focused Make/CTest gate and maintainer/user wording that states the exact selected proof and retained non-claims. |
| 195.6 Validation | Run the focused gate, source-list checks as needed, `make format && make lint && make test` after C/H edits, and relevant docs checks. |

## Baseline Evidence

| Source | Finding |
| --- | --- |
| Sprint 187 reliability gates | Sprint 195 must select exactly one owner and prove failure status, cleanup, stale-output behavior, retry, and global-state restoration without claiming exhaustive reliability. |
| Sprint 187 owner inventory | Existing reliability evidence is limited to iterative repeated-run handles and `sparse_matmul()` workspace allocation. |
| Sprint 193 invariant contract | Process-global state restoration before assertion-macro early returns remains a live review requirement. |
| `src/sparse_alloc_internal.*` | Existing private fail-after hooks control internal allocation wrappers and reset process-global hook state. |
| `Makefile` / `CMakeLists.txt` | Focused allocation-failure gates and CTest labels already exist for iterative and matmul proof lanes. |
| `docs/maintainer_guide.md` | Maintainer docs already preserve narrow reliability claims and broad allocation-failure non-claims. |

## Allocation and Cleanup Signals

Approximate source-scan results for Day 2 ranking:

| Candidate owner | Allocation signal | Cleanup/failure signal | Interpretation |
| --- | ---: | ---: | --- |
| `src/sparse_ldlt_csc.c` | 55 | 29 | Highest allocation density; strong payoff but high algorithmic and scope-control risk. |
| `src/sparse_lu_csr.c` | 37 | 36 | High allocation and cleanup density; good candidate if narrowed to one public entry point. |
| `src/sparse_qr.c` | 33 | 37 | High cleanup density and user impact; recent QR work raises review risk. |
| `src/sparse_etree.c` | 30 | 32 | Structural owner with many partial-state paths and likely contained proof potential. |
| `src/sparse_ldlt.c` | 29 | 19 | Public direct-solver owner; output and retry semantics need tracing. |
| `src/sparse_chol_csc.c` | 19 | 25 | Public SPD direct-solver owner with existing correctness evidence. |
| `src/sparse_lu.c` | 18 | 15 | Public LU solve/factor workspace owner with possible narrower solve-lane proof. |
| `src/sparse_matrix.c` | 14 | 22 | Highest general user impact, but broad constructor/insertion/conversion coverage could exceed scope. |

## Prior Pattern Map

| Pattern | Current owner | Reuse in Sprint 195 |
| --- | --- | --- |
| Fail-after hook | `src/sparse_alloc_internal.c` | Use for wrapped allocation sites; reset before assertions can early-return. |
| Repeated-run handle proof | `tests/test_iterative_handle_helpers.h` | Model handle state preservation and retry-after-reset assertions. |
| Workspace stale-output proof | `tests/test_matmul.c` | Model fail-site table, stale-output assertions, and focused gate registration. |
| Hook smoke tests | `tests/test_sparse_matrix.c` | Model countdown/reset semantics without creating public API claims. |
| Focused gate registration | `make matmul-allocation-failure-gate` plus its Python registration guard | Use similar guard if Sprint 195 adds a new focused owner gate. |

## Initial Candidate Disposition

Day 2 should start from these candidates:

1. `src/sparse_lu_csr.c` public LU CSR factor/solve lane.
2. `src/sparse_etree.c` or analysis-adjacent symbolic owner lane.
3. Linked-list LU solve workspace lane in `src/sparse_lu.c`.
4. Cholesky CSC selected factor/solve lane in `src/sparse_chol_csc.c`.
5. A narrow core sparse matrix constructor path in `src/sparse_matrix.c`.

The default Day 1 exclusions are broad QR, broad LDLT CSC, broad sparse
matrix construction/insertion/conversion, broad SVD, and any multi-owner
allocation proof. Those may be reconsidered only if Day 2 finds a narrow,
deterministic, selected-owner lane.

## Day 2 Questions

1. Which candidate has enough wrapper-controlled allocation sites to avoid a
   broad allocator conversion?
2. Which candidate has the clearest stale-output or partial-publication
   contract?
3. Which candidate can prove retry success against existing fixtures or
   oracles?
4. Which candidate can use an existing test binary or small focused gate with
   low registration churn?
5. Which candidate most improves user-visible reliability confidence while
   preserving selected-owner-only claim boundaries?

## Validation

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

`git diff --check` passes.
