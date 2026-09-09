# Sprint 200 Day 1: Candidate Intake

## Purpose

Establish the Sprint 200 reliability scope, map project-plan items to initial
artifacts, inventory existing allocation-failure proof gates, and build a
candidate ledger before selecting exactly one owner on Day 2.

## Scope Trace

| Epic item | Day 1 interpretation |
| --- | --- |
| 200.1 Owner Selection | Rank candidates and defer the exact selection to Day 2. |
| 200.2 Invariant Record | Prepare invariant categories for cleanup, publication, stale output, retry, caller-owned input, and unsupported breadth. |
| 200.3 Harness Integration | Prefer existing deterministic allocation hooks; extend only for the selected owner. |
| 200.4 Regression Tests | Plan failed-allocation, cleanup, stale-output, retry, and caller-owned-input coverage. |
| 200.5 Focused Gate | Reuse the existing focused-gate and Python registration-guard pattern. |
| 200.6 Docs And Validation | Keep claims selected-owner-only and defer full validation until code or test changes exist. |

## Existing Gate Inventory

| Gate | Existing owner | Reuse in Sprint 200 |
| --- | --- | --- |
| `make iterative-allocation-failure-gate` | Iterative repeated-run handles. | Exclude as already covered; reuse global-hook reset discipline. |
| `make matmul-allocation-failure-gate` | `sparse_matmul()` workspace allocation. | Reuse stale-output and retry proof style. |
| `make symbolic-allocation-failure-gate` | Selected `sparse_symbolic_cholesky()` output allocation. | Reuse selected-owner proof structure and registration guard. |

## Prior Sprint 195 Handoff

Sprint 195 completed one selected symbolic Cholesky allocation-failure proof.
Its retrospective explicitly left these future reliability owners:

| Residual owner | Day 1 Sprint 200 interpretation |
| --- | --- |
| `sparse_symbolic_lu()` allocation failures | Leading candidate because it is a symbolic owner adjacent to the completed proof but not already covered. |
| `sparse_analyze()` allocation failures | Leading candidate because it owns analysis lifecycle state, but it may be too composed for one proof. |
| Standalone etree/postorder/colcount helpers | Candidate if Day 2 needs a narrower symbolic owner. |
| Direct solvers and matrix construction | Candidate class only; Day 2 must isolate one entry point if chosen. |
| OS OOM, concurrent allocation-hook behavior, and hosted CI ownership | Explicit non-goals for this selected-owner sprint unless separately scoped later. |

## Source Scan

The Day 1 source scan counted allocation tokens and cleanup/failure tokens as
rough signals only.

| Candidate source | Allocation signal | Cleanup/failure signal | Candidate reading |
| --- | ---: | ---: | --- |
| `src/sparse_ldlt_csc.c` | 58 | 118 | Allocation dense, but likely too broad without a narrow sub-owner. |
| `src/sparse_lu_csr.c` | 37 | 145 | Strong direct-solver candidate if scoped to one public entry point. |
| `src/sparse_qr.c` | 33 | 149 | High cleanup density, but likely high review cost. |
| `src/sparse_etree.c` | 30 | 145 | Strong symbolic candidate pool, especially symbolic LU or helper-level proof. |
| `src/sparse_ldlt.c` | 29 | 135 | Public direct-solver candidate with meaningful stale-output questions. |
| `src/sparse_chol_csc.c` | 28 | 71 | Possible direct Cholesky owner distinct from symbolic Cholesky. |
| `src/sparse_lu.c` | 18 | 96 | Possible narrow linked-list LU solve workspace owner. |
| `src/sparse_matrix.c` | 14 | 56 | High user impact but broad owner boundary risk. |
| `src/sparse_svd_partial.c` | 14 | 97 | Valuable but fixture and multi-output complexity may exceed this sprint. |
| `src/sparse_analysis.c` | 7 | 77 | Lower allocation signal, but Sprint 195 named it as an important residual. |

## Candidate Ledger

| Candidate owner | Strength | Concern | Day 2 input |
| --- | --- | --- | --- |
| `sparse_symbolic_lu()` | Explicit residual; symbolic output cleanup can likely reuse Sprint 195 patterns. | May compose several helper paths and matrix construction behavior. | Score first. |
| `sparse_analyze()` | High user-facing lifecycle value. | Composed analysis state may make owner boundary and stale-output semantics broad. | Score first with scope caution. |
| Helper-level etree/postorder/colcount owner | More bounded than full symbolic LU. | May be less user-visible and may not exercise public output publication. | Keep as fallback symbolic owner. |
| One direct-solver output publication owner | High user impact and stale-output value. | Must avoid claiming all direct solvers. | Consider only one public entry point. |
| LU CSR selected entry point | Dense allocation/cleanup surface. | Direct allocation and factor-growth breadth may add harness cost. | Score if symbolic owners are less feasible. |
| Linked-list LU solve workspace | Narrower than full LU factorization. | Caller-owned output mutation complicates stale-output expectations. | Keep as narrow solver fallback. |
| Core matrix constructor path | Highest common user surface. | Broad matrix API could overwhelm a selected proof. | Lower priority unless a single constructor boundary is obvious. |

## Day 2 Questions

1. Which candidate is not already covered and has the clearest selected-owner
   boundary?
2. Which candidate is reachable through existing deterministic allocation
   hooks with minimal wrapper conversion?
3. Which candidate has a clean stale-output or publication contract?
4. Which candidate can prove successful retry against existing fixtures?
5. Which candidate can receive a focused gate without broad source-list churn?

## Validation

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

`git diff --check` is the Day 1 validation command.
