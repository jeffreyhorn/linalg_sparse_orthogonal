# Sprint 210 Day 1: Allocation Proof Intake

## Purpose

Establish the Sprint 210 selected allocation-failure proof scope, map Epic 19
items to initial evidence, inventory inherited Sprint 200 proof assumptions,
and build a candidate owner ledger before selecting exactly one owner on Day 2.

## Scope Trace

| Epic item | Day 1 interpretation |
| --- | --- |
| 210.1 Owner Selection | Rank Epic 19 candidate families and defer exact owner selection to Day 2. |
| 210.2 Lifecycle Invariant Record | Prepare status, cleanup, stale-output, caller-input, retry, and partial-publication invariant categories. |
| 210.3 Harness Extension | Prefer existing deterministic allocation hooks; extend only for the selected owner. |
| 210.4 Regression Tests | Plan failed-allocation, cleanup, stale-output, preservation, and retry coverage. |
| 210.5 Gate And Documentation | Reuse focused-gate and registration-guard patterns once the selected owner is known. |
| 210.6 Validation And Closeout | Keep Day 1 documentation-only; defer focused, family, source-list, docs, and full C quality checks until implementation changes exist. |

## Inherited Sprint 200 Assumptions

| Sprint 200 evidence | Sprint 210 reuse |
| --- | --- |
| Selected `sparse_symbolic_lu()` allocation-failure owner proof is closed. | Exclude symbolic LU from Sprint 210 owner selection. |
| `symbolic-lu-allocation-failure-gate` and registration guard exist. | Reuse focused-gate and registration-guard structure if Sprint 210 needs a new gate. |
| Sprint 200 proved deterministic allocation status, cleanup, stale-output suppression, caller-input preservation, repeated cleanup, and retry after reset for one selected owner. | Reuse the same invariant categories and reset discipline. |
| Sprint 200 residuals preserved broad allocation-failure, direct-solver, eigensolver, SVD, matrix construction/conversion/IO, package/install, hosted, platform, OS OOM, concurrent hook, release, and state-of-the-art non-claims. | Keep those non-claims explicit until Sprint 210 closes exactly one new owner. |

## Existing Gate Inventory

| Gate | Existing owner | Day 1 disposition |
| --- | --- | --- |
| `make iterative-allocation-failure-gate` | Iterative repeated-run handle allocation behavior. | Already covered; not a Sprint 210 candidate. |
| `make matmul-allocation-failure-gate` | `sparse_matmul()` workspace allocation behavior. | Already covered; reuse stale-output and retry proof style. |
| `make symbolic-allocation-failure-gate` | Selected symbolic Cholesky allocation behavior. | Already covered; reuse selected-owner gate pattern. |
| `make symbolic-lu-allocation-failure-gate` | Selected `sparse_symbolic_lu()` allocation behavior. | Closed by Sprint 200; not a Sprint 210 candidate. |

## Source Scan

The scan counted allocation-token lines and cleanup/failure-token lines as
rough Day 1 ranking signals only. These counts do not prove owner boundaries.

| Candidate source | Allocation signal | Cleanup/failure signal | Candidate reading |
| --- | ---: | ---: | --- |
| `src/sparse_ldlt_csc.c` | 58 | 118 | Strong LDLT/Cholesky numeric factorization candidate if narrowed. |
| `src/sparse_qr.c` | 33 | 149 | Strong QR workspace candidate with high cleanup density and review-surface risk. |
| `src/sparse_ldlt.c` | 29 | 135 | Public LDLT candidate with visible user impact. |
| `src/sparse_chol_csc.c` | 28 | 71 | Direct CSC Cholesky candidate distinct from symbolic proofs. |
| `src/sparse_svd_partial.c` | 14 | 97 | SVD workspace candidate with multi-output complexity. |
| `src/sparse_matrix.c` | 14 | 56 | Matrix construction/conversion candidate with high impact and broad-boundary risk. |
| `src/sparse_svd.c` | 13 | 104 | SVD candidate with output cleanup and numerical retry concerns. |
| `src/sparse_eigs_thick_restart.c` | 10 | 39 | Eigensolver workspace candidate with selected-algorithm potential. |
| `src/sparse_eigs.c` | 6 | 37 | Public eigensolver candidate, likely composed with workspace helpers. |
| `src/sparse_cholesky.c` | 6 | 60 | Public Cholesky candidate, possibly delegated. |
| `src/sparse_matrix_io.c` | 1 | 14 | Matrix import/export candidate with lower allocation density. |

## Candidate Ledger

| Candidate owner | Strength | Concern | Day 2 input |
| --- | --- | --- | --- |
| Selected LDLT CSC numeric factorization owner | Highest allocation signal and meaningful direct-solver reliability value. | Must avoid broad LDLT or all-direct-solver claims. | Score first. |
| Selected QR workspace owner | High cleanup density and strong least-squares/QR user value. | Recent QR review surface and multi-path behavior may increase churn. | Score first with review-surface caution. |
| Selected Cholesky CSC numeric owner | Distinct from symbolic Cholesky and likely fixture-friendly. | Must avoid broad Cholesky correctness or symbolic proof claims. | Score first if owner boundary is precise. |
| Selected SVD workspace owner | Important multi-output numerical surface. | Stale-output and retry evidence may be numerically complex. | Keep as candidate if a narrow helper/output owner is obvious. |
| Selected eigensolver workspace owner | Important advanced-user surface with rich failure semantics. | Handle/caller-buffer lifecycle may be composed and broad. | Keep as candidate with lifecycle caution. |
| Matrix construction/conversion owner | High common user impact. | Easy to overclaim broad sparse matrix allocation reliability. | Consider only one constructor/conversion boundary. |
| Matrix import/export owner | User-facing IO path with parse and allocation interactions. | Lower allocation signal; IO errors may dominate allocation behavior. | Consider if Day 2 prioritizes public IO semantics. |

## Initial Invariant Categories

| Category | Day 1 placeholder |
| --- | --- |
| Return status | Failed selected-owner allocation returns a documented error status and cannot look like success. |
| Cleanup | Failed allocation releases temporary and partially owned resources. |
| Stale-output suppression | Caller-visible outputs remain unchanged, null-cleared, or status-gated according to the selected owner contract. |
| Partial publication | Failed allocation does not publish a success-looking factor, matrix, workspace, handle, or result structure. |
| Caller-owned input | Caller inputs remain valid and unchanged unless the selected owner contract explicitly permits mutation. |
| Retry | After resetting deterministic failure injection, the same fixture succeeds and produces fresh output. |
| Unsupported breadth | The proof does not cover broad allocation reliability, OS OOM, concurrent hooks, hosted proof, package/install, platform, performance, release, external-library parity, or state-of-the-art reliability. |

## Validation Matrix

| Validation | Day 1 status | Notes |
| --- | --- | --- |
| `git diff --check` | Planned for Day 1 closeout. | Documentation-only changes. |
| Focused selected-owner gate | Not yet applicable. | Owner selection occurs on Day 2. |
| Relevant family tests | Not yet applicable. | Family depends on selected owner. |
| Source-list check | Not yet applicable. | No source or test registration edits on Day 1. |
| Full C quality gate | Not required. | No `.c` or `.h` files changed on Day 1. |

## Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Selecting too broad an owner | Sprint 210 should completely close one gap, not partially close a family. | Day 2 must freeze one owner and reject broad family claims. |
| Re-proving a closed owner | Sprint 210 must add a new proof outside selected symbolic LU. | Exclude iterative, matmul, symbolic Cholesky, and symbolic LU owners. |
| Allocation hooks do not reach the selected owner | Failure-injection tests only prove hook-reachable paths. | Prefer wrapped allocation paths or budget a narrow wrapper conversion. |
| Output semantics are ambiguous | Cleanup tests can overclaim if publication rules are not known. | Day 3 records lifecycle invariants before implementation. |
| Numerical owner fixtures become broad | QR/SVD/eigs outputs can require large correctness surfaces. | Select one workspace/output owner with deterministic fixture scope. |
| Hook state leaks across tests | Process-global failure hooks can poison later tests. | Follow Sprint 200 reset-before-assertion discipline. |
| Documentation overclaims reliability | One selected proof is not broad reliability. | Keep docs selected-owner-only and preserve non-claims. |

## Open Questions For Day 2

1. Which candidate has the clearest selected-owner boundary outside symbolic LU?
2. Which candidate balances user impact, allocation density, cleanup risk, and
   retry clarity?
3. Which candidate can be reached through existing deterministic allocation
   hooks with minimal source churn?
4. Which candidate has a clean stale-output and partial-publication contract?
5. Which candidate can receive a focused gate without broad source-list or
   review-surface churn?

## Validation

Commands run:

```sh
git diff --check
git status --short
find docs/planning/EPIC_19/SPRINT_210 -type f | sort
git diff --name-only -- '*.c' '*.h'
```

Day 1 changes planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.
