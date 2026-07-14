# Sprint 122 Day 5 QR External Lane Requirements

## Purpose

Day 5 defines the QR external dense-reference lane problem before any Day 6
implementation or deferral decision. The goal is to identify candidate scenarios
that add bounded independent QR or least-squares evidence without duplicating
Sprint 121 deterministic fixture expansion or introducing a QR external parity
claim prematurely.

## QR Evidence Inventory

| Capability | Current Evidence Owner | Sprint 121 Coverage | Day 5 Requirement |
| --- | --- | --- | --- |
| Basic QR factorization | `tests/test_qr.c` | Small dense/sparse, identity, zero, triangular, permutation validity | Do not use external lane for low-level Householder or factor-shape smoke. |
| Reconstruction | `tests/test_qr.c`, `tests/test_qr_solve.c` | `A*P = Q*R` residuals across square, tall, wide, rank-1, near-singular, SuiteSparse, sparse mode | External lane must preserve permutation semantics if it checks factorization. |
| Q orthogonality | `tests/test_qr.c` | Tall, wide, economy, sparse mode, Q/QT application | External lane should not compare Q bases unless sign/basis semantics are designed. |
| Rank and nullspace | `tests/test_qr.c`, `tests/test_qr_solve.c` | Duplicate columns, rank-1, near duplicate, explicit tolerance, nullspace residual | External lane must not blur exact structural rank with numerical threshold policy. |
| Square solve | `tests/test_qr_solve.c` | Synthetic square systems and bounded QR-vs-LU agreement | External lane should not imply direct-solver or LAPACK parity. |
| Overdetermined compatible least squares | `tests/test_qr_solve.c` | Deterministic generated-RHS tall systems with near-zero residual | External lane should avoid duplicating exact generated-RHS coverage unless it adds an independent reference solution. |
| Overdetermined incompatible least squares | `tests/test_qr_solve.c` | Deterministic known-residual 4x2 fixture and residual reporting | Strongest candidate area because an independent dense normal-equation-free reference can validate residual/solution semantics. |
| Underdetermined minimum-norm | `tests/test_colamd.c`, `tests/test_qr_solve.c` | Known 2x4, 3x6, 5x10, 1xn, COLAMD-backed, QR-vs-pinv checks | Keep for Day 9 helper ownership; do not mix with Day 6 external QR lane unless ownership is clear. |
| Iterative refinement | `tests/test_qr_solve.c` | Residual non-increase and exact recomputation for well-conditioned, ill-conditioned, SuiteSparse, overdetermined, minimum-norm paths | External lane should not become a refinement oracle in Sprint 122. |
| Economy and sparse modes | `tests/test_qr.c` | Shape, solve, rank, orthogonality, and backend agreement | External lane should not claim backend parity or performance. |
| Reordering / fill | `tests/test_qr.c`, `tests/test_colamd.c` | AMD, COLAMD, none, SuiteSparse fill and solve residuals | Reordering is QR-adjacent; keep out of Day 6 external dense-reference scope. |

## QR External Candidate Scenario Table

| Candidate | Adds New Evidence? | Risk | Day 5 Disposition |
| --- | --- | --- | --- |
| `qr_overdetermined_incompatible_4x2_external_ls` | Yes. Independently validates least-squares solution and nonzero residual for a small incompatible tall system. | Moderate: reference must avoid implying LAPACK/normal-equation parity and must keep residual semantics explicit. | Strongest Day 6 candidate. |
| `qr_overdetermined_compatible_4x2_external_solve` | Partial. Confirms known solution but duplicates generated-RHS exact coverage. | Low implementation risk, high duplication risk. | Defer unless Day 6 wants a companion case. |
| `qr_square_3x3_external_solve` | Low. Square solve is already covered by exact synthetic and QR-vs-LU fixtures. | Could imply direct-solver parity. | Reject for Sprint 122 external lane. |
| `qr_rankdef_duplicate_5x4_external_ls` | Partial. Rank-deficient least-squares is important but needs rank and minimum-norm ownership clarity. | High: overlaps Day 9 helper ownership and rank threshold semantics. | Defer to future rank-deficient QR oracle work. |
| `qr_underdetermined_minnorm_2x4_external` | Partial. Important, but ownership is entangled with minimum-norm helper migration. | High: overlaps Day 9 and QR-vs-pinv non-claim semantics. | Defer to Day 9 or future minimum-norm owner. |
| `qr_q_factor_external_basis_check` | Low for Sprint 122. | High: sign, basis, and economy/full Q shape semantics are not designed. | Reject for Day 6. |
| `qr_suite_sparse_external_ls` | Potentially broad. | High: optional data, platform/runtime variability, and corpus interpretation. | Reject for Sprint 122 external lane. |

## Strongest Day 6 Candidate

`qr_overdetermined_incompatible_4x2_external_ls`

Candidate fixture intent:

- Matrix shape: 4x2 dense tall full-column-rank matrix.
- RHS: generated compatible component plus a known vector orthogonal to the
  column space, producing a deterministic nonzero least-squares residual.
- Product path: `sparse_qr_solve` or the existing QR solve path under
  `test_qr_solve`.
- Reference path: Python standard-library dense least-squares reference using a
  small normal-equation or Gram-system solve only if Day 6 records the trust
  boundary clearly.
- Compared quantities: solution vector and residual norm.
- Tolerances: solution max-diff below `1e-8`; residual norm max absolute
  difference below `1e-8`; optional relative residual comparison only if named.
- Build impact preference: stay inside existing `test_qr_solve`; no Makefile,
  CMake, or CTest membership change.

## Tolerance and Skip Requirements

| Requirement | Decision Input for Day 6 |
| --- | --- |
| Fixture size | Prefer 4x2 or similarly tiny dense fixture to keep Python helper deterministic and fast. |
| Solution tolerance | Use fixture-local absolute tolerance; do not generalize to arbitrary ill-conditioned least-squares systems. |
| Residual tolerance | Compare the named residual norm separately from solution error. |
| Rank assumptions | Record full-column-rank assumptions; do not assert broad rank-detection policy. |
| Optional dependency policy | Python standard library only; no NumPy, SciPy, LAPACK, BLAS, or external package dependency. |
| Missing `python3` | Skip through the existing external-reference helper pattern. |
| Windows behavior | Skip explicitly if the existing external-reference lane remains Windows-disabled. |
| Helper `ERROR` output | Treat as test failure, not skip. |

## Failure Interpretation Requirements

Day 6 must make any failure message identify:

- fixture key;
- reference helper status;
- product QR solve status;
- solution max difference;
- residual norm difference;
- whether the failure is reference generation, QR solve, solution mismatch,
  residual mismatch, unsupported platform, or optional-helper unavailability.

## Helper and Solver-Semantic Dependencies

| Dependency | Required Handling |
| --- | --- |
| Residual helper semantics | Keep absolute and relative residual meanings visible at the test call site. |
| QR helper ownership | Do not move minimum-norm helpers from `tests/test_colamd.c` during Day 6. |
| Rank-deficient ownership | Defer rank-deficient external QR work until rank threshold and minimum-norm ownership are explicit. |
| Q basis semantics | Do not compare Q columns externally without sign and basis rules. |
| CTest membership | Keep Day 6 inside an existing executable unless explicitly documenting reviewed count impacts. |
| Public wording | Do not update public solver-selection claims from Day 6 alone. |

## Day 6 Design Checklist

Day 6 should choose one of these outcomes:

| Outcome | Required Evidence |
| --- | --- |
| Implement `qr_overdetermined_incompatible_4x2_external_ls` | Define Python reference output protocol, add one existing-executable test, validate focused QR solve executable, run required quality chain, and record no broad QR parity claim. |
| Defer QR external lane | Explain which fixture, tolerance, reference, skip, failure, or ownership gate is not ready and assign future owner. |
| Reject QR external lane for Epic 11 | Explain why deterministic Sprint 121 QR evidence plus current scope is sufficient for this epic. |

## Non-Claim Register

Day 5 does not claim QR external parity. Any Day 6 implementation must continue
to avoid claims of LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos, Eigen,
direct-solver parity, broad least-squares optimality, minimum-norm global
optimality, performance, scalability, package, platform, ABI, public API, or
state-of-the-art behavior.

## Validation Notes

Day 5 planning content is documentation-only, but the branch currently includes
Day 4 `.c` changes. The branch-level validation gate therefore remains the full
code quality chain required by Day 4:

```sh
make format && make lint && make test
```

plus `git diff --check` and focused trailing-whitespace scans.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| QR lane design inputs are explicit before any implementation decision. | Complete | See evidence inventory, candidate table, tolerance requirements, and Day 6 checklist. |
| Candidates do not duplicate completed Sprint 121 QR fixture expansion. | Complete | Candidate dispositions reject or defer duplicate square, compatible, SuiteSparse, Q-basis, rank-deficient, and minimum-norm cases. |
| QR external parity remains a non-claim unless separately earned. | Complete | See non-claim register and helper dependency boundaries. |
