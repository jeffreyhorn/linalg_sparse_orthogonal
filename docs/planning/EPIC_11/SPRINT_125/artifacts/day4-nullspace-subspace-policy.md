# Sprint 125 Day 4 Nullspace and Subspace Policy

## Purpose

Day 4 defines the nullspace and subspace policy required before Sprint 125 can
accept rank-deficient QR nullspace or subspace evidence. The policy keeps rank,
nullity, vector residual, basis orientation, projection/subspace metrics,
minimum-norm, pseudoinverse, Q-basis, economy, and residual-only evidence
separate.

This is a policy artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, or public wording changes are made by Day
4.

## Inputs Reviewed

| Input | Policy Use |
| --- | --- |
| Sprint 125 Plan Day 4 | Requires current evidence inventory, nullity expectations, rank thresholds, sign/ordering rules, projection/subspace metric selection, tolerances, diagnostics, and non-claims. |
| Sprint 125 Day 1 dedupe map | Provides duplicate fences for completed Sprint 121-124 QR evidence. |
| Sprint 125 Day 2 trust gate | Separates residual-only evidence from nullspace, minimum-norm, pseudoinverse, and Q-basis claims. |
| Sprint 125 Day 3 residual evidence | Adds `qr_rankdef_duplicate_5x4_residual_only` and explicitly excludes nullspace/subspace claims. |
| Sprint 124 Day 2 rank policy | Defines rank-threshold, nullspace, minimum-norm, tolerance, and failure-interpretation boundaries. |
| `tests/test_qr.c` | Current owner for deterministic rank, nullspace, diagonal-threshold, Q-basis, economy, sparse-mode, reorder, and reconstruction evidence. |
| `tests/test_qr_solve.c` | Current owner for QR solve residual and bounded external QR solve fixtures; not the nullspace/subspace owner. |
| `tests/qr_external_dense_reference.py` | Potential future helper owner for tiny external subspace references if Day 5 accepts a lane. |

## Current QR Rank, Nullspace, and Subspace-Adjacent Evidence

| Evidence Class | Current Owner | Evidence Summary | Day 4 Interpretation |
| --- | --- | --- | --- |
| Duplicate-column rank deficiency | `tests/test_qr.c` | `test_qr_rank_deficient` expects rank 2 for a 4x3 duplicate-column fixture. | Internal rank evidence; not external nullspace proof. |
| Rank-only external duplicate-column fixture | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_rankdef_duplicate_5x4_rank_only` checks rank 3 at threshold `0.0`. | External rank-only evidence; nullity can be derived only inside a policy that pins the same rank threshold. |
| Residual-only external duplicate-column fixture | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_rankdef_duplicate_5x4_residual_only` checks one non-zero least-squares residual. | External residual-only evidence; no nullspace or subspace claim. |
| Rank-1 nullspace | `tests/test_qr.c` | `test_rank_1_nullspace` expects nullity 2 and verifies each returned vector satisfies `A*v ~= 0`. | Valid internal vector-residual evidence; does not prove external basis equality. |
| Known nullspace vector | `tests/test_qr.c` | `test_known_nullspace` verifies the returned basis vector is a null vector for a duplicate-column fixture. | Useful diagnostic baseline, but raw vector orientation should not be externally compared. |
| Rectangular rank-deficient nullspace | `tests/test_qr.c` | `test_rank_rect_deficient` expects rank 2 for a 3x5 fixture and verifies three returned null vectors. | Internal nullity/vector-residual evidence for wide shape; future external evidence needs subspace metrics. |
| Dependent-row rank fixture | `tests/test_qr.c` | `test_qr_rank_dependent_row_fixture` verifies rank, reconstruction, nullity 1, and null residual. | Candidate family only if Day 5 proves it adds trust beyond existing deterministic evidence. |
| Diagonal rank threshold fixture | `tests/test_qr.c` | `test_qr_rank_diagonal_threshold_fixture` expects ranks 3, 2, and 1 at fixture-local thresholds. | Threshold-policy input for Days 6-7; not a nullspace evidence lane. |
| Economy rank-deficient QR | `tests/test_qr.c` | `test_economy_rank_deficient` checks economy QR detects rank deficiency. | Sprint 126 Q/economy input; not Sprint 125 nullspace evidence. |
| Sparse-mode rank-deficient parity | `tests/test_qr.c` | `test_sparse_mode_rank_deficient` compares dense/sparse QR rank and result behavior. | Sparse-mode owner; not an external nullspace/subspace lane. |

## Nullity and Rank-Threshold Policy

| Policy Point | Sprint 125 Rule |
| --- | --- |
| Nullity formula | Nullity may be asserted as `n - rank` only for a named fixture with an explicit expected rank and rank threshold. |
| Global rank threshold | Do not introduce a global QR rank threshold. Every external nullspace/subspace fixture must pin its threshold locally. |
| Structural rank fixtures | Exact duplicate-column or dependent-row fixtures are preferred because expected rank and nullity can be explained without dense-library dependencies. |
| Near-rank-deficient fixtures | Defer to Days 6-7 unless the fixture has explicit threshold families, expected ranks, and stability policy. |
| Rank mismatch interpretation | A rank mismatch blocks nullspace/subspace interpretation; it is not converted into a basis failure. |
| Residual-only fixtures | Residual agreement does not establish rank or nullity. |

## Sign and Ordering Policy

| Basis Issue | Policy |
| --- | --- |
| Sign ambiguity | Individual null vectors may be multiplied by `-1`; raw sign-sensitive equality is not acceptable by default. |
| Basis ordering | Nullspace basis columns can be permuted; column-order equality is not acceptable unless a fixture explicitly proves deterministic ordering. |
| Rotations within a subspace | For nullity greater than 1, any orthonormal rotation spans the same subspace; raw vector equality is invalid. |
| Degenerate/ill-conditioned rank | Near-threshold nullspaces require Day 6-7 threshold policy before any basis or subspace check. |
| Acceptable raw vector exception | Raw vector comparison is allowed only for a future tiny fixture with nullity 1, a fixed normalization rule, deterministic sign convention, and a documented reason subspace metrics are unnecessary. |
| Default comparison | Use projection/subspace metrics, not raw basis columns. |

## Projection and Subspace Metric Policy

| Metric | Use | Acceptance Rule |
| --- | --- | --- |
| Null residual `||A*z_i||_2` | Per-vector diagnostic that each returned vector is in the nullspace. | Useful but insufficient by itself for full external subspace equivalence. |
| Orthonormality `||Z^T Z - I||` | Validates returned product basis quality. | Required when the product basis is compared as a subspace. |
| Projector distance `||Z Z^T - Z_ref Z_ref^T||_F` | Preferred external subspace comparison for orthonormal bases. | Primary Day 5 metric for nullity greater than 1. |
| Two-way projection residual | Check each basis projects into the other subspace: `||(I - P_ref)Z||` and `||(I - P)Z_ref||`. | Acceptable alternative when storing full projector values is too noisy. |
| Principal angles | Mathematically strong but heavier to implement. | Defer unless a future helper justifies the extra protocol complexity. |
| Raw vector max difference | Basis equality check. | Disallowed by default because sign/order/rotation ambiguity can produce false failures. |

## Fixture-Local Tolerance Policy

| Fixture Class | Rank Threshold | Nullity | Metric Tolerance | Notes |
| --- | ---:| ---:| ---:| --- |
| Exact duplicate-column tall fixture | `0.0` or fixture-pinned value | `n - expected_rank` | projector or two-way projection residual `<= 1e-8`; null residual `<= 1e-10` | Best Day 5 candidate if external helper can emit stable subspace data. |
| Exact dependent-row fixture | `0.0` or fixture-pinned value | `n - expected_rank` | projector or two-way projection residual `<= 1e-8`; null residual `<= 1e-10` | Defer unless duplicate-column evidence is insufficient. |
| Wide rank-deficient fixture | fixture-pinned value | `n - expected_rank` | projector metric `<= 1e-8`; residual tolerance fixture-local | Higher risk because it overlaps underdetermined/minimum-norm semantics. |
| Near-rank-deficient fixture | explicit threshold family | threshold-specific | defer | Belongs to Days 6-7 before nullspace evidence. |
| SuiteSparse fixture | corpus-specific | corpus-specific | defer | Requires Days 8-9 optional corpus and support-tier policy. |

## Diagnostics Policy

Any accepted nullspace/subspace evidence must report:

- fixture key;
- matrix shape;
- expected rank and threshold;
- product rank and nullity;
- reference nullity;
- maximum null residual `||A*z_i||_2`;
- product basis orthonormality error;
- reference basis orthonormality error if applicable;
- selected subspace metric and tolerance;
- skip reason for unsupported platform or helper unavailability;
- whether failure is rank, nullity, helper protocol, product basis,
  reference basis, subspace metric, tolerance, or unsupported optional data.

## Day 5 Acceptance Criteria

Day 5 may accept rank-deficient QR nullspace/subspace evidence only if all of
the following are true:

1. The fixture has explicit expected rank, nullity, and rank threshold.
2. The external helper can produce a stable reference subspace using only the
   Python standard library.
3. The product and reference bases are compared through projector or two-way
   projection metrics, not raw column equality.
4. The artifact documents sign, ordering, and rotation ambiguity.
5. The test asserts only rank/nullity/subspace metrics and null residuals, not
   minimum-norm, pseudoinverse, Q-basis, economy, SuiteSparse, backend, or
   performance behavior.
6. Focused helper and test validation commands are known before code edits.

Day 5 should explicitly defer a candidate if it requires near-threshold policy,
SuiteSparse corpus policy, raw vector equality, underdetermined
minimum-norm semantics, or broad public claim wording.

## Non-Claim Register

Day 4 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad dense-library parity;
- broad QR factorization, QR solve, rank-deficient solve, nullspace, Q-basis,
  economy, sparse-mode, reorder, backend, corpus, or performance parity;
- raw nullspace basis equality, unique basis orientation, sign, ordering, or
  vector-column external parity;
- global QR rank-threshold policy;
- minimum-norm optimality, QR-vs-SVD-pseudoinverse behavior, COLAMD, fallback,
  refinement, or SuiteSparse minimum-norm behavior;
- package, ABI, platform, public API, CMake, Makefile, CI, CTest, performance,
  scalability, memory, or state-of-the-art behavior.

## Validation Notes

Day 4 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_125`
   and already touched maintainer/test/helper files

No `.c`, `.h`, Python helper, build, public API, or public wording files
changed for Day 4, so no new code quality gate is required. The branch already
passed the full `make format && make lint && make test` gate after Day 3's
code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 3 has explicit acceptance criteria. | Complete | See nullity/rank-threshold policy and Day 5 acceptance criteria. |
| Basis-dependent evidence cannot be mistaken for raw vector equality. | Complete | See sign/ordering policy and projection/subspace metric policy. |
| Future tests have stable tolerance and diagnostic rules. | Complete | See fixture-local tolerance policy and diagnostics policy. |
