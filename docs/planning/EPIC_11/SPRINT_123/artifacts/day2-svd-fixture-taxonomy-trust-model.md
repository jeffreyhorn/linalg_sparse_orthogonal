# Sprint 123 Day 2 SVD Fixture Taxonomy and Trust Model

## Purpose

Day 2 defines what broader SVD external evidence can safely mean before Day 3
decides whether to implement another bounded fixture batch. The intent is to
avoid turning small pure-Python reference checks into broad LAPACK, NumPy,
SciPy, or dense-library parity claims.

This is a design artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, or public wording changes are made by
Day 2.

## Inputs Reviewed

| Input | Relevant Content |
| --- | --- |
| Sprint 123 Plan Day 2 | Requires current SVD external fixture inventory, deterministic coverage classification, trust model, tolerance/skip policy, and non-claim fencing. |
| Sprint 121 Day 4 taxonomy | Provides fixture metadata fields and classes for diagonal, threshold-rank, repeated-spectrum, duplicate-column, low-rank, pseudoinverse, partial-SVD, SuiteSparse, and expected-failure evidence. |
| Sprint 122 Day 3 SVD inventory | Defines prior candidate filters and the original strongest rank-deficient SVD external candidate. |
| Sprint 122 Day 4 SVD decision | Records completed `svd_rankdef_duplicate_5x4` implementation, tolerance split, skip behavior, and non-claims. |
| `tests/svd_external_dense_reference.py` | Current standard-library-only reference helper with `svd_rect_fullrank_6x4`, `svd_rankdef_duplicate_5x4`, and `partial_svd_diag6_k2`. |
| `tests/test_svd.c` | Current full-SVD, rank, pseudoinverse, low-rank, external-reference, full-UV, condition, and error-path proof owner. |
| `tests/test_svd_partial_helpers.h` | Current partial-SVD value, vector, subspace-adjacent, rectangular, wide, rank-deficient, and low-rank proof owner. |

## Current SVD External Fixture Inventory

| Fixture Key | Owner | Matrix Class | Compared Quantity | Current Trust Boundary | Duplicate Fence |
| --- | --- | --- | --- | --- | --- |
| `svd_rect_fullrank_6x4` | `tests/svd_external_dense_reference.py`, `tests/test_svd.c` | Small dense rectangular full-column-rank mixed-sign matrix | Full-SVD singular values only | Python standard-library `A^T A` plus Jacobi eigenvalue reference for one fixed fixture | Do not add another mixed dense full-column-rank rectangular fixture unless it adds a new shape or failure interpretation. |
| `svd_rankdef_duplicate_5x4` | `tests/svd_external_dense_reference.py`, `tests/test_svd.c` | Small dense rectangular exact-rank-deficient matrix with dependent columns | Full-SVD singular values and zero-tail tolerance | Same helper path, with positive-singular-value and zero-tail tolerances separated | Do not repeat exact rank-deficient singular-value evidence without adding a new semantic class. |
| `partial_svd_diag6_k2` | `tests/svd_external_dense_reference.py`, `tests/test_svd_partial_helpers.h` | Small diagonal partial-SVD top-k value fixture | Top two singular values only | Same helper path, truncated to top-k values | Not a full-SVD fixture; do not use it to claim vector, subspace, or convergence parity. |

## Deterministic Internal SVD Coverage Classes

| Coverage Class | Current Evidence Owner | External Fixture Implication |
| --- | --- | --- |
| Exact diagonal spectra | `test_svd_basic_sigma`, diagonal SVD, condition, and partial-SVD tests | External process adds little unless it validates reference protocol or output-shape behavior. |
| Threshold rank and near-singular values | rank threshold, condition, and Sprint 103 diagonal/rank claim tests | Candidate only if tiny singular-value interpretation and failure semantics are explicit. |
| Exact rank deficiency | duplicate/dependent rank tests plus `svd_rankdef_duplicate_5x4` | Already has one bounded external singular-value lane. New work must add a different rank-deficient shape or semantic. |
| Tall and wide full SVD | `test_svd_tall_10x5`, `test_svd_wide_5x10`, UV reconstruction tests | Wide output-shape semantics remain a real gap because current helper is easiest for `A^T A` column-space output. |
| Repeated spectra | `test_svd_repeated`, diagonal/repeated internal checks | External singular values are easy; vector/subspace uniqueness is not claimable. |
| Pseudoinverse identities | Moore-Penrose and rectangular/underdetermined pinv tests | Not a singular-value fixture unless Day 3 explicitly scopes identity metrics. |
| Low-rank dense/sparse output | dense and sparse low-rank tests, corpus safety, outer-product path | Singular-value checks do not prove low-rank output optimality; keep separate unless comparing tail-energy only. |
| Partial-SVD vectors and residuals | `tests/test_svd_partial_helpers.h` vector, `A*v`, reconstruction, wide, and SuiteSparse checks | Belongs to Sprint 123 Days 9-10, not Day 3 full-SVD fixture work. |
| SuiteSparse smoke evidence | nos4, west0067, bcsstk04, corpus safety owners | Optional-file and runtime variability make this unsuitable for a small Day 3 external fixture batch. |
| Error paths | null, bad-k, factored-matrix, low-rank error tests | Not numerical external oracle evidence. |

## Candidate External SVD Fixture Classes

| Candidate Class | Example Fixture Key | Adds New Evidence? | Trust/Risk | Day 2 Disposition |
| --- | --- | --- | --- | --- |
| Wide full-rank singular-value output | `svd_wide_fullrank_4x6` | Yes. Exercises wide shape and `min(m,n)` output semantics not covered by current external full-SVD fixtures. | Moderate. Reference helper must emit exactly four singular values rather than column-count padded values. | Strong Day 3 candidate if output-shape contract is pinned. |
| Near-dependent threshold singular values | `svd_near_dependent_5x4` | Yes, if tiny singular values are compared without claiming universal rank policy. | High. Requires separate positive, tiny-tail, and rank-threshold interpretation. | Candidate only if Day 3 wants threshold semantics; otherwise defer. |
| Repeated non-diagonal spectrum | `svd_repeated_spectrum_5x5` | Moderate. Adds non-diagonal repeated singular values without vector uniqueness claims. | Moderate. Singular values are stable, but vector/subspace parity must remain fenced. | Candidate if Day 3 prefers low-risk singular-value diversity. |
| Rectangular low-rank tail-energy fixture | `svd_lowrank_tail_6x5` | Moderate. Could compare singular values and tail-energy expectation. | High. Risks drifting into low-rank optimality claims. | Defer unless scoped as singular values only. |
| Pseudoinverse singular-value threshold fixture | `svd_pinv_threshold_4x3` | Low to moderate. Could support pinv threshold decisions later. | High. Moore-Penrose identities and minnorm are not proven by singular values alone. | Defer to pseudoinverse/minnorm owners. |
| SuiteSparse SVD external fixture | `svd_suitesparse_nos4_external` | Broad but tempting. | Very high. Optional corpus, runtime, platform, and broad-corpus interpretation risks. | Reject for Sprint 123 Day 3. |
| Singular-vector or subspace external check | `svd_vectors_repeated_subspace` | Different evidence class, not a simple fixture expansion. | Very high. Sign, basis, repeated-spectrum, and subspace-angle policy required. | Defer to future vector/subspace owner; do not include in Day 3 singular-value batch. |

## Reference Trust Model

| Layer | Trust Decision | Rationale |
| --- | --- | --- |
| Reference implementation | Continue using Python standard-library arithmetic for small fixed fixtures. | Keeps the lane dependency-light and independent of NumPy, SciPy, LAPACK, BLAS, or platform packages. |
| Algorithmic method | `A^T A` plus Jacobi eigenvalue iteration is acceptable for small singular-value fixtures. | This proves fixture-local agreement, not broad dense-SVD correctness. |
| Matrix size | Keep new fixtures small, preferably `min(m,n) <= 6`. | Avoids runtime, conditioning, and reference convergence ambiguity. |
| Output shape | Each fixture must state whether the helper emits `min(m,n)` values or a top-k subset. | Prevents wide-shape zero-padding or truncation mistakes. |
| Compared quantity | Full-SVD Day 3 candidates compare singular values only. | Vector, subspace, pseudoinverse, and low-rank output claims require separate metrics. |
| Missing helper dependency | Missing `python3` may skip through existing helper behavior. | Maintains consistency with existing external-reference lanes. |
| Windows behavior | Preserve explicit Windows skip unless a future sprint promotes platform proof. | No equal Windows reviewed-support claim follows from this lane. |
| Failure interpretation | Failures must identify fixture key, reference status, singular-value index or max difference, and whether the mismatch is positive or tail value. | Keeps failures actionable without turning them into broad library parity conclusions. |

## Tolerance and Skip Policy

| Evidence Type | Tolerance / Policy | Failure Interpretation |
| --- | --- | --- |
| Positive singular values | Default max absolute difference target remains `1e-8` unless Day 3 justifies a tighter or looser bound. | Product/reference mismatch for this fixture only. |
| Zero or tiny tails | Use a separate tail tolerance such as `1e-8`; do not silently fold into rank policy. | Tail mismatch indicates fixture-specific threshold drift, not universal rank failure. |
| Repeated singular values | Compare sorted values with equality slack; do not compare vector orientation. | Value ordering/multiplicity mismatch only. |
| Wide output shape | Helper must emit exactly `min(m,n)` singular values or state a top-k subset. | Output-count mismatch is a fixture protocol failure. |
| Near-dependent threshold | Require explicit positive/tiny/tail buckets and rank-threshold non-claim. | Failure may indicate roundoff or threshold-policy ambiguity; artifact must say which. |
| Missing Python helper | Skip through the existing external-reference helper. | Skip is environmental, not numerical success. |
| Helper `ERROR` output | Fail the test with the emitted reason. | Reference fixture protocol failure. |
| Windows path | Keep explicit skip unless promoted separately. | No Windows parity claim. |

## Day 3 Decision Inputs

Day 3 should choose one of these outcomes:

| Outcome | Required Evidence |
| --- | --- |
| Implement `svd_wide_fullrank_4x6` | Define matrix, emit exactly four singular values, compare values only, preserve no vector/subspace/pinv/low-rank claims, and run focused helper plus `test_svd` validation. |
| Implement a repeated-spectrum singular-value fixture | Define non-diagonal repeated-spectrum matrix, compare values only, and explicitly state singular vectors are non-unique and unclaimed. |
| Defer near-dependent threshold fixture | Record threshold ambiguity, rank-policy dependency, and future owner unless Day 3 pins tiny-tail semantics. |
| Defer all additional SVD external work | State that current Sprint 122 full-rank and rank-deficient lanes are sufficient before Sprint 124 corpus work, and carry remaining classes forward with promotion gates. |

## Non-Claim Register

Day 2 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad ecosystem parity;
- broad external dense-library SVD correctness;
- singular-vector, subspace, or repeated-spectrum basis parity;
- partial-SVD external vector/subspace/convergence parity;
- pseudoinverse or minimum-norm correctness beyond existing deterministic
  tests;
- low-rank global optimality;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- portable performance, scalability, memory behavior, or state-of-the-art
  behavior.

## Validation Notes

Day 2 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_123`

No `.c`, `.h`, Python helper, build metadata, public docs, or test membership
changed on Day 2.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 1 has explicit decision criteria. | Complete | See candidate external SVD fixture classes, trust model, tolerance policy, and Day 3 decision inputs. |
| Every SVD fixture candidate has a trust-boundary rationale. | Complete | See candidate table and reference trust model. |
| No external-library parity claim is introduced. | Complete | See non-claim register. |
