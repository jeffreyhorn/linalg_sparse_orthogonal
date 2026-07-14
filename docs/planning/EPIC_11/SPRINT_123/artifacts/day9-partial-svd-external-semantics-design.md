# Day 9 Partial-SVD External Semantics Design

## Purpose

Define the semantics required before extending partial-SVD external evidence beyond the existing `partial_svd_diag6_k2` top-k singular-value lane. This artifact separates value, vector, subspace, convergence-budget, repeated-spectrum, rectangular, rank-deficient, and low-rank semantics so Day 10 can either implement a bounded lane or defer it with precise proof gates.

## Current Evidence Inventory

| Evidence family | Current owner | Current proof shape | External? | Boundary |
| --- | --- | --- | --- | --- |
| Top-k singular values | `test_partial_svd_diag_10x10`, `test_partial_svd_dense_8x8`, `test_partial_svd_tall`, `test_partial_svd_wide`, `test_partial_svd_nos4`, `test_partial_svd_west0067` | Compares partial-SVD singular values to this library's full SVD or deterministic diagonal fixtures | Mostly no | Good regression evidence, but not independent dense-library parity. |
| Bounded top-k external values | `test_partial_svd_external_dense_reference_diag6_k2` plus `tests/svd_external_dense_reference.py` | Compares top two singular values of a 6x6 diagonal fixture to pure-Python dense reference output | Yes | Value-only; no vectors, subspaces, convergence-budget, low-rank, or broad parity claim. |
| Vector residuals | `test_partial_svd_vectors_ortho`, `test_partial_svd_vectors_Av`, `test_partial_svd_vectors_vs_full`, SuiteSparse vector tests | Orthogonality, `A*v ~= sigma*u`, reconstruction, and full-SVD comparison | No | Internal proof only; sign and basis orientation remain local to the implementation. |
| Rectangular vector behavior | `test_partial_svd_vectors_wide`, `test_partial_svd_vectors_rectangular_lowrank_recon` | Shape, vector residuals, and low-rank reconstruction on rectangular fixtures | No | Internal proof; not an external subspace or optimality comparison. |
| Rank-deficient values | `test_partial_svd_rank_deficient` | Top singular values on rank-deficient fixtures | No | Uses internal expectations; zero/near-zero threshold semantics are not externalized. |
| Ordering and k boundaries | `test_partial_svd_descending`, `test_partial_svd_k1`, `test_partial_svd_full_k`, bad-argument tests | Ordering and API boundary behavior | No | API and ordering proof, not external numerical parity. |
| Convergence/timing smoke | `test_partial_svd_timing`, SuiteSparse partial-SVD tests | Timing smoke and bounded residual windows | No | Not a performance claim or external convergence guarantee. |
| Low-rank approximation | dense and sparse low-rank tests in `tests/test_svd.c` | Dense/sparse output consistency and fixture-specific Frobenius checks | No | Not a global Eckart-Young or production low-rank optimality claim. |

## Semantics Matrix

| Semantic class | What it would prove | Required rule before implementation | Failure interpretation | Day 10 viability |
| --- | --- | --- | --- | --- |
| Top-k singular values | Partial-SVD returns the requested leading singular values for a fixed fixture | Fixture key, `k`, expected output count, descending order, absolute tolerance | Difference means bounded fixture regression, not broad parity failure | Already implemented for `partial_svd_diag6_k2`; only add another value fixture if it covers a clearly different shape. |
| Singular vectors | Returned `U` and `Vt` columns correspond to singular triplets | Sign-invariant comparison or residual-only metric, explicit vector output option, orientation convention | Residual failure is meaningful; raw sign mismatch alone is not | Defer unless the fixture uses residual metrics instead of direct vector equality. |
| Subspace agreement | Returned vectors span the expected leading singular subspace | Principal-angle or projection-distance metric, especially for repeated or clustered spectra | Basis mismatch inside the same subspace is not a failure | Defer until projection/angle helpers exist. |
| Repeated spectrum | Partial-SVD handles equal leading singular values | Compare subspaces or unordered value multisets; never compare individual vectors directly | Vector swaps/signs are expected; only subspace or value-set mismatch matters | Defer; current helper emits ordered values only. |
| Clustered spectrum | Partial-SVD remains stable when leading values are close | Define gap, convergence budget, tolerance, and whether near ties are accepted as ordered or set-based | A strict order failure may be algorithmic noise rather than correctness failure | Defer unless a conservative value-only fixture avoids near-tie ambiguity. |
| Rectangular shape | Partial-SVD handles tall/wide dimensions consistently | Matrix shape, `k <= min(m,n)`, output dimensions, and value/vector expectations | Shape mismatch is API regression; value mismatch is numerical regression | Candidate for a value-only Day 10 fixture if top-k values are distinct. |
| Rank deficiency | Partial-SVD handles numerical zeros and rank truncation | Rank threshold, expected positive singular values, zero tolerance, and whether `k` crosses rank | Near-zero differences need threshold interpretation; direct equality is unsafe | Candidate only for value/rank threshold proof, not vector proof. |
| Convergence budget | Algorithm reaches expected quality within a fixed budget | Options surface, iteration cap, random seed policy if any, residual tolerance, skip behavior | Budget failure can indicate convergence regression, not mathematical impossibility | Defer; current external helper has no convergence-budget protocol. |
| Low-rank optimality | Rank-k reconstruction error matches theoretical or external reference error | Frobenius/2-norm metric, comparison reference, truncation policy, output sparsification policy | Dense and sparse low-rank failures have different meanings | Defer; keep separate from partial-SVD top-k values. |

## Fixture Candidate Table

| Candidate | Class | Value | Risk | Day 10 recommendation |
| --- | --- | --- | --- | --- |
| `partial_svd_tall_diag_8x5_k3` | Rectangular value-only | Adds external top-k coverage for tall shape without vector semantics | Low; distinct diagonal values avoid basis ambiguity | Acceptable if Day 10 wants one bounded implementation. |
| `partial_svd_wide_diag_5x8_k3` | Rectangular value-only | Adds external top-k coverage for wide shape | Low; duplicates full-SVD wide external value logic unless clearly framed as partial-SVD | Acceptable but lower priority than tall because Sprint 123 already added a wide full-SVD external fixture. |
| `partial_svd_rankdef_diag_6x4_k3` | Rank-deficient value-only | Covers `k` crossing a zero or near-zero singular slot | Medium; threshold and zero handling must be explicit | Defer unless Day 10 defines rank/zero tolerance as the main claim. |
| `partial_svd_repeated_diag_6_k3` | Repeated-spectrum value/subspace | Exercises equal leading values | Medium-high; vector/subspace comparison is required for meaningful evidence beyond values | Defer until subspace metric exists. |
| `partial_svd_clustered_diag_6_k3` | Clustered-spectrum convergence | Exercises close singular values | High; convergence budget and ordering semantics are unresolved | Defer. |
| `partial_svd_vector_residual_diag_6_k2` | Vector residual | External fixture could check residuals rather than raw vector signs | Medium; needs helper protocol for vector output and residual metrics | Defer unless Day 10 stays residual-only and avoids direct vector comparison. |
| `partial_svd_lowrank_rect_5x4_k2` | Low-rank approximation | Connects partial-SVD output to rank-k reconstruction error | High; low-rank optimality and sparse output semantics are separate owners | Defer. |

## Sign, Subspace, Convergence, and Tolerance Policy

- Value-only fixtures may compare ordered singular values when values are well separated and the fixture explicitly defines `k`.
- Direct vector equality is not a valid external comparison because singular vectors may flip signs.
- Vector evidence must use sign-invariant residuals such as `||A v_i - sigma_i u_i||`, `||A^T u_i - sigma_i v_i||`, and orthogonality checks.
- Repeated or clustered spectra must use projection/subspace metrics rather than per-vector identity.
- Convergence-budget evidence must state the option surface, iteration cap, residual tolerance, fixture condition, and failure interpretation before implementation.
- Rank-deficient fixtures must state the zero/near-zero threshold and whether `k` intentionally crosses the numerical rank.
- Missing `python3` remains a skip through the external-reference helper; helper `ERROR` output remains a test failure.
- Windows external-reference helper behavior remains explicitly skipped unless the test framework changes.
- Value-only partial-SVD external tolerance remains `1e-8` unless Day 10 documents a fixture-specific reason.

## Duplicate Fences

- Do not duplicate `partial_svd_diag6_k2`; it already proves one bounded top-k external value lane.
- Do not re-use full-SVD external fixtures as partial-SVD proof unless `k`, output count, and partial-SVD-specific diagnostics are explicit.
- Do not use SuiteSparse smoke tests as external dense-reference parity.
- Do not move vector/subspace checks into the value-only external helper.
- Do not fold low-rank dense/sparse output checks into partial-SVD top-k value evidence.
- Do not treat timing smoke as a performance or convergence guarantee.

## Non-Claim Register

Day 9 preserves the following non-claims:

- no LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or vendor-backend parity claim;
- no broad partial-SVD external parity claim;
- no singular-vector or subspace external parity claim;
- no repeated-spectrum or clustered-spectrum external correctness claim;
- no broad convergence-budget guarantee;
- no low-rank global optimality claim;
- no package, platform, ABI, performance, or state-of-the-art claim.

## Day 10 Decision Checklist

Day 10 may implement a bounded lane only if it can answer all of the following:

1. Which semantic class is being tested: value, vector residual, subspace, convergence, rank-deficient, rectangular, or low-rank?
2. What is the exact fixture key, matrix, `k`, expected output shape, and tolerance?
3. Does the fixture duplicate `partial_svd_diag6_k2` or full-SVD external evidence?
4. Are signs, vector basis, repeated/clustered spectra, and rank thresholds either irrelevant or explicitly defined?
5. What does failure mean: fixture regression, convergence-budget miss, unsupported external helper, or broader claim failure?
6. Which files may change, and which files must remain untouched?
7. What focused helper and test commands prove the change?

## Recommendation

The lowest-risk Day 10 implementation candidate is a value-only rectangular diagonal fixture, preferably `partial_svd_tall_diag_8x5_k3`, because it extends partial-SVD external top-k evidence beyond the existing square diagonal fixture while avoiding sign, subspace, repeated-spectrum, clustered-spectrum, low-rank, and convergence-budget semantics.

All vector/subspace, rank-threshold, repeated/clustered spectrum, convergence-budget, and low-rank optimality lanes should remain deferred unless Day 10 narrows one lane to a residual-only or value-only proof with explicit failure semantics.
