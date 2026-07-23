# Sprint 130 Day 6 - Nonsymmetric Rectangular Evidence

## Purpose

Day 6 applies the Day 5 nonsymmetric rectangular gate. It adds one bounded
partial-SVD residual evidence lane for a deterministic non-diagonal 10x8
fixture, but narrows the accepted lane from the Day 5 candidate `k=4` to
`k=3` after spectrum preflight diagnostics showed the fourth singular value is
inside a near-zero clustered tail.

This is intentional claim control: the accepted evidence covers the stable
top-3 triplets only. It does not claim clustered, rank-deficient, subspace,
low-rank optimality, convergence-budget, wide nonsymmetric, or solver-selection
behavior.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 130 Day 5 nonsymmetric gate | Defines the candidate fixture, vector-residual metrics, orientation policy, and fallback rules. |
| Sprint 130 Day 2 metric map | Requires metric, tolerance, oracle, diagnostics, failure class, duplicate fence, and non-claims before implementation. |
| `tests/test_svd_partial_helpers.h` | Owns partial-SVD helper fixtures, vector-residual checks, and the existing nonsymmetric internal consistency test. |
| `tests/test_svd.c` | Owns external dense-reference fixture whitelisting and SVD test registration. |
| `tests/svd_external_dense_reference.py` | Owns external singular-value fixtures for bounded SVD parity checks. |
| `docs/maintainer_guide.md` | Owns maintainer-facing evidence wording and non-claims. |

## Spectrum Preflight

The Day 5 candidate proposed `partial_svd_nonsym_rect10x8_k4`. Before
implementation, the 10x8 fixture spectrum was measured with the external dense
reference helper:

| Index | Singular value |
| --- | --- |
| 1 | `16.36583372940877` |
| 2 | `8.362692244678056` |
| 3 | `4.369689127707603` |
| 4 | `5.457404879457828e-08` |
| 5 | `4.457615440305726e-08` |
| 6 | `4.216589889708944e-08` |
| 7 | `3.426068564941283e-08` |
| 8 | `0.0` |

The fourth through seventh values are a near-zero cluster. Publishing
individual top-4 triplet residual evidence would blur vector-residual evidence
with clustered-tail behavior that belongs to the repeated/clustered subspace
owner. Day 6 therefore accepts only the stable top-3 lane.

## Accepted Evidence Lane

| Field | Value |
| --- | --- |
| Vector-residual fixture | `partial_svd_vector_residual_nonsym_rect10x8_k3` |
| External singular-value fixture | `partial_svd_nonsym_rect10x8_k3` |
| Matrix | 10x8 deterministic non-diagonal matrix with entries `(i + 1) / (j + 1)` when `(i + j) % 3 != 0`, and zero otherwise. |
| `k` | `3` |
| Options | `compute_uv = 1`, `economy = 1`, default iteration and tolerance settings. |
| External oracle | Top-3 singular values emitted by `tests/svd_external_dense_reference.py`. |
| Product-owned residual metrics | `||A v_i - sigma_i u_i||_2` and `||A^T u_i - sigma_i v_i||_2`. |
| Orthogonality metrics | U-column and V-row orthogonality errors. |
| Shape diagnostics | Returned `m`, `n`, `k`, `sigma`, `U`, and `Vt` are checked. |
| Assert tolerance | `1e-8` for singular-value agreement, both residual equations, and both orthogonality checks. |

The tolerance is fixture-specific. It is accepted because the focused
diagnostic run stayed several orders of magnitude below `1e-8`:

| Metric | Focused diagnostic |
| --- | --- |
| Max singular-value difference | `2.842e-14` |
| Max `A v - sigma u` residual | `7.085e-15` |
| Max `A^T u - sigma v` residual | `1.219e-14` |
| U orthogonality error | `2.220e-16` |
| V orthogonality error | `7.571e-16` |

## Implementation Summary

| File | Change |
| --- | --- |
| `tests/svd_external_dense_reference.py` | Added `build_partial_svd_nonsym_rect10x8_k3`, fixture dispatch, and top-3 output slicing. |
| `tests/test_svd_partial_helpers.h` | Added the shared 10x8 nonsymmetric fixture builder and the external value plus triplet-residual test. Reused the builder in the existing internal nonsymmetric test. |
| `tests/test_svd.c` | Added the new helper key to the external-reference whitelist and registered the new vector-residual test. |
| `docs/maintainer_guide.md` | Added bounded nonsymmetric rectangular singular-value and vector-residual fixture names while preserving broad non-claims. |

## Validation

Focused validation completed before this artifact was written:

1. `python3 -m py_compile tests/svd_external_dense_reference.py`
2. `python3 tests/svd_external_dense_reference.py partial_svd_nonsym_rect10x8_k3`
3. `make build/test_svd && ./build/test_svd`

The focused SVD executable reported:

| Result | Value |
| --- | --- |
| Tests run | `111` |
| Tests failed | `0` |
| Assertions | `1861` |

Full quality validation also passed after documentation edits because Day 6
touched Python, C, and header-backed tests.

## Deferrals

| Deferred lane | Reason | Future owner and promotion gate |
| --- | --- | --- |
| `partial_svd_nonsym_rect10x8_k4` | The fourth singular value is in a near-zero clustered tail; individual vector residual evidence would overstate basis stability. | Days 7-8 clustered/repeated subspace owners must define projector or principal-angle metrics, or a future residual owner must choose a fixture with a stable fourth value. |
| Wide nonsymmetric rectangular residual | Day 6 covers one tall 10x8 nonsymmetric fixture only. | Future rectangular owner must define a wide non-diagonal fixture, external singular-value oracle, both triplet residuals, shape policy, and bounded wording. |
| Nonsymmetric subspace evidence | Raw vector equality is forbidden and projector/principal-angle metrics are not part of this lane. | Days 7-8 subspace owner. |
| Nonsymmetric rank-deficient evidence | The near-zero tail is not promoted into rank/null-space evidence without rank threshold and null-space policy. | Days 9-10 rank-deficient subspace owner. |
| Nonsymmetric low-rank optimality | Residual evidence does not prove Frobenius, spectral, reconstruction, sparse-output, or drop-tolerance optimality. | Day 12 low-rank owner. |
| Nonsymmetric convergence-budget behavior | Day 6 uses default partial-SVD options and does not define budget exhaustion semantics. | Day 13 convergence owner. |
| Public solver-selection wording | One bounded nonsymmetric fixture is insufficient for public solver-selection guidance. | Day 14 claim gate after all Sprint 130 evidence is reconciled. |

## Non-Claim Register

Day 6 does not claim:

- top-4 nonsymmetric rectangular triplet stability for the 10x8 fixture;
- wide nonsymmetric rectangular partial-SVD residual behavior;
- raw vector equality, sign, orientation, ordering, or unique-basis stability;
- repeated-spectrum or clustered-spectrum subspace behavior;
- rank-deficient range/null-space or threshold behavior;
- SuiteSparse corpus residual parity;
- low-rank global optimality;
- convergence-budget guarantees or partial-result semantics;
- public solver-selection wording readiness;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Apply the Day 5 gate to the selected nonsymmetric candidate. | Complete | The `k=4` candidate was preflighted and narrowed to `k=3` because of the near-zero clustered tail. |
| Implement one accepted evidence lane if fixture, oracle, metrics, tolerance, diagnostics, and non-claims are ready. | Complete | `partial_svd_vector_residual_nonsym_rect10x8_k3` checks external singular values, both triplet residuals, orthogonality, and shape diagnostics. |
| Otherwise write an explicit deferral package. | Complete | The top-4, wide nonsymmetric, subspace, rank-deficient, low-rank, convergence, and solver-selection lanes are deferred with owners. |
| Run focused validation for touched helper, SVD tests, and Python fixture changes. | Complete | Python compile/helper invocation and focused `test_svd` run passed. |
| Run full quality and diff hygiene validation after all Day 6 edits. | Complete | `make format && make lint && make test`, `git diff --check`, and the focused Sprint 130 markdown whitespace scan passed. |
