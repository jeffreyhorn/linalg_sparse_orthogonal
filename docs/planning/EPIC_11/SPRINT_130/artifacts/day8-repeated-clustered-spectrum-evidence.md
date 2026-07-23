# Sprint 130 Day 8 - Repeated And Clustered Spectrum Evidence

## Decision

Defer repeated and clustered partial-SVD subspace evidence for Day 8.

Day 7 accepted `partial_svd_repeated_diag6_k3_projector` as the lowest-risk
candidate only if the product could validate the whole repeated leading
subspace with analytic projector metrics. Day 8 preflighted that lane, and the
focused evidence check failed. The attempted test was removed, and
`docs/maintainer_guide.md` was not updated with any repeated/clustered
evidence claim.

## Attempted Lane

| Field | Attempted value |
| --- | --- |
| Fixture key | `partial_svd_repeated_diag6_k3_projector` |
| Matrix | 6x6 diagonal `diag(7, 7, 7, 3, 2, 1)` |
| `k` | `3` |
| Options | `compute_uv = 1`, `economy = 1`, default iteration and tolerance settings. |
| Oracle | Analytic top-3 singular values and analytic coordinate projectors onto coordinates `0..2`. |
| Primary metric | Left and right projector Frobenius errors against the analytic leading subspace. |
| Secondary metrics | Top-3 singular-value agreement, both singular-triplet residual equations, U/V orthogonality, shape and vector-availability checks. |
| Intended tolerance | `1e-8` for this exact diagonal repeated fixture only. |
| Post-block gap | `4`, between singular values `7` and `3`. |

## Preflight Result

The focused SVD run produced this diagnostic for the attempted projector lane:

```text
partial-SVD repeated diag6_k3 projector: sigma=5.000e+00, PU=2.000e+00, PV=2.000e+00, Av=9.938e-15, Atu=2.808e-15, U_ortho=6.661e-16, V_ortho=5.828e-16, gap=4.000e+00
```

The triplet residual and orthogonality diagnostics were small, but the
singular-value and projector checks failed the evidence contract:

| Metric | Intended bound | Observed |
| --- | --- | --- |
| Max top-3 singular-value difference from `7` | `< 1e-8` | `5.000e+00` |
| Left projector Frobenius error | `< 1e-8` | `2.000e+00` |
| Right projector Frobenius error | `< 1e-8` | `2.000e+00` |
| Max `A v - sigma u` residual | `< 1e-8` | `9.938e-15` |
| Max `A^T u - sigma v` residual | `< 1e-8` | `2.808e-15` |
| U orthogonality error | `< 1e-8` | `6.661e-16` |
| V orthogonality error | `< 1e-8` | `5.828e-16` |

This means the current partial-SVD path can publish internally consistent
triplets for the returned vectors, but it does not establish the full
multiplicity-3 leading repeated subspace for this fixture under the Day 7
projector policy.

## Final Day 8 Action

| Surface | Action |
| --- | --- |
| `tests/test_svd_partial_helpers.h` | No accepted repeated/clustered test remains. The attempted projector helper/test was removed after failing the focused gate. |
| `tests/test_svd.c` | No repeated/clustered test registration remains. |
| `docs/maintainer_guide.md` | No repeated/clustered SVD evidence wording was added. |
| Sprint 130 artifacts | This deferral package records the failed preflight and carry-forward gates. |

## Deferrals

| Deferred lane | Reason | Future owner and promotion gate |
| --- | --- | --- |
| `partial_svd_repeated_diag6_k3_projector` | Focused preflight failed singular-value and projector metrics for the whole repeated leading block. | Future partial-SVD implementation/convergence owner must change the algorithm or option surface enough to recover the complete repeated block, then rerun projector evidence. |
| Partial selection inside a repeated block | Selecting `k=2` from a multiplicity-3 block needs containment or principal-angle-to-containing-subspace semantics. | Future subspace owner must define containment metrics and pass/fail interpretation. |
| Clustered-spectrum projector fixture | Requires declared within-cluster gap, post-cluster gap, ordered/set value policy, projector tolerance, and convergence-budget interpretation. | Clustered/convergence owner must define gap and budget policy before implementation. |
| Day 6 near-zero clustered tail | Needs rank threshold, zero singular-value tolerance, and range/null-space split before promotion. | Days 9-10 rank-deficient subspace owner. |
| Corpus clustered spectra | Optional data, conditioning, runtime, support tier, and residual windows are not owned by this lane. | Days 11-12 corpus owner. |
| Public solver-selection wording | No repeated or clustered evidence was accepted. | Day 14 claim gate should record no-update rationale unless later evidence lands. |

## Non-Claim Register

Day 8 does not claim:

- repeated-spectrum partial-SVD projector correctness;
- clustered-spectrum partial-SVD projector correctness;
- raw singular-vector equality for repeated or clustered spectra;
- stable ordering inside a repeated block;
- partial selection through a repeated block;
- clustered-spectrum convergence or budget behavior;
- rank-deficient range/null-space behavior;
- corpus clustered-spectrum parity;
- low-rank optimality;
- public solver-selection wording readiness;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity.

## Validation Plan

Because the accepted Day 8 outcome is documentation-only after removing the
failed attempted test, required validation is:

1. `make build/test_svd && ./build/test_svd`
2. `git diff --check`
3. focused Sprint 130 markdown trailing-whitespace scan

The focused SVD run is included because Day 8 briefly touched C/header tests
and must prove the failed attempted lane was fully removed.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Repeated/clustered policy is applied to the selected lane. | Complete | The Day 7 accepted lane was attempted and rejected after failing its projector/value evidence gate. |
| Accepted evidence avoids raw vector equality. | Complete | No repeated/clustered evidence was accepted; raw vector equality remains forbidden. |
| Clustered and ambiguous repeated cases have blocker and owner notes. | Complete | Deferral table records repeated projector, partial repeated-block, clustered, near-zero tail, corpus, and solver-selection gates. |
| Focused and documentation validation are run for the final Day 8 state. | Complete | `make format && make build/test_svd && ./build/test_svd`, `git diff --check`, and the focused Sprint 130 markdown whitespace scan passed. |
