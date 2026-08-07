# Day 2 Partial-SVD Residual Reaudit

## Scope

Day 2 re-ranks the partial-SVD residual candidates from the Sprint 140
intake and selects one bounded behavior family for complete closure. The
selection must fit the existing solver contract, avoid broad external parity
claims, and allow fixture/oracle design to proceed without relying on raw
singular-vector identity.

## Scoring Rubric

Scores use a 1-5 scale.

| Score field | Meaning |
| --- | --- |
| User-facing risk | Higher means the residual is more likely to mislead a downstream user. |
| Determinism | Higher means the fixture and oracle can be made repeatable locally. |
| Fixture complexity | Higher means more fixture, oracle, or comparison design is required. |
| Validation fit | Higher means existing Sprint 138/139 validation architecture can prove it. |
| Closure fit | Higher means the residual can be fully closed inside Sprint 140. |

## Residual Ranking

| Rank | Candidate | Current evidence | Remaining gap | Risk | Determinism | Fixture complexity | Validation fit | Closure fit | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Clustered/repeated top-k subspace plus convergence-budget closure | Full SVD has repeated-value coverage; partial-SVD helpers already prove singular values, vector residuals, projectors, and fail-closed `max_iter` behavior on simpler fixtures. | No maintained partial-SVD corpus lane proves clustered or repeated singular-value ambiguity with subspace-safe comparison and budget recovery. | 5 | 4 | 3 | 4 | 5 | Select |
| 2 | Rank-deficient range-projector corpus promotion | `partial_svd_rankdef_diag6x4_k2_range_projector` checks rank, singular values, U/V projectors, residuals, and orthogonality. | Evidence is helper-local and not tied to maintained corpus/oracle rows or selected Sprint 140 claim wording. | 4 | 5 | 2 | 4 | 4 | Backup |
| 3 | Existing max-iteration fail-closed diagonal budget proof | `partial_svd_max_iter_fail_closed_diag6_k2` proves tight-budget `SPARSE_ERR_NOT_CONVERGED` and default-budget recovery. | Standalone fixture does not close the repeated/clustered edge-case handoff; best used as the selected residual's budget dimension. | 4 | 5 | 2 | 4 | 3 | Fold into selected candidate |
| 4 | Near-zero singular-value and rank-threshold behavior | Rank helpers and tolerance-bearing APIs exist. | Closing this well would require a sharper public threshold policy and broader rank/null-space wording than Sprint 140 should claim. | 4 | 3 | 4 | 3 | 2 | Defer |
| 5 | Rectangular/nonsymmetric vector-residual expansion | Existing dense-reference fixtures cover tall and nonsymmetric rectangular partial-SVD lanes with residual checks. | Current evidence is not the highest uncovered risk and would drift toward broad rectangular/nonsymmetric parity claims if overextended. | 3 | 4 | 3 | 3 | 2 | Defer |
| 6 | Low-rank Frobenius or sparse drop-tolerance optimality | `partial_svd_lowrank_diag6x4_k2_frobenius_optimality` and sparse/dense low-rank consistency tests exist. | Product-facing low-rank optimality and sparse-output/drop-tolerance behavior are broader than one partial-SVD residual closure. | 3 | 4 | 4 | 3 | 2 | Defer |
| 7 | Optional SuiteSparse or external-data partial-SVD corpus | Optional external-data support-tier patterns exist elsewhere. | Requires support-tier promotion, availability policy, and cross-environment proof outside this sprint's bounded closure. | 5 | 2 | 5 | 1 | 1 | Out of scope |
| 8 | Broad LAPACK/NumPy/SciPy partial-SVD parity | Dense-reference helper exists for named fixtures only. | Broad external parity is an explicit non-claim and cannot be closed by a small deterministic fixture batch. | 5 | 1 | 5 | 1 | 1 | Out of scope |

## Evidence And Gap Map

| Candidate | Evidence to reuse | Gap to close or defer |
| --- | --- | --- |
| Clustered/repeated top-k subspace plus budget closure | `tests/test_svd_partial_helpers.h` vector-residual, projector, orthogonality, and max-iteration patterns; Sprint 138 corpus manifest/schema/oracle/report architecture; Sprint 139 fixture-local closure pattern. | Define one deterministic partial-SVD fixture key, expected singular values, projector/subspace comparison, vector-residual rows, tight-budget fail-closed status, default-budget recovery, and claim wording. |
| Rank-deficient range-projector promotion | Existing `partial_svd_rankdef_diag6x4_k2_range_projector` helper test and `sparse_svd_rank` check. | Promote to corpus/oracle only if selected candidate proves too ambiguous; otherwise keep as already-covered helper evidence. |
| Max-iteration fail-closed diagonal budget | Existing `partial_svd_max_iter_fail_closed_diag6_k2` helper test. | Use its status and allocation expectations as the model for selected fixture budget proof; do not make it the only Sprint 140 edge-case closure. |
| Near-zero rank threshold | Public `tol` and rank surfaces plus rank-deficient tests. | Requires a policy decision about numerical threshold behavior across fixtures and docs. |
| Rectangular/nonsymmetric expansion | `partial_svd_tall_diag_8x5_k3` and `partial_svd_nonsym_rect10x8_k3` dense-reference lanes. | Current helpers already cover bounded cases; further work does not address the repeated/clustered ambiguity gap. |
| Low-rank Frobenius/drop tolerance | Dense low-rank Frobenius helper and sparse low-rank consistency tests. | Needs low-rank product semantics and sparse approximation wording beyond selected partial-SVD residual proof. |
| Optional SuiteSparse/external data | Optional-data skip patterns and broader corpus architecture. | Needs reviewed support-tier promotion before pass evidence is meaningful. |
| Broad external parity | Named dense-reference helper script. | Explicit non-goal; retain as bounded reference evidence only. |

## Selected Priority Residual

Working name: `partial_svd_clustered_repeated_subspace_budget_v1`.

The selected residual is a deterministic clustered/repeated leading-spectrum
partial-SVD fixture with subspace-safe comparison and convergence-budget proof.
It should close the following bounded behavior family:

- singular values for the selected top-k fixture are correct within the chosen
  fixture tolerance;
- U and V comparisons use projector or subspace distance rather than raw vector
  identity;
- vector residuals satisfy `A*v ~= sigma*u` and `A^T*u ~= sigma*v`;
- U and V factors remain orthonormal within fixture tolerance;
- a deliberately tight iteration budget fails closed with
  `SPARSE_ERR_NOT_CONVERGED` and no published partial factor arrays;
- the default budget recovers on the same fixture and satisfies the selected
  singular-value, residual, and subspace checks.

This closure stays inside named fixture-local evidence. It does not claim broad
partial-SVD correctness, broad repeated-spectrum coverage, convergence-rate
behavior, performance, partial-result guarantees, or external-library parity.

## Backup Candidate

Working name: `partial_svd_rankdef_range_projector_budget_v1`.

If the clustered/repeated fixture cannot distinguish valid basis ambiguity from
a solver defect with a simple deterministic oracle, fall back to promoting the
existing rank-deficient range-projector lane into the maintained corpus/oracle
architecture. The backup is lower priority because it is already covered by a
focused helper test and does not directly close the Sprint 138/139 handoff for
clustered and repeated singular values.

## Out-Of-Scope Residuals

| Residual | Defer reason |
| --- | --- |
| Broad LAPACK/NumPy/SciPy parity | Explicit non-claim; named dense references are enough for fixture-local evidence. |
| Optional SuiteSparse partial-SVD corpus | Requires reviewed support-tier promotion and external-data availability policy. |
| Performance or convergence-rate proof | Sprint 140 budget checks prove status behavior, not runtime or iteration-complexity claims. |
| Near-zero threshold policy | Needs broader numerical policy design across rank, null-space, and tolerance surfaces. |
| Low-rank productization or sparse approximation optimality | Exceeds one residual closure and risks broad product claims. |
| Report-index normalization | Planned for Sprint 141; Sprint 140 can emit or inspect reports only as supporting evidence. |
| Raw singular-vector identity across repeated spectra | Invalid comparison target because basis rotations are acceptable in degenerate subspaces. |

## Day 3 Handoff

Day 3 should convert the selected residual into a closure contract with exact
fixture dimensions, singular-value pattern, expected rows, oracle rows,
tolerances, proof owner, validation commands, and documentation boundaries.
