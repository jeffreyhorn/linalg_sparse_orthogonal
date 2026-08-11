# Sprint 150 Day 2: QR Family Candidate Audit

## Purpose

Audit candidate QR fixture families and decide which are bounded enough to
consider for Sprint 150 complete closure. Day 2 does not select the families;
it prepares evidence for the Day 3 selection decision.

## Candidate Summary

| Candidate Family | Existing Evidence | Closure Value | Implementation Risk | Report Readiness | Claim-Boundary Clarity | Day 2 Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Rank-deficient rectangular | Strong owner-local coverage in `tests/test_qr.c`, `tests/test_qr_solve.c`, and `tests/test_qr_helpers.h`; one maintained Sprint 139 corpus fixture already exists. | 5 | 2 | 4 | 5 | 16 |
| Underdetermined minimum-norm | Strong exact-value and residual/norm owner-local coverage in `tests/test_qr_solve.c` and `tests/test_colamd.c`. | 5 | 3 | 3 | 4 | 15 |
| Reorder/COLAMD QR | Good owner-local functional coverage in `tests/test_colamd.c` and `tests/test_qr.c`, but semantics mix ordering validity, residual, fill, and non-performance wording. | 4 | 4 | 2 | 3 | 9 |

Scoring scale: closure value and readiness/clarity are better when higher;
implementation risk is better when lower. The score is
`closure value + report readiness + claim-boundary clarity - implementation risk`.

## Rank-Deficient Rectangular Candidate

### Current Coverage

| Evidence | File / Test Surface | Notes |
| --- | --- | --- |
| Maintained corpus seed | `tests/test_qr_corpus.c`; `tests/corpus/manifests/fixtures.tsv`; `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv` | Fixture-local rank `3`, nullity `1`, and normalized nullspace residual `<= 1e-10`. |
| Duplicate-column rank and residual | `tests/test_qr_solve.c` tests `qr_rankdef_duplicate_5x4_rank_only` and `qr_rankdef_duplicate_5x4_residual_only` | Existing dense-reference helper supplies rank or residual values, but rows are not source-controlled corpus metadata. |
| Dependent-row residual | `tests/test_qr_solve.c` test `qr_rankdef_dependent_row_4x3_residual_only` | Good residual-only candidate with explicit RHS and bounded tolerance. |
| Nullspace projector/subspace | `tests/test_qr.c` tests `qr_rankdef_duplicate_5x4_nullspace_projector`, `qr_rankdef_dependent_row_4x3_nullspace_projector`, and `qr_rankdef_wide_3x5_nullspace_subspace` | Uses projector/subspace metrics instead of raw basis identity. |
| Threshold-rank families | `tests/test_qr.c` tests `qr_rank_threshold_diag4_family`, `qr_rank_threshold_diag4_scaled_family`, `qr_rank_threshold_duplicate_5x4_perturbed_family`, and `qr_rank_threshold_dependent_row_4x3_perturbed_family` | Valuable, but global rank-threshold policy is a known non-claim and may be too broad for Sprint 150. |

### Metadata Needs

- Fixture rows for two or three selected rank-deficient rectangular fixtures,
  likely from duplicate-column, dependent-row, and wide-nullspace shapes.
- Generator rows with deterministic structure/value hashes and regeneration
  commands.
- Expected rows for rank, nullity, residual norm, and optionally projector or
  subspace distance.
- Claim-scope rows that explicitly stay fixture-family local.
- Non-claims for global rank-threshold policy, raw basis equality, broad
  rank-deficient solve behavior, SuiteSparse parity, and external-library
  parity.

### Oracle Readiness

Strong. Existing tests already use rank, nullity, residual, projector, and
subspace-safe comparisons. The existing Sprint 139 oracle path can be extended
by adding stable fixture keys and expected rows.

### Risks

- Mirroring test-local fixture builders into corpus metadata can drift, as
  Sprint 139 already exposed with the QR fixture `nnz` mismatch.
- Threshold-rank candidates risk widening into global tolerance policy if
  selected too early.
- Dense-reference helper outputs are generated on demand and need conversion to
  source-controlled expected rows before promotion.

## Underdetermined Minimum-Norm Candidate

### Current Coverage

| Evidence | File / Test Surface | Notes |
| --- | --- | --- |
| Exact 2x4 minimum norm | `tests/test_qr_solve.c::test_qr_solve_minnorm_underdetermined_known_solution`; `tests/test_colamd.c::test_minnorm_2x4_known` | Exact solution `[0.5, 0.5, 0.5, 0.5]`, residual, and norm `1.0`. |
| External dense reference 2x4 | `tests/test_qr_solve.c::test_qr_external_dense_reference_underdetermined_minnorm_2x4` | Compares solution entries, residual, and norm against helper output. |
| Exact 3x6 and 5x10 | `tests/test_colamd.c::test_minnorm_3x6`; `tests/test_colamd.c::test_minnorm_5x10` | Existing exact expected vectors, residual checks, and norm checks. |
| COLAMD minimum-norm | `tests/test_colamd.c::test_minnorm_with_colamd` | Checks residual, exact expected values, norm, and COLAMD option path for a 2x5 system. |
| Rank-deficient and zero-row minimum norm | `tests/test_colamd.c::test_minnorm_rank_deficient`; `tests/test_colamd.c::test_minnorm_zero_row` | Good edge candidates, but require careful consistency and status semantics. |
| Pseudoinverse cross-check | `tests/test_colamd.c::test_minnorm_vs_pinv` | Useful bounded cross-check, but SVD-pseudoinverse must not become a global oracle claim. |

### Metadata Needs

- Fixture rows for selected underdetermined matrices, RHS policy, rank/nullity
  where meaningful, and exact-solution availability.
- Generator rows for deterministic small systems.
- Expected rows for residual, solution norm, selected exact solution values, and
  status.
- Claim-scope rows that identify the solution norm and residual as the promoted
  metrics.
- Non-claims for global minimum-norm guarantee, SVD-pseudoinverse-as-global
  oracle, broad rank-deficient recovery, and performance/platform parity.

### Oracle Readiness

Moderate to strong. Exact small fixtures are already available and have clear
expected vectors and norms. The main missing piece is source-controlled expected
metadata and generated oracle/report rows that separate residual from norm and
exact-value checks.

### Risks

- Exact solution vectors are more brittle than residual/norm claims and may be
  sensitive to future algorithmic changes that still preserve minimum-norm
  semantics.
- Cross-checking against SVD pseudoinverse could widen claims if docs do not
  state it is a bounded local comparison.
- COLAMD minimum-norm overlaps with the reorder family; selection should avoid
  double-counting one fixture as two broad claims.

## Reorder/COLAMD QR Candidate

### Current Coverage

| Evidence | File / Test Surface | Notes |
| --- | --- | --- |
| QR+COLAMD solve | `tests/test_colamd.c::test_qr_colamd_solve` | Checks finite bounded residual and bounded `Ax-b` error for a 6x4 system. |
| QR COLAMD vs AMD/natural | `tests/test_colamd.c::test_qr_colamd_vs_amd` | Compares residuals for natural, AMD, and COLAMD; does not define broad ordering superiority. |
| QR COLAMD sparse mode | `tests/test_colamd.c::test_qr_colamd_sparse_mode` | Exercises sparse-mode QR with COLAMD and bounded residual output. |
| QR reorder in broad QR tests | `tests/test_qr.c::test_qr_reorder_amd_solve`, `test_qr_reorder_nos4_fillin`, and `test_qr_reorder_none` | Existing AMD/natural behavior and fill/residual checks. |
| COLAMD public API and fill tests | `tests/test_colamd.c` public API, west0067, steam1, and fill comparisons | Useful context but not QR-family proof by itself. |

### Metadata Needs

- Fixture rows for explicitly selected QR+COLAMD matrices and RHS policy.
- Generator rows for synthetic ordering-sensitive patterns or optional-data
  policy rows for SuiteSparse matrices if selected.
- Expected rows for status, residual, permutation validity, and possibly fill
  diagnostics.
- Claim-scope rows that avoid performance or ordering-optimality claims.
- Non-claims for COLAMD parity, broad reorder optimality, fill improvement
  guarantee, SuiteSparse corpus completeness, platform parity, and performance.

### Oracle Readiness

Weak to moderate. Existing tests prove bounded behavior, but corpus promotion
needs clearer semantics: residual/status is straightforward, while fill and
ordering comparisons risk sounding like performance or optimality claims.

### Risks

- Ordering/fill metrics are easy to overstate.
- Optional SuiteSparse paths are disabled or local-data dependent and cannot be
  pass evidence unless Sprint 150 explicitly promotes an optional-data policy.
- Selecting this family may consume most report/documentation effort because
  non-claim wording must be precise.

## Cross-Family Metadata Gap Table

| Required Row / Evidence | Rank-Deficient Rectangular | Underdetermined Minimum-Norm | Reorder/COLAMD QR |
| --- | --- | --- | --- |
| Fixture rows | Needed for duplicate, dependent-row, and/or wide-nullspace candidates | Needed for 2x4, 3x6, 5x10, rank-deficient, and/or zero-row candidates | Needed for synthetic QR+COLAMD candidates |
| Generator rows | Needed; deterministic small matrices already exist in helpers/scripts | Needed; deterministic small matrices already exist in tests | Needed; likely new synthetic generator metadata |
| Expected rows | Rank, nullity, residual, projector/subspace distance | Residual, solution norm, selected exact values, status | Residual, status, permutation/fill diagnostics if selected |
| Oracle extension | Extend `--include-solver-qr` beyond one fixture | Add minimum-norm operations to QR oracle rows | Add reorder/status/residual operations carefully |
| Focused proof owner | Extend `tests/test_qr_corpus.c` or add focused QR corpus helpers | Extend `tests/test_qr_corpus.c` with minimum-norm helpers | Likely add focused QR/COLAMD corpus tests rather than reuse broad `test_colamd` |
| Report rows | Ready after expected rows exist | Ready after minimum-norm operation semantics exist | Needs careful non-claim wording |
| Documentation | Straightforward from Sprint 139 pattern | Moderate; exact values vs semantic minimum-norm wording | Harder; avoid optimality/performance claims |

## Day 3 Selection Inputs

Recommended Day 3 posture:

1. Prefer selecting **rank-deficient rectangular** as the first Sprint 150
   family because it extends the Sprint 139 pattern directly and has the
   clearest subspace-safe semantics.
2. Strongly consider **underdetermined minimum-norm** as the second family
   because exact small fixtures already exist and the claim can be bounded to
   residual plus norm.
3. Treat **reorder/COLAMD QR** as a possible third family only if Day 3 can
   define a narrow status/residual claim and avoid fill/performance or
   ordering-optimality wording.
4. Avoid threshold-rank policy as a primary Sprint 150 family unless the sprint
   explicitly narrows it to named fixtures and tolerances.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Candidate families are compared with concrete repository evidence. | Complete | Tables cite current tests, helpers, manifests, and oracle/report scripts. |
| Each family has a closure/risk score. | Complete | Candidate summary scores rank-deficient, minimum-norm, and reorder/COLAMD families. |
| Family-selection inputs are ready for Day 3 without implementation bias. | Complete | Day 3 selection inputs recommend posture but defer final selection to the product decision artifact. |
