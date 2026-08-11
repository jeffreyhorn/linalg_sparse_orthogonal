# Sprint 151 Day 2: Partial-SVD Family Candidate Audit

## Purpose

Audit candidate partial-SVD fixture families and decide which ones are bounded
enough for complete Sprint 151 closure. Day 2 compares concrete repository
evidence, identifies metadata/oracle/report needs, scores closure value and
risk, and prepares Day 3 family selection without making the selection yet.

## Candidate Summary

| Candidate Family | Existing Evidence | Closure Value | Risk | Day 2 Score |
| --- | --- | ---: | ---: | ---: |
| Rank-deficient rectangular range projector | `test_partial_svd_rankdef_diag6x4_k2_range_projector` | High | Low | 17 |
| Sparse low-rank output | `test_lowrank_sparse_*`, `test_sparse_svd_lowrank_outer_product_*`, `example_svd_lowrank.c` | High | Medium | 15 |
| Convergence/fail-closed behavior | `test_partial_svd_max_iter_fail_closed_diag6_k2`, Sprint 140 tight-budget tests | Medium | Low | 14 |
| External dense-reference vector residuals | `partial_svd_diag6_k2`, `partial_svd_tall_diag_8x5_k3`, `partial_svd_nonsym_rect10x8_k3` | Medium | Medium | 12 |
| Repeated spectra beyond Sprint 140 | Sprint 140 corpus fixture, `test_svd_repeated` | Medium | Medium | 11 |

Scoring rubric:

- closure value: 0-5
- implementation risk: 0-5, inverted in final score
- report/oracle readiness: 0-5
- claim-boundary clarity: 0-5

Day 2 score is a qualitative total out of 20, not a selection decision.

## Candidate 1: Rank-Deficient Rectangular Range Projector

### Existing Evidence

Primary owner-local evidence:

- `tests/test_svd_partial_helpers.h`
- `test_partial_svd_rankdef_diag6x4_k2_range_projector`

The test builds a 6 by 4 diagonal matrix with diagonal values
`9, 6, 0, 0`, requests `k=2`, verifies `sparse_svd_rank(A, 1e-8) == 2`,
checks top-2 singular values `{9, 6}`, checks left and right coordinate-range
projectors, verifies `A v ~= sigma u`, verifies `A^T u ~= sigma v`, and
checks U/V orthogonality.

### Metadata Needs

- fixture row for a generated 6 by 4 rank-deficient diagonal matrix;
- generator row with deterministic diagonal values, `k=2`, expected rank `2`,
  and nullity `2`;
- expected rows for:
  - singular values;
  - rank;
  - left range projector;
  - right range projector;
  - vector residuals;
  - orthogonality;
  - default success status;
- claim-scope wording tied to range/projector evidence only;
- non-claims for broad null-space behavior and raw singular-vector identity.

### Oracle And Report Readiness

High. The fixture is deterministic, diagonal, and can reuse Sprint 140-style
generated-reference oracle rows. Projector and triplet-residual comparison
kinds already exist. Expected row names and tolerances can be made
source-controlled without adding new schema fields.

### Risks

- The fixture is diagonal and may look too synthetic unless documentation
  describes it as fixture-local range/projector evidence.
- Rank-deficient rectangular wording can be overread as broad null-space or
  pseudoinverse/minimum-norm behavior. Non-claims must be explicit.

### Day 2 Score

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Closure value | 5 | Directly addresses Sprint 151 rank-deficient rectangular family. |
| Low implementation risk | 5 | Existing owner-local test already proves the needed assertions. |
| Report/oracle readiness | 4 | Current comparison kinds fit; only fixture-specific oracle wiring is needed. |
| Claim-boundary clarity | 3 | Needs careful wording to avoid broad null-space claims. |
| Total | 17 | Strong candidate for Day 3 selection. |

## Candidate 2: Sparse Low-Rank Output

### Existing Evidence

Primary owner-local evidence:

- `tests/test_svd.c`
- `test_lowrank_sparse_diagonal`
- `test_lowrank_sparse_sparsity`
- `test_lowrank_sparse_vs_dense`
- `test_lowrank_sparse_rank1`
- `test_lowrank_sparse_rectangular`
- `test_lowrank_rectangular_dense_sparse_consistency`
- `test_sparse_svd_lowrank_outer_product_matches_dense`
- `test_sparse_svd_lowrank_outer_product_corpus_safety`
- `examples/example_svd_lowrank.c`

The strongest fixture candidates are deterministic diagonal/rectangular
low-rank cases where sparse low-rank output can be compared by shape, selected
values, Frobenius difference from dense low-rank output, and expected zeroed
tail entries.

### Metadata Needs

- one or two fixture rows for generated low-rank sparse-output cases;
- generator rows for deterministic diagonal or rectangular matrices;
- expected rows for:
  - sparse output status;
  - output shape;
  - selected exact values;
  - output `nnz` or bounded `nnz` policy;
  - dense/sparse Frobenius difference;
  - dense low-rank residual where applicable;
- claim-scope wording for sparse-output construction, not sparse optimality;
- non-claims for performance, storage optimality, drop-tolerance optimality,
  and broad low-rank approximation quality.

### Oracle And Report Readiness

Medium-high. Value and residual comparison kinds exist, but sparse-output
structure/value expectations need careful encoding. If a structural row uses
`nnz` or selected coordinate values, Day 4 must define whether it is a value
row, diagnostic row, or a new expected-result convention.

### Risks

- Sparse low-rank output is product-facing and can be mistaken for performance
  or storage-optimality evidence.
- Drop tolerance can create brittle `nnz` expectations unless fixture values
  are well separated from thresholds.
- Dense/sparse comparison requires deterministic ordering and tolerance
  wording.

### Day 2 Score

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Closure value | 5 | Addresses a visible public API surface and Sprint 151 sparse-output item. |
| Low implementation risk | 3 | Existing tests help, but reportable structural semantics need design. |
| Report/oracle readiness | 3 | Existing kinds likely fit, but exact row encoding needs Day 4 work. |
| Claim-boundary clarity | 4 | Can stay narrow with shape/value/Frobenius rows and explicit non-claims. |
| Total | 15 | Strong candidate if Day 3 keeps the fixture set small. |

## Candidate 3: Convergence And Fail-Closed Behavior

### Existing Evidence

Primary owner-local evidence:

- `tests/test_svd_partial_helpers.h`
- `test_partial_svd_max_iter_fail_closed_diag6_k2`
- Sprint 140 corpus tests for tight-budget fail-closed behavior and recovery
  after failure

The owner-local test builds a 6 by 6 diagonal matrix with diagonal values
`9, 6, 3, 1, 0.5, 0.25`, requests `k=2`, verifies a tight `max_iter=1`
budget returns `SPARSE_ERR_NOT_CONVERGED`, checks `sigma`, `U`, and `Vt` are
not published on failure, then verifies default-budget recovery succeeds with
expected singular values and triplet residuals.

### Metadata Needs

- fixture row for deterministic diagonal convergence fixture;
- generator row with diagonal values, `k=2`, tight-budget policy, and
  expected default top-k values;
- expected rows for:
  - tight-budget status;
  - no partial arrays on failure;
  - default status;
  - default singular values;
  - default triplet residuals;
  - recovery-after-failure behavior if selected;
- claim-scope wording for fail-closed behavior only;
- non-claims for convergence rate, portable iteration counts, and partial
  result guarantees.

### Oracle And Report Readiness

High. Sprint 140 already established status and diagnostic rows for tight
budget behavior. This candidate mostly reuses that pattern on a non-repeated
diagonal fixture.

### Risks

- Could duplicate Sprint 140 too closely unless Day 3 defines what new
  behavior is being closed.
- Any iteration-count wording must remain out of scope because portability is
  not guaranteed.

### Day 2 Score

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Closure value | 3 | Useful, but Sprint 140 already owns one tight-budget fail-closed lane. |
| Low implementation risk | 5 | Existing test and comparison semantics are mature. |
| Report/oracle readiness | 4 | Status/diagnostic rows already fit. |
| Claim-boundary clarity | 2 | Needs clear distinction from Sprint 140 to avoid duplicate closure. |
| Total | 14 | Good supporting candidate, but likely not enough as the only expansion. |

## Candidate 4: External Dense-Reference Vector Residuals

### Existing Evidence

Primary owner-local evidence:

- `tests/test_svd_partial_helpers.h`
- `test_partial_svd_external_dense_reference_vector_residual_diag6_k2`
- `test_partial_svd_external_dense_reference_vector_residual_tall8x5_k3`
- `test_partial_svd_external_dense_reference_vector_residual_nonsym_rect10x8_k3`
- `tests/svd_external_dense_reference.py`

These tests use optional external reference data for singular values and then
verify project partial-SVD triplet residuals and orthogonality on square,
tall, and nonsymmetric rectangular fixtures.

### Metadata Needs

- fixture rows for one or more deterministic external-reference fixture keys;
- optional-data provenance rows if external data is treated as required input;
- expected rows for singular values, triplet residuals, and orthogonality;
- support-tier and skip/defer policy wording;
- claim-scope wording that external dense references are named fixture-local
  checks, not broad library parity.

### Oracle And Report Readiness

Medium. The comparison metrics are ready, but optional external data and
Windows skip behavior introduce support-tier complexity. If selected, Day 3
should prefer deterministic generated fixtures over optional external helper
dependencies unless the goal is explicitly external-reference provenance.

### Risks

- Easy to overclaim as LAPACK/NumPy/SciPy parity.
- Optional external data can turn into skip/defer rows rather than pass
  evidence.
- Windows helper skips complicate hosted-platform interpretation.

### Day 2 Score

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Closure value | 4 | Adds shape variety: square, tall, nonsymmetric rectangular. |
| Low implementation risk | 2 | Optional data and platform skips require careful support-tier policy. |
| Report/oracle readiness | 3 | Metrics are ready; provenance/report policy is less ready. |
| Claim-boundary clarity | 3 | Possible, but external parity overreach risk is high. |
| Total | 12 | Useful later or as optional support, but not the safest core Sprint 151 closure. |

## Candidate 5: Repeated Spectra Beyond Sprint 140

### Existing Evidence

Primary evidence:

- Sprint 140 corpus fixture `partial_svd_clustered_repeated_diag8x6_k3_v1`
- `tests/test_svd.c` repeated-spectrum full-SVD coverage
- `test_svd_repeated`

Sprint 140 already covers clustered/repeated top-3 singular values,
projectors, triplet residuals, orthogonality, default success, tight-budget
failure, and recovery for one generated 8 by 6 fixture.

### Metadata Needs

- another repeated-spectrum fixture only if it adds a clearly distinct
  property, such as a different rank, truncation boundary, or rectangular
  projector behavior;
- expected rows for singular values, projectors, residuals, and status;
- explicit non-claims for raw singular-vector identity and broad
  repeated-spectrum coverage.

### Oracle And Report Readiness

High for mechanics because Sprint 140 already owns the pattern. Medium for
value because a second repeated-spectrum fixture can duplicate the Sprint 140
claim unless it is tied to a distinct family.

### Risks

- Duplicates Sprint 140 without closing a new gap.
- Repeated subspaces are the highest-risk area for accidental raw-vector,
  sign, orientation, or basis-order claims.

### Day 2 Score

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Closure value | 2 | Sprint 140 already closed the strongest repeated-spectrum seed. |
| Low implementation risk | 4 | Mechanics are known. |
| Report/oracle readiness | 4 | Existing oracle/report pattern fits. |
| Claim-boundary clarity | 1 | High risk of duplicate or over-broad repeated-spectrum wording. |
| Total | 11 | Defer unless Day 3 finds a distinct repeated-spectrum gap worth closing. |

## Cross-Candidate Gap Table

| Gap | Rank-Def Rect | Sparse Low-Rank | Fail-Closed | External Ref | Repeated |
| --- | --- | --- | --- | --- | --- |
| New fixture metadata needed | Yes | Yes | Yes | Yes | Maybe |
| New generator rows needed | Yes | Yes | Yes | Maybe | Maybe |
| New expected TSV needed | Yes | Yes | Yes | Yes | Maybe |
| Existing focused proof owner can extend | Yes | Yes, with care | Yes | Maybe | Yes |
| Comparison contract already mostly available | Yes | Partial | Yes | Yes | Yes |
| New schema fields likely required | No | Probably no, but row encoding needs design | No | Maybe for provenance | No |
| Oracle/report generator changes needed | Yes | Yes | Yes | Yes if selected | Yes if selected |
| Documentation risk | Medium | Medium-high | Low-medium | High | High |
| Platform/package/performance overclaim risk | Low | High | Medium | High | Medium |

## Day 3 Selection Inputs

Recommended Day 3 posture:

1. Select `partial_svd_rankdef_diag6x4_k2_range_projector` or a nearby
   generated derivative as the highest-confidence rank-deficient rectangular
   family.
2. Select one sparse low-rank output fixture only if Day 3 can keep the claim
   to shape, selected values, `nnz` policy, and dense/sparse Frobenius
   consistency without storage-optimality or performance wording.
3. Consider the fail-closed diagonal fixture as a supporting family if it is
   framed as a distinct non-repeated convergence-budget lane rather than a
   duplicate of Sprint 140.
4. Defer external dense-reference vector residuals unless Sprint 151 decides
   to explicitly own optional-data provenance and hosted-platform skip policy.
5. Defer additional repeated-spectrum fixtures unless they add a distinct
   rank/truncation/subspace boundary that Sprint 140 did not already close.

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Raw singular-vector identity sneaks into expected rows. | Invalid repeated/degenerate subspace claim. | Use projector/subspace and residual rows; make raw-vector identity a stop condition. |
| Sparse low-rank rows imply performance or storage optimality. | Public claim overreach. | Keep sparse-output claims to fixture-local shape/value/Frobenius/`nnz` behavior and explicit non-claims. |
| Optional external reference rows are interpreted as broad parity. | External-library parity overclaim. | Defer or mark local/optional with explicit skip and provenance rules. |
| Fail-closed rows duplicate Sprint 140. | Sprint adds little net coverage. | Select only if the fixture proves a distinct non-repeated convergence-budget boundary. |
| Generated report rows become stale. | Report-index evidence drift. | Reuse Sprint 150 generated-output cleanup and freshness checks. |

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Candidate families are compared with concrete repository evidence. | Complete | Each candidate cites existing test/helper/corpus/report files and owner-local test names. |
| Each family has a closure/risk score. | Complete | Five candidate families are scored out of 20 with risk notes. |
| Family-selection inputs are ready for Day 3 without implementation bias. | Complete | Day 3 inputs recommend posture but defer final selection until claim-scope decisions are recorded. |
