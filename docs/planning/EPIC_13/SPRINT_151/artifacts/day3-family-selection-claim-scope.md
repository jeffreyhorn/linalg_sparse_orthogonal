# Sprint 151 Day 3: Family Selection And Claim Scope

## Purpose

Select the partial-SVD fixture families for Sprint 151 complete closure and
define claim scopes, non-claims, implementation owners, rollback rules, and
deferred candidates before comparison-contract and metadata work begins.

## Selection Decision

Sprint 151 selects three bounded partial-SVD families for complete closure:

| Selected Family | Planned Fixture Key | Current Owner-Local Seed | Selection Rationale |
| --- | --- | --- | --- |
| Rank-deficient rectangular range projector | `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | `test_partial_svd_rankdef_diag6x4_k2_range_projector` | Highest Day 2 score, directly closes rank-deficient rectangular projector evidence without raw singular-vector identity. |
| Sparse low-rank output | `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | `test_lowrank_rectangular_dense_sparse_consistency` | High user-facing API value and deterministic rectangular sparse-output semantics can be bounded to shape, selected values, and dense/sparse consistency. |
| Non-repeated convergence fail-closed | `partial_svd_fail_closed_diag6_k2_v1` | `test_partial_svd_max_iter_fail_closed_diag6_k2` | Adds a distinct non-repeated convergence-budget fixture so Sprint 151 does not only repeat the Sprint 140 clustered/repeated tight-budget lane. |

Sprint 151 explicitly defers:

- optional external dense-reference vector-residual fixtures;
- additional repeated-spectrum fixtures beyond the Sprint 140 clustered seed;
- broad sparse-output/drop-tolerance optimality;
- broad convergence-rate or portable iteration-count behavior.

## Family 1: Rank-Deficient Rectangular Range Projector

### Fixture Definition

Planned fixture key:

```text
partial_svd_rankdef_diag6x4_k2_range_projector_v1
```

Planned generator key:

```text
partial_svd_rankdef_diag6x4_k2_range_projector_generator_v1
```

Fixture shape and values:

- rows: `6`
- columns: `4`
- `k=2`
- diagonal values: `9, 6, 0, 0`
- expected rank: `2`
- expected nullity: `2`

### Claim Scope

This family may claim only fixture-local evidence for:

- partial-SVD default success for the named fixture;
- top-2 singular values `{9, 6}`;
- reported SVD rank `2` at the declared tolerance;
- left coordinate-range projector distance within tolerance;
- right coordinate-range projector distance within tolerance;
- triplet residuals `A v ~= sigma u` and `A^T u ~= sigma v`;
- U/V orthogonality within tolerance.

### Non-Claims

This family does not claim:

- broad rank-deficient partial-SVD correctness;
- raw singular-vector identity;
- sign, orientation, phase, or basis-order parity;
- broad null-space behavior;
- pseudoinverse or minimum-norm behavior;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art status.

### Implementation Map

| Surface | Planned Owner |
| --- | --- |
| fixture row | `tests/corpus/manifests/fixtures.tsv` |
| generator row | `tests/corpus/manifests/generators.tsv` |
| expected rows | `tests/corpus/expected/partial_svd_rankdef_diag6x4_k2_range_projector_v1.tsv` |
| deterministic generator | `scripts/validate_corpus_schema.py` |
| oracle/report rows | `scripts/run_corpus_oracle.py --include-partial-svd` |
| proof owner | `tests/test_svd_partial_corpus.c` |
| helper owner | `tests/test_svd_partial_shared_helpers.h` |
| docs | README, algorithm docs, cookbook, maintainer guide, corpus docs, oracle schema docs |

## Family 2: Sparse Low-Rank Output

### Fixture Definition

Planned fixture key:

```text
partial_svd_lowrank_rect5x7_k3_sparse_output_v1
```

Planned generator key:

```text
partial_svd_lowrank_rect5x7_k3_sparse_output_generator_v1
```

Fixture shape and values:

- rows: `5`
- columns: `7`
- `k=3`
- diagonal values: `8, 4, 2, 1, 0`
- sparse low-rank drop tolerance: `0`
- expected retained diagonal entries: `8, 4, 2`
- expected zeroed tail diagonal entry: `1`
- expected dense low-rank Frobenius error: `1.0`
- expected sparse-vs-dense low-rank Frobenius difference: `0.0`

### Claim Scope

This family may claim only fixture-local evidence for:

- sparse low-rank output status for the named fixture;
- output shape `5x7`;
- selected retained and zeroed coordinate values;
- dense low-rank Frobenius error for the named fixture;
- sparse low-rank output agreement with dense low-rank output for the named
  fixture at `drop_tol=0`.

### Non-Claims

This family does not claim:

- broad low-rank approximation optimality;
- broad sparse-output correctness;
- storage optimality;
- drop-tolerance optimality;
- sparse-output performance;
- broad rectangular low-rank behavior;
- external-library parity;
- platform, package, ABI, or state-of-the-art status.

### Implementation Map

| Surface | Planned Owner |
| --- | --- |
| fixture row | `tests/corpus/manifests/fixtures.tsv` |
| generator row | `tests/corpus/manifests/generators.tsv` |
| expected rows | `tests/corpus/expected/partial_svd_lowrank_rect5x7_k3_sparse_output_v1.tsv` |
| deterministic generator | `scripts/validate_corpus_schema.py` |
| oracle/report rows | `scripts/run_corpus_oracle.py --include-partial-svd` |
| proof owner | `tests/test_svd_partial_corpus.c` |
| helper owner | `tests/test_svd_partial_shared_helpers.h` or local focused helper additions |
| docs | README, algorithm docs, cookbook, maintainer guide, corpus docs, oracle schema docs |

## Family 3: Non-Repeated Convergence Fail-Closed

### Fixture Definition

Planned fixture key:

```text
partial_svd_fail_closed_diag6_k2_v1
```

Planned generator key:

```text
partial_svd_fail_closed_diag6_k2_generator_v1
```

Fixture shape and values:

- rows: `6`
- columns: `6`
- `k=2`
- diagonal values: `9, 6, 3, 1, 0.5, 0.25`
- tight budget: `max_iter=1`
- default budget: implementation default
- expected tight-budget status: `SPARSE_ERR_NOT_CONVERGED`
- expected tight-budget arrays: no `sigma`, `U`, or `Vt` published
- expected default top-2 singular values: `{9, 6}`

### Claim Scope

This family may claim only fixture-local evidence for:

- tight-budget non-convergence status on the named non-repeated fixture;
- no partial `sigma`, `U`, or `Vt` arrays are published on tight-budget
  failure;
- default-budget recovery succeeds after a failure attempt;
- default-budget top-2 singular values and triplet residuals match the
  declared tolerance.

### Non-Claims

This family does not claim:

- convergence rate;
- portable iteration counts;
- partial-result guarantees after non-convergence;
- broad convergence behavior;
- broad fail-closed behavior for arbitrary matrices;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art status.

### Implementation Map

| Surface | Planned Owner |
| --- | --- |
| fixture row | `tests/corpus/manifests/fixtures.tsv` |
| generator row | `tests/corpus/manifests/generators.tsv` |
| expected rows | `tests/corpus/expected/partial_svd_fail_closed_diag6_k2_v1.tsv` |
| deterministic generator | `scripts/validate_corpus_schema.py` |
| oracle/report rows | `scripts/run_corpus_oracle.py --include-partial-svd` |
| proof owner | `tests/test_svd_partial_corpus.c` |
| helper owner | `tests/test_svd_partial_shared_helpers.h` |
| docs | README, algorithm docs, cookbook, maintainer guide, corpus docs, oracle schema docs |

## Deferred Candidates

| Deferred Candidate | Reason |
| --- | --- |
| External dense-reference vector residuals | Optional helper, external provenance, Windows skip behavior, and broad parity overclaim risk make this a poor core Sprint 151 closure candidate. |
| Repeated spectra beyond Sprint 140 | Sprint 140 already closes the strongest repeated/clustered seed; another repeated fixture would likely duplicate the existing claim unless a later sprint names a distinct boundary. |
| Drop-tolerance sparse-output optimality | Too easy to overclaim storage/performance optimality; Sprint 151 keeps sparse-output evidence to `drop_tol=0` dense/sparse consistency. |
| Broad nonsymmetric rectangular external-reference behavior | Useful owner-local evidence, but not selected because optional data and external-reference wording would widen the sprint surface. |

## Expected Row Families

Day 4 should refine exact row names and comparison strings, but Day 3 selects
these expected row families:

| Fixture | Expected Row Families |
| --- | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | default status, singular values, rank, left projector, right projector, vector residuals, orthogonality |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | sparse-output status, output shape, selected values, dense low-rank Frobenius error, sparse-vs-dense Frobenius difference |
| `partial_svd_fail_closed_diag6_k2_v1` | tight-budget status, tight-budget no partial arrays, default status, default singular values, default vector residuals, recovery-after-failure status |

## Rollback Rules

Rollback the selected fixture or downgrade it to a deferred candidate if any
of these conditions occur:

- the fixture requires raw singular-vector equality, sign parity, orientation
  parity, phase parity, or arbitrary basis ordering;
- default success or residual tolerances are unstable under `make test`;
- sparse low-rank selected coordinate or Frobenius expectations depend on
  incidental insertion order rather than deterministic fixture semantics;
- `drop_tol=0` sparse-vs-dense comparison is not stable at `1e-10` or a
  documented looser tolerance;
- tight-budget failure publishes partial arrays or becomes platform/compiler
  dependent;
- generated fixture hashes drift without intentional generator-version updates;
- corpus schema, focused proof-owner tests, oracle/report checks, or required
  full C gates fail after implementation;
- documentation cannot state the claim without implying broad partial-SVD,
  external-library, platform, package, ABI, performance, or state-of-the-art
  support.

## Days 4-11 Implementation Plan

| Day | Selected-Family Owner Work |
| --- | --- |
| Day 4 | Define exact comparison contract for singular values, rank, projectors, residuals, sparse-output shape/value/Frobenius rows, and fail-closed status/diagnostic rows. |
| Day 5 | Design fixture, generator, expected, support-tier, claim-scope, and non-claim rows for the three selected families. |
| Day 6 | Add manifest rows, generator validation, and expected TSV skeletons. |
| Day 7 | Implement deterministic oracle inputs and expected observed-result encodings. |
| Day 8 | Design focused `tests/test_svd_partial_corpus.c` additions and helper cleanup. |
| Day 9 | Implement focused proof-owner tests and run required C gates if `.c` or `.h` files change. |
| Day 10 | Design normalized report rows and freshness expectations. |
| Day 11 | Implement partial-SVD oracle/report expansion and normalization checks. |

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Selected families can be fully closed within Sprint 151. | Complete | Three deterministic generated families selected; external-reference and extra repeated-spectrum work deferred. |
| Every claim has a matching proof owner and report/update owner. | Complete | Each selected family maps to corpus manifests, expected rows, generator validation, oracle/report generation, focused proof owner, helpers, and docs. |
| Unsupported partial-SVD claims are explicit before comparison-contract work starts. | Complete | Non-claims and rollback rules reject raw vectors, broad correctness, external parity, platform/package/ABI, performance, and state-of-the-art claims. |
