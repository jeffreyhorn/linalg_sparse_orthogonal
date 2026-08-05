# Sprint 138 Day 3 - Taxonomy Review & Claim Boundaries

## Purpose

Day 3 reviews the Day 2 fixture taxonomy against the current repository,
finalizes the maintained corpus taxonomy for Sprint 138, defines promotion
gates, and records fixture-local claim boundaries before storage layout begins.

This remains a documentation-only review artifact. It does not implement
corpus paths, manifests, generators, oracle rows, validation commands, tests,
or public documentation updates.

## Current Repository Comparison

| Surface reviewed | Existing evidence | Day 3 finding |
| --- | --- | --- |
| `tests/test_qr.c` | Inline QR tests already cover square, tall, wide, rank-deficient, rank-1, nearly singular, diagonal, single-row, single-column, and external dense-reference helper keys. | Useful fixture candidates exist, but they are test-local fixtures without maintained corpus metadata, stable manifest rows, or oracle row semantics. |
| `tests/test_qr_helpers.h` | QR helper constructors provide reusable small generated matrices. | The first durable lane can reuse the same style later, but Sprint 138 should promote a corpus generator contract instead of treating helper code as the manifest. |
| `tests/test_svd.c` and `tests/test_svd_partial_helpers.h` | SVD and partial-SVD tests cover rank-deficient matrices, rectangular low-rank behavior, vector residuals, external dense references, SuiteSparse smoke checks, and fail-closed iteration budget behavior. | The taxonomy must preserve SVD handoff fields, but Day 3 does not promote an SVD lane because Sprint 140 owns repeated/clustered-spectrum closure. |
| `tests/qr_external_dense_reference.py` and `tests/svd_external_dense_reference.py` | Python helpers provide selected dense reference values for test-local fixture keys. | External reference helpers are useful precedent for oracle commands, but a maintained corpus row still needs command, commit, support tier, tolerance, and non-claim metadata. |
| `tests/data/*.mtx` | Small Matrix Market fixtures cover identity, diagonal, symmetric, tridiagonal, unsymmetric, pattern, and bad-header data. | These remain available future fixture classes; they are not selected as the first lane because generated QR rank/nullity evidence is narrower and easier to close. |
| `tests/data/suitesparse/*.mtx` | Bundled SuiteSparse-style files cover several larger real matrices used by existing tests. | These must not be treated as broad SuiteSparse parity. Optional or bundled external-style data needs explicit availability, support-tier, and claim-boundary rows before corpus promotion. |
| `examples/` | Examples expose QR least-squares/minimum-norm, SVD/low-rank, condition, Matrix Market, and SuiteSparse-flavored workflows. | Examples are adoption evidence, not corpus manifests. They can consume later corpus documentation after fixture-local claims are implemented. |
| Sprint 137 Day 8 templates | Define fixture, generated-matrix, optional-data, oracle, and failure interpretation fields. | The templates are sufficient for Sprint 138; Day 3 keeps their field names as the implementation contract. |
| Sprint 137 Day 9 report templates | Define report row identity, freshness, support tier, status, claim scope, and non-claims. | Corpus and oracle rows must remain compatible with Sprint 141 report normalization. |
| Sprint 137 Day 12 claim freeze | Blocks broad parity, state-of-the-art, package/platform, performance, coverage, and broad QR/SVD claims. | Day 3 preserves the freeze. First-lane corpus evidence remains fixture-local only. |

## Final Maintained Corpus Taxonomy

The Day 2 taxonomy is accepted with one constraint: Sprint 138 implementation
must use existing Sprint 137 Day 8 fields first. Draft classes that can be
derived from existing fields stay derived until a later artifact proves they
need first-class schema fields.

| Axis | Final Sprint 138 handling | Required field or derivation |
| --- | --- | --- |
| Symmetry | Accepted as manifest metadata. | `symmetry` |
| Definiteness | Accepted as manifest metadata. Rectangular matrices use `rectangular`. | `definiteness` |
| Rank | Accepted as manifest metadata when rank participates in the claim. | `rank_status`, `expected_rank`, `nullity` |
| Rectangularity | Accepted as a query class, derived from dimensions. | `rows`, `cols`; derive `square`, `tall`, or `wide` in validation/report logic when needed. |
| Conditioning | Accepted as manifest metadata. | `conditioning_class` |
| Scaling | Accepted as manifest metadata. | `scale_class` |
| Sparsity pattern | Accepted as manifest metadata. | `sparsity_class`, `nnz` |
| Graph shape | Deferred as first-class metadata. | Derive from fixture family or future graph fixture metadata only after a graph/order lane needs it. |
| RHS policy | Accepted as manifest metadata. | `rhs_policy` |
| Expected behavior | Accepted as manifest metadata. | `expected_behavior` |
| Data provenance | Accepted as manifest metadata. | `storage_kind`, `matrix_path`, `generator_key` |
| Failure class | Kept in oracle rows. | `failure_class` and `comparison_status` in oracle output rows. |

## Candidate Field Disposition

| Candidate field | Decision | Reason |
| --- | --- | --- |
| `shape_class` | Do not implement as a stored manifest field in Sprint 138. | `rows` and `cols` are required and can derive square/tall/wide without adding schema drift. |
| `graph_shape` | Do not implement for the first lane. | The selected first lane is QR-focused; graph/order fixtures are residuals and can promote graph-specific fields later. |
| `expected_failure_class` | Do not implement as a fixture field in Sprint 138. | Failure interpretation belongs to oracle rows until a concrete expected-failure fixture needs fixture-level classification. |

## Selected First Durable Lane

Day 3 confirms the Day 2 selected first lane:

| Field | Final Sprint 138 value |
| --- | --- |
| Fixture family | `qr_rank_deficient` |
| Fixture key | `qr_rank_deficient_6x4_nullspace_v1` |
| Generator key | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| Storage kind | `generated` |
| Shape | Tall rectangular, derived from `rows=6`, `cols=4` |
| Expected rank | `3` |
| Nullity | `1` |
| Symmetry | `none` |
| Definiteness | `rectangular` |
| Rank status | `rank_deficient` |
| Conditioning class | `moderate` |
| Scale class | `unit` |
| Sparsity class | `structured_sparse` |
| RHS policy | `generated_rhs` |
| Expected behavior | `success` |
| Support tier before hosted proof | `local_only` until validation evidence promotes it. |
| Claim scope | Fixture-local generated reference rank/nullity and normalized null-vector residual metadata. |
| Non-claims | No raw-basis parity; no broad QR correctness; no global minimum-norm guarantee; no SuiteSparse parity; no broad corpus completeness; no SVD correctness claim. |

This lane supports Sprint 139 because it gives the QR owner stable
rank/nullity metadata and a deterministic rectangular generated fixture for
projector or two-way projection comparison. It does not broaden to external
corpus parity because it uses no optional external data and carries explicit
non-claims.

## Fixture-Class Promotion Gates

| Gate | Requirement | Blocks |
| --- | --- | --- |
| Taxonomy fit | Fixture row must map to accepted Sprint 138 axes without redefining field meanings. | Ad hoc fixture classes and schema drift. |
| Stable identity | Fixture key and family must be stable and referenced by tests, oracle rows, reports, and docs. | Anonymous inline tests being counted as corpus coverage. |
| Reproducibility | Generated fixtures must carry generator key, version, parameters, expected structure/value hashes, regeneration command, and change policy. | Non-reproducible generated rows. |
| Oracle semantics | Oracle row must include expected result, observed result, tolerance, command, source commit, support tier, comparison status, claim scope, and non-claims. | Pass/fail rows without interpretable evidence. |
| Support-tier evidence | Support tier must match the command/platform actually run. Local rows remain `local_only` unless reviewed platform evidence exists. | Unsupported platform promotion. |
| Skip/defer handling | Optional data and intentionally deferred rows must be `skip` or `defer`, never silent pass. | External-data and residual classes being counted as solver successes. |
| Claim boundary | Fixture must list what it may prove and what remains blocked. | Broad correctness, parity, coverage, performance, package, platform, or state-of-the-art claims. |
| Validation path | A maintained command must exercise the fixture or report the current skip/defer state. | Corpus rows with no reproducible check. |
| Documentation handoff | Maintainer docs must describe fixture-local meaning before adoption docs use the row. | Public wording that outpaces evidence. |

## Fixture-Local Claim Boundary Table

| Class or lane | Allowed claim scope | Required non-claims |
| --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1` | One deterministic generated QR fixture has expected rank/nullity and nullspace/subspace residual behavior under its oracle tolerance and support tier. | No raw-basis parity; no broad QR correctness; no minimum-norm closure; no SuiteSparse parity; no broad corpus completeness; no SVD correctness claim. |
| Future `svd_clustered` lane | One selected SVD fixture may support singular-value/subspace comparison and convergence-budget behavior after Sprint 140 implements it. | No broad SVD correctness; no external solver parity; no broad repeated/clustered spectrum coverage; no performance claim. |
| Future direct-solver SPD/indefinite lanes | One selected direct-solver fixture may support fixture-local success or diagnostic behavior. | No broad Cholesky/LDLT/LU correctness; no package/platform or state-of-the-art claim. |
| Future optional external-data rows | Available optional data may support only the configured fixture-local check; unavailable data supports skip-policy evidence only. | No SuiteSparse parity; no external-library parity; no solver pass when skipped; no broad corpus coverage. |
| Future graph/order rows | One graph/order fixture may support fixture-local ordering, separator, or structural behavior. | No general graph partitioning quality, fill-reduction superiority, or external parity claim. |
| Future report-index rows | A normalized row may support freshness and row interpretation for its subject. | No release proof; no broad correctness proof; no coverage completeness; no portable performance proof. |

## QR, Partial-SVD, and Report Handoff Notes

| Consumer | Day 3 handoff |
| --- | --- |
| Sprint 139 QR | Use `qr_rank_deficient_6x4_nullspace_v1` as the corpus-backed QR seed. Compare rank, nullity, residuals, and subspace/projector behavior. Avoid raw basis equality as the primary oracle. |
| Sprint 140 partial SVD | Reuse the accepted taxonomy axes and oracle row semantics, especially rank, rectangularity, conditioning, scale, tolerance, convergence-budget status, and subspace comparison. Select clustered/repeated-spectrum fixtures in Sprint 140, not Sprint 138. |
| Sprint 141 report indexes | Treat corpus fixture rows and oracle rows as eligible report families only when they include source commit, command, support tier, status, claim scope, non-claims, and freshness-compatible metadata. |

## Final Residual List

| Residual | Owner sprint | Promotion prerequisite |
| --- | --- | --- |
| Partial-SVD repeated/clustered-spectrum fixture family | Sprint 140 | Solver-specific fixture choice, singular-value/subspace oracle, convergence-budget semantics, and non-claims. |
| Optional external/SuiteSparse corpus rows | Later corpus owner | Availability policy, license/terms review, skip/defer rows, validation command, and no-parity claim boundary. |
| Direct-solver SPD/semidefinite/indefinite/singular corpus lanes | Later solver/corpus owner | Fixture family selection, expected success or diagnostic semantics, and solver-specific oracle tolerance. |
| Graph/order fixture taxonomy expansion | Later graph/order owner | Decision on whether `graph_shape` becomes stored metadata or remains fixture-family derived. |
| Random/probabilistic generated fixtures | Later corpus owner | Seed, generator version, stability hashes, tolerance policy, and deterministic replay proof. |
| Performance/backend sentinel corpus interaction | Sprint 142 | Runtime/backend precedence, sentinel row semantics, and portable-performance non-claims. |
| Report normalization and freshness automation | Sprint 141 | Concrete Sprint 138 corpus/oracle paths and row fields. |
| Public adoption wording based on corpus evidence | Sprint 145 | Implemented corpus rows, solver handoffs, report normalization, and claim-boundary validation. |

## Day 3 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected taxonomy does not imply broad corpus completeness. | Complete | First-lane non-claims, promotion gates, and residual list keep the selected lane to one deterministic QR fixture. |
| QR, partial-SVD, and report dependencies remain supported. | Complete | Handoff table preserves QR rank/nullity/subspace, SVD rank/conditioning/subspace, and report freshness fields. |
| Claim boundaries are written before storage layout begins. | Complete | Fixture-local claim boundary table and promotion gates are recorded before Day 4 path design. |
