# Sprint 138 Day 2 - Fixture Taxonomy Draft

## Purpose

Day 2 drafts the maintained numerical-corpus taxonomy and selects the first
durable fixture lane for Sprint 138 implementation. The draft uses the Sprint
137 Day 8 corpus and oracle templates as the controlling vocabulary, then adds
only review-time candidate fields where the current templates do not yet expose
the project-plan class directly.

This is a design artifact. It does not create fixtures, generators, oracle
rows, reports, public claims, or solver behavior changes.

## Taxonomy Principles

| Principle | Decision |
| --- | --- |
| Fixture-local evidence | Each fixture row must describe what that fixture proves and what it does not prove. |
| Stable keys | Fixture families and fixture keys must remain durable enough for report indexes and future Sprint 139/140 handoffs. |
| Deterministic first | The first durable lane should use a deterministic generated fixture before optional external corpus data. |
| Existing templates first | Sprint 137 Day 8 field meanings control Day 2 terminology unless Day 3 explicitly promotes a candidate extension. |
| Skip is not pass | Optional-data skips, unsupported classes, and known residuals must remain separate from solver passes. |
| Support tier explicit | Corpus rows must carry support-tier meaning and must not imply broader package, platform, or release support. |
| Claims remain bounded | Corpus evidence may support fixture-local claims only until later sprints promote broader evidence. |

## Maintained Matrix-Class Taxonomy Draft

| Class axis | Allowed values or draft values | Required metadata | Solver and report relevance |
| --- | --- | --- | --- |
| Symmetry | `none`, `symmetric`, `structural_symmetric`, `hermitian_not_applicable` | `symmetry` | Distinguishes general QR/SVD inputs from Cholesky, LDLT, graph, and ordering fixtures. |
| Definiteness | `spd`, `semidefinite`, `indefinite`, `singular`, `rectangular`, `unknown` | `definiteness` | Controls direct-solver eligibility, expected diagnostic failures, and unsupported fixture rows. |
| Rank | `full_rank`, `rank_deficient`, `numerically_rank_deficient`, `unknown` | `rank_status`, `expected_rank`, `nullity` | Required for QR rank-deficiency evidence, SVD comparison semantics, least-squares behavior, and nullspace handoff. |
| Rectangularity | Draft derived values: `square`, `tall`, `wide` | `rows`, `cols`; candidate derived field `shape_class` | Needed by QR, least-squares, minimum-norm, and partial-SVD fixtures; can be derived from dimensions unless Day 3 promotes a field. |
| Conditioning | `well_conditioned`, `moderate`, `ill_conditioned`, `near_singular`, `not_applicable` | `conditioning_class` | Separates stable deterministic checks from tolerance-sensitive residuals and near-singular diagnostics. |
| Scaling | `unit`, `scaled`, `mixed_scale`, `not_applicable` | `scale_class` | Preserves tolerance context and prevents mixed-scale fixtures from silently widening accuracy claims. |
| Sparsity pattern | `diagonal`, `banded`, `block`, `graph_laplacian`, `random_sparse`, `structured_sparse`, `other` | `sparsity_class`, `nnz` | Provides enough structure for maintained generators, report grouping, and future graph/order fixture classification. |
| Graph shape | Draft values: `path`, `grid`, `star`, `tree`, `separator`, `block_graph`, `unknown` | `sparsity_class`; candidate field `graph_shape` | Useful for future graph, ordering, and separator corpus lanes; not required for the first QR lane. |
| RHS policy | `none`, `single_rhs`, `multi_rhs`, `generated_rhs`, `stored_rhs` | `rhs_policy` | Required for solve, least-squares, and example-consumer oracle rows; optional for decomposition-only rows. |
| Expected behavior | `success`, `diagnostic_failure`, `unsupported`, `non_convergence`, `skip` | `expected_behavior`; oracle `failure_class` when observed | Separates passing evidence from known residuals, unsupported cases, optional-data skips, and future failure investigations. |
| Data provenance | `inline`, `generated`, `matrix_market`, `optional_external` | `storage_kind`, `matrix_path`, `generator_key` as applicable | Keeps generated, stored, and optional external data paths explicit for reproducible reports. |

## Fixture Metadata Field Map

The first implementation lane should use these Sprint 137 Day 8 fields without
renaming them:

| Metadata group | Fields | Day 2 use |
| --- | --- | --- |
| Fixture identity | `fixture_key`, `fixture_family`, `introduced_in`, `owner` | Stable row identity and sprint ownership. |
| Storage and provenance | `storage_kind`, `matrix_path`, `generator_key` | Distinguishes generated fixtures from stored and optional external data. |
| Matrix dimensions | `rows`, `cols`, `nnz` | Provides rectangularity and sparsity evidence without a new required field. |
| Matrix class | `symmetry`, `definiteness`, `rank_status`, `expected_rank`, `nullity`, `conditioning_class`, `scale_class`, `sparsity_class` | Covers the project-plan classes for symmetry, definiteness, rank, conditioning, scaling, and sparsity. |
| Solver setup | `rhs_policy`, `expected_behavior` | Identifies whether the fixture supports solver, decomposition, skip, or failure evidence. |
| Claim boundary | `claim_scope`, `non_claims`, `support_tier` | Prevents fixture-local rows from widening public claims. |
| Validation | `validation_command` | Anchors row freshness and future report-index checks. |

Candidate review fields for Day 3:

| Candidate field | Reason | Day 2 disposition |
| --- | --- | --- |
| `shape_class` | Makes `square`, `tall`, and `wide` queries easier for QR, least-squares, and SVD reports. | Treat as derived from `rows` and `cols` unless Day 3 promotes it. |
| `graph_shape` | Helps future graph/order corpus lanes distinguish path, grid, tree, and separator families. | Leave out of the first lane unless Day 3 decides graph fixtures need first-class metadata. |
| `expected_failure_class` | Mirrors oracle `failure_class` for fixtures whose intended result is a diagnostic failure. | Keep failure class in oracle rows until a concrete fixture requires fixture-level prediction. |

## Solver Family Mapping

| Solver family or surface | Required class coverage | Sprint 138 relevance |
| --- | --- | --- |
| LU / CSR LU | Square general matrices, singular diagnostics, scaling, sparsity pattern. | Future corpus lane; not the first durable lane. |
| Cholesky / IC | Symmetric positive definite, semidefinite residuals, graph Laplacian structure. | Future corpus lane; taxonomy preserves `spd`, `semidefinite`, and `graph_laplacian`. |
| LDLT | Symmetric indefinite and singular structures. | Future corpus lane; taxonomy preserves `indefinite`, `singular`, and diagnostic failure classes. |
| QR | Rectangularity, rank deficiency, expected rank, nullity, RHS policy, conditioning. | Primary Sprint 139 handoff; first durable lane is selected here. |
| SVD / partial SVD | Rectangularity, rank, clustered or repeated spectra, conditioning, scaling, subspace semantics. | Sprint 140 handoff; taxonomy reserves fields but does not make first-lane SVD claims. |
| Iterative solvers | Conditioning, scaling, RHS policy, non-convergence, unsupported and skip states. | Future residual lane; taxonomy separates `non_convergence` from pass rows. |
| Eigensolvers | Symmetry, definiteness, graph shape, clustered eigenvalues, tolerance policy. | Future residual lane; shares conditioning and graph-class metadata. |
| Runtime/backend reports | Support tier, validation command, platform, compiler, configuration, skip/defer reasons. | Later report indexes consume the fixture/oracle row fields, but Sprint 138 does not prove platform parity. |

## First Durable Fixture Lane Selection

The first durable fixture lane should be a generated QR rank-deficient
rectangular fixture:

| Field | Draft value |
| --- | --- |
| Fixture family | `qr_rank_deficient` |
| Fixture key | `qr_rank_deficient_6x4_nullspace_v1` |
| Storage kind | `generated` |
| Rows / cols | `6` / `4` |
| Rank status | `rank_deficient` |
| Expected rank | `3` |
| Nullity | `1` |
| Symmetry | `none` |
| Definiteness | `rectangular` |
| Conditioning class | `moderate` |
| Scale class | `unit` |
| Sparsity class | `structured_sparse` |
| RHS policy | `generated_rhs` |
| Expected behavior | `success` |
| Support tier | Maintained deterministic fixture lane; fixture-local evidence only. |

This lane is narrow enough to close in Sprint 138 because it needs one
deterministic generator, one fixture metadata row, and one fixture-local oracle
path before broader corpus volume. It also gives Sprint 139 the rank/nullity
and rectangularity evidence needed for QR residual work.

## First-Lane Non-Claims

| Non-claim | Boundary |
| --- | --- |
| Broad QR correctness | The first lane covers one maintained rank-deficient generated fixture only. |
| Raw basis parity | Future QR evidence should compare projection/subspace behavior, not raw basis columns unless explicitly justified. |
| Minimum-norm or least-squares closure | The selected fixture may carry RHS metadata, but it does not close all solve modes. |
| SuiteSparse or external corpus parity | No optional external corpus row is enabled by the first lane. |
| Broad SVD correctness | Partial-SVD metadata is preserved for Sprint 140, but the first lane is not an SVD proof. |
| Corpus completeness | One generated lane is a durable seed, not a complete matrix zoo. |
| Public state-of-the-art claim | Fixture-local evidence does not promote competitive or state-of-the-art wording. |

## Out-of-Scope Residual Classes

| Residual class | Reason deferred |
| --- | --- |
| Optional SuiteSparse or external QR/SVD fixtures | Requires licensing, availability, skip/defer, and external-data policy before maintained use. |
| Partial-SVD clustered or repeated singular-value fixtures | Sprint 140 owns the solver-specific comparison semantics. |
| SPD and semidefinite direct-solver fixtures | Useful later, but not needed for the first QR lane. |
| Indefinite LDLT diagnostic fixtures | Requires dedicated expected-failure policy and solver-family review. |
| Random sparse generator families | Deterministic generated structure is preferable before probabilistic corpus volume. |
| Large performance or backend sentinel fixtures | Sprint 138 corpus rows are numerical evidence, not portable performance evidence. |
| Graph separator and ordering fixtures | Taxonomy reserves graph shape, but first-lane implementation remains QR-focused. |
| Matrix Market parser failure corpus | Parser diagnostics are a separate fixture family from numerical solver oracles. |
| Iterative non-convergence fixtures | Requires iteration-budget semantics and failure interpretation outside first-lane scope. |
| Platform/package/install fixtures | Package and platform confidence paths remain outside corpus taxonomy closure. |

## QR/SVD Dependency Notes

| Consumer | Dependency carried forward |
| --- | --- |
| Sprint 139 QR | First-lane metadata gives QR a stable generated rank-deficient rectangular fixture with expected rank and nullity. Sprint 139 should define projection, nullspace, and residual comparisons without relying on raw basis column equality. |
| Sprint 140 partial SVD | The taxonomy keeps rank, conditioning, scaling, rectangularity, and future clustered-spectrum classes visible, but Day 2 intentionally does not select a partial-SVD lane. |
| Sprint 141 report indexes | Fixture identity, support tier, validation command, expected behavior, and non-claim fields align with the Sprint 137 oracle/report templates. |

## Day 2 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Taxonomy covers the project-plan matrix classes. | Complete | Symmetry, definiteness, rank, rectangularity, conditioning, scaling, sparsity pattern, graph shape, RHS policy, and expected failures are represented. |
| First fixture lane is narrow enough to close in Sprint 138. | Complete | The selected lane is one deterministic generated QR rank-deficient 6x4 fixture with one expected rank/nullity contract. |
| Out-of-scope fixture classes are explicit residuals. | Complete | Residual table lists optional external data, SVD clustered fixtures, direct-solver fixtures, random generators, performance sentinels, graph/order fixtures, parser failures, iterative non-convergence, and package/platform fixtures. |
