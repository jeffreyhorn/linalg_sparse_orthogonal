# Sprint 131 Day 4 - Corpus Taxonomy Policy

## Purpose

Day 4 defines corpus tags for structure, numerical properties, ownership,
availability, support tier, oracle output, failure interpretation, and
claim-boundary status. The taxonomy is meant to drive later report indexes
without changing test semantics or promoting fixtures silently.

This is a policy artifact only. No code, helper, benchmark, coverage,
dead-code, report-generation, public API, or public wording changes are made
by Day 4.

## Tagging Principles

- Tags describe evidence; they do not create evidence.
- Every reviewed corpus row needs a fixture key, source path or construction
  rule, owner, evidence class, validation command, and non-claim boundary.
- Generated fixtures are tagged as local analytic or generated stress inputs
  until an artifact records independent corpus status.
- Checked-in SuiteSparse-derived files are corpus inputs, but default to smoke
  or report context unless independent oracle metadata exists.
- Expected failures, skips, and unsupported behavior are first-class tags so
  report indexes do not count them as positive evidence.
- Benchmark, coverage, dead-code, sentinel, and guardrail artifacts require
  report/evidence tags separate from numerical correctness tags.

## Structural Tags

| Tag family | Allowed values | Required when | Promotion or deferral semantics |
| --- | --- | --- | --- |
| `shape` | `square`, `tall`, `wide`, `rectangular`, `vector`, `empty`, `malformed` | Every matrix-like source | `malformed` sources may only support parser-negative evidence. |
| `storage_format` | `matrix-market-coordinate`, `generated-sparse`, `generated-dense`, `csr`, `csc`, `graph`, `dense-helper`, `unknown` | Every source | `unknown` blocks reviewed promotion. |
| `market_field` | `real`, `integer`, `pattern`, `invalid`, `not-applicable` | Matrix Market files | `pattern` supports structure only unless numeric values are defined elsewhere. |
| `market_symmetry` | `general`, `symmetric`, `skew-symmetric`, `hermitian`, `invalid`, `not-applicable` | Matrix Market files | Symmetric files must state stored versus expanded nonzero semantics before indexing. |
| `definiteness` | `spd`, `symmetric-indefinite`, `nonsymmetric`, `not-spd`, `unknown`, `not-applicable` | Solver and corpus fixtures | `unknown` blocks solver-specific reviewed claims that depend on definiteness. |
| `rank_model` | `full-rank`, `exact-rank-deficient`, `threshold-rank`, `singular`, `unknown`, `not-applicable` | Rank, solve, SVD, QR, eigensolver, and low-rank evidence | `unknown` blocks rank/nullity, minimum-norm, and subspace promotion. |
| `graph_pattern` | `path-1d`, `grid-2d`, `mesh-3d`, `complete`, `bipartite`, `cliques-bridge`, `corpus-graph`, `not-graph` | Graph/reorder sources | Graph tags must not imply numerical solve coverage. |
| `ordering` | `natural`, `amd`, `colamd`, `nd`, `qg-amd`, `postordered`, `multiple`, `not-applicable` | Reorder, factorization, benchmark, and guardrail evidence | Ordering-specific evidence is not transferable to all ordering modes. |

## Numerical Tags

| Tag family | Allowed values | Required when | Promotion or deferral semantics |
| --- | --- | --- | --- |
| `scale` | `unit`, `small`, `large-entry`, `scaled`, `ill-scaled`, `unknown` | Numerical solve, residual, factorization, SVD, QR, and eigensolver evidence | `unknown` requires fixture-local tolerance rationale. |
| `conditioning` | `well-conditioned`, `near-singular`, `ill-conditioned`, `threshold-sensitive`, `unknown` | Residual, rank, SVD, QR, and iterative evidence | `unknown` blocks broad tolerance reuse. |
| `spectrum_shape` | `separated`, `repeated`, `clustered`, `zero-tail`, `unknown`, `not-applicable` | SVD/eigensolver/condition/rank evidence | `repeated` or `clustered` requires projector/subspace policy before vector or basis claims. |
| `nullity` | `zero`, `positive`, `threshold-dependent`, `unknown`, `not-applicable` | QR/SVD rank, nullspace, minnorm, and singularity evidence | `unknown` blocks nullspace/minimum-norm promotion. |
| `density` | `diagonal`, `banded`, `sparse`, `medium-sparse`, `dense`, `pattern-only`, `unknown` | Every matrix-like source | `unknown` blocks scale/report indexing. |
| `known_solution` | `exact-rhs`, `dense-reference`, `analytic-residual`, `product-observed`, `none`, `unknown` | Solve and residual evidence | `product-observed` is smoke or internal consistency, not independent oracle evidence. |
| `tolerance_policy` | `exact-absolute`, `relative-residual`, `fixture-specific`, `threshold-specific`, `report-only`, `none`, `unknown` | Any pass/fail numerical comparison | `unknown` blocks reviewed promotion. |

## Evidence And Oracle Tags

| Tag family | Allowed values | Required when | Promotion or deferral semantics |
| --- | --- | --- | --- |
| `evidence_class` | `parser-negative`, `load-structure`, `solve-vector`, `residual`, `rank`, `nullspace`, `projector`, `singular-values`, `vector-residual`, `lowrank-reconstruction`, `lowrank-optimality`, `convergence-budget`, `benchmark-report`, `coverage-report`, `deadcode-report`, `guardrail-report`, `documentation-policy` | Every indexed artifact row | Report-only classes must not count as numerical correctness evidence. |
| `oracle_source` | `analytic`, `external-helper`, `published-metadata`, `cross-solver`, `product-observed`, `none`, `unknown` | Every reviewed evidence row | `product-observed`, `none`, and `unknown` cannot support independent oracle claims. |
| `oracle_output` | `solve-vector`, `singular-values`, `rank-scalar`, `threshold-rank-triples`, `residual-norm`, `projector-values`, `expected-failure`, `report-index-row`, `not-applicable` | External-reference and report rows | Output class must match assertion class before promotion. |
| `failure_class` | `expected-parse-error`, `expected-singular`, `expected-not-spd`, `expected-shape-error`, `expected-badarg`, `expected-nonconvergence`, `helper-skip`, `helper-error`, `optional-data-skip`, `env-setup-skip`, `slow-optin-skip`, `numeric-mismatch`, `report-freshness-mismatch`, `none` | Every skip, expected failure, or external-reference lane | Expected failures and skips are not positive evidence unless their evidence class says so. |

## Ownership Tags

| Tag family | Allowed values | Required when | Promotion or deferral semantics |
| --- | --- | --- | --- |
| `solver_family` | `sparse-io`, `direct-lu`, `lu-csr`, `cholesky`, `chol-csc`, `ldlt`, `qr`, `qr-solve`, `svd`, `partial-svd`, `eigs`, `iterative`, `graph`, `reorder`, `dense`, `integration`, `benchmark`, `tooling`, `docs` | Every fixture and report row | Multi-family rows need explicit primary owner. |
| `fixture_owner` | file path or `unknown` | Every fixture row | `unknown` blocks reviewed promotion. |
| `oracle_owner` | helper path, artifact path, citation, `product`, `none`, or `unknown` | Every evidence row | `product` must be marked internal/smoke unless paired with independent source. |
| `validation_owner` | command, target, or `unknown` | Every reviewed or supplemental row | `unknown` blocks report-index generation. |
| `report_owner` | report script/path, artifact path, `none`, or `unknown` | Reportable rows | `unknown` blocks generated index inclusion except as a gap row. |
| `docs_owner` | docs path or `none` | Rows that affect wording | Public/maintainer wording changes require docs owner and non-claim scan. |

## Availability Tags

| Tag | Meaning | Default support tier | Promotion or deferral semantics |
| --- | --- | --- | --- |
| `local-analytic` | Generated or checked-in fixture with construction rule and exact facts. | Candidate reviewed fixture | Promotable when owner, oracle, tolerance, validation, and non-claims are complete. |
| `checked-in-parser` | Checked-in parser or structural fixture. | Reviewed parser/structure or unsupported numerical | Numerical claims require separate numeric metadata. |
| `checked-in-smoke` | Checked-in corpus file used for bounded smoke/regression. | Smoke | Requires independent oracle metadata before reviewed corpus evidence. |
| `checked-in-reviewed` | Checked-in fixture with accepted bounded evidence package. | Reviewed | Must preserve exact evidence class and fixture key. |
| `checked-in-expensive` | Checked-in data whose runtime or memory cost is too high for default unit evidence. | Supplemental or benchmark | Default reviewed promotion requires runtime budget and skip policy. |
| `optional-local` | Locally available but absent from default repo/CI. | Supplemental or experimental | Missing data must skip with source/version/checksum policy. |
| `optional-external` | Downloaded or external corpus data. | Supplemental or deferred | Reviewed promotion requires availability, checksum/version, source, oracle, runtime, and skip policy. |

## Support-Tier Tags

| Tag | Meaning | Allowed claims |
| --- | --- | --- |
| `unsupported` | Expected failure, parser negative, invalid API use, or unsupported mode. | Error-path behavior only. |
| `smoke` | Product regression or corpus load/run check with limited metadata. | Fixture loaded or local behavior stayed within owner-specific smoke metric. |
| `reviewed` | Evidence has owner, metadata, oracle/trust boundary, tolerance, validation, and non-claims. | Exact bounded claim named by the artifact. |
| `supplemental` | Useful report or optional context outside mandatory reviewed gate. | Context only; no default support guarantee. |
| `experimental` | Opt-in or exploratory surface. | Experimental behavior only. |
| `benchmark` | Timing/measurement/report surface. | Local measurement or schema/report evidence, not correctness or portable performance. |
| `deferred` | Candidate lacks required metadata, oracle, runtime, or owner. | No claim; carries blocker and future-owner criteria. |

## Reviewed Promotion Checklist

A row may move to `support_tier=reviewed` only when all required fields below
are explicit:

1. Stable fixture/report key and source path or construction rule.
2. Structural, numerical, availability, support-tier, and evidence-class tags.
3. Primary fixture owner, oracle owner, validation owner, and docs owner when
   wording changes.
4. Oracle source and output class, or explicit reason evidence is analytic.
5. Tolerance, threshold, runtime, skip, and failure interpretation.
6. Validation command and expected artifact freshness rule.
7. Non-claim boundary that prevents broader solver, corpus, benchmark,
   coverage, platform, or public wording overclaim.

If any required field is missing, the row stays `deferred`, `smoke`,
`supplemental`, or `unsupported` with blocker text.

## Demotion Rules

| Trigger | Required demotion |
| --- | --- |
| Fixture loses independent oracle metadata. | `reviewed` to `smoke` or `deferred`. |
| Runtime exceeds default unit budget without opt-in policy. | `reviewed` to `supplemental`, `benchmark`, or `deferred`. |
| Missing data is treated as pass without a required/optional distinction. | `reviewed` to `deferred`. |
| Output class and assertion class do not match. | `reviewed` to `deferred`. |
| Report artifact has no freshness rule or owner. | `reviewed`/`supplemental` report row to `deferred`. |
| Public wording exceeds bounded evidence. | Wording change blocked; evidence row remains bounded or is demoted if needed. |

## Tag Families For Report Indexes

Generated indexes should treat these fields as minimum row schema:

| Field | Required? | Notes |
| --- | --- | --- |
| `key` | yes | Stable fixture or report key. |
| `source` | yes | Path, helper key, script output, or construction artifact. |
| `evidence_class` | yes | Must distinguish correctness, expected failure, report, and policy rows. |
| `support_tier` | yes | Must be one of the support-tier tags above. |
| `availability` | yes | Must distinguish checked-in, optional, expensive, and analytic sources. |
| `solver_family` | yes | Primary owner family. |
| `fixture_owner` | yes | File path or explicit `unknown`. |
| `oracle_source` | yes | `unknown` means not promotable. |
| `oracle_output` | yes for external/helper rows | Prevents value helpers from implying vector/projector claims. |
| `validation_owner` | yes | Command or target. |
| `failure_class` | yes for skips/errors | Expected failures and skips remain visible. |
| `claim_boundary` | yes | Short text for what the row does and does not prove. |

## Non-Claims Preserved

This taxonomy does not claim:

- broad Matrix Market or SuiteSparse corpus coverage;
- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or ecosystem parity;
- raw Q, U, V, eigenvector, singular-vector, sign, orientation, or basis
  stability when only projector/residual/value evidence exists;
- portable performance, scalability, memory, coverage, or dead-code proof;
- public solver-selection readiness beyond existing bounded maintainer
  evidence;
- package, ABI, CMake, CI, install-header, or platform expansion.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Tags are specific enough to drive report indexes. | Complete | Tag families include structural, numerical, evidence/oracle, ownership, availability, support-tier, failure, and minimum index row fields. |
| Support tiers do not imply unsupported public claims. | Complete | Support-tier table and non-claims separate smoke, reviewed, supplemental, experimental, benchmark, deferred, and unsupported evidence. |
| Every tag family has promotion and deferral semantics. | Complete | Each tag table states promotion or deferral semantics, with reviewed promotion checklist and demotion rules. |

## Day 5 Handoff

Day 5 should apply this taxonomy manually to representative fixtures across
direct solvers, iterative solvers, QR, SVD, partial SVD, eigensolvers, graph
partitioning, and integration tests. Any ambiguous tag should become a blocker
or refinement before Sprint 131 generates an index.

