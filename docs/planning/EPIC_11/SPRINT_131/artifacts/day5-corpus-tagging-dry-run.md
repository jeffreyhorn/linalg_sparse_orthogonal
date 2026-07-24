# Sprint 131 Day 5 - Corpus Tagging Dry Run

## Purpose

Day 5 applies the Day 4 taxonomy to representative fixtures across solver,
oracle, graph, integration, and report-adjacent surfaces. The dry run checks
whether representative rows can be tagged without changing test semantics and
records blockers before Sprint 131 designs generated indexes.

This is a documentation-only dry run. It does not change fixtures, tests,
helpers, scripts, benchmark semantics, public wording, or support tiers.

## Representative Tagged Fixtures

| Key | Source | Solver family | Structural tags | Numerical tags | Evidence/oracle tags | Availability/support | Owner and validation | Claim boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bad_header` | `tests/data/bad_header.mtx` | `sparse-io` | `shape=malformed`, `storage_format=matrix-market-coordinate`, `market_field=invalid`, `market_symmetry=invalid` | `density=unknown`, `known_solution=none`, `tolerance_policy=none` | `evidence_class=parser-negative`, `oracle_source=none`, `oracle_output=expected-failure`, `failure_class=expected-parse-error` | `availability=checked-in-parser`, `support_tier=unsupported` | owner `tests/test_sparse_io.c`; validation `make build/test_sparse_io && ./build/test_sparse_io` if behavior changes | Parser rejection only; no numerical corpus evidence. |
| `identity_5` | `tests/data/identity_5.mtx` | `direct-lu` | `shape=square`, `storage_format=matrix-market-coordinate`, `market_field=real`, `market_symmetry=general`, `definiteness=spd`, `rank_model=full-rank` | `scale=unit`, `conditioning=well-conditioned`, `density=diagonal`, `known_solution=exact-rhs`, `tolerance_policy=exact-absolute` | `evidence_class=solve-vector`, `oracle_source=analytic`, `oracle_output=solve-vector`, `failure_class=none` | `availability=local-analytic`, `support_tier=reviewed` for current bounded IO/LU rows only | owners `tests/test_sparse_io.c`, `tests/test_known_matrices.c`; validation focused owner tests | Exact small identity fixture only; no broad Matrix Market or LU parity. |
| `tridiagonal_20` | `tests/data/tridiagonal_20.mtx` | `direct-lu` | `shape=square`, `storage_format=matrix-market-coordinate`, `market_field=real`, `market_symmetry=general`, `definiteness=spd`, `rank_model=full-rank` | `scale=small`, `conditioning=well-conditioned`, `density=banded`, `known_solution=exact-rhs`, `tolerance_policy=relative-residual` | `evidence_class=residual`, `oracle_source=analytic`, `oracle_output=solve-vector`, `failure_class=none` | `availability=local-analytic`, `support_tier=reviewed` for current LU/refinement residual rows only | owner `tests/test_known_matrices.c`; validation focused known-matrix tests | Fixture-local tridiagonal solve/refinement residual evidence only. |
| `nos4-chol-external` | `tests/data/suitesparse/nos4.mtx` plus `tests/chol_external_dense_reference.py` | `chol-csc` | `shape=square`, `storage_format=matrix-market-coordinate`, `market_field=real`, `market_symmetry=symmetric`, `definiteness=spd`, `rank_model=full-rank` | `scale=unknown`, `conditioning=unknown`, `density=sparse`, `known_solution=dense-reference`, `tolerance_policy=fixture-specific` | `evidence_class=solve-vector`, `oracle_source=external-helper`, `oracle_output=solve-vector`, `failure_class=helper-skip` when helper/platform unavailable | `availability=checked-in-reviewed`, `support_tier=reviewed` for bounded Cholesky CSC solve only | owner `tests/test_chol_csc.c`; validation `make build/test_chol_csc && ./build/test_chol_csc` if touched | One checked-in SuiteSparse-derived Cholesky CSC solve fixture; no broad SuiteSparse or dense-library parity. |
| `lu_singular_square_4` | `tests/lu_external_dense_reference.py` fixture key | `direct-lu` | `shape=square`, `storage_format=generated-dense`, `definiteness=nonsymmetric`, `rank_model=singular` | `scale=small`, `conditioning=near-singular`, `density=dense`, `known_solution=none`, `tolerance_policy=none` | `evidence_class=parser-negative` is not right; use `evidence_class=solve-vector` only for negative solve contract, `oracle_source=external-helper`, `oracle_output=expected-failure`, `failure_class=expected-singular` | `availability=local-analytic`, `support_tier=unsupported` | owner `tests/test_sparse_lu.c`; validation focused sparse LU tests | Expected singular failure; not a successful solve row. |
| `qr_rankdef_duplicate_5x4_nullspace_projector` | `tests/qr_external_dense_reference.py` fixture key | `qr` | `shape=tall`, `storage_format=generated-dense`, `definiteness=nonsymmetric`, `rank_model=exact-rank-deficient` | `scale=small`, `conditioning=threshold-sensitive`, `nullity=positive`, `density=dense`, `known_solution=dense-reference`, `tolerance_policy=fixture-specific` | `evidence_class=projector`, `oracle_source=external-helper`, `oracle_output=projector-values`, `failure_class=helper-skip` when helper/platform unavailable | `availability=checked-in-reviewed`, `support_tier=reviewed` for bounded projector lane only | owner `tests/test_qr.c`; validation focused QR tests | Nullspace projector evidence only; no raw Q-basis sign/orientation claim. |
| `partial_svd_nonsym_rect10x8_k3` | `tests/svd_external_dense_reference.py` and `tests/test_svd_partial_helpers.h` | `partial-svd` | `shape=tall`, `storage_format=generated-dense`, `definiteness=nonsymmetric`, `rank_model=full-rank` | `scale=small`, `conditioning=unknown`, `spectrum_shape=unknown`, `density=dense`, `known_solution=dense-reference`, `tolerance_policy=fixture-specific` | `evidence_class=singular-values`; vector-residual rows require separate `evidence_class=vector-residual`, `oracle_source=external-helper`, `oracle_output=singular-values`, `failure_class=helper-skip` when helper/platform unavailable | `availability=local-analytic`, `support_tier=reviewed` only for bounded named partial-SVD lane | owner `tests/test_svd_partial_helpers.h`; validation focused SVD tests | Singular-value helper is value-only; vector residual checks do not prove raw singular-vector parity. |
| `bcsstk04-lobpcg-preconditioned` | `tests/data/suitesparse/bcsstk04.mtx` | `eigs` | `shape=square`, `storage_format=matrix-market-coordinate`, `market_field=real`, `market_symmetry=symmetric`, `definiteness=spd`, `rank_model=full-rank` | `scale=unknown`, `conditioning=unknown`, `spectrum_shape=clustered`, `density=sparse`, `known_solution=product-observed`, `tolerance_policy=fixture-specific` | `evidence_class=residual`, `oracle_source=product-observed`, `oracle_output=not-applicable`, `failure_class=optional-data-skip` if missing | `availability=checked-in-smoke`, `support_tier=smoke` unless independent eigen metadata is added | owner `tests/test_eigs_lobpcg.c`; validation focused eigensolver tests if touched | Local Ritz residual/preconditioner comparison only; no ARPACK/SciPy/eigensolver parity. |
| `Pres_Poisson-guardrail` | `tests/data/suitesparse/Pres_Poisson.mtx` through `scripts/large_matrix_guardrails.sh` | `reorder` | `shape=square`, `storage_format=matrix-market-coordinate`, `market_field=real`, `market_symmetry=symmetric`, `graph_pattern=corpus-graph`, `ordering=multiple` | `scale=unknown`, `conditioning=unknown`, `density=sparse`, `known_solution=none`, `tolerance_policy=report-only` | `evidence_class=guardrail-report`, `oracle_source=none`, `oracle_output=report-index-row`, `failure_class=report-freshness-mismatch` when report is stale | `availability=checked-in-expensive`, `support_tier=supplemental` or reviewed structural guardrail only under existing guardrail policy | owner `scripts/large_matrix_guardrails.sh`; validation `make large-matrix-guardrails` if script/report changes | Structural/report guardrail only; not broad scalability, memory, or performance proof. |
| `integration-kkt` | `integration_build_kkt` in `tests/test_integration_fixtures.h` | `integration` | `shape=square`, `storage_format=generated-sparse`, `definiteness=symmetric-indefinite`, `rank_model=full-rank` when construction dimensions satisfy nonsingularity assumptions | `scale=small`, `conditioning=unknown`, `density=sparse`, `known_solution=exact-rhs` when generated RHS is used, `tolerance_policy=fixture-specific` | `evidence_class=solve-vector` or lifecycle/residual depending on owner test, `oracle_source=analytic`, `oracle_output=solve-vector`, `failure_class=none` | `availability=local-analytic`, `support_tier=smoke` until row-specific validation owner and claim boundary are recorded | owner `tests/test_integration.c`; validation focused integration tests if touched | Integration lifecycle fixture; not standalone LDLT/direct-solver corpus evidence unless promoted separately. |

## Taxonomy Refinements From Dry Run

| Refinement | Reason | Day 5 decision |
| --- | --- | --- |
| Add or allow `evidence_class=expected-error` in a future revision. | `lu_singular_square_4` is not naturally a parser-negative row and should not look like a positive solve-vector row. | For Day 5, keep `failure_class=expected-singular` as the controlling tag and mark support as `unsupported`; Day 6+ index schema should include expected-error rows explicitly. |
| Distinguish `checked-in-reviewed` from fixture-wide reviewed status. | `nos4` can be reviewed for one Cholesky CSC external solve but only smoke for many other owners. | Require claim boundary and owner-specific evidence class on every row. |
| Require `oracle_output` even when `oracle_source=product-observed`. | Product-observed eigensolver/SVD smoke can otherwise look like independent dense-reference evidence. | Index row schema should record `oracle_source=product-observed` plus `oracle_output=not-applicable` or exact metric. |
| Split report rows from fixture rows. | `Pres_Poisson` is both a checked-in matrix and a large-matrix guardrail input. | Day 6 should design report indexes with report key, report owner, generation command, freshness, and input corpus fields separate from fixture metadata. |
| Record stored versus expanded nonzero counts. | Symmetric Matrix Market files have stored-entry counts that differ from expanded logical entries. | Minimum schema keeps both `stored_entries` and optional `expanded_nnz`. |

## Ambiguity And Missing-Metadata Register

| Ambiguity or gap | Affected rows | Blocker | Future owner |
| --- | --- | --- | --- |
| Expected-error evidence class is not a named Day 4 value. | `lu_singular_square_4`, non-SPD Cholesky, parser and shape/API negative tests. | Need index schema value for expected error-path evidence distinct from parser-negative and successful solve rows. | Day 6 report-index requirements and Day 7 design. |
| Fixture-level support tier can be confused with owner-specific support tier. | `nos4`, `bcsstk04`, `west0067`, `Pres_Poisson`. | Same matrix has reviewed, smoke, benchmark, and supplemental interpretations depending on owner. | Day 6 and Day 12 ownership map. |
| Conditioning metadata is missing for most SuiteSparse-derived rows. | `nos4`, `bcsstk04`, `Pres_Poisson`, larger SuiteSparse fixtures. | Cannot reuse tolerances broadly or promote corpus parity. | Future corpus metadata owner after Sprint 131. |
| Product-observed oracle rows are easy to overcount. | Eigensolver/SVD/partial-SVD SuiteSparse smoke and low-rank mode-equivalence checks. | Need index display that marks product-observed as internal/smoke. | Day 6 requirements and Day 8 coverage architecture. |
| Report freshness is not a fixture property. | Large-matrix guardrails, benchmark reports, coverage, dead-code reports. | Generated index rows need report timestamps/commands separate from matrix tags. | Day 6-7 report-index design. |
| Integration fixtures are multi-owner by construction. | `integration_build_tridiag_spd`, `integration_build_unsym_4x4`, `integration_build_kkt`. | Need primary owner per evidence row before reviewed promotion. | Day 12 ownership map. |

## Minimum Generated-Index Row Schema

### Required Fields

| Field | Meaning |
| --- | --- |
| `key` | Stable fixture, helper, report, or expected-error key. |
| `row_type` | `fixture`, `external-reference`, `expected-error`, `skip`, `report`, or `policy`. |
| `source` | Path, helper fixture key, report output, or construction owner. |
| `solver_family` | Primary owner family. |
| `fixture_owner` | Source/test owner or explicit `unknown`. |
| `evidence_class` | Correctness, report, policy, or error-path class. |
| `support_tier` | `unsupported`, `smoke`, `reviewed`, `supplemental`, `experimental`, `benchmark`, or `deferred`. |
| `availability` | Local analytic, checked-in parser, checked-in smoke, checked-in reviewed, checked-in expensive, optional local, or optional external. |
| `oracle_source` | Analytic, external helper, published metadata, cross-solver, product observed, none, or unknown. |
| `oracle_output` | Output class or `not-applicable`. |
| `validation_owner` | Focused command, make target, script target, or explicit `unknown`. |
| `failure_class` | Expected failure, skip, helper error, numeric mismatch, report freshness mismatch, or `none`. |
| `claim_boundary` | One-sentence bounded claim and non-claim. |

### Optional Fields

| Field | Meaning |
| --- | --- |
| `path` | Checked-in path when different from source key. |
| `shape` | Matrix dimensions or `malformed`. |
| `stored_entries` | Matrix Market stored entry count. |
| `expanded_nnz` | Logical nonzero count after symmetric expansion, when known. |
| `market_field` | Matrix Market field. |
| `market_symmetry` | Matrix Market symmetry. |
| `definiteness` | SPD, indefinite, nonsymmetric, not-SPD, unknown, or not applicable. |
| `rank_model` | Full rank, exact deficient, threshold rank, singular, unknown, or not applicable. |
| `density` | Diagonal, banded, sparse, medium sparse, dense, pattern-only, or unknown. |
| `conditioning` | Conditioning tag or unknown. |
| `tolerance_policy` | Exact, relative, fixture-specific, threshold-specific, report-only, none, or unknown. |
| `report_owner` | Script or artifact owner for report rows. |
| `freshness_rule` | Command/timestamp/checksum policy for report rows. |
| `docs_owner` | Documentation path for wording changes. |

## Reviewed Corpus Row Promotion Checklist

Before any row can be promoted to `support_tier=reviewed`, it must satisfy:

1. Stable key and row type are defined.
2. Source path or construction owner is recorded.
3. Primary solver family and fixture owner are recorded.
4. Evidence class, oracle source, oracle output, and validation owner match.
5. Structural and numerical tags required by the evidence class are not
   `unknown`.
6. Tolerance, threshold, runtime, skip, and failure class are explicit.
7. The row explains what the evidence proves and what it does not prove.
8. If public or maintainer wording changes, docs owner and non-claim scan are
   recorded.
9. If the row uses SuiteSparse-derived or optional data, support tier,
   availability, runtime, and missing-data behavior are explicit.
10. If the row is a report row, freshness rule and report owner are explicit.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Representative fixtures can be tagged without changing test semantics. | Complete | Dry-run table covers parser, direct LU, Cholesky CSC, QR, partial SVD, eigensolver, reorder/guardrail, and integration rows. |
| Ambiguous tags have blockers and future owners. | Complete | Ambiguity register assigns blockers and Day 6/7/12/future owners. |
| Generated-index schema has required and optional fields. | Complete | Minimum generated-index row schema separates required and optional fields. |

## Day 6 Handoff

Day 6 should use this dry run to define report-index requirements. The key
design pressure is separating fixture metadata from report metadata while
preserving owner-specific support tiers and making expected-error, skip, and
product-observed smoke rows visible instead of silently counting them as
reviewed evidence.

