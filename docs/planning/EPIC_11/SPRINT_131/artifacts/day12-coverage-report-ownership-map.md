# Sprint 131 Day 12 - Coverage and Report Ownership Map

## Purpose

Day 12 consolidates Sprint 131 owner labels for corpus, coverage, reports,
indexes, guardrails, and claim gates. It maps each recurring assurance area to
files, scripts, generated outputs, validation commands, and future queues, then
records explicit orphan or residual status where ownership is not yet ready.

This is a documentation-only ownership artifact. It does not change
maintainer-guide wording, source code, tests, scripts, Makefile targets,
coverage thresholds, benchmark semantics, CI, generated report schemas, or
public claims.

## Owner Label Vocabulary

| Owner label | Scope | Primary files or outputs |
| --- | --- | --- |
| `corpus-taxonomy-owner` | Fixture/report tags, support tiers, promotion/demotion rules, and non-claim boundaries. | Sprint 131 Day 4-5 artifacts, future corpus index artifacts. |
| `sparse-io-corpus-owner` | Parser and Matrix Market IO fixtures. | `tests/data/*.mtx`, `tests/test_sparse_io.c`, `docs/matrix_market.md`. |
| `direct-solver-corpus-owner` | LU, CSR LU, Cholesky, LDLT, QR, expected failures, and direct external-reference fixtures. | Direct solver tests and external helper scripts. |
| `svd-corpus-owner` | SVD, partial-SVD, bidiag, singular-value helper rows, vector-residual rows, and SVD support tiers. | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, `tests/test_bidiag.c`, `tests/svd_external_dense_reference.py`. |
| `eigs-corpus-owner` | Eigensolver generated and checked-in smoke/report rows. | `tests/test_eigs*.c`, eigensolver source/test fixtures. |
| `graph-reorder-corpus-owner` | Graph, reorder, ND, qg-AMD, and large structural fixture rows. | `tests/test_graph*.c`, `tests/test_reorder*.c`, `tests/data/suitesparse/*.mtx`. |
| `coverage-core-structures` | Core sparse structure coverage risk and residual queue. | `src/sparse_matrix.c`, `src/sparse_csr.c`, `src/sparse_vector.c`, `src/sparse_dense.c`, owner tests. |
| `coverage-direct-solvers` | Direct solver coverage risk and residual queue. | LU, Cholesky, LDLT, QR source and owner tests. |
| `coverage-iterative-preconditioners` | Iterative solver, ILU, IC, stagnation, and matrix-free coverage risk. | `src/sparse_iterative.c`, `src/sparse_ilu.c`, `src/sparse_ic.c`, owner tests. |
| `coverage-eigensolvers` | Eigensolver coverage risk and residual queue. | `src/sparse_eigs.c`, eigensolver tests. |
| `coverage-svd-bidiag` | SVD and bidiag coverage risk and residual queue. | `src/sparse_svd.c`, `src/sparse_bidiag.c`, SVD/bidiag tests. |
| `coverage-symbolic-graph` | Symbolic, reorder, graph, and ND coverage risk. | Analysis, etree, reorder, COLAMD, graph source and owner tests. |
| `coverage-workflow` | Coverage targets, backend behavior, threshold, reports, source filters, and reset requirement. | `Makefile`, `.github/workflows/ci.yml`, `coverage/` outputs. |
| `deadcode-workflow` | Dead-code raw evidence, classification, report completeness, serial execution, and residual workflow gaps. | `Makefile`, `scripts/deadcode_workflow.sh`, `scripts/deadcode_report.py`, `build/deadcode/`. |
| `large-matrix-guardrails` | Guardrail lane IDs, reviewed/supplemental split, manifest, index, and bounded CSV-shape report behavior. | `Makefile`, `scripts/large_matrix_guardrails.sh`, `build/bench-reports/large-matrix-guardrails/`. |
| `report-index-owner` | Cross-report schemas, freshness fields, stale-report policy, row identity, and generated-versus-curated decisions. | Sprint 131 Day 6-11 artifacts and future generated index scripts. |
| `benchmark-report-owner` | Canonical benchmark, sentinel, and benchmark-local report schemas and non-claims. | `benchmarks/README.md`, `scripts/bench_canonical_report.sh`, `scripts/performance_sentinels.sh`, benchmark binaries. |
| `external-oracle-owner` | External-reference helper protocol, output class, fixture keys, skips, and helper non-claims. | `tests/*_external_dense_reference.py`, owner tests, Sprint 120-130 artifacts. |
| `maintainer-guide-owner` | Maintainer wording, non-claim guardrails, and public/maintainer interpretation. | `docs/maintainer_guide.md`, related user-facing docs when changed. |
| `sprint-planning-owner` | Sprint artifacts, retrospectives, residual queues, and historical traceability. | `docs/planning/EPIC_11/SPRINT_131/`. |

## Recurring Assurance Ownership Map

| Assurance area | Owner | Files and outputs | Validation owner | Current status |
| --- | --- | --- | --- | --- |
| Corpus taxonomy and support tiers | `corpus-taxonomy-owner` | Day 4-5 taxonomy and dry-run artifacts | Documentation hygiene unless index/schema changes | Owned, curated. |
| Matrix Market parser fixtures | `sparse-io-corpus-owner` | `tests/data/*.mtx`, `tests/test_sparse_io.c`, `docs/matrix_market.md` | Focused sparse IO tests if behavior changes | Owned reviewed/parser or unsupported rows. |
| Checked-in SuiteSparse-derived corpus rows | Solver-family corpus owner plus `corpus-taxonomy-owner` | `tests/data/suitesparse/*.mtx` | Owner tests or report targets by row | Partially owned; many rows remain smoke/report until metadata is complete. |
| Direct solver external-reference rows | `direct-solver-corpus-owner` and `external-oracle-owner` | Cholesky, LDLT, LU, QR helper scripts and owner tests | Focused direct solver tests | Owned within helper-specific protocols. |
| SVD and partial-SVD oracle rows | `svd-corpus-owner` and `external-oracle-owner` | SVD helper, SVD tests, partial helper header | Focused SVD tests | Owned for bounded singular-value/vector-residual lanes only. |
| Eigensolver smoke/report rows | `eigs-corpus-owner` | `tests/test_eigs*.c`, checked-in fixtures | Focused eigensolver tests | Owned as smoke unless independent metadata is added. |
| Integration fixtures | Primary solver owner per row plus `corpus-taxonomy-owner` | `tests/test_integration*.c`, fixture helpers | Focused integration tests | Explicitly multi-owner; reviewed promotion requires primary owner per row. |
| Coverage family risks | Day 8 coverage owner labels | Source families and owner tests | Focused owner tests; `make coverage*` remains supplemental | Owned by family; residual gaps have blockers. |
| Coverage workflow | `coverage-workflow` | Makefile coverage targets, CI supplemental coverage job, `coverage/` outputs | Coverage command only when coverage/report behavior changes | Owned supplemental workflow. |
| Dead-code reports | `deadcode-workflow` | `build/deadcode/report.md`, `report.tsv`, raw evidence files | `make deadcode-report` and `make deadcode-check` when workflow/report changes | Owned triage/report completeness surface. |
| Large-matrix guardrail index | `large-matrix-guardrails` | `index.tsv`, `manifest.txt`, reviewed logs, bounded CSV | `make large-matrix-guardrails` when script/report changes | Owned first generated index path. |
| Cross-report index strategy | `report-index-owner` | Day 6-11 artifacts, future normalized schema | Docs hygiene now; future script tests when implemented | Owned as deferred architecture. |
| Canonical benchmark reports | `benchmark-report-owner` | `build/bench-reports/canonical/`, benchmark scripts/docs | Existing benchmark report command when touched | Owned benchmark/report surface. |
| Performance sentinels | `benchmark-report-owner` | `build/bench-reports/sentinels/`, sentinel scripts | `make performance-sentinels` when touched | Owned local report surface with bounded wall-check semantics. |
| External-reference helper index | `external-oracle-owner` and `report-index-owner` | Helper scripts and future curated/generated index | Focused owner tests plus helper protocol validation | Deferred; no generated index yet. |
| Planning artifacts | `sprint-planning-owner` | Sprint 131 plan, working notes, artifacts, retrospective | Docs hygiene | Owned curated traceability. |
| Maintainer interpretation | `maintainer-guide-owner` | `docs/maintainer_guide.md` | Non-claim scan and docs hygiene when changed | Owned; no Day 12 wording update needed. |

## Orphaned Output Register

| Output or row family | Orphan status | Blocker | Future owner |
| --- | --- | --- | --- |
| Broad checked-in SuiteSparse corpus index | Orphaned as a generated index | Missing per-row conditioning, oracle provenance, support tier, runtime, and missing-data policy. | `corpus-taxonomy-owner` plus solver-family owners. |
| External-reference helper generated index | Orphaned as generated output | Helper protocols are documented, but no generated row emitter exists and output classes differ by helper. | `external-oracle-owner` plus `report-index-owner`. |
| Cross-report normalized index | Orphaned as implementation | Coverage, dead-code, benchmark, guardrail, oracle, and planning rows have different freshness and failure semantics. | `report-index-owner`. |
| Coverage generated index | Deferred, not orphaned | Day 8 fields exist, but no generator and coverage remains tree-mutating/supplemental. | `coverage-workflow`. |
| Dead-code freshness index | Deferred, not orphaned | `report.tsv` has bucket schema but no manifest-style branch/commit freshness fields. | `deadcode-workflow`. |
| Supplemental large-matrix recurring validation | Deferred, not orphaned | Supplemental lanes are opt-in and threshold-free; no recurring runtime budget or support-tier promotion decision. | `large-matrix-guardrails`. |
| Integration fixture reviewed corpus rows | Partially orphaned | Multi-owner fixtures need primary owner, evidence class, oracle, tolerance, and claim boundary per row. | `corpus-taxonomy-owner` plus affected solver owner. |
| Product-observed eigensolver/SVD rows | Partially orphaned | Product-observed metrics can be overcounted without independent oracle metadata. | `eigs-corpus-owner` or `svd-corpus-owner`. |
| Automated stale-report scanner | Orphaned as tooling | No common metadata contract across report families yet. | `report-index-owner`. |
| Maintainer wording refresh | Not orphaned | Existing wording already matches accepted Day 8-11 boundaries. | `maintainer-guide-owner` if future evidence changes. |

## Supplemental-To-Reviewed Promotion Criteria

Any smoke, supplemental, benchmark, experimental, or deferred row can move to
reviewed recurring assurance only when all criteria below are satisfied:

1. Stable row key, source path or construction rule, and primary owner.
2. Explicit evidence class, support tier, availability, failure class, and
   claim boundary.
3. Solver-family or report-family owner accepts the row as within scope.
4. Oracle source and output class match the assertion class, or the row states
   why analytic evidence is sufficient.
5. Numerical tags required by the row are known, including shape, rank model,
   definiteness, conditioning or tolerance rationale, and density when relevant.
6. Runtime budget, optional-data behavior, skip behavior, and platform
   assumptions are explicit.
7. Validation command is reproducible and appropriate for the support tier.
8. Freshness rule is explicit for report rows.
9. Maintainer or public wording has a docs owner and non-claim scan if changed.
10. Promotion does not broaden to solver-family, corpus, coverage, benchmark,
    platform, performance, memory, or ecosystem parity.

Demotion remains required when oracle metadata is lost, runtime or optional
data behavior no longer matches the support tier, freshness is absent, or
wording exceeds the accepted evidence boundary.

## Maintainer Wording Decision

No maintainer-facing wording change is required on Day 12.

Rationale:

- `docs/maintainer_guide.md` already states that coverage remains
  supplemental and should not be treated as an active reviewed baseline.
- The guide already describes the dead-code workflow as conservative evidence,
  with `deadcode-check` acting as report-completeness rather than
  zero-findings or removal-ready proof.
- The guide already describes `large-matrix-guardrails` as bounded structural
  guardrails and threshold-free supplemental report context, not broad
  large-matrix performance proof.
- Day 10 accepted the existing generated large-matrix guardrail index without
  schema or semantics changes.
- Day 11 defined freshness as report traceability only, not a stronger CI,
  release, scalability, timing, memory, coverage, or corpus guarantee.

A future maintainer-guide update is justified only if a later sprint changes a
target, schema, support tier, CI role, public claim, or recurring freshness
requirement.

## Future-Owner Queue

| Future work | Blocker | Dependency | Owner |
| --- | --- | --- | --- |
| Create a generated corpus index | Need row-level metadata for SuiteSparse, integration, product-observed, and expected-error rows. | Day 4-5 taxonomy plus solver-family owner review. | `corpus-taxonomy-owner`. |
| Add coverage index generator | Need command/backend/freshness fields and tree-mutating reset policy in generated rows. | Day 8 coverage architecture. | `coverage-workflow`. |
| Add dead-code report freshness metadata | Need manifest-style branch/commit/timestamp fields without weakening bucket classification. | Future dead-code index decision. | `deadcode-workflow`. |
| Normalize guardrail index fields | Need cross-report schema decision that preserves current `index.tsv` semantics. | Day 11 freshness policy and future schema design. | `report-index-owner`. |
| Validate supplemental guardrail mode recurring use | Need runtime and support-tier policy for opt-in lanes. | Large-matrix owner baseline work. | `large-matrix-guardrails`. |
| Generate external-reference helper index | Need stable helper row emitter and output-class schema. | External helper protocol owner review. | `external-oracle-owner`. |
| Resolve integration fixture ownership | Need primary owner and evidence class per integration row. | Future corpus index design. | `corpus-taxonomy-owner` plus solver owners. |
| Define stale-report scanner | Need common metadata contract across report families. | Future normalized index schema. | `report-index-owner`. |
| Revisit maintainer wording | Need accepted semantics change or new recurring gate. | Future sprint evidence. | `maintainer-guide-owner`. |

## Day 13 Handoff

Day 13 should use this ownership map to publish the final residual assurance
queue. It should verify that every residual corpus, coverage, report,
dead-code, large-matrix, oracle, guardrail, and validation item has a blocker,
dependency, and owner, then run the final documentation hygiene checks.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every recurring assurance area has an owner or explicit orphan status. | Complete | Recurring ownership map and orphaned-output register cover corpus, coverage, reports, indexes, guardrails, dead-code, oracle, planning, and maintainer wording. |
| No orphaned output lacks blocker and future owner. | Complete | Orphaned-output register and future-owner queue list blockers and owners for each unresolved area. |
| Maintainer wording changes, if any, trace directly to accepted decisions. | Complete | Day 12 records a no-update rationale because current maintainer wording already matches accepted Sprint 131 decisions. |
