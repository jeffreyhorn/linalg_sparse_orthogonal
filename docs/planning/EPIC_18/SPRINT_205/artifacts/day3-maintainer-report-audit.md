# Day 3: Maintainer And Report Surface Audit

**Sprint:** 205 - Support Matrix and Adoption Quick-Reference Consolidation  
**Theme:** Audit maintainer, benchmark, report, and planning-adjacent docs for
support truth drift and diagnostics vocabulary inconsistency.  
**Time estimate:** 12 hours  
**Branch:** `sprint-205`  
**Base commit:** `9a2b4ff6`

## Purpose

Day 3 completes the Sprint 205 audit pass by reviewing maintainer-only,
benchmark/report, selected-target manifest, schema, and planning-adjacent
surfaces. The goal is to keep the future quick reference and support
consolidation grounded in the correct evidence owners without turning planning
or maintainer proof detail into user-facing support claims.

## Reviewed Maintainer And Report Surfaces

| Surface | Role | Day 3 classification |
| --- | --- | --- |
| `docs/maintainer_guide.md` | Maintainer interpretation for support tiers, claim boundaries, generated API, benchmark governance, report freshness, platform evidence, deferred queues, and guard expectations. | Authoritative for proof interpretation, but too dense for the public quick reference. Keep as maintainer evidence owner. |
| `benchmarks/README.md` | User-facing benchmark/report interpretation, canonical report commands, methodology fields, and performance non-claims. | Authoritative for benchmark users after workflow selection. Do not collapse into README or quick-reference proof wording. |
| `tests/corpus/schemas/report_index_fields.md` | Schema contract for report-family fields, selected-target fields, freshness policies, and guardrails. | Authoritative schema vocabulary. Public docs should link to higher-level report docs rather than copy schema detail. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Source of truth for selected oracle, comparison, and benchmark target identity, workflow metadata, claim scope, and non-claims. | Authoritative selected-target contract. Any support/quick-reference claim that names selected evidence must stay consistent with this manifest. |
| `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` | Epic-level outcome and validation snapshot through Sprint 204. | Evidence summary only. Link from maintainer docs when useful; do not make it user workflow. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Residual queue with closure conditions and non-claims. | Residual handoff only. Do not expose as first-use adoption guidance. |

## Source-Of-Truth Split

| Question | Public owner | Maintainer/report owner | Notes for Sprint 205 |
| --- | --- | --- | --- |
| "What is supported or validated?" | `INSTALL.md#support-readiness-matrix` | `docs/maintainer_guide.md` for interpretation | Keep public labels concise; keep proof details in maintainer/report docs. |
| "Which selected report targets are freshness-bearing?" | Public docs should link to report docs | `tests/corpus/manifests/selected_report_targets.tsv` | Manifest is contract truth; quick reference should not duplicate row lists. |
| "How should benchmark rows be interpreted?" | `benchmarks/README.md` | `docs/maintainer_guide.md`, report schema, selected target manifest | Preserve threshold-free/local/hosted-selected distinctions. |
| "What does generated API evidence prove?" | `docs/api_reference.md` and `INSTALL.md` | `docs/maintainer_guide.md`, API docs guards | Keep Sprint 204 local-only policy. |
| "What work remains residual?" | No first-use public owner | `EPIC_18_RESIDUAL_QUEUE.md` and sprint artifacts | Planning artifacts are evidence, not adoption docs. |

## Diagnostics Vocabulary Inventory

| Term | Current contexts | Potential conflict | Day 9 handling |
| --- | --- | --- | --- |
| `status` | Iterative solver result structs, report-index rows, selected benchmark rows, comparison rows, validation logs. | `status=measurement` in benchmark rows is not pass/fail; solver status describes algorithm outcome. | Define context-qualified status terms. |
| `result` | Solver result structs, validation command results, report row outputs. | Users may read validation result as solver result. | Use `solver result`, `validation result`, or `report row` explicitly. |
| `residual` / `residual_norm` | Direct solve checks, iterative final residual, Ritz residual, QR/SVD comparison rows, benchmark/report diagnostics. | Same phrase means different computed quantities across families. | Define family-specific residual vocabulary. |
| `convergence` | Iterative solvers, eigensolver workflows, benchmark/sentinel reports. | Iteration convergence is not benchmark pass/fail and not external parity. | Tie convergence wording to the owning workflow. |
| `fresh` / `freshness` | Report-index selected gates, generated API docs, benchmark canonical report freshness. | Freshness means current generated evidence, not support promotion or release proof. | Preserve "freshness is currency, not support widening". |
| `pass` / `fail` | Tests, validation gates, report rows, generated comparison status. | Source-controlled contract rows must not use `pass`; observed generated rows own pass/fail status. | Keep schema distinction visible. |
| `skip` | Optional-data reports, environment prerequisites, validation commands. | Skips can explain absence but are not pass evidence. | Use `skip` only with owner and reason. |
| `defer` / `deferred` | Residual queues, report rows, product decisions, unsupported support surfaces. | Deferral can be mistaken for planned support. | Pair deferred status with exact closure condition and non-claim. |
| `local_only` | Generated API, selected report targets, local benchmark reports, allocation-failure evidence. | Local-only can mean local generated output, local proof, or local report evidence. | Define by surface: API, report, benchmark, or proof owner. |
| `hosted_selected` | Selected benchmark support tier and hosted workflow metadata. | Hosted selected evidence is not broad hosted support or portable performance. | Keep selected row and workflow artifact scope visible. |
| `claim_boundary` / `claim_scope` | Benchmark/report metadata, selected target manifest, maintainer guide. | Users may not distinguish positive claim scope from retained non-claims. | Public quick reference should avoid raw schema terms unless linked. |
| `non_claims` | Manifest rows, maintainer guide, public caveats. | Long lists create friction but protect scope. | Centralize long lists in owner docs and link from concise public text. |

## Repeated Maintainer Caveats

| Caveat family | Current maintainer/report locations | Keep local or centralize? | Rationale |
| --- | --- | --- | --- |
| Package-manager and ABI non-claims | Maintainer guide, INSTALL, API reference, residual queue. | Centralize public truth in INSTALL; keep maintainer interpretation local. | Public users need one answer; maintainers need closure conditions. |
| Windows selected freshness re-deferrals | Maintainer guide, INSTALL, README, selected-target schema, residual queue. | Keep detailed deferral in maintainer/schema/residual docs; public docs should use short guarded/deferred labels. | The exact promotion rule is subtle and should not be copied everywhere. |
| Generated API local-only non-publication | Maintainer guide, API reference, INSTALL, API docs guards. | Keep owner detail in API reference and maintainer guide; link from support matrix. | Sprint 204 guard stack is specific and should remain owner-owned. |
| Benchmark portable-performance non-claims | Maintainer guide, benchmarks README, selected target manifest, README/cookbook/solver docs. | Keep detailed methodology in benchmarks README and manifest; public quick reference should route after workflow selection. | Benchmark rows are easy to overread as performance proof. |
| Report freshness selected/unselected split | Maintainer guide, report schema, selected target manifest, benchmark docs. | Keep in schema/manifest/maintainer docs; public docs should avoid target lists. | The manifest is already the source of truth. |
| State-of-the-art non-claims | Maintainer guide, public docs, residual queue. | Keep public concise; maintainer/residual docs own closure conditions. | Broad claim would require a dedicated evidence sprint. |

## Planning-Evidence Routing Notes

| Planning artifact | Use from Sprint 205 | Do not use as |
| --- | --- | --- |
| `EPIC_18_RETROSPECTIVE.md` | High-level current outcome and validation snapshot. | A first-use guide or support matrix replacement. |
| `EPIC_18_RESIDUAL_QUEUE.md` | Closure conditions for residual package, platform, API publication, benchmark, and state-of-the-art gaps. | A public support page. |
| Sprint 198-204 retrospectives | Evidence for inherited decisions and non-claims. | User-facing install or solver instructions. |
| Sprint 205 artifacts | Review evidence and implementation rationale. | Source of truth after public docs/guards are updated. |

## Claim-Safety Findings

| Finding | Severity | Notes |
| --- | --- | --- |
| Maintainer/report support truth is internally consistent with the public support matrix. | Low | No stale support promotion was found in the audited maintainer/report surfaces. |
| Diagnostics vocabulary is overloaded across solver, benchmark, report, and validation contexts. | Moderate | Day 9 should normalize vocabulary before public wording changes. |
| Selected-target manifest is the strongest anti-drift owner for report claims. | Low | Day 11 guards should prefer manifest-backed checks over duplicated target lists. |
| Planning artifacts are useful evidence but too historical for public adoption. | Moderate friction | Day 5-Day 8 should link to maintainer docs rather than directly to planning artifacts in user paths. |
| Public quick-reference wording must not copy raw schema terms without explanation. | Moderate | `support_tier`, `claim_boundary`, and `freshness_policy` are maintainer/report terms. |

## Day 3 Recommendations For Later Days

1. Keep public support truth in `INSTALL.md#support-readiness-matrix`; use
   maintainer/report surfaces for proof interpretation and closure conditions.
2. Design the quick reference with user labels, not raw manifest/schema labels.
3. Treat `selected_report_targets.tsv` as authoritative for selected target
   identity, hosted workflow metadata, support tier, claim scope, and
   non-claims.
4. Do not expose Epic planning artifacts as first-use routes; route public
   users to README, cookbook, solver selection, INSTALL, API reference,
   examples, and benchmarks.
5. Build Day 9 diagnostics vocabulary around context-qualified terms:
   solver result, validation result, report row, residual type, freshness
   state, support status, and claim boundary.
6. Prefer guard updates that check owner boundaries rather than exact long
   paragraphs copied into multiple docs.

## Validation And Hygiene

| Check | Day 3 result |
| --- | --- |
| Public docs edited | No. Day 3 is audit-only. |
| Maintainer/report docs edited | No. |
| User-facing claim changed | No. |
| `.c` or `.h` files changed | No. Full C gate not required for Day 3. |
| Generated output changed | No generated output intentionally created. |
| `git diff --check` | Planned after artifact creation. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 205.1 includes maintainer and report surfaces, not only public docs. | Met. Maintainer guide, benchmark docs, report schema, selected target manifest, Epic retrospective, and residual queue are covered. |
| Diagnostics vocabulary conflicts are identified before wording changes. | Met. Overloaded diagnostics and report terms are inventoried before Day 9 wording changes. |
| Planning artifacts remain evidence, not replacement user documentation. | Met. Planning-evidence routing notes explicitly keep planning artifacts out of first-use public documentation. |

## Day 3 Disposition

Day 3 is complete. Day 4 should design the compact quick reference using the
Day 2 public audit and this maintainer/report audit, while keeping support
truth anchored in INSTALL and proof interpretation anchored in maintainer,
benchmark, manifest, and report-schema owners.
