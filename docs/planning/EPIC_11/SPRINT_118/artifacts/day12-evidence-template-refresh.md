# Sprint 118 Day 12 Evidence Template Refresh

## Purpose

Day 12 publishes the refreshed evidence templates designed on Day 11 for
Sprints 119-127. These templates preserve proof, validation, drift,
non-claim, and handoff discipline for Epic 11 implementation and closeout
work.

The templates are blank reusable artifacts. Day 12 does not perform
source-boundary movement, oracle expansion, performance sentinel changes,
package/ABI implementation, or adoption cleanup reserved for future sprints.

## Published Template Files

| Template | Primary owner sprints | Purpose |
|---|---|---|
| `templates/source-movement-evidence-template.md` | Sprints 119-123 | Source movement, private-owner extraction, internal-header reshaping, and giant-test splits. |
| `templates/oracle-expansion-evidence-template.md` | Sprints 120-122 | Direct, iterative, eigensolver, SVD, QR, rank, corpus, dense-reference, external-reference, and cross-solver proof. |
| `templates/performance-sentinel-evidence-template.md` | Sprints 122-123 | Benchmark/report/sentinel, backend/runtime, report-index, stale-report, and local-measurement evidence. |
| `templates/package-abi-decision-template.md` | Sprints 124-125 | Static-first continuation, shared-library/ABI decisions, install/export proof, platform tiers, and package-manager disposition. |
| `templates/adoption-cleanup-evidence-template.md` | Sprints 126-127 | Public docs, examples, cookbook routes, link/path checks, and claim-boundary cleanup. |
| `templates/template-usage-notes.md` | Sprints 119-127 | Template selection, required inputs, validation rules, claim discipline, and owner map. |

## Required Fields Implemented

| Required field group | Implemented in templates |
|---|---|
| Scope | All five evidence templates. |
| Baseline | All five evidence templates. |
| Proof values | All five evidence templates. |
| Change plan or decision plan | Source, package/ABI, adoption; equivalent planning fields in oracle and performance templates. |
| Validation | All five evidence templates plus usage notes. |
| Drift | All five evidence templates. |
| Non-claims | All five evidence templates plus usage notes. |
| Handoff | All five evidence templates. |

## Cross-References Preserved

| Sprint 118 source | Day 12 use |
|---|---|
| `artifacts/day3-baseline-quality-recheck.md` | Required validation and CTest parity baseline for future template fills. |
| `artifacts/day4-ci-tier-platform-truth.md` | Package/platform tier and staged-exclusion truth for package and adoption templates. |
| `artifacts/day6-residual-owner-map.md` | Proof gates and Sprint 119-127 owner expectations. |
| `artifacts/day8-product-truth-map.md` | Baseline claims, candidate claims, explicit non-claims, and drift guardrails. |
| `artifacts/day9-hotspot-metrics.md` | Source/test metrics and reproducibility commands for source movement templates. |
| `artifacts/day10-hotspot-owner-handoff.md` | Ranked movement/split targets and no-move/defer guidance. |
| `artifacts/day11-evidence-template-design.md` | Template design rationale and Day 12 implementation checklist. |

## Future-Sprint Handoff Rules

| Rule | Rationale |
|---|---|
| Fill the relevant template before broad implementation when the template is acting as a design gate. | Prevents source movement, oracle work, package decisions, or adoption edits from outrunning proof. |
| Update the filled template after validation with observed commands, counts, and residuals. | Keeps planning evidence tied to real command output. |
| Link filled templates from future working notes and retrospectives. | Keeps owner decisions discoverable during Sprint 127 closeout. |
| Preserve explicit non-claims even when focused evidence passes. | Prevents bounded evidence from becoming ecosystem, platform, ABI, or performance overclaims. |
| Carry incomplete template sections into residual deferred debt. | Keeps deferred requirements visible instead of silently dropping them. |

## Validation Boundary

Day 12 changed only Sprint 118 planning documentation:

- no C source files were modified;
- no public headers were modified;
- no Makefile, CMake, workflow, package, benchmark, script, install, or test
  surfaces were modified;
- required validation is documentation hygiene only.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 5 is complete. | Complete. |
| Refreshed source-movement template is published. | Complete. |
| Refreshed oracle-expansion template is published. | Complete. |
| Refreshed performance-sentinel template is published. | Complete. |
| Refreshed package/ABI decision template is published. | Complete. |
| Refreshed adoption-cleanup template is published. | Complete. |
| Template usage notes are published. | Complete. |
| Each template includes proof, validation, drift, and non-claim fields. | Complete. |
| Future sprint handoffs reference the refreshed templates. | Complete. |
