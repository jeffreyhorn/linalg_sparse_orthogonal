# Sprint 118 Day 11 Evidence Template Refresh Design

## Purpose

Day 11 designs the refreshed evidence templates that Day 12 will publish for
Sprints 119-127. The design builds on the Sprint 100 reusable templates and
the Sprint 118 baseline, residual, truth-map, and hotspot artifacts.

The refreshed templates must keep Epic 11 implementation work evidence-bound:
source movement, oracle expansion, performance sentinels, package/ABI
decisions, and adoption cleanup should each record proof values, validation
commands, drift risk, non-claims, and future handoff requirements.

## Existing Template Inventory

| Existing artifact | Reusable pattern | Gap for Epic 11 |
|---|---|---|
| `docs/planning/EPIC_10/SPRINT_100/artifacts/day9-solver-comparison-template.md` | Separates correctness, convergence, timing, unsupported cases, non-claims, and validation summary. | Does not cover source movement, giant-test splitting, before/after ownership maps, or exact source-list/CMake impact. |
| `docs/planning/EPIC_10/SPRINT_100/artifacts/day10-benchmark-coverage-performance-template.md` | Separates benchmark reports, coverage, performance sentinels, local timing, reset needs, and reviewed/supplemental status. | Needs Epic 11 report-index, stale-report, backend/runtime, and sentinel-owner fields. |
| `docs/planning/EPIC_10/SPRINT_100/artifacts/day11-platform-packaging-evidence-template.md` | Captures package proof, platform tiers, ABI decisions, downstream consumers, expected counts, and staged exclusions. | Needs a sharper static-first-vs-shared decision path and explicit package-manager/non-claim carry-forward fields for Sprint 124-125. |
| Sprint 114-117 retrospectives | Strong residual deferred debt, validation, lessons, non-claim, and handoff sections. | Retrospectives are closeout artifacts, not pre-change evidence templates; future sprints need fill-before-work templates. |
| Sprint 118 Day 6 residual owner map | Defines proof gates for source movement, oracle fixtures, giant-test splits, corpus/report indexes, performance, package/ABI, platform, adoption, and public claims. | Needs reusable blank templates for owner sprints to fill consistently. |
| Sprint 118 Day 8 product truth map | Freezes baseline claims, candidate claims, explicit non-claims, and evidence references. | Needs claim-drift fields in every future template so implementation artifacts do not overstate outcomes. |
| Sprint 118 Day 10 hotspot owner handoff | Ranks source/test targets and defines source-movement and giant-test split prerequisites. | Needs a template that records before/after line counts, responsibility maps, CTest membership, and rollback plans. |

## Template Gap List

| Gap | Why it matters | Day 12 template response |
|---|---|---|
| Source movement evidence is not standardized. | Sprints 119-123 may move private owners, split source files, or reshape helper contracts. Without a template, source-list/CMake and rollback proof can be missed. | Create a source-movement evidence template. |
| Giant-test split evidence is mixed with generic source movement. | Splitting tests has different risks: CTest membership, fixture reuse, failure localization, and before/after responsibility maps. | Include giant-test split sections inside the source-movement template or as a required subsection. |
| Oracle expansion lacks corpus and trust-boundary fields. | New direct/iterative/SVD/QR/eigensolver evidence must preserve tolerance, fixture, expected-failure, and external-reference limits. | Create an oracle-expansion evidence template. |
| Performance sentinels need report-index and backend/runtime fields. | Sprint 123 work should not turn local timings into portable performance claims. | Create a performance-sentinel evidence template. |
| Package/ABI decisions need an explicit decision tree. | Sprint 124 can either implement shared-library/ABI support or preserve static-first support; both require proof. | Create a package/ABI decision evidence template. |
| Adoption cleanup needs claim-boundary and link/path proof. | Sprint 126 docs and examples can accidentally promote candidate claims or break routes. | Create an adoption-cleanup evidence template. |
| Claim drift is not first-class in all templates. | Every future implementation artifact can affect README, install, benchmark, solver-selection, examples, or support wording. | Add drift and public-claim fields to every template. |
| Non-claims can be omitted when work succeeds. | Passing evidence in one bounded lane does not earn ecosystem parity, portable speed, dynamic ABI, package-manager, GPU, or distributed-memory claims. | Add explicit non-claim fields to every template. |
| Handoff and residuals need consistent closeout. | Owner sprints may partially complete work; residuals must not disappear. | Add residual/handoff sections to every template. |

## Refreshed Template Set

Day 12 should publish these reusable templates under
`docs/planning/EPIC_11/SPRINT_118/templates/`:

| Template | Primary users | Purpose |
|---|---|---|
| `source-movement-evidence-template.md` | Sprints 119-123 | Plan and close source movement, private-owner extraction, internal-header reshaping, and giant-test split work. |
| `oracle-expansion-evidence-template.md` | Sprints 120-122 | Record direct, iterative, eigensolver, SVD, QR, rank, corpus, and external/dense-reference evidence. |
| `performance-sentinel-evidence-template.md` | Sprints 122-123 | Record benchmark/report/sentinel changes, local measurement context, backend/runtime state, and stale-report handling. |
| `package-abi-decision-template.md` | Sprints 124-125 | Record static-first continuation, shared-library/ABI decisions, install/export proof, platform tiers, and package-manager disposition. |
| `adoption-cleanup-evidence-template.md` | Sprints 126-127 | Record docs/examples/header wording changes, link/path checks, claim-boundary scan, and user-facing handoff. |
| `template-usage-notes.md` | Sprints 119-127 | Explain which template to use for each touched surface and what validation gates apply. |

## Required Fields Shared By Every Template

| Field group | Required fields |
|---|---|
| Scope | Sprint/day, artifact owner, touched surfaces, explicit out-of-scope surfaces. |
| Baseline | Starting files, line counts or current evidence where relevant, current product truth references, current non-claims. |
| Proof values | Behaviors or claims protected by the work, invariants, unsupported cases, expected failures, and public API impact. |
| Change plan | Exact files or docs to change, build-system impact, consumer impact, rollback or defer plan. |
| Validation | Required commands by touched surface, focused commands, full quality chain trigger, expected CTest count, reviewed/supplemental/local classification. |
| Drift | Public claim impact, docs/examples/install/support wording impact, benchmark/performance wording impact, platform/package wording impact. |
| Non-claims | Explicit claims still not earned after the work succeeds. |
| Handoff | Completed work, deferred work, residual debt, next owner sprint, and evidence links. |

## Source-Movement Template Outline

```markdown
# <Sprint/Day> Source Movement Evidence

## Scope
## Starting Owner Metrics
## Behavior Boundary
## Old/New File Plan
## Internal Header And Private API Contract
## Source-List, Makefile, And CMake Impact
## Public API And Claim Impact
## Focused Consumer Proof
## CTest Membership And Expected Count
## Validation Commands
## Rollback Or Defer Plan
## Giant-Test Split Map
## Non-Claims Preserved
## Residual Handoff
```

Required proof emphasis:

- before/after responsibility map;
- exact old/new files;
- source-list and CMake impact;
- focused tests for every consumer;
- CTest count evidence;
- failure-localization improvement for test splits;
- no hidden public API change unless explicitly designed and validated.

## Oracle-Expansion Template Outline

```markdown
# <Sprint/Day> Oracle Expansion Evidence

## Scope
## Solver Or Behavior Family
## Fixture Taxonomy
## Matrix/RHS/Eigenpair Construction
## Oracle Or Reference Source
## Trust Boundary
## Tolerance And Acceptance Model
## Correctness Metrics
## Convergence Metrics
## Unsupported Or Expected-Failure Cases
## Validation Commands
## Public Claim Impact
## Non-Claims Preserved
## Residual Handoff
```

Required proof emphasis:

- solver-specific tolerances remain visible;
- dense/external/cross-solver references are bounded;
- correctness, convergence, and timing remain separate;
- unsupported cases are explicit;
- no broad ecosystem parity claim is implied.

## Performance-Sentinel Template Outline

```markdown
# <Sprint/Day> Performance Sentinel Evidence

## Scope
## Benchmark Or Report Surface
## Machine, Compiler, Backend, And Thread Context
## Fixture And Runtime Budget
## Metrics And Units
## Threshold Or Report-Only Status
## Baseline Source
## Report Index Or Stale-Report Handling
## Correctness Context
## Validation Commands
## Public Claim Impact
## Non-Portable Interpretation
## Residual Handoff
```

Required proof emphasis:

- local measurement context is mandatory;
- benchmark correctness context does not replace tests/oracles;
- thresholded sentinels and report-only benchmarks stay separate;
- portable speed and vendor-backend parity remain non-claims.

## Package/ABI Decision Template Outline

```markdown
# <Sprint/Day> Package And ABI Decision Evidence

## Scope
## Decision
## Static-First Contract
## Shared-Library And ABI Contract
## Installed Artifact Expectations
## Package Metadata And Version Behavior
## Downstream Consumer Proof
## Platform Tier Impact
## Expected Test Counts And Staged Exclusions
## Validation Commands
## Public Claim Impact
## Package-Manager Disposition
## Non-Claims Preserved
## Residual Handoff
```

Required proof emphasis:

- static-first continuation and shared-library support are separate decisions;
- installed artifacts and explicitly absent artifacts are listed;
- `pkg-config` and CMake consumer proof are separate;
- platform tiers and staged exclusions are updated only with proof;
- package-manager support remains a non-claim unless real recipes and
  consumer proof exist.

## Adoption-Cleanup Template Outline

```markdown
# <Sprint/Day> Adoption Cleanup Evidence

## Scope
## User-Facing Route Changed
## Current Product Truth References
## Claim-Boundary Scan
## Files And Links Changed
## Example Or Cookbook Proof
## Install/Package/Platform Wording Impact
## Benchmark/Performance Wording Impact
## Validation Commands
## Link And Path Checks
## Non-Claims Preserved
## Residual Handoff
```

Required proof emphasis:

- compressed-first remains the public product center;
- mutable shell remains compatibility, not performance center;
- docs/examples do not promote candidate claims before owner evidence exists;
- link/path checks are recorded;
- non-claims remain explicit where public wording touches support boundaries.

## Day 12 Implementation Checklist

| Step | Day 12 action |
|---:|---|
| 1 | Create `docs/planning/EPIC_11/SPRINT_118/templates/`. |
| 2 | Add the five refreshed blank templates. |
| 3 | Add `template-usage-notes.md` explaining which template applies to source movement, oracle work, performance/report work, package/ABI/platform decisions, and adoption cleanup. |
| 4 | Ensure every template includes scope, baseline, proof values, validation, drift, non-claims, and handoff sections. |
| 5 | Cross-reference Day 6 proof gates, Day 8 product truth map, Day 10 handoff, and Sprint 119-127 owners. |
| 6 | Keep templates blank and reusable; do not perform owner-sprint implementation work during Sprint 118. |
| 7 | Run documentation hygiene checks after adding the template files. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 5 has a bounded update design. | Complete. |
| Existing-template inventory is recorded. | Complete. |
| Template gaps are listed. | Complete. |
| Refreshed template outlines are drafted. | Complete. |
| Required evidence fields are defined. | Complete. |
| Day 12 implementation checklist is ready. | Complete. |
| Templates preserve evidence visibility and non-claim discipline. | Complete. |
| Future sprints can use the design without rediscovering required fields. | Complete. |
