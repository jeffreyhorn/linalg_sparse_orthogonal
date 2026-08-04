# Sprint 137 Day 13 - Handoff Synthesis & Sprint 138 Readiness

## Purpose

Day 13 reconciles the Sprint 137 baseline, residual, selection, template,
quality, and claim-freeze artifacts into a Sprint 138-ready handoff. It also
records dependency-ordered notes for Sprints 139-146 so later implementation
sprints do not need to redo baseline or gap-selection work.

This artifact is documentation-only. It does not implement the Sprint 138
corpus lane and does not widen public claims.

## Reconciled Artifact Index

| Artifact | Status | Handoff role |
| --- | --- | --- |
| `day1-scope-artifact-setup.md` | Complete | Establishes Sprint 137 artifact structure, inherited inputs, day ownership, validation expectations, and claim fences. |
| `day2-source-test-maintainability-baseline.md` | Complete | Provides source/test/benchmark/example size and maintainability baseline for later implementation risk. |
| `day3-build-package-ci-report-baseline.md` | Complete | Provides build, package, CI, platform, report, benchmark, and support-tier baseline. |
| `day4-epic11-residual-intake.md` | Complete | Converts Epic 11 residuals into active candidates, duplicates, already-covered items, optional work, and non-claims. |
| `day5-residual-owner-nongoal-map.md` | Complete | Assigns active residuals to owner workstreams, dependencies, promotion gates, non-goals, and stop conditions. |
| `day6-gap-selection-criteria.md` | Complete | Defines complete-closure criteria, scoring rubric, anti-goals, claim gates, and feasibility guardrails. |
| `day7-gap-selection-decision.md` | Complete | Selects Sprint 138-146 gap targets, records scored candidates, deferrals, rejections, dependencies, and claim boundaries. |
| `day8-corpus-oracle-evidence-templates.md` | Complete | Defines corpus fixture, generated-matrix, optional-data, oracle row, and failure interpretation templates. |
| `day9-report-index-freshness-templates.md` | Complete | Defines report metadata, freshness, stale-report semantics, normalization eligibility, and report non-claims. |
| `day10-package-abi-platform-claim-templates.md` | Complete | Defines package/ABI decision, downstream proof, platform promotion, unsupported-artifact, and public claim templates. |
| `day11-quality-surface-map.md` | Complete | Defines required checks, supplemental checks, hosted-CI dependencies, and stop conditions by touched surface. |
| `day12-public-claim-freeze.md` | Complete | Freezes current public claim boundaries and records non-claims before implementation sprints begin. |

## Contradiction and Duplicate Review

| Topic | Review result | Resolution |
| --- | --- | --- |
| Sprint 138 scope | Day 7 selects a maintained corpus/oracle contract; Day 8 defines templates for the same target. | Consistent. Sprint 138 starts from Day 8 templates and does not need to reselect corpus scope. |
| QR scope | Day 7 selects rank-deficient nullspace/subspace QR closure; Day 8 reserves fixture/oracle fields needed for rank/nullity and subspace rows. | Consistent. Sprint 139 can use Sprint 138 corpus rows rather than inventing QR-specific row semantics. |
| Partial-SVD scope | Day 7 selects repeated/clustered spectra with convergence-budget semantics; Day 8 supports generated fixtures and subspace/tolerance rows. | Consistent. Sprint 140 must narrow to the selected fixture family and preserve SVD non-claims. |
| Report scope | Day 7 selects report normalization/freshness; Day 9 defines row-meaning-preserving templates and excludes unsafe families. | Consistent. Sprint 141 should not flatten row families or treat freshness as product proof. |
| Runtime/backend scope | Day 7 selects precedence plus one sentinel lane; Day 9 and Day 11 keep sentinel evidence local and support-tier bounded. | Consistent. Sprint 142 must choose one sentinel after audit and avoid portable performance claims. |
| Package/ABI scope | Day 7 selects static-first follow-through; Day 10 records static-first as the selected shape and rejects shared ABI/package-manager work for Epic 12. | Consistent. Sprint 143 should not reopen shared-library implementation unless the project plan changes. |
| Platform scope | Day 7 selects Windows CMake install/downstream as the one platform lane; Day 10 defines the exact lane and fallback semantics. | Consistent. Sprint 144 should not promote general Windows parity, POSIX/pthread staged tests, or macOS package parity. |
| Adoption timing | Day 7 schedules adoption after evidence-bearing sprints; Day 12 freezes current public wording until proof lands. | Consistent. Sprint 145 rewrites from earned evidence only. |
| Quality gates | Day 1 and Day 11 both require full C quality for `.c`/`.h` changes. | Consistent. Day 11 is the authoritative implementation-sprint quality map. |
| Public claims | Day 6-7 claim gates and Day 12 freeze all block unsupported state-of-the-art, parity, platform, package, ABI, and performance claims. | Consistent. No immediate public-doc cleanup is required before Sprint 138. |

No unresolved contradiction was found across Day 1-12 artifacts. Duplicate
coverage exists by design where later artifacts refine earlier ones:

- Day 6 defines criteria; Day 7 applies them.
- Day 8 defines corpus/oracle templates; Day 9 consumes them for report rows.
- Day 10 defines claim and product templates; Day 12 freezes current public
  wording until those templates are satisfied.
- Day 11 consolidates validation rules first sketched in Day 1.

## Sprint 138 Corpus Handoff

Sprint 138 can begin from the following fixed inputs:

| Input | Required use in Sprint 138 |
| --- | --- |
| Selected target | Implement a maintained numerical corpus/oracle contract with one durable deterministic fixture lane and explicit skip/defer semantics. |
| Primary owner | Corpus/oracle owner. |
| Supporting owners | Report-index owner, QR owner, partial-SVD owner, and Adoption/docs owner where row interpretation or docs are touched. |
| Required templates | Use Day 8 corpus fixture, deterministic generated-matrix, optional-data skip/defer, oracle row, and failure interpretation templates. |
| Required quality map | Use Day 11 to select validation based on touched surfaces. Any `.c` or `.h` change requires `make format && make lint && make test`. |
| Required claim boundary | Use Day 12 claim freeze. Corpus/oracle rows remain fixture-local evidence and do not imply broad SuiteSparse coverage, external-library parity, package/platform support, performance proof, or state-of-the-art status. |

### Sprint 138 Minimum Implementation Checklist

Sprint 138 should not close until it provides:

1. A maintained corpus manifest path and serialization format.
2. At least one deterministic generated-matrix fixture lane.
3. Stable fixture keys referenced by tests, oracle rows, reports, and docs.
4. Generator metadata with version, algorithm, parameters, canonical format,
   hash policy, regeneration command, and change policy.
5. Optional-data state handling where unavailable data is `skip` or `defer`,
   never `pass`.
6. At least one oracle row path with expected result, observed result,
   tolerance, support tier, command, fixture key, source commit, and comparison
   status.
7. Failure interpretations for oracle mismatch, generator mismatch, stale
   report, optional-data skip, unsupported platform, deferred row, and known
   residual xfail.
8. Focused validation command for the maintained corpus/oracle lane.
9. Documentation that explains fixture-local evidence and preserved non-claims.

### Sprint 138 Stop Conditions

- Corpus fixtures are added without manifest metadata.
- Generated matrices cannot be reproduced from versioned metadata.
- Optional external data can be counted as pass evidence when unavailable.
- Oracle rows omit command, commit, support tier, tolerance, or non-claims.
- Report rows are introduced before row meanings are defined.
- Public docs imply broad corpus completeness, SuiteSparse parity,
  external-library parity, platform support, package support, performance
  proof, or state-of-the-art status.
- Touched code or headers fail the Day 11 quality requirements.

## Later-Sprint Handoff Notes

| Sprint | Handoff from Sprint 137 | Dependency from prior sprint |
| --- | --- | --- |
| 139 QR priority residual | Close rank-deficient nullspace/subspace QR behavior using projector or two-way projection metrics, rank/nullity metadata, fixture-local tolerance, focused tests, docs, and non-claims. | Requires Sprint 138 corpus/oracle fixture and oracle row semantics. |
| 140 partial-SVD residual | Close repeated/clustered-spectrum behavior with singular-value/subspace comparison, convergence-budget semantics, focused tests, docs, and non-claims. | Requires Sprint 138 corpus/oracle semantics and any comparison lessons from Sprint 139. |
| 141 report normalization/freshness | Implement row-meaning-preserving normalized report index and stale-report checks using Day 9 templates. | Requires concrete corpus/oracle rows from Sprint 138 and selected solver evidence from Sprints 139-140. |
| 142 runtime/backend governance | Define runtime/backend precedence and one normalized local sentinel lane. | Requires Sprint 141 report metadata and freshness semantics. |
| 143 package/ABI follow-through | Execute static-first package decision, strengthen optional static mode proof, and preserve shared ABI/package-manager deferrals. | Requires Day 10 templates and Sprint 142 runtime/backend support boundaries where optional modes intersect. |
| 144 platform lane | Attempt Windows CMake install/downstream reviewed lane, or publish blockers and keep it supplemental. | Requires Sprint 143 static-first package semantics and hosted Windows proof. |
| 145 adoption simplification | Rewrite first-use docs and examples from earned evidence only. | Requires settled solver, corpus/report, runtime/backend, package, and platform claims from Sprints 138-144. |
| 146 closeout | Publish final evidence inventory, validation, claims/non-claims, residual queue, retrospective, and state-of-the-art assessment. | Requires all selected Epic 12 sprint closeouts or explicit deferrals. |

## Owner and Evidence Map

| Owner | Next required evidence |
| --- | --- |
| Corpus/oracle owner | Sprint 138 manifest, deterministic fixture lane, optional-data policy, oracle rows, validation command, docs, and residuals. |
| QR owner | Sprint 139 selected QR fixtures, subspace metric, tolerance/rank/nullity metadata, focused tests, docs, and QR residual disposition. |
| Partial-SVD owner | Sprint 140 selected clustered-spectrum fixtures, comparison semantics, convergence-budget proof, focused tests, docs, and SVD residual disposition. |
| Report-index owner | Sprint 141 normalized metadata, stale-report scanner, report-family exclusions, docs, and non-claims. |
| Runtime/backend owner | Sprint 142 runtime/backend precedence, one sentinel lane, backend-state semantics, docs, and performance non-claims. |
| Package/ABI owner | Sprint 143 static-first decision record, downstream proof, optional static mode matrix, unsupported-artifact checks, docs, and ABI residuals. |
| Platform owner | Sprint 144 Windows CMake install/downstream evidence, expected counts, failure semantics, support-tier docs, and fallback decision. |
| Adoption/docs owner | Sprint 145 adoption workflow rewrite, examples/cookbook alignment, docs validation, and no unsupported claim widening. |
| Closeout owner | Sprint 146 final validation, claim recalibration, residual publication, retrospective, and next-epic handoff. |

## Sprint 138 Readiness Decision

Sprint 138 is ready to start without redoing baseline or gap selection.

Readiness evidence:

- baseline and residual intake are complete;
- active residuals have owners and non-goals;
- complete-closure criteria and gap selection are written;
- corpus/oracle evidence templates are implementation-ready;
- report, package, platform, quality, and public claim templates are available
  for downstream sprint dependencies;
- no contradictions were found that block Sprint 138;
- public claim wording is frozen and no immediate cleanup is required before
  corpus implementation.

## Day 13 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 138 can begin without redoing baseline or gap selection. | Complete | Reconciled artifact index, Sprint 138 handoff, minimum implementation checklist, stop conditions, and readiness decision. |
| All later sprint handoffs are dependency ordered. | Complete | Later-sprint handoff notes order Sprints 139-146 from corpus through solver, report, runtime, package, platform, adoption, and closeout dependencies. |
| Contradictions across Sprint 137 artifacts are resolved or explicitly noted. | Complete | Contradiction and duplicate review records no unresolved contradictions and identifies intentional refinement relationships. |
