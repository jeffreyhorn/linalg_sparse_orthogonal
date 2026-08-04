# Sprint 137 Day 6 - Gap-Selection Criteria

## Purpose

Day 6 defines the rules Sprint 137 will use to select Epic 12 gap targets.
The criteria deliberately favor complete closure of selected gaps over broad
partial progress across many residuals.

This artifact does not select the gaps. Day 7 applies this rubric to the Day 5
owner map and records the Sprint 138-146 implementation targets, deferrals,
and claim boundaries.

## Complete-Closure Definition

A gap counts as closed only when all required closure dimensions are satisfied.
Evidence can be narrow, but it must be explicit, reproducible, and aligned with
the public claim being made.

| Dimension | Closure requirement |
| --- | --- |
| Scope boundary | The gap has a named fixture, report family, platform lane, package mode, solver behavior, or documentation surface that can be finished inside the assigned sprint budget. |
| Primary owner | One owner workstream is accountable for implementation, validation, documentation, and residual handoff. Supporting owners are listed when dependencies cross workstreams. |
| Implementation path | The required code, script, CI, package, or documentation changes are identified before the gap is promoted. |
| Evidence source | The proof source is named: focused test, generated report, install proof, hosted CI lane, corpus manifest, oracle output, or reviewed documentation validation. |
| Validation command | The command or hosted lane needed to prove the claim is known and can be repeated. |
| Row and output semantics | Fixtures, reports, package metadata, or examples describe what each row/output means without flattening different evidence families into one generic schema. |
| Support tier | Linux, macOS, Windows, optional-data, optional-backend, and supplemental/reviewed boundaries are explicit. |
| Documentation alignment | README, INSTALL, tutorial, cookbook, solver-selection, benchmark, algorithm, and maintainer docs are updated only where the evidence earns new wording. |
| Non-claim preservation | The work records what remains unproven, including external-library parity, portable performance, broad package-manager support, shared ABI, and state-of-the-art status where applicable. |
| Residual disposition | Any remaining work is either closed, deferred with a reason, rejected as a non-goal, or carried forward with a dependency. |

## Scoring Model

Score each candidate gap from 0 to 3 in each category. The maximum score is 21.
The score is advisory; any stop condition or missing closure requirement can
still block selection.

| Category | 0 points | 1 point | 2 points | 3 points |
| --- | --- | --- | --- | --- |
| User value | Mostly internal cleanup with no clear user-facing result. | Helps maintainers or a narrow advanced workflow. | Improves a documented user workflow or reduces adoption friction. | Directly improves install, solver trust, diagnostics, or first-use success for a maintained workflow. |
| State-of-the-art relevance | Does not affect the current state-of-the-art gap assessment. | Adds context but does not close a competitive shortcoming. | Closes a bounded proof or product gap that supports future competitive claims. | Closes a high-impact gap required before any credible state-of-the-art reconsideration. |
| Dependency readiness | Requires multiple unresolved decisions or upstream artifacts. | Has known dependencies but at least one is not yet specified. | Depends on one prior artifact or selected scope that Sprint 137 can provide. | Can start from current evidence and has a clear closure path. |
| Testability and proofability | Cannot be validated without broad external infrastructure. | Can be manually inspected but lacks a repeatable proof command. | Has a repeatable focused command or generated artifact path. | Has focused tests, report output, install proof, or hosted CI evidence that can become a maintained gate. |
| Platform and package risk | Implies unsupported platform, ABI, loader, or package-manager behavior. | Has high platform/package uncertainty and would need fallback wording. | Has bounded platform/package risk with explicit support-tier limits. | Avoids platform/package expansion or includes strong install/export/downstream proof. |
| Documentation and adoption impact | Would mainly add internal notes. | Clarifies maintainer-only interpretation. | Improves public docs after implementation lands. | Removes a major public ambiguity in install, solver selection, reports, support tiers, or examples. |
| Complete-closure feasibility | Cannot fit in one assigned sprint without shallow partial work. | Fits only if major proof or documentation is deferred. | Fits with careful narrowing and explicit residuals. | Fits inside the assigned sprint with implementation, tests, docs, and claim gates together. |

### Score Interpretation

| Score | Selection meaning |
| --- | --- |
| 17-21 | Strong candidate for Day 7 selection if no stop condition applies. |
| 13-16 | Candidate only if it aligns with dependency order and has a narrow closure target. |
| 9-12 | Defer, split, or reduce scope before selection. |
| 0-8 | Reject for Epic 12 unless the project plan changes. |

## Candidate Pre-Screen Rules

| Candidate family | Day 6 pre-screen |
| --- | --- |
| Corpus/oracle architecture | Treat as foundational. It should be selected only as a maintained row-semantics and oracle contract, not as a broad fixture dump. |
| QR residual closure | Select at most one priority QR residual family unless Day 7 proves the second is a direct byproduct of the same fixtures and validation. |
| Partial-SVD residual closure | Select at most one priority partial-SVD residual family unless the second shares the same comparison semantics and proof owner without broadening claims. |
| Report normalization and freshness | Select only row-meaning-preserving normalization. Reject schemas that flatten benchmark, package, coverage, dead-code, oracle, and sentinel evidence into one ambiguous row type. |
| Runtime/backend governance | Select a bounded precedence or sentinel claim. Reject portable speedup, backend parity, or memory/scalability superiority claims without reproducible cross-platform evidence. |
| Package/ABI productization | Day 7 must choose either a bounded implementation path or an explicit static-first deferral. Do not select full shared ABI, package-manager distribution, and all-platform loader proof together unless budget and proof are concrete. |
| Platform promotion | Select one platform lane for complete promotion or staged-test closure. Do not promote macOS and Windows broadly from supplemental wording alone. |
| Adoption simplification | Schedule after evidence-bearing sprints. Adoption docs should explain earned behavior, not lead implementation. |
| Maintainability cleanup | Select only cleanup required to close a chosen proof gap. Reject broad decomposition of large tests or sources as a standalone Epic 12 success metric. |

## Anti-Goals

- Touching every residual family shallowly.
- Adding fixtures without taxonomy, expected-result metadata, skip behavior,
  tolerance policy, support tier, and validation command.
- Treating generated report indexes as broad correctness, release, coverage,
  or performance proof.
- Flattening report schemas so different row meanings become ambiguous.
- Adding hard timing thresholds without a local baseline, variance policy,
  runtime budget, and backend/platform interpretation.
- Enabling shared-library builds without artifact naming, export/import policy,
  ABI boundary, install/export metadata, loader behavior, downstream proof, and
  documentation.
- Promoting macOS or Windows support tiers from documentation wording alone.
- Using docs updates to imply new solver behavior, platform support, package
  support, external-library parity, or state-of-the-art status.
- Splitting large tests or source files without improving a selected proof
  owner, fixture owner, or claim gate.

## Claim Gate Matrix

| Claim family | Evidence required | Validation required | Documentation required | Blocked claims until satisfied |
| --- | --- | --- | --- | --- |
| Corpus/oracle | Fixture taxonomy, provenance, expected-result semantics, skip/defer policy, support tier, and row schema. | Focused corpus/oracle generator or validation command plus link/path hygiene for reports. | Maintainer interpretation and public boundaries for fixture-local evidence. | Broad external corpus coverage, SuiteSparse parity, and corpus completeness. |
| QR | Named fixture, rank/nullity or residual semantics, tolerance policy, output semantics, and proof owner. | Focused QR test or oracle/consistency check that runs in maintained gates. | Solver-selection and algorithm wording limited to the selected QR behavior. | General QR superiority, raw basis equality, minimum-norm guarantees outside tested fixtures, and external-library parity. |
| Partial-SVD | Selected edge class, singular value/vector/subspace comparison policy, convergence-budget semantics, tolerance policy, and fixture metadata. | Focused partial-SVD test and any selected report/oracle command. | SVD docs distinguish full SVD, partial SVD, convergence, and parity non-claims. | Broad ARPACK/SciPy parity, global ordering guarantees, sparse-output claims, and unbounded convergence claims. |
| Report/freshness | Shared metadata fields, source commit, generator command, support tier, row meaning, skip/defer reason, and freshness status. | Report generation or stale-report scanner command for maintained report families. | Maintainer guide and report docs state interpretation limits. | Release proof, correctness proof, coverage completeness, and portable performance proof. |
| Runtime/backend | Runtime precedence rule, backend-state vocabulary, fixture/metric semantics, runtime budget, and variance policy. | Focused sentinel or benchmark command with local-platform metadata. | Benchmark/runtime docs keep local-only and backend-availability boundaries. | Portable speedups, backend parity, scalability, and memory-use superiority. |
| Package/ABI | Product decision, build/install/export metadata, unsupported-artifact checks, downstream consumer proof, and support tier. | Make/CMake/pkg-config/install/downstream proof appropriate to the selected package path. | README, INSTALL, CMake/package docs, and maintainer docs agree on static-first or ABI status. | Shared ABI, package-manager support, dynamic loader behavior, and cross-platform install parity. |
| Platform | Hosted-runner evidence, expected test/package counts, failure semantics, support-tier wording, and owner for failures. | Reviewed or supplemental CI lane identified by platform and generator. | Support matrix explains reviewed, supplemental, and staged scopes. | Platform parity, reviewed status, or test promotion not backed by hosted evidence. |
| Adoption | Earned workflow behavior, maintained examples, docs links, support-tier alignment, and no unsupported claim expansion. | Link/path validation plus relevant example, install, or docs proof. | README, tutorial, cookbook, solver-selection, INSTALL, and maintainer guide agree. | New solver, package, platform, performance, or state-of-the-art claims from wording alone. |

## Feasibility Against Epic 12 Budget

Sprints 138-146 remain after Sprint 137. Day 7 should allocate one complete
closure target per implementation sprint unless two targets share the same
proof owner, fixtures, validation command, and documentation surface.

Budget guardrails:

- Keep Sprint 138 focused on the maintained corpus/oracle contract because QR,
  partial-SVD, report, and stale-report claims depend on row semantics.
- Keep Sprint 139 to one priority QR residual with focused tests, docs, and
  residual disposition.
- Keep Sprint 140 to one priority partial-SVD residual with focused tests,
  comparison semantics, docs, and residual disposition.
- Keep report normalization to row-meaning-preserving report families and
  defer any family that cannot expose honest metadata inside one sprint.
- Keep runtime/backend governance to a bounded precedence or sentinel scope.
- Choose either package/ABI implementation or static-first deferral cleanup;
  do not select a full shared ABI plus package-manager distribution unless the
  closure requirements are already concrete.
- Promote one platform lane completely before attempting broad platform parity.
- Delay adoption simplification until evidence-bearing changes have landed.

## Day 7 Inputs

Day 7 should produce:

- A score for each active candidate family from the Day 5 owner map.
- A selected target for each Sprint 138-146 implementation slot.
- A rejection or deferral reason for every active candidate not selected.
- A dependency order showing how corpus/oracle, solver, report, runtime,
  package, platform, and adoption work build on each other.
- Claim-boundary notes that preserve unsupported state-of-the-art, parity,
  package, ABI, platform, and performance non-claims.

## Day 6 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Gap selection can be justified from written criteria. | Complete | Scoring model, score interpretation, pre-screen rules, and Day 7 inputs define how candidates are selected, deferred, or rejected. |
| The rubric favors complete closure over broad partial work. | Complete | Complete-closure dimensions, complete-closure feasibility scoring, anti-goals, and budget guardrails require implementation, validation, docs, and residual disposition together. |
| Unsupported state-of-the-art expansion is blocked by explicit gates. | Complete | Claim gate matrix and anti-goals block broad state-of-the-art, parity, performance, ABI, package, and platform claims until evidence exists. |
