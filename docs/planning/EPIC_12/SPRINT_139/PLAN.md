# Sprint 139 Plan: QR Priority Residual Closure

**Sprint Duration:** 14 days
**Goal:** Completely close the selected QR residual with corpus-backed
fixtures, oracle evidence, focused proof ownership, and updated claim wording.
This sprint implements the Sprint 139 section of
`docs/planning/EPIC_12/PROJECT_PLAN.md`.

**Starting Point:** Sprint 139 begins from:
- Sprint 138 maintained corpus architecture, fixture manifests, expected-result
  rows, oracle/report command, and skip/defer semantics
- Sprint 137 evidence contracts, claim boundaries, and quality-surface map
- current QR tests, examples, solver documentation, and residual queues
- the Sprint 138 closeout handoff for QR fixture expansion

The sprint must:
- re-audit QR residuals and select one bounded priority residual to close
- add deterministic corpus fixtures for the selected QR residual family
- add oracle comparison rows with explicit tolerance, skip, and failure
  semantics
- create or extract a focused QR proof owner without weakening existing QR
  coverage
- update solver, algorithm, cookbook, and maintainer documentation with earned
  QR wording and preserved non-claims
- run focused QR/corpus validation and broader quality gates when code changes
  require them
- publish closed QR claims, remaining non-claims, and Sprint 140 partial-SVD
  handoff requirements

**End State:** Sprint 139 leaves behind:
- closed priority QR residual
- QR corpus fixture batch
- QR oracle comparison row(s)
- focused QR proof owner
- updated QR docs and claim boundaries
- validation evidence for touched surfaces
- Sprint 140 partial-SVD handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 139 project-plan estimate.

---

## Day 1: QR Residual Intake

**Title:** Residual Intake
**Theme:** Establish Sprint 139 scope, inherited QR evidence, and closure
criteria before choosing the priority residual
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 139 section of
   `docs/planning/EPIC_12/PROJECT_PLAN.md`.
2. Review Sprint 137 evidence contracts and Sprint 138 corpus/oracle handoff
   artifacts.
3. Inventory current QR tests, QR examples, solver-selection docs, algorithm
   docs, and corpus rows.
4. Create Sprint 139 working notes and artifact directory structure.
5. Map Sprint 139 Items 1-7 to day-level owners.
6. Record initial claim boundaries, non-claims, and validation expectations.

### Deliverables
- Sprint 139 working-notes baseline
- artifact directory structure
- inherited QR evidence inventory
- item-to-day owner map
- initial closure criteria and non-claim register

### Completion Criteria
- every Sprint 139 project-plan item has a day-level owner
- inherited QR/corpus evidence is visible before fixture or code changes begin
- closure criteria distinguish earned QR evidence from remaining non-claims

---

## Day 2: QR Residual Reaudit

**Title:** Residual Reaudit
**Theme:** Re-rank open QR residuals across correctness, corpus, claim, and
maintenance risk
**Time estimate:** 12 hours

### Tasks
1. Review rank-deficient, rectangular, least-squares, minimum-norm, nullspace,
   COLAMD, and SuiteSparse/corpus QR behavior.
2. Compare each residual against current tests, docs, examples, and corpus
   coverage.
3. Score each candidate by user-facing risk, feasibility, validation cost, and
   ability to close fully in one sprint.
4. Identify residuals that require external data, broad SuiteSparse parity, or
   partial-SVD follow-up and keep them out of scope.
5. Draft the residual ranking artifact.
6. Record the selected closure candidate and backup candidate.

### Deliverables
- QR residual ranking table
- evidence and gap map for each candidate
- selected priority residual
- out-of-scope residual list
- backup candidate notes

### Completion Criteria
- the selected residual can be closed without broadening unsupported claims
- lower-priority residuals have explicit defer reasons
- fixture and oracle design can proceed from a single bounded QR behavior

---

## Day 3: Closure Design

**Title:** Closure Design
**Theme:** Define the exact QR behavior, fixture class, oracle semantics, and
proof-owner boundary for the selected residual
**Time estimate:** 12 hours

### Tasks
1. Define the selected residual's success, diagnostic failure, and tolerance
   boundary cases.
2. Choose deterministic fixture dimensions, sparsity patterns, rank/nullity
   properties, and right-hand-side policies.
3. Define expected results, comparison kinds, tolerance kinds, and failure
   classes.
4. Decide whether the proof owner is extracted from `tests/test_qr.c` or added
   as a focused helper/test lane.
5. Define source-list, CMake, and test-runner surfaces that may need updates.
6. Write the closure design artifact.

### Deliverables
- selected residual closure design
- fixture class definitions
- oracle comparison design
- proof-owner boundary
- touched-surface and validation map

### Completion Criteria
- fixtures and oracle rows have unambiguous expected semantics
- proof ownership is scoped before implementation begins
- validation requirements are known for docs-only, script, and C/H changes

---

## Day 4: Fixture Batch Design

**Title:** Fixture Design
**Theme:** Specify deterministic QR fixtures that cover success, failure, and
tolerance boundaries for the selected residual
**Time estimate:** 12 hours

### Tasks
1. Design deterministic generated or stored fixture rows for the selected QR
   residual family.
2. Define fixture keys, generator keys, metadata fields, claim scopes, and
   non-claims.
3. Define expected-result row IDs and short forms that remain unique.
4. Map each fixture to success, failure, or tolerance-boundary behavior.
5. Decide which fixtures become first-class source-controlled evidence and
   which remain staged or deferred.
6. Write the fixture batch design artifact.

### Deliverables
- QR fixture batch specification
- generator and fixture naming map
- expected-result row plan
- claim/non-claim table
- staged/deferred fixture notes

### Completion Criteria
- every planned fixture has a precise QR behavior and claim scope
- row IDs can be validated without ambiguity
- no fixture implies broad QR, SuiteSparse, or corpus completeness

---

## Day 5: Fixture Batch Implementation

**Title:** Fixture Batch
**Theme:** Add the deterministic QR fixture rows and generator metadata under
the maintained corpus layout
**Time estimate:** 12 hours

### Tasks
1. Add or update corpus fixture manifest rows for the selected QR fixtures.
2. Add or update generator manifest rows and deterministic generator support as
   needed.
3. Add expected-result rows for the selected fixture batch.
4. Update corpus README/schema notes when new row semantics are introduced.
5. Run corpus schema validation and TSV consistency checks.
6. Capture implementation notes and any fixture residuals.

### Deliverables
- source-controlled QR fixture rows
- generator metadata or stored-fixture metadata
- expected-result rows
- corpus documentation updates if needed
- fixture implementation validation notes

### Completion Criteria
- new fixture rows pass maintained schema validation
- expected-result rows preserve explicit non-claims
- generated or stored fixture metadata is reproducible and reviewable

---

## Day 6: Oracle Comparison Design

**Title:** Oracle Design
**Theme:** Define dense or external-reference comparison behavior for the QR
fixture batch
**Time estimate:** 12 hours

### Tasks
1. Choose the oracle reference approach for the selected QR residual.
2. Define exact observed values, residual computations, tolerance thresholds,
   and status interpretation.
3. Define skip/defer behavior for any optional or external comparison input.
4. Define oracle row provenance fields, command shape, and report-row
   freshness expectations.
5. Confirm failure classes for mismatch, unsupported comparison, stale report,
   and unavailable optional data.
6. Write the oracle comparison design artifact.

### Deliverables
- QR oracle comparison design
- tolerance and failure-class table
- optional-data skip/defer policy if needed
- report freshness requirements
- command ownership notes

### Completion Criteria
- comparison semantics are explicit before runner changes
- optional or external references cannot be counted as pass evidence when
  unavailable
- the oracle design supports the selected residual without broad claims

---

## Day 7: Oracle Comparison Implementation

**Title:** Oracle Lane
**Theme:** Implement the QR oracle comparison row(s) and report output for the
selected fixture batch
**Time estimate:** 12 hours

### Tasks
1. Update the corpus oracle runner or add a focused QR oracle path for the
   selected fixture batch.
2. Emit oracle rows with expected/observed values, tolerance semantics,
   support tier, provenance, and claim boundaries.
3. Emit report-index rows and skip/defer rows when applicable.
4. Ensure command defaults and paths remain robust from any working directory.
5. Run schema validation, oracle generation, and report metadata checks.
6. Write the oracle implementation artifact.

### Deliverables
- QR oracle comparison row(s)
- report-index rows for QR evidence
- updated oracle runner or focused QR oracle helper
- validation output summary
- oracle implementation artifact

### Completion Criteria
- oracle rows are mechanically parseable and reproducible
- passing rows reflect only the selected fixture-local QR behavior
- generated reports include freshness and provenance fields

---

## Day 8: Proof Owner Design

**Title:** Proof Owner
**Theme:** Design the focused QR test/helper ownership split without weakening
existing coverage
**Time estimate:** 12 hours

### Tasks
1. Review `tests/test_qr.c` and related QR helper surfaces for ownership
   boundaries.
2. Decide whether to extract existing checks or add a focused test/helper for
   the selected residual.
3. Define test names, fixture reuse policy, assertion semantics, and failure
   messages.
4. Map required source-list, CMake, Make, and CI/test-suite updates.
5. Confirm that existing QR coverage remains intact.
6. Write the proof-owner design artifact.

### Deliverables
- QR proof-owner design
- test/helper ownership map
- build-system touch plan
- retained-coverage checklist
- focused assertion plan

### Completion Criteria
- proof-owner scope is narrower than broad QR correctness
- existing QR tests are not weakened or silently bypassed
- build and source-list implications are explicit before code changes

---

## Day 9: Proof Owner Implementation

**Title:** Proof Lane
**Theme:** Implement the focused QR proof owner and connect it to maintained
test/build surfaces
**Time estimate:** 12 hours

### Tasks
1. Add or extract the focused QR test/helper for the selected residual.
2. Add fixture-driven assertions for success, failure, and tolerance-boundary
   behavior where applicable.
3. Update Make, CMake, source lists, or test manifests if new C files are
   introduced.
4. Preserve existing `tests/test_qr.c` behavior unless intentionally
   ownership-split with equivalent coverage.
5. Run focused QR tests and build-surface checks required by touched files.
6. Write the proof implementation artifact.

### Deliverables
- focused QR proof owner
- test/build integration updates
- retained coverage evidence
- focused test output summary
- proof implementation artifact

### Completion Criteria
- the selected QR residual has a focused proof owner
- test failures identify the selected behavior and fixture clearly
- existing QR coverage remains present or is explicitly transferred

---

## Day 10: Solver Documentation Update

**Title:** Solver Wording
**Theme:** Update QR-facing solver documentation with earned claims and
preserved non-claims
**Time estimate:** 12 hours

### Tasks
1. Review solver-selection, QR algorithm, cookbook, README, and maintainer
   guidance for QR wording.
2. Add earned wording for the selected residual only after evidence exists.
3. Update examples or cookbook notes that reference the selected QR behavior.
4. Preserve non-claims for broad QR correctness, SuiteSparse parity, raw-basis
   parity, corpus completeness, and unsupported fixture families.
5. Update links to corpus/oracle/report artifacts.
6. Write the solver documentation update artifact.

### Deliverables
- updated QR solver documentation
- updated cookbook or algorithm references if needed
- maintained claim/non-claim wording
- artifact link map
- documentation update notes

### Completion Criteria
- public wording matches earned evidence exactly
- unsupported QR behavior remains fenced by explicit non-claims
- documentation points to reproducible fixture/oracle evidence

---

## Day 11: Maintainer Guidance & Residual Queue

**Title:** Maintainer Guidance
**Theme:** Record how maintainers should regenerate QR evidence and interpret
remaining residuals
**Time estimate:** 12 hours

### Tasks
1. Update corpus, oracle, report, or QR maintainer notes with the new QR lane.
2. Document regeneration commands, expected outputs, and stale-report signals.
3. Record remaining QR residuals and why they are not closed in Sprint 139.
4. Identify which remaining residuals depend on partial-SVD Sprint 140 work.
5. Preserve support-tier and optional-data interpretation rules.
6. Write the maintainer guidance artifact.

### Deliverables
- QR maintainer guidance updates
- regeneration command notes
- remaining QR residual queue
- Sprint 140 dependency notes
- stale-report and support-tier interpretation notes

### Completion Criteria
- maintainers can regenerate and interpret the QR evidence without guessing
- remaining residuals are explicit and prioritized
- Sprint 140 dependencies are visible before closeout

---

## Day 12: Focused Validation

**Title:** Validation Pass
**Theme:** Execute focused QR, corpus, documentation, and build validation for
all touched surfaces
**Time estimate:** 12 hours

### Tasks
1. Run corpus schema validation and QR oracle/report generation checks.
2. Run focused QR tests and any newly introduced test target.
3. Run source-list, Make, CMake, or build parity checks required by code or
   test ownership changes.
4. Run documentation link, whitespace, TSV consistency, and generated-artifact
   hygiene checks.
5. If `.c` or `.h` files changed, run the required full quality gates:
   `make format && make lint && make test`.
6. Record command results, skips, failures, and rerun requirements.

### Deliverables
- focused validation command log
- QR test evidence
- corpus/oracle/report validation evidence
- docs and TSV hygiene evidence
- full quality-gate evidence when required

### Completion Criteria
- all required checks for touched surfaces pass
- any skipped supplemental check has an explicit reason and non-claim
- generated artifacts are not accidentally promoted without freshness context

---

## Day 13: Claim Closure & Handoff

**Title:** Claim Closure
**Theme:** Publish the closed QR claim, remaining non-claims, and Sprint 140
partial-SVD handoff requirements
**Time estimate:** 12 hours

### Tasks
1. Summarize the selected residual, evidence added, and claim now closed.
2. Summarize fixture, oracle, proof-owner, and documentation deliverables.
3. Publish remaining QR non-claims and residuals.
4. Define Sprint 140 partial-SVD handoff requirements and dependencies.
5. Confirm validation evidence supports the final wording.
6. Write the claim closure and handoff artifact.

### Deliverables
- closed QR claim summary
- remaining QR non-claim list
- Sprint 140 partial-SVD handoff
- validation-to-claim traceability table
- closeout readiness notes

### Completion Criteria
- the closed claim is traceable to fixture, oracle, proof, and validation
  evidence
- remaining non-claims are visible and not contradicted by docs
- Sprint 140 can start without rediscovering QR dependencies

---

## Day 14: Sprint 139 Closeout

**Title:** Closeout
**Theme:** Finalize Sprint 139 artifacts, validation summary, and residual
handoff package
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 139 artifacts for consistency with the project-plan
   items.
2. Confirm all deliverables have source, test, corpus, documentation, or
   residual evidence.
3. Re-run required final validation for touched surfaces or cite the latest
   passing command set.
4. Update working notes with final decisions, deferred work, and validation
   status.
5. Prepare the retrospective input summary for the sprint closeout request.
6. Write the final closeout validation summary artifact.

### Deliverables
- final Sprint 139 artifact inventory
- validation summary
- residual and deferred-work summary
- retrospective input notes
- Sprint 140 handoff confirmation

### Completion Criteria
- Sprint 139 Items 1-7 are complete or explicitly deferred with reasons
- validation status is current and tied to touched surfaces
- the branch is ready for retrospective creation and pull-request packaging
