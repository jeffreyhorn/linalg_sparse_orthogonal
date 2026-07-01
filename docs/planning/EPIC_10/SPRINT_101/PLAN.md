# Sprint 101 Plan: Compressed-First Product Model & Storage Front Door

**Sprint Duration:** 14 days
**Goal:** Make compressed CSR/CSC workflows the unmistakable product center
while preserving mutable matrix-shell compatibility as a supported secondary
path. This sprint implements the Sprint 101 section of
`docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 101 begins from:
- the Sprint 100 claim map and evidence templates
- `docs/planning/EPIC_10/SPRINT_100/artifacts/day6-state-of-the-art-target.md`
- `docs/planning/EPIC_10/SPRINT_100/artifacts/day8-claim-dependency-model.md`
- `docs/planning/EPIC_10/SPRINT_100/artifacts/day12-public-claim-audit.md`
- `docs/planning/EPIC_10/SPRINT_100/artifacts/day13-sprint100-handoff-package.md`
- `docs/planning/EPIC_10/SPRINT_100/artifacts/day14-closeout-and-validation.md`

The strongest Sprint 101 pressure is to improve the product model without
pretending the mutable linked-list shell disappears. The sprint must:
- audit current construction, import/export, mutation, publication, and solver
  entry paths before changing APIs
- define a bounded CSR/CSC-first front-door design
- land implementation only after ownership and compatibility rules are clear
- prove lifecycle, ownership, error handling, and solver entry behavior
- update public docs and examples to describe mutable matrix-shell use as
  supported compatibility rather than the only product center
- preserve Sprint 100 non-claims around full shell replacement, broad
  state-of-the-art status, and unearned external-comparison claims

**End State:** Sprint 101 leaves behind:
- a compressed-first storage/workflow audit
- a bounded CSR/CSC front-door design
- implementation evidence for selected constructor/import improvements
- lifecycle and ownership tests for compressed-first paths
- compatibility wording for mutable matrix-shell users
- updated docs/examples where needed
- validation artifacts and Sprint 102 handoff criteria

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 101 project-plan estimate.

---

## Day 1: Sprint 101 Scope & Baseline Setup

**Title:** Scope Baseline
**Theme:** Convert the Sprint 101 project-plan section and Sprint 100 handoff
requirements into one bounded implementation package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 101 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read the Sprint 100 handoff, claim model, public claim audit, and
   closeout requirements.
3. Inventory required Sprint 101 workstreams:
   - storage surface audit
   - compressed-first API design
   - constructor/import implementation
   - lifecycle and ownership clarification
   - mutable matrix-shell compatibility documentation
   - regression proof
   - validation and Sprint 102 handoff
4. Create Sprint 101 working notes and an artifacts directory.
5. Record validation expectations for docs-only, code-touch, test-touch, and
   example-touch days.

### Deliverables
- Sprint 101 workstream inventory
- working-notes baseline
- initial artifacts directory structure
- validation expectation list

### Completion Criteria
- Sprint 101 work is bounded before audit or implementation begins
- every project-plan item has day-level ownership
- Sprint 100 claim boundaries are visible in working notes

---

## Day 2: Public Storage Surface Audit

**Title:** Storage Audit
**Theme:** Audit public construction, compressed import/export, mutation, and
publication paths for linked-list-first product costs
**Time estimate:** 12 hours

### Tasks
1. Inventory public matrix creation, insert/remove, copy, CSR/CSC import, and
   CSR/CSC export APIs.
2. Trace current README, tutorial, examples, and public-header guidance for
   linked-list-first versus compressed-first entry points.
3. Identify where compressed-input callers still pay avoidable explanation,
   conversion, mutation, or ownership cost.
4. Mark each surface as already compressed-first, compatibility-shell, unclear,
   or candidate for refinement.
5. Write the public storage surface audit artifact.

### Deliverables
- construction/import/export surface map
- linked-list-first cost table
- compressed-first candidate list
- compatibility-shell preservation notes

### Completion Criteria
- all public storage entry points have a current classification
- candidate improvements are ranked by user value and compatibility risk
- no implementation starts before the audit is recorded

---

## Day 3: Solver Entry Path Audit

**Title:** Solver Entry Audit
**Theme:** Audit direct, iterative, eigensolver, SVD, and analysis entry paths
for compressed-first adoption costs
**Time estimate:** 12 hours

### Tasks
1. Inventory solver entry points that accept `SparseMatrix`, CSR, CSC, or
   analysis/factor objects.
2. Trace how CSR/CSC data reaches LU, Cholesky, LDLT, QR, SVD, iterative, and
   eigensolver paths.
3. Identify solver families where compressed-first construction already works
   versus where callers still need linked-list ceremony.
4. Identify ownership, mutation, factor-state, and error-handling ambiguities.
5. Write the solver entry path audit artifact.

### Deliverables
- solver entry path map
- compressed-first readiness table by family
- ownership and mutation ambiguity list
- implementation candidate ranking

### Completion Criteria
- solver-path risks are separated from storage-constructor risks
- Sprint 102 direct-solver oracle work has clear dependencies from Sprint 101
- no broad solver-parity claim is introduced

---

## Day 4: Compressed-First API Design

**Title:** API Design
**Theme:** Define the bounded CSR/CSC-first API additions or refinements before
implementation
**Time estimate:** 12 hours

### Tasks
1. Reconcile Day 2 and Day 3 audit candidates.
2. Select the highest-value API refinements that fit Sprint 101.
3. Define ownership, lifetime, copy/adopt, and error semantics for each chosen
   refinement.
4. Define compatibility behavior for existing mutable matrix-shell callers.
5. Write the compressed-first API design artifact.

### Deliverables
- selected API refinement list
- ownership and lifetime contract
- compatibility behavior table
- non-goal list for unselected API ideas

### Completion Criteria
- selected implementation scope fits the sprint
- each chosen API behavior has validation requirements
- mutable shell compatibility remains explicit

---

## Day 5: Implementation Boundary Freeze

**Title:** Boundary Freeze
**Theme:** Freeze the implementation batch, test plan, docs plan, and risk
controls before code changes
**Time estimate:** 12 hours

### Tasks
1. Convert the Day 4 design into a specific file-level implementation plan.
2. Identify touched source, header, test, example, and documentation owners.
3. Define focused validation commands for each touched family.
4. Define full validation requirements for any `.c` or `.h` change.
5. Write the implementation boundary artifact.

### Deliverables
- file-level implementation plan
- focused validation plan
- compatibility and rollback notes
- code-touch quality gate list

### Completion Criteria
- implementation starts from an approved boundary
- code, test, docs, and examples have clear ownership
- required quality checks are known before edits begin

---

## Day 6: Constructor and Import Batch 1

**Title:** Import Batch 1
**Theme:** Land the first bounded compressed-first constructor/import
implementation batch
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-priority constructor or import refinement from the
   Day 5 boundary.
2. Preserve existing public API compatibility.
3. Add or update focused tests for success, bad input, and ownership behavior.
4. Update public header comments only where the changed behavior requires it.
5. Record implementation evidence and focused validation results.

### Deliverables
- constructor/import implementation batch
- focused regression tests
- updated API-local comments if needed
- Day 6 implementation evidence artifact

### Completion Criteria
- selected compressed-first behavior is implemented
- compatibility-shell callers remain supported
- required focused checks pass before continuing

---

## Day 7: Post-Batch Audit & Rerank

**Title:** Post-Batch Audit
**Theme:** Re-audit the landed constructor/import batch and rerank remaining
Sprint 101 work
**Time estimate:** 12 hours

### Tasks
1. Review the Day 6 implementation against the Day 4 design and Day 5
   boundary.
2. Check for unexpected ownership, lifetime, mutation, or documentation drift.
3. Rerank remaining constructor/import candidates.
4. Decide whether a second implementation batch is justified.
5. Write the post-batch audit and rerank artifact.

### Deliverables
- post-implementation audit
- remaining candidate rerank
- second-batch decision
- validation status notes

### Completion Criteria
- Day 6 work is reconciled before new edits
- remaining scope is either selected or deferred explicitly
- no candidate claim is promoted without evidence

---

## Day 8: Lifecycle and Ownership Design

**Title:** Lifecycle Design
**Theme:** Clarify ownership, lifetime, mutation, and repeated-run rules for
compressed matrices and solver handles
**Time estimate:** 12 hours

### Tasks
1. Audit current ownership and lifecycle wording for CSR/CSC import/export,
   matrix shells, analysis objects, factor objects, and solver handles.
2. Identify where compressed-first paths need clearer copy/adopt/free rules.
3. Define repeated-run direct, iterative, and eigensolver handle implications.
4. Define mutation and factored-state rules for compressed-first callers.
5. Write the lifecycle and ownership design artifact.

### Deliverables
- ownership/lifetime rule table
- repeated-run lifecycle clarification
- mutation and factored-state rule map
- test and docs follow-through queue

### Completion Criteria
- compressed-first ownership rules are explicit enough to test
- repeated-run implications are clear without broad solver redesign
- compatibility-shell wording remains accurate

---

## Day 9: Lifecycle and Ownership Batch

**Title:** Lifecycle Batch
**Theme:** Land focused lifecycle, ownership, and error-handling follow-through
for compressed-first paths
**Time estimate:** 12 hours

### Tasks
1. Implement selected lifecycle or ownership clarifications from Day 8.
2. Add or update focused tests for ownership, invalid input, no-op, and
   mutation/factored-state behavior.
3. Update public headers or docs where changed behavior needs call-site
   clarity.
4. Run focused validation for touched families.
5. Record the lifecycle batch evidence artifact.

### Deliverables
- lifecycle/ownership implementation or docs batch
- focused ownership and error tests
- updated call-site wording if needed
- validation evidence

### Completion Criteria
- compressed-first ownership behavior is test-backed
- mutable matrix-shell compatibility remains passing
- validation requirements are satisfied before docs widening

---

## Day 10: Compatibility Path Documentation Design

**Title:** Compatibility Design
**Theme:** Design public wording that presents mutable matrix-shell workflows
as supported compatibility rather than the only product center
**Time estimate:** 12 hours

### Tasks
1. Review README, tutorial, examples, and relevant headers after Day 6-9 work.
2. Identify wording that still makes linked-list mutation feel like the only
   first-class route.
3. Draft the compressed-first workflow narrative and compatibility-shell
   boundary.
4. Decide which public docs and examples need edits in Sprint 101.
5. Write the compatibility documentation design artifact.

### Deliverables
- public wording audit
- compressed-first narrative draft
- compatibility-shell wording rules
- docs/example edit list

### Completion Criteria
- mutable matrix-shell support is preserved and demoted only in product
  narrative
- docs edits are scoped to earned implementation evidence
- no broad full-shell-replacement claim appears

---

## Day 11: Docs and Examples Follow-Through

**Title:** Docs Batch
**Theme:** Update public docs and examples to reflect the earned
compressed-first product model
**Time estimate:** 12 hours

### Tasks
1. Update selected README, tutorial, example, or header wording from Day 10.
2. Add or adjust examples only where they demonstrate implemented
   compressed-first behavior.
3. Preserve one-shot and mutable-shell examples as supported compatibility
   paths.
4. Run docs/example validation appropriate to touched files.
5. Write the docs/examples follow-through artifact.

### Deliverables
- updated compressed-first public wording
- updated examples if justified
- compatibility-shell wording preserved
- validation notes

### Completion Criteria
- public docs match implemented behavior
- examples compile or have an explicit validation path
- Sprint 100 claim boundaries are preserved

---

## Day 12: Regression Proof Expansion

**Title:** Regression Proof
**Theme:** Complete focused regression proof for compressed-first construction,
ownership, error handling, and solver entry behavior
**Time estimate:** 12 hours

### Tasks
1. Review tests added or touched on Days 6 and 9.
2. Add any missing focused tests for constructor/import, solver entry,
   ownership, bad input, and compatibility behavior.
3. Confirm Make/CMake test registration implications for any new tests.
4. Run focused tests and required quality checks for touched code.
5. Write the regression proof artifact.

### Deliverables
- focused compressed-first regression tests
- Make/CMake registration notes if applicable
- focused validation results
- remaining test gap list

### Completion Criteria
- selected compressed-first behavior has regression coverage
- no test-count or source-list drift is introduced
- required quality checks pass or the sprint stops for clarification

---

## Day 13: Full Validation & Product-Model Reconciliation

**Title:** Validation
**Theme:** Run required validation and reconcile implementation, docs, tests,
and claim wording before closeout
**Time estimate:** 12 hours

### Tasks
1. Run the full required quality chain if any `.c` or `.h` files changed:
   `make format && make lint && make test`.
2. Run any focused example, docs, or CMake checks required by touched surfaces.
3. Reconcile public wording against the Day 12 public claim audit and Sprint
   100 handoff.
4. Record final earned, deferred, and non-claim states for Sprint 101.
5. Write the validation and reconciliation artifact.

### Deliverables
- full validation results
- public wording reconciliation notes
- earned/deferred/non-claim state table
- Sprint 102 dependency notes

### Completion Criteria
- all required checks pass before closeout
- compressed-first claims are tied to implementation and tests
- no Sprint 102 dependency is left implicit

---

## Day 14: Sprint 101 Closeout & Handoff

**Title:** Closeout
**Theme:** Close Sprint 101 with a validated compressed-first product-model
baseline and clear handoff to Sprint 102
**Time estimate:** 12 hours

### Tasks
1. Confirm every Sprint 101 project-plan item has a deliverable.
2. Write the Sprint 101 closeout artifact and artifact index.
3. Record Sprint 102 direct-solver oracle prerequisites created or deferred by
   Sprint 101.
4. Record any residual compatibility-shell, docs, example, or test follow-up.
5. Prepare retrospective inputs and final validation notes.

### Deliverables
- Sprint 101 closeout artifact
- complete artifact index
- Sprint 102 handoff requirements
- retrospective input and residual queue

### Completion Criteria
- Sprint 101 artifacts are complete and internally consistent
- validation requirements are satisfied or explicitly blocked
- Sprint 102 can start from stable compressed-first ownership and lifecycle
  rules
