# Sprint 111 Plan: API Usability, Documentation & Example Coherence

**Sprint Duration:** 14 days
**Goal:** Make the project easier to adopt by giving users a concise
compressed-first path, solver-selection guidance, Matrix Market behavior
documentation, and examples that match the actual product contracts. This
sprint implements the Sprint 111 section of
`docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 111 begins from:
- the Sprint 101 compressed-first workflow and storage front-door decisions
- Sprint 102-105 solver, comparison, benchmark, and scalability evidence
- Sprint 110 Matrix builder and Matrix Market I/O split validation
- Sprint 110 eigensolver behavior-owner and proof-owner follow-through
- existing public headers, README material, examples, benchmark notes, and
  maintainer proof artifacts that now need clearer audience boundaries

The sprint must:
- audit adoption surfaces from a first-time user perspective
- create solver-selection and matrix-format guidance grounded in actual
  supported behavior
- update examples around CSR/CSC-first workflows
- document Matrix Market ownership, duplicate-entry, zero/default, pattern,
  symmetric expansion, errno, and runtime behavior without creating a public
  Matrix I/O module or builder API claim
- clarify benchmark interpretation without overclaiming portability
- move maintainer-only proof language away from user-facing surfaces where
  practical
- close with doc/example validation and a residual handoff

**End State:** Sprint 111 leaves behind:
- a user-journey audit and adoption-surface cleanup queue
- a concise solver-selection and matrix-format guide
- compressed-first examples for common workflows
- tighter Matrix Market behavior, ownership, error, runtime, and benchmark
  documentation
- cleaner separation between user documentation and maintainer proof language
- validation evidence for touched docs, examples, and any code-adjacent samples

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 111 project-plan estimate.

---

## Day 1: Sprint 111 Scope & User Journey Inventory

**Title:** Adoption Inventory
**Theme:** Establish the first-time user path and documentation surface map
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 111 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Inventory README, tutorial, examples, install docs, benchmark docs, public
   headers, and generated/reference documentation.
3. Identify first-time user entry points for build, matrix creation, Matrix
   Market loading, direct solve, iterative solve, eigensolve, SVD, reordering,
   and benchmarking.
4. Mark maintainer-only proof artifacts that should not anchor user adoption.
5. Create Sprint 111 working notes and artifacts directory.
6. Write the user-journey inventory artifact.

### Deliverables
- adoption-surface inventory
- first-time user path map
- maintainer-only proof surface list
- Sprint 111 working-notes baseline
- Day 1 user-journey inventory artifact

### Completion Criteria
- every Sprint 111 item has an initial documentation or example owner
- all user-facing entry points are mapped
- maintainer-only surfaces are distinguishable from adoption surfaces
- downstream days can proceed without rediscovering the documentation layout

---

## Day 2: Documentation Gap Audit

**Title:** Gap Audit
**Theme:** Compare current documentation against actual product contracts
**Time estimate:** 12 hours

### Tasks
1. Review public headers and examples for API names, ownership rules, option
   defaults, error reporting, and lifecycle expectations.
2. Compare README and tutorial claims against current source and tests.
3. Identify stale solver, storage-format, Matrix Market, benchmark, and
   workflow guidance.
4. Classify gaps as user-blocking, confusing, maintainer-only, or future-work
   material.
5. Identify documentation changes that require example or smoke validation.
6. Write the documentation gap audit artifact.

### Deliverables
- user-facing gap list
- stale or risky claim inventory
- validation-needs list for example changes
- audience classification notes
- Day 2 documentation gap audit artifact

### Completion Criteria
- documentation risks are tied to concrete files or sections
- no planned guide claims exceed implemented behavior
- example-impacting changes have validation expectations
- guide-writing days have a prioritized source of truth

---

## Day 3: Solver Selection Guide Outline

**Title:** Solver Guide Outline
**Theme:** Define the guide structure and supported decision tree
**Time estimate:** 12 hours

### Tasks
1. Draft the guide audience, scope, and non-goals.
2. Define matrix-format guidance for CSR, CSC, dense-adjacent workflows, and
   Matrix Market input.
3. Outline direct solver choices, reorder/fill considerations, decomposition
   reuse, and fallback paths.
4. Outline iterative solver selection, preconditioner expectations, convergence
   caveats, and diagnostics.
5. Outline eigensolver and SVD guidance without overclaiming state-of-the-art
   parity beyond available evidence.
6. Write the solver guide outline artifact.

### Deliverables
- solver-selection guide outline
- matrix-format decision tree
- direct and iterative solver decision notes
- eigen/SVD guidance boundaries
- Day 3 guide outline artifact

### Completion Criteria
- the guide structure covers all project-plan solver-selection requirements
- claims are bounded by existing public APIs and validation evidence
- compressed-first workflows are the default path
- unknown or future behavior is excluded from user guidance

---

## Day 4: Solver Selection Guide Draft

**Title:** Solver Guide Draft
**Theme:** Turn the decision tree into concise user-facing documentation
**Time estimate:** 12 hours

### Tasks
1. Create or update the solver-selection guide in the appropriate docs
   location.
2. Document CSR/CSC-first matrix preparation and when conversion is expected.
3. Document direct solve, iterative solve, eigensolve, SVD, reorder/fill, and
   decomposition reuse choices.
4. Include short, stable references to examples and relevant public headers.
5. Avoid maintainer proof language and unsupported performance claims.
6. Record guide decisions in working notes.

### Deliverables
- initial solver-selection guide
- compressed-first matrix-format guidance
- links or references to planned examples
- unsupported-claim exclusion notes

### Completion Criteria
- guide text is concise enough for adoption use
- each solver family has clear selection guidance
- every example reference points to an existing or planned Sprint 111 example
- documentation does not claim public Matrix I/O or builder APIs

---

## Day 5: Compressed-First Example Audit

**Title:** Example Audit
**Theme:** Select examples that should demonstrate common CSR/CSC workflows
**Time estimate:** 12 hours

### Tasks
1. Inventory existing examples, sample snippets, tests used as examples, and
   README walkthroughs.
2. Identify examples that start from less-preferred or unclear matrix setup
   paths.
3. Select minimal robust workflows for CSR construction, CSC direct solve,
   Matrix Market load, iterative solve, eigensolve, SVD, and reorder/fill use.
4. Define validation commands for examples that compile or execute locally.
5. Decide which examples should be updated versus newly added.
6. Write the compressed-first example audit artifact.

### Deliverables
- example inventory
- update/add decision list
- selected compressed-first workflows
- example validation command list
- Day 5 example audit artifact

### Completion Criteria
- each planned example supports a real user workflow
- examples are scoped to public API behavior
- validation expectations are clear before edits begin
- examples avoid maintainer-only proof scaffolding

---

## Day 6: CSR/CSC Construction Examples

**Title:** Construction Examples
**Theme:** Add or update minimal examples for compressed matrix creation
**Time estimate:** 12 hours

### Tasks
1. Update or add a minimal CSR construction example.
2. Update or add a minimal CSC construction or conversion example if supported
   by public APIs.
3. Show ownership and cleanup patterns explicitly.
4. Keep examples small enough for users to copy and adapt.
5. Run focused example formatting or compile checks for touched samples.
6. Update working notes with example decisions and residuals.

### Deliverables
- CSR-first construction example
- CSC-oriented construction or conversion example
- ownership and cleanup notes in examples
- focused validation evidence

### Completion Criteria
- examples compile or are otherwise validated according to local conventions
- memory ownership is clear
- examples do not require private headers
- examples match solver-guide recommendations

---

## Day 7: Solver Workflow Examples

**Title:** Solver Examples
**Theme:** Demonstrate common direct, iterative, and reuse workflows
**Time estimate:** 12 hours

### Tasks
1. Update or add a direct-solve example using the documented compressed-first
   path.
2. Update or add an iterative-solve example with convergence and cleanup notes.
3. Include a decomposition-reuse or reorder/fill example if a concise public
   workflow exists.
4. Avoid exposing test-only helper patterns as user API.
5. Run focused validation for changed examples.
6. Cross-check examples against the solver-selection guide.

### Deliverables
- direct-solve example
- iterative-solve example
- optional reuse or reorder/fill example
- validation notes

### Completion Criteria
- solver examples follow public APIs only
- guide text and examples agree on matrix format and lifecycle
- validation passes for changed samples
- examples remain minimal and copyable

---

## Day 8: Eigen/SVD and Matrix Market Examples

**Title:** Advanced Examples
**Theme:** Cover higher-level workflows without broadening support claims
**Time estimate:** 12 hours

### Tasks
1. Update or add an eigensolver example with bounded expectations and cleanup.
2. Update or add an SVD example if there is a stable concise user workflow.
3. Update or add a Matrix Market load/use example that does not imply a public
   Matrix I/O module.
4. Clarify option defaults and error handling where examples touch them.
5. Run focused validation for changed examples.
6. Record any advanced workflow that should stay documented only as future work.

### Deliverables
- eigensolver example
- SVD example or explicit no-example rationale
- Matrix Market load/use example
- advanced-workflow validation notes

### Completion Criteria
- advanced examples do not overclaim portability or performance
- Matrix Market example uses only public entry points
- error and cleanup behavior is visible
- validation passes or a specific no-example rationale is documented

---

## Day 9: Matrix Market Behavior Documentation

**Title:** Matrix Market Docs
**Theme:** Document exact Matrix Market behavior and ownership boundaries
**Time estimate:** 12 hours

### Tasks
1. Document ownership and cleanup responsibilities for loaded matrices.
2. Document zero/default option behavior without implying a public builder API.
3. Document duplicate-entry last-write behavior.
4. Document final-zero elision, pattern handling, and symmetric expansion.
5. Document errno and runtime behavior at the level supported by public
   contracts.
6. Record no-public-Matrix-I/O-module wording decisions.

### Deliverables
- Matrix Market behavior documentation
- ownership and cleanup wording
- duplicate, zero, pattern, symmetric, errno, and runtime notes
- public/private boundary notes

### Completion Criteria
- Matrix Market behavior matches Sprint 110 implementation evidence
- docs avoid public builder or Matrix I/O module claims
- behavior details are discoverable from adoption surfaces
- wording is consistent with public headers and examples

---

## Day 10: Header and Tutorial Coherence Pass

**Title:** Header Coherence
**Theme:** Align public headers, tutorials, and guides around the same contracts
**Time estimate:** 12 hours

### Tasks
1. Review public header comments touched by solver, Matrix Market, or ownership
   guidance.
2. Update tutorial or README references to point users toward the new guide and
   examples.
3. Remove or relocate maintainer-only proof language from adoption surfaces
   where practical.
4. Check that public wording does not expose private source-owner details.
5. Run focused doc lint or formatting checks available in the repo.
6. Update working notes with any remaining audience-boundary debt.

### Deliverables
- public header wording updates, if needed
- README/tutorial coherence updates
- maintainer/user split cleanup notes
- focused doc-check evidence

### Completion Criteria
- headers, tutorials, guide, and examples use consistent terminology
- maintainer proof language no longer distracts from adoption surfaces
- private implementation owners remain private in public docs
- doc checks pass for touched documentation

---

## Day 11: Benchmark Interpretation Documentation

**Title:** Benchmark Docs
**Theme:** Explain local benchmark and comparison artifacts responsibly
**Time estimate:** 12 hours

### Tasks
1. Inventory benchmark scripts, artifacts, comparison docs, and existing claims.
2. Document how to read local benchmark outputs and comparison artifacts.
3. Clarify machine, compiler, matrix corpus, dependency, and configuration
   sensitivity.
4. Tie benchmark interpretation back to solver-selection guidance.
5. Remove or qualify portability and competitive claims that exceed evidence.
6. Write the benchmark interpretation artifact.

### Deliverables
- benchmark interpretation documentation
- comparison-artifact reading notes
- portability and configuration caveats
- Day 11 benchmark documentation artifact

### Completion Criteria
- users can understand what benchmark outputs do and do not prove
- comparison claims are evidence-bounded
- docs do not imply universal performance results
- solver guide references benchmark material accurately

---

## Day 12: Maintainer/User Surface Split

**Title:** Audience Split
**Theme:** Move proof-owner and planning language away from adoption surfaces
**Time estimate:** 12 hours

### Tasks
1. Identify remaining public docs that lead with maintainer proof language.
2. Move detailed proof-owner, source-boundary, and deferred-debt wording to
   planning or maintainer-oriented locations where practical.
3. Keep user-facing docs focused on supported workflows, constraints, and
   examples.
4. Preserve traceability to evidence without requiring users to read sprint
   artifacts.
5. Run focused documentation checks.
6. Update working notes with before/after audience boundaries.

### Deliverables
- cleaned adoption surfaces
- maintainer-oriented proof references
- evidence traceability notes
- focused doc-check evidence

### Completion Criteria
- user docs read as adoption material first
- maintainer proof records remain available in planning artifacts
- no evidence is deleted without an appropriate replacement or reference
- docs remain coherent after movement or wording changes

---

## Day 13: Integrated Documentation and Example Validation

**Title:** Integrated Validation
**Theme:** Validate the Sprint 111 documentation and example set as one product
surface
**Time estimate:** 12 hours

### Tasks
1. Run all applicable documentation, formatting, example, and smoke checks for
   touched files.
2. Check links or relative references introduced by the new guide and examples.
3. Verify examples and guide text agree on public API names, ownership, option
   defaults, and cleanup.
4. Confirm Matrix Market docs still avoid public builder or Matrix I/O module
   claims.
5. Capture validation output and unresolved risks.
6. Write the integrated validation artifact.

### Deliverables
- doc/example validation evidence
- link/reference check notes
- guide/example consistency checklist
- Matrix Market claim no-drift notes
- Day 13 validation artifact

### Completion Criteria
- all applicable checks pass
- no broken references are introduced
- examples, guide, public headers, and README agree
- residual risks are documented before closeout

---

## Day 14: Sprint 111 Closeout and Handoff

**Title:** Closeout
**Theme:** Finish Sprint 111 with clear artifacts, residuals, and downstream
handoff
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 111 artifacts, working notes, docs, and examples.
2. Confirm each project-plan item has evidence of completion or a documented
   residual.
3. Summarize changed adoption surfaces and user-facing workflow improvements.
4. Record residual deferred debt for Sprint 112 and beyond if needed.
5. Run final applicable documentation and example checks.
6. Prepare the sprint closeout artifact.

### Deliverables
- completed Sprint 111 artifact set
- adoption-surface change summary
- residual deferred-debt list
- final validation evidence
- Sprint 111 closeout artifact

### Completion Criteria
- all seven Sprint 111 project-plan items are closed or explicitly deferred
- residuals are dependency-ordered and non-duplicative
- final checks pass
- Sprint 112 handoff is clear and actionable
