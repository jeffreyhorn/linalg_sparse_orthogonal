# Sprint 106 Plan: Large-Source & Giant-Test Maintainability Phase 7

**Sprint Duration:** 14 days
**Goal:** Continue extracting the largest implementation and proof owners so
future solver, backend, and platform changes are smaller and safer. This
sprint implements the Sprint 106 section of
`docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 106 begins from:
- the Sprint 100 hotspot baseline, evidence contract, and validation templates
- Sprint 101 compressed-first storage and API front-door ownership notes
- Sprint 102 direct solver robustness and external oracle work
- Sprint 103 iterative, eigensolver, and SVD comparison surfaces
- Sprint 104 backend/runtime modernization and performance-sentinel guidance
- Sprint 105 reorder, graph, large-matrix guardrails, and scalability handoff
- the current source-list checker, Make/CMake parity expectations, and
  reviewed CI surface

The strongest Sprint 106 pressure is to reduce risk in large source and
giant-test owners without creating broad rewrites or destabilizing reviewed
solver behavior. The sprint must:
- re-rank large source and giant-test files from the live tree
- extract one high-value ownership seam from the largest CSC direct-family path
- extract focused seams from LU, QR, eigensolver, or iterative hotspots
- split reusable fixtures and helpers out of the largest proof owners
- keep Make, CMake, source-list checking, and reviewed parity exact
- update maintainer guidance for the new ownership layout
- close with before/after metrics, validation evidence, and Sprint 107 handoff

**End State:** Sprint 106 leaves behind:
- a refreshed maintainability ranking for large source and giant-test owners
- smaller high-risk CSC direct-family implementation owners
- smaller focused LU, QR, eigensolver, or iterative implementation owners
- reusable fixture/helper owners separated from giant tests
- exact source-list and CMake follow-through for extracted files
- maintainer documentation that reflects the new file ownership model
- validation artifacts and a residual extraction queue

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 106 project-plan estimate.

---

## Day 1: Sprint 106 Scope & Maintainability Baseline

**Title:** Maintainability Baseline
**Theme:** Convert the Sprint 106 project-plan section and prior Epic 10
handoffs into one bounded extraction package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 106 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 100 hotspot baseline artifacts and Sprint 102-105 closeout
   notes for extraction candidates, validation expectations, and deferred
   maintainability risks.
3. Inventory the Sprint 106 workstreams:
   - extraction target re-rank
   - LDLT/CSC source extraction
   - LU, QR, eigensolver, or iterative source extraction
   - giant-test fixture extraction
   - source-list and CMake follow-through
   - maintainer guidance
   - validation and closeout
4. Create Sprint 106 working notes and artifacts directory.
5. Record validation expectations for docs-only, source-touch, test-touch,
   build-system-touch, and mixed extraction days.

### Deliverables
- Sprint 106 workstream inventory
- working-notes baseline
- initial artifacts directory structure
- validation expectation list

### Completion Criteria
- every Sprint 106 project-plan item has day-level ownership
- prior Epic 10 extraction handoffs are visible in working notes
- validation expectations are explicit before audit or extraction work begins

---

## Day 2: Extraction Target Re-rank

**Title:** Target Re-rank
**Theme:** Re-rank large source and giant-test files by churn risk, ownership
ambiguity, and failure-localization value
**Time estimate:** 12 hours

### Tasks
1. Generate current size, line-count, symbol, and test-owner inventories for
   source, header, test, benchmark, and script files.
2. Review recent churn and failure-prone areas from Sprint 102-105 notes and
   the live git history.
3. Classify large owners by family:
   - LDLT and CSC direct solver paths
   - LU, QR, eigensolver, SVD, and iterative paths
   - graph, reorder, and large-matrix proof owners
   - shared fixture, oracle, and integration-test helpers
4. Rank extraction candidates by risk reduction, locality, validation cost,
   API impact, and Make/CMake follow-through cost.
5. Write the extraction target re-rank artifact.

### Deliverables
- ranked source and test owner inventory
- churn and ownership ambiguity notes
- fix-now vs defer extraction queue
- first-pass validation cost estimate

### Completion Criteria
- extraction targets are ranked from live repository evidence
- the highest-value CSC direct-family candidate is identified
- deferred candidates include explicit reasons and future sprint placement

---

## Day 3: LDLT/CSC Extraction Boundary

**Title:** CSC Boundary
**Theme:** Freeze the ownership seam for the highest-value CSC direct-family
extraction before editing implementation files
**Time estimate:** 12 hours

### Tasks
1. Inspect the selected LDLT/CSC source path, public headers, internal helper
   boundaries, and direct solver tests.
2. Identify the smallest cohesive helper group suitable for extraction without
   changing solver semantics.
3. Define file names, ownership responsibilities, include dependencies, and
   private/public boundaries.
4. Map all Make, CMake, source-list, and test references that will need updates.
5. Write the LDLT/CSC extraction boundary artifact with focused validation
   commands.

### Deliverables
- LDLT/CSC extraction design record
- dependency and include-boundary map
- source-list and CMake update checklist
- focused validation command list

### Completion Criteria
- the CSC extraction seam is narrow enough to review safely
- no public API change is implied unless explicitly documented
- all build-system follow-through points are known before source edits start

---

## Day 4: LDLT/CSC Extraction Batch

**Title:** CSC Extraction
**Theme:** Extract the selected CSC direct-family helper owner and preserve
existing direct solver behavior
**Time estimate:** 12 hours

### Tasks
1. Create the new internal source/header owner for the selected LDLT/CSC helper
   group.
2. Move implementation code with minimal logic changes and clear ownership
   naming.
3. Update includes, internal prototypes, Makefile source lists, CMake source
   lists, and source-list checker expectations.
4. Run focused direct solver and source-list validation.
5. Record before/after file metrics and validation output in working notes.

### Deliverables
- extracted LDLT/CSC helper owner
- updated build-system and source-list references
- focused validation results
- before/after file metric notes

### Completion Criteria
- extracted code builds through both maintained build surfaces
- direct solver behavior remains covered by focused tests
- the original large owner has a measurable reduction in responsibility

---

## Day 5: LDLT/CSC Test and Oracle Follow-Through

**Title:** CSC Proof Follow-Through
**Theme:** Tighten test, oracle, and documentation ownership around the new
LDLT/CSC extraction
**Time estimate:** 12 hours

### Tasks
1. Review direct solver tests and oracle helpers touched by the CSC extraction.
2. Move or rename test helpers only where it improves failure localization.
3. Add focused regression coverage for the extracted helper boundary if a gap
   appears during extraction.
4. Update direct solver maintainer notes or source comments that describe the
   old ownership layout.
5. Run focused direct solver, source-list, and formatting checks for touched
   files.

### Deliverables
- tightened CSC proof ownership
- focused regression coverage or explicit no-new-test rationale
- updated direct solver ownership notes
- Day 5 validation notes

### Completion Criteria
- tests point failures at the new helper owner when practical
- no stale ownership description remains for the extracted path
- focused validation passes before moving to the second extraction family

---

## Day 6: LU/QR/Eigs/Iterative Extraction Boundary

**Title:** Secondary Boundary
**Theme:** Select one or two focused seams from LU, QR, eigensolver, or
iterative hotspots
**Time estimate:** 12 hours

### Tasks
1. Compare Day 2 rankings with the residual risk after the CSC extraction.
2. Inspect candidate LU, QR, eigensolver, and iterative source owners.
3. Choose one larger seam or two smaller seams with low API risk and high
   review value.
4. Define file names, helper responsibilities, include dependencies, and build
   follow-through requirements.
5. Write the secondary extraction boundary artifact.

### Deliverables
- selected LU, QR, eigensolver, or iterative extraction target
- ownership and dependency map
- source-list and validation checklist
- explicit rationale for skipped candidates

### Completion Criteria
- selected seams are narrow, cohesive, and testable
- build-system updates are known before edits begin
- skipped hotspots remain in the residual extraction queue

---

## Day 7: Secondary Source Extraction Batch 1

**Title:** Secondary Batch 1
**Theme:** Extract the first selected LU, QR, eigensolver, or iterative source
seam
**Time estimate:** 12 hours

### Tasks
1. Create the new internal owner for the first selected secondary seam.
2. Move helper code with minimal behavioral change.
3. Update includes, local prototypes, Makefile source lists, CMake source
   lists, and source-list checker expectations.
4. Run focused tests for the affected solver family.
5. Record before/after file metrics and validation output.

### Deliverables
- first secondary extracted source owner
- updated build-system references
- focused solver-family validation
- before/after ownership metric notes

### Completion Criteria
- the affected solver family still passes focused validation
- source-list and CMake parity remain exact
- extracted ownership improves locality without broad refactoring

---

## Day 8: Secondary Source Extraction Batch 2

**Title:** Secondary Batch 2
**Theme:** Complete the second focused seam or deepen the first extraction only
where the Day 6 boundary justifies it
**Time estimate:** 12 hours

### Tasks
1. Re-check Day 7 results against the Day 6 extraction boundary.
2. Extract the second selected seam or complete the remaining part of the first
   seam.
3. Update all build-system, include, and source-list references.
4. Run focused family tests plus source-list validation.
5. Record any changed residual risk or deferred cleanup decisions.

### Deliverables
- second secondary extracted owner or completed primary secondary extraction
- updated Make/CMake/source-list references
- focused validation results
- updated residual extraction queue

### Completion Criteria
- no planned secondary extraction is left half-moved
- validation remains focused and passing
- residual cleanup is documented rather than hidden in code comments

---

## Day 9: Giant-Test Fixture Boundary

**Title:** Fixture Boundary
**Theme:** Identify reusable fixtures and helpers trapped inside the largest
direct, graph, and integration test owners
**Time estimate:** 12 hours

### Tasks
1. Inventory large tests and helper blocks for direct solver, graph/reorder,
   integration, oracle, and generated-fixture responsibilities.
2. Identify reusable setup, assertion, matrix-construction, and oracle helpers
   that can move without changing test intent.
3. Define fixture/helper file ownership and naming rules.
4. Map Make, CMake, test registration, and include updates required for helper
   extraction.
5. Write the giant-test fixture boundary artifact.

### Deliverables
- giant-test helper inventory
- selected fixture/helper extraction targets
- test ownership and naming plan
- test validation checklist

### Completion Criteria
- fixture extraction targets are separated from test behavior changes
- direct, graph, and integration test owners are represented
- validation commands are known before editing tests

---

## Day 10: Direct and Graph Fixture Extraction

**Title:** Fixture Batch 1
**Theme:** Split reusable direct solver and graph/reorder fixtures from giant
test owners
**Time estimate:** 12 hours

### Tasks
1. Create or extend focused test helper owners for selected direct solver and
   graph/reorder fixtures.
2. Move shared fixture setup and assertion helpers out of giant tests.
3. Update affected tests to include and use the extracted helpers.
4. Preserve test names, registration, and reviewed CTest surface unless an
   explicit change is required.
5. Run focused direct, graph, reorder, and source-list validation.

### Deliverables
- extracted direct solver fixture/helper owner
- extracted graph/reorder fixture/helper owner
- updated giant tests with smaller local responsibilities
- focused validation notes

### Completion Criteria
- giant tests are smaller without weakening assertions
- test registration and reviewed CTest shape remain intentional
- focused validation passes for affected test families

---

## Day 11: Integration and Oracle Fixture Extraction

**Title:** Fixture Batch 2
**Theme:** Split reusable integration and oracle helpers while keeping proof
intent readable at call sites
**Time estimate:** 12 hours

### Tasks
1. Extract selected integration or oracle helpers from the largest remaining
   giant-test owner.
2. Keep helper names specific enough that test intent remains clear without
   reading the helper implementation first.
3. Update Make, CMake, test helper includes, and source-list expectations.
4. Run focused integration, oracle, and affected solver tests.
5. Record before/after test-owner metrics.

### Deliverables
- extracted integration or oracle helper owner
- updated giant tests with clearer local proof intent
- build-system follow-through
- before/after test-owner metric notes

### Completion Criteria
- extracted helpers reduce duplication or size meaningfully
- call sites remain understandable
- focused tests and source-list checks pass

---

## Day 12: Source-List and CMake Parity Reconciliation

**Title:** Parity Reconciliation
**Theme:** Reconcile all extracted source and test owners across Make, CMake,
source-list checking, and reviewed CI assumptions
**Time estimate:** 12 hours

### Tasks
1. Audit every file added, moved, or split during Sprint 106.
2. Compare Makefile object lists, CMake targets, test registration, install or
   export surfaces, and source-list checker rules.
3. Run source-list and CMake configure/build checks appropriate to the touched
   files.
4. Fix stale references, missing registration, duplicate ownership, or
   accidental unreviewed test exposure.
5. Write the source-list and CMake reconciliation artifact.

### Deliverables
- source-list parity reconciliation notes
- Make and CMake update summary
- reviewed test-surface confirmation
- validation output references

### Completion Criteria
- every extracted file is owned by the intended build surface
- source-list checking reflects the new layout
- reviewed CMake and Make parity assumptions remain exact

---

## Day 13: Maintainer Guidance and Metrics Update

**Title:** Guidance Update
**Theme:** Update maintainer documentation and before/after metrics for the new
source and test ownership model
**Time estimate:** 12 hours

### Tasks
1. Update maintainer guidance for extracted CSC, secondary solver, and fixture
   ownership.
2. Add notes about where future helpers should live and when not to grow giant
   source or test owners.
3. Compile before/after metrics for large source files, giant tests, helper
   ownership, and validation surfaces.
4. Reconcile Sprint 106 working notes with completed work and residual items.
5. Write the maintainability metrics and documentation artifact.

### Deliverables
- updated maintainer guidance
- before/after maintainability metrics
- residual extraction and cleanup queue
- Day 13 documentation validation notes

### Completion Criteria
- documentation points maintainers to the new ownership layout
- metrics demonstrate the impact and limits of Sprint 106
- deferred work is explicit and ready for Sprint 107 planning

---

## Day 14: Validation and Sprint Closeout

**Title:** Closeout
**Theme:** Run required checks, reconcile artifacts, and close Sprint 106 with
clear handoff evidence
**Time estimate:** 12 hours

### Tasks
1. Run required quality checks for touched file types, including the full
   `make format && make lint && make test` gate when `.c` or `.h` files changed.
2. Run focused CMake, source-list, and reviewed parity checks needed by the
   extraction work.
3. Reconcile working notes, artifacts, documentation, and validation output.
4. Write the Sprint 106 closeout artifact with before/after ownership metrics.
5. Prepare Sprint 107 handoff notes for any remaining large-source or
   giant-test risks.

### Deliverables
- final validation output
- Sprint 106 closeout artifact
- before/after file and test owner metrics
- Sprint 107 handoff queue

### Completion Criteria
- required quality checks pass
- all Sprint 106 artifacts and working notes agree with the final tree
- residual risks are documented with owner, reason, and suggested next step
