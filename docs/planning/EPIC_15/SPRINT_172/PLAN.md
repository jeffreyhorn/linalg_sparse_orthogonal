# Sprint 172 Plan: Public Header Coherence Batch 2

**Sprint Duration:** 14 days
**Goal:** Normalize one additional high-impact public header family and add
lightweight guardrails against documentation drift. This sprint implements the
Sprint 172 section of `docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md` and the title "Sprint 172: Public
Header Coherence Batch 2"; the active merged Sprint 172 project-plan section
lives in `docs/planning/EPIC_15/PROJECT_PLAN.md` and has the same title.

**Starting Point:** Sprint 172 begins from:

- prior header cleanup batches from Epic 14;
- Sprint 167 evidence-ledger and claim-gate conventions;
- Sprint 170 static-first package and shared-library ABI product decision;
- Sprint 171 package-manager deferral guard and adoption/package boundaries;
- current public header, generated API, tutorial, cookbook, and maintainer
  documentation surfaces.

The sprint must:

- select the next highest-impact public header family based on adoption risk
  and API complexity;
- normalize ownership, lifecycle, error-code, tolerance, workspace, and
  threading language;
- reorganize declarations into coherent sections without changing API
  behavior;
- align examples or usage docs with the cleaned header;
- add or extend a lightweight guard against header documentation drift where
  practical;
- run format/lint/test if code headers change, plus focused docs and guard
  checks.

**End State:** Sprint 172 leaves behind:

- one cleaned public header family;
- updated usage docs or examples for the selected header family;
- mechanical or reviewable guard coverage for header documentation drift;
- validation records showing API behavior stayed unchanged;
- Sprint 172 working notes, daily artifacts, and closeout records.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 172 project-plan estimate.

---

## Day 1: Sprint Intake And Header Baseline

**Title:** Header Intake
**Theme:** Establish Sprint 172 scope, inherited claim boundaries, and public
header cleanup constraints
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 172 section of
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.
2. Review prior Epic 14 header cleanup artifacts and Sprint 167 evidence
   ledger guidance.
3. Review Sprint 170/Sprint 171 package, ABI, and package-manager handoff
   boundaries.
4. Create Sprint 172 working notes and artifact directory structure.
5. Record the prompt path/source-artifact mismatch.
6. Write the Day 1 header-intake artifact.

### Deliverables

- Sprint 172 working-notes baseline
- artifact directory structure
- source artifact note
- inherited header/package/ABI/adoption boundary summary
- Day 1 header-intake artifact

### Completion Criteria

- Sprint 172 scope is tied to the active Epic 15 project plan
- public-header cleanup constraints are explicit before selecting a family
- package-manager, shared-library, ABI, runtime-loader, and broad platform
  non-claims remain protected

---

## Day 2: Public Header Candidate Inventory

**Title:** Header Inventory
**Theme:** Inventory candidate public header families and classify adoption
risk, API complexity, and documentation drift
**Time estimate:** 12 hours

### Tasks

1. Inventory installed public headers under `include/`.
2. Review existing API reference, tutorial, cookbook, examples, and maintainer
   guide references for each candidate family.
3. Classify ownership, lifecycle, error-code, tolerance, workspace, threading,
   callback, and allocation-language risk.
4. Identify declaration-order or sectioning problems.
5. Rank candidate families by adoption risk and cleanup value.
6. Write the Day 2 header-candidate inventory artifact.

### Deliverables

- public-header candidate matrix
- adoption-risk and API-complexity ranking
- documentation-drift notes
- declaration-organization risk list
- Day 2 header-inventory artifact

### Completion Criteria

- candidate header families are visible and comparable
- the selection decision is evidence-based
- no header edits are made before the cleanup target is selected

---

## Day 3: Header Family Selection Decision

**Title:** Header Selection
**Theme:** Select the highest-impact public header family for cleanup and
define support/non-claim boundaries
**Time estimate:** 12 hours

### Tasks

1. Compare Day 2 candidates against adoption risk, API complexity,
   documentation drift, and cleanup feasibility.
2. Select exactly one public header family for Sprint 172 cleanup.
3. Define in-scope contract language: ownership, lifecycle, errors,
   tolerance, workspace, threading, callbacks, and allocation behavior.
4. Define out-of-scope claims: ABI stability, shared-library support,
   package-manager support, broad platform parity, broad performance, and
   state-of-the-art coverage.
5. Define impacted examples/docs/guard surfaces.
6. Write the Day 3 header-selection artifact.

### Deliverables

- selected public header family
- supported contract-language scope
- unsupported-claim list
- impacted file list
- Day 3 selection artifact

### Completion Criteria

- exactly one header family is selected
- behavior-preservation constraints are explicit
- unsupported package/ABI/platform claims remain non-claims

---

## Day 4: Contract Cleanup Design

**Title:** Contract Design
**Theme:** Design the selected header’s ownership, lifecycle, error, tolerance,
workspace, and threading cleanup before editing
**Time estimate:** 12 hours

### Tasks

1. Read the selected header and its implementation/tests/examples.
2. Map current public declarations to intended documentation sections.
3. Draft normalized language for ownership, lifecycle, error-code,
   tolerance, workspace, threading, callback, and allocation behavior.
4. Identify comments that should be retained, rewritten, moved, or removed.
5. Define behavior-preservation and declaration-compatibility constraints.
6. Write the Day 4 contract-cleanup design artifact.

### Deliverables

- selected-header contract map
- normalized language plan
- declaration section plan
- behavior-preservation constraints
- Day 4 contract-design artifact

### Completion Criteria

- header edits are scoped before implementation
- comment changes are tied to real public API semantics
- no API behavior or binary/package support claim is changed by design

---

## Day 5: Contract Cleanup Implementation

**Title:** Contract Cleanup
**Theme:** Normalize selected header contract language without changing API
behavior
**Time estimate:** 12 hours

### Tasks

1. Update selected header comments for ownership and lifecycle semantics.
2. Normalize error-code and failure-mode language.
3. Normalize tolerance and workspace language where relevant.
4. Normalize threading, callback, and allocation language where relevant.
5. Preserve declarations and behavior unless Day 4 explicitly authorized a
   non-behavioral reorder.
6. Write the Day 5 contract-cleanup artifact.

### Deliverables

- cleaned selected header contract comments
- behavior-preservation notes
- impacted declaration list
- Day 5 implementation artifact

### Completion Criteria

- selected header contract language is clearer and internally consistent
- API declarations remain behavior-compatible
- no unsupported ABI/package/platform claim is introduced

---

## Day 6: Declaration Organization Design

**Title:** Declaration Design
**Theme:** Design coherent declaration ordering and section headings for the
selected header
**Time estimate:** 12 hours

### Tasks

1. Review selected header declaration order after Day 5 cleanup.
2. Group declarations by workflow and dependency order.
3. Define section headings and ordering rules.
4. Identify declarations that must not move because of readability or
   generated-doc constraints.
5. Define diff-review expectations for non-behavioral reordering.
6. Write the Day 6 declaration-organization design artifact.

### Deliverables

- declaration grouping plan
- section-heading plan
- non-move exception list
- Day 6 declaration-design artifact

### Completion Criteria

- declaration reordering is reviewable before implementation
- public API behavior remains unchanged
- generated API/readability constraints are explicit

---

## Day 7: Declaration Organization Implementation

**Title:** Declaration Organization
**Theme:** Reorder selected header declarations into coherent sections without
behavior changes
**Time estimate:** 12 hours

### Tasks

1. Apply the Day 6 declaration ordering plan.
2. Add or normalize section headings where they improve generated API
   readability.
3. Keep declarations, names, signatures, macros, enums, and structs
   behavior-compatible.
4. Update local comments affected by the reorder.
5. Run focused header/diff checks.
6. Write the Day 7 declaration-organization artifact.

### Deliverables

- organized selected header
- section-heading updates
- focused header/diff validation notes
- Day 7 declaration-organization artifact

### Completion Criteria

- selected header has coherent sections
- declaration changes are non-behavioral
- focused validation passes or stops for user input

---

## Day 8: Example And Usage Documentation Design

**Title:** Usage Design
**Theme:** Design example or documentation alignment for the cleaned header’s
recommended workflow
**Time estimate:** 12 hours

### Tasks

1. Review examples and docs that reference the selected header family.
2. Identify one recommended workflow that should be clearer after cleanup.
3. Define README/tutorial/cookbook/API-reference/maintainer-guide edit scope.
4. Preserve source install, package-manager, shared-library, ABI,
   runtime-loader, platform, and performance non-claims.
5. Define documentation claim scans for Day 9.
6. Write the Day 8 usage-documentation design artifact.

### Deliverables

- usage-doc update plan
- recommended workflow mapping
- claim-scan plan
- Day 8 usage-design artifact

### Completion Criteria

- documentation edits are scoped before implementation
- docs will reinforce the selected header workflow
- unsupported adoption/package/ABI/platform claims remain separate

---

## Day 9: Example And Usage Documentation Implementation

**Title:** Usage Update
**Theme:** Update examples or usage docs for the cleaned header’s recommended
workflow
**Time estimate:** 12 hours

### Tasks

1. Update the selected docs or examples from Day 8.
2. Keep example changes minimal and behavior-preserving unless tests require
   otherwise.
3. Add links from broad docs to the cleaned header workflow where useful.
4. Preserve package-manager, shared-library, dynamic ABI, runtime-loader,
   platform, and performance non-claims.
5. Run targeted documentation claim scans.
6. Write the Day 9 usage-update artifact.

### Deliverables

- updated examples or usage docs
- targeted claim-scan results
- behavior-preservation notes
- Day 9 usage-update artifact

### Completion Criteria

- usage docs align with the cleaned header
- unsupported claims remain explicit or absent
- docs/examples are ready for guard design

---

## Day 10: Mechanical Guard Design

**Title:** Guard Design
**Theme:** Design lightweight guard coverage for selected-header declaration
and documentation drift
**Time estimate:** 12 hours

### Tasks

1. Review existing script/test/report-index guard patterns.
2. Decide whether the guard should be shell, Python, Make target, or
   review-only checklist.
3. Define positive checks for selected-header sections and expected docs links.
4. Define negative checks for unsupported package/ABI/platform/adoption
   wording where practical.
5. Define validation commands and failure output.
6. Write the Day 10 guard-design artifact.

### Deliverables

- guard design
- positive and negative check list
- validation command list
- Day 10 guard-design artifact

### Completion Criteria

- guard behavior is scoped before implementation
- guard cannot silently change API behavior
- unsupported package/ABI/platform claims remain protected

---

## Day 11: Mechanical Guard Implementation

**Title:** Guard Implementation
**Theme:** Add or extend lightweight guard coverage for selected-header docs
and declaration coherence
**Time estimate:** 12 hours

### Tasks

1. Implement the selected guard or reviewable checklist artifact.
2. Add positive checks for selected-header section/docs coverage.
3. Add negative checks for unsupported claim wording where practical.
4. Wire the guard into the appropriate local validation surface if warranted.
5. Run focused guard validation.
6. Write the Day 11 guard-implementation artifact.

### Deliverables

- implemented guard or checklist
- focused validation output
- wiring decision record
- Day 11 guard-implementation artifact

### Completion Criteria

- selected-header drift is guarded or explicitly review-gated
- guard output is clear enough for maintainers
- focused validation passes or stops for user input

---

## Day 12: Integrated Header Validation

**Title:** Header Validation
**Theme:** Run required quality gates, header-specific checks, docs checks, and
guard validation
**Time estimate:** 12 hours

### Tasks

1. Run `make format && make lint && make test` if any `.c` or `.h` files were
   modified.
2. Run focused guard checks added or changed during Sprint 172.
3. Run package-manager/static-package deferral guards if docs touch adoption,
   package, ABI, runtime-loader, or platform boundaries.
4. Run targeted docs and claim scans.
5. Record validation output and failures, if any.
6. Write the Day 12 integrated-validation artifact.

### Deliverables

- full C quality-gate decision/results
- focused guard validation results
- docs/claim-scan results
- Day 12 validation artifact

### Completion Criteria

- required quality gates pass before closeout work
- header cleanup is behavior-compatible
- failures stop the sprint for user input

---

## Day 13: Integrated Claim Review

**Title:** Claim Review
**Theme:** Reconcile selected header cleanup, examples/docs, guards, and
validation into one coherent API claim boundary
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 172 artifacts and working notes.
2. Reconcile selected-header changes against examples/docs, guard coverage,
   and validation results.
3. Confirm no generated API artifacts or build outputs are staged
   unintentionally.
4. Run targeted package-manager, shared-library, ABI, runtime-loader,
   platform, performance, and state-of-the-art claim scans.
5. Identify residuals and Sprint 173 handoff needs.
6. Write the Day 13 integrated-claim-review artifact.

### Deliverables

- integrated header claim review
- generated-output staging check
- claim-scan results
- residual and handoff list
- Day 13 claim-review artifact

### Completion Criteria

- selected header claim boundary is internally coherent
- docs/examples/guards match the cleaned header
- no generated artifacts are staged unintentionally

---

## Day 14: Sprint Closeout And Sprint 173 Handoff

**Title:** Sprint Closeout
**Theme:** Reconcile Sprint 172 deliverables, final validation, and handoff to
Sprint 173 generated API HTML publication work
**Time estimate:** 12 hours

### Tasks

1. Reconcile Sprint 172 outcomes against project-plan items 172.1 through
   172.6.
2. Confirm selected header cleanup, docs/examples alignment, guard coverage,
   and validation artifacts are complete.
3. Re-run final lightweight hygiene checks and required focused checks.
4. Confirm no generated output is staged unintentionally.
5. Prepare Sprint 173 handoff notes for generated API HTML publication.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables

- final Sprint 172 validation record
- project-plan item reconciliation
- generated-output staging check
- Sprint 173 handoff
- Day 14 sprint-closeout artifact

### Completion Criteria

- one high-impact public header family is cleaned and validated
- documentation and guards match the selected header contract
- Sprint 173 can begin from a clearer public API documentation boundary
