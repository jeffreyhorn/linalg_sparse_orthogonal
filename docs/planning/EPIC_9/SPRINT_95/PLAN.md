# Sprint 95 Plan: Public Narrative, Docs & Workflow Coherence

**Sprint Duration:** 14 days
**Goal:** Remove sprint-era chronology from permanent product surfaces and make
the public workflow narrative smaller, clearer, and more coherent. This sprint
implements the Sprint 95 section of
`docs/planning/EPIC_9/PROJECT_PLAN.md`.

**Starting Point:** Sprint 95 begins from:
- the Sprint 90 public narrative target
- the Sprint 91 public workflow and capability changes
- the Sprint 94 widened capability surface, which must be described without
  turning permanent docs into sprint history
- a documentation tree that has accumulated useful proof, workflow, and support
  content but still exposes too much chronology, duplicated narrative, and
  proof-owner naming to public readers

The strongest Sprint 95 pressure is not to add more explanation everywhere. It
is to make the permanent public story smaller and more product-shaped by:
- auditing the current README, tutorial, example, install, benchmark, support,
  public header, and proof-owner surfaces
- defining clear audience ownership for adoption, support, benchmark, and
  maintainer material
- rewriting the highest-value public narrative surfaces first
- removing sprint-era residue from headers and examples only where it affects
  public understanding
- renaming or regrouping proof owners where sprint names make the product story
  harder to follow
- reconciling support and workflow surfaces with the cleaned ownership model
- closing with validation artifacts and a residual queue that keeps permanent
  docs from becoming another sprint archive

**End State:** Sprint 95 leaves behind:
- a public-surface audit and ranked cleanup queue
- an audience and narrative ownership model
- cleaned README, tutorial, and highest-value public docs
- header and example wording that avoids sprint-era residue on touched surfaces
- clearer proof-owner or test grouping names for the highest-value cases
- consolidated install, benchmark, maintainer, and support workflow surfaces
- a validated Sprint 95 closeout package and Sprint 96 handoff queue

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `164` hours, matching the Sprint 95 project-plan estimate.

---

## Day 1: Public-Surface Inventory

**Title:** Surface Inventory
**Theme:** Build the live map of permanent product surfaces that still read like
sprint history
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 95 section of
   `docs/planning/EPIC_9/PROJECT_PLAN.md` and the Sprint 90, Sprint 91, and
   Sprint 94 planning assumptions that define the public narrative target.
2. Inventory the permanent user-facing surfaces most likely to contain
   chronology or duplicated story:
   - `README.md`
   - tutorial and quick-start docs
   - examples and example descriptions
   - install and package docs
   - benchmark and support docs
   - public headers with user-visible narrative comments
3. Separate permanent public surfaces from planning, retrospective, and
   intentionally historical documents.
4. Capture obvious sprint-era residue, duplicated onboarding claims, stale
   workflow descriptions, and confusing proof-owner references.
5. Open Sprint 95 working notes with the initial surface list and evidence.

### Deliverables
- public-surface inventory
- initial chronology and duplication evidence list
- Sprint 95 working-notes baseline

### Completion Criteria
- the highest-value permanent public surfaces are identified
- intentionally historical planning docs are excluded from cleanup scope
- Sprint 95 starts from a live evidence map, not a generic docs-polish list

---

## Day 2: Audit Ranking & Cleanup Queue

**Title:** Public-Surface Audit
**Theme:** Rank the worst public narrative contradictions by reader impact and
cleanup risk
**Time estimate:** 12 hours

### Tasks
1. Re-scan the Day 1 surface list for the strongest problem classes:
   - sprint chronology on permanent product pages
   - duplicated quick-start or adoption stories
   - benchmark claims that do not point to the right proof owner
   - install/support workflows that repeat or conflict
   - public comments that explain development history instead of stable API
     behavior
2. Rank findings by user impact, truth risk, and implementation cost.
3. Identify which cleanup candidates are rewrite-only, which require proof-owner
   naming changes, and which need validation because filenames or test targets
   may move.
4. Define the fix-now queue for Sprint 95 and a lower-priority residual queue.
5. Write a Day 2 audit artifact summarizing the ranked queue.

### Deliverables
- ranked public-surface audit artifact
- fix-now vs residual cleanup queue
- proof-risk and validation-risk notes

### Completion Criteria
- the public cleanup problem is reduced to a ranked queue
- the highest-value Sprint 95 targets are explicit before rewrite work begins
- risky naming or proof-owner changes are separated from plain prose cleanup

---

## Day 3: Audience Ownership Model

**Title:** Audience Model
**Theme:** Define the permanent audience split before rewriting public docs
**Time estimate:** 12 hours

### Tasks
1. Define the stable audience split for the cleaned documentation set:
   - adoption and quick-start readers
   - API users and example readers
   - install and packaging consumers
   - benchmark and performance readers
   - maintainers and proof reviewers
2. Decide which surface owns each narrative:
   - first-use capability story
   - support and install workflow
   - benchmark interpretation
   - proof and validation references
   - maintainer-only history
3. Write naming and style rules for permanent product surfaces:
   - describe current behavior before history
   - link to proof owners without narrating every sprint
   - keep historical context in planning docs
   - avoid duplicate long-form explanations
4. Identify public surfaces that need ownership headers or shorter linking
   patterns.
5. Record the Day 3 ownership model.

### Deliverables
- audience ownership design artifact
- naming and style rules for Sprint 95 edits
- mapping from narratives to owning surfaces

### Completion Criteria
- every major public narrative has one intended owner
- permanent docs have a clear rule for when to link planning history instead of
  repeating it
- rewrite days can follow one shared voice and ownership model

---

## Day 4: README Narrative Boundary

**Title:** README Boundary
**Theme:** Make the README the concise adoption front door, not a sprint ledger
**Time estimate:** 12 hours

### Tasks
1. Re-read the README against the Day 2 audit and Day 3 audience model.
2. Decide the README's permanent responsibilities:
   - concise project identity
   - current supported capability story
   - shortest build/test/install path
   - pointers to tutorial, examples, benchmarks, and maintainer docs
3. Identify README content that should be deleted, shortened, or moved behind
   links.
4. Draft the cleaned README structure without changing technical truth.
5. Record any claims that require cross-checking against tests, examples,
   package docs, or benchmarks before the rewrite lands.

### Deliverables
- README rewrite outline
- claim-check list for README edits
- move/delete list for historical or duplicated content

### Completion Criteria
- README ownership is fixed before the prose rewrite
- every preserved README claim has an intended proof or destination
- no sprint-era chronology is required to understand the product front door

---

## Day 5: README Cleanup Batch

**Title:** README Cleanup
**Theme:** Land the first high-value public rewrite batch
**Time estimate:** 12 hours

### Tasks
1. Rewrite the highest-value README sections from the Day 4 outline.
2. Remove or replace sprint-era chronology from permanent README content.
3. Collapse duplicated adoption, support, and workflow paragraphs.
4. Ensure links point to the intended owner surfaces from the Day 3 model.
5. Run the documentation-focused review needed for changed links and examples.

### Deliverables
- cleaned README public narrative batch
- updated link and claim references
- notes for any README follow-up that should not block the first rewrite batch

### Completion Criteria
- the README reads as product documentation rather than sprint closeout
- duplicated public narrative is materially reduced
- retained claims remain true and point to stable owner surfaces

---

## Day 6: Tutorial & Quick-Start Cleanup

**Title:** Tutorial Cleanup
**Theme:** Make the primary learning path coherent with the cleaned README
**Time estimate:** 12 hours

### Tasks
1. Audit tutorial and quick-start docs against the Day 3 audience model.
2. Remove historical explanations that duplicate README or planning history.
3. Reorder tutorial content around current user workflows:
   - create or load a matrix
   - solve or analyze
   - validate or interpret output
   - find examples and support docs
4. Ensure tutorial terminology matches the cleaned README.
5. Record any example or header wording that needs follow-up on later days.

### Deliverables
- cleaned tutorial or quick-start narrative batch
- terminology alignment notes
- follow-up list for examples and headers

### Completion Criteria
- the tutorial complements the README instead of repeating it
- first-use workflow is visible without sprint context
- unresolved example/header wording is explicitly queued

---

## Day 7: Public Docs Coherence Pass

**Title:** Docs Coherence
**Theme:** Align remaining high-value public docs with the new ownership model
**Time estimate:** 12 hours

### Tasks
1. Review the highest-value permanent docs outside README and tutorial:
   - install docs
   - usage notes
   - benchmark overview
   - support or troubleshooting docs
2. Remove duplicated narrative that now belongs to README or tutorial.
3. Replace sprint-era explanation with stable current-state wording.
4. Add concise cross-links only where they reduce duplication.
5. Update the ranked audit queue to mark completed and deferred surfaces.

### Deliverables
- public docs coherence cleanup batch
- updated audit queue
- cross-link and ownership notes

### Completion Criteria
- the main public docs no longer compete to tell the same story
- high-value permanent docs use current-state language
- residual narrative cleanup is explicitly tracked

---

## Day 8: Public Header Narrative Cleanup

**Title:** Header Cleanup
**Theme:** Keep public headers focused on API contracts, not implementation
history
**Time estimate:** 12 hours

### Tasks
1. Review touched or high-visibility public headers against the Day 3 style
   rules.
2. Identify comments that expose sprint chronology, development rationale, or
   stale capability framing where API contract wording is enough.
3. Rewrite only the highest-value header comments to preserve stable behavior,
   return codes, constraints, and examples.
4. Avoid broad comment churn in headers that are not part of the public
   narrative problem.
5. Run the appropriate formatting and documentation review for touched headers.

### Deliverables
- public header narrative cleanup batch
- list of untouched header surfaces with rationale
- updated validation notes for any `.h` changes

### Completion Criteria
- touched public headers describe stable contracts rather than sprint history
- header changes are scoped to user-visible narrative value
- any code-format or lint implications are recorded before closeout

---

## Day 9: Example Surface Cleanup

**Title:** Example Cleanup
**Theme:** Make examples read as reusable workflows instead of sprint proofs
**Time estimate:** 12 hours

### Tasks
1. Review example names, comments, and companion docs from the Day 6 follow-up
   list.
2. Remove sprint-era language from example descriptions.
3. Align example comments with the cleaned README and tutorial workflow.
4. Preserve examples that intentionally demonstrate current behavior or support
   claims.
5. Identify any example renames that are worth doing later, separating them
   from low-value churn.

### Deliverables
- example narrative cleanup batch
- example naming and residual rename notes
- updated user-workflow cross-reference map

### Completion Criteria
- examples present current workflows without requiring sprint context
- example prose reinforces the cleaned adoption path
- risky or low-value renames are deferred explicitly

---

## Day 10: Proof-Owner Naming Design

**Title:** Proof Naming Design
**Theme:** Decide which sprint-named tests or proof owners should become
product-oriented
**Time estimate:** 12 hours

### Tasks
1. Audit sprint-named integration, proof, and validation owners most visible to
   public or maintainer workflows.
2. Separate proof owners into:
   - stable product-oriented owners worth renaming or regrouping now
   - sprint-named regression owners that should remain historical
   - low-value names that should not churn in Sprint 95
3. Define rename and regrouping rules:
   - preserve test coverage and build target behavior
   - update references in docs and scripts
   - avoid breaking historical planning references unnecessarily
4. Choose the smallest high-value rename or regrouping batch.
5. Write the Day 10 proof-owner naming design artifact.

### Deliverables
- proof-owner naming audit
- rename/regrouping rules
- selected high-value proof naming batch

### Completion Criteria
- Sprint 95 has a bounded proof naming plan before files move
- public and maintainer references are included in the rename scope
- churn-only proof renames are rejected explicitly

---

## Day 11: Proof Naming Cleanup Batch

**Title:** Proof Naming Cleanup
**Theme:** Land the selected product-oriented proof owner cleanup safely
**Time estimate:** 11 hours

### Tasks
1. Apply the selected proof-owner rename or regrouping batch from Day 10.
2. Update build, test, documentation, and support references to the changed
   owners.
3. Preserve compatibility notes where historical sprint names still matter.
4. Run targeted checks for renamed files, targets, or proof groups.
5. Update the audit queue with completed and deferred naming work.

### Deliverables
- product-oriented proof-owner cleanup batch
- updated references
- targeted validation notes

### Completion Criteria
- renamed or regrouped proof owners are discoverable by product capability,
  not sprint chronology
- references and build/test hooks remain coherent
- deferred proof naming work is tracked instead of half-started

---

## Day 12: Support Surface Consolidation

**Title:** Support Consolidation
**Theme:** Reconcile install, benchmark, and maintainer surfaces with the
cleaned public narrative
**Time estimate:** 11 hours

### Tasks
1. Review install, benchmark, package, and maintainer docs against the Day 3
   ownership model.
2. Consolidate duplicated workflow instructions.
3. Make benchmark and support docs point to the right proof and reporting
   owners.
4. Remove sprint-era explanation from surfaces meant for users or maintainers
   doing current work.
5. Record any intentionally historical support content that should remain in
   planning docs only.

### Deliverables
- support-surface consolidation batch
- install/benchmark/maintainer ownership notes
- updated support cross-link map

### Completion Criteria
- support surfaces have a clear owner split
- install, benchmark, and maintainer workflows no longer duplicate the public
  front door
- public support docs describe current use rather than development chronology

---

## Day 13: Validation & Residual Queue

**Title:** Validation Sweep
**Theme:** Validate the cleaned narrative and freeze the residual public-docs
queue
**Time estimate:** 11 hours

### Tasks
1. Run the strongest appropriate validation for changed docs, headers, examples,
   tests, or support scripts.
2. Re-check links, command references, example names, and proof-owner
   references touched during Sprint 95.
3. Review the Day 2 audit queue and mark:
   - completed cleanup
   - explicitly deferred cleanup
   - intentional historical surfaces
   - any follow-up that belongs to Sprint 96 or a later epic
4. Write a validation and residual queue artifact.
5. Prepare closeout notes for Day 14.

### Deliverables
- validation results
- residual public-narrative queue
- closeout preparation notes

### Completion Criteria
- changed surfaces have been validated with the right checks
- no incomplete rename or reference update remains hidden
- the residual queue distinguishes future work from intentional non-claims

---

## Day 14: Sprint 95 Closeout

**Title:** Closeout
**Theme:** Close Sprint 95 with evidence, artifacts, and a clean handoff
**Time estimate:** 11 hours

### Tasks
1. Re-read the Sprint 95 project-plan section against the completed artifacts.
2. Confirm each item has been addressed or explicitly deferred:
   - Public-Surface Audit
   - Narrative Ownership Design
   - README/Tutorial Cleanup Batch
   - Header and Example Narrative Cleanup
   - Test/Proof Naming Cleanup
   - Support-Surface Consolidation
   - Validation and Closeout
3. Write the Sprint 95 retrospective and handoff notes.
4. Confirm the public narrative is smaller, clearer, and less chronological
   than the starting state.
5. Record the Sprint 96 handoff queue and close the working notes.

### Deliverables
- Sprint 95 retrospective
- Sprint 95 handoff queue
- final validation and artifact index

### Completion Criteria
- Sprint 95 closes from validated evidence, not aspirational docs polish
- all project-plan items have a clear done/deferred status
- Sprint 96 receives a bounded handoff instead of a broad narrative-cleanup
  backlog
