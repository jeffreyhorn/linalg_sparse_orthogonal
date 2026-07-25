# Sprint 135 Plan: Adoption Surface Simplification & Documentation Productization

**Sprint Duration:** 14 days
**Goal:** Simplify the adoption surface after Epic 10 by separating first-use
guides from maintainer history and making compressed-first workflows easier to
find.

**Starting Point:** Sprint 135 begins from:
- Sprint 131 report index decisions and generated-report index conventions
- Sprint 133 static-first package, ABI, and shared-library product decisions
- Sprint 134 Linux/macOS/Windows package and platform support-tier truth
- current README, tutorial, solver-selection, algorithm, benchmark, install,
  maintainer, and example documentation
- current compressed-first direct, iterative, Matrix Market, SVD, eigensolver,
  and benchmark examples
- existing documentation hygiene, path checks, and claim-boundary validation
  expectations

The sprint must:
- audit the full adoption surface for duplicated, stale, historical, or
  maintainer-only material
- design a clear split between concise current algorithm reference and
  historical measurement appendix content
- implement the algorithm-document split or a bounded first phase with
  redirects and link/path checks
- productize compressed-first workflows so first-use direct, iterative, Matrix
  Market, SVD, eigensolver, and benchmark paths are easier to follow
- surface generated report indexes and local-measurement interpretation in
  concise adoption language
- validate links, paths, claims, examples, and support-tier wording without
  widening package, ABI, platform, or performance claims
- publish adoption simplification metrics and residual documentation work

**End State:** Sprint 135 leaves behind:
- adoption-surface audit and simplification map
- algorithm-document split design
- implemented algorithm split or bounded first phase
- compressed-first cookbook updates
- benchmark and report-index adoption docs
- link, path, and claim-boundary validation evidence
- closeout metrics and residual docs queue

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `164` hours, matching the Sprint 135 project-plan estimate.

---

## Day 1: Adoption Sprint Intake

**Title:** Adoption Intake
**Theme:** Establish Sprint 135 scope, artifact structure, documentation
surfaces, and claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 135 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Review Sprint 131 report-index decisions and Sprint 133-134 package,
   platform, and support-tier truth.
3. Inventory current adoption surfaces: README, tutorial, solver selection,
   examples, algorithm docs, benchmark docs, install docs, and maintainer
   guide.
4. Map Sprint 135 project-plan Items 1-7 to day-level owners.
5. Create the Sprint 135 working-notes baseline and artifact directory.
6. Record claim fences for first-use guidance, maintainer history, package
   support, platform support, report interpretation, and performance wording.

### Deliverables
- Sprint 135 working-notes baseline
- artifact directory structure
- adoption-surface inventory
- item-to-day owner map
- support-tier and documentation claim-boundary notes

### Completion Criteria
- every Sprint 135 project-plan item has a day-level owner
- prior report-index, package, ABI, and platform truth is preserved before
  documentation changes begin
- first-use, reference, maintainer, historical, and generated-report surfaces
  are visible before simplification decisions begin

---

## Day 2: Adoption Surface Audit

**Title:** Surface Audit
**Theme:** Audit first-use, reference, example, benchmark, install, and
maintainer documentation for overlap and adoption friction
**Time estimate:** 12 hours

### Tasks
1. Inspect README, tutorial, solver-selection, install, benchmark, algorithm,
   and maintainer docs for duplicated onboarding material.
2. Inspect example and cookbook-style content for discoverability of
   compressed-first workflows.
3. Classify each document section as first-use guide, concise reference,
   generated-report index, maintainer history, or historical measurement
   appendix candidate.
4. Identify stale references, overlong adoption paths, repeated support-tier
   explanations, and places where maintainer-only context interrupts first-use
   workflows.
5. Record link/path dependencies that must be preserved during later doc
   movement.
6. Write the adoption surface audit artifact.

### Deliverables
- adoption-surface map
- overlap and duplication list
- first-use friction list
- maintainer-history extraction candidates
- link/path dependency notes

### Completion Criteria
- every major adoption document has an assigned role
- duplicated or displaced material is captured before edits begin
- compressed-first discoverability gaps are named by workflow family

---

## Day 3: Algorithm Split Design

**Title:** Algorithm Design
**Theme:** Design the split between concise current algorithm reference and
historical measurement appendix material
**Time estimate:** 12 hours

### Tasks
1. Inspect current algorithm documentation and related benchmark/report
   references.
2. Identify content that belongs in concise current reference: supported
   APIs, solver behavior, compressed-first expectations, known limitations,
   and current selection guidance.
3. Identify historical measurement content that should move to appendix or
   archive-style docs.
4. Define target filenames, headings, redirects, backlinks, and compatibility
   notes for moved sections.
5. Define link-check and claim-boundary validation required after the split.
6. Write the algorithm doc split design artifact.

### Deliverables
- algorithm reference target structure
- historical measurement appendix structure
- move/redirect map
- link and backlink preservation plan
- validation plan for the split

### Completion Criteria
- current-reference and historical-appendix responsibilities are separated
- no historical performance detail is left as first-use adoption guidance by
  default
- the implementation batch has a bounded edit plan and validation checklist

---

## Day 4: Algorithm Split Preparation

**Title:** Split Prep
**Theme:** Prepare the algorithm-document split with minimal-risk movement,
anchors, and link updates
**Time estimate:** 12 hours

### Tasks
1. Stage the target algorithm reference and historical appendix file layout.
2. Draft section headings, anchor names, compatibility notes, and redirect
   language before moving large blocks.
3. Identify links from README, tutorial, examples, solver-selection, benchmark
   docs, and maintainer guide that need updates.
4. Decide whether the sprint can implement the full split or a bounded first
   phase.
5. Record risks around broken anchors, duplicated claims, and historical
   performance language drifting into current guidance.
6. Write the split preparation artifact.

### Deliverables
- file layout preparation notes
- anchor and heading inventory
- inbound-link update queue
- full-split versus bounded-phase decision
- split risk notes

### Completion Criteria
- target docs can be edited without guessing ownership boundaries
- inbound links are known before movement begins
- the selected split scope fits inside the remaining sprint budget

---

## Day 5: Algorithm Split Batch 1

**Title:** Split Batch 1
**Theme:** Implement the first algorithm-document split batch and preserve
adoption links
**Time estimate:** 12 hours

### Tasks
1. Move or rewrite selected current algorithm material into the concise
   reference target.
2. Move or isolate selected historical measurement material into the appendix
   target.
3. Update inbound links for the moved sections.
4. Add short redirect or orientation notes where old locations must remain
   discoverable.
5. Run focused path, anchor, and whitespace checks for touched documentation.
6. Write the Batch 1 implementation artifact.

### Deliverables
- first algorithm split edit batch
- updated inbound links
- historical appendix seed or expanded appendix
- redirect/orientation notes
- focused validation evidence

### Completion Criteria
- moved content has exactly one clear primary home
- first-use links point to concise current guidance first
- historical material remains reachable without dominating adoption docs

---

## Day 6: Algorithm Split Batch 2

**Title:** Split Batch 2
**Theme:** Complete or stabilize the algorithm split and clean residual
duplication
**Time estimate:** 12 hours

### Tasks
1. Continue the algorithm split for remaining high-value sections.
2. Remove or consolidate duplicated current-reference wording created by the
   split.
3. Update cross-links from solver-selection, benchmark, maintainer, and
   example docs.
4. Check that package, ABI, platform, solver, and performance claims remain
   bounded by prior sprint decisions.
5. Run focused documentation validation for the split.
6. Write the Batch 2 implementation artifact.

### Deliverables
- completed or bounded algorithm split phase
- duplication cleanup
- updated cross-links
- claim-boundary scan notes
- residual algorithm docs queue

### Completion Criteria
- the selected split scope is implemented end to end
- link targets are coherent from adoption and maintainer entry points
- residual algorithm docs work is explicit rather than hidden in mixed-purpose
  pages

---

## Day 7: Compressed-First Cookbook Design

**Title:** Cookbook Design
**Theme:** Design concise compressed-first adoption paths across direct,
iterative, Matrix Market, SVD, eigensolver, and benchmark workflows
**Time estimate:** 12 hours

### Tasks
1. Inventory current compressed-first examples and documentation references.
2. Map the expected first-use path for direct solvers, iterative solvers,
   Matrix Market input, low-rank SVD, eigensolvers, and benchmark execution.
3. Identify gaps where users must infer compressed-first setup from scattered
   examples or maintainer notes.
4. Design cookbook structure, ordering, filenames, examples, and links.
5. Define validation for example paths and support-tier wording.
6. Write the compressed-first cookbook design artifact.

### Deliverables
- cookbook target structure
- workflow-by-workflow adoption map
- example and doc link inventory
- gap and rewrite queue
- validation plan

### Completion Criteria
- every requested compressed-first workflow has a planned adoption path
- cookbook scope is separated from exhaustive API reference material
- example links and support claims are ready for implementation

---

## Day 8: Compressed-First Cookbook Batch 1

**Title:** Cookbook Batch 1
**Theme:** Implement direct, iterative, and Matrix Market compressed-first
adoption paths
**Time estimate:** 12 hours

### Tasks
1. Add or reorganize cookbook guidance for compressed-first direct solver
   setup.
2. Add or reorganize cookbook guidance for compressed-first iterative solver
   setup.
3. Add or reorganize cookbook guidance for Matrix Market input and conversion
   paths.
4. Link maintained examples from the cookbook without duplicating source-level
   implementation details.
5. Run focused docs hygiene and path checks for touched cookbook links.
6. Write the Batch 1 cookbook implementation artifact.

### Deliverables
- direct solver compressed-first cookbook path
- iterative solver compressed-first cookbook path
- Matrix Market compressed-first cookbook path
- updated links to maintained examples
- focused validation evidence

### Completion Criteria
- first-use readers can find direct, iterative, and Matrix Market paths from
  adoption docs
- cookbook text stays concise and links to detailed references where needed
- no package or platform support claim is widened while reorganizing examples

---

## Day 9: Compressed-First Cookbook Batch 2

**Title:** Cookbook Batch 2
**Theme:** Implement SVD, eigensolver, and benchmark compressed-first adoption
paths
**Time estimate:** 12 hours

### Tasks
1. Add or reorganize cookbook guidance for compressed-first SVD workflows.
2. Add or reorganize cookbook guidance for compressed-first eigensolver
   workflows.
3. Add or reorganize cookbook guidance for compressed-first benchmark
   workflows and local measurement entry points.
4. Link maintained examples and benchmark docs from the cookbook.
5. Scan cookbook language for performance, backend, and report-interpretation
   overclaims.
6. Write the Batch 2 cookbook implementation artifact.

### Deliverables
- SVD compressed-first cookbook path
- eigensolver compressed-first cookbook path
- benchmark compressed-first cookbook path
- updated report and benchmark links
- claim-boundary scan notes

### Completion Criteria
- all requested compressed-first workflow families have concise adoption paths
- benchmark guidance points to measurement interpretation rather than implying
  broad performance guarantees
- maintained examples remain the source of executable details

---

## Day 10: Benchmark and Report Index Docs

**Title:** Report Index Docs
**Theme:** Surface generated report indexes and local-measurement
interpretation in concise adoption language
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 131 report index artifacts and current benchmark/report docs.
2. Identify generated report indexes that should be visible from adoption
   docs.
3. Write concise guidance for interpreting local measurement results,
   generated indexes, and benchmark artifacts.
4. Update benchmark docs, README links, or cookbook links as needed.
5. Preserve boundaries around backend, performance, package, and platform
   claims.
6. Write the benchmark/report index docs artifact.

### Deliverables
- report-index adoption guidance
- local-measurement interpretation notes
- updated benchmark or cookbook links
- support and performance claim-boundary notes
- residual report-docs queue

### Completion Criteria
- generated report indexes are discoverable from first-use docs
- benchmark interpretation text is concise and evidence-bounded
- report docs do not duplicate maintainer history or imply unsupported claims

---

## Day 11: Adoption Navigation Alignment

**Title:** Navigation Alignment
**Theme:** Align README, tutorial, install, cookbook, reference, report, and
maintainer navigation after the split
**Time estimate:** 12 hours

### Tasks
1. Re-audit top-level navigation from README and tutorial entry points.
2. Update adoption links so first-use paths lead to cookbook, concise
   reference, install, solver-selection, and report-index guidance in a
   predictable order.
3. Move maintainer-only or historical links out of first-use flows where they
   interrupt adoption.
4. Ensure install and package links preserve Sprint 133-134 support-tier truth.
5. Run focused link/path and whitespace checks for touched documentation.
6. Write the navigation alignment artifact.

### Deliverables
- updated adoption navigation
- first-use link order notes
- maintainer-history link placement notes
- install and package support-tier alignment notes
- focused validation evidence

### Completion Criteria
- README and tutorial entry points lead to simplified adoption paths
- historical and maintainer material remains findable without being the
  default first-use path
- package/platform support wording remains aligned with Sprint 133-134 truth

---

## Day 12: Link and Claim Validation

**Title:** Validation Sweep
**Theme:** Run documentation hygiene, link/path checks, and claim-boundary
scans across the productized adoption surface
**Time estimate:** 11 hours

### Tasks
1. Run documentation whitespace and diff hygiene checks.
2. Run available link/path scans for touched docs, examples, generated report
   references, and moved algorithm sections.
3. Scan README, install docs, cookbook, algorithm reference, benchmark docs,
   maintainer guide, and Sprint 135 artifacts for package/platform claim drift.
4. Scan benchmark and report language for unsupported performance claims.
5. Record any validation failures, fixes, or explicit residual risks.
6. Write the link and claim validation artifact.

### Deliverables
- documentation hygiene evidence
- link/path validation evidence
- package and platform claim-boundary scan
- performance and report-interpretation claim scan
- residual validation queue

### Completion Criteria
- touched adoption docs pass available hygiene and link/path checks
- support-tier and performance claims remain evidence-bounded
- unresolved validation gaps are named with owners and follow-up paths

---

## Day 13: Integrated Adoption Review

**Title:** Adoption Review
**Theme:** Review the simplified adoption surface as a first-use reader and a
maintainer
**Time estimate:** 11 hours

### Tasks
1. Walk the README-to-install-to-cookbook-to-reference path as a first-use
   reader.
2. Walk the benchmark/report-index path as a reader interpreting local
   measurement output.
3. Walk the maintainer-history and historical-appendix paths as a maintainer.
4. Verify compressed-first direct, iterative, Matrix Market, SVD, eigensolver,
   and benchmark paths are all discoverable.
5. Fix small navigation, wording, and duplication issues discovered during the
   integrated review.
6. Write the integrated adoption review artifact.

### Deliverables
- first-use adoption walkthrough notes
- benchmark/report walkthrough notes
- maintainer-history walkthrough notes
- compressed-first workflow discoverability matrix
- final small-fix queue or applied cleanup

### Completion Criteria
- adoption docs have a coherent reader path from first contact to examples
- maintainer and historical material is discoverable without crowding
  first-use guidance
- all compressed-first workflow families have visible entry points

---

## Day 14: Sprint Closeout

**Title:** Adoption Closeout
**Theme:** Publish adoption simplification metrics, validation summary, and
residual documentation work
**Time estimate:** 10 hours

### Tasks
1. Summarize adoption simplification outcomes, moved content, new links, and
   residual documentation work.
2. Record metrics such as audited surfaces, split documents, cookbook paths,
   report links surfaced, and validation checks run.
3. Reconcile Sprint 135 artifacts against project-plan Items 1-7.
4. Confirm package/platform/performance/support-tier claim boundaries one last
   time.
5. Write the closeout and Sprint 136 handoff artifact.
6. Prepare the retrospective input queue.

### Deliverables
- Sprint 135 closeout artifact
- adoption simplification metrics
- validation summary
- residual documentation queue
- Sprint 136 handoff notes

### Completion Criteria
- all Sprint 135 deliverables are represented by artifacts or explicit
  residual decisions
- validation evidence and claim-boundary status are summarized
- next-sprint documentation risks are visible and actionable
