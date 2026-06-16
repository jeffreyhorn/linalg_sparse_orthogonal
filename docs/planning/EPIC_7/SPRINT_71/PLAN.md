# Sprint 71 Plan: Public Surface De-chronologization & Reference Cleanup

**Sprint Duration:** 14 days  
**Goal:** Remove the strongest remaining sprint-history and policy-overload
from public docs and reference surfaces so the repo reads more like a mature
library and less like a sprint archive. This sprint implements the Sprint 71
section of `docs/planning/EPIC_7/PROJECT_PLAN.md`.

**Starting Point:** Sprint 70 closed with an explicit Epic 7 starting
contract:
- the strongest local reviewed baseline remains `make quality-review-full`
- the public cleanup queue is now ranked ahead of product-model and capability
  implementation work
- `README.md` and `INSTALL.md` remain the strongest public contradiction
  centers
- `include/sparse_cholesky.h` remains the strongest header/reference cleanup
  candidate
- `docs/tutorial.md`, `examples/README.md`, `benchmarks/README.md`, and
  `docs/maintainer_guide.md` remain support-only surfaces unless first-batch
  cleanup truly forces them to move
- the Sprint 70 non-goal fence is already fixed:
  - no broad rewrite of the library core
  - no unsupported marketing claims
  - no capability or platform widening without proof
  - no generic cleanup campaign detached from ranked seams

The highest-value Sprint 71 work is therefore not implementation change. It is
bounded public/reference cleanup focused on removing chronology and duplicated
policy spill from the strongest permanent user-facing surfaces while
preserving the truthfulness, ownership, and validation fences fixed in Sprint
70.

**End State:** Sprint 71 leaves behind a cleaner public/reference package:
- a re-ranked public-surface and header/reference cleanup map
- a bounded front-door and install cleanup batch
- a bounded public-header narrative cleanup batch
- a reconciled tutorial/example/benchmark support package
- maintainer-guide authority preserved with less duplicated policy elsewhere
- a Sprint 71 closeout package that lets Sprint 72 start from cleaner public
  and reference surfaces without reopening the Sprint 70 architecture fence

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
158 hours, matching the Sprint 71 estimate and staying below the 168-hour
limit.

---

## Day 1: Sprint 71 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 71 project-plan scope plus the Sprint 70 closeout
into a bounded public-surface cleanup sprint  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 71 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Sprint 70 retrospective, and the Sprint 70 closeout artifact.
2. Reconfirm the preserved Sprint 71 constraints:
   - no reopening of the Sprint 70 architecture contract
   - no implementation or behavior work disguised as public cleanup
   - no widening of platform, packaging, or benchmark claims
   - no generic repo-wide chronology cleanup campaign
3. Define the Sprint 71 workstreams explicitly:
   - public-surface history audit
   - front-door and install cleanup
   - public header narrative cleanup
   - tutorial/example/benchmark support-surface reconciliation
   - maintainer-guide re-centering
   - truth-surface review and closeout
4. Record the strongest likely Sprint 71 touch surfaces:
   - `README.md`
   - `INSTALL.md`
   - `include/sparse_cholesky.h`
   - likely support surfaces from Sprint 70
5. Open Sprint 71 working notes and record the intended landing order,
   required artifacts, and docs-only validation expectations.

### Deliverables
- Sprint 71 scope inventory
- Public-surface cleanup map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 71 starts from the Sprint 70 architecture contract rather than
  reopening broad Epic 7 planning
- The cleanup workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or landing work proceeds

---

## Day 2: Validation Baseline & Truth-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the docs-only validation contract and the truth surfaces
that Sprint 71 cleanup must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 71 authority split:
   - docs-only days use targeted sanity checks only
   - any future `*.c` / `*.h` work would still require
     `make format`, `make lint`, and `make test`
3. Recheck the live truth surfaces Sprint 71 must not distort:
   - README / INSTALL product claims
   - maintainer-guide policy claims
   - benchmark/report wording
   - platform/install wording
4. Refresh the targeted doc-sanity set most likely to matter in Sprint 71:
   - diff review
   - terminology/alignment scans
   - touched-surface `wc -l`
   - branch-state rechecks
5. Record the authoritative docs-only validation split in the working notes.

### Deliverables
- Refreshed validation notes
- Sprint 71 doc-sanity checklist
- Preserved truth-surface checklist

### Completion Criteria
- Sprint 71 uses the same reviewed/truthfulness reading fixed in Sprint 70
- The docs-only sanity contract is explicit before cleanup starts
- No validation ambiguity remains around touched public/reference surfaces

---

## Day 3: Public-Surface History Audit I

**Title:** Public Audit I  
**Theme:** Re-rank the strongest remaining chronology and policy-density seams
across user-facing public docs  
**Time estimate:** 12 hours

### Tasks
1. Audit the current top public surfaces:
   - `README.md`
   - `INSTALL.md`
   - `docs/tutorial.md`
   - `examples/README.md`
   - `benchmarks/README.md`
2. Classify the strongest remaining burdens:
   - sprint-history spill
   - policy duplication
   - installation or workflow over-explanation
   - ownership explanation repeated in too many places
3. Record where the current docs are already strong and where they still read
   more like delivery archives than durable product/reference docs.
4. Rank the strongest public contradiction centers by:
   - user-facing cost
   - readability cost
   - duplication density
   - likely Sprint 71 payoff
5. Write the first public-surface audit artifact.

### Deliverables
- Initial public-surface audit
- Ranked chronology/policy-density list
- First public cleanup hotspot map

### Completion Criteria
- The broad Sprint 71 public-cleanup problem is reduced to a concrete file
  ranking
- The strongest user-facing chronology and duplication seams are explicit
- Day 4 can proceed from a real current-state public-surface ranking

---

## Day 4: Public-Surface History Audit II & First Landing Boundary

**Title:** Public Audit II  
**Theme:** Refine the public-surface ranking and freeze the first cleanup
boundary for the sprint  
**Time estimate:** 12 hours

### Tasks
1. Re-rank the Day 3 public surfaces against:
   - top-level user value
   - duplication density
   - truthfulness sensitivity
   - likely cleanup leverage
2. Separate:
   - first-batch landing surfaces
   - support surfaces that may move only if the first batch forces it
   - lower-value or explicitly deferred public cleanup surfaces
3. Identify the strongest first Sprint 71 cleanup fence:
   - front door
   - install story
   - reference header center
   - support-surface follow-through
4. Record the strongest non-goals:
   - no repo-wide user-doc rewrite
   - no benchmark, test, or platform claim widening
   - no public-header cleanup wave without a ranked center
5. Fix the Day 4 public-surface boundary in writing.

### Deliverables
- Refined public-surface ranking
- First cleanup boundary
- Deferred/support-surface map

### Completion Criteria
- The first Sprint 71 landing fence is explicit before design begins
- Lower-value or higher-risk cleanup is clearly separated from the first lane
- Support surfaces are bounded rather than assumed

---

## Day 5: Front-Door & Install Cleanup Design

**Title:** Product-Story Design  
**Theme:** Define the bounded cleanup contract for `README.md` and
`INSTALL.md` before edits land  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 70 truthfulness and non-goal fences against:
   - `README.md`
   - `INSTALL.md`
2. Design the first cleanup batch around:
   - compact product-story front door
   - stable release/install claims only
   - reduced policy duplication
   - preserved example/benchmark/test ownership wording
3. Decide what remains in place and what must move to support or maintainer
   surfaces instead.
4. Fix the exact first-batch non-touch set:
   - public headers
   - implementation files
   - proof-owner tests
   - platform workflow surfaces
5. Record the Day 5 design artifact.

### Deliverables
- Front-door/install cleanup design
- First-batch non-touch set
- Preserved claim/authority checklist

### Completion Criteria
- The first cleanup batch is explicitly designed before edits begin
- README and INSTALL cleanup is bounded to stable product/release claims
- The authority split for support and policy text is fixed in writing

---

## Day 6: Front-Door & Install Cleanup Batch

**Title:** Product-Story Batch  
**Theme:** Land the highest-value public cleanup on the front-door and install
surfaces without widening claims  
**Time estimate:** 12 hours

### Tasks
1. Edit `README.md` to reduce the strongest remaining chronology and duplicated
   policy spill.
2. Edit `INSTALL.md` to keep the install/release story concise and stable.
3. Preserve the Sprint 70 truthfulness fence across:
   - static-first packaging
   - reviewed vs supplemental proof
   - examples vs benchmarks vs tests ownership
4. Record the touched-surface notes and the landed batch artifact.
5. Run the targeted docs-only sanity set:
   - diff review
   - terminology/alignment scans
   - touched-surface `wc -l`
   - branch-status recheck

### Deliverables
- Landed README/INSTALL cleanup batch
- Updated working notes
- Batch artifact with preserved-truth checklist

### Completion Criteria
- The strongest public contradiction centers are materially cleaner
- No header, implementation, or workflow surface widens unexpectedly
- The front-door/install story remains truthful and more product-like

---

## Day 7: Post-Landing Audit & Header/Support Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank what is left after the first docs batch and fix the next
cleanup center  
**Time estimate:** 10 hours

### Tasks
1. Re-read the post-Day-6 public surfaces and compare them against the Day 5
   design assumptions.
2. Decide whether the strongest next seam is:
   - `include/sparse_cholesky.h`
   - tutorial/example/benchmark support drift
   - maintainer-guide recentering
3. Separate:
   - required next batch
   - support-only follow-through
   - surfaces that remain explicitly deferred
4. Confirm that the Sprint 70 contract still holds after the first landing.
5. Record the rerank artifact and Day 8 target.

### Deliverables
- Post-landing audit notes
- Updated cleanup ranking
- Exact next-batch target

### Completion Criteria
- The next cleanup center is fixed from the post-Day-6 state
- No fake second batch is implied where the first landing already solved the
  strongest contradiction
- Support/deferred surfaces remain explicit

---

## Day 8: Public Header Narrative Cleanup Design

**Title:** Header Design  
**Theme:** Define the bounded narrative-cleanup contract for the strongest
remaining public header surface  
**Time estimate:** 12 hours

### Tasks
1. Re-read `include/sparse_cholesky.h` against the Sprint 70 fences and the
   post-Day-7 rerank.
2. Design the bounded cleanup around:
   - API-local truth only
   - reduced sprint-history and ABI-history spill
   - preserved backend/error semantics
   - preserved benchmark/test/reference ownership reading
3. Decide whether any support surface must move with the header:
   - tutorial
   - examples
   - benchmarks
   - maintainer guide
4. Fix the exact non-touch set for the header batch.
5. Record the Day 8 design artifact.

### Deliverables
- Header-cleanup design
- Support-only follow-through map
- Header-batch non-touch set

### Completion Criteria
- The header cleanup is explicitly bounded before edits begin
- API-local caveats remain intact while avoidable chronology is identified
- Support surfaces move only if genuinely forced

---

## Day 9: Public Header Narrative Cleanup Batch

**Title:** Header Batch  
**Theme:** Land the bounded public-header cleanup on the strongest reference
surface  
**Time estimate:** 12 hours

### Tasks
1. Edit `include/sparse_cholesky.h` to remove the densest non-essential
   chronology and policy spill.
2. Preserve the local reference truth:
   - public direct-workflow interpretation
   - backend-contract semantics
   - retained proof-owner relationships
3. Land any required support-only follow-through if the Day 8 design proved it
   necessary.
4. Record the touched-surface notes and the landed batch artifact.
5. Run the targeted docs-only sanity set:
   - diff review
   - terminology/alignment scans
   - touched-surface `wc -l`
   - branch-status recheck

### Deliverables
- Landed header cleanup batch
- Updated working notes
- Batch artifact with preserved-reference-truth checklist

### Completion Criteria
- The strongest header/reference contradiction center is materially cleaner
- The cleaned header remains technically precise and API-local
- No unrelated public-header cleanup wave is triggered

---

## Day 10: Tutorial / Example / Benchmark Cross-Surface Design

**Title:** Support Design  
**Theme:** Define the bounded support-surface reconciliation needed after the
front-door and header cleanup  
**Time estimate:** 12 hours

### Tasks
1. Re-read:
   - `docs/tutorial.md`
   - `examples/README.md`
   - `benchmarks/README.md`
   against the landed Day 6 and Day 9 state.
2. Identify the strongest remaining support drift:
   - repeated-run teaching flow
   - examples vs benchmarks vs tests ownership
   - benchmark-side proof interpretation
3. Decide the exact support batch boundary and support-only non-touch set.
4. Reconfirm that `docs/maintainer_guide.md` remains policy authority rather
   than the primary cleanup center unless the support batch forces it.
5. Record the Day 10 design artifact.

### Deliverables
- Support-surface reconciliation design
- Exact support-batch boundary
- Maintainer-guide follow-through decision

### Completion Criteria
- The support cleanup is explicitly designed before edits begin
- The examples/benchmarks/tutorial split is preserved and sharpened
- Maintainer-guide movement is bounded by need, not by symmetry

---

## Day 11: Tutorial / Example / Benchmark Reconciliation Batch

**Title:** Support Batch  
**Theme:** Land the bounded teaching/proof support cleanup after the front-door
and header batches  
**Time estimate:** 12 hours

### Tasks
1. Edit the Day 10 support surfaces to reduce duplicated planning history and
   preserve the teaching/proof split.
2. Keep:
   - tutorial = teaching flow
   - examples = adoption entry points
   - benchmarks = retained workflow/performance proof
   - tests = regression/oracle/property guarantees
3. Land maintainer-guide follow-through only if the Day 10 design proved it
   necessary.
4. Record the touched-surface notes and the landed support-batch artifact.
5. Run the targeted docs-only sanity set:
   - diff review
   - terminology/alignment scans
   - touched-surface `wc -l`
   - branch-status recheck

### Deliverables
- Landed support-surface reconciliation batch
- Updated working notes
- Support-batch artifact with preserved-ownership checklist

### Completion Criteria
- The support surfaces agree with the cleaned front door and header
- The teaching/proof split is clearer without widening claims
- Any maintainer-guide movement stays bounded and justified

---

## Day 12: Maintainer Guide Re-centering & Truth-Surface Review

**Title:** Policy Review  
**Theme:** Re-center policy authority and recheck all touched truth surfaces
before closeout  
**Time estimate:** 10 hours

### Tasks
1. Re-read the touched Sprint 71 public/support surfaces against:
   - `docs/maintainer_guide.md`
   - `README.md`
   - `INSTALL.md`
   - benchmark/test ownership language
2. Tighten the maintainer guide only where policy authority or deferred
   rationale must move back out of user-facing surfaces.
3. Confirm that no contradiction remains across:
   - public product story
   - install/release story
   - benchmark/test/example authority split
   - Sprint 70 truthfulness fence
4. Record the truth-surface review artifact.
5. Reconfirm the exact Day 13-14 closeout and retrospective set.

### Deliverables
- Maintainer-guide recentering notes
- Truth-surface review artifact
- Final closeout queue

### Completion Criteria
- Policy authority is clearer and less duplicated
- No unresolved contradiction remains before closeout
- Day 13 can proceed from a stable Sprint 71 package

---

## Day 13: Sprint 71 Package Review & Closeout Prep

**Title:** Package Review  
**Theme:** Re-read the full Sprint 71 package for coherence, gaps, and handoff
quality before closeout  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 71 working notes and all audit/design/landing artifacts.
2. Confirm that the Sprint 71 package covers:
   - public-surface history audit
   - front-door/install cleanup
   - public header narrative cleanup
   - tutorial/example/benchmark reconciliation
   - maintainer-guide re-centering
   - truth-surface review
3. Check for contradictions between the Sprint 71 artifacts and:
   - `docs/planning/EPIC_7/PROJECT_PLAN.md`
   - the Sprint 70 architecture contract
   - the live repo wording on product/install/benchmark/platform truth
4. Tighten any remaining wording drift in planning artifacts only.
5. Record the final Day 13 review notes and the Day 14 handoff queue.

### Deliverables
- Coherence review notes
- Final Sprint 71 artifact checklist
- Day 14 handoff queue

### Completion Criteria
- Sprint 71 artifacts agree with each other and with the Sprint 70 contract
- No unresolved contradiction remains in the package before closeout
- Day 14 can close from a coherent reviewed planning state

---

## Day 14: Sprint 71 Closeout & Handoff

**Title:** Closeout and Handoff  
**Theme:** Close Sprint 71 with one explicit cleaned public/reference package
and a clear carry-forward queue for Sprint 72  
**Time estimate:** 10 hours

### Tasks
1. Write the Sprint 71 closeout artifact summarizing what was fixed in:
   - public docs
   - install surface
   - header/reference cleanup
   - support-surface reconciliation
   - truth-surface review
2. Rank the strongest carry-forward items for Sprint 72 and beyond:
   - product-model convergence
   - configuration modernization
   - capability modernization
   - backend/performance maturity
   - later permanent-surface cleanup
3. Recheck whether `docs/planning/EPIC_7/PROJECT_PLAN.md` needs any Sprint 71
   correction after the deeper public-surface cleanup work.
4. Confirm the final Sprint 71 branch state and documentation footprint.
5. Record the final handoff notes for the next sprint.

### Deliverables
- Sprint 71 closeout artifact
- Ranked Sprint 72 carry-forward queue
- Final handoff notes

### Completion Criteria
- Sprint 71 ends with a cleaner public/reference package rather than a loose
  set of doc edits
- Sprint 72 can begin from a ranked, bounded carry-forward queue
- Any project-plan correction need is explicitly resolved before handoff
