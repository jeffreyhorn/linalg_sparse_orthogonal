# Sprint 70 Plan: Epic 7 Baseline, State-of-the-Art Target & Architecture Contract

**Sprint Duration:** 14 days  
**Goal:** Freeze the post-Epic-6 baseline, define the real Epic 7
state-of-the-art target, and fix the product-model, capability, and
validation/platform non-goal fence before implementation starts. This sprint
implements the Sprint 70 section of
`docs/planning/EPIC_7/PROJECT_PLAN.md`.

**Starting Point:** Epic 6 closed with the library in a strong but still
bounded state:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- the repo now has a coherent public product story, bounded backend-aware
  performance surfaces, canonical benchmark governance, and a truthful
  packaging/platform contract
- the strongest remaining ceilings are structural rather than foundational:
  core product-model convergence, capability breadth, residual configuration
  modernization, deeper performance maturity, platform/release convergence,
  and remaining large-source/test debt
- the next priority is not another isolated subsystem sprint; it is freezing
  the exact Epic 7 target and non-goal fence before implementation starts

The highest-value next work is not feature delivery. It is an audit and
planning sprint focused on reconfirming the live baseline, mapping the largest
remaining product-model and capability gaps, freezing the final validation and
platform contract, and turning the Epic 7 review into an implementation-ready
architecture contract.

**End State:** Sprint 70 leaves behind one coherent Epic 7 starting package:
- a revalidated post-Epic-6 baseline and rerun contract
- an explicit state-of-the-art target and non-goal fence
- a ranked product-model, capability, public-surface, and proof-surface gap map
- a frozen validation/platform/install truthfulness contract for Epic 7
- closeout artifacts and working notes that let Sprint 71 start from a sharp
  implementation boundary rather than another broad review pass

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
152 hours, matching the Sprint 70 estimate and staying below the 168-hour
limit.

---

## Day 1: Sprint 70 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 70 project-plan scope plus the Epic 6 closeout into
a bounded Epic 7 baseline and audit map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 70 section of
   `docs/planning/EPIC_7/PROJECT_PLAN.md`, the Epic 6 retrospective, the Epic
   6 review, and the strongest Epic 6 closeout artifacts.
2. Reconfirm the preserved Sprint 70 constraints:
   - no fake “state-of-the-art” claims not grounded in the live repo
   - no broad implementation work disguised as baseline setup
   - no reopening solved Epic 6 lanes unless a touched audit seam truly forces
     it
   - no weakening of the reviewed truthfulness contract
3. Define the Sprint 70 workstreams explicitly:
   - baseline recheck
   - product-model gap inventory
   - capability ceiling audit
   - public-surface and proof-surface audit
   - validation and platform contract freeze
   - closeout and handoff
4. Record the strongest likely Sprint 70 touch surfaces:
   - README, maintainer guide, install, workflows, and public headers
   - core matrix and direct-solver implementation seams
   - benchmark/reporting and install/package proof surfaces
   - project-level Epic 7 planning outputs
5. Open Sprint 70 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 70 scope inventory
- Epic 7 baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 70 starts from the Epic 6 validated close rather than reopening older
  subsystem work
- The baseline, audit, and contract-freeze workstreams are explicit before
  deeper audit begins
- The sprint non-goal fence is fixed before design or implementation planning
  proceeds

---

## Day 2: Validation Baseline & Rerun Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed baseline and rerun set that Epic 7 planning
and later implementation sprints must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity counts
   - current quality/truthfulness wording
2. Reconfirm the mandatory gate for later `*.c` / `*.h` days:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial architecture, capability,
   benchmark-governance, or platform work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Epic 7:
   - core integration and family-local proof surfaces
   - examples and maintained benchmark/reporting surfaces
   - install/package and canonical report surfaces
   - large-source and giant-test owners most likely to move later
5. Record the authoritative validation split for docs-only, bounded code-day,
   and substantial integration days.

### Deliverables
- Refreshed validation notes
- Epic 7 rerun list
- Final code-day validation checklist

### Completion Criteria
- Sprint 70 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before deeper audits begin
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Product-Model Gap Inventory I

**Title:** Product-Model Audit I  
**Theme:** Audit the strongest remaining linked-list/product-model and
conversion-heavy workflow seams in the live library  
**Time estimate:** 12 hours

### Tasks
1. Audit the public and implementation surfaces that define the current matrix
   and factor/workspace story:
   - `README.md`
   - `include/sparse_matrix.h`
   - `src/sparse_matrix.c`
   - strongest direct CSC/CSR-backed solver paths
2. Classify the largest remaining product-model tensions:
   - mutable linked-list construction versus compressed numeric work
   - repeated conversion or publication friction
   - caller ownership and lifecycle ambiguity
   - mutation/copy surprise on advanced workflows
3. Record where the current public model remains a strength and where it reads
   more like compatibility burden.
4. Rank the strongest current workflow seams by:
   - user cost
   - performance cost
   - implementation risk
   - likely Epic 7 payoff
5. Write the first product-model audit artifact.

### Deliverables
- Initial product-model gap inventory
- Ranked workflow and ownership-tension list
- First product-model hotspot map

### Completion Criteria
- The broad Epic 7 product-model problem is reduced to a concrete file and
  ownership map
- The strongest conversion-heavy and mutation-heavy seams are explicit before
  rerank/design work
- Day 4 can proceed from a real current-state product-model ranking

---

## Day 4: Product-Model Gap Inventory II & First Boundary

**Title:** Product-Model Audit II  
**Theme:** Rerank the product-model seams and freeze the first architecture
boundary that later Epic 7 implementation should respect  
**Time estimate:** 12 hours

### Tasks
1. Re-rank the Day 3 product-model surfaces against:
   - user-facing workflow importance
   - numeric-path performance cost
   - compatibility burden
   - proof burden
2. Separate:
   - first-phase convergence targets
   - later capability-adjacent targets
   - residual compatibility surfaces that should stay deferred
3. Identify the strongest first product-model implementation fence for Epic 7:
   - mutable matrix state
   - compressed working formats
   - factor/workspace ownership
   - publication and round-trip boundaries
4. Record the strongest non-goals:
   - no broad rewrite of every matrix entry point
   - no fake abstraction layer added only for aesthetics
   - no capability-surface widening disguised as product-model cleanup
5. Fix the Day 4 product-model boundary in writing.

### Deliverables
- Refined product-model ranking
- First architecture boundary
- Deferred compatibility/residual map

### Completion Criteria
- The product-model target set is smaller and sharper than the original Epic 7
  review wording
- The first architecture boundary is explicit before later sprints design code
  changes
- Lower-value or higher-risk product-model cleanup is clearly separated from
  the first Epic 7 lane

---

## Day 5: Capability Ceiling Audit I

**Title:** Capability Audit I  
**Theme:** Re-rank the strongest current capability ceilings by product impact
and implementation risk  
**Time estimate:** 10 hours

### Tasks
1. Audit the live capability boundaries across:
   - `idx_t` and index-width assumptions
   - scalar-type assumptions
   - eigensolver scope
   - public API and packaging implications
2. Identify the strongest current capability ceilings:
   - 32-bit index limits
   - real-only scalar support
   - algorithm-surface omissions that most affect state-of-the-art positioning
3. Distinguish between:
   - public product caveats
   - implementation-local assumptions
   - build/package compatibility implications
4. Rank the highest-value first capability seams for later Epic 7 work.
5. Capture the first capability audit artifact.

### Deliverables
- Initial capability inventory
- Ranked capability-ceiling list
- First modernization shortlist

### Completion Criteria
- The broad “capability expansion” goal is reduced to a concrete ranked ceiling
  list
- The strongest product-facing and implementation-facing capability burdens are
  explicit
- Day 6 can proceed from a real current-state capability map

---

## Day 6: Capability Ceiling Audit II & Modernization Fence

**Title:** Capability Audit II  
**Theme:** Separate the first realistic Epic 7 capability modernization lane
from larger deferred ambitions  
**Time estimate:** 12 hours

### Tasks
1. Re-rank the Day 5 capability ceilings against user value, ecosystem impact,
   implementation risk, and proof burden.
2. Separate:
   - first bounded modernization candidates
   - medium-term capability targets
   - explicitly deferred capability ambitions
3. Identify the strongest exact first modernization fence:
   - index-width seam candidates
   - scalar-surface preparation candidates
   - docs/package/test implications
4. Fix the strongest capability non-goals:
   - no claim of full type-generic conversion in one sprint
   - no fake API expansion without end-to-end proof
   - no broad algorithm wishlist disconnected from the live library shape
5. Record the Day 6 capability fence in writing.

### Deliverables
- Refined capability ranking
- First modernization fence
- Deferred capability queue

### Completion Criteria
- The capability target set is smaller and sharper than the original review
  language
- The first modernization fence is explicit before implementation planning
  continues
- Deferred capability ambitions are separated cleanly from the likely first
  landing

---

## Day 7: Public-Surface and Proof-Surface Audit I

**Title:** Surface Audit I  
**Theme:** Map the strongest remaining documentation, header, benchmark, and
giant-test contradictions that later Epic 7 work must respect  
**Time estimate:** 10 hours

### Tasks
1. Audit the strongest maintained public and proof surfaces across:
   - README, tutorial, install, examples, and benchmark docs
   - maintainer guide and key public headers
   - giant proof-owner tests and canonical benchmark/report surfaces
2. Classify each surface by:
   - user-facing product-story importance
   - policy-density and chronology burden
   - contradiction risk
   - likely Epic 7 cleanup payoff
3. Identify the strongest current contradictions:
   - duplicate or drifted product explanations
   - header wording that implies broader maturity than the repo validates
   - examples/benchmarks/tests ownership drift
   - giant proof surfaces that still carry excess history burden
4. Rank the most valuable later Epic 7 public-surface and proof-surface
   candidates.
5. Write the first surface audit artifact.

### Deliverables
- Public/proof-surface inventory
- Ranked contradiction list
- Initial cleanup shortlist

### Completion Criteria
- The broad public-surface/proof-surface problem is reduced to a concrete
  contradiction map
- The strongest user-facing and proof-facing drift points are explicit before
  contract freeze
- Day 8 can proceed from a real current-state surface ranking

---

## Day 8: Public-Surface and Proof-Surface Audit II

**Title:** Surface Audit II  
**Theme:** Separate the highest-value future cleanup lanes from lower-value
surface churn and freeze the support-surface fence  
**Time estimate:** 10 hours

### Tasks
1. Re-rank the Day 7 surface set against the Epic 7 target and the product,
   capability, and benchmark stories now in force.
2. Separate:
   - public front-door cleanup candidates
   - header/reference cleanup candidates
   - benchmark/proof-surface cleanup candidates
   - lower-value residuals that should remain deferred
3. Confirm which files likely belong in:
   - public product-surface cleanup
   - policy/reference cleanup
   - giant-test or benchmark-maintainer follow-through only
4. Fix the likely later support-surface fence in writing.
5. Record the residual queue that Sprint 70 should not absorb.

### Deliverables
- Refined surface ranking
- Support-surface fence
- Deferred surface-cleanup queue

### Completion Criteria
- The future cleanup target set is smaller and sharper than the initial audit
- The support-surface fence is explicit before the final architecture contract
  is written
- Lower-value surface churn is clearly separated from the likely Epic 7 lanes

---

## Day 9: Validation and Platform Contract I

**Title:** Contract Audit I  
**Theme:** Reconfirm the live validation, install, packaging, and platform
truthfulness contract that Epic 7 must preserve  
**Time estimate:** 10 hours

### Tasks
1. Audit the current contract across:
   - `make quality-review-full`
   - CMake parity and reviewed test counts
   - install/package regression surfaces
   - canonical benchmark/report surfaces
   - Linux, macOS, and Windows workflow wording
2. Identify the strongest remaining truthfulness sensitivities:
   - platform reviewed-subset asymmetry
   - install/package proof ownership
   - dead-code and supplemental-lane staging
   - benchmark/report interpretation boundaries
3. Confirm what later Epic 7 work can move without reopening fake platform or
   release claims.
4. Rank the highest-risk contract contradictions to preserve against future
   work.
5. Capture the first validation/platform contract artifact.

### Deliverables
- Validation/platform contract inventory
- Ranked truthfulness-sensitivity list
- First contract-preservation notes

### Completion Criteria
- The live truthfulness contract is explicit before Sprint 70 closeout
- The strongest platform/install/reporting sensitivities are named directly
- Day 10 can proceed from a real contract-preservation map

---

## Day 10: Validation and Platform Contract II

**Title:** Contract Audit II  
**Theme:** Freeze the exact validation, package, and platform non-goal fence
for later Epic 7 implementation sprints  
**Time estimate:** 10 hours

### Tasks
1. Re-rank the Day 9 contract surfaces against Epic 7 implementation risk.
2. Separate:
   - maintained reviewed truth
   - supplemental-but-truthful proof lanes
   - explicitly deferred platform or release ambitions
3. Fix the strongest contract non-goals:
   - no fake shared-library maturity claims
   - no inflated Windows parity claims
   - no benchmark or example wording that steals test-owned guarantees
   - no install/package story beyond reviewed evidence
4. Record the exact contract-preservation checklist for later Epic 7 sprints.
5. Freeze the Day 10 validation/platform fence in writing.

### Deliverables
- Refined contract ranking
- Validation/platform non-goal fence
- Later-sprint preservation checklist

### Completion Criteria
- Later Epic 7 work has an explicit truthfulness fence before implementation
  starts
- Deferred platform and release ambitions are separated cleanly from maintained
  reviewed truth
- The benchmark/install/platform contract is explicit enough to prevent drift

---

## Day 11: State-of-the-Art Target Synthesis

**Title:** Target Synthesis  
**Theme:** Turn the baseline and audit outputs into one explicit Epic 7
state-of-the-art target and success rubric  
**Time estimate:** 10 hours

### Tasks
1. Synthesize the Day 3-10 audits into one explicit Epic 7 target model across:
   - product model
   - capability surface
   - performance maturity
   - public/product documentation
   - platform/release maturity
   - assurance and proof architecture
2. Distinguish between:
   - true “state-of-the-art” target qualities
   - strong-but-bounded target qualities
   - intentionally deferred beyond-Epic-7 ambitions
3. Define the strongest Epic 7 success criteria and anti-goals.
4. Record the final ranked carry-in queue for Sprints 71-79.
5. Capture the target-synthesis artifact.

### Deliverables
- Explicit Epic 7 target statement
- Success-criteria and anti-goal list
- Ranked carry-in queue for later sprints

### Completion Criteria
- Epic 7 has one explicit target rather than broad aspirational wording
- The difference between target quality and deferred ambition is written down
- The later sprint sequence can be grounded in a real success rubric

---

## Day 12: Architecture Contract & Non-Goal Fence

**Title:** Architecture Contract  
**Theme:** Write the final product-model, capability, validation, and
platform-preservation contract that later Epic 7 sprints must obey  
**Time estimate:** 12 hours

### Tasks
1. Consolidate the Sprint 70 findings into one architecture and contract
   artifact.
2. Fix the exact preserved boundaries for:
   - product-model convergence
   - capability modernization
   - configuration modernization
   - backend/performance work
   - benchmark governance
   - packaging/platform truthfulness
3. Fix the strongest cross-cutting non-goals:
   - no broad rewrite of the library core
   - no unsupported product marketing claims
   - no capability promises without end-to-end proof
   - no drift from maintained reviewed validation truth
4. Record the exact future-sprint planning implications.
5. Review the draft contract against the Epic 7 project plan and earlier audit
   artifacts.

### Deliverables
- Sprint 70 architecture contract artifact
- Cross-cutting non-goal fence
- Later-sprint planning implications

### Completion Criteria
- Epic 7 has one explicit architecture contract before implementation starts
- The non-goal fence is concrete enough to keep later sprints bounded
- The contract matches the live repo and the Epic 7 plan rather than generic
  aspirations

---

## Day 13: Sprint 70 Baseline Package Review

**Title:** Baseline Package Review  
**Theme:** Re-read the full Sprint 70 package for coherence, gaps, and
handoff quality before closeout  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 70 working notes and all audit/contract artifacts.
2. Confirm that the Sprint 70 package covers:
   - baseline recheck
   - product-model gap inventory
   - capability ceiling audit
   - public/proof-surface audit
   - validation/platform contract freeze
   - state-of-the-art target synthesis
3. Check for contradictions between the Sprint 70 artifacts and:
   - `docs/planning/EPIC_7/PROJECT_PLAN.md`
   - the Epic 7 review and todo
   - the live repo wording on product/package/platform truth
4. Tighten any remaining wording drift in the planning artifacts only.
5. Record the final Day 13 review notes and the Day 14 handoff queue.

### Deliverables
- Coherence review notes
- Final Sprint 70 artifact checklist
- Day 14 handoff queue

### Completion Criteria
- Sprint 70 artifacts agree with each other and with the Epic 7 project plan
- No unresolved contradiction remains in the baseline package before closeout
- Day 14 can close from a coherent reviewed planning state

---

## Day 14: Sprint 70 Closeout & Handoff

**Title:** Closeout and Handoff  
**Theme:** Close Sprint 70 with one explicit Epic 7 starting contract and a
clear carry-forward package for Sprint 71  
**Time estimate:** 12 hours

### Tasks
1. Write the Sprint 70 closeout artifact summarizing what was fixed in
   planning:
   - baseline
   - target
   - architecture contract
   - non-goal fence
   - validation/platform truthfulness contract
2. Rank the strongest carry-forward items for Sprint 71 and beyond:
   - de-chronologization
   - product-model convergence
   - capability modernization
   - backend/performance maturity
   - packaging/platform convergence
3. Recheck whether `docs/planning/EPIC_7/PROJECT_PLAN.md` needs any Sprint 70
   correction after the deeper baseline work.
4. Confirm the final Sprint 70 branch state and documentation footprint.
5. Record the final handoff notes for the next sprint.

### Deliverables
- Sprint 70 closeout artifact
- Ranked Sprint 71 carry-forward queue
- Final handoff notes

### Completion Criteria
- Sprint 70 ends with one explicit Epic 7 starting contract rather than a
  loose review bundle
- The next sprint can begin from a ranked, bounded carry-forward queue
- Any project-plan correction need is explicitly resolved before handoff
