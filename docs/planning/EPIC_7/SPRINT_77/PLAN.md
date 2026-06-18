# Sprint 77 Plan: Packaging, ABI & Cross-Platform Quality Convergence Phase 2

**Sprint Duration:** 14 days  
**Goal:** Reduce the remaining release/install/platform asymmetries that still
limit the repo's maturity as a broadly deployable sparse library. This sprint
implements the Sprint 77 section of
`docs/planning/EPIC_7/PROJECT_PLAN.md`.

**Starting Point:** Sprint 76 closed with a validated benchmark-governance
package and preserved the bounded Epic 7 truthfulness fence. Sprint 77 starts
from the strongest remaining release/install/platform pressure:
- the strongest local reviewed baseline remains `make quality-review-full`
- Sprint 66 fixed the static-first packaging and platform truth contract
- Sprint 70 fixed the Epic 7 architecture and non-goal fence
- Sprint 74 preserved bounded capability modernization without widening the
  release claim surface
- Sprint 76 did not change the package/install/platform contract, so Sprint 77
  can focus directly on:
  - release-surface re-audit
  - platform-gap rerank
  - highest-value packaging/productization improvements
  - bounded Windows/macOS proof follow-through
  - workflow and contract reconciliation
  - install/package/platform regression coverage
  - validation and closeout
- the preserved Epic 7 non-goal fence is still in force:
  - no fake shared-library or dynamic-ABI maturity claim
  - no broader cross-platform review claim than maintained evidence supports
  - no widened install-validation truth detached from actual proof
  - no packaging rewrite disguised as “quality convergence”

The highest-value Sprint 77 work is therefore not a broad release overhaul.
It is a bounded packaging/platform convergence phase on the maintained
static-first contract, with clearer downstream-consumer guidance, narrower
platform asymmetry on the highest-value lanes, and stronger proof alignment
across maintained commands, docs, and install validation.

**End State:** Sprint 77 leaves behind a smaller and more coherent
release/install/platform gap:
- a refreshed release-surface and ABI claim audit
- one explicit ranked platform-gap map
- one landed bounded packaging/productization batch
- stronger Windows/macOS proof or supplemental confidence where justified
- aligned CI/workflow/docs reading of the maintained platform contract
- a validated Sprint 77 closeout package that lets Sprint 78 start from a
  stronger packaging/platform baseline instead of a mixed release backlog

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
156 hours, staying within the day-cap limit while covering the Sprint 77
implementation scope.

---

## Day 1: Sprint 77 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 77 project-plan scope plus the Sprint 66, Sprint
70, and Sprint 76 handoff into a bounded packaging/platform sprint  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 77 section of
   `docs/planning/EPIC_7/PROJECT_PLAN.md`, the Sprint 66 platform closeout,
   the Sprint 70 architecture contract, and the Sprint 76 closeout artifact.
2. Reconfirm the preserved Sprint 77 constraints:
   - no fake shared-library maturity claim
   - no widened platform-review claim without proof
   - no packaging rewrite that breaks the maintained static-first contract
   - no capability or backend reopening unless the release surface truly
     forces it
3. Define the Sprint 77 workstreams explicitly:
   - release-surface re-audit
   - platform-gap rerank
   - packaging/productization batch
   - Windows/macOS proof follow-through
   - workflow and contract reconciliation
   - regression-coverage follow-through
   - validation and closeout
4. Record the strongest likely Sprint 77 touch surfaces:
   - install/package documentation
   - build and export metadata
   - CI/workflow contract surfaces
   - install/package/platform proof scripts
5. Open Sprint 77 working notes and record the intended landing order,
   required artifacts, and validation expectations.

### Deliverables
- Sprint 77 scope inventory
- Release/install/platform workstream map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 77 starts from the Sprint 66, Sprint 70, and Sprint 76 contract
  rather than reopening broad Epic 7 planning
- The implementation workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or landing work proceeds

---

## Day 2: Validation Baseline & Truth-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the implementation-day validation contract and the
highest-signal rerun surfaces Sprint 77 must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 77 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial packaging/platform/workflow batches default to
     `make quality-review-full`
   - docs-only audit/design days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 77 is most likely to stress:
   - install/package regression scripts
   - reviewed CMake proof-owner tests and examples
   - maintained benchmark/report commands that still define the shipped
     build/install surface
   - CI/workflow-facing commands in `Makefile`
4. Refresh the targeted rerun set most likely to matter in Sprint 77.
5. Record the authoritative validation split in the working notes.

### Deliverables
- Refreshed validation notes
- Sprint 77 rerun checklist
- Preserved proof-surface checklist

### Completion Criteria
- Sprint 77 uses the same reviewed/truthfulness reading fixed in Sprint 70
- The code-day validation contract is explicit before release/platform work
  starts
- No rerun ambiguity remains around the likely touched packaging/workflow/docs
  seams

---

## Day 3: Release-Surface Re-audit

**Title:** Release Audit  
**Theme:** Re-rank the live release/install/platform surface by downstream
value, maintenance cost, and truthfulness risk  
**Time estimate:** 12 hours

### Tasks
1. Audit the current release surface across:
   - static-first package/install story
   - ABI/version and exported-surface wording
   - downstream-consumer guidance
   - install/package proof ownership
   - platform-confidence summaries
2. Classify the strongest remaining release-surface contradictions:
   - release-shape drift
   - reviewed-platform asymmetry
   - install-validation interpretation drift
   - consumer-onboarding friction
3. Record where the current package story is already coherent and where it
   still behaves like a staged contract rather than a mature maintained one.
4. Rank the strongest contradiction centers by:
   - downstream value
   - compatibility risk
   - proof clarity
   - bounded Sprint 77 payoff
5. Write the release-surface re-audit artifact.

### Deliverables
- Release-surface re-audit
- Ranked contradiction list
- First packaging/platform hotspot map

### Completion Criteria
- The broad Sprint 77 release problem is reduced to a concrete seam ranking
- The highest-value packaging/platform contradictions are explicit
- Day 4 can proceed from a real current-state ranking

---

## Day 4: Platform Gap Boundary

**Title:** Boundary Freeze  
**Theme:** Refine the ranking and freeze the first packaging/platform fence
for the sprint  
**Time estimate:** 11 hours

### Tasks
1. Re-rank the Day 3 release/platform hotspots against:
   - downstream leverage
   - review clarity
   - compatibility risk
   - bounded cleanup payoff
2. Separate:
   - first-batch landing surfaces
   - support surfaces that move only if the first batch forces them
   - later or explicitly deferred platform/release work
3. Identify the strongest first Sprint 77 fence:
   - release/install productization lane
   - platform-proof lane
   - workflow/contract lane
4. Record the strongest non-goals:
   - no broad packaging-system rewrite
   - no fake shared-library maturity
   - no broad ABI promise widening
   - no wider reviewed cross-platform claim than evidence allows
5. Fix the Day 4 boundary in writing.

### Deliverables
- Refined platform-gap ranking
- First landing boundary
- Deferred/support-surface map

### Completion Criteria
- The first Sprint 77 landing fence is explicit before design begins
- Lower-value or higher-risk packaging/platform work is clearly separated from
  the first lane
- Support surfaces are bounded rather than assumed

---

## Day 5: Packaging/Productization Design

**Title:** Packaging Design  
**Theme:** Define the bounded implementation contract for the first
release/install improvement batch before edits begin  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 66 and Day 4 contract against the first-batch surfaces.
2. Decide the ownership split for the strongest first landing:
   - package/export metadata owner
   - install-proof owner
   - consumer-guidance owner
   - policy/truth owner
3. Define the guarantees the batch must preserve:
   - static-first release shape
   - current downstream build assumptions unless explicitly improved
   - current truthful ABI/version reading
   - current bounded install/platform claims
4. Fix the exact first-batch non-touch set:
   - unrelated solver/backend work
   - broader capability claims
   - unsupported dynamic/shared-library marketing
   - cross-platform proof widening not backed by touched evidence
5. Record the Day 5 design artifact.

### Deliverables
- Packaging/productization design
- First-batch non-touch set
- Preserved compatibility checklist

### Completion Criteria
- The first packaging batch is explicitly designed before edits begin
- The landing is bounded to release/install/productization clarity
- Compatibility and non-goal fences are fixed in writing

---

## Day 6: Packaging/Productization Batch

**Title:** Packaging Batch  
**Theme:** Land the highest-value bounded release/install improvements
justified by the refreshed audit  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 5 bounded packaging/productization changes.
2. Keep the batch inside the first-fence touch set and preserve the static-
   first truth contract.
3. Reconcile touched release/install wording or metadata across the minimal
   required surfaces only.
4. Run the required validation gate for the touched surface type.
5. Record the landing artifact and note any support-only follow-through
   needed for Day 7 or Day 8.

### Deliverables
- Landed packaging/productization batch
- Updated bounded release/install surface
- Recorded validation result for the landing day

### Completion Criteria
- One real release/install improvement batch lands
- The batch does not widen into unsupported ABI/platform claims
- The touched surface still reads as one coherent static-first contract

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-audit the release/platform surface after the first landing to
find the strongest remaining seam  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Day 6 landed surfaces.
2. Re-rank the remaining Sprint 77 contradictions against:
   - downstream friction removed
   - proof gaps still visible
   - support-surface drift
   - bounded follow-through value
3. Decide whether the strongest remaining seam is now:
   - Windows/macOS proof asymmetry
   - workflow/CI contract drift
   - install/package regression coverage
4. Fix the exact Day 8 design center in writing.
5. Record the rerank artifact.

### Deliverables
- Post-landing audit
- Updated seam ranking
- Exact next-batch center

### Completion Criteria
- Sprint 77 does not repeat the first landing unnecessarily
- The strongest remaining seam is explicitly reranked from the landed state
- Day 8 starts from a current-state ranking rather than the original backlog

---

## Day 8: Windows/macOS Proof Design

**Title:** Proof Design  
**Theme:** Define the bounded platform-confidence follow-through batch now
justified by the reranked release surface  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Day 7 rerank artifact.
2. Audit the exact Windows and macOS confidence gap most likely to offer
   bounded payoff:
   - reviewed exclusions
   - supplemental local proof
   - dead-code or build-shape interpretation
   - install-validation asymmetry
3. Fix the exact Day 9 touch set and support-only surfaces.
4. Define the required preserved truth:
   - no broader reviewed-platform claim than evidence supports
   - no disguised shared-library maturity claim
   - no fake Windows Makefile parity claim if only CMake proof improved
5. Record the Day 8 design artifact.

### Deliverables
- Platform-proof design
- Day 9 touch set
- Preserved truthfulness checklist

### Completion Criteria
- The second implementation lane is explicitly bounded
- The proof batch is justified by the post-Day-6 state rather than assumed
- The platform-claim fence is fixed before landing work begins

---

## Day 9: Platform Proof Follow-Through Batch

**Title:** Proof Batch  
**Theme:** Land the bounded Windows/macOS proof or supplemental-confidence
improvements justified by the Day 8 design  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 8 bounded proof or workflow follow-through.
2. Keep the batch inside the preserved platform-claim fence.
3. Update only the support surfaces truly required by the landing.
4. Run the required validation gate for the touched surface type.
5. Record the landing artifact, retained proof, and any remaining support
   drift.

### Deliverables
- Landed bounded platform-confidence batch
- Updated proof or workflow surfaces
- Recorded validation result for the landing day

### Completion Criteria
- One real platform-confidence seam improves
- The landing does not widen the reviewed-platform claim beyond evidence
- The touched surfaces still read as one coherent platform contract

---

## Day 10: Workflow & Contract Reconciliation Design

**Title:** Workflow Design  
**Theme:** Define the bounded CI/workflow/docs reconciliation batch that
matches the landed packaging/platform state  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Day 6 and Day 9 landed surfaces plus the current maintained
   command surface.
2. Decide the exact workflow/contract follow-through set across:
   - `Makefile`
   - install/package docs
   - maintainer policy
   - any CI-facing command summaries
3. Fix the exact Day 11 touch set.
4. Preserve the truthfulness split between:
   - reviewed lanes
   - supplemental local proof
   - install/package regression proof
5. Record the Day 10 design artifact.

### Deliverables
- Workflow/contract reconciliation design
- Day 11 touch set
- Preserved reviewed-vs-supplemental split

### Completion Criteria
- The workflow/docs follow-through batch is explicitly bounded
- The contract reconciliation matches the actual landed state
- Day 11 can move without reopening earlier design choices

---

## Day 11: Workflow & Contract Reconciliation Batch

**Title:** Workflow Batch  
**Theme:** Align maintained commands, docs, and policy surfaces with the
landed packaging/platform contract  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 10 bounded workflow and contract reconciliation batch.
2. Keep the work scoped to maintained command and contract surfaces only.
3. Verify whether any additional support-only follow-through is truly needed.
4. Run the appropriate validation gate for the touched surface type.
5. Record the landing artifact and the final support-surface status.

### Deliverables
- Landed workflow/contract reconciliation batch
- Updated maintained command/docs/policy surfaces
- Recorded validation result for the landing day

### Completion Criteria
- Maintained commands and docs now agree with the landed Sprint 77 contract
- No unsupported reviewed-platform or ABI claim is introduced
- The support-surface status is explicit before final proof alignment

---

## Day 12: Regression Coverage & Proof Alignment

**Title:** Proof Alignment  
**Theme:** Confirm the touched release/install/platform proofs sit in the
right owners and fix the exact Day 13 validation queue  
**Time estimate:** 10 hours

### Tasks
1. Re-read the touched Sprint 77 proof owners:
   - install/package scripts
   - reviewed CMake proof-owner tests
   - touched workflow and contract surfaces
2. Decide whether any new focused regression code is truly needed or whether
   the real remaining gap is proof-owner policy wording.
3. Reconcile the touched proof-owner map with the maintainer-policy surface.
4. Fix the exact Day 13 validation queue in writing.
5. Record the Day 12 alignment artifact.

### Deliverables
- Proof-owner alignment pass
- Final Sprint 77 validation queue
- Explicit no-op note if no new regression code is required

### Completion Criteria
- The touched release/install/platform proof sits in the right owners
- The final validation queue is explicit before Day 13
- No proof ambiguity remains around the landed Sprint 77 package

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Validate the full Sprint 77 package from the landed Day 12 state  
**Time estimate:** 12 hours

### Tasks
1. Run the standard code-day validation gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed baseline:
   - `make quality-review-full`
3. Reconfirm the reviewed CMake parity anchor.
4. Re-run the focused Sprint 77 follow-ons from the Day 12 queue:
   - touched install/package scripts
   - representative reviewed tests/examples
   - any touched workflow/reporting commands
5. Record the retained outputs, reviewed anchors, and any non-blocking
   runtime notes.

### Deliverables
- Full validation-sweep artifact
- Explicit reviewed anchor set
- Final Sprint 77 validated baseline

### Completion Criteria
- All required validation passes
- Reviewed parity remains exact
- The branch has one explicit validated close baseline before Day 14

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Finalize Sprint 77, fix the preserved contract, and hand off the
ranked next queue for Sprint 78  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Sprint 77 project-plan section and the Day 13 validation
   artifact.
2. Summarize the final Sprint 77 package:
   - refreshed release/platform audit
   - landed packaging/productization work
   - bounded platform-proof follow-through
   - workflow/contract reconciliation
   - proof-owner alignment
   - validated Day 13 close state
3. Fix the preserved non-goals and truthfulness fence in writing.
4. Rank the Sprint 78 carry-forward queue from the actual landed state.
5. Record the closeout artifact and final working-notes handoff.

### Deliverables
- Sprint 77 closeout artifact
- Final handoff notes
- Ranked Sprint 78 carry-forward queue

### Completion Criteria
- Sprint 77 closes from a validated baseline
- The preserved packaging/platform truth fence is explicit
- The next sprint inherits a coherent contract instead of a mixed release
  backlog
