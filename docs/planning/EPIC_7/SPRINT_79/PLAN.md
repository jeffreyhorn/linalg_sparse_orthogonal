# Sprint 79 Plan: Numerical Assurance Expansion, Final Integration & Epic 7 Closeout

**Sprint Duration:** 14 days  
**Goal:** Close Epic 7 from a measured validated baseline after the product,
configuration, performance, capability, packaging, and maintainability work has
been reconciled across all maintained surfaces. This sprint implements the
Sprint 79 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`.

**Starting Point:** Sprint 78 closed with a validated maintainability package
and a ranked residual hotspot queue. Sprint 79 starts from the final bounded
Epic 7 closeout pressure:
- Sprint 71-78 already moved the highest-value public-surface, product-model,
  configuration, capability, backend, benchmark, packaging, and maintainability
  seams
- the strongest local reviewed baseline remains `make quality-review-full`
- the highest-value remaining Sprint 79 work is now:
  - assurance-gap rerank
  - one bounded differential/oracle/property batch
  - one final cross-surface integration sweep
  - one full final validation sweep
  - Epic 7 summary and residual finalization
  - Epic 7 retrospective and handoff
  - bounded closeout reconciliation time if the integrated sweep reveals a real
    truthfulness seam
- the preserved Epic 7 non-goal fence is still in force:
  - no fake “final polish” batch that widens claims beyond maintained evidence
  - no new broad subsystem redesign hidden inside closeout work
  - no benchmark-threshold or platform-parity claim without reviewed proof
  - no retrospective or summary package that rewrites the actual residual queue

The highest-value Sprint 79 work is therefore not “add more features before the
end.” It is one bounded numerical-assurance expansion plus one integrated
truth-surface closeout so Epic 7 ends from a measured and explicitly reconciled
state.

**End State:** Sprint 79 leaves behind a final Epic 7 closeout package:
- a refreshed assurance-gap ranking
- one landed bounded oracle/property/lifecycle-stress improvement
- one fully reconciled docs/examples/headers/benchmarks/workflows/package/proof
  reading of the Epic 7 tree
- one final validated baseline with reviewed parity anchors
- one explicit Epic 7 summary, residual queue, retrospective, and post-Epic-7
  handoff

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
156 hours, staying within the day-cap limit while covering the Sprint 79
closeout scope.

---

## Day 1: Sprint 79 Scope Audit & Final Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 79 project-plan section plus the Sprint 78 closeout
into one bounded Epic 7 closeout sprint  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 79 section of
   `docs/planning/EPIC_7/PROJECT_PLAN.md`, the Sprint 78 closeout package, and
   the strongest prior Epic 7 truthfulness fences.
2. Reconfirm the preserved Sprint 79 constraints:
   - no new broad subsystem work
   - no widened product or platform claim without proof
   - no fake summary that hides unresolved residual seams
   - no closeout churn detached from measured validation value
3. Define the Sprint 79 workstreams explicitly:
   - assurance-gap rerank
   - bounded oracle/property batch
   - final cross-surface integration sweep
   - full validation
   - Epic 7 summary and residual finalization
   - retrospective and handoff
4. Record the strongest likely Sprint 79 touch surfaces:
   - proof-owner tests
   - public/reference docs
   - headers
   - benchmarks/benchmark policy surfaces
   - package/workflow surfaces
5. Open Sprint 79 working notes and record the intended landing order,
   artifacts, and validation expectations.

### Deliverables
- Sprint 79 scope inventory
- Final Epic 7 workstream map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 79 starts from the Sprint 78 validated closeout instead of reopening
  broad Epic 7 planning
- The final assurance and integration workstreams are explicit before deeper
  audit begins
- The closeout non-goal fence is fixed before design or landing work proceeds

---

## Day 2: Validation Baseline & Truth-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the final implementation-day validation contract and the
highest-signal proof surfaces Sprint 79 must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 79 authority split:
   - bounded `*.c` / `*.h` landing days require `make format`, `make lint`,
     and `make test`
   - substantial assurance or integration batches default to
     `make quality-review-full`
   - docs-only audit/design/review days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 79 is most likely to stress:
   - final oracle/property owners
   - reviewed CMake proof-owner tests and representative examples
   - install/package and benchmark command surfaces
4. Refresh the targeted rerun set most likely to matter in Sprint 79.
5. Record the authoritative validation split in the working notes.

### Deliverables
- Refreshed validation notes
- Sprint 79 rerun checklist
- Preserved proof/truth-surface checklist

### Completion Criteria
- Sprint 79 uses the same reviewed/truthfulness reading fixed by the earlier
  Epic 7 sprints
- The code-day validation contract is explicit before assurance work starts
- No rerun ambiguity remains around the likely touched closeout seams

---

## Day 3: Assurance Gap Re-audit

**Title:** Assurance Audit  
**Theme:** Re-rank the remaining highest-value oracle, property, lifecycle, and
platform-confidence gaps after the main Epic 7 implementation work  
**Time estimate:** 12 hours

### Tasks
1. Audit the current residual assurance surfaces across:
   - external-oracle value
   - property-test value
   - lifecycle-stress value
   - platform-confidence relevance
2. Classify the strongest remaining assurance contradictions:
   - missing external cross-checks
   - under-owned lifecycle stress paths
   - residual property gaps
   - support-surface overclaim relative to proof
3. Record where the current proof package is already coherent and where it
   still leaves the strongest final residual gap.
4. Rank the highest-value assurance seams by:
   - closeout value
   - boundedness
   - runtime cost
   - truth-surface payoff
5. Write the assurance-gap audit artifact.

### Deliverables
- Assurance-gap re-audit
- Ranked residual contradiction list
- First final-assurance map

### Completion Criteria
- The broad Sprint 79 assurance problem is reduced to a concrete seam ranking
- The highest-value final proof gap is explicit
- Day 4 can proceed from a real current-state ranking

---

## Day 4: Assurance Boundary Freeze

**Title:** Assurance Boundary  
**Theme:** Refine the ranking and freeze the first final assurance fence for
the sprint  
**Time estimate:** 11 hours

### Tasks
1. Re-rank the Day 3 assurance seams against:
   - closeout payoff
   - proof clarity
   - runtime cost
   - bounded landing value
2. Separate:
   - first-batch landing surfaces
   - support surfaces that move only if the batch forces them
   - later or explicitly deferred final-closeout work
3. Identify the strongest first Sprint 79 fence:
   - differential/oracle lane
   - lifecycle-stress lane
   - support-only integration lane
4. Record the strongest non-goals:
   - no broad new feature work
   - no giant proof campaign across all families
   - no claim widening detached from validation
   - no summary work before the proof gap is bounded
5. Fix the Day 4 boundary in writing.

### Deliverables
- Refined assurance-gap ranking
- First landing boundary
- Deferred/support-surface map

### Completion Criteria
- The first Sprint 79 assurance fence is explicit before design begins
- Lower-value or higher-cost assurance work is clearly separated from the first
  lane
- Support surfaces are bounded rather than assumed

---

## Day 5: Differential / Oracle Batch Design

**Title:** Assurance Design  
**Theme:** Define the bounded implementation/proof contract for the strongest
final oracle/property landing before edits begin  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Day 4 assurance boundary against the strongest first-batch
   files.
2. Decide the ownership split for the first landing:
   - primary proof owner
   - support proof owner
   - public/support interpretation owner
   - validation-sensitive owner
3. Define the guarantees the batch must preserve:
   - current public behavior
   - current reviewed validation surface
   - current runtime-bounded interpretation
   - current platform truthfulness
4. Fix the exact first-batch non-touch set:
   - unrelated solver families
   - unrelated benchmark or package mechanics
   - unrelated maintainability backlog
   - broad docs churn
5. Record the Day 5 design artifact.

### Deliverables
- Differential/oracle batch design
- First-batch non-touch set
- Preserved compatibility/truthfulness checklist

### Completion Criteria
- The first assurance batch is explicitly designed before edits begin
- The landing is bounded to one real closeout proof improvement
- Compatibility and non-goal fences are fixed in writing

---

## Day 6: Differential / Oracle Batch

**Title:** Oracle Batch  
**Theme:** Land the highest-value bounded assurance improvement justified by
the refreshed audit  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded proof/oracle/property landing from the Day 5 design.
2. Keep the landing local to the first-batch proof and support surfaces.
3. Add or update focused proof only where the bounded batch truly requires it.
4. Run the required code-day validation gate:
   - `make format`
   - `make lint`
   - `make test`
5. Run `make quality-review-full` if the batch crosses a substantial proof or
   integration boundary.
6. Record the landing artifact and retained outputs.

### Deliverables
- Landed bounded assurance batch
- Recorded validation results
- Explicit retained proof/owner notes

### Completion Criteria
- The branch gains one real final assurance improvement
- The landing stays inside the Day 5 fence
- Validation passes before the sprint reranks the remaining closeout pressure

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining final Epic 7 pressure after the bounded
assurance landing  
**Time estimate:** 10 hours

### Tasks
1. Re-read the landed Day 6 batch and its validation results.
2. Recheck whether the strongest remaining closeout seam is now:
   - cross-surface integration
   - residual proof drift
   - support-surface truthfulness
   - summary/residual package drift
3. Separate:
   - required next batch
   - support-only follow-through
   - explicit no-op surfaces
4. Fix the next strongest closeout lane in writing.
5. Record the rerank artifact.

### Deliverables
- Post-landing rerank
- Next-batch target map
- Support-only/no-op map

### Completion Criteria
- The strongest remaining Sprint 79 seam is explicit after the Day 6 landing
- The branch does not drift into summary work before the integrated surface is
  reranked
- Day 8 starts from a current-state closeout map

---

## Day 8: Cross-Surface Integration Audit

**Title:** Integration Audit  
**Theme:** Re-read the fully integrated Epic 7 tree and reduce the remaining
support-surface problem to one bounded contradiction map  
**Time estimate:** 11 hours

### Tasks
1. Audit the strongest integrated support surfaces:
   - `README.md`
   - `INSTALL.md`
   - `docs/tutorial.md`
   - `examples/README.md`
   - `benchmarks/README.md`
   - `docs/maintainer_guide.md`
   - strongest touched public headers
2. Recheck that examples, benchmarks, headers, workflows, and package surfaces
   still agree with the landed Epic 7 package.
3. Classify the strongest remaining integration contradictions:
   - ownership wording drift
   - duplicated authority
   - stale support interpretation
   - missing final residual caveat
4. Rank any remaining integration seams by bounded closeout value.
5. Write the integration-audit artifact.

### Deliverables
- Final cross-surface integration audit
- Ranked support-surface contradiction list
- Candidate Day 9 touch set

### Completion Criteria
- The broad integration problem is reduced to a concrete support-surface map
- The strongest remaining integration contradiction is explicit
- Day 9 can proceed from one exact integration fence

---

## Day 9: Cross-Surface Integration Batch

**Title:** Integration Batch  
**Theme:** Land the strongest bounded integration reconciliation justified by
the Day 8 audit  
**Time estimate:** 12 hours

### Tasks
1. Land the bounded support-surface or maintainer-policy reconciliation batch.
2. Keep the batch inside the Day 8 fence:
   - touch only the exact required integration surfaces
   - avoid reopening implementation or broad product work
3. Preserve the authority split across:
   - public docs
   - public headers
   - examples
   - benchmarks
   - maintainer policy
4. Run the docs-only sanity set or code-day validation as appropriate to the
   touched surfaces.
5. Record the landing artifact and preserved authority split.

### Deliverables
- Landed integration reconciliation batch
- Updated support-surface authority map
- Recorded validation/sanity results

### Completion Criteria
- The strongest integrated support contradiction is closed
- The batch stays inside the Day 8 fence
- No new ownership ambiguity remains across the touched closeout surfaces

---

## Day 10: Epic 7 Summary & Residual Design

**Title:** Summary Design  
**Theme:** Define the final Epic 7 summary, residual queue, and closeout
package structure before the final summary batch lands  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Day 9 integrated tree against the full Epic 7 project-plan
   history.
2. Decide the exact summary package outputs Sprint 79 must leave behind:
   - final Epic 7 summary
   - residual queue
   - project-plan note if needed
   - post-Epic-7 handoff framing
3. Fix the exact non-goal fence for the summary lane:
   - no rewriting actual residual debt
   - no fake “everything solved” language
   - no historical revisionism
4. Define whether any final support surface must still move before the summary
   batch.
5. Record the Day 10 design artifact.

### Deliverables
- Epic 7 summary/residual design
- Final non-goal fence
- Day 11 touch-set decision

### Completion Criteria
- The final summary package is explicitly designed before edits begin
- The residual queue is treated as a truth surface, not as marketing copy
- Day 11 starts from one exact summary/closeout fence

---

## Day 11: Epic 7 Summary & Residual Batch

**Title:** Summary Batch  
**Theme:** Land the bounded Epic 7 summary, residual finalization, and any
required project-plan closeout note  
**Time estimate:** 11 hours

### Tasks
1. Land the final summary/residual package from the Day 10 design.
2. Update `PROJECT_PLAN.md` only if the integrated closeout truly requires it.
3. Keep the batch bounded to:
   - summary surfaces
   - residual queue surfaces
   - explicit closeout truthfulness notes
4. Avoid widening into new implementation or proof work.
5. Record the Day 11 artifact.

### Deliverables
- Landed Epic 7 summary package
- Final residual queue
- Any needed project-plan closeout note

### Completion Criteria
- Epic 7 has one explicit truthful summary package
- The residual queue is fixed in writing
- The batch does not widen into late-cycle implementation churn

---

## Day 12: Final Proof Alignment & Validation Queue Freeze

**Title:** Proof Alignment  
**Theme:** Freeze the final validation queue and close any last bounded
proof-owner ambiguity before the final sweep  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Day 6 and Day 9-11 landed surfaces against the final support and
   proof-owner map.
2. Decide whether any last focused regression or support edit is truly needed,
   or whether the remaining Day 12 result is a bounded no-op note.
3. Fix the exact Day 13 validation queue in writing:
   - full gates
   - reviewed parity anchors
   - targeted follow-ons
4. Reconfirm the branch closes from measured proof, not from assumption.
5. Record the Day 12 alignment artifact.

### Deliverables
- Final proof/support alignment pass
- Exact Day 13 validation queue
- Explicit no-op note if no further edit is needed

### Completion Criteria
- The final validation queue is explicit before Day 13
- No ownership ambiguity remains around the Epic 7 closeout package
- Sprint 79 does not start Day 13 from an implied proof surface

---

## Day 13: Full Final Validation Sweep

**Title:** Validation Sweep  
**Theme:** Validate the full Epic 7 closeout package from the Day 12 aligned
state  
**Time estimate:** 12 hours

### Tasks
1. Run the standard code-day validation gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed baseline:
   - `make quality-review-full`
3. Reconfirm the reviewed CMake parity anchor.
4. Re-run the focused Sprint 79 follow-ons from the Day 12 queue:
   - touched proof owners
   - representative reviewed examples
   - any touched support/workflow/package surfaces
5. Record the retained outputs, reviewed anchors, and any non-blocking runtime
   notes.

### Deliverables
- Full final-validation artifact
- Explicit reviewed anchor set
- Final Epic 7 validated baseline

### Completion Criteria
- All required validation passes
- Reviewed parity remains exact
- The branch has one explicit validated Epic 7 close baseline before Day 14

---

## Day 14: Epic 7 Retrospective, Handoff & Closeout Buffer

**Title:** Closeout  
**Theme:** Finalize Epic 7, land the retrospective and handoff package, and
absorb any last measured truthfulness adjustment revealed by the integrated
closeout  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Sprint 79 project-plan section and the Day 13 validation
   artifact.
2. Summarize the final Epic 7 package:
   - assurance rerank
   - landed bounded assurance batch
   - final integration reconciliation
   - validated close baseline
   - final summary and residual package
3. Produce the retrospective and explicit post-Epic-7 carry-forward queue.
4. If the integrated closeout reveals one last bounded truthfulness adjustment,
   use the closeout buffer to land it without widening scope.
5. Record the final closeout artifact and handoff notes.

### Deliverables
- Epic 7 retrospective
- Post-Epic-7 handoff package
- Ranked carry-forward queue
- Explicit final closeout artifact

### Completion Criteria
- Epic 7 ends from one explicit validated close baseline
- The retrospective and handoff package are complete
- Post-Epic-7 work inherits a ranked residual queue rather than an implied
  backlog
