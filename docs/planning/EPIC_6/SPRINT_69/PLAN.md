# Sprint 69 Plan: Public Product Surface Finalization, Integration & Epic 6 Closeout

**Sprint Duration:** 14 days  
**Goal:** Finalize the public docs/examples/reference story, run the final
cross-surface integration sweep, and close Epic 6 from a measured baseline.
This sprint implements the Sprint 69 section of
`docs/planning/EPIC_6/PROJECT_PLAN.md`.

**Starting Point:** Sprint 68 closed with the strongest remaining giant-test
and second-layer assurance work reduced and validated:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- the hardest large-`n` CSC-backed Cholesky lifecycle lanes now have stronger
  oracle and seeded property coverage
- public product surfaces, benchmark governance, packaging/install truth, and
  large-source/test maintainability have all advanced enough that Sprint 69 can
  focus on final integration and product-surface closure rather than reopening
  broad backend, packaging, or test-architecture redesign
- the next Epic 6 priority is no longer another isolated subsystem sprint; it
  is reconciling the final public product story across docs, headers, examples,
  benchmarks, tests, and maintainer surfaces, then closing the epic from a
  measured validated baseline

The highest-value next work is not broad new functionality. It is a bounded
finalization sprint focused on auditing the final public product surface,
landing the highest-value last-mile docs/examples simplifications, reconciling
cross-surface claims, validating the final integrated branch state, and writing
the Epic 6 closeout and residual handoff package.

**End State:** Sprint 69 leaves behind one coherent Epic 6 closeout package:
- a final audited and simplified public product story
- aligned README/tutorial/examples/benchmark/header/maintainer claims
- a measured final cross-surface validation baseline
- final Epic 6 summary, residual limits, and handoff artifacts
- a closed Epic 6 branch state that matches the maintained reviewed truth

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
144 hours, matching the Sprint 69 estimate and staying below the 168-hour
limit.

---

## Day 1: Sprint 69 Scope Audit & Public Surface Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 69 project-plan scope plus the Sprint 68 validated
close into a bounded public-surface and Epic-closeout implementation map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 69 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 68 retrospective, and
   the strongest Sprint 68 closeout artifacts.
2. Reconfirm the preserved Sprint 69 constraints:
   - no fake product simplification that weakens the maintained truthfulness
     contract
   - no broad implementation work disguised as public-surface cleanup
   - no inflated cross-platform confidence claims beyond reviewed evidence
   - no reopening settled Sprint 60-68 seams unless a touched public surface
     truly forces it
3. Define the Sprint 69 workstreams explicitly:
   - public surface audit
   - docs/examples productization
   - cross-surface compatibility sweep
   - full validation
   - Epic 6 summary and handoff
   - project-level residual finalization
4. Record the strongest likely Sprint 69 touch surfaces:
   - README and tutorial
   - examples and benchmark docs
   - public headers and maintainer guide
   - project-level planning and residual artifacts
5. Open Sprint 69 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 69 scope inventory
- Public-surface baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 69 starts from the Sprint 68 validated close rather than reopening
  older implementation or packaging lanes
- The public-surface and closeout workstreams are explicit before deeper audit
  begins
- The sprint non-goal fence is fixed before design or code edits land

---

## Day 2: Validation Baseline & Final Rerun Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed baseline and final rerun set that Sprint 69
public-surface and closeout work must preserve  
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
3. Reconfirm the stronger default for substantial cross-surface integration or
   closeout work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Sprint 69:
   - core integration and family-local test surfaces
   - examples and maintained benchmark/reporting surfaces
   - install/package and canonical report surfaces that define the final Epic 6
     product story
5. Record the authoritative validation split for docs-only, bounded code-day,
   and substantial integration/closeout days.

### Deliverables
- Refreshed validation notes
- Sprint 69 rerun list
- Final code-day validation checklist

### Completion Criteria
- Sprint 69 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before finalization work begins
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Public Surface Audit I

**Title:** Surface Audit I  
**Theme:** Re-rank the final public product surfaces by contradiction density,
adoption value, and closeout payoff  
**Time estimate:** 12 hours

### Tasks
1. Audit the strongest maintained public surfaces across:
   - `README.md`
   - `docs/tutorial.md`
   - `examples/README.md`
   - `benchmarks/README.md`
   - key public headers
   - `docs/maintainer_guide.md`
2. Classify each surface by:
   - user-facing product-story importance
   - stale chronology or wording risk
   - cross-surface contradiction density
   - likely final simplification payoff
3. Identify the strongest current contradictions:
   - duplicated or drifted product explanations
   - unclear ownership between examples, benchmarks, and tests
   - headers or docs that imply stronger guarantees than the validated repo
4. Rank the most valuable Sprint 69 product-surface candidates.
5. Write the audit artifact with the explicit hotspot map.

### Deliverables
- Live public-surface inventory
- Ranked product-surface candidate list
- Initial finalization shortlist

### Completion Criteria
- The broad “public product surface finalization” claim is reduced to a
  concrete file and contradiction map
- The strongest user-facing and maintainer-facing drift points are explicit
  before design begins
- Day 4 can proceed from a real current-state surface ranking instead of
  generic docs concerns

---

## Day 4: Public Surface Audit II & Landing Boundary

**Title:** Surface Audit II  
**Theme:** Separate the must-land Sprint 69 product surfaces from lower-value
cleanup candidates and freeze the first landing boundary  
**Time estimate:** 12 hours

### Tasks
1. Re-rank the Day 3 surface set against the Epic 6 closeout target.
2. Separate:
   - first landing public product surfaces
   - second landing compatibility/reconciliation surfaces
   - later residuals that should stay out of Sprint 69
3. Confirm which touched files likely belong in:
   - docs/examples productization
   - compatibility sweep support
   - Epic closeout and residual finalization only
4. Fix the first Sprint 69 implementation target set in writing.
5. Record the residual queue that Sprint 69 should not absorb.

### Deliverables
- Refined surface ranking
- First landing boundary
- Deferred residual map

### Completion Criteria
- The Sprint 69 target set is smaller and sharper than the original epic-level
  review
- The first landing boundary is explicit before design begins
- Lower-value cleanup is clearly separated from the bounded Sprint 69 lane

---

## Day 5: Docs/Examples Productization Design

**Title:** Productization Design  
**Theme:** Define the final simplification and adoption-edit contract for the
highest-value public surfaces  
**Time estimate:** 10 hours

### Tasks
1. Design the final productization contract for the selected first landing:
   - top-level product narrative
   - examples/tutorial handoff
   - benchmarks/tests ownership wording
   - maintainer-guide authority split
2. Define the preserved truthfulness rules:
   - no marketing-style claims beyond reviewed evidence
   - no example or benchmark wording that steals test-owned guarantees
   - no widened platform or ABI promises
3. Decide which pieces belong in:
   - README/tutorial/examples edits
   - benchmark/header/maintainer support edits
   - residual or project-level closeout only
4. Record the exact safety contract for the first implementation batch.
5. Fix the likely file fence for the Day 6-7 landing set.

### Deliverables
- Productization design artifact
- Explicit truthfulness contract
- First implementation fence

### Completion Criteria
- The final public-surface simplification story is explicit before edits start
- Ownership boundaries are separated clearly enough to prevent churn
- The converged design is tight enough to support a bounded landing

---

## Day 6: Docs/Examples Productization Batch 1

**Title:** Productization Batch I  
**Theme:** Land the first bounded final public-surface simplification batch on
the highest-value docs/examples surfaces  
**Time estimate:** 10 hours

### Tasks
1. Implement the first productization batch inside the Day 5 fence.
2. Keep the batch bounded to:
   - highest-value top-level public-story corrections
   - example/tutorial handoff cleanup
   - minimal support wording needed on adjacent maintained surfaces
3. Remove or tighten stale chronology only where the touched text would
   otherwise stay misleading.
4. Run targeted sanity checks appropriate to the touched surfaces.
5. Record landed behavior, touched files, and any residual queue sharpened by
   the implementation.

### Deliverables
- First landed productization batch
- Updated docs/examples wording
- Post-landing notes

### Completion Criteria
- The highest-value public product contradictions are reduced without widening
  into unrelated implementation work
- The touched surfaces preserve the maintained truthfulness fence
- Day 7 can audit from a landed state instead of staying purely speculative

---

## Day 7: Docs/Examples Productization Batch 2

**Title:** Productization Batch II  
**Theme:** Finish the bounded docs/examples productization set and align the
remaining first-landing support surfaces  
**Time estimate:** 12 hours

### Tasks
1. Complete the remaining first-landing productization edits.
2. Align adjacent surfaces only where the first batch would otherwise leave
   obvious drift:
   - benchmark docs
   - maintainer guide
   - public header wording if required
3. Recheck the public-surface ownership split:
   - tests own regression/oracle/property guarantees
   - examples own workflow/adoption teaching
   - benchmarks own workflow/performance proof
4. Run targeted sanity checks and direct rereads on the touched surfaces.
5. Record the landed state and sharpen the compatibility-sweep queue.

### Deliverables
- Completed first-landing public-surface batch
- Cross-surface wording notes
- Compatibility-sweep carry-forward list

### Completion Criteria
- The first public-surface landing is coherent enough that later work can
  focus on cross-surface reconciliation rather than basic product-story repair
- The ownership split is explicit across touched surfaces
- The residual queue is smaller and more concrete after the landing

---

## Day 8: Cross-Surface Compatibility Sweep Design

**Title:** Compatibility Design  
**Theme:** Define the final API/docs/examples/benchmarks/tests/platform
reconciliation contract before the final sweep lands  
**Time estimate:** 10 hours

### Tasks
1. Audit the live post-Day-7 state across:
   - docs and examples
   - benchmark docs and canonical report wording
   - public headers and maintainer guide
   - workflow/platform claim surfaces
2. Design the final compatibility sweep contract around:
   - API/docs/examples consistency
   - benchmark/test ownership consistency
   - install/package/platform claim consistency
   - project-plan and residual-story alignment
3. Define which contradictions must be fixed in Sprint 69 versus documented as
   final residual limits.
4. Fix the exact touched-file fence for Day 9-10.
5. Record the preserved non-widening rules for the final reconciliation batch.

### Deliverables
- Compatibility-sweep design artifact
- Exact reconciliation fence
- Final contradiction list

### Completion Criteria
- The cross-surface sweep is explicit before edits start
- Required fixes are separated from final residual documentation
- The Day 9-10 landing set is small enough to stay bounded

---

## Day 9: Final Cross-Surface Compatibility Batch

**Title:** Compatibility Batch  
**Theme:** Land the bounded final reconciliation batch across the highest-value
remaining product surfaces  
**Time estimate:** 12 hours

### Tasks
1. Implement the final compatibility batch inside the Day 8 fence.
2. Reconcile the strongest remaining contradictions across:
   - API/docs/examples/benchmarks/tests wording
   - platform and reviewed-subset claims
   - install/package and canonical-report interpretation
3. Keep the batch bounded to wording, reference, and ownership alignment unless
   a touched code or header seam is truly necessary.
4. Run the appropriate sanity checks or code-day validation based on the files
   touched.
5. Record the landed reconciliation state and remaining final validation
   questions.

### Deliverables
- Final compatibility batch
- Reconciled public/maintainer surfaces
- Post-landing audit notes

### Completion Criteria
- The strongest remaining cross-surface contradictions are closed from the live
  repo state
- The sprint still stays bounded to product-surface and closeout work
- The branch is ready for validation-focused follow-through rather than more
  generic reconciliation

---

## Day 10: Post-Landing Audit & Final Validation/Handoff Design

**Title:** Pre-Close Audit  
**Theme:** Re-rank the residual queue after the compatibility batch and fix the
exact final validation and handoff plan  
**Time estimate:** 10 hours

### Tasks
1. Audit the live post-Day-9 branch state against the Sprint 69 end-state
   target.
2. Decide whether any bounded follow-through edit remains truly necessary
   before final validation.
3. Fix the exact final validation set:
   - maintained reviewed gates
   - truthfulness anchors
   - targeted examples, tests, benchmarks, install/package, and report surfaces
4. Fix the exact final handoff set:
   - sprint closeout artifact
   - Epic 6 summary inputs
   - residual-limit updates
   - project-level summary artifacts
5. Record the final Day 11-14 sequence explicitly.

### Deliverables
- Post-landing audit artifact
- Final validation plan
- Final handoff plan

### Completion Criteria
- The remaining queue is smaller and more concrete than a generic “final docs
  pass”
- The exact validation and closeout steps are explicit before they run
- Sprint 69 is positioned to close from a measured baseline rather than from
  vague confidence

---

## Day 11: Final Cross-Surface Follow-Through

**Title:** Final Follow-Through  
**Theme:** Land any single bounded residual fix required to make the final
validation and closeout story truthful  
**Time estimate:** 10 hours

### Tasks
1. Implement one bounded follow-through batch only if the Day 10 audit proves
   it is necessary.
2. Keep the batch limited to:
   - final wording truthfulness
   - final reference alignment
   - final support/header/docs/plan cleanup required by the validated product
     story
3. Avoid widening into new feature or architecture work.
4. Run the appropriate sanity checks or code-day validation based on touched
   files.
5. Record the final pre-validation state.

### Deliverables
- Any required final follow-through batch
- Updated pre-validation notes
- Final touched-surface summary

### Completion Criteria
- No unresolved contradiction remains that would make the final validation or
  Epic 6 closeout misleading
- The branch is ready for the full maintained validation sweep
- The sprint remains bounded even if a small last-mile fix lands

---

## Day 12: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the final maintained quality gates, truthfulness anchors, and
targeted follow-ons from the integrated Epic 6 branch state  
**Time estimate:** 8 hours

### Tasks
1. Run the full maintained gate set:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm the reviewed truthfulness anchors:
   - reviewed CMake parity count
   - Makefile/CMake parity
   - final reviewed CMake pass count
3. Run the targeted Sprint 69 follow-on set:
   - integration and representative family-local proofs
   - key examples
   - canonical maintained benchmark/report surfaces
   - install/package/reporting surfaces if they remain part of the final story
4. Record retained representative outputs and any non-blocking observations.
5. Freeze the validation baseline for the closeout days.

### Deliverables
- Final validation artifact
- Measured final Epic 6 baseline
- Retained representative outputs

### Completion Criteria
- The full maintained quality gate passes from the integrated final branch
  state
- The reviewed truthfulness anchors remain exact
- The closeout days can rely on a measured validated baseline instead of a
  stale earlier sprint result

---

## Day 13: Epic 6 Summary, Residual Finalization & Handoff Package

**Title:** Epic Summary  
**Theme:** Write the final sprint closeout, Epic 6 summary inputs, and
project-level residual finalization package from the validated baseline  
**Time estimate:** 12 hours

### Tasks
1. Summarize the final Sprint 69 delivered state from the Day 12 baseline.
2. Write the Sprint 69 closeout and handoff artifact.
3. Update the final Epic 6 residual limits and project-level summary inputs:
   - final carry-forward queue
   - final deferred limits
   - project-plan or summary artifacts if they truly need correction
4. Reconcile the final Epic-level interpretation across:
   - sprint notes
   - sprint artifacts
   - project-level planning/summary surfaces
5. Record any explicit non-blocking residuals that remain after Epic 6 close.

### Deliverables
- Sprint 69 closeout artifact
- Epic 6 summary inputs
- Final residual-limit package

### Completion Criteria
- Epic 6 closeout inputs are written from the measured validated state
- Final residuals are explicit instead of being left as implied history
- The only work left is the final close and handoff confirmation

---

## Day 14: Sprint 69 Closeout & Epic 6 Final Handoff

**Title:** Closeout  
**Theme:** Close Sprint 69 and hand off Epic 6 from the final validated,
integrated product baseline  
**Time estimate:** 6 hours

### Tasks
1. Re-read the Day 12 validation artifact and Day 13 summary package.
2. Confirm the final Sprint 69 closeout narrative:
   - public product story finalized
   - cross-surface claims reconciled
   - validated baseline preserved
   - Epic 6 handoff complete
3. Reconfirm whether `PROJECT_PLAN.md` or any project-level summary surface
   needs a final correction.
4. Record the final branch-close assumptions and residual queue.
5. Prepare the sprint for retrospective and PR closeout.

### Deliverables
- Sprint 69 final closeout notes
- Final Epic 6 handoff state
- Retrospective-ready branch summary

### Completion Criteria
- Sprint 69 closes from the Day 12 validated baseline
- The final Epic 6 product story and residual limits are explicit in writing
- The branch is ready for retrospective, PR packaging, and merge without
  further design work
