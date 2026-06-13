# Sprint 66 Plan: Packaging, ABI & Platform Quality Convergence

**Sprint Duration:** 14 days  
**Goal:** Tighten the packaging and release story while reducing the remaining
cross-platform quality asymmetries that most affect product maturity, without
weakening the reviewed truthfulness contract. This sprint implements the
Sprint 66 section of `docs/planning/EPIC_6/PROJECT_PLAN.md`.

**Starting Point:** Sprint 65 closed with the benchmark-governance and
bounded solver-efficiency package landed and validated:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- the canonical maintained performance surface is now explicit and bounded
- the self-contained default build remains authoritative
- the next Epic 6 priority is no longer benchmark-governance churn; it is
  packaging, ABI, install, and platform-quality convergence

The next highest-value work is not another generic performance sprint.
It is a bounded productization sprint focused on clarifying the packaging and
ABI story, reassessing the remaining macOS/Windows/dead-code residuals, landing
the highest-value install/release improvements, reconciling CI and docs around
the new contract, and closing from the reviewed baseline.

**End State:** Sprint 66 leaves behind one coherent packaging and platform
quality package:
- a precise audit of the current install, ABI, static/shared, and release
  surface
- a refreshed map of the live macOS, Windows, and dead-code residuals
- a bounded packaging/productization implementation batch on the highest-value
  touched seams
- bounded dead-code and platform follow-through justified by the audit
- reconciled docs, workflows, and maintained commands around the converged
  packaging/platform contract
- focused install/package/platform regression coverage
- full validation and closeout from the landed state

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
148 hours, staying within the Sprint 66 estimate and below the 168-hour limit.

---

## Day 1: Sprint 66 Scope Audit & Packaging/Productization Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 66 project-plan scope plus the Sprint 65 validated
close into a bounded packaging and platform-quality implementation map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 66 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 65 retrospective, and
   the strongest Sprint 65 closeout artifacts.
2. Reconfirm the preserved Sprint 66 constraints:
   - no fake platform closure beyond reviewed evidence
   - no ABI or packaging claims unsupported by the shipped install surface
   - no broad build-system rewrite disguised as productization work
   - no weakening of the Linux/macOS/Windows truthfulness split
3. Define the Sprint 66 workstreams explicitly:
   - packaging and ABI audit
   - platform residual recheck
   - packaging/productization batch
   - dead-code and platform follow-through
   - CI and contract reconciliation
   - install/package regression coverage
   - validation and closeout
4. Record the strongest likely Sprint 66 touch surfaces:
   - install/package docs and build files
   - CI/workflow truth surfaces
   - platform and dead-code helper seams
5. Open Sprint 66 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 66 scope inventory
- Packaging/productization baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 66 starts from the Sprint 65 validated close rather than reopening
  benchmark-governance-first work
- The packaging and platform workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or code edits land

---

## Day 2: Validation Baseline & Install/Platform Rerun Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed baseline and rerun set that Sprint 66
packaging, platform, and CI changes must preserve  
**Time estimate:** 8 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity counts
   - current quality/truthfulness wording
2. Reconfirm the mandatory gate for later `*.c` / `*.h` days:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial packaging, CI, or
   platform-quality work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Sprint 66:
   - install/package/build entry surfaces
   - direct/examples/benchmarks that prove the touched release contract still
     works
   - workflow and parity paths that should not drift
5. Record the authoritative validation split for docs-only, bounded code-day,
   and substantial packaging/platform days.

### Deliverables
- Refreshed validation notes
- Sprint 66 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 66 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before implementation work begins
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Packaging & ABI Surface Audit

**Title:** Packaging Audit I  
**Theme:** Re-rank the live install, export, static/shared, versioning, and ABI
surfaces from the current repo state  
**Time estimate:** 11 hours

### Tasks
1. Inventory the current packaging and install surface across:
   - `CMakeLists.txt`
   - generated install/export surfaces
   - `INSTALL.md`
   - README package/install claims
2. Classify the live productization gaps:
   - static-only versus shared-library expectations
   - versioning and ABI claim clarity
   - install/export completeness
   - consumer ergonomics
3. Identify the strongest current contradictions:
   - claims that read broader than the shipped install surface
   - packaging behavior that differs between docs and build files
   - ABI/version surfaces that are implicit rather than explicit
4. Rank the highest-value Sprint 66 packaging/productization candidates.
5. Write the audit artifact with the explicit packaging and ABI map.

### Deliverables
- Live packaging/ABI inventory
- Ranked packaging-gap draft
- Initial packaging-batch candidate list

### Completion Criteria
- The broad “packaging and ABI convergence” claim is reduced to a concrete map
- The strongest install/export/versioning contradictions are explicit before
  redesign begins
- Day 4 can proceed from a real current-state productization audit instead of
  generic packaging concerns

---

## Day 4: Platform Residual Recheck

**Title:** Platform Audit  
**Theme:** Reassess the live macOS, Windows, and dead-code residual queue
against the current reviewed truthfulness contract  
**Time estimate:** 11 hours

### Tasks
1. Re-read the current Linux/macOS/Windows truth surfaces in:
   - `README.md`
   - `docs/maintainer_guide.md`
   - workflows
   - `Makefile`
2. Reassess the remaining platform residuals:
   - macOS dead-code state
   - Windows reviewed-wrapper gaps
   - Windows dead-code residuals
   - serialized dead-code topology
3. Separate:
   - real Sprint 66 platform-quality work
   - later stretch/non-goal platform work
   - operational truth surfaces that should stay unchanged
4. Fix the first platform/dead-code target set in writing.
5. Record the residual queue that Sprint 66 should not absorb.

### Deliverables
- Refined platform/dead-code residual map
- Ranked platform follow-through target set
- Deferred residual inventory

### Completion Criteria
- The platform queue is sharper than the original epic-level review
- The first platform/dead-code target set is explicit before design begins
- Non-goal platform work is clearly separated from the bounded Sprint 66 lane

---

## Day 5: Packaging/Productization Design

**Title:** Packaging Design  
**Theme:** Define the converged packaging, install, export, and release-shape
contract for the selected Sprint 66 surface  
**Time estimate:** 11 hours

### Tasks
1. Design the maintained Sprint 66 packaging contract for:
   - static build truth
   - any bounded shared-library or versioning surfaces, if justified
   - install/export consumer story
   - release-claim wording
2. Define the preserved compatibility rules:
   - no overstated ABI guarantees
   - no broadened cross-platform claim without reviewed evidence
   - no install/export widening without validation ownership
3. Decide which contract pieces belong in:
   - build files
   - install docs
   - maintainer policy
   - workflows/CI
4. Record the exact safety contract for the first implementation batch.
5. Fix the likely file fence for the Day 6-10 landing set.

### Deliverables
- Packaging/productization design artifact
- Explicit safety/compatibility contract
- First implementation fence

### Completion Criteria
- The packaging and install story is explicit before edits start
- Build files, docs, and CI ownership are separated clearly enough to prevent
  drift
- The converged productization story is defined tightly enough to support later
  platform follow-through

---

## Day 6: Platform & Dead-Code Follow-Through Design

**Title:** Platform Design  
**Theme:** Convert the residual platform queue into one bounded implementation
plan that stays inside the reviewed truth fence  
**Time estimate:** 9 hours

### Tasks
1. Fix which platform/dead-code residuals move in Sprint 66 and which stay
   deferred.
2. Define what each bounded follow-through batch should prove:
   - dead-code report completeness
   - reviewed workflow truthfulness
   - Windows/macOS wording alignment
   - bounded operational cleanup
3. Define what is intentionally *not* Sprint 66:
   - fake platform closure
   - broad unsandboxed dead-code topology changes
   - broad wrapper redesign beyond the audited seams
4. Record the exact implementation fence for the platform-quality batch.
5. Confirm where focused regression coverage will later need to land.

### Deliverables
- Platform/dead-code implementation fence
- Deferred residual list
- Regression-coverage target shortlist

### Completion Criteria
- The platform-quality batch is bounded before code lands
- The dead-code and workflow truth surfaces are explicitly separated
- Later regression coverage and docs follow-through have a concrete starting set

---

## Day 7: Exact Landing Fence & Regression Plan

**Title:** Landing Design  
**Theme:** Fix the exact touched-file fence, proof plan, and validation order
before packaging or platform changes land  
**Time estimate:** 10 hours

### Tasks
1. Collapse the Day 5-6 design into one exact landing plan for:
   - build/install/package files
   - docs surfaces
   - workflows
   - optional bounded code seams if required
2. Separate:
   - required first-batch files
   - optional support files only if proof burden forces them
   - explicit non-touch set for Sprint 66
3. Define the proof plan:
   - install/package regression checks
   - workflow truth checks
   - reviewed baseline gates
4. Define the Day 8-12 sequence so implementation order is explicit.
5. Record the exact validation order for the final landing days.

### Deliverables
- Exact touched-file fence
- Proof and validation plan
- Ordered Day 8-12 landing sequence

### Completion Criteria
- The first code/doc batch is fully bounded before implementation starts
- The proof homes and validation order are explicit
- Sprint 66 can land without improvising the touched surface late

---

## Day 8: Packaging/Productization Batch 1

**Title:** Packaging Batch I  
**Theme:** Land the highest-value first packaging and install convergence slice
inside the audited fence  
**Time estimate:** 12 hours

### Tasks
1. Implement the first bounded packaging/productization slice on the required
   Day 7 file set.
2. Keep the batch focused on the highest-value install/export/release-shape
   contradiction from the audit.
3. Preserve the current truthfulness fence:
   - no broader claim than the landed surface supports
   - no unproved ABI promise
   - no platform widening hidden inside packaging edits
4. Add or adjust focused regression proof only where the landing requires it.
5. Run the required validation gate for the touched surface.

### Deliverables
- First landed packaging/productization batch
- Focused regression proof for the landed slice
- Day 8 implementation artifact

### Completion Criteria
- One real productization contradiction is resolved in code/docs/build files
- The touched proof surface demonstrates the landed behavior
- The batch stays inside the Day 7 fence and passes the required gate

---

## Day 9: Packaging/Productization Batch 2

**Title:** Packaging Batch II  
**Theme:** Land the remaining bounded Sprint 66 packaging and release-shape
follow-through without widening the contract  
**Time estimate:** 12 hours

### Tasks
1. Land the second bounded packaging batch from the Day 7 queue.
2. Reconcile any remaining install/export/versioning wording that must move
   together with the landed implementation.
3. Keep the batch bounded:
   - no general shared-library framework unless the design explicitly justified it
   - no platform-quality spillover unless it is part of the planned fence
4. Add or tighten focused regression proof only where the landed surface needs
   it.
5. Run the required validation gate for the touched surface.

### Deliverables
- Second landed packaging/productization batch
- Updated proof for the landed slice
- Day 9 implementation artifact

### Completion Criteria
- The bounded packaging queue is materially smaller and more truthful than at
  sprint start
- The implementation, docs, and proof remain aligned
- The batch passes the required gate without widening beyond plan

---

## Day 10: Platform / Dead-Code Follow-Through Batch

**Title:** Platform Batch  
**Theme:** Land the highest-value bounded platform-quality and dead-code
follow-through justified by the audit  
**Time estimate:** 12 hours

### Tasks
1. Implement the selected platform/dead-code follow-through batch.
2. Keep the landing focused on:
   - reviewed truthfulness alignment
   - dead-code/report operational clarity
   - bounded Windows/macOS residual cleanup
3. Preserve the explicit non-goal fence:
   - no fake platform closure
   - no broad dead-code topology rewrite
   - no generic workflow churn unrelated to the audited seam
4. Add or tighten focused regression proof where the landing requires it.
5. Run the stronger reviewed validation gate if the touched surface justifies it.

### Deliverables
- Landed platform/dead-code follow-through batch
- Focused proof or workflow confirmation
- Day 10 implementation artifact

### Completion Criteria
- One real platform/dead-code contradiction is resolved
- The truthfulness contract is clearer after the batch than before it
- The batch stays bounded and passes the required gate

---

## Day 11: CI & Contract Reconciliation

**Title:** CI Reconciliation  
**Theme:** Align workflows, maintained commands, and contract wording with the
landed Sprint 66 packaging and platform state  
**Time estimate:** 10 hours

### Tasks
1. Reconcile any touched workflow, Makefile, and docs wording surfaces with the
   landed packaging/platform contract.
2. Remove stale commentary or assumptions revealed by the Day 8-10 batches.
3. Keep the batch bounded to touched truth surfaces only:
   - no speculative CI expansion
   - no benchmark-governance drift
   - no unrelated build-system cleanup
4. Recheck the reviewed command surface and workflow wording for consistency.
5. Record the exact remaining Day 12-14 queue.

### Deliverables
- Reconciled workflow and command truth surfaces
- Reduced stale-contract commentary
- Day 11 reconciliation artifact

### Completion Criteria
- The maintained command and workflow story reads coherently after the landed
  implementation batches
- No touched contract surface contradicts the shipped productization state
- The remaining queue is narrowed to regression coverage, validation, and close

---

## Day 12: Install / Package Regression Coverage

**Title:** Regression Coverage  
**Theme:** Add focused verification for the touched install, package, workflow,
and platform surfaces before final validation  
**Time estimate:** 10 hours

### Tasks
1. Add or tighten the focused regression coverage justified by the Day 8-11
   landings.
2. Keep proof burden bounded to the actual touched surfaces:
   - install/package behavior
   - workflow truth surfaces
   - platform/dead-code operational behavior
3. Avoid widening into unrelated assurance expansion.
4. Run the required validation gate for the touched proof surface.
5. Record the exact Day 13 rerun set and retained outputs to capture.

### Deliverables
- Focused install/package/platform regression coverage
- Reduced proof gap on touched surfaces
- Day 12 coverage artifact

### Completion Criteria
- The touched productization and platform surfaces have focused verification
- The final validation sweep has an explicit rerun set before it begins
- No unrelated assurance sprawl enters the sprint late

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full required validation gate and rerun the targeted
Sprint 66 proof surfaces from the final landed tree  
**Time estimate:** 12 hours

### Tasks
1. Run the full required gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Confirm the reviewed anchors remain exact:
   - reviewed CMake parity count
   - Makefile/CMake parity
   - full reviewed CMake `ctest`
3. Rerun the targeted Sprint 66 proof surfaces:
   - install/package/build entry checks
   - representative examples
   - representative benchmarks
   - touched workflow/platform proof binaries
4. Capture the retained outputs needed for the closeout and retrospective.
5. Record any non-blocking validation notes explicitly.

### Deliverables
- Full validation artifact
- Retained benchmark/example/proof outputs
- Explicit validated baseline for closeout

### Completion Criteria
- All required gates pass
- The reviewed anchors remain exact
- Sprint 66 has a documented validated baseline for Day 14 closeout

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Convert the validated Sprint 66 branch into one coherent handoff
package for the next Epic 6 sprint  
**Time estimate:** 11 hours

### Tasks
1. Collapse the sprint result into one explicit outcome package:
   - packaging/ABI audit
   - platform residual recheck
   - landed productization and platform batches
   - CI/contract reconciliation
   - regression coverage
   - validated close
2. Fix the preserved truthfulness fence and deferred queue in writing.
3. Recheck the Sprint 66 section of `docs/planning/EPIC_6/PROJECT_PLAN.md` for
   any correction now that the sprint is complete.
4. Write the Day 14 closeout and handoff artifact from the validated Day 13
   baseline only.
5. Confirm the branch is ready for retrospective and PR work.

### Deliverables
- Sprint 66 closeout artifact
- Explicit deferred queue and handoff notes
- Final Day 14 working-notes close

### Completion Criteria
- Sprint 66 reads as one coherent packaging and platform-quality sprint
- The deferred queue is explicit instead of implicit
- The branch closes from a validated baseline and is ready for retrospective work
