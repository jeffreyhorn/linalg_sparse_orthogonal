# Sprint 39 Plan: Epic 3 Stabilization, Final Audit & Closeout

**Sprint Duration:** 14 days  
**Goal:** Finish Epic 3 with a final repository-wide audit, close remaining quality gaps from the sprint notes, and leave behind stable standards and artifacts that keep the cleaned-up state from regressing. This sprint implements the Sprint 39 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`.

**Starting Point:** Sprint 38 closed with a validated regression-proofing baseline, a truthful coverage/readiness contract, a closed dead-code compile-db exclusion list, a stronger local reviewed baseline via `make quality-review-full`, and a README readiness checklist. Sprint 39 starts from that stable state and focuses on final audit and closeout work: last-pass warning verification, residual dead-code disposition, final cross-platform reconciliation, maintainer-standard consolidation, cleanup of transitional scaffolding, and a final Epic 3 end-state record.

**End State:** Sprint 39 leaves Epic 3 in a stable audited state: reviewed warning claims are revalidated, residual dead-code buckets are dispositioned or explicitly justified, cross-platform limits are named honestly, maintainer standards are consolidated, temporary sprint-only scaffolding is removed where appropriate, and the repo has a final end-state baseline plus an Epic 3 summary suitable for later feature work.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 124 hours, matching the Sprint 39 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 39 Scope Audit & Final-Baseline Inventory

**Title:** Final Baseline  
**Theme:** Convert the Sprint 39 project-plan items into a concrete final-audit scope  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 39 section of `docs/planning/EPIC_3/PROJECT_PLAN.md` plus the Sprint 38 handoff/retrospective so the sprint stays anchored to the final documented prerequisites.
2. Confirm the validated baseline that must remain true through Sprint 39: direct maintained gates pass, `make quality-review-full` remains authoritative locally, reviewed CMake parity remains auditable, and dead-code report/check stays truthful about its staged limits.
3. Inventory the remaining explicit Sprint 39 surfaces: final warning audit, final dead-code audit, final cross-platform audit, standards/documentation closeout, temporary-scaffolding cleanup, Epic summary, and final validation.
4. Record the open residual queues already named by prior sprints rather than assuming new cleanup debt exists.
5. Open Sprint 39 working notes and capture the baseline assumptions, current gate boundaries, and first-pass audit order.

### Deliverables
- Sprint 39 final-baseline inventory
- Initial audit-order map
- Named residual queues for warnings, dead-code, cross-platform limits, and standards/doc closeout

### Completion Criteria
- Sprint 39 starts from a documented Sprint 38 validated baseline
- The final-audit scope is separated from already-closed Sprint 38 work
- The remaining Epic 3 closeout surfaces are identified before implementation begins

---

## Day 2: Final Warning Audit

**Title:** Warning Audit  
**Theme:** Reconfirm the reviewed warning-clean baseline against the authoritative Sprint 30 workflow  
**Time estimate:** 8 hours

### Tasks
1. Re-run the Sprint 30 compile-hygiene lens conceptually against the current repo and identify the authoritative warning-check surfaces to use this sprint.
2. Audit current warning claims in docs, handoffs, and maintainer guidance against the actual Apple Clang CMake full-tree model and the narrower Makefile cross-checks.
3. Distinguish real warning regressions from wording drift or non-authoritative path mismatches.
4. Record any remaining warning-risk surfaces that still need implementation work versus those already closed.
5. Write the audit note that defines the final warning-closeout batch.

### Deliverables
- Final warning audit note
- Explicit authoritative/non-authoritative warning-surface map
- Initial fix/defer/closed classification for warning-related closeout work

### Completion Criteria
- Final warning claims are mapped before edits begin
- Warning-truthfulness issues are separated from real compile regressions
- The warning-closeout batch is bounded to concrete surfaces

---

## Day 3: Final Dead-Code Audit

**Title:** Dead-Code Audit  
**Theme:** Reassess the residual dead-code buckets after Sprint 38’s zero-gap compile-db closeout  
**Time estimate:** 10 hours

### Tasks
1. Re-run the dead-code reporting lens over the current post-Sprint-38 state.
2. Audit the residual report buckets: audited public keeps, `cppcheck` supporting-signal rows, and `non-deadcode-static-analysis-noise`.
3. Reconfirm the current serialized-execution limitation and keep it separate from content-level findings.
4. Distinguish what can be fully dispositioned in Sprint 39 from what should remain documented as intentional/noisy/non-actionable.
5. Write the audit note that defines the final dead-code-closeout batch.

### Deliverables
- Final dead-code audit note
- Residual-bucket classification map
- Initial resolve/justify/defer classification for dead-code closeout work

### Completion Criteria
- The dead-code closeout queue is explicit before edits begin
- Content-level findings are separated from workflow-topology limits
- The final dead-code batch is bounded enough to implement deliberately

---

## Day 4: Final Cross-Platform Audit

**Title:** Platform Audit  
**Theme:** Reconfirm the enforced/staged/excluded platform contract before closeout edits  
**Time estimate:** 10 hours

### Tasks
1. Re-audit Linux, macOS, and Windows quality surfaces against the Sprint 36 contract and the current workflow/docs state.
2. Reconfirm what is truly enforced, what remains staged, what is supplemental, and what remains intentionally excluded.
3. Identify any wording drift between workflow files, README contract text, and closeout docs.
4. Separate real platform regressions from intentionally preserved limitations.
5. Write the audit note that defines the final cross-platform reconciliation batch.

### Deliverables
- Final cross-platform audit note
- Updated enforced/staged/excluded parity map
- Initial fix/keep/defer classification for platform-closeout work

### Completion Criteria
- Cross-platform closeout starts from a current audited contract
- Residual platform limits are explicit before edits begin
- Later platform wording/fix work is bounded clearly enough to implement safely

---

## Day 5: Warning Closeout Batch

**Title:** Warning Batch  
**Theme:** Close the highest-value remaining warning-truthfulness or warning-regression slice  
**Time estimate:** 8 hours

### Tasks
1. Choose the smallest highest-value warning-closeout batch from the Day 2 audit.
2. Implement the fix in the authoritative surfaces, whether that means code, build config, or maintainer docs.
3. Keep the batch narrow enough that any resulting validation deltas remain attributable.
4. Revalidate the touched warning-related paths directly.
5. Record the residual warning queue, if any, after the batch.

### Deliverables
- Final warning-closeout batch
- Residual warning queue
- Updated notes describing the authoritative warning end state

### Completion Criteria
- The highest-value warning-closeout item is resolved
- Warning claims are more truthful and/or the reviewed warning baseline is tighter
- Remaining warning work, if any, is smaller and explicitly named

---

## Day 6: Dead-Code Closeout Batch

**Title:** Dead-Code Batch  
**Theme:** Resolve or explicitly disposition the strongest remaining dead-code findings  
**Time estimate:** 10 hours

### Tasks
1. Choose the first dead-code closeout batch from the Day 3 audit, limited to the strongest actionable or clearest justification items.
2. Implement the strongest safe improvement to code, report classification, or documentation for the chosen residual findings.
3. Preserve the staged serialized-execution contract while refining content-level disposition.
4. Revalidate the authoritative serial dead-code path directly.
5. Record the residual dead-code queue that remains for final summary/justification.

### Deliverables
- Final dead-code closeout batch
- Residual dead-code queue
- Updated notes describing the final actionable-vs-justified boundary

### Completion Criteria
- At least the highest-value residual dead-code item is resolved or explicitly justified
- The authoritative serial dead-code path still passes
- Remaining dead-code debt is smaller and honestly described

---

## Day 7: Cross-Platform Reconciliation Batch

**Title:** Platform Batch  
**Theme:** Close the highest-value workflow/docs/platform wording mismatch  
**Time estimate:** 10 hours

### Tasks
1. Choose the first cross-platform reconciliation batch from the Day 4 audit.
2. Implement the highest-value platform-closeout changes, whether in workflow files, contract docs, or narrowly-scoped portability/support surfaces.
3. Preserve honest staged/excluded boundaries instead of forcing fake uniformity.
4. Revalidate the touched platform-facing contract surfaces directly.
5. Record the residual platform queue for final closeout language.

### Deliverables
- Final cross-platform reconciliation batch
- Residual platform queue
- Updated notes describing the final enforced/staged/excluded contract

### Completion Criteria
- The highest-value platform mismatch is resolved
- Cross-platform wording is more internally consistent
- Remaining platform limitations are smaller and explicitly staged/excluded

---

## Day 8: Standards & Maintainer-Doc Audit

**Title:** Standards Audit  
**Theme:** Inventory the final maintainer standards that should survive Epic 3  
**Time estimate:** 8 hours

### Tasks
1. Audit current maintainer-facing standards/docs for warnings, designated initializers, dormant-test truthfulness, dead-code workflow, reviewed-quality wrappers, and cross-platform contract text.
2. Identify duplication, stale sprint-specific wording, and missing authoritative references.
3. Decide which document(s) should own each long-term standard after Epic 3 ends.
4. Separate lasting maintainer standards from sprint-only narrative artifacts.
5. Write the audit note that defines the standards/documentation closeout batch.

### Deliverables
- Standards/documentation audit note
- Ownership map for lasting maintainer standards
- Initial consolidate/compress/remove classification for standards closeout work

### Completion Criteria
- Maintainer-standard ownership is explicit before edits begin
- Sprint-only wording is separated from long-term guidance
- The standards closeout batch is bounded clearly enough to implement safely

---

## Day 9: Standards & Documentation Closeout Batch

**Title:** Standards Batch  
**Theme:** Consolidate the lasting Epic 3 maintainer standards into their authoritative homes  
**Time estimate:** 8 hours

### Tasks
1. Implement the maintainer-standard consolidation chosen from the Day 8 audit.
2. Compress duplicate guidance and remove stale sprint-specific wording where the authoritative contract is already clear.
3. Keep the resulting documentation compact, stable, and aligned with the current enforced/staged quality model.
4. Revalidate the touched docs/maintainer workflow surfaces directly.
5. Record any residual doc standardization work that still belongs only in final summary language.

### Deliverables
- Standards/documentation closeout batch
- Residual standards queue
- Updated notes describing the stable maintainer-doc end state

### Completion Criteria
- Long-term maintainer standards are clearer and less duplicated
- Sprint-only guidance is reduced where appropriate
- Remaining standards/doc work is minimal and explicitly bounded

---

## Day 10: Temporary Scaffolding Audit & Cleanup Design

**Title:** Scaffolding Audit  
**Theme:** Identify which transitional notes, helpers, or allowances should not survive Epic 3 closeout  
**Time estimate:** 10 hours

### Tasks
1. Audit the repo for temporary allowlists, transitional notes, sprint-only helper structure, and other closeout-sensitive scaffolding.
2. Distinguish useful permanent helper layers from temporary Epic 3 implementation residue.
3. Evaluate whether any remaining scaffolding is load-bearing for truthfulness, reviewability, or local maintenance.
4. Choose the smallest safe cleanup batch that reduces clutter without destabilizing the current baseline.
5. Write the cleanup design note that defines the Day 11 batch.

### Deliverables
- Temporary-scaffolding audit/design note
- Explicit keep/remove/defer list for transitional cleanup
- Defined cleanup batch for Day 11

### Completion Criteria
- Transitional cleanup starts from a current audited inventory
- Useful permanent support layers are separated from sprint-only residue
- The cleanup batch is bounded safely before edits begin

---

## Day 11: Temporary Scaffolding Cleanup Batch

**Title:** Cleanup Batch  
**Theme:** Remove or collapse the highest-value transitional scaffolding that should not remain post-Epic 3  
**Time estimate:** 8 hours

### Tasks
1. Implement the cleanup batch chosen on Day 10.
2. Remove or simplify temporary scaffolding while preserving the current reviewed/dead-code/platform contracts.
3. Revalidate the touched quality/reporting/doc surfaces directly.
4. Record any consciously-retained transitional items that must remain for later feature work.
5. Update notes with the final keep/remove boundary after the batch.

### Deliverables
- Transitional cleanup batch
- Residual consciously-retained scaffolding list
- Updated notes describing the final cleanup boundary

### Completion Criteria
- The highest-value removable scaffolding is gone
- The maintained baseline still behaves the same in the touched surfaces
- Any retained transitional items are explicitly justified

---

## Day 12: Epic 3 Summary Report

**Title:** Epic Summary  
**Theme:** Produce the concise Epic 3 narrative of what changed, what is enforced, and what remains limited  
**Time estimate:** 8 hours

### Tasks
1. Summarize the warning, dead-code, cross-platform, test-truthfulness, docs, and reviewed-gate outcomes across Epic 3.
2. Keep the summary concise and future-facing rather than turning it into another sprint diary.
3. State what is now enforced, what remains staged/excluded, and what residual risks remain for future work.
4. Link the summary back to the final authoritative artifacts rather than duplicating all detail.
5. Record the summary deliverable in working notes and closeout docs.

### Deliverables
- Epic 3 summary report
- Concise enforced/staged/residual-risk narrative
- Updated notes linking the summary to authoritative artifacts

### Completion Criteria
- Epic 3 has a concise summary suitable for later feature work handoff
- The summary reflects the actual final contracts, not aspirational claims
- Residual risks remain visible without overwhelming the closeout docs

---

## Day 13: Final Validation Sweep

**Title:** Final Validation  
**Theme:** Re-run the full maintained quality baseline and capture the final Epic 3 end-state  
**Time estimate:** 10 hours

### Tasks
1. Run the final direct maintained quality commands.
2. Run the strongest local reviewed baseline via `make quality-review-full`.
3. Run the authoritative serial dead-code path and capture the end-state residual bucket counts.
4. Record timings, final reviewed CMake parity numbers, and any important caveats in a dedicated artifact.
5. Reconcile the final validation output against the Sprint 39 closeout claims.

### Deliverables
- Final validation sweep artifact
- Final timings/logs for direct, reviewed, and dead-code paths
- Final measured Epic 3 end-state baseline

### Completion Criteria
- The maintained direct and reviewed paths pass
- The authoritative serial dead-code path passes
- The final Epic 3 baseline is captured in a reviewable artifact

---

## Day 14: Epic 3 Closeout & Handoff

**Title:** Epic Closeout  
**Theme:** Write the final handoff, retrospective, and closeout state for Epic 3  
**Time estimate:** 8 hours

### Tasks
1. Write the Sprint 39 handoff and retrospective based on the Day 13 validated baseline.
2. Finalize the Epic 3 closeout language: what is now stable, what is enforced, what remains staged/excluded, and what future work should inherit.
3. Update `docs/planning/EPIC_3/PROJECT_PLAN.md` only if the final audit surfaces a truly new deferred item not already owned by later work.
4. Ensure the summary, closeout docs, and working notes all point to the same authoritative final baseline.
5. Commit the Day 14 closeout state.

### Deliverables
- Sprint 39 handoff
- Sprint 39 retrospective
- Final Epic 3 closeout state and any necessary project-plan routing update

### Completion Criteria
- Epic 3 closes from a documented validated baseline
- Any surviving deferred work is explicitly routed or explicitly absent
- The repo has a final closeout package suitable for handing back to normal feature work
