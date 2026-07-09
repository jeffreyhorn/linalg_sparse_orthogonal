# Sprint 116 Plan: Adoption Surface Residual QA & Claim Guardrails

**Sprint Duration:** 14 days
**Goal:** Close Sprint 111's residual adoption-surface debt before final Epic
10 integration by validating external documentation references, preserving
audience boundaries, and keeping performance/support claims evidence-bounded.

**Starting Point:** Sprint 116 begins from:
- Sprint 111 solver-selection, Matrix Market, benchmark, README, tutorial, and
  example documentation
- Sprint 112 package/platform support truth
- Sprint 113 behavior and proof-owner closeout evidence
- Sprint 114 residual proof-owner follow-through and source-boundary non-claims
- Sprint 115 package/platform residual decisions, including package, ABI,
  Windows, macOS, install, and package-manager non-claims

The sprint must:
- validate external references used by adoption-facing documentation
- keep README quality/CI wording compact and user-facing
- review benchmark documentation scanability and lane naming
- decide whether `docs/algorithm.md` needs public/adoption cleanup or remains
  technical background
- keep performance wording tied to measured evidence
- verify adoption docs do not advertise unreviewed package/platform,
  proof-owner, internal-helper, or state-of-the-art claims
- close with documentation hygiene and final handoff to Sprint 117

**End State:** Sprint 116 leaves behind:
- external-reference QA artifact
- README quality and CI boundary artifact
- benchmark scanability and indexing decision
- algorithm-reference positioning decision
- evidence-bounded performance wording pass
- adoption non-claims checklist
- validation and final Epic 10 closeout handoff artifact

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `56` hours, matching the Sprint 116 project-plan estimate.

---

## Day 1: Adoption QA Intake and Scope Fence

**Title:** Adoption Intake
**Theme:** Establish adoption-facing scope, duplicate fence, and artifact map
**Time estimate:** 4 hours

### Tasks
1. Re-read the Sprint 116 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 111, Sprint 112, Sprint 113, Sprint 114, and Sprint 115
   retrospective handoffs that affect adoption-facing claims.
3. Inventory adoption-facing documents:
   `README.md`, `INSTALL.md`, `docs/tutorial.md`,
   `docs/solver_selection.md`, `docs/matrix_market.md`,
   `docs/algorithm.md`, `benchmarks/README.md`, and `examples/README.md`.
4. Explicitly exclude implementation, package-manager recipe, ABI, source
   movement, and helper-abstraction work from Sprint 116.
5. Create Sprint 116 working notes and artifact directory.
6. Write the adoption QA intake artifact.

### Deliverables
- Sprint 116 working-notes baseline
- artifact directory
- adoption-surface inventory
- duplicate-work exclusion list
- day-level owner map

### Completion Criteria
- all Sprint 116 project-plan items have day-level owners
- Sprint 115 package/platform decisions are available as claim guardrails
- no implementation or package/platform support work is pulled into Sprint 116

---

## Day 2: External Reference Inventory

**Title:** Reference Inventory
**Theme:** Collect adoption-facing external links before network validation
**Time estimate:** 4 hours

### Tasks
1. Search adoption-facing docs for external URLs and named external resources.
2. Classify links by audience impact:
   Matrix Market, SuiteSparse, benchmarks, packages, toolchains, and examples.
3. Identify links that are informational versus links required for a workflow.
4. Record any links that should not be network-checked because they are
   examples, generated references, or intentionally offline.
5. Write the external-reference inventory artifact.

### Deliverables
- external-reference inventory
- link classification table
- network-check candidate list
- excluded-link rationale

### Completion Criteria
- Day 3 can run link QA without rediscovering sources
- all adoption-facing external references have an owner or exclusion reason
- no documentation content is changed before validation

---

## Day 3: External Reference QA Follow-Through

**Title:** Reference QA
**Theme:** Validate external references and fence stale or unstable links
**Time estimate:** 6 hours

### Tasks
1. Network-check Day 2 candidate links.
2. Replace, fence, or remove stale links only where adoption value is affected.
3. Preserve links that are still accurate but document external volatility if
   needed.
4. Avoid broad content edits unrelated to reference QA.
5. Run focused documentation hygiene for touched docs.
6. Write the external-reference QA artifact.

### Deliverables
- external-reference QA artifact
- optional focused documentation updates
- stale-link disposition table
- validation notes

### Completion Criteria
- external reference status is documented
- stale or unstable adoption-facing links are addressed or explicitly fenced
- touched documentation passes hygiene checks

---

## Day 4: README Quality and CI Boundary Audit

**Title:** README Boundary
**Theme:** Audit README quality/support wording without turning it into a handbook
**Time estimate:** 4 hours

### Tasks
1. Review README quality, CI, install, support-tier, and evidence wording.
2. Compare README claims against Sprint 112-115 package/platform truth.
3. Identify places where README is too maintainer-oriented for first-use
   adoption.
4. Identify any unsupported install, platform, ABI, package-manager, or
   benchmark claims.
5. Write the README boundary audit artifact.

### Deliverables
- README quality/CI boundary audit
- unsupported-claim candidates
- compactness and audience notes
- Day 5 edit checklist

### Completion Criteria
- Day 5 has a concrete edit or no-edit decision
- README package/platform claims match Sprint 115 truth
- README remains scoped to adoption, not maintainer policy

---

## Day 5: README Quality and CI Boundary Follow-Through

**Title:** README Follow-Through
**Theme:** Apply only necessary README wording fixes
**Time estimate:** 4 hours

### Tasks
1. Apply the Day 4 edit checklist if wording fixes are needed.
2. Keep changes compact and adoption-facing.
3. Leave maintainer-policy detail in `docs/maintainer_guide.md`.
4. Preserve evidence-bounded CI and package/platform wording.
5. Run documentation hygiene for touched files.
6. Write the README follow-through artifact.

### Deliverables
- README follow-through artifact
- optional README updates
- claim-boundary validation notes
- no-edit rationale if no update is needed

### Completion Criteria
- README support and CI wording is compact and evidence-bounded
- no unsupported package/platform, ABI, or benchmark claim is introduced
- touched documentation passes hygiene checks

---

## Day 6: Benchmark Documentation Scanability Audit

**Title:** Benchmark Audit
**Theme:** Review benchmark documentation for scanability and live lane clarity
**Time estimate:** 4 hours

### Tasks
1. Review `benchmarks/README.md` for live lane names, report mechanics,
   interpretation entry points, and indexes.
2. Compare benchmark wording against current benchmark and CI support truth.
3. Identify sections that are hard to scan for adoption-facing users.
4. Identify any universal performance claims or stale lane names.
5. Write the benchmark scanability audit artifact.

### Deliverables
- benchmark scanability audit
- lane-name and report-mechanics notes
- performance-claim candidates
- Day 7 edit checklist

### Completion Criteria
- benchmark docs have a clear edit or no-edit path
- performance language remains tied to measured local evidence
- no benchmark workflow changes are implied

---

## Day 7: Benchmark Documentation Follow-Through

**Title:** Benchmark Follow-Through
**Theme:** Apply bounded benchmark documentation cleanup if needed
**Time estimate:** 4 hours

### Tasks
1. Apply Day 6 benchmark documentation edits only if scanability or claim
   accuracy requires them.
2. Clarify indexes, lane names, report mechanics, or interpretation entry
   points where needed.
3. Avoid new benchmark commands, CI lanes, or performance claims.
4. Run documentation hygiene for touched files.
5. Write the benchmark follow-through artifact.

### Deliverables
- benchmark scanability and indexing decision
- optional benchmark documentation updates
- validation evidence
- residual notes for Sprint 117 if needed

### Completion Criteria
- benchmark docs remain scanable for adoption-facing users
- no universal performance or unsupported CI claim is introduced
- touched documentation passes hygiene checks

---

## Day 8: Algorithm Reference Positioning Audit

**Title:** Algorithm Positioning
**Theme:** Decide whether algorithm docs are adoption reference or technical background
**Time estimate:** 4 hours

### Tasks
1. Review `docs/algorithm.md` from a first-time user and adoption QA
   perspective.
2. Identify whether any section is referenced as public adoption guidance.
3. Check for unsupported state-of-the-art, benchmark, package/platform, or
   internal-helper claims.
4. Decide whether the document needs cleanup or should remain technical
   background.
5. Write the algorithm positioning artifact.

### Deliverables
- algorithm-reference positioning decision
- unsupported-claim candidates
- public-versus-background boundary notes
- Day 9 edit checklist or no-edit rationale

### Completion Criteria
- `docs/algorithm.md` has a clear adoption-facing role
- no maintainer-proof-first wording is promoted into adoption guidance
- Day 9 can apply focused cleanup if needed

---

## Day 9: Algorithm Reference Follow-Through

**Title:** Algorithm Follow-Through
**Theme:** Apply focused algorithm-document cleanup if needed
**Time estimate:** 4 hours

### Tasks
1. Apply Day 8 cleanup only where adoption claims or positioning need
   correction.
2. Preserve technical background content that remains accurate and useful.
3. Avoid large rewrites or algorithmic implementation claims.
4. Run documentation hygiene for touched files.
5. Write the algorithm follow-through artifact.

### Deliverables
- algorithm positioning follow-through artifact
- optional algorithm documentation updates
- validation evidence
- residual handoff if broader reference cleanup is needed

### Completion Criteria
- algorithm documentation role is explicit enough for Epic closeout
- no unsupported state-of-the-art or platform claim remains unfenced
- touched documentation passes hygiene checks

---

## Day 10: Performance Wording Evidence Audit

**Title:** Performance Evidence
**Theme:** Audit performance language against measured evidence
**Time estimate:** 4 hours

### Tasks
1. Review README, solver-selection guide, benchmark docs, and final support
   wording for performance claims.
2. Identify universal speed, state-of-the-art, or benchmark-generalization
   wording.
3. Map each meaningful performance claim to local measured evidence or mark it
   for downgrade.
4. Confirm package/platform and toolchain caveats are not used as performance
   claims.
5. Write the performance wording audit artifact.

### Deliverables
- performance wording evidence map
- unsupported performance-claim candidates
- downgrade or no-edit checklist
- Sprint 117 closeout notes

### Completion Criteria
- performance wording is tied to evidence or marked for cleanup
- no universal benchmark claim is left unexamined
- Day 11 can apply bounded wording fixes

---

## Day 11: Performance Wording Follow-Through

**Title:** Performance Follow-Through
**Theme:** Apply evidence-bounded performance wording cleanup
**Time estimate:** 4 hours

### Tasks
1. Apply Day 10 wording changes where evidence does not support the claim.
2. Keep performance language specific to measured local conditions.
3. Preserve useful benchmark guidance without overstating competitive
   calibration.
4. Run documentation hygiene for touched files.
5. Write the performance wording follow-through artifact.

### Deliverables
- evidence-bounded performance wording artifact
- optional documentation updates
- validation evidence
- remaining performance residuals for Sprint 117

### Completion Criteria
- public performance wording avoids universal speed claims
- all changed claims remain tied to local evidence
- touched documentation passes hygiene checks

---

## Day 12: Adoption Non-Claims Checklist

**Title:** Non-Claims Checklist
**Theme:** Verify adoption docs do not advertise unreviewed surfaces
**Time estimate:** 4 hours

### Tasks
1. Audit adoption-facing docs for public Matrix I/O module, public builder API,
   shared-library/ABI, expanded platform-support, universal benchmark, and
   maintainer-proof-first claims.
2. Check Sprint 115 package/platform non-claims against adoption docs.
3. Check Sprint 114 proof-owner non-claims against adoption docs.
4. Record each non-claim as present, absent, fenced, or needing cleanup.
5. Write the adoption non-claims checklist artifact.

### Deliverables
- adoption non-claims checklist
- package/platform claim-fence table
- proof-owner/internal-helper claim-fence table
- Day 13 cleanup list

### Completion Criteria
- adoption non-claims are explicit and auditable
- Sprint 117 has a clear residual list if any claims remain ambiguous
- no implementation or package support claim is introduced

---

## Day 13: Adoption Claim Guardrail Follow-Through

**Title:** Claim Guardrails
**Theme:** Apply final adoption-surface claim cleanup before validation
**Time estimate:** 4 hours

### Tasks
1. Apply Day 12 cleanup list where wording fixes are required.
2. Keep edits limited to adoption-facing claim boundaries.
3. Re-check README, INSTALL, solver-selection, Matrix Market, tutorial,
   examples, benchmark, and algorithm docs touched during the sprint.
4. Run documentation hygiene for touched files.
5. Write the claim-guardrail follow-through artifact.

### Deliverables
- adoption claim-guardrail follow-through artifact
- optional documentation updates
- residual claim-boundary notes for Sprint 117
- validation evidence

### Completion Criteria
- adoption-facing docs do not advertise unreviewed support surfaces
- remaining residuals are explicitly handed to Sprint 117
- touched documentation passes hygiene checks

---

## Day 14: Validation and Handoff

**Title:** Validation Handoff
**Theme:** Validate Sprint 116 docs and hand off final adoption truth
**Time estimate:** 4 hours

### Tasks
1. Review all Sprint 116 artifacts, working notes, and touched documentation.
2. Run required documentation hygiene checks.
3. Confirm no `.c`, `.h`, build metadata, workflows, package metadata, or
   implementation files changed unless explicitly required earlier.
4. Capture final adoption QA metrics and changed-surface summary.
5. Publish Sprint 117 final integration handoff.
6. Write the validation and handoff artifact.

### Deliverables
- final validation evidence
- changed-surface summary
- adoption truth handoff
- Sprint 117 residual list
- Day 14 closeout artifact

### Completion Criteria
- required documentation checks pass
- adoption and claim-boundary truth is explicit
- Sprint 116 closes without unsupported package/platform, performance,
  proof-owner, or state-of-the-art claims
