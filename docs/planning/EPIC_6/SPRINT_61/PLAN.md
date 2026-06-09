# Sprint 61 Plan: Configuration Surface Modernization Phase 1

**Sprint Duration:** 14 days  
**Goal:** Replace the highest-value process-global environment-variable tuning
controls with typed option surfaces and explicit precedence rules. This sprint
implements the Sprint 61 section of
`docs/planning/EPIC_6/PROJECT_PLAN.md`.

**Starting Point:** Sprint 60 closed with the Epic 6 baseline, architecture
contract, and validation/platform contract frozen:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- the repeated-run workflow fence is already fixed and must not widen
- the strongest near-term configuration target is now explicit:
  `SPARSE_ND_*`, `SPARSE_FM_*`, and adjacent advisory controls

The next highest-value work is no longer broad audit or target-definition
work. It is a bounded implementation sprint centered on typed configuration
surfaces, precedence rules, compatibility behavior, and supporting regression
and docs updates.

**End State:** Sprint 61 leaves behind the first coherent Phase 1 typed
configuration package:
- a ranked live env-var inventory grounded in the repo state
- explicit public/internal typed option design and precedence rules
- first landed typed option surfaces for the highest-value reorder/ND controls
- bounded migration of selected analysis-time controls out of process-global
  env vars where justified
- explicit legacy compatibility behavior
- regression, docs, and validation coverage for the new configuration model

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
146 hours, staying within the Sprint 61 estimate and below the 168-hour limit.

---

## Day 1: Sprint 61 Scope Audit & Configuration Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 61 project-plan scope plus the Sprint 60 frozen
contract into a bounded implementation map  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Sprint 61 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 60 retrospective, and
   the strongest Sprint 60 contract artifacts.
2. Reconfirm the preserved Sprint 61 constraints:
   - no reopening the Epic 5 repeated-run workflow fence
   - no broad backend-policy rewrite in the same batch
   - no packaging/platform widening disguised as configuration work
   - no fake removal of all legacy env-var behavior in Phase 1
3. Define the Sprint 61 workstreams explicitly:
   - env-var inventory
   - typed option design
   - reorder/ND integration
   - analysis/postorder integration
   - compatibility behavior
   - regression/docs updates
   - validation and closeout
4. Record the strongest likely Sprint 61 touch surfaces:
   - public option-bearing headers
   - graph/reorder and analysis implementation seams
   - docs and maintainer truthfulness surfaces
   - proof surfaces likely to need expansion
5. Open Sprint 61 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 61 scope inventory
- Configuration-modernization baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 61 starts from the Sprint 60 frozen contract rather than reopening
  target-definition debates
- The implementation workstreams are explicit before deeper investigation
  begins
- The sprint non-goal fence is fixed before design or code edits land

---

## Day 2: Validation Baseline & Code-Day Gate Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed baseline and rerun set that Sprint 61 code
changes must preserve  
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
3. Reconfirm the stronger default for substantial control-plane or
   architecture-sensitive work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Sprint 61:
   - direct lifecycle proofs
   - graph/reorder-sensitive proofs
   - representative examples and benchmarks
   - parity/count anchors
5. Record the authoritative validation split for:
   - docs-only days
   - bounded code-touching days
   - substantial control-plane days

### Deliverables
- Refreshed validation notes
- Sprint 61 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 61 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before implementation work begins
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Env-Var Surface Inventory

**Title:** Env-Var Audit  
**Theme:** Re-rank the live `SPARSE_ND_*`, `SPARSE_FM_*`, and related controls
by value, user pain, and migration risk  
**Time estimate:** 11 hours

### Tasks
1. Inventory all live configuration env-var seams:
   - `SPARSE_ND_*`
   - `SPARSE_FM_*`
   - `SPARSE_SUPERNODAL_POSTORDER`
   - adjacent analysis/reorder advisory controls
2. Classify each control by:
   - likely public typed-option candidate
   - likely internal typed policy candidate
   - likely compatibility-only override candidate
   - likely stay-internal diagnostic/debug path
3. Rank the controls by:
   - user-facing value
   - architectural leverage
   - migration complexity
   - regression risk
4. Identify the strongest Phase 1 cut line for Sprint 61.
5. Write the inventory artifact with the explicit ranked control map.

### Deliverables
- Live env-var inventory
- Ranked Phase 1 candidate list
- Public/internal/compatibility classification draft

### Completion Criteria
- The broad “too env-var driven” claim is reduced to a concrete ranked list
- The strongest Phase 1 controls are explicit before option design begins
- Day 4 can proceed from a real migration target instead of from generic
  cleanup goals

---

## Day 4: Typed Options Design & Precedence Contract

**Title:** Options Design  
**Theme:** Define the public/internal typed option surfaces and exact
precedence rules before code changes begin  
**Time estimate:** 12 hours

### Tasks
1. Design the Phase 1 typed option model for the highest-value controls.
2. Define exact precedence rules across:
   - explicit typed option values
   - default typed option values
   - internal policy defaults
   - legacy env-var compatibility overrides
3. Decide which controls belong in:
   - public option structs
   - internal typed policy structs
   - compatibility translation helpers
4. Define the explicit compatibility behavior:
   - where env vars still map into typed options
   - where env vars remain the only backward-compat path
   - where no compatibility mapping is justified
5. Record the landing fence for the first integration batch.

### Deliverables
- Typed options design artifact
- Explicit precedence/override contract
- Phase 1 landing fence

### Completion Criteria
- The future control-plane behavior is explicit before implementation edits
- Public and internal ownership are separated clearly enough to prevent drift
- Compatibility behavior is defined tightly enough to support regression work

---

## Day 5: Header and Internal-Surface Landing Design

**Title:** Landing Design  
**Theme:** Convert the option design into a precise touched-file and API/impl
boundary plan  
**Time estimate:** 10 hours

### Tasks
1. Identify the exact public header surfaces to widen or normalize.
2. Identify the exact internal implementation seams to touch first:
   - reorder/ND entry points
   - analysis-time control plumbing
   - compatibility translation flow
3. Define the minimum viable Phase 1 public API additions.
4. Define the minimum viable internal helper additions or refactors.
5. Record the Day 6-7 code landing boundary and explicit non-goals.

### Deliverables
- Header/internal landing design
- Exact touched-surface map
- Day 6-7 implementation fence

### Completion Criteria
- The implementation batch has an explicit touched-file plan
- The first landing is bounded tightly enough to preserve momentum and safety
- Non-goals are clear before public-header or implementation edits start

---

## Day 6: Reordering Option Integration Batch I

**Title:** Integration I  
**Theme:** Land the first typed configuration surfaces for the highest-value
reorder/ND controls  
**Time estimate:** 12 hours

### Tasks
1. Add the first public/internal typed option fields or structures.
2. Thread the highest-value reorder/ND control path through the new typed
   option surface.
3. Preserve existing behavior where explicit typed options are not supplied.
4. Add or adjust compatibility translation for the selected env-var controls.
5. Run the required code-day validation gate and any targeted follow-ons.

### Deliverables
- First landed typed configuration surfaces
- First reorder/ND typed integration batch
- Validation results for the batch

### Completion Criteria
- At least one high-value reorder/ND control is available through the typed
  configuration path
- Existing default behavior remains stable
- Required validation passes before Day 7 proceeds

---

## Day 7: Reordering Option Integration Batch II

**Title:** Integration II  
**Theme:** Complete the Phase 1 reorder/ND landing and tighten the control
flow around it  
**Time estimate:** 12 hours

### Tasks
1. Land the remaining bounded Phase 1 reorder/ND controls selected in Day 4.
2. Tighten defaulting and precedence handling around the new typed path.
3. Normalize any touched public/internal naming or comments needed for clarity.
4. Add or expand regression coverage around the landed controls.
5. Run the required gate and any targeted reruns driven by the touched
   surfaces.

### Deliverables
- Completed Phase 1 reorder/ND integration batch
- Expanded regression support
- Validation results for the second batch

### Completion Criteria
- The selected reorder/ND Phase 1 surface is fully landed
- Precedence behavior is no longer ambiguous on the landed path
- Validation remains clean after the full integration batch

---

## Day 8: Analysis/Postorder Integration Audit

**Title:** Analysis Audit  
**Theme:** Re-audit the remaining analysis-time and postorder-related controls
after the first landing  
**Time estimate:** 9 hours

### Tasks
1. Re-read the landed control flow after Days 6-7.
2. Identify the next strongest analysis-time controls still living in env-var
   space.
3. Separate:
   - controls that should move in Sprint 61
   - controls that should stay compatibility-only for now
   - controls that should defer to a later sprint
4. Define the exact bounded Day 9 landing target.
5. Record any new risks exposed by the landed typed path.

### Deliverables
- Post-landing analysis/postorder audit
- Ranked next control slice
- Day 9 landing target

### Completion Criteria
- The post-Day-7 queue is smaller and more concrete than the original Sprint
  61 scope
- The next integration slice is explicit before more code moves
- No accidental broadening of scope is required to proceed

---

## Day 9: Analysis/Postorder Integration Design

**Title:** Analysis Design  
**Theme:** Convert the remaining justified analysis-time controls into a
bounded implementation plan  
**Time estimate:** 10 hours

### Tasks
1. Define the exact analysis/postorder control subset to move.
2. Design the public/internal plumbing for that subset.
3. Define exact precedence and compatibility behavior for the Day 10 batch.
4. Confirm which controls remain explicitly deferred.
5. Record the Day 10 code landing fence and regression obligations.

### Deliverables
- Analysis/postorder integration design
- Explicit Day 10 implementation fence
- Deferred-control list

### Completion Criteria
- The next code batch is precise rather than generic
- Compatibility behavior is explicit before the batch lands
- Deferred controls are named rather than silently dropped

---

## Day 10: Analysis/Postorder Integration Batch

**Title:** Integration III  
**Theme:** Land the bounded analysis-time and postorder-related typed control
surfaces selected in Day 9  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 9 analysis/postorder typed control path.
2. Preserve stable defaults on untouched caller paths.
3. Land the bounded compatibility or translation behavior still justified.
4. Add or adjust regression coverage for the new path.
5. Run the required code-day validation gate and targeted follow-ons.

### Deliverables
- Analysis/postorder typed integration batch
- Compatibility translation updates
- Validation results for the batch

### Completion Criteria
- The selected analysis/postorder controls are available through the new typed
  surface
- Stable defaults and backward behavior remain intact where promised
- Required validation passes before closeout work begins

---

## Day 11: Compatibility Layer & Regression Sweep

**Title:** Compatibility Sweep  
**Theme:** Tighten the legacy env-var behavior and prove the new precedence
model explicitly  
**Time estimate:** 11 hours

### Tasks
1. Review the full landed compatibility path:
   - typed option only
   - default internal policy
   - env-var compatibility override
2. Add or tighten regression tests around:
   - precedence
   - compatibility fallback
   - stable default behavior
3. Remove or clarify any stale wording/comments around the old env-var-only
   model.
4. Record the post-landing compatibility state.
5. Run the required validation gate and targeted reruns.

### Deliverables
- Compatibility-layer cleanup
- Precedence/override regression coverage
- Post-landing compatibility notes

### Completion Criteria
- The Phase 1 precedence story is explicitly proven
- The remaining env-var behavior is bounded and intentional
- No stale wording implies broader or different control behavior than shipped

---

## Day 12: Docs & Maintainer Story Update

**Title:** Docs Follow-Through  
**Theme:** Align caller-facing and maintainer-facing surfaces with the new
typed configuration model  
**Time estimate:** 9 hours

### Tasks
1. Update the highest-value public docs and headers for the landed
   configuration story.
2. Update maintainer guidance around:
   - precedence
   - compatibility behavior
   - preferred typed configuration path
3. Update example or benchmark references only if needed for truthfulness.
4. Record the exact residual deferred configuration queue.
5. Run docs/workflow sanity checks against the touched surfaces.

### Deliverables
- Updated public/maintainer configuration docs
- Explicit residual deferred queue
- Docs follow-through notes

### Completion Criteria
- The shipped typed configuration story is coherent across API, docs, and
  maintainer guidance
- The residual env-var queue is explicit rather than implied
- No caller-facing contradiction remains on the touched surfaces

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Revalidate the full Sprint 61 landed state from the strongest
reviewed baseline  
**Time estimate:** 12 hours

### Tasks
1. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Recheck:
   - reviewed CMake parity counts
   - Makefile/CMake parity
   - full reviewed CMake `ctest`
3. Run the targeted Sprint 61 follow-on set:
   - direct lifecycle proofs
   - graph/reorder-sensitive proofs
   - representative examples
   - representative benchmark drivers
4. Record representative retained behavior/results.
5. Capture any non-blocking warnings or residual notes honestly.

### Deliverables
- Full validation artifact
- Reviewed parity results
- Targeted follow-on results

### Completion Criteria
- Full required validation passes
- Reviewed parity anchors stay exact or any change is reconciled explicitly
- No hidden regression queue remains before closeout

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Package Sprint 61 into a clean Epic 6 handoff for later
configuration and control-plane work  
**Time estimate:** 7 hours

### Tasks
1. Re-read the Sprint 61 plan and final landed artifacts.
2. Summarize:
   - what typed configuration surfaces shipped
   - what precedence rules are now fixed
   - what compatibility behavior remains
   - what residual controls remain for later sprints
3. Confirm whether `PROJECT_PLAN.md` needs any correction from landed results.
4. Record the preserved non-goal fence after the first implementation sprint.
5. Write the Day 14 closeout artifact and final working-notes summary.

### Deliverables
- Sprint 61 closeout artifact
- Final working-notes synthesis
- Explicit next-sprint starting queue

### Completion Criteria
- Sprint 61 closes from a validated and clearly documented landed state
- The next configuration/control-plane queue is explicit
- Later Epic 6 work can continue without reopening Sprint 61 contract
  decisions
