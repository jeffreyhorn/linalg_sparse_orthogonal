# Sprint 47 Plan: Benchmark CLI Modernization, Auxiliary Surface Safety & Example/Tooling Cleanup

**Sprint Duration:** 14 days  
**Goal:** Bring benchmark CLIs, examples, and auxiliary tooling up to the same usability and safety standard as the core library while preserving the validated Epic 4 baseline. This sprint implements the Sprint 47 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`.

**Starting Point:** Sprint 46 closed with reusable repeated-run benchmark support already in place for iterative solvers and eigensolvers, Sprint 41 established the shared safety-helper conventions, and the benchmark/example/tooling surface still carries older parsing, auxiliary-safety, and documentation drift. Sprint 47 begins from that validated baseline and turns the auxiliary surface into a bounded cleanup target without reopening core solver architecture work.

**End State:** Sprint 47 leaves behind shared CLI parsing helpers, a modernized `bench_main` and aligned peer benchmark/tool surfaces, examples and auxiliary support code that follow the current safety conventions, refreshed benchmark/example documentation, and a validation record showing the broader auxiliary cleanup still satisfies the maintained local reviewed baseline.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 136 hours, matching the Sprint 47 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 47 Scope Audit & Baseline Refresh

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 47 project-plan items into a bounded benchmark/example/tooling execution map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 47 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`, the Sprint 40 validation anchor, the Sprint 41 helper/safety closeout, and the Sprint 45 / Sprint 46 benchmark reuse closeouts.
2. Reconfirm the preserved constraints Sprint 47 must not reopen:
   - preserve Sprint 40 validation truthfulness
   - reuse Sprint 41 safety helpers rather than adding a third parsing/allocation framework
   - preserve the current validated benchmark/example behavior while tightening the auxiliary surface
   - keep cleanup bounded away from unrelated core-solver redesign
3. Define the Sprint 47 workstreams explicitly:
   - CLI/helper seam inventory
   - shared parsing-helper design
   - `bench_main` modernization
   - reorder-mode parity cleanup
   - example safety audit and cleanup
   - auxiliary tooling cleanup
   - docs refresh
   - validation closeout
4. Record the highest-risk auxiliary seams:
   - legacy `atoi` / unchecked numeric parsing
   - reorder-mode drift across benchmarks
   - unchecked size/count arithmetic in example/support code
   - stale docs that no longer match live CLI/runtime behavior
5. Open Sprint 47 working notes and record scope, assumptions, and initial landing order.

### Deliverables
- Sprint 47 scope inventory
- Benchmark/example/tooling workstream map
- Working-notes baseline assumptions

### Completion Criteria
- Sprint 47 starts from the documented Epic 4 baseline rather than ad hoc CLI cleanup
- Preserved safety and validation constraints are explicit before implementation begins
- The benchmark/example/tooling targets are named before code changes start

---

## Day 2: CLI and Auxiliary Surface Inventory Refresh

**Title:** Surface Inventory  
**Theme:** Re-map the benchmark, example, and support-code seams before choosing the landing order  
**Time estimate:** 10 hours

### Tasks
1. Refresh the live seam inventory for:
   - `benchmarks/bench_main.c`
   - `benchmarks/bench_eigs.c`
   - related benchmark entry helpers
   - example argument / safety surfaces
   - script-side support code tied to benchmark/example use
2. Classify the current issues into bounded buckets:
   - numeric parsing and error-reporting drift
   - reorder-mode / label parity drift
   - unchecked auxiliary size/count handling
   - outdated helper or support-code patterns
   - docs drift versus live runtime behavior
3. Identify which auxiliary paths are the strongest shared-helper targets versus which are file-local or docs-only.
4. Confirm the first implementation order:
   - shared parsing helpers first
   - `bench_main` next
   - reorder-mode parity after parser stabilization
   - example/tooling cleanup after the benchmark shape is clear
5. Write the inventory artifact.

### Deliverables
- Refreshed benchmark/example/tooling seam inventory
- Shared-vs-local classification
- First landing-order notes

### Completion Criteria
- The auxiliary surface is reduced to named cleanup seams
- Shared parser/helper candidates are distinguished from file-local cleanup
- Later implementation order is grounded in the live code state

---

## Day 3: Shared CLI Parsing Helper Design

**Title:** CLI Helper Design  
**Theme:** Define the reusable parsing helper layer for positive integers, bounded integers, finite doubles, and enum-like modes  
**Time estimate:** 10 hours

### Tasks
1. Design the internal helper surface for benchmark/example CLI parsing:
   - positive integer parsing
   - bounded integer parsing
   - finite double parsing
   - enum-like string mode parsing
2. Define ownership and usage rules:
   - return/error contract
   - invalid-input messaging expectations
   - whether helper output is parse-only or parse-plus-range-check
   - shared helper versus file-local wrapper boundaries
3. Decide where the helper layer should live so it is reusable across benchmarks and examples without becoming a public library API.
4. Record how reorder-mode parsing should map to the current intended benchmark/library capabilities.
5. Write the parsing-helper design artifact.

### Deliverables
- Shared CLI parsing-helper design
- Error-reporting and ownership contract
- Shared-vs-local boundary notes

### Completion Criteria
- Sprint 47 has a concrete parsing-helper design before code edits
- Error and range behavior are explicit instead of ad hoc
- The design stays bounded away from public API expansion

---

## Day 4: Validation and Peer-Surface Landing Design

**Title:** Landing Design  
**Theme:** Bound the first implementation batches and the validation shape before editing code  
**Time estimate:** 8 hours

### Tasks
1. Define the initial implementation-day validation shape:
   - full required gate for all `*.c` / `*.h` changes
   - targeted benchmark/example compile and CLI sanity checks
   - stronger reviewed baseline for broader auxiliary batches when justified
2. Bound the likely first peer surfaces after `bench_main`:
   - `bench_eigs`
   - repeated-run benchmark drivers
   - small example argument or helper seams
3. Decide what should stay out of scope in Sprint 47:
   - broad benchmark framework redesign
   - new public CLI abstractions exported from the core library
   - large tutorial or README restructuring beyond the specific benchmark/example refresh
4. Confirm the mid-sprint order for parser, benchmark, parity, example, tooling, docs, and validation work.
5. Write the landing/validation design artifact.

### Deliverables
- Validation-plan artifact
- Mid-sprint landing order
- Explicit out-of-scope notes

### Completion Criteria
- The sprint has a clear validation contract before implementation begins
- The main implementation batches are sequenced from the live inventory
- Scope boundaries are explicit before code changes start

---

## Day 5: Shared CLI Parsing Helper Batch

**Title:** Parser Helper Batch  
**Theme:** Land the reusable parsing helper layer that later benchmark/example cleanup will consume  
**Time estimate:** 12 hours

### Tasks
1. Add the first bounded shared CLI parsing helper seam using the Day 3 design.
2. Implement the agreed helper cases for:
   - positive integers
   - bounded integers
   - finite doubles
   - enum-like mode parsing
3. Keep the batch narrow:
   - no broad benchmark rewrite yet
   - no documentation batch yet
   - no public library API changes
4. Update internal declarations/build wiring as needed.
5. Run the required code-quality gate and targeted helper-consumer sanity checks justified by the touched seam.

### Deliverables
- Shared CLI parsing helper layer
- Updated declarations/build wiring
- Validation result for the first helper batch

### Completion Criteria
- A real reusable parsing helper seam exists outside the individual CLIs
- The helper cases needed for benchmark/example cleanup are implemented
- The required validation passes

---

## Day 6: `bench_main` Parser Modernization Batch

**Title:** `bench_main` Batch  
**Theme:** Replace legacy parsing and tighten error reporting in the main benchmark CLI  
**Time estimate:** 12 hours

### Tasks
1. Migrate `bench_main` to the new shared parsing helper seam.
2. Replace legacy parsing patterns and align invalid-input handling with the Day 3 contract.
3. Tighten option/error reporting so unsupported or malformed inputs fail clearly.
4. Keep the batch bounded:
   - preserve current benchmark capability scope
   - do not widen into reorder-mode parity cleanup yet except where required by parser correctness
5. Run the required code-quality gate and targeted `bench_main` CLI sanity checks.

### Deliverables
- Modernized `bench_main` parser
- Improved benchmark CLI error reporting
- Validation result for the `bench_main` batch

### Completion Criteria
- `bench_main` no longer relies on the older ad hoc parsing style
- Error reporting is clearer and more consistent
- The required validation passes

---

## Day 7: `bench_main` Post-Landing Audit

**Title:** Parser Audit  
**Theme:** Audit the post-Day-6 benchmark state to confirm what remains for reorder parity and peer CLIs  
**Time estimate:** 8 hours

### Tasks
1. Review the post-Day-6 benchmark state and identify remaining drift in:
   - reorder-mode support
   - emitted labels / reporting
   - peer benchmark argument surfaces
2. Separate:
   - direct reorder-mode parity fixes
   - parser-helper consumers that are now trivial follow-ons
   - lower-priority peer surfaces that should stay outside the main Sprint 47 path
3. Confirm the bounded Day 8 target set for reorder-mode parity cleanup.
4. Record any helper-layer adjustments still needed before peer adoption.
5. Write the post-landing audit artifact.

### Deliverables
- Post-`bench_main` audit
- Bounded reorder-parity target list
- Peer-surface follow-on notes

### Completion Criteria
- The remaining benchmark queue is concrete rather than generic
- Reorder-mode parity targets are explicit before the next batch
- The benchmark cleanup sequence remains bounded

---

## Day 8: Reorder-Mode Parity Cleanup Batch

**Title:** Reorder Parity Batch  
**Theme:** Bring benchmark reorder-mode handling and emitted labels into parity with the intended capabilities  
**Time estimate:** 10 hours

### Tasks
1. Land the bounded reorder-mode parity fixes identified on Day 7.
2. Align supported modes, labels, and emitted reporting across the touched benchmark surfaces.
3. Preserve current benchmark semantics while removing known naming or parity drift.
4. Keep the batch bounded away from broad CLI redesign outside the touched parity seam.
5. Run the required code-quality gate and targeted benchmark CLI sanity checks.

### Deliverables
- Reorder-mode parity cleanup
- Updated emitted-label / reporting alignment
- Validation result for the parity batch

### Completion Criteria
- The touched benchmark surfaces present reorder capabilities consistently
- Labels and runtime behavior are aligned
- The required validation passes

---

## Day 9: Example Safety Audit & Batch Design

**Title:** Example Audit  
**Theme:** Audit the example surface for unchecked arithmetic, weak parsing, and outdated helper patterns before editing examples  
**Time estimate:** 10 hours

### Tasks
1. Audit the examples for:
   - unchecked size/count arithmetic
   - weak or duplicated argument parsing
   - outdated helper patterns now superseded by Sprint 41 or Sprint 47 seams
2. Classify the examples into:
   - direct shared-helper adoption targets
   - example-local helper cleanup
   - docs-only examples that do not justify code churn
3. Choose a bounded example batch for Day 10.
4. Record which example surfaces should remain intentionally untouched in Sprint 47.
5. Write the example audit artifact.

### Deliverables
- Example safety audit
- Bounded Day 10 target list
- Explicit defer/keep notes

### Completion Criteria
- The example cleanup queue is concrete rather than generic
- Direct shared-helper adoption candidates are explicit
- Day 10 is bounded before code changes begin

---

## Day 10: Example Safety Cleanup Batch

**Title:** Example Cleanup  
**Theme:** Align the selected examples to the current safety and helper conventions  
**Time estimate:** 10 hours

### Tasks
1. Migrate the bounded example set identified on Day 9.
2. Adopt the shared parsing/helper seam where it clearly improves consistency.
3. Fix touched unchecked arithmetic or weak auxiliary patterns in the selected examples.
4. Keep the batch narrow:
   - no broad example-style rewrite
   - no public library behavior changes
   - no unrelated benchmark churn
5. Run the required code-quality gate and targeted touched-example sanity checks.

### Deliverables
- Safer touched example surfaces
- Shared-helper adoption where justified
- Validation result for the example batch

### Completion Criteria
- The selected examples follow the current safety conventions more closely
- The batch stays bounded to the audited target set
- The required validation passes

---

## Day 11: Auxiliary Tooling Safety Cleanup Batch

**Title:** Tooling Cleanup  
**Theme:** Align benchmark/example support code and auxiliary tooling with the shared safety conventions  
**Time estimate:** 10 hours

### Tasks
1. Migrate the bounded auxiliary support-code surfaces identified earlier in the sprint.
2. Tighten any touched unchecked parsing, size/count handling, or outdated helper patterns.
3. Keep the batch focused on auxiliary safety and usability rather than framework redesign.
4. Reconcile any touched benchmark/example support seams so they match the new helper contract.
5. Run the required code-quality gate and targeted touched-tool sanity checks.

### Deliverables
- Auxiliary tooling safety cleanup
- Support-code/helper alignment
- Validation result for the tooling batch

### Completion Criteria
- The touched auxiliary surfaces follow the shared safety conventions
- The batch stays bounded to the identified support-code seams
- The required validation passes

---

## Day 12: Benchmark/Example Docs Refresh

**Title:** Docs Refresh  
**Theme:** Update benchmark and example documentation so it matches the modernized CLI and safer auxiliary behavior  
**Time estimate:** 8 hours

### Tasks
1. Refresh the relevant benchmark/example docs to match the live Sprint 47 behavior:
   - supported flags
   - error/usage expectations
   - reorder-mode descriptions where touched
   - repeated-run benchmark references where needed
2. Keep the docs batch bounded to the touched benchmark/example surfaces.
3. Record any intentionally deferred broader tutorial/README cleanup.
4. Capture maintainer-facing notes where the new helper/safety conventions affect future auxiliary work.
5. Write the docs-refresh artifact and update working notes.

### Deliverables
- Refreshed benchmark/example documentation
- Maintainer-facing notes for touched helper/safety conventions
- Explicit defer notes for broader docs work

### Completion Criteria
- The touched docs reflect live CLI/runtime behavior
- Sprint 47 leaves behind clearer benchmark/example usage guidance
- The docs batch stays bounded to the touched surfaces

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full quality gate and the targeted benchmark/example/tooling checks for the Sprint 47 surface  
**Time estimate:** 10 hours

### Tasks
1. Run the full required validation gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the stronger reviewed baseline:
   - `make quality-review-full`
3. Run the targeted benchmark/example follow-ons justified by the touched Sprint 47 surface.
4. Reconcile any small validation issues that surface and rerun the full authoritative gate if needed.
5. Record measured results, parity checks, and any remaining residual risks in the validation artifact and working notes.

### Deliverables
- Full Sprint 47 validation record
- Targeted benchmark/example/tooling follow-on results
- Reconciled measured baseline notes

### Completion Criteria
- The full required gate passes
- The stronger reviewed baseline passes
- The targeted auxiliary follow-ons are recorded and green

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Synthesize the Sprint 47 results, residual queue, and handoff constraints for the next Epic 4 phase  
**Time estimate:** 8 hours

### Tasks
1. Summarize what Sprint 47 actually landed across:
   - shared CLI parsing helpers
   - `bench_main` modernization
   - reorder-mode parity cleanup
   - example/tooling safety alignment
   - benchmark/example docs refresh
2. Record the validated end-state and the preserved constraints for later work.
3. Classify any residual deferred benchmark/example/tooling work that should be carried forward in future Epic 4 planning.
4. Update Sprint 47 working notes with the final synthesis.
5. Write the closeout/handoff artifact.

### Deliverables
- Sprint 47 closeout and handoff artifact
- Final working-notes synthesis
- Residual queue classification for later Epic 4 work

### Completion Criteria
- Sprint 47 ends with one coherent benchmark/example/tooling cleanup handoff
- The validated end-state and remaining queue are explicit
- Later Epic 4 work can pick up the auxiliary-surface state without re-auditing the sprint
