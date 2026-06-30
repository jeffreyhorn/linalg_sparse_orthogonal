# Sprint 99 Day 1: Closeout Baseline

## Purpose

Day 1 opens Sprint 99 by converting the final Epic 9 project-plan section and
the Sprints 90-98 evidence package into a bounded closeout execution model.
The goal is not to start a new implementation wave. The goal is to define the
inputs, workstreams, validation rules, and landing order needed to close Epic 9
from evidence.

## Sprint 99 Scope

Sprint 99 implements the Epic 9 final integration, competitive calibration,
and closeout phase centered on:

- end-state re-audit
- final competitive comparison sweep
- final fix/no-fix decision
- one bounded final fix batch only if evidence requires it
- residual queue finalization
- full validation and reporting sweep
- Sprint 99 retrospective, Epic 9 retrospective, and handoff package

Non-goals for Day 1:

- no source or header edits
- no build-system edits
- no workflow edits
- no benchmark or report-generation edits
- no test harness edits
- no comparison lane expansion
- no final-fix decision before the audit and evidence sweep
- no broad performance, platform, shared-library, complex-number, or
  mixed-precision claim expansion

## Starting Evidence

### Sprint 90 Claim Fence

Sprint 90 defined Epic 9 as a bounded state-of-the-art sparse linear algebra
library effort. It also froze the comparison and non-goal fence that Sprint 99
must preserve:

- maintained correctness comparison lane
- maintained package-shape comparison lane
- bounded runtime-reference lane
- no fake platform symmetry
- no fake broad shared-library maturity
- no fake broad complex or mixed-precision claim
- no fake benchmark-supremacy claim

Sprint 99 should audit whether the live tree now supports the intended bounded
target, not reinterpret the target into a broader product promise.

### Sprints 91-98 Landed Workstreams

| Sprint | Closeout signal for Sprint 99 |
|---:|---|
| 91 | compressed-first product convergence, public-story proof, and full validation landed |
| 92 | dense/backend maturity, LDLT backend adoption, observability, and full reviewed validation landed |
| 93 | runtime/threading reduction, runtime-control cleanup, and runtime evidence landed |
| 94 | scalar/index capability modernization and support-only solver-family alignment landed |
| 95 | public narrative, examples, install, benchmark, maintainer, and proof-owner naming cleanup landed |
| 96 | large-source and giant-test maintainability cleanup landed with source/test registrations updated |
| 97 | build/package convergence landed with source-list proof and static-first package decision preserved |
| 98 | external correctness and runtime/fill assurance expansion landed with CI/support-surface alignment |

Sprint 99 should treat these as completed evidence inputs while still checking
for live contradictions, stale claims, and residual queues.

### Immediate Sprint 98 Handoff

Sprint 98 handed off four closeout pressure areas:

1. **External correctness.** Broader LDLT CSC Matrix Market or indefinite
   corpus coverage, iterative solver comparison, eigensolver/LOBPCG comparison,
   and QR/SVD comparison all require separate architecture before expansion.
2. **Runtime/fill.** Repeated reorder/fill captures may justify a generated
   report target, but broad timing comparison and canonical report expansion
   remain deferred unless bounded.
3. **Coverage topology.** Coverage remains supplemental and tree-mutating, with
   `make clean` reset guidance after coverage modes.
4. **CI and support surfaces.** Linux remains strongest; macOS and Windows
   remain intentionally scoped proof surfaces rather than full parity claims.

## Final Closeout Workstream Map

| Workstream | Primary question | Day range | Expected output |
|---|---|---:|---|
| End-state re-audit | Do the original Epic 9 contradiction classes still exist in the live tree? | Days 2-3 | contradiction map and comparison scope |
| Competitive comparison sweep | What correctness, runtime/fill, package, usability, and workflow evidence supports final claims? | Days 3-5 | final evidence package inputs |
| Final fix decision | Is any last fix needed for truthful closeout? | Day 6 | fix/no-fix decision and boundary |
| Final bounded fix batch | Can the selected closeout blocker be resolved without scope drift? | Days 7-8 | bounded fix batch or no-op evidence |
| Residual queue finalization | What remains future work vs deliberate non-claim? | Day 9 | post-Epic-9 residual queue |
| Validation/reporting sweep | Does the final tree pass the strongest practical proof set? | Days 10-11 | reviewed validation and reporting results |
| Sprint/Epic closeout | What did Sprint 99 and Epic 9 actually deliver? | Days 12-14 | evidence package, retrospectives, handoff |

## Landing Order

1. Freeze inputs and validation expectations.
2. Re-audit contradiction classes against the live tree.
3. Freeze final comparison scope before executing evidence commands.
4. Execute correctness/runtime/fill evidence before support-surface evidence.
5. Decide on final fixes only from audited contradictions and evidence gaps.
6. Land bounded fixes only inside the written boundary.
7. Finalize residual queue before broad closeout writing.
8. Run strongest validation and reporting commands.
9. Write the evidence-backed Sprint 99 and Epic 9 closeout package.

## Validation Expectations by Touch Surface

| Touch surface | Expected validation |
|---|---|
| planning docs only | `git diff --check` and trailing-whitespace scan |
| public docs or maintainer docs | docs hygiene plus stale-claim scan where claims change |
| scripts | focused script command plus docs hygiene |
| benchmark/reporting commands | focused report/benchmark command and claim-boundary review |
| build, CMake, install/export | local reviewed equivalent and install/export proof where available |
| workflow files | local syntax-equivalent review where possible; CI remains final platform proof |
| `.c` or `.h` files | `make format && make lint && make test` |

## Day 1 Deliverables

- Sprint 99 scope inventory.
- Final closeout workstream map.
- Sprint 99 working-notes baseline.
- Validation expectations for each workstream.
- Authoritative input list.

## Day 1 Exit Criteria

- Sprint 99 starts from the merged Sprint 98 end state.
- Closeout work is bounded before the contradiction re-audit begins.
- Validation requirements are explicit before any final edits.
- The final-fix batch remains evidence-gated rather than assumed.
