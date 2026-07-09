# Sprint 117 Day 8 Final Comparison And Claim Cleanup

## Purpose

Day 8 packages the final comparison evidence from Day 7 and checks public and
support documentation for unsupported claims identified on Days 2-3. It records
whether public/support docs needed cleanup and closes Sprint 117 Project Plan
Items 3 and 4 for the current documentation-only branch state.

## Inputs

| Input | Role |
|---|---|
| `artifacts/day2-end-state-claim-inventory.md` | Claim inventory and evidence disposition. |
| `artifacts/day3-end-state-claim-decision.md` | Earned, bounded, deferred, and non-claim decision source. |
| `artifacts/day6-final-validation-package.md` | Final validation package and touched-surface proof. |
| `artifacts/day7-final-comparison-inventory.md` | Final comparison artifact owners, classifications, and provenance. |
| `docs/planning/EPIC_10/SPRINT_116/artifacts/day14-validation-handoff.md` | Adoption-surface final truth and public-claim guardrails. |

## Public/Support Surface Recheck

Day 8 rechecked the active public/support surfaces that can affect closeout
claims:

- `README.md`
- `INSTALL.md`
- `docs/*.md`
- `benchmarks/README.md`
- `examples/README.md`

The focused scan looked for risky claim families:

- unqualified state-of-the-art or replacement language;
- broad ecosystem, solver, backend, or external parity wording;
- portable performance or universal speed/reorder/fill claims;
- shared-library, dynamic ABI, package-manager, Windows install, or symmetric
  platform-parity claims;
- public Matrix I/O module or builder API wording;
- broad source-boundary/proof-owner debt closure wording.

## Cleanup Decision

No public/support documentation edit was required on Day 8.

The current public/support docs already fence the risky wording:

| Surface | Day 8 finding | Decision |
|---|---|---|
| `README.md` | Continuous-integration and benchmark wording stays tiered and local. `make bench-fast` is described as bounded runtime signal, and benchmark rows are local measurement artifacts, not portable guarantees. | Keep. |
| `INSTALL.md` | Platform support remains tiered. Linux is strongest reviewed truth, macOS and Windows are scoped, and Windows does not claim a separate reviewed install-validation lane. | Keep. |
| `docs/solver_selection.md` | Eigensolver and Matrix Market wording explicitly avoids portable state-of-the-art parity, public Matrix I/O module, and builder API claims. | Keep. |
| `docs/matrix_market.md` | Public surface is only `sparse_load_mm(...)` and `sparse_save_mm(...)`; no separate module or builder API is claimed. | Keep. |
| `docs/algorithm.md` | The top note frames the file as technical background, not an adoption guide, install/support contract, package/ABI reference, or portable performance guarantee. Reorder/fill and performance sections keep local measurement caveats. | Keep. |
| `benchmarks/README.md` | Benchmark docs explicitly classify benchmarks as local measurement tools and reject portable performance interpretation. | Keep. |
| `examples/README.md` | Matrix Market and example wording stays workflow-oriented and avoids Matrix I/O module or builder API claims. | Keep. |
| `docs/maintainer_guide.md` | Maintainer-facing interpretation names package, ABI, platform, solver, benchmark, and parity non-claims explicitly. | Keep. |

## Evidence-Bounded Claim Table

| Claim area | Final Day 8 allowed wording | Evidence source | Boundary preserved |
|---|---|---|---|
| Product maturity | Epic 10 improved productization, validation discipline, public docs, support tiers, and claim boundaries. | Days 2-7 artifacts and Sprint 100-116 handoffs. | No unqualified state-of-the-art replacement claim. |
| Direct solver evidence | Selected direct-solver families have named bounded external dense-reference evidence. | Sprint 102 and Day 7 comparison inventory. | No every-family direct-solver external parity claim. |
| Iterative/eigensolver/SVD evidence | Fixture-local residual, convergence, reconstruction, rank, and orthogonality evidence exists. | Sprints 103, 113, 114 and Day 7 inventory. | No ARPACK, LAPACK, SciPy, PETSc, Trilinos, or broad ecosystem parity claim. |
| Benchmark/performance evidence | Benchmark and sentinel outputs are local measurement and regression context. | Sprint 104, Sprint 105, Benchmark docs, Day 7 inventory. | No portable performance, universal speed, or cross-platform max-RSS claim. |
| Reorder/fill evidence | Named fixtures, fill metrics, and report-contract evidence support bounded interpretation. | Sprint 105 and Day 7 inventory. | No universal reorder/fill superiority claim. |
| Package/platform support | Static-first package support and tiered platform support remain the maintained product truth. | Sprints 112, 115, 116 and Day 7 inventory. | No shared-library ABI, package-manager, Windows install parity, or full macOS install/export parity claim. |
| Matrix Market | Public Matrix Market support is load/save functions with documented format boundaries. | Sprint 110, Sprint 111, Sprint 116, current docs. | No separate public Matrix I/O module or builder API claim. |
| Maintainability/source ownership | Touched-owner and proof-owner progress is documented where artifacts and validation exist. | Sprints 106-114 and Day 7 inventory. | No claim that all source-boundary or proof-owner debt is closed. |

## Unsupported-Claim Cleanup Record

| Unsupported claim family | Current public/support state | Day 8 action |
|---|---|---|
| Unqualified state-of-the-art replacement | Not present in active public/support docs. | No edit. |
| Broad external/ecosystem parity | Present only as explicit non-claim or bounded maintainer context. | No edit. |
| Portable performance superiority | Active benchmark and algorithm docs explicitly reject this interpretation. | No edit. |
| Universal reorder/fill superiority | Active docs fence fill/timing rows as named-fixture and local context. | No edit. |
| Shared-library or dynamic ABI support | Maintainer/install docs keep these as non-claims. | No edit. |
| Package-manager support | README/INSTALL wording avoids package-manager support implication. | No edit. |
| Windows install or Makefile parity | Windows remains reviewed CMake subset and CMake-first consumer story. | No edit. |
| Full macOS install/export parity | macOS install/export remains supplemental/static-first where discussed. | No edit. |
| Public Matrix I/O module or builder API | Matrix Market docs, solver guide, and examples explicitly reject this. | No edit. |
| All proof-owner/source-boundary debt closed | Not present as a public/support claim. | No edit. |

## Final Comparison Package

Day 8 packages the Day 7 comparison inventory as the retrospective-ready final
comparison package:

- comparison evidence is owned by named sprint artifacts and test/docs owners;
- public claim evidence is separated from local measurement context;
- supplemental package/install/guardrail evidence is not promoted to reviewed
  platform parity;
- absent Day 7 benchmark and coverage regeneration is not cited as new proof;
- public/support docs already match the final evidence boundaries.

## Items 3 And 4 Closeout

| Project Plan Item | Status | Evidence |
|---|---|---|
| Item 3, "Final Comparison Package" | Complete | Day 7 inventory plus this Day 8 package. |
| Item 4, "Unsupported-Claim Cleanup" | Complete | Focused public/support scan and no-edit cleanup record above. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Items 3 and 4 are complete. | Complete. |
| Public/support wording matches final evidence. | Complete. |
| Final comparison package is ready for retrospectives. | Complete. |
