# Sprint 118 Day 14 Closeout And Handoff Package

## Purpose

Day 14 closes Sprint 118 by indexing the sprint artifacts, summarizing the
post-Epic-10 baseline truth, and handing Sprints 119-127 the evidence gates,
templates, non-claim boundaries, and residual owner map they need before
implementation work begins.

Sprint 118 remained a baseline, residual-conversion, truth-freeze, metric,
template, and audit sprint. It did not perform eigensolver source movement,
direct/iterative/SVD/QR oracle expansion, performance sentinel changes,
package/ABI implementation, platform expansion, or broad adoption rewrites.

## Artifact Index

| Day | Artifact | Primary deliverable |
|---:|---|---|
| 1 | `artifacts/day1-sprint-intake.md` | Sprint intake, input inventory, scope boundaries, validation expectations, and day-level owner map. |
| 2 | `artifacts/day2-validation-inventory.md` | Reviewed and supplemental validation command inventory, platform expected-count/staged-exclusion notes, and Day 3 checklist. |
| 3 | `artifacts/day3-baseline-quality-recheck.md` | Baseline command output summary, Make/CMake parity, source-list/CTest count evidence, and validation boundary. |
| 4 | `artifacts/day4-ci-tier-platform-truth.md` | CI-tier support map, platform validation boundary table, package/install claim map, staged-exclusion register, and Sprint 124-125 candidates. |
| 5 | `artifacts/day5-residual-intake.md` | Deduplicated Epic 10 residual intake, duplicate fence, category map, already-covered work list, and residual candidate list. |
| 6 | `artifacts/day6-residual-owner-map.md` | Epic 11 residual owner table, dependency order, proof gates, future-epic deferrals, and Sprint 119-127 handoff candidates. |
| 7 | `artifacts/day7-product-truth-map-design.md` | Product-truth categories, evidence inventory, classification rules, template, and Day 8 checklist. |
| 8 | `artifacts/day8-product-truth-map.md` | Current product truth map, baseline claim list, candidate claim list, explicit non-claim list, and evidence cross-reference. |
| 9 | `artifacts/day9-hotspot-metrics.md` | Repository file counts, largest source/test owner tables, mixed-responsibility source list, giant-test proof-owner list, and reproducibility commands. |
| 10 | `artifacts/day10-hotspot-owner-handoff.md` | Ranked hotspot owner map, source-movement prerequisites, giant-test split prerequisites, Sprint 119-123 handoff notes, and defer/no-move candidates. |
| 11 | `artifacts/day11-evidence-template-design.md` | Existing-template inventory, template-gap list, refreshed template outlines, required evidence fields, and Day 12 implementation checklist. |
| 12 | `artifacts/day12-evidence-template-refresh.md` | Published refreshed templates and future-sprint usage rules. |
| 13 | `artifacts/day13-public-claim-drift-audit.md` | Public/support claim audit table, unsupported/stale claim list, candidate-only claim list, and Sprint 126-127 claim handoff. |
| 14 | `artifacts/day14-sprint-closeout-handoff.md` | Sprint closeout, artifact index, validation/product-truth summary, Sprint 119-127 requirements, and residual deferred debt. |

## Refreshed Template Index

| Template | Intended owners |
|---|---|
| `templates/source-movement-evidence-template.md` | Sprints 119-123 source movement, private-owner extraction, internal-header reshaping, and giant-test splits. |
| `templates/oracle-expansion-evidence-template.md` | Sprints 120-122 solver/corpus oracle and dense/external reference work. |
| `templates/performance-sentinel-evidence-template.md` | Sprints 122-123 benchmark/report/sentinel/backend/runtime work. |
| `templates/package-abi-decision-template.md` | Sprints 124-125 static-first, shared-library, ABI, install/export, platform, and package-manager decisions. |
| `templates/adoption-cleanup-evidence-template.md` | Sprints 126-127 docs/examples/cookbook/link/claim-boundary cleanup. |
| `templates/template-usage-notes.md` | Sprints 119-127 template selection, required inputs, validation rules, claim discipline, and owner map. |

## Validation Summary

| Surface | Sprint 118 evidence |
|---|---|
| Documentation hygiene | Repeated `git diff --check` and focused trailing-whitespace scans over `docs/planning/EPIC_11/SPRINT_118` passed on each documentation-only day. |
| Strongest local reviewed baseline | Day 3 ran `make quality-review-full`; it passed. |
| Makefile reviewed path | Day 3 passed `format-check`, `lint`, full `test`, and `deadcode-check` inside the reviewed path. |
| CMake reviewed parity path | Day 3 passed configure, clean build, `ctest -N`, Makefile/CMake test-count parity, and full CTest. |
| CTest count | Day 3 observed `54` CMake registrations. |
| Makefile/CMake parity | Day 3 observed `54` vs `54`. |
| Full CTest | Day 3 observed `54 / 54` passing, `0` failed, total real time `208.17 sec`. |
| Code/header changes | Sprint 118 documentation work did not modify `.c` or `.h` files. |
| Required C quality chain | Not required after Day 3 because later work remained documentation-only; Day 3 already ran the stronger full reviewed baseline. |

## Product Truth Summary

| Truth family | Current Sprint 118 disposition |
|---|---|
| Compressed-first workflows | Supported and preferred when callers already have CSR/CSC data; not yet a pure compressed-only product model. |
| Mutable shell | Supported compatibility and construction surface; not the performance-center claim for all operations. |
| Direct solvers | LU, Cholesky, LDLT, QR, CSR LU, CSC Cholesky/LDLT, one-shot, and selected repeated lifecycle support are baseline truth with bounded proof breadth. |
| Iterative solvers | CG, GMRES, MINRES, BiCGSTAB, block variants, preconditioners, diagnostics, and selected repeated handles are supported within documented limits. |
| Eigensolvers | Symmetric eigensolver workflows are supported; source-boundary and external-comparison breadth remain owner-sprint work. |
| SVD/QR/rank | Supported with current regression evidence; broad LAPACK/SciPy parity remains unclaimed. |
| Matrix Market | Load/save support is bounded to documented coordinate variants and explicit unsupported features. |
| Graph/reorder | RCM, AMD, ND, COLAMD-style surfaces, graph partition helpers, and typed options are supported with bounded caveats. |
| Package/platform | Static-first package story, `pkg-config`, and CMake `find_package` are maintained; Linux is strongest reviewed, macOS and Windows remain tiered. |
| Benchmarks/performance | Local benchmark/report/sentinel surfaces exist; no portable performance or vendor-backend claim. |
| Adoption/docs | Public routes exist and are honest, but scanability and compressed-first product identity remain Sprint 126 work. |

## Explicit Non-Claims Carried Forward

- No broad state-of-the-art sparse linear algebra replacement claim.
- No SuiteSparse, PETSc, Trilinos, ARPACK, LAPACK, NumPy/SciPy, GraphBLAS, or
  vendor-backend parity claim.
- No every-solver-family broad external-oracle coverage claim.
- No portable performance superiority claim.
- No universal reorder or fill-reduction superiority claim.
- No shared-library dynamic ABI stability claim.
- No package-manager distribution support claim.
- No symmetric Linux, macOS, and Windows reviewed parity claim.
- No Windows Makefile, install-validation, thread/fuzz/property, or full CTest
  parity claim.
- No GPU support claim.
- No distributed-memory support claim.
- No broad complex or mixed-precision maturity claim.

## Sprint 119-127 Handoff Requirements

| Sprint | Required Sprint 118 inputs | First proof gate |
|---|---|---|
| 119 | Day 6 residual owner map, Day 8 truth map, Day 10 hotspot handoff, source-movement template. | Eigensolver movement feasibility audit with exact old/new file plan, source-list/CMake impact, focused consumer proof, CTest count evidence, rollback plan, and no broad eigensolver parity claim. |
| 120 | Day 10 direct/iterative targets, Day 12 source/oracle templates, Day 8 non-claims. | Direct/iterative oracle design and giant-test split plan that preserves CTest membership, solver-specific tolerances, failure localization, and full C quality if `.c`/`.h` changes. |
| 121 | Day 10 SVD/QR targets, Day 12 oracle template, Day 8 SVD/QR caveats. | Fixture taxonomy and proof helpers that preserve rank, reconstruction, orthogonality, storage, leading-dimension, tolerance, and no LAPACK/SciPy parity claim. |
| 122 | Day 6 corpus/report owners, Day 11-12 templates, Day 9 risk metrics. | Corpus/report index model with reviewed/supplemental/local classification, stale-report handling, and coverage interpreted by owner risk rather than vanity percentage. |
| 123 | Day 8 benchmark truth, Day 10 graph/reorder/performance handoff, performance template. | Local sentinel/report design with machine/compiler/backend/thread context, threshold/report-only separation, focused validation, and no portable-performance claim. |
| 124 | Day 4 package/platform truth, Day 8 non-claims, package/ABI template. | Explicit static-first continuation or shared-library/ABI product decision with install/export/downstream proof or explicit deferral. |
| 125 | Day 4 staged-exclusion register, Day 8 platform truth, package/ABI template. | Platform install/export or staged-lane changes with reviewed/supplemental classification, expected-count impact, and support wording updates. |
| 126 | Day 8 truth map, Day 13 claim audit, adoption template. | Adoption cleanup that makes compressed-first identity clearer, reduces historical density, checks links/paths, and preserves non-claims. |
| 127 | All Sprint 118 artifacts plus Sprint 119-126 outcomes. | Final claim recalibration with strongest reviewed baseline, earned/unearned claim table, unsupported-claim cleanup, and residual publication. |

## Residual Deferred Debt Created Or Preserved By Sprint 118

Sprint 118 did not create implementation debt by changing code. It preserved
the following intentionally deferred work for owner sprints:

| Residual | Owner |
|---|---|
| Eigensolver private-owner movement and source-boundary decisions. | Sprint 119 |
| `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert setup/conversion, and `lanczos_iterate_op` movement or explicit deferral. | Sprint 119 |
| Direct/iterative generated-RHS oracle and giant-test split work. | Sprint 120 |
| SVD/QR/rank-deficient proof helper and dense/external reference pilot work. | Sprint 121 |
| Numerical corpus taxonomy, report index, and coverage architecture. | Sprint 122 |
| Performance/backend governance, local sentinel design, and graph/reorder guardrail interpretation. | Sprint 123 |
| Package/ABI product decision and package-manager disposition. | Sprint 124 |
| Linux/macOS/Windows install/export and staged-platform follow-through. | Sprint 125 |
| Compressed-first adoption identity, algorithm doc split, cookbook examples, and docs scanability. | Sprint 126 |
| Final earned/non-earned claim recalibration and post-Epic-11 residual publication. | Sprint 127 |

Future-epic or explicit non-claim candidates remain:

- package-manager support unless Sprint 124 earns real recipes and consumer
  proof;
- Windows Makefile parity unless Sprint 125 designs and validates it;
- GPU support;
- distributed-memory support;
- broad ecosystem parity;
- broad complex or mixed-precision maturity;
- portable performance superiority.

## Day 14 Closeout Checklist

| Requirement | Status |
|---|---|
| Complete Sprint 118 artifact index exists. | Complete. |
| Validation and product-truth summary exists. | Complete. |
| Residual owner handoff package exists. | Complete. |
| Hotspot and template handoff package exists. | Complete. |
| Claim-drift handoff package exists. | Complete. |
| Residual deferred debt list exists. | Complete. |
| Every Sprint 118 deliverable has an artifact or explicit deferral. | Complete. |
| Sprint 119 can begin with clear prerequisites, evidence gates, and non-claim boundaries. | Complete. |
