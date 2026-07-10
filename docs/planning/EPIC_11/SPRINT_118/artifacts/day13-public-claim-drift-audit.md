# Sprint 118 Day 13 Public Claim Drift Audit

## Purpose

Day 13 audits public and support documentation against the Day 8 product truth
map. The goal is to identify unsupported, stale, partially supported, or
candidate-only wording before Sprints 126-127 perform adoption cleanup and
final claim recalibration.

This audit does not rewrite public docs. It records drift risk, required
fences, and future owners while preserving the Sprint 118 boundary that
baseline work should not perform future implementation or broad adoption
rewrites.

## Audit Inputs

| Input | Use |
|---|---|
| `artifacts/day8-product-truth-map.md` | Baseline claims, candidate claims, explicit non-claims, and evidence references. |
| `artifacts/day4-ci-tier-platform-truth.md` | Platform, package, install, and staged-exclusion truth. |
| `artifacts/day6-residual-owner-map.md` | Future owner and non-claim handoff. |
| `artifacts/day10-hotspot-owner-handoff.md` | Maintainability and proof-owner claim boundaries. |
| `templates/adoption-cleanup-evidence-template.md` | Future adoption cleanup evidence fields for Sprint 126-127. |

## Audited Surfaces

| Surface | Audit disposition |
|---|---|
| `README.md` | Primary public front door; generally aligned, with one identity-positioning handoff for Sprint 126. |
| `INSTALL.md` | Static-first package and platform-tier wording is aligned with Day 4 and Day 8. |
| `docs/solver_selection.md` | Aligned; explicitly keeps benchmarks local and rejects nonsymmetric eigensolver/state-of-the-art parity. |
| `docs/matrix_market.md` | Aligned; exact supported/unsupported Matrix Market surface is clear. |
| `benchmarks/README.md` | Aligned; local measurement and benchmark/report boundaries are explicit. |
| `examples/README.md` | Aligned; examples remain teaching/adoption surfaces, not benchmark or oracle owners. |
| `docs/maintainer_guide.md` | Aligned; strongly preserves reviewed/supplemental/staged, package/ABI, oracle, and performance non-claims. |
| `docs/tutorial.md` | Spot-audited through claim scans; no unsupported claim found. |
| `docs/algorithm.md` | Caveated as non-adoption/non-performance-contract doc, but remains dense and historical; Sprint 126 owner. |

## Public/Support Claim Audit Table

| Claim area | Public/support wording observed | Day 8 truth comparison | Disposition | Future owner |
|---|---|---|---|---|
| Product identity | README opens with "orthogonal linked-list" and later presents compressed-first CSR/CSC routes. | Day 8 says compressed-first is preferred when callers already have CSR/CSC, while mutable shell remains compatibility. | Partially aligned; not unsupported, but user-facing identity can be clearer. | Sprint 126 adoption cleanup. |
| One-shot direct solvers | README and solver-selection guide list LU, Cholesky, LDLT, QR as first workflows. | Supported baseline truth with bounded proof breadth. | Aligned. | Sprint 120 only if oracle/proof wording expands. |
| Repeated direct lifecycle | README, solver-selection, examples, and tutorial describe analyze/factor/refactor for stable sparsity patterns. | Supported baseline truth. | Aligned; caveats are present. | Sprint 120 if test split/oracle work changes evidence. |
| Iterative handles | README and examples say repeated handles are bounded to CG, GMRES, and MINRES. | Matches Day 8 handle-scope caveat. | Aligned. | Sprint 120. |
| Eigensolver support | README and examples present symmetric eigensolver workflows, shift-invert, thick restart, and LOBPCG. | Supported baseline truth with source-boundary and external-comparison caveats. | Aligned; no nonsymmetric or ARPACK parity claim found. | Sprints 119-120. |
| SVD/QR/rank support | README, examples, and solver-selection present SVD, QR, rank, condition, pseudoinverse, low-rank, and min-norm workflows. | Supported baseline truth with LAPACK/SciPy parity caveats. | Aligned; no broad dense-library parity claim found. | Sprint 121. |
| Matrix Market I/O | Matrix Market docs list real, pattern, symmetric, and integer coordinate support, duplicate handling, and unsupported array/complex/hermitian/skew features. | Day 8 baseline is bounded to documented variants. | Aligned and more precise than the Day 8 abbreviated wording. | Sprint 126 only for cookbook routing. |
| Graph and reordering | README and docs describe RCM, AMD, ND, COLAMD, typed controls, and benchmark handoff. | Supported baseline with universal-fill caveats. | Aligned; algorithm doc has historical performance detail but also caveats local interpretation. | Sprint 123 for guardrails; Sprint 126 for docs split. |
| Package/install | INSTALL and README describe static-first install, `pkg-config`, and CMake `find_package(Sparse)`. | Matches Day 8 package truth. | Aligned. | Sprint 124. |
| ABI/shared library | INSTALL and maintainer guide explicitly say no broad shared-library or dynamic-ABI promise. | Matches explicit non-claim. | Aligned. | Sprint 124. |
| Package-manager support | INSTALL lists OS package commands for prerequisites only and does not claim library package-manager distribution. | Matches explicit non-claim. | Aligned. | Sprint 124 residual/future epic if pursued. |
| Platform support | README, INSTALL, and maintainer guide keep Linux strongest, macOS narrower/supplemental, Windows CMake subset. | Matches Day 4 and Day 8 platform truth. | Aligned. | Sprint 125. |
| Benchmark/performance | README, solver-selection, benchmarks README, maintainer guide, and algorithm doc repeatedly fence benchmarks as local measurement. | Matches Day 8 benchmark truth and explicit non-claims. | Aligned; no portable performance claim found. | Sprint 123; Sprint 126 docs split. |
| State-of-the-art/library parity | Solver-selection and maintainer guide explicitly reject portable state-of-the-art or ecosystem parity; README does not claim it. | Matches explicit non-claim. | Aligned. | Sprint 127 final recalibration. |
| External oracle breadth | Maintainer guide carefully limits external dense-reference evidence to named solver-family lanes. | Matches Day 8 caveat that external-oracle breadth is selected. | Aligned. | Sprints 120-122. |
| Adoption docs | README, examples, tutorial, solver-selection, install, benchmarks, and maintainer guide split ownership clearly. | Day 8 says docs are usable but dense. | Aligned, but scanability and historical density remain real. | Sprint 126. |

## Unsupported Or Stale Claim List

No immediate unsupported public claim requiring a Sprint 118 edit was found.

The scan did not find public wording that silently claims:

- broad state-of-the-art replacement;
- SuiteSparse, PETSc, Trilinos, ARPACK, LAPACK, NumPy/SciPy, GraphBLAS, or
  vendor-backend parity;
- every solver family has broad external-oracle coverage;
- portable performance superiority;
- universal reorder or fill-reduction superiority;
- shared-library dynamic ABI stability;
- package-manager distribution support;
- symmetric Linux, macOS, and Windows reviewed parity;
- Windows Makefile, install-validation, thread/fuzz/property, or full CTest
  parity;
- GPU support;
- distributed-memory support;
- broad complex or mixed-precision maturity.

## Partially Supported Or Candidate-Only Claim List

These are not current public-doc defects, but future sprints should keep them
fenced until owner evidence exists:

| Candidate wording area | Current disposition | Owner |
|---|---|---|
| Stronger compressed-first public identity | Current docs support compressed-first routes but README still opens with orthogonal linked-list identity. | Sprint 126. |
| Source-boundary maintainability improvement | Current docs do not claim it as earned; Day 10 only ranks targets. | Sprint 119 and Sprint 127. |
| Broader direct/iterative/eigensolver/SVD oracle confidence | Maintainer guide preserves named-lane trust boundaries. | Sprints 120-122. |
| Report-index and corpus architecture | Current docs mention report surfaces but do not claim a polished recurring assurance architecture. | Sprint 122. |
| Performance sentinel governance | Current docs describe local sentinels; no portable claim. | Sprint 123. |
| ABI/shared-library productization | Current docs explicitly defer. | Sprint 124. |
| Wider platform install validation | Current docs keep platform tiers and staged exclusions explicit. | Sprint 125. |
| Simplified adoption/cookbook path | Current docs are usable but dense. | Sprint 126. |
| Final earned claim table | Current Sprint 118 truth map is a baseline; final Epic 11 claims wait for owner sprint evidence. | Sprint 127. |

## Edit, Fence, Or Future-Owner Recommendations

| Recommendation | Action | Owner |
|---|---|---|
| Do not edit public docs during Sprint 118 Day 13. | Current wording is aligned enough for the baseline sprint, and broad adoption cleanup is explicitly owned later. | Sprint 118 no-op. |
| Make compressed-first the clearer first-viewport product story. | In Sprint 126, consider revising README's first sentence and first workflow path so compressed-first solving is the public center while mutable shell remains compatibility. | Sprint 126. |
| Split or restructure algorithm reference. | Preserve historical evidence but make current algorithm reference easier to scan and keep local measurement caveats attached. | Sprint 126. |
| Keep package/ABI wording static-first until a product decision is made. | Use the Day 12 package/ABI template before changing README, INSTALL, maintainer guide, or workflows. | Sprint 124. |
| Keep platform wording tiered. | Any Linux/macOS/Windows expansion needs reviewed-lane proof, expected-count updates, and staged-exclusion updates. | Sprint 125. |
| Keep benchmark wording local. | Any sentinel/report changes must preserve local machine/compiler/backend/thread context and no portable-performance claim. | Sprint 123. |
| Use final claim recalibration before Epic 11 closeout. | Re-run this audit after Sprints 119-126 have actual evidence outcomes. | Sprint 127. |

## Sprint 126-127 Handoff Notes

| Sprint | Handoff |
|---|---|
| 126 | Use `templates/adoption-cleanup-evidence-template.md` for README, algorithm docs, examples, tutorial, solver-selection, Matrix Market, benchmark docs, and install-doc wording changes. Preserve the Day 8 non-claim list while improving scanability. |
| 127 | Re-audit all public/support surfaces after owner-sprint outcomes. Publish earned claims, unearned candidate claims, explicit residuals, and post-Epic-11 non-claims. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 6 is complete. | Complete. |
| Public/support claim audit table is recorded. | Complete. |
| Unsupported or stale claim list is recorded. | Complete; no immediate unsupported public claim was found. |
| Candidate-only claim list is recorded. | Complete. |
| Edit/fence/future-owner recommendations are recorded. | Complete. |
| Sprint 126-127 adoption and claim handoff notes are recorded. | Complete. |
| Public wording does not silently exceed the current truth map. | Complete for the audited surfaces. |
| Unsupported claims have a cleanup owner or explicit non-claim disposition. | Complete. |
