# Sprint 118 Day 5 Residual Queue Intake and Duplicate Fence

## Purpose

Day 5 converts the final Epic 10 residual queue into a deduplicated Epic 11
intake list. It records the raw carry-forward work, removes duplicates across
Sprint 117, the Epic 10 retrospective, the Epic 11 review/todo, and the Epic
11 project plan, and classifies each item by category and scheduled owner.

## Inputs Reviewed

| Input | Residual role |
|---|---|
| `docs/planning/EPIC_10/EPIC_10_RETROSPECTIVE.md` | Authoritative post-Epic residuals, future-epic candidates, optional scanability work, and non-claim table. |
| `docs/planning/EPIC_10/SPRINT_117/RETROSPECTIVE.md` | Sprint 117 residual deferred debt and consciously constrained non-claims. |
| `docs/planning/EPIC_11/reviews/review-codex-2026-07-09.md` | Highest-priority Epic 11 gaps and state-of-the-art readiness limits. |
| `docs/planning/EPIC_11/reviews/todo-codex-2026-07-09.md` | Step-by-step gap-closure sequence and rules for deferral/non-claims. |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` | Sprint 118-127 scheduled owners and estimates. |
| Sprint 118 Day 2-4 artifacts | Current validation, platform, package, and support-tier truth. |

## Raw Residual Intake

| Raw item | Source evidence | Category |
|---|---|---|
| Move one eigensolver private owner with exact source-list, CMake, consumer, CTest, and rollback proof. | Epic 10 retrospective; Sprint 117 retrospective; Epic 11 todo. | Source owner |
| Revisit `s20_select_indices` movement with grow-m, thick-restart, and LOBPCG consumer proof. | Epic 10 retrospective; Sprint 117 retrospective; Epic 11 todo. | Source owner / proof owner |
| Revisit `s20_lift_ritz_vectors` movement after partial-publication ownership proof. | Epic 10 retrospective; Sprint 117 retrospective; Epic 11 todo. | Source owner / proof owner |
| Revisit shift-invert setup/conversion movement after LDLT lifecycle, operator selection, public error propagation, and cleanup ownership proof. | Epic 10 retrospective; Sprint 117 retrospective; Epic 11 todo. | Source owner / proof owner |
| Revisit `lanczos_iterate_op` movement with compile-unit proof for all consumers. | Epic 10 retrospective; Sprint 117 retrospective; Epic 11 todo. | Source owner |
| Preserve Sprint 114 non-package residual gates in adjacent package/platform work. | Epic 10 retrospective. | Package/platform / claim boundary |
| Decide whether QR, CG, GMRES, BiCGSTAB, and MINRES generated-RHS setup can share a direct/iterative oracle. | Sprint 117 retrospective; Epic 11 todo. | Oracle / proof owner |
| Decide whether SVD reconstruction, U/Vt orthogonality, Moore-Penrose, low-rank, sparse-vs-dense, and condition-number helpers can share a proof owner. | Sprint 117 retrospective; Epic 11 todo. | Oracle / proof owner |
| Split giant tests into focused proof owners. | Epic 11 review/todo. | Proof owner / maintainability |
| Build numerical oracle and corpus architecture. | Epic 11 review/todo. | Oracle |
| Improve benchmark, coverage, dead-code, source-list, and large-matrix report indexes. | Epic 11 review/todo; Epic 10 optional scanability work. | Reportability / performance |
| Strengthen local performance sentinels without claiming portable speed. | Epic 11 review/todo. | Performance / claim boundary |
| Promote Linux install proof to reviewed CI only with CI/runtime ownership and support wording updates. | Sprint 117 retrospective; Epic 10 retrospective; Day 4 artifact. | Package/platform |
| Promote macOS CMake install/export parity only with reviewed CI proof. | Sprint 117 retrospective; Epic 10 retrospective; Day 4 artifact. | Package/platform |
| Add Windows install-validation only with MSVC install, downstream consumer, reviewed-count, and non-claim proof. | Sprint 117 retrospective; Epic 10 retrospective; Day 4 artifact. | Package/platform |
| Port or split Windows thread/fuzz/property proof only with native Windows behavior and CTest count updates. | Sprint 117 retrospective; Epic 10 retrospective; Day 4 artifact. | Package/platform / proof owner |
| Add shared-library/dynamic ABI support only with build, package, loader, symbol, versioning, ABI-test, and platform proof. | Sprint 117 retrospective; Epic 10 retrospective; Epic 11 todo; Day 4 artifact. | Package/platform |
| Add package-manager support only with real recipes and install/consumer proof for each claimed manager/platform. | Sprint 117 retrospective; Epic 10 retrospective; Epic 11 todo; Day 4 artifact. | Package/platform |
| Split `docs/algorithm.md` into concise current reference plus historical measurement appendix. | Sprint 117 retrospective; Epic 10 retrospective; Epic 11 todo. | Adoption |
| Add generated benchmark artifact indexes in public or maintainer docs. | Sprint 117 retrospective; Epic 10 retrospective; Epic 11 todo. | Reportability / adoption |
| Preserve explicit non-claims for state-of-the-art replacement, ecosystem parity, every-family validation, portable speed, universal reorder/fill superiority, dynamic ABI, package managers, GPU, and distributed memory. | Epic 10 retrospective; Sprint 117 retrospective; Epic 11 review/todo. | Claim boundary |

## Duplicate Fence

| Deduplicated owner candidate | Duplicate raw items folded in | Scheduled disposition |
|---|---|---|
| Eigensolver source-boundary follow-through | private owner movement, `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert setup/conversion, `lanczos_iterate_op` | Sprint 119 Items 1-7. |
| Direct/iterative oracle and giant-test split | generated-RHS sharing, direct/iterative proof owners, QR/LDLT/LDLT CSC/iterative giant tests | Sprint 120 Items 1-7. |
| SVD/QR/rank-deficient oracle expansion | SVD helper owner, QR evidence, rank/low-rank/pseudoinverse helpers | Sprint 121 Items 1-7. |
| Corpus, coverage, and report architecture | numerical corpus, benchmark/coverage/dead-code/source-list report indexes, stale-report detection | Sprint 122 Items 1-7. |
| Performance/backend governance | hot path inventory, backend runtime contract, local sentinels, benchmark interpretation | Sprint 123 Items 1-7. |
| Package/ABI decision | shared library, dynamic ABI, static-first continuation, symbol/version checks, package-manager residuals | Sprint 124 Items 1-7. |
| Cross-platform install/staged lanes | Linux install CI, macOS install/export parity, Windows install validation, Windows staged tests, Windows Makefile gap | Sprint 125 Items 1-7. |
| Adoption/doc scanability | algorithm doc split, compressed-first cookbook, benchmark/report index docs | Sprint 126 Items 1-7. |
| Final claim and residual publication | non-claims, unsupported-claim cleanup, final validation, post-Epic-11 residuals | Sprint 127 Items 1-7. |

## Category Map

| Category | Deduplicated items | Primary sprint owners |
|---|---|---|
| Source owner | Eigensolver private movement, `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert, `lanczos_iterate_op`. | Sprint 119 |
| Proof owner | Giant tests, direct/iterative oracle ownership, SVD helper ownership, Windows staged thread/fuzz/property proof. | Sprints 120-121, 125 |
| Oracle | Direct/iterative generated-RHS oracle, SVD/QR dense or external reference, corpus taxonomy. | Sprints 120-122 |
| Performance | Hot path inventory, sentinels, backend runtime governance, benchmark interpretation. | Sprint 123 |
| Package/platform | Shared-library/ABI, package-manager, Linux install CI, macOS install/export parity, Windows install validation. | Sprints 124-125 |
| Adoption | Algorithm doc split, compressed-first cookbook, public docs simplification. | Sprint 126 |
| Reportability | Benchmark, coverage, dead-code, source-list, large-matrix artifact indexes. | Sprints 122, 126 |
| Claim boundary | State-of-the-art, ecosystem parity, portable performance, universal reorder/fill, dynamic ABI, package-manager, GPU, distributed-memory non-claims. | Sprints 118, 123-127 |

## Already-Scheduled Work List

| Residual | Already scheduled in Epic 11? | Notes |
|---|---|---|
| Eigensolver private owner movement and related source-boundary decisions | Yes, Sprint 119 | Includes movement feasibility, old/new files, source-list/CMake updates, rollback, validation, and non-claims. |
| `s20_select_indices` and `s20_lift_ritz_vectors` | Yes, Sprint 119 | Selection/lifting batch explicitly moves or defers based on proof. |
| Shift-invert setup/conversion | Yes, Sprint 119 | Boundary decision explicitly splits or defers after dependency proof. |
| `lanczos_iterate_op` | Yes, Sprint 119 | Covered by movement feasibility and first movement/source-boundary proof gates. |
| Shared direct/iterative generated-RHS oracle | Yes, Sprint 120 | Covered by oracle ownership audit, shared fixture design, and pilot. |
| Giant direct/iterative test owners | Yes, Sprint 120 | Covers selected QR, LDLT, LDLT CSC, iterative, BiCGSTAB, and MINRES split candidates. |
| Shared SVD proof-helper owner | Yes, Sprint 121 | Covered by SVD helper extraction and rank-deficient evidence expansion. |
| Numerical corpus architecture | Yes, Sprint 122 | Corpus inventory, taxonomy, and report index design. |
| Coverage/report architecture | Yes, Sprint 122 | Coverage architecture and generated index batch. |
| Local performance sentinels | Yes, Sprint 123 | Sentinel design, implementation batch, and non-portable interpretation. |
| Shared-library/dynamic ABI support or explicit deferral | Yes, Sprint 124 | Product decision plus implementation or deferral checks. |
| Package-manager support | Yes, Sprint 124 closeout/residual | Sprint 124 publishes residual package-manager work if not implemented. |
| Linux install CI | Yes, Sprint 125 | Explicit promote-or-defer decision. |
| macOS CMake install/export parity | Yes, Sprint 125 | Explicit add/strengthen/defer item. |
| Windows install validation | Yes, Sprint 125 | Explicit design and implement-or-defer item. |
| Windows thread/fuzz/property staged tests | Yes, Sprint 125 | Explicit staged test follow-through. |
| Algorithm doc split | Yes, Sprint 126 | Algorithm doc split design and batch. |
| Generated benchmark/report indexes in docs | Yes, Sprints 122 and 126 | Sprint 122 report index design/batch; Sprint 126 benchmark/report index docs. |
| Final claim recalibration and non-claim publication | Yes, Sprint 127 | Final competitive recalibration, unsupported-claim cleanup, and residual publication. |

## Unresolved Residual Candidate List

No immediate unscheduled residual from the Day 5 intake needs to be added to
the current Epic 11 project plan. The intake items are already represented in
Sprints 119-127 or are intentionally preserved as non-claims until an assigned
sprint earns evidence.

The following items should remain watch-listed because they may become
post-Epic-11 residuals if their owner sprints explicitly defer them:

- package-manager support after Sprint 124;
- shared-library/dynamic ABI if Sprint 124 chooses static-first continuation;
- Windows Makefile parity if Sprint 125 keeps it out of scope;
- GPU, distributed-memory, broad complex/mixed-precision, and ecosystem parity
  claims, which are explicit non-claims unless future work funds them.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Raw residual intake list is recorded. | Complete. |
| Duplicate residuals are folded into owner candidates. | Complete. |
| Every residual candidate has a category and evidence source. | Complete. |
| Already-scheduled Epic 11 work is identified. | Complete. |
| No completed Epic 10 work is reintroduced as unresolved debt. | Complete. |
| Unscheduled residual candidates are explicitly listed or marked absent. | Complete. |
