# Sprint 117 Day 13 Epic 10 Retrospective Draft

## Purpose

Day 13 drafts the Epic 10 retrospective from the project plan, review inputs,
Sprint 100-117 retrospectives, and Sprint 117 final validation, comparison,
claim-cleanup, and residual artifacts. This is draft source for Day 14
finalization, not the final `EPIC_10_RETROSPECTIVE.md`.

## Draft Epic Header

# Epic 10 Retrospective - Sprints 100-117 (linalg_sparse_orthogonal)

**Epic budget:** 18 sprint project-plan estimates = **2,906 hours nominal**
**Branch range:** `sprint-100` -> `sprint-117`
**Started:** Sprint 100 (Epic 10 baseline, state-of-the-art target, and
evidence contract)
**Closed:** Sprint 117 Day 14 (finalization pending)
**Goal:** Turn the post-Epic-9 library into a more product-grade sparse linear
algebra system through compressed-first workflows, stronger bounded solver
evidence, maintainable proof owners, clearer API/documentation surfaces,
stronger package/platform proof, and calibrated public claims.

> **Draft status:** Epic 10 draft retrospective content exists. Day 14 still
> needs to finalize the file, reconcile it with Sprint 117 validation and
> residual artifacts, run final touched-surface checks, and write the final
> closeout handoff.

## Draft Summary Table

| sprint | title | plan assessment | key outcome | project-plan h | actual h |
|---|---|---|---|---:|---:|
| 100 | Epic 10 Baseline, State-of-the-Art Target & Evidence Contract | Met | Bounded target, evidence templates, claim model, and baseline validation | 166 | n/t |
| 101 | Compressed-First Product Model & Storage Front Door | Met | Compressed-first public workflow and mutable-shell compatibility framing | 168 | n/t |
| 102 | Direct Solver Robustness & External Oracle Expansion | Met | Bounded direct-solver external/dense-reference evidence and failure-mode proof | 168 | n/t |
| 103 | Iterative, Eigensolver & SVD External Comparison | Met | Fixture-local iterative/eigensolver/SVD comparison and residual evidence | 168 | n/t |
| 104 | Performance Backend & Parallel Runtime Modernization | Met | Backend/runtime observability and local performance sentinel classification | 168 | n/t |
| 105 | Reordering, Graph & Large-Matrix Scalability | Met | Reorder/fill/graph guardrails and local large-matrix evidence boundaries | 168 | n/t |
| 106 | Large-Source & Giant-Test Maintainability Phase 7 | Met | Large-owner and giant-test maintainability follow-through with residual queue | 168 | n/t |
| 107 | Residual Maintainability Debt & Proof-Owner Cleanup | Met | Proof-owner cleanup and residual maintainability debt publication | 168 | n/t |
| 108 | Residual Proof-Owner & Source Boundary Follow-Through | Met | Proof-owner/source-boundary follow-through without unsupported movement claims | 168 | n/t |
| 109 | Residual Source Boundary & Proof-Owner Debt Closeout | Met | Source-boundary and proof-owner closeout with dependency gates | 168 | n/t |
| 110 | Residual Matrix I/O, Behavior Owners & Proof-Owner Follow-Through | Met | Matrix I/O/public-surface and behavior-owner boundaries clarified | 168 | n/t |
| 111 | API Usability, Documentation & Example Coherence | Met | API docs, examples, solver guidance, and adoption routes tightened | 168 | n/t |
| 112 | Packaging, ABI & Cross-Platform Validation Expansion | Met | Static-first package truth, install/export proof, and platform tiers sharpened | 168 | n/t |
| 113 | Residual Behavior & Proof-Owner Closeout | Met | Residual behavior proof and direct/eigensolver/SVD proof-owner cleanup | 168 | n/t |
| 114 | Residual Eigensolver, Direct/Iterative & SVD Proof-Owner Follow-Through | Met | Eigensolver/direct/iterative/SVD proof-owner residuals bounded and validated | 168 | n/t |
| 115 | Residual Package Platform Parity & ABI Productization Decision | Met | Package/platform/ABI decisions documented without unsupported support claims | 168 | n/t |
| 116 | Adoption Surface Residual QA & Claim Guardrails | Met | Adoption surfaces cleaned and claim guardrails reinforced | 56 | n/t |
| 117 | Final Integration, Competitive Calibration & Epic 10 Closeout | Met through Day 13 | Final validation, comparison package, residual queue, and sprint retrospective | 164 | n/t |
| **Total** | | | | **2,906** | **n/t** |

## Draft Cumulative Metrics

| metric | draft value |
|---|---:|
| planned sprints completed | `18 / 18` through Sprint 117 Day 13 |
| project-plan nominal hours | `2,906` |
| sprint plans present | `18` |
| sprint working-notes files present | `18` |
| sprint retrospectives present | `18` |
| sprint artifact files under `EPIC_10/SPRINT_*/artifacts/` before Day 13 draft | `270` |
| sprint retrospective lines before Day 13 draft | `4163` |
| sprint working-notes lines before Day 13 draft | `13161` |
| strongest final local reviewed baseline command | `make quality-review-full` |
| final reviewed CMake registered tests | `54` |
| final Makefile/CMake test-count parity | `54` vs `54` |
| final reviewed CMake CTest result | `54 / 54` passed |
| final unsupported broad public/support edits required by Sprint 117 Day 8 | `0` |
| final post-Epic residual queue items | `6` post-Epic residuals, `8` future-epic candidates, `2` optional scanability items |

## Draft Earned Claim Table

| Claim | Evidence | Earned wording |
|---|---|---|
| Epic 10 improved product-grade maturity. | Sprint 100 target/evidence contract; Sprint 101-117 retrospectives; Sprint 117 validation and cleanup artifacts. | Epic 10 improved productization, validation discipline, support-tier clarity, documentation, and claim boundaries. |
| Compressed-first workflows are a primary supported product path. | Sprint 101 compressed-first work and later docs/examples coherence. | Callers with CSR/CSC data have clearer compressed-first construction/export and solver workflow routes while mutable matrix-shell compatibility remains supported. |
| Selected direct-solver evidence is stronger. | Sprint 102 direct-solver oracle artifacts and Sprint 117 comparison inventory. | Selected direct-solver families have bounded named dense-reference or external-process evidence. |
| Iterative/eigensolver/SVD evidence is deeper but fixture-local. | Sprints 103, 113, and 114; Sprint 117 comparison inventory. | Fixture-local residual, convergence, reconstruction, rank, and orthogonality evidence improved. |
| Backend/runtime and performance interpretation are clearer. | Sprint 104 backend/runtime artifacts; Sprint 116 performance wording cleanup; Sprint 117 Day 7-8 artifacts. | Runtime/backend behavior and local sentinel/reporting surfaces are better documented as local evidence. |
| Reorder/fill/large-matrix evidence is governed. | Sprint 105 guardrails and Sprint 117 comparison inventory. | Reorder/fill reports and guardrails provide named-fixture structural and local measurement context. |
| Maintainability/proof-owner debt was reduced in selected owners. | Sprints 106-110, 113-114, and Sprint 117 residual publication. | Selected large-source, giant-test, proof-owner, Matrix I/O, and behavior-owner surfaces became more explicit and reviewable. |
| API, documentation, examples, and adoption routes are clearer. | Sprints 111 and 116; Sprint 117 claim cleanup. | Public docs and examples better route users by workflow while preserving support boundaries. |
| Static-first package and tiered platform support are explicit. | Sprints 112 and 115; Sprint 117 final cleanup. | Static-first package support and tiered platform scope are the maintained support story. |
| Final closeout claims are bounded by evidence. | Sprint 117 Days 2-10 and retrospective. | Epic 10 closes with earned claims, explicit non-claims, and a residual queue. |

## Draft Unearned / Non-Claim Table

| Non-claim | Reason it remains unearned |
|---|---|
| Unqualified state-of-the-art replacement for established sparse linear algebra ecosystems. | Evidence is intentionally bounded to productization and selected proof surfaces. |
| SuiteSparse, PETSc, Trilinos, ARPACK, LAPACK, SciPy, NumPy, or vendor backend parity. | No broad external ecosystem parity suite was implemented or validated. |
| Every-family external solver validation. | Direct, iterative, eigensolver, and SVD evidence remains selected and fixture-local. |
| Portable performance superiority or universal reorder/fill superiority. | Benchmarks and guardrails remain local measurement context. |
| Cross-platform timing or max-RSS thresholds. | Platform-local measurement differences remain unreviewed as portable gates. |
| Reviewed Linux install CI lane. | Local/static install proof exists, but reviewed Linux install CI was not promoted. |
| Full reviewed macOS CMake install/export parity. | macOS support remains scoped and supplemental where appropriate. |
| Windows install-validation, thread/fuzz/property, or Makefile parity. | Windows remains reviewed CMake-first consumer subset. |
| Shared-library package support or dynamic ABI guarantee. | Product contract, build, loader, symbol, versioning, and ABI proof are deferred. |
| Package-manager support. | No manager recipes and install/consumer proofs were added. |
| Public Matrix I/O module or public Matrix Market builder API. | Public Matrix Market support remains load/save functions. |
| Complete closure of proof-owner/source-boundary debt. | Several source-boundary and proof-owner items remain explicit post-Epic residuals. |

## Draft Validation Evidence

Final Sprint 117 validation is the Epic 10 closeout anchor:

- `make quality-review-full` passed on Day 5.
- Makefile reviewed path passed:
  - `format-check`;
  - `lint`;
  - `test`;
  - `deadcode-check`.
- CMake reviewed parity path passed:
  - configure;
  - clean build;
  - `ctest -N`;
  - Makefile/CMake test-count parity;
  - full CTest.
- CMake registered tests: `54`.
- Makefile/CMake test-count parity: `54` vs `54`.
- CTest result: `54 / 54` passed, `0` failed.
- CTest real time: `242.37 sec`.
- Sprint 117 remained documentation-only through Day 13:
  - changed `.c` files: `0`;
  - changed `.h` files: `0`;
  - changed Make/CMake/workflow/package/script files: `0`;
  - changed benchmark/source/test/include files: `0`.

Day 14 should rerun the required documentation hygiene checks after finalizing
the Epic 10 retrospective.

## Draft Lessons Learned

1. **Evidence templates helped prevent claim drift.**
   Sprint 100's comparison, benchmark, package, platform, and non-claim
   templates kept later sprints from treating local evidence as broad product
   claims.

2. **Bounded product claims are more useful than broad ambition.**
   Epic 10 made the library more product-grade without pretending to replace
   mature sparse ecosystems.

3. **Maintainer-facing proof owners need explicit residual queues.**
   The project improved many proof owners, but source movement remains risky
   unless exact source-list, CMake, CTest, consumer, and rollback proof is
   captured.

4. **Package/platform truth needs tiered language.**
   Static-first support, Linux/macOS/Windows differences, and install/export
   proof are clearer when they are not collapsed into a single parity claim.

5. **Performance artifacts need classification before publication.**
   Local benchmark and sentinel evidence are valuable, but only when readers
   can see what is local measurement context and what is reviewed proof.

6. **Closeout sprints should start from a running residual register.**
   Sprint 117 succeeded, but synthesizing deferred debt across multiple
   sprints is easier when residual disposition is updated continuously.

## Draft Post-Epic Carry-Forward Queue

### Post-Epic Residuals

- Move one eigensolver private owner with exact source-list, CMake, consumer,
  CTest, and rollback proof.
- Revisit `s20_select_indices` movement with grow-m, thick-restart, and LOBPCG
  consumer proof.
- Revisit `s20_lift_ritz_vectors` movement after grow-m and thick-restart
  partial-publication states share a proven owner.
- Revisit shift-invert setup/conversion movement after LDLT lifecycle,
  operator selection, public error propagation, and cleanup ownership proof.
- Revisit `lanczos_iterate_op` movement with compile-unit proof for all
  consumers.
- Carry Sprint 114 non-package residual gates into any future adjacent
  package/platform work.

### Future-Epic Candidates

- Shared direct/iterative generated-RHS oracle for QR, CG, GMRES, BiCGSTAB,
  and MINRES.
- Shared SVD proof-helper owner.
- Reviewed Linux install CI lane.
- Reviewed macOS CMake install/export parity.
- Windows install-validation lane.
- Windows thread/fuzz/property proof split or port.
- Shared-library and dynamic ABI support.
- Package-manager support.

### Optional Scanability Work

- Split `docs/algorithm.md` into a concise public algorithm reference plus a
  historical measurement appendix.
- Add generated benchmark artifact indexes in public or maintainer docs.

## Day 14 Finalization Checklist

- Create `docs/planning/EPIC_10/EPIC_10_RETROSPECTIVE.md` from this draft.
- Reconcile final Epic 10 claims with:
  - `SPRINT_117/RETROSPECTIVE.md`;
  - `SPRINT_117/artifacts/day6-final-validation-package.md`;
  - `SPRINT_117/artifacts/day8-final-comparison-cleanup.md`;
  - `SPRINT_117/artifacts/day10-residual-queue-and-nonclaims.md`.
- Recompute final Epic 10 artifact, retrospective, and working-note metrics
  after the final retrospective file exists.
- Run final required checks for touched surfaces:
  - `git diff --check`;
  - focused trailing-whitespace scan over `docs/planning/EPIC_10/SPRINT_117`
    and `docs/planning/EPIC_10/EPIC_10_RETROSPECTIVE.md`.
- Confirm no `.c` or `.h` files changed, or run
  `make format && make lint && make test` if that changes before closeout.
- Write the final Day 14 closeout handoff artifact.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Epic retrospective content exists before final day validation. | Complete. |
| Earned and unearned claims are evidence-bounded. | Complete. |
| Post-epic residual queue matches Day 10 publication. | Complete. |
