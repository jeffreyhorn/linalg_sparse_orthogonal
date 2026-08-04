# Project Plan: Corpus-Grade Evidence, Report Governance & Productization Follow-Through -- Sprints 137-146 (Epic 12)

Epic 12 starts from the completed Epic 11 closeout. The library is broad,
well-tested, and honest about support boundaries, but it still should not claim
unqualified state-of-the-art status. Epic 12 focuses on closing selected
high-value gaps completely: maintained numerical corpus/oracle evidence,
priority QR and partial-SVD residuals, normalized report/freshness
architecture, runtime/backend governance, package/ABI product decisions,
platform promotion, and adoption simplification.

This plan is based on:

- `reviews/review-codex-2026-08-03.md`
- `reviews/todo-codex-2026-08-03.md`
- `docs/planning/EPIC_11/EPIC_11_RETROSPECTIVE.md`
- `docs/planning/EPIC_11/SPRINT_136/artifacts/day12-residual-queue-publication.md`
- prior Epic retrospectives and carry-forward work

Each sprint remains 14 days and stays within the requested cap of 168 hours.

---

## Sprint 137: Epic 12 Baseline, Gap Selection & Evidence Contract

**Duration:** 14 days (~166 hours)
**Goal:** Freeze the post-Epic-11 baseline, select the gaps Epic 12 will close
completely, and create the evidence contracts that govern implementation.

### Prerequisites from Previous Sprints

- Epic 11 closeout merged
- Epic 11 retrospective and Sprint 136 residual queue available
- PR #151 review fixes merged into `master`

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Baseline Metrics Capture | Capture current source/test sizes, CMake counts, Windows subset count, package proofs, report artifacts, benchmark lanes, and support tiers. | 20 |
| 2 | Residual Queue Reconciliation | Convert Epic 11 residuals into Epic 12 owners, dependencies, duplicate fences, and non-goals. | 24 |
| 3 | Gap Selection Decision | Select the QR, partial-SVD, corpus/report, runtime/backend, package/ABI, and platform gaps that can be fully closed inside Epic 12. | 26 |
| 4 | Evidence Contract Templates | Define templates for corpus fixtures, oracle rows, report indexes, stale-report checks, package/ABI decisions, platform promotion, and claim gates. | 28 |
| 5 | Quality Surface Map | Map which checks are required for docs, scripts, Make/CMake, CI, package, report, benchmark, and `.c`/`.h` changes. | 18 |
| 6 | Public Claim Freeze | Reconfirm current README, INSTALL, benchmark, solver-selection, and maintainer docs contain no unsupported state-of-the-art, ABI, platform, or performance claims. | 20 |
| 7 | Sprint Closeout | Publish Sprint 137 artifacts, working notes, residuals, and handoff requirements for corpus implementation. | 30 |

### Deliverables

- post-Epic-11 baseline package
- Epic 12 gap-selection decision
- residual owner and non-goal map
- evidence contract templates
- quality surface map
- Sprint 138 handoff requirements

**Total estimate:** ~166 hours

---

## Sprint 138: Maintained Numerical Corpus Architecture

**Duration:** 14 days (~168 hours)
**Goal:** Build the maintained numerical corpus architecture and first durable
oracle/report lane before adding broad fixture volume.

### Prerequisites from Previous Sprints

- Sprint 137 evidence contracts complete
- selected corpus/oracle scope approved

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Fixture Taxonomy Design | Define maintained matrix classes for symmetry, definiteness, rank, rectangularity, conditioning, scaling, sparsity pattern, graph shape, and expected failures. | 24 |
| 2 | Corpus Storage Layout | Create deterministic fixture, generated-matrix, optional-data, manifest, skip, and expected-result layout under maintained paths. | 24 |
| 3 | Oracle Row Schema | Define durable oracle row fields for family, fixture, command, expected result, observed result, tolerance, support tier, skip reason, and source commit. | 24 |
| 4 | First Corpus Lane Implementation | Add the first sustained corpus lane with deterministic fixtures and one maintained oracle/report command. | 34 |
| 5 | Skip and Optional Data Semantics | Implement explicit skip/defer semantics for unavailable optional external data without converting skips into false pass evidence. | 20 |
| 6 | Focused Validation | Run the new corpus command, affected tests/scripts, and required quality checks for touched surfaces. | 20 |
| 7 | Documentation and Handoff | Document corpus ownership, row interpretation, stale-report assumptions, and Sprint 139 QR handoff requirements. | 22 |

### Deliverables

- maintained corpus taxonomy
- corpus storage and manifest layout
- first sustained oracle/report lane
- skip/defer semantics
- corpus maintainer documentation
- Sprint 139 QR fixture handoff

**Total estimate:** ~168 hours

---

## Sprint 139: QR Priority Residual Closure

**Duration:** 14 days (~168 hours)
**Goal:** Completely close the selected QR residual with corpus-backed
fixtures, oracle evidence, focused proof ownership, and updated claim wording.

### Prerequisites from Previous Sprints

- Sprint 138 corpus lane available
- selected QR residual and proof criteria documented

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | QR Residual Reaudit | Re-rank QR residuals for rank-deficient, rectangular, least-squares, minimum-norm, nullspace, COLAMD, and SuiteSparse/corpus behavior. | 18 |
| 2 | QR Fixture Batch | Add deterministic corpus fixtures for the selected QR residual family, including success, failure, and tolerance boundaries. | 28 |
| 3 | QR Oracle Comparison | Add dense or external-reference comparison for the selected QR lane with explicit tolerance, skip, and failure semantics. | 30 |
| 4 | QR Proof-Owner Split | Extract the new QR proof lane from `tests/test_qr.c` or create a focused helper/test owner without weakening existing coverage. | 28 |
| 5 | Solver Docs Update | Update solver-selection, algorithm, cookbook, and maintainer guidance with earned QR wording and preserved non-claims. | 18 |
| 6 | Validation | Run focused QR/corpus tests, source-list/CMake parity if code moved, and full quality gates if `.c`/`.h` changed. | 26 |
| 7 | Closeout and Residuals | Publish closed QR claim, remaining QR non-claims, and handoff requirements for partial-SVD work. | 20 |

### Deliverables

- closed priority QR residual
- QR corpus fixtures
- QR oracle comparison row(s)
- focused QR proof owner
- updated QR docs and claim boundaries
- Sprint 140 partial-SVD handoff

**Total estimate:** ~168 hours

---

## Sprint 140: Partial-SVD Edge-Case & Convergence Residual Closure

**Duration:** 14 days (~168 hours)
**Goal:** Completely close the selected partial-SVD residual with edge-case
fixtures, convergence-budget proof, oracle semantics, and docs.

### Prerequisites from Previous Sprints

- Sprint 138 corpus lane available
- Sprint 139 QR validation lessons available
- selected partial-SVD residual and comparison semantics documented

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Partial-SVD Residual Reaudit | Re-rank partial-SVD residuals for rank, repeated singular values, near-zero values, rectangularity, convergence budget, ordering, vectors, and subspaces. | 20 |
| 2 | Edge-Case Fixture Batch | Add deterministic corpus fixtures for the selected partial-SVD residual family. | 28 |
| 3 | Comparison Semantics | Implement singular-value, vector, subspace, ordering, tolerance, partial-convergence, and skip interpretation for the selected lane. | 32 |
| 4 | Convergence-Budget Tests | Add focused tests that prove budget behavior and failure diagnostics without masking non-convergence. | 26 |
| 5 | Proof-Owner Cleanup | Extract or refactor only the partial-SVD helper/test ownership needed to keep the new lane maintainable. | 22 |
| 6 | Validation | Run focused SVD/partial-SVD/corpus tests and full quality gates if `.c`/`.h` changed. | 24 |
| 7 | Docs and Closeout | Update SVD docs, solver-selection wording, non-claims, and Sprint 141 report-index handoff. | 16 |

### Deliverables

- closed priority partial-SVD residual
- partial-SVD corpus fixtures
- comparison semantics and oracle rows
- convergence-budget tests
- updated partial-SVD docs and non-claims
- Sprint 141 report normalization handoff

**Total estimate:** ~168 hours

---

## Sprint 141: Report Index Normalization & Freshness Gates

**Duration:** 14 days (~168 hours)
**Goal:** Normalize maintained report metadata across corpus, benchmark,
sentinel, guardrail, coverage, dead-code, and package lanes where row meaning
can be preserved.

### Prerequisites from Previous Sprints

- Sprint 138 corpus report schema available
- Sprint 139 QR and Sprint 140 partial-SVD report rows available
- existing benchmark/sentinel/guardrail/dead-code report artifacts available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Report Family Inventory | Inventory canonical benchmark, performance sentinel, large-matrix guardrail, coverage, dead-code, package, corpus, and oracle outputs. | 22 |
| 2 | Shared Metadata Contract | Define common fields for report family, generator, command, commit, platform, compiler, configuration, support tier, row meaning, freshness, and skip/defer reason. | 28 |
| 3 | Normalized Index Generator | Implement a maintained generator for report families whose row meanings can be normalized honestly. | 34 |
| 4 | Stale-Report Gate | Add a freshness check that identifies stale generated reports without pretending local measurements are release proofs. | 24 |
| 5 | Documentation Alignment | Update benchmark, maintainer, and corpus docs with normalized report semantics and non-claims. | 20 |
| 6 | Validation | Run report generators/checks, affected scripts, docs checks, and quality gates for touched script/code surfaces. | 24 |
| 7 | Closeout | Publish normalized report index evidence and handoff runtime/backend rows to Sprint 142. | 16 |

### Deliverables

- report family inventory
- shared metadata contract
- normalized report index generator
- stale-report check
- updated report interpretation docs
- Sprint 142 runtime/backend handoff

**Total estimate:** ~168 hours

---

## Sprint 142: Runtime Backend Governance & Sentinel Expansion

**Duration:** 14 days (~168 hours)
**Goal:** Convert runtime/backend behavior into a clearer maintained contract
and expand sentinels where they provide useful local regression evidence.

### Prerequisites from Previous Sprints

- Sprint 141 normalized report fields available
- current backend/runtime controls and sentinel rows inventoried

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Runtime Control Audit | Audit OpenMP, backend dispatch, dense helper selection, eigensolver backend selection, direct-solver dispatch, environment variables, and typed options. | 22 |
| 2 | Precedence Contract | Define maintained precedence for typed options, compile-time flags, environment compatibility overrides, backend fallback, and deterministic behavior. | 28 |
| 3 | Typed-Control Batch | Promote the highest-value remaining environment-only or implicit runtime controls into typed options, or explicitly classify them as maintainer-only. | 34 |
| 4 | Sentinel Expansion | Add normalized local sentinel rows for selected hot paths or backend decisions without creating portable timing claims. | 26 |
| 5 | Docs and Examples | Update README, benchmark docs, maintainer guide, and examples affected by runtime/backend contract changes. | 18 |
| 6 | Validation | Run focused runtime/backend tests, sentinels, report freshness checks, and full quality gates if `.c`/`.h` changed. | 24 |
| 7 | Closeout | Publish earned runtime/backend claims, non-claims, and package/ABI prerequisites for Sprint 143. | 16 |

### Deliverables

- runtime/backend control audit
- maintained precedence contract
- typed-control or explicit deferral batch
- normalized sentinel rows
- updated runtime/backend docs
- Sprint 143 package/ABI handoff

**Total estimate:** ~168 hours

---

## Sprint 143: Shared-Library ABI Decision & Static-First Contract Follow-Through

**Duration:** 14 days (~168 hours)
**Goal:** Make and implement the Epic 12 package/ABI product decision:
shared-library ABI support with proof, or stricter static-first-only contract
with no ambiguity.

### Prerequisites from Previous Sprints

- Sprint 137 package/ABI decision criteria available
- Sprint 142 runtime/backend support boundaries available
- current install, CMake, pkg-config, and static-deferral proofs green

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | ABI Feasibility Audit | Audit public headers, symbol surface, versioning, visibility, CMake exports, pkg-config metadata, loader behavior, and platform risks. | 24 |
| 2 | Product Decision | Decide whether Epic 12 implements shared-library ABI support or preserves static-first-only support for another epic. | 18 |
| 3 | Implementation Batch | Implement the selected path: shared-library build/install/export proof, or stricter static-first deferral guards and docs cleanup. | 42 |
| 4 | Downstream Consumer Proof | Add or strengthen downstream consumer tests for Make, pkg-config, CMake, version constraints, loader behavior where applicable, and unsupported-artifact checks. | 28 |
| 5 | CI/Packaging Alignment | Update Linux/macOS/Windows CI jobs and support-tier comments for the selected package contract. | 20 |
| 6 | Documentation Alignment | Update README, INSTALL, CMake package docs, pkg-config comments, maintainer guide, and non-claim wording. | 18 |
| 7 | Validation and Closeout | Run install/package tests, CMake install/export tests, static/shared guards as applicable, and full quality gates for code/build changes. | 18 |

### Deliverables

- explicit package/ABI product decision
- implemented selected package path
- downstream consumer proof
- CI/support-tier alignment
- updated package docs
- Sprint 144 platform promotion handoff

**Total estimate:** ~168 hours

---

## Sprint 144: Platform Promotion Lane Closure

**Duration:** 14 days (~168 hours)
**Goal:** Fully promote one high-value platform support lane, or explicitly
reject promotion with source-level blockers and proof requirements.

### Prerequisites from Previous Sprints

- Sprint 143 package/ABI decision implemented
- current Linux, macOS, and Windows CI tier comments available
- selected platform lane scoped for complete closure

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Platform Lane Selection | Choose one lane for complete closure: macOS reviewed install/export parity, Windows reviewed install/downstream parity, Windows staged test portability, or Linux source-of-truth strengthening. | 20 |
| 2 | Source/Script Portability Fixes | Fix source, test, script, path, shell, pthread/POSIX, CMake, or PowerShell blockers for the selected lane. | 36 |
| 3 | CI Promotion Implementation | Update CI workflow jobs, expected counts, failure messages, support-tier comments, and artifact proof for the selected lane. | 30 |
| 4 | Package/Report Integration | Connect the selected platform lane to package/report/freshness artifacts where applicable. | 20 |
| 5 | Documentation Alignment | Update README, INSTALL, maintainer guide, and platform support docs to reflect the earned support tier. | 18 |
| 6 | Validation | Run local feasible checks, workflow syntax checks, package/report checks, and full quality gates if `.c`/`.h` changed. | 24 |
| 7 | Closeout | Publish platform promotion evidence, residual platform non-claims, and adoption handoff for Sprint 145. | 20 |

### Deliverables

- selected platform lane closed or explicitly rejected
- source/script/CI updates for selected lane
- support-tier documentation
- package/report integration evidence
- Sprint 145 adoption handoff

**Total estimate:** ~168 hours

---

## Sprint 145: Adoption Surface Simplification & High-Level Workflow Front Door

**Duration:** 14 days (~168 hours)
**Goal:** Simplify first-use adoption after the corpus, QR, partial-SVD,
report, runtime, package, and platform decisions have landed.

### Prerequisites from Previous Sprints

- Sprint 139 QR claims settled
- Sprint 140 partial-SVD claims settled
- Sprint 141 report semantics settled
- Sprint 142 runtime/backend contract settled
- Sprint 143 package/ABI decision settled
- Sprint 144 platform tier settled

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Adoption Friction Audit | Audit README, cookbook, tutorial, solver-selection, examples, INSTALL, benchmark docs, and public headers for first-use friction and stale support-tier wording. | 22 |
| 2 | High-Level Workflow Design | Design a concise high-level workflow front door for build/install, choose solver, run solve, inspect diagnostics, and move to advanced controls. | 24 |
| 3 | Example/Cookbook Batch | Add or revise maintained examples and cookbook entries for the earned QR, partial-SVD, corpus/report, runtime, and package/platform behavior. | 30 |
| 4 | README/INSTALL Simplification | Reduce first-use density while preserving exact support boundaries and links to deeper maintainer references. | 24 |
| 5 | Public Header Pass | Remove or move maintainer-only historical detail from the highest-impact public headers without weakening contracts. | 22 |
| 6 | Validation | Run docs link checks, example builds, install/downstream checks as needed, and full quality gates if `.c`/`.h` changed. | 24 |
| 7 | Closeout | Publish adoption claim map, residual doc debt, and Epic 12 closeout handoff. | 22 |

### Deliverables

- simplified first-use adoption path
- updated examples/cookbook entries
- README/INSTALL support-tier alignment
- public-header cleanup for selected surfaces
- Sprint 146 closeout handoff

**Total estimate:** ~168 hours

---

## Sprint 146: Epic 12 Final Validation, Claim Recalibration & Closeout

**Duration:** 14 days (~166 hours)
**Goal:** Validate the final Epic 12 product state, recalibrate claims, publish
closed gaps and residuals, and decide whether any state-of-the-art claim has
actually been earned.

### Prerequisites from Previous Sprints

- Sprint 137-145 deliverables complete or explicitly deferred
- final package/platform/report/adoption support tiers documented
- quality gates identified for all touched surfaces

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Final Evidence Inventory | Inventory all Epic 12 corpus, QR, partial-SVD, report, runtime, package, platform, adoption, and validation artifacts. | 20 |
| 2 | Full Quality Baseline | Run the strongest feasible reviewed local baseline plus package/report/corpus checks required by touched surfaces. | 28 |
| 3 | Cross-Platform/CI Reconciliation | Reconcile Linux, macOS, Windows, supplemental, staged, reviewed, local-only, and deferred evidence after final CI runs. | 20 |
| 4 | Claim and Non-Claim Audit | Rescan public docs and support docs for unsupported state-of-the-art, external parity, package, platform, performance, ABI, or report wording. | 24 |
| 5 | Residual Queue Publication | Publish remaining future work with owners, blockers, prerequisites, and promotion gates. | 24 |
| 6 | Epic 12 Retrospective | Write the Epic 12 retrospective with earned claims, non-claims, closed gaps, validation evidence, and state-of-the-art assessment. | 26 |
| 7 | Final Project Plan Reconciliation | Reconcile the Epic 12 project plan against completed sprint artifacts and prepare the next-epic handoff. | 24 |

### Deliverables

- final Epic 12 evidence inventory
- final validation package
- public claim/non-claim audit
- residual queue with promotion gates
- Epic 12 retrospective
- next-epic handoff

**Total estimate:** ~166 hours

