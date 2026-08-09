# Sprint 145 Day 2 Adoption Friction Audit

## Purpose

Audit first-use adoption friction, stale wording, duplicated entry points,
support-tier ambiguity, and public-header cleanup candidates before designing
the high-level workflow front door.

## Audit Scope

| Surface | Size or shape | Audit result |
| --- | ---: | --- |
| `README.md` | 910 lines | Strong evidence and feature inventory, but first-use path competes with large capability, API, performance, limitations, testing, project-structure, and installation sections. |
| `INSTALL.md` | 361 lines | Accurate static-first install and platform contract, but the shortest path is followed quickly by support split, maintained install contract, platform table, CMake, and verification detail. |
| `docs/cookbook.md` | 286 lines | Good task-oriented material, but no single canonical first-use recipe that ties build/install, matrix input, solver choice, diagnostics, and next step together. |
| `docs/tutorial.md` | 539 lines | Useful learning path, but broad solver coverage and many code blocks make it heavy as a first-use front door. |
| `docs/solver_selection.md` | 227 lines | Solid decision guide, but users need a shorter choose-solver summary before entering detailed evidence boundaries. |
| `examples/README.md` | 334 lines | Runnable examples are well documented, but the first-use sequence is spread across many programs rather than one compact workflow ladder. |
| `benchmarks/README.md` | 662 lines | Correctly advanced and evidence-focused; should remain behind deeper links for first-use readers. |
| `include/*.h` | 18 public headers, 5750 total lines | Several headers are adoption-relevant but dense; cleanup must be selected carefully and validated as C/header work. |
| `docs/maintainer_guide.md` | 1269 lines | Correct home for deeper evidence and maintenance policy; should absorb details removed from first-use surfaces. |

## Friction Inventory

| User path | Primary friction | Evidence | Initial fix direction |
| --- | --- | --- | --- |
| Build/install | README and INSTALL both carry install guidance, but README's installation content begins after extensive API/performance/testing sections. | `README.md` has `## Building` at line 222 and `## Installation` at line 863; `INSTALL.md` has `## Start Here` at line 9 and detailed support sections immediately after. | Day 3 should design a single first-use build/install path and Day 6-7 should route detailed package/platform proof to INSTALL and maintainer guide. |
| Choose solver | README, cookbook, tutorial, solver-selection guide, and examples all provide solver entry points with different levels of detail. | `README.md` has `## Choose a Workflow`; `docs/cookbook.md` links solver selection; `docs/tutorial.md` has `### Choose a Workflow First`; `docs/solver_selection.md` is dedicated. | Day 3 should define one concise solver-selection front door and Day 8 should consolidate diagnostics and advanced-control escalation. |
| Run solve | Examples are runnable but the simplest path spans README code blocks, `examples/README.md`, tutorial sections, and cookbook entries. | `examples/README.md` has `example_basic_solve`, `example_compressed_input`, and many solver examples; README has embedded code and example links. | Day 4-5 should choose a short maintained example ladder and link it consistently from README/INSTALL/cookbook/tutorial. |
| Inspect diagnostics | Diagnostics appear in examples, cookbook, solver guidance, and public API comments, but they are not a clear first-use step. | `docs/cookbook.md` references `sparse_err_t`; `examples/README.md` calls out diagnostics in compressed input; headers document errors in detail. | Day 8 should add a clear diagnostics handoff: check return codes, inspect solver stats/status, then move to advanced controls. |
| Advanced controls | Runtime/backend, repeated-run handles, benchmark interpretation, and report evidence are valuable but can overwhelm first-use paths. | README has runtime/backend controls at line 194, repeated-run lifecycle at line 421, performance at line 710; benchmark README is 662 lines. | Day 3 should define routing rules so advanced controls remain discoverable but outside the first-use front door. |
| Installed consumer | Static-first package proof is strong, but the user-facing route from local example to installed CMake consumer is split across INSTALL and examples. | `INSTALL.md` covers `pkg-config` and CMake consumers; `examples/README.md` has `examples/cmake_example`. | Day 7 should make the installed-consumer path obvious while preserving static-first and platform boundaries. |

## Stale Wording And Support-Tier Register

| Finding | Current interpretation | Action |
| --- | --- | --- |
| Historical Sprint 144 artifacts contain before-change macOS supplemental wording. | Correct historical evidence, not stale current public docs. | Exclude historical planning artifacts from stale public-surface cleanup scans unless the artifact claims to represent current state. |
| Current public support-tier wording distinguishes Linux, macOS, and Windows. | Good: Linux strongest reviewed baseline, macOS reviewed static-first install/export proof, Windows CMake-first and narrower. | Preserve this exact split when simplifying README and INSTALL. |
| Static-first package wording appears in README and INSTALL with repeated non-claims. | Accurate but dense for first-use readers. | Keep short first-use wording near the front door and route detailed non-claims to INSTALL/maintainer guide. |
| Benchmark/report wording correctly warns against broad portability/performance readings. | Good but advanced. | Keep benchmark/report details behind links from first-use docs. |
| QR and partial-SVD non-claims are present in public docs. | Good but sometimes placed near front-door feature lists. | Day 8 should preserve bounded claims while making the first solver-choice path shorter. |

## Public Header Cleanup Candidates

| Priority | Header | Friction signal | Risk | Proposed Day 9 decision |
| --- | --- | --- | --- | --- |
| High | `include/sparse_matrix.h` | 617 lines; core first-use matrix API; includes SuiteSparse threshold commentary and many internal/cache details. | Medium because ownership, mutation, and error semantics are contract-critical. | Audit for comments that can move to docs while preserving API contracts. |
| High | `include/sparse_iterative.h` | 773 lines; main iterative solver and diagnostics surface; dense options/workspace comments. | Medium because solver stats and workspace ownership are user-visible. | Select focused comments around diagnostics and repeated-run state for clarity. |
| High | `include/sparse_qr.h` | 391 lines; QR adoption surface after Sprint 139 claim closure. | Medium because bounded QR claim wording must remain precise. | Audit first-use QR comments and ensure evidence boundaries stay in docs. |
| High | `include/sparse_svd.h` | 260 lines; SVD/partial-SVD adoption surface after Sprint 140. | Medium because partial-SVD failure/partial-result semantics are important. | Audit comments for concise public contract and bounded claim wording. |
| Medium | `include/sparse_lu.h`, `include/sparse_cholesky.h`, `include/sparse_ldlt.h` | Direct solver entry points contain lifecycle, backend, and ABI history details. | Medium-high due API contract and ABI warning sensitivity. | Consider only if Day 9 finds obvious maintainer-only wording. |
| Medium | `include/sparse_analysis.h`, `include/sparse_eigs.h` | Advanced repeated-run surfaces are large and detailed. | High because advanced lifecycle semantics are easy to weaken. | Prefer routing from first-use docs rather than editing unless a clear adoption blocker appears. |
| Low | `include/sparse_vector.h`, `include/sparse_bidiag.h`, `include/sparse_ic.h`, `include/sparse_csr.h`, `include/sparse_dense.h`, `include/sparse_ilu.h`, `include/sparse_reorder.h`, `include/sparse_types.h`, `include/sparse_lu_csr.h` | Smaller or more specialized. | Varies. | Defer unless Day 11 coherence pass finds a direct conflict. |

## Claim Discovery And Overread Risks

| Evidence family | Discovery risk | Overread risk | Guardrail |
| --- | --- | --- | --- |
| QR | Earned fixture-local QR closure is split across solver docs, cookbook, examples, and maintainer guide. | Users may read QR examples as broad QR/SuiteSparse/LAPACK/NumPy/SciPy parity. | Keep a short QR front-door claim and link to solver-selection/evidence detail. |
| Partial-SVD | Partial-SVD evidence is precise but technical. | Users may infer broad repeated-spectrum, sparse-output, convergence-rate, or performance claims. | Present partial-SVD as bounded maintained confidence with failure semantics. |
| Reports | Normalized rows are source-controlled and useful. | Users may infer generated reports are fresh without rerunning commands. | Always pair report links with freshness wording. |
| Runtime/backend | Backend controls are useful but advanced. | Users may infer portable performance or backend superiority. | Route performance interpretation to benchmark docs and maintainer guide. |
| Package/platform | Static-first and support-tier wording is accurate but dense. | Users may infer shared libraries, package-manager support, Windows parity, or broad macOS parity. | Keep support split explicit but concise in first-use docs. |

## Ranked Fix Shortlist

| Rank | Candidate fix | Adoption value | Implementation risk | Validation cost | Claim-boundary risk | Recommended sprint day |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Design a single high-level workflow front door that routes build/install, choose solver, run solve, diagnostics, and advanced controls. | High | Low | Low | Low if it only routes existing evidence. | Day 3 |
| 2 | Create an example/cookbook ladder: basic solve, compressed input, solver-selection branch, diagnostics, installed consumer. | High | Medium | Medium because examples may need build/run checks. | Medium because examples must not broaden solver claims. | Days 4-5 |
| 3 | Simplify README so first-use workflow appears before dense capability/performance/testing detail. | High | Medium | Low-medium docs scans. | Medium because support-tier/non-claim wording must remain intact. | Day 6 |
| 4 | Simplify INSTALL around static-first first-use install and downstream consumption, with detailed proof routed lower. | High | Medium | Medium if install commands change. | Medium for package/platform overclaims. | Day 7 |
| 5 | Consolidate solver-selection and diagnostics front-door wording. | Medium-high | Medium | Low-medium docs scans. | Medium-high for QR/partial-SVD overreads. | Day 8 |
| 6 | Design selected public-header cleanup for `sparse_matrix.h`, `sparse_iterative.h`, `sparse_qr.h`, and `sparse_svd.h`. | Medium | Medium-high | High if headers changed. | Medium-high because comments are API contracts. | Days 9-10 |
| 7 | Keep benchmarks/report details as advanced links rather than first-use content. | Medium | Low | Low. | Low if non-claims are preserved. | Days 3, 11 |

## Current Versus Historical Wording Policy

Day 2 found no current public support-tier wording that must be fixed before
Day 3 design. Historical planning artifacts can and should retain before-change
phrasing. Sprint 145 stale-wording scans should focus on current public
surfaces unless explicitly validating a new sprint artifact that claims current
state.

## Day 2 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| First-use blockers are backed by concrete file references. | Complete | Friction inventory lists file names, headings, and line-count evidence. |
| Stale or ambiguous support-tier wording is separated from historical planning evidence. | Complete | Stale wording register distinguishes current public surfaces from historical Sprint 144 artifacts. |
| Candidate fixes are ranked before workflow design begins. | Complete | Ranked fix shortlist assigns value, risk, validation cost, claim risk, and likely sprint day. |

## Day 3 Handoff

Day 3 should design the high-level workflow front door around the top-ranked
fix: build/install, choose solver, run solve, inspect diagnostics, and move to
advanced controls. The design should specify which front-door content belongs
in README, INSTALL, cookbook, tutorial, solver selection, examples, and
maintainer guide before implementation starts.
