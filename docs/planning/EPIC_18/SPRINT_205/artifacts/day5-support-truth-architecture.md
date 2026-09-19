# Day 5: Support Truth Architecture

**Sprint:** 205 - Support Matrix and Adoption Quick-Reference Consolidation  
**Theme:** Design the centralized support/readiness routing model that will
replace repeated caveats with safe links.  
**Time estimate:** 12 hours  
**Branch:** `sprint-205`  
**Base commit:** `9a2b4ff6`

## Purpose

Day 5 defines the support/readiness ownership model that Day 6-Day 8 should
implement. The goal is to reduce repeated caveats while preserving the exact
claim boundaries established by Sprints 198-204 and the Day 2-Day 4 audits.

This is an architecture artifact only. It does not change user-facing docs.

## Central Ownership Model

| Support area | Public source of truth | Detail/proof owner | Sprint 205 rule |
| --- | --- | --- | --- |
| Current support/readiness status | `INSTALL.md#support-readiness-matrix` | `docs/maintainer_guide.md` | Keep INSTALL as the one public support truth. Other public docs should link here instead of repeating the full matrix. |
| First-use workflow routing | `docs/cookbook.md` quick reference after Day 6 | `docs/solver_selection.md`, examples, public headers | Use cookbook for compact routing; use solver selection for detailed solver evidence. |
| Package-manager status | `INSTALL.md#support-readiness-matrix` and package-manager row | Homebrew proof material, package deferral guards, Sprint 198 artifacts | Public docs may say "source install via Make/CMake"; do not claim Homebrew or broad package-manager support. |
| Shared-library and dynamic ABI | `INSTALL.md#support-readiness-matrix` and installed contract sections | Sprint 170 package/ABI decision and static-package guard | Keep explicit static-first wording near install examples. |
| Platform and Windows status | `INSTALL.md#support-readiness-matrix` | Windows workflow docs, PowerShell validator, maintainer guide, selected target manifest | Use concise public labels: `validated`, `guarded workflow`, `deferred`. Do not promote selected Windows freshness. |
| Generated API docs | `docs/api_reference.md` and INSTALL support row | API docs local-only/routing guards, maintainer guide | Public docs should route to `docs/api_reference.md`; generated HTML remains local-only. |
| Benchmark/report interpretation | `benchmarks/README.md` | `tests/corpus/manifests/selected_report_targets.tsv`, report schema, maintainer guide | Keep benchmark rows out of broad performance claims; link to benchmark docs after workflow selection. |
| Selected comparison/oracle targets | Public docs should link to solver/benchmark docs | `tests/corpus/manifests/selected_report_targets.tsv` and report schema | Do not duplicate selected target row lists in the quick reference. |
| Release/state-of-the-art status | `INSTALL.md#support-readiness-matrix` for current non-claims | Epic residual queue and maintainer guide | Keep public wording concise; do not imply release or state-of-the-art evidence. |

## Duplicate Caveat Conversion Plan

| Caveat location | Current friction | Conversion plan | Inline boundary required? |
| --- | --- | --- | --- |
| README installation section | Repeats static-first, Windows, package-manager, Homebrew, ABI, and selected Windows caveats. | Keep a short static-first summary and link to `INSTALL.md#support-readiness-matrix`. | Yes, retain one sentence that package-manager and dynamic ABI support are not claimed. |
| README generated API/build command area | Repeats local-only generated API policy. | Link to `docs/api_reference.md` and `make api-docs-freshness`; avoid re-explaining hosted/artifact/commit non-claims in full. | Yes, retain "local-only" near the command. |
| README benchmark/performance sections | Repeats local/selected benchmark caveats and non-performance claims. | Link to `benchmarks/README.md#reading-benchmark-results` and selected freshness command. | Yes, retain "not portable performance" near benchmark commands. |
| README Windows selected freshness paragraphs | Long but accurate support/freshness distinction. | Replace with short "guarded workflow / deferred promotion" language linked to INSTALL and maintainer guide. | Yes, retain no promoted selected Windows freshness. |
| Tutorial install/support handoff | Repeats static package ownership and support matrix boundary. | Keep one link to INSTALL support matrix and remove extra package detail when Day 7 edits. | No, if the link is adjacent and no install claim is otherwise made. |
| Tutorial benchmark handoff | Repeats benchmark local-evidence caveat. | Link to benchmark reading section and keep concise local measurement wording. | Yes, retain local measurement phrase near benchmark handoff. |
| Cookbook first-use ladder | Has support caveat plus future quick-reference placement. | After Day 6, let quick reference carry the routing; keep one support matrix link. | Yes, retain compact non-widening statement under the table. |
| Solver selection advanced/benchmark reminders | Repeats package/platform/performance non-claims. | Keep key warnings because users choose advanced paths there; link to support matrix and benchmarks. | Yes, solver selection remains a high-risk overclaim surface. |
| API reference generated HTML section | Repeats local-only generated API boundaries. | Keep local-only detail here because this is the API owner; other docs should link here. | Yes, this is the owner surface. |
| Examples README benchmark/support hints | Repeats support/readiness and benchmark interpretation routing. | Link to cookbook quick reference and support matrix; avoid expanding package/benchmark caveats. | No, unless an example row directly mentions install, benchmark, or platform support. |
| Benchmarks README no-performance caveats | Dense but owner-owned. | Keep local; do not replace with links because this is the benchmark interpretation authority. | Yes, this is the owner surface. |

## Link Text And Anchor Model

Use stable, descriptive link text:

| Link text | Target | Use |
| --- | --- | --- |
| "support/readiness matrix" | `INSTALL.md#support-readiness-matrix` | Any support, platform, package, ABI, generated API, benchmark, or Windows status question. |
| "problem-shape quick reference" | `docs/cookbook.md#problem-shape-quick-reference` | README, tutorial, and examples route into the compact table after Day 6. |
| "solver-selection guide" | `docs/solver_selection.md#choose-the-smallest-workflow` | Detailed solver choice and evidence boundaries. |
| "diagnostics handoff" | `docs/solver_selection.md#diagnostics-handoff` or `examples/README.md#diagnostics-handoff` | Local result interpretation before changing workflow. |
| "API reference" | `docs/api_reference.md` | Public headers and local generated API policy. |
| "benchmark result interpretation" | `benchmarks/README.md#reading-benchmark-results` | Benchmark/report rows and performance caveats. |
| "maintainer guide" | `docs/maintainer_guide.md` | Proof interpretation, guard policy, and closure criteria; use sparingly in public docs. |

Day 6 should create the `docs/cookbook.md#problem-shape-quick-reference`
anchor by adding a `## Problem-Shape Quick Reference` heading near the first-use
ladder.

## Required Inline Non-Claims

Some boundaries must stay inline because a link alone would invite
overinterpretation:

| Surface | Inline boundary to retain |
| --- | --- |
| Build/install command snippets | Static-first install; no shared-library or dynamic ABI claim. |
| Package/Homebrew mention | No Homebrew/core, bottles, Linuxbrew, public tap, package-manager distribution, or user-facing Homebrew install path. |
| Windows mention | Windows CMake validation does not imply Windows Makefile, Windows `pkg-config`, broad Windows parity, or selected report freshness promotion. |
| Benchmark/report command mention | Local/selected freshness is not portable performance, timing threshold, release, backend-superiority, or state-of-the-art evidence. |
| Generated API command mention | Generated HTML is local-only ignored output, not hosted, retained artifact, committed output, or release evidence. |
| Solver evidence summary | Fixture/selected evidence is not broad external-library parity or broad correctness proof. |

## Public-Doc Update Map

| File | Day 6-Day 8 update | Purpose |
| --- | --- | --- |
| `docs/cookbook.md` | Add `## Problem-Shape Quick Reference` table and compact lead-in. | Implement item 205.2 and central user routing. |
| `README.md` | Link Start Here/Adoption Map to the quick reference; shorten repeated install/Windows/benchmark/API caveats where safe. | Make quick reference discoverable and reduce front-door duplication. |
| `docs/tutorial.md` | Route early documentation map or first-use text to the quick reference; shorten duplicated support handoff wording if safe. | Keep tutorial focused on learning path. |
| `examples/README.md` | Add quick-reference route near Start Here; keep examples as runnable-owner docs. | Help users choose example by problem shape. |
| `docs/solver_selection.md` | Keep detailed owner; add reciprocal link to quick reference only if it improves navigation. | Preserve deep decision tree while reducing first-use pressure. |
| `docs/api_reference.md` | No Day 6 change expected unless links need support-truth calibration. | Preserve Sprint 204 local-only API owner. |
| `benchmarks/README.md` | No Day 6 change expected unless README/cookbook links need benchmark anchors. | Preserve benchmark interpretation authority. |
| `docs/maintainer_guide.md` | Later Day 7/Day 11 may add support-truth ownership notes if needed. | Keep maintainer proof interpretation aligned. |

## Maintainer/Guard Update Map

| Surface | Later update candidate | Reason |
| --- | --- | --- |
| Install docs guard | Ensure support matrix remains linked from quick reference and public support summaries. | Prevent support-truth drift. |
| API docs routing guard | Ensure generated API links route to `docs/api_reference.md`, not generated HTML. | Preserve Sprint 204 local-only policy. |
| Benchmark docs tests | Ensure benchmark quick-reference wording does not claim portable performance. | Preserve Sprint 202 selected benchmark boundary. |
| Windows claim validators | Ensure shortened Windows wording still says selected Windows freshness remains unpromoted where relevant. | Preserve Sprint 199/Sprint 203 re-deferrals. |
| New focused quick-reference guard | Consider validating cookbook quick-reference anchor, owner links, and forbidden broad claims. | Protect Sprint 205 consolidation work. |

## Implementation Constraints

Day 6-Day 8 should follow these constraints:

1. Do not change public headers or C sources for support wording work.
2. Do not add new support/readiness statuses unless Day 5 ownership maps them
   to existing evidence.
3. Do not introduce planning artifacts as primary public workflow links.
4. Do not duplicate selected target row lists outside manifest/report owner
   surfaces.
5. Do not remove inline non-claims near high-risk build/install, Windows,
   benchmark, generated API, or package-manager wording.
6. Prefer one authoritative link plus a short boundary sentence over long
   repeated caveat blocks.

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 205.3 has an implementation-ready routing model. | Met. The central ownership model, link text, anchor plan, public-doc map, and guard map are recorded here. |
| Replaced caveats have authoritative destinations. | Met. Each conversion candidate maps to INSTALL, cookbook, solver selection, API reference, benchmarks, maintainer guide, or manifest/schema owners. |
| Critical non-claims remain visible where users could otherwise overinfer support. | Met. Required inline non-claims are listed for install, package, Windows, benchmark, generated API, and solver evidence surfaces. |

## Day 5 Disposition

Day 5 is complete. Day 6 should implement the quick reference in
`docs/cookbook.md`, add discovery links from README/tutorial/examples, and
record the exact claim-boundary changes before Day 7 performs broader support
truth consolidation.
