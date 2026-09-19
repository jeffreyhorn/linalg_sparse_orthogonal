# Day 2: Public Documentation Audit

**Sprint:** 205 - Support Matrix and Adoption Quick-Reference Consolidation  
**Theme:** Audit user-facing documentation for duplicate caveats, stale support
truth, and adoption friction.  
**Time estimate:** 12 hours  
**Branch:** `sprint-205`  
**Base commit:** `9a2b4ff6`

## Purpose

Day 2 audits public documentation before any consolidation edits. The goal is
to identify where users encounter duplicated caveats, too much proof-owner
detail, missing support-truth links, or adoption friction. This artifact is an
evidence record only; it does not change public documentation.

## Reviewed Public Surfaces

| Surface | Audit coverage | Day 2 classification |
| --- | --- | --- |
| `README.md` | Start Here, Adoption Map, capability sections, build/API/Windows/benchmark/install/documentation sections. | Accurate but high-friction for support claims. It is a strong front door, but later sections repeat detailed package, Windows, API, and benchmark caveats that should route to authoritative owners where safe. |
| `INSTALL.md` | Start Here, Support Split, Support Readiness Matrix, install contract, platform notes, verification, package rows. | Current public support authority. It should remain the source of truth for support/readiness status, package, ABI, platform, local generated API, and package-manager boundaries. |
| `docs/tutorial.md` | Getting Started, documentation map, local link/install handoff, diagnostics/advanced handoffs. | Good learning path with correct links. Repeats support/readiness and benchmark caveats enough that Day 7 can replace some prose with links once the quick reference exists. |
| `docs/cookbook.md` | First-use ladder, data-first routes, solver-family recipes, benchmark handoff. | Strong candidate for quick-reference adjacency. It already has problem-shape routing and support caveats, but the first-use ladder can become a shorter adoption route. |
| `docs/solver_selection.md` | First-use solver route, matrix start table, smallest workflow table, diagnostics handoff, solver-family evidence. | Detailed solver decision authority. It should not become the compact quick reference wholesale because it includes extensive selected-evidence boundaries. |
| `docs/api_reference.md` | Source of truth, generated HTML, workflow guides, claim boundaries. | Accurate Sprint 204 local-only API route. Preserve as the API authority; avoid copying generated API caveats into unrelated docs except through links. |
| `examples/README.md` | Start Here, building, diagnostics handoff, program list, writing your own. | Good runnable-example owner. It should point to the quick reference/support matrix, but should not duplicate install/package/benchmark caveats. |
| `benchmarks/README.md` | Quick navigation, reading benchmark results, compile-only gate, workflow groups, report index handoff. | Benchmark/report authority. It contains necessary non-performance claim wording and should remain deep-dive evidence rather than the adoption front door. |

## Duplication And Friction Inventory

| Topic | Public surfaces where it appears | Status | Consolidation candidate |
| --- | --- | --- | --- |
| Support/readiness ownership | README, INSTALL, tutorial, cookbook, solver selection, API reference, examples. | Accurate and linked, but repeated. | Keep `INSTALL.md#support-readiness-matrix` authoritative; replace repeated explanatory prose with shorter links where safe. |
| Package-manager and Homebrew non-claims | README, INSTALL, API reference, solver selection, benchmark docs. | Accurate but verbose in README and INSTALL. | Keep full detail in INSTALL; README should summarize and link. Other docs should avoid provider lists unless needed. |
| Shared-library and dynamic ABI deferral | README, INSTALL, API reference, solver selection. | Accurate and important. | Keep in INSTALL/API claim boundaries; link from workflow docs. |
| Windows support and report-freshness caveats | README, INSTALL, solver selection, maintainer-linked surfaces. | Accurate but hard for users to classify because workflow evidence and support promotion are both described. | Keep support status in INSTALL; quick reference should use short labels such as "validated", "guarded workflow", and "deferred". |
| Generated API local-only policy | README, INSTALL, API reference. | Accurate Sprint 204 policy. | Keep detailed explanation in `docs/api_reference.md`; support matrix can link to it. README can stay concise. |
| Benchmark/performance non-claims | README, tutorial, cookbook, solver selection, benchmarks. | Accurate and necessary, but users see the caveat before they know which benchmark surface they need. | Keep benchmark interpretation in `benchmarks/README.md`; quick reference should route users there only after workflow selection. |
| Solver-family selected evidence boundaries | README, cookbook, solver selection. | Accurate but too detailed for first-use adoption. | Keep detailed evidence in solver selection; quick reference should name first workflow and link out. |
| Diagnostics handoff | README, tutorial, cookbook, solver selection, examples. | Useful but distributed. | Day 9 should normalize vocabulary and Day 6-Day 8 can route all first-use docs to one concise diagnostics handoff. |

## Support/Readiness Wording Status

| Surface | Current wording status | Evidence source | Day 2 disposition |
| --- | --- | --- | --- |
| README | Claim-safe but too broad for one front-door document in the installation, Windows, benchmark, generated API, and selected-evidence sections. | Links to INSTALL, benchmarks, API reference, solver selection, and maintainer guide. | Candidate for consolidation by replacing repeated proof-owner paragraphs with links after support truth is selected. |
| INSTALL | Claim-safe and authoritative. Some rows are necessarily dense because they combine user status, evidence owner, and non-claims. | Support readiness matrix, install validation tests, package and platform evidence. | Keep as source of truth; Day 5 should decide whether to add clearer labels or anchors rather than moving authority elsewhere. |
| Tutorial | Claim-safe and user-oriented. Repeats install/support and benchmark boundaries in several handoffs. | README, INSTALL, benchmark docs, solver selection. | Candidate for shorter references after quick-reference routing exists. |
| Cookbook | Claim-safe. First-use ladder is close to the desired quick reference but still data-first rather than complete adoption reference. | INSTALL, solver selection, benchmarks, examples. | Candidate location or source material for quick-reference design. |
| Solver selection | Claim-safe and detailed. It contains the clearest problem-shape table but also deep selected-evidence caveats. | Public headers, tests, selected report targets, benchmarks. | Keep as detailed owner; quick reference should link here, not duplicate it. |
| API reference | Claim-safe. Generated HTML wording is current with Sprint 204 local-only policy. | `make api-docs-freshness`, local-only/routing guards, public headers. | Preserve; any simplification needs API routing/local-only guard coverage. |
| Examples README | Claim-safe and practical. It routes support/readiness and benchmark interpretation to the right owners. | Examples, INSTALL, benchmarks, solver selection. | Keep example-focused; add quick-reference route later if selected. |
| Benchmarks README | Claim-safe and authoritative for benchmark/report interpretation. Dense by design. | Benchmark report scripts, selected performance metadata, report-index manifests. | Keep as benchmark authority; do not make it the adoption quick reference. |

## User Workflow Questions That Remain High-Friction

| User question | Current path | Friction |
| --- | --- | --- |
| "I have CSR/CSC/Matrix Market data; what should I run first?" | README Start Here -> cookbook ladder -> examples -> solver selection. | Correct but multi-hop; a compact quick reference can route this directly. |
| "Which solver family should I start with?" | README Choose a Workflow -> solver selection -> examples. | Correct, but selected-evidence caveats appear before the user may need them. |
| "Can I install this for another project?" | README Installation -> INSTALL Start Here -> support matrix -> consumer tutorial. | Correct; support matrix density makes first decision slower. |
| "Is Homebrew/package-manager support available?" | README Installation and INSTALL package-manager rows. | Correct answer is no broad support, but details are long and repeated. |
| "Does Windows support mean Windows report freshness?" | README and INSTALL explain guarded/re-deferred paths. | Accurate but subtle; quick labels and support matrix routing would reduce ambiguity. |
| "Where are API docs?" | README build commands -> `docs/api_reference.md` -> local generated API policy. | Correct; generated local-only boundary is repeated in multiple places. |
| "Can benchmark rows prove performance?" | README, cookbook, solver selection, benchmarks. | Correct no-claim wording appears often; route users to benchmark interpretation once. |
| "What diagnostic should I inspect first?" | README, examples, solver selection, tutorial, cookbook. | Many partial answers; Day 9 should make a shared vocabulary and Day 6-Day 8 should route to it. |

## Claim-Safety Findings

| Finding | Severity | Notes |
| --- | --- | --- |
| No obvious stale support promotion was found in the audited public surfaces. | Low | The docs consistently route package, ABI, platform, benchmark, and generated API boundaries to owner docs. |
| README contains the most repeated proof-owner detail. | Moderate friction, low claim risk | It is accurate, but installation, Windows, benchmark, generated API, and selected-evidence paragraphs make the front door long. |
| INSTALL is dense but should remain authoritative. | Moderate friction, low claim risk | The support/readiness matrix is the only public surface that cleanly separates user status, path, evidence owner, and retained non-claims. |
| Solver selection mixes quick first-use guidance with detailed selected-evidence boundaries. | Moderate friction, low claim risk | It should remain the deep-dive solver evidence owner; a shorter quick reference should link into it. |
| Benchmark caveats are necessarily repeated across adoption docs. | Moderate friction, low claim risk | Consolidation should not remove benchmark no-claim semantics without a benchmark docs link. |

## Day 2 Recommendations For Later Days

1. Keep `INSTALL.md#support-readiness-matrix` as the public support truth.
2. Design the Sprint 205 quick reference as a short routing table, not a new
   evidence ledger.
3. Use concise support labels consistently: `supported`, `validated`,
   `hosted-evidence`, `guarded-workflow`, `local-only`, `deferred`, and
   `not claimed`.
4. Keep detailed selected-evidence and non-claim paragraphs in owner docs:
   solver selection, benchmark docs, API reference, and INSTALL.
5. Replace repeated README/tutorial/cookbook/examples caveats with links only
   after Day 5 selects the support-truth architecture and Day 11 identifies
   guard coverage.
6. Avoid moving maintainer-only proof interpretation into the user quick
   reference; keep it linked through `docs/maintainer_guide.md`.

## Validation And Hygiene

| Check | Day 2 result |
| --- | --- |
| Public docs edited | No. Day 2 is audit-only. |
| User-facing claim changed | No. |
| `.c` or `.h` files changed | No. Full C gate not required for Day 2. |
| Generated output changed | No generated output intentionally created. |
| `git diff --check` | Planned after artifact creation. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 205.1 has public-doc evidence for every listed user-facing surface. | Met for the public-doc half of 205.1. README, INSTALL, tutorial, cookbook, solver selection, API reference, examples, and benchmarks are covered here. |
| Duplicated caveats are classified before consolidation starts. | Met. Duplicate support, package, Windows, generated API, benchmark, solver-evidence, and diagnostics caveats are classified above. |
| No claim is edited without knowing its current evidence source. | Met. No public claim was edited on Day 2; evidence owners are recorded before future changes. |

## Day 2 Disposition

Day 2 is complete. Day 3 should audit maintainer, benchmark/report, manifest,
and planning-adjacent surfaces so support truth and diagnostics vocabulary are
consolidated against both public and maintainer evidence before wording changes.
