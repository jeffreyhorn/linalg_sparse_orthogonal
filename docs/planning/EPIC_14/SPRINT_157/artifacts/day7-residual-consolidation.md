# Day 7 Residual Consolidation

## Scope

Day 7 converts the Epic 13 residual queue and Epic 14 review findings into one
claim-oriented backlog for Epic 14. The consolidation merges duplicate residuals
by the public claim or product boundary they would enable, not by the sprint or
file where the residual was first recorded.

This artifact does not select the final Epic 14 targets. Day 8 owns the final
selected-target and explicit-non-goal decision.

## Inputs Reviewed

| Input | Role in consolidation |
| --- | --- |
| `docs/planning/EPIC_13/EPIC_13_RETROSPECTIVE.md` | Highest-priority next-epic candidates and long-horizon deferrals. |
| `docs/planning/EPIC_13/SPRINT_156/artifacts/day11-residual-queue-publication.md` | Full 18-row residual queue with owner, blocker, prerequisite, and gate fields. |
| `docs/planning/EPIC_14/reviews/review-codex-2026-08-14.md` | Epic 14 gap assessment and state-of-the-art limits. |
| `docs/planning/EPIC_14/reviews/todo-codex-2026-08-14.md` | Step-by-step closure plan and explicit non-goals. |
| `docs/planning/EPIC_14/PROJECT_PLAN.md` | Sprint 157-166 implementation allocation. |
| Sprint 157 Day 1-6 artifacts | Current baseline for source/API, tests/CI, docs/claims, generated outputs, package, ABI, and platforms. |

## Consolidation Rules

1. Merge residuals by the claim surface they would enable.
2. Keep owner, blocker, prerequisite, and promotion gate visible for every
   consolidated residual.
3. Mark complete-closure candidates only when Epic 14 has a bounded sprint,
   deliverable, and validation path.
4. Preserve broad state-of-the-art, package-manager, shared-library, dynamic
   ABI, broad ecosystem parity, and portable performance superiority work as
   retained non-claims unless a later product decision explicitly expands
   scope.
5. Do not treat generated local rows, advisory benchmark rows, source-controlled
   report-owner rows, or planning artifacts as pass evidence by themselves.

## Consolidated Residual Register

| ID | Claim surface | Inherited residuals | Category | Owner role | Blocker | Prerequisites | Promotion gate or retained non-claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E14-R01 | Generated API reference publication | E13-R15; Epic 14 G1 | API/docs | Documentation/API owner | No tracked `docs/api/html/` tree; generated HTML freshness and page coverage are not currently reviewable from source control. | Run `make docs`, capture warnings, decide generated `sparse_version.h` input policy, verify intended public-header page coverage, and select committed, CI-published, or local-only policy. | Sprint 158 closes with committed generated HTML plus coverage evidence, or an explicit no-commit/local-only guard and docs decision. |
| E14-R02 | Hosted selected generated oracle/comparison evidence | E13-R06, E13-R07; Epic 14 G2 | Generated evidence/CI | CI, corpus, comparison, and report owners | Selected QR, partial-SVD, oracle, and comparison generated rows remain ignored local artifacts under `build/`. | Select claim-bearing families, measure runtime, define artifact retention, tighten hosted stale/missing/failing semantics, and keep advisory families out of scope. | Sprint 159 reviewed hosted lane runs selected freshness gates and publishes or summarizes artifacts without broadening solver claims. |
| E14-R03 | Bounded QR comparison breadth | E13-R09, E13-R10; Epic 14 G3 | Comparison/QR | QR and comparison owners | Current comparison evidence is one local QR minimum-norm fixture and does not support broad QR or ecosystem parity. | Select one QR fixture family, define basis-invariant metrics/tolerances, extend harness/report rows, and document dependency/skip/defer semantics. | Sprint 160 adds one bounded QR comparison family with passing freshness and fixture-local docs; broad QR parity remains rejected. |
| E14-R04 | Bounded partial-SVD comparison publication | E13-R11, E13-R12; Epic 14 G3 | Comparison/SVD | SVD and comparison owners | Partial-SVD has corpus evidence but no selected normalized external comparison family. | Select one subspace-safe fixture family, define singular-value/projector/residual/orthogonality/convergence/fail-closed metrics, extend harness/report rows. | Sprint 161 publishes one bounded partial-SVD comparison family with selected freshness; broad SVD parity and raw vector identity remain rejected. |
| E14-R05 | Windows package parity decision | E13-R01, E13-R02; Epic 14 G5 | Platform/package | Platform/package owner | Windows package support is reviewed CMake install/downstream only; Makefile and `pkg-config` execution parity remain non-claims. | Evaluate Windows `pkg-config` and Makefile parity separately, choose product scope, define provider/shell/link behavior if promoting. | Sprint 162 either implements a reviewed selected proof or strengthens the retained non-claim with exact checks and docs. |
| E14-R06 | Methodology-bound performance publication | E13-R08, E13-R14; Epic 14 G4 | Performance/reports | Benchmark and report owners | Benchmarks, sentinel rows, large-matrix guardrails, dead-code, and coverage outputs are mostly local/advisory or supplemental. | Select a bounded report subset, define methodology fields, classify hard/advisory rows, and preserve local-only semantics where applicable. | Sprint 163 publishes methodology-bound report evidence for selected rows; portable superiority remains rejected. |
| E14-R07 | Public header and API coherence cleanup | E13-R16; Epic 14 G7 | API/docs | Header owners | Public declarations are broad, generated HTML is absent, and some header comments still need declaration-preserving cleanup. | Select high-impact headers, capture normalized declarations before editing, preserve signatures, align docs and generated-docs policy. | Sprint 164 lands a header cleanup batch with zero declaration drift or explicit API review and required quality gates. |
| E14-R08 | Static-first package boundary hardening | E13-R04, E13-R05; Epic 14 G6 | ABI/package | Package/ABI owner | Shared-library and dynamic ABI are intentionally deferred; accidental metadata or wording could still imply support. | Audit package metadata, `BUILD_SHARED_LIBS` rejection, public structs, version docs, install docs, CMake package files, and `.pc` metadata. | Sprint 165 hardens static-first proof and publishes future shared-library/package-manager residuals; shared-library and dynamic ABI remain non-claims. |
| E14-R09 | Broad external ecosystem parity | E13-R13; Epic 14 G3/G9 | External parity | Solver and comparison owners | Optional NumPy/SciPy baselines and broader LAPACK, SuiteSparse, Eigen, PETSc, and Trilinos comparisons are not selected proof. | Dependency/version/provenance policy, target selection, tolerance policy, and hosted or reproducible execution model. | Retained long-horizon non-claim in Epic 14 except for the one QR and one partial-SVD bounded comparison families. |
| E14-R10 | Package-manager distribution | E13-R03; Epic 14 G6 | Package/release | Package/release owner | No recipe ownership, release workflow, update/uninstall policy, shared/static product decision, or channel validation. | Release/version policy, package recipe maintainers, external channel support, install/uninstall validation, and support docs. | Retained long-horizon non-claim; package-manager availability must not be implied by install/export proof. |
| E14-R11 | Full shared-library product support | E13-R04; Epic 14 G6 | ABI/package | Package/ABI owner | Export/import macros, symbol visibility, SONAME/install-name/RPATH/DLL metadata, shared consumers, and loader validation are absent. | Product decision, shared build/install design, symbol allowlist, platform loader policy, and installed shared consumer tests. | Retained long-horizon non-claim; Sprint 165 may only harden the static-first boundary and future gate. |
| E14-R12 | Dynamic ABI compatibility policy | E13-R05; Epic 14 G6/G10 | ABI/product | ABI owner | Public structs, callbacks, enums, allocator/lifetime rules, error state, and versioning lack a reviewed ABI promise. | ABI stability level, compatibility window, exported-header audit, binary compatibility tests, and release policy. | Retained long-horizon non-claim; exact package versions do not imply ABI compatibility. |
| E14-R13 | Broad state-of-the-art positioning | E13-R17; Epic 14 G10 | Claims/product | Epic/product owner | Evidence remains bounded by selected fixtures, local and hosted lanes, static-first package support, and narrow comparisons. | Broad external parity, performance methodology, package maturity, ABI policy, platform support, and recurring evidence. | Sprint 166 must publish a claim audit that rejects unsupported state-of-the-art claims unless recurring evidence exists. |
| E14-R14 | Runtime/backend API promotion and wider sentinel control | E13-R18; Epic 14 G4/G9 | Runtime/backend | Runtime/backend and benchmark owners | No selected typed runtime/backend API, ABI scope, or standalone sentinel expansion is in Epic 14. | API design, ABI review, metrics, budgets, docs, report rows, and support-tier policy. | Retained long-horizon non-claim unless tied to a selected API/ABI decision in a future epic. |
| E14-R15 | Maintainability and large-file pressure | Epic 14 G8 | Maintainability | Maintainer and test owners | Several proof-owner tests and scripts are large, making review and targeted fixes harder. | Focused proof-owner pattern, source-list consistency checks, and refusal to expand monoliths unless behavior requires it. | Epic 14 handles this as a quality rule, not a standalone rewrite: new work should use focused tests and preserve source-list checks. |
| E14-R16 | Optional data, coverage, dead-code, and advisory report semantics | E13-R08, E13-R13; Epic 14 G9/G10 | Reports/claims | Report and maintainer owners | Optional/skipped/advisory rows can be overread as pass evidence or coverage completeness. | Row-level support-tier wording, freshness semantics, skip/defer policy, and public claim audit. | Retained as claim-governance constraint; only explicitly selected families may become claim-bearing. |

## Complete-Closure Candidate Shortlist

| Candidate | Epic 14 sprint | Why it can close completely |
| --- | --- | --- |
| E14-R01 generated API reference publication | Sprint 158 | Bounded docs/tooling decision with concrete `make docs`, warning, page-coverage, and publication-policy outputs. |
| E14-R02 hosted selected generated evidence | Sprint 159 | Already has mature local commands; closure is CI promotion and support-tier wording, not new solver semantics. |
| E14-R03 one QR comparison family | Sprint 160 | Can select one fixture family and land bounded metrics, report rows, freshness, and docs. |
| E14-R04 one partial-SVD comparison family | Sprint 161 | Can publish one subspace-safe comparison family without broad SVD parity. |
| E14-R05 Windows package parity decision | Sprint 162 | Can end with either a reviewed selected proof or a stronger explicit retained non-claim. |
| E14-R06 methodology-bound performance publication | Sprint 163 | Can publish a bounded report with explicit methodology and non-superiority language. |
| E14-R07 public header/API coherence batch | Sprint 164 | Can select a finite header batch and prove declaration preservation. |
| E14-R08 static-first package boundary hardening | Sprint 165 | Can harden guards and docs while preserving shared-library/dynamic ABI deferrals. |
| E14-R13 final claim recalibration | Sprint 166 | Can map every earned claim to evidence and publish residuals for anything not closed. |

## Retained Long-Horizon Non-Goals

| Non-goal | Consolidated residual | Reason retained |
| --- | --- | --- |
| Package-manager distribution | E14-R10 | Requires release/channel ownership and package recipe support beyond Epic 14 scope. |
| Full shared-library product support | E14-R11 | Requires cross-platform binary product design and loader validation. |
| Dynamic ABI compatibility promise | E14-R12 | Requires ABI policy, compatibility tests, and release/version governance. |
| Broad external ecosystem parity | E14-R09 | Requires multiple dependency families, broader matrix coverage, and recurring evidence. |
| Portable performance superiority | E14-R06/E14-R13 | Sprint 163 can publish methodology-bound evidence, not superiority across machines. |
| Broad Windows Makefile parity | E14-R05 | Sprint 162 may explicitly retain this non-claim unless selected product scope changes. |
| Runtime/backend API promotion | E14-R14 | Requires API and ABI scope not selected by the current Epic 14 plan. |
| Unqualified state-of-the-art claim | E14-R13 | Current evidence can improve posture but does not support broad state-of-the-art status. |

## Promotion Gate Summary

| Gate type | Required evidence before claim widens |
| --- | --- |
| API docs | Generated docs command, warning triage, page coverage, and explicit publication policy. |
| Hosted generated evidence | Reviewed hosted lane, selected family list, artifact retention or deterministic summary, stale/missing/failing row semantics, and docs support-tier update. |
| Comparison | Fixture selection, metric contract, dependency provenance, skip/defer semantics, normalized rows, freshness check, and fixture-local docs. |
| Windows package | Product-scope decision, selected toolchain/provider if promoted, hosted proof or rejection guard, and synchronized README/INSTALL/maintainer/workflow wording. |
| Performance | Methodology fields, row classification, bounded runtime, report artifact, and explicit rejection of portable superiority. |
| Header/API cleanup | Before/after declaration capture, zero signature drift or reviewed API change, generated-doc policy application, and C/header quality gates if headers change. |
| Static package/ABI | Install/export proof, static deferral guard, metadata audit, and explicit shared-library/dynamic ABI residual publication. |
| Final claims | Public docs scan, evidence-owner mapping, plan reconciliation, and final residual queue. |

## Day 8 Inputs

Day 8 should use this consolidated register to make the target decision:

- selected complete-closure targets should come from E14-R01 through E14-R08
  and E14-R13 final claim recalibration;
- E14-R09 through E14-R12 and E14-R14 should remain explicit non-goals unless
  the sprint plan is re-scoped;
- E14-R15 and E14-R16 should be applied as quality and claim-governance rules
  across selected implementation sprints rather than as broad rewrite work.

## Completion Check

- Duplicate Epic 13 and Epic 14 residuals are consolidated by claim surface.
- Every consolidated residual has owner, blocker, prerequisite, and gate or
  retained non-claim fields.
- Complete-closure candidates are separated from long-horizon product and
  ecosystem work.
- Generated local output, advisory rows, package proof-owner rows, and planning
  artifacts remain excluded from pass evidence unless a future sprint promotes
  them explicitly.
