# Sprint 156 Day 11: Residual Queue Publication

## Purpose

Publish the final Epic 13 residual queue with owner roles, blockers,
prerequisites, and promotion gates. This artifact consolidates residuals from
Sprints 147-155 and Sprint 156 Days 1-10 so deferred work cannot be mistaken
for completed support.

## Inputs Reviewed

- Sprint 147-155 `RETROSPECTIVE.md` files.
- Sprint 147-155 closeout and handoff artifacts.
- Sprint 156 Day 1-10 artifacts, especially:
  - `day5-package-validation.md`;
  - `day6-platform-reconciliation.md`;
  - `day7-corpus-report-validation.md`;
  - `day8-comparison-reconciliation.md`;
  - `day9-adoption-api-reconciliation.md`;
  - `day10-claim-audit.md`.

## Consolidation Rules

- Merge duplicate residuals by the claim they would enable, not by the sprint
  where they were recorded.
- Keep each residual tied to an owner role and a concrete blocker.
- Require an executable or reviewable promotion gate before any public claim
  can widen.
- Prefer complete-gap closure candidates over broad partial progress.
- Preserve long-horizon research and ecosystem work as explicit deferrals.

## Final Epic 13 Residual Queue

| ID | Category | Residual | Owner role | Blocker | Prerequisites | Promotion gate |
| --- | --- | --- | --- | --- | --- | --- |
| E13-R01 | Platform/package | Windows Makefile install/uninstall parity | Platform/package owner | Windows reviewed support remains CMake-first; Makefile install behavior is not implemented or reviewed. | Decide whether Windows Makefile support is product scope; define MSVC/MinGW expectations and install layout. | Hosted Windows lane runs Make install/uninstall or explicit rejection proof; docs and support-tier wording updated. |
| E13-R02 | Platform/package | Windows `pkg-config` execution and downstream parity | Platform/package owner | No selected reviewed Windows `pkg-config` toolchain or downstream compile/link/run proof. | Select Windows `pkg-config` provider, path handling, shell environment, and static link flags. | Hosted Windows install lane resolves `sparse.pc`, compiles/links/runs a downstream consumer, and records non-claims. |
| E13-R03 | Platform/package | Package-manager distribution support | Package/release owner | No package-manager recipe ownership, update/uninstall policy, release workflow, or channel validation. | Stable static/shared product decision, version/release policy, recipe owners, install/uninstall proof. | Selected package-manager lane builds, installs, validates version/metadata, and documents support limits. |
| E13-R04 | ABI/package | Shared-library product support | Package/ABI owner | Shared-library requests are intentionally rejected; export/import, visibility, loader, and downstream shared-consumer policy do not exist. | Product decision, `SPARSE_API` or equivalent, symbol allowlist, shared build/install design, platform loader policy. | Linux `.so`, macOS `.dylib`, and Windows DLL/import-library lanes build, install, inspect symbols, and run installed shared consumers. |
| E13-R05 | ABI/package | Dynamic ABI compatibility policy | ABI owner | Public structs, callbacks, enum values, allocator/lifetime boundaries, error state, and version metadata lack an ABI promise. | Decide ABI stability level and compatibility window; audit all exported headers and package metadata. | ABI policy is documented, tested against compatibility fixtures, and tied to versioning/release checks. |
| E13-R06 | Reports/CI | Hosted promotion for selected local-only oracle and comparison rows | CI, corpus, comparison, and report owners | QR, partial-SVD, oracle, and comparison generated rows remain local-only ignored artifacts. | Runtime budget, hosted artifact retention policy, freshness semantics, and support-tier update plan. | Reviewed hosted lane runs selected freshness gates, uploads or records evidence, and updates support tiers. |
| E13-R07 | Reports/tooling | Row-level strict generated freshness semantics beyond aggregate selected gates | Report/tooling owner | Sprint 152 selected aggregate freshness is current, but row-level strict comparison policy is not generalized. | Define strict-vs-advisory semantics per generated family and stale-row failure policy. | Normalizer tests and selected commands enforce row-level strict freshness for claim-bearing families. |
| E13-R08 | Reports/tooling | Benchmark, sentinel, large-matrix, dead-code, and coverage report publication policy | Benchmark and maintainer owners | These generated families remain advisory or supplemental and do not publish claim-bearing freshness. | Define family-specific owners, runtime budget, artifact retention, thresholds, and non-claims. | Selected generated families have report rows, freshness checks, docs, and CI or explicit local-only policy. |
| E13-R09 | QR corpus | Broad QR rank-threshold, rank-deficient solve, minimum-norm, nullspace, economy, sparse-mode, reorder, and COLAMD corpus breadth | QR and reorder owners | Sprint 150 evidence is fixture-local and intentionally excludes broad QR behavior. | Select bounded families, stable metrics, tolerance policy, generators, expected rows, and non-claims. | Focused tests, generated oracle rows, normalized report checks, and docs land for each selected family. |
| E13-R10 | QR comparison | QR comparison breadth beyond `qr_underdetermined_minnorm_2x4` | QR and comparison owners | Only one selected source-controlled dense-reference comparison study exists. | Target selection, metrics, tolerances, dependency provenance, stale-output policy, and report rows. | Comparison freshness passes for each selected fixture and docs remain fixture-local. |
| E13-R11 | Partial-SVD corpus | Broader partial-SVD repeated-spectrum, raw-vector-safe, sparse-output optimality, convergence-rate, and partial-result semantics | SVD owner | Sprint 151 evidence is fixture-local and does not cover broad subspace/vector or convergence guarantees. | Select subspace-safe metrics, fixtures, iteration-budget semantics, sparse-output expectations, and non-claims. | Focused tests, generated oracle rows, report freshness, and docs land for each selected family. |
| E13-R12 | Partial-SVD comparison | Partial-SVD normalized comparison publication | SVD and comparison owners | No selected comparison-family publication exists for partial-SVD. | Define subspace-safe/repeated-spectrum-safe comparison metrics and selected baseline helpers. | Comparison runner emits normalized rows and selected freshness gate passes with explicit non-parity wording. |
| E13-R13 | External parity | Optional NumPy/SciPy baselines and broader LAPACK, SuiteSparse, Eigen, PETSc, and Trilinos comparisons | Solver and comparison owners | Optional dependencies are deferred and no ecosystem baseline is selected as proof. | Dependency version capture, skip/defer semantics, license/provenance review, target selection, and tolerance policy. | One bounded external baseline family at a time publishes pass/defer/fail rows and preserves non-parity boundaries. |
| E13-R14 | Performance | Portable performance methodology and superiority claims | Benchmark owner | Existing benchmark/report rows are local measurements and freshness diagnostics only. | Workload suite, hardware/compiler matrix, variance policy, thresholds, regression gates, and publication policy. | Recurring reviewed performance lane publishes methodology-bound results; public docs name exact scope and caveats. |
| E13-R15 | API/docs | Generated API HTML refresh and publication | Documentation/API owner | The repository has no checked-in generated API HTML tree under `docs/api/html/`; any local output is ignored by Git and cannot be treated as source-controlled reference evidence. | Run `make docs`, triage warnings, decide generated `sparse_version.h` input policy, verify page coverage, and decide whether generated output should be committed. | Generated API output is either committed with page-coverage evidence or the absence of checked-in generated HTML remains explicitly documented; docs remain source-header-first. |
| E13-R16 | API/docs | Public-header cleanup outside the Sprint 155 selected batch | Header owners | Sprint 155 declaration-preservation proof covers only selected comment-cleanup batches. | Select next header batch, capture before/after declarations, scan claims, and define quality gate. | Normalized declaration diff is zero or API change is explicitly reviewed; `make format && make lint && make test` passes for code/header changes. |
| E13-R17 | Claims/product | Broad state-of-the-art sparse linear algebra claim | Epic/product owner | Evidence remains bounded by selected local, hosted, package, comparison, and docs scopes. | External comparison breadth, hosted platform evidence, package maturity, ABI decision, performance methodology, support policy. | Product claim audit maps every state-of-the-art dimension to recurring evidence or keeps the claim rejected. |
| E13-R18 | Runtime/backend | Typed runtime/backend control promotion and additional sentinel rows | Runtime/backend and benchmark owners | Epic 13 did not select new typed-control API/ABI scope or standalone sentinel expansion. | Select a concrete control or sentinel family with API design, ABI review, metrics, budgets, and docs. | API/tests/docs/report rows land with support-tier and non-claim wording; performance claims remain separately gated. |

## Next-Epic Candidates

The strongest next-epic candidates are the residuals that can close complete
gaps without requiring a broad product repositioning:

| Priority | Candidate | Why this is a complete-gap closure candidate |
| --- | --- | --- |
| 1 | E13-R15 generated API HTML refresh | Bounded docs/tooling task with clear pass/fail output and direct user value. |
| 2 | E13-R06 hosted promotion for selected local-only oracle/comparison rows | Converts already-defined local evidence into reviewed hosted evidence without broadening solver semantics. |
| 3 | E13-R10 QR comparison breadth, one bounded fixture family | Extends the existing comparison harness with controlled scope and clear report/freshness gates. |
| 4 | E13-R12 partial-SVD comparison publication, one bounded fixture family | Complements Sprint 151 corpus work while preserving subspace-safe metrics. |
| 5 | E13-R01 or E13-R02 Windows package parity decision | Closes a visible platform/package gap if the project wants Windows parity beyond CMake-first support. |
| 6 | E13-R16 next public-header cleanup batch | Continues adoption/API coherence with repeatable declaration-preservation gates. |

These candidates are intentionally narrower than broad ecosystem parity or
state-of-the-art positioning. Each can end with a binary decision, a reviewed
artifact, or an explicit retained non-claim.

## Long-Horizon Deferrals

The following should stay long-horizon unless product scope changes:

- E13-R03 package-manager distribution across external channels.
- E13-R04 shared-library product support across Linux, macOS, and Windows.
- E13-R05 dynamic ABI compatibility policy.
- E13-R13 broad ecosystem parity against multiple external libraries.
- E13-R14 portable performance superiority.
- E13-R17 broad state-of-the-art positioning.
- E13-R18 typed runtime/backend API promotion unless tied to a selected
  user-facing control and ABI review.

These items can be valuable, but each requires cross-cutting product,
release, support, and validation decisions that exceed a narrow closeout
cleanup sprint.

## Non-Claim Preservation

Until a residual passes its promotion gate, public docs must continue to avoid
claiming:

- unqualified state-of-the-art sparse linear algebra status;
- broad external-library or ecosystem parity;
- broad QR, SVD, or partial-SVD correctness outside selected fixtures;
- raw QR basis or raw singular-vector identity parity;
- portable performance, backend superiority, or portable iteration-count
  guarantees;
- generated report pass evidence from source-controlled rows alone;
- shared-library support, dynamic ABI compatibility, or runtime-loader
  compatibility;
- package-manager availability;
- Windows Makefile parity, Windows `pkg-config` parity, or broad Windows
  platform parity.

## Completion Criteria Check

- Residuals are actionable and include owner, blocker, prerequisite, and gate
  fields.
- Duplicate sprint residuals are consolidated into claim-oriented categories.
- Next-epic candidates prioritize complete, bounded closure.
- Long-horizon product/research items remain explicit deferrals.
- Deferred work cannot be mistaken for completed support.
