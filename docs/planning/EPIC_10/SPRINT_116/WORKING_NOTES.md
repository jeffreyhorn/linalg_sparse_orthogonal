# Sprint 116 Working Notes

## Sprint Goal

Sprint 116 closes the residual adoption-surface QA debt before final Epic 10
integration. The sprint validates adoption-facing external references, keeps
README and benchmark guidance scannable, preserves audience boundaries, and
keeps performance, platform, package, and support claims tied to reviewed
evidence.

## Starting Constraints

- Limit this sprint to adoption-surface QA, wording, documentation hygiene,
  and handoff artifacts.
- Do not add implementation, package-manager recipes, ABI support, source
  movement, helper abstractions, package install lanes, or new platform parity
  support in Sprint 116.
- Treat Sprint 115 package/platform decisions as claim guardrails, not as
  permission to advertise unreviewed support.
- Do not repeat completed Sprint 111 documentation buildout; audit and adjust
  only where adoption QA finds stale links, unclear boundaries, or
  unsupported claims.
- If documentation only changes, run `git diff --check` and a focused
  trailing-whitespace scan over touched documentation. If code, workflow,
  Make/CMake, or script surfaces change, run checks appropriate to that
  touched surface before proceeding.

## Completed Work Excluded From Sprint 116 Scope

| Completed work | Source evidence | Sprint 116 handling |
|---|---|---|
| Solver-selection, Matrix Market, README, tutorial, benchmark, and example documentation buildout | Sprint 111 retrospective and artifacts | Use as adoption surface; do not rebuild unless QA finds a stale or unsupported claim. |
| Package/platform support truth | Sprint 112 retrospective and artifacts | Use as support wording guardrail; do not add install or platform proof lanes. |
| Behavior and proof-owner closeout | Sprint 113 retrospective and artifacts | Avoid advertising unproven internals; do not move behavior ownership. |
| Proof-owner and source-boundary follow-through | Sprint 114 retrospective and artifacts | Preserve non-claims around proof-owner, source movement, and helper abstractions. |
| Package/platform residual decisions | Sprint 115 retrospective and artifacts | Preserve static-first, no dynamic ABI, no package-manager, and reviewed-platform boundaries. |

## Adoption Surface Inventory

| Surface | Files / paths | Sprint 116 use |
|---|---|---|
| Project front door | `README.md` | External reference, quality, CI boundary, performance wording, and non-claim QA. |
| Install guidance | `INSTALL.md` | Package/platform support wording guardrail and install non-claim QA. |
| Tutorial path | `docs/tutorial.md` | Adoption workflow link and audience-boundary QA. |
| Solver selection guide | `docs/solver_selection.md` | Performance wording and evidence-bounded recommendation QA. |
| Matrix Market guide | `docs/matrix_market.md` | External reference, Matrix I/O boundary, and public API non-claim QA. |
| Algorithm reference | `docs/algorithm.md` | Decide whether it remains technical background or needs adoption-facing cleanup. |
| Benchmark guide | `benchmarks/README.md` | Scanability, lane naming, report mechanics, and performance-claim QA. |
| Examples guide | `examples/README.md` | Adoption workflow and external reference QA. |

## Sprint 115 Claim Guardrails

| Guardrail | Sprint 116 handling |
|---|---|
| No reviewed Linux install CI lane beyond existing evidence | Do not imply Linux install support exceeds reviewed proof. |
| No full reviewed macOS CMake install/export parity | Keep macOS support wording bounded to reviewed coverage. |
| No Windows install-validation parity | Keep Windows package/install claims staged or absent. |
| No Windows thread/fuzz/property parity | Do not advertise Windows parity for excluded proof lanes. |
| No shared-library package support or dynamic ABI guarantee | Avoid ABI and shared-library support claims. |
| No package-manager support | Do not imply Homebrew, vcpkg, distro, or other package-manager availability. |
| No public API/install-header expansion | Avoid presenting internal or unreviewed headers as adoption contracts. |
| Sprint 114 non-package residuals deferred | Do not pull eigensolver, direct/iterative, SVD, source-list, or proof-owner movement into Sprint 116. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Adoption QA intake, duplicate fence, working notes baseline, artifact map. | Item 1 |
| 2 | External reference inventory across adoption-facing docs. | Item 1 |
| 3 | Network-check references and make focused stale-link fixes. | Item 1 |
| 4 | README quality, support-tier, and CI wording review. | Item 2 |
| 5 | README claim-boundary fixes and quality artifact. | Item 2 |
| 6 | Benchmark documentation scanability and lane inventory. | Item 3 |
| 7 | Benchmark scanability/indexing decision and focused cleanup. | Item 3 |
| 8 | Algorithm reference audience and adoption-positioning review. | Item 4 |
| 9 | Algorithm reference decision and focused cleanup if needed. | Item 4 |
| 10 | Performance wording inventory across README, solver, benchmark, and support docs. | Item 5 |
| 11 | Evidence-bounded performance wording fixes and artifact. | Item 5 |
| 12 | Adoption non-claims checklist draft and public-surface audit. | Item 6 |
| 13 | Non-claim follow-through and final checklist completion. | Item 6 |
| 14 | Documentation hygiene, final validation, and Epic 10 handoff. | Item 7 |

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation only | `git diff --check`; trailing-whitespace scan over touched docs; local relative Markdown link check when links change. |
| External links | Network check for candidate adoption-facing URLs before content changes; document unreachable or intentionally volatile links. |
| README, install, benchmark, tutorial, example, solver, Matrix Market, or algorithm docs | Focused review for audience fit, claim boundaries, and stale references. |
| Code, headers, Make/CMake, workflows, or scripts | Run the relevant build/test/check lane before proceeding; if `.c` or `.h` changes, run `make format && make lint && make test`. |

## Day 1 Notes

- Created the Sprint 116 working-notes baseline and artifact directory.
- Re-read the Sprint 116 project-plan scope and Day 1 plan.
- Re-read Sprint 111 through Sprint 115 retrospective handoff headings and
  residual-debt locations to identify adoption-facing claim risks.
- Inventoried the adoption-facing document set:
  - `README.md`
  - `INSTALL.md`
  - `docs/tutorial.md`
  - `docs/solver_selection.md`
  - `docs/matrix_market.md`
  - `docs/algorithm.md`
  - `benchmarks/README.md`
  - `examples/README.md`
- Captured Sprint 115 package/platform decisions as claim guardrails for
  README, install, benchmark, and support wording.
- Explicitly excluded implementation, package-manager recipe, ABI, source
  movement, package/platform parity, and helper-abstraction work from Sprint
  116.
- Added Day 1 adoption QA intake artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day1-adoption-qa-intake.md`.

## Day 2 Notes

- Re-read the Sprint 116 Day 2 plan and confirmed the day is inventory only:
  no adoption documentation content changes before Day 3 link validation.
- Searched the adoption-facing document set for literal external URLs:
  - `README.md`
  - `INSTALL.md`
  - `docs/tutorial.md`
  - `docs/solver_selection.md`
  - `docs/matrix_market.md`
  - `docs/algorithm.md`
  - `benchmarks/README.md`
  - `examples/README.md`
- Found three literal external URL references:
  - `https://math.nist.gov/MatrixMarket/formats.html`
  - `https://sparse.tamu.edu/`
  - a second `https://sparse.tamu.edu/` reference in the SuiteSparse usage
    section.
- Classified named external resources without literal URLs, including
  SuiteSparse, Matrix Market, BLAS, LAPACK, OpenMP, CMake, Make, GCC, Clang,
  MSVC, macOS, Windows, Linux, Homebrew, `pkg-config`, lcov, Valgrind, AMD,
  COLAMD, and METIS.
- Marked Day 3 network-check candidates and excluded local, generated,
  command-only, and intentionally non-URL references from network validation.
- Added Day 2 external-reference inventory artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day2-external-reference-inventory.md`.

## Day 3 Notes

- Re-read the Sprint 116 Day 3 plan and Day 2 candidate list.
- Network-checked the two unique external URL targets with redirects followed:
  - `https://math.nist.gov/MatrixMarket/formats.html`
  - `https://sparse.tamu.edu/`
- Confirmed both URLs returned HTTP 200 and remained at their original final
  URLs.
- Applied the SuiteSparse result to both `docs/matrix_market.md` references
  because the same `https://sparse.tamu.edu/` URL appears twice.
- Left `docs/matrix_market.md` unchanged because no stale, redirected-away,
  or unstable adoption-facing links were found.
- Added Day 3 external-reference QA artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day3-external-reference-qa.md`.

## Day 4 Notes

- Re-read the Sprint 116 Day 4 plan.
- Reviewed `README.md` for first-use adoption fit, CI boundary wording,
  install/support-tier wording, benchmark/performance evidence wording, and
  unsupported package/platform/ABI/package-manager claims.
- Compared README package/platform wording against Sprint 115 residual
  guardrails and the current `INSTALL.md` maintained install contract.
- Confirmed README already preserves the key boundaries:
  - Linux is described as the strongest reviewed source of truth, not the only
    supported platform.
  - macOS is described through reviewed Apple Clang and supplemental Homebrew
    GCC/static-first install evidence.
  - Windows is described as a reviewed CMake subset and CMake-first consumer
    story, not full install-validation parity.
  - Shared-library packaging is explicitly deferred.
  - Benchmark rows are explicitly local measurement artifacts, not portable
    performance guarantees.
- Identified one compact Day 5 edit candidate: the build section says
  `INSTALL.md` covers "package-manager detail", which can read like
  package-manager support exists even though Sprint 115 explicitly deferred
  package-manager support.
- Added Day 4 README boundary audit artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day4-readme-boundary-audit.md`.

## Day 5 Notes

- Re-read the Sprint 116 Day 5 plan and Day 4 README edit checklist.
- Applied the single Day 4 README wording fix:
  - changed "package-manager detail" to "install-support detail" in the
    build section.
- Left the README CI/support-tier paragraph unchanged because it already
  preserves Linux, macOS, Windows, reviewed/staged, and benchmark boundaries.
- Left the README installation section unchanged because it already describes
  `pkg-config`, `find_package(Sparse)`, static archive support, and deferred
  shared-library packaging correctly.
- Left README benchmark/performance wording unchanged because it already says
  benchmark rows are branch-local measurement artifacts, not portable
  performance guarantees.
- Added Day 5 README follow-through artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day5-readme-follow-through.md`.

## Day 6 Notes

- Re-read the Sprint 116 Day 6 plan.
- Reviewed `benchmarks/README.md` for live lane names, report mechanics,
  interpretation entry points, scanability, and unsupported performance
  claims.
- Compared named benchmark/report targets against `Makefile`:
  - `make tooling-build`
  - `make bench-build`
  - `make bench-fast`
  - `make bench-reorder-sprint86`
  - `make bench-canonical-report`
  - `make performance-sentinels`
  - `make large-matrix-guardrails`
  - `make bench-suitesparse`
  - `make bench-eigs`
  - `make wall-check`
- Confirmed the guide already preserves the key claim boundaries:
  - benchmark rows are local measurement artifacts;
  - `bench-canonical-report` is threshold-free reporting, not a pass/fail
    timing gate;
  - `performance-sentinels` only has the existing `wall-check` hard gate;
  - `large-matrix-guardrails` separates reviewed structural lanes from
    supplemental reports;
  - tests own regression/oracle/property guarantees.
- Identified the Day 7 edit candidate: add a compact quick-navigation/index
  table near the top of `benchmarks/README.md` so adoption-facing readers can
  jump to result interpretation, compile-only gates, workflow groups,
  maintained category split, report bundles, and specific CLI sections without
  changing benchmark commands or claims.
- Added Day 6 benchmark scanability audit artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day6-benchmark-scanability-audit.md`.

## Day 7 Notes

- Re-read the Sprint 116 Day 7 plan and Day 6 benchmark edit checklist.
- Added a compact Quick Navigation table near the top of
  `benchmarks/README.md`.
- Kept the cleanup scanability-only:
  - no benchmark commands changed;
  - no CI lane or quality gate was added;
  - no report bundle semantics changed;
  - no performance claim was widened.
- Left target names, performance caveats, workflow groups, and report bundle
  semantics unchanged because Day 6 verified they were current and
  evidence-bounded.
- Added Day 7 benchmark follow-through artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day7-benchmark-follow-through.md`.

## Day 8 Notes

- Re-read the Sprint 116 Day 8 plan.
- Reviewed `docs/algorithm.md` from an adoption QA perspective.
- Checked adoption references to `docs/algorithm.md`; README links it as
  "Algorithm Description" for data structure, LU algorithm, and complexity
  analysis, while solver selection, examples, and benchmark docs own the
  first-use adoption path.
- Confirmed the document is primarily technical background and historical
  evidence:
  - algorithms, complexity, and data structures;
  - historical sprint measurements and advisory knobs;
  - local benchmark context and regression-gate rationale;
  - internal implementation notes.
- Found no package/platform, ABI, shared-library, package-manager, or
  cross-platform support claims in `docs/algorithm.md`.
- Found no unfenced state-of-the-art claim; performance claims are tied to
  named fixtures, local measurements, or explicit caveats.
- Identified the Day 9 edit candidate: add a compact positioning note near
  the top of `docs/algorithm.md` clarifying that it is technical background,
  not the first-use adoption guide, support contract, benchmark guarantee, or
  package/platform reference.
- Added Day 8 algorithm positioning audit artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day8-algorithm-positioning-audit.md`.

## Day 9 Notes

- Re-read the Sprint 116 Day 9 plan and Day 8 algorithm edit checklist.
- Added a compact positioning note near the top of `docs/algorithm.md`.
- Kept the follow-through focused:
  - no algorithm sections were rewritten;
  - no benchmark tables or historical measurements were changed;
  - no package/platform, ABI, install, or support wording was added;
  - no implementation behavior changed.
- Left broader performance wording cleanup for Days 10-11, as planned.
- Added Day 9 algorithm positioning follow-through artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day9-algorithm-follow-through.md`.

## Day 10 Notes

- Re-read the Sprint 116 Day 10 plan.
- Audited performance wording in:
  - `README.md`;
  - `docs/solver_selection.md`;
  - `benchmarks/README.md`;
  - `docs/algorithm.md`;
  - `INSTALL.md`.
- Confirmed the main adoption-facing performance wording is already bounded:
  - README keeps benchmarks local, branch-specific, and not portable
    guarantees;
  - solver selection says benchmarks are branch-local and
    configuration-sensitive;
  - benchmark docs explicitly reject portable performance conclusions across
    machines, compilers, operating systems, BLAS/dense backends, OpenMP
    runtimes, thread counts, corpora, and build options;
  - install docs do not turn platform support into performance claims.
- Identified one Day 11 cleanup candidate in `docs/algorithm.md`: the
  preconditioner table says ILU(0) has "Good (3-1000x speedup)", which is too
  broad without local-evidence fencing.
- Left broader algorithm-doc performance tables for Day 11 no-edit review
  because the new Day 9 positioning note frames them as technical background
  and most are already tied to named fixtures or historical artifacts.
- Added Day 10 performance wording evidence audit artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day10-performance-wording-audit.md`.

## Day 11 Notes

- Re-read the Sprint 116 Day 11 plan and Day 10 cleanup checklist.
- Applied the single performance wording cleanup in `docs/algorithm.md`:
  - replaced ILU(0) "Good (3-1000x speedup)" with
    "Workload-dependent acceleration; benchmark locally".
- Left README, solver-selection, benchmark, and install wording unchanged
  because Day 10 found those surfaces already evidence-bounded.
- Left broader algorithm-doc performance tables unchanged because they are
  now fenced by the Day 9 positioning note and generally tied to named
  fixtures, benchmark artifacts, or local-measurement context.
- Added Day 11 performance wording follow-through artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day11-performance-wording-follow-through.md`.

## Day 12 Notes

- Re-read the Sprint 116 Day 12 plan.
- Audited adoption-facing docs for unreviewed-surface claims:
  - `README.md`;
  - `INSTALL.md`;
  - `docs/tutorial.md`;
  - `docs/solver_selection.md`;
  - `docs/matrix_market.md`;
  - `docs/algorithm.md`;
  - `benchmarks/README.md`;
  - `examples/README.md`.
- Confirmed Matrix Market non-claims are explicit in `docs/matrix_market.md`,
  `docs/solver_selection.md`, and `examples/README.md`: load/save functions
  are public, but there is no separate public Matrix I/O module or public
  builder API.
- Confirmed package/platform non-claims are explicit in README and
  `INSTALL.md`: static-first install surface, shared-library packaging
  deferred, no dynamic ABI promise, Windows CMake-first reviewed subset, no
  separate Windows install-validation lane, and no full macOS install/export
  parity.
- Confirmed benchmark/performance non-claims are explicit in README,
  `benchmarks/README.md`, and `docs/algorithm.md`: benchmark rows are local
  measurement artifacts, not portable performance guarantees.
- Confirmed proof-owner/internal-helper wording is not promoted into the
  adoption path; `docs/algorithm.md` now explicitly frames itself as
  technical background.
- Added Day 12 adoption non-claims checklist artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day12-adoption-non-claims-checklist.md`.

## Day 13 Notes

- Re-read the Sprint 116 Day 13 plan and Day 12 cleanup list.
- Rechecked adoption-facing claim boundaries across README, INSTALL,
  solver-selection, Matrix Market, tutorial, examples, benchmark, and
  algorithm docs.
- Confirmed the Day 12 no-edit path still holds:
  - Matrix Market module/builder API language is fenced;
  - package/platform and install-validation boundaries are fenced;
  - shared-library and dynamic ABI support are not claimed;
  - package-manager support is not claimed;
  - benchmark/performance language remains local and evidence-bounded;
  - algorithm/proof-owner/internal-helper detail is not promoted into
    first-use adoption guidance.
- No Day 13 adoption wording fixes were required.
- Added Day 13 claim-guardrail follow-through artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day13-claim-guardrail-follow-through.md`.

## Day 14 Notes

- Re-read the Sprint 116 Day 14 plan and reviewed all Sprint 116 artifacts.
- Confirmed Sprint 116 produced the planned adoption QA artifacts:
  - adoption intake;
  - external-reference inventory and QA;
  - README boundary audit and follow-through;
  - benchmark scanability audit and follow-through;
  - algorithm positioning audit and follow-through;
  - performance wording audit and follow-through;
  - adoption non-claims checklist;
  - claim-guardrail follow-through;
  - final validation and handoff.
- Confirmed tracked documentation edits are limited to:
  - `README.md`;
  - `benchmarks/README.md`;
  - `docs/algorithm.md`.
- Confirmed no `.c`, `.h`, Make/CMake, workflow, package metadata, script, or
  implementation files changed.
- Reconfirmed old ambiguous wording remains absent:
  - `package-manager detail`;
  - `3-1000x` / `3-1000x speedup`.
- Added Day 14 validation and handoff artifact:
  `docs/planning/EPIC_10/SPRINT_116/artifacts/day14-validation-handoff.md`.
