# Sprint 155 Working Notes

## Goal

Sprint 155 closes the remaining adoption and documentation gap by aligning the
tutorial, selected public headers, and API reference surfaces around the
current earned claims without widening package, ABI, platform, performance, or
state-of-the-art claims.

## Starting Evidence

- Sprint 145 created the first-use adoption ladder across `README.md`,
  `INSTALL.md`, `examples/README.md`, `docs/cookbook.md`,
  `docs/solver_selection.md`, and selected high-impact public headers.
- Sprint 146 published Epic 12 closeout residuals `R8` and `R9` for tutorial
  alignment and broader public-header cleanup.
- Sprint 148 through Sprint 154 completed Epic 13 platform, install, QR,
  partial-SVD, generated-report, package/ABI, and first narrow comparison
  follow-through work.
- Sprint 154 handed off a narrow `qr-minnorm` external-process comparison
  lane and explicitly limited it to fixture-local evidence.
- The installed public header surface currently has `19` headers/templates
  under `include/`, including `include/sparse_version.h.in`.
- The generated API reference currently lives under `docs/api/html/`.

## Item-To-Day Owner Map

| Sprint 155 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: Tutorial Audit | Days 1-2 | Day 1 inventories sources and claim boundaries; Day 2 performs the source-to-source tutorial audit. |
| Item 2: Tutorial Alignment | Days 3-5 | Day 3 designs the target flow; Days 4-5 rewrite and reconcile tutorial sections. |
| Item 3: Header Cleanup Selection | Day 6 | Day 6 selects the high-impact public-header batch and defers the rest explicitly. |
| Item 4: Header Cleanup Batch | Days 7-9 | Day 7 defines cleanup guardrails; Days 8-9 clean selected headers and collect preservation evidence. |
| Item 5: API Reference Plan | Days 10-11 | Day 10 designs API reference publication guidance; Day 11 implements docs/index updates. |
| Item 6: Declaration Preservation | Day 12 | Day 12 reruns preservation scans and reconciles tutorial/header/API surfaces. |
| Item 7: Validation And Closeout | Days 13-14 | Day 13 runs required gates; Day 14 packages residuals and Sprint 156 handoff. |

## Stop Conditions

- Tutorial or API reference text describes the library as unqualified
  state-of-the-art sparse linear algebra.
- Tutorial, header, or API text turns fixture-local QR, partial-SVD, or
  `qr-minnorm` comparison evidence into broad external-library parity.
- Header cleanup changes declarations, signatures, typedefs, enums, macros,
  struct fields, installed header names, include guards, or exported symbols.
- Public comments imply shared-library packaging, dynamic ABI compatibility,
  runtime-loader behavior, package-manager distribution, Windows Makefile
  parity, Windows `pkg-config` parity, or broad Windows parity.
- Generated `docs/api/html/` content is treated as fresh release evidence
  without an explicit regeneration and publication policy.
- `.c` or public `.h` changes land without the required
  `make format && make lint && make test` quality gate.
- Declaration-preservation checks are skipped after public header edits.
- Optional comparison, benchmark, sentinel, coverage, dead-code, or report rows
  are counted as proof outside their documented support tier.

## Daily Log

### Day 1: Sprint Intake And Documentation Baseline

- Re-read the Sprint 155 section of
  `docs/planning/EPIC_13/PROJECT_PLAN.md`.
- Reviewed Sprint 145 and Sprint 146 residual/adoption closeouts from
  `docs/planning/EPIC_12/`.
- Reviewed Sprint 154 Day 14 handoff for the narrow QR minimum-norm comparison
  boundary that Sprint 155 must preserve.
- Created the Sprint 155 artifact directory.
- Created
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day1-documentation-baseline.md`.
- Inventoried the current documentation owner surfaces:
  - `README.md` owns the short first-use front door and support-tier summary;
  - `INSTALL.md` owns static-first install, downstream consumer, package, and
    platform-support boundaries;
  - `examples/README.md` owns runnable first-use example routing;
  - `docs/cookbook.md` owns data-first recipes;
  - `docs/solver_selection.md` owns solver-family decision guidance;
  - `docs/tutorial.md` is the residual fuller learning path targeted by Days
    2 through 5;
  - `docs/maintainer_guide.md` owns maintainer policy, report interpretation,
    packaging, and validation boundaries;
  - `docs/api/html/` contains the current generated Doxygen output.
- Inventoried the public header cleanup surface:
  - `19` installed public headers/templates under `include/`;
  - Sprint 145 already cleaned `include/sparse_matrix.h`,
    `include/sparse_iterative.h`, `include/sparse_qr.h`, and
    `include/sparse_svd.h`;
  - Day 6 should select the next highest-impact batch without declaration
    drift.
- Recorded earned claims and non-claim boundaries for tutorial, header, and
  API-reference work.
- Day 2 handoff: audit `docs/tutorial.md` against README, INSTALL, examples,
  cookbook, solver-selection, maintainer/report guidance, public headers, and
  API reference links.

### Day 2: Tutorial Audit

- Created
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day2-tutorial-audit.md`.
- Audited `docs/tutorial.md` against:
  - `README.md`;
  - `INSTALL.md`;
  - `examples/README.md`;
  - `docs/cookbook.md`;
  - `docs/solver_selection.md`;
  - `docs/maintainer_guide.md`;
  - `docs/matrix_market.md`;
  - `benchmarks/README.md`;
  - the public headers under `include/`;
  - the generated API reference surface under `docs/api/html/`;
  - Sprint 154's `qr-minnorm` comparison handoff.
- Recorded the source-to-source comparison matrix for tutorial opening,
  documentation map, build/link guidance, matrix construction, first solve,
  direct solvers, iterative solvers, preconditioning, SVD/partial-SVD,
  eigensolver, diagnostics, reports, install, support tiers, and API reference
  handoffs.
- Identified stale content:
  - partial-SVD evidence text only names the Sprint 140 clustered/repeated 8x6
    lane and omits the Sprint 151 rank-deficient projector, sparse low-rank
    output, and fail-closed recovery rows;
  - the include list is narrower than the current public API surface;
  - the tutorial opening does not mirror the current first-use ladder exactly.
- Identified missing content:
  - explicit `example_basic_solve` first-solve anchor;
  - data-first CSR/CSC/Matrix Market route tied to examples and cookbook;
  - workflow-local diagnostics ladder;
  - symmetric eigensolver tutorial handoff;
  - MINRES/IC(0) handoff through `example_ic_minres`;
  - concise install/downstream and API-reference handoffs.
- Identified claim-risky wording:
  - "ILU preconditioning dramatically reduces iteration counts" should become
    local/workload-dependent acceleration guidance;
  - partial-SVD evidence updates must stay fixture-local;
  - Sprint 154 comparison evidence must not become broad external-library
    parity.
- Day 3 handoff: design a target tutorial flow around build, first solve,
  data input, solver choice, diagnostics, install, and advanced controls while
  delegating install/package/platform, report/comparison, and exact API
  declaration detail to their owner surfaces.

### Day 3: Tutorial Flow Design

- Created
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day3-tutorial-flow-design.md`.
- Designed the target tutorial flow around:
  - README remains the short front door;
  - examples remain runnable anchors;
  - data input comes before solver depth;
  - solver choice stays problem-shaped and aligned with
    `docs/solver_selection.md`;
  - diagnostics stay workflow-local;
  - install/package detail stays delegated to `INSTALL.md`;
  - exact declarations, option structs, result fields, and ownership contracts
    stay delegated to public headers and the API reference plan.
- Proposed target tutorial sections:
  - Getting Started;
  - Documentation Map;
  - Build-Tree Setup And First Solve;
  - Link Or Install;
  - Start From Your Matrix;
  - Choose The Solver Workflow;
  - Direct Solver Walkthrough;
  - Iterative And Preconditioned Workflows;
  - SVD And Low-Rank Workflows;
  - Symmetric Eigensolver Workflows;
  - Matrix-Free Interface;
  - Diagnostics Handoff;
  - Advanced Controls, Reports, API Reference, And Install.
- Mapped target tutorial sections to owner docs including README, INSTALL,
  examples, cookbook, solver-selection, Matrix Market, benchmarks, maintainer
  guide, public headers, and generated API reference output.
- Created the example/cookbook cross-link plan for `example_basic_solve`,
  `example_compressed_input`, `example_matrix_market`, `example_analysis`,
  `example_least_squares`, `example_minnorm`, `example_ic_minres`,
  `example_svd_lowrank`, `example_condition`, `example_eigs`, and
  `example_matrix_free`.
- Recorded advanced-control boundaries:
  - runtime/backend options are workflow controls, not ABI or performance
    claims;
  - benchmarks and reports are local measurement/evidence surfaces;
  - report freshness commands belong in maintainer or advanced-report
    contexts;
  - generated Doxygen output needs a publication plan before being treated as
    current reference evidence.
- Day 4 handoff: rewrite the tutorial core path only: opening, documentation
  map, build-tree first solve, local link/install handoff, data-input routing,
  first-solve/example cross-links, and compact solver workflow table.

### Day 4: Tutorial Rewrite Batch 1

- Created
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day4-tutorial-core-rewrite.md`.
- Updated `docs/tutorial.md` core learning path:
  - reframed the opening around build, first maintained solve, data input,
    solver choice, diagnostics, install, and advanced controls;
  - revised the documentation map so it names owner surfaces more directly;
  - added the build-tree first-solve sequence `make`, `make examples`, and
    `./build/example_basic_solve`;
  - clarified `make examples-build` as compile-only example validation and
    `make test` as code-change validation rather than the first learning step;
  - kept the local build-tree link command while delegating installed
    consumers to `INSTALL.md`;
  - expanded include guidance for LDLT, IC, and symmetric eigensolver
    workflows while keeping exact declarations delegated to public headers and
    API reference;
  - moved data-input routing before solver depth and linked CSR/CSC and Matrix
    Market users to examples and cookbook surfaces;
  - added a compact solver workflow table with runnable anchors for direct,
    QR, iterative, eigensolver, SVD, and matrix-free routes.
- Preserved Day 4 claim boundaries:
  - no shared-library, dynamic ABI, package-manager, Windows parity,
    performance, broad QR/SVD/partial-SVD, external-library parity,
    generated-report freshness, or state-of-the-art claim was introduced.
- No `.c` or public `.h` files changed.
- Day 5 handoff: finish tutorial alignment by rewriting the preconditioning
  wording, refreshing or delegating partial-SVD evidence, adding workflow-local
  diagnostics, and adding compact advanced-control/report/API-reference
  handoffs.

### Day 5: Tutorial Rewrite Batch 2

- Created
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day5-tutorial-alignment-summary.md`.
- Updated `docs/tutorial.md` to finish the tutorial alignment pass:
  - replaced overbroad ILU preconditioning language with local,
    workload-dependent diagnostic wording;
  - refreshed partial-SVD evidence by naming the clustered/repeated 8x6 lane
    plus Sprint 151 rank-deficient projector, sparse low-rank output, and
    fail-closed recovery rows while delegating detailed boundaries to
    `docs/solver_selection.md`;
  - added a compact symmetric eigensolver section for `sparse_eigs_sym(...)`,
    AUTO backend starting point, `example_eigs`, and eigensolver diagnostics;
  - positioned matrix-free workflows as advanced iterative paths after
    standard matrix-shell workflows;
  - added workflow-local diagnostics guidance for construction, Matrix
    Market, direct, repeated direct, iterative, QR, SVD/partial-SVD,
    eigensolver, benchmark, and report contexts;
  - added advanced handoffs for runtime/backend controls, benchmarks/reports,
    install/downstream consumers, public headers/API reference, and maintainer
    evidence/report/package/support-tier interpretation.
- Rechecked tutorial wording against non-claims:
  - no shared-library, dynamic ABI, package-manager, Windows parity,
    performance, broad QR/SVD/partial-SVD/eigensolver, external-library
    parity, generated-report freshness, or state-of-the-art claim was
    introduced.
- No `.c` or public `.h` files changed.
- Day 6 handoff: select a high-impact public-header cleanup batch using the
  tutorial alignment results, with special attention to LDLT, eigensolver,
  IC/preconditioning, analysis lifecycle, Matrix Market/I/O, and public error
  contracts.

### Day 6: Header Cleanup Selection

- Created
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day6-header-cleanup-selection.md`.
- Inventoried all `19` installed public headers/templates under `include/`.
- Reviewed the Sprint 145 public-header cleanup design and validation pattern.
- Compared the Day 5 tutorial alignment results against the current public
  header surface.
- Found no dedicated single-purpose declaration-preservation script; recorded
  a Day 7 requirement to formalize repeatable git-diff/text-scan checks before
  public headers are edited.
- Scored candidate headers by user impact, cross-doc value, cleanup need, and
  low-risk feasibility.
- Selected the Sprint 155 public-header cleanup batch:
  - `include/sparse_ldlt.h`;
  - `include/sparse_ic.h`;
  - `include/sparse_eigs.h`;
  - `include/sparse_analysis.h`.
- Recommended Day 8 cleanup tranche:
  - `include/sparse_ldlt.h`;
  - `include/sparse_ic.h`.
- Recommended Day 9 cleanup tranche:
  - `include/sparse_eigs.h`;
  - `include/sparse_analysis.h`.
- Deferred Sprint 145-cleaned headers (`sparse_matrix.h`,
  `sparse_iterative.h`, `sparse_qr.h`, `sparse_svd.h`) unless later
  reconciliation finds a small targeted issue.
- Recorded preservation constraints for declarations, signatures, typedefs,
  enum values, macros, struct fields, include guards, ownership/default/error
  semantics, mutation/non-mutation guarantees, and non-claim boundaries.
- Day 7 handoff: write the cleanup contract and exact declaration-preservation
  command log before editing selected headers.

### Day 7: Header Cleanup Contract

- Created
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day7-header-cleanup-contract.md`.
- Defined allowed public-header cleanup edits:
  - comment rewrites;
  - shorter tutorial-scale examples;
  - replacement of sprint-history prose with current behavior;
  - ownership, lifetime, output-buffer, NULL/error, mutation, precondition,
    default, result-field, backend, and non-claim clarification;
  - concise cross-links to owner docs.
- Defined disallowed cleanup edits:
  - public declaration/signature/order changes;
  - typedef, enum, struct-field, macro, include-guard, installed-header, or
    exported-name changes;
  - removal of defaults, ownership/freeing requirements, error returns,
    mutation guarantees, or safety preconditions;
  - unsupported package, ABI, platform, performance, external-parity, or
    state-of-the-art wording.
- Defined declaration-preservation command plan for Day 8 and Day 9 before/after
  declaration-like captures.
- Defined focused claim scans and required post-header-edit gate:
  `make format && make lint && make test`.
- Added error-contract wording guidance for public header comments.
- Added maintainer checklist for future public-header cleanup.
- Updated `docs/maintainer_guide.md` with reusable public-header cleanup
  guidance under Documentation Ownership Rules.
- Day 8 handoff: capture declaration-like baselines, edit only
  `include/sparse_ldlt.h` and `include/sparse_ic.h`, keep edits comment-only,
  run the Day 7 declaration/claim scans, and run
  `make format && make lint && make test`.

### Day 8: LDLT And IC Header Cleanup

- Captured the Day 8 declaration-like baseline in
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day8-header-declarations-before.txt`.
- Cleaned only Doxygen/comment text in:
  - `include/sparse_ldlt.h`;
  - `include/sparse_ic.h`.
- Kept all public declarations, signatures, typedefs, enum values, struct
  fields, macros, include guards, installed header names, and exported names
  unchanged.
- Clarified LDLT call-site contracts for owned-factor lifecycle,
  `sparse_analysis.h` repeated-run handoff, backend selection, CSC telemetry,
  progress cancellation, reset-on-entry, free-after-success, and propagated
  error returns.
- Clarified IC call-site contracts for SPD scope, `sparse_ilu_t` storage,
  `sparse_ic_precond()` usage, reset-on-entry, identity row/column state,
  `r`/`z` aliasing, callback NULL handling, dimension mismatch, and NULL/zeroed
  free behavior.
- Captured the Day 8 after declaration scan in
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day8-header-declarations-after.txt`.
- Recorded normalized declaration preservation in
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day8-header-declarations-normalized-diff.txt`;
  the normalized diff is empty.
- Ran the unsupported-claim scan from the Day 7 contract; it returned no
  matches.
- Ran `git diff --check`; it passed.
- Ran `make format && make lint && make test`; it passed and ended with
  `All tests passed.`
- Wrote the Day 8 summary artifact:
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day8-header-cleanup-summary.md`.
- Day 9 handoff: repeat the Day 7 contract for `include/sparse_eigs.h` and
  `include/sparse_analysis.h`, including before/after declaration captures,
  claim scan, `git diff --check`, and `make format && make lint && make test`.
- Day 8 handoff: capture declaration baseline, edit only
  `include/sparse_ldlt.h` and `include/sparse_ic.h`, run declaration diff,
  claim scan, `git diff --check`, and the full C quality gate.

### Day 9: EIGS And Analysis Header Cleanup

- Captured the Day 9 declaration-like baseline in
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day9-header-declarations-before.txt`.
- Cleaned only Doxygen/comment text in:
  - `include/sparse_eigs.h`;
  - `include/sparse_analysis.h`.
- Kept all public declarations, signatures, typedefs, enum values, struct
  fields, macros, include guards, installed header names, and exported names
  unchanged.
- Shortened the eigensolver header's convergence, ownership, options/result,
  LOBPCG, preconditioner, soft-locking, and progress-callback prose while
  preserving residual, buffer-lifetime, backend, and cancellation boundaries.
- Shortened the repeated-run analysis header introduction and clarified the
  analyze/factor/refactor/free lifecycle, one-shot API handoff, same-pattern
  refactor scope, and ownership/freeing requirements.
- Captured the Day 9 after declaration scan in
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day9-header-declarations-after.txt`.
- Recorded normalized declaration preservation in
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day9-header-declarations-normalized-diff.txt`;
  the normalized diff is empty.
- Ran the unsupported-claim scan from the Day 7 contract; it returned no
  matches.
- Ran `git diff --check`; it passed before and after the full gate.
- Ran `make format && make lint && make test`; it passed and ended with
  `All tests passed.`
- Wrote the Day 9 summary artifact:
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day9-header-cleanup-summary.md`.
- Day 10 handoff: use the cleaned selected headers as input for API reference
  baseline and publication planning, preserving the static-first/package/ABI
  claim boundaries.

### Day 10: API Reference Baseline And Publication Plan

- Inventoried the current API reference surfaces:
  - `Doxyfile`;
  - `Makefile` target `docs`;
  - generated HTML under `docs/api/html/`;
  - `README.md`;
  - `docs/tutorial.md`;
  - `docs/maintainer_guide.md`.
- Confirmed local Doxygen availability with `doxygen --version`; the local
  version is `1.16.1`.
- Compared checked-in public headers under `include/` with generated HTML pages
  under `docs/api/html/`.
- Found generated HTML coverage gaps for:
  - `include/sparse_analysis.h`;
  - `include/sparse_eigs.h`;
  - `include/sparse_ic.h`;
  - `include/sparse_ldlt.h`;
  - `include/sparse_lu_csr.h`.
- Recorded that the installed generated version header
  `build/include/sparse_version.h` is outside the current `Doxyfile` input set
  because `Doxyfile` reads checked-in `include/*.h`.
- Decided Sprint 155 should add both:
  - direct user-facing API reference guidance;
  - maintainer-facing generated-reference publication and freshness rules.
- Deferred generated HTML refresh to Day 11 implementation review instead of
  creating a large generated-output diff during Day 10 planning.
- Wrote the Day 10 plan artifact:
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day10-api-reference-publication-plan.md`.
- Day 11 handoff: add a compact API reference entry point, link it from the
  adoption docs, add generated-reference publication guidance to the maintainer
  guide, and refresh generated HTML only if the generated diff is reviewable.

### Day 11: API Reference Guidance Implementation

- Added `docs/api_reference.md` as the compact user-facing API reference entry
  point.
- Linked `docs/api_reference.md` from:
  - `README.md`;
  - `docs/tutorial.md`;
  - `docs/cookbook.md`.
- Added generated API-reference publication and freshness guidance to
  `docs/maintainer_guide.md`.
- Kept generated HTML refresh deferred because Day 10 found the current
  checked-in `docs/api/html/` surface is partial and a refresh would create a
  large generated-output diff.
- Preserved claim boundaries around dynamic ABI, shared-library support,
  package-manager distribution, broad Windows parity, external-library parity,
  portable performance, and state-of-the-art coverage.
- Ran `git diff --check`; it passed.
- Checked new link targets for `docs/api_reference.md`, `docs/api/html/index.html`,
  and `include/`; they exist.
- Ran the unsupported-claim scan across the touched docs; matches were explicit
  non-claim wording only.
- Wrote the Day 11 implementation artifact:
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day11-api-reference-guidance-implementation.md`.
- Day 12 handoff: reconcile tutorial, public-header, and API-reference
  references; run link/claim scans; preserve declaration evidence from Days
  8-9; and decide whether generated-reference refresh remains deferred.

### Day 12: Preservation And Cross-Doc Reconciliation

- Generated the final current declaration scan for the edited public-header
  batch:
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day12-header-declarations-current.txt`.
- Compared the Day 8 full pre-cleanup declaration baseline against the current
  edited headers and recorded the normalized diff in
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day12-header-declarations-normalized-diff.txt`.
- Confirmed Day 8, Day 9, and Day 12 normalized declaration diffs are all
  empty.
- Checked installed-header expectations against CMake install behavior:
  checked-in `include/*.h` headers remain installed under
  `${CMAKE_INSTALL_INCLUDEDIR}/sparse`, while generated `sparse_version.h`
  remains installed separately from the build include directory.
- Reconciled API reference links across `README.md`, `docs/tutorial.md`,
  `docs/cookbook.md`, `docs/api_reference.md`, and
  `docs/maintainer_guide.md`.
- Replaced one stale tutorial phrase with a direct `api_reference.md` link.
- Confirmed the generated API HTML inventory remains partial for the current
  checked-in header set: `18` checked-in public headers, `13` generated
  `sparse__*_8h.html` pages, and missing generated pages for analysis, eigs,
  IC, LDLT, and LU CSR.
- Kept generated HTML refresh deferred and documented the partial/freshness
  boundary rather than producing a large generated-output diff.
- Updated `docs/maintainer_guide.md` so public-header/API-comment cleanup also
  checks `docs/api_reference.md` and generated HTML freshness.
- Ran the unsupported-claim scan across touched docs and edited headers;
  matches were explicit non-claim wording only.
- Ran `git diff --check`; it passed.
- Checked link targets for `docs/api_reference.md`, `docs/api/html/index.html`,
  and `include/`; they exist.
- Confirmed the stale phrase scan for `API reference surface` and
  `generated API reference` returned no matches.
- Wrote the Day 12 preservation and reconciliation artifact:
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day12-preservation-and-reconciliation.md`.
- Day 13 handoff: run integrated validation, including diff check, link-target
  checks, unsupported-claim scan, declaration-diff zero-size checks, and any
  broader docs/generated-reference gate selected for closeout.

### Day 13: Integrated Validation

- Ran the full required quality gate because the branch includes public header
  edits:
  `make format && make lint && make test`.
- The full gate passed and ended with `All tests passed.`
- `make lint` completed strict compile, `clang-tidy`, and `cppcheck`.
- `make test` completed the maintained local test suite.
- Refreshed the post-format declaration scan in
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day12-header-declarations-current.txt`.
- Refreshed the Day 12 normalized declaration diff in
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day12-header-declarations-normalized-diff.txt`.
- Confirmed Day 8, Day 9, and Day 12 normalized declaration diffs are all
  `0` bytes.
- Ran `git diff --check`; it passed.
- Checked link targets for `docs/api_reference.md`, `docs/api/html/index.html`,
  and `include/`; they exist.
- Confirmed the stale phrase scan for `API reference surface` and
  `generated API reference` returned no matches.
- Ran the unsupported-claim scan across README, tutorial, cookbook,
  maintainer guide, API reference, and the edited headers; matches were
  explicit non-claim wording only.
- Did not refresh `docs/api/html/`; the generated API reference remains an
  explicitly documented partial/freshness-governed surface.
- Wrote the Day 13 integrated validation artifact:
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day13-integrated-validation.md`.
- Day 14 handoff: package the Sprint 155 closeout, including tutorial
  alignment, header cleanup, API reference guidance, declaration-preservation
  evidence, full quality-gate success, and deferred generated HTML refresh.

### Day 14: Closeout And Sprint 156 Handoff

- Reviewed all Sprint 155 artifacts for completeness and consistency.
- Mapped Sprint 155 deliverables back to project-plan items:
  - tutorial audit;
  - tutorial alignment;
  - header cleanup selection;
  - selected public-header cleanup batch;
  - API reference plan;
  - declaration preservation;
  - validation and closeout.
- Summarized delivered user-facing improvements:
  - aligned tutorial;
  - `docs/api_reference.md` entry point;
  - README/tutorial/cookbook API-reference links;
  - maintainer generated-reference freshness policy.
- Summarized selected public-header cleanup for:
  - `include/sparse_ldlt.h`;
  - `include/sparse_ic.h`;
  - `include/sparse_eigs.h`;
  - `include/sparse_analysis.h`.
- Recorded validation evidence from Day 13:
  `make format && make lint && make test` passed and ended with
  `All tests passed.`
- Recorded residual and deferred work:
  - generated API HTML refresh;
  - missing generated API pages for analysis, eigs, IC, LDLT, and LU CSR;
  - generated installed `sparse_version.h` Doxygen inclusion decision;
  - remaining public-header cleanup outside selected batch;
  - final Epic 13-wide claim recalibration.
- Preserved explicit non-claims for dynamic ABI compatibility,
  shared-library support, package-manager distribution, broad Windows parity,
  external-library parity, portable performance, generated-reference
  completeness, and state-of-the-art sparse linear algebra status.
- Wrote the Day 14 closeout artifact:
  `docs/planning/EPIC_13/SPRINT_155/artifacts/day14-closeout-sprint156-handoff.md`.
- Sprint 156 handoff: inventory final Epic 13 evidence, run the strongest
  feasible validation baseline, reconcile platform/CI tiers, audit public
  claims/non-claims, publish residual promotion gates, and write the Epic 13
  retrospective.
