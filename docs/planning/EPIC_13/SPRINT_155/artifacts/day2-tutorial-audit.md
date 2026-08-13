# Sprint 155 Day 2 Tutorial Audit

## Purpose

Day 2 audits `docs/tutorial.md` against the current adoption, examples,
package, solver-selection, diagnostics, report, and support-tier owner
surfaces. The audit is intentionally pre-implementation: it identifies what
Days 3 through 5 should restructure or rewrite before any broad tutorial churn
lands.

## Source-To-Source Comparison Matrix

| Tutorial Area | Source Of Truth | Current State | Finding | Rewrite Priority |
| --- | --- | --- | --- | --- |
| Getting started flow | `README.md#start-here`, `examples/README.md#start-here`, `docs/cookbook.md#first-use-ladder` | Tutorial opens with create/load, choose workflow, validate output, then widen. | Direction is broadly correct, but it does not mirror the newer first-use ladder exactly: build, first maintained solve, data-first route, solver choice, diagnostics, install, advanced controls. | High |
| Documentation map | README adoption map, maintainer ownership rules | Tutorial links to solver-selection, cookbook, examples, install, benchmarks, algorithm docs, and maintainer guide. | Useful map, but it does not explain which document owns which claim. Day 3 should preserve the map and make ownership clearer without adding maintainer-policy blocks. | Medium |
| Build guidance | `README.md#building`, `INSTALL.md#start-here`, `INSTALL.md#quick-start-makefile` | Tutorial shows `make`, `make test`, and `make examples`. | Correct for build-tree use, but it skips the current first-use order from README and examples and does not mention `make examples-build` as the compile-only route. | Medium |
| Link guidance | README quick start, `INSTALL.md#using-via-pkg-config`, `INSTALL.md#using-from-a-cmake-project` | Tutorial shows build-tree `cc -O2 -Iinclude ... -Lbuild -lsparse_lu_ortho -lm`. | Correct for local build-tree snippets, but it should explicitly defer installed downstream consumers to `INSTALL.md` and not duplicate static package policy. | Medium |
| Header include list | public headers under `include/`, generated Doxygen output | Tutorial lists a subset of commonly used headers. | Missing current families such as `sparse_analysis.h`, `sparse_ldlt.h`, `sparse_eigs.h`, `sparse_ic.h`, `sparse_bidiag.h`, and Matrix Market/I/O context. Day 3 should decide whether to keep a compact list or point to API reference instead. | Medium |
| Matrix construction | `docs/cookbook.md#start-from-your-data`, `docs/solver_selection.md#start-from-your-matrix`, `docs/matrix_market.md` | Tutorial covers hand-written construction, CSR/CSC handoff, Matrix Market load, and matrix operations. | Mostly aligned. It should be moved earlier into the first-use route and connected more directly to `example_compressed_input` and Matrix Market example routing. | Medium |
| First solve | `examples/README.md#one-shot-direct-example_basic_solve`, README quick start | Tutorial jumps from matrix construction into LU/Cholesky/LDL/QR examples. | Missing an explicit "run the first maintained solve" step before API family detail. Day 4 should add a first-solve anchor that points to `example_basic_solve`. | High |
| Direct solver choice | `docs/solver_selection.md#direct-solvers`, examples README | Tutorial documents LU, Cholesky, LDLT, and QR. | Generally aligned, but it lacks the concise direct-solver table now used by solver-selection and cookbook. It also uses `LDL^T` prose but does not include an LDLT code path, leaving a gap for symmetric indefinite first-use. | Medium |
| QR evidence and comparison | `docs/solver_selection.md#qr-evidence-boundary`, Sprint 154 handoff, maintainer comparison freshness gate | Tutorial explains QR rank and minimum-norm solve. | It does not mention the Sprint 154 `qr-minnorm` comparison lane, which is acceptable for first-use, but if QR evidence is mentioned later it must preserve the fixture-local boundary and avoid external parity. | Low |
| Iterative solvers | `docs/solver_selection.md#iterative-solvers`, examples README | Tutorial covers CG, GMRES, and preconditioning. | Basic solver text is aligned, but it omits MINRES as a first-class repeated-run supported handle and does not point to `example_ic_minres`. | Medium |
| Preconditioning language | solver-selection diagnostics and benchmark caveats | Tutorial says "ILU preconditioning dramatically reduces iteration counts." | This is overbroad. It should be rewritten as local/workload-dependent acceleration guidance and tied to solver assumptions and diagnostics. | High |
| SVD and partial SVD | `docs/solver_selection.md#svd-and-low-rank-workflows`, Sprint 151 closeout | Tutorial covers full SVD, partial SVD, condition, rank, pseudoinverse, and low-rank APIs. | Partial-SVD evidence text is stale: it names only the generated 8x6 clustered/repeated fixture and omits Sprint 151 rank-deficient projector, sparse low-rank output, and fail-closed recovery rows. | High |
| Eigensolver workflows | README current capabilities, solver-selection eigensolver section, examples README | Tutorial has no symmetric eigensolver section. | Missing user-facing tutorial coverage for `sparse_eigs_sym(...)`, AUTO backend starting point, and `example_eigs` handoff. | High |
| Matrix-free interface | examples README, solver-selection | Tutorial includes matrix-free CG and GMRES snippets. | Useful, but should be positioned as an advanced iterative path after basic iterative diagnostics and linked to `example_matrix_free`. | Medium |
| Diagnostics | examples diagnostics handoff, solver-selection diagnostics handoff | Tutorial has a final generic error-code table and scattered residual text. | Missing the current workflow-local diagnostics ladder: construction, Matrix Market errno, direct residuals, iterative convergence, QR rank/residuals, SVD triplet/convergence, eigensolver Ritz residuals, benchmark/report context. | High |
| Reports and generated freshness | `docs/maintainer_guide.md#normalized-report-index-workflow`, `benchmarks/README.md#report-index-handoff` | Tutorial links to benchmarks for report interpretation but has no report/freshness section. | This is mostly correct for first-use. Day 5 should add a short advanced-report handoff only, not maintainer-level report policy. | Low |
| Install and package surface | `INSTALL.md` | Tutorial says use `INSTALL.md` for install/downstream workflows. | Correct boundary. Day 5 should add a concise install handoff after first workflow works, while preserving static-first and non-package-manager boundaries through links rather than duplicated policy. | Medium |
| Support tiers and platform claims | `INSTALL.md#supported-platforms`, maintainer guide | Tutorial does not discuss platform tiers. | Acceptable for first-use, but any build/install rewrite must avoid broad Windows parity, package-manager, shared-library, dynamic ABI, or runtime-loader claims. | Medium |
| API reference | `docs/api/html/`, `Makefile` docs target, maintainer ownership rules | Tutorial does not point directly to generated Doxygen output. | Missing API reference handoff. Day 10-11 should decide the final API-reference publication plan; Day 5 can reserve a small "when you need exact declarations" link. | Medium |

## Stale, Missing, Or Overbroad Content

### Stale

1. `docs/tutorial.md` partial-SVD evidence text still describes only the
   Sprint 140 clustered/repeated 8x6 lane. It should include the Sprint 151
   rank-deficient rectangular projector, sparse low-rank output, and
   fail-closed recovery rows, or defer full details to
   `docs/solver_selection.md`.
2. The tutorial include list is narrower than the current public surface and
   does not help readers find LDLT, eigensolver, IC, analysis, Matrix Market,
   or API-reference details.
3. The tutorial does not reflect the exact first-use ladder from README,
   examples, and cookbook.

### Missing

1. A first maintained solve anchor that tells readers to build and run
   `./build/example_basic_solve` before choosing advanced paths.
2. A data-first route that explicitly points from CSR, CSC, and Matrix Market
   input to `example_compressed_input`, `example_matrix_market`, cookbook, and
   solver-selection.
3. A workflow-local diagnostics section aligned with
   `docs/solver_selection.md#diagnostics-handoff`.
4. Symmetric eigensolver tutorial coverage and `example_eigs` handoff.
5. MINRES and IC(0) handoff through `example_ic_minres`.
6. A concise API-reference handoff for readers who need exact declarations,
   ownership, options, result structs, or return codes.
7. A concise install/downstream handoff after first workflow success.

### Overbroad Or Claim-Risky

1. "ILU preconditioning dramatically reduces iteration counts" should be
   rewritten. It implies a general performance/convergence claim without local
   workload caveats.
2. Partial-SVD evidence wording is currently narrow but stale. Updating it
   must not imply broad partial-SVD correctness, raw vector identity, external
   parity, performance, platform, package, ABI, or state-of-the-art support.
3. Any future mention of Sprint 154 comparison evidence must stay fixture-local
   to `qr_underdetermined_minnorm_2x4` and must not imply NumPy/SciPy/LAPACK
   parity.

## Claim-Risk Register

| Risk | Current Trigger | Required Treatment |
| --- | --- | --- |
| Portable preconditioner performance claim | "dramatically reduces iteration counts" in the preconditioning section | Rewrite as local, workload-dependent acceleration; route diagnostics to solver-selection and benchmarks. |
| Stale partial-SVD proof | Partial-SVD evidence mentions only one fixture family | Either update to the full Sprint 151 fixture set or defer evidence detail to solver-selection. |
| Tutorial becoming maintainer policy | Potential report/freshness and support-tier additions | Keep report/install/platform text as short handoffs to owner docs. |
| External comparison overclaim | Sprint 154 `qr-minnorm` lane may be tempting to mention broadly | Mention only in advanced evidence/report contexts, if at all, and keep fixture-local non-parity wording. |
| Header/API drift | Tutorial may list headers as the API reference | Use API reference guidance for exact declarations rather than expanding tutorial into a full header index. |

## Rewrite Backlog For Days 3-5

### P0: Must Fix During Tutorial Alignment

1. Reframe the opening around the current first-use ladder:
   build, first maintained solve, data input, solver choice, diagnostics,
   install, and advanced controls.
2. Add a first-solve anchor for `example_basic_solve` and the example README.
3. Add workflow-local diagnostics guidance or a concise diagnostics handoff
   section.
4. Rewrite the preconditioning claim to avoid portable iteration-count or
   performance implication.
5. Refresh or delegate partial-SVD evidence to the current Sprint 151 bounded
   fixture set.

### P1: Should Fix If It Fits Day 4-5

1. Add explicit data-first routing to CSR, CSC, and Matrix Market workflows.
2. Add symmetric eigensolver tutorial coverage and `example_eigs` handoff.
3. Add MINRES/IC(0) handoff through `example_ic_minres`.
4. Add a compact install/downstream handoff after first workflow success.
5. Add a compact API-reference handoff without duplicating generated Doxygen.

### P2: Defer Unless Needed For Coherence

1. Expand QR comparison evidence in tutorial. Prefer deferring to
   solver-selection or maintainer guide unless a user-facing QR section needs
   it.
2. Add report freshness commands to the tutorial. Prefer a short advanced
   report handoff to benchmarks/maintainer docs.
3. Turn the tutorial into a complete public header catalog. Prefer API
   reference and selected header cleanup instead.

## Day 3 Handoff

Day 3 should design a target tutorial flow with these section jobs:

1. Start from the maintained first-use ladder.
2. Keep data-input routing before deep solver detail.
3. Present solver families by problem shape, with examples as runnable anchors.
4. Keep diagnostics workflow-local.
5. Keep install/package/platform details delegated to `INSTALL.md`.
6. Keep report/comparison evidence delegated to benchmarks and maintainer docs.
7. Keep exact declarations and option structs delegated to API reference and
   public headers.

## Day 2 Completion Check

- Major tutorial sections were compared against current owner surfaces.
- Stale partial-SVD evidence was identified.
- Claim-risky preconditioning wording was identified.
- Missing first-solve, diagnostics, eigensolver, API-reference, and install
  handoffs were recorded.
- Rewrite backlog is small enough for Days 3 through 5.
