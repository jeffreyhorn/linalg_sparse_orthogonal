# Sprint 111 Working Notes

## Sprint Goal

Sprint 111 makes the project easier to adopt by turning the Epic 10 technical
evidence into a concise user-facing path: compressed-first matrix setup,
solver-selection guidance, Matrix Market behavior documentation, benchmark
interpretation, coherent examples, and a cleaner split between adoption
surfaces and maintainer proof records.

## Starting Constraints

- Keep public API and install-header surfaces stable unless a later explicit
  review proves a wording or declaration change is required.
- Do not describe a public Matrix I/O module or public Matrix builder API.
- Treat Matrix Market load/save as public functions with private source
  ownership, not as a newly public subsystem.
- Keep benchmark documentation evidence-bounded and local-environment aware.
- Keep examples on public APIs only; no private headers, helper targets, or
  planning-only proof scaffolding.
- Do not let maintainer proof language become the first adoption path.
- Documentation-only changes require `git diff --check` and a
  trailing-whitespace scan over touched docs.
- If later days modify examples or public headers, run the strongest checks
  required for the touched file types.

## Sprint 111 Adoption Surface Inventory

| Surface | Files | Current Role | Sprint 111 Disposition |
|---|---|---|---|
| README front door | `README.md` | First user path, capability overview, build commands, quick start, workflow selection. | Primary adoption entry; keep concise and route deeper topics to guides/examples. |
| Install docs | `INSTALL.md` | Cross-platform install and downstream consumer setup. | Keep as install authority; link from first-success and examples paths. |
| Tutorial | `docs/tutorial.md` | Longer API walkthrough and repeated-run lifecycle explanation. | Use as deeper workflow material after quick-start examples. |
| Matrix Market docs | `docs/matrix_market.md` | File format support, loader/saver behavior, SuiteSparse usage. | Tighten behavior and ownership wording without public module/builder claims. |
| Algorithm docs | `docs/algorithm.md` | Internal/algorithmic explanation for implementation behavior. | Use as reference, not the main adoption path. |
| Maintainer guide | `docs/maintainer_guide.md` | Quality policy, CI interpretation, maintainer workflow. | Keep maintainer-only; avoid sending first-time users here for basic usage. |
| Examples map | `examples/README.md` | Example index and public workflow descriptions. | Main practical next step after README; align with solver guide. |
| Example sources | `examples/*.c`, `examples/cmake_example/*` | Copyable public API usage references. | Update around compressed-first, solver, Matrix Market, and benchmark paths. |
| Benchmark docs | `benchmarks/README.md` | Benchmark commands, reports, and interpretation notes. | Clarify local measurement scope and comparison caveats. |
| Public headers | `include/*.h` | API contracts and option semantics. | Treat as source of truth for public names, ownership, options, and errors. |
| Generated reference | Doxygen via `make docs` | API reference generated from public headers. | Keep as reference output; do not make it the first adoption path. |
| Planning artifacts | `docs/planning/EPIC_*`, sprint artifacts, retrospectives | Evidence, deferred debt, implementation decisions, proof ownership. | Maintainer evidence only; use summaries in user docs where needed. |

## First-Time User Entry Points

| Need | Primary Entry | Follow-Up Surfaces | Notes |
|---|---|---|---|
| Build locally | `README.md` Building section | `INSTALL.md`, `examples/README.md` | README should remain the shortest successful path. |
| Install or consume downstream | `INSTALL.md` | `examples/cmake_example/` | Keep package and CMake details out of the quick-start body. |
| Matrix creation by insertion | `README.md` Quick Start | `docs/tutorial.md`, `include/sparse_matrix.h` | Keep as small first success, not the only recommended path. |
| CSR/CSC compressed input | `examples/example_compressed_input.c` | `examples/README.md`, `include/sparse_csr.h`, `include/sparse_matrix.h` | Sprint 111 should make this the obvious path for caller-owned compressed data. |
| Matrix Market loading | `docs/matrix_market.md` | `README.md`, `examples/README.md`, `include/sparse_matrix.h` | Must document behavior without public Matrix I/O module claims. |
| One-shot direct solve | `examples/example_basic_solve.c` | `README.md`, `include/sparse_lu.h`, `include/sparse_cholesky.h`, `include/sparse_ldlt.h`, `include/sparse_qr.h` | Keep as first successful solve. |
| Repeated direct solve | `examples/example_analysis.c` | `docs/tutorial.md`, `include/sparse_analysis.h` | Use when stable sparsity pattern and factor reuse matter. |
| Iterative solve | `examples/example_iterative.c` | `include/sparse_iterative.h`, `include/sparse_ilu.h`, `include/sparse_ic.h` | Split one-shot and reusable-handle messaging clearly. |
| Eigensolver | `examples/example_eigs.c` | `include/sparse_eigs.h`, benchmarks when measuring | Avoid turning benchmark evidence into universal performance claims. |
| SVD / low-rank | `examples/example_svd_lowrank.c` | `include/sparse_svd.h`, `include/sparse_bidiag.h` | Keep example small and user-facing. |
| Reordering/fill | `examples/example_colamd.c`, `examples/example_analysis.c` | `include/sparse_reorder.h`, `benchmarks/bench_reorder.c` | Explain when this is workflow guidance versus measurement. |
| Benchmarking | `benchmarks/README.md` | benchmark binaries, canonical reports | Present as local measurement and comparison context. |

## Maintainer-Only Surfaces

The following surfaces should not anchor first-time adoption, even when they
remain important evidence for future work:

- `docs/planning/EPIC_*/PROJECT_PLAN.md`
- `docs/planning/EPIC_*/SPRINT_*/WORKING_NOTES.md`
- `docs/planning/EPIC_*/SPRINT_*/RETROSPECTIVE.md`
- `docs/planning/EPIC_*/SPRINT_*/artifacts/*.md`
- `docs/planning/EPIC_*/reviews/*.md`
- CI count/drift discussions that exist to protect reviewed surfaces
- proof-owner cleanup notes for tests
- source-boundary and private-owner decisions
- residual deferred-debt queues

User-facing docs may link to summarized outcomes from those records only when
the summary directly helps adoption, troubleshooting, or responsible benchmark
interpretation.

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | User journey inventory, adoption surface map, maintainer-only surface split, and working-notes baseline. | Item 1 |
| 2 | Documentation gap audit against public headers, examples, README, tutorial, Matrix Market docs, and benchmark docs. | Item 1 |
| 3 | Solver-selection guide outline and decision-tree boundaries. | Item 2 |
| 4 | Solver-selection guide draft and compressed-first matrix-format guidance. | Item 2 |
| 5 | Compressed-first example audit and validation planning. | Item 3 |
| 6 | CSR/CSC construction example updates. | Item 3 |
| 7 | Direct, iterative, reuse, and reorder/fill workflow examples. | Item 3 |
| 8 | Eigensolver, SVD, and Matrix Market example updates. | Items 3 and 4 |
| 9 | Matrix Market behavior and ownership documentation. | Item 4 |
| 10 | Public header, tutorial, README, and guide coherence pass. | Items 4 and 6 |
| 11 | Benchmark interpretation documentation. | Item 5 |
| 12 | Maintainer/user surface split cleanup. | Item 6 |
| 13 | Integrated documentation and example validation. | Item 7 |
| 14 | Sprint closeout, residual queue, and downstream handoff. | Item 7 |

## Dependency Order

1. Adoption inventory must precede gap auditing and guide writing.
2. Gap audit must precede detailed solver guide and example edits.
3. Solver guide outline must precede final guide wording.
4. Example audit must precede example updates.
5. Matrix Market behavior docs must use Sprint 110 implementation evidence and
   must not precede the public/private boundary review.
6. Header/tutorial coherence must happen after guide and example language
   stabilizes.
7. Benchmark interpretation must stay after solver workflow wording is known.
8. Audience split cleanup must happen before integrated validation.
9. Integrated validation must precede closeout and residual handoff.

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation only | `git diff --check`; trailing-whitespace scan over touched docs. |
| Example `.c` files | Focused example build or `make examples` as applicable; `git diff --check`. |
| Public headers | Public API/install-header review plus the strongest code checks required by touched files. |
| Benchmark docs only | `git diff --check`; benchmark claim review against current commands/artifacts. |
| Mixed docs and examples | Apply the strongest check required by any touched file. |
| Implementation `.c` or `.h` files | Focused family tests plus `make format && make lint && make test`. |

## Day 1 Notes

- Created Sprint 111 working notes and artifact directory.
- Re-read Sprint 111 project-plan scope and established the docs/examples
  adoption workstream.
- Inventoried README, install docs, tutorial, Matrix Market docs, algorithm
  docs, maintainer guide, examples, benchmark docs, public headers, generated
  reference expectations, and planning artifacts.
- Mapped first-time user entry points for build, matrix creation, compressed
  input, Matrix Market loading, direct solve, repeated direct solve, iterative
  solve, eigensolve, SVD, reordering/fill, and benchmarking.
- Marked planning artifacts, proof-owner notes, source-boundary records, CI
  drift details, and residual-debt queues as maintainer-only surfaces.
- Established day-level ownership, dependency order, and validation
  expectations before Day 2 gap analysis.

## Day 2 Notes

- Audited adoption-facing documentation against public headers, examples, and
  the Sprint 110 Matrix Market implementation.
- Confirmed the compressed-first public contract in `include/sparse_csr.h`:
  simple `sparse_create_from_csr/csc(...)` constructors return `NULL` on
  invalid input or allocation failure, while diagnostic
  `sparse_from_csr/csc(...)` constructors return `sparse_err_t`; input arrays
  remain caller-owned and the returned `SparseMatrix` is independent.
- Confirmed Matrix Market public behavior from `include/sparse_matrix.h` and
  `src/sparse_matrix_io.c`:
  - save writes coordinate real general with `%.15g`;
  - load supports coordinate real, integer, and pattern value types with
    general or symmetric symmetry;
  - pattern entries default to `1.0`;
  - symmetric off-diagonal entries are mirrored;
  - unsupported format or coordinate errors return `SPARSE_ERR_PARSE`;
  - I/O failures return `SPARSE_ERR_IO` and set `sparse_errno()`;
  - successful load/save resets `sparse_errno()` to `0`;
  - loaded matrices are caller-owned and freed with `sparse_free(...)`.
- Identified the strongest user-facing gaps:
  - no single concise solver-selection guide yet;
  - Matrix Market docs omit duplicate-entry last-write behavior, final-zero
    elision, explicit ownership, and errno reset details;
  - README/tutorial/examples currently split workflow guidance across several
    places, which Day 3-4 should consolidate;
  - examples do not yet provide a dedicated Matrix Market workflow and should
    be audited before any source edits;
  - benchmark docs are accurate but dense and maintainer-oriented, so Day 11
    should add a clearer interpretation path.
- Classified gaps by audience:
  - user-blocking: missing concise solver-selection guide and missing precise
    Matrix Market behavior wording for load/use workflows;
  - confusing: duplicated workflow routing across README, tutorial, and
    examples;
  - maintainer-only: proof-owner and reviewed-lane detail in benchmark and
    maintainer docs;
  - future-work: any public Matrix I/O module or public builder API wording.
- Recorded validation needs:
  - documentation-only guide/docs changes need `git diff --check` and
    trailing-whitespace scans;
  - example source changes need at least focused example builds or
    `make examples`;
  - public header wording changes need public API/install-header review plus
    checks appropriate to touched files.

## Day 3 Notes

- Created the solver-selection guide outline and decision-tree artifact.
- Defined the guide audience as users choosing a public workflow, not
  maintainers interpreting proof ownership or source boundaries.
- Set the guide scope around:
  - matrix input and format choice;
  - one-shot direct solves;
  - repeated direct solve lifecycle;
  - iterative solver selection;
  - preconditioner expectations;
  - eigensolver and SVD usage boundaries;
  - reorder/fill workflows;
  - examples and benchmark handoff.
- Established compressed-first matrix guidance:
  - use CSR/CSC constructors when data already exists in compressed storage;
  - use mutable insertion for small hand-written matrices;
  - use Matrix Market loading for file exchange and SuiteSparse-style inputs;
  - keep dense-adjacent workflows scoped to public SVD/dense helper surfaces,
    not private implementation storage.
- Defined solver decision notes:
  - LU for general square one-shot solves;
  - Cholesky for SPD systems;
  - LDLT for symmetric indefinite systems;
  - QR for least-squares, rectangular, and rank-sensitive workflows;
  - explicit analyze/factor/solve/refactor only when sparsity pattern reuse is
    the reason to manage lifecycle objects.
- Defined iterative guide boundaries:
  - CG for SPD systems;
  - GMRES for general unsymmetric systems;
  - MINRES for symmetric indefinite systems;
  - BiCGSTAB as one-shot compatibility only;
  - repeated-run handles limited to CG, GMRES, and MINRES;
  - preconditioners must be introduced as problem-dependent acceleration, not
    correctness requirements.
- Defined eigen/SVD guide boundaries:
  - symmetric eigensolver guidance must stay on `sparse_eigs_sym(...)` and
    documented handle workflows;
  - backend notes should be selection aids, not state-of-the-art parity claims;
  - SVD guidance should cover rank, condition, pseudoinverse, and low-rank
    workflows without claiming external dense oracle coverage beyond existing
    evidence.
- Marked non-goals:
  - no public Matrix I/O module;
  - no public builder API;
  - no universal benchmark/performance claims;
  - no unsupported direct CSR/CSC solve API claim;
  - no broad external oracle or state-of-the-art proof claim.

## Day 4 Notes

- Added `docs/solver_selection.md` as the first concise user-facing
  solver-selection guide.
- Started the guide from matrix input format:
  - caller-owned CSR arrays;
  - caller-owned CSC arrays;
  - Matrix Market files;
  - small hand-written matrices;
  - existing public matrix shells.
- Made compressed-first construction the recommended path when coefficients
  already arrive as CSR or CSC arrays.
- Documented simple versus diagnostic compressed constructors and caller
  ownership:
  - `sparse_create_from_csr(...)` / `sparse_create_from_csc(...)`;
  - `sparse_from_csr(...)` / `sparse_from_csc(...)`;
  - input arrays remain caller-owned;
  - returned matrices are freed with `sparse_free(...)`.
- Documented direct solver selection:
  - LU for general square systems;
  - Cholesky for SPD systems;
  - LDLT for symmetric indefinite systems;
  - QR for rectangular, least-squares, and rank-sensitive workflows;
  - explicit analyze/factor/solve/refactor lifecycle only for stable-pattern
    repeated direct solves.
- Documented iterative solver selection:
  - CG for SPD systems;
  - GMRES for general unsymmetric systems;
  - MINRES for symmetric indefinite systems;
  - BiCGSTAB as a one-shot-only compatibility path;
  - preconditioners as problem-dependent acceleration tools.
- Documented eigensolver, SVD, reorder/fill, example handoff, and benchmark
  handoff boundaries without claiming portable performance, public Matrix I/O
  modules, public builder APIs, unsupported direct CSR/CSC solve APIs, or
  broad external-oracle/state-of-the-art proof.
- Left README/examples link integration for Day 10 coherence after Day 5-9
  example and Matrix Market wording work stabilizes.

## Day 5 Notes

- Audited the shipped example set and build registration before editing
  example source files.
- Confirmed Makefile behavior:
  - `EX_SRCS = $(wildcard examples/*.c)`;
  - new `examples/example_*.c` files are picked up automatically by
    `make examples`.
- Confirmed CMake behavior:
  - examples are listed explicitly with `add_executable(...)`;
  - any new compiled example must be added to `CMakeLists.txt`.
- Confirmed existing teaching examples:
  - `example_basic_solve` covers smallest one-shot LU;
  - `example_compressed_input` covers caller-owned CSR construction and
    one-shot LU;
  - `example_analysis` covers analyze-once / factor-many direct reuse;
  - `example_iterative` covers one-shot GMRES plus ILU(0);
  - `example_eigs` covers Matrix Market-backed symmetric eigensolver usage,
    shift-invert, and LOBPCG preconditioning;
  - `example_svd_lowrank` covers SVD, rank, condition, and low-rank output;
  - `example_colamd` covers COLAMD and QR-oriented ordering;
  - `example_ldlt`, `example_ic_minres`, `example_least_squares`,
    `example_minnorm`, `example_condition`, and `example_matrix_free` cover
    additional supported workflows.
- Identified Day 6 example work:
  - update `example_compressed_input` to show both CSR and CSC compressed-first
    construction;
  - keep input arrays caller-owned and returned matrices freed with
    `sparse_free(...)`;
  - keep the example small and public-API only.
- Identified Day 7 example work:
  - update `examples/README.md` so direct, iterative, reuse, QR/minnorm,
    LDLT, IC/MINRES, condition, matrix-free, SVD, eigensolver, and reorder
    examples align with `docs/solver_selection.md`;
  - avoid changing solver source behavior.
- Identified Day 8 example work:
  - add or identify a dedicated Matrix Market load/use route;
  - if adding `examples/example_matrix_market.c`, update CMake explicit
    registration and validate with `make examples`.
- Established example validation expectations:
  - existing example source edits: `make examples` and `git diff --check`;
  - new example source edits: `make examples`, CMake example registration
    review, and `git diff --check`;
  - documentation-only example README edits: `git diff --check` and
    trailing-whitespace scan.

## Day 6 Notes

- Updated `examples/example_compressed_input.c` to cover both public
  compressed-first construction routes:
  - `sparse_from_csr(...)` for diagnostic CSR import;
  - `sparse_create_from_csc(...)` for simple CSC import.
- Preserved the existing caller-owned-array demonstration for CSR and added the
  same check for CSC:
  - mutate the input compressed values after construction;
  - verify the returned `SparseMatrix` still owns an independent copy.
- Kept the example on public headers only:
  - `sparse_csr.h`;
  - `sparse_lu.h`;
  - `sparse_matrix.h`.
- Added a shared one-shot LU solve/report helper so the example teaches the
  construction difference without duplicating solve boilerplate.
- Kept cleanup explicit:
  - matrices returned from CSR/CSC construction are freed with
    `sparse_free(...)`;
  - caller-owned compressed arrays remain stack-owned in the example.
- Validation recorded for Day 6:
  - `make examples`;
  - `./build/example_compressed_input`;
  - `git diff --check`;
  - trailing-whitespace scan over touched Sprint 111 docs and the example
    source.

## Day 7 Notes

- Updated `examples/README.md` to align runnable solver examples with
  `docs/solver_selection.md`.
- Added a direct link from the examples front door to the solver-selection
  guide so users can choose a workflow before selecting a binary.
- Updated the compressed-input example description to reflect Day 6 CSR and
  CSC coverage.
- Made additional existing solver examples discoverable:
  - `example_ldlt` for symmetric indefinite direct solves;
  - `example_minnorm` for underdetermined minimum-norm QR;
  - `example_colamd` for COLAMD/reorder and QR-oriented ordering;
  - `example_condition` for condition-number workflows;
  - `example_ic_minres` for IC(0), CG, and MINRES;
  - `example_matrix_free` for callback-based GMRES.
- Preserved the public workflow boundaries:
  - examples include public headers only;
  - one-shot examples remain one-shot;
  - repeated-run direct workflow stays in `example_analysis`;
  - repeated iterative handle support remains limited to CG, GMRES, and
    MINRES;
  - benchmarks remain measurement surfaces, not correctness or portability
    proof.
- Validation recorded for Day 7:
  - `git diff --check`;
  - trailing-whitespace scan over touched docs;
  - no source changed during Day 7, so no example rebuild was required beyond
    the Day 6 source validation already recorded.

## Day 8 Notes

- Added `examples/example_matrix_market.c` as the dedicated Matrix Market
  load/use teaching example.
- Registered `example_matrix_market` in `CMakeLists.txt`; Makefile picks it up
  automatically through the `examples/*.c` wildcard.
- Kept the example on public APIs only:
  - `sparse_load_mm(...)`;
  - `sparse_errno()`;
  - `sparse_matvec(...)`;
  - `sparse_copy(...)`;
  - `sparse_lu_factor(...)`;
  - `sparse_lu_solve(...)`;
  - `sparse_free(...)`.
- Used `examples/example_alloc_helpers.h` for checked dynamic vector
  allocation.
- Updated `examples/README.md` and `docs/solver_selection.md` to point Matrix
  Market users to `example_matrix_market` instead of the earlier Day 8 plan
  placeholder.
- Left `example_eigs` and `example_svd_lowrank` source unchanged because they
  already provide stable concise user workflows; Day 8 clarified the example
  map rather than broadening advanced solver claims.
- Validation recorded for Day 8:
  - `make examples`;
  - `./build/example_matrix_market`;
  - `git diff --check`;
  - trailing-whitespace scan over touched docs and example source.

## Day 9 Notes

- Updated `docs/matrix_market.md` to document exact public Matrix Market
  behavior from `include/sparse_matrix.h`, `src/sparse_matrix_io.c`, and the
  regression tests.
- Added public API boundary wording:
  - Matrix Market support is exposed through `sparse_load_mm(...)` and
    `sparse_save_mm(...)`;
  - it is not a separate public Matrix I/O module;
  - it does not expose a public Matrix builder API.
- Documented loaded-matrix ownership:
  - successful loads produce caller-owned `SparseMatrix` objects;
  - callers free loaded matrices with `sparse_free(...)`;
  - failed loads leave `*mat_out` as `NULL`.
- Documented format behavior:
  - save writes coordinate real general with `%.15g`;
  - load supports real, integer, and pattern coordinate inputs with general or
    symmetric symmetry;
  - pattern entries default to `1.0`;
  - symmetric off-diagonal entries are mirrored and symmetric inputs must be
    square;
  - duplicate coordinates use last-entry-wins semantics;
  - final duplicate values of `0.0` are omitted from sparse storage.
- Documented error behavior:
  - `SPARSE_ERR_IO` captures system errno for `sparse_errno()`;
  - successful load/save resets `sparse_errno()` to `0`;
  - parse errors are represented by `SPARSE_ERR_PARSE`, not system errno.
- Validation recorded for Day 9:
  - `git diff --check`;
  - trailing-whitespace scan over touched docs.

## Day 10 Notes

- Updated README adoption handoffs so the compact workflow section points to
  `docs/solver_selection.md` for the fuller decision tree.
- Tightened README Matrix Market capability wording around the actual public
  functions:
  - `sparse_load_mm(...)`;
  - `sparse_save_mm(...)`.
- Updated `docs/tutorial.md` so the getting-started path links to the
  solver-selection guide and examples README before deeper reference material.
- Expanded the tutorial Matrix Market snippet to show:
  - `sparse_strerror(...)` for public error reporting;
  - `sparse_errno()` only for `SPARSE_ERR_IO`;
  - caller ownership and cleanup with `sparse_free(...)`;
  - handoff to `docs/matrix_market.md` for detailed format behavior.
- Replaced a tutorial maintainer-guide detour in the QR section with a
  user-facing solver-selection guide reference.
- Removed Sprint-planning language from the Matrix Market section of
  `docs/solver_selection.md` now that Day 9 behavior docs exist.
- Updated `include/sparse_matrix.h` Matrix Market load comments to match the
  public docs:
  - square requirement for symmetric input;
  - mirrored off-diagonal entries;
  - duplicate last-entry-wins behavior;
  - final-zero elision;
  - broader parse-error categories.
- Added Day 10 artifact:
  - `artifacts/day10-header-tutorial-coherence.md`.
- Validation recorded for Day 10:
  - `git diff --check`;
  - trailing-whitespace scan over touched Day 10 files.

## Day 11 Notes

- Added a `Reading Benchmark Results` section to `benchmarks/README.md`.
- Documented benchmark interpretation rules:
  - benchmark rows are local measurement artifacts;
  - compare workload identity before timing;
  - read residual/status/convergence/fill/path fields before speed fields;
  - use `manifest.txt` and `index.tsv` for command, branch, commit,
    compiler/platform, artifact, and label context.
- Clarified sensitivity to:
  - machine;
  - compiler;
  - operating system;
  - dense backend;
  - OpenMP runtime and thread count;
  - matrix corpus;
  - build options;
  - repeat count.
- Added a benchmark handoff table separating:
  - examples as API learning surfaces;
  - solver-selection docs as problem-shape guidance;
  - `bench-canonical-report` as threshold-free branch-local CSV output;
  - `performance-sentinels` as a bounded local sentinel bundle;
  - individual `bench_*` binaries as focused local measurements;
  - tests as regression/oracle/property evidence.
- Replaced README wording that said benchmarks "prove" workflow/performance
  with measurement wording tied to local configuration.
- Tightened `docs/solver_selection.md` benchmark handoff caveats around
  machine, compiler, backend selection, matrix corpus, build options, and
  thread settings.
- Added Day 11 artifact:
  - `artifacts/day11-benchmark-interpretation.md`.
- Validation recorded for Day 11:
  - `git diff --check`;
  - trailing-whitespace scan over touched Day 11 docs.

## Day 12 Notes

- Cleaned remaining adoption-surface proof-owner detail from README and
  tutorial.
- Updated README direct-solver evidence wording:
  - removed inline external dense-reference lane detail from the front door;
  - linked to `docs/maintainer_guide.md` for evidence boundaries and current
    test ownership;
  - kept README focused on choosing and running public workflows.
- Updated tutorial audience wording:
  - replaced "owner surface" phrasing with user-facing "next level of detail";
  - removed inline LU external dense-reference evidence detail;
  - removed inline Cholesky external dense-reference evidence detail;
  - replaced Sprint-era LDL^T KKT evidence wording with workflow guidance for
    one-shot versus explicit repeated-run lifecycle use.
- Preserved evidence traceability through:
  - `docs/maintainer_guide.md`;
  - named test owners already listed there;
  - Day 12 artifact before/after notes.
- Added Day 12 artifact:
  - `artifacts/day12-audience-boundary-split.md`.
- Validation recorded for Day 12:
  - `git diff --check`;
  - trailing-whitespace scan over touched Day 12 docs.

## Day 13 Notes

- Performed integrated validation across the Sprint 111 adoption surfaces:
  - README;
  - solver-selection guide;
  - Matrix Market docs;
  - tutorial;
  - benchmark README;
  - examples README;
  - Matrix Market public header comments;
  - compressed-input example;
  - Matrix Market example.
- Confirmed guide/example/header/README agreement on:
  - CSR/CSC input arrays remain caller-owned;
  - constructed and loaded matrices are caller-owned and freed with
    `sparse_free(...)`;
  - Matrix Market support is the public `sparse_load_mm(...)` /
    `sparse_save_mm(...)` surface;
  - `SPARSE_ERR_IO` is the Matrix Market error class that exposes captured
    system errno through `sparse_errno()`;
  - benchmark outputs are local measurement artifacts, not portable
    performance guarantees.
- Confirmed Matrix Market no-drift claims:
  - no public Matrix I/O module claim;
  - no public Matrix builder API claim;
  - behavior docs remain aligned with public header comments and
    `example_matrix_market`.
- Ran validation:
  - `make examples`;
  - `./build/example_compressed_input`;
  - `./build/example_matrix_market`;
  - `cmake -S . -B cmake-build`;
  - `cmake --build cmake-build --target example_compressed_input example_matrix_market`;
  - `./cmake-build/example_compressed_input`;
  - `./cmake-build/example_matrix_market`;
  - `git diff --check`;
  - trailing-whitespace scan over touched docs, examples, header, and Sprint
    111 artifacts;
  - local relative Markdown link existence check across README, solver guide,
    Matrix Market docs, tutorial, benchmark README, and examples README.
- Added Day 13 artifact:
  - `artifacts/day13-integrated-validation.md`.
- Residuals for Day 14:
  - external HTTP links were not network-checked;
  - full `make format && make lint && make test` was not rerun on Day 13
    because Days 11-13 were documentation-only after the Day 10 full quality
    pass; Day 13 focused on integrated docs/examples validation.

## Day 14 Notes

- Reviewed the full Sprint 111 artifact set, working notes, and adoption
  surface changes against the Sprint 111 project-plan items.
- Closed all seven Sprint 111 items:
  - user journey audit;
  - solver-selection guide;
  - compressed-first example batch;
  - Matrix Market behavior and ownership docs;
  - benchmark interpretation docs;
  - maintainer/user split cleanup;
  - validation and closeout.
- Summarized changed adoption surfaces:
  - README workflow routing;
  - new solver-selection guide;
  - tutorial handoffs;
  - Matrix Market behavior docs and public header comments;
  - benchmark interpretation docs;
  - examples README and compressed-first/Matrix Market examples;
  - CMake example registration.
- Ran final full quality validation:
  - `make format && make lint && make test`.
- Ran final hygiene validation:
  - `git diff --check`;
  - trailing-whitespace scan over touched docs, examples, public header, and
    Sprint 111 artifacts.
- Added Day 14 artifact:
  - `artifacts/day14-closeout-and-handoff.md`.
- Deferred non-blocking residuals for Sprint 112 or later:
  - optional external-link network validation;
  - keep README quality/CI wording compact;
  - preserve benchmark documentation scanability;
  - review `docs/algorithm.md` only if it becomes a public adoption surface;
  - keep performance wording tied to measured local evidence.
