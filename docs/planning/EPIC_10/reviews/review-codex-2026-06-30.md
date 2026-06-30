# Epic 10 Code Review - Codex - 2026-06-30

## Executive Verdict

`linalg_sparse_orthogonal` is a serious, unusually well-validated C sparse
linear algebra project. It has broad solver coverage, a disciplined quality
surface, static install/export proof, CMake consumer validation, substantial
examples, and a long record of explicit non-claims. It is far beyond a toy
library.

It is not yet a state-of-the-art sparse linear algebra library in the
SuiteSparse, PETSc, Trilinos, Eigen, Intel oneMKL, or CHOLMOD/UMFPACK sense.
The largest remaining gap is not one missing solver. It is product maturity:
compressed storage is still converging with a compatibility matrix shell, the
best external comparisons are bounded rather than family-wide, packaging is
static-first, platform parity is intentionally asymmetric, and several
implementation and proof owners remain too large for long-term velocity.

Epic 10 should therefore focus on earning narrower, verifiable product claims:
compressed-first workflows, direct and iterative solver oracle depth,
performance/backend evidence, packaging/API usability, cross-platform proof,
and maintainability extraction.

## Review Basis

This review is based on the live repository state after Epic 9 closeout and
PR #112 merge, including:

- `README.md`, `INSTALL.md`, `benchmarks/README.md`, and
  `docs/maintainer_guide.md`
- public headers in `include/`
- implementation files in `src/`
- tests, examples, benchmarks, scripts, and CI workflow topology
- Epic retrospectives, especially `docs/planning/EPIC_7`,
  `docs/planning/EPIC_8`, and `docs/planning/EPIC_9`
- Epic 9 residuals and explicit non-claims

Measured repository signals:

| signal | observed value |
|---|---:|
| files under `src`, `include`, `tests`, `benchmarks`, `examples`, `scripts`, and `docs/planning` | `1,933` |
| total lines across `src/*.c`, `include/*.h`, `tests/*.c`, `benchmarks/*.c`, and `examples/*.c` | `110,359` |
| strongest documented local review command | `make quality-review-full` |
| Epic 9 final reviewed Make/CMake parity | `54` vs `54` tests |
| Epic 9 final full reviewed CMake run | `54 / 54` passing |
| Epic 9 final Make install/export proof | `14 / 14` passing |
| Epic 9 final CMake install/export proof | `16 / 16`, `0` skips |
| documented coverage threshold | supplemental `80%` line coverage gate |

Largest implementation and proof owners observed:

| file | lines | risk |
|---|---:|---|
| `tests/test_ldlt_csc.c` | `3,878` | giant proof owner, hard to isolate failures |
| `tests/test_integration.c` | `3,421` | broad mixed responsibility |
| `tests/test_qr.c` | `3,234` | large direct-family proof owner |
| `tests/test_ldlt.c` | `2,977` | large direct-family proof owner |
| `tests/test_etree.c` | `2,962` | large graph/order proof owner |
| `tests/test_graph.c` | `2,925` | graph proof owner still carries history |
| `src/sparse_ldlt_csc.c` | `2,174` | largest source hotspot |
| `src/sparse_lu_csr.c` | `1,665` | large source hotspot |
| `src/sparse_qr.c` | `1,563` | large source hotspot |
| `src/sparse_ldlt.c` | `1,535` | large source hotspot |
| `src/sparse_eigs.c` | `1,534` | large source hotspot |
| `src/sparse_iterative.c` | `1,495` | large source hotspot |

## Efficiency

### Strengths

- The project has real compressed sparse computation coverage: CSR/CSC direct
  paths, LU, Cholesky, LDLT, QR, SVD, eigensolvers, iterative solvers,
  reorderers, graph support, and benchmark surfaces.
- Epic 9 improved compressed-first construction/import, repeated-run lifecycle
  ownership, dense backend seams, nested-dissection runtime, and bounded
  reorder/fill evidence.
- The Make/CMake parity proof and source-list checker reduce build drift.
- Optional OpenMP and sanitizer surfaces show concern for runtime behavior
  beyond basic correctness.

### Gaps

- The product still has a compatibility matrix shell and linked-list-first
  residue. Compressed formats are much stronger than before, but not yet the
  sole product center.
- Performance evidence is local and bounded. There is no broad, maintained
  comparison matrix against SuiteSparse, Eigen/Spectra, ARPACK-class
  eigensolvers, oneMKL sparse, CHOLMOD, KLU, UMFPACK, or GraphBLAS-style
  backends.
- Dense acceleration is still a bounded portable seam rather than a broad
  BLAS/LAPACK backend contract with crisp selection, observability, and
  failure modes.
- Parallelism is partial. OpenMP coverage exists, but there is no coherent
  product-level runtime model for thread pools, nested parallelism, backend
  interactions, or deterministic performance tuning.
- There is no GPU, distributed-memory, or block-sparse product story. That is
  acceptable if scoped out, but it prevents state-of-the-art claims.
- Timing benchmarks are explicitly not portable performance claims. This is
  honest, but it leaves the project short of competitive performance evidence.

## Maintainability

### Strengths

- Epic 9 landed important extraction work and source-list drift checking.
- The maintainer guide, quality targets, and validation scripts make the repo
  unusually navigable for a C library with this breadth.
- Historical retrospectives preserve decision context and non-claims.

### Gaps

- Several source files remain too large. `src/sparse_ldlt_csc.c`,
  `src/sparse_lu_csr.c`, `src/sparse_qr.c`, `src/sparse_eigs.c`, and
  `src/sparse_iterative.c` are still dense enough that local changes carry
  unnecessary regression risk.
- Several test owners are larger than many subsystems. They should become
  focused fixtures, family-local helpers, and scenario files.
- Permanent code and test surfaces still contain sprint-era chronology,
  temporary/fallback wording, and implementation history. The public docs were
  improved in Epic 9, but lower-level source/test comments still carry
  planning residue.
- Internal configuration, environment overrides, fallback paths, and proof
  scripts are powerful but hard to reason about as a unified product system.
- Build, benchmark, coverage, install, and CI scripts are numerous. Their
  behavior is documented, but the ownership model remains complex.

## Usability

### Strengths

- The README gives a practical workflow chooser.
- Examples cover basic, iterative, direct, decomposition, reordering,
  graph, and performance paths.
- Static install/export and CMake/pkg-config consumer proof are maintained.
- Public headers expose broad functionality directly to C users.

### Gaps

- The API remains low-level and broad. A new user must learn many structs,
  ownership rules, options objects, and solver-specific contracts before
  making confident choices.
- The linked-list matrix shell, compressed views, import/export helpers, and
  solver-specific compressed handles still read as multiple product models.
- Error handling and observability are not yet product-polished. Some APIs
  expose C-level status values while others rely on conventions or validation
  helpers.
- Exact-version CMake package wording and static-first installation are clear,
  but they are narrower than typical library consumer expectations.
- There are no higher-level language bindings, package-manager recipes, or
  turnkey examples for common application workflows.

## Documentation

### Strengths

- Documentation volume and honesty are strong.
- The project is unusually clear about reviewed versus supplemental checks,
  unsupported claims, package boundaries, and platform asymmetry.
- Install, benchmark, maintainer, and planning docs give maintainers enough
  context to avoid accidental overclaiming.

### Gaps

- The documentation set is large enough that it can overwhelm users. It is
  stronger as a maintainer archive than as a product onboarding path.
- Solver-selection guidance is still scattered. A user should be able to pick
  direct, iterative, eigensolver, reorder, and decomposition paths from one
  concise decision guide.
- Benchmark docs correctly avoid broad timing claims, but users need clearer
  examples of how to interpret local evidence.
- Planning artifacts are valuable, but their density makes it hard to see the
  current product truth without reading many retrospective files.

## Coherence

### Strengths

- The repo has an admirable habit of preserving truth boundaries. Claims,
  non-claims, and reviewed surfaces are explicit.
- Epic 9 improved public narrative coherence and naming, especially around
  support surfaces and proof ownership.

### Gaps

- The library still spans three identities: educational C sparse library,
  productizing compressed sparse solver library, and long-running planning
  program. Those identities are compatible only if docs and APIs clearly
  separate user-facing truth from historical process.
- The central product model is not yet simple enough. "Use compressed sparse
  matrices first" should become the obvious path, with mutable matrix-shell
  compatibility documented as secondary.
- Some benchmark, CI, and install proof names are still easier for maintainers
  than for external users.

## Test Coverage and Assurance

### Strengths

- Reviewed Make/CMake parity, CTest counts, install/export proof, source-list
  checking, sanitizer lanes, and coverage targets form a strong quality net.
- Tests are broad and include direct solvers, iterative solvers, eigensolvers,
  reorderers, graph support, edge cases, integration scenarios, and selected
  external dense-reference checks.
- Epic 9 added LDLT CSC external dense-reference assurance and bounded
  reorder/fill artifacts.

### Gaps

- Coverage remains supplemental rather than a reviewed universal gate.
- External oracle coverage is still selective. Cholesky and LDLT CSC are
  stronger than several other solver families.
- Windows remains a reviewed CMake subset and does not claim Makefile or
  install-validation parity.
- The project lacks a broad numerical corpus with matrices stratified by
  conditioning, symmetry, definiteness, sparsity pattern, scale, and expected
  solver behavior.
- There is limited automated performance regression governance. The project
  rightly avoids portable timing claims, but it still needs bounded local
  regression sentinels for hot paths.
- Giant tests make failure localization and review cost high.

## State-of-the-Art Assessment

The project can credibly claim to be a broad, well-tested, self-contained C
sparse linear algebra library with strong maintainer discipline. It cannot yet
credibly claim to be state of the art.

State-of-the-art sparse linear algebra libraries generally provide many of the
following:

- compressed-format-first APIs and data ownership
- strong direct solver backends with supernodal, multifrontal, or vendor
  backend options
- robust ordering integration and fill-reduction strategy guidance
- broad iterative/eigensolver comparison against external references
- tuned kernels, clear backend selection, and performance portability
- mature packaging, ABI/versioning, shared-library support, and downstream
  ecosystem integration
- language bindings or high-level interfaces for common users
- large external matrix corpora and recurring numerical robustness campaigns
- platform parity or clearly tiered support contracts

`linalg_sparse_orthogonal` has pieces of this, but not enough breadth or
external evidence to use the state-of-the-art label without qualification.

## Highest-Priority Gaps

1. Make compressed CSR/CSC workflows the unmistakable product center.
2. Expand external oracle comparisons across direct, iterative, eigensolver,
   SVD, QR, reorder, and graph paths.
3. Extract large source and giant-test owners until high-risk changes are
   local and reviewable.
4. Define a durable backend/runtime contract for BLAS/LAPACK-class optional
   acceleration, OpenMP behavior, and observability.
5. Improve public usability with a solver-selection guide, clearer ownership
   contracts, and sharper examples.
6. Expand package maturity beyond static-first proof or explicitly preserve it
   as a long-term tiered support decision.
7. Turn cross-platform validation into a clearer tier model with Windows and
   macOS expectations separated from Linux strongest truth.
8. Convert benchmark and coverage architecture into decision-grade evidence
   without pretending local timings are universal.

## Recommended Epic 10 Direction

Epic 10 should not be a feature grab bag. It should be a productization and
evidence epic:

- compress the product model around CSR/CSC
- add external comparison depth where claims are weakest
- reduce maintenance hotspots
- make performance/runtime/backend evidence clearer
- improve user-facing documentation and examples
- strengthen packaging and platform truth
- close with a competitive calibration that says exactly what claims were
  earned and what remains outside scope
