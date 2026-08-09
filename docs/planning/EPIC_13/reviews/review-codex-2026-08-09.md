# Codex Review: Epic 13 Planning Baseline

**Date:** 2026-08-09
**Reviewer:** Codex
**Scope:** Full project review for code efficiency, maintainability,
usability, documentation, coherence, test coverage, and readiness to support
state-of-the-art sparse linear algebra claims.

## Executive Assessment

The project is substantially more mature than the early planning history would
suggest. It has a broad C API, many solver families, Make and CMake build
paths, static-first install proof, platform-specific CI, examples, a maintainer
guide, corpus metadata, report normalization, and explicit non-claim language.
That discipline is valuable.

It still should not claim state-of-the-art sparse linear algebra status. The
main blockers are not just missing algorithms. They are incomplete competitive
evidence, limited maintained corpus breadth, Windows parity gaps,
documentation scale, large monolithic tests and solver files, static-only
packaging, and performance/report artifacts that remain local and mostly
advisory.

Epic 13 should focus on closing a small number of high-value gaps completely:
Windows platform parity, broader maintained numerical corpus coverage for QR
and partial-SVD, generated report freshness, shared-library ABI/productization,
and direct competitive evidence. Those are the areas most likely to move the
library toward credible state-of-the-art positioning without overclaiming.

## Review Inputs

Reviewed surfaces included:

- public docs: `README.md`, `INSTALL.md`, `docs/*.md`,
  `benchmarks/README.md`;
- public headers: `include/*.h`;
- implementation: `src/*.c`;
- tests: `tests/*.c`, `tests/corpus/**`;
- build/package: `Makefile`, `CMakeLists.txt`, `cmake/*.in`;
- CI: `.github/workflows/*.yml`;
- planning evidence: Epic 10-12 retrospectives and Sprint 137-146 artifacts.

Snapshot observations:

- repository has more than 120k lines across C, headers, scripts, tests,
  examples, benchmarks, and planning docs;
- the largest tests are still multi-thousand-line files such as
  `tests/test_qr.c`, `tests/test_ldlt_csc.c`, `tests/test_integration.c`,
  `tests/test_svd.c`, and `tests/test_iterative.c`;
- the largest implementation files include `src/sparse_ldlt_csc.c`,
  `src/sparse_lu_csr.c`, `src/sparse_ldlt.c`, `src/sparse_iterative.c`,
  `src/sparse_qr.c`, `src/sparse_eigs.c`, and `src/sparse_svd.c`;
- maintained corpus metadata currently contains two promoted fixture rows:
  `qr_rank_deficient_6x4_nullspace_v1` and
  `partial_svd_clustered_repeated_diag8x6_k3_v1`;
- Windows CI intentionally excludes `test_threads`,
  `test_sprint4_integration`, and `test_fuzz`;
- the package contract is static-first and intentionally rejects
  `BUILD_SHARED_LIBS=ON`.

## Strengths

### Engineering Discipline

- The codebase has both Make and CMake build paths.
- Warning policy is stronger than many C libraries: `-Wall`, `-Wextra`,
  `-Wpedantic`, `-Wshadow`, `-Wconversion`, and stricter CMake flags on
  non-MSVC compilers.
- CI covers Linux, macOS, and Windows with clearly documented reviewed,
  supplemental, staged, local-only, and deferred lanes.
- Static package install behavior is tested through Make install,
  `pkg-config`, CMake install/export, exact-version behavior, and downstream
  consumer examples.
- Public docs now avoid unsupported state-of-the-art, broad parity, portable
  performance, and dynamic ABI claims.

### Algorithm Breadth

- The project includes direct solvers, iterative solvers, reordering,
  preconditioners, eigensolvers, full SVD, partial SVD, matrix arithmetic,
  CSR/CSC conversion, Matrix Market I/O, and repeated-run workflows.
- There is useful dispatch layering for linked-list, CSR, and CSC paths.
- Runtime/backend governance has improved through typed controls and explicit
  local-control boundaries.

### Test And Evidence Culture

- There are many focused tests plus install/package scripts and corpus/report
  validators.
- The corpus/report architecture separates source-controlled metadata from
  generated local evidence.
- Recent sprints built explicit residual queues and promotion gates, which
  makes overclaiming less likely.

## Gaps And Shortcomings

### 1. State-Of-The-Art Claims Are Still Unsupported

The project has not run direct comparative studies against established sparse
linear algebra ecosystems. Current evidence does not support claims against
LAPACK, SuiteSparse, SciPy, NumPy, ARPACK, PETSc, Trilinos, Eigen, or vendor
BLAS/LAPACK stacks.

Specific gaps:

- no maintained cross-library comparison harness;
- no pinned external dependency versions;
- no cross-platform reproducibility matrix for comparisons;
- no fixture families broad enough to describe ecosystem parity;
- no performance methodology strong enough for portable performance claims;
- no memory-footprint or scalability comparison with accepted baselines.

### 2. Maintained Corpus Breadth Is Too Narrow

The corpus architecture is good, but promoted fixture volume is still small.
Two source-controlled fixture rows cannot support broad QR, partial-SVD, SVD,
iterative, eigensolver, reorder, direct-solver, or SuiteSparse-style claims.

Specific gaps:

- QR evidence has one Sprint 139 promoted corpus fixture plus many older
  fixture-local tests, but not a broad maintained family suite;
- partial-SVD evidence has one Sprint 140 promoted corpus fixture plus bounded
  helper tests, but not enough repeated-spectrum, rank-deficient, rectangular,
  sparse-output, and convergence families;
- optional external data remains skip/defer policy rather than external-corpus
  proof;
- generated oracle/report rows are local and not source-controlled pass
  evidence;
- there is no mature fixture triage dashboard that helps maintainers choose
  the next numerical gap.

### 3. Windows Parity Remains The Largest Platform Gap

Windows support is real but narrow. The reviewed Windows lane is CMake-first;
Makefile, `pkg-config`, reviewed install-validation parity, pthread/POSIX
tests, and fuzz/property coverage remain staged or unsupported.

Specific gaps:

- `test_threads` and `test_sprint4_integration` depend on pthread APIs;
- `test_fuzz` depends on POSIX temp-file behavior;
- Windows CI has an enforced CTest count, but each new test requires manual
  count maintenance;
- Windows install/downstream proof is supplemental rather than reviewed parity;
- Windows package docs correctly avoid parity claims, but users still face a
  narrower platform story.

### 4. Static-Only Packaging Blocks Product Maturity

The static-first package contract is coherent, but state-of-the-art libraries
usually support shared libraries, ABI policy, package-manager recipes, and
runtime-loader validation.

Specific gaps:

- CMake rejects `BUILD_SHARED_LIBS=ON`;
- no symbol visibility/export policy;
- no dynamic ABI compatibility policy;
- no runtime-loader tests;
- no package-manager recipes or update/uninstall validation;
- no shared/static selector contract.

### 5. Maintainability Is Strained By Large Tests And Solver Files

The test suite is broad but several test files are very large. Some solver
implementations are also large enough that local reasoning is difficult.

Specific gaps:

- `tests/test_qr.c` is around 4k lines;
- `tests/test_ldlt_csc.c`, `tests/test_integration.c`, `tests/test_svd.c`,
  `tests/test_ldlt.c`, `tests/test_etree.c`, and `tests/test_iterative.c` are
  each multi-thousand-line files;
- several implementation files exceed 1k lines and mix algorithm, workspace,
  fallback, diagnostics, and error handling;
- source lists are duplicated in Make and CMake, increasing drift risk;
- historical sprint comments are sometimes still embedded in code and build
  files, which helps traceability but increases noise.

### 6. Documentation Is Useful But Heavy

The documentation is honest and comprehensive, but its size makes onboarding
hard. README and maintainer guide carry a lot of non-claim detail, support-tier
history, and evidence boundaries.

Specific gaps:

- the tutorial remains explicitly residual from Epic 12;
- several public headers still have not received the selected cleanup style
  applied in Sprint 145;
- users may struggle to distinguish first-use docs from maintainer evidence
  docs;
- claim-boundary wording appears in many places and can drift;
- there is no generated public API reference published as a consumer artifact.

### 7. Performance Evidence Is Mostly Local And Advisory

Benchmarks and sentinels exist, but the project correctly refuses portable
performance claims. That means performance work is not yet strong enough for
competitive positioning.

Specific gaps:

- benchmark rows are local measurement context, not release evidence;
- sentinel rows are selective and partly threshold-free;
- no standard hardware/compiler matrix for release benchmark publication;
- no external-library performance comparisons;
- no memory-footprint, bandwidth, or scaling dashboard;
- full benchmark runs are too heavy for normal PR CI.

### 8. API Usability Still Reflects Accreted Growth

The public API is broad and increasingly documented, but some usability
patterns remain sharp:

- `sparse_get` and `sparse_get_phys` intentionally return silent zero for NULL,
  out-of-bounds, and absent entries, which is convenient but ambiguous;
- compile-time choices such as `SPARSE_IDX_BITS`, OpenMP, mutexes, and backend
  thresholds are not represented by a single runtime capabilities query;
- one-shot and repeated-run workflows require careful reading across README,
  cookbook, solver-selection docs, and public headers;
- cancellation mutability differs by solver family;
- dense scalar support remains real-only `double`, with complex and wider
  precision intentionally unimplemented.

### 9. Coherence Risks Remain Across Report, CI, Docs, And Claims

Epic 12 improved coherence, but the project now has many places where the same
support boundary is restated. That creates future drift risk.

Specific gaps:

- Windows expected CTest count is hard-coded in workflow policy;
- support-tier language exists in README, INSTALL, maintainer guide, CI
  comments, report rows, sprint artifacts, and retrospectives;
- generated-report freshness semantics depend on correct command use;
- corpus/report rows can be misunderstood as pass evidence by future edits;
- there is no single machine-readable claim ledger that public docs can verify
  against.

## State-Of-The-Art Readiness

The project is not currently state of the art in the defensible public-claim
sense.

It is closer to a disciplined, broad, educational-to-midlevel sparse linear
algebra library with increasingly serious packaging, CI, and evidence
boundaries. To approach state-of-the-art status, it needs:

- broader maintained numerical corpus coverage;
- external-library comparisons with pinned versions and documented caveats;
- competitive performance methodology;
- stronger Windows parity;
- shared-library ABI and package-manager productization;
- clearer public API reference and onboarding;
- continued modularization of large solver/test files.

The most credible near-term goal is not "state of the art" as a blanket claim.
It is a set of narrow state-of-practice claims for specific surfaces, each
backed by direct evidence.

## Recommended Epic 13 Focus

Epic 13 should close these gaps in priority order:

1. Windows platform parity for the staged tests and install-validation decision.
2. Broader maintained QR and partial-SVD corpus families.
3. Generated report freshness for selected families needed by claims.
4. Shared-library ABI feasibility and productization decision.
5. External comparison harness and first narrow comparative study.
6. API/documentation simplification around tutorial and remaining headers.

The epic should continue rejecting broad state-of-the-art claims unless direct
comparative evidence exists by closeout.
