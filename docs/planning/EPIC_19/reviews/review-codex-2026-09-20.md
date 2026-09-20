# Codex Review: Epic 19 Intake

**Date:** 2026-09-20  
**Reviewer:** Codex  
**Scope:** Full repository review for efficiency, maintainability, usability,
documentation, coherence, test coverage, and readiness to support a
state-of-the-art sparse linear algebra library claim.

## Executive Summary

The project has grown into a broad C sparse linear algebra library with a large
solver surface, strong local tests, extensive documentation, and unusually
explicit claim-boundary governance. It now includes linked-list matrix
construction, compressed import/export, direct solvers, iterative solvers,
eigensolvers, SVD paths, graph/reordering utilities, benchmarks, generated
reports, install/export validation, and selected hosted evidence.

The codebase is not yet defensibly state of the art. The remaining blockers
are not a lack of features alone; they are productization, portability,
algorithmic evidence, ABI/release maturity, maintainability, and benchmark
methodology gaps. The strongest path for Epic 19 is to fully close a small set
of high-value gaps instead of spreading work across every residual.

## Repository Strengths

### Breadth of numerical functionality

- Public headers under `include/` expose a wide set of workflows: matrix shell,
  CSR/CSC conversion, LU, Cholesky, LDL^T, QR, SVD, iterative solvers,
  eigensolvers, preconditioners, dense helpers, graph/reordering, and analysis
  lifecycles.
- The README now gives a clear first-use route and maps major workflows to
  examples, solver-selection docs, benchmarks, API reference, and maintainer
  guidance.
- The project has selected evidence for direct, iterative, QR, SVD, partial
  SVD, eigensolver, graph, package, and generated-report behaviors.

### Quality infrastructure

- The Makefile and CMake paths are both maintained, with reviewed Linux, macOS,
  and Windows CI lanes.
- The code builds with strict warnings and uses lint/static-analysis gates in
  the reviewed Makefile path.
- The test suite is large and covers many family-specific regressions. Current
  C/Python/shell test and tooling files total more than 150k lines.
- Documentation and claim guards protect package-manager, static package,
  support, generated API, Windows, selected benchmark, selected comparison, and
  helper-ownership boundaries.

### Documentation and support-boundary discipline

- `INSTALL.md#support-readiness-matrix` is a useful public support authority.
- `docs/maintainer_guide.md` records evidence interpretation, non-claims, and
  owner surfaces in detail.
- `docs/cookbook.md`, `docs/solver_selection.md`, `examples/README.md`, and
  README adoption routes make first-use navigation better than the raw API
  breadth would otherwise allow.
- Epic 18 closed several selected scopes without overstating broad support.

## Major Gaps And Shortcomings

### 1. State-of-the-art claim is not earned

The project correctly does not claim state-of-the-art sparse linear algebra
status. The missing evidence is material:

- no broad SuiteSparse, PETSc, Trilinos, Eigen, SciPy, NumPy, LAPACK, or
  ARPACK parity program;
- no external baseline matrix suite with pinned versions, build options,
  platforms, tolerances, and reproducible runner metadata;
- no timing thresholds or statistical methodology for performance claims;
- no broad memory-footprint, scalability, robustness, or numerical-stability
  claim across matrix families;
- no release, ABI, or package provenance story sufficient for third-party
  production adoption.

State-of-the-art status needs an explicit evidence program, not more scattered
feature additions.

### 2. Packaging remains incomplete for real users

The source install and static CMake/pkg-config surfaces are maintained, and
Sprint 198 proved a developer-mode local Homebrew source formula. That is not
the same as user-facing package-manager support.

Remaining shortcomings:

- no Homebrew/core readiness, bottles, public tap maintenance, Linuxbrew, or
  supported Homebrew installation path;
- no vcpkg, Conan, pkgsrc, distro package, binary package, or provider-specific
  release recipe;
- no package provenance policy beyond local static-source proof;
- no user-facing install claim outside Make/CMake static source install.

This is one of the highest productization gaps because it blocks evaluation by
external users who expect package-consumer workflows.

### 3. Shared-library, ABI, and release policy are deferred

`CMakeLists.txt` intentionally rejects `BUILD_SHARED_LIBS=ON`, and the docs
are clear that shared-library packaging and dynamic ABI support are deferred.
That honesty is good, but it is a major maturity gap.

Missing pieces:

- symbol visibility and export/import policy;
- Linux SONAME, macOS install-name/RPATH, and Windows DLL/import-library
  behavior;
- ABI compatibility rules and versioning policy;
- release checklist, artifact retention, changelog, and provenance;
- runtime-loader validation and downstream shared-library consumer tests.

Without this, the project remains static-first and not release-grade for many
downstream environments.

### 4. Windows support is still partly selected and partly deferred

Windows CMake/MSVC lanes exist and are valuable, but Windows evidence is still
carefully bounded. Selected Cholesky comparison freshness and QR incompatible
comparison promotion remain incomplete or re-deferred in the Epic 18 residual
queue.

Remaining gaps:

- no broad Windows Makefile parity;
- no Windows `pkg-config` execution parity;
- selected Windows freshness claims require manifest/support-tier promotion;
- QR incompatible comparison lacks hosted Windows/MSVC proof and artifact
  inspection;
- Windows benchmark freshness is not promoted.

This is a coherence gap for users reading platform support: Windows is present,
but support tiers remain nuanced and selected.

### 5. Allocation-failure and cleanup proof is still selected, not broad

The project has strong selected allocation-failure proofs for several owners,
including symbolic LU. That is not broad reliability coverage.

Remaining gaps:

- direct solver factorization internals still have allocation-heavy paths
  without deterministic failure proof;
- matrix construction/import/export paths need broader stale-output and retry
  proof;
- eigensolver/SVD/QR workspace paths need selected owner expansion;
- generated tooling and package/install flows do not have broad allocation
  reliability claims;
- OS OOM and concurrent allocation-hook behavior remain out of scope.

For a C numerical library, allocation failure behavior is a real reliability
axis and should be expanded one owner at a time.

### 6. Review surface remains large

Several implementation and test files remain difficult to review in one pass:

- large tests such as `tests/test_ldlt_csc.c`, `tests/test_etree.c`,
  `tests/test_integration.c`, `tests/test_qr.c`, `tests/test_ldlt.c`, and
  `tests/test_iterative.c`;
- large implementation files such as `src/sparse_ldlt_csc.c`,
  `src/sparse_lu_csr.c`, `src/sparse_ldlt.c`, `src/sparse_iterative.c`,
  `src/sparse_qr.c`, `src/sparse_eigs.c`, and `src/sparse_svd.c`;
- large Python report tools such as `scripts/run_external_comparison.py` and
  `scripts/normalize_report_index.py`.

Epic 18 reduced one selected SVD test surface, but broad maintainability still
needs targeted extraction, ownership guards, and focused tests.

### 7. Performance evidence is selected and threshold-free

The benchmark suite is useful and now has selected hosted freshness lanes.
However, the project intentionally avoids portable performance and timing
threshold claims.

Remaining gaps:

- no stable performance acceptance thresholds;
- no statistical methodology for variance, warmup, repeats, and runner class;
- no cross-platform performance matrix;
- no baseline comparisons against external libraries;
- no memory-bandwidth or cache-efficiency analysis for key kernels;
- orthogonal linked-list storage remains inherently less cache-friendly for
  many numerical kernels than CSR/CSC-first implementations.

This limits the project’s ability to claim high-performance or
state-of-the-art status.

### 8. Core storage architecture is flexible but not performance-first

The orthogonal linked-list shell is useful for mutation and dual traversal, but
it is not the standard high-performance representation for numerical kernels.
The code has added CSR/CSC backends and compressed-first workflows, yet the
public identity and many paths still revolve around linked-list storage.

Risks:

- pointer chasing limits cache locality;
- dual linked lists increase memory overhead;
- conversion and dispatch policy are complex;
- performance-critical paths need clearer compressed-kernel ownership;
- users may not know when to stay compressed-first.

The library can be useful, but a state-of-the-art sparse library needs
compressed formats and backend selection to be first-class and measured.

### 9. Usability is improved but still complex

The docs are extensive, but the volume is high and many claims are carefully
qualified. This protects correctness, but it can overwhelm users.

Observed issues:

- many workflows require choosing among one-shot, analyze/refactor, compressed
  first, iterative handles, eigensolver handles, and generated evidence
  surfaces;
- support status is accurate but nuanced;
- examples help, but there is no single installed "happy path" package story;
- generated API docs are local-only and not easily discoverable online;
- no stable release page or external docs site exists.

The project is easier to use than earlier epics, but still not simple for a
new external user.

### 10. Documentation is coherent but evidence-heavy

The documentation is unusually thorough. The downside is that many pages carry
long non-claim sections and historical sprint references. This is appropriate
for internal governance, but public docs should continue separating first-use
guidance from maintainer evidence.

Risk:

- evidence tables can make public docs feel defensive;
- historical references can obscure current support;
- guard-sensitive phrases create maintenance overhead;
- local-only generated API docs limit external API discovery.

### 11. Test coverage is broad but not yet a product-quality oracle suite

The test suite covers many families and edge cases, but the review found
several boundaries:

- broad randomized/property testing is limited compared with the API surface;
- many external references are selected fixtures rather than broad oracle
  parity;
- allocation failure is selected, not systematic;
- performance regressions are freshness/metadata oriented rather than
  threshold gates;
- Windows/macOS hosted coverage is selected and claim-bound.

This is strong regression coverage for a growing project, not yet a complete
conformance/performance validation program.

## Coherence Assessment

The project is coherent in its current claim model: selected evidence is
named, broad claims are avoided, and public/maintainer docs explain the
support tiers. The main coherence problem is that the implementation breadth
looks more ambitious than the product support maturity. Users may see a large
solver list and assume broad parity, while the evidence supports narrower
selected claims.

Epic 19 should keep the honest claim model while closing product gaps that
make support easier to state plainly.

## State-Of-The-Art Assessment

The project is not yet a state-of-the-art sparse linear algebra library.

It is a capable educational/research-oriented C library with unusually strong
planning discipline, selected evidence, and documentation. To become
state-of-the-art, it needs:

- package and release maturity;
- external-library baseline methodology;
- compressed-kernel performance focus;
- broad numerical conformance evidence;
- ABI/shared-library policy;
- hosted documentation/public API publication;
- complete platform story for selected promoted targets;
- repeatable benchmark thresholds or an explicitly reviewed methodology for
  why thresholds are not claimed.

## Recommended Epic 19 Focus

Epic 19 should close fewer gaps completely. Recommended closure themes:

1. user-facing package distribution for one provider path;
2. selected Windows freshness promotion for one already-guarded target;
3. one additional allocation-failure owner outside symbolic LU;
4. one additional large review-surface reduction;
5. one selected performance methodology upgrade with thresholds or explicit
   threshold-deferral proof;
6. generated API publication decision and implementation if selected;
7. shared-library/ABI/release readiness decision with at least one concrete
   proof path;
8. a state-of-the-art evidence blueprint that can be executed later without
   overclaiming in Epic 19.
