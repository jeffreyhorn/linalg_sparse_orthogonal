# Epic 11 Code Review - Codex - 2026-07-09

## Executive Verdict

`linalg_sparse_orthogonal` is now a broad, heavily validated, self-contained C
sparse linear algebra library with unusually strong planning discipline. Epic
10 materially improved the product story: compressed-first workflows are more
visible, selected solver families have stronger bounded evidence, package and
platform support tiers are explicit, public claims are better fenced, and the
final reviewed local validation anchor remained green.

The project is still not a state-of-the-art sparse linear algebra library in
the SuiteSparse, PETSc, Trilinos, Eigen, ARPACK/LAPACK, CHOLMOD/UMFPACK, KLU,
GraphBLAS, oneMKL, or high-performance package-ecosystem sense. The biggest
remaining issue is no longer broad missing functionality alone. It is the gap
between a serious research/productizing C library and a mature external-facing
numerical product:

- several proof owners and source files are still too large;
- solver evidence remains selected and fixture-local rather than ecosystem-wide;
- performance evidence is intentionally local rather than portable;
- package support is static-first and does not include shared-library ABI or
  package-manager integration;
- Windows and macOS support remain explicitly narrower than Linux;
- adoption surfaces are honest but still large and difficult for new users to
  navigate quickly.

Epic 11 should therefore focus on product hardening after Epic 10: finishing
remaining proof-owner/source-boundary debt, widening numerical oracle coverage
where it can be sustained, adding carefully scoped performance and robustness
evidence, deciding the ABI/package-manager path, and simplifying the adoption
surface without overclaiming.

## Review Basis

This review is based on the repository after Epic 10 closeout and PR #131
merge, including:

- public headers in `include/`
- implementation files in `src/`
- tests under `tests/`
- benchmarks, examples, Makefile, CMake, CI workflows, install scripts, and
  documentation
- prior reviews and todos under `docs/planning/EPIC_10/reviews/`
- Epic retrospectives through `docs/planning/EPIC_10/EPIC_10_RETROSPECTIVE.md`

Measured current signals:

| signal | observed value |
|---|---:|
| C/header files under `src`, `include`, `tests`, `benchmarks`, and `examples` | `183` |
| total lines across those C/header files | `119,638` |
| implementation files under `src` | `68` |
| public headers under `include` | `18` |
| tests under `tests` | `63` |
| benchmark sources under `benchmarks` | `18` |
| example sources under `examples` | `16` |
| final Epic 10 reviewed CMake registered tests | `54` |
| final Epic 10 Makefile/CMake parity | `54` vs `54` |
| final Epic 10 reviewed CTest result | `54 / 54` passed |
| final Epic 10 project-plan nominal hours | `2,906` |
| Epic 10 sprint artifacts | `272` |

Largest current source and proof owners:

| file | lines | review risk |
|---|---:|---|
| `tests/test_ldlt_csc.c` | `3,915` | giant direct-solver proof owner |
| `tests/test_integration.c` | `3,279` | broad mixed integration owner |
| `tests/test_qr.c` | `3,234` | giant QR proof owner |
| `tests/test_ldlt.c` | `3,006` | large LDLT proof owner |
| `tests/test_etree.c` | `2,962` | large etree/reorder proof owner |
| `tests/test_iterative.c` | `2,924` | giant iterative proof owner |
| `tests/test_svd.c` | `2,823` | giant SVD proof owner |
| `tests/test_graph.c` | `2,764` | large graph proof owner |
| `tests/test_chol_csc.c` | `2,554` | large CSC Cholesky proof owner |
| `tests/test_chol_csc_supernodal.c` | `2,504` | large supernodal proof owner |
| `tests/test_reorder_nd.c` | `2,304` | large ND proof owner |
| `tests/test_eigs.c` | `2,155` | large eigensolver proof owner |
| `src/sparse_ldlt_csc.c` | `2,095` | largest source hotspot |
| `src/sparse_lu_csr.c` | `1,594` | large compressed direct-solver hotspot |
| `src/sparse_ldlt.c` | `1,535` | large linked-list LDLT hotspot |
| `src/sparse_iterative.c` | `1,495` | large iterative hotspot |
| `src/sparse_qr.c` | `1,448` | large QR hotspot |
| `src/sparse_eigs.c` | `1,412` | large eigensolver hotspot |
| `src/sparse_svd.c` | `1,319` | large SVD hotspot |

## Efficiency Review

### Strengths

- The library covers a wide solver surface: LU, Cholesky, LDLT, QR, SVD,
  eigensolvers, iterative solvers, preconditioners, reordering, graph support,
  Matrix Market I/O, CSR/CSC paths, and repeated-run lifecycles.
- Compressed-first workflows are now visible in README, examples, docs, and
  implementation paths while the mutable matrix shell remains a compatibility
  owner.
- CSC Cholesky/LDLT and CSR LU dispatch paths reduce linked-list overhead for
  larger direct workloads.
- Reorder/fill and large-matrix guardrails exist and are explicitly scoped.
- Backend/runtime and benchmark surfaces are classified as local measurement
  context instead of overclaiming portable speed.

### Gaps

- The core public identity still starts with "orthogonal linked-list" in the
  README, while state-of-the-art sparse libraries are compressed-format-first
  in API, memory behavior, documentation, and implementation.
- The project has selected compressed dispatch paths, not a uniform
  compressed-first solve stack across all direct, iterative, eigensolver, SVD,
  reorder, and analysis operations.
- Performance evidence is deliberately not portable. That is truthful, but it
  means the project cannot claim competitive speed or backend maturity beyond
  local regression/sentinel context.
- Optional dense/runtime backend behavior exists but is not yet a full BLAS,
  LAPACK, SuiteSparse, GraphBLAS, or vendor backend contract.
- Parallelism is useful but partial. The product model does not yet give users
  a simple performance/runtime tuning story for OpenMP, thread counts, nested
  kernels, backend fallback, and deterministic behavior.
- There is no GPU, distributed-memory, block-sparse, or package-ecosystem
  acceleration story. That is fine as a non-claim, but it keeps the library
  below the state-of-the-art threshold.

## Maintainability Review

### Strengths

- Epic 10 made proof-owner and residual decisions explicit instead of hiding
  deferred debt.
- Source-list parity, CMake parity, dead-code checks, linting, and
  quality-review wrappers give maintainers a strong safety net.
- The repository has a rare amount of decision history, validation logs, and
  retrospective traceability.
- Several public docs now distinguish adoption guidance from maintainer proof.

### Gaps

- Giant tests remain the largest practical maintainability problem. The top
  seven test files total more than many complete C libraries.
- Large implementation files still mix algorithm logic, conversion, fallback,
  workspace, dispatch, error handling, and proof-owner history in ways that
  make local review expensive.
- Source comments and headers are often technically rich but too historical for
  first-use users. Some public headers explain sprint-era calibration in more
  depth than a caller needs.
- Build and validation surfaces are powerful but numerous. New maintainers must
  understand reviewed, supplemental, staged, local, CI-owned, and
  platform-specific lanes before changing support claims safely.
- Retrospectives are valuable but voluminous. Current product truth remains
  scattered across README, INSTALL, maintainer guide, solver selection,
  benchmark docs, Matrix Market docs, algorithm docs, and many sprint artifacts.

## Usability Review

### Strengths

- README, `docs/solver_selection.md`, `docs/tutorial.md`,
  `examples/README.md`, and `INSTALL.md` provide a credible adoption path.
- Examples cover direct solves, analysis/refactorization, compressed input,
  iterative solvers, eigensolvers, SVD, Matrix Market, and CMake consumption.
- Package docs are honest about static-first support and platform tiers.
- Error/status enums, options structs, progress/cancel callbacks, and
  observability fields give advanced C users meaningful control.

### Gaps

- The API is broad and low-level. A new user still faces many solver-specific
  structs, defaults, ownership rules, and matrix-state restrictions.
- Compressed-first is stronger than before but not yet the unmistakable "do
  this first" path in the API itself.
- Some APIs reject factored/reordered matrices, require identity permutations,
  or depend on caller-owned output buffers; these are valid C contracts but
  need simpler cookbook examples and consistency audits.
- There is no package-manager install path, language binding, or high-level
  wrapper for common application users.
- The public docs are truthful but still dense. State-of-the-art libraries
  usually offer a shorter beginner path plus deeper maintainer references.

## Documentation Review

### Strengths

- Documentation is unusually honest about non-claims, platform asymmetry,
  local benchmark interpretation, package limitations, and support tiers.
- Epic 10 produced strong retrospective and closeout documentation.
- Matrix Market, solver selection, benchmark, install, maintainer, and
  algorithm docs each have a clearer role than before.

### Gaps

- Documentation remains large and fragmented. It is easy for a maintainer to
  verify claims, but harder for a new user to quickly learn the current best
  path.
- `docs/algorithm.md` remains a mixture of current technical reference,
  historical measurements, implementation notes, and planning-era evidence.
- Benchmark artifacts are discoverable to maintainers, but generated benchmark
  report indexes are not yet surfaced as a polished documentation asset.
- Public headers sometimes carry deep benchmarking or historical rationale
  that would be better moved to maintainer docs.

## Coherence Review

### Strengths

- The repo has strong claim discipline. It does not pretend to have broad
  ecosystem parity, portable performance superiority, dynamic ABI guarantees,
  package-manager support, or symmetric platform parity.
- Epic 10 reduced contradictions between README, INSTALL, examples,
  benchmarks, and maintainer docs.
- The static-first package story and tiered platform support model are clear.

### Gaps

- The project still has multiple product identities:
  - orthogonal linked-list sparse matrix library;
  - compressed-first sparse solver library;
  - broad numerical methods workbench;
  - long-running planning/validation program.
- These identities can coexist, but Epic 11 should make the user-facing product
  identity simpler: compressed-first sparse solving with a compatible mutable
  shell and explicit proof boundaries.
- Some internal names and build targets are maintainer-precise but difficult
  for external users to interpret.

## Test Coverage and Assurance Review

### Strengths

- Test breadth is excellent: direct solvers, iterative methods, eigensolvers,
  SVD, reorderers, graph support, Matrix Market, install scripts, package
  consumers, fuzz/property lanes, OpenMP, edge cases, and integration paths.
- The strongest local reviewed baseline, CMake parity, and CI tiering are
  documented and active.
- External/dense-reference evidence exists for selected direct solvers and
  fixture-local residual/reconstruction evidence exists elsewhere.
- Dead-code and coverage tooling exist, with coverage kept supplemental rather
  than overclaimed.

### Gaps

- Tests are broad but concentrated. Failure localization and review burden are
  high in giant files.
- External oracle coverage is still uneven across solver families.
- Numerical corpus coverage is not yet a systematic product asset organized by
  conditioning, pattern, symmetry, definiteness, rank, scale, and failure mode.
- Windows remains a reviewed CMake subset with staged exclusions rather than a
  complete package/install/thread/fuzz/property parity story.
- Coverage remains supplemental and does not yet drive a reviewed architecture
  for untested branches or numerical corner cases.
- Performance regression governance is intentionally bounded. More structured
  local sentinels are needed before claims can become stronger.

## Packaging, ABI, and Platform Review

### Strengths

- Static-first install/export proof is real and documented.
- `pkg-config` and CMake downstream stories exist.
- Exact-version CMake package behavior is explicit.
- Linux/macOS/Windows support tiers are honest.

### Gaps

- No shared-library package support.
- No dynamic ABI compatibility guarantee.
- No package-manager support.
- No reviewed Linux install CI lane.
- No full reviewed macOS CMake install/export parity.
- No Windows install-validation parity.
- No Windows Makefile parity.
- Windows thread/fuzz/property proof remains staged.

## State-of-the-Art Assessment

The project can credibly claim to be a broad, well-tested, self-contained C
sparse linear algebra library with strong maintainer discipline and selected
product-grade surfaces. It cannot credibly claim to be state of the art without
qualification.

State-of-the-art sparse linear algebra libraries generally provide most of:

- compressed-format-first APIs and memory ownership;
- optimized direct solver backends with supernodal/multifrontal/vendor support;
- robust reordering and fill-reduction strategy guidance;
- broad external oracle and numerical corpus coverage;
- portable performance evidence and backend selection;
- mature shared/static packaging, ABI policy, and package-manager integration;
- language bindings or higher-level workflow surfaces;
- strong platform support contracts;
- large-matrix and numerical robustness campaigns.

`linalg_sparse_orthogonal` has real pieces of this, but Epic 11 should avoid
declaring state-of-the-art status. The right claim remains bounded product
maturity with explicit non-claims.

## Highest-Priority Epic 11 Gaps

1. Reduce the remaining largest proof-owner and source-boundary risks.
2. Convert selected residual source movements into proven extractions.
3. Build a systematic numerical oracle/corpus architecture across solver
   families.
4. Strengthen local performance regression sentinels without claiming portable
   speed.
5. Decide and implement or explicitly defer shared-library ABI and
   package-manager support.
6. Improve Windows/macOS install and staged validation parity where feasible.
7. Simplify public adoption docs and relocate historical/maintainer-only detail.
8. Clarify compressed-first as the product center across examples, docs, and
   high-value APIs.
9. Make benchmark, coverage, and dead-code reports easier to interpret as
   recurring quality artifacts.
10. Close Epic 11 with another evidence-bounded claim audit and residual queue.

