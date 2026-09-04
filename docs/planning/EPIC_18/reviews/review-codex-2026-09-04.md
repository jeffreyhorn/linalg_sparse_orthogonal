# Codex Project Review - 2026-09-04

## Scope

This review assesses `linalg_sparse_orthogonal` after Epic 17 landed on
`master`. It covers code efficiency, maintainability, usability,
documentation, coherence, test coverage, and readiness to be considered a
state-of-the-art sparse linear algebra library.

Reviewed evidence included:

- public headers under `include/`;
- implementation files under `src/`;
- tests, benchmarks, examples, scripts, package templates, and CI workflows;
- user-facing docs: `README.md`, `INSTALL.md`, `docs/api_reference.md`,
  `docs/solver_selection.md`, `docs/cookbook.md`, and `benchmarks/README.md`;
- maintainer evidence and claim-governance docs:
  `docs/maintainer_guide.md`;
- Epic 10 through Epic 17 retrospectives;
- the Epic 17 residual queue in
  `docs/planning/EPIC_17/EPIC_17_RESIDUAL_QUEUE.md`.

## Executive Assessment

The project has continued to improve its evidence discipline. Epic 17 closed
several selected gaps: PowerShell validation ownership, one guarded Windows
Cholesky workflow path, one bounded QR external comparison family, one Linux
selected benchmark freshness lane, one QR review-surface reduction, clearer
adoption/API documentation, and one selected symbolic Cholesky
allocation-failure proof. The project now has 49 library source files, 18
checked-in public headers, 59 `make test` C sources plus the separate smoke
test source, 16 benchmark drivers, explicit support/readiness tiers, and
unusually detailed non-claim governance.

The central limitation remains the same: the library is strong for an
experimental static-first C sparse linear algebra project with broad local
regression coverage, but it is not yet defensibly state of the art. The best
current claim is that selected workflows have bounded evidence with clear
support boundaries. A state-of-the-art claim would require much broader
comparative correctness, robustness, performance, packaging, ABI, and
platform evidence than the repository currently contains.

Epic 18 should therefore close a small number of high-value gaps completely
instead of spreading partial work across every remaining weakness. The most
valuable complete closures are:

1. resolve package-manager/Homebrew support by adding approved license
   metadata and proving the selected provider workflow;
2. promote or formally re-close the selected Windows Cholesky freshness lane
   with hosted evidence and manifest metadata;
3. add one more allocation-failure proof owner using the Sprint 195 pattern;
4. reduce one more large QR or direct-solver review surface;
5. extend selected benchmark freshness to one additional hosted platform;
6. promote the Windows QR incompatible comparison only if MSVC/CMake evidence
   supports it;
7. make a clear product decision for generated API HTML publication;
8. finish with calibrated release/state-of-the-art positioning rather than
   an unqualified marketing claim.

## Strengths

### Feature Breadth

The library exposes a large numerical surface for a small C project:

- orthogonal linked-list sparse matrix storage;
- CSR/CSC import/export and compressed-first construction;
- LU, Cholesky, LDLT, QR, iterative Krylov solvers, SVD, and symmetric
  eigensolvers;
- direct analyze/factor/refactor/solve lifecycle;
- reusable iterative and eigensolver handles;
- Matrix Market I/O;
- RCM, AMD, COLAMD, and nested-dissection reordering;
- ILU and IC preconditioning;
- examples, benchmarks, generated corpus reports, and selected external
  comparisons.

This breadth is a real strength. It also increases the need for rigorous
claim boundaries because no single test family proves the whole surface.

### Evidence And Claim Governance

The project is unusually disciplined about separating supported, validated,
hosted-evidence, local-only, guarded-workflow, deferred, and not-claimed
states. README, INSTALL, maintainer guidance, benchmark docs, corpus docs,
and planning artifacts repeatedly prevent unsupported promotion of package,
Windows, performance, ABI, release, and state-of-the-art claims.

This makes the project more trustworthy than many experimental numerical
libraries. The evidence story is verbose, but the honesty is valuable.

### Static-First Packaging

The static install path is coherent and validated:

- Make install and `pkg-config` metadata are tested on Unix-like systems;
- CMake install/export and downstream consumption are tested;
- Windows owns a CMake-first install/downstream route;
- shared-library requests are rejected rather than implicitly supported;
- package-manager support remains guarded until proof evidence exists.

### CI, Tooling, And Reports

The CI surface is substantial:

- Linux reviewed CMake parity and static package contract lanes;
- Linux supplemental runtime, sanitizer, and benchmark-fast paths;
- macOS reviewed Apple Clang compile/CMake/sanitize paths plus supplemental
  GCC;
- Windows MSVC CMake test and install/downstream lanes;
- selected hosted oracle/comparison freshness lanes;
- selected hosted benchmark freshness;
- PowerShell validation ownership;
- Doxygen coverage and local-only generated API guards;
- source-list, helper, package, report, and allocation-failure guards.

The generated report framework is a strong foundation for future evidence
publication if more targets are selected and promoted carefully.

### Test Volume

The test inventory is large: 59 `make test` C sources plus the separate smoke
test source and many Python/shell guards. Large areas have focused coverage:
QR, SVD, LDLT, CSC backends, eigensolvers, graph routines, package install,
report freshness, API docs, and selected allocation failures.

## Shortcomings And Gaps

### 1. State-Of-The-Art Claim Is Still Not Earned

The project does not yet support an unqualified state-of-the-art sparse
linear algebra claim.

Missing evidence includes:

- broad comparison against named mature libraries and exact versions;
- representative sparse matrix corpora beyond selected fixtures;
- solver-family robustness studies for ill-conditioned, singular,
  indefinite, rectangular, rank-deficient, and large-scale problems;
- performance methodology with thresholds, variance policy, hardware
  metadata, compiler flags, warmup/repeat rules, and hosted artifacts across
  platforms;
- memory-footprint and asymptotic scaling evidence;
- thread-scaling evidence beyond bounded OpenMP paths;
- production package-manager distribution;
- shared-library ABI policy and compatibility testing;
- release compatibility and deprecation policy.

The current state-of-the-art posture should remain a non-claim.

### 2. Efficiency Is Limited By Data Layout And Sparse Kernel Depth

The orthogonal linked-list representation is useful for mutation and
bidirectional traversal, but it is not the data layout used by the fastest
modern sparse numerical kernels. The project has added CSR/CSC paths for
selected kernels, but many workflows still pay linked-list traversal,
conversion, pointer-chasing, or dense-workspace costs.

Specific concerns:

- pointer-heavy node storage hurts cache locality and memory footprint;
- QR still has dense-workspace limitations for large sparse problems;
- selected direct backends exist, but broad sparse supernodal/multifrontal
  performance evidence is limited;
- OpenMP support is narrow and does not prove scalable sparse kernels;
- benchmark evidence is selected and threshold-free, not a performance
  regression contract;
- users do not get a compact memory-budget table by solver and matrix
  shape.

### 3. Maintainability Hotspots Remain Large

Current line-count hotspots include:

- `tests/test_ldlt_csc.c`: 3469 lines;
- `tests/test_integration.c`: 3279 lines;
- `tests/test_etree.c`: 3131 lines;
- `tests/test_qr.c`: 3040 lines;
- `tests/test_svd.c`: 3029 lines;
- `tests/test_ldlt.c`: 3006 lines;
- `tests/test_iterative.c`: 2929 lines;
- `tests/test_graph.c`: 2764 lines;
- `tests/test_chol_csc.c`: 2554 lines;
- `tests/test_chol_csc_supernodal.c`: 2504 lines;
- `tests/test_reorder_nd.c`: 2304 lines;
- `src/sparse_ldlt_csc.c`: 2095 lines.

The project has started reducing review surfaces, but many files still mix
fixtures, helpers, ownership policy, regression assertions, and broad
scenario coverage. This increases review risk and makes targeted changes
harder than they need to be.

### 4. Usability Is Better But Still Dense

The README and INSTALL routing are much clearer than in earlier epics, but
the project still asks users to understand many choices:

- linked-list vs CSR/CSC vs Matrix Market inputs;
- one-shot direct vs repeated direct lifecycle;
- direct solvers vs iterative handles vs eigensolver handles;
- support/readiness status vs proof-owner detail;
- local-only vs hosted-evidence vs guarded-workflow report status;
- benchmark measurement vs correctness evidence.

The support/readiness matrix helps, but there is still no compact "I have
this problem shape, use exactly this minimal code" quick-reference table
covering the most common production-like workflows.

### 5. Documentation Is Strong But Heavy

Documentation is extensive and generally accurate, but it carries a lot of
historical and claim-governance burden. This has two costs:

- public docs can feel intimidating to new users;
- maintainers must keep many repeated non-claim phrases synchronized.

The strongest long-term fix is not to remove boundaries, but to centralize
them more cleanly: public docs should route users to the current support
truth, while maintainer docs should own the detailed proof and residual
semantics.

### 6. Test Coverage Is Broad, But Some Gaps Are Still Explicit

Important residual coverage gaps remain:

- package-manager/Homebrew proof is blocked by missing approved root license
  metadata;
- selected Windows Cholesky freshness still needs hosted evidence plus
  selected manifest promotion;
- Windows QR incompatible freshness remains local-only;
- Windows/macOS selected benchmark freshness is not owned;
- allocation-failure proof is selected-owner only;
- optional NumPy/SciPy package baselines are deferred context, not pass
  evidence;
- broad external-library parity remains unproved;
- generated API HTML is local-only and not published;
- shared-library and dynamic ABI behavior remain deferred.

### 7. Coherence Risk: The Evidence System Is Distributed

The project now has many evidence owners: Makefile targets, CMake lists,
workflow YAML, Python validators, shell guards, generated manifests,
benchmark docs, corpus docs, maintainer guidance, public docs, sprint
artifacts, and project-plan status tables. The system is coherent today, but
it is easy to introduce drift when a change touches only one layer.

The next phase should add selective consolidation and stronger guardrails
instead of more unconnected documentation.

## State-Of-The-Art Assessment

The project is not currently a state-of-the-art sparse linear algebra
library. It is a capable and increasingly well-governed sparse linear algebra
C library with strong selected evidence.

Compared with mature ecosystems such as SuiteSparse, PETSc, Trilinos, Eigen,
SciPy, MKL/PARDISO, MUMPS, ARPACK-derived stacks, and modern GPU/distributed
libraries, the project lacks:

- comparable sparse kernel maturity;
- broad external differential correctness;
- production package distribution;
- shared-library ABI policy;
- performance publication and thresholds;
- broad platform parity;
- long-running robustness evidence;
- large public corpus validation;
- release compatibility process.

The most honest goal for Epic 18 is to make several selected support claims
fully true, not to claim broad parity.

## Highest-Value Epic 18 Opportunities

1. **Package-manager/Homebrew proof completion.** This is the clearest
   user-facing product gap and has one explicit blocker: approved license
   metadata.
2. **Selected Windows Cholesky freshness promotion.** The guarded workflow
   exists; complete closure requires hosted evidence review and manifest
   promotion.
3. **Additional allocation-failure proof.** The Sprint 195 pattern is
   reusable and would materially improve reliability evidence.
4. **Additional review-surface reduction.** One more large QR/direct-solver
   cluster can be made reviewable with focused guards.
5. **Windows/macOS selected benchmark freshness.** One additional hosted
   platform lane would improve methodology evidence without claiming portable
   performance.
6. **Windows QR incompatible comparison promotion.** This should be promoted
   only if MSVC/CMake proof and artifact normalization are solid.
7. **Generated API publication decision.** Either publish a hosted generated
   API artifact with freshness semantics or deliberately strengthen the
   local-only decision.
8. **Claim and release calibration.** Final documentation should keep broad
   state-of-the-art, release, ABI, package, platform, performance, and parity
   claims unpromoted unless the exact evidence exists.
