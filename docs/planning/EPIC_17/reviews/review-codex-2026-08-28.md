# Codex Project Review - 2026-08-28

## Scope

This review assesses `linalg_sparse_orthogonal` after Epic 16 landed on
`master`. It covers code efficiency, maintainability, usability,
documentation, coherence, test coverage, and readiness to be considered a
state-of-the-art sparse linear algebra library.

Reviewed evidence included:

- public headers under `include/`;
- implementation files under `src/`;
- tests, benchmarks, examples, scripts, and CI workflows;
- user-facing docs: `README.md`, `INSTALL.md`, `docs/api_reference.md`,
  `docs/maintainer_guide.md`, `benchmarks/README.md`;
- Epic 10 through Epic 16 retrospectives;
- the Epic 16 residual queue in
  `docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md`.

## Executive Assessment

The project has unusually strong evidence discipline for a small C sparse
linear algebra library. It has broad solver coverage, extensive tests,
benchmark drivers, local and hosted CI, static-first package validation,
documented support tiers, generated-report workflows, and explicit non-claims
that prevent marketing language from outrunning proof.

The main weakness is not feature count. The main weakness is that many claims
are still bounded, local, fixture-specific, or product-decision-limited. The
library is currently best described as a capable experimental/static-first C
sparse linear algebra library with strong local regression evidence. It is not
yet defensibly state of the art compared with SuiteSparse, PETSc, Trilinos,
Eigen/Spectra, SciPy, MKL/PARDISO, CHOLMOD, UMFPACK, MUMPS, or modern GPU and
distributed sparse stacks.

The most valuable next epic should close complete gaps, not chase a broad
claim. The highest-return complete closures are:

1. make Homebrew proof actually pass by resolving standalone license metadata;
2. promote one Windows-safe report freshness lane after obtaining PowerShell
   validation ownership;
3. add one bounded external comparison and one methodology-bound performance
   lane with real tolerances and hosted artifacts;
4. reduce one large implementation/test review surface;
5. simplify adoption and API documentation around stable public workflows;
6. finish with a calibrated state-of-the-art assessment rather than an
   unqualified state-of-the-art claim.

## Strengths

### Feature Breadth

The public surface covers a large set of sparse workflows:

- mutable orthogonal linked-list matrix shell;
- CSR/CSC import/export and compressed-first construction;
- LU, Cholesky, LDLT, QR, iterative Krylov solvers, SVD, and symmetric
  eigensolvers;
- direct repeated-run analyze/factor/solve lifecycle;
- iterative and eigensolver reusable handles;
- Matrix Market I/O;
- reordering, including RCM, AMD, COLAMD, and nested dissection;
- ILU/IC preconditioning;
- benchmarks, examples, and generated report tooling.

This is a serious amount of functionality for a C library.

### Evidence And Claim Governance

The project is disciplined about claim boundaries. The README, install docs,
maintainer guide, benchmark docs, API reference, and Epic 16 retrospective all
state that many surfaces are bounded or non-claims. That matters because the
feature list could otherwise imply broad solver correctness, package-manager
support, portable performance, ABI stability, Windows parity, or external
library parity that the evidence does not actually prove.

### Static-First Packaging

The static install surface is relatively coherent:

- Make and CMake install paths exist;
- `pkg-config` and exported CMake package metadata are validated;
- shared-library support is rejected explicitly rather than accidentally
  implied;
- Windows has a CMake-first install/downstream path.

This is a good product decision for a C project that does not yet have ABI
policy, symbol visibility, SONAME/install-name, DLL/import-library, or runtime
loader validation.

### Test Volume

The tree has substantial test volume. The current inventory includes 59 C test
files, 49 library source files, and 18 checked-in public headers. The largest
tests cover QR, LDLT CSC, integration, SVD, LDLT, etree, iterative solvers,
graph routines, Cholesky CSC, and reordering. The full Make test suite is part
of the regular quality gate, and CMake test-count parity is checked.

### CI And Tooling

The project has meaningful CI segmentation:

- Linux as the strongest reviewed source of truth;
- macOS reviewed Apple Clang and supplemental GCC coverage;
- Windows CMake/MSVC coverage;
- static package contract lanes;
- generated oracle and comparison freshness lanes;
- sanitizer, TSan, coverage, dead-code, format, lint, and CMake parity paths.

That is stronger than many small numerical C libraries.

## Shortcomings And Gaps

### 1. State-Of-The-Art Claim Is Not Earned

The project still lacks the evidence required for a state-of-the-art sparse
linear algebra claim.

Missing proof includes:

- broad comparison against named mature libraries and exact versions;
- representative matrix corpora beyond selected fixtures;
- numerical robustness analysis for ill-conditioned, singular, indefinite,
  rectangular, rank-deficient, and large-scale cases by solver family;
- performance methodology with thresholds, variance policy, hardware metadata,
  compiler flags, and repeated hosted artifacts;
- memory footprint and asymptotic scaling evidence;
- thread-scaling evidence beyond bounded paths;
- optional backend integration policy for BLAS/LAPACK/SuiteSparse-style dense
  or sparse kernels;
- production package distribution and ABI policy;
- long-term compatibility and release policy.

The current best claim is much narrower: selected workflows have local and
hosted evidence with explicit non-claims.

### 2. Efficiency Is Mixed

The orthogonal linked-list representation gives convenient row/column mutation
and traversal, but it is not the dominant high-performance representation for
modern sparse linear algebra. The project has added CSR/CSC paths for selected
direct kernels, but the public identity and many workflows still route through
linked-list storage.

Efficiency concerns:

- linked-list node layout has pointer-heavy memory overhead and poor cache
  locality;
- many solver paths convert into dense or compressed workspaces at runtime;
- QR currently uses dense column-major workspace for the default path, which
  limits large sparse usability;
- performance evidence is mostly local, selected, threshold-free, or
  sentinel-scoped;
- OpenMP claims are narrow and do not prove scalable parallel sparse kernels;
- there is no clear memory budget model exposed to users for each solver path;
- several benchmarks are too slow for routine CI and are compile-only or
  local opt-in.

The code has promising kernels, but state-of-the-art sparse libraries win on
data layout, sparse kernel specialization, supernodal/multifrontal algorithms,
ordering integration, memory locality, threading, and mature backend
selection. This project is not there yet.

### 3. Maintainability Hotspots Remain

The codebase is large and several files are hard to review end-to-end.
Measured line counts show major hotspots:

- `tests/test_qr.c`: 3970 lines;
- `tests/test_ldlt_csc.c`: 3469 lines;
- `tests/test_integration.c`: 3279 lines;
- `tests/test_svd.c`: 3029 lines;
- `tests/test_ldlt.c`: 3006 lines;
- `tests/test_etree.c`: 2962 lines;
- `tests/test_iterative.c`: 2929 lines;
- `tests/test_graph.c`: 2764 lines;
- `tests/test_chol_csc.c`: 2554 lines;
- `tests/test_chol_csc_supernodal.c`: 2504 lines;
- `src/sparse_ldlt_csc.c`: 2095 lines;
- `src/sparse_lu_csr.c`: 1594 lines;
- `src/sparse_ldlt.c`: 1535 lines;
- `src/sparse_iterative.c`: 1503 lines;
- `src/sparse_qr.c`: 1448 lines;
- `src/sparse_eigs.c`: 1336 lines;
- `src/sparse_svd.c`: 1319 lines.

Large source files are not inherently wrong, but they raise review risk when
they mix algorithm logic, workspace ownership, validation, dispatch, telemetry,
environment overrides, and cleanup policy in one owner.

Additional maintainability concerns:

- global or process-wide test overrides exist in several areas and need strict
  restore discipline;
- implementation files contain historical sprint comments that explain why
  code exists but also make current ownership harder to scan;
- helper extraction has improved one LDLT CSC test cluster, but many other
  giant tests remain unpartitioned;
- Makefile and CMake source lists must stay synchronized manually, though a
  guard exists;
- API structs rely on trailing-field compatibility and extensive documentation
  instead of a more explicit versioned options story.

### 4. Usability Is Improving But Still Heavy

The README now gives a clear adoption map, and examples cover many workflows.
However, the user-facing story remains dense because the library exposes many
solver families, many caveats, and many non-claims.

Usability gaps:

- beginners must choose among linked-list, CSR, CSC, Matrix Market,
  one-shot direct, repeated direct, iterative handles, eigensolver handles,
  and benchmarks;
- result/status interpretation differs across solver families;
- some APIs return silent zero for invalid reads, which is documented but can
  surprise users;
- there is no single minimal installed-consumer tutorial that proves the
  package from a fresh external project in the docs;
- diagnostics are available but not uniformly surfaced through one result
  vocabulary across direct, iterative, QR/SVD, and eigensolvers;
- no release notes or compatibility policy gives users confidence about
  upgrading.

### 5. Documentation Is Strong But Overloaded

Documentation quality is high, but the docs carry a lot of governance weight.
Many pages repeat non-claims about ABI, package managers, Windows, generated
HTML, performance, and state-of-the-art status.

Documentation gaps:

- claim boundaries are correct but verbose and distributed;
- generated HTML remains local-only, so the browsable API experience is weak
  unless a user runs Doxygen locally;
- docs do not provide a compact "production readiness" matrix separated from
  maintainer planning history;
- some public headers are very long because they include full workflow
  guidance, caveats, and compatibility notes;
- historical sprint references are useful for maintainers but too prominent
  for users trying to adopt the library.

### 6. Test Coverage Is Broad But Not Yet Deep Enough For Broad Claims

The test suite is large and valuable, but the evidence is still bounded.

Coverage gaps:

- selected external comparisons are fixture-local;
- allocation-failure proof covers selected lanes, not broad allocator paths;
- fuzzing exists but does not replace broad property-based differential
  testing;
- coverage is supplemental and tree-mutating, not a routinely enforced
  source-coverage contract;
- Windows report freshness is deferred;
- package-manager provider proof is blocked;
- performance correctness and performance regression policy remain mostly
  separate from solver correctness.

The suite is good for regression confidence. It is not yet a comprehensive
numerical validation campaign.

### 7. Platform And Packaging Are Product-Limited

The static-first decision is coherent, but it limits adoption relative to
state-of-the-art libraries.

Remaining gaps:

- no package-manager support;
- Homebrew proof blocked by missing standalone license metadata;
- no shared library;
- no dynamic ABI policy;
- no runtime loader validation;
- Windows Makefile and `pkg-config` execution parity remain non-claims;
- Windows generated report freshness remains deferred.

These are acceptable product constraints, but they prevent broad production
distribution claims.

### 8. Coherence Has Improved But Still Depends On Guardrails

The project has strong guardrails, but coherence is maintained by many scripts,
workflow comments, and planning records. That is a sign of maturity, but also
a sign that the surface is complex.

Coherence risks:

- claim language can drift across README, INSTALL, maintainer guide, API
  reference, package docs, benchmark docs, and planning artifacts;
- selected report metadata has one manifest authority, but generated reports
  still require careful regeneration and freshness interpretation;
- docs and tests contain many historical sprint references that are hard to
  separate from current product truth;
- public API grouping is broad, and headers sometimes serve as both exact API
  reference and narrative workflow guide.

## Ability To Become State Of The Art

The project has a path toward a credible specialized library, but not toward a
general state-of-the-art claim within one epic.

To credibly move toward that level, Epic 17 should focus on selected complete
closures:

- prove one package-manager path end-to-end;
- promote one Windows report freshness lane;
- publish one real hosted comparison/performance lane with methodology;
- reduce one or two large review surfaces;
- unify diagnostics and adoption docs for the most-used workflows;
- add a final calibrated state-of-the-art assessment with concrete remaining
  blockers.

After that, the project could claim stronger product readiness and selected
workflow evidence. It still should not claim broad state-of-the-art sparse
linear algebra without several more epics of numerical, performance, packaging,
parallel, and interoperability work.

## Prioritized Gap List

| Priority | Gap | Why it matters | Close completely by |
| ---: | --- | --- | --- |
| 1 | Standalone license metadata and Homebrew proof | Unblocks the selected package-manager proof path and improves adoption. | Sprint 188 |
| 2 | PowerShell validation and Windows report freshness | Turns the strongest retained Windows report residual into proof or a refreshed decision. | Sprint 190 |
| 3 | One hosted comparison/performance lane with methodology | Moves one state-of-the-art-adjacent claim from local/fixture-only to reviewed evidence. | Sprint 192 |
| 4 | One large source/test maintainability cluster | Reduces review risk in the areas most likely to hide defects. | Sprint 193 |
| 5 | User adoption and API simplification | Makes the large capability set easier to consume without overclaiming. | Sprint 194 |
| 6 | Coverage and allocation-failure breadth for one selected owner | Converts one bounded reliability weakness into proof. | Sprint 195 |
| 7 | Final claim calibration and residual publication | Prevents Epic 17 from overstating what it earned. | Sprint 196 |

## Recommended Epic 17 Thesis

Epic 17 should not try to make the project state of the art in the general
sense. It should make the project materially more credible by fully closing a
small number of high-value gaps: one package proof, one Windows report proof,
one external evidence lane, one maintainability cluster, one adoption/API
simplification pass, and one final calibrated closeout.

