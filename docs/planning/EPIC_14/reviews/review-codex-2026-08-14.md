# Epic 14 Code Review - Codex - 2026-08-14

## Scope

This review assesses the current `master` baseline after Epic 13 and the
creation of branch `planning/epic-14`. The review covers code efficiency,
maintainability, usability, documentation, coherence, test coverage, and the
project's ability to justify state-of-the-art sparse linear algebra claims.

Evidence reviewed:

- public entry points in `README.md`, `INSTALL.md`, `docs/api_reference.md`,
  and `docs/maintainer_guide.md`;
- build and install surfaces in `Makefile`, `CMakeLists.txt`,
  `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, and CI workflows;
- public headers under `include/`;
- implementation, tests, benchmarks, scripts, corpus metadata, and previous
  Epic 11-13 retrospectives;
- Epic 13 residual queue in
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day11-residual-queue-publication.md`.

## Executive Assessment

The project is much stronger than a typical experimental C sparse library. It
has broad solver coverage, a serious test suite, explicit support-tier
language, maintained install/export proof, corpus machinery, comparison
scripts, benchmark/report tooling, and careful non-claim discipline. The
project is especially mature in how it distinguishes reviewed, supplemental,
local-only, hosted, advisory, and deferred evidence.

It is not yet state of the art in the way SuiteSparse, PETSc, Trilinos, Eigen,
SciPy-backed workflows, or vendor BLAS/LAPACK-backed packages can claim mature
ecosystem status. The strongest defensible position is that this is a
self-contained C sparse linear algebra library with meaningful educational and
medium-scale engineering depth, bounded QR and partial-SVD corpus evidence,
static-first packaging, and unusually explicit evidence governance.

Epic 14 should therefore avoid broad rewrites. The best return is to close the
highest-friction remaining gaps completely:

1. generated API reference publication;
2. hosted promotion of selected local-only oracle/comparison evidence;
3. one wider QR comparison family;
4. one partial-SVD comparison publication;
5. Windows package parity decision for the remaining non-claims;
6. public-header/API coherence cleanup;
7. benchmark/report publication discipline and claim recalibration.

## Strengths

- **Broad algorithm surface:** LU, CSR LU, Cholesky, CSC Cholesky, LDL^T, QR,
  SVD, partial SVD, iterative solvers, eigensolvers, graph/reordering, Matrix
  Market I/O, CSR/CSC conversion, preconditioners, and analysis/refactor
  workflows are all represented.
- **Clear static-first product boundary:** install tests and CI prove the
  static archive package shape across Linux, macOS, and Windows CMake
  consumers, while shared-library and dynamic ABI claims are explicitly
  rejected.
- **Strong evidence hygiene:** docs repeatedly prevent local generated reports,
  optional data, benchmark rows, or comparison rows from being inflated into
  unsupported public claims.
- **Cross-platform maturity:** Windows now has reviewed CMake CTest coverage
  and reviewed CMake install/downstream validation; Linux and macOS carry
  stronger Make/CMake/package lanes.
- **Corpus and comparison architecture:** QR and partial-SVD have maintained
  corpus lanes, normalized report indexes, source-controlled expected rows,
  and selected external comparison infrastructure.
- **Quality infrastructure:** CI includes Makefile quality, CMake parity,
  sanitizers, dead-code reporting, package install proofs, and Windows CTest
  count enforcement.

## Main Gaps

### G1: Generated API HTML Is Not Published

`docs/api_reference.md` says `make docs` writes generated HTML under
`docs/api/html/`, and `docs/maintainer_guide.md` defines freshness rules for
that tree. `git ls-files docs/api` currently returns no tracked generated HTML.
That means the user-facing API reference is still header-first only.

Impact:

- users cannot browse a checked-in generated reference;
- reviewers cannot tell whether current header comments generate clean pages;
- API doc freshness remains a recurring residual.

Closure target:

- run `make docs`;
- capture and triage warnings;
- verify generated page coverage against intended public headers;
- either commit generated HTML with clear source-header-first wording or
  explicitly choose no checked-in HTML and add a recurring guard that preserves
  that decision.

### G2: Generated Evidence Remains Mostly Local-Only

The selected oracle and comparison freshness gates write ignored artifacts
under `build/`. The maintainer guide documents commands such as
`make report-index-oracle-freshness` and `make report-index-comparison-freshness`,
but Epic 13 closed with E13-R06 still open: selected QR, partial-SVD, oracle,
and comparison rows remain local-only.

Impact:

- PR reviewers cannot rely on hosted logs for all claim-bearing generated
  evidence;
- local evidence is real but easy to let go stale or omit;
- state-of-the-art or ecosystem-parity claims remain impossible.

Closure target:

- promote a small selected freshness bundle into hosted CI;
- upload or summarize generated artifacts;
- fail the lane when claim-bearing rows are stale, missing, or failing;
- keep optional and advisory families outside the claim surface.

### G3: External Comparison Breadth Is Too Narrow

The external comparison harness is real but currently centered on one selected
QR minimum-norm fixture family. It does not yet establish broad QR, SVD,
SuiteSparse, SciPy, NumPy, LAPACK, Eigen, PETSc, or Trilinos parity.

Impact:

- competitive claims stay unsupported;
- comparisons can show harness viability but not ecosystem standing;
- users choosing between libraries still lack enough evidence.

Closure target:

- add exactly one bounded QR comparison family and one bounded partial-SVD
  comparison family;
- publish fixture, metric, tolerance, dependency, skip/defer, and non-claim
  semantics;
- normalize rows and freshness checks.

### G4: Performance Evidence Is Methodology-Limited

The benchmark system is substantially better than ad hoc timing. It has
canonical surfaces, sentinel rows, labels, manifests, support tiers, and
warnings against portable performance interpretation. That is good discipline,
but it also confirms no portable performance claim exists.

Impact:

- performance is observable but not product-grade competitive evidence;
- threshold-free rows cannot detect regressions except where a specific
  guardrail exists;
- hardware/compiler variance policy is not enough for superiority claims.

Closure target:

- publish a methodology-bound performance report for selected canonical rows;
- define what is thresholded, what is report-only, and what is advisory;
- keep "portable performance superiority" rejected unless recurring
  cross-machine evidence exists.

### G5: Windows Package Parity Still Has Explicit Non-Claims

Windows has reviewed CMake coverage and CMake install/downstream validation.
The docs still correctly reject Windows Makefile parity and Windows
`pkg-config` execution parity.

Impact:

- Windows users have a narrower downstream story than Unix users;
- package docs must repeatedly explain the CMake-first boundary;
- future package changes can accidentally imply parity.

Closure target:

- choose whether Windows `pkg-config` or Makefile parity is product scope;
- either implement a reviewed proof lane or strengthen the retained non-claim
  with tests/docs that make the decision unambiguous.

### G6: Shared Library And Dynamic ABI Are Deferred

`CMakeLists.txt` rejects `BUILD_SHARED_LIBS=ON`, and docs explain missing
export/import, symbol visibility, SONAME/install-name, DLL/import-lib,
runtime-loader, and ABI policy work.

Impact:

- many downstream packaging ecosystems expect shared-library support;
- public structs and exported declarations are not governed by an ABI promise;
- package-manager distribution remains blocked.

Closure target:

- do not attempt full shared-library support in Epic 14 unless product scope
  changes;
- instead, close static-first ambiguity and preserve a precise future ABI gate.

### G7: Public API Is Broad And Still Verbose For New Users

The adoption docs have improved, but the public C API remains broad and
solver-family specific. Exact contracts live in many headers, with generated
HTML absent and some header cleanup still deferred.

Impact:

- first-use success depends on following the guide sequence;
- users may land in declarations before understanding workflow boundaries;
- public-header comments are part of the product surface and need continued
  consistency.

Closure target:

- finish another declaration-preserving header cleanup batch;
- strengthen cross-links between README, tutorial, cookbook, solver-selection,
  API reference, and headers;
- preserve a zero-signature-drift gate.

### G8: Maintainability Is Good But Large-File Pressure Is Real

The implementation and test tree is large. A line-count scan shows roughly
134k lines across source, headers, tests, scripts, examples, and benchmarks.
Several test owners exceed 2k to 3.9k lines, including `tests/test_qr.c`,
`tests/test_ldlt_csc.c`, `tests/test_integration.c`, `tests/test_svd.c`, and
`tests/test_ldlt.c`. Scripts such as `scripts/run_corpus_oracle.py` and
`scripts/normalize_report_index.py` are also large.

Impact:

- focused proof-owner tests are increasingly important;
- source-list duplication between Makefile and CMake remains a review hazard;
- broad test files are harder to reason about during targeted fixes.

Closure target:

- keep adding focused proof owners for new corpus/comparison work;
- do not expand already-large monolithic files unless the touched behavior
  truly belongs there;
- preserve existing source-list consistency checks.

### G9: Test Coverage Is Broad But Not Complete

The test suite is extensive and cross-platform, with C tests, corpus schema
checks, external dense reference helpers, fuzz/property coverage, sanitizers,
dead-code checks, install validation, and CI count enforcement. Remaining
limits are mostly claim-boundary limits rather than missing basic tests.

Impact:

- broad numerical correctness, broad external parity, broad performance, and
  package ecosystem coverage remain unproved;
- generated coverage/dead-code/report outputs are not all claim-bearing;
- optional dependency and optional data lanes require skip/defer discipline.

Closure target:

- promote selected generated rows into hosted evidence;
- add bounded comparison families;
- avoid pretending coverage percentage or artifact existence equals solver
  correctness.

### G10: Coherence Depends On Claim Discipline

The docs are coherent because they consistently state what is not supported.
That coherence is fragile: generated reports, benchmarks, package metadata,
and support-tier wording all touch related claims.

Impact:

- a small wording change can overclaim platform, ABI, performance, or external
  parity;
- planning artifacts are useful but large enough to bury the current truth;
- closeout audits need to remain a formal part of each epic.

Closure target:

- keep claim/non-claim audits mandatory in Epic 14;
- route public claims to recurring evidence;
- publish a final state-of-the-art assessment that rejects any unsupported broad
  claim.

## State-Of-The-Art Assessment

The project should not claim state-of-the-art sparse linear algebra status
today.

What it can claim:

- self-contained C sparse linear algebra with a broad educational/product
  surface;
- maintained static-first install/export for Unix and Windows CMake consumers;
- reviewed cross-platform CI with clear support tiers;
- fixture-local correctness evidence for selected QR and partial-SVD families;
- local benchmark/report infrastructure with explicit non-performance-claim
  boundaries;
- one narrow external comparison lane.

What it cannot claim yet:

- broad external-library parity;
- portable performance superiority;
- package-manager ecosystem maturity;
- shared-library or dynamic ABI stability;
- broad Windows package parity;
- broad solver-family numerical robustness across industrial matrices;
- state-of-the-art sparse direct, iterative, spectral, or SVD performance.

Epic 14 should improve the public posture by closing high-friction evidence
and documentation gaps, not by inflating the claim language.
