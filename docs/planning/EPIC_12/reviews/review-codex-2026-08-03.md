# Epic 12 Code Review - Codex - 2026-08-03

## Executive Verdict

`linalg_sparse_orthogonal` is a serious, broad, self-contained C sparse linear
algebra library with unusually strong validation discipline and planning
traceability. It now has credible direct, iterative, eigensolver, SVD,
reordering, graph, Matrix Market, benchmark, install, and documentation
surfaces. Epic 11 also made the product truth much cleaner: static-first
packaging is enforced, platform tiers are explicit, report interpretation is
bounded, and unsupported state-of-the-art claims are fenced.

It is still not a state-of-the-art sparse linear algebra library in the
SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, LAPACK, CHOLMOD/UMFPACK, KLU,
GraphBLAS, oneMKL, GPU, distributed, or package-ecosystem sense. The codebase
has moved from "feature breadth with validation debt" to "productizing library
with explicit support boundaries." The remaining gap is less about adding
every solver and more about closing several high-value gaps completely:

- numerical evidence is broad but still fixture-local and not corpus-grade;
- QR and partial-SVD residuals remain explicitly bounded rather than fully
  closed;
- generated reports are useful but not yet normalized into a stable index
  contract;
- performance evidence is local and sentinel-style rather than competitive or
  portable;
- shared-library, dynamic ABI, runtime-loader, and package-manager support are
  intentionally absent;
- platform confidence is tiered, with Linux strongest and macOS/Windows still
  narrower;
- the code and test base still contains giant ownership files that make review
  expensive;
- the public API is powerful but too low-level for many first-time users.

Epic 12 should not try to partially touch all of those gaps. It should close a
smaller set of gaps fully enough to earn stronger claims: a maintained
numerical corpus/oracle lane, a normalized report index with staleness gates,
fully closed QR/partial-SVD priority residuals, a concrete ABI/package decision
with proof, and promoted platform lanes where the support tier can genuinely
move.

## Review Basis

This review is based on the repository after Epic 11 closeout and PR #151
merge, including:

- public headers in `include/`
- implementation files in `src/`
- tests under `tests/`
- benchmarks and report scripts under `benchmarks/` and `scripts/`
- examples under `examples/`
- Makefile, CMake, pkg-config, install scripts, and CI workflows
- documentation under `README.md`, `INSTALL.md`, `docs/`, and planning
  retrospectives through `docs/planning/EPIC_11/EPIC_11_RETROSPECTIVE.md`

Measured current signals:

| Signal | Observed value |
|---|---:|
| C/header lines under `src`, `include`, `tests`, `benchmarks`, and `examples` | `123,327` |
| implementation `.c` files under `src` | `49` |
| private implementation headers under `src` | `20` |
| public headers/templates under `include` | `19` |
| test `.c` files under `tests` | `58` |
| test helper headers under `tests` | `11` |
| benchmark `.c` files under `benchmarks` | `16` |
| example `.c` files under `examples` | `15` |
| CMake registered tests on Unix-like CMake path | `57` |
| Windows reviewed CMake subset declared in CI | `54` |

Largest current source and proof owners:

| File | Lines | Risk |
|---|---:|---|
| `tests/test_qr.c` | `3,970` | giant QR proof owner, hard to localize failures |
| `tests/test_ldlt_csc.c` | `3,915` | giant compressed direct-solver proof owner |
| `tests/test_integration.c` | `3,279` | broad mixed integration owner |
| `tests/test_svd.c` | `3,029` | giant SVD proof owner |
| `tests/test_ldlt.c` | `3,006` | large LDLT proof owner |
| `tests/test_etree.c` | `2,962` | large etree/reorder proof owner |
| `tests/test_iterative.c` | `2,924` | giant iterative proof owner |
| `tests/test_graph.c` | `2,764` | large graph proof owner |
| `tests/test_chol_csc.c` | `2,554` | large CSC Cholesky proof owner |
| `tests/test_chol_csc_supernodal.c` | `2,504` | large supernodal proof owner |
| `tests/test_reorder_nd.c` | `2,304` | large nested-dissection proof owner |
| `tests/test_eigs.c` | `2,155` | large eigensolver proof owner |
| `src/sparse_ldlt_csc.c` | `2,095` | largest direct-solver implementation hotspot |
| `src/sparse_lu_csr.c` | `1,594` | large compressed LU hotspot |
| `src/sparse_ldlt.c` | `1,535` | large linked-list LDLT hotspot |
| `src/sparse_iterative.c` | `1,495` | large iterative hotspot |
| `src/sparse_qr.c` | `1,448` | large QR hotspot |
| `src/sparse_eigs.c` | `1,336` | large eigensolver hotspot |
| `src/sparse_svd.c` | `1,319` | large SVD hotspot |

## Efficiency Review

### Strengths

- The library has real algorithm breadth: LU, CSR LU, Cholesky, CSC Cholesky,
  LDLT, CSC LDLT, QR, SVD, partial SVD, CG, GMRES, MINRES, BiCGSTAB, ILU,
  ILUT, IC, eigensolvers, reordering, graph routines, Matrix Market I/O, and
  CSR/CSC conversion.
- Compressed-first paths are now visible in README, cookbook, examples, direct
  solver dispatch, install docs, and benchmark docs.
- The direct-solver lifecycle supports analyze-once / factor-many workflows,
  which is an important practical feature for repeated sparse solves.
- Local benchmark and sentinel infrastructure exists, and docs correctly avoid
  claiming portable timing superiority.
- OpenMP support exists for selected paths, and CI includes sanitizer,
  ThreadSanitizer, wall-check, dead-code, install, CMake, and package-contract
  lanes.

### Gaps

- The public product identity still starts from an orthogonal linked-list
  matrix object. State-of-the-art sparse libraries are usually compressed,
  block, graph, or distributed data-structure-first at both API and runtime
  levels.
- Compressed-first is not uniform. Some solver families still convert through
  mutable matrix shells, rely on linked-list compatibility semantics, or expose
  compressed behavior only as a dispatch path.
- Performance evidence is local and non-portable by design. That is honest,
  but it means the project cannot claim competitive performance against
  SuiteSparse, PETSc, Trilinos, Eigen, oneMKL, or vendor BLAS/LAPACK stacks.
- Runtime governance remains partial: OpenMP behavior, backend fallback,
  deterministic execution, nested parallelism, and environment-vs-typed option
  precedence are documented but not yet a simple product contract.
- There is no GPU, distributed-memory, block-sparse, GraphBLAS, out-of-core, or
  mixed-precision acceleration story.

## Maintainability Review

### Strengths

- Source-list checks, Make/CMake parity, format/lint wrappers, dead-code
  reports, package proofs, and CI tiers provide strong guardrails.
- Internal headers and helper files have grown around real ownership
  boundaries instead of purely cosmetic splits.
- Planning artifacts preserve rationale, residuals, validation commands, and
  non-claim decisions. That makes future audits unusually traceable.
- Public claims are disciplined. The project is generally clear when evidence
  is reviewed, supplemental, staged, local-only, or deferred.

### Gaps

- Giant tests remain a major bottleneck. The largest six test files total more
  than 21k lines and mix fixtures, helpers, oracle checks, regression cases,
  failure-mode cases, and historical proof context.
- Several implementation files still mix algorithm kernels, dispatch,
  conversion, allocation, diagnostics, progress/cancel behavior, and fallback
  logic.
- The validation surface is powerful but cognitively expensive. New
  maintainers must understand reviewed, supplemental, staged, local,
  platform-specific, package, dead-code, coverage, and report lanes before
  safely changing claims.
- Documentation and retrospectives are accurate but voluminous. Product truth
  is easier to audit than to learn.

## Usability Review

### Strengths

- README now provides an adoption map, quick start, workflow chooser, and clear
  pointers to cookbook, solver selection, install, examples, benchmarks, and
  maintainer docs.
- Examples cover the main workflows: basic solve, analyze/refactor, compressed
  input, iterative solves, eigensolvers, SVD, Matrix Market, and CMake
  consumption.
- Public APIs expose useful C-level control through option structs, error
  codes, progress/cancel callbacks, preconditioner callbacks, and reusable
  handles.
- Static-first installation is maintained through Make install, `pkg-config`,
  CMake install/export, downstream consumer examples, and package-contract
  tests.

### Gaps

- A new numerical user still has to choose among many low-level solver-specific
  structs, tolerance rules, matrix-state restrictions, ownership rules, and
  support-tier caveats.
- There is no high-level "solve this sparse system" front door that selects a
  reasonable path, reports a clear diagnostic, and preserves advanced escape
  hatches.
- There are no language bindings, package-manager packages, binary releases,
  or stable shared-library ABI. Adoption remains source/static-library centric.
- Some APIs require fresh copies after mutation/factorization, identity
  permutations, caller-owned buffers, or family-local helper knowledge. These
  are acceptable C contracts but raise first-use friction.

## Documentation Review

### Strengths

- README, INSTALL, tutorial, cookbook, solver-selection, Matrix Market,
  algorithm, algorithm-history, benchmark, and maintainer docs now have clearer
  roles after Epic 11.
- Documentation is unusually honest about non-claims and support tiers.
- Install and CI docs explain Linux reviewed ownership, macOS supplemental
  confidence, Windows reviewed CMake subset, Windows staged exclusions, and
  static-first packaging.

### Gaps

- Generated report indexes are not yet a stable user-facing or maintainer
  contract. Canonical benchmarks, sentinels, guardrails, coverage, dead-code,
  and oracle artifacts still have separate row meanings and freshness rules.
- `docs/algorithm.md` is better than before but remains long and dense. It is
  more of a technical reference than a concise product-facing algorithm guide.
- The maintainer guide is comprehensive but large. It should become an indexed
  operations manual with stable gates and report schemas rather than a growing
  narrative file.
- Public docs still have to repeat many non-claims because the implementation
  does not yet close ABI, package-manager, platform parity, external oracle,
  and portable-performance gaps.

## Coherence Review

### Strengths

- The repo is coherent about support boundaries. It does not pretend that
  static packaging equals dynamic ABI, local benchmark rows equal portable
  performance, or supplemental CI equals reviewed platform parity.
- The current product can be described truthfully as a self-contained,
  static-first C sparse linear algebra library with broad solver coverage and
  strong local validation.
- Epic 11 converted many ambiguous future promises into explicit residuals and
  non-claims.

### Gaps

- The project still has four competing identities:
  - orthogonal linked-list matrix library;
  - compressed-first sparse solver library;
  - broad numerical methods workbench;
  - long-running evidence/planning program.
- Those identities can coexist internally, but the public product should
  simplify around one primary promise: maintained sparse linear algebra
  workflows with explicit support tiers.
- State-of-the-art language should remain absent until the project has
  external comparison, corpus, package, ABI, platform, and performance proof
  comparable to mature libraries.

## Test Coverage Review

### Strengths

- The test suite is large and covers many solver families and failure modes.
- CMake currently registers 57 Unix-like tests, and Windows CI explicitly owns
  a narrower 54-test reviewed subset.
- The project has dense/external reference helper scripts for several direct
  and decomposition families.
- Sanitizer, ThreadSanitizer, dead-code, package install, CMake install/export,
  fast benchmark, coverage, and large-matrix guardrail workflows exist.

### Gaps

- Coverage is still not corpus-grade. Many correctness claims are fixture
  bounded rather than representative across matrix families, sizes,
  conditioning, sparsity patterns, rank, and failure modes.
- External oracle coverage is not broad enough to support ecosystem parity
  wording for QR, SVD, partial SVD, eigensolvers, iterative convergence, or
  direct solver robustness.
- Property/fuzz coverage is present but not uniformly reviewed across
  platforms. Windows still excludes staged POSIX/pthread-dependent lanes.
- Large tests make failures harder to triage and make future coverage
  expansion more expensive than it should be.
- Coverage percentage should remain an internal assurance tool, not a proof of
  behavioral completeness.

## State-of-the-Art Assessment

The project is not state of the art today. It is best understood as a
well-engineered, self-contained sparse linear algebra library with broad
features and strong local validation discipline.

To credibly move toward state-of-the-art status, the project would need at
least these capabilities:

1. A maintained numerical corpus with representative matrix taxonomy,
   deterministic expected outcomes, external oracles, and sustained CI/report
   integration.
2. Competitive benchmark methodology with reproducible environments,
   statistically meaningful comparisons, and explicit comparisons against
   mature libraries.
3. A stable package story beyond static source builds: shared-library ABI,
   loader behavior, versioning, binary artifacts, and package-manager channels.
4. Stronger platform parity across Linux, macOS, and Windows.
5. A cleaner primary user API with compressed-first workflows and high-level
   solver selection.
6. Continued source/test decomposition so maintainability can keep pace with
   evidence expansion.

Epic 12 can make meaningful progress by fully closing selected gaps. It should
not claim broad state-of-the-art status at closeout unless those proof
standards are actually met.

## Highest-Value Epic 12 Gap Closures

1. **Numerical corpus and oracle contract.**
   Build one sustained corpus lane with taxonomy, external/dense references,
   fixtures, skip semantics, and report interpretation.

2. **QR and partial-SVD residual closure.**
   Pick the highest-value QR and partial-SVD residual families and close them
   completely with tests, docs, and non-claim updates.

3. **Report normalization and freshness.**
   Normalize benchmark, sentinel, guardrail, coverage, dead-code, and oracle
   report metadata where row meaning allows it; add stale-report gates.

4. **Runtime/backend governance.**
   Convert local sentinel and runtime/backend behavior into a durable support
   contract with typed options, clear environment precedence, and reproducible
   report rows.

5. **Package/ABI/productization decision.**
   Either implement shared-library ABI support with proof or explicitly carry
   static-first-only as an enforced product decision with no ambiguity.

6. **Platform promotion.**
   Promote the highest-value macOS/Windows install or staged-test lanes only
   when CI proof and source portability work justify the support-tier change.

7. **API/adoption simplification.**
   Add a smaller high-level adoption front door and reduce public-doc density
   after the evidence and package decisions are settled.

