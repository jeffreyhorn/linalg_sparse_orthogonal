# Code Review

**Date:** 2026-06-15  
**Reviewer:** Codex  
**Scope:** Full-project review of the current `linalg_sparse_orthogonal` tree
after Epic 6 closeout, with emphasis on code efficiency, maintainability,
usability, documentation, coherence, test coverage, and its ability to be a
state-of-the-art sparse linear algebra library.

## Executive Summary

The repository is now a serious single-node sparse linear algebra library with
real engineering discipline:

- the algorithm surface is broad,
- the reviewed validation contract is real and repeatedly exercised,
- the project has explicit packaging, benchmark, and maintainer-policy
  surfaces,
- and Epic 6 closed most of the gaps that previously made the library feel
  partially productized.

I did **not** find an obvious release-blocking correctness flaw in this review.
The strongest remaining problems are no longer missing core solvers or absent
quality discipline. They are structural product-maturity gaps:

- the core public matrix model is still too conversion-heavy and mutation-heavy
  for a best-in-class sparse library,
- the capability surface is still narrower than a state-of-the-art library
  should be,
- advanced configuration is still split between typed options and
  process-global compatibility/debug env vars,
- the backend/performance story is still only in an early bounded form,
- platform/release maturity remains asymmetric,
- and the public/test/documentation surfaces still carry too much sprint-era
  history and too many permanent review burdens.

**Bottom-line assessment:** this is now an excellent engineering-grade,
real-valued, single-node sparse linear algebra library. It is **not yet
state of the art as a shipping sparse linear algebra library**. The remaining
distance is mainly in:

- core data/product model maturity,
- capability breadth and portability,
- backend/performance depth,
- release/platform convergence,
- and public-surface/test-surface simplification.

## Dimension Assessment

| Dimension | Assessment | Notes |
|---|---|---|
| Code efficiency | Strong but uneven | Many hot paths are real and benchmarked, but the primary public matrix model and backend layer still cap peak efficiency. |
| Maintainability | Good, with concentrated debt | The repo is much cleaner than in prior epics, but several source and test hotspots remain large and history-heavy. |
| Usability | Improved but still expert-leaning | One-shot vs explicit lifecycle is clearer, but advanced callers still need too much product-memory and copy discipline. |
| Documentation | Rich but over-dense | The docs are informative, but public surfaces still contain too much sprint history and policy detail. |
| Coherence | Better than Epic 5, not fully simplified | README/tutorial/examples/benchmarks now agree, but compatibility and policy seams are still visible to callers. |
| Test coverage | Broad and serious | Coverage breadth is strong, but the proof architecture is still monolithic in places and not equally reviewed on all platforms. |
| State-of-the-art readiness | Not there yet | The library is credible and mature, but still narrower and more self-contained than best-in-class sparse libraries. |

## Strengths

1. **Algorithm breadth is already substantial.**
   The repository covers:
   - LU
   - Cholesky
   - LDL^T
   - QR
   - SVD
   - iterative Krylov solvers
   - symmetric sparse eigensolvers
   - multiple reorderings
   - analyze/factor/refactor workflows

2. **The validation contract is unusually explicit for a library of this size.**
   The maintained reviewed baseline, CMake parity path, dead-code workflow,
   install/package regression surfaces, and canonical benchmark/reporting
   surfaces are all named and documented.

3. **Epic 6 closed major productization gaps successfully.**
   The repo now has:
   - typed high-value analysis/reorder options
   - a clearer direct repeated-run lifecycle
   - a first bounded backend-aware dense-kernel seam
   - a canonical maintained benchmark surface
   - a truthful packaging/platform story
   - a final coherent public product story across README/tutorial/examples/tests

4. **The repository is honest about its limits.**
   It no longer hides staged exclusions or platform asymmetries behind generic
   “cross-platform support” language. That honesty is a strength.

## Findings

### 1. High: the primary public matrix/product model is still the strongest ceiling on both usability and performance

Representative references:

- `README.md`
- `include/sparse_matrix.h`
- `src/sparse_matrix.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`
- `src/sparse_qr.c`

Why this matters:

- The library’s public center of gravity is still `SparseMatrix`, an
  orthogonal linked-list / cross-linked structure with slab-pool allocation.
- That structure is flexible and useful for construction and mutation, but the
  library’s highest-value numeric paths increasingly route through compressed
  working formats:
  - CSC Cholesky
  - CSC LDL^T
  - CSR LU
  - dense/blocked helper paths
- The result is a persistent split:
  - the public matrix model is mutation-friendly and pointer-heavy
  - the fast numeric model is compressed, cache-friendlier, and often built
    only after one or more conversions

What falls short:

- best-in-class sparse libraries are usually built around CSC/CSR-style data
  ownership and explicit factor/workspace ownership, not a conversion-heavy
  linked-list public center
- the linked-list model keeps paying in:
  - conversion cost
  - cache locality
  - matrix-copy discipline
  - caller mental overhead
  - hidden compatibility state

Evidence in the current tree:

- `README.md` still leads with the orthogonal linked-list structure
- `include/sparse_matrix.h` is one of the largest public headers (`583` lines)
  and still carries both representation detail and compatibility contract
  detail
- the CSC/CSR direct paths still convert back into the linked-list-facing
  public matrix story after numeric work

Primary improvement:

- re-tier the product model so construction/editing, compressed working
  formats, and explicit factor/workspace ownership have cleaner long-term
  boundaries
- reduce how much of the public performance story depends on round-tripping
  through `SparseMatrix`

### 2. High: the capability surface is still materially narrower than a state-of-the-art sparse library

Representative references:

- `README.md`
- `include/sparse_types.h`
- `include/sparse_eigs.h`

Why this matters:

- The README explicitly states two major capability ceilings:
  - `idx_t` is `int32_t`
  - only real `double` values are supported
- The eigensolver surface is limited to symmetric problems.
- Those are reasonable product boundaries for a strong engineering library, but
  they are still substantial market/capability limits for a library aspiring
  to state-of-the-art status.

What falls short:

- `int32_t` indices cap dimensions and nonzero counts at roughly 2.1 billion
- real-only scalar support excludes complex-valued numerical workloads
- eigensolver breadth is strong for symmetric problems, but there is no
  unsymmetric sparse eigensolver story

Evidence in the current tree:

- `README.md:729-732` documents the `int32_t` and real-only limits directly
- `include/sparse_types.h` hardcodes `typedef int32_t idx_t;`
- `README.md` and `include/sparse_eigs.h` document only the symmetric sparse
  eigensolver surface

Primary improvement:

- introduce a real index-width and scalar-type roadmap, with at least one
  bounded end-to-end modernization path rather than leaving these as static
  caveats

### 3. High: the advanced configuration story is still split between typed options and process-global compatibility/debug env vars

Representative references:

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_graph.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_refine.c`
- `src/sparse_reorder_amd_qg.c`
- `include/sparse_svd.h`

Why this matters:

- Epic 6 Phase 1 successfully moved the highest-value analysis/reorder controls
  onto typed options.
- But the repo still contains a large residual env-var surface for:
  - FM heuristics
  - debug/profile toggles
  - SVD low-rank routing
  - compatibility overrides

What falls short:

- advanced policy still partly depends on ambient process state
- multiple workflows cannot cleanly hold independent policy at once
- debug/profile and residual tuning knobs still bleed into permanent code paths
- the typed configuration story is now “good but incomplete,” not “finished”

Evidence in the current tree:

- `src/sparse_reorder_nd.c` still parses multiple `SPARSE_ND_*` overrides
- `src/sparse_graph*.c` still reads a large `SPARSE_FM_*` family
- `src/sparse_reorder_amd_qg.c` still reads `SPARSE_QG_PROFILE`
- `include/sparse_svd.h` still documents `SPARSE_SVD_LOWRANK_OUTER`

Primary improvement:

- finish a second configuration modernization pass:
  - public typed where caller-meaningful
  - internal typed where algorithm-policy-local
  - compatibility env-var lane kept narrow and explicit
  - debug/profile controls moved out of the product-facing narrative

### 4. High: the performance architecture is still only a first-phase modernization, not a full state-of-the-art backend story

Representative references:

- `CMakeLists.txt`
- `README.md`
- `include/sparse_cholesky.h`
- `src/sparse_dense.c`
- `benchmarks/README.md`

Why this matters:

- Epic 6 landed a real first backend-aware dense-kernel seam.
- But the repo still does not look like a modern sparse performance stack with
  flexible backend strategy:
  - the package surface is explicitly static-first
  - the backend-aware dense-kernel seam is narrow and Cholesky-local
  - compile-time heuristics still drive important behavior
  - progress callback parity and broader backend wiring remain incomplete

What falls short:

- no general optional BLAS/LAPACK-style dense backend story
- no broader dense-kernel abstraction across QR/SVD/LDL^T
- limited threading/task model beyond selected OpenMP surfaces
- canonical benchmark reporting exists, but not deeper statistical or
  longitudinal performance governance

Evidence in the current tree:

- `CMakeLists.txt` explicitly forces a maintained static archive surface
- `include/sparse_cholesky.h` documents the bounded backend-aware CSC
  supernodal lane
- `src/sparse_dense.c` still exposes test-only override seams rather than a
  broader runtime/backend strategy
- `README.md` still documents callback gaps for CSC supernodal Cholesky /
  LDL^T and thick-restart Lanczos

Primary improvement:

- deepen the backend/performance architecture beyond one lane:
  - optional dense backend integration
  - shared policy/selection surfaces
  - callback parity on backend-aware paths
  - more systematic performance measurement and regression interpretation

### 5. Medium: platform, packaging, and release maturity are truthful but still too asymmetric for a top-tier library

Representative references:

- `CMakeLists.txt`
- `INSTALL.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `README.md`

Why this matters:

- The repo has a real install/export story and an honest platform story.
- But the platform/release contract is still visibly uneven:
  - Windows reviewed subset is smaller (`50` tests vs `53`)
  - Windows still excludes `test_fuzz`
  - Windows still does not claim reviewed install validation
  - macOS dead-code remains staged
  - the maintained release shape is static-only

What falls short:

- state-of-the-art libraries usually present a more converged
  install/validation/release matrix
- the current story is reliable, but still reads more like a carefully managed
  developer-distribution contract than a broadly deployable library surface

Primary improvement:

- either converge the remaining platform asymmetries or tighten the maintained
  release contract further so those asymmetries stop leaking into the public
  product story

### 6. Medium: maintainability debt is now concentrated in a smaller set of large source and test files, but that debt is still substantial

Representative references:

- `src/sparse_ldlt_csc.c` (`2130` lines)
- `src/sparse_iterative.c` (`1985` lines)
- `src/sparse_lu_csr.c` (`1665` lines)
- `src/sparse_qr.c` (`1563` lines)
- `src/sparse_chol_csc.c` (`1536` lines)
- `src/sparse_eigs.c` (`1534` lines)
- `tests/test_chol_csc.c` (`4608` lines)
- `tests/test_ldlt_csc.c` (`3680` lines)
- `tests/test_qr.c` (`3197` lines)
- `tests/test_graph.c` (`2900` lines)
- `tests/test_iterative.c` (`2802` lines)
- `tests/test_svd.c` (`2766` lines)

Why this matters:

- Epic 6 reduced the worst code hotspots, but many permanent files are still
  large enough to make safe review and future refactor work expensive.
- The largest debt is now concentrated rather than diffuse, which is good, but
  it is still operationally costly.

What falls short:

- giant files still combine:
  - behavior proof
  - local helpers
  - old chronology
  - compatibility checks
  - performance-path assertions
- some of the hardest future changes will still land in files that are
  difficult to review incrementally

Primary improvement:

- continue bounded decomposition and helper extraction on the remaining top
  source and test hotspots

### 7. Medium: the test surface is broad, but its architecture and portability still lag its raw coverage breadth

Representative references:

- `tests/test_framework_optin.c`
- `tests/test_fuzz.c`
- `tests/test_integration.c`
- `tests/test_reorder_nd.c`
- `.github/workflows/windows-ci.yml`
- `README.md`

Why this matters:

- The repository is not under-tested. That is not the problem.
- The remaining problems are:
  - monolithic proof owners
  - sprint-era naming still embedded in permanent test binaries
  - platform-reviewed coverage gaps
  - limited external oracle/differential proof relative to the breadth of the
    solver surface

What falls short:

- Windows still excludes `test_fuzz`
- several `test_sprint*` integration binaries remain part of the permanent
  reviewed surface
- many giant tests still read as chronology-preserving archives instead of
  modern behavior-grouped proof surfaces

Primary improvement:

- reorganize the permanent proof surface around behavior families rather than
  sprint history, and converge reviewed platform coverage on the highest-value
  assurance lanes

### 8. Medium: public documentation and public headers are coherent, but still too history-heavy and policy-dense

Representative references:

- `README.md`
- `benchmarks/README.md`
- `examples/README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `include/sparse_matrix.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_cholesky.h`

Why this matters:

- The docs are rich and technically good.
- But many public and semi-public surfaces still carry too much sprint-era
  explanation, compatibility history, and policy detail.

Evidence in the current tree:

- `README.md` still has `31` visible `Sprint`/`Day` references
- `benchmarks/README.md` still has `5`
- `INSTALL.md` still has `4`
- public headers and permanent tests also retain many sprint-history notes

What falls short:

- stable public docs should read like product/reference documentation, not like
  a compressed delivery log
- public headers should capture durable call-site contracts, not long-form
  sprint rationales and performance diary notes

Primary improvement:

- de-chronologize the public surface:
  - user docs for usage
  - maintainer docs for policy
  - planning docs for history/rationale

### 9. Medium: benchmark governance is better than before, but still not strong enough to support best-in-class performance claims

Representative references:

- `benchmarks/README.md`
- `make bench-canonical-report`
- `.github/workflows/ci.yml`

Why this matters:

- Epic 6 created a canonical benchmark surface and a threshold-free report
  path. That is good.
- But the current performance-governance story is still intentionally modest:
  - `bench-fast` is a bounded runtime signal
  - `bench-canonical-report` is artifact capture, not a true regression gate
  - portable timing truth is still largely manual and reviewer-mediated

What falls short:

- no longitudinal baseline store
- no machine-class-aware statistical comparison layer
- no stronger relationship between canonical benchmarks and release-level
  performance claims

Primary improvement:

- build a second-phase benchmark governance layer that can support longitudinal
  baselines, trend interpretation, and narrowly justified thresholding

### 10. Medium: advanced workflow usability is real but not yet fully productized

Representative references:

- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- `examples/example_analysis.c`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

Why this matters:

- The advanced repeated-run surfaces now exist and are better documented than
  before.
- But the adoption path still leans heavily on one-shot examples and
  maintainer-policy reading for advanced workflows.

What falls short:

- there is no equivalent small runnable example for:
  - iterative handle reuse
  - eigensolver handle reuse
  - typed configuration beyond the direct-analysis lane
- some advanced capabilities remain “documented and benchmarked” more than
  “discoverable and naturally teachable”

Primary improvement:

- add a small number of advanced adoption examples and simplify the advanced
  workflow teaching story so callers do not have to infer too much from
  benchmarks, tests, or headers

## State-of-the-Art Assessment

Today the library is best described as:

- **very strong engineering-grade sparse linear algebra library**
- **credible research/production crossover codebase**
- **not yet a state-of-the-art general sparse library**

Main reasons it is not yet state of the art:

1. The primary public matrix model is still linked-list-first rather than
   compressed-format/product-workflow-first.
2. Capability breadth is still constrained by:
   - 32-bit indices
   - real-only scalar support
   - symmetric-only eigensolvers
3. Backend/performance modernization is real but still early-phase.
4. The release/platform/packaging contract is still narrower than a
   best-in-class library surface.
5. Too much public/test/header content still exposes sprint history and
   compatibility archaeology.

## Recommended Epic 7 Priorities

1. Rework the product model around cleaner compressed-format and explicit
   factor/workspace ownership boundaries.
2. Finish configuration modernization for the remaining env-var-driven lanes.
3. Deepen the backend/performance architecture and the benchmark-governance
   story together.
4. Converge the platform/release/packaging surface further.
5. Attack the remaining large source/test hotspots and permanent
   sprint-history debt.
6. Start a real capability-expansion path for:
   - index width
   - scalar-type breadth
   - and higher-end library posture

## Bottom Line

Epic 6 closed the productization backlog very effectively. The project is now
coherent, well tested, and measurably maintained.

But the next frontier is different from the last two epics. The main work is
no longer “finish wiring the public story.” It is:

- reduce structural ceilings imposed by the core matrix/product model,
- deepen the performance and release architecture,
- simplify the public and proof surfaces,
- and expand the capability envelope enough that the library can credibly
  compete as a state-of-the-art sparse linear algebra system rather than only
  as an unusually strong self-contained C implementation.
