# Code Review - 2026-06-24

## Scope

This review assesses the current `linalg_sparse_orthogonal` tree across:

- code efficiency
- maintainability
- usability
- documentation quality
- cross-surface coherence
- test coverage
- state-of-the-art sparse linear algebra readiness

The assessment is based on the live post-Epic-8 `master` tree, including:

- core source files under `src/`
- public headers under `include/`
- tests, examples, benchmarks, and install/export proof
- build and package surfaces (`Makefile`, `CMakeLists.txt`, `sparse.pc.in`)
- CI workflows
- public support docs
- Epic 8 sprint/epic closeout material

## Baseline

The strongest current maintained validation baseline is still the Epic 8 close
state:

- `make quality-review-full`
- reviewed CMake parity:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - reviewed CMake `ctest` = `53 / 53`
- maintained install/export proof:
  - `bash tests/test_install.sh` = `13` passed, `0` failed
  - `bash tests/test_cmake_install.sh` = `15` passed, `0` failed,
    `0` skipped
- canonical maintained benchmark/reporting proof:
  - `make bench-canonical-report`

The project is therefore entering Epic 9 from a disciplined and well-proven
engineering baseline, not from a broken or drifting one.

## Executive Verdict

This is a serious and unusually disciplined C sparse linear algebra codebase.
It is far better documented, better tested, and more honest about its limits
than a typical small scientific-computing repository.

It is still **not a state-of-the-art sparse linear algebra library**.

The strongest reasons are structural:

1. the public/core product model is still anchored on an orthogonal
   linked-list shell instead of a compressed-first compute model
2. the dense/backend performance ceiling is still bounded by a scalar builtin
   implementation plus a narrow optional Accelerate lane
3. the capability surface is still materially narrower than leading sparse
   numerical libraries
4. concurrency, threading, and runtime scalability are still partial and
   family-local rather than product-wide
5. documentation and code surfaces still leak too much sprint-era chronology
   into permanent product surfaces
6. maintainability hotspots remain large enough to slow future work
7. packaging and cross-platform product maturity remain intentionally
   asymmetric

Bottom-line assessment:

- engineering rigor: high
- product maturity: medium
- performance ceiling: medium-low
- maintainability: medium
- usability for expert users: medium
- usability for new adopters: medium
- documentation truthfulness: high
- documentation clarity: medium
- test coverage breadth: high
- state-of-the-art competitiveness: not yet

## What The Project Does Unusually Well

Before the gaps, it is important to mark the real strengths:

1. **Validation culture is excellent.**
   The repo has a strong reviewed baseline, explicit Make/CMake parity,
   install/export proof, and a maintained benchmark/reporting surface.

2. **Algorithm breadth is already large for a single C library.**
   The project ships direct solvers, iterative solvers, SVD, symmetric sparse
   eigensolvers, preconditioners, reordering, and a sizeable benchmark/test
   surface.

3. **The codebase is unusually truthful about its limits.**
   The docs do not fake shared-library maturity, fake platform parity, or fake
   capability breadth the code does not actually ship.

4. **Proof ownership is strong.**
   The repository has `53` test binaries and about `1822` named test cases
   spread across direct, iterative, eigensolver, graph/reorder, install, and
   integration surfaces.

5. **The Epic 8 closeout materially moved the repo.**
   The project is more compressed-first on touched paths, has a real bounded
   external SPD comparison lane, has a sharper package contract, and has a
   cleaner front-door adoption path than it did before Sprint 80.

These strengths are important because they mean Epic 9 should be about
structural product convergence, not about inventing basic engineering hygiene
from scratch.

## Repository Snapshot

### Highest-signal implementation hotspots

| file | lines | review meaning |
|---|---:|---|
| `src/sparse_ldlt_csc.c` | 2694 | still the single largest mixed-role direct-family implementation owner |
| `src/sparse_iterative.c` | 1854 | large solver-family implementation and lifecycle surface |
| `src/sparse_lu_csr.c` | 1665 | large family-local direct-solver implementation |
| `src/sparse_qr.c` | 1563 | large algorithm-family implementation |
| `src/sparse_ldlt.c` | 1535 | large linked-list LDL^T owner |
| `src/sparse_eigs.c` | 1534 | large eigensolver orchestration surface |
| `src/sparse_svd.c` | 1319 | large dense/sparse algorithm surface |
| `src/sparse_matrix.c` | 1297 | still a major shell + mutation + utility owner |
| `src/sparse_chol_csc.c` | 1279 | still a large orchestration/completion hotspot |

### Highest-signal giant-test hotspots

| file | lines | review meaning |
|---|---:|---|
| `tests/test_chol_csc.c` | 4987 | very broad family-local proof owner |
| `tests/test_ldlt_csc.c` | 3680 | large direct-family proof surface |
| `tests/test_qr.c` | 3234 | large algorithm-family proof owner |
| `tests/test_integration.c` | 3197 | major public lifecycle/parity owner |
| `tests/test_etree.c` | 2962 | large symbolic/reordering proof surface |
| `tests/test_graph.c` | 2925 | large graph/reordering proof surface |
| `tests/test_ldlt.c` | 2921 | large linked-list direct-family proof surface |
| `tests/test_iterative.c` | 2841 | large solver-family proof owner |
| `tests/test_reorder_nd.c` | 2340 | reviewed runtime long-pole proof owner |

### Support-surface density

| file | lines | review meaning |
|---|---:|---|
| `README.md` | 1113 | improved front door, but still very dense |
| `Makefile` | 908 | broad hand-maintained build/test/benchmark workflow surface |
| `docs/maintainer_guide.md` | 727 | authoritative but still history-heavy |
| `docs/tutorial.md` | 473 | secondary user path with overlap risk |
| `CMakeLists.txt` | 416 | full second build topology and package surface |
| `benchmarks/README.md` | 399 | clear benchmark owner, but another distinct narrative surface |
| `INSTALL.md` | 315 | install surface is explicit and strong, but still another major owner |

These numbers do not automatically mean the code is poor. They do mean the
remaining complexity and future review cost are concentrated and easy to
locate.

## Findings

### 1. Critical: the public/core product model is still the main efficiency ceiling

**Evidence**

- `README.md` still introduces the library as an orthogonal linked-list sparse
  matrix library.
- `include/sparse_matrix.h` keeps the public matrix shell centered on mutable
  row/column linked structure.
- `src/sparse_matrix.c` still owns node-pool allocation, linked insert/remove,
  and much of the matrix utility surface.
- Epic 8 improved compressed-first touched paths, but the final Epic 8 closeout
  still explicitly describes the result as "less linked-list-first," not as a
  compressed-first product model.

**Why this matters**

This design remains a real product differentiator for pedagogy and mutation
flexibility. It is also the single biggest reason the library does not yet
look like a top-tier sparse compute product. Modern high-end sparse libraries
are usually compressed-first at the public product center, with mutable build
surfaces bounded around that. Here the relationship is still inverted:

- linked-list shell = public center
- compressed CSC/CSR paths = internal compute accelerators

That split can be made to work, but it creates persistent costs:

- extra conversion/publication complexity
- weaker locality and higher allocator pressure on shell-centric workflows
- a conceptual split between what users touch first and what the fast kernels
  actually want

**Gap**

The library still reads like a linked-list sparse matrix product with
compressed compute subsystems attached, not like a compressed sparse numerical
product with a bounded mutable shell.

### 2. Critical: the dense/backend ceiling is still too low for a top-tier performance claim

**Evidence**

- `src/sparse_dense.c` still owns the builtin dense GEMM/GEMV/factor/solve
  helpers in scalar C.
- Epic 8 widened only one bounded optional Accelerate-backed lane for touched
  direct-family dense kernels.
- `README.md`, `benchmarks/README.md`, and the Epic 8 retrospective all still
  describe backend work as intentionally bounded and optional.
- There is no broad portable BLAS/LAPACK-class backend lane, no vendor-neutral
  runtime backend matrix, and no multithreaded dense-kernel contract across
  the main direct paths.

**Why this matters**

Sparse direct performance depends heavily on the dense panel/update kernels in
the supernodal paths. A state-of-the-art sparse library normally has one or
more of:

- a mature BLAS/LAPACK-backed path
- vendor-kernel integration
- broader portable backend dispatch
- multithreaded dense kernels
- platform-spanning accelerated paths

This project instead has:

- strong backend-aware architecture discipline
- one bounded Darwin-only optional acceleration slice
- a careful but narrow builtin scalar fallback

That is honest and well engineered. It is not yet competitive with the backend
ceiling expected from a state-of-the-art sparse numerical library.

**Gap**

The codebase has backend architecture, but not backend maturity.

### 3. High: the capability surface is still materially narrower than leading sparse libraries

**Evidence**

- `include/sparse_types.h` still keeps `sparse_scalar_t` real-only.
- the public scalar/index modernization from Sprint 83 widened ownership
  semantics, but intentionally did not widen into broad complex or
  mixed-precision support.
- `README.md` still bounds reusable iterative handles to `CG`, `GMRES`, and
  `MINRES`.
- `README.md` still bounds sparse eigensolvers to the symmetric side only.
- Epic 8 explicitly closed with capability breadth still listed as a bounded
  non-claim.

**Why this matters**

State-of-the-art sparse libraries usually compete through some combination of:

- complex scalar support
- mixed precision
- broader eigensolver coverage
- broader reusable solver/workspace lifecycles
- richer large-index and ABI maturity

This project improved the capability story's honesty and structure. It did not
yet remove the main capability limits.

**Gap**

The capability contract is clearer and better factored than before, but still
too narrow for a top-tier sparse linear algebra claim.

### 4. High: concurrency and runtime scalability are still partial, family-local, and internally complicated

**Evidence**

- `README.md` advertises OpenMP for SpMV and Lanczos MGS, not broad
  product-wide parallel sparse compute.
- `README.md` also documents that many factorization and permutation surfaces
  remain single-threaded or caller-disciplined.
- `src/sparse_reorder_nd_internal.h` explicitly documents process-wide or
  current-thread internal controls that are not thread-safe and not ABI-safe.
- `src/sparse_ldlt_csc_internal.h` still has thread-safety caveats on some
  backend-selection internals.
- the final reviewed close baseline still carries a large runtime long pole in
  `tests/test_reorder_nd.c`.

**Why this matters**

The project has good selective parallelism and good honesty about where it is
safe. But state-of-the-art sparse libraries are expected to have a cleaner and
more uniform story around:

- threading model
- backend parallelism
- runtime scalability
- thread-safe configuration and instrumentation

The current situation is narrower and more operationally delicate:

- some parallelism is real
- some thread safety is optional or caller-disciplined
- some internal tuning/instrumentation controls are intentionally not
  production-grade

**Gap**

The library has meaningful parallel and concurrency features, but not a mature
product-wide parallel/runtime model.

### 5. High: maintainability hotspots remain large enough to slow future work

**Evidence**

- the top source hotspots remain between roughly `1279` and `2694` lines
- the top proof hotspots remain between roughly `2340` and `4987` lines
- the remaining largest files are not only large; many are mixed-role owners
  that combine orchestration, policy, algorithm detail, and helper logic

**Why this matters**

Epic 8 materially improved maintainability, but the repo still has too many
surfaces where reviewability, defect isolation, and onboarding cost are higher
than they should be. The largest examples are exactly the families the repo is
most likely to extend again:

- LDL^T CSC
- iterative solvers
- QR
- eigensolvers
- graph/reorder proof

This affects both correctness velocity and usability of the codebase for new
contributors.

**Gap**

The hotspot map is better than before, but still too concentrated for an
ambitious long-lived sparse library.

### 6. High: documentation and code surfaces still leak too much sprint-era chronology

**Evidence**

- `README.md` still contains many public-facing sprint references and detailed
  historical notes.
- `docs/maintainer_guide.md` still contains many "after Sprint X" style
  anchors.
- `include/` and `src/` still contain many historical sprint references in
  comments.
- there are still many sprint-named tests:
  - `tests/test_sprint4_integration.c`
  - `tests/test_sprint5_integration.c`
  - `tests/test_sprint29_integration.c`
  - and many more
- `tests/test_reorder_nd.c`, `tests/test_ldlt_csc.c`, and other family-local
  proof owners still contain large historical comment blocks tied to sprint
  chronology.

**Why this matters**

This is now one of the biggest coherence and usability issues in the repo.
Epic 8 improved the public front door, but the permanent product surfaces still
often read like a high-quality planning archive rather than like a finished
technical product.

That creates several costs:

- new users must parse historical context they do not need
- new contributors must reverse-engineer which sprint notes are durable
  technical rationale and which are just change history
- public headers and tests carry historical naming that obscures actual
  product ownership

**Gap**

The repository is highly documented, but still not documented enough in a
product-oriented, history-light form.

### 7. High: build, packaging, and workflow topology remain more manual and duplicated than a mature product should want

**Evidence**

- `Makefile` manually enumerates library sources, tests, benchmarks, and many
  workflow targets.
- `CMakeLists.txt` duplicates the library and test topology separately.
- both surfaces remain first-class product paths.
- `CMakeLists.txt` still keeps the package static-only even when
  `BUILD_SHARED_LIBS=ON` is requested.
- workflows remain intentionally asymmetric:
  - Linux = strongest reviewed truth
  - macOS = narrower reviewed lane plus supplemental package confidence
  - Windows = reviewed CMake-first subset only

**Why this matters**

The repo deserves credit for proving both build surfaces and for documenting
their asymmetry honestly. The problem is long-term product cost:

- topology duplication
- workflow duplication
- source-list maintenance burden
- larger CI surface
- reduced downstream packaging maturity

This is appropriate for a careful research-grade library. It is still weaker
than what a more mature sparse product would usually ship.

**Gap**

The build and package architecture is disciplined, but still too manual and
too duplicated for the product maturity target implied by "state of the art."

### 8. Medium: test coverage is broad and disciplined, but still fragmented and operationally heavy

**Evidence**

- `53` maintained test binaries
- about `1822` named `RUN_TEST(...)` invocations in the `tests/` tree
- broad coverage across install/export, graph/reorder, direct families,
  iterative solvers, eigensolvers, examples, and integration
- reviewed runtime still heavy:
  - reviewed total = `375.43 sec`
  - `test_reorder_nd` = `215.72 sec`
- Windows still excludes some proof surfaces from its reviewed subset

**Why this matters**

The breadth is a major strength. The fragmentation is the weakness:

- sprint-named tests obscure ownership
- some proof surfaces are giant family-local binaries
- reviewed runtime concentration remains non-trivial
- the cross-platform reviewed subset is still narrower than the Linux truth

This is therefore not a "coverage is weak" finding. It is a "coverage is
strong but still expensive, fragmented, and unevenly productized" finding.

**Gap**

The project has high test breadth, but still needs proof-surface
consolidation, runtime reduction, and naming/ownership cleanup.

### 9. Medium: usability has improved, but the user path still spans too many large surfaces

**Evidence**

- `README.md` = `1113` lines
- `docs/tutorial.md` = `473` lines
- `examples/README.md` = `218` lines
- `INSTALL.md` = `315` lines
- `benchmarks/README.md` = `399` lines

**Why this matters**

Epic 8 improved the front door substantially. But the product still asks users
to move across several long surfaces to form a full mental model:

- README for first adoption
- examples README for next-step map
- tutorial for fuller workflow
- INSTALL for package/install detail
- benchmark docs for retained workflow/performance proof

That split is truthful, but it is still somewhat heavy for first-time users,
especially because the library has many solver families and workflow modes.

**Gap**

Usability is no longer the biggest problem, but the product still lacks a
small, crisp, unified mental model for common user journeys.

### 10. Medium: the external-comparison and benchmark story is still too bounded for a top competitive claim

**Evidence**

- Epic 8 closed with one maintained bounded external SPD comparison lane on
  two retained fixtures.
- the runtime comparison lane remains bounded to the touched Sprint 86 slice
  and intentionally does not become a broad performance claim.
- the final Epic 8 retrospective explicitly keeps broader external comparison
  depth in the residual queue.

**Why this matters**

For a state-of-the-art claim, the project needs more than internal quality and
one bounded external differential lane. It needs a stronger comparison
package around:

- solver correctness breadth
- runtime behavior on representative classes
- reorder/fill-quality comparison
- product-shape comparison with established sparse stacks

This repo is finally in a position to do that honestly. It just has not done
enough of it yet.

**Gap**

The comparison story is now credible, but still too narrow to establish
best-in-class or state-of-the-art standing.

## Category Assessment

| category | assessment | explanation |
|---|---|---|
| Efficiency | Medium-low | strongest ceilings remain linked-list-first public model, narrow dense/backend acceleration, and a still-large reorder/ND runtime long pole |
| Maintainability | Medium | disciplined ownership and artifacts, but too many large mixed-role implementation and proof owners remain |
| Usability | Medium | front door is much better, but full workflow understanding still spans too many large surfaces |
| Documentation | Medium-high | highly truthful and deep, but too history-heavy and too sprint-referential in permanent surfaces |
| Coherence | Medium | support surfaces are better split than before, but the product story still has shell-vs-compute, one-shot-vs-reuse, and history-vs-product tensions |
| Test coverage | High | breadth is excellent, install/export proof is strong, but proof topology remains fragmented and runtime-heavy |
| State-of-the-art readiness | Not yet | real strengths are present, but core storage, backend maturity, capability breadth, packaging symmetry, and competitive comparison depth are still below top-tier sparse-library expectations |

## Bottom-Line Gap Summary

Epic 9 should not start from a generic "improve everything" brief. The current
repo has a clear ranked gap set:

1. compressed-first product-model convergence
2. portable dense/backend maturity and broader runtime scalability
3. capability breadth beyond the current bounded real-only contract
4. documentation/coherence cleanup, especially removal of sprint-era residue
5. maintainability hotspot reduction in the largest mixed-role code and proof
   owners
6. build/package/workflow convergence and reduction of duplication
7. broader external comparison and more competitive runtime/correctness
   calibration

If those gaps are closed in order, the repo can move from "rigorous and
bounded scientific sparse library" to "credible modern sparse linear algebra
product." If they are not, it will remain unusually well engineered, but still
structurally below state-of-the-art sparse library expectations.
