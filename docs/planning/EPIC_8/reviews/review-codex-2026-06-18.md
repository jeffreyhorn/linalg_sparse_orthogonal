# Code Review — 2026-06-18

## Scope

This review assesses the current `linalg_sparse_orthogonal` tree across:

- code efficiency
- maintainability
- usability
- documentation quality
- cross-surface coherence
- test coverage
- state-of-the-art sparse linear algebra readiness

The assessment is based on the live `master` tree after Epic 7 closeout,
including the core source files, public headers, build/package surfaces, CI
workflows, benchmarks, examples, maintainer policy, Epic 7 closeout notes, and
the current reviewed validation contract (`make quality-review-full`,
reviewed CMake parity, install/export proof scripts).

## Executive Verdict

The project is unusually strong on engineering discipline, truthfulness of
claims, validation culture, artifact trail, and proof ownership. It is much
more rigorous than a typical hobby sparse library.

It is **not yet a state-of-the-art sparse linear algebra library**.

The biggest reasons are structural rather than cosmetic:

1. the public/core storage model is still anchored on an orthogonal linked-list
   shell instead of a compressed-first compute model
2. the dense numeric backend is still a builtin scalar C implementation rather
   than a serious BLAS/LAPACK/vendor-kernel layer
3. the numerical capability surface is still bounded to real-only scalar types,
   compile-time index width, and a narrower algorithm/product envelope than
   leading sparse libraries
4. platform/package proof remains intentionally asymmetric and static-first
5. maintainability hotspots are still very large even after Epic 7 cleanup

My bottom-line assessment is:

- **engineering rigor:** high
- **product maturity:** medium
- **performance ceiling:** medium-low
- **maintainability:** medium
- **usability for expert users:** medium
- **usability for new adopters:** medium-low
- **state-of-the-art competitiveness:** not yet

## Sprint 80 Alignment Update

Sprint 80 Days 1-7 did not change the review verdict, but they did tighten the
contract this review should be read against:

- the strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity remains explicit at `ctest -N --test-dir build/quality-review-cmake = 53`
- the first maintained external correctness target is now fixed to a bounded
  CHOLMOD-class SPD Cholesky comparison lane
- BLAS/LAPACK-class references are fixed as performance-reference support, not
  as a broad maintained correctness contract
- the maintained benchmark face remains intentionally compact and the canonical
  report remains threshold-free
- Epic 8 is now explicitly fenced against fake platform parity, shared-library
  maturity, broad capability-genericity, or “rewrite the whole library”
  interpretations

That means the findings below should be read as a ranked contradiction map with
an explicit claim fence, not as an invitation to widen Epic 8 into every
possible modernization lane at once.

## Repository Snapshot

### Highest-signal implementation hotspots

| file | lines | review meaning |
|---|---:|---|
| `src/sparse_ldlt_csc.c` | 2182 | still a large mixed-role direct-solver implementation even after Sprint 78 cleanup |
| `src/sparse_iterative.c` | 1985 | very large solver-family surface with likely future extraction pressure |
| `src/sparse_lu_csr.c` | 1665 | large family-local implementation surface |
| `src/sparse_chol_csc.c` | 1564 | still a major orchestration/completion hotspot |
| `src/sparse_qr.c` | 1563 | large algorithm-family implementation surface |

### Highest-signal giant-test hotspots

| file | lines | review meaning |
|---|---:|---|
| `tests/test_chol_csc.c` | 4724 | very broad family-local proof owner |
| `tests/test_ldlt_csc.c` | 3680 | still a large direct-family proof surface |
| `tests/test_qr.c` | 3197 | large algorithm-family proof surface |
| `tests/test_etree.c` | 2962 | large symbolic/reordering proof owner |
| `tests/test_graph.c` | 2925 | large graph/reordering proof owner |
| `tests/test_integration.c` | 2689 | major public lifecycle/parity owner |

These numbers do not automatically mean the code is poor. They do mean the
remaining maintainability costs are concentrated and easy to locate.

## Findings

### 1. Critical: the public/core storage model is still the main efficiency ceiling

**Evidence**

- `README.md:3` introduces the library as an orthogonal linked-list sparse
  matrix library.
- `include/sparse_matrix.h:6-25` keeps `SparseMatrix` as the public mutable
  orthogonal linked-list shell and explicitly separates that from the repeated
  direct lifecycle.
- `src/sparse_matrix.c:18-63` implements node-pool allocation around pointer
  nodes.
- `src/sparse_matrix.c:296-357` inserts by walking row and column lists.
- `src/sparse_matrix.c:360-390` removes entries by walking linked structure.

**Why this matters**

This design is excellent for pedagogy, mutation flexibility, and a clean
cross-linked data model. It is not a state-of-the-art compute/storage model for
modern sparse numerical work. A compressed-first model is the standard high-end
shape because it gives much better locality, simpler kernel feeding, lower
allocator pressure, and cleaner interop with external backends.

The current architecture partly compensates for this by routing large direct
work to CSC/CSR-backed paths, but that means the public shell and the real
numeric working model are still different things. That split is manageable and
documented, but it is still a structural cost.

**Gap**

The library still behaves like a linked-list sparse matrix product with
compressed compute accelerators attached, not like a compressed sparse compute
product with a bounded mutable-construction shell.

### 2. Critical: the dense numeric backend is not strong enough for a top-tier performance claim

**Evidence**

- `src/sparse_dense.c:50-112` implements `dense_gemm` as a scalar triple loop.
- `src/sparse_dense.c:114-156` implements `dense_gemv` as a scalar loop.
- `src/sparse_dense.c:158-255` implements dense Cholesky factor/solve helpers
  in builtin scalar C.
- `src/sparse_dense.c:258-325` exposes a builtin dense-kernel descriptor rather
  than a mature external backend layer.
- `benchmarks/README.md:209-239` shows the backend-aware Cholesky benchmark is
  still proving a bounded builtin descriptor / batched panel seam rather than a
  richer backend ecosystem.

**Why this matters**

For sparse direct methods, especially supernodal paths, the dense update and
panel kernels are where a large part of the wall time moves. A sparse library
that wants to be state of the art normally has one or more of:

- BLAS/LAPACK integration
- serious runtime backend dispatch
- vendor-kernel integration
- multithreaded dense kernels
- wider backend-specific tuning

This codebase instead has a careful but narrow builtin implementation. That is
an honest product choice, but it hard-limits the performance ceiling.

**Gap**

The project has backend-aware architecture, but not yet backend maturity.

### 3. High: the capability surface is still too narrow for a state-of-the-art library

**Evidence**

- `include/sparse_types.h:20-60` keeps index width as a compile-time 32/64-bit
  choice.
- `include/sparse_types.h:62-82` defines `sparse_scalar_t` as `double` and
  explicitly documents the surface as real-only and bounded.
- `README.md:33-46` exposes only symmetric sparse eigensolvers.
- `README.md:22-27` explicitly bounds reusable iterative handles to `CG`,
  `GMRES`, and `MINRES`.

**Why this matters**

Modern sparse libraries typically compete on some combination of:

- complex-number support
- mixed precision or broader scalar families
- richer sparse eigensolver breadth
- broader runtime capability selection
- more mature large-index / ABI stories

This project is still intentionally bounded on those fronts. Epic 7 improved
the seams and made the limits clearer, but it did not remove the limits.

**Gap**

The capability contract is clearer than before, but still materially narrower
than best-in-class scientific sparse libraries.

### 4. High: cross-platform quality and packaging proof remain intentionally asymmetric

**Evidence**

- `CMakeLists.txt:10-18` explicitly ignores `BUILD_SHARED_LIBS=ON` and keeps
  the maintained package surface static-only.
- `CMakeLists.txt:87-128` builds a static library only.
- `.github/workflows/windows-ci.yml:3-11` documents that Windows is still a
  reviewed CMake-first subset only.
- `.github/workflows/windows-ci.yml:27-58` hardcodes a smaller reviewed test
  surface and states that `test_fuzz` is outside the reviewed Windows subset.
- `.github/workflows/macos-ci.yml:3-10` documents Apple Clang as the reviewed
  lane plus supplemental GCC.
- `.github/workflows/macos-ci.yml:91-96` explicitly says the install/`pkg-config`
  job is confidence-building only, not reviewed install/export parity.

**Why this matters**

The repo is commendably truthful about this. The problem is not misleading
documentation. The problem is product competitiveness. A high-end sparse
library is expected to ship a more converged packaging and cross-platform
story, especially for downstream consumption.

**Gap**

The project has a good honesty model for cross-platform quality, but still a
limited product model for cross-platform consumption.

### 5. High: build and packaging topology are duplicated and still maintenance-heavy

**Evidence**

- `Makefile:42-84` enumerates library sources manually.
- `Makefile:86-140` enumerates tests manually.
- `Makefile:142-159` enumerates benchmarks manually.
- `CMakeLists.txt:87-128` duplicates the library source topology in a separate
  maintained build system.
- `README.md:92-130` presents both Make and CMake as first-class paths.

**Why this matters**

The dual-surface build story is useful and well-tested, but it creates ongoing
maintenance cost:

- source list duplication
- command-surface duplication
- install/export contract duplication
- CI matrix complexity

The recent Epic 7 install-path race fix in the Makefile is a concrete example
of the kind of drift risk this creates.

**Gap**

The project has strong build validation, but the build architecture is still
more manual and duplicated than a mature long-term product should want.

### 6. High: maintainability hotspots remain large enough to slow future work

**Evidence**

- `src/sparse_ldlt_csc.c:1`
- `src/sparse_iterative.c:1`
- `src/sparse_chol_csc.c:1`
- `src/sparse_qr.c:1`
- `tests/test_chol_csc.c:1`
- `tests/test_ldlt_csc.c:1`
- `tests/test_qr.c:1`
- `tests/test_integration.c:1`

**Why this matters**

Epic 7 improved some of the biggest seams, especially:

- `src/sparse_ldlt_csc.c`
- `tests/test_chol_csc.c`

That helped, but it did not finish the job. These files are still large enough
that:

- review throughput is slower
- local reasoning costs are higher
- regression ownership is more concentrated than ideal
- architectural cleanup remains easy to postpone

**Gap**

The library has moved from “unmanaged hotspot sprawl” to “known bounded
hotspots,” but those hotspots still need another full maintainability pass.

### 7. Medium: test coverage is deep, but still too self-hosted for a strongest-possible assurance claim

**Evidence**

- The tree uses real SuiteSparse fixtures heavily:
  - `tests/test_chol_csc.c:394`
  - `tests/test_ldlt.c:2082`
  - `tests/test_iterative.c:499`
  - `tests/test_eigs.c:824`
- The repo also has explicit property/oracle owners:
  - `tests/test_integration.c:1`
  - `tests/test_fuzz.c:1`
- However, the reviewed tree and build/package surfaces do not show maintained
  CHOLMOD / UMFPACK / MKL / OpenBLAS / LAPACK differential harnesses as
  correctness or performance oracles.

**Why this matters**

The test culture is one of the strongest parts of the project. The remaining
gap is not “more unit tests.” It is “more independent reference testing.”

Right now most correctness proof is:

- internal oracle tests
- residual checks
- lifecycle/property checks
- regression-specific fixtures

That is good engineering. It is weaker than a maintained differential harness
against mature external sparse backends or reference solvers.

**Gap**

The project has strong internal proof, but limited external numerical proof.
Sprint 80 now narrows the first maintained external target to a bounded
CHOLMOD-class SPD direct-solver lane, which is the right first corrective
move, but that proof is not landed yet.

### 8. Medium: benchmark governance is disciplined, but performance governance is still intentionally weak

**Evidence**

- `benchmarks/README.md:130-160` explicitly makes `make bench-canonical-report`
  threshold-free and artifact-oriented rather than pass/fail.
- `benchmarks/README.md:160-239` keeps the maintained benchmark surface narrow
  and carefully split from oracle ownership.
- `README.md:84-86` also keeps the canonical maintained benchmark surface
  intentionally compact and threshold-free.

**Why this matters**

This is a good truthfulness model. It avoids fake portability claims and
spurious benchmark gates. But it also means the repo does not yet have a
strong, automated, competitive performance-governance story.

That leaves the project in an awkward middle state:

- better benchmark discipline than many libraries
- weaker automated performance regression control than a true
  high-performance production library

**Gap**

The project can explain performance honestly, but it cannot yet enforce or
market state-of-the-art performance strongly.
Sprint 80 now fixes the intended reading more explicitly: canonical reporting
remains threshold-free, `bench-fast` remains the bounded runtime lane, and
`wall-check` remains the narrow thresholded regression gate. That is a strong
truthfulness model, but it still leaves performance competitiveness less
automated than a top-tier production library.

### 9. Medium: the documentation set is strong but overloaded for adoption

**Evidence**

- `README.md:1-130` mixes product overview, workflow model, CI contract,
  benchmark governance, cancellation semantics, and build/install guidance.
- `README.md:77-88` requires users to understand a fairly nuanced workflow
  taxonomy very early.
- `README.md:74-75` documents materially different cancellation semantics by
  family/path.
- `benchmarks/README.md:130-239` is careful and coherent, but policy-dense.

**Why this matters**

For expert maintainers, this documentation is excellent. For new adopters, it
is a lot to absorb before they can answer simple questions like:

- Which solver path should I use?
- What storage model should I start from?
- What package shape am I actually consuming?
- Which benchmarks are proof versus examples versus exploratory tools?

**Gap**

The docs optimize for completeness and truthfulness more than for fast
adoption.

### 10. Medium: coherence is good internally, but too many internal policies leak into public surfaces

**Evidence**

- `include/sparse_matrix.h:51-96` exposes a large amount of threshold and
  benchmark-derived commentary in a public header.
- `README.md:34-46` documents backend-selection heuristics and thresholds at
  front-door level.
- `README.md:62-64` documents detailed ND compatibility / environment override
  policy in the feature list.

**Why this matters**

The repo is coherent in the sense that its docs and code agree with each other.
The cost is that user-facing surfaces often read like policy surfaces.

That is better than being misleading, but worse than having a narrower, cleaner
public contract with deeper details moved behind:

- stronger defaults
- smaller option surfaces
- advanced docs kept separate from front-door docs

**Gap**

The public surface is too policy-heavy for a library that wants to feel mature
and easy to consume.

## Strengths

The project has real strengths that should not be lost while closing the gaps:

1. **Validation discipline is unusually strong.**  
   The maintained reviewed baseline, reviewed CMake parity, install/export
   scripts, workflow truthfulness, and bounded benchmark governance are all far
   above average for a C library of this size.

2. **The maintainers are honest about product limits.**  
   The tree repeatedly avoids fake claims about platform parity, shared-library
   maturity, backend breadth, benchmark thresholds, or capability coverage.

3. **Proof ownership is excellent.**  
   Epic 7 left the project in a state where many important behaviors clearly
   belong to specific tests, scripts, or workflows.

4. **The repo already has a serious sparse-numerics feature base.**  
   LU, Cholesky, LDL^T, QR, iterative methods, eigensolvers, SVD, reordering,
   analysis/refactorization, and benchmark/reporting support is substantial.

5. **The docs are coherent even when dense.**  
   The biggest documentation issue is overload, not contradiction.

## Category Assessment

| category | assessment | notes |
|---|---|---|
| Efficiency | `medium-low` | serious sparse functionality, but pointer-heavy public shell and scalar dense kernels cap the ceiling |
| Maintainability | `medium` | good audit trail and ownership culture, but large hotspots remain |
| Usability | `medium-low` | powerful for expert users, heavy cognitive load for new adopters |
| Documentation | `high` | detailed, careful, and truthful; needs simplification more than expansion |
| Coherence | `medium-high` | strong internal consistency, but policy density leaks into public surfaces |
| Test Coverage | `high` | strong regression/oracle/property culture; weaker external differential coverage |
| Packaging / Platform | `medium-low` | static-first and intentionally bounded; not yet broadly converged |
| State-of-the-Art Readiness | `not yet` | strong engineering platform, insufficient storage/backend/capability/product maturity |

## State-of-the-Art Assessment

If the question is:

> “Is this a serious sparse linear algebra project?”

The answer is **yes**.

If the question is:

> “Is this already a state-of-the-art sparse linear algebra library in the same
> class as the strongest production sparse stacks?”

The answer is **no**.

The main blockers are:

- the linked-list-first public/core model
- the builtin scalar dense-kernel ceiling
- limited capability breadth
- incomplete cross-platform/package convergence
- incomplete external differential/performance proof
- remaining maintainability concentration

The project is best described as:

> a highly disciplined, well-tested, feature-rich sparse linear algebra library
> with a strong validation culture and honest product boundaries, but not yet a
> top-tier sparse compute platform.

Sprint 80's current fence sharpens that description further:

- Epic 8 is allowed to close real storage, backend, capability, assurance, and
  maintainability ceilings
- Epic 8 is not allowed to market those lanes as already complete before the
  corresponding proof and product surfaces actually move

## Recommended Priority Order

If the goal is to close the biggest gaps credibly, the order should be:

1. compressed-first product/storage modernization
2. dense/backend performance architecture modernization
3. capability breadth expansion
4. external differential and numerical-assurance expansion
5. hotspot source/test decomposition
6. reordering/runtime long-pole reduction
7. packaging/shared-library/platform convergence
8. front-door workflow and documentation simplification
9. final external comparison and claim calibration

That sequence is the basis for the Epic 8 todo and sprint plan.
