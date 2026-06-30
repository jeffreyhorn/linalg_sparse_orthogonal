# Sprint 100 Day 6 State-of-the-Art Target Draft

## Purpose

Day 6 defines the Epic 10 target for "state-of-the-art" work in a way that is
specific to this repository. The target is intentionally evidence-backed and
bounded. It prevents Epic 10 from importing every obligation of larger sparse
linear algebra ecosystems while still forcing serious product maturity work.

## One-Sentence Target

Epic 10 targets a **product-grade, self-contained C sparse linear algebra
library with compressed-first workflows, stronger external oracle evidence,
clear package/platform support tiers, and calibrated competitive claims**.

## Comparison Set

The comparison set is a maturity backdrop, not a promise to match every feature
or performance result.

| reference class | representative projects | comparison dimension |
|---|---|---|
| sparse direct solver suites | SuiteSparse, CHOLMOD, KLU, UMFPACK | compressed-first storage, ordering, direct-solver robustness, external corpus depth |
| iterative/eigensolver libraries | ARPACK, Spectra, LOBPCG implementations, Eigen iterative solvers | convergence evidence, residual reporting, restart/preconditioner behavior, oracle comparisons |
| scientific computing frameworks | PETSc, Trilinos | platform support, package maturity, solver selection, runtime/backend configurability |
| vendor/math backends | Intel oneMKL, Accelerate, BLAS/LAPACK providers | backend selection, observability, fallback behavior, performance-portability limits |
| graph/reorder ecosystems | METIS-class partitioning, AMD/COLAMD references, GraphBLAS-style systems | reorder/fill reporting, graph partition evidence, sparse operation maturity |
| user-facing C/C++ libraries | Eigen, smaller C sparse libraries | API usability, examples, documentation, install/consumer ergonomics |

## Evidence Dimensions

| dimension | Epic 10 expectation | evidence required before claim is earned |
|---|---|---|
| algorithmic quality | solver results are correct on named, stratified fixtures with clear failure modes | tests, external dense/reference comparisons, residual criteria, and unsupported-case records |
| API usability | compressed-first and solver-selection paths are clear to first-time users | examples, public header contracts, solver guide, ownership/error docs |
| storage/product model | CSR/CSC workflows are the obvious product center while mutable matrix-shell compatibility remains explicit | public constructors/imports, lifecycle tests, docs, and compatibility wording |
| maintainability | large source and giant-test risk decreases in touched families | file ownership maps, extraction artifacts, source-list parity, focused validation |
| backend/runtime behavior | builtin fallback and optional acceleration/runtime controls are inspectable and bounded | backend descriptors, observability tests, benchmark fields, non-claim wording |
| package maturity | install/export, CMake, pkg-config, versioning, and support tiers are explicit | install scripts, CMake consumer proof, package docs, platform-tier tables |
| platform proof | Linux/macOS/Windows support tiers are truthful and validated to their intended depth | reviewed/supplemental CI maps, expected counts, staged exclusion registers |
| benchmark evidence | local performance/reporting is decision-grade but not overclaimed | canonical reports, bounded sentinels, artifact metadata, local-timing caveats |
| external comparison depth | comparisons widen family by family with owned fixtures and tolerance models | one-page proof architecture, external scripts/oracles, fixture taxonomy |

## Capability Categories

### Must-Have for Epic 10 Earned Product-Maturity Claims

| capability | rationale | likely sprint owner |
|---|---|---|
| compressed-first CSR/CSC front door | product model cannot remain linked-list-first in its primary reading | Sprint 101 |
| direct solver external oracle expansion | direct solvers are a core sparse library claim | Sprint 102 |
| iterative/eigensolver/SVD comparison architecture | current evidence is strong internally but weak externally | Sprint 103 |
| backend/runtime contract | performance claims require clear fallback and observability | Sprint 104 |
| reorder/fill and large-matrix evidence | sparse performance depends heavily on ordering and graph behavior | Sprint 105 |
| large-source and giant-test extraction | maintainability is currently a state-of-the-art blocker | Sprint 106 |
| solver-selection and compressed-first docs | usability must match capability breadth | Sprint 107 |
| package/platform support tiers | state-of-the-art-adjacent claims need truthful consumer support | Sprint 108 |
| final competitive calibration | unsupported broad claims must be removed or explicitly deferred | Sprint 109 |

### Stretch Candidates

These are valuable if scope and evidence allow them, but they should not block
Epic 10 closeout unless Sprint 100 Day 8 promotes them into earned-claim
dependencies.

| capability | condition for promotion |
|---|---|
| broader LDLT CSC Matrix Market or indefinite corpus comparison | fixture taxonomy, runtime budget, and oracle behavior are defined first |
| generated reorder/fill report target | repeated manual captures justify a maintained generated artifact |
| shared-library/ABI proof | Sprint 108 explicitly chooses to build and validate this support tier |
| wider package-manager recipes | only if install/export proof and support ownership are explicit |
| stronger benchmark sentinels | thresholds are local, justified, stable, and not marketed as portable superiority |
| broader Windows parity | CI count, exclusions, install path, and validation cost are made explicit |

### Explicit Non-Goals Unless Replanned

| non-goal | reason |
|---|---|
| GPU sparse kernels | no implementation, packaging, backend, or CI proof exists |
| distributed-memory sparse solvers | out of scope for the current self-contained C library |
| universal vendor backend parity | optional backend work remains bounded and fallback-first |
| broad complex-number maturity | current capability surface is real-first and not broadly complex-generic |
| broad mixed-precision maturity | no family-wide mixed-precision implementation/proof exists |
| full replacement of the mutable matrix shell | compatibility shell remains part of the public product model |
| shared-library ABI guarantee | static-first package proof is current truth unless Sprint 108 changes it |
| symmetric Linux/macOS/Windows parity | reviewed platform lanes intentionally differ |
| portable timing superiority | benchmark timings are local calibration, not universal claims |
| every-solver-family external correctness comparison | current external lanes are bounded and family-local |

## Disallowed Broad Claims Before Sprint 109 Evidence

The following wording must not appear as an earned project claim until final
evidence supports it:

- "state-of-the-art sparse linear algebra library" without qualification
- "SuiteSparse parity" or "SuiteSparse replacement"
- "PETSc/Trilinos-class platform support"
- "portable performance superiority"
- "vendor backend parity"
- "shared-library ABI stable"
- "Windows parity" without naming the reviewed CMake subset and exclusions
- "every solver family externally validated"
- "complex support" or "mixed precision" as broad product claims
- "GPU-ready" or "distributed-ready"
- "universal reorder/fill superiority"
- "coverage is reviewed universal proof"

## Draft Epic 10 Claim Language

Allowed opening claim:

> Epic 10 is a productization and evidence epic for a broad, self-contained C
> sparse linear algebra library. It aims to make compressed CSR/CSC workflows
> more central, widen external oracle evidence on priority solver families,
> reduce large-owner maintainability risk, clarify backend/runtime behavior,
> and publish truthful package/platform support tiers.

Disallowed opening claim:

> Epic 10 makes the project a state-of-the-art replacement for mature sparse
> linear algebra ecosystems.

## Final Success Reading

Epic 10 succeeds if Sprint 109 can truthfully say:

- compressed-first workflows are clearer and better tested;
- comparison evidence is deeper on selected solver families;
- benchmark/reporting artifacts are easier to interpret without overclaiming;
- large source and giant-test ownership is measurably improved;
- package and platform support tiers are explicit;
- unsupported state-of-the-art claims are absent;
- remaining non-goals are written as deliberate future work.

