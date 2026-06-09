# Sprint 60 Day 8: Configuration & Performance Surface Audit II

## Purpose

Extend the Day 7 control-surface audit into the backend-sensitive and
performance-story layer:

- sparse direct workflows
- iterative/eigensolver workflows
- benchmark drivers and README stories
- build, packaging, and platform surfaces that shape performance claims

The goal is not to produce new measurements. It is to freeze a realistic
implementation order for later Epic 6 sprints.

## Day 8 Surface Findings

### 1. Direct backend sensitivity is already explicit, but split across three different policy layers

The direct side already exposes meaningful backend-sensitive surfaces:

- Cholesky:
  - `sparse_chol_backend_t`
  - `used_csc_path`
- LDL^T:
  - `sparse_ldlt_backend_t`
  - `used_csc_path`
- shared direct lifecycle:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - CSC-vs-linked-list decisions inside analysis/numeric paths

But the actual policy still spans three layers at once:

1. explicit per-call typed backend override
2. AUTO runtime dispatch
3. compile-time threshold policy such as `SPARSE_CSC_THRESHOLD`

Interpretation:

- the direct surface is already reasonably observable because callers can force
  a backend and inspect telemetry
- the real Epic 6 issue is not missing backend visibility
- the issue is scattered policy ownership:
  - some policy is typed
  - some is internal AUTO logic
  - some is public compile-time macro policy

### 2. Iterative and eigensolver backend sensitivity is narrower and healthier than the direct side

Iterative repeated-run support is already bounded and explicit:

- one-shot first
- handles only for:
  - `CG`
  - `GMRES`
  - `MINRES`

The iterative performance/control issue is therefore not backend sprawl. It is
mostly:

- OpenMP build-shape for shared kernels
- public-handle workflow proof
- residual maintainability around shared solver infrastructure

Eigensolvers are more mixed:

- explicit backend selector already exists
- `backend_used` already reports AUTO's choice
- `used_csc_path_ldlt` already exposes the shift-invert LDL^T path
- AUTO still depends on compile-time thresholds:
  - `SPARSE_EIGS_THICK_RESTART_THRESHOLD`
  - `SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD`

Interpretation:

- iterative control is mostly product-bounded already
- eigensolvers are closer to the direct side: visible controls exist, but AUTO
  routing policy still leaks through public compile-time thresholds

### 3. Benchmark drivers are already organized as workflow-proof surfaces

The benchmark layer is better than the broad Epic 6 review might imply.

Today it already separates into coherent groups:

- one-shot compatibility/comparison
- direct repeated-run lifecycle
- iterative public-handle reuse
- eigensolver public-handle reuse

This is a real product-strength:

- `bench_refactor` and `bench_refactor_csc` prove the direct repeated-run path
- `bench_iterative_reuse` proves the bounded iterative handle surface
- `bench_eigs_reuse` proves the bounded eigensolver handle surface

Interpretation:

- later Epic 6 work should preserve these workflow-proof binaries
- the benchmark problem is not "missing drivers"
- the benchmark problem is governance and consistency above the driver layer

### 4. Performance governance is still fragmented across docs, wrappers, and per-binary conventions

The repo has strong evidence surfaces, but weaker centralized policy for how
performance claims should be interpreted.

Current governance is distributed across:

- `README.md`
- `benchmarks/README.md`
- `Makefile`
- CI workflow comments
- per-benchmark CSV or text conventions

What is still fragmented:

- which benchmarks are canonical product baselines
- which outputs are machine-readable in a stable way
- which benchmark surfaces are regression-sensitive
- which runs are smoke/sanity versus claim-bearing
- which performance claims belong in README-level product messaging

Interpretation:

- this is real Epic 6 work
- but it is policy-layer work above the current benchmark binaries, not a
  reason to replace them with a new framework

### 5. Packaging/platform maturity still trails the quality and workflow story

The repo already has:

- install/export support
- pkg-config support
- `find_package(Sparse)` support
- reviewed CMake parity
- explicit staged Linux/macOS/Windows contract wording

But the build/package shape still reads more like a strong developer-install
surface than a full product distribution surface:

- primary library target is still:
  - `add_library(sparse_lu_ortho STATIC ...)`
- platform truthfulness is explicit, but asymmetric
- reviewed Windows support is still a subset/wrapper story rather than full
  parity
- macOS dead-code remains staged

Interpretation:

- packaging/platform work is definitely part of Epic 6
- but it should follow the control/backend contract work, not precede it

### 6. Some README performance narrative is still denser than the product story needs

The top-level docs now have a stronger workflow front door than before, but the
deeper performance sections still mix:

- current product facts
- detailed threshold rationale
- historical sprint-era measurement explanation
- backend-specific caveat density

Interpretation:

- there is still documentation-density cleanup to do later
- but the strongest remaining performance-story gap is policy coherence, not
  lack of raw detail

## Unified Configuration/Performance Map

Combining Day 7 and Day 8, the repo now breaks into five control/performance
bands.

### 1. Healthy public typed-control band

Already strong enough to preserve and extend carefully:

- direct lifecycle typed options
- one-shot direct solver typed options
- iterative typed options
- eigensolver typed options
- SVD typed options
- explicit backend selectors
- direct/eigensolver path telemetry

### 2. Must-converge configuration band

Strongest later public/internal cleanup target:

- ND/FM strategy and pass-budget controls
- process-global algorithm knobs that should migrate toward typed ownership if
  they remain supported

### 3. Must-rationalize backend-policy band

Second strongest implementation target:

- AUTO backend policy still leaking through public compile-time thresholds
- direct and eigensolver threshold policy needs clearer ownership
- public explicit backend forcing should remain

### 4. Governance/policy band

Needs product-quality consolidation rather than new math kernels:

- canonical benchmark surfaces
- machine-readable result conventions
- regression-sensitive performance policy
- claim-bearing versus smoke-only benchmark usage
- packaging/platform truthfulness rules

### 5. Residual density/maintainability band

Important, but clearly lower priority than the first four:

- long-form docs density
- residual hotspot decomposition
- residual giant-test seams
- platform follow-through beyond the already-stated truth surface

## Ranked Future Implementation Queue

### 1. Highest priority: typed configuration convergence for ND/FM and adjacent advisory controls

Why first:

- Day 7 showed this is the strongest remaining env-var-driven productization
  gap
- it affects the most consequential advanced fill/performance control plane
- it is the clearest mismatch with the otherwise typed solver-facing surface

Likely owning sprints later:

- early Epic 6 architecture and control-convergence sprints

### 2. Second priority: backend/AUTO policy rationalization

Why second:

- the repo already has explicit backend selection and telemetry
- the remaining issue is policy ownership, not missing capability
- this work can build directly on the Day 7-8 control-placement rule

Likely scope:

- direct CSC crossover policy
- eigensolver AUTO crossover policy
- clearer separation between public forcing and internal heuristics

### 3. Third priority: benchmark-governance consolidation

Why third:

- the benchmark binaries already prove the right workflows
- the current missing layer is policy:
  - canonical baselines
  - regression-sensitive tiers
  - stable output conventions

Non-goal:

- no broad benchmark-framework rewrite unless a real limitation appears

### 4. Fourth priority: packaging/platform/release-shape convergence

Why fourth:

- important for product maturity
- but secondary to getting configuration and backend policy coherent first
- should preserve the reviewed truthfulness contract already established in
  Epic 5

Likely scope:

- build-target/release-shape decisions
- staged platform follow-through where justified
- explicit residual limits where parity is still not real

### 5. Fifth priority: documentation/performance-story compression

Why fifth:

- the repo already has strong evidence and many details
- the main issue is coherence and density, not missing explanation
- this should follow real policy consolidation so the docs can describe a
  settled surface

## What This Means for Later Epic 6 Sprints

The strongest later implementation ownership now separates cleanly:

- backend/control public options:
  - own the ND/FM and adjacent control convergence work
- benchmark governance:
  - own canonical baselines, output conventions, and regression sensitivity
- performance-baseline policy:
  - own claim-bearing versus smoke-only benchmark distinctions
- packaging/platform follow-through:
  - own build/distribution/platform truthfulness after control policy is
    clearer

This also clarifies what is *not* the first move:

- not a broad solver-family API rewrite
- not a new benchmark framework
- not packaging-first polish ahead of control-plane coherence
- not generic docs trimming without deeper policy cleanup

## Day 8 Exit State

Sprint 60 now has a unified configuration/performance surface map:

- typed solver controls are already a strong base
- ND/FM process-global tuning is the strongest productization gap
- backend AUTO policy is the second strongest gap
- benchmark drivers are already good workflow-proof surfaces
- the main performance-story weakness is governance and policy coherence above
  those drivers
- packaging/platform maturity is real Epic 6 work, but not the first lever

That is enough to move to Day 9, where these findings can be converted into an
explicit architecture contract rather than remaining as audit observations.
