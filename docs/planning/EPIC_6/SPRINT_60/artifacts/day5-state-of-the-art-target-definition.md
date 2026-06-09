# Sprint 60 Day 5: State-of-the-Art Target Definition

## Purpose

Define what “state of the art” should actually mean for this project after the
Day 3-4 ranked Epic 6 inventory, and lock the explicit non-goal fence that
will keep later implementation sprints honest.

## Epic 6 Target Definition

For this repository, **state of the art** should mean:

- a strong, coherent, single-node sparse linear algebra product surface
- easier and safer direct-solver workflow adoption
- typed advanced configuration for the highest-value control surfaces
- a bounded modern performance/backend architecture on selected hot paths
- a clearer performance-governance and benchmark story
- a more product-like packaging/platform contract
- stronger second-layer assurance on the hardest workflows

It should **not** mean:

- distributed-memory or cluster/HPC scope
- immediate parity with vendor-tuned specialized backend libraries
- unlimited platform guarantees beyond reviewed measurement
- broad new algorithm-family expansion as the epic’s main story

## Primary Epic 6 Goal Bands

1. **Direct-solver usability convergence**
   - reduce mutable-matrix surprise and tighten the relationship between
     one-shot and explicit repeated-run direct workflows
2. **Typed advanced configuration**
   - replace the highest-value env-var-driven controls with typed option
     surfaces and explicit precedence rules
3. **Bounded backend/performance architecture**
   - add a real architecture seam for selected dense-kernel, threading-policy,
     and acceleration-sensitive hot paths
4. **Performance/platform/packaging maturity**
   - make benchmark/performance claims easier to govern and the
     packaging/platform story more product-like
5. **Assurance and residual maintainability follow-through**
   - deepen oracle/property/differential confidence where it matters most and
     reduce the residual hotspot/test debt that blocks the higher-value work

## Preserved Product Fence

Epic 6 target-setting does **not** reopen the Epic 5 solved workflow boundary:

- repeated direct solves remain the explicit analysis/factors lifecycle
- iterative handles remain bounded to:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver handle remains bounded to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows remain one-shot compatibility
  surfaces unless a later sprint explicitly re-charters them

## Explicit Non-Goal Fence

Epic 6 explicitly does **not** target:

- distributed-memory / cluster / MPI sparse linear algebra scope
- broad vendor-backend parity as a headline goal
- universal repeated-run support for every solver family
- a large new algorithm-family wave as the center of gravity of the epic
- fake cross-platform closure without measured reviewed support
- maintainability-only cleanup that does not materially help product,
  architecture, validation, or adoption outcomes

## Candidate Epic 6 Success Scorecard

Epic 6 should count as successful if it leaves the repo with:

- a more coherent direct-solver story that reduces matrix-state surprise
- a typed advanced-configuration story for the highest-value controls
- a real bounded backend/performance seam on selected hot paths
- a smaller and clearer set of canonical performance-governance surfaces
- a stronger packaging/platform contract with explicit residual limits
- stronger second-layer assurance on the hardest lifecycle/CSC/repeated-run
  workflows
- smaller remaining architecture/test hotspots where Epic 6 touched the most
  leverage-heavy seams
- a final API/docs/examples/benchmarks/validation story that reads as one
  coherent product surface

## Day 5 Exit State

Sprint 60 now has a stable target and a stable non-goal fence. Later Epic 6
implementation sprints can now be judged against explicit product outcomes
instead of vague “more mature” or “more state of the art” language.
