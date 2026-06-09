# Sprint 60 Day 7: Configuration & Performance Surface Audit I

## Purpose

Reduce the live configuration and performance surface to concrete productization
gaps before Epic 6 implementation work starts. The main question for Day 7 is
not whether the repo has "lots of knobs"; it is which knobs already belong to a
coherent public API, which ones still leak through process-global behavior, and
which ones should remain internal even if they survive future cleanup.

## High-Level Surface Map

### 1. The solver-facing typed option surface is already real

The library already has a meaningful public typed-control layer for most
front-door workflows:

- direct repeated-run lifecycle:
  - `sparse_analysis_opts_t`
- one-shot direct families:
  - `sparse_lu_opts_t`
  - `sparse_cholesky_opts_t`
  - `sparse_ldlt_opts_t`
  - `sparse_qr_opts_t`
- iterative families:
  - `sparse_iter_opts_t`
  - `sparse_gmres_opts_t`
- eigensolvers:
  - `sparse_eigs_opts_t`
- SVD:
  - `sparse_svd_opts_t`

This matters because Epic 6 is not starting from a repo with no typed control
model. The existing gap is narrower and more specific: some of the
highest-value performance and ordering controls still sit outside that typed
surface.

### 2. The compile-time/build control surface is relatively small

The live build-switch layer is not the main productization problem.

Clear build-time controls:

- `SPARSE_OPENMP`
- `SPARSE_MUTEX`
- `SANITIZE`

These read like genuine build-shape switches rather than accidental runtime
policy leaks.

The more ambiguous compile-time controls are different:

- `SPARSE_NODES_PER_SLAB`
- `SPARSE_DROP_TOL`
- `SPARSE_CSC_THRESHOLD`
- `SPARSE_EIGS_THICK_RESTART_THRESHOLD`
- `SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD`

Those affect allocator behavior, drop policy, or AUTO backend routing. They are
not just build plumbing; they shape runtime numerical or performance behavior.

### 3. The process-global runtime control surface is concentrated in graph/order tuning

The strongest env-var-driven surface is the nested-dissection and FM tuning
stack, not the direct/iterative/eigensolver call layer.

Algorithm-selection env vars:

- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`
- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_SVD_LOWRANK_OUTER`

Numeric/performance tuning env vars:

- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_ANNEALING_SCHEDULE`
- `SPARSE_FM_THICK_RESTART_PERTURB`
- `SPARSE_FM_GAIN_NOISE_SCHEDULE`

Debug/profile env vars:

- `SPARSE_QG_PROFILE`
- `SPARSE_ND_PROFILE`
- `SPARSE_HCC_DEBUG`
- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_THICK_RESTART_DEBUG`
- `SPARSE_FM_ANNEALING_DEBUG`
- `SPARSE_FM_GAIN_NOISE_DEBUG`

Interpretation:

- the solver-facing product surface is mostly typed
- the graph-ordering and experimental tuning surface is still process-global
- the repo therefore has two different configuration models at once

### 4. Process-global behavior is reinforced by thread-local runtime state

The env-var issue is not only discoverability. The implementation also carries
process-global or thread-local state that makes the control plane harder to
reason about as a product surface:

- `_Thread_local` ND profiling accumulators in `src/sparse_reorder_nd.c`
- `_Thread_local` FM runtime state in `src/sparse_graph_refine.c`
- `_Thread_local` forced-HEM override in `src/sparse_graph_coarsen.c`
- orchestration-level runtime save/restore through
  - `sparse_graph_fm_runtime_get(...)`
  - `sparse_graph_fm_runtime_set(...)`

That is an architecture seam, not just a documentation problem. It means some
behavior is:

- scoped to the process or thread rather than to a call or object
- harder to compose safely in multi-workload applications
- harder to make explicit in examples and API contracts

## Ranked Productization Gaps

### 1. Strongest gap: ND/FM tuning is still outside the typed product model

The biggest configuration productization gap is not CG/GMRES/LOBPCG. Those
already expose typed public options. The biggest gap is that one of the most
algorithmically consequential subsystems, nested dissection plus FM refinement,
still depends on environment-variable parsing for:

- algorithm family selection
- pass counts
- schedule choice
- fallback strategy
- separator policy

That is the clearest "must become more explicit later" control surface in the
repo.

### 2. Strong second gap: AUTO performance policy still leaks through public compile-time macros

Some AUTO-routing policy is still documented and overridden through public
compile-time macros rather than through a clearer internal policy layer:

- `SPARSE_CSC_THRESHOLD`
- `SPARSE_EIGS_THICK_RESTART_THRESHOLD`
- `SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD`

This is a different kind of issue than the env vars:

- env vars create process-global runtime drift
- compile-time public macros leak implementation policy into the build contract

The current public API already gives callers explicit backend selectors when
they need deterministic behavior. That weakens the case for keeping AUTO-policy
thresholds as prominent public tuning surfaces indefinitely.

### 3. Smaller but real gap: a few experimental workflow toggles read more product-like than they really are

Two live examples:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_SVD_LOWRANK_OUTER`

Both are real, but they read more like advisory or experimental strategy knobs
than like first-class stable user controls. If they survive into later Epic 6
work, they need a cleaner ownership decision:

- typed public option
- internal-only retained experiment
- documentation-only advisory that should shrink rather than expand

### 4. Build-shape switches are not the main Epic 6 control problem

`SPARSE_OPENMP`, `SPARSE_MUTEX`, and `SANITIZE` are visible, but they are not
the same class of gap as the env-var-driven algorithm plane. They already read
like honest build configuration.

Epic 6 should avoid conflating:

- build-shape switches
- runtime algorithm policy
- public per-call solver options

## Candidate Public/Internal Split

### Must-be-public later

These are the strongest candidates for a future typed public control layer if
Epic 6 wants a more product-like configuration story:

- ND algorithm-family choice:
  - coarsening
  - coarsest bisection
  - root bisection
  - separator lift strategy
- FM refinement budget/strategy:
  - finest/intermediate pass counts
  - bounded strategy selection where the behavior is still considered supported
- optional supernodal postorder behavior, if it remains caller-meaningful

The most natural ownership lane is likely the direct-analysis/reorder path,
because these controls affect fill-reducing ordering and symbolic preparation
far more than the numeric direct-solver families themselves.

### Must-be-internal later

These controls should stay internal, or become narrower maintainership/debug
surfaces rather than product knobs:

- `SPARSE_QG_PROFILE`
- `SPARSE_ND_PROFILE`
- `SPARSE_HCC_DEBUG`
- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_THICK_RESTART_DEBUG`
- `SPARSE_FM_ANNEALING_DEBUG`
- `SPARSE_FM_GAIN_NOISE_DEBUG`
- forced-HEM retry plumbing
- FM runtime save/restore internals

Likewise, the build switches should remain build-time:

- `SPARSE_OPENMP`
- `SPARSE_MUTEX`
- `SANITIZE`

### Architecture-risk seams

These are not mere cleanup items:

- env-var parsing deep in algorithm implementations means the same public call
  can change behavior by process state rather than call-local state
- thread-local runtime mutation creates hidden control flow around FM/ND
  refinement
- compile-time threshold macros mix public headers with internal performance
  policy
- the repo currently has inconsistent control placement:
  - solver options are typed
  - graph strategy is env-var driven
  - backend AUTO heuristics partly leak through compile-time public macros

### Documentation-only drift

Some of the control-surface problem is wording density rather than architecture:

- long-form docs still expose advisory env vars very prominently
- historical tuning narrative still sits close to caller-facing explanation in
  a few places
- some compile-time threshold discussion is more detailed than a polished
  product front door needs

But that is secondary. The core Day 7 result is that the biggest remaining
configuration gap is real implementation ownership, not documentation style.

## Cross-Check Against the Epic 6 Review

The Day 7 live-tree audit supports the main Epic 6 review claims:

- advanced tuning is still too env-var/process-global driven
- product control placement is the highest-risk coherence seam
- the repo already has strong typed options in some families, so the right fix
  is convergence rather than a full control-plane rewrite

It also narrows those review claims:

- the main problem is concentrated in ND/FM and a few advisory performance
  toggles
- the build-switch layer is not the main offender
- solver-family front doors are already in better shape than the original
  broad review wording might suggest

## Day 7 Exit State

Sprint 60 now has a concrete control-surface map:

- typed solver options are already a strong base
- graph-ordering/runtime tuning is still the dominant env-var-driven gap
- public compile-time thresholds leak AUTO-performance policy
- debug/profile controls can be separated cleanly from any future typed public
  control work

That gives Day 8 a concrete continuation path:

- refine the compile-time/runtime/public/internal split
- decide which controls belong with direct analysis/reordering
- decide which heuristics should stay internal even if callers can force
  explicit backends today
