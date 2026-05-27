# Sprint 45 Working Notes

## Day 1

**Objective:** Turn the Sprint 45 project-plan scope plus the Sprint 40/41/42
execution rules and the Sprint 44 closeout baseline into a concrete iterative
workspace starting point by confirming the preserved reviewed contracts,
naming the Sprint 45 workstreams explicitly, and defining the authoritative
iterative solver, test, example, and benchmark inputs before workspace-reuse
implementation begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short`
   - `git rev-parse --abbrev-ref HEAD`
2. Re-read the Sprint 45 plan and the main prerequisite planning artifacts:
   - `sed -n '189,253p' docs/planning/EPIC_4/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_44/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
3. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Reconfirm the current maintained reviewed/dead-code command surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`
5. Measure the live iterative hotspot and the current iterative-support
   concentration:
   - `wc -l src/sparse_iterative.c src/sparse_eigs.c tests/test_iterative.c tests/test_block_solvers.c tests/test_bicgstab.c tests/test_minres.c tests/test_stagnation.c examples/example_iterative.c examples/example_matrix_free.c benchmarks/bench_convergence.c benchmarks/bench_refactor.c`
6. Refresh the live iterative workspace / allocation seam markers:
   - `rg -n "sparse_solve_cg|sparse_solve_gmres|matrix_free|block|minres|workspace|malloc|calloc" src/sparse_iterative.c | sed -n '1,260p'`
   - `sed -n '1,260p' src/sparse_iterative.c`
   - `rg -n "typedef struct .*workspace|workspace_alloc|workspace_free|bicgstab_workspace_t|gmres|cg" src/sparse_iterative.c`

### Day 1 Findings

#### 1. Sprint 45 starts from a preserved Sprint 40/41/42/44 baseline, not from baseline repair work

The inherited starting contract remains explicit and stable:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- maintained dead-code/reporting paths already exist:
  - `make deadcode-report`
  - `make deadcode-check`
- dead-code execution remains serialized
- Sprint 41 already left behind the shared internal arithmetic/allocation seam:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`
- Sprint 42 already left behind:
  - internal lifecycle/factor-state scaffolding
  - shared matrix-state guard helpers
  - compatibility-preserving internal-first refactor rules
- Sprint 44 already preserved the stronger Epic 4 structural baseline, so
  Sprint 45 is free to target iterative repeated-solve efficiency directly

Interpretation:

- Sprint 45 is not a quality-baseline sprint
- Sprint 45 is an iterative-workspace and repeated-solve efficiency sprint on
  top of an already-validated Epic 4 baseline

#### 2. The live iterative surface is still concentrated in one real hotspot, but it already contains one reusable-workspace precedent

The live implementation/test concentration is:

- `src/sparse_iterative.c` = `2357` lines
- adjacent next-sprint comparison target:
  - `src/sparse_eigs.c` = `3151`
- primary iterative regression surfaces:
  - `tests/test_iterative.c` = `2795`
  - `tests/test_block_solvers.c` = `507`
  - `tests/test_bicgstab.c` = `1586`
  - `tests/test_minres.c` = `1588`
  - `tests/test_stagnation.c` = `1361`
- main maintained iterative examples:
  - `examples/example_iterative.c` = `144`
  - `examples/example_matrix_free.c` = `122`
- strongest likely repeated-solve benchmark surfaces:
  - `benchmarks/bench_convergence.c` = `421`
  - `benchmarks/bench_refactor.c` = `159`

The live allocation map also shows a useful asymmetry:

- CG, matrix-free CG, GMRES, block CG, and MINRES still allocate one-shot
  work bundles directly in `src/sparse_iterative.c`
- BiCGSTAB already uses a dedicated reusable-ish internal workspace owner:
  - `bicgstab_workspace_t`
  - `bicgstab_workspace_alloc(...)`
  - `bicgstab_workspace_free(...)`

Interpretation:

- Sprint 45 does not start from zero; it already has one internal workspace
  precedent in the iterative subsystem
- the real repeated-allocation target set is still CG / GMRES / block /
  MINRES-centric rather than "all iterative solvers equally"

#### 3. The strongest Sprint 45 workstreams are explicit before code changes begin

Day 1 confirms the sprint's eight bounded workstreams directly from the plan:

- iterative workspace seam inventory
- reusable workspace API design
- shared workspace-backed internal helper layer
- CG / GMRES migration
- block iterative migration
- compatibility wrapper preservation
- repeated-solve benchmark batch
- validation closeout

Interpretation:

- the front half of the sprint should stay internal-first:
  - seam inventory
  - workspace design
  - shared helper layer
  - primary iterative migration
- the back half should then pivot into:
  - block-path adoption
  - wrapper normalization
  - repeated-solve benchmark evidence

#### 4. Sprint 45 already has a clear preserve-not-reopen boundary

Sprint 45 should not reopen:

- public iterative API redesign
- explicit public workspace APIs
- eigensolver workspace reuse work that belongs to Sprint 46
- dead-code topology changes
- cross-platform CI contract changes
- broad benchmark framework rewrites unrelated to repeated-solve comparisons
- broad documentation/tutorial refresh that depends on future public API work

Interpretation:

- the correct Sprint 45 shape is:
  - land internal reusable iterative workspaces
  - preserve one-shot public APIs as compatibility wrappers
  - add bounded repeated-solve benchmark evidence
- broader memory-model or public-API changes remain later Epic 4 work

#### 5. The strongest Day 1 iterative seams are already visible in the live code

The highest-value live targets are:

- scalar CG packed workspace:
  - `r`
  - `z`
  - `p`
  - `Ap`
- matrix-free CG packed workspace
- GMRES restart / Arnoldi / Hessenberg workspace bundle
- block CG `n * nrhs` work bundles plus per-column side buffers
- MINRES packed workspace
- stagnation / residual-history support that should remain compatible with the
  new workspace ownership model

Interpretation:

- Day 2 should classify shared packed-buffer patterns separately from
  solver-specific state and wrapper logic
- the workspace design should be driven by these real allocation shapes rather
  than by a generic "add context objects everywhere" approach

#### 6. The Day 1 landing order is fixed before implementation starts

The correct early sprint order is:

1. baseline and iterative seam inventory
2. reusable workspace API design
3. shared iterative buffer-layer design
4. shared buffer landing
5. CG / GMRES migration
6. block-path migration
7. wrapper + benchmark work

Interpretation:

- Sprint 45 should preserve Sprint 40's core rule: structural refactors should
  be guided by measured seams and explicit ownership boundaries before code
  movement lands

## Day 2

**Objective:** Refresh the internal seam inventory inside
`src/sparse_iterative.c` so Sprint 45's workspace landing order is grounded in
the live post-Sprint-44 file rather than only in the project-plan labels, with
explicit separation between shared packed-buffer patterns, solver-specific
state, wrapper-only paths, and the already-existing BiCGSTAB workspace seam.

### Commands Run

1. Re-read the Sprint 45 Day 2 plan section:
   - `sed -n '56,92p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
2. Re-read the Day 1 baseline artifact:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_45/artifacts/day1-scope-and-iterative-baseline.md`
3. Re-read the public iterative surface and the current internal BiCGSTAB
   workspace precedent:
   - `sed -n '1,260p' include/sparse_iterative.h`
   - `sed -n '1,260p' src/sparse_bicgstab_internal.h`
4. Refresh the live iterative seam markers and function map:
   - `rg -n "sparse_solve_cg|sparse_solve_cg_mf|sparse_solve_gmres|sparse_solve_gmres_mf|sparse_cg_solve_block|sparse_gmres_solve_block|sparse_solve_minres|sparse_minres_solve_block|sparse_solve_bicgstab|sparse_bicgstab_solve_block|sparse_solve_bicgstab_mf|stag_|reshist_|workspace_alloc|workspace_free|malloc|calloc" src/sparse_iterative.c`
5. Re-read the main scalar / matrix-free / GMRES regions directly:
   - `sed -n '140,980p' src/sparse_iterative.c`
6. Re-read the block, MINRES, and BiCGSTAB regions directly:
   - `sed -n '980,2360p' src/sparse_iterative.c`

### Day 2 Findings

#### 1. The iterative subsystem now reduces cleanly to six workspace seam classes

The current file maps cleanly to these regions:

- shared support state:
  - `stag_tracker_t`
  - `reshist_t`
  - verbose/progress helpers
- scalar CG family:
  - `sparse_solve_cg(...)`
  - `sparse_solve_cg_mf(...)`
- GMRES family:
  - `sparse_solve_gmres(...)`
  - `sparse_solve_gmres_mf(...)`
- block / multi-RHS family:
  - `sparse_cg_solve_block(...)`
  - `sparse_gmres_solve_block(...)`
  - `sparse_minres_solve_block(...)`
  - `sparse_bicgstab_solve_block(...)`
- MINRES family:
  - `sparse_solve_minres(...)`
- existing separate-workspace precedent:
  - `sparse_solve_bicgstab(...)`
  - `sparse_solve_bicgstab_mf(...)`
  - `bicgstab_workspace_t`

Interpretation:

- Sprint 45 is not solving one flat "iterative allocation" problem
- it is solving a shared packed-buffer and repeated-solve problem across a
  few distinct solver families, with BiCGSTAB already sitting in a partially
  solved bucket

#### 2. The strongest shared extraction targets are the packed contiguous vector bundles

The clearest shared allocation patterns are:

- four-vector `n`-sized bundles:
  - CG
  - matrix-free CG
- GMRES full packed bundle:
  - Arnoldi basis `v`
  - Hessenberg `h`
  - Givens vectors `cs` / `sn`
  - Hessenberg RHS / solve scratch `g` / `y`
  - temporary work vector `w`
- block `n * nrhs` bundles:
  - `R`
  - `Z`
  - `P`
  - `AP`
- MINRES packed `nvecs * n` bundle:
  - six-vector unpreconditioned path
  - eight-vector preconditioned path

Interpretation:

- the first reusable workspace design should center on packed bundle ownership
  and typed slice views
- the shared seam is more about "contiguous work-buffer families" than about
  abstract solver contexts

#### 3. Some important state should remain solver-local even after workspace reuse lands

The main solver-local state that should not be forced into a generic shared
schema is:

- CG recurrence scalars:
  - `rz`
  - `alpha`
  - `beta`
- GMRES restart/Arnoldi control:
  - `m`
  - `total_iter`
  - restart loop state
  - left-vs-right preconditioning behavior
- MINRES Lanczos / QR state:
  - `cs`, `sn`
  - `cs_old`, `sn_old`
  - `phi_bar`, `beta_old`
  - direction-vector rotation order
- BiCGSTAB recurrence and half-step stabilization state

Interpretation:

- Sprint 45 should share buffer ownership and reset logic
- it should not try to unify solver math state into one generic iterative
  state machine

#### 4. The block solver family splits into one true workspace target and three wrapper-style follow-ons

The live block paths are not all equivalent:

- `sparse_cg_solve_block(...)` is a real direct workspace target:
  - its own `n * nrhs` bundles
  - shared block SpMV path
  - per-column side buffers
- `sparse_gmres_solve_block(...)` is currently a wrapper-style column loop over
  scalar GMRES
- `sparse_minres_solve_block(...)` is currently a wrapper-style column loop over
  scalar MINRES
- `sparse_bicgstab_solve_block(...)` is currently a wrapper-style column loop
  over scalar BiCGSTAB

Interpretation:

- the main Sprint 45 block-workspace target is block CG
- block GMRES / MINRES / BiCGSTAB are primarily compatibility-wrapper and
  repeated-call composition surfaces unless the sprint later chooses to widen
  scope deliberately

#### 5. The strongest wrapper-vs-reusable-core split is already visible

The clearest one-shot wrapper / reusable-core separations are:

- matrix-backed wrappers over matrix-free kernels:
  - `sparse_solve_gmres(...)` -> `sparse_solve_gmres_mf(...)`
- direct per-column block wrappers:
  - `sparse_gmres_solve_block(...)`
  - `sparse_minres_solve_block(...)`
  - `sparse_bicgstab_solve_block(...)`

Interpretation:

- later Sprint 45 wrapper work should formalize and preserve this layering
- it should not invent a second wrapper model after the reusable internals land

#### 6. BiCGSTAB is the main "use as precedent, not first migration target" seam

BiCGSTAB already owns:

- explicit workspace type
- dedicated alloc/free seam
- contiguous vector ownership

But it still differs from the Sprint 45 main target set because:

- the workspace lives in a solver-specific internal header
- it is not yet a generalized iterative workspace model
- the main repeated-allocation pain still sits in CG / GMRES / block CG /
  MINRES

Interpretation:

- Day 3 should treat BiCGSTAB as design evidence
- it should not let BiCGSTAB pull the first migration batch away from the
  bigger one-shot allocation seams

#### 7. The first-phase adoption order is now concrete

The correct first-phase workspace rollout is:

1. shared support/buffer layer
2. scalar CG and matrix-free CG
3. GMRES and matrix-free GMRES
4. block CG
5. MINRES if the shared layer remains cleanly extensible
6. wrapper normalization and repeated-solve measurement

Explicit later/defer classification:

- use as precedent, not first landing:
  - BiCGSTAB
- likely Sprint 45 wrapper/composition surfaces rather than primary workspace
  migration targets:
  - block GMRES
  - block MINRES
  - block BiCGSTAB

Interpretation:

- Sprint 45 now has a bounded internal rollout order driven by the live file
- the next design work should focus on one reusable packed-buffer model that
  fits CG first and still scales to GMRES/block/MINRES
