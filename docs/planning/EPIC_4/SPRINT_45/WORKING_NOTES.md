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

## Day 3

**Objective:** Turn the Day 2 seam inventory into a concrete internal reusable
workspace model for Sprint 45 by defining the ownership objects, typed solver
views, reset/resize rules, wrapper boundary, and the role of preconditioners,
matrix-free operators, and repeated stable-dimension solves before any code
edits land.

### Commands Run

1. Re-read the Sprint 45 Day 3 plan section:
   - `sed -n '94,126p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
2. Re-read the Day 2 seam inventory artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_45/artifacts/day2-iterative-workspace-seam-inventory.md`
3. Re-read the current BiCGSTAB internal workspace precedent:
   - `sed -n '1,220p' src/sparse_bicgstab_internal.h`
4. Re-read the shared allocation-helper surface that the new design should
   build on:
   - `sed -n '1,220p' src/sparse_alloc_internal.h`
5. Refresh the live iterative object/function landmarks in
   `src/sparse_iterative.c`:
   - `rg -n "typedef struct|cg_defaults|gmres_defaults|stag_tracker_t|reshist_t|bicgstab_workspace_t|sparse_solve_cg|sparse_solve_gmres|sparse_cg_solve_block|sparse_solve_minres" src/sparse_iterative.c`

### Day 3 Findings

#### 1. Sprint 45 should use one shared storage owner plus typed solver views, not one giant generic iterative context

The strongest bounded design shape is:

- one shared internal storage owner for contiguous reusable memory
- typed solver views layered on top of that owner:
  - CG view
  - GMRES view
  - block-CG view
  - later-extensible MINRES view
- wrapper-owned temporary workspaces for one-shot public entry points

Interpretation:

- the reusable model should standardize allocation, capacity, and slicing
- it should not flatten all solver control state into a single public- or
  internal-facing mega-struct

#### 2. The shared owner should be dimension/capacity-centric, not algorithm-centric

The storage owner should primarily track:

- contiguous `double` buffer
- optional contiguous auxiliary integer/bool buffer when a solver family needs
  it
- current capacity metadata:
  - `n`
  - `nrhs`
  - restart / Krylov dimension
  - allocated scalar counts for each underlying contiguous region
- a "prepared for solver family X" notion via typed prepare functions rather
  than via a generic runtime tag switch in every iteration step

Interpretation:

- ownership should answer:
  - how much memory is reserved
  - whether the current solve can reuse it
  - where each typed slice begins
- algorithm code should still consume typed pointers rather than raw offsets

#### 3. The first-phase typed view set is now explicit

The first-phase internal views should be:

- `cg_workspace_view`
  - `r`
  - `z`
  - `p`
  - `Ap`
- `gmres_workspace_view`
  - `v`
  - `h`
  - `cs`
  - `sn`
  - `g`
  - `y`
  - `w`
- `block_cg_workspace_view`
  - `R`
  - `Z`
  - `P`
  - `AP`
  - `bnorms`
  - `rz`
  - `conv`
  - `rnorms`
- `minres_workspace_view`
  - `v`
  - `v_old`
  - `w`
  - `d0`
  - `d1`
  - `d2`
  - optional `z`
  - optional `z_tmp`

Interpretation:

- the solver-specific APIs should consume views with already-sliced members
- the shared storage owner should remain hidden behind these typed preparation
  steps

#### 4. Reset/reuse should be explicit and cheap between stable-dimension solves

The correct ownership/lifecycle contract is:

- create once for a bounded family/shape
- prepare a typed view for a requested `(n, restart, nrhs, precond-shape)`
- reuse when the new request fits within current capacity
- resize/reallocate only when the request exceeds current capacity
- destroy explicitly at the owning wrapper or repeated-solve caller boundary

Reset rules should be narrow:

- solver code resets its working slices before each solve
- the owner does not promise mathematical-state preservation across solves
- residual-history output buffers remain caller-owned and outside the workspace
- stagnation tracker state should be reset per solve even if its backing
  storage is reused

Interpretation:

- Sprint 45 should optimize allocator churn, not preserve old iterative state
- "reuse" means reuse of capacity and contiguous storage, not resume-from-old
  Krylov-state semantics

#### 5. Resize/reject rules should stay internal and conservative in Sprint 45

The correct Day 3 rule set is:

- internal reusable helpers may resize when a request outgrows capacity
- wrapper-owned temporary workspaces should simply create, use, and destroy
- there is no Sprint 45 public "workspace mismatch" API surface
- failures remain existing internal failure classes:
  - overflow -> `SPARSE_ERR_ALLOC`
  - allocation failure -> `SPARSE_ERR_ALLOC`

Interpretation:

- Sprint 45 should not add public partial-state or mismatch-management
  semantics
- resize policy remains an internal implementation detail of the reusable
  workspace layer

#### 6. Preconditioners and matrix-free operators affect prepare rules, not the shared owner model

The main interaction rules are:

- matrix-backed and matrix-free CG should share the same CG workspace layout
- matrix-backed and matrix-free GMRES should share the same GMRES workspace
  layout
- optional preconditioner-dependent extra vectors stay encoded in the typed
  view preparation contract
- the shared owner should not know anything about callback function pointers or
  matrix objects beyond the size/capacity request

Interpretation:

- the typed prepare layer should absorb shape differences
- the low-level storage owner should stay allocation-focused rather than solver
  callback-aware

#### 7. BiCGSTAB should remain a separate precedent seam in Sprint 45

BiCGSTAB proves that:

- internal reusable contiguous solver-owned workspaces fit this codebase
- explicit alloc/free helpers are already acceptable style

But BiCGSTAB should remain separate for this sprint because:

- it already has a stable dedicated internal header
- its main value to Sprint 45 is precedent, not priority
- forcing the first shared owner design to absorb BiCGSTAB immediately would
  widen the batch before CG / GMRES / block-CG land

Interpretation:

- Sprint 45 should design with BiCGSTAB in mind
- it should not require Day 5 or Day 6 to migrate BiCGSTAB immediately

#### 8. The wrapper boundary is now explicit

Sprint 45 should keep:

- current public one-shot APIs as convenience wrappers
- wrapper-local temporary workspace ownership for existing entry points
- repeated-solve efficiency primarily as an internal capability plus benchmark
  proof in this sprint

What remains internal-only in Sprint 45:

- storage-owner types
- typed workspace-view types
- prepare/reset/resize helpers

Interpretation:

- Day 5+ implementation should land the internal capability first
- public explicit workspace APIs remain outside Sprint 45 scope

## Day 4

**Objective:** Turn the Day 3 internal workspace model into a bounded shared
iterative buffer-layer implementation plan by defining which helpers belong in
the common layer, which stay solver-local, what zero/reset behavior the common
layer owns, and what validation shape the first code batches must satisfy.

### Commands Run

1. Re-read the Sprint 45 Day 4 plan section:
   - `sed -n '127,162p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
2. Re-read the Day 3 workspace API design artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_45/artifacts/day3-reusable-workspace-api-design.md`
3. Re-read the Sprint 40 validation anchor and Sprint 41 prep rules:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
4. Refresh the live iterative helper/allocation markers:
   - `rg -n "sparse_matvec_block|vec_|stag_|reshist_|callback|progress_cb|precond|malloc|calloc|workspace" src/sparse_iterative.c`
5. Re-read the strongest likely repeated-solve benchmark surfaces:
   - `sed -n '1,220p' benchmarks/bench_convergence.c`
   - `sed -n '1,220p' benchmarks/bench_refactor.c`

### Day 4 Findings

#### 1. The shared buffer layer should be one small internal seam, not a second iterative subsystem

The bounded shared layer should own:

- checked reserve/grow logic for contiguous storage
- typed prepare helpers for:
  - CG
  - GMRES
  - block CG
  - later-extensible MINRES
- slice derivation from one owned contiguous allocation
- reset/zero helpers only where buffer reuse semantics need them to stay
  consistent

It should not own:

- solver stopping logic
- progress/logging callbacks
- preconditioner callback invocation
- matrix or matvec ownership
- residual-history output ownership

Interpretation:

- Day 5 should land one focused internal buffer layer
- it should not widen into a parallel "iterative framework" rewrite

#### 2. The common helper vocabulary is now explicit

The strongest shared helper set is:

- owner lifecycle:
  - init
  - free
  - reserve/ensure capacity
- typed preparation:
  - prepare CG view
  - prepare GMRES view
  - prepare block-CG view
  - optional later prepare MINRES view
- narrow reset helpers:
  - zero contiguous slices before a solve when the solver depends on a clean
    initial state
  - reset reusable support state for stagnation tracking when storage is reused

Interpretation:

- the common layer should export view-preparation helpers, not raw offset math
- the common layer should own the allocation and slicing contract that the
  solver bodies currently hand-roll

#### 3. Several important helpers should remain solver-local even after the shared layer lands

Keep solver-local:

- `stag_tracker_t` behavior and update rules
- `reshist_t` and residual-history recording
- `iter_report(...)`
- GMRES-specific early-exit checks
- MINRES QR/Lanczos update sequencing
- block solver convergence aggregation logic
- wrapper-level per-column dispatch loops

Interpretation:

- these are algorithm-flow or policy helpers, not shared contiguous-storage
  helpers
- Sprint 45 should not over-generalize them into the common layer

#### 4. Zeroing/reset behavior should be explicit but narrow

The common layer should guarantee:

- newly grown capacity is safe to initialize before use
- typed prepare helpers leave view metadata valid
- callers can request clean zeroed storage where the solver currently depends
  on `calloc`-style semantics

But the common layer should not promise:

- preservation of old vector contents across solves
- hidden residual-history retention
- automatic solver-state reconstruction after partial use

Interpretation:

- reset semantics should be "make the workspace ready for a fresh solve"
- they should not imply resumable iterative state

#### 5. The validation shape for implementation days is now concrete

For any `*.c` / `*.h` change:

- mandatory floor:
  - `make format`
  - `make lint`
  - `make test`

For substantial shared-layer or multi-solver migration batches:

- stronger default:
  - `make quality-review-full`

Targeted follow-on checks when the touched surface justifies them:

- direct touched iterative binaries:
  - `./build/test_iterative`
  - `./build/test_block_solvers`
  - `./build/test_minres`
  - `./build/test_bicgstab`
  - `./build/test_stagnation`
- touched example reruns if `examples/` change:
  - `./build/example_iterative`
  - `./build/example_matrix_free`
- repeated-solve benchmark reruns once the benchmark batch lands:
  - touched iterative benchmark binaries, likely centered on
    `bench_convergence`

Interpretation:

- Day 5 through Day 11 now have a clear proof model
- the sprint does not need to guess at validation scope per batch

#### 6. The benchmark work should stay bounded to repeated-solve evidence, not generic solver benchmarking

The live benchmark surfaces imply:

- `bench_convergence.c` is the strongest likely repeated-solve comparison
  surface for iterative methods
- `bench_refactor.c` is useful mainly as a shape precedent for repeated-run
  comparison framing, not as a direct implementation target

Interpretation:

- Sprint 45's benchmark batch should compare one-shot repeated calls against
  reusable-workspace repeated calls
- it should avoid a broad benchmark harness redesign

#### 7. The first landing order is now fully fixed

The correct implementation order is:

1. shared internal buffer layer
2. scalar CG / matrix-free CG
3. GMRES / matrix-free GMRES
4. block CG
5. wrapper normalization
6. repeated-solve benchmark batch
7. MINRES only if the shared layer stays cleanly extensible

Interpretation:

- the first implementation batch should prove the shared layer in the smallest
  high-value path first
- Day 6 should not broaden into block or benchmark work early

## Day 5

**Objective:** Land the first real Sprint 45 code batch by introducing the
private reusable iterative workspace owner and proving it in one bounded live
solver path, while preserving the internal-first / compatibility-wrapper
contract and validating the touched iterative surface completely.

### Commands Run

1. Re-read the Day 3 and Day 4 design artifacts before code changes:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_45/artifacts/day3-reusable-workspace-api-design.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_45/artifacts/day4-shared-buffer-layer-design-and-validation-plan.md`
2. Re-read the live iterative hotspot and current scalar-CG path:
   - `sed -n '1,260p' src/sparse_iterative.c`
   - `sed -n '146,340p' src/sparse_iterative.c`
   - `sed -n '340,520p' src/sparse_iterative.c`
3. Land the new internal workspace layer and the first solver adoption:
   - created `src/sparse_iterative_workspace_internal.h`
   - created `src/sparse_iterative_workspace_internal.c`
   - updated `src/sparse_iterative.c`
   - updated `Makefile`
   - updated `CMakeLists.txt`
4. Run the required gate for `*.c` / `*.h` changes:
   - `make format`
   - `make lint`
   - `make test`
5. After a first lint-pass cleanup issue, rerun the authoritative gate from
   the top:
   - `make format`
   - `make lint`
   - `make test`
6. Run targeted touched-surface follow-ons:
   - `./build/test_iterative`
   - `./build/test_stagnation`
7. Inspect the final change surface and branch state:
   - `git diff --stat`
   - `git status --short`

### Day 5 Findings

#### 1. Sprint 45 now has a real shared iterative workspace owner

The new private helper layer is now live:

- `src/sparse_iterative_workspace_internal.h`
- `src/sparse_iterative_workspace_internal.c`

That layer now owns:

- reusable contiguous double storage
- reusable integer side-buffer storage
- checked reserve/grow behavior
- typed workspace preparation for:
  - CG
  - GMRES
  - block CG
  - MINRES

Interpretation:

- the shared layer landed as an internal storage/view seam, not as a second
  iterative subsystem
- Sprint 45 now has the common allocation/capacity foundation it needs for the
  later solver migrations

#### 2. The first proof integration stayed correctly narrow

The first live adoption was:

- scalar `sparse_solve_cg(...)`

That path now:

- initializes a `sparse_iter_workspace_t`
- prepares a typed `sparse_cg_workspace_view_t`
- binds:
  - `r`
  - `z`
  - `p`
  - `Ap`
- frees the shared owner on all exits instead of managing a raw local `work`
  buffer

Interpretation:

- Day 5 proved the new seam in the smallest high-value path first
- the batch did not broaden prematurely into GMRES, block solvers, or
  benchmark work

#### 3. The shared-vs-local boundary held during the code landing

The new shared layer owns:

- contiguous storage ownership
- checked shape/count math
- typed slice preparation
- narrow zero/reset behavior where needed by prepare helpers

The following stayed solver-local:

- CG recurrence math
- stagnation tracking policy
- residual-history ownership
- callback/progress behavior
- preconditioner invocation choreography

Interpretation:

- the Day 4 boundary was preserved in code, not just in planning language
- Sprint 45 is still doing structural reuse work rather than algorithm rewrite

#### 4. The build surfaces now know about the new iterative helper layer

The new private source is wired into both maintained build systems:

- `Makefile`
- `CMakeLists.txt`

Interpretation:

- later Sprint 45 migrations can adopt the helper layer without additional
  build-system churn
- the shared workspace seam is now part of the normal maintained library
  surface

#### 5. Validation passed cleanly after one small migration-boundary cleanup

Because `*.c` and `*.h` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

All passed on the authoritative rerun.

The only issue surfaced during the first lint pass was:

- stale cleanup references in `src/sparse_iterative.c` after switching scalar
  CG from a raw local `work` buffer to the shared workspace owner

That was fixed immediately, then the full gate was rerun from the top and
passed.

Targeted touched-surface reruns also passed:

- `./build/test_iterative`
- `./build/test_stagnation`

Interpretation:

- the new shared layer did not destabilize the touched iterative regression
  surfaces
- the first workspace landing now has a clean validated baseline for Day 6

#### 6. The next migration order is now clearer after the first code batch

Day 5 leaves the following clean next steps:

1. matrix-free CG adoption of the same owner/view seam
2. GMRES adoption of the already-landed typed prepare helper
3. block-CG adoption of the shared double/int owner path
4. later repeated-solve benchmark evidence once the main live paths use the
   new seam

Interpretation:

- Day 5 converted the Sprint 45 workspace plan from design-only to live code
- later sprint days can now focus on controlled migration breadth rather than
  on inventing the common layer

## Day 6

**Objective:** Convert the main remaining scalar repeated-solve paths onto the
Day 5 shared workspace seam without widening into block paths, benchmarks, or
public API changes, and close the batch from a fully validated baseline.

### Commands Run

1. Re-read the Sprint 45 Day 6 plan section and the Day 5 handoff:
   - `sed -n '191,218p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_45/artifacts/day5-shared-iterative-buffer-layer-batch1.md`
2. Re-read the live matrix-free CG and GMRES paths before editing:
   - `sed -n '340,760p' src/sparse_iterative.c`
   - `sed -n '760,1260p' src/sparse_iterative.c`
3. Reconfirm the landed Day 5 helper surface:
   - `sed -n '1,220p' src/sparse_iterative_workspace_internal.h`
   - `sed -n '1,260p' src/sparse_iterative_workspace_internal.c`
4. Land the bounded Day 6 migration in `src/sparse_iterative.c`:
   - matrix-free CG
   - matrix-free GMRES
   - matrix-backed GMRES via its existing wrapper delegation
5. Run the required gate for `*.c` changes:
   - `make format`
   - `make lint`
   - `make test`
6. Run targeted touched-surface follow-ons:
   - `./build/test_iterative`
   - `./build/test_stagnation`
   - `./build/example_matrix_free`
7. Inspect the final change surface and branch state:
   - `git diff --stat`
   - `git status --short`

### Day 6 Findings

#### 1. Sprint 45's primary scalar iterative paths now share one reusable-workspace model

The Day 6 migration now covers:

- `sparse_solve_cg_mf(...)`
- `sparse_solve_gmres_mf(...)`

And because the matrix-backed GMRES entry point is already a wrapper over the
matrix-free core:

- `sparse_solve_gmres(...)` now participates automatically in the same shared
  workspace path

Interpretation:

- Sprint 45's primary repeated-solve scalar paths are no longer split between
  raw one-shot heap bundles and the new shared owner
- the iterative subsystem now has one coherent CG/GMRES workspace story across
  matrix-backed and matrix-free variants

#### 2. Matrix-free CG adopted the Day 5 shared seam cleanly

`sparse_solve_cg_mf(...)` now:

- initializes `sparse_iter_workspace_t`
- prepares `sparse_cg_workspace_view_t`
- binds:
  - `r`
  - `z`
  - `p`
  - `Ap`
- frees the shared owner on all exits

Interpretation:

- the Day 5 CG view was correctly general enough for both matrix-backed and
  matrix-free CG
- no extra CG-specific workspace redesign was needed for Day 6

#### 3. GMRES adopted the shared seam without reopening algorithm control

`sparse_solve_gmres_mf(...)` now uses the shared GMRES typed view for:

- Arnoldi basis storage
- Hessenberg storage
- Givens rotation scratch
- Hessenberg-space residual/solve vectors
- the main temporary work vector

But it still keeps solver-local:

- restart-loop control
- Arnoldi / lucky-breakdown flow
- callback/progress handling
- preconditioner-side branching
- final true-residual checks

Interpretation:

- the Day 3 / Day 4 shared-vs-local boundary held in the GMRES migration too
- Day 6 changed storage ownership, not GMRES behavior policy

#### 4. The Day 5 helper layer proved sufficient as designed

Day 6 did not require:

- new public APIs
- a second helper layer
- special-case matrix-free workspace objects
- block-path helper expansion

The existing Day 5 internal layer was sufficient for:

- CG matrix-free adoption
- GMRES matrix-free adoption
- matrix-backed GMRES via wrapper composition

Interpretation:

- Sprint 45's implementation order is working as intended
- Day 5's shared owner + typed-view design was broad enough for the first
  primary-solver migration batch

#### 5. Validation closed cleanly for the touched CG/GMRES paths

Because `*.c` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

All passed.

Targeted touched-surface follow-ons also passed:

- `./build/test_iterative`
- `./build/test_stagnation`
- `./build/example_matrix_free`

Representative live outcomes:

- direct iterative reruns kept all matrix-free CG / GMRES tests green
- `example_matrix_free` converged in `3` iterations both with and without the
  diagonal preconditioner, with solution error around `1e-13`

Interpretation:

- the shared-workspace migration did not destabilize the main iterative
  repeated-solve paths
- Sprint 45 now has a clean post-Day-6 baseline for the Day 7 audit

#### 6. The remaining Sprint 45 queue is now narrower and more honest

After Day 6, the main remaining iterative reuse queue is:

- post-primary-path audit
- block-CG as the real multi-RHS workspace target
- wrapper-normalization review for block GMRES / MINRES / BiCGSTAB
- repeated-solve benchmark evidence after the internal migration path is more
  complete

Interpretation:

- Day 7 should audit residual allocation churn rather than reopening CG/GMRES
- Day 8 should stay focused on the real multi-RHS workspace seam instead of
  broad wrapper churn

## Day 7

**Objective:** Audit the post-Day-6 iterative state so Sprint 45's remaining
workspace queue is reduced to real live targets, with explicit separation
between true block-workspace migration work, wrapper-style paths, and solver-
local or specialized seams that should remain later follow-ons.

### Commands Run

1. Re-read the Sprint 45 Day 7 plan section:
   - `sed -n '218,250p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
2. Re-read the Day 6 migration artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_45/artifacts/day6-cg-gmres-migration-batch1.md`
3. Refresh the live iterative function/seam map after Day 6:
   - `rg -n "sparse_solve_cg_mf|sparse_solve_gmres_mf|sparse_cg_solve_block|sparse_gmres_solve_block|sparse_solve_minres|sparse_minres_solve_block|sparse_solve_bicgstab|sparse_bicgstab_solve_block|malloc|calloc|sparse_iter_workspace" src/sparse_iterative.c`
4. Re-read the remaining block and MINRES regions directly:
   - `sed -n '956,1288p' src/sparse_iterative.c`
   - `sed -n '1288,1715p' src/sparse_iterative.c`
5. Re-read the current private iterative workspace and BiCGSTAB precedent
   surfaces:
   - `sed -n '1,260p' src/sparse_iterative_workspace_internal.h`
   - `rg -n "prepare_block_cg|prepare_minres|block_cg_workspace_view|minres_workspace_view|bicgstab_workspace_t|sparse_bicgstab_solve_block|sparse_solve_bicgstab_mf" src/sparse_iterative_workspace_internal.h src/sparse_iterative_workspace_internal.c src/sparse_bicgstab_internal.h src/sparse_iterative.c`
6. Inspect branch state before writing the audit:
   - `git status --short`
   - `git rev-parse --short HEAD`

### Day 7 Findings

#### 1. The post-Day-6 queue is no longer a generic iterative-workspace backlog

After the Day 5 and Day 6 landings, the remaining iterative surface now splits
cleanly into three buckets:

- one real remaining multi-RHS workspace target:
  - `sparse_cg_solve_block(...)`
- wrapper/composition paths:
  - `sparse_gmres_solve_block(...)`
  - `sparse_minres_solve_block(...)`
  - `sparse_bicgstab_solve_block(...)`
- solver-local or specialized later seams:
  - scalar MINRES packed workspace
  - the existing BiCGSTAB workspace model
  - support state such as stagnation/history tracking

Interpretation:

- Sprint 45 no longer needs to think in terms of "migrate all iterative paths"
- the real Day 8 target is narrower and clearer than the original project-plan
  phrasing

#### 2. Block CG is the only strong remaining direct workspace migration target

`sparse_cg_solve_block(...)` still owns a true repeated-allocation bundle:

- `R`
- `Z`
- `P`
- `AP`
- `bnorms`
- `rz`
- `conv`
- `rnorms`

That bundle aligns directly with the already-landed:

- `sparse_block_cg_workspace_view_t`
- `sparse_iter_workspace_prepare_block_cg(...)`

Interpretation:

- Day 8 should target block CG first
- the next migration batch does not need a new helper redesign before it can
  start

#### 3. Block GMRES, block MINRES, and block BiCGSTAB are mostly wrapper surfaces

The live code shape matters:

- `sparse_gmres_solve_block(...)` is a per-column loop over scalar GMRES
- `sparse_minres_solve_block(...)` is a per-column loop over scalar MINRES
- `sparse_bicgstab_solve_block(...)` is a per-column loop over scalar
  BiCGSTAB

Interpretation:

- these are not the same kind of workspace problem as block CG
- they belong in the wrapper/composition review bucket unless a very small
  follow-on naturally falls out later

#### 4. MINRES remains a valid later workspace candidate, but not the best Day 8 target

Scalar MINRES still uses its own one-shot 6/8-vector packed workspace and the
private helper layer already contains `sparse_iter_workspace_prepare_minres(...)`.

But compared with block CG:

- MINRES still carries more solver-specific control/state coupling
- block MINRES itself is currently just a wrapper path

Interpretation:

- MINRES remains a real later workspace extension candidate
- it should not displace block CG as the next Sprint 45 implementation target

#### 5. BiCGSTAB remains a precedent seam, not the next migration target

BiCGSTAB already uses:

- `bicgstab_workspace_t`

for both scalar and matrix-free paths.

Interpretation:

- Sprint 45 should continue treating BiCGSTAB as:
  - a useful precedent
  - a compatibility/reference seam
- not as the next major migration target

#### 6. No new helper-layer redesign is needed before Day 8

The current private helper surface already includes:

- `sparse_iter_workspace_prepare_block_cg(...)`
- `sparse_iter_workspace_prepare_minres(...)`

Interpretation:

- the helper layer is already sufficient for the next real block-CG landing
- the right Day 8 move is adoption, not API growth

#### 7. The real next queue is now explicitly sequenced

The correct remaining Sprint 45 order is:

1. block-CG migration
2. wrapper/composition review
3. repeated-solve benchmark evidence
4. optional later MINRES extension only if it stays small

Interpretation:

- Day 7 did the job it needed to do: it turned the residual queue into a live
  code-driven sequence rather than a generic backlog

## Day 8

**Objective:** Land the one real remaining direct workspace migration target
identified on Day 7 by moving `sparse_cg_solve_block(...)` onto the already-
landed shared block-CG workspace seam, while keeping Sprint 45 away from block
GMRES/MINRES/BiCGSTAB churn and preserving current one-shot public behavior.

### Commands Run

1. Re-read the Sprint 45 Day 8 plan section and the Day 7 audit:
   - `sed -n '249,285p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_45/artifacts/day7-primary-workspace-landing-audit.md`
2. Re-read the live block-CG implementation and the private workspace helper
   seam:
   - `sed -n '956,1205p' src/sparse_iterative.c`
   - `sed -n '1,260p' src/sparse_iterative_workspace_internal.h`
   - `sed -n '120,210p' src/sparse_iterative_workspace_internal.c`
3. Implement the bounded block-CG migration in:
   - `src/sparse_iterative.c`
4. Run the required code-quality gate:
   - `make format`
   - `make lint`
   - `make test`
5. Run targeted touched-surface follow-ons:
   - `./build/test_iterative`
   - `./build/test_block_solvers`
6. Confirm final state:
   - `git status --short`
   - `git rev-parse --short HEAD`

### Day 8 Findings

#### 1. Block CG now participates in the same reusable workspace model as the primary scalar paths

`sparse_cg_solve_block(...)` now:

- initializes `sparse_iter_workspace_t`
- prepares `sparse_block_cg_workspace_view_t`
- binds:
  - `R`
  - `Z`
  - `P`
  - `AP`
  - `bnorms`
  - `rz`
  - `conv`
  - `rnorms`
- frees the shared owner on all touched exit paths

Interpretation:

- Sprint 45 no longer has a split repeated-allocation story between:
  - scalar CG / matrix-free CG
  - and the true multi-RHS CG path
- Day 5's typed block-CG prepare seam was sufficient as designed; no helper
  redesign was needed before adoption

#### 2. The batch reduced real multi-RHS allocation churn without turning into a new block-only framework

The old block-CG path owned direct per-call heap allocation for:

- four packed `n * nrhs` vector bundles
- three per-column side buffers
- one convergence-state buffer

Day 8 replaced that with the already-landed shared owner/view seam rather than
introducing another specialized block allocator.

Interpretation:

- the sprint still uses one coherent iterative workspace model
- Day 8 reduced real repeated allocation churn instead of just rearranging
  wrapper code

#### 3. The bounded Day 7 boundary held

This batch did **not** widen into:

- block GMRES
- block MINRES
- block BiCGSTAB
- MINRES scalar workspace migration
- benchmark/example implementation churn
- public iterative API changes

Interpretation:

- Sprint 45 stayed focused on the one strong remaining direct workspace target
- wrapper/composition review remains a later queue instead of being forced into
  the same implementation batch

#### 4. Validation passed on both the full gate and the touched iterative/block surfaces

Because `src/sparse_iterative.c` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

All passed.

Targeted touched-surface follow-ons also passed:

- `./build/test_iterative`
- `./build/test_block_solvers`

Representative direct rerun outcomes:

- `test_iterative`
  - all visible CG, CG_mf, GMRES, and GMRES_mf cases passed
- `test_block_solvers`
  - all `15` tests passed
  - `test_block_cg_iteration_count` remained green with:
    - `block_cg iters=17`
    - `single_cg iters=17`

Interpretation:

- the shared-workspace landing preserved the current block-CG behavior contract
- the migration is backed by both the authoritative repo-wide gate and the
  direct touched binary reruns

#### 5. The residual Sprint 45 queue is now clearly wrapper/benchmark oriented

After Day 8, the strongest remaining Sprint 45 buckets are:

- wrapper/composition review:
  - `sparse_gmres_solve_block(...)`
  - `sparse_minres_solve_block(...)`
  - `sparse_bicgstab_solve_block(...)`
- repeated-solve benchmark evidence
- optional later MINRES extension only if it stays small

Interpretation:

- Day 8 closes the main direct workspace migration queue
- the sprint can now pivot from core storage adoption into wrapper clarity and
  efficiency evidence

## Day 9

**Objective:** Make the remaining one-shot block iterative compatibility
wrappers read more explicitly as convenience delegation layers over the scalar
solver entries, while keeping wrapper behavior unchanged and avoiding any new
workspace or algorithm redesign.

### Commands Run

1. Re-read the Sprint 45 Day 9 plan section and the Day 8 handoff:
   - `sed -n '285,325p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_45/artifacts/day8-block-iterative-migration-batch.md`
2. Re-read the live scalar and block wrapper surfaces in:
   - `src/sparse_iterative.c`
3. Confirm the wrapper/test surface concentration:
   - `rg -n "sparse_(solve|cg_solve|gmres_solve|minres_solve|bicgstab_solve)_block|sparse_solve_gmres|sparse_solve_minres|sparse_solve_bicgstab" src/sparse_iterative.c tests/test_block_solvers.c tests/test_minres.c tests/test_bicgstab.c`
4. Implement the bounded wrapper normalization in:
   - `src/sparse_iterative.c`
5. Run the required code-quality gate:
   - `make format`
   - `make lint`
   - `make test`
6. Run targeted touched-wrapper follow-ons:
   - `./build/test_block_solvers`
   - `./build/test_minres`
   - `./build/test_bicgstab`
7. Confirm final state:
   - `git status --short`
   - `git rev-parse --short HEAD`

### Day 9 Findings

#### 1. The block wrapper layer now routes through one explicit internal compatibility helper

Day 9 added one small internal wrapper helper that now owns the common
per-column pattern for:

- `sparse_gmres_solve_block(...)`
- `sparse_minres_solve_block(...)`
- `sparse_bicgstab_solve_block(...)`

That helper now owns:

- column iteration
- per-column scalar-solver delegation
- max-iteration / max-residual aggregation
- aggregate converged / stagnated / breakdown reporting
- first hard-error propagation

Interpretation:

- the wrapper relationship is now explicit in code instead of repeated three
  times
- Day 9 normalized compatibility behavior without changing the solver kernels

#### 2. The batch stayed in the wrapper/composition bucket

This landing did **not** widen into:

- scalar solver algorithm changes
- new workspace APIs
- MINRES workspace migration
- BiCGSTAB workspace redesign
- benchmark/example work
- public API signature changes

Interpretation:

- Sprint 45 stayed inside the post-Day-8 queue it had actually earned
- the Day 9 result is wrapper clarity, not another core iterative refactor

#### 3. The scalar entries remain the behavioral truth while the block wrappers stay convenience layers

After Day 9, the block compatibility shape is clearer:

- block GMRES = repeated delegation to scalar GMRES
- block MINRES = repeated delegation to scalar MINRES
- block BiCGSTAB = repeated delegation to scalar BiCGSTAB

Interpretation:

- the block wrappers now read more honestly as composition surfaces
- later Sprint 45 benchmark work can compare repeated-solve behavior without
  any ambiguity about where the real solver/workspace behavior lives

#### 4. Validation passed on both the full gate and the touched wrapper binaries

Because `src/sparse_iterative.c` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

All passed.

Targeted touched-wrapper follow-ons also passed:

- `./build/test_block_solvers`
- `./build/test_minres`
- `./build/test_bicgstab`

Representative direct rerun outcomes:

- `test_block_solvers`
  - all `15` tests passed
- `test_minres`
  - all `43` tests passed
- `test_bicgstab`
  - all `58` tests passed

Interpretation:

- the wrapper normalization preserved current block compatibility behavior
- the touched wrapper surfaces are directly re-validated beyond the full repo
  gate

#### 5. The remaining Sprint 45 queue is now benchmark-oriented

After Day 9, the strongest remaining Sprint 45 buckets are:

- repeated-solve benchmark design/evidence
- optional later MINRES extension only if it stays obviously small

Interpretation:

- the wrapper/composition cleanup queue is now materially smaller
- Sprint 45 can shift from internal structure cleanup to measured repeated-solve
  evidence

## Day 10

**Objective:** Define the smallest honest repeated-solve benchmark slice that
can demonstrate allocator-churn reduction from the Sprint 45 iterative
workspace landings, while avoiding broader benchmark harness churn or unstable
performance claims.

### Commands Run

1. Re-read the Sprint 45 Day 10 plan section and the Day 9 handoff:
   - `sed -n '325,365p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_45/artifacts/day9-wrapper-compatibility-batch.md`
2. Audit the live benchmark/example surfaces relevant to repeated iterative
   solves:
   - `rg -n "iterative|gmres|cg|repeat|reuse|workspace|benchmark" benchmarks/bench_convergence.c benchmarks/bench_refactor.c benchmarks/bench_bicgstab.c benchmarks/bench_main.c examples/example_iterative.c examples/example_matrix_free.c`
3. Re-read the strongest benchmark structure precedents directly:
   - `sed -n '1,260p' benchmarks/bench_convergence.c`
   - `sed -n '1,220p' benchmarks/bench_refactor.c`
   - `sed -n '460,575p' benchmarks/bench_main.c`
4. Reconfirm the Epic 4 planning/review intent for repeated-solve evidence:
   - `rg -n "benchmark|repeated|repeat|reuse|workspace" docs/planning/EPIC_4/PROJECT_PLAN.md docs/planning/EPIC_4/reviews/todo-codex-2026-05-21.md`
5. Confirm final state:
   - `git status --short`
   - `git rev-parse --short HEAD`

### Day 10 Findings

#### 1. The current benchmark surface still measures one-shot solver calls, not the new reusable-workspace seam directly

The live benchmark surfaces split clearly:

- `bench_convergence.c`
  - convergence tables and residual-vs-iteration history
  - one-shot calls to:
    - `sparse_solve_cg(...)`
    - `sparse_solve_gmres(...)`
- `bench_main --iterative`
  - one-shot iterative timing summaries
  - again centered on public one-shot entry points
- `bench_bicgstab.c`
  - solver comparison benchmark, but not a reusable-workspace measurement seam

Interpretation:

- the existing iterative benchmarks are useful baselines
- they are not yet direct evidence of Sprint 45's internal workspace reuse
- Day 11 should add a repeated-call comparison slice rather than repurposing
  the current convergence-oriented tables wholesale

#### 2. `bench_refactor.c` is the right structural precedent for Day 11

`bench_refactor.c` already has the comparison shape Sprint 45 needs:

- Approach A:
  - repeated one-shot calls
- Approach B:
  - reusable internal state on the same stable shape/problem
- bounded wall-clock comparison
- no broad harness abstraction

Interpretation:

- Day 11 should copy this comparison style rather than inventing a new
  benchmark framework
- the right Sprint 45 evidence form is:
  - repeated one-shot iterative solve
  - repeated reusable-workspace-backed iterative solve
  - stable dimensions and stable operator/preconditioner context

#### 3. The strongest Day 11 target set is scalar CG plus scalar GMRES

The best first repeated-solve targets are:

- scalar CG
- scalar GMRES

Reasons:

- both now participate in the shared iterative workspace seam
- both already have benchmark visibility in:
  - `bench_convergence.c`
  - `bench_main --iterative`
- both avoid the extra interpretation burden of block wrappers or solver-family
  edge cases

Interpretation:

- Day 11 should center on one SPD repeated-CG case and one general repeated-
  GMRES case
- this is enough to demonstrate the Sprint 45 workspace model on the main
  migrated scalar paths

#### 4. Block CG is the only reasonable optional add-on, and only if it stays obviously small

After Day 8, block CG is the only block path that represents a true direct
workspace migration rather than primarily a wrapper/composition surface.

Interpretation:

- if Day 11 has an optional third case, it should be block CG
- block GMRES, block MINRES, and block BiCGSTAB should not be the first
  repeated-solve benchmark targets because their main Day 9 significance is
  wrapper normalization, not unique workspace ownership

#### 5. The benchmark comparison model should stay narrow and claim-safe

The right comparison model is:

- same matrix/problem shape
- same tolerance/options
- same operator/preconditioner context
- repeated call loop with stable dimensions
- report:
  - wall time
  - iteration counts / convergence summary
  - allocation-path interpretation notes

It should **not** claim:

- universal speedups
- machine-independent runtime guarantees
- allocator counts unless Day 11 adds a bounded trustworthy counter

Interpretation:

- Day 11 should produce evidence, not marketing claims
- Sprint 45 needs a narrow repeated-solve measurement slice, not a generalized
  performance-reporting framework

#### 6. The likely Day 11 landing site is a small dedicated benchmark, not a broad `bench_main` CLI expansion

The cleanest implementation shapes are:

- a new small dedicated repeated-solve benchmark source
- or a very small dedicated mode in an existing iterative benchmark file

The least attractive Day 11 move is:

- broadening `bench_main` CLI surface during Sprint 45

Interpretation:

- CLI modernization belongs later in Epic 4
- Day 11 should minimize parser churn and maximize comparison clarity

#### 7. The concrete Day 11 plan is now bounded

The right Day 11 scope is:

1. add one repeated-solve CG case
2. add one repeated-solve GMRES case
3. optionally add one block-CG case only if it stays very small
4. use the `bench_refactor` A/B comparison pattern
5. record measured outputs in the artifact without overstating them

Interpretation:

- Day 10 succeeded: it turned a vague “benchmark evidence” task into a
  specific, low-churn Day 11 plan

## Day 11

**Objective:** Land the narrow repeated-solve benchmark slice defined on Day 10
so Sprint 45 has direct measured evidence for the reusable iterative workspace
seam on the migrated scalar solver paths, without widening into benchmark CLI
or broad framework churn.

### Commands Run

1. Re-read the Sprint 45 Day 11 plan section and the Day 10 benchmark design:
   - `sed -n '365,405p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_45/artifacts/day10-repeated-solve-benchmark-design.md`
2. Audit the current benchmark/build wiring and the Day 9/Day 10 iterative
   surfaces:
   - `rg -n "bench_.*\\.c|BENCH_SRCS|bench_" Makefile CMakeLists.txt`
   - `sed -n '188,208p' Makefile`
   - `sed -n '232,272p' CMakeLists.txt`
   - `sed -n '1,80p' src/sparse_iterative.c`
   - `sed -n '146,340p' src/sparse_iterative.c`
   - `sed -n '567,955p' src/sparse_iterative.c`
   - `sed -n '1,260p' src/sparse_iterative_workspace_internal.c`
3. Land the bounded Day 11 code batch:
   - add `src/sparse_iterative_internal.h`
   - add `benchmarks/bench_iterative_reuse.c`
   - update `src/sparse_iterative.c`
   - update `Makefile`
   - update `CMakeLists.txt`
4. Run the required authoritative gate because `*.c` and `*.h` files changed:
   - `make format`
   - `make lint`
   - `make test`
5. Run direct Day 11 follow-ons:
   - `./build/test_iterative`
   - `./build/bench_iterative_reuse`
6. Confirm final state:
   - `git status --short`
   - `git rev-parse --short HEAD`

### Day 11 Findings

#### 1. Sprint 45 now has a dedicated repeated-solve benchmark surface

Day 11 landed a small dedicated benchmark at:

- `benchmarks/bench_iterative_reuse.c`

It is intentionally narrow:

- scalar CG repeated-solve case
- scalar GMRES repeated-solve case
- one-shot public call loop vs reusable-workspace-backed internal call loop

Interpretation:

- Sprint 45 now has direct benchmark evidence for the migrated scalar workspace
  seam
- the batch avoided `bench_main` CLI churn and broad benchmark framework work

#### 2. The internal reusable-workspace seam is now explicit enough for direct repeated-call benchmarking

Day 11 added a narrow private internal header:

- `src/sparse_iterative_internal.h`

That header exposes only the two internal repeated-solve benchmark entry
points:

- `sparse_solve_cg_with_workspace_internal(...)`
- `sparse_solve_gmres_with_workspace_internal(...)`

Interpretation:

- the benchmark does not need to reach into implementation internals ad hoc
- the new seam is still private and benchmark-oriented, not a public API change

#### 3. The scalar one-shot entries now read clearly as compatibility wrappers over the reusable workspace model

After Day 11:

- `sparse_solve_cg(...)`
  - owns the one-shot local workspace lifecycle
  - delegates solver work to `sparse_solve_cg_with_workspace_internal(...)`
- `sparse_solve_gmres(...)`
  - owns the one-shot local workspace lifecycle
  - delegates to `sparse_solve_gmres_with_workspace_internal(...)`
- `sparse_solve_gmres_mf(...)`
  - now follows the same one-shot wrapper pattern over the reusable workspace
    preparation path

Interpretation:

- Sprint 45 preserved the public one-shot API contract
- repeated-solve benchmarking now targets the same solver logic through a
  reusable private workspace owner instead of duplicating algorithm code

#### 4. The measured repeated-solve result is honest but modest on this local run

Direct benchmark output was:

- CG repeated-solve case:
  - `cg-tridiag-300`
  - one-shot = `24.7220 ms`
  - reuse = `24.7000 ms`
  - speedup = `1.00x`
  - both paths:
    - `17` iterations
    - relative residual `5.192e-11`
    - converged
- GMRES repeated-solve case:
  - `gmres-unsym-220`
  - one-shot = `17.4030 ms`
  - reuse = `17.1030 ms`
  - speedup = `1.02x`
  - both paths:
    - `12` iterations
    - relative residual `7.364e-11`
    - converged

Interpretation:

- the reuse path preserved solver behavior exactly on the benchmarked cases
- allocator-churn reduction is now directly measurable
- the local timing win is small rather than dramatic, which is the right
  claim-safe Sprint 45 result to record

#### 5. The direct touched iterative surface remained green beyond the full repo gate

Direct follow-on reruns passed:

- `./build/test_iterative`
- `./build/bench_iterative_reuse`

Representative direct rerun outcomes:

- `test_iterative`
  - all `76` tests passed
- `bench_iterative_reuse`
  - both repeated-solve benchmark cases completed successfully
  - one-shot and reuse paths matched on convergence/iteration counts

Interpretation:

- the Day 11 batch is validated both through the full repo gate and through the
  exact iterative surfaces it touched

#### 6. Sprint 45 is now positioned for closeout rather than more structural workspace churn

After Day 11, the sprint has now landed:

- shared iterative workspace owner
- scalar CG / matrix-free CG migration
- scalar GMRES / matrix-free GMRES migration
- block-CG migration
- wrapper normalization for block GMRES / MINRES / BiCGSTAB
- direct repeated-solve benchmark evidence for scalar CG and GMRES

Interpretation:

- the remaining Sprint 45 work is now validation/closeout oriented
- no new broad iterative workspace redesign queue surfaced from the benchmark
  batch

## Day 12

**Objective:** Document the new internal iterative workspace contract for later
Epic 4 work, audit the residual repeated-allocation seams still visible in the
iterative surface, and fix the exact Day 13 validation sweep shape before
closeout.

### Commands Run

1. Re-read the Sprint 45 Day 12/13 plan section and the Day 11 handoff:
   - `sed -n '330,420p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_45/artifacts/day11-repeated-solve-benchmark-batch.md`
2. Sweep the live iterative and benchmark surfaces for remaining allocation /
   workspace seams:
   - `rg -n "malloc\\(|calloc\\(|sparse_malloc_|sparse_calloc_|workspace|reshist|stag_|malloc_array|calloc_array" src/sparse_iterative.c src/sparse_bicgstab_internal.h benchmarks/bench_iterative_reuse.c include/sparse_iterative.h`
3. Re-read the current private workspace and residual solver seams directly:
   - `sed -n '1,260p' src/sparse_iterative_workspace_internal.h`
   - `sed -n '1,320p' src/sparse_bicgstab_internal.h`
   - `sed -n '1140,1335p' src/sparse_iterative.c`
   - `sed -n '1320,1665p' src/sparse_iterative.c`
   - `sed -n '1700,2295p' src/sparse_iterative.c`
4. Confirm final state:
   - `git status --short`
   - `git rev-parse --short HEAD`

### Day 12 Findings

#### 1. Sprint 45 now has a clear internal workspace contract rather than a loose implementation trend

The live private workspace contract is now centered on:

- `src/sparse_iterative_workspace_internal.h`
- `src/sparse_iterative_workspace_internal.c`
- `src/sparse_iterative_internal.h`

The contract is now explicit:

- one shared internal owner holds contiguous reusable storage plus capacity
  metadata
- typed solver views are prepared from that owner for:
  - CG
  - GMRES
  - block CG
  - MINRES
- public one-shot scalar entries remain compatibility wrappers that:
  - initialize a local workspace
  - delegate to the reusable internal solver seam
  - free the workspace on return

Interpretation:

- Sprint 45 now has a real maintainer-facing model for repeated-solve
  iterative work
- the new internal seam is explicit enough for later extension without forcing
  a public API redesign in the same sprint

#### 2. The shared workspace seam owns storage and typed views, not solver behavior policy

The current shared workspace layer owns:

- contiguous storage ownership
- checked reserve/grow behavior
- typed view preparation
- reusable capacity across stable dimensions

It does **not** own:

- recurrence scalars
- callback/progress policy
- residual-history policy
- stagnation tracking
- preconditioner behavior
- block-wrapper per-column orchestration

Interpretation:

- later Epic 4 work should keep extending this seam as a storage/layout owner
- solver stopping logic and algorithm policy should stay in the solver-local
  code rather than being pushed down into the shared workspace layer

#### 3. The migrated direct workspace-reuse set is now explicit

After Day 11, the live direct reusable-workspace adoption set is:

- scalar CG
- matrix-free CG
- scalar GMRES
- matrix-free GMRES
- block CG

Interpretation:

- the main repeated-allocation targets Sprint 45 set out to address are now
  materially covered
- the sprint no longer has an implicit “maybe more CG/GMRES migration” queue

#### 4. The remaining iterative surface now falls into three clear residual classes

Residual Class A: wrapper/composition surfaces, not primary workspace targets

- block GMRES
- block MINRES
- block BiCGSTAB

Interpretation:

- these are now mainly compatibility/delegation layers over scalar solves
- they are not the next natural workspace-reuse landing zone

Residual Class B: specialized later solver-local workspace seams

- scalar MINRES
  - still owns a local packed `work` allocation inside `sparse_solve_minres(...)`
- scalar BiCGSTAB
- matrix-free BiCGSTAB
  - already use the separate `bicgstab_workspace_t` precedent in
    `src/sparse_bicgstab_internal.h`

Interpretation:

- MINRES is the clearest still-local repeated-allocation seam left in
  `src/sparse_iterative.c`
- BiCGSTAB is not “missing workspace reuse”; it already has a separate internal
  workspace model and should be treated as a later unification/evolution
  question, not as an unfinished Sprint 45 gap

Residual Class C: support-surface and later-epic non-goals

- repeated-run eigensolver workspace reuse
- public explicit iterative workspace APIs
- broader public docs/tutorial refresh for repeated-solve guidance
- broader benchmark CLI modernization

Interpretation:

- Sprint 45 intentionally stops short of these larger outward-facing or
  eigensolver-oriented queues
- they should be carried as later Epic 4 work rather than treated as hidden
  Sprint 45 incompleteness

#### 5. The Day 13 validation sweep shape is now fixed explicitly

The right Day 13 authoritative sweep is:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

The targeted iterative/benchmark follow-ons justified by the touched Sprint 45
surface are:

- `./build/test_iterative`
- `./build/test_block_solvers`
- `./build/test_minres`
- `./build/test_bicgstab`
- `./build/test_stagnation`
- `./build/bench_iterative_reuse`
- `./build/example_matrix_free`

Interpretation:

- Day 13 now has a concrete validation floor plus a bounded touched-surface
  rerun set
- the sweep should re-check both the migrated workspace paths and the wrapper /
  specialized seams that Sprint 45 interacted with indirectly

#### 6. Sprint 45’s non-goals are now explicit enough for closeout

Sprint 45 intentionally does **not** solve:

- eigensolver repeated-run workspace reuse
- public explicit iterative workspace handles
- broad benchmark harness redesign
- broad tutorial / README refresh for repeated-solve guidance

Interpretation:

- the sprint now has a clear “done vs later” boundary
- Day 14 closeout can hand the remaining repeated-run efficiency work forward
  without pretending the iterative story is fully universalized

## Day 13

**Objective:** Run the authoritative full validation pass for the Sprint 45
iterative workspace and repeated-solve changes, then reconfirm the reviewed
truthfulness anchors and the direct iterative/benchmark surfaces touched during
the sprint.

### Commands Run

1. Run the mandatory code-change floor:
   - `make format`
   - `make lint`
   - `/usr/bin/time -p make test`
2. Run the stronger reviewed local baseline:
   - `/usr/bin/time -p make quality-review-full`
3. Run the Day 12-targeted iterative and benchmark follow-ons:
   - `./build/test_iterative`
   - `./build/test_block_solvers`
   - `./build/test_minres`
   - `./build/test_bicgstab`
   - `./build/test_stagnation`
   - `./build/bench_iterative_reuse`
   - `./build/example_matrix_free`
4. Confirm final state:
   - `git status --short`
   - `git rev-parse --short HEAD`

### Day 13 Findings

#### 1. The full required code-change floor passed cleanly

The mandatory validation floor was:

- `make format`
- `make lint`
- `make test`

All passed.

The timed `make test` rerun completed at:

- `real 83.44`

Interpretation:

- Sprint 45 remains green at the standard code-change floor
- no iterative workspace or benchmark batch introduced a new repository-level
  failure

#### 2. The strongest local reviewed baseline also passed cleanly

The stronger proof run was:

- `make quality-review-full`

It passed at:

- `real 664.75`

That run also completed the reviewed-path dead-code tail successfully:

- `deadcode-check: report completeness checks passed`
- `quality-review: passed (format-check + lint + test + deadcode-check)`
- `quality-review-full: passed (quality-review + quality-review-cmake)`

Interpretation:

- Sprint 45 cleared the strongest routine local reviewed baseline
- the sprint does not close only from direct touched tests; it also clears the
  established reviewed wrapper and reviewed CMake parity path

#### 3. The maintained truthfulness anchors remained exact

The reviewed-parity contract stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake test-count parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 160.79 sec`

Interpretation:

- Sprint 45 did not disturb the maintained local reviewed baseline
- the iterative workspace and repeated-solve changes remain aligned with the
  established Makefile/CMake parity contract

#### 4. The direct iterative and benchmark follow-ons all passed

The Day 12-targeted follow-ons passed:

- `./build/test_iterative`
- `./build/test_block_solvers`
- `./build/test_minres`
- `./build/test_bicgstab`
- `./build/test_stagnation`
- `./build/bench_iterative_reuse`
- `./build/example_matrix_free`

Representative direct rerun outcomes:

- `test_iterative`
  - all `76` tests passed
- `test_block_solvers`
  - all `15` tests passed
  - `block_cg iters=17  single_cg iters=17`
- `test_minres`
  - all `43` tests passed
- `test_bicgstab`
  - all `58` tests passed
- `test_stagnation`
  - all `46` tests passed
- `example_matrix_free`
  - both GMRES runs converged in `3` iterations
  - solution error stayed around `2.7e-13`

Interpretation:

- the migrated direct workspace paths, wrapper/composition surfaces, and
  residual specialized seams all stayed green under focused reruns
- the benchmark/example evidence still composes cleanly with the main solver
  validation surface

#### 5. The repeated-solve benchmark remained behavior-stable but timing-sensitive

The direct Day 13 benchmark rerun produced:

- CG repeated-solve case:
  - one-shot = `26.5910 ms`
  - reuse = `25.9270 ms`
  - speedup = `1.03x`
  - both paths:
    - `17` iterations
    - relative residual `5.192e-11`
    - converged
- GMRES repeated-solve case:
  - one-shot = `18.0780 ms`
  - reuse = `19.3130 ms`
  - speedup = `0.94x`
  - both paths:
    - `12` iterations
    - relative residual `7.364e-11`
    - converged

Interpretation:

- the repeated-solve benchmark remains behavior-stable across reruns
- the timing effect is modest and varies run-to-run
- that confirms the right Sprint 45 claim:
  - the reusable-workspace seam is real and measurable
  - Sprint 45 should not overstate it as a stable universal speedup

#### 6. No new reconciliation queue surfaced

Day 13 did **not** surface:

- reviewed-baseline drift
- Makefile/CMake parity drift
- new iterative solver regressions
- new benchmark harness problems
- a need to reopen the Sprint 45 residual classification

Interpretation:

- Sprint 45 can now close from a measured, validated state
- Day 14 should be a clean closeout/handoff day rather than another
  reconciliation batch

## Day 14

**Objective:** Close Sprint 45 from the Day 13 validated baseline, summarize
the reusable-workspace package as one coherent iterative handoff, and record
exactly what later Epic 4 work inherits next.

### Commands Run

1. Re-read the Sprint 45 Day 14 plan section and the Day 13 validated state:
   - `sed -n '420,470p' docs/planning/EPIC_4/SPRINT_45/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_45/artifacts/day13-full-validation-sweep.md`
2. Re-read the current Day 13 closeout state from working notes:
   - `tail -n 140 docs/planning/EPIC_4/SPRINT_45/WORKING_NOTES.md`
3. Re-read a recent closeout/handoff artifact for format and scope control:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_44/artifacts/day14-closeout-and-handoff.md`
4. Confirm final state:
   - `git status --short`
   - `git rev-parse --short HEAD`

### Day 14 Findings

#### 1. Sprint 45 now hands off one coherent iterative workspace package

Sprint 45 closes with these connected outcomes, not isolated edits:

- shared internal iterative workspace owner
- typed reusable views for:
  - CG
  - GMRES
  - block CG
  - MINRES
- migrated direct reusable-workspace paths:
  - scalar CG
  - matrix-free CG
  - scalar GMRES
  - matrix-free GMRES
  - block CG
- compatibility-preserving one-shot wrapper structure for the touched scalar
  entries
- normalized wrapper/composition surfaces for:
  - block GMRES
  - block MINRES
  - block BiCGSTAB
- repeated-solve benchmark evidence for scalar CG and GMRES

Interpretation:

- Sprint 45 ends with a real internal repeated-solve efficiency package
- later work inherits a coherent workspace model rather than scattered
  allocation cleanups

#### 2. Sprint 45 closes from the Day 13 validated baseline

The sprint closes from:

- `make format` → passed
- `make lint` → passed
- `make test` → passed
- `make quality-review-full` → passed

The preserved truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`

Interpretation:

- Sprint 45 did not trade iterative reuse progress for baseline drift
- the internal workspace landings remain aligned with the maintained reviewed
  local contract

#### 3. The repeated-solve benchmark outcome is now stable enough to hand off honestly

After Day 11 and Day 13, the benchmark contract is now clear:

- repeated one-shot and reusable-workspace-backed paths match on:
  - iteration counts
  - convergence flags
  - residuals
- the runtime effect is measurable but modest
- the timing direction can vary across local reruns

Interpretation:

- later Epic 4 work should treat Sprint 45’s benchmark as bounded behavioral
  evidence plus modest local runtime evidence
- it should **not** inherit an over-claimed “universal iterative speedup”
  narrative

#### 4. The main later inherited queues are now explicit

The strongest next inherited iterative-efficiency queues are:

- scalar MINRES workspace migration / unification with the shared owner
- later unification or evolution of the separate BiCGSTAB workspace precedent
- eigensolver repeated-run workspace reuse
- any future public explicit iterative workspace API design only when a later
  sprint chooses that outward-facing scope directly

Interpretation:

- Sprint 45 narrowed the remaining queue materially
- later repeated-run work is now specialized and bounded rather than broad

#### 5. Sprint 45 intentionally does not widen into public docs or CLI redesign

Sprint 45 intentionally leaves later:

- broader benchmark CLI modernization
- README/tutorial/public repeated-solve guidance refresh
- public explicit workspace handles

Interpretation:

- the sprint solved the internal efficiency seam first
- later sprints can decide whether those outward-facing surfaces are justified
  by enough stable internal reuse benefit

#### 6. No immediate `PROJECT_PLAN.md` adjustment is needed

The Day 12 residual audit and Day 13 sweep did **not** surface:

- a missed Sprint 45 implementation queue
- a newly urgent public-API requirement
- a new validation or truthfulness obligation not already represented in the
  Epic 4 roadmap

Interpretation:

- Sprint 45 does not need a `PROJECT_PLAN.md` adjustment at closeout
- the remaining iterative and eigensolver repeated-run work is already within
  the intended later Epic 4 direction
