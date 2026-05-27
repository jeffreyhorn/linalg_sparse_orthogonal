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
