# Sprint 46 Working Notes

## Day 1

**Objective:** Turn the Sprint 46 project-plan scope plus the Sprint 40/41/42
execution rules and the Sprint 45 iterative-workspace closeout into a concrete
eigensolver starting point by confirming the preserved reviewed contracts,
naming the Sprint 46 workstreams explicitly, and defining the authoritative
eigensolver solver, test, example, and benchmark inputs before workspace-reuse
implementation begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short`
   - `git rev-parse --abbrev-ref HEAD`
2. Re-read the Sprint 46 plan and the main prerequisite planning artifacts:
   - `sed -n '223,253p' docs/planning/EPIC_4/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_46/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_45/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
3. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Reconfirm the current maintained reviewed/dead-code command surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`
5. Measure the live eigensolver hotspot and current eigensolver-support
   concentration:
   - `wc -l src/sparse_eigs.c tests/test_eigs.c tests/test_eigs_thick_restart.c tests/test_eigs_lobpcg.c benchmarks/bench_eigs.c benchmarks/bench_iterative_reuse.c examples/example_eigs.c src/sparse_iterative_workspace_internal.h src/sparse_iterative_workspace_internal.c`
6. Refresh the live eigensolver seam markers:
   - `rg -n "lanczos|thick[_-]restart|lobpcg|workspace|malloc|calloc|basis|ritz|projected|restart" src/sparse_eigs.c`

### Day 1 Findings

#### 1. Sprint 46 starts from a preserved Sprint 40/41/42/45 baseline, not from baseline repair work

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
  - compatibility-preserving internal-first refactor rules
  - shared matrix/lifecycle state guard rules
- Sprint 45 already left behind the reusable iterative workspace model:
  - `src/sparse_iterative_workspace_internal.h`
  - `src/sparse_iterative_workspace_internal.c`

Interpretation:

- Sprint 46 is not a quality-baseline sprint
- Sprint 46 is an eigensolver workspace and repeated-run efficiency sprint on
  top of an already-validated Epic 4 baseline and an already-landed internal
  reusable-workspace precedent

#### 2. The live eigensolver surface is still concentrated in one real hotspot

The live implementation/test concentration is:

- `src/sparse_eigs.c` = `3151` lines
- primary eigensolver regression surfaces:
  - `tests/test_eigs.c` = `1269`
  - `tests/test_eigs_thick_restart.c` = `1161`
  - `tests/test_eigs_lobpcg.c` = `1196`
- main maintained eigensolver example:
  - `examples/example_eigs.c` = `284`
- strongest likely repeated-run benchmark surface:
  - `benchmarks/bench_eigs.c` = `958`
- direct reusable-workspace precedent from Sprint 45:
  - `src/sparse_iterative_workspace_internal.h` = `76`
  - `src/sparse_iterative_workspace_internal.c` = `215`

Interpretation:

- Sprint 46 does not need another exploratory sprint before it begins
  eigensolver workspace work
- the real repeated-run target is still one concentrated subsystem rather than
  a wide repo-spanning efficiency pass

#### 3. The strongest repeated-run eigensolver targets are explicit before code changes begin

The live seam map shows the main repeated-allocation / repeated-run families
are:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG

The main high-value workspace shapes already visible in `src/sparse_eigs.c`
are:

- `n * m` Krylov basis bundles
- tridiagonal / Ritz / restart scratch
- `n * k` and `n * block_size` block bundles
- dense projected-subproblem intermediates
- thick-restart state carryover buffers

Interpretation:

- Sprint 46 should target reusable basis/scratch ownership first, not broad
  public API redesign
- Day 2 should classify shared packed-buffer patterns separately from
  solver-local spectral control state and wrapper-only behavior

#### 4. Sprint 46 already contains one useful partial precedent, but not the full reusable-workspace model yet

The live code already shows:

- thick-restart carries dedicated restart-state ownership:
  - `lanczos_restart_state_t`
  - `lanczos_restart_state_free(...)`
- but grow-m Lanczos, thick-restart outer-loop scratch, and LOBPCG still
  allocate large work buffers directly inside `src/sparse_eigs.c`
- Sprint 45 already proved the internal reusable-workspace direction in a
  neighboring solver family

Interpretation:

- Sprint 46 is not inventing the first eigensolver state owner from nothing
- Sprint 46 still needs a broader shared reusable workspace/state layer before
  repeated-run efficiency claims become real across the main eigensolver paths

#### 5. The Sprint 46 workstreams are explicit and already bounded by the plan

Day 1 confirms the sprint's eight bounded workstreams directly from the plan:

- eigensolver seam inventory
- reusable workspace/state design
- shared buffer layer
- grow-m / thick-restart migration
- LOBPCG migration
- wrapper preservation
- repeated-run benchmark batch
- memory-behavior documentation and validation closeout

Interpretation:

- the front half of the sprint should stay internal-first:
  - seam inventory
  - workspace/state design
  - shared buffer layer
  - primary eigensolver migration
- the back half should then pivot into:
  - benchmark evidence
  - memory-behavior documentation
  - validation closeout

#### 6. Sprint 46 inherits a clear preserve-not-reopen boundary

Sprint 46 should not reopen:

- public eigensolver API redesign
- explicit public workspace/state APIs
- iterative-solver workspace redesign that already closed in Sprint 45
- dead-code topology changes
- cross-platform CI contract changes
- broad benchmark CLI redesign
- broad documentation/tutorial refresh that depends on future public API work

Interpretation:

- the correct Sprint 46 shape is:
  - land internal reusable eigensolver workspaces/state
  - preserve one-shot public APIs as compatibility wrappers
  - add bounded repeated-run benchmark evidence
  - document the internal memory-behavior contract for maintainers

#### 7. The Day 1 landing order is fixed before implementation starts

The correct early sprint order is:

1. baseline and seam inventory
2. reusable workspace/state design
3. shared buffer-layer design
4. shared buffer landing
5. grow-m / thick-restart migration
6. LOBPCG migration
7. wrapper + benchmark + memory-contract closeout

Interpretation:

- Sprint 46 should preserve Sprint 40's core rule: structural refactors should
  be guided by measured seams and explicit ownership boundaries before code
  movement lands

## Day 2

**Objective:** Refresh the internal seam inventory inside
`src/sparse_eigs.c` so Sprint 46's workspace landing order is grounded in the
live post-Sprint-45 file rather than only in the project-plan labels, with
explicit separation between shared packed-buffer patterns, solver-specific
state, optional/preconditioner-dependent buffers, and one-shot wrapper logic.

### Commands Run

1. Re-read the Sprint 46 Day 2 plan section:
   - `sed -n '56,94p' docs/planning/EPIC_4/SPRINT_46/PLAN.md`
2. Re-read the Day 1 baseline artifact:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_46/artifacts/day1-scope-and-eigensolver-baseline.md`
3. Re-read the public and internal eigensolver surfaces:
   - `sed -n '1,260p' include/sparse_eigs.h`
   - `sed -n '1,260p' src/sparse_eigs_internal.h`
4. Refresh the live eigensolver seam markers and function map:
   - `rg -n "lanczos_iterate|lanczos_thick_restart_iterate|s21_thick_restart_outer_loop|s21_lobpcg_rr_step|s21_lobpcg_solve|malloc|calloc|sparse_malloc_array|workspace|restart_state|basis|projected|sel_idx|theta|alpha|beta|subdiag" src/sparse_eigs.c`
5. Re-read the main grow-m / thick-restart / LOBPCG allocation regions
   directly:
   - `sed -n '240,1315p' src/sparse_eigs.c`
   - `sed -n '1760,2385p' src/sparse_eigs.c`
   - `sed -n '2639,3155p' src/sparse_eigs.c`

### Day 2 Findings

#### 1. The eigensolver subsystem now reduces cleanly to four workspace seam classes

The current file maps cleanly to these regions:

- shared spectral helper/support paths:
  - `lanczos_iterate_op(...)`
  - Ritz extraction / selection helpers
  - dense arrowhead / Jacobi helpers
  - residual recomputation and restart-state assembly support
- grow-m Lanczos path:
  - `sparse_eigs_sym(...)` grow-m branch
  - full basis `V`
  - tridiagonal / Ritz / selection scratch
- thick-restart Lanczos path:
  - `lanczos_restart_state_t`
  - `lanczos_thick_restart_iterate(...)`
  - `s21_thick_restart_outer_loop(...)`
  - bounded restart-phase and arrowhead scratch
- LOBPCG path:
  - `s21_lobpcg_rr_step(...)`
  - `s21_lobpcg_solve(...)`
  - block `(n * bs)` bundles and dense projected-subproblem scratch

Interpretation:

- Sprint 46 is not one flat "reuse all eigensolver buffers" problem
- the shared buffer layer should be designed around these seam classes rather
  than by forcing all allocations into one identical view model

#### 2. The strongest shared extraction targets are real, but they are narrower than the full solver bodies

The most reusable shared allocation shapes are:

- graph-sized basis / vector bundles:
  - `n * m`
  - `n * k`
  - `n * block_size`
- tridiagonal / Ritz / restart scratch:
  - `alpha`
  - `beta`
  - `theta_*`
  - `subdiag`
  - `sel_idx`
- dense projected-subproblem intermediates:
  - `K * K`
  - `cap * cap`
- packed temporary bundles used to derive solver-local typed views

Interpretation:

- Day 3 should design one shared buffer owner around checked capacity and typed
  slicing
- the shared seam should own size/capacity and reset rules, not full solver
  control flow

#### 3. Solver-local state that should stay local is now explicit

The live code shows several solver-local state groups that should not be
collapsed into the shared buffer layer:

- grow-m Lanczos control:
  - `m`
  - `m_cap`
  - outer grow/retry policy
  - Wu/Simon convergence gating
- thick-restart control:
  - `k_locked`
  - `m_restart`
  - restart acceptance / lock selection
  - arrowhead phase orchestration
- LOBPCG control:
  - block Rayleigh-Ritz sequencing
  - soft-lock policy
  - preconditioner-composed residual/update flow

Interpretation:

- Sprint 46 should reuse buffer ownership, not erase algorithm boundaries
- Day 3 should keep solver-local math/control state outside the shared owner

#### 4. Optional and preconditioner-dependent buffers are concentrated in the LOBPCG family

The live asymmetry is now explicit:

- grow-m Lanczos and thick-restart Lanczos are primarily fixed-structure basis
  and spectral-scratch consumers
- LOBPCG carries the main optional / mode-dependent workspace behavior:
  - `P_new` present only when the conjugate-direction block is active
  - preconditioned residual flow via `W`
  - block-size-dependent dense `G`, `Y`, and `theta_full` scratch

Interpretation:

- the first shared workspace/state layer should be able to describe optional
  buffers cleanly
- LOBPCG should be a second major migration phase after the Lanczos family,
  not forced into the first shared landing batch

#### 5. The one-shot wrapper versus reusable-core split is now clear

The public entry surface is still intentionally one-shot:

- `sparse_eigs_sym(...)` remains the compatibility-facing entry point

The reusable-core candidates sit below that layer:

- grow-m Lanczos internals
- thick-restart outer loop and restart-state support
- LOBPCG outer-loop and RR-step work buffers

Interpretation:

- Sprint 46 should preserve the Sprint 45 pattern:
  - one-shot public entry point remains
  - reusable internal workspace/state paths sit underneath it
- no public workspace API is needed for the first Sprint 46 landing

#### 6. The first migration order is fixed by the live file

The correct adoption order is:

1. shared eigensolver buffer/state owner
2. grow-m Lanczos
3. thick-restart Lanczos
4. LOBPCG

The main later / lower-priority bucket is:

- broader helper cleanup that is support-only rather than repeated-run-critical

Interpretation:

- Day 5 should land only the shared owner and typed view seam
- Day 6 should target the main Lanczos families
- LOBPCG should follow once the shared owner has already proven itself on the
  simpler repeated-run eigensolver paths

## Day 3

**Objective:** Define the bounded internal reusable eigensolver
workspace/state object model for Sprint 46 so later code changes have explicit
ownership, reset, sizing, and wrapper-boundary rules before any implementation
lands.

### Commands Run

1. Re-read the Sprint 46 Day 3 plan section:
   - `sed -n '94,136p' docs/planning/EPIC_4/SPRINT_46/PLAN.md`
2. Re-read the Day 2 seam inventory:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_46/artifacts/day2-eigensolver-workspace-seam-inventory.md`
3. Re-read the Sprint 45 reusable-workspace precedent:
   - `sed -n '1,220p' src/sparse_iterative_workspace_internal.h`
   - `sed -n '1,260p' src/sparse_iterative_workspace_internal.c`
4. Re-read the narrower BiCGSTAB workspace precedent:
   - `sed -n '1,180p' src/sparse_bicgstab_internal.h`
5. Refresh the live eigensolver state/prepare seam markers:
   - `rg -n "typedef struct|prepare|ensure|reset|workspace|view" src/sparse_iterative_workspace_internal.h src/sparse_iterative_workspace_internal.c src/sparse_bicgstab_internal.h src/sparse_eigs.c`

### Day 3 Findings

#### 1. Sprint 46 should use one shared eigensolver buffer owner with typed solver-family views

The best first-phase internal shape is:

- one shared eigensolver storage owner for contiguous allocations and capacity
  tracking
- typed family views layered over that owner:
  - grow-m Lanczos view/state
  - thick-restart Lanczos view/state
  - LOBPCG view/state

Interpretation:

- Sprint 46 should follow the Sprint 45 pattern of one internal owner plus
  typed prepare helpers
- it should not create three unrelated allocation frameworks inside
  `src/sparse_eigs.c`

#### 2. The shared owner should be capacity-centric, while solver-family state should stay separate

The shared owner should track:

- `n` capacity
- Lanczos basis/restart capacities
- block-size capacity
- double scratch capacity
- `idx_t` scratch capacity
- optional int/flag capacity when needed

The family-specific state should track:

- grow-m:
  - active `m`
  - `m_cap`
  - current selected-take counts
- thick-restart:
  - `m_restart`
  - `k_locked`
  - restart-state ownership
- LOBPCG:
  - `block_size`
  - effective block/subspace dimensions
  - optional `P` ownership state

Interpretation:

- the owner should answer "is the memory large enough and sliced correctly?"
- the family state should answer "what is this algorithm doing with that
  memory right now?"

#### 3. Reset/reuse rules are now explicit

The first Sprint 46 internal contract should be:

- create:
  - zero-initialized owner/state is legal
  - first prepare call allocates as needed
- reset between repeated runs:
  - preserve capacity
  - zero or reinitialize only the slices whose next run requires clean state
  - never preserve old Krylov/Ritz/search-direction mathematical state as a
    semantic feature
- resize:
  - internal prepare helpers may grow capacity when a repeated-run workload
    exceeds current bounds
  - no public resize API is introduced
- destroy:
  - one free path resets the owner/state back to zero form

Interpretation:

- repeated stable-dimension workloads should amortize allocation cost
- Sprint 46 reuse means capacity reuse, not iterative continuation from prior
  eigensolver history

#### 4. Optional and mode-dependent buffers need explicit rules, especially for LOBPCG

The shared model should handle:

- fixed always-present Lanczos bundles
- thick-restart extras only when that family is prepared
- optional LOBPCG `P`/search-direction storage only when the active path
  requires it
- shift-invert / preconditioner composition without transferring ownership of:
  - LDLT factors
  - preconditioner callbacks
  - operator contexts

Interpretation:

- workspace/state owners should own buffers only
- operator/preconditioner/factor contexts remain caller-owned or solver-local
  external dependencies

#### 5. The internal-only versus wrapper-facing boundary is now fixed

Internal-only in Sprint 46:

- shared eigensolver workspace owner
- typed prepare helpers
- family-specific reusable state/view structs
- capacity and reset helpers

Wrapper-facing but still internal:

- one-shot `sparse_eigs_sym(...)` routing into reusable-core internals
- output/result initialization and compatibility reporting

Not part of Sprint 46:

- public workspace/state structs
- public init/free/reset APIs
- public repeated-run eigensolver handles

Interpretation:

- Sprint 46 should preserve the Sprint 42 / Sprint 45 compatibility pattern
- one-shot public APIs remain the public contract while internals become
  reusable

#### 6. The first code landing order is now concrete

The correct code-day order is:

1. shared owner + typed prepare helpers
2. grow-m Lanczos adoption
3. thick-restart adoption
4. LOBPCG adoption
5. wrapper/benchmark/documentation closeout

Interpretation:

- Day 5 should land a narrow owner/view seam first
- Day 6 should prove that seam on the simpler Lanczos families before LOBPCG
  joins it

## Day 4

**Objective:** Bound the shared eigensolver buffer-backed helper layer and the
implementation-day validation contract so Day 5 can land one narrow internal
workspace seam instead of reopening allocation policy, wrapper policy, or
validation scope mid-implementation.

### Commands Run

1. Re-read the Sprint 46 Day 4 plan section:
   - `sed -n '136,178p' docs/planning/EPIC_4/SPRINT_46/PLAN.md`
2. Re-read the Day 3 reusable-workspace/state design:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_46/artifacts/day3-reusable-eigensolver-workspace-state-design.md`
3. Re-read the Sprint 40 validation anchor:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
4. Refresh the maintained eigensolver test/example/benchmark/build surfaces:
   - `rg -n "test_eigs|test_eigs_thick_restart|test_eigs_lobpcg|bench_eigs|example_eigs|quality-review-full|deadcode-check" Makefile CMakeLists.txt .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml`
5. Refresh the live shared-vs-local allocation seam markers:
   - `rg -n "malloc|calloc|free|theta_|alpha|beta|sel_idx|workspace|restart_state|prepare" src/sparse_eigs.c`

### Day 4 Findings

#### 1. The shared eigensolver helper layer should be narrow and capacity-oriented

The shared layer should own:

- checked sizing for common eigensolver buffer shapes
- contiguous internal storage ownership
- reserve/grow helpers
- typed prepare helpers for:
  - grow-m Lanczos views
  - thick-restart Lanczos views
  - LOBPCG views
- narrow reset/zero helpers for slices that require clean-state reuse

Interpretation:

- Day 5 should land one bounded internal owner/view seam
- it should not migrate dense spectral math helpers or solver control flow into
  the shared layer just because they allocate memory nearby

#### 2. Several high-signal helpers should remain solver-local even after the shared owner lands

Keep local to their families:

- `lanczos_iterate_op(...)`
- `s20_ritz_pairs(...)`
- `s20_select_indices(...)`
- arrowhead assembly / dense Jacobi logic
- thick-restart lock-selection and restart-state choreography
- LOBPCG RR-step sequencing and soft-lock policy
- shift-invert factor and operator-composition handling

Interpretation:

- the shared layer is about ownership and slicing
- eigensolver math kernels and algorithm sequencing stay in their current
  solver-family codepaths

#### 3. The common owner must support two different kinds of reuse cleanly

The shared layer has to support:

- stable-dimension repeated runs:
  - same `n`
  - same or smaller `k`
  - same or smaller restart/block settings
- bounded internal growth when later runs exceed prior dimensions

It should not support:

- preserving old Krylov, Ritz, restart, or search-direction mathematical state
  as a cross-run semantic feature

Interpretation:

- reuse remains capacity reuse, not solver-history reuse
- Day 5 / Day 6 should make that distinction explicit in helper naming and
  reset behavior

#### 4. The implementation-day validation contract is now fixed

Mandatory for any `*.c` / `*.h` change:

- `make format`
- `make lint`
- `make test`

Strong default for substantial shared-layer or migration batches:

- `make quality-review-full`

Targeted eigensolver follow-ons when the touched surface justifies them:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`
- `./build/bench_eigs`

Interpretation:

- Day 5 and later code days should assume the full C gate always applies
- the reviewed wrapper baseline should be used whenever the batch touches the
  shared owner or multiple eigensolver families
- direct example/benchmark reruns stay targeted rather than universal

#### 5. The first code landing order is now locked in

The correct implementation order remains:

1. shared eigensolver owner + typed prepare helpers
2. grow-m Lanczos migration
3. thick-restart migration
4. LOBPCG migration
5. wrapper, benchmark, and memory-contract closeout

Interpretation:

- Day 5 should not broaden into algorithm migration yet
- Day 6 should prove the shared owner on the simpler Lanczos families before
  LOBPCG joins the reusable path

## Day 5

**Objective:** Land the first real Sprint 46 code batch by introducing the
private reusable eigensolver workspace/state owner and proving it in one
bounded live grow-m Lanczos path before widening the migration to
thick-restart, LOBPCG, examples, or benchmarks.

### Commands Run

1. Re-read the Sprint 46 Day 5 plan section:
   - `sed -n '136,178p' docs/planning/EPIC_4/SPRINT_46/PLAN.md`
2. Re-read the Day 4 shared-buffer design and validation contract:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_46/artifacts/day4-shared-buffer-layer-design-and-validation-plan.md`
3. Re-read the closest reusable-workspace precedents and current eigensolver
   allocation regions:
   - `sed -n '1,260p' src/sparse_iterative_workspace_internal.h`
   - `sed -n '1,320p' src/sparse_iterative_workspace_internal.c`
   - `sed -n '1,240p' src/sparse_bicgstab_internal.h`
   - `sed -n '240,1385p' src/sparse_eigs.c`
4. Refresh maintained build wiring before landing the new module:
   - `sed -n '1,220p' Makefile`
   - `sed -n '1,220p' CMakeLists.txt`
5. Implement the shared eigensolver workspace/state layer and the first live
   grow-m adoption:
   - `apply_patch` on:
     - `src/sparse_eigs_workspace_internal.h`
     - `src/sparse_eigs_workspace_internal.c`
     - `src/sparse_eigs.c`
     - `Makefile`
     - `CMakeLists.txt`
6. Run the mandatory full C/C-header gate:
   - `make format`
   - `make lint`
   - `make test`
7. Run the stronger reviewed baseline for the shared-layer landing:
   - `make quality-review-full`
8. Run the targeted eigensolver follow-ons justified by the touched seam:
   - `./build/test_eigs`
   - `./build/test_eigs_thick_restart`
   - `./build/test_eigs_lobpcg`

### Day 5 Findings

#### 1. Sprint 46 now has a real shared eigensolver workspace/state layer, not just a design

The first bounded shared seam now exists in:

- `src/sparse_eigs_workspace_internal.h`
- `src/sparse_eigs_workspace_internal.c`

That layer is intentionally narrow in this first batch:

- one reusable internal owner for:
  - double-backed work buffers
  - `idx_t` side buffers
  - `int` side buffers
  - cached eigensolver shape/capacity metadata
- typed prepare helpers for:
  - grow-m Lanczos
  - thick-restart Lanczos
  - LOBPCG

Interpretation:

- Sprint 46 now has the capacity-oriented internal owner/view seam that Day 3
  and Day 4 were aiming for
- this did not require public API changes or a second allocation framework

#### 2. The first live adoption is deliberately the simpler grow-m Lanczos path

`sparse_eigs_sym(...)` now proves the shared seam through the grow-m branch:

- it initializes a private `sparse_eigs_workspace_t`
- it prepares a typed `sparse_eigs_growm_workspace_view_t`
- it binds the former local allocation bundle through shared typed slices:
  - `V`
  - `alpha`
  - `beta`
  - `v0`
  - `theta_long`
  - `subdiag`
  - `Y_long`
  - `sel_idx`
- it frees the shared owner on all exits instead of manually freeing the
  former per-call grow-m heap bundle

Interpretation:

- the first proof point is a real live eigensolver path, not a dead internal
  scaffold
- the batch stayed bounded by proving the shared owner on the simpler Lanczos
  family before widening into thick-restart or LOBPCG control-heavy paths

#### 3. The shared layer stayed on the Day 4 ownership boundary

Day 5 moved only clearly shared sizing, allocation, and packed-slice logic into
the new seam.

It did **not** move:

- `lanczos_iterate_op(...)`
- Ritz extraction / selection helpers
- arrowhead or dense Jacobi helpers
- thick-restart lock/restart choreography
- LOBPCG RR-step sequencing
- result/reporting semantics

Interpretation:

- the shared layer owns memory, checked sizing, capacity growth, and typed view
  preparation
- eigensolver math kernels and algorithm control remain in `src/sparse_eigs.c`
  for later bounded migrations

#### 4. The batch remained intentionally narrow

Day 5 did **not** yet migrate:

- thick-restart Lanczos call sites
- LOBPCG call sites
- public wrappers or explicit public workspace APIs
- eigensolver benchmark or example code

Interpretation:

- this was the right Day 5 landing shape:
  - add the shared owner
  - prove it in one live high-value path
  - keep the rest of the solver-family migration queue for Day 6 and later

#### 5. The shared-layer landing validated cleanly

Because `*.c` and `*.h` files changed, the required gate was:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline for this shared-layer landing was also run:

- `make quality-review-full`

Targeted eigensolver follow-ons were also run:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`

One small implementation issue surfaced during the first lint pass:

- the initial LOBPCG prepare helper formed `view->X` through a null
  `view->P_new` branch in the no-`P` case, which `cppcheck` flagged as a null
  pointer arithmetic path

That was fixed immediately by making the `with_p` and no-`P` slice derivation
branches explicit, and the authoritative rerun from the top passed fully.

Interpretation:

- the shared owner is already clean under the maintained static-analysis gate
- the first shared-buffer batch closes from a measured reviewed baseline rather
  than from partial local testing

## Day 6

**Objective:** Convert the main Lanczos migration target that still owned a
per-call heap bundle — the thick-restart outer loop — to the shared reusable
internal workspace/state seam, while preserving the already-migrated grow-m
path, keeping `lanczos_restart_state_t` as the family-specific control/state
owner, and explicitly not widening into LOBPCG yet.

### Commands Run

1. Re-read the Sprint 46 Day 6 plan section:
   - `sed -n '179,224p' docs/planning/EPIC_4/SPRINT_46/PLAN.md`
2. Re-read the Day 5 shared-buffer closeout:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_46/artifacts/day5-shared-eigensolver-buffer-layer-batch1.md`
3. Re-read the landed shared workspace surface and the live thick-restart
   allocation region:
   - `sed -n '1,260p' src/sparse_eigs_workspace_internal.h`
   - `sed -n '1,360p' src/sparse_eigs_workspace_internal.c`
   - `sed -n '1600,1865p' src/sparse_eigs.c`
   - `sed -n '2110,2305p' src/sparse_eigs.c`
4. Refresh the live eigensolver seam markers around thick-restart:
   - `rg -n "lanczos_thick_restart_iterate|s21_thick_restart_outer_loop|lanczos_restart_state_t|malloc|calloc|workspace_prepare|growm_view|thick_restart" src/sparse_eigs.c`
5. Implement the bounded thick-restart migration:
   - `apply_patch` on:
     - `src/sparse_eigs.c`
6. Run the mandatory full C/C-header gate:
   - `make format`
   - `make lint`
   - `make test`
7. Run the stronger reviewed baseline for the multi-family Lanczos migration:
   - `make quality-review-full`
8. Run the targeted eigensolver follow-ons justified by the touched solver
   paths:
   - `./build/test_eigs`
   - `./build/test_eigs_thick_restart`
   - `./build/test_eigs_lobpcg`
   - `./build/example_eigs`

### Day 6 Findings

#### 1. The main remaining thick-restart heap bundle is now on the shared reusable seam

The live Day 5 gap was the thick-restart outer loop in
`s21_thick_restart_outer_loop(...)`, which still built a per-call bundle for:

- `V`
- `alpha`
- `beta`
- `v0`
- `residual_vec`
- `T_arrow`
- `theta_arrow`
- `Y_arrow`
- `sel_idx`
- temporary locked-state buffers

Day 6 now routes that bundle through the shared owner via:

- `sparse_eigs_workspace_t`
- `sparse_eigs_thick_restart_workspace_view_t`
- `sparse_eigs_workspace_prepare_thick_restart(...)`

Interpretation:

- the main repeated-run thick-restart basis/scratch path no longer depends only
  on ad hoc outer-loop heap allocation
- Sprint 46 now has both primary Lanczos families on the shared reusable seam

#### 2. The family-specific thick-restart control/state stayed separate, as designed

Day 6 did **not** collapse the algorithm-specific thick-restart control into the
shared owner.

It intentionally preserved:

- `lanczos_restart_state_t`
- `lanczos_restart_state_assemble(...)`
- `lanczos_restart_state_free(...)`
- lock-selection and restart choreography
- residual recomputation / restart assembly flow

Interpretation:

- the shared owner remains about capacity, ownership, and typed slicing
- family-specific restart semantics remain in the thick-restart code path, which
  matches the Day 3 design boundary

#### 3. The grow-m proof from Day 5 remains intact and unchanged in shape

Grow-m Lanczos was already migrated in Day 5 through:

- `sparse_eigs_workspace_prepare_growm(...)`
- `sparse_eigs_growm_workspace_view_t`

Day 6 did not reopen that path.  Instead it completed the first Lanczos-family
pairing by bringing thick-restart onto the same owner/view model.

Interpretation:

- the migration order is holding:
  1. shared owner
  2. grow-m proof
  3. thick-restart adoption
- this keeps Sprint 46's front half disciplined and bounded

#### 4. The batch stayed within the Day 6 boundary

Day 6 did **not** yet migrate:

- LOBPCG call sites
- public wrappers or explicit public workspace/state APIs
- eigensolver benchmark work
- maintainer memory-contract closeout

Interpretation:

- this was the correct Day 6 landing shape:
  - finish the primary Lanczos migration
  - keep LOBPCG and repeated-run evidence for later Sprint 46 batches

#### 5. The multi-family Lanczos migration validated from the full reviewed baseline

Because `*.c` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline for this shared-layer/multi-family Lanczos
migration was also run:

- `make quality-review-full`

Targeted eigensolver follow-ons were also run:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`

Interpretation:

- Day 6 closes from the maintained full reviewed baseline, not from partial
  smoke coverage
- the direct eigensolver binaries remain the right touched-surface proof for
  this batch

## Day 7

**Objective:** Audit the post-Day-6 eigensolver state so the remaining Sprint
46 queue is driven by the live code after the shared Lanczos workspace
landings, with an explicit Day 8 LOBPCG target set and a clear separation
between real remaining repeated-allocation work, wrapper/composition surfaces,
and helper/state paths that should stay local.

### Commands Run

1. Re-read the Sprint 46 Day 7 plan section:
   - `sed -n '225,275p' docs/planning/EPIC_4/SPRINT_46/PLAN.md`
2. Re-read the Day 6 Lanczos migration closeout:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_46/artifacts/day6-lanczos-migration-batch1.md`
3. Refresh the live shared-owner surface and the remaining eigensolver seam
   markers:
   - `sed -n '1,260p' src/sparse_eigs_workspace_internal.h`
   - `sed -n '1,360p' src/sparse_eigs_workspace_internal.c`
   - `rg -n "malloc|calloc|workspace_prepare|sparse_eigs_workspace|lanczos_restart_state_t|lobpcg|block_size|bench|example|with_workspace|prepare_" src/sparse_eigs.c src/sparse_eigs_workspace_internal.h src/sparse_eigs_workspace_internal.c include/sparse_eigs.h src/sparse_eigs_internal.h`
4. Re-read the main residual LOBPCG allocation regions directly:
   - `sed -n '2580,3075p' src/sparse_eigs.c`
5. Re-read the closest prior audit shape for comparison:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_45/artifacts/day7-primary-workspace-landing-audit.md`

### Day 7 Findings

#### 1. Sprint 46 no longer has a generic “eigensolver workspace migration” queue

After Day 6, the live state now breaks into three concrete buckets:

- primary families already on the shared workspace seam:
  - grow-m Lanczos
  - thick-restart Lanczos
- one real remaining main workspace migration target:
  - LOBPCG
- later wrapper/benchmark/documentation follow-ons:
  - one-shot public dispatch/wrapper edges
  - repeated-run benchmark evidence
  - maintainer memory-behavior closeout

Interpretation:

- the front half of Sprint 46 has already completed the main Lanczos-family
  shared-workspace landing
- the remaining implementation queue is now concrete rather than generic

#### 2. The strongest remaining direct repeated-allocation target is the LOBPCG family

The live LOBPCG path still owns the clearest repeated-allocation bundles in:

- `s21_lobpcg_rr_step(...)`
- `s21_lobpcg_solve(...)`

The main remaining buffer groups are:

- RR-step subspace/intermediate bundle:
  - `Q`
  - `AQ`
  - `G`
  - `Y`
  - `theta_full`
  - `sel_idx`
  - `X_new`
  - optional `P_new`
- outer-loop block bundle:
  - `X`
  - `R`
  - `W`
  - `AX`
  - `theta`
  - `converged`
  - lazily allocated `P`

Interpretation:

- the shared owner/view model already has an explicit LOBPCG prepare seam for
  these shapes
- Day 8 should target these two LOBPCG allocation regions directly rather than
  reopening Lanczos work or jumping ahead to wrappers/benchmarks

#### 3. The remaining non-LOBPCG helper allocations are not the next migration target

The live code still contains some local helper allocations outside the shared
owner:

- Lanczos/refinement helper scratch:
  - `lanczos_iterate_op(...)` local `w`
  - `lanczos_thick_restart_iterate(...)` local `w`
  - `s29_refine_eigenpairs(...)` local `Av` / `y`
- arrowhead/tridiagonal helper scratch:
  - `s21_arrowhead_to_tridiag(...)`
  - dense Householder helper buffers
- restart-state owned buffers:
  - `lanczos_restart_state_t`

Interpretation:

- these are real allocations, but they are not the main repeated-run target for
  Sprint 46 Day 8
- they either remain family-local by design or are small helper scratch rather
  than the main repeated bundle the shared seam was created to absorb

#### 4. Wrapper and public-surface work remain later follow-ons, not Day 8 work

The live public path is still intentionally simple:

- `sparse_eigs_sym(...)` remains the compatibility-facing one-shot entry point
- backend AUTO/explicit dispatch still routes internally to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - LOBPCG

The benchmark/example surfaces also remain later buckets:

- `benchmarks/bench_eigs.c`
- `examples/example_eigs.c`

Interpretation:

- Day 8 should keep its focus on the remaining internal LOBPCG bundle
- wrapper cleanup, repeated-run benchmark evidence, and memory-behavior notes
  should remain sequenced after the last major family migration lands

#### 5. No new internal-header redesign is needed before Day 8

The current private helper surface already contains the LOBPCG owner/view seam:

- `sparse_eigs_workspace_prepare_lobpcg(...)`
- `sparse_eigs_lobpcg_workspace_view_t`

Interpretation:

- Day 8 does not need another helper-layer redesign first
- the right next move is live adoption of the already-landed LOBPCG prepare seam
  in the main LOBPCG path

#### 6. The Day 8 target set is now explicit and bounded

Day 8 should be bounded to:

1. `s21_lobpcg_rr_step(...)`
2. `s21_lobpcg_solve(...)`
3. adoption of `sparse_eigs_workspace_prepare_lobpcg(...)`
4. preservation of current one-shot/public behavior
5. no benchmark or wrapper churn unless a follow-on is obviously trivial

Interpretation:

- the remaining eigensolver migration order is now stable:
  1. shared owner
  2. grow-m
  3. thick-restart
  4. LOBPCG
  5. wrapper/benchmark/memory-contract closeout

## Day 8

**Objective:** Migrate the remaining primary eigensolver workspace target,
LOBPCG, onto the shared reusable internal owner/view seam by converting both
the RR-step bundle and the outer-loop block bundle to
`sparse_eigs_workspace_prepare_lobpcg(...)`, while preserving current
one-shot/public behavior and keeping the batch bounded away from wrappers,
benchmarks, and broader closeout work.

### Commands Run

1. Re-read the Sprint 46 Day 8 plan section and the Day 7 narrowed target set:
   - `sed -n '275,325p' docs/planning/EPIC_4/SPRINT_46/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_46/artifacts/day7-primary-workspace-landing-audit.md`
2. Re-read the shared LOBPCG workspace prepare surface:
   - `sed -n '1,260p' src/sparse_eigs_workspace_internal.h`
   - `sed -n '1,320p' src/sparse_eigs_workspace_internal.c`
3. Re-read the live LOBPCG allocation regions directly:
   - `sed -n '2580,3075p' src/sparse_eigs.c`
4. Re-read the internal eigensolver helper declarations:
   - `sed -n '1,260p' src/sparse_eigs_internal.h`
5. Implement the LOBPCG workspace migration batch:
   - `src/sparse_eigs_workspace_internal.h`
   - `src/sparse_eigs_workspace_internal.c`
   - `src/sparse_eigs_internal.h`
   - `src/sparse_eigs.c`
6. Run the required code-quality gate:
   - `make format`
   - `make lint`
   - `make test`
7. Run the stronger reviewed baseline for this shared-layer/family-migration
   batch:
   - `make quality-review-full`
8. Run the targeted touched-surface follow-ons:
   - `./build/test_eigs`
   - `./build/test_eigs_thick_restart`
   - `./build/test_eigs_lobpcg`
   - `./build/example_eigs`

### Day 8 Findings

#### 1. LOBPCG is now on the shared reusable workspace seam

The Day 8 batch moved both remaining primary LOBPCG allocation regions onto the
already-landed shared owner/view model:

- `s21_lobpcg_rr_step(...)`
- `s21_lobpcg_solve(...)`

The migrated path now prepares a reusable:

- `sparse_eigs_workspace_t`
- `sparse_eigs_lobpcg_workspace_view_t`

Interpretation:

- Sprint 46 no longer has a remaining primary eigensolver-family workspace
  migration target
- the shared eigensolver owner now covers:
  - grow-m Lanczos
  - thick-restart Lanczos
  - LOBPCG

#### 2. The RR-step bundle now consumes a typed shared view instead of owning a per-call heap bundle

`s21_lobpcg_rr_step(...)` no longer allocates and frees its own:

- `Q`
- `AQ`
- `G`
- `Y`
- `theta_full`
- `sel_idx`
- `X_new`
- optional `P_new`

Instead, it now receives those through
`sparse_eigs_lobpcg_workspace_view_t`.

Interpretation:

- the RR-step path is now aligned with the same owner/view pattern already used
  by the migrated Lanczos families
- the function remains algorithm-local in control flow, but no longer owns the
  repeated heap churn the shared seam was created to absorb

#### 3. The outer-loop block bundle now binds through the same shared view

`s21_lobpcg_solve(...)` now prepares the shared owner and binds the former
outer-loop block bundle through typed slices:

- `X`
- `R`
- `W`
- `P`
- `AX`
- `theta`
- `converged`

The prior lazy `P` allocation is gone from the outer loop; first-iteration
behavior is now preserved by an explicit `have_p` / `use_p` contract instead of
by “allocate later” semantics.

Interpretation:

- Day 8 reduced the last main repeated LOBPCG allocation churn without changing
  public behavior
- first-iteration LOBPCG semantics stayed explicit and readable even after the
  owner/view conversion

#### 4. The helper-layer extension stayed minimal and justified

The only meaningful helper-layer widening for Day 8 was to make the existing
LOBPCG view model carry the persistent `P` slice in addition to the already
planned view-owned temporary bundles.

Interpretation:

- Day 8 did not reopen helper-layer design
- it only widened the existing LOBPCG typed view enough to support the live
  outer-loop migration cleanly

#### 5. The batch stayed inside the Day 7 boundary

Day 8 completed:

- LOBPCG RR-step migration
- LOBPCG outer-loop migration
- shared-owner adoption through
  `sparse_eigs_workspace_prepare_lobpcg(...)`

Day 8 intentionally did **not** widen into:

- public wrapper/API changes
- repeated-run benchmark work
- example/tutorial refresh
- maintainer memory-behavior closeout

Interpretation:

- the batch remained the right bounded Sprint 46 migration step
- wrapper/benchmark/closeout work can now proceed from a fully migrated
  eigensolver-family workspace baseline instead of from a mixed state

#### 6. The reviewed validation baseline stayed green after the final family migration

Because `*.c` and `*.h` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

All passed.

The stronger reviewed baseline for this shared-layer/final-family migration
batch also passed:

- `make quality-review-full`

The targeted eigensolver follow-ons also passed:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`

Interpretation:

- the full eigensolver workspace migration now closes from a reviewed green
  baseline rather than only from a narrow local proof

## Day 9

**Objective:** Normalize `sparse_eigs_sym(...)` so it reads explicitly as a
compatibility-preserving one-shot wrapper over the already-migrated reusable
internal backend paths, without widening Sprint 46 into another solver,
workspace, benchmark, or public-API redesign batch.

### Commands Run

1. Re-read the Sprint 46 Day 9 plan section:
   - `sed -n '330,375p' docs/planning/EPIC_4/SPRINT_46/PLAN.md`
2. Re-read the Day 8 closeout and the current wrapper/public surface:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_46/artifacts/day8-lobpcg-workspace-migration-batch.md`
   - `sed -n '1,260p' include/sparse_eigs.h`
3. Re-read the live public entry and backend dispatch path:
   - `sed -n '780,1325p' src/sparse_eigs.c`
   - `rg -n "backend_used|AUTO|opts == NULL|result->|sparse_eigs_sym|dispatch" src/sparse_eigs.c`
4. Re-read the closest prior compatibility-batch pattern:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_45/artifacts/day9-wrapper-compatibility-batch.md`
5. Implement the bounded wrapper/compatibility cleanup:
   - `src/sparse_eigs.c`
6. Run the required code-quality gate:
   - `make format`
   - `make lint`
   - `make test`
7. Run the targeted wrapper-focused eigensolver follow-ons:
   - `./build/test_eigs`
   - `./build/test_eigs_thick_restart`
   - `./build/test_eigs_lobpcg`
   - `./build/example_eigs`

### Day 9 Findings

#### 1. `sparse_eigs_sym(...)` now reads more clearly as a public compatibility wrapper

Day 9 moved the public-entry scaffolding in `sparse_eigs_sym(...)` behind small
internal helper functions for:

- public default option construction
- entry validation
- result-field initialization
- AUTO/explicit backend selection
- backend delegation

Interpretation:

- the public one-shot entry now reads more directly as a wrapper/composition
  surface over reusable internal backends
- the algorithm-specific backends remain the behavioral truth

#### 2. Backend math ownership stayed with the existing internal solver paths

Day 9 did **not** move or redesign the actual backend implementations:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG

The new wrapper helper layer only routes to the already-existing internal
implementations and preserves the existing refinement handoff.

Interpretation:

- Day 9 normalized composition structure, not eigensolver math
- the reusable-workspace migration from Days 5-8 remains the implementation
  owner for the real backend behavior

#### 3. Public defaults, validation, and result-telemetry setup are now explicit wrapper responsibilities

The following wrapper-facing responsibilities are now isolated more clearly from
the backend bodies:

- `opts == NULL` library-default mapping
- null/shape/option validation
- result telemetry reset:
  - `n_requested`
  - `n_converged`
  - `iterations`
  - `residual_norm`
  - `used_csc_path_ldlt`
  - `peak_basis_size`
  - `backend_used`

Interpretation:

- the public wrapper contract is now easier to audit
- later benchmark and maintainer-closeout work can reference one explicit
  wrapper boundary instead of a monolithic public entry

#### 4. AUTO/explicit backend selection is now one explicit compatibility seam

Day 9 isolated the current backend choice logic into one explicit helper that
preserves the existing behavior for:

- explicit LOBPCG
- explicit thick-restart
- AUTO LOBPCG routing
- AUTO thick-restart routing
- grow-m fallback

Interpretation:

- the user-facing one-shot entry now has one clearly named dispatch decision
  point
- no public API redesign or dispatch-policy change was introduced

#### 5. The batch stayed bounded to wrapper/composition cleanup

Day 9 completed:

- public wrapper structure cleanup
- explicit wrapper-vs-backend ownership separation
- compatibility-preserving dispatch normalization

Day 9 intentionally did **not** widen into:

- new public explicit workspace APIs
- repeated-run benchmark work
- eigensolver algorithm changes
- helper-layer/workspace redesign
- public docs/tutorial refresh

Interpretation:

- the batch stayed inside the Sprint 46 Day 9 boundary
- Sprint 46 is now ready to pivot into repeated-run benchmark evidence from a
  cleaner public-wrapper baseline

#### 6. The validation baseline stayed green after the wrapper cleanup

Because `*.c` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

All passed.

The targeted wrapper-focused eigensolver follow-ons also passed:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`

Interpretation:

- the wrapper/composition cleanup did not disturb public defaults, AUTO/explicit
  routing, or the example-facing one-shot flow

## Day 10

**Objective:** Define the smallest honest repeated-run benchmark slice for the
new reusable eigensolver workspace/state model so Day 11 can add measured
evidence without reopening broad `bench_eigs` CLI churn, backend-sweep sprawl,
or public-facing documentation work.

### Commands Run

1. Re-read the Sprint 46 Day 10 plan section:
   - `sed -n '375,430p' docs/planning/EPIC_4/SPRINT_46/PLAN.md`
2. Re-read the Day 9 wrapper closeout:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_46/artifacts/day9-compatibility-wrapper-batch.md`
3. Re-read the current permanent eigensolver benchmark driver:
   - `sed -n '1,260p' benchmarks/bench_eigs.c`
4. Re-read the closest repeated-run benchmark shape from Sprint 45:
   - `sed -n '1,260p' benchmarks/bench_iterative_reuse.c`
5. Sweep the current eigensolver benchmark/example/test references for repeated-run
   relevance and surface coupling:
   - `rg -n "backend|repeat|repeats|wall|median|csv|compare|LOBPCG|THICK_RESTART|AUTO|bench" benchmarks/bench_eigs.c examples/example_eigs.c tests/test_eigs.c tests/test_eigs_thick_restart.c tests/test_eigs_lobpcg.c include/sparse_eigs.h src/sparse_eigs.c`

### Day 10 Findings

#### 1. The current benchmark surface is broader than the Sprint 46 Day 11 target

`benchmarks/bench_eigs.c` is a broad backend/corpus sweep driver. It already
covers:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG
- multiple `which` modes
- multiple SuiteSparse/KKT cases
- CSV and compare modes
- preconditioner sweeps

Interpretation:

- it is a good existing benchmark surface
- it is **not** the right place for a first Sprint 46 repeated-run proof if the
  goal is to keep Day 11 narrow and honest

#### 2. The right Day 11 comparison shape is the Sprint 45-style A/B repeated-run driver

`benchmarks/bench_iterative_reuse.c` provides the cleaner model for the Day 11
shape:

- one-shot path
- reusable-workspace path
- repeated stable-dimension calls
- median wall-time comparison
- behavior-level parity checks
- no universal speedup claims

Interpretation:

- Day 11 should follow that style for eigensolvers rather than broadening
  `bench_eigs.c` into a repeated-run benchmarking framework
- the benchmark evidence should stay focused on allocator-churn reduction, not
  on replacing the permanent backend-sweep driver

#### 3. The narrow repeated-run target set should center on the three migrated primary families

After Days 5-9, the clean repeated-run targets are now:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG

But the design should still be staged:

- required Day 11 cases:
  - grow-m Lanczos
  - thick-restart Lanczos
- bounded add-on only if it stays obviously small:
  - LOBPCG

Interpretation:

- grow-m and thick-restart are the most direct stable repeated-run comparison
  cases
- LOBPCG should be included only if the Day 11 batch stays narrow and does not
  drag in preconditioner-selection or block-size experiment churn

#### 4. The benchmark cases should use stable dimensions and fixed option shapes

The repeated-run benchmark cases should be selected around fixed and repeatable:

- matrix
- `k`
- backend
- `which`
- restart/block-size settings

The cleanest first cases are:

- `nos4`, `k = 5`, `which = LARGEST`, explicit grow-m Lanczos
- `bcsstk14`, `k = 5`, `which = LARGEST`, explicit thick-restart Lanczos
- optional bounded LOBPCG add-on:
  - `bcsstk04`, `k = 3`, `which = SMALLEST`, explicit LOBPCG with IC(0)

Interpretation:

- these cases align with the existing benchmark/example/test corpus
- they avoid introducing new fixture-selection churn just to prove repeated-run
  reuse

#### 5. The claim scope must stay behavior-first and modest

The Day 11 repeated-run benchmark should measure:

- one-shot wall time
- reusable internal path wall time
- median comparison over repeated stable-dimension runs
- iteration-count parity
- convergence parity
- residual/output parity

It should explicitly avoid:

- universal speedup claims
- corpus-wide backend ranking claims
- benchmark CLI redesign
- claims about asymptotic improvements beyond reduced repeated allocation churn

Interpretation:

- the right Sprint 46 proof is “the migrated reusable internal paths support a
  narrow repeated-run A/B comparison without changing behavior”
- anything stronger belongs in later benchmark work, not this sprint

#### 6. Day 11 should add a new narrow repeated-run driver rather than mutate `bench_eigs` heavily

The best implementation shape for Day 11 is:

- a new bounded repeated-run benchmark driver, parallel to Sprint 45’s approach
- reuse existing public one-shot entry points plus the internal reusable backend
  seams as needed
- keep `bench_eigs.c` intact as the broader corpus/backend sweep driver

Interpretation:

- Sprint 46 can add repeated-run evidence without destabilizing or overloading
  the permanent multi-mode benchmark driver
- benchmark layering remains cleaner:
  - `bench_eigs.c` = broad backend/corpus sweep
  - new Sprint 46 driver = repeated-run reuse comparison evidence

#### 7. The Day 11 target set is now explicit and bounded

Day 11 should be bounded to:

1. one new repeated-run eigensolver benchmark driver
2. required A/B cases:
   - grow-m Lanczos
   - thick-restart Lanczos
3. optional LOBPCG add-on only if it stays obviously small
4. measured output notes in the Day 11 artifact
5. no broad `bench_eigs` CLI churn

Interpretation:

- Sprint 46 now has a concrete, honest repeated-run benchmark target
- the benchmark slice stays tied directly to the migrated eigensolver paths

## Day 11

**Objective:** Land the narrow repeated-run eigensolver benchmark slice defined
on Day 10 so Sprint 46 has direct measured evidence for the migrated reusable
workspace seam, without widening into `bench_eigs.c` CLI churn, public API
changes, or broader benchmark-framework work.

### Commands Run

1. Re-read the Day 10 benchmark design and the Sprint 45 repeated-run benchmark
   precedent:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_46/artifacts/day10-repeated-run-benchmark-design.md`
   - `sed -n '1,260p' benchmarks/bench_iterative_reuse.c`
2. Re-read the live eigensolver wrapper/backend seams and workspace-owner
   surface:
   - `sed -n '760,1520p' src/sparse_eigs.c`
   - `sed -n '2060,2420p' src/sparse_eigs.c`
   - `sed -n '1,260p' src/sparse_eigs_internal.h`
   - `sed -n '1,260p' src/sparse_eigs_workspace_internal.h`
   - `sed -n '1,260p' src/sparse_eigs_workspace_internal.c`
3. Re-read the benchmark/build wiring surfaces:
   - `sed -n '1,240p' benchmarks/bench_eigs.c`
   - `sed -n '130,180p' Makefile`
   - `sed -n '250,285p' CMakeLists.txt`
4. Implement the Day 11 benchmark and bounded private reuse seam:
   - `src/sparse_eigs.c`
   - `src/sparse_eigs_internal.h`
   - `benchmarks/bench_eigs_reuse.c`
   - `Makefile`
   - `CMakeLists.txt`
5. Run the required code-quality gate:
   - `make format`
   - `make lint`
   - `make test`
6. Run the targeted Day 11 eigensolver follow-ons:
   - `./build/test_eigs`
   - `./build/test_eigs_thick_restart`
   - `./build/test_eigs_lobpcg`
   - `./build/example_eigs`
   - `./build/bench_eigs_reuse`

### Day 11 Findings

#### 1. Sprint 46 now has a dedicated repeated-run eigensolver benchmark driver

Day 11 added:

- `benchmarks/bench_eigs_reuse.c`

The benchmark follows the Sprint 45 A/B repeated-run shape:

- one-shot public eigensolver path
- reusable internal workspace-backed path
- repeated stable-dimension calls
- median wall-time comparison
- behavior-level parity reporting

The benchmark stayed intentionally narrow:

- grow-m Lanczos on `nos4`
- thick-restart Lanczos on `bcsstk14`

Interpretation:

- Sprint 46 now has direct repeated-run evidence for the migrated Lanczos-family
  reuse seam
- the broad permanent sweep driver `benchmarks/bench_eigs.c` remained untouched

#### 2. A bounded internal reuse entry now exists without changing the public API

Day 11 added:

- `sparse_eigs_sym_with_workspace_internal(...)`

This internal helper mirrors the existing public entry’s:

- validation
- shift-invert setup
- AUTO/explicit backend selection
- result-field contract

while accepting a caller-owned `sparse_eigs_workspace_t` for the migrated
Lanczos-family backends.

Interpretation:

- the benchmark uses a real internal seam rather than ad hoc implementation
  reach-through
- `sparse_eigs_sym(...)` remains the compatibility-facing one-shot public API

#### 3. The public one-shot eigensolver path now composes around the shared implementation

Day 11 refactored the public one-shot path and the new internal reusable path
to share one implementation layer.

The shared implementation now owns:

- public defaults/validation
- shift-invert preprocessing
- backend dispatch
- refinement post-pass handoff

The reusable internal path is currently active for:

- grow-m Lanczos
- thick-restart Lanczos

LOBPCG intentionally kept its existing local allocation model for Day 11.

Interpretation:

- the repeated-run benchmark compares one-shot vs reuse across the same
  behavioral implementation path
- Day 11 avoided reopening the optional LOBPCG scope extension

#### 4. The measured repeated-run result is modest but clean

Direct benchmark output on this local run was:

- grow-m Lanczos case:
  - fixture: `nos4`
  - backend: explicit `SPARSE_EIGS_BACKEND_LANCZOS`
  - `k = 5`
  - repeats: `40`
  - one-shot median: `1.3680 ms`
  - reuse median: `1.3610 ms`
  - speedup: `1.01x`
  - last-run parity:
    - `115` iterations in both paths
    - converged in both paths
    - residual `4.326e-14` in both paths
    - `|lambda|max diff = 0.000e+00`
- thick-restart Lanczos case:
  - fixture: `bcsstk14`
  - backend: explicit `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`
  - `k = 5`
  - repeats: `8`
  - one-shot median: `49.7370 ms`
  - reuse median: `47.7710 ms`
  - speedup: `1.04x`
  - last-run parity:
    - `105` iterations in both paths
    - converged in both paths
    - residual `7.864e-14` in both paths
    - `|lambda|max diff = 0.000e+00`

Interpretation:

- the repeated-run reusable path preserved solver behavior exactly on the
  benchmarked cases
- the local timing gain is modest rather than dramatic
- that is still a valid Sprint 46 Day 11 result because the batch was about
  direct repeated-run measurement, not forced speedup claims

#### 5. The batch stayed inside the Day 10 boundary

Day 11 completed:

- one new repeated-run eigensolver benchmark driver
- one bounded internal reusable-workspace benchmark seam
- required grow-m and thick-restart repeated-run cases
- measured output capture for the Day 11 artifact

Day 11 intentionally did **not** widen into:

- broad `bench_eigs.c` CLI redesign
- public explicit eigensolver workspace APIs
- broad example/tutorial refresh
- mandatory LOBPCG repeated-run benchmarking

Interpretation:

- the benchmark slice stayed cleanly tied to the migrated primary reuse paths
- Sprint 46 remains on track for closeout without another broad eigensolver
  redesign batch

#### 6. The validation baseline stayed fully green after the benchmark landing

Because `*.c` and `*.h` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

All passed.

The targeted Day 11 eigensolver follow-ons also passed:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`

Interpretation:

- the benchmark landing preserved the existing eigensolver behavior and example
  surface
- Sprint 46 can move to documentation/residual audit work from a reviewed green
  baseline
