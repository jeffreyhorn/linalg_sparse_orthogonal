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
