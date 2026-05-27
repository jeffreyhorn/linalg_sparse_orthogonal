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
