# Sprint 49 Working Notes

## Day 1

**Objective:** Turn the Sprint 49 project-plan scope plus the Sprint
40/42/45/46/48 execution rules into a concrete public-lifecycle and final
Epic 4 starting point by confirming the preserved reviewed contracts, naming
the Sprint 49 workstreams explicitly, and defining the authoritative public
lifecycle/workspace, compatibility-wrapper, example/benchmark, and validation
inputs before final API exposure begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 49 project-plan source and the new sprint plan:
   - `sed -n '327,360p' docs/planning/EPIC_4/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_49/PLAN.md`
3. Re-read the immediate prerequisite closeouts:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_48/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_46/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_45/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Measure the live public-lifecycle, internal-workspace, example, benchmark,
   and regression hotspot sizes:
   - `wc -l include/*.h src/sparse_iterative*.c src/sparse_iterative*.h src/sparse_eigs*.c src/sparse_eigs*.h examples/example_iterative.c examples/example_matrix_free.c examples/example_eigs.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c tests/test_iterative.c tests/test_block_solvers.c tests/test_minres.c tests/test_bicgstab.c tests/test_stagnation.c tests/test_eigs.c tests/test_eigs_thick_restart.c tests/test_eigs_lobpcg.c`
7. Refresh the live lifecycle/workspace seam markers:
   - `rg -n "workspace|reuse|lifecycle|callback|cancel|one-shot|wrapper|compatibility" include src examples benchmarks tests -g '!build'`
8. Re-read the main public and internal lifecycle/workspace surfaces:
   - `sed -n '1,360p' include/sparse_analysis.h`
   - `sed -n '1,260p' include/sparse_iterative.h`
   - `sed -n '1,260p' include/sparse_eigs.h`
   - `sed -n '1,260p' src/sparse_matrix_internal.h`
   - `sed -n '1,240p' src/sparse_iterative_internal.h`
   - `sed -n '1,240p' src/sparse_eigs_internal.h`
9. Re-read one recent Day 1 artifact pattern for format calibration:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_46/artifacts/day1-scope-and-eigensolver-baseline.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_48/artifacts/day1-scope-and-quality-contract-baseline.md`

### Day 1 Findings

#### 1. Sprint 49 starts from a preserved Sprint 40/42/45/46/48 baseline, not from baseline repair work

The inherited starting contract remains explicit and stable:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Sprint 42 already left the lifecycle/cancellation groundwork:
  - factor-state scaffolding
  - compatibility-preserving internal-first rules
- Sprint 45 already left an internal iterative reusable-workspace seam
- Sprint 46 already left an internal eigensolver reusable-workspace seam
- Sprint 48 already clarified the maintainer-policy / migration-doc home

Interpretation:

- Sprint 49 is not a reviewed-baseline recovery sprint
- Sprint 49 is the bounded final public-lifecycle exposure and Epic 4
  integration sprint on top of an already-validated structural baseline

#### 2. The core Sprint 49 gap is now precise: public reusable-handle precedent exists, but iterative/eigensolver reuse is still internal-only

The live public surface already contains one explicit reusable-lifecycle model:

- `include/sparse_analysis.h`
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_refactor_numeric(...)`
  - `sparse_factor_free(...)`

But the newer repeated-run improvements remain internal-facing:

- iterative internal reusable-workspace seam:
  - `src/sparse_iterative_workspace_internal.h`
  - `src/sparse_iterative_workspace_internal.c`
  - `src/sparse_iterative_internal.h`
- eigensolver internal reusable-workspace seam:
  - `src/sparse_eigs_workspace_internal.h`
  - `src/sparse_eigs_workspace_internal.c`
  - `src/sparse_eigs_internal.h`

Interpretation:

- Sprint 49 does not need to invent lifecycle language from scratch
- it needs to reconcile the older public analysis/factor lifecycle model with
  the newer internal iterative/eigensolver reuse work and expose only the
  bounded public refinements that are now safe

#### 3. The direct Sprint 49 hotspots are still concentrated in the iterative and eigensolver public surfaces

The live implementation and public-header sizes make the main Day 1 API
hotspots explicit:

- `include/sparse_iterative.h` = `585`
- `include/sparse_eigs.h` = `592`
- `include/sparse_analysis.h` = `334`
- `src/sparse_iterative.c` = `2276`
- `src/sparse_eigs.c` = `3060`
- `src/sparse_iterative_workspace_internal.c` = `215`
- `src/sparse_eigs_workspace_internal.c` = `267`

The main caller-facing support surfaces are also explicit:

- `examples/example_iterative.c` = `144`
- `examples/example_matrix_free.c` = `122`
- `examples/example_eigs.c` = `285`
- `benchmarks/bench_iterative_reuse.c` = `251`
- `benchmarks/bench_eigs_reuse.c` = `201`

Interpretation:

- Sprint 49 should treat the iterative/eigensolver public headers plus their
  wrapper implementations as the main direct landing zone
- examples/benchmarks are compatibility and migration-proof surfaces, not the
  first design surface

#### 4. The regression surface for final lifecycle exposure is already concentrated and measurable

The live regression concentration is explicit:

- iterative family:
  - `tests/test_iterative.c` = `2795`
  - `tests/test_block_solvers.c` = `507`
  - `tests/test_minres.c` = `1588`
  - `tests/test_bicgstab.c` = `1586`
  - `tests/test_stagnation.c` = `1361`
- eigensolver family:
  - `tests/test_eigs.c` = `1269`
  - `tests/test_eigs_thick_restart.c` = `1161`
  - `tests/test_eigs_lobpcg.c` = `1196`

Interpretation:

- Sprint 49 already has a clear compatibility/regression proof surface
- the final public-lifecycle landing does not need a new test-discovery sprint

#### 5. The final public API work must stay compatibility-preserving

The live headers and internal wrappers still make the current public contract
clear:

- iterative public entry points are still one-shot solver APIs
- eigensolver public entry remains:
  - `sparse_eigs_sym(...)`
- internal repeated-run benchmarking already reuses caller-owned internal
  workspace/state without exposing that surface publicly

Interpretation:

- Sprint 49 should preserve the old one-shot calling style as a supported path
- any new explicit lifecycle/workspace exposure should layer on top of, not
  replace, the current public entry points

#### 6. Migration-path documentation is a first-class workstream, not an afterthought

Day 1 evidence shows Sprint 49 already has the raw ingredients for a useful
migration story:

- older explicit public reusable lifecycle:
  - analysis / numeric factorization / refactor
- newer internal repeated-run lifecycle:
  - iterative reusable workspace
  - eigensolver reusable workspace
- existing example and benchmark surfaces demonstrating repeated-run value

Interpretation:

- Sprint 49 migration docs should explain when the existing one-shot path is
  still the right choice and when the explicit lifecycle/workspace path is
  preferable
- the docs should be grounded in the actual landed public contract, not generic
  “handles are faster” claims

#### 7. The final residual review already has a concrete target set

The project-plan queue for Sprint 49 is not only API landing. It also requires
revisiting:

- `review-codex-2026-05-21.md`
- later inherited residuals from the lifecycle/workspace/documentation sprints
- the final cross-surface compatibility state after public exposure

Interpretation:

- Sprint 49 must reserve real bandwidth for residual classification and final
  Epic 4 integration reporting
- it should not spend the whole sprint only on header/API edits

#### 8. The front-half order of the sprint is fixed before implementation starts

The correct early sprint order is:

1. baseline and public-surface inventory
2. lifecycle API design
3. header/API landing
4. implementation/wrapper integration
5. migration docs
6. cross-surface compatibility sweep
7. residual review
8. final validation and closeout

Interpretation:

- Sprint 49 should preserve the Epic 4 pattern that public-facing cleanup lands
  only after seam mapping and bounded implementation design are explicit
