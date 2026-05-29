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

## Day 2

**Objective:** Refresh the public lifecycle/workspace seam inventory so Sprint
49's lifecycle API design, bounded public landing, migration-path
documentation, compatibility sweep, and final residual review are sequenced
from the live post-Sprint-48 repo state rather than only from the project-plan
labels.

### Commands Run

1. Re-read the Sprint 49 Day 2 plan section:
   - `sed -n '57,96p' docs/planning/EPIC_4/SPRINT_49/PLAN.md`
2. Re-read the Day 1 baseline artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_49/artifacts/day1-scope-and-lifecycle-api-baseline.md`
3. Re-read the main public lifecycle precedent and the current one-shot public
   solver/eigensolver headers:
   - `sed -n '1,220p' include/sparse_analysis.h`
   - `sed -n '1,260p' include/sparse_iterative.h`
   - `sed -n '1,260p' include/sparse_eigs.h`
4. Refresh the live caller-facing usage markers:
   - `rg -n "sparse_analyze|sparse_factor_numeric|sparse_refactor_numeric|sparse_solve_cg|sparse_solve_gmres|sparse_eigs_sym|with_workspace_internal|bench_iterative_reuse|bench_eigs_reuse" README.md docs examples benchmarks tests include src -g '!build'`
5. Re-read the direct repeated-run comparison drivers:
   - `sed -n '1,240p' benchmarks/bench_iterative_reuse.c`
   - `sed -n '1,240p' benchmarks/bench_eigs_reuse.c`
6. Re-read the main public examples that currently teach solver/eigensolver
   usage:
   - `sed -n '1,240p' examples/example_iterative.c`
   - `sed -n '1,240p' examples/example_eigs.c`

### Day 2 Findings

#### 1. The public lifecycle story now breaks cleanly into three distinct classes rather than one generic API backlog

The live repo already exposes three different public-facing patterns:

- reusable public lifecycle already present:
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_refactor_numeric(...)`
  - `sparse_factor_free(...)`
- compatibility-oriented one-shot iterative entry points:
  - `sparse_solve_cg(...)`
  - `sparse_solve_gmres(...)`
  - matrix-free and block convenience variants nearby
- compatibility-oriented one-shot eigensolver entry point:
  - `sparse_eigs_sym(...)`

Interpretation:

- Sprint 49 is not choosing between "handles everywhere" and "one-shot
  everywhere"
- it is reconciling these three already-existing public classes into a coherent
  final lifecycle/workspace story

#### 2. The real first landing targets are the iterative and eigensolver public surfaces, not the analysis/factor public precedent

`include/sparse_analysis.h` already teaches a stable explicit reusable-handle
workflow. The missing public-side work is concentrated instead in:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

Interpretation:

- `sparse_analysis.h` is mainly the public precedent and terminology anchor
- Sprint 49's first direct landing targets are the iterative/eigensolver public
  headers and their wrapper implementation surfaces

#### 3. The internal repeated-run paths are explicit and isolated enough to support a bounded public exposure

The direct repeated-run benchmark drivers prove that the reusable paths already
exist as stable internal seams:

- iterative:
  - `sparse_solve_cg_with_workspace_internal(...)`
  - `sparse_solve_gmres_with_workspace_internal(...)`
  - `benchmarks/bench_iterative_reuse.c`
- eigensolver:
  - `sparse_eigs_sym_with_workspace_internal(...)`
  - `benchmarks/bench_eigs_reuse.c`

Interpretation:

- Sprint 49 does not need another internal groundwork sprint before public
  exposure
- it can treat the internal repeated-run helpers as the backing seam and focus
  on a compatibility-preserving public contract

#### 4. Examples and README still teach one-shot usage, which makes them later verification surfaces rather than the first design surface

The main public examples and top-level docs currently present:

- iterative usage via:
  - `examples/example_iterative.c`
  - `README.md`
- eigensolver usage via:
  - `examples/example_eigs.c`
  - `README.md`

Those surfaces demonstrate valuable caller expectations, but they do not define
the public contract by themselves.

Interpretation:

- examples and README are migration-proof and compatibility-proof surfaces
- they should be updated after the bounded header/source landing is stable
- they are not the right first design surface for Sprint 49

#### 5. The remaining Sprint 49 work now reduces cleanly to five bounded seam buckets

The live repo state now reduces the remaining queue to:

- explicit lifecycle/workspace public exposure
- compatibility-preserving one-shot wrapper routing
- caller guidance and migration rules
- cross-surface documentation/example/benchmark/test agreement
- final residual-review and Epic 4 bookkeeping

Interpretation:

- Sprint 49 no longer has a generic "final integration" backlog
- each remaining deliverable class now has a concrete ownership surface

#### 6. The first landing order is fixed by the live public-vs-internal split

The correct execution order from the current repo state is:

1. lifecycle API design
2. bounded public header / source landing
3. migration-path documentation
4. cross-surface compatibility sweep
5. final residual review
6. final validation and closeout

Interpretation:

- Sprint 49 should not start by editing examples, README, or benchmark drivers
- it should first define and land the bounded public lifecycle/workspace shape
  the rest of the surfaces must then describe

## Day 3

**Objective:** Define the bounded public lifecycle/workspace model Sprint 49
can safely expose now that the lifecycle scaffolding, internal reusable
workspace seams, and documentation-policy homes already exist, while keeping
the old one-shot solver/eigensolver entry points as supported first-class
compatibility surfaces.

### Commands Run

1. Re-read the Sprint 49 Day 3 plan section:
   - `sed -n '97,131p' docs/planning/EPIC_4/SPRINT_49/PLAN.md`
2. Re-read the Day 2 seam inventory:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_49/artifacts/day2-public-lifecycle-surface-inventory.md`
3. Re-read the public reusable lifecycle precedent and the current one-shot
   public iterative/eigensolver surfaces:
   - `sed -n '1,220p' include/sparse_analysis.h`
   - `sed -n '220,520p' include/sparse_iterative.h`
   - `sed -n '237,620p' include/sparse_eigs.h`
4. Re-read the maintainer-facing policy-home guidance relevant to public
   contract placement:
   - `sed -n '1,220p' docs/maintainer_guide.md`
5. Refresh the current public/internal usage split markers:
   - `rg -n "sparse_analyze|sparse_factor_numeric|sparse_refactor_numeric|sparse_solve_cg|sparse_solve_gmres|sparse_eigs_sym|with_workspace_internal|bench_iterative_reuse|bench_eigs_reuse" README.md docs examples benchmarks tests include src -g '!build'`
6. Re-read the repeated-run benchmark and example surfaces that will later need
   to agree with the final public model:
   - `sed -n '1,240p' benchmarks/bench_iterative_reuse.c`
   - `sed -n '1,240p' benchmarks/bench_eigs_reuse.c`
   - `sed -n '1,240p' examples/example_iterative.c`
   - `sed -n '1,240p' examples/example_eigs.c`

### Day 3 Findings

#### 1. Sprint 49 should expose a public lifecycle layer, not a public internal-workspace mirror

The Day 2 seam map plus the live headers make the main design rule explicit:

- `sparse_analysis.h` already teaches a public reusable-handle lifecycle
- the repeated-run iterative/eigensolver helpers already exist internally
- the internal helper names are implementation-oriented and benchmark-oriented

Interpretation:

- Sprint 49 should not publish the exact `*_with_workspace_internal(...)`
  surfaces or the raw internal workspace owner structs
- it should expose a bounded public lifecycle layer that composes with the
  existing repeated-run internals while preserving a cleaner public contract

#### 2. The public contract should follow one common lifecycle shape across iterative and eigensolver work

The best bounded public model now follows the same high-level lifecycle already
present in `sparse_analysis.h`:

1. initialize / zero a public handle
2. prepare or configure for a stable-dimension repeated-run path
3. run one or more solves / eigensolver calls through that handle
4. reset or reuse without preserving old Krylov/subspace state
5. free explicit owned resources

Interpretation:

- Sprint 49 should align iterative/eigensolver public lifecycle wording with
  the existing analyze/factor precedent
- it should avoid surfacing algorithm-specific storage layout details as part of
  the public contract

#### 3. Compatibility wrappers must remain first-class rather than transitional leftovers

The current public headers, examples, and README still teach one-shot usage.
That caller shape remains valid and important.

Interpretation:

- the existing one-shot entries should stay supported public entry points
- Sprint 49 should describe them explicitly as compatibility-oriented or
  convenience-oriented one-shot wrappers over the new lifecycle layer
- Sprint 49 should not frame them as deprecated or second-class

#### 4. Option structs and result structs should stay stable caller surfaces rather than being redesigned around handle ownership

The existing public iterative/eigensolver APIs already expose:

- option structs
- result structs
- caller-owned output buffers for eigensolvers
- designated-initializer usage patterns

Interpretation:

- Sprint 49 should preserve those option/result surfaces as the primary caller
  configuration/result contract
- the new lifecycle layer should compose around them rather than replacing them
  with large new configuration objects

#### 5. Reset/reuse semantics must be explicit and cheap

The internal repeated-run model already proves the intended behavior:

- preserve allocation capacity across repeated stable-dimension runs
- do not preserve prior iteration/subspace/search state as a feature
- let each run start fresh numerically while amortizing allocation churn

Interpretation:

- the public lifecycle design should make reuse capacity explicit
- it should make numerical-state persistence explicitly unsupported
- it should keep resize/reprepare semantics bounded to stable dimensions unless
  later work chooses to widen them

#### 6. The main Day 3 non-goals are now explicit

Sprint 49 should not do any of the following under the banner of public
lifecycle exposure:

- broad solver API redesign
- publicizing every internal helper or workspace view
- removing or deprecating the one-shot solver/eigensolver entry points
- rewriting examples/README before the public header/source contract is landed
- introducing a public API promise around internal storage layout

Interpretation:

- the design is now bounded tightly enough to guide Day 5/6 implementation
  without encouraging public-surface sprawl

## Day 4

**Objective:** Convert the Day 3 public lifecycle target into a concrete
implementation order, validation contract, and scope boundary so the header/API
landing and wrapper integration work can proceed without widening into generic
example, benchmark, or post-Epic-4 churn.

### Commands Run

1. Re-read the Sprint 49 Day 4 plan section:
   - `sed -n '132,171p' docs/planning/EPIC_4/SPRINT_49/PLAN.md`
2. Re-read the Day 3 public lifecycle API design:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_49/artifacts/day3-public-lifecycle-api-design.md`
3. Re-read the Sprint 40 validation anchor:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
4. Re-read the current benchmark/example README scope boundaries:
   - `sed -n '1,240p' benchmarks/README.md`
   - `sed -n '1,240p' examples/README.md`

### Day 4 Findings

#### 1. Any Sprint 49 code/header landing must use the full required C/header gate by default

The Sprint 40 validation anchor still governs Sprint 49 implementation work:

- any `*.c` / `*.h` change requires:
  - `make format`
  - `make lint`
  - `make test`

Interpretation:

- Sprint 49 should treat this as the non-negotiable floor for Day 5/6 code
  landing
- there is no special exception just because the work is mostly public-API
  exposure rather than algorithm invention

#### 2. Substantial public-API landing batches should also default to the reviewed local baseline

The Day 3 design affects:

- public headers
- wrapper routing
- caller-visible lifecycle semantics
- likely later docs/example/benchmark alignment

Interpretation:

- Day 5/6 substantial public-API batches should default to:
  - `make quality-review-full`
- Sprint 49 should preserve the stronger reviewed baseline whenever the change
  spans public header and implementation surfaces together

#### 3. Targeted follow-on checks should be driven by touched surface, not run universally

The current repo already has focused verification surfaces that map naturally to
the expected Sprint 49 work:

- examples:
  - `./build/example_iterative`
  - `./build/example_matrix_free`
  - `./build/example_eigs`
- repeated-run benchmarks:
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`
- compile-only tooling gate:
  - `make tooling-build`
- solver/eigensolver regression binaries:
  - `./build/test_iterative`
  - `./build/test_block_solvers`
  - `./build/test_minres`
  - `./build/test_bicgstab`
  - `./build/test_stagnation`
  - `./build/test_eigs`
  - `./build/test_eigs_thick_restart`
  - `./build/test_eigs_lobpcg`

Interpretation:

- Sprint 49 should rerun these only when the touched surface justifies them
- examples and repeated-run benchmarks are high-value follow-ons after public
  lifecycle landing, but they are not universal every-day gates

#### 4. The landing order must stay header/API first, then migration/docs/examples/benchmarks

The Day 2 seam map and Day 3 design still force the correct order:

1. public header / API surface
2. implementation / wrapper integration
3. migration-path documentation
4. cross-surface compatibility sweep
5. residual review
6. final validation

Interpretation:

- Sprint 49 should not start by rewriting examples or README to "guess" the
  final public contract
- the public header/source shape must become real first

#### 5. Benchmark/example surfaces are later verification layers, not implementation drivers

The current benchmark/example docs stay intentionally local in scope:

- `benchmarks/README.md` focuses on benchmark-local command usage
- `examples/README.md` focuses on example-local public usage references

Interpretation:

- Sprint 49 should keep benchmark/example changes bounded to agreement with the
  landed API
- it should not turn Day 5/6 into a broad educational-surface rewrite

#### 6. Sprint 49’s out-of-scope boundary is now explicit enough to protect the implementation days

The following are not part of the intended Sprint 49 implementation landing:

- post-Epic-4 feature expansion
- large new benchmark framework work
- new solver families
- broad tutorial rewrite unrelated to the final lifecycle shape
- exposing raw internal workspace layout as public API
- replacing the existing one-shot public entries instead of preserving them

Interpretation:

- Day 5/6 code work now has a concrete fence that should prevent public-surface
  sprawl while still allowing a real public lifecycle landing

## Day 5 — Public Lifecycle API Landing Batch I

### Summary

Sprint 49 now has the first real public repeated-run lifecycle declarations for
the two main internal reuse seams already established by Sprint 45 and Sprint
46:

- iterative repeated-run handle surface in `include/sparse_iterative.h`
- eigensolver repeated-run handle surface in `include/sparse_eigs.h`

This Day 5 batch stayed intentionally header/API-only, matching the Day 4
landing order:

1. public header / API surface
2. implementation / wrapper integration
3. migration-path documentation
4. cross-surface compatibility sweep
5. residual review
6. final validation

Interpretation:

- Day 5 is the public contract landing
- Day 6 remains the implementation/wrapper integration day
- no attempt was made to widen into examples, benchmarks, or migration docs
  before the public surface existed

### Landed Public Surface

The iterative header now exposes a bounded repeated-run lifecycle seam:

- `sparse_iter_handle_t`
- `sparse_iter_handle_init(...)`
- `sparse_iter_handle_free(...)`
- `sparse_iter_handle_prepare_cg(...)`
- `sparse_iter_handle_prepare_gmres(...)`
- `sparse_solve_cg_with_handle(...)`
- `sparse_solve_gmres_with_handle(...)`

The eigensolver header now exposes the parallel bounded seam:

- `sparse_eigs_handle_t`
- `sparse_eigs_handle_init(...)`
- `sparse_eigs_handle_free(...)`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`

Design properties preserved:

- public terminology is lifecycle-centric rather than raw internal-helper
  centric
- one-shot `sparse_solve_cg(...)`, `sparse_solve_gmres(...)`, and
  `sparse_eigs_sym(...)` remain first-class supported compatibility entries
- no raw internal workspace owners, typed internal views, or storage-layout
  promises were made public

### Important Boundary Decisions

The public handle structs are intentionally opaque at the public level:

- each handle currently exposes only `internal_state`
- callers are directed to zero-init or use the init helper
- prepare/run/free is the public contract, not storage-field manipulation

The solve-path declarations deliberately stay narrow:

- iterative public repeated-run declarations cover the already-migrated direct
  CG/GMRES paths
- eigensolver repeated-run declarations cover the main symmetric eigensolver
  public path
- block/minres/BiCGSTAB public repeated-run expansion did not land early
- matrix-free repeated-run public declarations did not land early

Interpretation:

- Day 5 delivered the smallest coherent public lifecycle surface that Day 6 can
  back with real implementation
- Sprint 49 did not turn the header batch into a broad public solver-family
  redesign

### Validation

Because `*.h` changed, the required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

Because this was a substantial public-API landing, the stronger reviewed
baseline also ran:

- `make quality-review-full`

That also passed, including:

- reviewed CMake parity still at `53`
- Makefile/CMake test-count parity still `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 597.27 sec`

Targeted touched-family follow-ons also passed:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/example_iterative`
- `./build/example_eigs`

Representative direct results:

- `test_iterative`: all `76` tests passed
- `test_eigs`: all `25` tests passed
- `example_iterative`: GMRES converged in `25` iterations unpreconditioned and
  `9` with ILU(0)
- `example_eigs`: all three shipped demos converged and reported stable
  residuals

### Day 5 Position

The public contract is now real enough for Day 6 to do the bounded
implementation/wrapper integration:

- wire the public handle declarations to the existing internal reuse seams
- preserve result initialization, error-path safety, and free semantics
- keep one-shot compatibility wrappers intact
- avoid widening into README/examples/benchmarks as an implementation driver

Bottom line:

- Day 5 successfully landed the first public lifecycle API surface
- it preserved compatibility by leaving the one-shot public model intact
- it stayed inside the intended Sprint 49 fence
- the full required and reviewed validation baselines are green

## Day 6 — Public Lifecycle API Integration Batch II

### Goal

Back the new Day 5 public lifecycle declarations with real implementation and
compatibility-preserving wrapper routing, without widening the public promise
set beyond the intended repeated-run handle contract.

### Files Touched

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_eigs_internal.h`

### Main Integration Result

Sprint 49 now has a real public repeated-run implementation path, not just a
declared handle surface.

The Day 6 batch stayed within the Day 3/4 fence:

- iterative public handles now own and reuse the existing internal iterative
  workspace seam
- eigensolver public handles now own and reuse the existing internal
  eigensolver workspace seam
- one-shot public wrappers remain first-class compatibility entries, but now
  route through the new public handle path instead of standing apart from it
- no raw internal workspace layout, typed internal views, or benchmark-only
  helpers were made public

### Iterative Public Handle Integration

`src/sparse_iterative.c` now implements the new public lifecycle surface:

- `sparse_iter_handle_init(...)`
- `sparse_iter_handle_free(...)`
- `sparse_iter_handle_prepare_cg(...)`
- `sparse_iter_handle_prepare_gmres(...)`
- `sparse_solve_cg_with_handle(...)`
- `sparse_solve_gmres_with_handle(...)`

Key behavior:

- `sparse_iter_handle_t` owns a `sparse_iter_workspace_t` through
  `internal_state`
- prepare calls allocate or grow capacity through the same checked internal
  workspace prepare helpers already used by the Sprint 45 repeated-run seam
- run calls delegate to the existing internal workspace-backed CG/GMRES solve
  paths
- one-shot `sparse_solve_cg(...)` and `sparse_solve_gmres(...)` now create a
  temporary handle, run through the public handle path, and free it afterward

Interpretation:

- the public repeated-run iterative contract is now real rather than
  documentary
- the one-shot iterative API remains supported by construction rather than by
  parallel duplicated logic

### Eigensolver Public Handle Integration

`src/sparse_eigs.c` and `src/sparse_eigs_internal.h` now implement the public
repeated-run eigensolver path:

- `sparse_eigs_handle_init(...)`
- `sparse_eigs_handle_free(...)`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`

Key behavior:

- `sparse_eigs_handle_t` owns a `sparse_eigs_workspace_t` through
  `internal_state`
- prepare calls pre-allocate the correct backend-shaped capacity for:
  - grow-m Lanczos
  - thick-restart Lanczos
  - LOBPCG
- `sparse_eigs_sym_with_handle(...)` delegates to the existing shared backend
  implementation with caller-owned reusable workspace
- one-shot `sparse_eigs_sym(...)` now routes through the public handle path
  before freeing the temporary handle

Important Day 6 widening that stayed internal-only:

- `s21_lobpcg_solve(...)` now accepts an optional reusable workspace pointer so
  the new public eigensolver handle path covers LOBPCG cleanly instead of only
  the Lanczos families
- that change remained in internal code and did not widen the public API beyond
  the already-declared lifecycle contract

### Important Boundary Decisions

The public implementation batch deliberately did **not** land:

- public matrix-free repeated-run handle entries
- public block/minres/BiCGSTAB repeated-run entries
- benchmark/example migration as an API driver
- README/tutorial migration text
- raw internal-storage field semantics as public API

That was the correct fence:

- Day 6 needed to make Day 5’s public declarations true
- it did not need to turn Sprint 49 into a broad public solver-family redesign

### Validation

Because `*.c` and `*.h` changed, the required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

Because this was a substantial public-API integration batch, the stronger
reviewed baseline also ran:

- `make quality-review-full`

That also passed, including:

- reviewed CMake parity still at `53`
- Makefile/CMake test-count parity still `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 422.68 sec`

Targeted touched-family follow-ons also passed:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/example_iterative`
- `./build/example_eigs`

Representative direct results:

- `test_iterative`: all `76` tests passed
- `test_eigs`: all `25` tests passed
- `test_eigs_lobpcg`: all `26` tests passed
- `example_iterative`: GMRES converged in `25` iterations unpreconditioned and
  `9` with ILU(0)
- `example_eigs`: all three shipped demos converged cleanly, including the
  LOBPCG `bcsstk04` section at `3 / 3`

### Day 6 Position

Sprint 49 now has both sides of the bounded public lifecycle story:

- Day 5 made the public repeated-run contract explicit
- Day 6 wired that contract to the real internal reuse seams
- one-shot wrappers remain compatibility-preserving convenience surfaces

That leaves the next queue much clearer:

- migration-path documentation
- cross-surface compatibility proof across docs/examples/benchmarks/tests
- final residual review and Epic 4 closeout validation

Bottom line:

- Day 6 successfully made the new public lifecycle API real
- it preserved compatibility by routing existing one-shot entries through the
  same handle-backed implementation path
- it widened internal LOBPCG reuse just enough to keep the public repeated-run
  eigensolver path coherent
- the full required, reviewed, and targeted follow-on validation baselines are
  green

## Day 7 — Post-Landing API Audit

### Goal

Re-audit the live post-landing public lifecycle/workspace surface so the
remaining Sprint 49 queue is concrete before migration docs or compatibility
cleanup starts.

### Audited Surfaces

Public contract / implementation:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

Caller-facing and compatibility surfaces:

- `README.md`
- `examples/README.md`
- `benchmarks/README.md`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `tests/`

### Main Audit Result

The public lifecycle landing itself is complete and still bounded.

What is already true:

- public iterative and eigensolver handle declarations exist
- the implementation/wrapper path is live behind those declarations
- existing one-shot public entries remain first-class and route through the
  new handle-backed model where appropriate
- no raw internal workspace layout or typed internal view leaked into the
  public contract

Interpretation:

- Sprint 49 is no longer blocked on public API exposure itself
- the remaining work is now caller-guidance and cross-surface agreement work,
  not another lifecycle implementation batch

### What Is Complete

The following Day 5/6 goals now look done rather than still implied:

- bounded lifecycle-centric public terminology
- compatibility-preserving one-shot wrapper continuity
- public prepare / run / free contract for the first iterative and
  eigensolver repeated-run paths
- internal implementation reuse behind the public lifecycle surface

No meaningful API-sprawl was found:

- public exposure is still limited to the intended CG/GMRES and symmetric
  eigensolve repeated-run handles
- matrix-free, block, MINRES, and BiCGSTAB repeated-run public exposure did
  not accidentally widen
- README/tutorial/example churn did not leak into the Day 5/6 code batches

### What Is Not Done Yet

The strongest remaining gaps are now outside the core public headers:

1. migration-path documentation is still missing
2. cross-surface agreement on the final caller story is still incomplete
3. direct regression coverage for the new public handle entries is still absent

Evidence from the live repo state:

- the public handle names appear in `include/` and Sprint 49 notes/artifacts,
  but not yet in the main user-facing docs
- `README.md` does not yet explain:
  - old one-shot usage remains supported
  - when explicit handles are preferable
  - what “prepare once / repeated run” means for callers
- `examples/README.md` still presents the current examples as one-shot public
  usage references only
- the repeated-run benchmarks still exercise internal reuse seams:
  - `bench_iterative_reuse.c` uses
    `sparse_solve_*_with_workspace_internal(...)`
  - `bench_eigs_reuse.c` uses
    `sparse_eigs_sym_with_workspace_internal(...)`
- there is still no direct `tests/` coverage for:
  - `sparse_iter_handle_*`
  - `sparse_solve_*_with_handle(...)`
  - `sparse_eigs_handle_*`
  - `sparse_eigs_sym_with_handle(...)`

### Naming / Ownership Drift

No serious naming drift remains in the public headers themselves, but there is
ownership drift across adjacent surfaces:

- public headers now define the final repeated-run caller contract
- benchmarks still describe repeated-run evidence in internal-workspace terms
- examples and README do not yet present the new public repeated-run path as
  part of the supported caller model

Interpretation:

- Day 8 should focus on migration-path clarity
- Day 9 should focus on mapping which examples/benchmarks/tests/docs need the
  smallest agreement sweep
- Day 10 should then land the smallest coherent compatibility cleanup

### Day 8 Boundary

The migration-doc batch should stay focused on caller guidance, not on broad
surface churn.

Strongest likely Day 8 targets:

- `README.md`
- one bounded supporting docs surface if needed:
  - `examples/README.md`
  - or a small tutorial/maintainer-guide cross-reference

What Day 8 should explain explicitly:

- the one-shot path remains fully supported
- explicit handles are the repeated-run opt-in path
- reuse preserves allocation capacity, not numerical iteration state
- repeated-run handles are worth using on stable-dimension repeated solves
- callers do not need to adopt handles just to stay supported

### Day 9/10 Boundary

The compatibility sweep should stay behavior-level and targeted.

Strongest Day 9 audit / Day 10 batch candidates:

- reuse benchmarks
- direct public-handle regression coverage in `tests/`
- any nearby README/examples wording that would still contradict the final
  caller story after the migration-doc batch lands

Important non-goal:

- do not turn the compatibility sweep into a broad example or benchmark
  framework rewrite

### Day 7 Position

Sprint 49 is now in the right final shape:

- public lifecycle API exposure is materially done
- the next value is in migration guidance and cross-surface agreement
- the remaining queue is concrete enough to keep the last code/doc batches
  bounded

Bottom line:

- Day 7 confirmed the public lifecycle landing is complete and bounded
- the strongest remaining gaps are migration docs, direct public-handle
  regression coverage, and reuse-benchmark alignment with the final public
  caller model
- Day 8 should document the old-vs-new usage path
- Day 9/10 should then reconcile the highest-value examples/benchmarks/tests
  around that final model

## Day 8 — Migration-Path Documentation Batch

### Goal

Document the final old-vs-new caller path clearly enough that existing users
can stay on the one-shot APIs while repeated-run callers can see when explicit
handles are worth using.

### Files Touched

- `README.md`
- `examples/README.md`

### Main Documentation Result

Sprint 49 now has a real migration-path explanation instead of only header
contracts and sprint-local notes.

The Day 8 batch stayed intentionally narrow:

- `README.md` is now the primary user-facing explanation of:
  - one-shot compatibility
  - explicit repeated-run handles
  - when reuse is worth it
  - the basic prepare / run / free lifecycle
- `examples/README.md` now clarifies that the shipped examples remain simple
  one-shot public references even though explicit repeated-run handles now
  exist

Interpretation:

- existing callers can now read the top-level docs and know they do **not**
  need to migrate just to stay supported
- repeated-run callers can now find the public-handle path without digging only
  through header comments

### README Migration Guidance

The new top-level README section now makes the final public shape explicit:

- one-shot entries remain first-class:
  - `sparse_solve_cg(...)`
  - `sparse_solve_gmres(...)`
  - `sparse_eigs_sym(...)`
- repeated-run handles are the opt-in path for stable-dimension repeated work:
  - iterative:
    - `sparse_iter_handle_t`
    - `sparse_iter_handle_init(...)`
    - `sparse_iter_handle_prepare_cg(...)`
    - `sparse_iter_handle_prepare_gmres(...)`
    - `sparse_solve_cg_with_handle(...)`
    - `sparse_solve_gmres_with_handle(...)`
    - `sparse_iter_handle_free(...)`
  - eigensolver:
    - `sparse_eigs_handle_t`
    - `sparse_eigs_handle_init(...)`
    - `sparse_eigs_handle_prepare(...)`
    - `sparse_eigs_sym_with_handle(...)`
    - `sparse_eigs_handle_free(...)`

The guidance also states the most important behavioral truth:

- reuse preserves allocation capacity, not old numerical iteration state

That is the right Day 8 level of detail:

- enough for caller migration-path clarity
- not so much that the README becomes a duplicate of the public headers

### Example-Surface Handoff

`examples/README.md` now states the intended scope explicitly:

- the shipped examples still lean on one-shot public APIs
- that is deliberate, because those APIs remain first-class and simpler for
  most callers
- repeated-run handles exist, but they are an opt-in path for stable-dimension
  repeated work rather than a replacement for the shipped one-shot examples

This was the right supporting touch:

- it keeps example scope honest
- it avoids forcing Day 8 into a broad example rewrite
- it leaves Day 9/10 free to decide whether any concrete example should change
  later for compatibility-sweep reasons

### Important Boundary Decisions

The migration-doc batch deliberately did **not** yet land:

- benchmark-driver wording changes
- direct public-handle regression tests
- tutorial rewrite
- maintainer-guide expansion
- broad example rewrites to use the new handles

That was the correct fence:

- Day 8 needed to explain the final caller model
- it did not need to start the broader cross-surface agreement sweep early

### Targeted Sanity Checks

This was a docs-only batch, so I did not run `make format`, `make lint`, or
`make test`.

I ran targeted Day 8 sanity checks instead:

- `rg -n "Repeated-Run Lifecycle Handles|sparse_iter_handle_|sparse_eigs_handle_|one-shot public APIs|opt-in path" README.md examples/README.md`
- `wc -l README.md examples/README.md`
- spot-read the new README migration section in context

All were clean.

### Day 8 Position

The migration-path explanation is now in place, which makes the next queue
cleaner:

- Day 9 can audit the highest-value remaining drift across benchmarks/tests/docs
- Day 10 can then land the smallest coherent agreement batch

Bottom line:

- Day 8 successfully documented the old-vs-new repeated-run caller path
- it preserved the one-shot compatibility story explicitly
- it gave explicit handles a real top-level documentation home
- it stayed tightly bounded to the README plus one supporting example-scope
  surface

## Day 9 — Cross-Surface Compatibility Audit

### Goal

Map the smallest high-signal compatibility sweep now that the public lifecycle
API is landed and the migration path is documented.

### Audited Surfaces

Docs / examples:

- `README.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/tutorial.md`

Benchmarks:

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`

Tests:

- `tests/test_iterative.c`
- `tests/test_eigs.c`
- broader `tests/` public-contract references via search

Public headers:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

### Main Audit Result

The remaining cross-surface drift is now narrower than the Day 7 audit first
suggested.

What now looks good enough to leave alone for Day 10:

- the main README migration guidance is in place
- `examples/README.md` now explains why the shipped examples still lean on the
  one-shot path
- the public headers already describe the handle contract clearly
- broad tutorial churn is not required for Epic 4 closeout

What still looks like the highest-value remaining compatibility drift:

1. repeated-run benchmarks still present internal reuse seams rather than the
   final public handle path
2. direct public-handle regression coverage is still absent from `tests/`

Interpretation:

- examples and top-level docs are no longer the strongest Day 10 targets
- the highest-value Day 10 work should center on benchmarks plus focused tests

### Example-Surface Disposition

The example surface now reads acceptably for Sprint 49 closeout:

- examples remain simple one-shot public usage references
- that is now documented explicitly
- no example currently claims that one-shot is the *only* supported caller
  model

Therefore:

- no broad example rewrite is needed for Day 10
- only a small example touch would be justified if it were required to support
  a nearby benchmark/test clarification, which does not currently appear
  necessary

### Benchmark-Surface Drift

The strongest remaining benchmark disagreement is real and concrete.

`bench_iterative_reuse.c` still proves repeated-run evidence through:

- `sparse_solve_cg_with_workspace_internal(...)`
- `sparse_solve_gmres_with_workspace_internal(...)`

`bench_eigs_reuse.c` still proves repeated-run evidence through:

- `sparse_eigs_sym_with_workspace_internal(...)`

Why this now matters more than it did before Sprint 49:

- these were the correct seams when repeated-run support was still internal
- after Day 5/6, the final caller-facing repeated-run contract is the public
  handle path
- continuing to benchmark only the internal seams leaves the public repeated-run
  story only partially reflected in the repo surface

That makes the reuse benchmarks strong Day 10 candidates.

### Test-Surface Drift

The second strongest remaining gap is also concrete:

- there is still no direct regression coverage for:
  - `sparse_iter_handle_init(...)`
  - `sparse_iter_handle_prepare_cg(...)`
  - `sparse_iter_handle_prepare_gmres(...)`
  - `sparse_solve_cg_with_handle(...)`
  - `sparse_solve_gmres_with_handle(...)`
  - `sparse_eigs_handle_init(...)`
  - `sparse_eigs_handle_prepare(...)`
  - `sparse_eigs_sym_with_handle(...)`

Current test coverage is still mainly through:

- one-shot iterative/eigensolver behavior surfaces
- family-level regression and integration tests

That is a good safety floor, but it does not yet pin the final public repeated-
run contract directly.

This makes focused public-handle regression coverage the other strong Day 10
candidate.

### Docs-Surface Drift

`benchmarks/README.md` remains mostly benchmark-local and does not yet mention
the public repeated-run handle path.

That is now a secondary issue rather than the primary one:

- if Day 10 touches the reuse benchmarks, a small corresponding
  `benchmarks/README.md` clarification would be justified
- a broader docs pass is not needed

`docs/tutorial.md` still teaches the one-shot iterative path and matrix-free
path without the new repeated-run handle discussion.

That is acceptable for Sprint 49:

- the tutorial is still functionally correct
- the README now owns the primary old-vs-new migration explanation
- rewriting the tutorial would be broader than the highest-value remaining
  sweep

### Day 10 Target List

The smallest coherent high-signal Day 10 batch now looks like:

Primary targets:

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- focused direct public-handle regression additions in:
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`

Secondary touch only if needed:

- `benchmarks/README.md`

Intended Day 10 behavior boundary:

- preserve the same numerical behavior and repeated-run evidence goals
- update the repeated-run path to reflect the final public handle model
- add compact direct public-handle regression coverage
- avoid broad example or tutorial churn

### Important Non-Goals

Day 10 should still avoid:

- converting all examples to handle usage
- broad benchmark framework work
- large test refactors unrelated to the public handle contract
- tutorial expansion beyond a tiny local clarification, if any

### Day 9 Position

Sprint 49 is now down to a very small final compatibility queue:

- reuse benchmarks should likely move from internal repeated-run seams to the
  final public handle path
- direct regression coverage should pin the new public handle contract
- any docs touch beyond that should stay minimal and local

Bottom line:

- Day 9 reduced the compatibility sweep to one strong implementation bucket and
  one strong regression bucket
- examples and the main README now look good enough to leave alone
- Day 10 should focus on reuse benchmarks, direct public-handle tests, and at
  most one tiny benchmark-doc clarification

## Day 10 — Cross-Surface Compatibility Sweep Batch

### Goal

Land the smallest coherent agreement batch across benchmarks and regression
tests so the final Sprint 49 repeated-run story reflects the new public handle
contract instead of only the earlier internal reuse seams.

### Touched Surfaces

Benchmarks:

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`

Tests:

- `tests/test_iterative.c`
- `tests/test_eigs.c`

Untouched by design:

- `README.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/tutorial.md`

### Main Day 10 Result

The compatibility sweep landed cleanly without widening into example or docs
churn.

What changed:

1. the repeated-run benchmarks now prove the final public handle path instead
   of internal-only workspace entry points
2. direct regression coverage now pins the new iterative and eigensolver public
   handle APIs

What stayed intentionally unchanged:

- one-shot public APIs remain the default example/docs path
- no benchmark framework redesign
- no tutorial rewrite
- no broad example conversion to explicit handle usage

### Benchmark Agreement Outcome

`bench_iterative_reuse.c` now routes its repeated-run path through:

- `sparse_iter_handle_t`
- `sparse_iter_handle_prepare_cg(...)`
- `sparse_iter_handle_prepare_gmres(...)`
- `sparse_solve_cg_with_handle(...)`
- `sparse_solve_gmres_with_handle(...)`

`bench_eigs_reuse.c` now routes its repeated-run path through:

- `sparse_eigs_handle_t`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`

Interpretation:

- the benchmark evidence now matches the final caller-facing repeated-run API
- internal workspace seams remain implementation detail rather than the visible
  benchmark contract
- Sprint 49 now closes with the reuse benchmarks proving the public lifecycle
  path that Day 5/6 exposed

### Direct Public-Handle Regression Coverage

`tests/test_iterative.c` now adds compact direct handle coverage for:

- explicit prepare-and-reuse for CG
- public GMRES handle validation
- zero-initialized on-demand handle growth for GMRES

`tests/test_eigs.c` now adds compact direct handle coverage for:

- explicit prepare-and-reuse for symmetric eigensolve
- public eigensolver handle validation
- zero-initialized on-demand handle growth

That is the right Day 10 test boundary:

- the public lifecycle contract is now pinned directly
- the batch avoids large refactors or duplicated family-level solver coverage
- one-shot regression coverage continues to protect the compatibility wrappers

### Validation

Because `*.c` changed, the required gate ran:

```bash
make format
make lint
make test
```

All passed.

Focused Day 10 follow-ons also passed:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Representative direct results:

- `test_iterative`: `78 / 78` passed
- `test_eigs`: `27 / 27` passed
- iterative repeated-run benchmark:
  - CG: `43.9660 ms` one-shot vs `46.9730 ms` handle reuse, `0.94x`
  - GMRES: `30.5820 ms` one-shot vs `28.4040 ms` handle reuse, `1.08x`
- eigensolver repeated-run benchmark:
  - grow-m: `2.1650 ms` one-shot vs `2.0980 ms` handle reuse, `1.03x`
  - thick-restart: `74.7310 ms` one-shot vs `78.1250 ms` handle reuse,
    `0.96x`

Behavior-level parity remained intact:

- iterative reuse cases matched one-shot iteration counts and residuals
- eigensolver reuse cases matched one-shot iterations, convergence,
  `n_converged`, residuals, and eigenvalues

### Day 10 Position

Sprint 49 now has the intended cross-surface compatibility state:

- public lifecycle headers are landed
- implementation and compatibility wrappers are landed
- migration guidance is documented
- reuse benchmarks now exercise the final public handle contract
- direct public-handle regression coverage is in place

Bottom line:

- Day 10 closed the strongest remaining implementation drift
- it also closed the strongest remaining public-handle regression gap
- the batch stayed small enough to preserve the Sprint 49 final-integration
  fence
