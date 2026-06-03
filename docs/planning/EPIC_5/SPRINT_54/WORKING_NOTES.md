# Sprint 54 Working Notes

## Day 1

**Objective:** Turn the Sprint 54 project-plan scope plus the Epic 4 public
handle closeout and the Sprint 53 validated direct-solver close state into a
concrete repeated-run solver-lifecycle starting point by confirming the
preserved reviewed baseline, naming the Sprint 54 implementation workstreams
explicitly, and defining the authoritative iterative/eigensolver header,
implementation, benchmark, regression, example, and documentation hotspots
before any solver-lifecycle expansion or exclusion decisions begin.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 54 project-plan source and the new sprint plan:
   - `sed -n '153,182p' docs/planning/EPIC_5/PROJECT_PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_54/PLAN.md`
3. Re-read the strongest inherited public-handle and repeated-run closeout
   sources:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_49/artifacts/day14-closeout-and-handoff.md`
   - `rg -n "Residual limits / follow-up journal|repeated-run|handle|MINRES|BiCGSTAB|LOBPCG" docs/planning/EPIC_4/EPIC_4_RETROSPECTIVE.md docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md docs/planning/EPIC_5/PROJECT_PLAN.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Measure the live iterative/eigensolver header, implementation, benchmark,
   regression, example, and caller-facing hotspot sizes:
   - `wc -l include/sparse_iterative.h include/sparse_eigs.h src/sparse_iterative.c src/sparse_eigs.c src/sparse_iterative_workspace_internal.c src/sparse_eigs_workspace_internal.c tests/test_iterative.c tests/test_eigs.c tests/test_eigs_lobpcg.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c examples/example_iterative.c examples/example_eigs.c README.md examples/README.md docs/maintainer_guide.md`
7. Reconfirm the live repeated-run public-handle and remaining-family
   references:
   - `rg -n "sparse_iter_handle|sparse_eigs_handle|MINRES|BiCGSTAB|LOBPCG|repeated-run|one-shot" include src tests benchmarks examples README.md docs/maintainer_guide.md`

### Day 1 Findings

#### 1. Sprint 54 starts from a real and validated public repeated-run baseline, not from a handle-invention sprint

The inherited starting state is already explicit and stable:

- Epic 4 already closed with bounded public repeated-run lifecycle handles for:
  - iterative solvers
  - eigensolvers
- Sprint 53 already closed from:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- the inherited caller-facing contract is already real:
  - one-shot iterative/eigensolver APIs remain first-class entry points
  - explicit repeated-run handles are opt-in lifecycle surfaces
  - handle reuse preserves allocation capacity/setup, not stale numerical
    Krylov / Ritz / search state

Interpretation:

- Sprint 54 is not a public-handle invention sprint
- Sprint 54 is not a baseline-repair sprint
- Sprint 54 is a steady-state support-boundary and final solver-lifecycle
  completion sprint

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible on all substantial solver-lifecycle batches

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

Interpretation:

- Sprint 54 should keep using the exact “strongest local reviewed baseline”
  phrasing
- substantial public repeated-run API batches should continue to treat the
  reviewed CMake count and parity contract as truthfulness anchors

#### 3. The real Sprint 54 queue is concentrated in support-boundary asymmetry, not generic solver reuse work

The Sprint 54 plan items and live repo state narrow to six bounded work
classes:

1. public solver lifecycle audit
2. inclusion/exclusion decision batch for remaining families
3. iterative handle expansion where justified
4. eigensolver lifecycle tightening
5. public reuse benchmark alignment
6. regression/example/docs adoption plus validation closeout

Interpretation:

- the main problem is now deciding what the repo should support publicly
  rather than discovering new internal reuse seams
- Sprint 54 should stay centered on support truthfulness and bounded
  implementation, not broad solver-API redesign

#### 4. The strongest architectural asymmetry is still on the iterative side, not the eigensolver side

The live public surfaces already show the asymmetry clearly:

- iterative handle support is explicit for:
  - CG
  - GMRES
- iterative one-shot-only public families still include:
  - MINRES
  - BiCGSTAB
  - selected block workflows
- the eigensolver side already has one bounded public repeated-run handle
  surface, while remaining drift is more likely to be:
  - support-boundary wording
  - example/benchmark/test agreement
  - advanced backend lifecycle explanation

Interpretation:

- the highest-value Day 3-6 work is likely centered on the iterative-family
  decision and expansion seams
- the eigensolver side is more likely to need tightening and caller-surface
  agreement than a large new handle expansion

#### 5. The live hotspot map is already concentrated enough to name directly

The main touched surfaces are clear before any new solver-lifecycle edits
begin:

- public headers:
  - `include/sparse_iterative.h` = `718`
  - `include/sparse_eigs.h` = `680`
- main implementations:
  - `src/sparse_iterative.c` = `2361`
  - `src/sparse_eigs.c` = `3233`
  - `src/sparse_iterative_workspace_internal.c` = `215`
  - `src/sparse_eigs_workspace_internal.c` = `267`
- strongest proof surfaces:
  - `tests/test_iterative.c` = `2865`
  - `tests/test_eigs.c` = `1329`
  - `tests/test_eigs_lobpcg.c` = `1196`
  - `benchmarks/bench_iterative_reuse.c` = `250`
  - `benchmarks/bench_eigs_reuse.c` = `202`
- strongest caller-facing adoption surfaces:
  - `examples/example_iterative.c` = `144`
  - `examples/example_eigs.c` = `285`
  - `README.md` = `972`
  - `examples/README.md` = `116`
  - `docs/maintainer_guide.md` = `294`

Interpretation:

- Sprint 54 is correctly centered on the big iterative/eigensolver public
  headers, source files, and test binaries
- the strongest proof concentration remains `test_iterative.c`,
  `test_eigs.c`, and the repeated-run benchmark drivers

#### 6. The inherited review queue already says the repeated-run public story is real but uneven

The Epic 5 review and the Sprint 54 plan already point to the same remaining
gap:

- public repeated-run support is real
- it is still uneven across solver families
- examples remain intentionally one-shot-first
- benchmark proof exists but remains concentrated in dedicated repeated-run
  drivers

Interpretation:

- Sprint 54 does not need to rediscover whether there is a repeated-run gap
- it needs to convert that already-known medium-severity gap into an explicit
  supported-vs-excluded public boundary plus matching proof surfaces

#### 7. The Sprint 54 workstreams are now explicit before code changes begin

The Day 1 implementation workstreams are:

1. repeated-run solver baseline and validation recheck
2. public solver lifecycle audit
3. support-boundary decision batch
4. bounded iterative handle expansion if justified
5. bounded eigensolver lifecycle tightening
6. benchmark/test/example/docs alignment
7. validation and closeout

Interpretation:

- the Sprint 54 queue is already narrowed to solver-lifecycle completion
  slices, not broad research
- the correct Day 1 close is a clean repeated-run solver baseline and
  authoritative-input package
