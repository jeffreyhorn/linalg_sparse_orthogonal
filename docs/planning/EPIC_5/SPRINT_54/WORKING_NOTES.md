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

## Day 2

**Objective:** Reconfirm the maintained reviewed baseline and truthfulness
anchors Sprint 54 must preserve, then define the smallest authoritative
validation boundary for the later iterative/eigensolver implementation days and
the high-signal rerun set those code-touch batches should use.

### Commands Run

1. Re-read the Sprint 54 Day 2 plan item and the current sprint notes:
   - `sed -n '71,120p' docs/planning/EPIC_5/SPRINT_54/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_54/WORKING_NOTES.md`
2. Reconfirm the maintained reviewed CMake truthfulness anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
3. Reconfirm the maintained reviewed wrapper authority surface:
   - `make -n quality-review-full`
4. Re-read the live quality-contract wording sources:
   - `rg -n "strongest local reviewed baseline|quality-review-full|quality-review-cmake|deadcode-check" README.md docs/maintainer_guide.md Makefile .github/workflows -g '!build'`
5. Reconfirm the main repeated-run iterative/eigensolver follow-on binaries
   already present in the build tree:
   - `ls build/test_iterative build/test_eigs build/test_eigs_lobpcg build/example_iterative build/example_eigs build/bench_iterative_reuse build/bench_eigs_reuse`
6. Reconfirm the remaining-family decision surfaces already present in the
   build tree:
   - `ls build/test_minres build/test_bicgstab build/bench_bicgstab build/example_ic_minres`
7. Measure the live size of those remaining-family proof/adoption surfaces:
   - `wc -l tests/test_minres.c tests/test_bicgstab.c benchmarks/bench_bicgstab.c examples/example_ic_minres.c`

### Day 2 Findings

#### 1. The strongest local reviewed baseline and truthfulness anchors remain exact

The maintained Sprint 54 baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The authority split is still the same:

- `make quality-review-full`
  - strongest local reviewed baseline
- `make quality-review`
  - reviewed Makefile path
- `make quality-review-cmake`
  - reviewed CMake parity path
- `make deadcode-check`
  - report-completeness gate, not a zero-findings gate

Interpretation:

- Sprint 54 should keep using the exact “strongest local reviewed baseline”
  phrasing
- the sprint should treat the reviewed CMake count and parity contract as
  truthfulness anchors rather than as loose guidance

#### 2. The later solver-lifecycle code-day gate is simple and should stay explicit

The mandatory gate for later `*.c` / `*.h` solver-lifecycle work remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial public repeated-run API or
solver-family integration batches remains:

- `make quality-review-full`

Interpretation:

- Sprint 54 does not need a custom validation contract
- it should inherit the same strong gate used on the earlier public lifecycle
  and direct-solver landing sprints

#### 3. The main repeated-run handle rerun set is already explicit and present in the build tree

The highest-signal rerun binaries for the already-supported public handle paths
are present:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/example_iterative`
- `./build/example_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Interpretation:

- Sprint 54 can directly validate the existing supported handle families
  without inventing new proof surfaces first
- these binaries are the right default reruns when a batch touches the current
  public handle paths or their docs/benchmarks

#### 4. The remaining-family decision surfaces should also be part of the authoritative rerun boundary

The families most likely to be included, excluded, or clarified during Sprint
54 already have their own live proof/adoption surfaces:

- `./build/test_minres`
- `./build/test_bicgstab`
- `./build/bench_bicgstab`
- `./build/example_ic_minres`

Their current file sizes confirm they are substantial enough to matter during
support-boundary decisions:

- `tests/test_minres.c` = `1588`
- `tests/test_bicgstab.c` = `1586`
- `benchmarks/bench_bicgstab.c` = `173`
- `examples/example_ic_minres.c` = `232`

Interpretation:

- Sprint 54 should not treat MINRES and BiCGSTAB as purely abstract design
  questions
- if Day 4-6 batches touch their public lifecycle story, these binaries should
  be part of the targeted rerun set

#### 5. The authoritative Sprint 54 rerun list is now cleanly split by purpose

The Day 2 authoritative rerun boundary is:

- reviewed baseline / truthfulness anchors:
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
- default code-day gate for `*.c` / `*.h`:
  - `make format`
  - `make lint`
  - `make test`
- public handle path follow-ons:
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
  - `./build/example_iterative`
  - `./build/example_eigs`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`
- remaining-family decision follow-ons:
  - `./build/test_minres`
  - `./build/test_bicgstab`
  - `./build/bench_bicgstab`
  - `./build/example_ic_minres`

Interpretation:

- Sprint 54 now has one small authoritative rerun set for both supported
  handle families and the families whose steady-state support status is still
  being decided
- no validation ambiguity remains around later decision, expansion, or
  tightening days

## Day 3

**Objective:** Audit the live public repeated-run iterative and eigensolver
surfaces so Sprint 54 can separate already-supported handle paths from
families that still remain one-shot-only or internal-only, then reduce the
remaining repeated-run problem to a ranked set of explicit support-boundary
and implementation seams before Day 4 decisions begin.

### Commands Run

1. Re-read the Sprint 54 Day 3 plan item and the current sprint notes:
   - `sed -n '121,180p' docs/planning/EPIC_5/SPRINT_54/PLAN.md`
   - `sed -n '1,420p' docs/planning/EPIC_5/SPRINT_54/WORKING_NOTES.md`
2. Re-read the public repeated-run iterative handle contract:
   - `sed -n '210,620p' include/sparse_iterative.h`
3. Re-read the public repeated-run eigensolver handle contract:
   - `sed -n '534,720p' include/sparse_eigs.h`
4. Reconfirm the live public repeated-run entry points and one-shot-only
   remaining-family surfaces:
   - `rg -n "sparse_iter_handle_prepare|with_handle|sparse_solve_minres|sparse_solve_bicgstab|sparse_minres_solve_block|sparse_bicgstab_solve_block|sparse_eigs_handle_prepare|sparse_eigs_sym_with_handle" src include tests benchmarks examples README.md docs/maintainer_guide.md`
5. Re-scan the current benchmark/example/docs surfaces for support-boundary
   drift:
   - `rg -n "bench_iterative_reuse|bench_eigs_reuse|example_iterative|example_eigs|example_ic_minres|MINRES|BiCGSTAB|LOBPCG|repeated-run|one-shot" benchmarks examples README.md docs/maintainer_guide.md`
6. Re-read the main public repeated-run benchmark and example surfaces:
   - `sed -n '1,180p' benchmarks/bench_iterative_reuse.c`
   - `sed -n '1,220p' benchmarks/bench_eigs_reuse.c`
   - `sed -n '1,140p' examples/README.md`
   - `sed -n '430,470p' README.md`
   - `sed -n '1,180p' examples/example_iterative.c`
   - `sed -n '1,220p' examples/example_eigs.c`
   - `sed -n '1,220p' examples/example_ic_minres.c`
7. Audit the internal reusable-workspace boundary behind the remaining
   iterative families:
   - `rg -n "prepare_minres|prepare_bicgstab|block_cg|block_gmres|block_minres|block_bicgstab|workspace_prepare" src/sparse_iterative_workspace_internal.c src/sparse_iterative.c src/sparse_iterative_workspace_internal.h`
   - `sed -n '1,180p' src/sparse_iterative_workspace_internal.h`
   - `sed -n '160,320p' src/sparse_iterative_workspace_internal.c`
   - `sed -n '1360,2140p' src/sparse_iterative.c`

### Day 3 Findings

#### 1. The supported public repeated-run surface is explicit and still intentionally narrow

The live public handle support is concrete and bounded:

- iterative public handle support exists for:
  - CG
  - GMRES
- eigensolver public handle support exists for:
  - symmetric eigensolves through the shared `sparse_eigs_sym_with_handle(...)`
    surface
- the public docs still describe one-shot APIs as first-class and the handle
  paths as opt-in repeated-run surfaces

Interpretation:

- Sprint 54 is not filling a “no public repeated-run support” gap
- it is deciding whether to keep this support set narrow or extend it to a few
  remaining families

#### 2. MINRES is the strongest remaining iterative candidate because it already has an internal reusable-workspace seam

The iterative implementation audit shows:

- `sparse_iter_workspace_prepare_minres(...)` already exists in the internal
  workspace layer
- MINRES already sits closer to the CG/GMRES reusable-workspace model than the
  public headers suggest
- MINRES still has no public handle prepare/run entry points

Interpretation:

- the strongest “public repeated-run asymmetry” is not that MINRES lacks any
  reusable seam
- it is that MINRES reusable workspace is still internal-only even though the
  public handle model already exists for closely related iterative families

#### 3. BiCGSTAB is a different class of gap from MINRES

The BiCGSTAB path is still more isolated:

- BiCGSTAB uses its own `bicgstab_workspace_t` allocation/free path
- it does not plug into the public iterative handle owner or the shared
  `sparse_iter_workspace_t` prepare helpers the way CG/GMRES/MINRES do
- public surfaces already include:
  - scalar BiCGSTAB
  - block BiCGSTAB
  - matrix-free BiCGSTAB

Interpretation:

- BiCGSTAB is a real public repeated-run asymmetry
- but it is also a more implementation-heavy target than MINRES because the
  reusable seam is not already aligned with the existing public handle owner

#### 4. Block iterative workflows are better understood as compatibility surfaces than first public handle targets

The current block APIs remain one-shot or per-column wrappers:

- block CG has a reusable internal workspace view, but the public repeated-run
  story still centers on scalar handles
- block GMRES, block MINRES, and block BiCGSTAB are currently independent or
  per-column compatibility surfaces, not explicit handle-based lifecycle paths
- the current examples and README do not frame block workflows as the main
  repeated-run public story

Interpretation:

- block iterative workflows are poor first candidates for Sprint 54 public
  handle expansion
- they are stronger candidates for explicit bounded exclusion unless a clear
  user-facing lifecycle case emerges on Day 4

#### 5. The eigensolver side is structurally closer to “done” than the iterative side

The eigensolver repeated-run surface is already comparatively coherent:

- one public handle surface exists for symmetric eigensolves
- that surface already fronts the main public `sparse_eigs_sym(...)` entry
- the public handle contract is generic enough to cover backend-shaped working
  sets through `sparse_eigs_handle_prepare(...)`
- the main proof surfaces in `bench_eigs_reuse.c` currently cover:
  - grow-m Lanczos
  - thick-restart Lanczos

Interpretation:

- the main eigensolver gap is more likely:
  - proof drift
  - example/docs drift
  - LOBPCG repeated-run caller-surface underrepresentation
- it is less likely to require a broad new public API shape

#### 6. The examples and benchmarks still underrepresent the full repeated-run public story

The caller-facing surfaces remain intentionally one-shot-first:

- `examples/README.md` explicitly says the shipped examples still lean on the
  one-shot public APIs
- `example_iterative.c` is still a GMRES one-shot/preconditioner demo
- `example_eigs.c` is still a one-shot eigensolver demo, including explicit
  LOBPCG backend usage without the public handle path
- `example_ic_minres.c` is a one-shot MINRES/block-MINRES teaching surface
- repeated-run public-handle benchmarks remain narrow:
  - `bench_iterative_reuse.c` only proves CG and GMRES
  - `bench_eigs_reuse.c` only proves grow-m and thick-restart Lanczos

Interpretation:

- the repeated-run public story is real but still concentrated in dedicated
  benchmark drivers and README bullets
- examples and benchmarks currently do not make a strong case that MINRES,
  BiCGSTAB, or LOBPCG belong on the same steady-state public repeated-run
  support tier as the currently handled families

#### 7. The remaining Sprint 54 problem now reduces to five seam classes instead of a generic “finish the remaining families” bucket

The audited queue now reduces cleanly to:

1. public iterative-handle support asymmetry:
   - CG/GMRES supported
   - MINRES internal seam exists but is public-surface missing
   - BiCGSTAB remains both public-surface and implementation-shape asymmetric
2. block-workflow support-boundary ambiguity
3. eigensolver repeated-run proof/example drift, especially around LOBPCG
4. repeated-run benchmark support-set drift
5. example/README support-boundary drift

Interpretation:

- Sprint 54 can start Day 4 from a small ranked seam list
- the remaining work is materially more concrete than the raw project-plan
  placeholder

#### 8. The ranked Day 4 target list is now explicit

The highest-value Day 4 decisions should center on:

1. MINRES: strongest candidate for public repeated-run inclusion
2. BiCGSTAB: explicit inclusion vs explicit bounded exclusion
3. block iterative workflows: likely bounded exclusion unless a compelling
   repeated-run caller story exists
4. eigensolver tightening: likely keep one public handle surface, but expand
   proof/docs if needed rather than inventing a new API family
5. examples/benchmarks: treat as support-surface alignment work after the
   inclusion/exclusion boundary is fixed

Interpretation:

- Day 4 should decide the support boundary first
- implementation should follow only after these ranked decisions are explicit
