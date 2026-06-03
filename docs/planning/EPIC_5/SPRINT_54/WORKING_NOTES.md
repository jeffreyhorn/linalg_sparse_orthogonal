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

## Day 4

**Objective:** Convert the Day 3 audit into an explicit steady-state public
repeated-run support boundary so Sprint 54 can implement only the highest-value
in-scope lifecycle work and treat the rest as conscious compatibility
boundaries rather than accidental omissions.

### Commands Run

1. Re-read the Sprint 54 Day 4 plan item and the current sprint notes:
   - `sed -n '150,235p' docs/planning/EPIC_5/SPRINT_54/PLAN.md`
   - `sed -n '1,520p' docs/planning/EPIC_5/SPRINT_54/WORKING_NOTES.md`
2. Re-read the Day 3 audit artifact:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_54/artifacts/day3-public-solver-lifecycle-audit.md`
3. Re-read the Epic 5 review section that framed the repeated-run gap:
   - `sed -n '170,230p' docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
4. Re-scan the current public docs and headers where the support boundary will
   need to remain truthful:
   - `rg -n "MINRES|BiCGSTAB|block|LOBPCG|repeated-run handle|public handle|one-shot" README.md examples/README.md include/sparse_iterative.h include/sparse_eigs.h docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`

### Day 4 Findings

#### 1. Sprint 54 should include MINRES in the public repeated-run support set

MINRES is the strongest in-scope inclusion target because:

- it already has an internal reusable-workspace seam:
  - `sparse_iter_workspace_prepare_minres(...)`
- it is symmetric with the current CG/GMRES public-handle story in caller
  shape:
  - scalar iterative solve
  - stable-dimension repeated workloads
  - clear prepare/run/free lifecycle
- it carries strong user value:
  - symmetric indefinite systems are already a documented first-class solver
    case
  - the repo already has large dedicated MINRES regression and example
    surfaces

Interpretation:

- MINRES is the one remaining iterative family whose public-handle omission now
  looks more accidental than intentional
- Sprint 54 should treat MINRES repeated-run exposure as in-scope

#### 2. Sprint 54 should exclude BiCGSTAB from public repeated-run handle exposure

BiCGSTAB should remain outside the Sprint 54 repeated-run handle expansion
boundary because:

- its reusable seam is still implementation-shaped around a separate
  `bicgstab_workspace_t` path instead of the existing public iterative handle
  owner
- its public surface is already broad:
  - scalar
  - block
  - matrix-free
- adding a public handle for BiCGSTAB would therefore be a larger design and
  proof commitment than the MINRES case
- the current review finding only requires that the support boundary become
  obviously intentional, not that every solver family be made uniform

Interpretation:

- BiCGSTAB is an explicit bounded exclusion for Sprint 54
- it should remain a one-shot-first compatibility surface rather than a hidden
  “not yet wired” handle omission

#### 3. Selected block iterative workflows should remain excluded from public repeated-run handle exposure

Block iterative workflows should remain outside the Sprint 54 handle expansion
boundary because:

- their current public story is compatibility-first, not lifecycle-first
- they span multiple algorithm shapes:
  - shared block CG
  - per-column GMRES
  - per-column MINRES
  - per-column BiCGSTAB
- adding a coherent public block-handle story would broaden Sprint 54 into a
  separate API-design sprint

Interpretation:

- Sprint 54 should preserve block workflows as supported one-shot compatibility
  surfaces
- they should be explicitly excluded from the public repeated-run handle
  support set rather than implicitly deferred

#### 4. The eigensolver side should stay on one public repeated-run handle surface, with LOBPCG alignment in-scope

The eigensolver decision is narrower:

- keep the single public repeated-run eigensolver handle surface
- do not invent a second backend-specific public API family
- treat LOBPCG repeated-run alignment as in-scope for:
  - proof
  - docs/examples
  - benchmark coverage
- treat backend-specific lifecycle surfacing beyond that as out of scope

Interpretation:

- Sprint 54 should tighten the existing public eigensolver handle story
- it should not fragment the public API by backend

#### 5. The final Sprint 54 support boundary is now explicit

The steady-state public repeated-run support boundary for Sprint 54 is:

- supported public repeated-run iterative handles:
  - CG
  - GMRES
  - MINRES
- supported public repeated-run eigensolver handles:
  - symmetric eigensolves through the existing `sparse_eigs_handle_t` surface
  - including better LOBPCG alignment at the proof/docs layer if needed
- intentionally excluded from public repeated-run handle exposure in Sprint 54:
  - BiCGSTAB
  - block iterative workflows
  - backend-specific eigensolver public API families

Interpretation:

- Sprint 54 now has a concrete “include vs exclude” line instead of a fuzzy
  completeness goal
- the support boundary is now small enough to implement without reopening the
  broader handle model

#### 6. The implementation order is now fixed from the chosen boundary

The correct Sprint 54 landing order is now:

1. MINRES public-handle exposure
2. iterative regression proof for the final supported iterative handle set
3. eigensolver lifecycle/proof/docs tightening, especially around LOBPCG
4. repeated-run benchmark alignment to the final supported set
5. example/README adoption and explicit exclusion wording
6. final validation and closeout

Interpretation:

- implementation should start with the only newly included iterative family
- exclusion wording and benchmark/example alignment should follow the code
  boundary, not precede it

#### 7. Sprint 54 is now materially smaller and clearer than the raw placeholder

Day 4 cuts the raw placeholder down to one real public iterative expansion
target plus bounded alignment work:

- one new iterative-handle family in-scope:
  - MINRES
- one iterative family explicitly out of scope for handle exposure:
  - BiCGSTAB
- block workflows explicitly out of scope
- eigensolver work narrowed to support-tightening rather than API expansion

Interpretation:

- Sprint 54 now has a credible bounded implementation plan
- the remaining queue is small enough to execute without pretending all solver
  families should or will become uniform immediately

## Sprint 54 Day 5 - iterative handle expansion batch 1

Date: 2026-06-03
Commit intent: expose MINRES on the existing public iterative repeated-run
handle surface without broadening Sprint 54 beyond the Day 4 boundary.

### What changed

- Added public iterative-handle surface for MINRES in
  `include/sparse_iterative.h`:
  - `sparse_iter_handle_prepare_minres(...)`
  - `sparse_solve_minres_with_handle(...)`
- Refactored MINRES implementation in `src/sparse_iterative.c` around a shared
  internal workspace-backed execution seam so:
  - one-shot `sparse_solve_minres(...)` remains first-class
  - explicit repeated-run callers can reuse the existing
    `sparse_iter_handle_t` owner
  - zero-init handle growth still works on demand
- Added direct regression proof in `tests/test_iterative.c` covering:
  - null-handle / null-prepare validation
  - explicit prepare + repeated reuse
  - zero-init on-demand growth
  - repeated-run parity on solution and iteration counts

### Why this stayed inside the Sprint 54 fence

- MINRES was the only new iterative public-handle family included by Day 4.
- The landing reused the existing public iterative-handle model rather than
  inventing a new owner or changing the one-shot API shape.
- BiCGSTAB and block iterative workflows remained untouched and out of scope.

### Validation

Required gates:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 123.22 sec`

Focused Day 5 follow-ons:

- `./build/test_iterative` -> `79 / 79`
- `./build/test_minres` -> `43 / 43`
- `./build/example_ic_minres`
- `./build/bench_iterative_reuse`

Representative direct results:

- `example_ic_minres`:
  - MINRES on the `42x42` KKT system converged in `39` iterations
  - Jacobi-preconditioned MINRES converged in `26` iterations
- `bench_iterative_reuse`:
  - `cg-tridiag-300`: `1.00x`
  - `gmres-unsym-220`: `1.05x`

### Day 5 outcome

Sprint 54 now has a coherent supported public repeated-run iterative-handle
set for:

- `CG`
- `GMRES`
- `MINRES`

The remaining Sprint 54 queue is now smaller and cleaner:

- iterative proof/alignment for the final supported set
- eigensolver lifecycle/proof/docs tightening, especially around LOBPCG
- benchmark/example/README adoption and explicit exclusion wording

## Sprint 54 Day 6 - iterative contract tightening batch

Date: 2026-06-03
Commit intent: tighten the supported public iterative repeated-run handle
contract around validation, explicit prepare/reuse, and underprepared
on-demand growth without broadening Sprint 54 beyond the Day 4 boundary.

### What changed

- Tightened the top-level iterative handle wording in
  `include/sparse_iterative.h` so the public repeated-run contract now reads
  coherently with the Day 5 MINRES landing instead of still sounding like a
  CG/GMRES-only handle surface.
- Expanded `tests/test_iterative.c` so the supported iterative handle set now
  has more symmetric direct public proof:
  - `CG`:
    - null prepare validation
    - null handle solve validation
    - explicit prepare + repeated reuse
    - zero-init on-demand growth
  - `GMRES`:
    - null prepare validation
    - null handle solve validation
    - explicit prepare + repeated reuse
    - same-handle growth from a smaller prepared dimension/restart to a later
      larger solve
    - zero-init on-demand growth
  - `MINRES`:
    - preserved null validation and explicit prepare + reuse
    - same-handle growth from a smaller prepared dimension to a later larger
      solve
    - zero-init on-demand growth

### Why this stayed inside the Sprint 54 fence

- No new public solver family was added beyond the Day 5 MINRES inclusion.
- `BiCGSTAB` remained outside the public repeated-run handle support set.
- Block iterative workflows remained untouched and excluded from handle
  exposure.
- The patch tightened proof and support symmetry for the already supported
  iterative-handle families rather than reopening solver-API design.

### Validation

Required Day 6 gates:

- `make format`
- `make lint`
- `make test`

All passed.

Focused Day 6 follow-ons:

- `./build/test_iterative` -> `79 / 79`
- `./build/test_minres` -> `43 / 43`
- `./build/example_ic_minres`
- `./build/bench_iterative_reuse`

Representative direct results:

- `test_iterative` now directly passes the strengthened public-handle proofs:
  - `test_cg_public_handle_validation_reuse_and_on_demand`
  - `test_gmres_public_handle_prepare_reuse_and_growth`
  - `test_minres_public_handle_prepare_reuse_and_growth`
- `example_ic_minres` stayed stable:
  - MINRES on the `42x42` KKT system converged in `39` iterations
  - Jacobi-preconditioned MINRES converged in `26` iterations
- `bench_iterative_reuse` stayed aligned with the supported public-handle
  benchmark surface:
  - `cg-tridiag-300`: `1.12x`
  - `gmres-unsym-220`: `1.05x`

### Day 6 outcome

Sprint 54’s supported iterative repeated-run handle set is now stronger at the
contract/proof layer, not just at the API listing layer:

- `CG`
- `GMRES`
- `MINRES`

The remaining queue can now move on from iterative-handle proof symmetry to:

- eigensolver lifecycle/proof/docs tightening, especially around LOBPCG
- later benchmark/example/README adoption for the final support boundary

## Sprint 54 Day 7 - eigensolver lifecycle tightening batch

Date: 2026-06-03
Commit intent: tighten the supported public repeated-run eigensolver handle
contract and direct proof surface without expanding Sprint 54 into new
backend-specific public API families.

### What changed

- Tightened `include/sparse_eigs.h` so the public repeated-run eigensolver
  handle contract now says the real supported backend set explicitly:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit LOBPCG
- The header now also states the intended public boundary more directly:
  - repeated-run prepare/run/free still lives on one handle surface
  - Sprint 54 does not introduce backend-specific public handle types
  - explicit LOBPCG still uses the same public repeated-run lifecycle path
- Expanded `tests/test_eigs.c` with:
  - `test_public_handle_lobpcg_prepare_reuse_and_growth`
- The new regression proves the supported LOBPCG repeated-run handle path
  directly:
  - explicit `SPARSE_EIGS_BACKEND_LOBPCG`
  - explicit prepare on a smaller problem
  - repeated reuse on the same prepared shape
  - later on-demand growth to a larger problem and larger `k`
  - preserved `backend_used == SPARSE_EIGS_BACKEND_LOBPCG`

### Why this stayed inside the Sprint 54 fence

- No new eigensolver API family was added.
- No backend-specific public handle type was introduced.
- No new benchmark mode or example workflow was required to land the core Day 7
  contract/proof tightening.
- The batch preserved the Day 4 support boundary:
  - symmetric eigensolver repeated-run handle surface remains the single public
    lifecycle path
  - Sprint 54 keeps focusing on lifecycle/proof/docs tightening rather than
    broad eigensolver API expansion

### Validation

Required Day 7 gates:

- `make format`
- `make lint`
- `make test`

Focused Day 7 follow-ons:

- `./build/test_eigs` -> `28 / 28`
- `./build/test_eigs_lobpcg` -> `26 / 26`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`

Representative direct results:

- `test_eigs` now directly passes the new public repeated-run LOBPCG proof:
  - `test_public_handle_lobpcg_prepare_reuse_and_growth`
- `example_eigs` stayed stable on the explicit LOBPCG path:
  - `bcsstk04` converged `3 / 3` smallest eigenpairs in `62` outer iterations
  - `backend_used = LOBPCG`
  - `reported residual_norm = 8.808e-09`
- `bench_eigs_reuse` preserved repeated-run parity on the supported public
  handle path:
  - `growm-nos4-k5`: `0.96x`
  - `thick-bcsstk14-k5`: `1.04x`
  - both repeated-run cases kept exact eigenvalue parity with the one-shot path

### Day 7 outcome

Sprint 54’s supported public repeated-run eigensolver handle story is now
clearer and better proved:

- one public eigensolver handle surface
- explicit support for grow-m Lanczos, thick-restart Lanczos, and explicit
  LOBPCG through that surface
- direct repeated-run proof for the LOBPCG branch, not just generic handle
  coverage

The remaining queue can now move on from eigensolver lifecycle/proof tightening
to:

- repeated-run benchmark support-set alignment
- example/README adoption and explicit support-boundary wording

## Sprint 54 Day 8 - public reuse benchmark alignment audit

Date: 2026-06-03
Commit intent: audit the repeated-run benchmark surfaces against the final
Sprint 54 public support boundary before changing benchmark drivers, so Day 9
can land only the smallest benchmark-alignment batch instead of reopening
framework work.

### Commands run

1. Re-read the Day 8 plan target:
   - `sed -n '296,328p' docs/planning/EPIC_5/SPRINT_54/PLAN.md`
2. Re-read the current Sprint 54 close state around Days 6-7:
   - `tail -n 140 docs/planning/EPIC_5/SPRINT_54/WORKING_NOTES.md`
3. Audit the live iterative reuse benchmark:
   - `sed -n '1,260p' benchmarks/bench_iterative_reuse.c`
4. Audit the live eigensolver reuse benchmark:
   - `sed -n '1,260p' benchmarks/bench_eigs_reuse.c`
5. Re-read the benchmark-local docs:
   - `sed -n '1,220p' benchmarks/README.md`
6. Re-read the caller-facing repeated-run support wording for later audit
   cross-check:
   - `sed -n '240,310p' README.md`
   - `sed -n '340,365p' README.md`
   - `sed -n '1,180p' examples/README.md`
7. Cross-check benchmark, README, and examples wording:
   - `rg -n "MINRES|GMRES|CG|LOBPCG|thick-restart|grow-m|public handle|repeated-run|reuse" benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c benchmarks/README.md README.md examples/README.md`

### Day 8 findings

#### 1. The reuse benchmarks already prove public handle paths, not internal-only seams

The strongest Day 8 positive result is that neither benchmark is stale in the
worst way:

- `bench_iterative_reuse.c`
  - uses `sparse_iter_handle_prepare_cg(...)`
  - uses `sparse_iter_handle_prepare_gmres(...)`
  - compares one-shot paths against `*_with_handle(...)`
- `bench_eigs_reuse.c`
  - uses `sparse_eigs_handle_prepare(...)`
  - uses `sparse_eigs_sym_with_handle(...)`
  - keeps direct parity checks between one-shot and repeated-run results

Interpretation:

- Day 8 did not uncover any reuse benchmark that is still proving only an
  internal workspace seam
- the benchmark drift is about support-set completeness, not public-contract
  dishonesty

#### 2. The highest-value iterative benchmark drift is now explicit: MINRES is supported publicly but missing from `bench_iterative_reuse`

The final supported iterative repeated-run set after Day 6 is:

- `CG`
- `GMRES`
- `MINRES`

But `bench_iterative_reuse.c` still only measures:

- `CG`
- `GMRES`

Interpretation:

- this is now a real public-support-set drift, not a theoretical future idea
- Day 9 should add one bounded `MINRES` repeated-run case to the existing reuse
  driver instead of inventing a separate benchmark framework

#### 3. The strongest eigensolver benchmark drift is also concrete: `bench_eigs_reuse` still lacks an explicit LOBPCG repeated-run case

After Day 7, the public eigensolver repeated-run contract explicitly covers:

- grow-m Lanczos
- thick-restart Lanczos
- explicit LOBPCG

But `bench_eigs_reuse.c` still proves only:

- grow-m Lanczos on `nos4`
- thick-restart Lanczos on `bcsstk14`

Interpretation:

- the eigensolver benchmark surface now under-covers the final public support
  set
- the best Day 9 fix is a bounded explicit LOBPCG repeated-run case rather
  than a broad expansion of `bench_eigs`
- the likely stable shape is one explicit LOBPCG case with fixed backend and
  fixed option shape, not a new general-purpose benchmark matrix

#### 4. `benchmarks/README.md` understates the public repeated-run benchmark proof surface

The benchmark README currently documents:

- `bench_main`
- `bench_scaling`
- `bench_fillin`
- `bench_convergence`
- `bench_svd`
- `bench_refactor`
- `bench_refactor_csc`
- `bench_colamd`
- `bench_bicgstab`
- `bench_chol_csc`
- `bench_ldlt_csc`
- `bench_eigs`

But it does not currently name the two dedicated public repeated-run benchmark
drivers:

- `bench_iterative_reuse`
- `bench_eigs_reuse`

Interpretation:

- the benchmark-local docs currently underrepresent the public repeated-run
  benchmark surfaces that Sprint 54 is explicitly relying on
- this is a small documentation sync target for Day 9, not a Day 8 code issue

#### 5. The correct non-goal boundary is now explicit

Day 8 also made the benchmark non-goal line sharper:

- do not add `BiCGSTAB` repeated-run-handle benchmarking
  - because `BiCGSTAB` stays outside the Sprint 54 public handle boundary
- do not add block iterative reuse benchmarks
  - because block workflows remain compatibility surfaces, not first-class
    repeated-run public handle surfaces
- do not turn `bench_eigs_reuse` into a full backends/preconditioners sweep
  - that job already belongs to `bench_eigs`
- do not redesign the benchmark framework or CLI just to close the Sprint 54
  support-set gap

Interpretation:

- Day 9 should update benchmark proof surfaces, not benchmark architecture

### Ranked Day 9 target list

1. `bench_iterative_reuse.c`
   - add one bounded repeated-run `MINRES` case so the iterative reuse
     benchmark matches the final supported iterative handle set
2. `bench_eigs_reuse.c`
   - add one bounded explicit LOBPCG repeated-run case so the eigensolver
     reuse benchmark matches the final supported handle set
3. `benchmarks/README.md`
   - document `bench_iterative_reuse` and `bench_eigs_reuse`
   - state their intentionally narrow public-handle proof role

### Day 8 outcome

Sprint 54’s benchmark queue is now materially smaller and clearer:

- no benchmark is still proving only an internal reuse seam
- the remaining drift is a narrow support-set completeness problem
- the smallest alignment batch is now explicit:
  - `MINRES` on the iterative reuse side
  - explicit `LOBPCG` on the eigensolver reuse side
  - small benchmark README synchronization after that

## Sprint 54 Day 9 - public reuse benchmark alignment batch

Date: 2026-06-03
Commit intent: align the repeated-run benchmark proof surfaces with the final
Sprint 54 supported public solver-lifecycle set without expanding benchmark
scope beyond the Day 8 fence.

### What changed

- Expanded `benchmarks/bench_iterative_reuse.c` so the iterative public-handle
  reuse benchmark now covers:
  - `CG`
  - `GMRES`
  - `MINRES`
- Added a bounded generated symmetric-indefinite KKT fixture for the new
  `MINRES` repeated-run case:
  - `42x42`
  - same one-shot vs explicit public-handle comparison shape as the existing
    iterative reuse cases
- Expanded `benchmarks/bench_eigs_reuse.c` so the eigensolver public-handle
  reuse benchmark now covers:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- Refactored the eigensolver reuse benchmark just enough to support both:
  - file-backed cases
  - one bounded generated diagonal SPD case for explicit `LOBPCG`
- Updated `benchmarks/README.md` so the benchmark-local docs now name:
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- The README batch also states their intended narrow proof role explicitly:
  - public-handle-path proof for the supported reuse set
  - not a claim of public repeated-run-handle support for `BiCGSTAB`
  - not a broad replacement for `bench_eigs`

### Why this stayed inside the Sprint 54 fence

- No new public solver family was exposed.
- `BiCGSTAB` remained outside the public repeated-run handle support boundary.
- Block iterative workflows remained outside the repeated-run handle target set.
- No benchmark-framework or CLI redesign was introduced.
- The batch only closed the support-set completeness gap identified on Day 8:
  - `MINRES` on the iterative side
  - explicit `LOBPCG` on the eigensolver side

### Validation

Required Day 9 gates:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full` was not required by the Sprint 54 Day 9 landing
  contract, because this batch only touched benchmark sources and benchmark
  docs

All required gates passed.

Focused Day 9 follow-ons:

- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`
- `./build/example_ic_minres`
- `./build/example_eigs`

Representative direct results:

- `bench_iterative_reuse` now matches the final supported iterative public
  handle set:
  - `cg-tridiag-300`: `1.00x`
  - `gmres-unsym-220`: `1.05x`
  - `minres-kkt-42`: `1.01x`
  - the new `MINRES` repeated-run case kept exact iteration/residual parity:
    - one-shot: `39` iterations, `3.870e-11`
    - reuse: `39` iterations, `3.870e-11`
- `bench_eigs_reuse` now matches the final supported eigensolver public handle
  set:
  - `growm-nos4-k5`: `1.00x`
  - `thick-bcsstk14-k5`: `0.98x`
  - `lobpcg-diag40-k3`: `1.00x`
  - the new explicit `LOBPCG` repeated-run case kept exact eigenvalue parity:
    - `|lambda|max diff = 0.000e+00`
    - one-shot / reuse both: `45` iterations, `6.696e-11`
- `example_ic_minres` stayed stable on the bounded `MINRES` teaching path:
  - `MINRES`: `39` iterations, `3.87e-11`
  - `Jacobi-MINRES`: `26` iterations, `4.16e-11`
- `example_eigs` stayed stable on the explicit `LOBPCG` teaching path:
  - `bcsstk04`: `3 / 3` smallest eigenpairs
  - `62` outer iterations
  - `residual_norm = 8.808e-09`

### Day 9 outcome

Sprint 54’s benchmark proof surfaces now match the final supported repeated-run
solver lifecycle set instead of lagging it:

- iterative public reuse benchmark set:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver public reuse benchmark set:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`

The remaining queue can now move on from benchmark support-set completeness to:

- example/README support-boundary adoption
- final validation and closeout

## Sprint 54 Day 10 - regression and example adoption batch I

Date: 2026-06-03
Commit intent: tighten the highest-value direct proof and user-facing wording
for the final repeated-run solver boundary without reopening broader tutorial,
example-corpus, or solver-family scope.

### What changed

- Expanded `tests/test_eigs.c` with:
  - `test_public_handle_thick_restart_prepare_reuse_and_growth`
- The new regression proves the last still-implicit supported eigensolver
  repeated-run branch directly:
  - explicit `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`
  - explicit prepare on a smaller problem
  - repeated reuse on the same prepared shape
  - later on-demand growth to a larger problem and larger `k`
  - preserved `backend_used == SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`
- Updated the top-level `README.md` repeated-run sections so they now state the
  final supported iterative-handle set explicitly:
  - `CG`
  - `GMRES`
  - `MINRES`
- The README batch also makes the bounded exclusions explicit:
  - `BiCGSTAB` remains outside the public repeated-run handle surface
  - block iterative workflows remain compatibility surfaces
- Updated the `README.md` API summary surfaces so they match the landed state:
  - `sparse_iterative.h` summary row
  - repeated-run handle list
  - key-functions iterative section
- Updated `examples/README.md` so the shipped-example contract now matches the
  final support boundary:
  - examples remain intentionally one-shot-first
  - iterative handle support is named explicitly
  - eigensolver handle support is named explicitly
  - exclusions for `BiCGSTAB` and block iterative handles are stated directly

### Why this stayed inside the Sprint 54 fence

- No new public API family was added.
- No example source was broadened into a dedicated public-handle demo.
- `BiCGSTAB` stayed out of scope for public repeated-run handles.
- Block iterative workflows stayed out of scope for public repeated-run
  handles.
- The batch only closed the highest-value remaining proof/docs adoption drift:
  - explicit thick-restart public-handle proof
  - strongest README/examples support-boundary wording

### Validation

Required Day 10 gates:

- `make format`
- `make lint`
- `make test`

All passed.

Focused Day 10 follow-ons:

- `./build/test_eigs` -> `29 / 29`
- `./build/test_eigs_lobpcg` -> `26 / 26`
- `./build/example_eigs`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/bench_eigs_reuse`

Representative direct results:

- `test_eigs` now directly passes the full bounded public repeated-run backend
  proof set:
  - `test_public_handle_prepare_and_reuse`
  - `test_public_handle_thick_restart_prepare_reuse_and_growth`
  - `test_public_handle_lobpcg_prepare_reuse_and_growth`
- `bench_eigs_reuse` stayed aligned with the final supported eigensolver
  handle set:
  - `growm-nos4-k5`: `1.00x`
  - `thick-bcsstk14-k5`: `1.05x`
  - `lobpcg-diag40-k3`: `1.05x`
  - all three kept exact eigenvalue parity:
    - `|lambda|max diff = 0.000e+00`
- `example_eigs` stayed stable on the explicit LOBPCG teaching path:
  - `bcsstk04`: `3 / 3` smallest eigenpairs
  - `62` outer iterations
  - `backend_used = LOBPCG`
  - `residual_norm = 8.808e-09`
- `example_iterative` remained the intended one-shot teaching surface:
  - GMRES: `25` iterations, `9.56e-11`
  - ILU(0)-GMRES: `9` iterations, `3.14e-11`
- `example_ic_minres` remained the bounded MINRES teaching surface:
  - `MINRES`: `39` iterations, `3.87e-11`
  - `Jacobi-MINRES`: `26` iterations, `4.16e-11`

### Day 10 outcome

Sprint 54’s highest-value public solver-lifecycle surfaces now have both:

- direct proof for the full intended eigensolver handle backend set
- user-facing wording that matches the final support boundary

The remaining queue can now move on from first adoption/proof cleanup to:

- any final residual proof/docs sweep
- compatibility audit
- validation closeout

## Sprint 54 Day 11 - regression and example adoption batch II

Date: 2026-06-03
Commit intent: close the last high-value explicit repeated-run proof gap and
remove the last stale high-signal wording seams before the compatibility audit.

### What changed

- Expanded `tests/test_eigs.c` with:
  - `test_public_handle_growm_prepare_reuse_and_growth`
- The new regression now proves the remaining supported repeated-run
  eigensolver backend branch directly:
  - explicit `SPARSE_EIGS_BACKEND_LANCZOS`
  - explicit prepare on a smaller problem
  - repeated reuse on the same prepared shape
  - later on-demand growth to a larger problem and larger `k`
  - preserved `backend_used == SPARSE_EIGS_BACKEND_LANCZOS`
- Updated the README project-structure line for `sparse_iterative.h` so it no
  longer understates the landed handle set:
  - repeated-run handles for `CG` / `GMRES` / `MINRES`
- Updated the small top-of-tutorial include comment so it now matches the live
  iterative-family surface:
  - `CG`
  - `GMRES`
  - `MINRES`

### Why this stayed inside the Sprint 54 fence

- No new public API family was added.
- No new solver backend was exposed beyond the already supported surface.
- No broad tutorial rewrite was started.
- The batch only closed the final high-signal proof/docs drift:
  - explicit grow-m public-handle proof
  - last stale high-signal summary lines

### Validation

Required Day 11 gates:

- `make format`
- `make lint`
- `make test`

All passed.

Focused Day 11 follow-ons:

- `./build/test_eigs` -> `30 / 30`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`
- `rg` sanity checks over the touched README/tutorial wording

Representative direct results:

- `test_eigs` now directly passes the full explicit public repeated-run
  eigensolver backend proof set:
  - `test_public_handle_growm_prepare_reuse_and_growth`
  - `test_public_handle_thick_restart_prepare_reuse_and_growth`
  - `test_public_handle_lobpcg_prepare_reuse_and_growth`
- `bench_eigs_reuse` stayed aligned with the full supported backend set:
  - `growm-nos4-k5`: `1.07x`
  - `thick-bcsstk14-k5`: `1.01x`
  - `lobpcg-diag40-k3`: `0.99x`
  - all three kept exact eigenvalue parity:
    - `|lambda|max diff = 0.000e+00`
- `example_eigs` stayed stable on the explicit LOBPCG teaching path:
  - `bcsstk04`: `3 / 3` smallest eigenpairs
  - `62` outer iterations
  - `backend_used = LOBPCG`
  - `residual_norm = 8.808e-09`

### Day 11 outcome

Sprint 54’s remaining high-value proof/docs gaps are now closed:

- direct public-handle proof explicitly covers:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- the highest-signal README/tutorial summary lines now match the landed support
  surface instead of implying an older narrower state

That leaves the branch ready for:

- compatibility audit
- final validation sweep
- closeout

## Sprint 54 Day 12 - post-landing compatibility audit

Date: 2026-06-03
Commit intent: audit the landed Sprint 54 branch against the preserved
public-handle fence and chosen exclusion boundaries, then fix the Day 13
validation checklist from the landed state.

### Audit scope

The Day 12 audit re-checked the highest-signal surfaces across:

- public headers
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
- caller-facing docs
  - `README.md`
  - `examples/README.md`
  - `docs/tutorial.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- proof surfaces
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- example surfaces
  - `example_iterative`
  - `example_ic_minres`
  - `example_eigs`

### Day 12 findings

#### 1. The landed branch still matches the preserved one-shot vs handle fence

The main compatibility rule still holds everywhere that matters:

- one-shot solver APIs remain first-class
- repeated-run handles remain opt-in paths for stable-dimension repeated runs
- the shipped examples still read as intentionally one-shot-first rather than
  as accidental omissions

This stayed consistent across:

- `README.md`
- `examples/README.md`
- the public handle sections in `include/sparse_iterative.h`
- the public handle sections in `include/sparse_eigs.h`

#### 2. The supported iterative repeated-run handle set remains honest and bounded

The final intended iterative handle surface still reads consistently as:

- `CG`
- `GMRES`
- `MINRES`

And the intended exclusions still read as exclusions rather than broken partial
implementations:

- `BiCGSTAB`
- block iterative workflows

No benchmark, example, or README section audited on Day 12 overclaimed handle
support for those excluded families.

#### 3. The supported eigensolver repeated-run handle set remains honest and bounded

The final intended eigensolver handle surface still reads consistently as:

- grow-m Lanczos
- thick-restart Lanczos
- explicit `LOBPCG`

The direct test surface, benchmark surface, and user-facing docs now all agree
on that same three-backend set:

- `tests/test_eigs.c`
- `benchmarks/bench_eigs_reuse.c`
- `README.md`
- `examples/README.md`

#### 4. Reuse semantics remain honestly bounded

The repeated-run lifecycle wording still preserves the intended honesty line:

- reuse preserves allocation capacity
- reuse does not preserve old numerical Krylov / Ritz / search-direction state
- one-shot APIs remain supported and are not deprecated by Sprint 54

The same honesty line also remains consistent with the benchmark proof and the
example adoption surfaces.

#### 5. No blocker-level drift surfaced

Day 12 did not surface a blocker-level mismatch between:

- code
- tests
- benchmarks
- examples
- top-level docs

The only remaining queue is future-facing rather than corrective:

- larger tutorial modernization if later epics want explicit repeated-run
  teaching examples
- any later public-handle expansion beyond the bounded Sprint 54 fence

### Day 13 validation checklist

The final validation checklist is now fixed from the landed Day 9-11 state:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- targeted Sprint 54 follow-ons:
  - `./build/test_iterative`
  - `./build/test_minres`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

### Day 12 outcome

The landed Sprint 54 branch still matches the preserved public repeated-run
solver fence:

- one-shot APIs remain first-class
- repeated-run handles remain bounded opt-in paths
- excluded families still read as intentional exclusions, not half-supported
  promises
- the final validation checklist is now explicit and ready for Day 13

## Sprint 54 Day 13 - validation sweep

Date: 2026-06-03
Commit intent: run the full required gate, confirm the reviewed Makefile/CMake
truthfulness anchors, and rerun the high-signal iterative/eigensolver
repeated-run follow-ons from the landed Sprint 54 state.

### Commands run

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- `./build/test_iterative`
- `./build/test_minres`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

### Findings

#### 1. The full required gate passed from the landed Sprint 54 state

Day 13 completed the full required gate successfully:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Interpretation:

- Sprint 54 reached a real validated close state rather than only a
  compatibility-audited state

#### 2. The reviewed Makefile/CMake truthfulness anchors stayed exact

Measured Day 13 reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- reviewed CMake total time from `make quality-review-full` = `144.25 sec`

Interpretation:

- the maintained reviewed closeout contract is still stable after the final
  supported repeated-run handle completion work

#### 3. The supported iterative repeated-run handle proof set stayed green

Targeted iterative reruns all passed:

- `./build/test_iterative`
  - `79 / 79`
- `./build/test_minres`
  - `43 / 43`

The shipped iterative examples also stayed healthy:

- `./build/example_iterative`
  - unpreconditioned GMRES: `25` iterations, residual `9.56e-11`
  - ILU(0)-preconditioned GMRES: `9` iterations, residual `3.14e-11`
- `./build/example_ic_minres`
  - MINRES on the `42x42` KKT system: `39` iterations, residual `3.87e-11`
  - Jacobi-MINRES: `26` iterations, residual `4.16e-11`
  - block MINRES on the `28x28` KKT system: residual `8.06e-16`

Measured iterative reuse benchmark results:

- `./build/bench_iterative_reuse`
  - `cg-tridiag-300`: `1.12x`
  - `gmres-unsym-220`: `0.85x`
  - `minres-kkt-42`: `1.28x`

Interpretation:

- the final supported iterative handle set:
  - `CG`
  - `GMRES`
  - `MINRES`
- stayed correct and measurable across direct tests, examples, and the public
  reuse benchmark surface

#### 4. The supported eigensolver repeated-run handle proof set stayed green

Targeted eigensolver reruns all passed:

- `./build/test_eigs`
  - `30 / 30`
- `./build/test_eigs_lobpcg`
  - `26 / 26`

The shipped eigensolver example also stayed healthy:

- `./build/example_eigs`
  - nos4 largest-eigenvalue case: `5 / 5` pairs in `115` Lanczos iterations
  - KKT nearest-`sigma` case: `3 / 3` pairs in `6` Lanczos iterations
  - explicit LOBPCG on `bcsstk04`: `3 / 3` pairs in `62` outer iterations
  - LOBPCG reported residual = `8.808e-09`

Measured eigensolver reuse benchmark results:

- `./build/bench_eigs_reuse`
  - `growm-nos4-k5`: `1.00x`
  - `thick-bcsstk14-k5`: `0.99x`
  - `lobpcg-diag40-k3`: `1.00x`
  - all three kept exact eigenvalue parity:
    - `|lambda|max diff = 0.000e+00`

Interpretation:

- the final supported eigensolver handle backend set:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- stayed aligned across direct proof, examples, and the public reuse benchmark

### Day 13 outcome

Sprint 54 now has a validated measured close state:

- the full required gate passed
- the reviewed Makefile/CMake truthfulness anchors remained exact
- the supported iterative and eigensolver repeated-run follow-ons stayed green
- no new reconciliation queue surfaced during validation
