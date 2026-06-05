# Sprint 55 Working Notes

## Day 1

**Objective:** Turn the Sprint 55 project-plan scope plus the Sprint 54
validated repeated-run solver close state into a concrete large-source
decomposition starting point by confirming the preserved reviewed baseline,
naming the Sprint 55 implementation workstreams explicitly, and defining the
authoritative eigensolver/iterative implementation, proof, and caller-surface
hotspots before any extraction work begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 55 project-plan source and the new sprint plan:
   - `sed -n '186,215p' docs/planning/EPIC_5/PROJECT_PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_55/PLAN.md`
3. Re-read the strongest inherited repeated-run solver closeout source:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_54/artifacts/day14-closeout-and-handoff.md`
4. Re-read the Epic 5 large-source review/todo guidance:
   - `sed -n '110,150p' docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
   - `sed -n '116,136p' docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md`
   - `rg -n "large-source|decomposition|sparse_eigs\\.c|sparse_iterative\\.c|historical comment|comment reduction" docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md docs/planning/EPIC_5/PROJECT_PLAN.md docs/planning/EPIC_5/SPRINT_54/RETROSPECTIVE.md`
5. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
6. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
7. Measure the live eigensolver/iterative implementation and proof hotspots:
   - `wc -l src/sparse_eigs.c src/sparse_iterative.c src/sparse_iterative_workspace_internal.c src/sparse_eigs_workspace_internal.c src/sparse_iterative_internal.h src/sparse_eigs_internal.h tests/test_eigs.c tests/test_eigs_lobpcg.c tests/test_iterative.c benchmarks/bench_eigs_reuse.c benchmarks/bench_iterative_reuse.c include/sparse_eigs.h include/sparse_iterative.h examples/example_eigs.c examples/example_iterative.c README.md docs/maintainer_guide.md`
8. Reconfirm the live repeated-run and solver-family references across the
   caller-facing surfaces:
   - `rg -n "MINRES|LOBPCG|repeated-run|one-shot|handle" README.md examples/README.md docs/maintainer_guide.md include/sparse_iterative.h include/sparse_eigs.h`

### Day 1 Findings

#### 1. Sprint 55 starts from a validated repeated-run solver package, not from renewed lifecycle design work

The inherited starting state is already explicit and stable:

- Sprint 54 closed with:
  - explicit steady-state repeated-run solver support boundaries
  - public iterative handles for `CG`, `GMRES`, and `MINRES`
  - tightened eigensolver repeated-run proof for grow-m, thick-restart, and
    explicit `LOBPCG`
  - aligned reuse benchmarks and caller-facing repeated-run wording
- Sprint 54 also closed from:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- the inherited caller-facing contract is already real:
  - one-shot solver APIs remain first-class entry points
  - repeated-run handles remain bounded opt-in surfaces
  - reuse preserves allocation/setup capacity, not stale numerical state

Interpretation:

- Sprint 55 is not a public lifecycle redesign sprint
- Sprint 55 is not a validation-recovery sprint
- Sprint 55 is a bounded maintainability and ownership sprint

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible on all substantial decomposition batches

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

Interpretation:

- Sprint 55 should keep using the exact `strongest local reviewed baseline`
  phrasing
- substantial extraction batches should continue treating the reviewed CMake
  count and parity contract as truthfulness anchors

#### 3. The Epic 5 large-source review queue is now even more concentrated in `src/sparse_eigs.c` and `src/sparse_iterative.c`

The Epic 5 review already called out:

- `src/sparse_eigs.c` (`3233` lines)
- `src/sparse_iterative.c` (`2361` lines)

The live repo state now shows:

- `src/sparse_eigs.c` = `3233`
- `src/sparse_iterative.c` = `2377`

Interpretation:

- `src/sparse_eigs.c` remains the single clearest first target for bounded
  extraction
- `src/sparse_iterative.c` has grown since the review and is now a stronger
  maintainability hotspot than the original Epic 5 review snapshot implied
- Sprint 55 should treat both files as live large-source risks, not merely as
  review artifacts

#### 4. The real Sprint 55 queue is decomposition-first, not feature-first

The Sprint 55 plan items and live repo state narrow to seven bounded work
classes:

1. `sparse_eigs.c` seam audit
2. eigensolver decomposition batch 1
3. eigensolver decomposition batch 2
4. `sparse_iterative.c` seam audit
5. iterative decomposition batch 1
6. historical comment reduction on touched permanent implementation files
7. validation and closeout

Interpretation:

- Sprint 55 should reduce ownership ambiguity in the two biggest remaining
  solver files before broadening any other Epic 5 queue
- the sprint should explicitly prefer helper-vs-orchestration splits over
  generic “split by size” edits

#### 5. The live hotspot map is already concentrated enough to name directly

The main touched surfaces are clear before any extraction work begins:

- public headers:
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `687`
- main implementations:
  - `src/sparse_iterative.c` = `2377`
  - `src/sparse_eigs.c` = `3233`
  - `src/sparse_iterative_workspace_internal.c` = `215`
  - `src/sparse_eigs_workspace_internal.c` = `267`
  - `src/sparse_iterative_internal.h` = `26`
  - `src/sparse_eigs_internal.h` = `620`
- strongest proof surfaces:
  - `tests/test_iterative.c` = `2993`
  - `tests/test_eigs.c` = `1522`
  - `tests/test_eigs_lobpcg.c` = `1196`
  - `benchmarks/bench_iterative_reuse.c` = `370`
  - `benchmarks/bench_eigs_reuse.c` = `253`
- strongest caller-facing adoption surfaces:
  - `examples/example_iterative.c` = `144`
  - `examples/example_eigs.c` = `285`
  - `README.md` = `987`
  - `docs/maintainer_guide.md` = `294`

Interpretation:

- the strongest implementation risk seams are still concentrated in the two
  large solver translation units, not in their current small workspace-helper
  side files
- the proof surfaces have also grown materially and will need careful parity
  preservation during extraction

#### 6. The inherited support boundary is already fixed, which gives Sprint 55 a clean non-goal fence

The inherited repeated-run solver support boundary remains:

- iterative handles:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver handles:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- one-shot solver APIs remain first-class peer entry points
- `BiCGSTAB` and block iterative workflows remain intentionally outside the
  public repeated-run handle set

Interpretation:

- Sprint 55 should preserve that public support boundary while changing
  implementation ownership underneath it
- public API expansion is not the right success criterion for this sprint

#### 7. Historical implementation narrative remains a real and named maintainability problem

The Epic 5 review and todo notes remain explicit that touched permanent
implementation files still carry unnecessary sprint-history narrative:

- `src/sparse_eigs.c` was explicitly called out in the review
- the todo list still names stale sprint-history reduction as its own work item

Interpretation:

- Sprint 55 should not treat comment cleanup as optional polish
- on touched permanent implementation files, the cleanup goal is:
  - preserve durable algorithm commentary
  - remove temporary sprint-history narrative

## Day 1 Close

Sprint 55 now has an explicit starting point:

- preserved reviewed baseline
- inherited validated repeated-run solver support fence
- named large-source maintainability hotspots
- clear decomposition-first workstreams
- explicit non-goal fence against public API redesign

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening Sprint 49-54 public lifecycle decisions.

## Day 2

**Objective:** Reconfirm the maintained reviewed baseline and truthfulness
anchors Sprint 55 must preserve, then define the smallest authoritative
validation boundary for the later eigensolver/iterative extraction days and
the high-signal rerun set those code-touch batches should use.

### Commands Run

1. Re-read the Sprint 55 Day 2 plan item and the current sprint notes:
   - `sed -n '79,122p' docs/planning/EPIC_5/SPRINT_55/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_55/WORKING_NOTES.md`
2. Reconfirm the maintained reviewed CMake truthfulness anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
3. Reconfirm the maintained reviewed wrapper authority surface:
   - `make -n quality-review-full`
4. Re-read the live quality-contract wording sources:
   - `rg -n "strongest local reviewed baseline|quality-review-full|quality-review-cmake|deadcode-check" README.md docs/maintainer_guide.md Makefile .github/workflows -g '!build'`
5. Reconfirm the main Sprint 55 follow-on binaries already present in the
   build tree:
   - `ls build/test_iterative build/test_eigs build/test_eigs_lobpcg build/test_minres build/example_iterative build/example_eigs build/bench_iterative_reuse build/bench_eigs_reuse`
6. Measure the live size of those main proof/adoption surfaces:
   - `wc -l tests/test_minres.c tests/test_eigs.c tests/test_eigs_lobpcg.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c examples/example_iterative.c examples/example_eigs.c`

### Day 2 Findings

#### 1. The strongest local reviewed baseline and truthfulness anchors remain exact

The maintained Sprint 55 baseline remains:

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

- Sprint 55 should keep using the exact `strongest local reviewed baseline`
  phrasing
- the reviewed CMake count and Makefile/CMake parity contract remain the
  authoritative truthfulness anchors for later extraction days

#### 2. The later decomposition code-day gate is simple and should stay explicit

The mandatory gate for later `*.c` / `*.h` decomposition work remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial implementation ownership batches
remains:

- `make quality-review-full`

Interpretation:

- docs-only audit/design/summary days do not need the full code-day gate
- substantial extraction batches should continue to run both the direct gate
  and the stronger reviewed baseline path

#### 3. The live quality-contract wording still matches the maintained split across README, maintainer guide, and Makefile

The quality-contract wording remains aligned across the main authority
surfaces:

- `README.md`
  - user-facing command map
  - strongest local reviewed baseline wording
  - explicit `deadcode-check` completeness-gate wording
- `docs/maintainer_guide.md`
  - maintainer-facing authority framing
  - reviewed CMake parity anchor
  - dead-code interpretation boundary
- `Makefile`
  - executable reviewed-target authority
  - current rerun guidance
  - current test-count parity checks

Interpretation:

- Sprint 55 does not need to reopen any quality-contract wording work on Day 2
- the maintained reviewed baseline language is already stable enough to carry
  forward unchanged

#### 4. The high-signal Sprint 55 rerun set is now fixed explicitly from the live build tree

The main Sprint 55 follow-on binaries already present in `build/` are:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/test_minres`
- `./build/example_iterative`
- `./build/example_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Interpretation:

- Sprint 55 can keep its rerun set focused on the iterative/eigensolver
  families actually touched by the large-source decomposition work
- no extra cross-domain direct-solver rerun set is needed for the sprint's
  default extraction batches

#### 5. The proof and adoption surfaces are now large enough that parity preservation is part of the extraction work itself

The live proof/adoption surface sizes are now:

- `tests/test_minres.c` = `1588`
- `tests/test_eigs.c` = `1522`
- `tests/test_eigs_lobpcg.c` = `1196`
- `benchmarks/bench_iterative_reuse.c` = `370`
- `benchmarks/bench_eigs_reuse.c` = `253`
- `examples/example_iterative.c` = `144`
- `examples/example_eigs.c` = `285`

Interpretation:

- Sprint 55 extraction work should assume that proof-surface legibility and
  parity matter alongside implementation-file size reduction
- the rerun set is not ceremonial; it is the main defense against accidental
  behavior drift while ownership moves under the hood

## Day 2 Close

Sprint 55 now has an explicit validation boundary:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit code-day gate
- explicit stronger reviewed-baseline default
- authoritative iterative/eigensolver rerun set from the live build tree

That is enough to move to the Day 3 `sparse_eigs.c` seam audit without any
remaining ambiguity around validation expectations.

## Day 3

**Objective:** Reduce `src/sparse_eigs.c` to a concrete extraction map by
separating the live eigensolver ownership bands, ranking the real bounded seam
options, and choosing the first extraction target on maintainability grounds
rather than line-count reduction alone.

### Commands Run

1. Re-read the Sprint 55 Day 3 plan item and the current sprint notes:
   - `sed -n '123,159p' docs/planning/EPIC_5/SPRINT_55/PLAN.md`
   - `sed -n '1,420p' docs/planning/EPIC_5/SPRINT_55/WORKING_NOTES.md`
2. Reconfirm the live eigensolver implementation / proof hotspot sizes:
   - `wc -l src/sparse_eigs.c src/sparse_eigs_internal.h tests/test_eigs.c tests/test_eigs_lobpcg.c benchmarks/bench_eigs_reuse.c examples/example_eigs.c`
3. Build a top-level `src/sparse_eigs.c` function map:
   - `rg -n "^(static |sparse_err_t |void )" src/sparse_eigs.c`
4. Re-read the file head and the largest backend-specific cluster:
   - `sed -n '1,220p' src/sparse_eigs.c`
   - `sed -n '2280,3233p' src/sparse_eigs.c`
5. Re-read the current internal eigensolver declarations:
   - `sed -n '1,260p' src/sparse_eigs_internal.h`
   - `sed -n '260,620p' src/sparse_eigs_internal.h`
6. Recheck the strongest dedicated proof surfaces around the likely extraction
   seams:
   - `sed -n '1,260p' tests/test_eigs.c`
   - `sed -n '1,240p' benchmarks/bench_eigs_reuse.c`
7. Reconfirm the live seam keywords across implementation and proof files:
   - `rg -n "static |sparse_eigs_handle|lobpcg|thick|grow|workspace|prepare|ritz|residual|backend" src/sparse_eigs.c src/sparse_eigs_internal.h include/sparse_eigs.h tests/test_eigs.c tests/test_eigs_lobpcg.c benchmarks/bench_eigs_reuse.c examples/example_eigs.c`

### Day 3 Findings

#### 1. `src/sparse_eigs.c` already contains three large ownership bands, but they still live in one permanent file

The live top-level map now separates cleanly into three major bands:

- generic Lanczos and public-entry orchestration:
  - top-of-file helpers through:
    - `s46_sparse_eigs_sym_impl(...)`
    - `sparse_eigs_sym(...)`
    - `sparse_eigs_sym_with_handle(...)`
    - `sparse_eigs_sym_with_workspace_internal(...)`
- thick-restart Lanczos and restart-state machinery:
  - `lanczos_restart_state_free(...)`
  - `s21_arrowhead_to_tridiag(...)`
  - `lanczos_restart_pick_locked(...)`
  - `lanczos_restart_state_assemble(...)`
  - `lanczos_thick_restart_iterate(...)`
  - `s21_dense_sym_jacobi(...)`
  - `s21_build_dense_arrowhead(...)`
  - `s21_recompute_residual(...)`
  - `s21_thick_restart_outer_loop(...)`
- LOBPCG backend:
  - `s21_lobpcg_orthonormalize_block(...)`
  - `s21_lobpcg_rr_step(...)`
  - `s21_lobpcg_solve(...)`

Interpretation:

- the file is no longer ambiguous about where the ownership bands are
- the main problem is that those bands still share one permanent translation
  unit instead of cleaner file-level seams

#### 2. The cleanest first extraction target is the LOBPCG backend, not the outer public-entry layer

The LOBPCG region is the strongest first extraction candidate because it is:

- contiguous in `src/sparse_eigs.c`
- already grouped in `src/sparse_eigs_internal.h`
- solver-family-specific rather than cross-cutting
- directly covered by a dedicated proof file:
  - `tests/test_eigs_lobpcg.c`
- caller-visible through the same public handle and one-shot surface, but not
  mixed deeply into the public orchestration layer itself

Interpretation:

- a LOBPCG extraction can materially reduce `src/sparse_eigs.c` size while
  preserving the public eigensolver contract intact
- it is the strongest first helper/module slice because it improves ownership
  rather than just moving generic glue

#### 3. The second-best extraction target is the thick-restart restart-state block, but it is a higher-risk second batch

The thick-restart cluster is also large and contiguous, but it is more
entangled than LOBPCG because it shares:

- Lanczos basis and Ritz-selection assumptions
- restart-state types already declared in `src/sparse_eigs_internal.h`
- generic dense-arrowhead and residual recomputation helpers
- direct interaction with the main public backend dispatch layer

Interpretation:

- this is a strong second batch candidate
- it should follow the first extraction so the remaining orchestration layer is
  smaller before the more interdependent restart-state cluster moves

#### 4. The outer public-entry and handle layer should remain in `src/sparse_eigs.c` for Sprint 55 Phase 1

The public-entry band currently owns:

- backend AUTO/explicit selection
- one-shot vs handle entry routing
- validation and result initialization
- shift-invert setup and public contract preservation

Interpretation:

- moving that layer first would not give the best maintainability return
- it is cross-cutting orchestration, not the cleanest backend-owned slice
- Sprint 55 Phase 1 should keep `src/sparse_eigs.c` centered on that public
  orchestration role while moving family-specific backend bodies out around it

#### 5. `src/sparse_eigs_internal.h` is already broad enough to support extraction, but it also exposes where comment cleanup is overdue

The internal header already groups:

- generic Lanczos helpers
- thick-restart types and helpers
- LOBPCG helpers
- internal repeated-run workspace entry points

That is enough structure to support a module split, but it also confirms that
the eigensolver code still carries too much sprint-history narrative directly
inside permanent implementation comments.

Interpretation:

- Sprint 55 does not need a new abstraction vocabulary before extraction begins
- it does need to reduce sprint-history prose in touched eigensolver files as
  part of the implementation work

#### 6. The dedicated proof surfaces already line up with the likely extraction boundaries

The strongest proof surfaces match the likely backend seams:

- generic public lifecycle and backend proof:
  - `tests/test_eigs.c` = `1522`
  - `benchmarks/bench_eigs_reuse.c` = `253`
  - `examples/example_eigs.c` = `285`
- dedicated LOBPCG backend proof:
  - `tests/test_eigs_lobpcg.c` = `1196`

Interpretation:

- the proof layout already favors a first LOBPCG extraction
- Sprint 55 can preserve behavior confidence by treating the dedicated LOBPCG
  test file as the primary proof surface for the first batch

#### 7. The ranked eigensolver extraction order is now concrete

The bounded seam ranking is:

1. LOBPCG backend extraction
2. thick-restart restart-state / arrowhead cluster extraction
3. residual cleanup of the remaining `src/sparse_eigs.c` orchestration layer
4. later reconsideration of smaller generic helper splits only if they improve
   ownership further

Rejected Day 3 candidates:

- move the public-entry / handle layer first
  - rejected because it is the highest cross-cutting glue layer, not the
    cleanest backend-owned slice
- split only by helper count or comment size
  - rejected because that would be mechanical churn, not ownership

## Day 3 Close

Sprint 55 now has a concrete eigensolver extraction map:

- `src/sparse_eigs.c` reduced to named ownership bands
- LOBPCG selected as the strongest first extraction target
- thick-restart selected as the strongest second batch target
- public orchestration retained in `src/sparse_eigs.c` for Phase 1
- historical comment reduction explicitly tied to the touched eigensolver files

That is enough to move to the Day 4 first-batch design work without ambiguity
about the first extraction boundary.

## Day 4

**Objective:** Freeze the first eigensolver extraction boundary by turning the
Day 3 LOBPCG-first ranking into an exact file-boundary, declaration, and
validation-preservation design before any permanent implementation code moves.

### Commands Run

1. Re-read the Sprint 55 Day 4 plan item and current sprint notes:
   - `sed -n '160,194p' docs/planning/EPIC_5/SPRINT_55/PLAN.md`
   - `sed -n '1,520p' docs/planning/EPIC_5/SPRINT_55/WORKING_NOTES.md`
2. Re-read the Day 3 eigensolver seam audit:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_55/artifacts/day3-sparse-eigs-seam-audit.md`
3. Reconfirm the current eigensolver side-file inventory:
   - `ls src | rg '^sparse_eigs'`
4. Re-read the public entry / thick-restart boundary where the first extraction
   must stop:
   - `sed -n '1450,1515p' src/sparse_eigs.c`
5. Re-read the full LOBPCG block that the first batch would move:
   - `sed -n '2650,3233p' src/sparse_eigs.c`
6. Re-read the reusable workspace typed-view header that the LOBPCG extraction
   already depends on:
   - `sed -n '1,220p' src/sparse_eigs_workspace_internal.h`

### Day 4 Findings

#### 1. The first extraction should create a backend-owned source file, not a new public surface

The strongest first-batch source split is now explicit:

- keep in `src/sparse_eigs.c`:
  - public one-shot and handle entry points
  - backend AUTO/explicit selection
  - shared validation and result setup
  - grow-m Lanczos path
  - thick-restart path and restart-state machinery
- move to a new backend-owned source file:
  - `src/sparse_eigs_lobpcg.c`

The functions that should move together are:

- `s21_lobpcg_orthonormalize_block(...)`
- `s21_lobpcg_rr_step(...)`
- `s21_lobpcg_solve(...)`
- `s21_lobpcg_init_X(...)`

Interpretation:

- the first batch is a real backend/module extraction, not a public API change
- the retained `src/sparse_eigs.c` role becomes clearer: public orchestration
  plus non-LOBPCG shared backend dispatch

#### 2. Sprint 55 Phase 1 should not introduce a new private header just for the first batch

The current internal structure already provides:

- `src/sparse_eigs_internal.h`
  - prototypes for generic Lanczos, thick-restart, and LOBPCG helpers
- `src/sparse_eigs_workspace_internal.h`
  - the reusable typed workspace views the LOBPCG code already needs

That is enough for the first batch.

Interpretation:

- the first extraction should keep LOBPCG declarations in the existing
  `src/sparse_eigs_internal.h`
- introducing `src/sparse_eigs_lobpcg_internal.h` in the same batch would mix
  two ownership changes:
  - source extraction
  - private-header taxonomy redesign
- Sprint 55 can defer header narrowing to a later batch if the post-extraction
  shape makes it worthwhile

#### 3. The first extraction invariants are now concrete

The first batch must preserve:

- public handle semantics:
  - `sparse_eigs_handle_prepare(...)`
  - `sparse_eigs_sym_with_handle(...)`
  - reuse preserves allocation/setup capacity, not stale Ritz/search state
- backend selection/reporting behavior:
  - explicit `SPARSE_EIGS_BACKEND_LOBPCG`
  - AUTO behavior unchanged
  - `result->backend_used` unchanged
- workspace and growth behavior:
  - current `sparse_eigs_workspace_prepare_lobpcg(...)` contract
  - zero-init/local-workspace fallback behavior
  - on-demand handle/workspace reuse behavior
- benchmark and example parity:
  - `benchmarks/bench_eigs_reuse.c`
  - `examples/example_eigs.c`
- dedicated proof parity:
  - `tests/test_eigs_lobpcg.c`
  - public-handle LOBPCG proof in `tests/test_eigs.c`

Interpretation:

- the first batch succeeds only if it leaves the public repeated-run and
  one-shot eigensolver behavior observably unchanged
- file-count improvement alone is not enough

#### 4. The first batch should keep the thick-restart block untouched except for call-site continuity

The Day 3 ranking still holds:

- thick-restart remains the second strongest extraction target
- it is more entangled with generic Lanczos assumptions than LOBPCG

Interpretation:

- Day 5 should avoid opportunistic thick-restart cleanup unless the build
  requires trivial include/prototype adjustments
- mixing LOBPCG extraction and thick-restart refactoring in one batch would
  weaken the ownership proof of the first extraction

#### 5. Comment cleanup should be scoped to touched LOBPCG blocks and the retained public dispatch seam

The minimal comment policy for the first batch is now explicit:

- preserve:
  - algorithm meaning
  - invariants
  - convergence and workspace semantics
- reduce where touched:
  - sprint chronology
  - landing-history narrative
  - comments that explain prior sprint ordering instead of present code truth

Interpretation:

- Sprint 55 should not try to clean the entire eigensolver file in the first
  extraction batch
- it should clean the touched LOBPCG region and any adjacent retained dispatch
  comments that become obviously stale after the move

#### 6. The Day 5 touched-file set is now bounded

The first-batch expected touched set is:

- `src/sparse_eigs.c`
- `src/sparse_eigs_lobpcg.c` (new)
- `src/sparse_eigs_internal.h`
- possibly `tests/test_eigs.c` only if a small include or comment adjustment is
  needed

The batch should avoid by default:

- `include/sparse_eigs.h`
- `src/sparse_eigs_workspace_internal.h`
- `tests/test_eigs_lobpcg.c`
- `benchmarks/bench_eigs_reuse.c`
- `examples/example_eigs.c`

Interpretation:

- Day 5 can stay small and ownership-focused
- proof surfaces should mostly validate the extraction, not need redesign

## Day 4 Close

Sprint 55 now has an explicit first eigensolver extraction design:

- first batch target:
  - move the LOBPCG backend into `src/sparse_eigs_lobpcg.c`
- retained main-file role:
  - public orchestration plus non-LOBPCG backend dispatch
- declaration strategy:
  - reuse `src/sparse_eigs_internal.h` for Phase 1
- validation goal:
  - preserve one-shot, handle, benchmark, example, and backend-reporting
    behavior exactly

That is enough to begin the Day 5 implementation batch without reopening the
first extraction boundary.

## Day 5

**Objective:** Land the first bounded `src/sparse_eigs.c` decomposition batch by
extracting the LOBPCG backend into its own permanent source file while keeping
the public repeated-run eigensolver contract, backend-selection/reporting
behavior, and proof surfaces observably unchanged.

### Commands Run

1. Finish the Day 5 permanent implementation batch:
   - `apply_patch` updates to:
     - `CMakeLists.txt`
     - `Makefile`
     - `src/sparse_eigs.c`
     - `src/sparse_eigs_internal.h`
     - `src/sparse_eigs_lobpcg.c` (new)
2. Run the required code-day formatting gate:
   - `make format`
3. Run the required code-day lint gate:
   - `make lint`
4. Run the required full test gate:
   - `make test`
5. Run the stronger reviewed baseline for the substantial extraction batch:
   - `make quality-review-full`
6. Run the Day 5 touched-family follow-ons:
   - `./build/test_eigs`
   - `./build/test_eigs_lobpcg`
   - `./build/example_eigs`
   - `./build/bench_eigs_reuse`
7. Re-measure the landed eigensolver ownership split:
   - `wc -l src/sparse_eigs.c src/sparse_eigs_lobpcg.c src/sparse_eigs_internal.h`

### Day 5 Findings

#### 1. The first bounded eigensolver extraction landed exactly on the Day 4 seam

The permanent code move is now real:

- new source file:
  - `src/sparse_eigs_lobpcg.c`
- moved backend-owned function set:
  - `s21_lobpcg_orthonormalize_block(...)`
  - `s21_lobpcg_rr_step(...)`
  - `s21_lobpcg_solve(...)`
  - `s21_lobpcg_init_X(...)`

And the retained `src/sparse_eigs.c` role stayed bounded:

- public one-shot and handle entry points
- backend AUTO/explicit selection
- shared validation/result setup
- generic Lanczos helpers
- grow-m Lanczos path
- thick-restart path and restart-state machinery
- top-level backend dispatch/orchestration

Interpretation:

- Sprint 55 Day 5 achieved a real source decomposition, not just comment motion
- the first extraction stayed maintainability-first and ownership-first rather
  than reopening the public solver contract

#### 2. The batch reused the existing internal header strategy instead of mixing in a new private-header redesign

The extraction reused the existing:

- `src/sparse_eigs_internal.h`

No new LOBPCG-private header was introduced.

Interpretation:

- the Day 4 design rule held in practice
- Batch 1 changed one ownership axis:
  - source-file placement
- it did not mix in a second taxonomy redesign around private-header
  partitioning

#### 3. The main large-file hotspot is materially smaller while public behavior stays anchored

The landed line-count split is now:

- `src/sparse_eigs.c`:
  - Day 1 baseline: `3233`
  - after Day 5 extraction: `2660`
- extracted `src/sparse_eigs_lobpcg.c`:
  - `401`

Interpretation:

- the biggest remaining eigensolver file is materially smaller now
- the extracted code lives in a backend-owned translation unit instead of
  remaining embedded in the public orchestration file
- Sprint 55 made real maintainability progress without needing a public-header
  or benchmark/example redesign in the same batch

#### 4. The preserved contract stayed observably unchanged across the strongest public proof surfaces

The preserved invariants were revalidated directly:

- public repeated-run handle proof:
  - `./build/test_eigs` -> `30 / 30`
- dedicated LOBPCG proof:
  - `./build/test_eigs_lobpcg` -> `26 / 26`
- shipped example:
  - `./build/example_eigs`
- public-handle reuse benchmark:
  - `./build/bench_eigs_reuse`

Representative direct results stayed stable:

- `example_eigs`:
  - explicit `LOBPCG` on `bcsstk04` still converged `3 / 3` smallest pairs in
    `62` outer iterations
  - reported residual stayed `8.808e-09`
- `bench_eigs_reuse`:
  - `growm-nos4-k5` -> `1.10x`
  - `thick-bcsstk14-k5` -> `0.99x`
  - `lobpcg-diag40-k3` -> `1.02x`
  - all retained exact eigenvalue parity:
    - `|lambda|max diff = 0.000e+00`

Interpretation:

- the first extraction preserved one-shot, handle, benchmark, and example
  behavior closely enough that Day 5 did not surface a reconciliation queue
- the explicit LOBPCG backend still behaves exactly like a first-class member
  of the public repeated-run surface after the move

#### 5. The full required validation stack stayed green, including the reviewed parity path

The Day 5 code-day gates all passed:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline also passed:

- `make quality-review-full`

And the maintained truthfulness anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 248.97 sec`

Interpretation:

- the first decomposition batch is validated as a structural ownership change,
  not just as a local compile success
- the reviewed parity path remained truthful with the extracted new source file
  in place

## Day 5 Close

Sprint 55 now has a real first eigensolver decomposition batch:

- `src/sparse_eigs.c` is smaller and more orchestration-focused
- the LOBPCG backend now lives in `src/sparse_eigs_lobpcg.c`
- the existing internal header strategy carried the move cleanly
- the public repeated-run eigensolver contract remained stable across tests,
  examples, and reuse benchmarks

That is enough to move into the next decomposition day without reopening the
Batch 1 ownership boundary.

## Day 6

**Objective:** Re-audit the post-Day-5 eigensolver ownership shape and freeze
the second bounded eigensolver batch around the thick-restart restart-state
cluster so Day 7 can land a real follow-on decomposition instead of a generic
cleanup pass.

### Commands Run

1. Re-read the Day 6 / Day 7 sprint-plan boundary:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_55/PLAN.md`
2. Re-read the landed Day 5 artifact:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_55/artifacts/day5-eigensolver-decomposition-batch1.md`
3. Re-audit the post-Day-5 thick-restart block and retained public dispatch
   seam:
   - `sed -n '1450,2445p' src/sparse_eigs.c`
4. Re-audit the current internal declaration surface for the thick-restart
   block:
   - `sed -n '150,560p' src/sparse_eigs_internal.h`
5. Reconfirm the remaining file-size and proof context:
   - `wc -l src/sparse_eigs.c src/sparse_eigs_internal.h tests/test_eigs_thick_restart.c`
6. Reconfirm the strongest thick-restart references across the landed
   eigensolver sources and proof surface:
   - `rg -n "thick-restart|restart-state|arrowhead|cluster" docs/planning/EPIC_5/SPRINT_55/PLAN.md docs/planning/EPIC_5/SPRINT_55/WORKING_NOTES.md src/sparse_eigs.c src/sparse_eigs_internal.h tests/test_eigs_thick_restart.c`

### Day 6 Findings

#### 1. The post-Day-5 eigensolver ownership map now makes the thick-restart cluster the clearest second batch

After the LOBPCG extraction, the residual large owned block inside
`src/sparse_eigs.c` is now the Sprint 21 thick-restart cluster:

- restart-state lifecycle:
  - `lanczos_restart_state_free(...)`
- spectrum-preservation helper:
  - `s21_arrowhead_to_tridiag(...)`
- restart-state assembly helpers:
  - `s21_pick_locked(...)`
  - `s21_recompute_residual(...)`
  - `s21_build_dense_arrowhead(...)`
- phase / outer-loop execution:
  - `lanczos_thick_restart_iterate(...)`
  - `s21_thick_restart_outer_loop(...)`

Interpretation:

- Day 5 made the second batch easier to see cleanly
- the next ownership improvement target is no longer ambiguous:
  - the thick-restart restart-state and arrowhead machinery is now the
    strongest remaining backend-owned block in `src/sparse_eigs.c`

#### 2. The second batch should still be a real helper/module split, not merely comment cleanup or header gardening

The strongest residual maintainability issue is not:

- public dispatch wording
- header prose alone
- generic dense-helper naming

It is that the thick-restart implementation still sits inside the main
orchestration file even though it has a coherent backend-owned helper cluster.

Interpretation:

- Day 7 should still land a real source extraction
- a docs/comment-only or declaration-only Day 7 would under-deliver against the
  post-Day-5 ownership map

#### 3. The exact second-batch split is now concrete

The preferred second-batch new file is:

- `src/sparse_eigs_thick_restart.c`

Move into that file:

- `lanczos_restart_state_free(...)`
- `s21_arrowhead_to_tridiag(...)`
- `s21_pick_locked(...)`
- `s21_recompute_residual(...)`
- `s21_build_dense_arrowhead(...)`
- `lanczos_thick_restart_iterate(...)`
- `s21_thick_restart_outer_loop(...)`

Keep in `src/sparse_eigs.c`:

- public one-shot and handle entry points
- shared result setup / validation
- AUTO and explicit backend selection
- generic Lanczos helpers
- grow-m Lanczos path
- shift-invert and shared operator composition
- the call sites that dispatch into thick-restart

Interpretation:

- the second batch can improve ownership materially without reopening the
  public orchestration layer
- the retained main file continues to read as the shared eigensolver front door
  rather than a backend dump

#### 4. The second batch should continue reusing the existing internal header, not mix in a second taxonomy redesign

The Day 5 rule still holds for Batch 2:

- keep using:
  - `src/sparse_eigs_internal.h`

Do not combine Day 7 with:

- a new `src/sparse_eigs_thick_restart_internal.h`
- broad private-header narrowing
- generic helper relocation unrelated to the thick-restart move

Interpretation:

- the next clean ownership improvement is source-file extraction first
- private-header taxonomy cleanup can remain a later optional follow-on once the
  post-Day-7 shape is known

#### 5. Some helpers must stay shared even if the thick-restart backend moves

The post-Day-5 audit makes the non-goal boundary clearer:

- keep shared:
  - `s21_dense_sym_jacobi(...)`
  - `s20_select_indices(...)`
  - generic Lanczos kernels and reorthogonalization helpers
  - shared workspace preparation and public-handle orchestration

Reason:

- these helpers are used across grow-m, thick-restart, and LOBPCG-adjacent
  proof/orchestration paths
- moving them in the same batch would blur the backend-owned seam and weaken
  the Day 7 extraction story

Interpretation:

- Day 7 should move the thick-restart backend-owned cluster only
- it should not try to turn the second batch into a generic “all eigensolver
  helpers rehome” rewrite

#### 6. The strongest proof surface for Day 7 is already fixed

The main touched proof surface for the second batch should be:

- `tests/test_eigs_thick_restart.c`

Secondary validation surfaces remain:

- `tests/test_eigs.c`
- `tests/test_eigs_lobpcg.c`
- `examples/example_eigs.c`
- `benchmarks/bench_eigs_reuse.c`

Interpretation:

- Day 7 should validate the extracted thick-restart backend primarily through
  the dedicated thick-restart test surface
- the broader eigensolver example/benchmark/public-handle surfaces should act
  as parity checks rather than redesign targets

#### 7. The second batch should also trim the most stale Sprint 21 chronology comments inside the moved block

The thick-restart block still contains dense landing-history narrative such as:

- sprint-day chronology
- “Day X did Y” implementation history
- planning-order notes that no longer help code ownership

Interpretation:

- Day 7 should preserve durable algorithm meaning and invariants
- while moving the thick-restart cluster, it should reduce the most stale
  sprint-history prose inside that moved backend-owned block
- it should not try to normalize the entire remaining `src/sparse_eigs.c`
  comment body in the same batch

## Day 6 Close

Sprint 55 now has an explicit second eigensolver batch design:

- strongest next target:
  - thick-restart restart-state / arrowhead cluster extraction
- preferred new file:
  - `src/sparse_eigs_thick_restart.c`
- retained main-file role:
  - public orchestration plus shared Lanczos-family dispatch
- declaration strategy:
  - keep using `src/sparse_eigs_internal.h`
- proof focus:
  - `tests/test_eigs_thick_restart.c` first, then the broader eigensolver
    parity surfaces

That is enough to start the Day 7 implementation batch from the landed Day 5
reality instead of the original pre-extraction estimate.

# Sprint 55 Day 7 - eigensolver decomposition batch 2

Date: 2026-06-04
Branch: `sprint-55`

## Goal

Land the second bounded `src/sparse_eigs.c` decomposition batch by moving the
thick-restart restart-state / arrowhead / bounded-memory outer-loop backend
cluster into its own permanent source file while preserving the public
eigensolver contract, backend routing, reuse semantics, and the existing proof
surfaces.

## Work completed

### 1. Extracted the thick-restart backend into its own permanent source file

The Day 7 batch moved the thick-restart implementation cluster into:

- `src/sparse_eigs_thick_restart.c`

The extracted implementation set is now:

- `lanczos_restart_state_free(...)`
- `s21_arrowhead_to_tridiag(...)`
- `lanczos_restart_pick_locked(...)`
- `lanczos_restart_state_assemble(...)`
- `lanczos_thick_restart_iterate(...)`
- `s21_build_dense_arrowhead(...)`
- `s21_recompute_residual(...)`
- `s21_thick_restart_outer_loop(...)`

### 2. Kept the shared front-door eigensolver file focused on public/shared ownership

After the extraction, the retained `src/sparse_eigs.c` ownership is now clearer:

- public one-shot and handle entry points
- shared validation and result setup
- AUTO/explicit backend selection
- generic Lanczos helpers
- grow-m Lanczos path
- shared dense Jacobi helper
- shift-invert/shared operator composition
- the LOBPCG dispatch/orchestration call sites

Interpretation:

- the second extraction materially improved source ownership
- `src/sparse_eigs.c` now reads more like the shared eigensolver front door
  instead of a mixed orchestration-plus-backend dump

### 3. Reused the existing internal header instead of opening a second private-header redesign

The batch kept the Sprint 55 Day 6 declaration strategy:

- reused:
  - `src/sparse_eigs_internal.h`

The only header widening needed was to make the shared helper surface explicit
for the extracted thick-restart file:

- `s20_lanczos_starting_vector(...)`
- `s20_spectrum_scale(...)`
- `s20_lift_ritz_vectors(...)`
- `s21_thick_restart_outer_loop(...)`

Interpretation:

- the source split landed without a second taxonomy rewrite
- the batch stayed inside the planned decomposition-first fence

### 4. The main ownership reduction is now measurable

Current post-Day-7 line counts:

- `src/sparse_eigs.c` = `1727`
- `src/sparse_eigs_thick_restart.c` = `934`

Relative to the post-Day-5 baseline:

- `src/sparse_eigs.c`: `2660` -> `1727`

Interpretation:

- Batch 2 is a real decomposition step, not a cosmetic move
- the remaining main eigensolver file is now substantially smaller than the
  Sprint 55 Day 1 baseline

### 5. The first landing needed two real splice fixes before validation completed

The initial extraction exposed two real follow-up issues:

- `src/sparse_eigs_thick_restart.c` still ended with a dangling section-banner
  fragment
- `src/sparse_eigs.c` lost the opening `/*` for the retained LOBPCG banner

Both were fixed before validation. No algorithm change was needed after that;
the remaining work was shared-helper visibility plus build-system integration.

## Touched permanent files

- `CMakeLists.txt`
- `Makefile`
- `src/sparse_eigs.c`
- `src/sparse_eigs_internal.h`
- `src/sparse_eigs_thick_restart.c` (new)

## Validation

Required code-day validation passed:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline also passed:

- `make quality-review-full`

Reviewed truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 248.71 sec`

## Focused follow-ons

The strongest eigensolver parity surfaces also passed:

- `./build/test_eigs` -> `30 / 30`
- `./build/test_eigs_thick_restart` -> `20 / 20`
- `./build/test_eigs_lobpcg` -> `26 / 26`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`

Representative retained behavior:

- `example_eigs`:
  - explicit `LOBPCG` on `bcsstk04` still converged `3 / 3` smallest pairs in
    `62` outer iterations
  - reported residual stayed `8.808e-09`
- `bench_eigs_reuse`:
  - `growm-nos4-k5` -> `1.02x`
  - `thick-bcsstk14-k5` -> `0.97x`
  - `lobpcg-diag40-k3` -> `0.96x`
  - all retained exact eigenvalue parity:
    - `|lambda|max diff = 0.000e+00`

## Day 7 Close

Sprint 55 Day 7 successfully landed the second bounded eigensolver extraction:

- thick-restart backend ownership now lives in `src/sparse_eigs_thick_restart.c`
- `src/sparse_eigs.c` is materially smaller and more orchestration-focused
- the existing internal header strategy was sufficient for the move
- public eigensolver behavior, repeated-run parity, examples, and benchmarks
  remained stable under full validation

That closes the planned Phase 1 eigensolver decomposition pair:

- Day 5: LOBPCG extraction
- Day 7: thick-restart extraction

The remaining Sprint 55 queue can now audit the post-extraction state instead
of debating whether the second ownership split should happen at all.

## Day 8

**Objective:** Reduce `src/sparse_iterative.c` to concrete ownership seams
 before any iterative code movement begins, rank the real extraction targets by
 maintainability value and behavioral risk, and define the first bounded
 iterative extraction boundary from the landed post-Day-7 state.

### Commands Run

1. Re-read the Sprint 55 Day 8 plan item plus the current sprint notes:
   - `sed -n '245,290p' docs/planning/EPIC_5/SPRINT_55/PLAN.md`
   - `tail -n 220 docs/planning/EPIC_5/SPRINT_55/WORKING_NOTES.md`
2. Re-read the Day 7 landed state:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_55/artifacts/day7-eigensolver-decomposition-batch2.md`
3. Re-audit the live iterative implementation and internal declaration seams:
   - `wc -l src/sparse_iterative.c src/sparse_iterative_internal.h src/sparse_iterative_workspace_internal.h benchmarks/bench_iterative_reuse.c tests/test_iterative.c tests/test_minres.c examples/example_iterative.c`
   - `rg -n "^static |^sparse_err_t |^void |^double |^idx_t " src/sparse_iterative.c`
   - `sed -n '1,260p' src/sparse_iterative_internal.h`
   - `sed -n '1,260p' src/sparse_iterative_workspace_internal.h`
4. Re-read the strongest public repeated-run and family-local proof/adoption
   surfaces:
   - `sed -n '1,260p' benchmarks/bench_iterative_reuse.c`
   - `sed -n '2660,2865p' tests/test_iterative.c`
   - `sed -n '1,260p' tests/test_minres.c`
   - `sed -n '1,220p' examples/example_iterative.c`

### Day 8 Findings

#### 1. The iterative decomposition problem now reduces cleanly to six named ownership seams

The live `src/sparse_iterative.c` body no longer needs to be treated as one
generic large-file problem. It now separates into:

1. public handle orchestration and handle-growth helpers
2. shared staging / residual-history / reporting utilities
3. `CG` execution path
4. `GMRES` execution path
5. `MINRES` execution path
6. block-wrapper and `BiCGSTAB` family-local compatibility surfaces

Interpretation:

- Sprint 55 no longer needs a size-first decomposition strategy for the
  iterative side
- the right next move is an ownership-first extraction that leaves the shared
  public front-door layer intact

#### 2. The strongest first extraction target is `MINRES`, not the largest remaining code region

The most credible first iterative extraction target is the `MINRES` family:

- internal reusable-workspace preparation already exists:
  - `sparse_iter_workspace_prepare_minres(...)`
- public repeated-run handle support already exists:
  - `sparse_iter_handle_prepare_minres(...)`
  - `sparse_solve_minres_with_handle(...)`
- family-local numerical proof already exists:
  - `tests/test_minres.c`
- public repeated-run proof already exists:
  - `tests/test_iterative.c`

Interpretation:

- `MINRES` is already a coherent ownership band
- extracting it would reduce `src/sparse_iterative.c` without reopening the
  public handle contract or the one-shot/default caller story
- this is a better maintainability target than a mechanically larger but more
  entangled split

#### 3. `GMRES` remains important, but it is a worse first split than `MINRES`

The live `GMRES` cluster is materially larger and clearly valuable, but it is
also more entangled with:

- matrix-free adapters
- restart-state orchestration
- public handle reuse path
- block-column wrapper reuse

Interpretation:

- `GMRES` remains a strong later extraction candidate
- it is not the best first move for Sprint 55 Phase 1 because it carries more
  orchestration coupling risk than `MINRES`

#### 4. `BiCGSTAB` is a real seam, but it should stay outside the first extraction boundary

`BiCGSTAB` still sits on a distinct family-local reusable-workspace model
rather than the public iterative handle owner:

- `sparse_bicgstab_internal.h`
- `bicgstab_workspace_t`

And Sprint 54 already fixed its public repeated-run support boundary as:

- supported handles:
  - `CG`
  - `GMRES`
  - `MINRES`
- excluded handle families:
  - `BiCGSTAB`
  - block iterative workflows

Interpretation:

- `BiCGSTAB` is not the right first extraction target for Sprint 55 Phase 1
- extracting it early would mix decomposition work with a less unified
  ownership model and a consciously excluded public-handle family

#### 5. The block wrappers are a secondary seam, not the primary Sprint 55 target

The block portion of `src/sparse_iterative.c` is real, but the dominant shape
is still wrapper-oriented:

- shared per-column orchestration
- thin family-local block adapters for `CG`, `GMRES`, `MINRES`, and
  `BiCGSTAB`

Interpretation:

- a future block-wrapper extraction could improve locality
- doing that first would not reduce the main reasoning burden as much as
  extracting one numerically coherent repeated-run family like `MINRES`

#### 6. The first iterative extraction boundary is now explicit enough for Day 9

The recommended first iterative extraction target is:

- new owned implementation file:
  - `src/sparse_iterative_minres.c`

Move target set:

- `sparse_solve_minres_with_workspace_internal(...)`
- `sparse_solve_minres(...)`
- `sparse_solve_minres_with_handle(...)`
- `solve_block_minres_column(...)`
- `sparse_minres_solve_block(...)`

Retain in `src/sparse_iterative.c`:

- public handle init/free and growth helpers
- shared staging/residual/reporting utilities
- `CG`
- `GMRES`
- block-shared wrapper scaffolding
- `BiCGSTAB`

Interpretation:

- this would be a real ownership improvement
- it would not require public API changes
- it would preserve the Sprint 54 steady-state support fence

#### 7. The Day 8 non-goal fence is now explicit

Day 8 also rules out several weaker split strategies:

- do not split by arbitrary line ranges
- do not start with tiny utility-only moves that leave the main reasoning load
  unchanged
- do not reopen the Sprint 54 public repeated-run support boundary
- do not treat `BiCGSTAB` extraction as equivalent to the supported
  handle-backed families
- do not combine the first iterative extraction with a broad comment-taxonomy
  rewrite

Interpretation:

- the next implementation batch should be maintainability-shaped, not purely
  mechanical
- the main success criterion is cleaner ownership with stable solver behavior

## Day 8 Close

Sprint 55 now has an explicit iterative decomposition map:

- named ownership seams inside `src/sparse_iterative.c`
- ranked extraction targets
- a clear first bounded extraction recommendation:
  - `MINRES`
- an explicit defer list:
  - `GMRES` later
  - block wrappers later
  - `BiCGSTAB` outside the first extraction fence

That is enough to begin Day 9 from a real implementation boundary instead of a
generic large-file reduction goal.

## Day 9

**Objective:** Freeze the first iterative extraction boundary at file/helper
 level, define the exact private-declaration and invariants strategy for the
 first `MINRES` move, and record the landing checklist before any iterative
 implementation files are edited.

### Commands Run

1. Re-read the Sprint 55 Day 9 plan item plus the landed Day 8 state:
   - `sed -n '290,340p' docs/planning/EPIC_5/SPRINT_55/PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_55/artifacts/day8-sparse-iterative-seam-audit.md`
   - `tail -n 260 docs/planning/EPIC_5/SPRINT_55/WORKING_NOTES.md`
2. Re-read the current iterative private-header surfaces:
   - `sed -n '1,220p' src/sparse_iterative_internal.h`
   - `sed -n '1,260p' src/sparse_iterative_workspace_internal.h`
3. Re-read the exact `MINRES` implementation band and the nearby block helper:
   - `sed -n '1240,1388p' src/sparse_iterative.c`
   - `sed -n '1360,1795p' src/sparse_iterative.c`
4. Reconfirm the current live size of the main iterative implementation and
   proof/adoption surfaces:
   - `wc -l src/sparse_iterative.c src/sparse_iterative_internal.h src/sparse_iterative_workspace_internal.h tests/test_iterative.c tests/test_minres.c benchmarks/bench_iterative_reuse.c examples/example_iterative.c`
5. Re-scan the touched iterative files for stale sprint-history narrative that
   Day 10 should trim only inside the moved `MINRES` ownership block:
   - `rg -n "Sprint|Day [0-9]+" src/sparse_iterative.c src/sparse_iterative_internal.h src/sparse_iterative_workspace_internal.h`

### Day 9 Findings

#### 1. The first iterative extraction should be narrower than the Day 8 sketch

Day 8 was correct about the first family target:

- `MINRES`

But the exact Day 10 landing boundary should be tighter than the original
sketch:

- move the core `MINRES` solver family
- keep the block-wrapper orchestration in `src/sparse_iterative.c`

Interpretation:

- the first iterative batch should optimize for clean ownership with minimal
  helper widening
- moving the block wrapper in the same batch would force the generic
  block-column helper to become an extra cross-file seam too early

#### 2. The exact Day 10 file split is now explicit

Recommended new file:

- `src/sparse_iterative_minres.c`

Move into that new file:

- `sparse_solve_minres_with_workspace_internal(...)`
- `sparse_solve_minres(...)`
- `sparse_solve_minres_with_handle(...)`

Retain in `src/sparse_iterative.c`:

- public handle init/free and growth helpers
- shared staging / residual-history / reporting helpers
- `CG`
- `GMRES`
- shared block-column orchestration:
  - `iter_block_column_solver_fn`
  - `solve_block_independent_columns(...)`
- block wrapper entry points, including:
  - `solve_block_minres_column(...)`
  - `sparse_minres_solve_block(...)`
- `BiCGSTAB`

Interpretation:

- the moved file owns the coherent scalar/handle `MINRES` family
- the retained file keeps the shared front-door and block-wrapper scaffolding
- this is a cleaner Phase 1 boundary than moving both family code and shared
  block orchestration at once

#### 3. The private-header strategy should widen the existing internal header, not add a new one

The first iterative extraction does not need a new private-header taxonomy.

Keep using:

- `src/sparse_iterative_internal.h`
- `src/sparse_iterative_workspace_internal.h`

Expected Day 10 declaration widening:

- add:
  - `sparse_solve_minres_with_workspace_internal(...)`

Do not add:

- `src/sparse_iterative_minres_internal.h`

Interpretation:

- the existing iterative internal header already serves as the workspace-backed
  internal-entry surface
- adding `MINRES` there matches the current `CG` / `GMRES` internal pattern
- this keeps the first batch ownership-focused instead of taxonomy-focused

#### 4. The invariants for the first iterative move are now fixed

Day 10 must preserve all of the following:

- public repeated-run handle semantics for:
  - `CG`
  - `GMRES`
  - `MINRES`
- one-shot/default caller behavior for `sparse_solve_minres(...)`
- handle growth and reuse behavior for `sparse_solve_minres_with_handle(...)`
- workspace typing and capacity ownership through:
  - `sparse_iter_workspace_prepare_minres(...)`
- result/reporting behavior:
  - `iterations`
  - `residual_norm`
  - `converged`
  - `stagnated`
  - `breakdown`
- no benchmark/example contract drift on:
  - `bench_iterative_reuse`
  - `example_ic_minres`

Interpretation:

- the first iterative extraction is an ownership change, not a behavior change
- Day 10 should treat any observed parity drift as a bug, not as a permitted
  side effect of the split

#### 5. The first iterative batch now has an explicit comment-cleanup policy

The moved `MINRES` block currently carries stale sprint-history narrative such
as:

- `Sprint 29 Day 7` progress/cancel comments

Day 10 should:

- preserve durable algorithm commentary
- preserve comments that explain numerical invariants or convergence checks
- remove or rewrite stale sprint-history narration only inside the touched
  moved `MINRES` block

Day 10 should not:

- try to normalize the entire remaining `src/sparse_iterative.c` comment body
- mix the first extraction with a whole-file comment-style rewrite

Interpretation:

- comment cleanup remains part of the maintainability goal
- but it must stay bounded to the moved ownership band

#### 6. The first iterative landing checklist is now concrete

Expected Day 10 touched permanent files:

- `src/sparse_iterative.c`
- `src/sparse_iterative_minres.c` (new)
- `src/sparse_iterative_internal.h`
- `Makefile`
- `CMakeLists.txt`

Primary proof surfaces:

- `tests/test_minres.c`
- `tests/test_iterative.c`

Secondary parity surfaces:

- `benchmarks/bench_iterative_reuse.c`
- `examples/example_iterative.c`
- `build/example_ic_minres`

Required validation:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Interpretation:

- Day 10 now has an explicit minimal file set
- the proof surface priority is fixed before code movement begins

## Day 9 Close

Sprint 55 now has an explicit first iterative implementation design:

- new file:
  - `src/sparse_iterative_minres.c`
- move only the scalar/handle `MINRES` family in Batch 1
- keep block-wrapper scaffolding in the main iterative file
- widen the existing internal header instead of adding a new one
- preserve the full Sprint 54 repeated-run support fence and proof surfaces
- trim stale sprint-history comments only inside the moved `MINRES` block

That is enough to begin Day 10 from a precise ownership map and landing
checklist instead of refining the boundary mid-implementation.

## Day 10

**Objective:** Land the first bounded `src/sparse_iterative.c` extraction by
 moving the scalar/handle `MINRES` family into its own permanent source file,
 keeping block-wrapper scaffolding in the retained main iterative file, and
 preserving the full repeated-run/public-handle proof and reviewed validation
 contract.

### Commands Run

1. Re-read the Day 9 landing design plus the live `MINRES` band:
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_55/artifacts/day9-iterative-decomposition-batch1-design.md`
   - `sed -n '1360,1795p' src/sparse_iterative.c`
   - `rg -n "sparse_eigs_lobpcg.c|sparse_eigs_thick_restart.c|sparse_iterative.c" Makefile CMakeLists.txt`
2. Re-audit the shared helper dependencies that the moved `MINRES` block still
   needed:
   - `sed -n '1,260p' src/sparse_iterative.c`
   - `rg -n "static (inline )?(double|void|int|sparse_err_t) (vec_|matvec_|stag_|reshist_|iter_report|s29_iter_now_s)" src/sparse_iterative.c`
   - `rg -n "cg_defaults|gmres_defaults|s49_iter_handle_ensure|s29_iter_now_s|stag_tracker_t|reshist_t|iter_report" src/sparse_iterative.c src/sparse_iterative_internal.h`
3. Land the first iterative extraction batch:
   - split the scalar/handle `MINRES` family into `src/sparse_iterative_minres.c`
   - widen `src/sparse_iterative_internal.h`
   - update `Makefile` and `CMakeLists.txt`
4. Run the required code-day gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
5. Run focused iterative follow-ons and ownership checks:
   - `./build/test_iterative`
   - `./build/test_minres`
   - `./build/example_ic_minres`
   - `./build/bench_iterative_reuse`
   - `wc -l src/sparse_iterative.c src/sparse_iterative_minres.c src/sparse_iterative_internal.h`

### Day 10 Findings

#### 1. The first iterative extraction landed as the intended narrow `MINRES` batch

The new owned iterative file is now:

- `src/sparse_iterative_minres.c`

Moved ownership:

- `sparse_solve_minres_with_workspace_internal(...)`
- `sparse_solve_minres(...)`
- `sparse_solve_minres_with_handle(...)`

Retained in `src/sparse_iterative.c`:

- public handle init/free and growth helpers
- shared staging / residual-history / reporting helpers
- `CG`
- `GMRES`
- shared block-column scaffolding
- block MINRES wrappers
- `BiCGSTAB`

Interpretation:

- the first iterative split stayed inside the Day 9 fence
- the main file kept the shared front-door/block-wrapper role
- the new file owns the coherent scalar/handle `MINRES` family

#### 2. The shared-helper/header widening stayed minimal but real

The first extraction exposed one genuine shared-helper seam:

- `MINRES` still needed:
  - `s29_iter_now_s(...)`
  - `s49_iter_handle_ensure(...)`
  - `stag_*`
  - `reshist_*`
  - `iter_report(...)`

The landed solution stayed within the planned header strategy:

- widened:
  - `src/sparse_iterative_internal.h`
- did not add:
  - a new private `MINRES` header

Interpretation:

- Batch 1 remained ownership-focused rather than taxonomy-focused
- the internal iterative header now carries the shared helper declarations
  needed for split implementation ownership

#### 3. The ownership reduction is now measurable

Current post-Day-10 line counts:

- `src/sparse_iterative.c` = `1985`
- `src/sparse_iterative_minres.c` = `308`
- `src/sparse_iterative_internal.h` = `79`

Relative to the pre-Day-10 state:

- `src/sparse_iterative.c`: `2377` -> `1985`

Interpretation:

- this is a real decomposition step, not a comment-only cleanup pass
- the retained main iterative file is materially smaller and more
  orchestration-focused than the Sprint 55 Day 1 baseline

#### 4. The moved `MINRES` block also dropped its stale sprint-history narration

The moved `MINRES` ownership band no longer carries the stale
`Sprint 29 Day 7` progress/cancel narrative inside the extracted backend body.

Interpretation:

- the comment cleanup stayed bounded to the moved ownership band
- durable algorithm commentary was preserved
- the batch did not turn into a whole-file comment rewrite

#### 5. The full required validation and reviewed baseline both stayed green

Required code-day validation passed:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline also passed:

- `make quality-review-full`

Reviewed truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real)` = `244.60 sec`

Interpretation:

- the iterative extraction did not disturb the maintained reviewed parity path
- the Day 10 batch preserved the repo’s strongest local reviewed baseline

#### 6. The strongest iterative follow-ons remained stable after the split

Focused follow-ons passed:

- `./build/test_iterative` -> `79 / 79`
- `./build/test_minres` -> `43 / 43`
- `./build/example_ic_minres`
- `./build/bench_iterative_reuse`

Representative direct results:

- `example_ic_minres`:
  - `MINRES` on the `42x42` KKT system converged in `39` iterations
  - Jacobi-`MINRES` converged in `26` iterations
- `bench_iterative_reuse`:
  - `cg-tridiag-300` -> `1.05x`
  - `gmres-unsym-220` -> `1.04x`
  - `minres-kkt-42` -> `1.11x`

Interpretation:

- the public repeated-run iterative handle story stayed intact
- the split did not introduce parity drift on the strongest `MINRES` proof and
  adoption surfaces

## Day 10 Close

Sprint 55 Day 10 successfully landed the first bounded iterative extraction:

- `MINRES` scalar/handle ownership now lives in `src/sparse_iterative_minres.c`
- the retained `src/sparse_iterative.c` is smaller and more
  orchestration-focused
- the existing internal header strategy was sufficient after one bounded shared
  helper widening
- block wrappers stayed in the main file as planned
- the full required validation and reviewed parity path remained green

That closes the planned Batch 1 implementation step without reopening the
Sprint 54 repeated-run solver support boundary.

# Sprint 55 Day 11 - historical comment reduction sweep

Date: 2026-06-04
Branch: `sprint-55`

## Goal

Do the bounded historical-comment cleanup planned for Day 11:

- re-scan the Sprint 55 touched permanent implementation files
- remove stale sprint/day chronology comments
- preserve durable algorithm, ownership, and invariant commentary
- rerun the required code-day validation gate

## Files reviewed and touched

The sweep stayed inside the Sprint 55 touched implementation set:

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_eigs_internal.h`
- `src/sparse_eigs_thick_restart.c`

No public headers, tests, benchmarks, examples, or build wiring changed.

## What changed

The patch was comment-only and focused on permanent maintainability:

- removed stale `Sprint ... Day ...` narrative from the iterative/eigensolver
  implementation files touched earlier in Sprint 55
- rewrote those comments as durable explanations of:
  - why `_POSIX_C_SOURCE` is requested
  - what the progress callbacks mean
  - what the shared Lanczos/MGS helpers own
  - what the grow-m, thick-restart, arrowhead, and LOBPCG sections are
    responsible for
  - what the restart-state and shift-invert seams guarantee
- kept algorithm commentary that still helps future maintainers reason about:
  - recurrence invariants
  - workspace ownership
  - spectrum / residual semantics
  - backend dispatch boundaries

The intended truthfulness check is now clean:

- `rg -n "Sprint|Day [0-9]+" src/sparse_eigs.c src/sparse_eigs_internal.h src/sparse_eigs_thick_restart.c src/sparse_iterative.c`
  returned no matches after the cleanup

## Measured outcome

This was a real reduction sweep rather than churn:

- `git diff --stat`:
  - `src/sparse_eigs.c` = `429` changed lines
  - `src/sparse_eigs_internal.h` = `106` changed lines
  - `src/sparse_eigs_thick_restart.c` = `110` changed lines
  - `src/sparse_iterative.c` = `12` changed lines
- total patch shape:
  - `217` insertions
  - `440` deletions

Current post-Day-11 line counts:

- `src/sparse_eigs.c` = `1534`
- `src/sparse_eigs_internal.h` = `631`
- `src/sparse_eigs_thick_restart.c` = `914`
- `src/sparse_iterative.c` = `1985`

Interpretation:

- the sweep materially reduced stale narrative in the extracted eigensolver
  ownership bands
- the retained iterative main file kept its Day 10 ownership size while losing
  the remaining sprint-history comments
- the cleanup stayed within the Day 11 maintainability fence and did not reopen
  decomposition scope

## Validation

Required Day 11 code-day validation passed:

- `make format`
- `make lint`
- `make test`

Interpretation:

- the comment-only patch did not disturb the maintained compile/lint/test
  contract
- the touched implementation files remained formatting-clean and tool-clean

## Day 11 Close

Sprint 55 Day 11 successfully completed the planned historical comment
reduction sweep:

- stale sprint/day narrative is gone from the Sprint 55 touched permanent
  implementation files
- durable algorithm and ownership commentary remains in place
- the patch stayed comment-only and bounded
- the required Day 11 validation gate remained green

That leaves Day 12 free to audit the decomposed source ownership state rather
than to clean up leftover Sprint 55 implementation narration.

# Sprint 55 Day 12 - post-landing compatibility audit

Date: 2026-06-04
Branch: `sprint-55`

## Goal

Audit the landed Sprint 55 branch against the preserved solver/lifecycle
compatibility fence and confirm that the decomposition work improved ownership
rather than merely moving code across files.

## Audit inputs used

Primary public and caller-facing surfaces rechecked:

- `README.md`
- `examples/README.md`
- `docs/tutorial.md`
- `benchmarks/README.md`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

Primary implementation and ownership surfaces rechecked:

- `src/sparse_eigs.c`
- `src/sparse_eigs_lobpcg.c`
- `src/sparse_eigs_thick_restart.c`
- `src/sparse_eigs_internal.h`
- `src/sparse_iterative.c`
- `src/sparse_iterative_minres.c`
- `src/sparse_iterative_internal.h`

Build-wiring confirmation surfaces:

- `Makefile`
- `CMakeLists.txt`

## Compatibility-fence audit

### 1. No public API redesign surfaced

The public headers and README still describe the same preserved public repeated-
run solver boundary:

- iterative repeated-run handles remain intentionally bounded to:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver repeated-run handles remain intentionally bounded to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- explicit retained exclusions still read as exclusions, not hidden drift:
  - `BiCGSTAB`
  - block iterative workflows

Interpretation:

- Sprint 55 moved implementation ownership only
- it did not widen or narrow the public solver support boundary fixed in Sprint
  54

### 2. No behavior-visible lifecycle change surfaced

The caller-facing docs still describe the same lifecycle semantics:

- one-shot solver APIs remain first-class
- repeated-run handles remain opt-in paths
- handle reuse preserves allocation capacity, not stale numerical iteration
  state
- examples remain intentionally one-shot-first

Interpretation:

- Sprint 55 did not introduce any new public lifecycle model
- the decomposition work stayed underneath the already-validated behavior

### 3. Examples, benchmarks, and tutorial wording stayed aligned

The high-signal non-header surfaces still agree with the preserved contract:

- `examples/README.md`
  - still states the bounded iterative/eigensolver handle sets explicitly
- `benchmarks/README.md`
  - still treats the reuse drivers as narrow proof surfaces rather than general
    solver bake-offs
- `docs/tutorial.md`
  - still names the iterative family as `CG`, `GMRES`, `MINRES`

Interpretation:

- no public-documentation drift appeared while the code was being split

## Ownership-gain audit

### 1. The eigensolver split is now materially real

Current file ownership shape:

- retained orchestration/shared file:
  - `src/sparse_eigs.c` = `1534`
- extracted backend-owned files:
  - `src/sparse_eigs_lobpcg.c` = `401`
  - `src/sparse_eigs_thick_restart.c` = `914`
- shared private declaration surface:
  - `src/sparse_eigs_internal.h` = `631`

Relative to the Day 1 baseline:

- `src/sparse_eigs.c`: `3233` -> `1534`

Interpretation:

- this is no longer one large eigensolver file with conceptual bands
- the LOBPCG and thick-restart backends now own real permanent source files
- the retained main file is now clearly the public/shared front door

### 2. The iterative split is also materially real

Current file ownership shape:

- retained orchestration/shared file:
  - `src/sparse_iterative.c` = `1985`
- extracted backend-owned file:
  - `src/sparse_iterative_minres.c` = `308`
- shared private declaration surface:
  - `src/sparse_iterative_internal.h` = `79`

Relative to the Day 1 baseline:

- `src/sparse_iterative.c`: `2377` -> `1985`

Interpretation:

- `MINRES` scalar/handle ownership now lives in its own permanent file
- the retained iterative main file is smaller and more orchestration-focused
- the split improved maintainability without reopening the public handle model

### 3. The build system now reflects the same ownership reality

Both build surfaces explicitly include the extracted files:

- `Makefile`
  - `src/sparse_iterative_minres.c`
  - `src/sparse_eigs_lobpcg.c`
  - `src/sparse_eigs_thick_restart.c`
- `CMakeLists.txt`
  - `src/sparse_iterative_minres.c`
  - `src/sparse_eigs_lobpcg.c`
  - `src/sparse_eigs_thick_restart.c`

Interpretation:

- the decomposition is not local-only or tool-path-only
- the maintained Makefile/CMake ownership surfaces agree on the landed split

## Residual follow-up queue

The remaining large-source follow-ons are now explicit and future-facing rather
than Sprint 55 blockers:

- later iterative decomposition candidates:
  - `GMRES`
  - block-wrapper scaffolding
- later eigensolver cleanup/decomposition candidates:
  - additional trimming inside the retained `src/sparse_eigs.c`
  - possible future private-header taxonomy cleanup if it buys clarity
- still intentionally deferred:
  - `BiCGSTAB` extraction as a first-class public repeated-run handle topic
  - broad public API redesign
  - large doc/tutorial rewrites unrelated to ownership

## Day 13 checklist

Final validation should run from the landed Day 12 state:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Targeted Sprint 55 follow-ons:

- `./build/test_iterative`
- `./build/test_minres`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/example_iterative`
- `./build/example_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

## Day 12 Close

Sprint 55 Day 12 confirms the landed branch still matches the preserved public
solver/lifecycle fence:

- no public API redesign surfaced
- no solver support-boundary drift surfaced
- no behavior-visible repeated-run lifecycle change surfaced
- the source splits are now explicit and defensible ownership improvements

No blocker-level drift remains before Day 13 validation.
