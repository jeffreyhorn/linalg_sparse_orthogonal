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
