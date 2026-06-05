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
