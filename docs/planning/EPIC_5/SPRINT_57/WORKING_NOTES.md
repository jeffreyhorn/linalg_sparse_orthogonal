# Sprint 57 Working Notes

## Day 1

**Objective:** Turn the Sprint 57 project-plan scope plus the Sprint 56
validated decomposition close state into a concrete giant-test and
lifecycle-regression starting point by confirming the preserved reviewed
baseline, naming the Sprint 57 maintainability and proof workstreams
explicitly, and defining the authoritative large-test, benchmark, example,
and caller-surface hotspots before any refactor or regression expansion
begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 57 project-plan source and the new sprint plan:
   - `sed -n '250,277p' docs/planning/EPIC_5/PROJECT_PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_57/PLAN.md`
3. Re-read the strongest inherited Sprint 56 closeout sources:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_56/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_56/RETROSPECTIVE.md`
4. Re-read the Epic 5 review/todo guidance for the giant-test and
   lifecycle-regression queue:
   - `rg -n "Giant-Test Refactor|giant-test|test refactor|lifecycle regression|factor-many|test_chol_csc|test_ldlt_csc|test_svd|test_integration" docs/planning/EPIC_5/reviews docs/planning/EPIC_5/PROJECT_PLAN.md docs/planning/EPIC_5/SPRINT_56/RETROSPECTIVE.md`
   - `sed -n '135,165p' docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md`
   - `sed -n '145,170p' docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
5. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
6. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
7. Measure the live large-test ranking:
   - `python3 - <<'PY' ... Path('tests').glob('test_*.c') ... PY`
8. Measure the main Sprint 57 proof and caller-facing hotspot surfaces:
   - `wc -l tests/test_chol_csc.c tests/test_svd.c tests/test_ldlt_csc.c tests/test_qr.c tests/test_iterative.c tests/test_integration.c tests/test_etree.c benchmarks/bench_refactor_csc.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c examples/example_analysis.c examples/example_iterative.c examples/example_eigs.c README.md docs/maintainer_guide.md`

### Day 1 Findings

#### 1. Sprint 57 starts from a validated decomposition baseline, not from renewed public lifecycle or solver-design work

The inherited starting state is already explicit and stable:

- Sprint 56 closed with:
  - bounded CSC direct-solver and SVD decomposition complete enough to reduce:
    - `src/sparse_ldlt_csc.c`: `2723 -> 2127`
    - `src/sparse_chol_csc.c`: `2194 -> 1532`
    - `src/sparse_svd.c`: `1728 -> 1319`
  - no public header/API redesign
  - no behavior-visible repeated-run lifecycle drift
  - no solver-family support-boundary drift
- Sprint 56 also closed from:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- the inherited caller-facing contract remains unchanged:
  - one-shot solver APIs remain first-class entry points
  - repeated direct-solver lifecycle support remains the validated Sprint 50-53
    shape
  - repeated-run iterative/eigensolver handles remain the validated Sprint 54
    shape

Interpretation:

- Sprint 57 is not a public design sprint
- Sprint 57 is not a validation-recovery sprint
- Sprint 57 is a bounded test-maintainability and regression-proof sprint

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible throughout giant-test and regression work

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

Interpretation:

- Sprint 57 should keep using the exact `strongest local reviewed baseline`
  phrasing
- substantial giant-test refactor and regression batches should continue
  treating the reviewed CMake count and Makefile/CMake parity contract as the
  main truthfulness anchors

#### 3. The Epic 5 review queue is now concentrated in giant tests rather than production implementation files

The project plan and Epic 5 review/todo notes already fixed the next
maintainability problem:

- split or helper-extract the largest test binaries:
  - `test_chol_csc`
  - `test_svd`
  - `test_ldlt_csc`
  - `test_qr`
  - `test_etree`
  - `test_iterative`
- add direct lifecycle coverage for the final public direct-solver model
- keep benchmark and example parity checks where they prove caller stories

The live repo state confirms that the queue is still current:

- `tests/test_chol_csc.c` = `4643`
- `tests/test_svd.c` = `3746`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_qr.c` = `3197`
- `tests/test_iterative.c` = `2993`
- `tests/test_etree.c` = `2962`
- `tests/test_integration.c` = `1803`

Interpretation:

- Sprint 57 should treat giant-test maintainability as the dominant remaining
  Epic 5 ownership problem
- several core test binaries are now larger than most retained production
  files, which makes the maintenance cost real rather than hypothetical

#### 4. The real Sprint 57 queue is split between giant-test maintainability and final lifecycle/factor-many proof strengthening

The Sprint 57 plan items and live repo state reduce cleanly to six bounded
work classes:

1. large-test audit
2. direct-solver test refactor batch
3. iterative / eigensolver test refactor batch
4. lifecycle regression expansion
5. factor-many / compatibility regression expansion
6. validation and closeout

Interpretation:

- Sprint 57 should improve ownership and readability in the biggest proof
  surfaces before widening regression coverage indiscriminately
- the lifecycle and factor-many expansion work should be additive proof work,
  not feature-first expansion disguised as testing

#### 5. The live hotspot map is concentrated enough to name directly before refactor work starts

The strongest Sprint 57 proof and caller-facing surfaces are now explicit:

- giant tests:
  - `tests/test_chol_csc.c` = `4643`
  - `tests/test_svd.c` = `3746`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_qr.c` = `3197`
  - `tests/test_iterative.c` = `2993`
  - `tests/test_etree.c` = `2962`
  - `tests/test_integration.c` = `1803`
- benchmark surfaces:
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_iterative_reuse.c` = `370`
  - `benchmarks/bench_eigs_reuse.c` = `253`
- caller-facing example surfaces:
  - `examples/example_eigs.c` = `285`
  - `examples/example_analysis.c` = `210`
  - `examples/example_iterative.c` = `144`
- summary/truthfulness docs:
  - `README.md` = `987`
  - `docs/maintainer_guide.md` = `294`

Interpretation:

- the strongest direct-solver maintainability and proof seams sit in
  `test_chol_csc`, `test_ldlt_csc`, and `test_integration`
- the strongest solver-family test seams sit in `test_iterative` and the SVD
  hotspot remains large enough that it can compete with iterative/eigensolver
  work for refactor priority
- benchmark and example parity checks should stay visible because they still
  prove caller stories more directly than raw helper-level coverage

#### 6. The inherited public compatibility fence gives Sprint 57 a clean non-goal boundary

The inherited fence remains:

- no public API redesign
- no reopening the direct-solver lifecycle contract
- no reopening the repeated-run iterative/eigensolver support boundary
- no feature-first solver expansion disguised as test work
- preserve reviewed validation and truthfulness anchors

Interpretation:

- Sprint 57 should refactor and expand proof underneath the already-validated
  public surfaces
- helper extraction and regression additions are the success criteria, not new
  user-visible capability

#### 7. Factor-many and lifecycle coverage should now be treated as final proof-shaping work, not as an abstract backlog

The inherited Sprint 51-56 work already established:

- public analyze-once / factor-many precedent
- indefinite CSC factor-many follow-through
- repeated-run iterative/eigensolver support boundaries

Interpretation:

- Sprint 57 should focus on high-signal final regression cases that directly
  exercise:
  - final public direct lifecycle surfaces
  - one-shot compatibility expectations
  - factor-many workflows
  - repeated-run caller stories already taught by examples and benchmarks
- the right coverage shape is behavior-level and caller-shaped, not a broad
  helper-driven test explosion

## Day 1 Close

Sprint 57 now has an explicit starting point:

- preserved reviewed baseline
- inherited validated public-contract fence from Sprint 56
- named giant-test and caller-facing hotspot surfaces
- clear maintainability-first and proof-expansion workstreams
- explicit non-goal fence against public API or feature expansion

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening Sprint 50-56 public contract decisions.
