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

## Day 2

**Objective:** Reconfirm the maintained reviewed baseline and truthfulness
anchors Sprint 57 must preserve, then define the smallest authoritative
validation boundary for the later giant-test refactor and lifecycle/factor-many
regression days and the high-signal rerun set those code-touch batches should
use.

### Commands Run

1. Re-read the Sprint 57 Day 2 plan item and the current sprint notes:
   - `sed -n '78,130p' docs/planning/EPIC_5/SPRINT_57/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_57/WORKING_NOTES.md`
2. Reconfirm the maintained reviewed CMake truthfulness anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
3. Reconfirm the maintained reviewed wrapper authority surface:
   - `make -n quality-review-full`
4. Re-read the live quality-contract wording sources:
   - `rg -n "strongest local reviewed baseline|quality-review-full|quality-review-cmake|deadcode-check" README.md docs/maintainer_guide.md Makefile .github/workflows -g '!build'`
5. Reconfirm the main Sprint 57 follow-on binaries already present in the
   build tree:
   - `ls build/test_chol_csc build/test_ldlt_csc build/test_svd build/test_qr build/test_iterative build/test_integration build/bench_refactor_csc build/bench_iterative_reuse build/bench_eigs_reuse build/example_analysis`
6. Measure the live size of those main proof/adoption surfaces:
   - `wc -l tests/test_chol_csc.c tests/test_ldlt_csc.c tests/test_svd.c tests/test_qr.c tests/test_iterative.c tests/test_integration.c benchmarks/bench_refactor_csc.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c examples/example_analysis.c`

### Day 2 Findings

#### 1. The strongest local reviewed baseline and truthfulness anchors remain exact

The maintained Sprint 57 baseline remains:

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

- Sprint 57 should keep using the exact `strongest local reviewed baseline`
  phrasing
- the reviewed CMake count and Makefile/CMake parity contract remain the main
  authoritative truthfulness anchors for later giant-test and regression days

#### 2. The later test-refactor and regression code-day gate is simple and should stay explicit

The mandatory gate for later `*.c` / `*.h` test-refactor and regression work
remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial test-surface changes remains:

- `make quality-review-full`

Interpretation:

- docs-only audit/design/summary days do not need the full code-day gate
- substantial giant-test refactor or regression-expansion batches should run
  both the direct gate and the stronger reviewed baseline path

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

- Sprint 57 does not need to reopen any quality-contract wording work on Day 2
- the maintained reviewed baseline language is already stable enough to carry
  forward unchanged

#### 4. The high-signal Sprint 57 rerun set is now fixed explicitly from the live build tree

The main Sprint 57 follow-on binaries already present in `build/` are:

- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/test_svd`
- `./build/test_qr`
- `./build/test_iterative`
- `./build/test_integration`
- `./build/bench_refactor_csc`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`
- `./build/example_analysis`

Interpretation:

- Sprint 57 can keep its rerun set focused on the direct-solver, giant-test,
  and repeated-run caller-story surfaces actually touched by the planned work
- no broader default rerun set is needed on Day 2

#### 5. The proof and caller-story surfaces are large enough that parity preservation is part of the refactor work itself

The live proof/adoption surface sizes are now:

- `tests/test_chol_csc.c` = `4643`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_svd.c` = `3746`
- `tests/test_qr.c` = `3197`
- `tests/test_iterative.c` = `2993`
- `tests/test_integration.c` = `1803`
- `benchmarks/bench_refactor_csc.c` = `611`
- `benchmarks/bench_iterative_reuse.c` = `370`
- `benchmarks/bench_eigs_reuse.c` = `253`
- `examples/example_analysis.c` = `210`

Interpretation:

- Sprint 57 giant-test refactor work should assume that proof-surface
  legibility and caller-story parity matter alongside line-count reduction
- the rerun set is a real guard against accidental behavior drift while helper
  ownership and regression coverage evolve

## Day 2 Close

Sprint 57 now has an explicit validation boundary:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit code-day gate
- explicit stronger reviewed-baseline default
- authoritative giant-test and caller-story rerun set from the live build tree

That is enough to move to the Day 3 direct-solver giant-test audit without any
remaining ambiguity around validation expectations.

## Day 3

**Objective:** Reduce the strongest direct-solver giant tests to a concrete
helper and proof-group extraction map by separating the live ownership bands,
ranking the real bounded seam options, and fixing the strongest first refactor
target before any code movement begins.

### Commands Run

1. Re-read the Sprint 57 Day 3 plan item and current sprint notes:
   - `sed -n '130,180p' docs/planning/EPIC_5/SPRINT_57/PLAN.md`
   - `sed -n '1,520p' docs/planning/EPIC_5/SPRINT_57/WORKING_NOTES.md`
2. Inventory the helper/test function map for the direct-solver giant tests:
   - `python3 - <<'PY' ... test_chol_csc.c / test_ldlt_csc.c / test_integration.c ... PY`
3. Inspect the live section/comment boundaries inside those tests:
   - `python3 - <<'PY' ... section/comment scan ... PY`
4. Inspect the runner ordering and grouped `RUN_TEST(...)` blocks:
   - `rg -n "RUN_TEST|main\\(" tests/test_chol_csc.c tests/test_ldlt_csc.c tests/test_integration.c`
   - `sed -n '4300,4475p' tests/test_chol_csc.c`
   - `sed -n '3520,3695p' tests/test_ldlt_csc.c`
   - `sed -n '1720,1815p' tests/test_integration.c`
5. Re-read the top-of-file ownership/comments and the existing shared
   test-helper seam:
   - `ls tests`
   - `sed -n '1,260p' tests/test_solver_helpers.h`
   - `sed -n '1,120p' tests/test_chol_csc.c`
   - `sed -n '1,120p' tests/test_ldlt_csc.c`
   - `sed -n '1,120p' tests/test_integration.c`
6. Measure rough helper density in the two CSC giant tests:
   - `python3 - <<'PY' ... helper/assert count ... PY`

### Day 3 Findings

#### 1. The direct-solver giant-test problem reduces cleanly to three different seam shapes, not one generic “big test” pattern

The three target files have materially different ownership shapes:

- `tests/test_chol_csc.c`
  - broad CSC-family proof binary with:
    - alloc/grow/roundtrip/perm/validate scaffolding
    - scalar elimination and solve groups
    - supernodal detection/elimination/writeback clusters
    - dispatch and compatibility tail
- `tests/test_ldlt_csc.c`
  - broad CSC-family proof binary with:
    - alloc/row-adjacency/from-sparse scaffolding
    - analysis-aware repeated-run groups
    - supernodal indefinite groups
    - native-kernel / symmetric-swap / solve groups
- `tests/test_integration.c`
  - smaller cross-family caller-story binary with:
    - baseline LU workflows
    - progress/cancel coverage
    - public direct lifecycle proof
    - repeated-run mismatch/failure-preservation proof
    - iterative/eigensolver progress coverage tail

Interpretation:

- Sprint 57 should not treat all three files as needing the same refactor move
- the CSC family binaries are the real giant-test ownership problem
- `test_integration.c` is mainly a proof-group clarity and later lifecycle
  coverage surface, not the best first extraction target

#### 2. `test_chol_csc.c` is the strongest first refactor target because its largest proof cluster is both contiguous and behaviorally cohesive

The live `test_chol_csc.c` structure already separates into named bands:

- low-risk foundational scaffolding:
  - alloc/grow: `39-154`
  - conversion/roundtrip/permutation/analysis/validate: `155-1012`
  - workspace/scalar elimination/solve: `1013-1997`
- high-mass supernodal and CSC-dispatch cluster:
  - detect-supernodes and postorder: `1998-2515`
  - dense helper and supernodal elimination cross-checks: `2529-3996`
  - writeback and dispatch tail: `4120-4455`

The file also carries `137` test functions with only about `12` helper/assert
functions, which means the main maintainability pain is not “too many generic
helpers”; it is that too many related supernodal proof groups still live in one
translation unit.

Interpretation:

- the strongest first landing should follow the supernodal / dispatch proof
  boundary, not a generic helper-only cleanup
- `test_chol_csc.c` is large enough that a bounded first split can improve
  reviewability materially without scattering unrelated scalar-path proof

#### 3. `test_ldlt_csc.c` is the strongest second target because it already has clear family-local clusters but a denser helper/fixture seam

The live `test_ldlt_csc.c` structure already separates into named bands:

- alloc/row-adjacency and conversion scaffolding: `1-1274`
- analysis-aware repeated-run and supernodal indefinite groups: `1275-1707`
- scalar/native elimination and native-kernel groups: `1784-3166`
- solve / inertia / singularity groups: `3177-3523`

Compared with Cholesky, LDLT has more local helper and fixture density:

- `96` test functions
- about `19` helper/assert/builder functions

Interpretation:

- `test_ldlt_csc.c` is a good second refactor target because it has both:
  - a real giant-file problem
  - a clearer local helper/builder seam than `test_chol_csc.c`
- it is not the best first target because its proof surface mixes more native,
  swap, supernodal, and analysis-aware indefinite ownership in one family

#### 4. `test_integration.c` should stay mostly intact during the first direct-test batch

The live `test_integration.c` structure is already much smaller and more
caller-shaped:

- baseline LU workflows: `20-384`
- progress/cancel plus explicit-analysis/direct-lifecycle groups: `385-1495`
- QR / iterative / eigensolver progress tail: `1502-1800`

Interpretation:

- the direct lifecycle and factor-many tests inside `test_integration.c` are
  already grouped contiguously enough to support later additive coverage
  without making Day 5’s first refactor depend on an integration-file split
- mechanically splitting `test_integration.c` early would risk scattering the
  strongest cross-family caller-story surface without meaningfully reducing the
  largest direct-solver maintenance pain

#### 5. The existing shared helper seam should be extended cautiously, not turned into a generic test framework

The live shared helper layer is still intentionally narrow:

- `tests/test_solver_helpers.h`
  - L2 residual helpers for solver/integration tests

Interpretation:

- Sprint 57 should extend shared helpers only where they clearly support
  repeated direct-solver proof without creating a broad generic framework
- the first direct-test refactor should prefer:
  - family-local helper extraction
  - behavior-cluster separation
  over pushing many CSC-specific details into a global shared test header

#### 6. The first refactor target should be chosen by proof ownership clarity, not only by line count

Ranked candidate landing order after audit:

1. `tests/test_chol_csc.c`
   - first target: supernodal / writeback / dispatch proof cluster
2. `tests/test_ldlt_csc.c`
   - second target: analysis-aware supernodal / native-kernel family seam
3. `tests/test_integration.c`
   - later target only if lifecycle expansion proves a small helper boundary is
     worth it

Rejected as the first move:

- purely mechanical “extract generic asserts first” cleanup
  - too low-impact on reviewability
- early `test_integration.c` split
  - too much cross-family caller-story scattering for too little giant-file
    relief

## Day 3 Close

Sprint 57's direct-solver giant-test problem is now reduced to concrete seam
classes and a ranked landing order:

- `test_chol_csc.c` is the strongest first refactor target
- `test_ldlt_csc.c` is the strongest second target
- `test_integration.c` should stay mostly intact during the first direct-test
  batch
- the first landing should follow a real supernodal/direct-proof boundary, not
  a generic helper-only cleanup

That is enough to move to the Day 4 direct-solver test refactor design without
any remaining ambiguity around the first bounded target.

## Day 4

**Objective:** Freeze the first bounded direct-solver giant-test refactor
boundary before editing permanent proof surfaces by defining the exact
`test_chol_csc.c` ownership split, helper strategy, preserved invariants, and
minimal cleanup policy for the first landing batch.

### Commands Run

1. Re-read the Sprint 57 Day 4 plan item, current notes, and Day 3 audit:
   - `sed -n '180,240p' docs/planning/EPIC_5/SPRINT_57/PLAN.md`
   - `sed -n '520,780p' docs/planning/EPIC_5/SPRINT_57/WORKING_NOTES.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_57/artifacts/day3-direct-solver-giant-test-seam-audit.md`
2. Re-read two recent bounded-design artifacts for shape and invariants:
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_55/artifacts/day9-iterative-decomposition-batch1-design.md`
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_56/artifacts/day7-cholesky-csc-decomposition-design.md`
3. Reconfirm the existing test/build pattern and helper-header precedent:
   - `rg -n "#include \\\"test_.*\\.h\\\"|#include \\\".*helpers.*\\\"" tests`
   - `rg -n "test_chol_csc|test_ldlt_csc|test_integration" Makefile CMakeLists.txt`
4. Inspect the selected `test_chol_csc.c` proof cluster and runner grouping:
   - `sed -n '4548,4645p' tests/test_chol_csc.c`
   - `sed -n '1998,2518p' tests/test_chol_csc.c`
   - `sed -n '2520,3998p' tests/test_chol_csc.c`
5. Locate the current family-local helper seams in that cluster:
   - `python3 - <<'PY' ... detect_supernodes_alloc / day8_count_supernodes / day9_assert_batched_matches_scalar / day11_build_spd ... PY`
6. Measure rough test/helper density inside the selected sub-clusters:
   - `python3 - <<'PY' ... detect/postorder vs dense+supernodal core vs writeback+dispatch ... PY`

### Day 4 Findings

#### 1. The first direct-test landing should stay build-neutral and follow the repo’s existing local helper-header pattern

The live build pattern is clear:

- `test_chol_csc`, `test_ldlt_csc`, and `test_integration` are each compiled
  as single-source test binaries
- the repo already uses local helper headers where that keeps a binary shape
  stable:
  - `tests/test_solver_helpers.h`
  - `benchmarks/bench_backend_compare_helpers.h`
  - `examples/example_alloc_helpers.h`

Interpretation:

- Day 5 should not start by creating a new multi-translation-unit test binary
- the first bounded `test_chol_csc.c` refactor should use a narrow local helper
  header include rather than a new Makefile/CMake test-target topology

#### 2. The selected first seam remains the `test_chol_csc.c` supernodal / writeback / dispatch family

The selected first target stays:

- `tests/test_chol_csc.c`

The first owned proof family remains:

- supernode detection and reporting
- postorder corpus-safety regression
- dense-helper and supernodal elimination cross-checks
- extract/writeback helpers and tests
- writeback and dispatch tail

Measured rough density inside the selected ranges:

- detect/postorder cluster (`1998-2518`)
  - `9` tests
  - `2` family-local helpers
- dense+supernodal core (`2520-3998`)
  - `33` tests
  - `7` family-local helpers
- writeback+dispatch tail (`4120-4455`)
  - `13` tests
  - `1` family-local helper

Interpretation:

- this is large enough to justify a real family extraction
- the cluster already has enough local helper identity that moving it as one
  owned group is cleaner than trying to tease out one or two generic asserts

#### 3. The exact Day 5 ownership split should be helper-header extraction first, not test-binary splitting

Recommended new local helper file:

- `tests/test_chol_csc_supernodal_helpers.h`

Move into it:

- family-local helper functions that are only meaningful for the selected
  supernodal/writeback/dispatch proof family, including:
  - `detect_supernodes_alloc(...)`
  - `day8_count_supernodes(...)`
  - `day9_assert_batched_matches_scalar(...)`
  - `day11_build_spd(...)`
- any small adjacent family-local utility helpers needed only by the moved
  proof cluster

Keep in `tests/test_chol_csc.c`:

- alloc/grow/conversion/permutation/analysis/validate scaffolding
- workspace/scalar elimination/solve/factor-solve groups
- the selected supernodal/writeback/dispatch tests themselves in the first
  batch
- `main()` and current `RUN_TEST(...)` grouping

Interpretation:

- Batch 1 should reduce ownership clutter first by moving the local helper seam
  out of the main file while leaving the test binary and its runner shape
  stable
- a later follow-through day can decide whether the supernodal proof group
  itself should become a second owned include-style block once the helper
  extraction proves useful

#### 4. The first batch should not widen into global CSC test utilities

Do not move in Batch 1:

- broad residual or matrix-builder helpers into `tests/test_solver_helpers.h`
- generic direct-solver CSC helpers shared across Cholesky and LDLT
- any `test_ldlt_csc.c` or `test_integration.c` logic

Reason:

- Sprint 57 still needs the first batch to clarify one family’s ownership
  before trying to generalize helper contracts across multiple giant tests
- widening too early would blur whether the landing actually improved review
  cost in `test_chol_csc.c`

#### 5. The first batch invariants are now explicit

Day 5 must preserve:

- test names and their current intent
- `RUN_TEST(...)` ordering and binary identity for `test_chol_csc`
- output and reporting truthfulness
- corpus fixture coverage, especially:
  - `nos4`
  - `bcsstk04`
  - `bcsstk14`
  - `Kuu`
- one-shot scalar-path versus repeated-run / dispatch proof boundaries
- existing supernodal residual thresholds and scalar↔batched parity checks

This is an ownership and readability change, not a behavior change.

#### 6. The cleanup policy for the first direct-test batch should stay tightly bounded

Day 5 should:

- preserve durable assertion commentary
- preserve comments that explain structural expectations, residual thresholds,
  or corpus-safety meaning
- trim stale sprint-history narrative where it is encountered inside the moved
  supernodal family-local helper block

Day 5 should not:

- normalize the entire `test_chol_csc.c` comment body
- rewrite all historical comments in the retained scalar-path sections
- turn the helper extraction into a repo-wide CSC test-comment cleanup

## Day 4 Close

Sprint 57 now has an exact first direct-solver test landing boundary:

- target file:
  - `tests/test_chol_csc.c`
- first move:
  - extract the supernodal family’s local helper seam into
    `tests/test_chol_csc_supernodal_helpers.h`
- keep the binary shape, runner, and main proof groups stable in Batch 1
- preserve all proof meaning and fixture coverage while reducing local
  ownership clutter

That gives Day 5 a precise, build-neutral, maintainability-first landing plan.
