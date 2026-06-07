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

---

# Day 5 - direct-solver test refactor batch 1

## Summary

Landed the first bounded Sprint 57 direct-solver giant-test refactor by
extracting the `test_chol_csc.c` supernodal family's local helper seam into a
new build-neutral helper header while keeping the test binary, `main()`, and
`RUN_TEST(...)` ordering stable.

## Goals

- execute the Day 4 helper-header extraction plan exactly
- reduce local ownership clutter in `tests/test_chol_csc.c` without changing
  proof meaning
- preserve the existing `test_chol_csc` binary shape and fixture coverage
- validate the batch with the required code-day gate plus the stronger reviewed
  baseline

## Implementation Notes

### 1. The selected helper seam moved into a new local header

Created:

- `tests/test_chol_csc_supernodal_helpers.h`

Moved the family-local helper layer into it:

- `detect_supernodes_alloc(...)`
- `day8_count_supernodes(...)`
- `day9_assert_batched_matches_scalar(...)`
- `day11_build_spd(...)`

Interpretation:

- this is the exact helper seam chosen on Day 4
- the moved functions are meaningful only for the supernodal / writeback /
  dispatch proof family
- the helper header stays local to `test_chol_csc.c` rather than widening into
  a generic CSC test utility surface

### 2. `test_chol_csc.c` kept the proof bodies and runner shape

Retained in `tests/test_chol_csc.c`:

- all existing test bodies
- `main()`
- current grouped `RUN_TEST(...)` ordering
- the existing supernodal / writeback / dispatch proof surfaces themselves

Only structural changes in the main file:

- added a forward declaration for `day8_chol_csc_match(...)` so the new helper
  header can use the existing family-local comparator without reordering the
  test file
- added the new helper-header include
- removed the in-file bodies of the extracted helpers

Interpretation:

- the landing stayed build-neutral
- the batch changed ownership/readability, not proof topology

### 3. The first batch stayed inside the non-goal fence

Did not change:

- `Makefile`
- `CMakeLists.txt`
- `tests/test_ldlt_csc.c`
- `tests/test_integration.c`
- `tests/test_solver_helpers.h`
- test names, residual thresholds, or fixture coverage

This matters because the first Sprint 57 direct-test batch was supposed to
prove that a local ownership seam could land cleanly before any broader
test-topology or cross-family helper redesign.

## Resulting Ownership State

Touched files:

- `tests/test_chol_csc.c`
- `tests/test_chol_csc_supernodal_helpers.h`

Current measured sizes:

- `tests/test_chol_csc.c` = `4552` lines
- `tests/test_chol_csc_supernodal_helpers.h` = `96` lines

Net effect:

- the largest direct-solver giant test now has a real owned helper seam for
  the highest-value supernodal family
- the first Sprint 57 landing reduced main-file helper clutter without
  fragmenting the test binary

## Validation

Required code-day gate:

- `make format`
- `make lint`
- `make test`

All passed.

Stronger reviewed baseline:

- `make quality-review-full`

Passed.

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 266.18 sec`

Focused touched-surface reruns:

- `./build/test_chol_csc` -> `137 / 137`
- `./build/test_integration` -> `37 / 37`
- `./build/test_cholesky` -> `21 / 21`
- `./build/example_analysis` -> residual `4.44e-16`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `speedup_refactor = 1.11x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

## Day 5 Conclusion

Sprint 57 now has its first landed direct-solver giant-test maintainability
batch:

- the target stayed `tests/test_chol_csc.c`
- the chosen Day 4 seam landed exactly as
  `tests/test_chol_csc_supernodal_helpers.h`
- the `test_chol_csc` binary shape and proof meaning stayed intact
- the required quality gate and the stronger reviewed baseline both passed

This gives Day 6 a cleaner base for deciding whether the next direct-solver
giant-test step should stay within Cholesky CSC or shift to the next highest
value proof surface.

---

# Day 6 - direct-solver refactor follow-through audit

## Summary

Re-audited the direct-solver giant-test queue from the landed Day 5 state and
reduced the remaining work to one explicit deferred seam plus one intentional
"leave it dense" decision. The Day 5 helper extraction materially changed the
shape of the queue: `test_chol_csc.c` is still large, but it is now
proof-heavy rather than helper-heavy, so the next highest-value follow-through
target shifts to `tests/test_ldlt_csc.c` instead of forcing a second Cholesky
batch immediately.

## Goals

- shape the remaining direct-test queue from the landed Day 5 reality
- identify the next highest-value direct-test seam without reopening build
  topology or proof meaning
- decide which direct giant-test surfaces can intentionally stay dense in
  Sprint 57
- leave a clean handoff into the Day 7 solver-family audit

## Audit Findings

### 1. Day 5 changed `test_chol_csc.c` from helper-cluttered to proof-heavy

Current touched direct-test sizes:

- `tests/test_chol_csc.c` = `4552` lines
- `tests/test_ldlt_csc.c` = `3680` lines
- `tests/test_integration.c` = `1803` lines

The key qualitative difference after Day 5 is that the selected
supernodal-family helper seam is now gone from the Cholesky giant file:

- `tests/test_chol_csc_supernodal_helpers.h` now owns:
  - `detect_supernodes_alloc(...)`
  - `day8_count_supernodes(...)`
  - `day9_assert_batched_matches_scalar(...)`
  - `day11_build_spd(...)`

What remains in `tests/test_chol_csc.c` is still large, but it is dominated by
cohesive proof groups:

- supernode detection + postorder corpus-safety proof
- dense/supernodal elimination proof
- writeback proof
- dispatch proof

Interpretation:

- another immediate Cholesky helper-only extraction would now buy much less
  than Day 5 did
- the remaining Cholesky density is mostly test-body density, not accidental
  local-helper clutter

### 2. The next highest-value direct-test seam is now in `test_ldlt_csc.c`

`tests/test_ldlt_csc.c` still contains a large contiguous helper-heavy front
half:

- supernode detection
- extract/writeback round-trips
- batched supernodal LDLT cross-checks
- analysis-aware indefinite two-pass workflow proof

The strongest local helper seam is concentrated around:

- `build_dense_ldlt_with_pivots(...)`
- `snapshot_supernode_state(...)`
- `ldlt_csc_factor_state_matches(...)`
- `build_kkt_5x5(...)`
- `build_kkt_10x10(...)`
- `s20_two_pass_indefinite_factor(...)`
- `s20_solve_residual(...)`

Interpretation:

- unlike post-Day-5 `test_chol_csc.c`, LDLT CSC still has a cleaner remaining
  helper-ownership win
- if Sprint 57 needed a second direct-solver test batch, the best boundary is
  now a local LDLT CSC helper header rather than another Cholesky move

### 3. `test_integration.c` should intentionally stay dense in Sprint 57

`tests/test_integration.c` is only `1803` lines and already functions as the
cross-family caller-story hub for:

- public lifecycle sequencing
- factor-many and refactor failure preservation
- progress/cancellation behavior
- one-shot versus repeated-run parity

Interpretation:

- this file should not be split as direct-test follow-through
- Sprint 57 should keep it intact for the later Day 10-11 lifecycle and
  factor-many regression expansion work

## Updated Direct-Test Seam Map

### Intentionally completed in Sprint 57 direct-test work

- `tests/test_chol_csc.c`
  - landed helper extraction via
    `tests/test_chol_csc_supernodal_helpers.h`
  - no further direct-test refactor required before the sprint pivots

### Best remaining direct-test seam, but consciously deferred

- `tests/test_ldlt_csc.c`
  - next-best future seam:
    - a local helper header for the supernode / analysis-aware indefinite
      proof family
  - deferred because Sprint 57 now gets higher value from pivoting to the
    iterative/eigensolver giant tests and the final lifecycle-regression work

### Intentionally dense and preserved

- `tests/test_integration.c`
  - keep intact as the later lifecycle / factor-many proof hub

## Day 6 Conclusion

Day 5 materially changed the direct-test queue:

- the first Cholesky seam is landed and validated
- the remaining Cholesky density is now mostly cohesive proof density
- the strongest remaining direct seam is LDLT CSC, not another immediate
  Cholesky helper peel
- `test_integration.c` stays intact for the later regression-expansion days

That means Sprint 57 can pivot cleanly to Day 7's solver-family giant-test
audit without leaving the direct-test queue ambiguous.

---

# Day 7 - solver-family giant-test audit and design

## Summary

Reduced the iterative/eigensolver giant-test queue to explicit seam classes and
froze the first bounded solver-family refactor boundary. The strongest first
target is now `tests/test_svd.c`, not `tests/test_iterative.c` or
`tests/test_qr.c`, because it combines the largest remaining line count with
the cleanest backend-owned family split.

## Goals

- reduce the remaining solver-family giant-test problem to named seams
- rank the iterative/eigensolver candidate targets by real maintainability
  value rather than line count alone
- select the first bounded solver-family refactor target
- define the exact first ownership boundary before Day 8 code movement

## Audit Findings

### 1. Live solver-family hotspot ranking

Current sizes:

- `tests/test_svd.c` = `3746`
- `tests/test_qr.c` = `3197`
- `tests/test_iterative.c` = `2993`

The highest-value Sprint 57 solver-family targets remain real giant tests, but
their maintainability shapes are not equivalent.

### 2. `test_iterative.c` is large, but already centered on a shared front-door model

`tests/test_iterative.c` currently groups around one public repeated-run
surface with adjacent one-shot and matrix-free variants:

- CG baseline and preconditioned proof
- GMRES baseline and restart/preconditioning proof
- right-preconditioning proof
- matrix-free CG/GMRES proof
- public-handle repeated-run proof for:
  - CG
  - GMRES
  - MINRES

Maintainability conclusion:

- the file is large, but its density is dominated by solver-behavior proof
  rather than a large accidental helper seam
- the strongest remaining local helpers are generic matrix builders and
  callback/preconditioner shims
- extracting those first would buy less clarity than the comparable move in
  `test_svd.c`

### 3. `test_qr.c` is broad, but its helper layer is already consolidated

`tests/test_qr.c` has a wide proof surface:

- basic Householder / reconstruction
- least-squares solve
- SuiteSparse validation
- rank / null-space
- economy mode
- sparse-mode
- refinement

But the file already centralizes its cross-cutting helpers near the top:

- reconstruction-error helpers
- true-residual helpers
- sparse-vs-dense comparison helpers

Maintainability conclusion:

- `test_qr.c` is still large, but it is already closer to a consciously dense
  proof file than to a cluttered helper file
- a first Sprint 57 solver-family move here would likely become a mechanical
  split rather than a high-value ownership improvement

### 4. `test_svd.c` has the cleanest backend-owned seam

`tests/test_svd.c` is the strongest first target because it separates into
large, behaviorally cohesive families:

- Golub-Kahan extraction / validation
- bidiagonal / full-SVD convergence and driver proof
- partial-SVD backend proof
- partial-SVD vector proof
- low-rank / pseudoinverse / condition-number applications
- Sprint 29 outer-product / full-mode follow-through

The most compelling first owned slice is the partial-SVD family:

- partial-SVD backend proof (`Day 11-12` groups)
- partial-SVD vector proof (`Sprint 9 Day 4-5` groups)

This family is attractive because:

- it is large and contiguous
- it is backend-specific rather than truly generic
- it already carries its own local proof identity
- it can be separated without reopening the top-level `test_svd` binary shape

## Ranked Solver-Family Target Order

1. `tests/test_svd.c`
   - first target
   - strongest backend-owned seam
2. `tests/test_iterative.c`
   - second target if Sprint 57 still needs another solver-family test batch
3. `tests/test_qr.c`
   - intentionally deferred unless a later sprint needs broader QR proof
     normalization

## Selected First Refactor Boundary

### Target file

- `tests/test_svd.c`

### First owned seam

- the partial-SVD family, spanning:
  - partial-SVD backend proof
  - partial-SVD vector proof

### Recommended new local helper/proof file

- `tests/test_svd_partial_helpers.h`

The first Sprint 57 solver-family landing should stay build-neutral and use an
include-style local helper/proof seam, not a new test target.

## Exact Day 8 landing fence

### Move into `tests/test_svd_partial_helpers.h`

- partial-SVD-family local helpers and any narrow support code used only by:
  - `test_partial_svd_*`
  - `test_partial_svd_vectors_*`

### Keep in `tests/test_svd.c`

- Golub-Kahan extraction / validation groups
- bidiagonal and full-SVD driver groups
- low-rank / pseudoinverse / condition-number groups
- Sprint 29 outer-product and full-mode follow-through
- `main()` and current `RUN_TEST(...)` order

## Preserved invariants

Day 8 must preserve:

- the `test_svd` binary shape
- `main()` ownership in `tests/test_svd.c`
- test names and their current proof meaning
- low-rank / full-mode / partial-SVD output truthfulness
- corpus fixture coverage, especially:
  - `nos4`
  - `west0067`
  - `bcsstk04`
  - `steam1`
  - `orsirr_1`

This is a maintainability move, not an SVD behavior change.

## Day 7 Conclusion

Sprint 57 now has an explicit first solver-family refactor boundary:

- first target:
  - `tests/test_svd.c`
- first owned seam:
  - the partial-SVD family
- preferred Day 8 landing:
  - a build-neutral local include seam in
    `tests/test_svd_partial_helpers.h`

That leaves Day 8 with a concrete implementation boundary instead of a generic
"refactor one of the big solver tests" instruction.

## Sprint 57 Day 8 - solver-family test refactor batch 1

Date: 2026-06-06 18:09:02 CDT
Branch: `sprint-57`

### Goal

Land the first bounded solver-family giant-test refactor by extracting the
partial-SVD proof family out of `tests/test_svd.c` into the Day 7-selected
build-neutral local helper seam, while keeping the `test_svd` binary shape and
proof meaning intact.

### Implementation

Touched files:

- `tests/test_svd.c`
- `tests/test_svd_partial_helpers.h`

Landed boundary:

- `tests/test_svd.c` now includes `tests/test_svd_partial_helpers.h`
- the extracted owned seam is the partial-SVD family:
  - `test_partial_svd_*`
  - `test_partial_svd_vectors_*`
- `main()` and the current `RUN_TEST(...)` ordering remain in
  `tests/test_svd.c`
- no `Makefile` / `CMakeLists.txt` changes were needed

Measured ownership reduction:

- `tests/test_svd.c`: `3746 -> 2766`
- new `tests/test_svd_partial_helpers.h`: `915`

### Validation

Required gate:

- `make format`
- `make lint`
- `make test`

All passed.

Reviewed baseline:

- `make quality-review-full`

Passed with maintained anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 225.66 sec`

Focused SVD follow-ons:

- `./build/test_svd` -> `97 / 97`
- `./build/test_sprint8_integration` -> `7 / 7`
- `./build/bench_svd`
- `./build/example_svd_lowrank`

Representative retained outputs:

- `bench_svd`
  - `nos4` partial/full = `2.1x`
  - `west0067` partial/full = `2.2x`
  - `bcsstk04` partial/full = `1.5x`
  - `steam1` partial/full = `11.3x`
  - `orsirr_1` partial/full = `236.2x`
- `example_svd_lowrank`
  - sparse low-rank `k=2`: `22 -> 6` nnz
  - compression = `3.7x`

### Day 8 result

The first solver-family refactor landed cleanly and stayed inside the Day 7
fence:

- build-neutral include seam
- same `test_svd` binary shape
- same `main()` ownership
- same `RUN_TEST(...)` ordering
- no SVD behavior, fixture, or caller-surface drift

This leaves Sprint 57 with a real solver-family ownership reduction instead of
just a design note:

- `tests/test_svd.c` is no longer the largest untouched solver-family proof
  surface by the same margin
- the partial-SVD backend/vector proof family is now an explicit owned seam

## Sprint 57 Day 9 - solver-family test refactor batch 2

Date: 2026-06-06 18:25:49 CDT
Branch: `sprint-57`

### Goal

Land the second bounded solver-family maintainability improvement by extracting
the public repeated-run iterative handle proof cluster out of
`tests/test_iterative.c`, keeping the `test_iterative` binary shape, proof
meaning, and support-boundary coverage intact.

### Re-audit result

Post-Day-8 solver-family shape:

- `tests/test_svd.c` is materially smaller and now reads as intentionally
  proof-dense rather than helper-dense
- `tests/test_iterative.c` remains the strongest next solver-family seam
  because its repeated-run public-handle cluster is contiguous and
  behaviorally cohesive
- `tests/test_qr.c` remains intentionally deferred

Selected Day 9 seam:

- `tests/test_iterative.c`
- public repeated-run handle proof cluster:
  - `test_cg_public_handle_validation_reuse_and_on_demand`
  - `test_gmres_public_handle_prepare_reuse_and_growth`
  - `test_minres_public_handle_prepare_reuse_and_growth`

### Implementation

Touched files:

- `tests/test_iterative.c`
- `tests/test_iterative_handle_helpers.h`

Landed boundary:

- `tests/test_iterative.c` now includes
  `tests/test_iterative_handle_helpers.h`
- the extracted owned seam is exactly the public repeated-run handle family
- `main()` and current `RUN_TEST(...)` ordering remain in
  `tests/test_iterative.c`
- no `Makefile` / `CMakeLists.txt` changes were needed

Measured ownership reduction:

- `tests/test_iterative.c`: `2993 -> 2802`
- new `tests/test_iterative_handle_helpers.h`: `197`

### Validation

Required gate:

- `make format`
- `make lint`
- `make test`

All passed.

Focused iterative follow-ons:

- `./build/test_iterative` -> `79 / 79`
- `./build/test_minres` -> `43 / 43`
- `./build/bench_iterative_reuse`
- `./build/example_ic_minres`

Representative retained outputs:

- `bench_iterative_reuse`
  - `cg-tridiag-300` = `1.08x`
  - `gmres-unsym-220` = `1.11x`
  - `minres-kkt-42` = `1.27x`
- `example_ic_minres`
  - `MINRES` on KKT `42x42` = `39` iterations
  - `Jacobi-MINRES` = `26` iterations
  - speedup = `1.5x`

Reviewed baseline:

- `make quality-review-full`

Passed with maintained anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 207.54 sec`

### Day 9 result

The second solver-family batch stayed inside the Day 7-9 fence:

- build-neutral include seam
- same `test_iterative` binary shape
- same `main()` ownership
- same `RUN_TEST(...)` ordering
- no repeated-run support-boundary drift

Intentionally deferred remaining density:

- the larger CG / GMRES / matrix-free proof surface in `tests/test_iterative.c`
  now reads as intentionally dense coverage rather than an obvious next helper
  clutter seam
- `tests/test_qr.c` remains deferred unless a later sprint needs broader
  solver-family proof cleanup

## Sprint 57 Day 10 - lifecycle regression expansion batch 1

Date: 2026-06-06 18:56:28 CDT
Branch: `sprint-57`

### Goal

Expand the public direct repeated-run lifecycle proof in the highest-signal
remaining gap area without changing the Sprint 50-56 support boundary:

- repeated solve behavior on one analyzed/factored direct path
- explicit free-to-zero behavior for `sparse_factors_t`
- explicit free-to-zero behavior for `sparse_analysis_t`
- no widening into new lifecycle semantics or API changes

### Re-audit result

Post-Day-9 lifecycle proof shape:

- `tests/test_integration.c` is still the right Day 10 surface because it owns
  the public direct caller story rather than family-local numeric detail
- existing public lifecycle proof already covered:
  - zeroed-factor solve rejection
  - mismatched analysis/factors rejection
  - zero-init refactor acceptance
  - same-pattern refactor and old-factor preservation on failure
- the strongest still-implicit public contract seam was:
  - repeated `sparse_factor_solve(...)` calls on one valid
    `sparse_analysis_t` + `sparse_factors_t`
  - explicit post-free zeroed state for both public lifecycle structs

### Implementation

Touched files:

- `tests/test_integration.c`

Landed batch:

- added `test_public_lifecycle_repeated_solve_and_free_zeroed`
- the new test:
  - analyzes one SPD tridiagonal matrix
  - performs numeric factorization once
  - solves two distinct right-hand sides through the same public lifecycle
    state
  - verifies both recovered solutions exactly
  - frees `sparse_factors_t` and proves the public struct returns to a zeroed
    state
  - frees `sparse_analysis_t` and proves the public struct returns to a zeroed
    state
  - calls both free entry points again on the zeroed state to prove no-op
    safety

Test-surface effect:

- `./build/test_integration` now passes `38 / 38`
- the new coverage lives alongside the existing public lifecycle rejection and
  refactor tests instead of creating a new binary or broadening family-local
  proof surfaces

### Validation

Required gate:

- `make format`
- `make lint`
- `make test`

All passed.

Focused direct follow-ons:

- `./build/test_integration` -> `38 / 38`
- `./build/example_analysis`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative retained outputs:

- `example_analysis`
  - solve residual = `4.44e-16`
- `bench_refactor_csc nos4`
  - `speedup_refactor = 2.52x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

Reviewed baseline:

- `make quality-review-full`

Passed with maintained anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 203.98 sec`

### Day 10 result

The Day 10 lifecycle batch stayed inside the plan fence:

- no public API changes
- no new solver-family behavior
- no support-boundary drift
- one bounded addition to the public direct repeated-run caller proof

The main Day 10 gain is that the public direct lifecycle story is now less
implicit in two high-signal places:

- repeated solve reuse on one analyzed/factored path is now direct proof, not
  just inference from refactor tests
- `sparse_factor_free(...)` and `sparse_analysis_free(...)` now have direct
  zeroed-state proof in the lifecycle regression surface

## Sprint 57 Day 11 - factor-many and compatibility regression batch

Date: 2026-06-06 19:26:07 CDT
Branch: `sprint-57`

### Goal

Add the highest-signal remaining factor-many / compatibility proof without
changing the Sprint 50-56 public direct lifecycle contract:

- same-pattern refactor-many expectations
- repeated-run versus one-shot compatibility parity
- benchmark-facing workflow truthfulness

### Re-audit result

Post-Day-10 lifecycle proof shape:

- `tests/test_integration.c` remains the right Day 11 surface because it owns
  the public direct caller story rather than family-local kernel detail
- existing lifecycle coverage already proved:
  - zero-init refactor acceptance
  - mismatched existing-factor rejection
  - old-factor preservation on failed refactor
  - repeated solve reuse
  - zeroed free behavior
- the strongest still-implicit factor-many seam was:
  - same-pattern refactor-many parity with the still-supported one-shot
    Cholesky compatibility path across successive value updates

### Implementation

Touched files:

- `tests/test_integration.c`

Landed batch:

- added `test_public_lifecycle_refactor_same_pattern_matches_one_shot_cholesky`
- the new test:
  - analyzes one SPD tridiagonal matrix once with
    `SPARSE_FACTOR_CHOLESKY` + `SPARSE_REORDER_AMD`
  - performs one initial numeric factorization on the shared public path
  - applies two successive same-pattern diagonal-value updates on fresh
    matrices
  - refactors the shared `sparse_factors_t` across both updates
  - solves each updated system on the public repeated-run path
  - factors matching fresh matrices through
    `sparse_cholesky_factor_opts(...)`
  - proves the repeated-run and one-shot paths recover the same solution on
    both updates

Test-surface effect:

- `./build/test_integration` now passes `39 / 39`
- the public lifecycle regression cluster now directly proves the benchmark-
  facing analyze-once / refactor-many compatibility assumption instead of
  leaving it implied by separate one-shot and lifecycle tests

### Validation

Required gate:

- `make format`
- `make lint`
- `make test`

All passed.

Focused factor-many follow-ons:

- `./build/test_integration` -> `39 / 39`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative retained outputs:

- `example_analysis`
  - solve residual = `4.44e-16`
- `bench_refactor`
  - `tridiag-200` = `1.72x`
  - `tridiag-500` = `1.39x`
  - `bcsstk04` = `1.52x`
  - `nos4` = `1.49x`
- `bench_refactor_csc nos4`
  - `speedup_refactor = 1.08x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

Reviewed baseline:

- `make quality-review-full`

Passed with maintained anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 202.80 sec`

### Day 11 result

The Day 11 batch stayed inside the plan fence:

- no public API changes
- no benchmark-driver changes
- no one-shot compatibility wording changes
- one bounded addition to the public direct factor-many regression surface

The main Day 11 gain is that the repeated-run direct path and the one-shot
Cholesky compatibility path are now compared directly on the exact
same-pattern value-update story that the benchmark and example surfaces teach.

## Sprint 57 Day 12 - post-expansion compatibility audit

Date: 2026-06-06 19:46:12 CDT
Branch: `sprint-57`

### Goal

Re-audit the landed Sprint 57 branch after the test refactors and lifecycle
regression additions, then lock the residual queue and final validation
checklist from the actual branch state.

### Audit scope

Re-read from the landed tree:

- direct giant-test surfaces
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_integration.c`
- solver-family giant-test surfaces
  - `tests/test_svd.c`
  - `tests/test_iterative.c`
  - helper seams extracted into:
    - `tests/test_chol_csc_supernodal_helpers.h`
    - `tests/test_svd_partial_helpers.h`
    - `tests/test_iterative_handle_helpers.h`
- caller-facing workflow wording
  - `README.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `docs/tutorial.md`

### Main audit result

The landed Sprint 57 branch still matches the preserved Sprint 50-56 steady-
state contract:

- no public API redesign surfaced
- no solver-family support-boundary drift surfaced
- no benchmark/example contract drift surfaced
- no behavior-visible direct repeated-run lifecycle drift surfaced

The strongest structural compatibility fact is still explicit:

- `master...HEAD` has no `include/` changes at all

That means Sprint 57 has remained a proof-surface and maintainability sprint,
not a hidden product-surface sprint.

### Landed branch shape

Direct-solver proof surfaces:

- `tests/test_chol_csc.c` is smaller and less helper-dense after the Day 5
  supernodal helper extraction
- `tests/test_integration.c` now owns the strongest public direct lifecycle /
  factor-many caller-story cluster
- `tests/test_ldlt_csc.c` remains intentionally dense and is now the clearest
  deferred direct-solver giant-test seam rather than an unnoticed drift

Solver-family proof surfaces:

- `tests/test_svd.c` is materially smaller after the partial-SVD family
  extraction
- `tests/test_iterative.c` is modestly smaller after the public handle-family
  extraction
- `tests/test_qr.c` remains unchanged and intentionally deferred

Caller-facing workflow wording:

- README still states:
  - analyze once
  - factor / solve
  - refactor / solve many on same-pattern value changes
  - one-shot LU / Cholesky / LDL^T remain first-class
- benchmark docs still match the live proof surfaces:
  - `bench_refactor`
  - `bench_refactor_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- examples/tutorial still keep shipped examples one-shot-first while naming the
  bounded repeated-run support surfaces explicitly

### Residual queue

The remaining queue is now bounded and consciously deferred:

- direct-solver giant-test follow-on:
  - `tests/test_ldlt_csc.c` helper-density seam
- solver-family giant-test follow-on:
  - `tests/test_qr.c` only if a later sprint needs broader proof-surface
    cleanup
- broader integration density:
  - `tests/test_integration.c` remains intentionally dense because it is now
    the main public caller-story surface

No blocker-level contract drift surfaced from the Day 12 audit.

### Final validation checklist

Day 13 should validate exactly the landed branch state with:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`

Targeted Sprint 57 follow-ons:

- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/test_svd`
- `./build/test_iterative`
- `./build/test_integration`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

### Day 12 result

Sprint 57’s landed branch now reads cleanly as:

- giant-test maintainability improvement
- direct repeated-run lifecycle proof tightening
- factor-many / one-shot compatibility proof tightening

without any hidden product-surface drift before the final validation sweep.
